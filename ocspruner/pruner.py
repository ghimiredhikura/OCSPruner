import torch
import torch.nn as nn
from functools import partial

import vainf_torch_pruning.torch_pruning as tp
from .importance import GroupFOTaylerImportance, GroupNormV2Importance
from vainf_torch_pruning.torch_pruning._helpers import _FlattenIndexMapping
from vainf_torch_pruning.torch_pruning import function

import matplotlib.pyplot as plt
import numpy as np
import math

class OCSPruner(tp.pruner.GroupNormPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        reg=1e-4,
        alpha=4,
        channel_groups=dict(),
        ignored_layers=None,
        group_cost = None,
    ):
        super(OCSPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            ignored_layers=ignored_layers,
        )
        self.reg = reg
        self.alpha = alpha
        self.groups_prune_candidate = list(self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types))
        self.groups_prune = []
        self.groups_noprune = []
        self.groups_pruning_indices = {}
        self.group_cost = group_cost
        self.example_inputs = example_inputs

        plt.style.use('ggplot')

    @torch.no_grad()
    def get_all_groups_norm(self, groups, p=2):
        group_norm_all = []
        for group in groups:
            group_norm = self.get_group_norm(group, p)
            group_norm_all.extend(group_norm.cpu().numpy())
        return group_norm_all

    @torch.no_grad()
    def get_prune_groups_total_norm(self, p=2):
        group_norm_all = []
        for group in self.groups_prune:
            group_norm = self.get_group_norm(group, p)
            group_norm_all.extend(group_norm)
        return sum(group_norm_all)

    @torch.no_grad()
    def plot_group_norm_hist(self, prune_groups, noprune_groups=None, save_as="hist.jpg",p=2):
        group_norm_all = self.get_all_groups_norm(prune_groups, p)
        HIST_BINS = np.linspace(0, 5, 100)
        plt.hist(group_norm_all, HIST_BINS, color='red', alpha=0.5)
        if noprune_groups is not None:
            group_norm_all = self.get_all_groups_norm(noprune_groups, p)
            plt.hist(group_norm_all, HIST_BINS, color='blue', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_as)
        plt.close()

    @torch.no_grad()
    def plot_all_group_norms(self):
        group_norm_all = []
        x_ticks = []  # Adjusted x-axis tick positions
        x_tick_labels = []  # Labels for the ticks
        index = 0
        
        for i, group in enumerate(self.groups_prune_candidate):
            group_norm = self.get_group_norm(group, 2) * (1.0 - self.group_cost[f'group{i+1}'][-1])
            group_norm_values = group_norm.cpu().numpy()
            
            # Append group_norm_values to group_norm_all
            group_norm_all.extend(group_norm_values)
            
            # Calculate the midpoint index for the current group
            midpoint_index = index + len(group_norm_values) / 2
            x_ticks.append(midpoint_index)
            x_tick_labels.append(f'Group {i+1}')
            
            # Update the index
            index += len(group_norm_values)
        
        # Create a figure and axis object
        fig, ax = plt.subplots()
        # Plot the bar chart for all data
        ax.bar(range(len(group_norm_all)), group_norm_all)
        # Set the x-tick positions and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel('Groups')
        ax.set_ylabel('Group Norm')
        ax.set_title('Group Norm Histogram')
        plt.tight_layout()
        plt.savefig("norm.jpg")
        plt.show()  # This will display the plot
        plt.close()

    @torch.no_grad()
    def get_group_norm(self, group, p=2):
        ch_groups = self.get_channel_groups(group)
        group_norm = 0
        # Get group norm
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(p).sum(1)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm+=local_norm
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                w = (layer.weight).transpose(0, 1).flatten(1)
                if (
                    w.shape[0] != group_norm.shape[0]
                ):  
                    if hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping):
                        # conv - latten
                        w = w.view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # group conv
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)               
                local_norm = w.abs().pow(p).sum(1)
                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm += local_norm[idxs]
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    #print(i, "bn", w.shape)
                    local_norm = w.abs().pow(p)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm
        current_channels = len(group_norm)
        if ch_groups>1:
            group_norm = group_norm.view(ch_groups, -1).sum(0)
            group_stride = current_channels//ch_groups
            group_norm = torch.cat([group_norm+group_stride*i for i in range(ch_groups)], 0)
        group_norm = group_norm**(1/p)
        return group_norm

    @torch.no_grad()
    def scale_weights_towards_zero(self, weight_scale_fact=1.0):
        for i, group in enumerate(self.groups_prune):
            ch_groups = self.get_channel_groups(group)
            # Scale weights towards zero
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]:
                    layer.weight.data[idxs] *= weight_scale_fact
                elif prune_fn in [
                    function.prune_conv_in_channels,
                    function.prune_linear_in_channels,
                ]:
                    # regularize input channels
                    if prune_fn==function.prune_conv_in_channels and layer.groups>1:
                        scale = scale[:len(idxs)//ch_groups]
                        idxs = idxs[:len(idxs)//ch_groups]
                    layer.weight.data[:, idxs] *= weight_scale_fact
                elif prune_fn == function.prune_batchnorm_out_channels:
                    # regularize BN
                    if layer.affine is not None:
                        layer.weight.data[idxs] *= weight_scale_fact

    def update_pruning_groups(self, groups_pruning_indices, need_noprune_group_too=False):
        self.groups_pruning_indices = groups_pruning_indices
        self.groups_prune = list(self.get_pruning_groups_global(groups_pruning_indices))
        if need_noprune_group_too:
            self.groups_noprune = list(self.get_nopruning_groups_global())

    def accumulate_importance(self):
        """Accumulate the neuron importance score for the layer groups to be pruned.
        """
        for group in self.groups_prune_candidate:
            ch_groups = self.get_channel_groups(group)
            # invoke these two functionns in sequence to accumulate FOTayler importance over the minibatches
            self.importance._calculate_first_order_taylor(group, ch_groups)
            self.importance.update_importance(group)

    def estimate_importance(self, group, ch_groups=1):
        if hasattr(self.importance, 'get_importance'):
            return self.importance.get_importance(group, ch_groups=ch_groups)
        else: 
            return self.importance(group, ch_groups=ch_groups)

    def get_pruning_groups_global(self, groups_pruning_indices):
        #if groups_pruning_indices is None:
        #    groups_pruning_indices = self.get_groupwise_pruning_indices()
       
        for i, group in enumerate(self.groups_prune_candidate):
            if f'group{i+1}' not in groups_pruning_indices.keys():
                continue
            module = group[0][0].target.module
            pruning_fn = group[0][0].handler
            pruning_indices = groups_pruning_indices[f'group{i+1}']            
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self.DG.check_pruning_group(group):
                yield group

    def get_nopruning_groups_global(self):
        for i, group in enumerate(self.groups_prune_candidate):
            if f'group{i+1}' not in self.groups_pruning_indices.keys():
                yield group
            else:
                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                pruning_indices = self.groups_pruning_indices[f'group{i+1}']
                out_channels = self.DG.get_out_channels(module)
                nopruning_indices = [x for x in range(out_channels) if x not in pruning_indices]
                group = self.DG.get_pruning_group(module, pruning_fn, nopruning_indices)
                if self.DG.check_pruning_group(group):
                    yield group

    def get_all_group_imp_scores(self, layer_max_prune_limit=0.7, dynamic_pruning_limit_fact=1.0):
        global_importance = []        
        for i, group in enumerate(self.groups_prune_candidate):
            module = group[0][0].target.module
            ch_groups = self.get_channel_groups(group)
            imp = self.estimate_importance(group, ch_groups=ch_groups) * (1-self.group_cost[f'group{i+1}'][-1])
            # preserve top n% channels in each group
            out_channels = self.DG.get_out_channels(module)
            n_donot_prune = out_channels - int(out_channels * self.calculate_pruning_limit(out_channels, dynamic_pruning_limit_fact, layer_max_prune_limit))
            if ch_groups > 1:
                imp = imp[:len(imp)//ch_groups]
            topk_thresh = torch.topk(imp, k=n_donot_prune).values[-1]
            imp[imp >= topk_thresh] = 100.0
            global_importance.append((group, ch_groups, imp))
        return global_importance

    def get_group_pruning_indices(self, all_groups_with_imp, all_imp, idx):
        groups_pruning_indices = {}
        for i, (group, ch_groups, imp) in enumerate(all_groups_with_imp):
            pruning_indices = (imp <= all_imp[idx]).nonzero().view(-1)
            module = group[0][0].target.module
            if ch_groups > 1:
                group_size = self.DG.get_out_channels(module)//ch_groups
                pruning_indices = torch.cat([pruning_indices+group_size*i for i in range(ch_groups)], 0)
            if self.round_to:
                n_pruned = len(pruning_indices)
                n_pruned = n_pruned - (n_pruned % self.round_to)
                pruning_indices = pruning_indices[:n_pruned]
            pruning_indices = pruning_indices.tolist()
            if len(pruning_indices) < 1:
                continue
            groups_pruning_indices[f'group{i+1}'] = pruning_indices
        return groups_pruning_indices

    def calculate_pruning_limit(self, C, k=0.015, max_limit=0.7):
        return max_limit * (1 - np.exp(-k * C))

    def pruning(self, group_pruning_indices):
        self.model.eval()
        base_ops, _ = tp.utils.count_ops_and_params(self.model, example_inputs=self.example_inputs)

        groups = self.get_pruning_groups_global(group_pruning_indices)
        for group in groups:
            group.prune()

        pruned_ops, _ = tp.utils.count_ops_and_params(self.model, example_inputs=self.example_inputs)
        speed_up = float(base_ops) / pruned_ops
        return speed_up
    
    def get_model_info(self):
        self.model.eval()
        base_ops, params = tp.utils.count_ops_and_params(self.model, example_inputs=self.example_inputs)
        return base_ops, params

    def get_pruned_net_structure(self):
        pruned_net_structure = {}
        for i, group in enumerate(self.groups_prune_candidate):
            key_name = f'group{i+1}'
            module = group[0][0].target.module
            out_nchannels = self.DG.get_out_channels(module)
            all_ch_indices = [x for x in range(0, out_nchannels)]
            pruned_net_structure[key_name] = all_ch_indices
            if key_name in self.groups_pruning_indices.keys():
                remain_indices = [x for x in all_ch_indices if x not in self.groups_pruning_indices[key_name]]
                pruned_net_structure[key_name] = remain_indices
        return pruned_net_structure
    
def get_pruner(model, args, example_inputs):
    if args.method == "group_norm_sl":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg)
    elif args.method == "group_norm_sl_ocspruner":
        imp = GroupNormV2Importance(p=2)
        pruner_entry = partial(OCSPruner, reg=args.reg)
    elif args.method == "group_norm_sl_ocspruner_p1":
        imp = GroupNormV2Importance(p=1)
        pruner_entry = partial(OCSPruner, reg=args.reg)
    else:
        raise NotImplementedError

    # ignore output layers
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)
    #ignored_layers.append(model.features[0][0])
    #ignored_layers.append(model.conv1)

    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        ignored_layers=ignored_layers
    )
    return pruner