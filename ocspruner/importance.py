
import torch
import torch.nn as nn
import vainf_torch_pruning.torch_pruning as tp
from vainf_torch_pruning.torch_pruning import function
from vainf_torch_pruning.torch_pruning._helpers import _FlattenIndexMapping
import math

class GroupFOTaylerImportance():
    """For channel importance score calculation."""
    def __init__(self):
        self._neuron_metric = {}
        self._accumulate_count = 0
        self.importance_score = 0
        self.p = 2

    def update_importance(self, group):
        """Accumulate the neuron importance score for the layer groups to be pruned.
        Args:
            layers (Dict[str, torch.nn.Module]): Mapping the layer name to the corresponding
                layer module.
            prune_groups (List[Tuple[str, ...]]): A list of layer groups where each group of
                layers are pruned together. Each group of layer names is saved in a tuple.
            layer_bn (Dict[str, Any]): Mapping the name of a convolution layer to the name of
                its following batch normalization layer. Default is None, which means no
                batch normalization layer in the model.
        """
        if (
            not torch.isnan(self.importance_score).any()
            and not torch.isinf(self.importance_score).any()
        ):
            if group not in self._neuron_metric:
                self._neuron_metric[group] = self.importance_score
            else:
                self._neuron_metric[group] += self.importance_score
            self._accumulate_count += 1
        else:
            return
 
    def calc_average_importance(self):
        """Get the average importance score from the accumulation."""
        for group_name in self._neuron_metric.keys():
            # Average the score over past iterations.
            self._neuron_metric[group_name] /= self._accumulate_count

    def reset_importance(self):
        """Reset the neuron importance to 0."""
        #for group in self._neuron_metric.keys():
        #    self._neuron_metric[group][:] = 0
        self._neuron_metric.clear()
        self._accumulate_count = 0

    def get_importance(self, group, ch_groups=1):
        """Get the neuron importance metric value."""
        if group in self._neuron_metric.keys():
            return self._neuron_metric[group]
        else:
            print("FOTayler Score is not available for Group", group) 
            return self._calculate_group_norm_importance(group, ch_groups)

    @torch.no_grad()
    def _calculate_first_order_taylor(self, group, ch_groups=1):
        """First order of Taylor expansion importance calculation.
        Get the importance score for neuron using first order of Taylor expansion
        according to https://arxiv.org/abs/1906.10771.
        """
        group_imp = 0
        group_cc_size = 0
        # Get first order taylor group importance 
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels 
            if prune_fn in [ function.prune_conv_out_channels, function.prune_linear_out_channels ]:
                w = layer.weight.data[idxs].flatten(1)
                g = layer.weight.grad.data[idxs].flatten(1)
                local_imp = (w * g).sum(1).abs()
                local_imp = local_imp / math.sqrt(w.shape[1])
                if ch_groups > 1:
                    local_imp = local_imp.view(ch_groups, -1).sum(0)
                    local_imp = local_imp.repeat(ch_groups) 
                group_imp += local_imp
                group_cc_size += 1
            # Conv in channels
            elif prune_fn in [ function.prune_conv_in_channels, function.prune_linear_in_channels ]:
                is_conv_flatten_linear = False
                if isinstance(layer, nn.ConvTranspose2d):
                    w = (layer.weight.data).flatten(1)
                    g = (layer.weight.grad.data).flatten(1)
                else:
                    w = (layer.weight.data).transpose(0, 1).flatten(1)
                    g = (layer.weight.grad.data).transpose(0, 1).flatten(1)
                if (w.shape[0] != group_imp.shape[0]):
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        # conv-flatten
                        w = w[idxs].view(group_imp.shape[0], w.shape[0] // group_imp.shape[0], w.shape[1]).flatten(1)
                        g = g[idxs].view(group_imp.shape[0], g.shape[0] // group_imp.shape[0], g.shape[1]).flatten(1)
                        is_conv_flatten_linear = True
                elif ch_groups > 1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                    # non-grouped conv with group convs
                    w = w.view(w.shape[0] // group_imp.shape[0],
                                group_imp.shape[0], w.shape[1]).transpose(0,1).flatten(1)
                    g = w.view(g.shape[0] // group_imp.shape[0],
                                group_imp.shape[0], g.shape[1]).transpose(0,1).flatten(1)
                local_imp = (w * g).sum(1).abs()
                local_imp = local_imp / math.sqrt(w.shape[1])
                if ch_groups > 1:
                    if len(local_imp) == len(group_imp):
                        local_imp = local_imp.view(ch_groups, -1).sum(0)
                    local_imp = local_imp.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_imp = local_imp[idxs]
                group_imp += local_imp
                group_cc_size += 1
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN with bias (https://github.com/NVlabs/Taylor_pruning/blob/master/pruning_engine.py#L317)
                if layer.affine: 
                    criteria_w = layer.weight.data[idxs] * layer.weight.grad.data[idxs]
                    #criteria_b = layer.bias.data[idxs] * layer.bias.grad.data[idxs]
                    #local_imp = (criteria_w + criteria_b).abs()
                    local_imp = (criteria_w).abs()
                    if ch_groups > 1:
                        local_imp = local_imp.view(ch_groups, -1).sum(0)
                        local_imp = local_imp.repeat(ch_groups)
                    group_imp += local_imp
                    group_cc_size += 1
        group_imp = group_imp / math.sqrt(group_cc_size)
        group_imp = group_imp / group_imp.max()
        self.importance_score = group_imp

    @torch.no_grad()
    def _calculate_group_norm_importance(self, group, ch_groups=1):

        group_norm = 0

        #Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                local_norm = local_norm / math.sqrt(w.shape[1])
                #print(local_norm.shape, layer, idxs, ch_groups)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm+=local_norm
                #if layer.bias is not None:
                #    group_norm += layer.bias.data[idxs].pow(2)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if isinstance(layer, nn.ConvTranspose2d):
                    w = (layer.weight).flatten(1)  
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)             
                if (w.shape[0] != group_norm.shape[0]):  
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        #conv-flatten
                        w = w[idxs].view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        is_conv_flatten_linear = True
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # non-grouped conv with group convs
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)           
                local_norm = w.abs().pow(self.p).sum(1)
                local_norm = local_norm / math.sqrt(w.shape[1])
                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_norm = local_norm[idxs]
                group_norm += local_norm
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm

        group_imp = group_norm**(1/self.p)
        group_imp = group_imp / group_imp.max()
        return group_imp 
    
class GroupNormV2Importance():
    def __init__(self, p=2):
        self.p = p
        
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_norm = 0
        group_cc_size = 0
        #Get group norm
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Conv out_channels
            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:  
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = w.abs().pow(self.p).sum(1)
                local_norm = local_norm / math.sqrt(w.shape[1])
                #local_norm = local_norm / (w.shape[1])

                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_norm+=local_norm
                group_cc_size += 1
                #if layer.bias is not None:
                #    group_norm += layer.bias.data[idxs].pow(2)
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, 'transposed') and layer.transposed:
                    w = (layer.weight).flatten(1)  
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)             
                if (w.shape[0] != group_norm.shape[0]):  
                    if (hasattr(dep, 'index_mapping') and isinstance(dep.index_mapping, _FlattenIndexMapping)):
                        #conv-flatten
                        w = w[idxs].view(
                            group_norm.shape[0],
                            w.shape[0] // group_norm.shape[0],
                            w.shape[1],
                        ).flatten(1)
                        is_conv_flatten_linear = True
                    elif ch_groups>1 and prune_fn==function.prune_conv_in_channels and layer.groups==1:
                        # non-grouped conv with group convs
                        w = w.view(w.shape[0] // group_norm.shape[0],
                                group_norm.shape[0], w.shape[1]).transpose(0, 1).flatten(1)           
                local_norm = w.abs().pow(self.p).sum(1)
                local_norm /= math.sqrt(w.shape[1])
                #local_norm = local_norm / (w.shape[1])

                if ch_groups>1:
                    if len(local_norm)==len(group_norm):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                if not is_conv_flatten_linear:
                    local_norm = local_norm[idxs]
                group_norm += local_norm
                group_cc_size += 1
            # BN
            elif prune_fn == function.prune_batchnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    local_norm = w.abs().pow(self.p)
                    if ch_groups>1:
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                        local_norm = local_norm.repeat(ch_groups)
                    group_norm += local_norm
                    group_cc_size += 1

        #group_norm = group_norm**(1/self.p)
        group_imp = group_norm / group_cc_size
        #group_imp = group_imp / math.sqrt(group_cc_size)
        #group_imp = group_imp / group_imp.max()
        return group_imp 