import vainf_torch_pruning.torch_pruning as tp

from .pruner import get_pruner
from copy import deepcopy

import math

def prune_channel_group(pruner, group, channel_idxs):
    module = group[0][0].target.module
    pruning_fn = group[0][0].handler
    pruning_group = pruner.DG.get_pruning_group(module, pruning_fn, channel_idxs)
    if pruner.DG.check_pruning_group(pruning_group):
        pruning_group.prune()

def calculate_channel_cost_ingroup(model, args, example_input):

    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_input)    
    pruner_org = get_pruner(model, args, example_input) 

    FLOPs_groups = {}
    FLOPs_groups[f'base_model'] = (base_ops, -1)
    model_pruned = deepcopy(model)

    for idx in range(len(pruner_org.groups_prune_candidate)):
        pruner = get_pruner(model_pruned, args, example_input) 
        group = pruner.groups[idx]
        channel_idxs = group[0][1]
        prune_channel_group(pruner, group, channel_idxs[:-1])
        pruned_ops, _ = tp.utils.count_ops_and_params(model_pruned, example_inputs=example_input)
        cost = float(base_ops-pruned_ops)/len(channel_idxs[:-1])
        FLOPs_groups[f'group{idx+1}'] = (cost, len(channel_idxs))
        model_pruned = deepcopy(model)

    # extract the values from the dictionary
    values = list(FLOPs_groups.values())[1:]
    
    # calculate the sum of exponential values
    exp_sum = sum([math.exp(v[0] / 1e6) for v in values])

    # calculate the softmax probabilities
    probs = [math.exp(v[0]/1e6)/exp_sum for v in values]

    # update the last item in the value field with softmax probabilities
    for i, (k, v) in enumerate(FLOPs_groups.items()):
        v = v + (-1,) if k == 'base_model' else v + (probs[i-1],)
        FLOPs_groups[k] = v
    '''
    # what is the ops if we remove n channels from group 0? 
    n = 20
    contrib_f, n_channels = FLOPs_groups[f'group{1}']
    remain_ops_lut = base_ops - contrib_f * n

    # lets check physically removing the n channels from group 0
    group = org_groups[0]
    channel_idxs = group[0][1]
    prune_channel_group(pruner_org, group, channel_idxs[:n])
    remain_ops_real, _ = tp.utils.count_ops_and_params(model, example_inputs=example_input)
    print(remain_ops_real, remain_ops_lut)
    # Note: this does not work if we remove channels from multiple groups
    '''
    return FLOPs_groups