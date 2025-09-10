
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def structure_sim(list1, list2):
    n1 = len(list1)
    n2 = len(list2)
    return float(n1-n2) / (n1+n2)

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

net_history = []
n_history = 3
def get_net_similarity(pruner, jaccard_sim=True):
    pruned_net = pruner.get_pruned_net_structure()
    net_history.append(pruned_net)
    if len(net_history) < n_history:
        return 0
    if len(net_history) > n_history:
        net_history.pop(0)
    average_sim = 0
    for (key1, val1), (key2, val2) in zip(net_history[0].items(), net_history[n_history-1].items()):
        if jaccard_sim:
            average_sim += jaccard(val1, val2)
        else:
            average_sim += structure_sim(val1, val2)
    average_sim = average_sim / len(net_history[0])

    return average_sim if jaccard_sim else 1.0-average_sim

pi_all = []
def draw_pi_sim(sim_val, save_path):
    pi_all.append(sim_val)
    pi_all_running_avg = running_average(pi_all, n_history)
    #return pi_all_running_avg[-1]
    
    x = np.arange(len(pi_all_running_avg))
    y = np.array(pi_all_running_avg)
    mask = y > -1
    plt.plot(x[mask], y[mask])
    plt.ylim(0, 1.01)
    plt.xlim(0, len(pi_all))
    plt.savefig(save_path)
    plt.close()
    return pi_all_running_avg[-1]

def running_average(numbers, n):
    """
    Generates a new list with the same length as the input list, where each element is the running average of the last n
    values in the input list.
    """
    result = []
    window_sum = 0
    window = []
    for i, x in enumerate(numbers):
        window.append(x)
        window_sum += x
        if i >= n:
            window_sum -= window.pop(0)
        result.append(window_sum / min(i + 1, n))
    return result

def get_conv_layer_filter_count(model):
    conv_filter_count = {}    
    for idx, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.Linear):
            conv_filter_count[name] = m.weight.shape[0]
    conv_filter_count_new = {}
    for key, value in conv_filter_count.items():
        new_key = key.replace('module.', '') # Remove the 'module' text from the key if exist
        conv_filter_count_new[new_key] = value
    return conv_filter_count_new

def plot_prune_retained_filter_ratios(model_org, model_pruned, save_as='visualize_model.pdf'):
    # Increase font sizes for different text elements
    # Set font to Times New Roman
    rc('font', family='Times New Roman')
    plt.rcParams.update({'font.size': 16})  # Change the default font size

    #dict1 = {'block0.0': 64, 'block0.3': 64, 'block1.0': 128, 'block1.3': 128, 'block2.0': 256, 'block2.3': 256, 'block2.6': 256, 'block3.0': 512, 'block3.3': 512, 'block3.6': 512, 'block4.0': 512, 'block4.3': 512, 'block4.6': 512}
    #dict2 = {'block0.0': 10, 'block0.3': 43, 'block1.0': 84, 'block1.3': 121, 'block2.0': 177, 'block2.3': 147, 'block2.6': 143, 'block3.0': 57, 'block3.3': 85, 'block3.6': 52, 'block4.0': 73, 'block4.3': 52, 'block4.6': 61}

    dict1 = get_conv_layer_filter_count(model_org)
    dict2 = get_conv_layer_filter_count(model_pruned)

    keys = list(dict1.keys())
    heights = np.array([dict2.get(key, 0) for key in keys])
    colors = ['#663300' if dict2.get(key) is not None else '#0000ff' for key in keys]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    labels = ax.get_xticklabels()
    for label in labels:
        label.set(rotation=60, horizontalalignment='right', verticalalignment='top')

    ax.tick_params(axis='y', length=0, pad=10, labelcolor='dimgrey', zorder=1)
    ax.tick_params(axis='x', colors='dimgrey', length=5, width=1)  # Change x-axis tick color to light black
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.bar(keys, list(dict1.values()), width=0.5, bottom=0.1, color='#b36b00', alpha=0.5, linewidth=1, edgecolor='black', zorder=2)
    plt.bar(keys, heights, color=colors, width=0.5, bottom=0.1, linewidth=1, edgecolor='black', zorder=2)

    ax.set_ylabel('Total Number of Filters', fontsize=24)  # Increase font size for y-axis label
    ax.set_xlabel('Layers', fontsize=24)  # Increase font size for x-axis label

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.legend(['Pruned Filters', 'Preserved Filters'], loc='upper left', fontsize=24)  # Increase font size for legend
    plt.savefig(save_as, dpi=600)
    plt.close()   