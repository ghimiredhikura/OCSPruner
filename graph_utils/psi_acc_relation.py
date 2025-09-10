import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import seaborn as sns
import os
import re

def load_data(file_path):
    epochs = []
    sim_vals = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            epochs.append(int(parts[-2]))
            sim_vals.append(float(parts[-1]))
    return epochs, sim_vals

def parse_accuracy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            match = re.search(r'Best Acc=([\d.]+)', line)
            if match:
                return float(match.group(1))
    return None

def process_folder(data_dir, root_folders, subfolder):
    best_accs = []
    for root_folder in root_folders:
        folder_path = os.path.join(data_dir, root_folder, subfolder)
        file_path = os.path.join(folder_path, 'cifar10-resnet56.txt')
        if os.path.exists(file_path):
            acc = parse_accuracy(file_path)
            if acc is not None:
                best_accs.append(acc)
    return best_accs

def parse_data(data_dir):
    root_folders = ['exp_1', 'exp_2', 'exp_3']
    subfolder_names = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '94', '110', '120', '130']#, '140', '150']

    pruning_epoch_vec = []
    mean_vec = []
    std_vec = []

    for subfolder_name in subfolder_names:
        best_accs = process_folder(data_dir, root_folders, subfolder_name)
        avg_acc = np.mean(best_accs) * 100.0
        std_acc = np.std(best_accs) * 100.0

        pruning_epoch_vec.append(int(subfolder_name))
        mean_vec.append(avg_acc)
        std_vec.append(std_acc)

    return pruning_epoch_vec, mean_vec, std_vec

def draw_avg_with_std(data_dir, save_path):

    # List of psi.txt file names
    file_names = ['psi_exp_1.txt', 'psi_exp_2.txt', 'psi_exp_3.txt']
    # List of full file paths
    file_paths = [os.path.join(data_dir, file_name) for file_name in file_names]

    all_sim_vals = []

    for file_path in file_paths:
        epochs, sim_vals = load_data(file_path)
        all_sim_vals.append(sim_vals)

    data_len = max(len(x) for x in all_sim_vals)

    # Pad shorter lists with NaNs to match the maximum length
    for i in range(len(all_sim_vals)):
        if len(all_sim_vals[i]) < data_len:
            padding_length = data_len - len(all_sim_vals[i])
            all_sim_vals[i] += [np.nan] * padding_length

    avg_sim_vals = np.nanmean(all_sim_vals, axis=0)
    std_sim_vals = np.nanstd(all_sim_vals, axis=0)

    # Configure Matplotlib to use Times New Roman font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    sns.set_theme(style="whitegrid", rc={"axes.labelsize": 12, "axes.titlesize": 12})
    font = {'fontname': 'Times New Roman', 'weight': 'bold'}

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=250)

    # Data for Top@1 Accuracy
    stable_pruning_epoch = 94 # we average this epoch from three trials. 
    epochs_acc_org, avg_values_org, std_values_org = parse_data(data_dir)

    data_len = epochs_acc_org[-1]
    x = np.arange(data_len)

    avg_sim_vals = avg_sim_vals[:data_len]
    std_sim_vals = std_sim_vals[:data_len]

    epochs_acc, avg_values, std_values = epochs_acc_org[:10], avg_values_org[:10], std_values_org[:10]
    epochs_acc_1, avg_values_1, std_values_1 = epochs_acc_org[9:], avg_values_org[9:], std_values_org[9:]
    
    ax.errorbar(epochs_acc, avg_values, yerr=std_values, fmt='-o', capsize=3, label='Top-1 Acc. (%)', color='#8B0000')
    ax.errorbar(epochs_acc_1, avg_values_1, yerr=std_values_1, fmt='-o', capsize=3, label='', color='#8B0000', alpha=0.2)

    # Color mapping and labeling for the 94th position
    for i, (a, b) in enumerate(zip(epochs_acc, avg_values)):
        if a == stable_pruning_epoch:  # 94th position (since indexing starts from 0)
            ax.errorbar(a, b, yerr=std_values[i], fmt='-', capsize=3, color='blue', marker='*', markersize=12, markeredgecolor='blue', markerfacecolor='blue', ecolor='purple')
            ax.annotate('Stable Pruning Epoch', xy=(a, b), xytext=(a-41, b+0.04),
                        arrowprops=dict(facecolor='black', arrowstyle='->', color='blue'), **font, size=14)
    
    ax.set_ylabel('Top-1 Acc. (%)', **font, size=14, color='#8B0000')
    ax.set_xlabel('Training/Pruning @Epochs', **font, size=14)

    ax.set_ylim(93.25, 94.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis='x', color='lightgray', alpha=0.3)
    ax.grid(True, axis='y', color='lightgray', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(10))

    ax2 = ax.twinx() # Create a twin Axes sharing the x-axis
    ax2.errorbar(x[:95], avg_sim_vals[:95], yerr=None, fmt='', capsize=5, label='Sub-network\nStability Score', color='#006265')
    #ax2.legend(loc='upper left')
    ax2.fill_between(x[:95], avg_sim_vals[:95] - std_sim_vals[:95], avg_sim_vals[:95] + std_sim_vals[:95], color='#006265', alpha=0.2)
    cutoff_value = 93
    # Plot after cutoff with reduced alpha
    ax2.errorbar(x[x > cutoff_value], avg_sim_vals[x > cutoff_value], yerr=None, fmt='', capsize=5, color='#006265', alpha=0.2)
    ax2.fill_between(x[x > cutoff_value], avg_sim_vals[x > cutoff_value] - std_sim_vals[x > cutoff_value], avg_sim_vals[x > cutoff_value] + std_sim_vals[x > cutoff_value], color='#006265', alpha=0.1)

    ax2.set_xlabel('Pruning @Epochs', **font, size=14)
    ax2.set_ylabel('Sub-network Stability Score', **font, size=14, color='#17becf')
    #ax2.set_xlabel('Pruning @Epochs')
    #ax2.set_ylabel('Sub-network Stability Score')
    ax2.set_ylim(0.7, 1.005)  # Set y-limit from 0.5 to 1.01
    ax2.set_xlim(-1, data_len+1)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, axis='x', color='lightgray', alpha=0.3)
    ax2.grid(True, axis='y', color='lightgray', alpha=0.3)
    ax2.xaxis.set_major_locator(MultipleLocator(10))

    colors = {
        'warmup': '#FFEDC2',      # Warm pale yellow
        'searching': '#AED9E0',   # Cool soft teal
        'stable': '#D4A5A5',      # Muted rose (a desaturated red)
        'white': '#FFFFFF',      # White
        'gray': '#D3D3D3'         # Light gray
    }
    #ax.axvspan(0, 30, color=colors['gray'], alpha=0.1)
    ymin, ymax = ax.get_ylim()
    ax.fill_between([30, 130], ymin, ymax, hatch='xx', facecolor='none', alpha=0.1, edgecolor='#EEEEEE', linewidth=0.5)
    ax.axvspan(0, 30, color=colors['white'], alpha=0.25)
    ax.axvspan(30, stable_pruning_epoch, color=colors['white'], alpha=0.25)
    ax.axvspan(stable_pruning_epoch, max(epochs_acc_org), color=colors['white'], alpha=0.25)

    #ax.axvline(x=30, color='black', linestyle='--', linewidth=1, alpha=0.2)

    # Adding text annotations with Times New Roman font
    ax.text(15, 93.33, 'Standard\nTraining', ha='center', va='center', color='black', fontsize=14, **font)
    ax.text(55, 93.33, 'Training with Structured\nSparsity Regularization', ha='center', va='center', color='black', fontsize=14, **font)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()# Define the font properties for the legend
    font_legend = {'family': 'Times New Roman', 'size': 14}
    # Setting legend with Times New Roman font
    legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', prop=font_legend)

    # Set tick parameters to use Times New Roman font
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in (ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_color('#8B0000')
    
    # Set tick parameters to use Times New Roman font
    ax2.tick_params(axis='both', which='major', labelsize=12)
    for label in (ax2.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_color('#006265')
    
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')

    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Directory containing the psi.txt files
data_dir = 'abla_r56'
# Output file path
out_graph = 'psi_accuracy_connection.pdf'
# Draw average curve with standard deviation
draw_avg_with_std(data_dir, out_graph)