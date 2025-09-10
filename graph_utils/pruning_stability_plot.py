import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from matplotlib import ticker

# Set the style using Seaborn
sns.set(style="whitegrid")

# Set font to Times New Roman
rc('font', family='Times New Roman')
# Increase font sizes for different text elements
plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})  # Change the default font size

pruning_stability_raw = [
    0.0000, 0.5067, 0.6899, 0.7429, 0.7893, 0.8158, 0.8395, 0.8472, 0.8661, 0.8497, 0.8560, 0.8656, 0.8640,
    0.8577, 0.8593, 0.8650, 0.8545, 0.8642, 0.8906, 0.9077, 0.9293, 0.9422, 0.9588, 0.9683, 0.9812, 0.9829,
    0.9874, 0.9929, 0.9964, 0.9978, 0.9998, 0.9973, 0.9983, 0.9969, 0.9982, 0.9994, 0.9993, 0.9978, 0.9985,
    1.0000, 0.9992, 0.9992, 0.9992
]

pruning_stability_avg = [
    0.0000, 0.1689, 0.3989, 0.6465, 0.7407, 0.7827, 0.8149, 0.8342, 0.8509, 0.8544, 0.8573, 0.8571, 0.8619,
    0.8625, 0.8603, 0.8607, 0.8596, 0.8612, 0.8698, 0.8875, 0.9092, 0.9264, 0.9434, 0.9564, 0.9694, 0.9775,
    0.9839, 0.9878, 0.9923, 0.9957, 0.9980, 0.9983, 0.9985, 0.9975, 0.9978, 0.9982, 0.9989, 0.9988, 0.9985,
    0.9988, 0.9992, 0.9992, 0.9992 
]

pruning_stability_no_reg_raw = [
    0.0000, 0.5262, 0.7027, 0.7527, 0.7878, 0.8318, 0.8533, 0.8632, 0.8637, 0.8631, 0.8596, 0.8698, 0.8686,
    0.8712, 0.8708, 0.8691, 0.8651, 0.8693, 0.8632, 0.8546, 0.8626, 0.8543, 0.8548, 0.8556, 0.8487, 0.8547,
    0.8465, 0.8534, 0.8503, 0.8589, 0.8535, 0.8599, 0.8613, 0.8476, 0.8582, 0.8508, 0.8560, 0.8486, 0.8586, 
    0.8455, 0.8603, 0.8605, 0.8599]#, 0.8617, 0.8679, 0.8690, 0.8627, 0.8707, 0.8631]#, 0.8563, 0.8581
#] 

pruning_stability_no_reg_avg = [
    0.0000, 0.1754, 0.4096, 0.6605, 0.7477, 0.7908, 0.8243, 0.8494, 0.8601, 0.8633, 0.8621, 0.8642, 0.8660, 
    0.8699, 0.8702, 0.8704, 0.8683, 0.8678, 0.8658, 0.8623, 0.8601, 0.8571, 0.8572, 0.8549, 0.8530, 0.8530,
    0.8499, 0.8515, 0.8500, 0.8542, 0.8542, 0.8574, 0.8582, 0.8563, 0.8557, 0.8522, 0.8550, 0.8518, 0.8544,
    0.8509, 0.8548, 0.8554, 0.8602]#, 0.8607, 0.8631, 0.8662, 0.8665, 0.8675, 0.8655]#, 0.8634, 0.8592
#] 

# Create a list of training epochs (assuming the data starts from epoch 1)
training_epochs = list(range(1, len(pruning_stability_avg) + 1))

# Create the plot
plt.figure(figsize=(8, 6))

plt.plot(training_epochs, pruning_stability_raw, linestyle='-.', linewidth=1, color='r', alpha=0.5)
plt.plot(training_epochs, pruning_stability_no_reg_raw, linestyle='--', linewidth=1, color='b', alpha=0.7)

# Plot the pruning_stability curve
plt.plot(training_epochs, pruning_stability_avg, linestyle='-', linewidth=2, color='r', label='With Regularization')
# Plot the pruning_stability_no_reg curve
plt.plot(training_epochs, pruning_stability_no_reg_avg, linestyle='-', linewidth=2, color='b', label='Without Regularization')

plt.xlabel('Training Epochs ($t$)', fontsize=18, fontweight='bold')
plt.ylabel('Sub-network Stability Score ($J^t_{avg}$)', fontsize=18, fontweight='bold')

# Adding markers and annotations for structured sparsity regularization
# Iterate through the list to find the index
start_sparsity_epoch = 17
for i in range(3, len(pruning_stability_avg)):
    if (pruning_stability_avg[i] - pruning_stability_avg[i - 3]) <= 0.0001:
        start_sparsity_epoch = i
        break

end_sparsity_threshold = 0.999
end_sparsity_epoch = next(epoch for epoch, value in enumerate(pruning_stability_avg) if value >= end_sparsity_threshold)

# Example annotations
start_annotation = 'Begin Structured\nSparsity Regularization\n($t_{sl-start}$)'
end_annotation = 'End Structured\nSparsity Regularization\n(Stable Pruning Epoch $t^*$)'

# Define the bounding box properties
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="tab:brown", facecolor="snow")

# Create annotations with a bounding box and arrow from the middle
plt.annotate(start_annotation, xy=(start_sparsity_epoch, 0.83),
            xytext=(start_sparsity_epoch + 2, 0.74),  # Adjusted xytext position
            arrowprops=dict(arrowstyle="->", color='tab:brown'),  # Arrow from the middle
            fontsize=17,
            fontname='Times New Roman',
            fontweight='bold',
            bbox=bbox_props)

plt.annotate(end_annotation, xy=(end_sparsity_epoch, 0.75),
            xytext=(end_sparsity_epoch - 2, 0.65),  # Adjusted xytext position for right alignment
            horizontalalignment='right',  # Set the annotation to be right-aligned
            arrowprops=dict(arrowstyle="->", color='tab:brown'),  # Arrow from the middle
            fontsize=17,
            fontname='Times New Roman',
            fontweight='bold',
            bbox=bbox_props)

# Add vertical lines
plt.axvline(x=start_sparsity_epoch, color='tab:brown', linestyle='--', alpha=0.7)
plt.axvline(x=end_sparsity_epoch, color='tab:brown', linestyle='--', alpha=0.7)

plt.scatter(start_sparsity_epoch, pruning_stability_avg[start_sparsity_epoch-1], color='tab:green', marker='*', s=100, linewidths=1.5, zorder=2)
plt.scatter(end_sparsity_epoch, pruning_stability_avg[end_sparsity_epoch-1], color='tab:green', marker='*', s=100, linewidths=1.5,zorder=2 )

# Set x-axis and y-axis limits
plt.xlim(0, len(pruning_stability_avg))  # Set x-axis limits from 1 to the maximum epoch value
plt.ylim(0.6, 1.01)  # Set y-axis limits from 0 to 1 for pruning stability


# Set tick labels' font size and weight
ytick_positions = [0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.999]
plt.yticks(ytick_positions, fontsize=14, fontname='Times New Roman')
plt.xticks(fontsize=14, fontname='Times New Roman')

# Increase the number of tick marks on x and y axes
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(base=5))  # Show tick marks at every 10 units

plt.legend(fontsize=17, loc='upper left')

plt.tight_layout()

# Save the plot to a file (e.g., PDF or PNG) for use in your conference paper
plt.savefig('pruning_stability_plot.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()