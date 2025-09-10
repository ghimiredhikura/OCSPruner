import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# Increase font sizes for different text elements
# Set font to Times New Roman
rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 16})  # Change the default font size

def plot_right_subplot(ax):
    # Data
    data = [
        #("RCP", 47.99, 93.94, 1),
        ("PGMPF", 34.00, 93.60, 'Y'),
        ("CPGCN", 26.93, 93.08, 'Y'),
        ("LAASP", 39.54, 93.79, 'N'),
        ("OTOv1", 26.80, 93.50, 'N'),
        ("DLRFC", 23.05, 93.64, 'Y'),
        ("OTOv2", 23.70, 93.20, 'N'),
        ("DCFF", 23.13, 93.47, 'N'),
    ]

    # Data
    OCSP_data = [
        ("OCSPPruner (Ours)", 39.98, 93.98, "N"),
        ("OCSPPruner (Ours)", 26.01, 93.88, 'N'),
        ("OCSPPruner (Ours)", 21.22, 93.76, 'N'),
    ]

    # Extract x and y values
    x = [point[1] for point in data]
    y = [point[2] for point in data]

    distinct_colors = ["#00008B", "#008B8E", "green", "#0077B2", "blue", "#4B0082", "#000000", "#CC2200"]

    data_colors = distinct_colors[0:len(data)]
    essp_gsl_color = distinct_colors[0]
    #fig, ax = plt.subplots(dpi=200)
    ax.grid(True, linestyle='dotted', color=(0.9, 0.9, 0.9), zorder=-1)  # Customize linestyle, color (light gray), and set zorder to 0

    # Add Training Speed Up legend
    speed_up_factors = [0.50, 1.00, 1.50]
    speed_up_sizes = [75 * (factor ** 2) for factor in speed_up_factors]  # Exponential scaling
    legend_labels = [f"{factor} x" for factor in speed_up_factors]

    for i, size in enumerate(speed_up_sizes):
        ax.scatter(0, 75 - i, color='darkgray', s=size, label=legend_labels[i])

    # Increase font size for annotations
    annotation_fontsize = 10
    for annotation in ax.texts:
        annotation.set_fontsize(annotation_fontsize)

    # Set blue color for "ours" method
    #blue_color = "#8B0000"
    blue_color = 'red'

    for i, point in enumerate(OCSP_data):
        marker = '*' if point[3] == 'N' else 'o'
        ax.scatter(point[1], point[2], color='lightgray', marker='o', s=85, label="OCSP (Ours)")
        if marker == '*':
            ax.scatter(point[1], point[2], c=blue_color, marker=marker, s=85)

    # Plot data points with assigned colors and sizes
    for i, point in enumerate(data):
        marker = '*' if point[3] == 'N' else 'o'
        if marker == '*':
            ax.scatter(point[1], point[2], c='lightgray', marker='o', s=85)
            ax.scatter(point[1], point[2], c=data_colors[i], marker=marker, s=85)
        else:
            ax.scatter(point[1], point[2], c=data_colors[i], marker=marker, s=85)

    # Plot dotted line for OCSP (Ours)
    essp_gsl_x = [point[1] for point in OCSP_data]
    essp_gsl_y = [point[2] for point in OCSP_data]
    ax.plot(essp_gsl_x, essp_gsl_y, '--', color=essp_gsl_color)

    # Increase font size for axis labels
    ax.set_xlabel("FLOPs (%)", fontsize=16)
    ax.set_ylabel("Top-1 Acc. (%)", fontsize=16)

    # Set x-axis limits
    ax.set_ylim(93.05, 94.05)
    ax.set_xlim(20.5, 40.50)
    # Set white background6
    fig.set_facecolor('white')
    ax.plot(essp_gsl_x, essp_gsl_y, '--', color=blue_color)  # Dotted line color for "ours" method

    txt = "OCSPruner (Ours)"
    label_width = len(txt) * 10

    ax.annotate(txt, (39.94, 93.98), xytext=(-120, 8), textcoords='offset points', color=blue_color)
    ax.annotate(txt, (26.01, 93.88), xytext=(-80, 10), textcoords='offset points', color=blue_color)
    ax.annotate(txt, (21.22, 93.76), xytext=(-3, -17), textcoords='offset points', color=blue_color)

    #txt = "RCP [33]"
    #label_width = len(txt) * 10
    #ax.annotate(txt, (47.99, 93.94), xytext=(10, -4), textcoords='offset points', color=data_colors[0])
    color_index = 0
    txt = "PGMPF [4]"
    ax.annotate(txt, (34.00, 93.60), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "CPGCN [25]"
    ax.annotate(txt, (26.93, 93.08), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "LAASP [17]"
    ax.annotate(txt, (39.54, 93.79), xytext=(-90, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "OTOv1 [5]"
    ax.annotate(txt, (26.80, 93.50), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "DLRFC [24]"
    ax.annotate(txt, (23.05, 93.64), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "OTOv2 [6]"
    ax.annotate(txt, (23.70, 93.20), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "DCFF [37]"
    ax.annotate(txt, (23.13, 93.47), xytext=(-40, -20), textcoords='offset points', color=data_colors[color_index])

    plt.tight_layout(pad=0.3)

    # Create legend for markers
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Yes', markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='No', markerfacecolor='black', markersize=15),
    ]
    ax.legend(handles=legend_elements, loc='lower right', title="Use pre-trained?", fontsize=14)

    ax.set_title("VGG16 on CIFAR-10")

# Create subplots
fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), dpi=100)

# Plot left subplot
plot_right_subplot(ax)

# Display the graph
plt.savefig('cifar_prune_compare.pdf', dpi=600)
plt.show()