import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

# Increase font sizes for different text elements
# Set font to Times New Roman
rc('font', family='Times New Roman')
# Update font size globally
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

def plot_imagenet(ax):
    # Data
    data = [
        ("CLR-RNF (TNNLS 2022)",    59.6,   74.85,  0.66,   'Y'),
        ("FPGM (CVPR 2019)",        57.8,   75.03,  1.0,    'N'),
        ("LAASP (IVC 2023)",        57.7,   75.85,  1.04,   'N'),
        ("HRank (CVPR 2020)",       56.18,  74.98,  0.5,    'Y'),
        ("Tayler (CVPR 2019)",      54.95,  74.5,   0.5,    'Y'),
        ("White-Box (TNNLS 2022)",  54.34,  75.32,  0.66,   'Y'),
        ("DepGraph (CVPR 2023)",    51.82,  75.83,  0.5,    'Y'),
        ("PGMPF (AAAI 2022)",       46.5,   75.11,  0.66,   'Y'),
        ("GReg-2 (ICLR 2021)",      43.33,  75.36,  0.66,   'Y'),
        ("PaT (CVPR 2022)",         41.3,   74.85,  1,      'N'),
        ("CC (CVPR 2021)",          37.32,  74.54,  0.66,   'Y'),
        ("OTOv1 (NIPS 2021)",       34.5,   74.70,   1,      'N'),
        ("GNN-RL (ICML 2022)",      47.00,  74.28,  0.66,   'Y'),
        ("DTP (ICCV 2023)",         32.68,  74.26,  0.66,   'Y'),
        ("NuSPM (WACV 2024)",       41.3,   75.25,  1.0,    'N'),
    ]

    # Data
    OCSP_data = [
        ("OCSPPruner (Ours)", 56.99, 76, 1.28, 'N'),
        ("OCSPPruner (Ours)", 51.74, 75.95, 1.30, 'N'),
        ("OCSPPruner (Ours)", 42.98, 75.49, 1.38, 'N'),
        ("OCSPPruner (Ours)", 33.05, 74.724, 1.41, 'N'),
    ]

    marker_scale = 175
    # Separate OCSP (Ours) data for dotted line
    essp_gsl_sizes = [(point[3] ** 2) * marker_scale for point in OCSP_data]  # Extract sizes

    # Predefined distinct colors
    distinct_colors = [
        "#9C7ED7", "#00008B", "#008B8E", "#370044", "#00000F",
        "#1A237E", "#0077B2", "#004D40", "#007D6C",
        "blue", "#692079", "#000000", "#6A007F", "green",
        "#03A9F4", "#500050", "#246628", "#A34C9E", "#A03037",
        "#A03037", "#A03037",
        "#A03037", "#A03037",
        "#A03037", "#A03037",
        "#A03037", "#A03037",
        "#A03037", "#A03037",
        "#A03037", "#A03037",
    ]

    # Assign colors to data points
    #data_colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(data))]
    data_colors = distinct_colors[0:len(data)]
    # Find color for OCSP (Ours) method
    essp_gsl_color = distinct_colors[0]

    # Make sure grid goes behind everything
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='dotted', color=(0.9, 0.9, 0.9), zorder=-1)

    # Add Training Speed Up legend
    speed_up_factors = [0.50, 1.00, 1.50]
    speed_up_sizes = [marker_scale * (factor ** 2) for factor in speed_up_factors]  # Exponential scaling
    legend_labels = [f"{factor} x" for factor in speed_up_factors]

    for i, size in enumerate(speed_up_sizes):
        ax.scatter(0, marker_scale - i, color='darkgray', s=size, label=legend_labels[i])

    # Increase font size for annotations
    annotation_fontsize = 10
    for annotation in ax.texts:
        annotation.set_fontsize(annotation_fontsize)

    # Increase font size for legend
    legend_speed_up = ax.legend(loc='upper left', title="Training Speed Up", fontsize=16)
    plt.setp(legend_speed_up.get_title(), fontsize='16')  # Set legend title font size

    # Set blue color for "ours" method
    blue_color = 'red' #"#8B0000"
    for i, point in enumerate(OCSP_data):
        marker = '*' if point[4] == 'N' else 'o'
        ax.scatter(point[1], point[2], c=(0.6, 0.6, 0.6), marker='o', s=essp_gsl_sizes[i], label="OCSP (Ours)")
        if marker == '*':
            ax.scatter(point[1], point[2], c=blue_color, marker=marker, s=essp_gsl_sizes[i])

    # Plot data points with assigned colors and sizes
    for i, point in enumerate(data):
        marker = '*' if point[4] == 'N' else 'o'
        if marker == '*':
            #ax.scatter(point[1], point[2], c='lightgray', marker='o', s=[(point[3] ** 2) * marker_scale])
            ax.scatter(point[1], point[2], c=(0.6, 0.6, 0.6), marker='o', s=[(point[3] ** 2) * marker_scale])
            ax.scatter(point[1], point[2], c=data_colors[i], marker=marker, s=[(point[3] ** 2) * marker_scale])
        else:
            ax.scatter(point[1], point[2], c=data_colors[i], marker=marker, s=[(point[3] ** 2) * marker_scale])

    # Plot dotted line for OCSP (Ours)
    essp_gsl_x = [point[1] for point in OCSP_data]
    essp_gsl_y = [point[2] for point in OCSP_data]
    ax.plot(essp_gsl_x, essp_gsl_y, '--', color=essp_gsl_color)

    # Increase font size for axis labels
    ax.set_xlabel("FLOPs (%)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Top-1 Acc. (%)", fontsize=14, fontweight='bold')

    # Set x-axis limits
    ax.set_ylim(74.10, 76.15)
    ax.set_xlim(31, 66.50)
    # Set white background6
    fig.set_facecolor('white')
    from matplotlib.ticker import FuncFormatter

    # Generate 8 evenly spaced ticks
    yticks = np.linspace(74.10, 76.15, 8)
    ax.set_yticks(yticks)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))

    ax.tick_params(axis='both', which='major', labelsize=14)  # major ticks

    ax.plot(essp_gsl_x, essp_gsl_y, '--', color=blue_color)  # Dotted line color for "ours" method

    txt = "OCSPruner (Ours)"
    label_width = len(txt) * 10
    ax.annotate(txt, (32.96, 74.69), xytext=(-25, 10), textcoords='offset points', color=blue_color, rotation=49)
    ax.annotate(txt, (42.98, 75.45), xytext=(-50, -8), textcoords='offset points', color=blue_color, rotation=41)
    ax.annotate(txt, (51.74, 75.95), xytext=(-90, 15), textcoords='offset points', color=blue_color)
    ax.annotate(txt, (56.99, 76.02), xytext=(56.99-label_width+90, 10), textcoords='offset points', color=blue_color)

    color_index = 0
    txt = "CLR-RNF [37]"
    label_width = len(txt) * 10
    ax.annotate(txt, (59.6, 74.85), xytext=(-30, -20), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "FPGM [23]"
    ax.annotate(txt, (57.8, 75.03), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "LAASP [18]"
    ax.annotate(txt, (57.7, 75.85), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "HRank [36]"
    ax.annotate(txt, (56.18, 74.98), xytext=(-55, -17), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "Tayler [43]"
    ax.annotate(txt, (54.95, 74.5), xytext=(-82, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "White-Box [62]"
    ax.annotate(txt, (54.34, 75.32), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "DepGraph [13]"
    ax.annotate(txt, (51.82, 75.83), xytext=(-25, -15), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "PGMPF [4]"
    ax.annotate(txt, (46.5, 75.11), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "GReg-2 [54]"
    ax.annotate(txt, (43.33, 75.36), xytext=(10, -6), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "PaT [50]"
    ax.annotate(txt, (41.3, 74.85), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "CC [33]"
    ax.annotate(txt, (37.32, 74.54), xytext=(-10, -17), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "OTOv1 [5]"
    ax.annotate(txt, (34.5, 74.70), xytext=(5, 5), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "GNN-RL [60]"
    ax.annotate(txt, (47.00,  74.28), xytext=(-45, -18), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "DTP [35]"
    ax.annotate(txt, (32.68,  74.26), xytext=(10, -4), textcoords='offset points', color=data_colors[color_index])

    color_index += 1
    txt = "NuSPM [29]"
    ax.annotate(txt, (41.30,  75.25), xytext=(10, -6), textcoords='offset points', color=data_colors[color_index])

    plt.tight_layout(pad=0.2)

    # Custom handler to create composite legend markers
    class CompositeLegendHandler:
        def __init__(self, marker1, marker2, dx=16, dy=5):
            self.marker1 = marker1
            self.marker2 = marker2
            self.dx = dx
            self.dy = dy

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            # Get the current position of the handlebox
            x0, y0 = handlebox.xdescent, handlebox.ydescent

            # Adjust positions of the markers
            self.marker1.set_xdata([x0 + self.dx])
            self.marker1.set_ydata([y0 + self.dy])
            self.marker2.set_xdata([x0 + self.dx])
            self.marker2.set_ydata([y0 + self.dy])

            handlebox.add_artist(self.marker1)
            handlebox.add_artist(self.marker2)

            return [self.marker1, self.marker2]

    # Create the transparent circle and star markers for the legend
    circle_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.6, 0.6, 0.6), markeredgecolor=(0.6, 0.6, 0.6), markersize=13)
    star_marker = Line2D([0], [0], marker='*', color='black', markerfacecolor='black', markersize=13)

    # Create the composite legend entry
    composite_marker = (circle_marker, star_marker)

    # Create legend for markers with the composite entry
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Yes', markerfacecolor='black', markersize=13),
        composite_marker  # Composite marker for 'No'
    ]

    legend_labels = ['Yes', 'No']

    legend_pretrained = ax.legend(
        legend_elements, 
        legend_labels, 
        loc='lower right', 
        title="Use pre-trained?", 
        fontsize=16, 
        handler_map={composite_marker: CompositeLegendHandler(circle_marker, star_marker)}
    )

    # Add both legends to the plot
    ax.add_artist(legend_speed_up)
    ax.add_artist(legend_pretrained)
    #ax.set_title("ResNet50 on ImageNet")

# Create subplots
fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), dpi=100)

# Plot left subplot
plot_imagenet(ax)

# Display the graph
plt.savefig('imagenet_prune_compare.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
plt.savefig('imagenet_prune_compare.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()