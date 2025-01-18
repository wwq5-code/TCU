import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def radar_factory(num_vars):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    return np.concatenate((theta, [theta[0]]))

def plot_radar(ax, theta, values_list, labels, colors, methods, title, max_vals):
    """
    Plot multiple data series on a single radar (spider) chart.
    (No legend here; we'll create a single shared legend at the figure level.)
    """
    for values, color, method_name in zip(values_list, colors, methods):
        vals = np.concatenate((values, [values[0]]))
        line = ax.plot(theta, vals, color=color, linewidth=3, label=method_name)
        # ax.fill(theta, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(theta[:-1]), labels)
    ax.set_rlabel_position(0)

    ax.set_ylim(0, 105)
    ax.set_title(title, y=1.03, fontsize=13)

    # Hide radial tick labels if desired:
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # repeat the first angle to close the plot

    # Add labels
    for i, category in enumerate(categories):
        angle = angles[i]
        if angle > np.pi / 2 and angle < 3 * np.pi / 2:
            ha = 'right'  # Flip text alignment
        else:
            ha = 'left'
        ax.text(angle, max(values) * 1.21, category, fontsize=12, ha=ha, va='center', rotation=i*72-90, color='black')
        ax.text(angle, max(values) * 1.15, max_vals[i], fontsize=10, ha=ha, va='center', rotation=i * 72 - 90, color='red')


# -----------------------------
# Example data
# -----------------------------

categories = [
    r"$\mathbf{RTE(s)}$",
    r"$\mathbf{UA(\%)}$",
    r"$\mathbf{RA(\%)}$",
    r"$\mathbf{TA(\%)}$",
    r"$\mathbf{MIA(\%)}$"
]
N = len(categories)

# "Class-wise forgetting"
retrain_values_cw     = [4,     100 ,  99.49, 99.23,   63.99]
finetuning_values_cw  = [100,   66,  99.37, 98.25,    58.00]
l1_sparse_values_cw   = [88,  66,  99.40, 98.95,    56.99]

# "Random data forgetting"
retrain_values_random   = [4,   9.9*10,  94.66, 89.32,   94.98]
finetuning_values_random= [100,   9.9*10,  87.82, 84.68,   64.99]
l1_sparse_values_random = [95,  9.9*10,  94.18, 89.02,    69.99]

# Methods and colors
methods = ['Retrain', 'GA', 'MBU']
colors  = ['green', 'red', 'blue']

# -----------------------------
# Create figure and subplots
# -----------------------------
fig = plt.figure(figsize=(11, 5))
theta = radar_factory(N)

max_vals_random = [0.183, 3.00, 100.00, 100.00, 100.00]

ax1 = plt.subplot(1, 2, 1, projection='polar')
plot_radar(
    ax=ax1,
    theta=theta,
    values_list=[retrain_values_cw, finetuning_values_cw, l1_sparse_values_cw],
    labels=categories,
    colors=colors,
    methods=methods,
    title="(a) On MNIST",
    max_vals=max_vals_random
)


max_vals_random = [1.52, 10.00, 100.00, 100.00, 100.00]

ax2 = plt.subplot(1, 2, 2, projection='polar')
plot_radar(
    ax=ax2,
    theta=theta,
    values_list=[retrain_values_random, finetuning_values_random, l1_sparse_values_random],
    labels=categories,
    colors=colors,
    methods=methods,
    title="(b) On CIFAR10",
    max_vals=max_vals_random
)

# -----------------------------
# Create a single legend for the entire figure
# -----------------------------
# Collect handles/labels from the first subplot (or second – they share the same 3 methods).
handles, labels = ax1.get_legend_handles_labels()

fig.legend(
    handles, labels,
    loc='lower center',         # place it at the bottom center
    bbox_to_anchor=(0.5, 0.0),# adjust vertical spacing as needed
    ncol=len(methods),          # one column per method
    fontsize=13,
    frameon=True
)

plt.tight_layout()
# Because we put the legend below the figure, we might need
# a bit more space at the bottom:
plt.subplots_adjust(bottom=0.15,top=0.912)

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('radar_image.pdf', format='pdf', dpi=200)


plt.show()
