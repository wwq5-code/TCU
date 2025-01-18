import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def radar_factory(num_vars):
    """
    Create the angles for a radar plot, returning an array `theta`
    that goes from 0 to 2*pi, plus a repeat of the first angle.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    return np.concatenate((theta, [theta[0]]))

def plot_radar_with_scales(
    ax, theta, raw_values_list, max_vals, categories, colors, methods, title
):
    """
    Plot multiple data series on a single radar chart where
    each axis can have its own maximum.

    - raw_values_list: list of lists, each sublist is raw data for one method.
    - max_vals: the per-axis maxima (same length as categories).
    - categories: axis labels (strings).
    """
    # 1) Normalize each raw data value by that axis's max
    #    so that all values end up in [0..1].
    normed_values_list = []
    for raw_values in raw_values_list:
        normed = [val / mv for val, mv in zip(raw_values, max_vals)]
        normed_values_list.append(normed)

    # 2) Plot each method's polygon
    for normed_vals, color, method_name in zip(normed_values_list, colors, methods):
        # Close the polygon by repeating the first value
        vals_closed = np.concatenate((normed_vals, [normed_vals[0]]))
        ax.plot(theta, vals_closed, color=color, linewidth=3, label=method_name)
        ax.fill(theta, vals_closed, alpha=0.1, color=color)

    # 3) Manually draw the radial grid (0..1 circles) and spokes
    #    We'll just do a few circles (25%, 50%, 75%, 100%).
    #    If you prefer a different style, adjust here.
    n_points = 300
    angles_full = np.linspace(0, 2*np.pi, n_points)
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_full, [r]*n_points, color='gray', ls='--', alpha=0.5, lw=0.8)
    # spokes (one for each category)
    for angle in theta[:-1]:
        ax.plot([angle, angle], [0, 1], color='gray', lw=0.8)

    # 4) Turn off the default numeric angle ticks (since we'll place custom text)
    ax.set_xticks([])
    ax.set_yticks([])

    # 5) Place the per-axis label + max value
    #    We'll position the category label (in black) a bit inward
    #    and the numeric max (in red) just beyond radius=1.
    for angle, cat_label, mv in zip(theta[:-1], categories, max_vals):
        # numeric max in red
        ax.text(angle, 1.05, f"{mv:.2f}", color='red',
                ha='center', va='center', fontsize=10, fontweight='bold')
        # category label in black, slightly inward
        ax.text(angle, 0.88, cat_label, color='black',
                ha='center', va='center', fontsize=12)

    # 6) Set radial limits [0..1] after normalization
    ax.set_ylim(0, 1)

    # 7) Title
    ax.set_title(title, y=1.03, fontsize=11)

# -------------------------------------------------------
# Example usage (building on your original code structure)
# -------------------------------------------------------
fig = plt.figure(figsize=(10, 5))

# We have 5 categories (in the order you gave them):
categories = [
    r"    $\mathbf{RTE(s)}$",
    r"$\mathbf{UA(\%)}$",
    r"$\mathbf{RA(\%)}$",
    r"$\mathbf{TA(\%)}$",
    r"$\mathbf{MIA(\%)}$"
]
N = len(categories)
theta = radar_factory(N)  # angles array

# Example per-axis maxima for subplot (a)
# (You'd replace these with your actual per-axis limits.)
# e.g. RTE(s) ~ 2.5, UA(%)=100, RA(%)=100, TA(%)~94.83, MIA(%)=100
max_vals_cw = [2.5, 100.0, 100.0, 94.83, 100.0]

# Example data for "Class-wise forgetting"
retrain_values_cw     = [2.3,  1.0,   90.0, 90.0,  80.0]  # raw data
finetuning_values_cw  = [2.5, 10.0,   95.0, 80.0,  75.0]
l1_sparse_values_cw   = [2.1, 70.0,   85.0, 94.0,  100.0]

# Subplot (a)
ax1 = plt.subplot(1, 2, 1, projection='polar')
plot_radar_with_scales(
    ax=ax1,
    theta=theta,
    raw_values_list=[
        retrain_values_cw,
        finetuning_values_cw,
        l1_sparse_values_cw
    ],
    max_vals=max_vals_cw,
    categories=categories,
    colors=['green', 'red', 'blue'],
    methods=['Retrain', 'GA', 'MBU'],
    title="(a) On MNIST"
)

# Example per-axis maxima for subplot (b)
max_vals_random = [2.5, 100.0, 100.0, 94.83, 100.0]

# Example data for "Random data forgetting" or "CIFAR10"
retrain_values_random   = [1.8,   0.5,   70.0,   50.0,  20.0]
finetuning_values_random= [2.4,  10.0,   80.0,   40.0,  30.0]
l1_sparse_values_random = [2.5,  90.0,   90.0,   80.0, 100.0]

# Subplot (b)
ax2 = plt.subplot(1, 2, 2, projection='polar')
plot_radar_with_scales(
    ax=ax2,
    theta=theta,
    raw_values_list=[
        retrain_values_random,
        finetuning_values_random,
        l1_sparse_values_random
    ],
    max_vals=max_vals_random,
    categories=categories,
    colors=['green', 'red', 'blue'],
    methods=['Retrain', 'GA', 'MBU'],
    title="(b) On CIFAR10"
)

# Single legend at the bottom
handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=13,
    frameon=True
)

plt.tight_layout()
# Adjust the bottom so it doesn't cut off the legend
plt.subplots_adjust(bottom=0.15, top=0.912)
plt.show()
