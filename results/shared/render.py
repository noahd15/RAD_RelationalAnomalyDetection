import numpy as np
import matplotlib.pyplot as plt

variants = [
    "RAD LLM no BCE",
    "RAD LLM + BCE",
    "GAE no rules",
    "GAE RF sampled",
    "GAE RF features",
]

datasets = {
    "LANL": {
        "AUROC": [(0.996, 0.001), (0.983, 0.031), (0.994, 0.004), (0.985, 0.020), (0.993, 0.008)],
        "AUPRC": [(0.659, 0.053), (0.552, 0.025), (0.370, 0.069), (0.530, 0.051), (0.423, 0.123)],
        "P@k":   [(0.601, 0.012), (0.572, 0.009), (0.484, 0.097), (0.560, 0.035), (0.509, 0.114)],
    },
    "Amazon": {
        "AUROC": [(0.693, 0.010), (0.691, 0.007), (0.682, 0.005), (0.694, 0.006), (0.589, 0.030)],
        "AUPRC": [(0.043, 0.003), (0.031, 0.005), (0.028, 0.002), (0.031, 0.001), (0.025, 0.002)],
        "P@k":   [(0.050, 0.007), (0.042, 0.006), (0.038, 0.007), (0.044, 0.006), (0.047, 0.011)],
    },
    "H&M": {
        "AUROC": [(0.676, 0.018), (0.685, 0.017), (0.654, 0.023), (0.657, 0.027), (0.560, 0.040)],
        "AUPRC": [(0.031, 0.003), (0.032, 0.004), (0.031, 0.004), (0.032, 0.004), (0.022, 0.003)],
        "P@k":   [(0.046, 0.008), (0.042, 0.014), (0.045, 0.013), (0.045, 0.010), (0.035, 0.009)],
    },
}

metrics = ["AUROC", "AUPRC", "P@k"]

fig, axes = plt.subplots(
    2, 3,
    figsize=(11, 4.2),
    sharex="col",
    gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.06, "wspace": 0.25}
)

bar_width = 0.14
x = np.arange(len(metrics))
offsets = (np.arange(len(variants)) - 2) * bar_width

for col, (dataset_name, values) in enumerate(datasets.items()):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    for i, variant in enumerate(variants):
        means = [values[m][i][0] for m in metrics]
        stds = [values[m][i][1] for m in metrics]

        ax_top.bar(
            x + offsets[i],
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=variant,
            edgecolor="black",
            linewidth=0.4,
        )
        ax_bot.bar(
            x + offsets[i],
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            edgecolor="black",
            linewidth=0.4,
        )

    # Dataset-specific broken-axis ranges
    if dataset_name == "LANL":
        ax_top.set_ylim(0.90, 1.02)
        ax_bot.set_ylim(0.25, 0.75)
    else:
        ax_top.set_ylim(0.52, 0.73)
        ax_bot.set_ylim(0.00, 0.075)

    ax_top.set_title(dataset_name, fontweight="bold")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(metrics)

    # Hide touching spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.tick_params(top=False)

    # Diagonal break marks
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_bot.transAxes, color="k", clip_on=False, linewidth=0.8)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

axes[0, 0].set_ylabel("Score")
axes[1, 0].set_ylabel("Score")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("combined_ablation_barplots_broken_axis.pdf", bbox_inches="tight")
plt.savefig("combined_ablation_barplots_broken_axis.png", dpi=300, bbox_inches="tight")
plt.show()