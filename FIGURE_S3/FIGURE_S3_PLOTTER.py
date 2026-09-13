import pickle
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S3_WINDOW_SENSITIVITY.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S3.png"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

x = data["window_width_ML"]
roughness = data["roughness"]
bootstrap = data["bootstrap_sd_ML"]
selected = data["selected_window_width_ML"]

colors = {
    "NEG": "#0072B2",
    "NEU": "black",
    "POS": "#E69F00",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for state in ["NEG", "NEU", "POS"]:
    axes[0].plot(
        x,
        roughness[state],
        "-o",
        color=colors[state],
        markersize=5,
        linewidth=1.5,
        label=state
    )

    axes[1].plot(
        x,
        bootstrap[state],
        "-o",
        color=colors[state],
        markersize=5,
        linewidth=1.5,
        label=state
    )

for ax in axes:
    ax.axvline(
        selected,
        color="gray",
        linestyle="--",
        linewidth=1.0
    )

    ax.set_xlabel("Window width / ML")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["0.05", "0.075", "0.1", "0.125", "0.15", "0.175", "0.2"],
        rotation=45,
        ha="right"
    )
    ax.tick_params(direction="in", top=True, right=True)

axes[0].set_ylabel("Normalized derivative\nroughness")
axes[1].set_ylabel("Bootstrap SD of\ninterval center / ML")

axes[0].text(
    0.96, 0.95, "(a)",
    transform=axes[0].transAxes,
    ha="right", va="top",
    fontsize=20, fontweight="bold"
)

axes[1].text(
    0.96, 0.95, "(b)",
    transform=axes[1].transAxes,
    ha="right", va="top",
    fontsize=20, fontweight="bold"
)

# ============================================================
# LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

# First arrange the two panels
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Find the horizontal center of the actual two-panel plot area
left = axes[0].get_position().x0
right = axes[1].get_position().x1
legend_center_x = (left + right) / 2

# Position legend just above the panels
top = max(ax.get_position().y1 for ax in axes)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(legend_center_x, top - 0.02)
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()