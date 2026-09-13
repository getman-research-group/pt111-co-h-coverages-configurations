import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S7_FUNCTIONALS.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S7.png"

# ============================================================
# Load data
# ============================================================

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

labels = data["labels"]
energies = data["binding_energy_eV"]
exp_low, exp_high = data["experimental_range_eV"]

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(16, 8))

ax.bar(
    labels,
    energies,
    color="red",
    edgecolor="black",
    linewidth=1.0
)

ax.axhspan(
    exp_low,
    exp_high,
    color="yellow",
    alpha=0.5,
    label="Experimental Range"
)

ax.set_ylabel("CO* Binding Energy / eV", fontsize=20)
ax.set_ylim(-2.0, 0.0)

ax.xaxis.tick_top()
ax.tick_params(
    axis="x",
    rotation=45,
    labelsize=20,
    labeltop=True,
    labelbottom=False,
)
ax.tick_params(axis="y", labelsize=20)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("left")

ax.legend(loc="lower right", fontsize=20)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
