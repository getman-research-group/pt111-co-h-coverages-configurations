import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PKL_PATH = BASE_DIR / "FIGURE_S5_PARITY.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S5.png"

# The CSV files do not contain an explicit Training/Testing flag.
# Therefore this script makes a reproducible random split.
TEST_FRACTION = 0.30
RANDOM_SEED = 42

COLORS = {
    "NEG": "#0072B2",
    "NEU": "black",
    "POS": "#E69F00",
}

SYSTEM_ORDER = ["NEG", "NEU", "POS"]

TITLES = {
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive",
}

PANEL_LABELS = [
    ["a)", "b)", "c)"],
    ["d)", "e)", "f)"],
]

X_MIN = -1.9
X_MAX = 1.05
Y_MIN = -1.9
Y_MAX = 1.05

TICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

POINT_SIZE = 30
EDGE_WIDTH = 0.7
LINE_WIDTH = 1.15


# ============================================================
# LOAD PORTABLE PKL
# ============================================================

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

if data.get("format") != "ERG_PARITY_PORTABLE_V1":
    raise ValueError("Unexpected PKL format.")


# ============================================================
# REPRODUCIBLE TRAIN / TEST SPLIT
# ============================================================

def split_indices(n, test_fraction, seed):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_test = max(1, int(round(n * test_fraction)))
    test_idx = sorted(indices[:n_test])
    train_idx = sorted(indices[n_test:])

    return train_idx, test_idx


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15.5, 10.2),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.0, "hspace": 0.0},
)

for col, system in enumerate(SYSTEM_ORDER):
    system_data = data["systems"][system]
    columns = system_data["columns"]

    x_all = np.asarray(columns["DFT_Formation_Energy"], dtype=float)
    y_all = np.asarray(columns["CE_Predicted_Formation_Energy"], dtype=float)

    # Different seed per charge state, but fully reproducible.
    train_idx, test_idx = split_indices(
        len(x_all),
        TEST_FRACTION,
        RANDOM_SEED + col,
    )

    for row, indices in enumerate([train_idx, test_idx]):
        ax = axes[row, col]

        x = x_all[indices]
        y = y_all[indices]

        # Parity line.
        ax.plot(
            [X_MIN, X_MAX],
            [X_MIN, X_MAX],
            linestyle="--",
            linewidth=LINE_WIDTH,
            color="black",
            zorder=1,
        )

        # Scatter points.
        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            facecolor=COLORS[system],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            zorder=2,
        )

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)

        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=6,
            width=1.0,
            labelsize=18,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        # Panel label.
        ax.text(
            0.05,
            0.93,
            PANEL_LABELS[row][col],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=23,
            fontweight="bold",
        )

    axes[0, col].set_title(
        TITLES[system],
        fontsize=22,
        pad=10,
        fontweight="normal",
    )


# ============================================================
# SHARED LABELS
# ============================================================

fig.supxlabel(
    r"$E_{\sigma}^{\mathrm{form,DFT}}$ / eV/site",
    fontsize=24,
    x=0.5,
    y=0.006,
)

fig.supylabel(
    r"$E_{\sigma}^{\mathrm{form,CE}}$ / eV/site",
    fontsize=24,
    x=0.002,
    y=0.50,
)

fig.text(
    0.965,
    0.745,
    "Training",
    rotation=-90,
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
)

fig.text(
    0.965,
    0.325,
    "Testing",
    rotation=-90,
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
)


# ============================================================
# LAYOUT + SAVE
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.955,
    bottom=0.105,
    top=0.935,
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
