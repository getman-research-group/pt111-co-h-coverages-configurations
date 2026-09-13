import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S1_CO_BE_CHARGE.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S1.png"


# ============================================================
# FONT SETTINGS
# ============================================================

FONT_FAMILY = "Dejavu Sans"     # e.g. "Arial", "Times New Roman", "DejaVu Sans"

AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18

# Optional global font settings
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": TICK_LABEL_FONTSIZE,
})


# ============================================================
# PLOT SETTINGS
# ============================================================

FIGSIZE = (10, 7)

LINEWIDTH = 2.5
MARKERSIZE = 8

HIGHLIGHT_SIZE = 220
HIGHLIGHT_COLOR = "red"

X_LABEL = r"Charge per Surface Pt Atom / h$^+$"
Y_LABEL = r"CO* Binding Energy / eV"


# ============================================================
# LOAD DATA
# ============================================================

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

x = data["charge_per_surface_pt"]
y = data["co_binding_energy_eV"]
highlight = data["highlight_indices"]


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=FIGSIZE)


# ============================================================
# MAIN LINE
# ============================================================

ax.plot(
    x,
    y,
    "-o",
    linewidth=LINEWIDTH,
    markersize=MARKERSIZE
)


# ============================================================
# HIGHLIGHTED POINTS
# ============================================================

ax.scatter(
    [x[i] for i in highlight],
    [y[i] for i in highlight],
    s=HIGHLIGHT_SIZE,
    color=HIGHLIGHT_COLOR,
    zorder=3
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    X_LABEL,
    fontsize=AXIS_LABEL_FONTSIZE
)

ax.set_ylabel(
    Y_LABEL,
    fontsize=AXIS_LABEL_FONTSIZE
)


# ============================================================
# AXIS LIMITS
# ============================================================

ax.set_xlim(-0.183, 0.183)
ax.set_ylim(-2.92, -1.18)


# ============================================================
# TICKS
# ============================================================

ax.set_xticks([
    -0.15,
    -0.10,
    -0.05,
    0.00,
    0.05,
    0.10,
    0.15
])

ax.set_yticks([
    -1.2,
    -1.4,
    -1.6,
    -1.8,
    -2.0,
    -2.2,
    -2.4,
    -2.6,
    -2.8
])

ax.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_LABEL_FONTSIZE
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()