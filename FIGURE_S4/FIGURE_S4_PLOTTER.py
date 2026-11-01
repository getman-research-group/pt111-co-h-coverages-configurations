import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S4_GCMC_EQUILIBRATION.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S4.png"


# ============================================================
# Plot settings
# ============================================================

FIGSIZE = (16, 8.5)
DPI = 300

ENERGY_COLOR = "blue"
CO_COLOR = "green"
H_COLOR = "red"

MAIN_LINEWIDTH = 0.8
INSET_LINEWIDTH = 0.9

# Inset locations in main-axis-relative coordinates:
# [left, bottom, width, height]
INSET_POSITIONS = [
    [0.29, 0.49, 0.17, 0.24],
    [0.535, 0.49, 0.17, 0.24],
    [0.78, 0.49, 0.17, 0.24],
]


# ============================================================
# Load portable PKL
# ============================================================

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

cols = data["columns"]
n_sites = float(data.get("n_sites", 900))

step = np.asarray(cols["Step"], dtype=float)
phase = np.asarray(cols["Phase"], dtype=str)
co_cov = np.asarray(cols["CO_Coverage"], dtype=float)
h_cov = np.asarray(cols["H_Coverage"], dtype=float)
total_energy = np.asarray(cols["Total_Energy_eV"], dtype=float)

energy_per_site = total_energy / n_sites


# ============================================================
# Locate convergence window
# ============================================================

conv_mask = phase == "convergence_window"

if not np.any(conv_mask):
    raise ValueError("No rows with Phase == 'convergence_window' were found.")

conv_steps = step[conv_mask]
conv_energy = energy_per_site[conv_mask]
conv_co = co_cov[conv_mask]
conv_h = h_cov[conv_mask]

conv_start = conv_steps.min()
conv_end = conv_steps.max()

sd_co = np.std(conv_co, ddof=1)
sd_h = np.std(conv_h, ddof=1)
sd_energy = np.std(conv_energy, ddof=1)


# ============================================================
# Figure
# ============================================================

fig, ax1 = plt.subplots(figsize=FIGSIZE)

ax2 = ax1.twinx()

# Main traces
line_energy, = ax1.plot(
    step,
    energy_per_site,
    color=ENERGY_COLOR,
    lw=MAIN_LINEWIDTH,
    label=r"$E^{\mathrm{form,CE}}$"
)

line_co, = ax2.plot(
    step,
    co_cov,
    color=CO_COLOR,
    lw=MAIN_LINEWIDTH,
    label=r"$\theta_{\mathrm{CO}*}$"
)

line_h, = ax2.plot(
    step,
    h_cov,
    color=H_COLOR,
    lw=MAIN_LINEWIDTH,
    label=r"$\theta_{\mathrm{H}*}$"
)

# Convergence boundary
ax1.axvline(
    conv_end,
    color="black",
    linestyle="--",
    linewidth=1.0
)


# ============================================================
# Main axis formatting
# ============================================================

ax1.set_xlabel("Number of Moves", fontsize=24)
ax1.set_ylabel(
    r"$E^{\mathrm{form,CE}}$ / eV/site",
    fontsize=24
)

ax2.set_ylabel(
    r"$\theta_{\mathrm{CO}*}, \theta_{\mathrm{H}*}$ / ML",
    fontsize=24
)

ax1.set_xlim(step.min() - 600, step.max() + 600)
ax1.set_ylim(-5.8, 0.5)
ax2.set_ylim(-0.025, 0.49)

ax1.tick_params(axis="both", labelsize=22)
ax2.tick_params(axis="y", labelsize=22)

# Legend
handles = [line_energy, line_co, line_h]
labels = [h.get_label() for h in handles]

ax1.legend(
    handles,
    labels,
    loc="center right",
    bbox_to_anchor=(1.0, 0.16),
    fontsize=21,
    frameon=True
)


# ============================================================
# Insets: separate convergence windows, matching Figure S5
# ============================================================

production_mask = phase == "production"


def add_convergence_inset(values, mean, sd, label, color, unit, bounds):
    inset = ax1.inset_axes(bounds)
    inset.plot(conv_steps, values, color=color, linewidth=INSET_LINEWIDTH)
    inset.set_xlim(conv_start, conv_end)

    value_min, value_max = values.min(), values.max()
    value_range = value_max - value_min
    if value_range == 0:
        value_range = 1.0 / n_sites
    inset.set_ylim(value_min - 0.10 * value_range, value_max + 0.10 * value_range)
    inset.set_xticks([conv_start, conv_end])
    inset.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.4f" if unit == "ML" else "%.3f"))
    inset.tick_params(axis="both", labelsize=12, pad=2, length=3)
    inset.grid(alpha=0.16)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color(color)
    inset.set_title(
        f"{label} | {conv_end:,.0f} moves\nSD = {sd:.5f} {unit}",
        fontsize=13, color=color, pad=7,
    )
    ax1.text(
        bounds[0] + bounds[2] / 2, 0.40,
        f"Mean: {mean:.5f} {unit}",
        ha="center", va="bottom", color=color, fontsize=14,
        transform=ax1.transAxes,
    )


for values, full_values, sd, label, color, unit, bounds in zip(
    [conv_co, conv_h, conv_energy],
    [co_cov, h_cov, energy_per_site],
    [sd_co, sd_h, sd_energy],
    [r"$\theta_{\mathrm{CO}*}$", r"$\theta_{\mathrm{H}*}$", r"$E^{\mathrm{form,CE}}$"],
    [CO_COLOR, H_COLOR, ENERGY_COLOR],
    ["ML", "ML", "eV/site"],
    INSET_POSITIONS,
):
    add_convergence_inset(
        values, np.mean(full_values[production_mask]), sd,
        label, color, unit, bounds,
    )

ax1.text(
    0.62, 0.87,
    f"{len(conv_steps):,}-move windows; means from production",
    ha="center", transform=ax1.transAxes, fontsize=14, color="#444444",
)


# ============================================================
# Final layout and save
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.93,
    bottom=0.11,
    top=0.97
)

plt.savefig(
    OUTPUT_PATH,
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()
