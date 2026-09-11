import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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

# Inset location in figure-relative coordinates:
# [left, bottom, width, height]
INSET_POSITION = [0.37, 0.535, 0.41, 0.34]


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
    label=r"$E^{\mathrm{form,CE}}/N_{\mathrm{sites}}$"
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
    r"$E^{\mathrm{form,CE}}/N_{\mathrm{sites}}$ (eV/site)",
    fontsize=24
)

ax2.set_ylabel(
    r"$\theta_{\mathrm{CO}*}, \theta_{\mathrm{H}*}$ | ML",
    fontsize=24
)

ax1.set_xlim(step.min() - 600, step.max() + 600)
ax1.set_ylim(-31.15, -24.82)
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
# Inset: 1000-move convergence window
# ============================================================

axins = fig.add_axes(INSET_POSITION)
axins2 = axins.twinx()

axins.plot(
    conv_steps,
    conv_energy,
    color=ENERGY_COLOR,
    lw=INSET_LINEWIDTH
)

axins2.plot(
    conv_steps,
    conv_co,
    color=CO_COLOR,
    lw=INSET_LINEWIDTH
)

axins2.plot(
    conv_steps,
    conv_h,
    color=H_COLOR,
    lw=INSET_LINEWIDTH
)

axins.set_title("1000-Move Convergence Window", fontsize=23, pad=5)

axins.set_xlabel("Number of Moves", fontsize=20)
axins.set_ylabel(
    r"$E^{\mathrm{form,CE}}/N_{\mathrm{sites}}$ (eV/site)",
    fontsize=20
)
axins2.set_ylabel(
    r"$\theta_{\mathrm{CO}*}, \theta_{\mathrm{H}*}$ | ML",
    fontsize=20
)

axins.tick_params(axis="both", labelsize=18)
axins2.tick_params(axis="y", labelsize=18)

# Match the visual scale in the reference plot closely
energy_pad = max(0.003, 0.12 * (conv_energy.max() - conv_energy.min()))
axins.set_ylim(
    conv_energy.min() - energy_pad,
    conv_energy.max() + energy_pad
)

axins2.set_ylim(0.10, 0.50)
axins.set_xlim(conv_start, conv_end)

stats_text = (
    rf"$\sigma(\theta_{{CO}})$ = {sd_co:.5f}" + "\n"
    rf"$\sigma(\theta_{{H}})$ = {sd_h:.5f}" + "\n"
    rf"$\sigma(E/N)$ = {sd_energy:.6f} eV/site"
)

ax1.text(
    0.245,
    0.34,
    stats_text,
    transform=ax1.transAxes,
    fontsize=14,
    va="bottom",
    ha="left",
    linespacing=1.25,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
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
