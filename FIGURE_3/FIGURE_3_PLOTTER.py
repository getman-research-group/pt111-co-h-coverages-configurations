import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / 'FIGURE_3_CO_ISLANDING_DISTRIBUTION.pkl'
output_path = BASE_DIR / 'FIGURE_3_CO_ISLANDING_DISTRIBUTION.png'

cmap_name = "turbo"
use_log_color = True

visible_min_island_size = 3
visible_max_island_size = 500

# Best-fit shared x-axis range
use_manual_xlim = False
manual_xlim = (0.10, 0.55)
x_padding_fraction = 0.03

# Manual color limits
use_manual_vmax = False
manual_vmax = 0.10

use_manual_vmin = False
manual_vmin = 1e-4

# Marker settings
marker_size = 18
marker_shape = "o"
marker_alpha = 0.85

charge_order = ["NEG", "NEU", "POS"]

charge_titles = {
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive"
}

fit_ranges = {
    "NEG": (0.25, 0.35),
    "NEU": (0.25, 0.35),
    "POS": (0.30, 0.40)
}

curve_y_offset = 8.0

# ============================================================
# FONT SETTINGS
# ============================================================

mpl.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})

# ============================================================
# LOAD DATA
# ============================================================

with open(file_path, "rb") as f:
    scatter_data = pickle.load(f)

# ============================================================
# GLOBAL ARRAYS FOR NORMALIZATION AND LIMITS
# ============================================================

all_x = []
all_c = []

for charge_label in charge_order:
    data = scatter_data[charge_label]
    X = np.asarray(data["X"], dtype=float)
    Y = np.asarray(data["Y"], dtype=float)
    C = np.asarray(data["C"], dtype=float)

    scatter_data[charge_label] = {
        "X": X,
        "Y": Y,
        "C": C
    }

    if len(X) > 0:
        all_x.extend(X)
        all_c.extend(C)

all_x = np.array(all_x, dtype=float)
all_c = np.array(all_c, dtype=float)

if len(all_x) == 0:
    raise ValueError("No scatter points found in the PKL file.")

# ============================================================
# BEST-FIT COMMON X-RANGE AND COLOR NORMALIZATION
# ============================================================

if use_manual_xlim:
    x_min, x_max = manual_xlim
else:
    x_min = np.nanmin(all_x)
    x_max = np.nanmax(all_x)

    x_range = x_max - x_min
    x_pad = x_padding_fraction * x_range

    x_min = x_min - x_pad
    x_max = x_max + x_pad

if use_log_color:
    vmin = manual_vmin if use_manual_vmin else np.nanmin(all_c)
    vmax = manual_vmax if use_manual_vmax else np.nanmax(all_c)

    norm = mpl.colors.LogNorm(
        vmin=vmin,
        vmax=vmax
    )
else:
    vmin = 0
    vmax = manual_vmax if use_manual_vmax else np.nanmax(all_c)

    norm = mpl.colors.Normalize(
        vmin=vmin,
        vmax=vmax
    )

cmap = plt.get_cmap(cmap_name)

# ============================================================
# HELPER FUNCTION
# ============================================================

def exp_model(theta, A, B, C, theta0):
    return A * np.exp(B * (theta - theta0)) + C

# ============================================================
# MAKE ONE COMBINED IMAGE FOR ALL THREE CHARGES
# WITH A SAME-HEIGHT COLORBAR IN THE SAME FIGURE
# ============================================================

fig = plt.figure(figsize=(19.0, 6.5))

gs = fig.add_gridspec(
    nrows=1,
    ncols=5,
    width_ratios=[1.0, 1.0, 1.0, 0.08, 0.055],
    wspace=0.0
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[0, 2])
]

axes[1].sharey(axes[0])
axes[2].sharey(axes[0])

cbar_ax = fig.add_subplot(gs[0, 4])

fig.subplots_adjust(
    left=0.055,
    right=0.920,
    bottom=0.18,
    top=0.88
)

for ax in axes[1:]:
    ax.tick_params(labelleft=False)

for ax, charge_label in zip(axes, charge_order):

    data = scatter_data[charge_label]

    X = data["X"]
    Y = data["Y"]
    C = data["C"]

    ax.scatter(
        X,
        Y,
        c=C,
        s=marker_size,
        marker=marker_shape,
        cmap=cmap,
        norm=norm,
        alpha=marker_alpha,
        edgecolors="none",
        zorder=2
    )

    # ========================================================
    # Largest island size for each coverage bin
    # ========================================================

    unique_x = np.array(sorted(np.unique(X)))

    largest_x = []
    largest_y = []

    for xval in unique_x:
        mask = np.isclose(X, xval)

        if not np.any(mask):
            continue

        largest_x.append(xval)
        largest_y.append(np.nanmax(Y[mask]))

    largest_x = np.array(largest_x)
    largest_y = np.array(largest_y)

    # ========================================================
    # Select fit range
    # ========================================================

    x0, x1 = fit_ranges[charge_label]

    fit_mask = (
        (largest_x >= x0) &
        (largest_x <= x1)
    )

    x_fit = largest_x[fit_mask]
    y_fit = largest_y[fit_mask]

    # ========================================================
    # Exponential fit
    # ========================================================

    if len(x_fit) >= 4:

        theta0 = x0

        C0_guess = max(0, np.min(y_fit) - 10)
        A0 = max(1, y_fit[0] - C0_guess)
        B0 = 20.0

        try:
            popt, pcov = curve_fit(
                lambda theta, A, B, C0_fit: exp_model(
                    theta, A, B, C0_fit, theta0
                ),
                x_fit,
                y_fit,
                p0=[A0, B0, C0_guess],
                maxfev=20000
            )

            A, B, Cfit = popt

            x_curve = np.linspace(x_fit.min(), x_fit.max(), 300)
            y_curve = exp_model(x_curve, A, B, Cfit, theta0)

            y_pred = exp_model(x_fit, A, B, Cfit, theta0)

            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

            eq_text = (
                rf"$n_{{\mathrm{{max}}}}={A:.1f}"
                rf"e^{{{B:.1f}(\theta_{{\mathrm{{CO}}*}}-{theta0:.2f})}}"
                rf"{Cfit:+.1f}$"
                "\n"
                rf"$R^2={r2:.3f}$"
            )

            ax.text(
                0.03, 0.97,
                eq_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=20,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    pad=0.3
                ),
                zorder=100
            )

            ax.plot(
                x_curve,
                y_curve + curve_y_offset,
                color="black",
                linewidth=0.5,
                linestyle="-",
                zorder=30
            )

            ax.plot(
                x_curve,
                y_curve + curve_y_offset,
                color="black",
                linewidth=2.0,
                linestyle="--",
                zorder=31
            )

        except RuntimeError:
            pass

    local_x_min = np.nanmin(X)
    local_x_max = np.nanmax(X)

    local_x_range = local_x_max - local_x_min
    local_x_pad = x_padding_fraction * local_x_range

    ax.set_title(charge_titles[charge_label], pad=10)
    ax.set_xlim(
        local_x_min - local_x_pad,
        local_x_max + local_x_pad
    )
    ax.set_ylim(
        visible_min_island_size,
        visible_max_island_size
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(direction="in", length=5, width=1)

axes[0].set_ylabel(r"$n$", labelpad=4)
fig.supxlabel(r"$\theta_{\mathrm{CO}*}$", x=0.47, y=0.055)

# ============================================================
# SHARED COLORBAR IN THE SAME FIGURE
# ============================================================

sm = mpl.cm.ScalarMappable(
    cmap=cmap,
    norm=norm
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    cax=cbar_ax
)

cbar.set_label(r"$\chi_n$", labelpad=8)
cbar.ax.tick_params(direction="in", length=5, width=1)

fig.savefig(
    output_path,
    dpi=300
)

plt.close(fig)
