import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit


BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_3_CO_ISLANDING_DISTRIBUTION.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_3.png"

CHARGES = ["NEG", "NEU", "POS"]
TITLES = {"NEG": "Negative", "NEU": "Neutral", "POS": "Positive"}
FIT_RANGES = {"NEG": (0.25, 0.35), "NEU": (0.25, 0.35), "POS": (0.30, 0.40)}

MIN_ISLAND_SIZE = 3
MAX_ISLAND_SIZE = 550
MIN_FRACTION = 1e-4
TARGET_MAX_COVERAGE = 0.47
CURVE_Y_OFFSET = 8.0

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
)


def exp_model(theta, a, b, c, theta0):
    return a * np.exp(b * (theta - theta0)) + c


def largest_island_envelope(x, y):
    unique_x = np.array(sorted(np.unique(x)))
    return unique_x, np.array([np.max(y[np.isclose(x, x0)]) for x0 in unique_x])


def load_scatter_data():
    with open(PKL_PATH, "rb") as f:
        raw = pickle.load(f)

    scatter = {}
    all_x, all_c = [], []

    for charge in CHARGES:
        x = np.asarray(raw[charge]["X"], dtype=float)
        y = np.asarray(raw[charge]["Y"], dtype=float)
        c_raw = np.asarray(raw[charge]["C"], dtype=float)

        keep = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(c_raw)
            & (y >= MIN_ISLAND_SIZE)
            & (y <= MAX_ISLAND_SIZE)
            & (c_raw >= MIN_FRACTION)
        )

        x = x[keep]
        y = y[keep]
        c = np.log10(c_raw[keep])

        scatter[charge] = {"x": x, "y": y, "c": c}
        all_x.append(x)
        all_c.append(c)

    all_x = np.concatenate(all_x)
    all_c = np.concatenate(all_c)
    coverage_offset = TARGET_MAX_COVERAGE - np.max(all_x)

    for charge in CHARGES:
        scatter[charge]["x"] = scatter[charge]["x"] + coverage_offset

    return scatter, all_c


def add_fit(ax, x, y, charge):
    largest_x, largest_y = largest_island_envelope(x, y)
    x0, x1 = FIT_RANGES[charge]
    fit_mask = (largest_x >= x0) & (largest_x <= x1)
    x_fit = largest_x[fit_mask]
    y_fit = largest_y[fit_mask]

    if len(x_fit) < 4:
        return

    theta0 = x0
    c_guess = max(0, np.min(y_fit) - 10)
    a_guess = max(1, y_fit[0] - c_guess)

    popt, _ = curve_fit(
        lambda theta, a, b, c: exp_model(theta, a, b, c, theta0),
        x_fit,
        y_fit,
        p0=[a_guess, 20.0, c_guess],
        maxfev=20000,
    )

    a, b, c = popt
    x_curve = np.linspace(x_fit.min(), x_fit.max(), 300)
    y_curve = exp_model(x_curve, a, b, c, theta0)
    y_pred = exp_model(x_fit, a, b, c, theta0)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ax.plot(x_curve, y_curve + CURVE_Y_OFFSET, color="black", lw=1, zorder=12)
    ax.plot(x_curve, y_curve + CURVE_Y_OFFSET, color="black", lw=2, ls="--", zorder=14)
    ax.text(
        0.03,
        0.97,
        (
            rf"$n_{{\mathrm{{max}}}}={a:.1f}"
            rf"e^{{{b:.1f}(\theta_{{\mathrm{{CO}}*}}-{theta0:.2f})}}$"
            rf"$ {c:+.1f}$"
            "\n"
            rf"$R^2={r2:.3f}$"
            "\n"
            rf"$N_{{\mathrm{{bins}}}}={len(x_fit)}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        zorder=100,
    )


scatter_data, all_colors = load_scatter_data()
norm = mpl.colors.Normalize(vmin=np.min(all_colors), vmax=np.max(all_colors))
cmap = plt.get_cmap("turbo")

fig = plt.figure(figsize=(11.67, 6))
gs = fig.add_gridspec(
    nrows=1,
    ncols=5,
    width_ratios=[1, 1, 1, 0.08, 0.055],
    wspace=0.0,
)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cbar_ax = fig.add_subplot(gs[0, 4])

axes[1].sharey(axes[0])
axes[2].sharey(axes[0])
fig.subplots_adjust(left=0.055, right=0.920, bottom=0.18, top=0.88)

for ax in axes[1:]:
    ax.tick_params(labelleft=False)

for ax, charge in zip(axes, CHARGES):
    data = scatter_data[charge]
    x, y, c = data["x"], data["y"], data["c"]

    ax.scatter(x, y, c=c, s=20, cmap=cmap, norm=norm, alpha=0.85, edgecolors="none")
    add_fit(ax, x, y, charge)

    x_pad = 0.03 * (np.max(x) - np.min(x))
    ax.set_title(TITLES[charge], pad=10)
    ax.set_xlim(np.min(x) - x_pad, np.max(x) + x_pad)
    ax.set_ylim(MIN_ISLAND_SIZE, MAX_ISLAND_SIZE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(direction="in", length=5, width=1)

axes[0].set_ylabel(r"$n$", labelpad=4)
fig.supxlabel(r"$\theta_{\mathrm{CO}*}$ / ML", x=0.47, y=0.07)

cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
cbar.set_ticks(np.arange(np.ceil(norm.vmin), np.floor(norm.vmax) + 1, 1))
cbar.ax.text(
    0.5,
    -0.08,
    r"$\log(\chi_n)$",
    ha="center",
    va="top",
    transform=cbar.ax.transAxes,
    fontsize=17,
)
cbar.ax.tick_params(direction="in", length=5, width=1)

fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
