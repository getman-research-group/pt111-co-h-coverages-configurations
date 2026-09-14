import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


BASE_DIR = Path(__file__).resolve().parent
PANEL_A_PKL = BASE_DIR / "FIGURE_4A_THETA_LOW.pkl"
PANEL_B_PKL = BASE_DIR / "FIGURE_4B_THETA_H.pkl"
PANEL_C_PKL = BASE_DIR / "FIGURE_4C_THETA_COH_1NN.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_4.png"

CHARGES = ["NEG", "NEU", "POS"]
COLORS = {"NEG": "#0072B2", "NEU": "#000000", "POS": "#E69F00"}
LABELS = {"NEG": "Negative", "NEU": "Neutral", "POS": "Positive"}
POLY_ORDER = {"low": 1, "h": 3, "pairs": 2}

mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
    }
)


def load_dataframe_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and obj.get("__type__") == "DataFrame":
        return pd.DataFrame(obj["data"], columns=obj["columns"], index=obj.get("index"))
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame(obj)


def fit_polynomial(x, y, order):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) <= order:
        return None
    return np.poly1d(np.polyfit(x[keep], y[keep], order))


def plot_series(ax, x, y, yerr, color, alpha=1.0, label=None, order=1, zorder=3):
    poly = fit_polynomial(x, y, order)
    x_fit = np.linspace(np.min(x), np.max(x), 300) if poly is not None else x
    y_fit = poly(x_fit) if poly is not None else y

    ax.plot(x_fit, y_fit, lw=1, color=color, alpha=alpha, label=label, zorder=zorder)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=4,
        capsize=3,
        elinewidth=1,
        color=color,
        alpha=alpha,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=zorder + 1,
    )


def plot_single_quantity_panel(ax, df, ylabel, order):
    for charge in CHARGES:
        part = df[df["Charge"] == charge].sort_values("CO_Coverage")
        x = part["CO_Coverage"].to_numpy(dtype=float)
        y = part["Y_Plot"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                part["Y_Error_Low"].to_numpy(dtype=float),
                part["Y_Error_High"].to_numpy(dtype=float),
            ]
        )
        plot_series(ax, x, y, yerr, COLORS[charge], label=LABELS[charge], order=order)

    ax.set_ylabel(ylabel)
    ax.grid(False)


def plot_panel_c(ax, df):
    ax2 = ax.twinx()

    for charge in CHARGES:
        color = COLORS[charge]

        for quantity, target_ax, alpha, order, zorder in [
            ("CO_H_1NN", ax, 1.0, POLY_ORDER["pairs"], 3),
            ("CO_times_H", ax2, 0.35, POLY_ORDER["pairs"], 1),
        ]:
            part = df[
                (df["Charge"] == charge) & (df["Quantity"] == quantity)
            ].sort_values("CO_Coverage")
            x = part["CO_Coverage"].to_numpy(dtype=float)
            y = part["Y_Plot"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    part["Y_Error_Low"].to_numpy(dtype=float),
                    part["Y_Error_High"].to_numpy(dtype=float),
                ]
            )
            plot_series(target_ax, x, y, yerr, color, alpha=alpha, order=order, zorder=zorder)

    ax.set_ylabel(r"$\theta_{\mathrm{CO^*-H^*~1NN}}$ / ML")
    ax2.set_ylabel(r"$\theta_{\mathrm{CO}*}\theta_{\mathrm{H}*}$ / ML$^2$")

    ax.set_ylim(0.00, 0.14)
    ax.set_yticks(np.arange(0.00, 0.12 + 0.001, 0.04))
    ax2.set_ylim(0.00, 0.14)
    ax2.set_yticks(np.arange(0.00, 0.12 + 0.001, 0.04))

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)
    return ax2


panel_a_df = load_dataframe_pkl(PANEL_A_PKL)
panel_b_df = load_dataframe_pkl(PANEL_B_PKL)
panel_c_df = load_dataframe_pkl(PANEL_C_PKL)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(6, 11),
    sharex=True,
    gridspec_kw={"hspace": 0.0},
)

plot_single_quantity_panel(axes[0], panel_a_df, r"$\theta_{low}$ / ML", POLY_ORDER["low"])
plot_single_quantity_panel(axes[1], panel_b_df, r"$\theta_{\mathrm{H}*}$ / ML", POLY_ORDER["h"])
panel_c_right_ax = plot_panel_c(axes[2], panel_c_df)

axes[0].set_ylim(0.00, 0.16)
axes[0].set_yticks(np.arange(0.02, 0.14 + 0.001, 0.04))
axes[1].set_ylim(0.12, 0.44)
axes[1].set_yticks(np.arange(0.16, 0.42 + 0.001, 0.08))

for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("")
    ax.tick_params(direction="in", length=5, width=1)

panel_c_right_ax.tick_params(direction="in", length=5, width=1)

for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)

for ax, label in zip(axes, ["a)", "b)", "c)"]):
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )

quantity_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        lw=1,
        marker="o",
        markersize=4,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.8,
        alpha=1.0,
        label=r"$\theta_{\mathrm{CO^*-H^*~1NN}}$",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        lw=1,
        marker="o",
        markersize=4,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.8,
        alpha=0.35,
        label=r"$\theta_{\mathrm{CO}*}\theta_{\mathrm{H}*}$",
    ),
]
axes[2].legend(
    handles=quantity_handles,
    loc="lower right",
    ncol=2,
    frameon=False,
    handlelength=2.2,
    columnspacing=1.2,
    handletextpad=0.5,
)

fig.legend(
    [Line2D([0], [0], color=COLORS[c], lw=2) for c in CHARGES],
    [LABELS[c] for c in CHARGES],
    loc="upper center",
    bbox_to_anchor=(0.50, 0.975),
    ncol=3,
    frameon=False,
)

fig.supxlabel(r"$\theta_{\mathrm{CO}*}$ / ML", x=0.50, y=0.015, fontsize=14)

plt.subplots_adjust(left=0.18, right=0.84, bottom=0.08, top=0.93, hspace=0.0)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {OUTPUT_PATH}")
