import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

panel_a_pkl = BASE_DIR / 'FIGURE_4A_THETA_LOW.pkl'
panel_b_pkl = BASE_DIR / 'FIGURE_4B_THETA_H.pkl'
panel_c_pkl = BASE_DIR / 'FIGURE_4C_THETA_COH_1NN.pkl'

output_path = BASE_DIR / 'FIGURE_4_SPATIAL_ANALYSIS.png'

charges_to_plot = ["NEG", "NEU", "POS"]

charge_colors = {
    "NEG": "#0072B2",   # Blue
    "NEU": "#000000",   # Black
    "POS": "#E69F00",   # Orange
}

charge_labels = {
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive",
}


poly_order_empty = 1
poly_order_h = 3
poly_order_pairs = 2

# ============================================================
# FONT SETTINGS
# ============================================================

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# ============================================================
# FUNCTIONS
# ============================================================

def fit_polynomial(x, y, order):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) <= order:
        return None

    coeffs = np.polyfit(x, y, order)
    return np.poly1d(coeffs)


def plot_panel_a(ax, df):
    for charge in charges_to_plot:
        plot_df = df[df["Charge"] == charge].copy()
        plot_df = plot_df.sort_values("CO_Coverage")

        x = plot_df["CO_Coverage"].to_numpy(dtype=float)
        y = plot_df["Y_Plot"].to_numpy(dtype=float)
        yerr_lower = plot_df["Y_Error_Low"].to_numpy(dtype=float)
        yerr_upper = plot_df["Y_Error_High"].to_numpy(dtype=float)
        yerr = np.vstack([yerr_lower, yerr_upper])

        color = charge_colors[charge]

        poly = fit_polynomial(x, y, poly_order_empty)

        if poly is None:
            x_fit = x
            y_fit = y
        else:
            x_fit = np.linspace(np.min(x), np.max(x), 300)
            y_fit = poly(x_fit)

        ax.plot(
            x_fit,
            y_fit,
            linewidth=1,
            color=color,
            label=charge_labels[charge],
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_ylabel(r"$\theta_{low}$ / ML")
    ax.grid(False)


def plot_panel_b(ax, df):
    for charge in charges_to_plot:
        plot_df = df[df["Charge"] == charge].copy()
        plot_df = plot_df.sort_values("CO_Coverage")

        x = plot_df["CO_Coverage"].to_numpy(dtype=float)
        y = plot_df["Y_Plot"].to_numpy(dtype=float)
        yerr_lower = plot_df["Y_Error_Low"].to_numpy(dtype=float)
        yerr_upper = plot_df["Y_Error_High"].to_numpy(dtype=float)
        yerr = np.vstack([yerr_lower, yerr_upper])

        color = charge_colors[charge]

        poly = fit_polynomial(x, y, poly_order_h)

        if poly is None:
            x_fit = x
            y_fit = y
        else:
            x_fit = np.linspace(np.min(x), np.max(x), 300)
            y_fit = poly(x_fit)

        ax.plot(
            x_fit,
            y_fit,
            linewidth=1,
            color=color,
            label=charge_labels[charge],
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_ylabel(r"$\theta_{\mathrm{H}*}$ / ML")
    ax.grid(False)


def plot_panel_c(ax, df):
    for charge in charges_to_plot:
        color = charge_colors[charge]
        charge_name = charge_labels[charge]

        # ----------------------------------------------------
        # CO_H_1NN
        # ----------------------------------------------------
        df_1nn = df[
            (df["Charge"] == charge) &
            (df["Quantity"] == "CO_H_1NN")
        ].copy().sort_values("CO_Coverage")

        x1 = df_1nn["CO_Coverage"].to_numpy(dtype=float)
        y1 = df_1nn["Y_Plot"].to_numpy(dtype=float)
        yerr1_lower = df_1nn["Y_Error_Low"].to_numpy(dtype=float)
        yerr1_upper = df_1nn["Y_Error_High"].to_numpy(dtype=float)
        yerr1 = np.vstack([yerr1_lower, yerr1_upper])

        poly1 = fit_polynomial(x1, y1, poly_order_pairs)

        if poly1 is None:
            x1_fit = x1
            y1_fit = y1
        else:
            x1_fit = np.linspace(np.min(x1), np.max(x1), 300)
            y1_fit = poly1(x1_fit)

        ax.plot(
            x1_fit,
            y1_fit,
            linewidth=1,
            color=color,
            alpha=1.0,
            linestyle="-",
            label=charge_name,
            zorder=3,
        )

        ax.errorbar(
            x1,
            y1,
            yerr=yerr1,
            fmt="o",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color=color,
            alpha=1.0,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=4,
        )

        # ----------------------------------------------------
        # CO_times_H
        # ----------------------------------------------------
        df_prod = df[
            (df["Charge"] == charge) &
            (df["Quantity"] == "CO_times_H")
        ].copy().sort_values("CO_Coverage")

        x2 = df_prod["CO_Coverage"].to_numpy(dtype=float)
        y2 = df_prod["Y_Plot"].to_numpy(dtype=float)
        yerr2_lower = df_prod["Y_Error_Low"].to_numpy(dtype=float)
        yerr2_upper = df_prod["Y_Error_High"].to_numpy(dtype=float)
        yerr2 = np.vstack([yerr2_lower, yerr2_upper])

        poly2 = fit_polynomial(x2, y2, poly_order_pairs)

        if poly2 is None:
            x2_fit = x2
            y2_fit = y2
        else:
            x2_fit = np.linspace(np.min(x2), np.max(x2), 300)
            y2_fit = poly2(x2_fit)

        ax.plot(
            x2_fit,
            y2_fit,
            linewidth=1,
            color=color,
            alpha=0.35,
            linestyle="-",
            zorder=1,
        )

        ax.errorbar(
            x2,
            y2,
            yerr=yerr2,
            fmt="o",
            markersize=3,
            capsize=3,
            elinewidth=1,
            color=color,
            alpha=0.35,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=2,
        )

    ax.set_ylabel(
        r"$\theta_{\mathrm{CO^*-H^*~1NN}}$ or "
        r"$\theta_{\mathrm{CO}*}\theta_{\mathrm{H}*}$ / ML"
    )

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(False)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataframe_pkl(path):
    """
    Load either:
    1. the portable DataFrame PKL format, or
    2. a normal pandas DataFrame PKL.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Portable DataFrame format
    if (
        isinstance(obj, dict)
        and obj.get("__type__") == "DataFrame"
        and "columns" in obj
        and "data" in obj
    ):
        return pd.DataFrame(
            obj["data"],
            columns=obj["columns"],
            index=obj.get("index", None)
        )

    # Normal pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    # Fallback
    return pd.DataFrame(obj)


panel_a_df = load_dataframe_pkl(panel_a_pkl)
panel_b_df = load_dataframe_pkl(panel_b_pkl)
panel_c_df = load_dataframe_pkl(panel_c_pkl)

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(5, 9),
    sharex=True,
    gridspec_kw={"hspace": 0.0},
)

plot_panel_a(axes[0], panel_a_df)
plot_panel_b(axes[1], panel_b_df)
plot_panel_c(axes[2], panel_c_df)

# Set exact y-axis limits and ticks
axes[0].set_ylim(0.00, 0.16)
axes[0].set_yticks(np.arange(0.02, 0.14 + 0.001, 0.04))

axes[1].set_ylim(0.12, 0.44)
axes[1].set_yticks(np.arange(0.16, 0.42 + 0.001, 0.08))

axes[2].set_ylim(0.00, 0.14)
axes[2].set_yticks(np.arange(0.00, 0.12 + 0.001, 0.04))

for ax in axes:
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# Remove individual legends from all panels
for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("")

for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)



# Panel labels
panel_labels = ["a)", "b)", "c)"]

for ax, label in zip(axes, panel_labels):
    ax.text(
        0.02, 0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )

# ============================================================
# Panel c legend (quantities)
# ============================================================

quantity_handles = [
    Line2D(
        [0], [0],
        color="black",
        lw=1,
        marker="o",
        markersize=3,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.8,
        alpha=1.0,
        label=r"$\theta_{\mathrm{CO^*-H^*~1NN}}$",
    ),
    Line2D(
        [0], [0],
        color="black",
        lw=1,
        marker="o",
        markersize=3,
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

# Central legend above all panels
handles = [
    Line2D([0], [0], color=charge_colors[c], lw=2)
    for c in charges_to_plot
]
labels = [charge_labels[c] for c in charges_to_plot]

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.58, 0.975),
    ncol=3,
    frameon=False,
)

fig.supxlabel(
    r"$\theta_{\mathrm{CO}*}$ / ML",
    x=0.58,
    y=0.03
)

plt.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.08,
    top=0.93,
    hspace=0.0,
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure: {output_path}")