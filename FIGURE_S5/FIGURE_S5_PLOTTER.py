import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S5_GCMC_LATTICE_COMPARISON.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S5.png"
CONVERGENCE_WINDOW = 1000
COLORS = ("#D55E00", "#0072B2", "#009E73")
STYLES = ("-", "-", "--")
COLUMNS = ("CO_Coverage", "H_Coverage", "Energy_Per_Site_eV")
LABELS = (
    r"$\theta_{\mathrm{CO}*}$ / ML",
    r"$\theta_{\mathrm{H}*}$ / ML",
    r"$E^{\mathrm{form,CE}}$ / eV/site",
)


def add_inset(ax, data, lattice, convergence, column, color, style, bounds):
    window = data.loc[data.Step.between(convergence - CONVERGENCE_WINDOW + 1, convergence)]
    inset = ax.inset_axes(bounds)
    inset.plot(window.Step, window[column], color=color, linestyle=style, linewidth=0.9)
    inset.set_xlim(window.Step.iloc[0], window.Step.iloc[-1])
    value_min, value_max = window[column].min(), window[column].max()
    value_range = value_max - value_min
    if value_range == 0:
        value_range = 1.0 / int(lattice.split("x")[0])**2
    inset.set_ylim(value_min - 0.10 * value_range, value_max + 0.10 * value_range)
    inset.set_xticks([window.Step.iloc[0], window.Step.iloc[-1]])
    inset.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    inset.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    inset.yaxis.set_major_formatter(FormatStrFormatter("%.4f" if "Coverage" in column else "%.3f"))
    inset.tick_params(axis="both", labelsize=10, pad=2, length=3)
    inset.grid(alpha=0.16)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color(color)
    unit = "ML" if "Coverage" in column else "eV/site"
    lattice_label = lattice.replace("x", " × ")
    inset.set_title(
        f"{lattice_label} | {convergence:,} moves\n"
        f"SD = {window[column].std():.5f} {unit}",
        fontsize=10.5, color=color, pad=7,
    )
    mean = data.loc[data.Step > convergence, column].mean()
    mean_y = 0.015 if "Coverage" in column else 0.30
    ax.text(bounds[0] + bounds[2] / 2, mean_y,
            f"Mean: {mean:.5f} {unit}", ha="center", va="bottom",
            color=color, fontsize=11, transform=ax.transAxes)


def main():
    with PKL_PATH.open("rb") as stream:
        values = pickle.load(stream)
    lattices = list(values)
    trajectories = [pd.DataFrame({column: values[lattice][column]
                                 for column in ("Step", *COLUMNS)})
                    for lattice in lattices]
    convergence_steps = [values[lattice]["Convergence_Step"] for lattice in lattices]

    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    max_step = max(int(data.Step.iloc[-1]) for data in trajectories)
    for index, (data, lattice, convergence, color, style) in enumerate(zip(
            trajectories, lattices, convergence_steps, COLORS, STYLES)):
        for ax, column in zip(axes, COLUMNS):
            ax.plot(data.Step, data[column], color=color, linestyle=style,
                    linewidth=0.85, alpha=0.88, label=lattice.replace("x", " × "))
            ax.axvline(convergence, color=color, linestyle=":", linewidth=1.1, alpha=0.8)
            ax.axvspan(convergence - CONVERGENCE_WINDOW + 1, convergence,
                       color=color, alpha=0.09)
            inset_y = 0.16 if "Coverage" in column else 0.45
            add_inset(ax, data, lattice, convergence, column, color, style,
                      [0.405 + index * 0.215, inset_y, 0.155, 0.30])

    for letter, ax, label in zip("abc", axes, LABELS):
        ax.set_ylabel(label, fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(alpha=0.18)
        ax.set_xlim(0, max_step)
        ax.text(-0.065, 1.015, f"({letter})", transform=ax.transAxes, weight="bold")
        heading_y = 0.92 if ax is axes[-1] else (0.60 if ax is axes[1] else 0.63)
        ax.text(0.6975, heading_y, f"{CONVERGENCE_WINDOW:,}-move windows; means from production",
                ha="center", transform=ax.transAxes, fontsize=12, color="#444444")
    axes[0].set_ylim(-0.02, 0.51)
    axes[1].set_ylim(-0.008, 0.155)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.015),
                   borderaxespad=0, ncol=3, frameon=False, fontsize=17,
                   handlelength=3.5, columnspacing=3)
    axes[-1].set_xlabel("Number of Moves", fontsize=18)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.tight_layout(rect=(0.005, 0, 1, 1), h_pad=1.8)
    fig.savefig(OUTPUT_PATH, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
