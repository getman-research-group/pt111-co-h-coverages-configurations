import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path



# ============================================================
# USER SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PKL_PATH = BASE_DIR / 'FIGURE_S2_PRE_COVERAGE_BINDING_ENERGY.pkl'
OUTPUT_PATH = BASE_DIR / 'FIGURE_S2.png'

colors = {
    "NEG": "#0072B2",
    "NEU": "#000000",
    "POS": "#E69F00",
}

labels = {
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive",
}

charge_order = ["NEG", "NEU", "POS"]


# ============================================================
# BAR GEOMETRY
# ============================================================

BAR_WIDTH = 0.24
GROUP_SPACING = 1.0


# ============================================================
# FIGURE SETTINGS
# ============================================================

FIGURE_WIDTH = 10.0
FIGURE_HEIGHT = 13.0

LABEL_FONT_SIZE = 17
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 17
LEGEND_FONT_SIZE = 18


# ============================================================
# LOAD DATA
# ============================================================

import pickle

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

required_columns = {
    # CO binding-energy data
    "CO_COV_FOR_CO",
    "H_COV_FOR_CO",
    "CO_BE_NEG",
    "CO_BE_NEU",
    "CO_BE_POS",

    # H binding-energy data
    "CO_COV_FOR_H",
    "H_COV_FOR_H",
    "H_BE_NEG",
    "H_BE_NEU",
    "H_BE_POS",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        "The PKL file is missing the following columns:\n"
        + "\n".join(sorted(missing_columns))
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cov_str(value):
    """
    Return coverage string.

    Zero is displayed as 0.
    Nonzero values are displayed as x / 9.
    """

    value = int(value)

    if value == 0:
        return "0"

    return rf"{value}\,/\,9"


def make_coverage_label(
    co_numerator,
    h_numerator,
    adsorbate,
):
    """
    Create the vertical label above one bar group.
    """

    co_numerator = int(co_numerator)
    h_numerator = int(h_numerator)

    if co_numerator == 0 and h_numerator == 0:
        return (
            rf"$E_{{\mathrm{{{adsorbate}}}*}}"
            rf"^{{\mathrm{{ads}}}}$"
        )

    return (
        rf"$\theta_{{\mathrm{{CO}}*}}={cov_str(co_numerator)}$"
        "\n"
        rf"$\theta_{{\mathrm{{H}}*}}={cov_str(h_numerator)}$"
    )


def prepare_panel_data(
    source_df,
    co_column,
    h_column,
    energy_columns,
):
    """
    Extract, clean, and sort data for one panel.

    The isolated (0,0) case is always placed first.
    """

    panel_columns = [
        co_column,
        h_column,
        *energy_columns.values(),
    ]

    panel_df = source_df[panel_columns].dropna().copy()

    panel_df[co_column] = panel_df[co_column].astype(int)
    panel_df[h_column] = panel_df[h_column].astype(int)

    panel_df["_isolated_first"] = np.where(
        (panel_df[co_column] == 0)
        & (panel_df[h_column] == 0),
        0,
        1,
    )

    panel_df = (
        panel_df.sort_values(
            by=[
                "_isolated_first",
                co_column,
                h_column,
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .drop(columns="_isolated_first")
        .reset_index(drop=True)
    )

    isolated_mask = (
        (panel_df[co_column] == 0)
        & (panel_df[h_column] == 0)
    )

    if isolated_mask.sum() == 0:
        raise ValueError(
            f"No isolated row with {co_column}=0 and "
            f"{h_column}=0 was found."
        )

    if isolated_mask.sum() > 1:
        raise ValueError(
            f"More than one isolated row with {co_column}=0 and "
            f"{h_column}=0 was found."
        )

    isolated_row = panel_df.loc[isolated_mask].iloc[0]

    return panel_df, isolated_row


def plot_binding_energy_panel(
    ax,
    panel_df,
    isolated_row,
    co_column,
    h_column,
    energy_columns,
    adsorbate,
):
    """
    Plot one binding-energy panel.
    """

    number_of_groups = len(panel_df)

    group_positions = (
        np.arange(number_of_groups) * GROUP_SPACING
    )

    bar_offsets = {
        "NEG": -BAR_WIDTH,
        "NEU": 0.0,
        "POS": BAR_WIDTH,
    }

    left_plot_edge = group_positions[0] - 0.65
    right_plot_edge = group_positions[-1] + 0.65

    isolated_group_x = group_positions[0]


    # ========================================================
    # GROUPED BARS
    # ========================================================

    for charge in charge_order:

        values = panel_df[
            energy_columns[charge]
        ].to_numpy(dtype=float)

        bar_positions = (
            group_positions + bar_offsets[charge]
        )

        ax.bar(
            bar_positions,
            values,
            width=BAR_WIDTH,
            color=colors[charge],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )


    # ========================================================
    # DASHED ISOLATED-ADSORBATE REFERENCE LINES
    # ========================================================

    for charge in charge_order:

        isolated_energy = float(
            isolated_row[energy_columns[charge]]
        )

        isolated_bar_center = (
            isolated_group_x + bar_offsets[charge]
        )

        isolated_bar_right_edge = (
            isolated_bar_center + BAR_WIDTH / 2
        )

        ax.hlines(
            y=isolated_energy,
            xmin=isolated_bar_right_edge,
            xmax=right_plot_edge,
            colors=colors[charge],
            linestyles="--",
            linewidth=1.6,
            zorder=2,
        )


    # ========================================================
    # TOP COVERAGE LABELS
    # ========================================================

    coverage_labels = [
        make_coverage_label(
            co_numerator=co,
            h_numerator=h,
            adsorbate=adsorbate,
        )
        for co, h in zip(
            panel_df[co_column],
            panel_df[h_column],
        )
    ]

    ax.set_xticks(group_positions)

    ax.set_xticklabels(
        coverage_labels,
        fontsize=LABEL_FONT_SIZE,
        rotation=90,
        ha="center",
        va="bottom",
    )

    ax.tick_params(
        axis="x",
        which="major",
        labeltop=True,
        labelbottom=False,
        top=False,
        bottom=False,
        length=0,
        pad=20,
    )


    # ========================================================
    # Y-AXIS LABEL
    # ========================================================

    ax.set_ylabel(
        rf"$E_{{\mathrm{{{adsorbate}}}*}}^{{\mathrm{{ads}}}}"
        rf"(\theta_{{\mathrm{{CO}}*}},"
        rf"\theta_{{\mathrm{{H}}*}})$ (eV)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )


    # ========================================================
    # AXIS LIMITS
    # ========================================================

    all_energies = panel_df[
        list(energy_columns.values())
    ].to_numpy(dtype=float)

    energy_min = np.nanmin(all_energies)

    if adsorbate == "CO":
        bottom_padding = 0.07
    else:
        bottom_padding = 0.05

    ax.set_ylim(
        energy_min - bottom_padding,
        0.0,
    )

    ax.set_xlim(
        left_plot_edge,
        right_plot_edge,
    )


    # ========================================================
    # ZERO LINE
    # ========================================================

    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        zorder=2,
    )


    # ========================================================
    # GENERAL FORMATTING
    # ========================================================

    ax.tick_params(
        axis="y",
        direction="out",
        length=5,
        width=1.0,
        labelsize=TICK_FONT_SIZE,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.grid(False)


# ============================================================
# DEFINE COLUMN SETS
# ============================================================

co_energy_columns = {
    "NEG": "CO_BE_NEG",
    "NEU": "CO_BE_NEU",
    "POS": "CO_BE_POS",
}

h_energy_columns = {
    "NEG": "H_BE_NEG",
    "NEU": "H_BE_NEU",
    "POS": "H_BE_POS",
}


# ============================================================
# PREPARE CO DATA
# ============================================================

co_df, co_isolated_row = prepare_panel_data(
    source_df=df,
    co_column="CO_COV_FOR_CO",
    h_column="H_COV_FOR_CO",
    energy_columns=co_energy_columns,
)


# ============================================================
# PREPARE H DATA
# ============================================================

h_df, h_isolated_row = prepare_panel_data(
    source_df=df,
    co_column="CO_COV_FOR_H",
    h_column="H_COV_FOR_H",
    energy_columns=h_energy_columns,
)


# ============================================================
# CREATE VERTICAL FIGURE
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
)


# ============================================================
# TOP PANEL: CO
# ============================================================

plot_binding_energy_panel(
    ax=axes[0],
    panel_df=co_df,
    isolated_row=co_isolated_row,
    co_column="CO_COV_FOR_CO",
    h_column="H_COV_FOR_CO",
    energy_columns=co_energy_columns,
    adsorbate="CO",
)


# ============================================================
# BOTTOM PANEL: H
# ============================================================

plot_binding_energy_panel(
    ax=axes[1],
    panel_df=h_df,
    isolated_row=h_isolated_row,
    co_column="CO_COV_FOR_H",
    h_column="H_COV_FOR_H",
    energy_columns=h_energy_columns,
    adsorbate="H",
)


# ============================================================
# Y TICKS
# ============================================================

axes[0].set_yticks(
    np.arange(
        -1.50,
        0.001,
        0.25,
    )
)

axes[1].set_yticks(
    np.arange(
        -0.50,
        0.001,
        0.10,
    )
)


# ============================================================
# COMMON LEGEND
# ============================================================

legend_handles = [
    Patch(
        facecolor=colors[charge],
        edgecolor="black",
        linewidth=0.7,
        label=labels[charge],
    )
    for charge in charge_order
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=3,
    frameon=False,
    fontsize=LEGEND_FONT_SIZE,
    handlelength=1.8,
    columnspacing=2.8,
)


# ============================================================
# PANEL SPACING
# ============================================================

plt.subplots_adjust(
    left=0.13,
    right=0.98,
    top=0.87,
    bottom=0.10,
    hspace=0.42,
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
)

plt.show()

print(f"Figure saved to:\n{OUTPUT_PATH}")