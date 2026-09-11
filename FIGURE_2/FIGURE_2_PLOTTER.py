import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from pathlib import Path


# ============================================================
# Font Settings
# ============================================================

mpl.rcParams.update({
    'font.size': 21,
    'axes.titlesize': 21,
    'axes.labelsize': 21,
    'xtick.labelsize': 21,
    'ytick.labelsize': 21,
    'legend.fontsize': 21,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3
})


# ============================================================
# Input
# ============================================================

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / 'FIGURE_2_CO_COVERAGE_HEATMAP.pkl'

output_path = BASE_DIR / 'FIGURE_2_CO_COVERAGE_HEATMAP.png'


co_filter_values = [
    100,
    1000,
    10000
]

z_keys = [
    'AVG_NEG',
    'AVG_NEU',
    'AVG_POS'
]

titles = [
    'Negative',
    'Neutral',
    'Positive'
]


# ============================================================
# Load Portable PKL
# ============================================================

with open(file_path, 'rb') as f:
    payload = pickle.load(f)

df_all = pd.DataFrame(
    payload['data'],
    columns=payload['columns']
)


# ============================================================
# Rename Columns
# ============================================================

df_all = df_all.rename(columns={
    'Temperature (K)': 'TEMP',

    'CO Concentration (ppm)': 'CO',

    'H2 Concetration (ln(P_H2/P_CO))': 'H',

    'Average CO* coverage on negatively charged surface (ML)':
        'AVG_NEG',

    'Average CO* coverage on neutral surface (ML)':
        'AVG_NEU',

    'Average CO* coverage on positively charged surface (ML)':
        'AVG_POS'
})


# ============================================================
# Data Preparation Helper
# ============================================================

def prepare_data(df, z_keys, co_vals):

    all_pivot_rows = []

    global_vmin = float('inf')
    global_vmax = float('-inf')

    for co_value in co_vals:

        # Filter for selected CO concentration
        df_filtered = df[
            df['CO'] == co_value
        ].copy()

        # Create one pivot table for each charge state
        pivots = [
            df_filtered.pivot(
                index='H',
                columns='TEMP',
                values=z
            )
            for z in z_keys
        ]

        all_pivot_rows.append(pivots)

        # Find global min and max so all panels
        # use the same color scale
        for pivot in pivots:

            current_min = pivot.min().min()
            current_max = pivot.max().max()

            global_vmin = min(
                global_vmin,
                current_min
            )

            global_vmax = max(
                global_vmax,
                current_max
            )

    return (
        all_pivot_rows,
        global_vmin,
        global_vmax
    )


# ============================================================
# Plotting Helper
# ============================================================

def draw_block(
    pivot_rows,
    global_vmin,
    global_vmax,
    fig,
    gs,
    block_label,
    colorbar_label,
    show_titles=False
):

    final_im = None

    middle_left_ax = None

    right_row_axes = []


    # ========================================================
    # Draw Heatmap Panels
    # ========================================================

    for i, (pivot_row, co_val) in enumerate(
        zip(pivot_rows, co_filter_values)
    ):

        for j, (pivot, title) in enumerate(
            zip(pivot_row, titles)
        ):

            ax = fig.add_subplot(
                gs[i, j]
            )


            # =================================================
            # Save Important Axes
            # =================================================

            if i == 1 and j == 0:

                middle_left_ax = ax

            if j == 2:

                right_row_axes.append(ax)


            # =================================================
            # Sort Data
            # =================================================

            pivot = pivot.sort_index()

            pivot = pivot.reindex(
                sorted(pivot.columns),
                axis=1
            )


            # =================================================
            # Mesh Grid
            # =================================================

            X, Y = np.meshgrid(
                pivot.columns.values,
                pivot.index.values
            )


            # =================================================
            # Heatmap
            # =================================================

            im = ax.pcolormesh(
                X,
                Y,
                pivot.values,
                shading='auto',
                cmap='coolwarm',
                vmin=global_vmin,
                vmax=global_vmax,
                edgecolors='black',
                linewidth=0.5
            )


            # =================================================
            # Plot Borders
            # =================================================

            for spine in ax.spines.values():

                spine.set_edgecolor(
                    'black'
                )

                spine.set_linewidth(
                    2.5
                )


            # =================================================
            # Column Titles
            # =================================================

            if show_titles and i == 0:

                ax.set_title(
                    title,
                    pad=8,
                    fontsize=21
                )


            # =================================================
            # Y-axis Ticks
            # =================================================

            if j == 0:

                ax.set_yticks(
                    pivot.index.values
                )

                ax.tick_params(
                    axis='y',
                    pad=2
                )

            else:

                ax.set_yticks([])

                ax.set_yticklabels([])


            # =================================================
            # Temperature Axis
            # =================================================

            if i == 2:

                ax.set_xticks(
                    pivot.columns.values
                )

                ax.tick_params(
                    axis='x',
                    pad=2,
                    labelrotation=90
                )

                for label in ax.get_xticklabels():

                    label.set_horizontalalignment(
                        'center'
                    )

                    label.set_verticalalignment(
                        'top'
                    )


                # =============================================
                # X-axis Label
                # =============================================

                if j == 1:

                    ax.set_xlabel(
                        r'$T$ / K',
                        labelpad=16
                    )

            else:

                ax.set_xticks([])

                ax.set_xticklabels([])


            final_im = im


    # =========================================================
    # Left-side ln(PH2 / PCO) Label
    #
    # y = 0 is in data coordinates, so the center of the label
    # is aligned exactly with the 0 tick of the middle row.
    # =========================================================

    if middle_left_ax is not None and block_label:

        middle_left_ax.text(
            -0.35,
            0,
            block_label,
            transform=middle_left_ax.get_yaxis_transform(),
            rotation=90,
            va='center',
            ha='center',
            fontsize=24,
            clip_on=False
        )


    # =========================================================
    # Right-side CO Pressure Values
    #
    # First vertical slot:
    #
    # 10^2
    # 10^3
    # 10^4
    # =========================================================

    for ax, co_val in zip(
        right_row_axes,
        co_filter_values
    ):

        exponent = int(
            np.log10(co_val)
        )

        row_label = (
            rf'$10^{{{exponent}}}$'
        )

        ax.text(
            1.12,
            0.5,
            row_label,
            transform=ax.transAxes,
            fontsize=21,
            rotation=90,
            va='center',
            ha='center',
            clip_on=False
        )


    # =========================================================
    # Shared P_CO / ppm Label
    #
    # Second vertical slot on the right.
    # =========================================================

    middle_right_ax = right_row_axes[1]

    middle_right_ax.text(
        1.30,
        0.5,
        r'$P_{\mathrm{CO}}$ / ppm',
        transform=middle_right_ax.transAxes,
        fontsize=21,
        rotation=90,
        va='center',
        ha='center',
        clip_on=False
    )


    # =========================================================
    # Horizontal Colorbar
    #
    # Positioned below the temperature axis label.
    # =========================================================

    cax = fig.add_axes([
        0.13,   # left
        0.065,  # bottom
        0.69,   # width
        0.025   # height
    ])

    cbar = fig.colorbar(
        final_im,
        cax=cax,
        orientation='horizontal'
    )

    cbar.set_label(
        colorbar_label,
        fontsize=21,
        labelpad=6
    )

    cbar.ax.tick_params(
        labelsize=21,
        width=0.8,
        length=3
    )

    cbar.outline.set_linewidth(
        0.8
    )


# ============================================================
# Prepare CO Coverage Data
# ============================================================

pivots_CO, vmin_CO, vmax_CO = prepare_data(
    df_all,
    z_keys,
    co_filter_values
)


# ============================================================
# Create Figure
# ============================================================

fig = plt.figure(
    figsize=(10, 8.9)
)


# ============================================================
# Grid
#
# Only the three heatmap columns are in GridSpec.
# The colorbar is placed independently below the figure.
# ============================================================

gs = GridSpec(
    3,
    3,
    figure=fig,
    width_ratios=[
        1,
        1,
        1
    ],
    wspace=0.0,
    hspace=0.0
)


# ============================================================
# Margins
# ============================================================

fig.subplots_adjust(
    left=0.13,
    right=0.82,
    bottom=0.23,
    top=0.92
)


# ============================================================
# Draw Heatmaps
# ============================================================

draw_block(
    pivots_CO,
    vmin_CO,
    vmax_CO,
    fig,
    gs,

    block_label=(
        r'$\ln\left('
        r'\frac{P_{\mathrm{H}_2}}'
        r'{P_{\mathrm{CO}}}'
        r'\right)$'
    ),

    colorbar_label=(
        r'$\theta_{\mathrm{CO*}}$ / ML'
    ),

    show_titles=True
)


# ============================================================
# Save PNG
# ============================================================

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight'
)




# ============================================================
# Show Figure
# ============================================================

plt.show()