from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


# ============================================================
# USER SETTINGS
# ============================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_POSCAR = (
    BASE_DIR
    / "POSCAR"
)

OUTPUT_IMAGE = BASE_DIR / "SPATIAL_ANALYSIS_EXAMPLE.png"

PAIR_CUTOFF = 1.71       # Angstrom; O-H neighbor cutoff
ISLAND_CUTOFF = 2.83     # Angstrom; O-O DBSCAN epsilon
MIN_ISLAND_SIZE = 3      # Clusters smaller than this are not treated as islands
FIGURE_DPI = 500

O_SYMBOL = "O"
H_SYMBOL = "H"


# ============================================================
# PERIODIC GEOMETRY
# ============================================================
def wrap_xy(atoms: Atoms) -> Atoms:
    """Return a copy wrapped into the primary cell along x and y."""
    wrapped = atoms.copy()
    wrapped.set_pbc((True, True, False))

    scaled = wrapped.get_scaled_positions(wrap=False)
    scaled[:, 0] %= 1.0
    scaled[:, 1] %= 1.0
    wrapped.set_scaled_positions(scaled)

    return wrapped


def minimum_image_xy_vector(
    position_i: np.ndarray,
    position_j: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Return the minimum-image x-y displacement vector from i to j."""
    xy_matrix = np.column_stack((cell[0, :2], cell[1, :2]))
    delta_xy = np.asarray(position_j[:2]) - np.asarray(position_i[:2])

    delta_fractional = np.linalg.solve(xy_matrix, delta_xy)
    delta_fractional -= np.rint(delta_fractional)

    return xy_matrix @ delta_fractional


def periodic_xy_images(
    positions_xy: np.ndarray,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a 3 x 3 set of periodic x-y images.

    Returns
    -------
    image_positions:
        Cartesian x-y positions of all images.
    original_local_indices:
        For each image point, the local index of its primary-cell atom.
    """
    shifts = []
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            shifts.append(da * cell[0, :2] + db * cell[1, :2])

    image_positions = []
    original_local_indices = []

    for shift in shifts:
        image_positions.append(positions_xy + shift)
        original_local_indices.extend(range(len(positions_xy)))

    return (
        np.vstack(image_positions),
        np.asarray(original_local_indices, dtype=int),
    )


# ============================================================
# O-H PAIR DETECTION WITH KDTREE
# ============================================================
def detect_O_H_pairs(
    atoms: Atoms,
    cutoff: float = PAIR_CUTOFF,
    o_symbol: str = O_SYMBOL,
    h_symbol: str = H_SYMBOL,
) -> list[dict]:
    """Detect periodic O-H pairs using scipy.spatial.KDTree."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.cell)

    o_indices = np.asarray(
        [i for i, symbol in enumerate(symbols) if symbol == o_symbol],
        dtype=int,
    )
    h_indices = np.asarray(
        [i for i, symbol in enumerate(symbols) if symbol == h_symbol],
        dtype=int,
    )

    if len(o_indices) == 0 or len(h_indices) == 0:
        return []

    h_positions_xy = positions[h_indices, :2]
    h_image_positions, h_image_to_local = periodic_xy_images(
        h_positions_xy,
        cell,
    )

    h_tree = KDTree(h_image_positions)
    pairs: list[dict] = []

    for o_index in o_indices:
        o_position_xy = positions[o_index, :2]
        image_hits = h_tree.query_ball_point(o_position_xy, r=cutoff)

        # A physical H atom may occur more than once among the image hits.
        # Keep only its closest periodic image for this O atom.
        closest_hit_by_h: dict[int, float] = {}

        for image_hit in image_hits:
            h_local_index = int(h_image_to_local[image_hit])
            h_index = int(h_indices[h_local_index])
            distance = float(
                np.linalg.norm(h_image_positions[image_hit] - o_position_xy)
            )

            previous_distance = closest_hit_by_h.get(h_index)
            if previous_distance is None or distance < previous_distance:
                closest_hit_by_h[h_index] = distance

        for h_index, distance in closest_hit_by_h.items():
            if distance <= cutoff + 1.0e-12:
                pairs.append(
                    {
                        "o_index": int(o_index),
                        "h_index": h_index,
                        "distance": distance,
                    }
                )

    pairs.sort(key=lambda pair: (pair["o_index"], pair["h_index"]))
    return pairs


# ============================================================
# O ISLAND DETECTION WITH DBSCAN
# ============================================================
def periodic_O_distance_matrix(
    o_positions: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Build the periodic x-y distance matrix used by DBSCAN."""
    n_o = len(o_positions)
    distance_matrix = np.zeros((n_o, n_o), dtype=float)

    for i in range(n_o):
        for j in range(i + 1, n_o):
            vector = minimum_image_xy_vector(
                o_positions[i],
                o_positions[j],
                cell,
            )
            distance = float(np.linalg.norm(vector))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix


def detect_O_islands(
    atoms: Atoms,
    cutoff: float = ISLAND_CUTOFF,
    min_island_size: int = MIN_ISLAND_SIZE,
    o_symbol: str = O_SYMBOL,
) -> list[dict]:
    """
    Detect periodic O islands using sklearn.cluster.DBSCAN.

    DBSCAN uses a precomputed minimum-image distance matrix. Setting
    min_samples=1 makes every connected O-O group a DBSCAN cluster. Clusters
    smaller than min_island_size are then excluded from the island list.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.cell)

    o_indices = np.asarray(
        [i for i, symbol in enumerate(symbols) if symbol == o_symbol],
        dtype=int,
    )

    if len(o_indices) == 0:
        return []

    o_positions = positions[o_indices]
    distance_matrix = periodic_O_distance_matrix(o_positions, cell)

    labels = DBSCAN(
        eps=cutoff,
        min_samples=1,
        metric="precomputed",
    ).fit_predict(distance_matrix)

    islands: list[dict] = []

    for cluster_label in sorted(set(labels)):
        local_members = np.where(labels == cluster_label)[0]

        if len(local_members) < min_island_size:
            continue

        atom_indices = sorted(int(o_indices[i]) for i in local_members)
        islands.append(
            {
                "size": len(atom_indices),
                "atom_indices": atom_indices,
            }
        )

    islands.sort(
        key=lambda island: (-island["size"], island["atom_indices"][0])
    )
    return islands


# ============================================================
# REPORTING
# ============================================================
def print_results(pairs: list[dict], islands: list[dict]) -> None:
    """Print pair count and the count of islands at each island size."""
    print(f"\nTotal O-H pairs : {len(pairs)}")
    print(f"Total O islands  : {len(islands)}")

    print("\nIsland sizes and counts")
    if not islands:
        print("No islands detected.")
        return

    size_counts = Counter(island["size"] for island in islands)

    for size in sorted(size_counts):
        count = size_counts[size]
        island_word = "island" if count == 1 else "islands"
        print(f"Size {size:>4d} : {count:>4d} {island_word}")


# ============================================================
# VISUALIZATION
# ============================================================
def cell_polygon(cell: np.ndarray) -> np.ndarray:
    """Return the x-y polygon of the primary simulation cell."""
    origin = np.zeros(2)
    a_vector = cell[0, :2]
    b_vector = cell[1, :2]
    return np.vstack((origin, a_vector, a_vector + b_vector, b_vector, origin))


def visualize_pairs_and_islands(
    atoms: Atoms,
    pairs: list[dict],
    islands: list[dict],
    output_path: str | Path = OUTPUT_IMAGE,
    dpi: int = FIGURE_DPI,
) -> None:
    """
    Make the requested simple top-view visualization.

    O in a DBSCAN island : red
    Other O              : black
    H in an O-H pair     : blue
    Other H              : white
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    cell = np.asarray(atoms.cell)

    o_indices = [i for i, symbol in enumerate(symbols) if symbol == O_SYMBOL]
    h_indices = [i for i, symbol in enumerate(symbols) if symbol == H_SYMBOL]

    island_o_indices = {
        atom_index
        for island in islands
        for atom_index in island["atom_indices"]
    }
    paired_h_indices = {pair["h_index"] for pair in pairs}

    nonisland_o = [i for i in o_indices if i not in island_o_indices]
    island_o = [i for i in o_indices if i in island_o_indices]
    unpaired_h = [i for i in h_indices if i not in paired_h_indices]
    paired_h = [i for i in h_indices if i in paired_h_indices]

    fig, ax = plt.subplots(figsize=(8, 8))

    polygon = cell_polygon(cell)
    ax.plot(
        polygon[:, 0],
        polygon[:, 1],
        color="black",
        linewidth=1.5,
        zorder=1,
    )

    if nonisland_o:
        ax.scatter(
            positions[nonisland_o, 0],
            positions[nonisland_o, 1],
            s=90,
            color="#A23B72",
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )

    if island_o:
        ax.scatter(
            positions[island_o, 0],
            positions[island_o, 1],
            s=90,
            color="#A23B72",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
        )

    if unpaired_h:
        ax.scatter(
            positions[unpaired_h, 0],
            positions[unpaired_h, 1],
            s=70,
            color="none",
            edgecolors="none",
            linewidths=1.0,
            zorder=5,
        )

    if paired_h:
        ax.scatter(
            positions[paired_h, 0],
            positions[paired_h, 1],
            s=70,
            color="#56B4E9",
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
        )

    corners = polygon[:-1]
    x_span = float(np.ptp(corners[:, 0]))
    y_span = float(np.ptp(corners[:, 1]))
    margin = 0.05 * max(x_span, y_span)

    ax.set_xlim(corners[:, 0].min() - margin, corners[:, 0].max() + margin)
    ax.set_ylim(corners[:, 1].min() - margin, corners[:, 1].max() + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect periodic O-H pairs with KDTree and periodic O islands "
            "with DBSCAN."
        )
    )
    parser.add_argument(
        "poscar",
        nargs="?",
        default=DEFAULT_POSCAR,
        help=f"Input structure readable by ASE (default: {DEFAULT_POSCAR})",
    )
    parser.add_argument(
        "--pair-cutoff",
        type=float,
        default=PAIR_CUTOFF,
        help=f"O-H x-y cutoff in Angstrom (default: {PAIR_CUTOFF})",
    )
    parser.add_argument(
        "--island-cutoff",
        type=float,
        default=ISLAND_CUTOFF,
        help=f"DBSCAN epsilon in Angstrom (default: {ISLAND_CUTOFF})",
    )
    parser.add_argument(
        "--min-island-size",
        type=int,
        default=MIN_ISLAND_SIZE,
        help=f"Minimum island size (default: {MIN_ISLAND_SIZE})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_IMAGE,
        help=f"Output image name (default: {OUTPUT_IMAGE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path = Path(args.poscar)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if args.pair_cutoff <= 0 or args.island_cutoff <= 0:
        raise ValueError("Both cutoffs must be positive.")
    if args.min_island_size < 1:
        raise ValueError("--min-island-size must be at least 1.")

    atoms = read(input_path)
    atoms = wrap_xy(atoms)

    pairs = detect_O_H_pairs(
        atoms,
        cutoff=args.pair_cutoff,
    )
    islands = detect_O_islands(
        atoms,
        cutoff=args.island_cutoff,
        min_island_size=args.min_island_size,
    )

    print_results(pairs, islands)

    visualize_pairs_and_islands(
        atoms,
        pairs,
        islands,
        output_path=args.output,
    )

    print(f"\nSaved visualization: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()