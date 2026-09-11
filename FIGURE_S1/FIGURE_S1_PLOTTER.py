import pickle
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PKL_PATH = BASE_DIR / "FIGURE_S1_CO_BE_CHARGE.pkl"
OUTPUT_PATH = BASE_DIR / "FIGURE_S1.png"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

x = data["charge_per_surface_pt"]
y = data["co_binding_energy_eV"]
highlight = data["highlight_indices"]

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(
    x,
    y,
    "-o",
    linewidth=2.5,
    markersize=8
)

ax.scatter(
    [x[i] for i in highlight],
    [y[i] for i in highlight],
    s=220,
    color="red",
    zorder=3
)

ax.set_xlabel("Charge per Surface Pt Atom")
ax.set_ylabel("CO Binding Energy (eV)")

ax.set_xlim(-0.183, 0.183)
ax.set_ylim(-2.92, -1.18)

ax.set_xticks([-0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15])
ax.set_yticks([-1.2, -1.4, -1.6, -1.8, -2.0, -2.2, -2.4, -2.6, -2.8])

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
