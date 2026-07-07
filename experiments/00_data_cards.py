"""Write data cards for the three standardized datasets + the common grid card."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
from ocean_tokenizer import data, config as C

os.makedirs(C.REPORTS, exist_ok=True)

names = {"woa23": "data_card_woa23.md",
         "cesm2": "data_card_cesm2.md",
         "cesm2_le_full": "data_card_cesm2_le.md"}

for name, fn in names.items():
    card = data.data_card(name)
    with open(os.path.join(C.REPORTS, fn), "w") as f:
        f.write(card)
    print("wrote", fn)

grid = data.CommonGrid()
with open(os.path.join(C.REPORTS, "common_grid.md"), "w") as f:
    f.write("# Common Analysis Grid\n\n")
    f.write(f"- ground truth source: `{C.GT_SOURCE}`\n")
    f.write(f"- lat: {grid.nlat} pts ({grid.lat.min():.1f}..{grid.lat.max():.1f})\n")
    f.write(f"- lon: {grid.nlon} pts ({grid.lon.min():.1f}..{grid.lon.max():.1f}, {C.LON_CONVENTION})\n")
    f.write(f"- depth: {grid.ndepth} target levels (m): "
            + ", ".join(f"{d:.0f}" for d in grid.depth) + "\n")
    f.write(f"- ocean fraction: {grid.ocean.mean():.3f}\n")
print("wrote common_grid.md")
print(grid)
