"""
Pad each exported figure to an exact 4:3 aspect ratio with transparent margins,
so the figures display as 4:3 in the manuscript without distorting the plot or text.
Run after the MATLAB run_all export and after make_schematic.py.

    .venv/Scripts/python.exe scripts/pad_figures.py
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "manuscript" / "figures_and_data" / "figures" / "matlab" / "export"
DEFAULT_TARGET = 4 / 3
# Per-figure target aspect ratio (width / height); default is 4:3.
TARGETS = {"Fig1_feature_importance.png": 1 / 1.2}   # SHAP figure is 1:1.2 (portrait)


def pad_to(path, target):
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    ar = w / h
    if abs(ar - target) < 0.004:
        return False
    if ar < target:                       # too narrow -> add width
        new_w, new_h = round(h * target), h
    else:                                 # too wide -> add height
        new_w, new_h = w, round(w / target)
    canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 0))
    canvas.paste(im, ((new_w - w) // 2, (new_h - h) // 2), im)
    canvas.save(path)
    return True


def pad_to_43(path):
    return pad_to(path, TARGETS.get(path.name, DEFAULT_TARGET))


def main():
    targets = sorted(EXPORT.glob("*.png"))
    schematic = ROOT / "plots" / "pipeline_schematic.png"
    if schematic.exists():
        targets.append(schematic)
    for p in targets:
        changed = pad_to_43(p)
        im = Image.open(p)
        print(f"  {p.name:28s} -> {im.size[0]}x{im.size[1]} (ar {im.size[0]/im.size[1]:.3f})"
              f"{'  [padded]' if changed else ''}")


if __name__ == "__main__":
    main()
