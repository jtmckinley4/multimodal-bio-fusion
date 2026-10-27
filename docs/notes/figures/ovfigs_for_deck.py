"""Render the manuscript's own figures for the slide deck.

The deck should show the same figures as the Overleaf, not new ones. These are vector PDFs in
docs/notes/figures/figures_overleaf/, written for a 6.5 inch text column; on a 13.3 inch slide
they sit at roughly their native size, so the type lands around 7 to 8 point. The only change is
the page background, #fcfcfb in the manuscript and #f6f5f4 on the slide canvas, so the figure does
not read as a pasted-in rectangle.

    python ovfigs_for_deck.py <figures_overleaf dir> <output dir>
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

SRC = sys.argv[1]
OUT = sys.argv[2]
DPI = 300
PAGE_BG = (252, 252, 251)      # #fcfcfb, the manuscript's surface
SLIDE_BG = (246, 245, 244)     # #f6f5f4, the deck's canvas

# name -> crop as fractions (left, right) or (left, right, top, bottom), None for the whole figure
WANTED = {
    "r34_fusion_forest": None,
    "r34_mrfp": None,
    "r34_lucaone": None,
    # panel b only, from the start of its row labels, without the panel title the slide repeats
    "r34_gated_leak": (0.468, 1.0, 0.085, 1.0),
    "r34_phaseB": None,
    "r34_reproduction": None,
    "r34_halflife": None,
    "r34_isoformer": None,
}

os.makedirs(OUT, exist_ok=True)
for name, crop in WANTED.items():
    pdf = os.path.join(SRC, name + ".pdf")
    if not os.path.exists(pdf):
        print(f"{name}: missing, skipped")
        continue
    stem = os.path.join(OUT, name)
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-f", "1", "-l", "1", pdf, stem], check=True)
    rendered = next(p for p in sorted(os.listdir(OUT)) if p.startswith(name + "-") and p.endswith(".png"))
    im = Image.open(os.path.join(OUT, rendered)).convert("RGB")
    if crop:
        w, h = im.size
        top, bottom = (crop[2], crop[3]) if len(crop) == 4 else (0.0, 1.0)
        im = im.crop((int(crop[0] * w), int(top * h), int(crop[1] * w), int(bottom * h)))
    a = np.array(im)
    # the page background is flat, so an exact match is enough; antialiased edges keep their blend
    mask = np.all(np.abs(a.astype(int) - np.array(PAGE_BG)) <= 1, axis=-1)
    a[mask] = SLIDE_BG
    Image.fromarray(a).save(stem + ".png")
    os.remove(os.path.join(OUT, rendered))
    print(f"{name}: {im.size[0]} x {im.size[1]} px, aspect {im.size[0] / im.size[1]:.3f}, "
          f"{mask.mean() * 100:.0f}% background recolored")
