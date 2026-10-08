"""Rebuild the editable architecture figure from the implemented CESM2 model.

Run with the project Python environment. All labels remain native SVG text;
CairoSVG provides the accompanying PNG and vector PDF.
"""

from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg


ROOT = Path(__file__).resolve().parent
STEM = ROOT / "fig_matched_architecture_20261007"
W, H = 1600, 1100
INK = "#24343F"
MUTED = "#61737F"
BLUE = "#527A96"
GREEN = "#4F827D"
OCHRE = "#A67D49"
PARTS = []


def txt(x, y, text, size=20, weight=400, color=INK, anchor="start", **attrs):
    extra = " ".join(f'{escape(str(k), quote=True)}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
    PARTS.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}" {extra}>{escape(text)}</text>')


def line(d, color=MUTED, width=2.2, arrow=True, dashed=False, bridge=False):
    if bridge:
        PARTS.append(f'<path d="{d}" fill="none" stroke="white" stroke-width="8"/>')
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    marker = f' marker-end="url(#{color[1:]})"' if arrow else ""
    PARTS.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash}{marker}/>')


def box(x, y, width, height, title, rows, color=BLUE, fill="#F4F8FA", row_size=19):
    PARTS.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')
    PARTS.append(f'<path d="M{x+16} {y+54} H{x+width-16}" stroke="{color}" stroke-opacity="0.25"/>')
    txt(x+18, y+34, title, 21, 600)
    for i, row in enumerate(rows):
        txt(x+18, y+84+i*29, row, row_size, color=INK)


PARTS.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="figure-title figure-desc">')
PARTS.append('<title id="figure-title">OI-anchored multimodal ocean reconstruction: implemented candidate architecture</title>')
PARTS.append('<desc id="figure-desc">Per-depth Argo measurements and optional synthetic surface fields are encoded into a shared latent Transformer. An independent coordinate query decoder predicts residual and marginal Gaussian scale. In parallel, frozen spherical optimal interpolation and exact-depth local innovations provide numerical evidence for an anchored temperature and salinity prediction. The local Transformer variant updates each query representation only from its own neighborhood.</desc>')
PARTS.append(f'<defs><style>text{{font-family:"Liberation Sans",Arial,sans-serif;}}</style>')
for color in (INK, MUTED, BLUE, GREEN, OCHRE):
    PARTS.append(f'<marker id="{color[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="{color}"/></marker>')
PARTS.append('</defs><rect width="1600" height="1100" fill="white"/>')

txt(45, 57, "OI-anchored multimodal ocean reconstruction", 34, 600)
txt(45, 93, "Implemented candidate system · architecture selection remains validation-driven", 20, color=MUTED)
txt(45, 145, "A", 22, 700, BLUE)
txt(75, 145, "Learned representation and independent coordinate queries", 21, 600, BLUE)

# Draw routes first so that boxes do not conceal any endpoint.
line("M300 286 H355", BLUE)
line("M600 286 H655", BLUE)
line("M925 286 H980", BLUE)
line("M1250 286 H1310", BLUE)

# Optional surface information also enters the query directly, bypassing latent.
line("M240 415 V533 H980", BLUE, dashed=True)
txt(350, 519, "Optional surface features at the requested location", 17, color=BLUE)
line("M1115 455 V415", BLUE)

# Exact numerical innovations leave the observation input before compression.
line("M45 272 H20 V744 H45", OCHRE)
line("M300 749 H355", OCHRE)
line("M172 885 V925 H790 V885", OCHRE)
txt(230, 913, "Same original depth values and masks", 17, color=OCHRE)

# Frozen OI enters both query projection and the final mean, as in decode().
line("M600 735 H625 V574 H980", OCHRE)
txt(646, 560, "OI correction a − b", 17, color=OCHRE)
line("M478 885 V961 H1060 V885", OCHRE)
txt(650, 949, "Frozen numerical anchor a", 18, color=OCHRE)

# Query coordinates control the OI depth and both neighbor lookups.
line("M1115 605 V635 H470 V675", MUTED, width=1.9, bridge=True)
line("M790 635 V675", MUTED, width=1.9)
PARTS.append(f'<circle cx="790" cy="635" r="3" fill="{MUTED}"/>')
txt(820, 622, "Query location + depth", 17, color=MUTED)

# Direct original numerical local evidence produces a candidate and statistics.
line("M925 751 H980", OCHRE)
txt(940, 737, "c, s", 17, color=OCHRE, anchor="middle")
line("M900 675 V649 H1280 V366 H1310", GREEN)
txt(1086, 665, "Local statistics s", 17, color=GREEN)

# Learned residual and scale heads receive query h and local statistics.
line("M1430 415 V675", BLUE)
txt(1447, 553, "r(h), σ", 19, color=BLUE)

box(45, 180, 255, 235, "Available observations", [], color=MUTED, fill="#F7F9FA")
txt(63, 260, "Argo T/S values + masks", 20, 600)
txt(63, 291, "Position, depth and month", 19)
PARTS.append('<path d="M63 314 H282" stroke="#D1DBDF"/>')
txt(63, 346, "Optional surface fields", 19, 600, BLUE)
txt(63, 376, "SST / SSS / steric SSH", 19)

box(355, 180, 245, 235, "Observation encoder", [
    "One token per depth",
    "T/S innovation MLPs",
    "Coordinate embeddings",
    "Separate surface tokens",
    "Modality-balanced prior",
])
box(655, 180, 270, 235, "Shared ocean latent", [
    "Observation cross-attention",
    "Latent self-attention",
    "Dense FFN or Soft MoE",
    "96 latents · width 192*",
    "MoE: 4 experts × 2 slots",
], color=GREEN, fill="#F2F8F6", row_size=18)
box(980, 180, 270, 235, "Independent decoder", [
    "Query → latent attention",
    "Per-query FFN",
    "No query–query attention",
    "Query representation h",
    "Encode once; query in chunks",
], row_size=18)
box(1310, 180, 245, 235, "Learned output heads", [
    "T/S residual r(h)",
    "Gaussian scale σ(h, s)",
    "Scale uses local coverage",
    "and innovation variance",
    "Marginal uncertainty",
], row_size=18)
box(980, 455, 270, 150, "Requested coordinate", [
    "Position, depth and month",
    "Optional surface features",
    "Background b; OI term a − b",
], color=MUTED, fill="#F7F9FA", row_size=18)

txt(45, 648, "B", 22, 700, OCHRE)
txt(75, 648, "Original numerical evidence", 21, 600, OCHRE)

box(45, 675, 255, 210, "Exact-depth innovations", [
    "Original T/S values − b",
    "Separate validity masks",
    "Read before token encoding",
    "No depth-band averaging",
], color=OCHRE, fill="#FAF7F1", row_size=18)
box(355, 675, 245, 210, "Frozen spherical OI", [
    "Same-depth T/S separately",
    "20 or 40 valid neighbors",
    "Gaussian covariance solve",
    "Numerical mean a",
], color=OCHRE, fill="#FAF7F1", row_size=18)
box(655, 675, 270, 210, "Exact-depth local path", [], color=OCHRE, fill="#FAF7F1")
txt(673, 756, "32 nearest source profiles", 19)
txt(673, 782, "Raw innovations + masks", 19)
txt(673, 808, "Learned weights → c; statistics s", 17)
PARTS.append('<path d="M673 821 H907" stroke="#DCD0BF"/>')
txt(673, 843, "Local Transformer option:", 17, 600, GREEN)
txt(673, 868, "neighbor attention updates h†", 17, color=GREEN)

box(980, 675, 575, 210, "Anchored T/S prediction", [], color=BLUE, fill="#F4F8FA")
txt(1267, 763, "μ = a + r(h) + g(h, s) ⊙ [c − (a − b)]", 23, 500, anchor="middle", style="font-family:DejaVu Sans,Arial,sans-serif")
txt(1000, 800, "Signed gate g = tanh(MLP); zero when no valid local evidence", 18)
txt(1000, 835, "Temperature / salinity mean and marginal Gaussian uncertainty", 18)
txt(1000, 865, "Rescale μ and σ using training-only depth / variable statistics", 17, color=MUTED)

# A restrained legend and explicit scope prevent stronger claims than the run.
line("M47 999 H85", BLUE, arrow=False)
txt(96, 1006, "Learned representation", 17, color=MUTED)
line("M322 999 H360", OCHRE, arrow=False)
txt(371, 1006, "Original numerical evidence", 17, color=MUTED)
line("M639 999 H677", GREEN, arrow=False)
txt(688, 1006, "Local statistics / optional contextualization", 17, color=MUTED)
line("M1121 999 H1159", BLUE, arrow=False, dashed=True)
txt(1170, 1006, "Optional surface branch", 17, color=MUTED)
txt(45, 1042, "CESM2: monthly reconstruction · 20 scored depths (5–984.7 m) · b = 0 in standardized-anomaly space", 18, color=MUTED)
txt(45, 1070, "* Dense64 uses 64 latents / width 64. All Soft MoE experts execute. † Local Transformer is registered in the Argo-only arm.", 16, color=MUTED)
PARTS.append('</svg>')

svg = "\n".join(PARTS)
ET.fromstring(svg)
STEM.with_suffix(".svg").write_text(svg, encoding="utf-8")
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(STEM.with_suffix(".png")), output_width=W*2, output_height=H*2)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(STEM.with_suffix(".pdf")))

caption = """Figure. Implemented candidate architecture for CESM2 multimodal ocean reconstruction. Per-depth Argo temperature and salinity innovations, validity masks, coordinates, and optional synthetic SST/SSS/steric-SSH features are encoded into a shared latent Transformer. Each block reads the observation memory, performs latent self-attention, and uses a Dense FFN or Soft MoE FFN. An independent coordinate query decoder produces a query representation h, with no attention between target queries. Optional surface features also enter the query directly. Frozen spherical optimal interpolation (OI) provides the numerical anchor a and its correction a − b to the query projection. A parallel local path reads the original numerical innovations at the exact requested depth from 32 nearest source profiles, forms a learned weighted candidate c, and computes coverage and innovation-variance statistics s. The anchored mean is μ = a + r(h) + g(h, s) ⊙ [c − (a − b)], with a signed tanh gate. Residual and gate heads are initialized to zero, so the initial mean equals OI. The bounded marginal-Gaussian scale head reads h and s. In the Local Transformer variant, attention operates within each query's own local neighborhood and updates h before residual and scale prediction; this variant is registered only for Argo-only reconstruction. Soft MoE executes all four experts with two slots each; it is not sparse top-k routing. The large registered configuration has 96 latents and width 192; Dense64 has 64 latents and width 64. Here b = 0 in standardized-anomaly space, with no separate satellite first-guess MLP. Both mean and scale are converted to physical units with training-only statistics. This monthly, retrospective experiment is trained and scored at 20 fixed depths from 5 to 984.7 m; the diagram does not assert validated forecasting or arbitrary-depth interpolation. It depicts implemented candidates, not a preselected final backbone.

Source: src/ocean_tokenizer/latent_ocean.py; src/ocean_tokenizer/synthetic_latent_experiment.py; reports/synthetic/matched_architecture_protocol_20261007.md. Rebuild: reports/synthetic/fig_matched_architecture_20261007.py. SVG labels and paths are editable vector elements. Generated 2026-10-07.
"""
STEM.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
print(f"Wrote {STEM}.svg/.png/.pdf/.caption.txt")
