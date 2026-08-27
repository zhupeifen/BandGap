# BandGap v2.0.3

A figures-and-metadata release. **No analysis, data, or reported value changes from v2.0.2** — every
number in the manuscript and Supporting Information is identical. What changes is that the figure
scripts now redraw the figures as they appear in the submitted paper, and that the archive records the
Zenodo DOIs.

## Why this release exists

The v2.0.2 tag was cut before three follow-up commits landed, so the archived v2.0.2 snapshot does not
reproduce four of the figures in the submitted manuscript, and its `README.md` and `CITATION.cff` do
not carry the Zenodo DOI that v2.0.2 was minted with — the commit recording that DOI came after the
tag. This release closes both gaps.

## Figure scripts brought in line with the manuscript

- `scripts/mechanism_causal.py` (Figure S4) plotted "Optimism (by-chem / random MAE)" against
  "Solid-solution density" and annotated the reference line "no optimism". All three terms were
  retired in the revision: the metric is the grouped/random MAE ratio, and a formula is a proxy for a
  composition family rather than evidence of solid-solution disorder. The axes now read
  **grouped/random MAE ratio** against **mean compositions per chemistry family**, matching the
  figure's own caption. The plotted values are unchanged.
- `scripts/make_schematic.py` (Figure 1) labelled the gate threshold **τ**, which the paper uses for
  the tolerance factor; the gate threshold is **θ**. The gate box was also narrower than the text
  inside it, so `[P ≥ θ] × ĝap` ran past its border, as did the two input boxes' contents. Fixed.
- `scripts/make_checklist_flowchart.py` (Figure 7) said "fractional share" where the paper says
  **fractional-formula share**.
- `scripts/graphical_abstract.py` aimed the two-stage arrow at the bar top, where the 0.585 value
  label sits, so the arrowhead landed on the digits; it now meets the bar's left shoulder. Its panel
  (a) title said "semiconductor error" where the paper says **non-zero-gap error**.

The regenerated renders in `plots/` are updated to match.

## Metadata

- `README.md` and `CITATION.cff` record the archive DOIs. `CITATION.cff` cites the **concept DOI**
  `10.5281/zenodo.22102802`, which always resolves to the latest release, so a release can never again
  name a version DOI that does not exist when the tag is cut. Per-version DOIs are listed in the
  README as they are minted.

## Verifying against the paper

Running the four scripts above on this release reproduces Figures 1, 7 and S4 and the graphical
abstract exactly as submitted. `scripts/reproduce_all.py` regenerates the core CPU tables and figures;
the accelerator-dependent CrabNet and graph-network runs use the SLURM scripts and environment files
in `scripts/`.

## Provenance

Supersedes v2.0.2 (version DOI `10.5281/zenodo.22114447`), which supersedes v2.0.1
(`10.5281/zenodo.22104943`). The concept DOI `10.5281/zenodo.22102802` resolves to the latest.
