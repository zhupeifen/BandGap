# Honest Evaluation of Machine-Learned Band Gaps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22102802.svg)](https://doi.org/10.5281/zenodo.22102802)

Code and data generation for:

> **Evaluation of Machine-Learned Band Gaps: Two-Stage Modeling and the Effects of
> Data Representation and Split Strategy**
> Evan Solecki and Peifen Zhu, University of Missouri.

This repository holds the reproducible code: every table and figure in the paper regenerates
from the scripts here. (The manuscript text is maintained separately and is not part of this
repository.) The work is a **methods / benchmarking** study (it does
not propose a new algorithm) showing how two routine evaluation choices inflate reported
band-gap accuracies across **six public datasets**, and what honest reporting changes.

## TL;DR — two findings

1. **Aggregate accuracy hides poor regression on semiconductors.** Band-gap datasets are
   *zero-inflated* (most candidates are metals), so aggregate accuracy mostly measures metal
   classification. On one dataset a single-stage model gets 0.93 aggregate accuracy while its
   error on the actual semiconductors is ~1 eV. Reporting **non-zero-material MAE** exposes this;
   a two-stage classify-then-regress model recovers much of the hidden performance.

2. **Split sensitivity depends on the sampled composition families.** Fractional-formula
   share is an operational source-formula proxy, not a definitive solid-solution label. In an
   exploratory six-dataset comparison, the two datasets with non-zero shares have the largest
   grouped/random MAE ratios, but the association is uncertain (Spearman rho = 0.85, exact
   two-sided permutation p = 0.067). The ratio measures a change of task from interpolation to
   held-out-chemistry extrapolation, not invalidity of the random-split estimate.

We also subject our own model to the same scrutiny: a leak-free **nested cross-validation**
shows most of the apparent two-stage gain comes from the *features*, not the architecture
(the method itself contributes 5–15%, significant only on composition-only data).

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/` | All analysis (CV, nested CV, ablation, grouped splits, learning curves, SHAP, the six-dataset runs) |
| `reports/` | A standalone six-dataset Word report |
| `plots/` | Figures (parity plots, generalization, split-sensitivity analysis, SHAP, learning curves) |
| `data/` | Downloaded datasets and stored analysis outputs; see the source records and retrieval scripts |

## Datasets (six)

| Dataset | n | Target | Non-zero | Source |
|---------|---|--------|----------|--------|
| Expt gap | 6,354 | experimental | 61% | matminer `expt_gap` |
| MP gap | 106,113 | PBE | 57% | matminer `mp_gap` |
| Wolverton oxides | 4,914 | PBE | 21% | matminer `wolverton_oxides` |
| Castelli perovskites | 18,928 | GLLB-SC | 4% | matminer `castelli_perovskites` |
| Double perovskites | 1,306 | GLLB-SC | 100% | matminer `double_perovskites_gap` |
| Tol-screened perovskites | 67,916 | HSE-level ML surrogate | 100% | Materials Data Facility, DOI 10.18126/dp3z-bp06 |

## Setup & reproduce

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt   # exact pinned versions
```

```bash
# regenerate every table and figure
.venv/Scripts/python.exe scripts/reproduce_all.py            # full
.venv/Scripts/python.exe scripts/reproduce_all.py --quick    # skip the slow CV sweeps
```

Scripts resolve data via `DATA_DIR = <repo>/data`, so they run from any working directory.
The retrieval scripts fetch the five matminer datasets. The Tol CSV is third-party CC-BY 4.0
data obtained from the Materials Data Facility and is not redistributed here.

## Key scripts

- `cv_evaluation.py` — non-zero metrics, baseline vs. two-stage (Tables 1–2)
- `nested_cv.py` — leak-free nested CV + paired significance (the feature-vs-method decomposition)
- `feature_ablation.py` / `feature_importance.py` — where the signal lives (ablation + SHAP)
- `cv_grouped_splits.py` / `redundancy_correlation.py` — grouped-split and fractional-formula analysis
- `learning_curve.py` — data-limitation curves
- `mp_gap_analysis.py` — 106k-material large-scale validation

## Citation

Cite the concept DOI [10.5281/zenodo.22102802](https://doi.org/10.5281/zenodo.22102802), which always
resolves to the latest release. See `CITATION.cff`.

Per-version DOIs, minted by Zenodo when each tag is pushed:

| Release | Version DOI |
|---|---|
| v2.0.3 | [10.5281/zenodo.22119613](https://doi.org/10.5281/zenodo.22119613) |
| v2.0.2 | [10.5281/zenodo.22114447](https://doi.org/10.5281/zenodo.22114447) |
| v2.0.1 | [10.5281/zenodo.22104943](https://doi.org/10.5281/zenodo.22104943) |

> Solecki, E.; Zhu, P. *Evaluation of Machine-Learned Band Gaps: Two-Stage Modeling and the Effects of
> Data Representation and Split Strategy.* Computational Materials Science, under revision
> (COMMAT-D-26-02664). Code: https://doi.org/10.5281/zenodo.22102802
