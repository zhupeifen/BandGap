# XGBoost Band-Gap Research

Predicting the electronic **band gap** of perovskites and oxides from elemental,
compositional, and structural features using gradient-boosted trees (XGBoost).

The central modeling theme is handling **zero-inflated** band-gap data: most
candidate materials are metals (gap = 0), and a single regressor trained on all
of them learns to predict ~0 and fails on the materials that actually have a gap.
The fix here is a **two-stage model** — a metal/non-metal classifier that gates a
regressor trained only on non-zero-gap materials.

## Setup

The scripts use a local virtual environment with the full materials-ML stack.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt # macOS/Linux
```

Run any script with the venv's Python, e.g.:

```bash
.venv/Scripts/python.exe scripts/binary_band_gap_improved.py
```

Each script resolves its data via `DATA_DIR = <repo>/data`, so it works regardless
of the current working directory.

### Data

The datasets live in `data/` and are **not tracked in git** (hundreds of MB). They
must be supplied separately. Scripts expect:

| File | Dataset | Target |
|------|---------|--------|
| `wolverton_oxides.json` | Wolverton oxides | `gap pbe` |
| `castelli_perovskites.json` | Castelli perovskites | `gap gllbsc` |
| `Tol_screened_ensemble_final.csv` | Tol-screened perovskites | `Band gap(HSE-mf1)` |
| `Elemental_properties.xlsx` | Per-element lookup table | — |

(The JSON files load via `matminer.utils.io.load_dataframe_from_json`.)

## Scripts

### Band-gap models
| Script | Dataset | Approach |
|--------|---------|----------|
| `scripts/binary_band_gap_prediction.py` | Wolverton oxides | Original two-stage (classifier × regressor on all data) |
| `scripts/binary_band_gap_improved.py` | Wolverton oxides | **Improved** two-stage + Magpie features; prints baseline vs. improved |
| `scripts/material_prediction.py` | Wolverton oxides | **Improved** two-stage; original single-stage kept inline as baseline |
| `scripts/dataset_test.py` | Castelli perovskites | **Improved** two-stage; original single-stage kept inline as baseline |
| `scripts/large_dataset_test.py` | Tol perovskites | Single regressor (no zeros to gate) + generalization check |
| `scripts/tol_generalization_test.py` | Tol perovskites | Leakage-controlled (grouped) split stress-test |

### Utilities / sandbox
- `scripts/sql_converter.py` — MySQL→SQLite dump converter (legacy, unused).
- `scripts/simple_test_scripts/` — unrelated XGBoost learning exercises (image
  classification, tic-tac-toe, tree visualization).
- `xgboost_test.py`, `test_code2.py` — AFT / custom-objective demos on toy data.

## The two-stage fix

For a zero-inflated dataset, the improved scripts:

1. Train a **metal / non-metal classifier** on all materials
   (`scale_pos_weight` handles class imbalance).
2. Train the **regressor only on non-zero-gap materials**, on a `log1p` target.
3. **Gate**: `prediction = P(non-metal ≥ threshold) × regressor`. The threshold is
   lowered to **0.25** to recover misgated small-gap materials — a true non-zero
   material gated to 0 is a guaranteed miss (min gap ≫ 0.07 eV tolerance).
4. Add **Magpie** elemental descriptors (matminer `ElementProperty`) derived from
   the formula, alongside the original structural/energetic features.

Each improved script reports the original single-stage **baseline** alongside the
improved model on the same split, and saves a **parity plot** to `plots/`.

> Note: `vbm`/`cbm` columns are deliberately excluded as features in
> `dataset_test.py` — since `gap = cbm − vbm`, using them would be target leakage.

## Results

Non-zero-material metrics, baseline → improved (same split per script):

| Script (dataset) | non-zero MAE (eV) | non-zero acc <0.07 eV |
|------------------|-------------------|------------------------|
| `binary_band_gap_improved.py` (Wolverton) | 0.558 → **0.375** (−33%) | 0.168 → 0.199 |
| `material_prediction.py` (Wolverton) | 0.451 → **0.354** (−22%) | 0.184 → 0.222 |
| `dataset_test.py` (Castelli) | 0.988 → **0.600** (−39%) | 0.034 → 0.090 |

The **Tol perovskite** model (`large_dataset_test.py`) has no zero gaps and is
already near-perfect (R² ≈ 0.9998). A grouped-split stress test confirms this is
real generalization, not dense-grid interpolation: holding out entire chemistries,
it still reaches **acc 0.95 / MAE 0.024 eV**.

### Caveats
- The hard `<0.07 eV` accuracy is near its data-imposed ceiling on the noisy
  datasets; **MAE/RMSE** are the more meaningful improvement metrics.
- Castelli is data-limited (only 735 non-zero materials of 18,928), so its
  absolute performance is bounded by sample size, not modeling choices.
- `Band gap(HSE-mf1)` is a smooth fitted surrogate, which is why it is so
  learnable; results would not transfer directly to noisy experimental gaps.
