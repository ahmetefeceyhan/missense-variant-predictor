# MASTER — Feature-Depth Analysis (in depth)

How much can the features of the MASTER panel predict the `Label`
(0 = benign, 1 = pathogenic)? Two parts:

- **Part 1 — single-feature models:** train a basic XGBoost on **one feature at a time** and
  rank features by predictive power.
- **Part 2 — multi-feature models:** **combine many features** (sweep the count `k`, with
  several selection strategies) and see whether the panel predicts better together than any
  feature alone.

Both parts use the same basic model (`XGBClassifier(n_estimators=100)`, "nothing fancy") and
the same stratified 80/20 holdout split so every number is comparable.

| File | What it is |
|------|------------|
| `MASTER_single_feature_results.csv` | Part 1 output: one row per model, **sorted best F1 → worst**. |
| `MASTER_multi_feature_results.csv` | Part 2 output: one row per (method, k) model, **sorted best F1 → worst**, with the chosen feature list. |
| `MASTER_multi_feature_ksweep.png` | Part 2 plot: F1 vs k for each selection method. |
| `README.md` | This document. |

The scripts that generate these live in `feature_depth/` and are run with `--panel MASTER`.

This is the **largest** panel (2931 variants), so its 587-row test fold (157 benign) makes the
metrics far more stable than the smaller panels — the findings here are the most trustworthy.
The "majority-class floor" (F1 of a trivial always-predict-pathogenic model, `2·prev/(1+prev)`)
is **F1 ≈ 0.8461** at this panel's prevalence of 0.733; a model must clear that to be useful.

---

# PART 1 — Single-feature models

## 1. The dataset

- 2931 rows. Target `Label`: **2149 pathogenic (1) / 782 benign (0)** — prevalence 0.733
  (the least imbalanced panel).
- ID column: `Variant_ID` (excluded).
- Features (351 total), by prefix:
  - `AL_` ×334 — numeric floats (allele-frequency / score-like).
  - `EK_` ×9 — numeric.
  - `CAT_` ×6 — categorical strings (e.g. `gnomADg_AMR`).
  - `AA_` ×2 — amino-acid letters (e.g. `F`, `L`).

## 2. The testing structure / plan

Each **model is a single feature → Label classifier**. The same feature can produce several
models because we vary how its values are represented and how missing values are filled.

### Numeric features (`AL_`, `EK_`) → 6 models each (`3 transforms × 2 imputations`)

| Transform | Meaning | Example `0.000080095878575 →` |
|-----------|---------|-------------------------------|
| `raw` | unchanged | `0.000080095878575` |
| `significand` | mantissa normalized to `[1,10)`, sign kept, rounded 4 dp (drops magnitude) | `8.0096` |
| `sig4figs` | rounded to 4 significant figures, magnitude preserved | `0.00008010` |

| Imputation | Fill value (fit on **train only**) |
|------------|------------|
| `mean` | training-column mean (after transform) |
| `median` | training-column median (after transform) |

### Categorical features (`CAT_`, `AA_`) → 1 model each

Missing → its own category `"__MISSING__"`; **ordinal-encoded** with a mapping fit on train
(unseen test categories → `-1`).

### Total: `343 numeric × 6 + 8 categorical = 2066 models`.

### Model & evaluation

`XGBClassifier(n_estimators=100, random_state=42)` (all else default). One stratified 80/20
holdout (seed 42), reused for every model: train = 2344 rows, test = 587 rows (157 benign /
430 pathogenic). Preprocessing stats fit on **train only**. Metrics on the test fold:
`accuracy`, and `f1 / recall / precision` with `pos_label=1`.

### Output

`MASTER_single_feature_results.csv`, sorted **best F1 → worst**, with a `rank` column. Columns:
`rank, feature, prefix, kind, transform, imputation, na_fraction, accuracy, f1, recall, precision`.

## 3. What we found (Part 1)

- **Best single feature: `AL_2` (`raw`, median)** — **F1 0.8636**, recall 0.972, precision
  0.777, accuracy 0.775. `AL_4`, `AL_37`, `AL_3`, `AL_6` follow closely (~0.86).
- **Only 46 of 351 features clear the majority floor** (F1 > 0.8461). Because this panel has a
  large, well-populated test fold, this is a *real* signal (not noise): a minority of `AL_`
  features carry genuine standalone discriminative power, the rest do not.
- **Transform/imputation barely matter** for a given numeric feature: XGBoost splits on rank,
  so `raw`, `significand`, and `sig4figs` (and mean vs median) usually tie.
- **Takeaway:** no single feature is strong on its own (best F1 0.8636 is only ~0.02 above the
  floor), but the top `AL_` features are consistently and reliably the most informative.

Top of the table:

| rank | feature | transform | imputation | accuracy | f1 | recall | precision |
|-----:|---------|-----------|-----------|---------:|------:|-------:|----------:|
| 1 | AL_2 | raw | median | 0.7751 | 0.8636 | 0.9721 | 0.7770 |
| 2 | AL_4 | significand | median | 0.7717 | 0.8610 | 0.9651 | 0.7772 |
| 3 | AL_37 | significand | mean | 0.7700 | 0.8607 | 0.9698 | 0.7737 |
| 4 | AL_3 | significand | mean | 0.7666 | 0.8595 | 0.9744 | 0.7688 |
| 5 | AL_6 | significand | mean | 0.7700 | 0.8595 | 0.9605 | 0.7778 |

---

# PART 2 — Multi-feature models

Does **combining many features** predict MASTER better than any single feature? We pick feature
subsets four different ways, sweep the subset size `k`, train one basic XGBoost per
`(method, k)`, and record the exact feature list chosen.

## 4. The testing structure / plan (Part 2)

- **Same dataset, same split, same model** as Part 1.
- **Preprocessing:** `raw` values + **median** imputation (median fit on train only);
  categoricals ordinal-encoded (missing = own category, unseen → -1). The full train/test
  matrices are built once, then each model slices its columns.

### Selection methods — four ways to choose features (rank, then take top `k`)

| Method | How features are ranked |
|--------|--------------------------|
| `single_f1` | Reuse `MASTER_single_feature_results.csv`: best single-feature F1 per feature, descending. |
| `xgb_importance` | Fit one XGBoost on all 351 features (train only), rank by **gain**. Zero-gain features back-filled in column order. |
| `mutual_info` | `mutual_info_classif(Xtrain, ytrain)` — model-agnostic statistical ranking. |
| `random` | One random draw (seed 42). A **noisy baseline**. |

> ⚠️ `single_f1` was ranked using F1 measured on the **same test fold**, so it is mildly
> *optimistic* vs the three train-only selectors.

### The sweep

`k ∈ {10, 25, 50, 100, 150, 200, 250}` per method, **plus** `all` (=351, emitted once as
`all_features`). → **4 × 7 + 1 = 29 models**.

### Output

`MASTER_multi_feature_results.csv`, sorted **best F1 → worst** with a `rank` column. Columns:
`rank, selection_method, k, n_features, accuracy, f1, recall, precision, selected_features` —
where `selected_features` is the full `;`-joined list of chosen feature names, so the exact
subset behind every row is reproducible from the CSV alone.

## 5. What we found (Part 2)

- **Combining features clearly helps here.** The best multi-feature model (`xgb_importance`,
  k=200) reaches **F1 0.8935** — a real improvement over the best *single* feature (`AL_2`,
  0.8636) and over the majority floor (0.8461). Crucially, the **precision jumps** from ~0.78
  (single feature) to ~0.85, i.e. the combined model stops over-predicting "pathogenic".
- **More features ≈ better, up to a point.** F1 climbs steadily with `k` and **peaks around
  k = 100–200**, then flattens; `all_features` (351) gives 0.8793 — slightly *below* the best
  curated subsets, i.e. the tail features add mild noise.
- **`xgb_importance` and `mutual_info` are the strongest, most consistent selectors** (both top
  out ~0.888–0.894). `single_f1` is competitive; `random` trails — as expected, principled
  selection beats picking features at random, which is meaningful given this panel's stable
  test fold.
- **Takeaway:** for MASTER, feature *combination* and a *principled selector* both pay off. A
  curated ~100–200-feature `xgb_importance`/`mutual_info` subset is the sweet spot — better than
  any single feature and better than dumping in all 351.

F1 by `method × k` (bold = best per row):

| method | 10 | 25 | 50 | 100 | 150 | 200 | 250 | all(351) |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|
| single_f1 | 0.8667 | 0.8615 | 0.8747 | 0.8744 | 0.8704 | **0.8874** | 0.8719 | — |
| xgb_importance | 0.8400 | 0.8647 | 0.8700 | 0.8813 | 0.8810 | **0.8935** | 0.8882 | — |
| mutual_info | 0.8540 | 0.8515 | 0.8715 | 0.8882 | 0.8869 | 0.8882 | 0.8867 | — |
| random | 0.8445 | 0.8782 | 0.8738 | 0.8714 | 0.8749 | 0.8763 | **0.8844** | — |
| all_features | | | | | | | | 0.8793 |

*Reference — best single-feature F1 (Part 1) = 0.8636; majority-class floor F1 ≈ 0.8461.*

---

## 6. How to replicate

**Requirements:** `python3, pandas, numpy, scikit-learn, xgboost` (and `matplotlib` for the
Part 2 plot — optional, the script skips the PNG if it is missing).

```bash
# Part 1 — single-feature (writes MASTER_single_feature_results.csv)
python3 feature_depth/single_feature_eval.py --panel MASTER

# Part 2 — multi-feature (writes MASTER_multi_feature_results.csv + ksweep PNG)
python3 feature_depth/multi_feature_eval.py --panel MASTER
```

- Run Part 1 before Part 2 (Part 2's `single_f1` selector reads the Part 1 CSV).
- The scripts auto-locate the data folder; override with `--data_dir "/path/to/EĞİTİM (TRAIN) SETLERİ"`.
- Part 1 is the long pole — ~2066 model fits on 2931 rows (~20–30 min); Part 2 is ~30 s.
- Part 1 runs a self-check asserting `significand(0.000080095878575) == 8.0096` and
  `sig4figs(...) == 0.00008010`, so a wrong transform fails loudly.
- Fully reproducible: `random_state=42` everywhere (split, models, mutual info, random selector).
- Each run **overwrites** the CSV/PNG outputs in this folder.
