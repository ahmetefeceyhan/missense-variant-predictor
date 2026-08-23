# CFTR — Feature-Depth Analysis (in depth)

How much can the features of the CFTR panel predict the `Label`
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
| `CFTR_single_feature_results.csv` | Part 1 output: one row per model, **sorted best F1 → worst**. |
| `CFTR_multi_feature_results.csv` | Part 2 output: one row per (method, k) model, **sorted best F1 → worst**, with the chosen feature list. |
| `CFTR_multi_feature_ksweep.png` | Part 2 plot: F1 vs k for each selection method. |
| `README.md` | This document. |

The scripts that generate these live in `feature_depth/` and are run with `--panel CFTR`.

---

## ⚠️ Read this first — CFTR is a very small panel

CFTR has only **111 variants** (90 pathogenic / 21 benign). The stratified 80/20 holdout
therefore leaves a **test fold of just 23 rows — only 4 of them benign**. A single
misclassified sample moves F1 by ~0.04–0.05, so **all metrics below are extremely noisy** and
should be read as rough, relative signal — not stable estimates. The "majority-class floor"
(F1 of a trivial always-predict-pathogenic model, `2·prev/(1+prev)`) is **F1 ≈ 0.8955** at
this panel's prevalence of 0.811; treat that as the baseline any model must clear to be useful.

---

# PART 1 — Single-feature models

## 1. The dataset

- 111 rows. Target `Label`: **90 pathogenic (1) / 21 benign (0)** — imbalanced (prevalence 0.811).
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
holdout (seed 42), reused for every model: train = 88 rows, test = 23 rows (4 benign / 19
pathogenic). Preprocessing stats fit on **train only**. Metrics on the test fold: `accuracy`,
and `f1 / recall / precision` with `pos_label=1`.

### Output

`CFTR_single_feature_results.csv`, sorted **best F1 → worst**, with a `rank` column. Columns:
`rank, feature, prefix, kind, transform, imputation, na_fraction, accuracy, f1, recall, precision`.

## 3. What we found (Part 1)

- **Best single feature: F1 0.9500** — a three-way tie (`AL_215`, `AL_21`, `AL_6`, all `raw`),
  recall 1.000, precision 0.905, accuracy 0.913. That is just **one** benign sample better than
  the majority floor (0.8955), so it is barely distinguishable from "predict pathogenic".
- **Transform/imputation barely matter** for a given numeric feature: XGBoost splits on rank,
  so `raw`, `significand`, and `sig4figs` (and mean vs median) usually tie.
- The ranking is **dominated by noise**: with only 4 benign test rows, ~274 of 351 features
  "clear" the floor, which is not meaningful separation — it mostly reflects that almost any
  feature lets the model keep predicting the majority class.
- **Takeaway:** at this sample size, single-feature F1 is not a reliable signal of feature
  quality for CFTR. The top of the table (`AL_215`, `AL_21`, `AL_6`) is the best guess, but the
  confidence interval is wide.

Top of the table:

| rank | feature | transform | imputation | accuracy | f1 | recall | precision |
|-----:|---------|-----------|-----------|---------:|------:|-------:|----------:|
| 1 | AL_215 | raw | mean | 0.9130 | 0.9500 | 1.0000 | 0.9048 |
| 2 | AL_21 | raw | mean | 0.9130 | 0.9500 | 1.0000 | 0.9048 |
| 3 | AL_6 | raw | mean | 0.9130 | 0.9500 | 1.0000 | 0.9048 |
| 4 | AL_237 | sig4figs | mean | 0.8696 | 0.9268 | 1.0000 | 0.8636 |
| 5 | AL_166 | significand | mean | 0.8696 | 0.9268 | 1.0000 | 0.8636 |

---

# PART 2 — Multi-feature models

Does **combining many features** predict CFTR better than any single feature? We pick feature
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
| `single_f1` | Reuse `CFTR_single_feature_results.csv`: best single-feature F1 per feature, descending. |
| `xgb_importance` | Fit one XGBoost on all 351 features (train only), rank by **gain**. Zero-gain features back-filled in column order. |
| `mutual_info` | `mutual_info_classif(Xtrain, ytrain)` — model-agnostic statistical ranking. |
| `random` | One random draw (seed 42). A **noisy baseline**. |

> ⚠️ `single_f1` was ranked using F1 measured on the **same test fold**, so it is mildly
> *optimistic* vs the three train-only selectors.

### The sweep

`k ∈ {10, 25, 50, 100, 150, 200, 250}` per method, **plus** `all` (=351, emitted once as
`all_features`). → **4 × 7 + 1 = 29 models**.

### Output

`CFTR_multi_feature_results.csv`, sorted **best F1 → worst** with a `rank` column. Columns:
`rank, selection_method, k, n_features, accuracy, f1, recall, precision, selected_features` —
where `selected_features` is the full `;`-joined list of chosen feature names, so the exact
subset behind every row is reproducible from the CSV alone.

## 5. What we found (Part 2)

> ⚠️ Same small-N caveat, **even stronger** here: with 4 benign test rows, F1 differences of
> ~0.04–0.05 are single-sample noise. The spread below is mostly not signal.

- **Combining features does NOT beat the best single feature.** The best multi-feature model
  (`single_f1`, k=10) reaches **F1 0.9231** — *below* the best single-feature F1 of 0.9500, and
  every model sits at or near the majority floor (F1 ≈ 0.83–0.92).
- **No clear `k` trend, no clear winning selector.** `mutual_info` is flat at 0.90 across all
  `k`; `xgb_importance` hovers ~0.87–0.90; `single_f1` starts highest (0.9231 at k=10) but
  **degrades as more features are added** (down to 0.84 by k=250); `random` is all over the place
  — exactly the fingerprint of noise rather than a real selection effect.
- **`all_features` (k=351) gives F1 0.9000** — no better than small subsets.
- **Takeaway:** for CFTR there is no measurable benefit to combining features, and no stable
  optimal feature count. With this little data the model is pinned near the majority-class floor
  regardless of how many features it sees. Any apparent "best" subset here is within the noise
  band and should not be trusted as a real finding.

F1 by `method × k`:

| method | 10 | 25 | 50 | 100 | 150 | 200 | 250 | all(351) |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|
| single_f1 | 0.9231 | 0.8947 | 0.8649 | 0.8947 | 0.8500 | 0.8421 | 0.8421 | — |
| xgb_importance | 0.9000 | 0.8718 | 0.9000 | 0.8718 | 0.9000 | 0.9000 | 0.9000 | — |
| mutual_info | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | — |
| random | 0.8947 | 0.8333 | 0.8421 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | — |
| all_features | | | | | | | | 0.9000 |

*Reference — best single-feature F1 (Part 1) = 0.9500; majority-class floor F1 ≈ 0.8955.*

---

## 6. How to replicate

**Requirements:** `python3, pandas, numpy, scikit-learn, xgboost` (and `matplotlib` for the
Part 2 plot — optional, the script skips the PNG if it is missing).

```bash
# Part 1 — single-feature (writes CFTR_single_feature_results.csv)
python3 feature_depth/single_feature_eval.py --panel CFTR

# Part 2 — multi-feature (writes CFTR_multi_feature_results.csv + ksweep PNG)
python3 feature_depth/multi_feature_eval.py --panel CFTR
```

- Run Part 1 before Part 2 (Part 2's `single_f1` selector reads the Part 1 CSV).
- The scripts auto-locate the data folder; override with `--data_dir "/path/to/EĞİTİM (TRAIN) SETLERİ"`.
- Part 1 runs a self-check asserting `significand(0.000080095878575) == 8.0096` and
  `sig4figs(...) == 0.00008010`, so a wrong transform fails loudly.
- Fully reproducible: `random_state=42` everywhere (split, models, mutual info, random selector).
- Each run **overwrites** the CSV/PNG outputs in this folder.
