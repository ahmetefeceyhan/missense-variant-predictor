# KANSER — Feature-Depth Analysis (in depth)

How much can the features of the KANSER panel predict the `Label`
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
| `KANSER_single_feature_results.csv` | Part 1 output: one row per model, **sorted best F1 → worst**. |
| `KANSER_multi_feature_results.csv` | Part 2 output: one row per (method, k) model, **sorted best F1 → worst**, with the chosen feature list. |
| `KANSER_multi_feature_ksweep.png` | Part 2 plot: F1 vs k for each selection method. |
| `README.md` | This document. |

The scripts that generate these live in `feature_depth/` and are run with `--panel KANSER`.

The 78-row test fold (24 benign) makes metrics moderately stable — more trustworthy than a
tiny panel, though differences under ~0.02 F1 are still within noise. The "majority-class
floor" (F1 of a trivial always-predict-pathogenic model, `2·prev/(1+prev)`) is **F1 ≈ 0.8171**
at this panel's prevalence of 0.691; a model must clear that to be useful.

---

# PART 1 — Single-feature models

## 1. The dataset

- 388 rows. Target `Label`: **268 pathogenic (1) / 120 benign (0)** — prevalence 0.691.
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
holdout (seed 42), reused for every model: train = 310 rows, test = 78 rows (24 benign /
54 pathogenic). Preprocessing stats fit on **train only**. Metrics on the test fold:
`accuracy`, and `f1 / recall / precision` with `pos_label=1`.

### Output

`KANSER_single_feature_results.csv`, sorted **best F1 → worst**, with a `rank` column. Columns:
`rank, feature, prefix, kind, transform, imputation, na_fraction, accuracy, f1, recall, precision`.

## 3. What we found (Part 1)

- **Best single feature: `AL_22` (`significand`, mean)** — **F1 0.8870**, recall 0.944,
  precision 0.836, accuracy 0.833. Notably **`EK_2` is #2** (F1 0.8829) — one of the few times
  an `EK_` feature competes with the best `AL_` features.
- **248 of 351 features clear the majority floor** (F1 > 0.8171). The floor is low here
  (prevalence 0.691), so clearing it is easier than on more imbalanced panels; still, the top
  features sit a meaningful ~0.07 above it.
- **Transform/imputation barely matter** for a given numeric feature: XGBoost splits on rank,
  so `raw`, `significand`, and `sig4figs` (and mean vs median) usually tie.
- **Takeaway:** several `AL_` features (and `EK_2`) carry real standalone signal, with the best
  reaching F1 ≈ 0.89 — the strongest single-feature performance of any panel relative to its
  floor.

Top of the table:

| rank | feature | transform | imputation | accuracy | f1 | recall | precision |
|-----:|---------|-----------|-----------|---------:|------:|-------:|----------:|
| 1 | AL_22 | significand | mean | 0.8333 | 0.8870 | 0.9444 | 0.8361 |
| 2 | EK_2 | raw | median | 0.8333 | 0.8829 | 0.9074 | 0.8596 |
| 3 | AL_209 | raw | median | 0.8077 | 0.8760 | 0.9815 | 0.7910 |
| 4 | AL_215 | raw | mean | 0.8077 | 0.8760 | 0.9815 | 0.7910 |
| 5 | AL_23 | sig4figs | mean | 0.8077 | 0.8718 | 0.9444 | 0.8095 |

---

# PART 2 — Multi-feature models

Does **combining many features** predict KANSER better than any single feature? We pick feature
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
| `single_f1` | Reuse `KANSER_single_feature_results.csv`: best single-feature F1 per feature, descending. |
| `xgb_importance` | Fit one XGBoost on all 351 features (train only), rank by **gain**. Zero-gain features back-filled in column order. |
| `mutual_info` | `mutual_info_classif(Xtrain, ytrain)` — model-agnostic statistical ranking. |
| `random` | One random draw (seed 42). A **noisy baseline**. |

> ⚠️ `single_f1` was ranked using F1 measured on the **same test fold**, so it is mildly
> *optimistic* vs the three train-only selectors.

### The sweep

`k ∈ {10, 25, 50, 100, 150, 200, 250}` per method, **plus** `all` (=351, emitted once as
`all_features`). → **4 × 7 + 1 = 29 models**.

### Output

`KANSER_multi_feature_results.csv`, sorted **best F1 → worst** with a `rank` column. Columns:
`rank, selection_method, k, n_features, accuracy, f1, recall, precision, selected_features` —
where `selected_features` is the full `;`-joined list of chosen feature names, so the exact
subset behind every row is reproducible from the CSV alone.

## 5. What we found (Part 2)

- **Combining features helps modestly.** The best multi-feature model (`random`, k=200) reaches
  **F1 0.9107**, above the best *single* feature (`AL_22`, 0.8870) and well above the floor
  (0.8171). `all_features` (351) gives 0.8929.
- **A small curated set already captures most of the gain.** `xgb_importance` hits **F1 0.9043
  with just k=10 features** and stays flat (~0.883) afterward — i.e. ~10 well-chosen features
  are nearly as good as hundreds. This is the most useful practical result for KANSER.
- **`xgb_importance` is the most efficient selector** (great at tiny k); `random` only catches
  up at large k (its k=200 win is partly luck on a 24-benign test fold). `mutual_info` improves
  gradually with k.
- **Takeaway:** for KANSER, combining features gives a real but modest lift over the best single
  feature, and a compact `xgb_importance` top-10 subset is an excellent, efficient choice.

F1 by `method × k` (bold = best per row):

| method | 10 | 25 | 50 | 100 | 150 | 200 | 250 | all(351) |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|
| single_f1 | 0.8850 | 0.8850 | **0.8947** | 0.8750 | 0.8750 | 0.8750 | 0.8750 | — |
| xgb_importance | **0.9043** | 0.8929 | 0.8829 | 0.8829 | 0.8829 | 0.8829 | 0.8829 | — |
| mutual_info | 0.8333 | 0.8598 | 0.8704 | 0.8807 | 0.8727 | 0.8727 | **0.8909** | — |
| random | 0.8246 | 0.8073 | 0.8393 | 0.8727 | 0.9009 | **0.9107** | 0.8929 | — |
| all_features | | | | | | | | 0.8929 |

*Reference — best single-feature F1 (Part 1) = 0.8870; majority-class floor F1 ≈ 0.8171.*

---

## 6. How to replicate

**Requirements:** `python3, pandas, numpy, scikit-learn, xgboost` (and `matplotlib` for the
Part 2 plot — optional, the script skips the PNG if it is missing).

```bash
# Part 1 — single-feature (writes KANSER_single_feature_results.csv)
python3 feature_depth/single_feature_eval.py --panel KANSER

# Part 2 — multi-feature (writes KANSER_multi_feature_results.csv + ksweep PNG)
python3 feature_depth/multi_feature_eval.py --panel KANSER
```

- Run Part 1 before Part 2 (Part 2's `single_f1` selector reads the Part 1 CSV).
- The scripts auto-locate the data folder; override with `--data_dir "/path/to/EĞİTİM (TRAIN) SETLERİ"`.
- Part 1 runs ~2066 model fits (~4 min); Part 2 is ~10 s.
- Part 1 runs a self-check asserting `significand(0.000080095878575) == 8.0096` and
  `sig4figs(...) == 0.00008010`, so a wrong transform fails loudly.
- Fully reproducible: `random_state=42` everywhere (split, models, mutual info, random selector).
- Each run **overwrites** the CSV/PNG outputs in this folder.
