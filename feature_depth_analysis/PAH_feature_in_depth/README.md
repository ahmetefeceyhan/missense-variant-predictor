# PAH — Feature-Depth Analysis (in depth)

How much can the features of the PAH (phenylketonuria) panel predict the `Label`
(0 = benign, 1 = pathogenic)? Two parts:

- **Part 1 — single-feature models:** train a basic XGBoost on **one feature at a time** and
  rank features by predictive power.
- **Part 2 — multi-feature models:** **combine many features** (sweep the count `k`, with
  several selection strategies) and see whether the panel predicts better together than any
  feature alone.

Both parts use the same dataset, the same basic model (`XGBClassifier(n_estimators=100)`,
"nothing fancy"), and the same stratified 80/20 holdout split so every number is comparable.

This folder is self-contained:

| File | What it is |
|------|------------|
| `PAH_single_feature_eval.py` | **Part 1** script (one model per feature × preprocessing variant). |
| `PAH_single_feature_results.csv` | Part 1 output: one row per model, **sorted best F1 → worst**. |
| `PAH_multi_feature_eval.py` | **Part 2** script (selection-method × k sweep). Imports helpers from the Part 1 script. |
| `PAH_multi_feature_results.csv` | Part 2 output: one row per (method, k) model, **sorted best F1 → worst**, with the chosen feature list. |
| `PAH_multi_feature_ksweep.png` | Part 2 plot: F1 vs k for each selection method. |
| `README.md` | This document. |

---

# PART 1 — Single-feature models

## 1. The dataset

- **Source file:** `universite-veri-seti/EĞİTİM (TRAIN) SETLERİ/YARISMA_TRAIN_PAH.csv`
- **Shape:** 372 rows × 353 columns.
- **Target `Label`:** 310 pathogenic (1) / 62 benign (0) — heavily **imbalanced** (~83% positive).
- **ID column:** `Variant_ID` (excluded from modeling).
- **Features (351 total), by prefix:**
  - `AL_` ×334 — numeric floats (allele-frequency / score-like), missingness 27–96%.
  - `EK_` ×9 — numeric.
  - `CAT_` ×6 — categorical strings (e.g. `gnomADg_AMR`, `AllofUs_OTH`).
  - `AA_` ×2 — amino-acid letters (e.g. `F`, `L`, `V`).

**Scope:** PAH dataset **only**. No augmentation/borrowing from MASTER, KANSER, or CFTR.

---

## 2. The plan / testing structure

Each **model is a single feature → Label classifier**. The same feature can produce several
models because we vary how its values are represented and how missing values are filled. Every
combination is its own model so we can compare representations head-to-head.

### Numeric features (`AL_`, `EK_`) → 6 models each

`3 value-transforms × 2 imputations = 6`.

**Value transforms** (the raw cells float "too much", e.g. `0.000080095878575`):

| Transform | Meaning | Example `0.000080095878575 →` |
|-----------|---------|-------------------------------|
| `raw` | unchanged | `0.000080095878575` |
| `significand` | mantissa normalized to `[1,10)`, sign kept, rounded 4 dp (drops the exponent/magnitude) | `8.0096` |
| `sig4figs` | rounded to 4 significant figures, magnitude preserved | `0.00008010` |

> `significand` matches the simplification idea from the request literally. Note it **discards
> magnitude**, so it can break rank order — included deliberately as a comparison variant.

**Imputation** (fill missing values), each computed **on the training split only**:

| Imputation | Fill value |
|------------|------------|
| `mean` | training-column mean (after transform) |
| `median` | training-column median (after transform) |

### Categorical features (`CAT_`, `AA_`) → 1 model each

- Missing values become their own category `"__MISSING__"`.
- **Ordinal-encoded** with a mapping fit on train only; categories unseen in the test split → `-1`.
- (Mean/median imputation does not apply to strings, so these get a single model.)

### Total models

`343 numeric × 6  +  8 categorical × 1  =  2058 + 8  =  2066 models`.

### The model itself ("nothing fancy")

```python
xgboost.XGBClassifier(n_estimators=100, random_state=42,
                      eval_metric="logloss", verbosity=0)
```

All other hyperparameters are XGBoost defaults. No class weighting, no early stopping. The
single feature is reshaped to `(n, 1)`.

### Evaluation

- **One stratified 80/20 holdout split** (`random_state=42`), **computed once and reused for
  every model** so the metrics are directly comparable.
  - train = 297 rows, test = 75 rows (test = 13 benign / 62 pathogenic).
- All preprocessing statistics (mean, median, category map) are fit on **train only** and
  applied to test — no leakage.
- Metrics on the test split: `accuracy`, and `f1 / recall / precision` with `pos_label=1`
  (pathogenic) and `zero_division=0`.

### Output

`PAH_single_feature_results.csv`, **sorted by F1 descending (best model first → worst last)**,
with a prepended `rank` column (1 = best). Ties broken by recall, then precision. Columns:

```
rank, feature, prefix, kind, transform, imputation, na_fraction,
accuracy, f1, recall, precision
```

---

## 3. What we found

> ⚠️ **Read the caveat first.** With a 310/62 class split, the test fold has only **13 benign**
> rows. A model that leans toward predicting "pathogenic" gets recall ≈ 1.0 and precision ≈ 0.83
> almost for free (F1 ≈ 0.905 is essentially the *floor*). So treat the ranking as **relative
> signal strength between features**, not as absolute accuracy.

- **Best single feature: `AL_66` (`raw`)** — **F1 0.9185**, recall 1.000, precision 0.849,
  accuracy 0.853. `AL_20` and `AL_300` follow at F1 ≈ 0.917.
- **Transform/imputation rarely matter** for a given numeric feature: XGBoost splits on rank,
  so `raw`, `significand`, and `sig4figs` (and mean vs median) usually produce **identical or
  near-identical** F1. Even `significand`, despite dropping magnitude, often ties — because a
  single feature's split points still separate the classes similarly.
- **Worst models** bottom out around **F1 ≈ 0.81 / accuracy ≈ 0.69** (e.g. `AL_323`, `AL_157`,
  `AL_318`) — these features actively mislead the classifier relative to the majority-class floor.
- **Categorical features** (`CAT_*`, `AA_*`) sit right at the majority-class floor (F1 ≈ 0.905,
  recall 1.0) — i.e. on their own they carry little discriminative signal here; `CAT_1` is the
  only one slightly off the floor.
- **Takeaway:** a handful of `AL_` features (e.g. `AL_66`, `AL_20`, `AL_300`) carry real
  single-feature signal above the baseline; most features individually do not beat predicting
  the majority class. This motivates the multi-feature / coverage-stratified approach used in
  the full model (`advisee_models/PAH_model.py`).

The top of the table (from `PAH_single_feature_results.csv`):

| rank | feature | transform | imputation | accuracy | f1 | recall | precision |
|-----:|---------|-----------|-----------|---------:|------:|-------:|----------:|
| 1 | AL_66 | raw | mean | 0.8533 | 0.9185 | 1.0000 | 0.8493 |
| 2 | AL_66 | raw | median | 0.8533 | 0.9185 | 1.0000 | 0.8493 |
| 3 | AL_20 | raw | median | 0.8533 | 0.9173 | 0.9839 | 0.8592 |
| 4 | AL_20 | sig4figs | median | 0.8533 | 0.9173 | 0.9839 | 0.8592 |
| 5 | AL_300 | raw | median | 0.8533 | 0.9173 | 0.9839 | 0.8592 |

---

## 4. How to replicate

**Requirements** (already present in the project's environment):

```
python3, pandas, numpy, scikit-learn, xgboost
# pip install pandas numpy scikit-learn xgboost
```

**Run** (from anywhere — the script auto-locates the `universite-veri-seti` data folder by
searching parent directories):

```bash
python3 PAH_feature_in_depth/PAH_single_feature_eval.py
```

Or point it at the data folder explicitly:

```bash
python3 PAH_feature_in_depth/PAH_single_feature_eval.py \
    --data_dir "/path/to/universite-veri-seti/EĞİTİM (TRAIN) SETLERİ"
```

- Runtime: ~3–4 minutes (2066 tiny XGBoost fits).
- The script first runs a **self-check** asserting `significand(0.000080095878575) == 8.0096`
  and `sig4figs(...) == 0.00008010`, so a wrong transform fails loudly.
- It **overwrites** `PAH_single_feature_results.csv` in this folder and prints the top-20
  models plus each feature's best variant.
- Fully reproducible: `random_state=42` for both the split and every model.

---

# PART 2 — Multi-feature models

Part 1 found that *individually* almost no feature beats the majority-class floor (only 18 of
351 features clear single-feature F1 0.9051). Part 2 asks the obvious follow-up: **does
combining many features predict PAH better than any single feature?** We pick feature subsets
four different ways, sweep the subset size `k`, train one basic XGBoost per `(method, k)`, and
record the exact feature list chosen.

## 5. The testing structure / plan (Part 2)

- **Same dataset, same split, same model** as Part 1 (stratified 80/20 holdout, seed 42;
  `XGBClassifier(n_estimators=100)`). All numbers are directly comparable to Part 1.
- **Preprocessing:** `raw` values + **median** imputation (median fit on train only);
  categoricals (`CAT_`,`AA_`) ordinal-encoded (missing = own category, unseen → -1). The full
  297×351 train / 75×351 test matrices are built **once**, then each model slices its columns.
  (We do *not* re-sweep transforms/imputations here — Part 1 showed they barely move trees.)

### Selection methods — four ways to choose features (rank, then take top `k`)

| Method | How features are ranked |
|--------|--------------------------|
| `single_f1` | Reuse `PAH_single_feature_results.csv`: best single-feature F1 per feature, descending. *Continuity with Part 1.* |
| `xgb_importance` | Fit one XGBoost on all 351 features (train only), rank by **gain** importance. Zero-gain features back-filled in column order. |
| `mutual_info` | `mutual_info_classif(Xtrain, ytrain)` — model-agnostic statistical ranking. |
| `random` | One random draw (seed 42). A **noisy baseline** — one draw only, so treat it as a sanity floor, not a tuned result. |

> ⚠️ `single_f1` was ranked using F1 measured on the **same test fold**, so it is mildly
> *optimistic* vs the three train-only selectors. Its ranks past ~18 are also arbitrary
> floor-ties (see Part 1). The clean cross-method comparison is `xgb_importance` /
> `mutual_info` / `random`.

### The sweep

`k ∈ {10, 25, 50, 100, 150, 200, 250}` for each method, **plus** `all` (=351, identical for
every method, emitted once as `all_features`). → **4 × 7 + 1 = 29 models**.

### Output

`PAH_multi_feature_results.csv`, sorted **best F1 → worst** with a `rank` column. Columns:

```
rank, selection_method, k, n_features, accuracy, f1, recall, precision, selected_features
```

`selected_features` is the **full `;`-joined list** of chosen feature names, so the exact
subset behind every row is reproducible from the CSV alone. The script also prints the full
29-row table and an F1 `method × k` pivot, and saves `PAH_multi_feature_ksweep.png`.

## 6. What we found (Part 2)

> ⚠️ Same caveat as Part 1, **stronger here**: 13 benign test rows means differences of
> 1–2 misclassified samples swing F1 by ~0.01–0.02. Most of the spread below is noise.

- **Combining features barely helps.** The best multi-feature model (`random`, k=150) reaches
  **F1 0.9254** — only marginally above the best *single* feature (`AL_66`, F1 0.9185), and
  `random` winning is itself a sign that the gap is **noise, not signal**.
- **Everything still hovers at the majority-class floor** (F1 ≈ 0.87–0.93, recall ≈ 0.92–1.0).
  No subset cleanly separates benign from pathogenic — consistent with Part 1's finding that
  the discriminative signal in this altered/binned panel is weak.
- **`xgb_importance` is the most reliable selector** and improves monotonically with `k`
  (F1 0.870 → 0.918 from k=10 → 200), i.e. more features ≈ slightly better, plateauing near
  `all_features` (F1 0.9104). `mutual_info` is weakest at small `k`.
- **`single_f1` selection underperforms** the train-only selectors at most `k` — evidence that
  the Part 1 floor-tie ranking is a poor feature picker (it front-loads many interchangeable
  floor-tied features).
- **Takeaway:** for PAH, *which* features you pick and *how many* matters less than the hard
  ceiling imposed by the weak signal + class imbalance. A handful of features already captures
  what's learnable; piling on more features mostly adds noise. This is why the production model
  (`advisee_models/PAH_model.py`) leans on **missingness/coverage stratification** rather than
  raw feature count.

F1 by `method × k` (from the run; bold = best per row):

| method | 10 | 25 | 50 | 100 | 150 | 200 | 250 | all(351) |
|--------|----:|----:|----:|----:|----:|----:|----:|----:|
| single_f1 | 0.870 | 0.904 | 0.897 | 0.889 | 0.879 | 0.896 | 0.897 | — |
| xgb_importance | 0.870 | 0.909 | 0.910 | 0.910 | 0.910 | **0.919** | 0.917 | — |
| mutual_info | 0.881 | 0.864 | 0.879 | 0.904 | 0.904 | 0.902 | 0.910 | — |
| random | 0.886 | 0.916 | 0.910 | 0.901 | **0.925** | 0.919 | 0.917 | — |
| all_features | | | | | | | | 0.910 |

*Reference — best single-feature F1 (Part 1) = 0.9185 (`AL_66`).*

## 7. How to replicate (Part 2)

Same requirements as Part 1, plus **`matplotlib`** for the plot (optional — the script skips
the PNG gracefully if it is missing).

```bash
python3 PAH_feature_in_depth/PAH_multi_feature_eval.py
```

- Runtime: ~10 seconds (29 models + a couple of selector fits).
- Imports its preprocessing/model helpers from `PAH_single_feature_eval.py` (same folder) and
  reads `PAH_single_feature_results.csv` for the `single_f1` ranking — run Part 1 first if that
  CSV is missing.
- **Overwrites** `PAH_multi_feature_results.csv` and `PAH_multi_feature_ksweep.png`.
- Fully reproducible: `random_state=42` everywhere (split, models, MI, the random selector).
