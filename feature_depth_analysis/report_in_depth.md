# Feature-Depth Analysis — In-Depth Report (all panels)

A single place that summarizes the feature-depth study across **all four panels** — PAH,
MASTER, KANSER, CFTR — and states the **best settings** found for each. Per-panel detail lives
in each panel's own folder (`<PANEL>_feature_in_depth/README.md`); this report compares them.

- **Question.** How well can the panel features predict `Label` (0 = benign, 1 = pathogenic),
  (a) one feature at a time, and (b) combining many features?
- **Model.** A deliberately basic `XGBClassifier(n_estimators=100, random_state=42)` — "nothing
  fancy", all other params default — so results reflect the *data*, not tuning.
- **Evaluation.** One stratified 80/20 holdout per panel (seed 42), reused for every model so
  numbers are directly comparable. Metrics on the held-out test fold: accuracy, and
  F1 / recall / precision for the pathogenic class (`pos_label=1`). All preprocessing fit on
  **train only** (no leakage).
- **Scope.** Each panel is analyzed on its **own dataset only** — no cross-panel augmentation.

> **The single most important caveat:** every panel is imbalanced, so a trivial
> "always predict pathogenic" model already scores a high F1 — the **majority-class floor**
> (`2·prev/(1+prev)`, computed on the test fold). A model is only interesting to the extent it
> beats that floor *with better precision*. Floors are high here, so absolute F1 is misleading;
> read F1 **relative to the floor**, and weight panels by how big (and therefore how stable)
> their test fold is.

---

## Methodology (shared by all panels)

**Part 1 — single-feature models.** For every non-ID feature, train a model that uses *only
that feature*.
- Numeric features (`AL_`, `EK_`): 6 models each = 3 value-transforms × 2 imputations.
  - transforms: `raw` (unchanged) · `significand` (mantissa to `[1,10)`, e.g.
    `0.000080095878575 → 8.0096`) · `sig4figs` (4 significant figures, e.g. `→ 0.00008010`).
  - imputation: `mean` or `median` (fit on train).
- Categorical features (`CAT_`, `AA_`): 1 model each, ordinal-encoded (missing = own category).
- ⇒ **2066 models per panel.** Output: `<PANEL>_single_feature_results.csv` (sorted best F1→worst).

**Part 2 — multi-feature models.** Select a feature subset, sweep its size `k`, train one model.
- Preprocessing: `raw` + **median** impute; categoricals ordinal-encoded.
- 4 selection methods: `single_f1` (reuse Part 1 ranking) · `xgb_importance` (gain) ·
  `mutual_info` · `random` (seed-42 baseline).
- `k ∈ {10, 25, 50, 100, 150, 200, 250, all}` ⇒ **29 models per panel**. Output:
  `<PANEL>_multi_feature_results.csv` (each row records the exact feature list) + a k-sweep PNG.

**Shared code:** `feature_depth/single_feature_eval.py` and `feature_depth/multi_feature_eval.py`
(run with `--panel <NAME>`). *(PAH was the prototype and keeps its own copy in
`PAH_feature_in_depth/`; the numbers are identical.)*

---

## Headline results (all panels)

| Panel | Rows | Prev. | Test fold (benign/path) | Floor F1\* | Best **single** F1 | Best **multi** F1 | All-features F1 |
|-------|-----:|------:|:-----------------------:|----------:|-------------------:|------------------:|----------------:|
| **MASTER** | 2931 | 0.733 | 587 (157 / 430) | 0.846 | 0.864 | **0.894** | 0.879 |
| **KANSER** | 388 | 0.691 | 78 (24 / 54) | 0.818 | 0.887 | **0.911** | 0.893 |
| **PAH** | 372 | 0.833 | 75 (13 / 62) | 0.905 | 0.919 | **0.925** | 0.910 |
| **CFTR** | 111 | 0.811 | 23 (**4** / 19) | 0.905 | **0.950** | 0.923 | 0.900 |

\* Floor = F1 of "always predict pathogenic" on that test fold. **Trust ordering by test-fold
size: MASTER ≫ KANSER > PAH > CFTR.** CFTR's 4-benign test fold makes its numbers essentially
anecdotal (±0.05 F1 per sample).

### Does combining features beat the best single feature?

| Panel | Single → Multi | Verdict |
|-------|:--------------:|---------|
| **MASTER** | 0.864 → 0.894 (**+0.030**), precision 0.78 → 0.85 | **Yes — real, meaningful gain.** The only panel where combination clearly pays off, and on the most reliable test fold. |
| **KANSER** | 0.887 → 0.911 (+0.024) | **Yes — modest gain.** A compact 10-feature set already gets there. |
| **PAH** | 0.919 → 0.925 (+0.006) | **No real gain.** Improvement is within noise; `random` "winning" confirms it. |
| **CFTR** | 0.950 → 0.923 (**−0.027**) | **No.** Too little data; everything sits at the floor, multi is worse. |

---

## Best settings, per panel

### MASTER — *combine features; use a principled selector* ✅ best evidence
- **Recommended:** `xgb_importance` top **~200** features (raw + median impute).
- **Result:** **F1 0.8935**, accuracy 0.835, recall 0.947, **precision 0.846** (vs 0.777 for
  the best single feature — the combined model stops over-predicting "pathogenic").
- F1 rises with `k`, peaks at k≈100–200, then flattens; `all_features` (0.879) is *worse* than
  the curated subset (tail features add noise). `xgb_importance` and `mutual_info` are the
  strongest, most consistent selectors; `random` trails.
- Best single feature (reference): `AL_2`, raw/median, F1 0.864.

### KANSER — *a small curated set is enough*
- **Recommended (efficient):** `xgb_importance` top **10** features → **F1 0.9043** with almost
  nothing; adding more features barely moves it.
- **Highest observed:** `random` k=200, F1 0.9107 (partly luck on a 24-benign fold — prefer the
  stable xgb_importance top-10).
- Best single feature: `AL_22`, significand/mean, F1 0.887 — and notably **`EK_2` is #2**
  (0.883), the only panel where an `EK_` feature rivals the best `AL_` features.

### PAH — *signal is weak; few features suffice*
- **Recommended:** no benefit to large feature sets. Best multi is `random` k=150 (F1 0.9254)
  but it's within the noise band of the best single feature `AL_66` (raw, F1 0.9185).
- Among selectors, `xgb_importance` is the only one that improves monotonically with `k`
  (plateaus ≈ all-features 0.910). Everything hovers at the majority floor (~0.905).
- Practical takeaway: a handful of `AL_` features (`AL_66`, `AL_20`, `AL_300`) is as good as it
  gets; this panel's discriminative signal is inherently limited.

### CFTR — *too small to conclude; do not combine*
- **Recommended:** the best single feature — `AL_6` / `AL_215` / `AL_21` (three-way tie),
  raw/mean, **F1 0.950**. Combining features does **not** help (best multi 0.923, all-features
  0.900).
- ⚠️ With only **4 benign test rows**, every metric here is dominated by noise (±0.05 F1 per
  misclassified sample). Treat all CFTR conclusions as provisional; collect more data before
  trusting any feature ranking.

---

## Cross-panel findings

1. **Panel size decides whether feature combination helps.** On the large, stable MASTER panel
   combining features gives a clear, precision-driven gain; on the tiny CFTR/PAH folds the
   "best" multi-feature model is indistinguishable from (or worse than) the best single feature.
   Sample size — not the algorithm — is the binding constraint.
2. **The value transform almost never matters.** `raw`, `significand`, and `sig4figs` (and
   mean vs median) produce near-identical results everywhere, because trees split on **rank**
   and all three transforms preserve rank within a feature (significand can break it but rarely
   changes the split). Best single features split ~evenly between `raw` and `significand`.
3. **`AL_` features dominate.** The best single feature is an `AL_` feature on every panel
   (KANSER's `EK_2` is the lone non-`AL_` near the top). `CAT_`/`AA_` features sit at the floor.
4. **Principled selection > random when there's enough data.** `xgb_importance` / `mutual_info`
   beat `random` cleanly on MASTER and are most efficient on KANSER (top-10 ≈ top-all). On the
   noisy panels `random` sometimes "wins" — a tell that the differences there are not real.
5. **More features is not better past a point.** Where combination helps, F1 peaks at a curated
   ~100–200-feature subset and `all_features` is slightly worse — the long tail of weak features
   adds noise.
6. **Everything is precision-limited, not recall-limited.** Across panels recall is high
   (~0.93–1.0) because the majority class is pathogenic; the lever that moves F1 above the floor
   is **precision**, which only the combined MASTER/KANSER models meaningfully improve.

---

## Bottom-line recommendations

| Panel | Best setting | F1 | Why |
|-------|--------------|---:|-----|
| **MASTER** | `xgb_importance` top-200, raw+median | **0.894** | Real gain, best precision, most reliable fold |
| **KANSER** | `xgb_importance` top-10, raw+median | **0.904** | Nearly best, very compact/efficient |
| **PAH** | single feature `AL_66` (raw) — or any small set | **0.919** | Combining adds nothing beyond noise |
| **CFTR** | single feature `AL_6`/`AL_215`/`AL_21` (raw) | **0.950** | Too small to combine; needs more data |

> These are **basic-XGBoost, single-split** numbers meant for *relative* comparison. For a
> production model, the next steps would be cross-validation (especially for KANSER/PAH/CFTR),
> class-imbalance handling, and—for PAH specifically—the missingness/coverage-stratified
> approach used in `advisee_models/PAH_model.py`.

---

## How to reproduce

```bash
# Per panel: Part 1 (single-feature) then Part 2 (multi-feature)
python3 feature_depth/single_feature_eval.py --panel MASTER
python3 feature_depth/multi_feature_eval.py  --panel MASTER
#   ...repeat with --panel KANSER, --panel CFTR, --panel PAH
```

- Requirements: `python3, pandas, numpy, scikit-learn, xgboost` (+ `matplotlib` for the plots).
- Outputs land in `<PANEL>_feature_in_depth/`. Part 1 is the long pole (~2066 fits; MASTER
  ~20–30 min, others a few minutes); Part 2 is ~10–30 s.
- Fully reproducible: `random_state=42` throughout (split, every model, mutual info, the random
  selector). Run Part 1 before Part 2 (Part 2's `single_f1` selector reads the Part 1 CSV).
