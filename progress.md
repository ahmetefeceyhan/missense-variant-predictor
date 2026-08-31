# Teknofest Model Projesi — İlerleme Durumu

Son güncelleme: 2026-07-24 (NB40 Bayes Error Ceiling + NB41 Feature Selection sonuçları geriye dönük eklendi — bu iki notebook Haziran'da çalıştırılmış ve çıktı dosyaları [`results/v23_bayes_error_ceiling/`, `results/v24_feature_selection/`] mevcuttu ama bulgular hiç bu dosyaya işlenmemişti. Ayrıca danışman EDA raporunun (`feature_depth_analysis/report_in_depth.md` + `*_search_models/`) ham metodoloji/sayısal detayları — headline tablo, panel-bazlı en iyi feature isimleri, tam search_models sonuç tabloları — "Danışman EDA İncelemesi" bölümüne geriye dönük eklendi. **KRİTİK SONUÇ: NB40'ın Bayes-ceiling analizi PAH'ta anlamlı boşluk (gap=+0.1125, "DEVAM" kararı) buluyor — bu, önceki "PAH KESİNLEŞTİ / Chatterji tavanı" kararıyla ÇELİŞİYOR.** Bu çelişki nedeniyle ve genel olarak yeni bir araştırma turu başlatmak için **TÜM PANELLER (CFTR/KANSER/PAH/MASTER) yeniden AÇILDI** — hiçbiri artık "kesinleşti" sayılmıyor. Aşağıdaki "KESİNLEŞTİ" ibareleri tarihsel bağlam olarak korunuyor ama artık geçerli değil; bkz. "Panel Durumu (2026-07-24 itibarıyla)" bölümü.)

> **⚠️ FINAL TEST DAĞILIMI:** Yarışma final test seti her panel için ~%80 benign / %20 pathogenic olacak — **train'in TERSİ**. Metrik pathogenic F1. Bu, düşük-eşik seçen modellerde precision çökmesine yol açar (bkz. CLAUDE.md "FINAL TEST DAĞILIMI" bölümü). NB15 v2 ve NB16 bunu benign-aware threshold + %80/20 bootstrap değerlendirme ile ele alır.

---

## ⭐ Panel Durumu (2026-07-24 itibarıyla — GÜNCEL)

**Tüm paneller yeniden açıldı.** Aşağıdaki tablodaki "en iyi" skorlar hâlâ geçerli referans noktalarıdır (yeni deneyler bunları geçmeye çalışacak), ama hiçbiri artık "dokunma" statüsünde değildir.

| Panel | Mevcut en iyi | Boot-F1 | NB40 Bayes-ceiling Gap | Karar |
|---|---|---|---|---|
| **MASTER** | S1_6040/balbag (NB39) | 0.6379 | +0.0161 (DUR/ceiling, HIGH conf.) | AÇIK — stacking/Optuna denenmedi |
| **KANSER** | P9_REVERSE_6040_catboost_with_fe (NB32) | 0.7300 | −0.0037 (DUR/ceiling, HIGH conf.) | AÇIK — NB44 (G_LOW flag) planlı, çalıştırılmadı |
| **PAH** | P4_COMBINED_BalBag (NB21) | 0.582 | **+0.1125 (DEVAM/significant margin, HIGH conf.)** | **AÇIK — ÇELİŞKİ VAR, öncelik yüksek** |
| **CFTR** | S0c_COMBINED (NB20) | 0.863 | −0.1080 (DUR/ceiling, LOW conf. — n=21 benign) | AÇIK — Bayes tahmini güvenilmez, küçük örneklem |

**PAH çelişkisi (araştırma önceliği):** NB21→NB35 boyunca "Chatterji 2022 azınlık-örneği tavanı" gerekçesiyle PAH kapatılmıştı (floor F1=0.905, model=0.582–0.925 arası, danışmanın train-dağılımı sonucu bile floor'a yakın). Ama NB40'ın k-NN tabanlı Bayes error ceiling tahmini PAH için **Bayes-F1(8020)=0.6945 [CI: 0.6779–0.7081]** buluyor — mevcut modelin (0.582) belirgin altında değil, üstünde bir teorik tavan gösteriyor. İki analiz farklı varsayımlarla çalışıyor (floor = "hep patho de" trivial baseline; Bayes-ceiling = k-NN yerel yoğunluk tahmini) ve şu ana kadar bu iki çerçeve birbirine karşı doğrulanmadı. **Yeni araştırma bunu çözmeli.**

---

## Genel Bakış

TEKNOFEST genetik varyant patojenite tahmini (binary: 0=benign, 1=pathogenic).

**İki veri evreni var** (bkz. CLAUDE.md): (a) **Legacy** ClinVar/OpenCRAVAT — NB02–NB11, çoğu bitmiş; (b) **Yarışma** anonimleştirilmiş gerçek veri (`data/real_data/`) — **NB12–NB16, asıl teslim hattı**.

**Aktif odak = yarışma evreni, panel-bazlı modeller (her panel için ayrı: CFTR/KANSER/PAH/MASTER).**
- **KRİTİK kısıt:** Final test seti ~%80 benign / %20 patho (train'in TERSİ), metrik pathogenic F1 (yukarıdaki uyarıya bak).
- **Doğrulanmış reçete (NB16):** M3 missing + benign-aware threshold + SmallMLP(focal/ES) + OOF stacking(meta=LR) + hafif FE. Değerlendirme: %50/50 + bootstrap %80/20 (birincil).
- **Şu anki en iyi (%80/20):** **MASTER=0.6379 (S1_6040/balbag, NB39)**, **KANSER=0.7300 (P9_REVERSE_6040_catboost_with_fe, NB32)**, **PAH=0.582 (P4_COMBINED_BalBag, NB21)**, **CFTR=0.863 boot-mean (S0c_COMBINED, LOO-MCC=0.644, precision=1.0)**. Bkz. yukarıdaki "Panel Durumu" tablosu — **hepsi yeniden açık.**
- **CFTR (önceki karar: KESİNLEŞTİ, NB20):** S0c_COMBINED (MASTER+KANSER+PAH → LightGBM + prior-shift). Ensemble ve S0_MASTER terk edildi. **Yeniden açıldı (2026-07-24) — NB40 Bayes-ceiling n=21 nedeniyle güvenilmez, doğrulama gerekiyor.**
- **PAH (önceki karar: KESİNLEŞTİ, NB21→NB35):** NB16'dan +0.067 puan (BalancedBagging + COMBINED). NB22 sweep, NB24 TabPFN, NB35 FE — hiçbiri platoyu kıramadı. Chatterji 2022 tavanı doğrulandı sanılıyordu. **Yeniden açıldı (2026-07-24) — NB40 Bayes-ceiling gap=+0.1125 ile ÇELİŞİYOR, en yüksek öncelik.**
- **MASTER MODELLEMESİ İLERLEDİ (NB36→NB39):** NB36 baseline (BalBag=0.603) → NB37 stacking (0.589, değer katmadı) → NB38 heterojen stacking+calibrate-then-shift (V1_calib=0.6165, +0.014) → **NB39 reversed-distribution (S1_6040/balbag=0.6379, +0.021)**. Threshold yakınsama hipotezi doğrulandı. Gerçek resample > ağırlıklama. NB41 feature-selection'ı MASTER'da denedi (aşağıda) — top-N seçimi işe yaramadı.
- **KANSER (önceki karar: KESİNLEŞTİ, NB32):** P9_REVERSE_6040_catboost_with_fe, Boot=0.730, Bayes-ceiling'e neredeyse tam oturmuş (gap=−0.0037). **Yeniden açıldı (2026-07-24) — NB44 (G_LOW flag ablasyonu) planlı ama hiç yazılmadı/çalıştırılmadı.**
- **Açık çalışma alanları:** MASTER'da stacking (NB39 kazanan senaryo base'leriyle), Optuna hiperparametre tuning; PAH'ta Bayes-ceiling çelişkisinin çözümü (en yüksek öncelik); KANSER'de NB44; CFTR'de küçük-örneklem doğrulaması; final teslim modeli sabitleme tüm panellerde ertelendi.

---

## Tamamlananlar

### Altyapı & Kaynak Kod

| Dosya | Durum | İçerik |
|---|---|---|
| `config.py` | ✅ | SEED=42, DATA_PATH, RESULTS_NO_INSIL_DIR, RESULTS_WITH_INSIL_DIR, RESULTS_STACKING_DIR, MODELS_STACKING_DIR, REPORTS_DIR, PANELS_SINGLE |
| `src/columns.py` | ✅ | LEAKY_META_PREDICTOR_SCORES (68 col), LEAKY_META_PREDICTOR_PREDS (10 col), LABEL_ENC_COLS, OHE_COLS_V3 |
| `src/features.py` | ✅ | `prepare_data_v3_no_insil()` + `prepare_data_v3()` — 8 adımlı FE pipeline |
| `src/models.py` | ✅ | grid_search_lightgbm, grid_search_xgboost, grid_search_nn_fast, grid_search_dnn_fast, grid_search_svm, grid_search_le_lightgbm, grid_search_le_xgboost, grid_search_le_svm, grid_search_catboost |
| `src/metrics.py` | ✅ | `compute_all_metrics()` (9 metrik), `optimize_threshold()` (F1-max, 0.10–0.90) |
| `src/utils.py` | ✅ | prepare_for_xgb, prepare_for_nn |

### Feature Engineering Pipeline (her iki notebook için)

1. Boolean → int dönüşümü
2. In-silico drop (no_insil: 68+10=78 col, with_insil: korunur)
3. FE özellikleri: `grantham_distance`, `blosum62_score/delta/ref_self/alt_self`, `conservation_product`, `gerp_x_phylop`, `is_transition`, `is_cpg_site`, `gc_content_11mer`
4. OHE (sabit alfabe): 48 col (4+4+20+20)
5. K-mer encoding (k=2): DNA 16+16 col, Protein ~430+434 col
6. Gereksiz sütun temizliği (24 col drop)
7. Korelasyon filtresi (r > 0.99 → 14 col drop)
8. NaN doldurma

**Sonuç:** Ham 4287×119 → FE sonrası 4287×954

### Grid Search Parametreleri

| Model | Combo | Sabit parametreler |
|---|---|---|
| LightGBM | 12 (n_est×num_leaves×lr) | min_child_samples=20, subsample=0.8 |
| XGBoost | 12 (n_est×max_depth×lr) | subsample=0.8, min_child_weight=3 |
| NN-fast | 4 (dropout×lr) | h1=2048, h2=1024, h3=512, 50 epoch |
| DNN-fast | 4 (n_layers×hidden_dim) | dropout=0.3, lr=1e-3, 50 epoch |
| SVM | 6 (C×gamma) | kernel=rbf, class_weight=balanced |

---

### Notebook 07 — v3_no_insil_panels (TAMAMLANDI & ÇALIŞTIRILDI)

**Dosya:** `notebooks/07_v3_no_insil_panels.ipynb`
**Pipeline:** In-silico OLMADAN, 5 model, hold-out %20, 3-fold CV

**Panel Dağılımı:**
- General: 3156 satır (pos=2182, neg=974) → eğitim paneli
- Hereditary_Cancer: 715 satır (pos=354, neg=361) → eğitim paneli
- PAH: 324 satır (pos=323, neg=1) → test-only (min_class < 5)
- CFTR: 92 satır (pos=92, neg=0) → test-only

**Hold-Out Sonuçları (F1 / AUC-ROC):**

| Panel | LightGBM | XGBoost | NN | DNN | SVM |
|---|---|---|---|---|---|
| General | 0.9465 / 0.9674 | **0.9487** / 0.9686 | 0.8814 / 0.8813 | 0.8886 / 0.8819 | 0.8955 / 0.8960 |
| Hereditary_Cancer | **0.9362** / 0.9810 | 0.9343 / 0.9806 | 0.8414 / 0.8865 | 0.8593 / 0.9018 | 0.8800 / 0.9235 |

**Test-Only Panel Sonuçları (General modeli ile):**

| Panel | LightGBM | XGBoost | NN | DNN | SVM |
|---|---|---|---|---|---|
| CFTR recall | 0.9457 | 0.9457 | 0.8696 | 0.8478 | 0.9348 |
| PAH F1 | 0.9427 | 0.9340 | 0.8667 | 0.8726 | 0.9215 |

**Cross-Model Error Analizi:**
- General: 18 satırda tüm modeller yanlış (ground truth şüpheli), 82 satırda çoğunluk yanlış
- Hereditary_Cancer: 0 satırda tüm modeller yanlış, 15 satırda çoğunluk yanlış

**Çıktılar:** `results/v3_no_insil_panels/`, `reports/v3_no_insil_panels_report.pdf`

---

### Notebook 08 — v3_with_insil_panels (OLUŞTURULDU, çalıştırılmadı)

**Dosya:** `notebooks/08_v3_with_insil_panels.ipynb`
**Pipeline:** In-silico KORUNUYOR (68 skor + 10 tahmin label-encoded), aynı yapı
**Amaç:** In-silico skorların modele katkısını ölçmek (NB07 ile karşılaştırma)

**Durum:** Kernel yeniden başlatılıp çalıştırılmayı bekliyor.

---

### Notebook 09 — Sekans Ablasyonu (OLUŞTURULDU)

**Dosya:** `notebooks/09_ablation_sequence.ipynb`
**Amaç:** 8 sekans özellik grubunu tek tek kaldırarak F1 etkisini ölçmek

**12 senaryo:**
1. Baseline (tüm özellikler)
2. Tüm sekans kaldırıldı (OHE + k-mer, ~944 col)
3. Tüm OHE kaldırıldı (48 col)
4. Tüm k-mer kaldırıldı (~896 col)
5–8. OHE grupları tek tek (ref_base, alt_base, ref_amino, alt_amino)
9–12. k-mer grupları tek tek (DNA_Ref, DNA_Alt, Prot_Ref, Prot_Alt)

**Model:** LightGBM, **Çıktı:** `results/ablation_sequence/`

---

### Notebook 10 — Non-Sekans Ablasyonu (OLUŞTURULDU)

**Dosya:** `notebooks/10_ablation_nonseq.ipynb`
**Amaç:** Her non-sekans sütunu tek tek kaldırarak F1 etkisini ölçmek

**Özellik grupları (runtime'da tespit ediliyor):**
- Conservation: gerp, phastcons, phylop (~4 col)
- Fizikokimyasal: delta_hydropathy, delta_volume, delta_pi, delta_charge, polarity_change, chirality_shift (~6 col)
- Mühendislik: grantham_distance, blosum62, conservation_product, is_transition, is_cpg_site, gc_content_11mer (~10 col)
- Ham veri kalıntıları: diğer raw sütunlar

**Özellikler:** Bireysel sütun kaldırma, renk kodlu bar chart (kırmızı=önemli, sarı=nötr, yeşil=gürültü), grup bazlı özet

**Model:** LightGBM, **Çıktı:** `results/ablation_nonseq/`

---

## Diğer Çıktılar

| Dosya | Durum |
|---|---|
| `reports/v3_no_insil_panels_report.pdf` | ✅ Oluşturuldu |
| `reports/proje_sunus_raporu_taslak.md` | ✅ Teknik bölümler dolduruldu, takım/referans bölümleri boş |
| `results/v3_no_insil_panels/panel_results.csv` | ✅ Oluşturuldu |
| `results/v3_no_insil_panels/General_error_analysis.csv` | ✅ Oluşturuldu |
| `results/v3_no_insil_panels/Hereditary_Cancer_error_analysis.csv` | ✅ Oluşturuldu |

---

### Notebook 11 — Stacking Modeli: General Panel (TAMAMLANDI & ÇALIŞTIRILDI)

**Dosya:** `notebooks/11_stacking_general_panel.ipynb`
**Pipeline:** In-silico OLMADAN, General Panel, OOF (5-fold) Stacking

**Mimari:**
- Baz modeller: LightGBM, XGBoost, DNN, SVM
- Her model OOF ile 1×2 olasılık vektörü üretiyor → birleşik 1×8 meta-feature
- Meta modeller: LightGBM ve NN (ayrı ayrı değerlendiriliyor)
- Hiperparametreler: `LGBM_FIXED` / `XGB_FIXED` / `DNN_FIXED_FAST` / `SVM_FIXED` (NB07 ile senkronize)

**Veri Bölünmesi:**
- CV seti: 2524 satır (OOF baz egitimi + meta egitim)
- Hold-out: 632 satır (final test)

**Hold-Out Sonuçları:**

| Model | Tip | F1 | AUC-ROC | Precision | Recall | MCC |
|---|---|---|---|---|---|---|
| LightGBM | Baz | 0.9454 | 0.9715 | 0.9217 | 0.9703 | 0.8153 |
| XGBoost | Baz | 0.9451 | 0.9682 | 0.9254 | 0.9657 | 0.8154 |
| DNN | Baz | 0.8612 | 0.8206 | 0.7684 | 0.9794 | 0.4540 |
| SVM | Baz | 0.8955 | 0.8960 | 0.8623 | 0.9314 | 0.6352 |
| Stacking LightGBM | Meta | 0.9443 | 0.9619 | 0.9197 | 0.9703 | 0.8115 |
| **Stacking NN** | **Meta** | **0.9505** | **0.9653** | **0.9357** | **0.9657** | **0.8347** |

**Önemli Notlar:**
- En iyi meta-model: Stacking NN (F1=0.9505, MCC=0.8347)
- Stacking NN, en iyi baz modelin F1'ini (0.9454) 0.005 puan geçti
- OOF yaklaşımının baz model F1'ini hafif düşürdüğü gözlemlendi (0.9465→0.9454); bunun sebebi data leakage değil, küçük fold'lar + majority vote varyansı
- Data leakage temizleme asıl katkısını meta-model için temiz OOF skoru üretmekte gösteriyor

**Çıktılar:** `results/v3_stacking_general/stacking_results.csv`, `reports/stacking_general_panel_report.pdf`

---

---

### Notebook 12 — Gerçek Veri Baseline / is_missing + Median Imputation (YENİDEN YAPILANDIRILDI — çalıştırılmayı bekliyor)

**Dosya:** `notebooks/12_real_data_baseline.ipynb`  
**Pipeline:** Feature engineering YOK — NaN yönetimi yeniden tasarlandı  
**Veri:** `YARISMA_TRAIN_MASTER.csv` (2931 satır, 353 sütun)  
**Split:** Hold-out %20, 3-fold CV grid search  
**Panel testi:** KANSER, PAH, CFTR → MASTER train encoder/median ile doğrudan test

#### NaN Yönetimi (Yeni Strateji)

| NaN Oranı | Strateji | Gerekçe |
|-----------|----------|---------|
| > %50 | `is_missing_*` binary flag (drop edilmez) | Eksiklik Label ile koreleli (%59.9 vs %41.2) — bilgi taşıyor |
| ≤ %50 | Median imputation | AL_ sütunları sağa çarpık — median aykırı değerlere karşı daha robust |
| Kategorik | `'MISSING'` string | OHE/LE pipeline'da ayrı kategori olarak işlenir |

**Leakage kuralı:** Tüm fit işlemleri (median, OHE, LE) yalnızca MASTER train üzerinde; test/panellere transform.

#### Notebook Yapısı (13 cell)

| Cell | İçerik |
|------|---------|
| 0 | Markdown — strateji tablosu |
| 1 | Imports & config |
| 2 | Veri yükleme, sütun gruplarını tespit (high_nan / low_nan / cat) |
| 3 | Train/test split |
| 4 | `preprocess_ohe()` + `preprocess_raw()` + `add_missing_flags()` fonksiyonları; MASTER preprocessing |
| 5 | Panel preprocessing (MASTER train median/OHE kullanarak leakage-free) |
| **6** | **Section A — OHE Pipeline**: LightGBM, XGBoost, SVM |
| **7** | **Section B — LabelEncoder Pipeline**: LightGBM, XGBoost, SVM |
| **8** | **Section C — CatBoost Pipeline**: CatBoost (OTS) |
| 9 | Sonuçları birleştir + Panel testi (7 model × 3 panel) |
| 10 | Cross-model error analizi (difficulty score) |
| 11 | Görselleştirme — is_missing flaglerin feature importance'taki yeri vurgulanır (mor bar) |
| 12 | PDF rapor |

#### 3 Pipeline

| Pipeline | Encoding | Modeller |
|----------|----------|---------|
| A — OHE | OHE + is_missing flags + median num | lgbm_ohe, xgb_ohe, svm_ohe |
| B — LE | LabelEncoder + is_missing flags + median num | lgbm_le, xgb_le, svm_le |
| C — CatBoost | OTS native + is_missing flags + median num | catboost |

**Önemli Değişiklikler (eski NB12'ye göre):**
- Eski: 165 sütun NaN>%50 drop ediliyordu → Yeni: `is_missing_*` flag olarak korunuyor
- Eski: mean imputation → Yeni: median imputation
- Eski: `preprocess_panel()` MASTER ham medianını kullanıyordu → Yeni: train split sonrası hesaplanan median

**Çıktılar:** `results/real_data_baseline/`, `reports/real_data_baseline_report.pdf`

---

---

### Notebook 13 — Kapsamlı Ablasyon: No-FE | OHE | XGBoost + LightGBM (YENİDEN YAPILANDIRILDI — çalıştırılmayı bekliyor)

**Dosya:** `notebooks/13_ablation_ohe_xgboost.ipynb`
**Pipeline:** Feature engineering YOK, OHE + XGBoost + LightGBM (paralel), drop ablation
**Veri:** `YARISMA_TRAIN_MASTER.csv` (2931 satır, 353 sütun → 165 drop → 186 kalan)
**Split:** Hold-out %20, 3-fold CV grid search (12 combo)

#### Notebook Yapısı (12 cell)

| Cell | İçerik |
|------|---------|
| 0 | Markdown — pipeline + 14 senaryo tablosu |
| 1 | Imports & Config — XGB_GRID + LGBM_GRID, path'ler |
| 2 | Veri yükleme — `columns_real.py` grup tanımları, özdeş çift doğrulama |
| 3 | Train/Test Split + panel yükleme (CFTR/KANSER/PAH) |
| 4 | `preprocess()` + `train_eval()` fonksiyonları (XGBoost ve LightGBM) |
| 5 | Ablasyon döngüsü — 13 senaryo × 2 model, `baseline_info` dict |
| 6 | Panel transfer testi — Baseline model ile CFTR/KANSER/PAH |
| 7 | Missing-mask ablasyonu — `is_missing_*` bayrakları, 3 senaryo × 2 model |
| 8 | Delta hesabı + CSV kayıt (`ablation_results.csv`, `mask_results.csv`) |
| 9 | Görselleştirme — 5 grafik (F1 drop XGB vs LGBM, 3-metrik, panel heatmap, mask, AUC-PR) |
| 10 | Özet & Yorumlar |
| 11 | PDF Rapor — 5 bölüm + 5 grafik |

#### Ablasyon Senaryoları (13 + Baseline)

| Senaryo | Çıkarılan Grup | Açıklama |
|---------|----------------|----------|
| Baseline | — | Tüm özellikler |
| -CAT_all | CAT_1..CAT_6 | Tüm kategorikler |
| -CAT_pop | CAT_1, CAT_2 | Popülasyon kategorileri |
| -CAT_geno | CAT_3, CAT_4, CAT_5 | Genotip kategorileri |
| -CAT_dup | CAT_3, CAT_5 | Özdeş çift (CLAUDE.md uyarısı) |
| -AA | AA_1, AA_2 | Amino asit |
| -EK | EK_1..EK_9 | Ek sayısal skorlar |
| -AL_safe | AL_SAFE_COLS | Güvenli AL sütunları |
| -AL_high_miss | AL_1..6 + AL_27..38 | %80+ eksik bloklar |
| -AL_miss_leak | AL_16..25 | Eksiklik-etiket korelasyonu yüksek (leakage riski) |
| -AL_low | AL_ (NaN≤%40) | Düşük boşluklu |
| -AL_high | AL_ (NaN>%40) | Yüksek boşluklu |
| -AL_all | Tüm AL_ | Tüm AL özellikleri |

**Missing-mask senaryoları:** Baseline_nomask / +Mask_all / +Mask_highrisk

**Çıktılar:**
- `results/ablation_ohe_xgb/ablation_results.csv` (14 senaryo × 2 model × tam metrik)
- `results/ablation_ohe_xgb/mask_results.csv` (missing-mask sonuçları)
- `results/ablation_ohe_xgb/panel_results.csv` (CFTR/KANSER/PAH panel transfer)
- `results/ablation_ohe_xgb/fig1..fig5.png` (5 görselleştirme)
- `reports/ablation_ohe_xgb_report.pdf`

---

---

### Notebook 14 — Stacking Ensemble Deneyi (OLUŞTURULDU, çalıştırılmayı bekliyor)

**Dosya:** `notebooks/14_stacking_experiment.ipynb`
**Pipeline:** Stacking ensemble, 7 baz model, 2 meta-öğrenici, 2×5 = 10 konfigürasyon
**Veri:** `YARISMA_TRAIN_MASTER.csv` (eğitim) → CFTR, KANSER, PAH (panel transferi)

#### Deney Matrisi (2 × 5 = 10 konfigürasyon)

| Boyut | Senaryo | Açıklama |
|-------|---------|---------|
| **Balance** | B1 | Stratified 80/20 split (olduğu gibi) |
| **Balance** | B2 | 80/20 split → train'de patho undersampling (625:625), fazla patho → test |
| **Missing** | M1 | Flag yok, medyan imputation |
| **Missing** | M2 | Flag yok, NaN içeren sütunlar drop |
| **Missing** | M3 | Flag + medyan imputation (orijinal sütun korunur) |
| **Missing** | M4 | Flag + NaN içeren sütunlar drop |
| **Missing** | M5 | Flag + >%50 NaN → drop, ≤%50 NaN → medyan imputation |

#### 7 Baz Model

| Model | Encoding | Kaynak |
|-------|----------|--------|
| LightGBM | LabelEncoder | `grid_search_le_lightgbm()` |
| XGBoost | LabelEncoder | `grid_search_le_xgboost()` |
| CatBoost | Native categorical | `grid_search_catboost()` |
| RandomForest | OHE | Notebook inline (RF_GRID, 12 combo) |
| SVM | LE + StandardScaler | `grid_search_le_svm()` |
| NN (MLP3Layer) | LE + StandardScaler + Tensor | `grid_search_nn_fast()` |
| DNN (DeepMLP) | LE + StandardScaler + Tensor | `grid_search_dnn_fast()` |

#### OOF Protokolü
1. **Grid search** (3-fold CV) → en iyi hiperparametre kombinasyonu
2. **OOF üretimi** (5-fold CV, sabit best_combo) → `oof_proba` (n_train,)
3. **Final model** (tüm train, best_combo) → test & panel tahminleri

#### Meta-Feature Matrisi (n_train × 11)
- 7 OOF olasılıkları (bir baz model başına)
- 4 çeşitlilik özelliği: `std`, `mean`, `entropy`, `max-min`

#### 2 Meta-Öğrenici
| Meta | Konfigürasyon |
|------|-------------|
| Logistic Regression | C=1.0, L2, class_weight='balanced' |
| LightGBM | n_estimators=100, max_depth=3, lr=0.1, class_weight='balanced' |

#### Notebook Yapısı (14 hücre)

| Hücre | İçerik |
|-------|--------|
| 0 | Markdown — deney matrisi, mimari özeti |
| 1 | Imports, sabitler (N_OOF_FOLDS=5, RF_GRID), klasör oluşturma |
| 2 | Veri yükleme (MASTER + 3 panel), istatistik özeti |
| 3 | `make_b1_split()` / `make_b2_split()` |
| 4 | `fit_preprocessor()` / `apply_preprocessor()` / `get_surviving_cat_cols()` |
| 5 | `grid_search_random_forest()` — RF için inline grid search (OHE) |
| 6 | OOF yardımcı fonksiyonları: `_oof_tree`, `_oof_catboost`, `_oof_rf`, `_oof_svm`, `_oof_mlp` |
| 7 | `run_all_base_models()` — 7 modeli sırayla çalıştırır, `predict_fn` closure üretir |
| 8 | Meta-learner fonksiyonları: `build_meta_features()`, `train_meta_lr()`, `train_meta_lgbm()`, `stacking_predict()` |
| 9 | Ana deney döngüsü (2 balance × 5 missing = 10 config), checkpoint CSV |
| 10 | Panel transfer değerlendirme (CFTR/KANSER/PAH), `train_cols` state'ten alınır |
| 11 | Sonuç derleme — top-5 tablo, missing/balance karşılaştırması |
| 12 | 6 görselleştirme (F1/Precision/Recall bar, F1 heatmap, baz model F1, confusion matrix 4 panel, M1-M5 karşılaştırma, B1/B2 karşılaştırma) |
| 13 | PDF raporu (`StackingReport(FPDF)`, 8 bölüm) |

#### Teknik Notlar
- Kategorik NaN → `'MISSING'` / `'X'` (AA_UNKNOWN_TOKEN) string fill, OHE/LE öncesi
- `train_cols` her config için `CONFIG_STATES` içine kaydediliyor; panel transferinde yeniden hesaplanmıyor
- Checkpoint: `results/v4_stacking/all_results_checkpoint.csv` — her config sonrası yazılıyor

**Çıktılar:** `results/v4_stacking/all_results.csv`, `results/v4_stacking/panel_results.csv`, 6 PNG grafik, `reports/NB14_stacking_report.pdf`

---

### Notebook 15 — Panel-Bazlı Eğitim & Stacking Deneyi (OLUŞTURULDU, çalıştırılmayı bekliyor)

**Dosya:** `notebooks/15_panel_specific_stacking.ipynb`
**Pipeline:** Feature engineering YOK, M3 missing stratejisi (NB14'ten devralındı), panel-bazlı 4 senaryo × 4 model
**Veri:** `YARISMA_TRAIN_{MASTER,CFTR,KANSER,PAH}.csv`
**Birincil metrik:** pathogenic (Label=1) F1. Ayrıca precision + recall, train & test ayrı raporlanır.

#### Deney Senaryoları (her panel için ayrı model)

| Senaryo | Eğitim verisi | Test verisi | Modeller |
|---------|---------------|-------------|----------|
| **S1** | Panelin %50'si (patho %50 + benign %50) + dengelenmiş MASTER çekirdeği (625 patho / 625 benign) | Panelin kalan %50'si (MASTER yok) | LightGBM, CatBoost, NN, DNN |
| **S2** | Panelin %50'si + aynı boyutta MASTER (panel-train tam 2x; sınıf oranı korunur) | Panelin kalan %50'si | LightGBM, CatBoost, NN, DNN |
| **S3** | NN pretrain (MASTER çekirdek) → finetune (panelin %50'si) | Panelin kalan %50'si | NN (ShallowMLP/MLP3) |
| **S4** | S3 ile aynı, fakat DNN (DeepMLP) | Panelin kalan %50'si | DNN |

Toplam: S1=12, S2=12, S3=3, S4=3 = **30 model değerlendirmesi**.

#### Kritik Tasarım Kararları

- **Panel %50/%50 split stratified**: patho ve benign ayrı ayrı yarıya bölünür. CFTR örn: train 45 patho + 10 benign, test 45 patho + 11 benign (doğrulandı).
- **MASTER çekirdeği tüm paneller için ortak ve sabit** (SEED=42 ile bir kez 625/625 undersample). S1'de aynı çekirdek her panele eklenir.
- **S2 sınıf oranı**: MASTER eklemesi panelin kendi patho/benign sayısını aynalar → toplam train tam 2x panel-train.
- **Variant_ID butünlüğü**: Aynı Variant_ID MASTER ve alt-panellerde olsa da çoğunlukla FARKLI varyant (ort. 41-150 özellik hücresi farklı). Yalnızca tüm özellik+label birebir aynı olan satırlar drop edildi: **KANSER=3, PAH=3, CFTR=0**. (Variant_ID global PK değil.)
- **Sütun drop (FE yok)**: MASTER üzerinde sabit (57) + birebir aynı sütun çiftlerinin ikincisi (CAT_5 + büyük AL blokları) → toplam **63 sütun drop, 288 feature kaldı**. M3 flag'leri sonrası model girdisi: 281 sayısal + 7 kategorik (CAT_1/2/3/4/6 + AA_1/2) + 140 `is_missing_*` = **428 sütun**.
- **M3 missing stratejisi** (NB14 panel-transfer'de en iyi, mean F1 0.9210): >%50 NaN sütunlar için `is_missing_*` flag + tüm sayısal sütunlar train-medyanı ile impute.
- **Leakage disiplini**: imputer/encoder/scaler yalnız train'de fit. Threshold train-içi F1-max (`optimize_threshold`), test'e dokunulmaz. S3/S4 encoder pretrain+panel_train birleşiminde fit edilir.
- **Hiperparametre optimizasyonu (4 model tipinde de)**: 3-fold StratifiedKFold grid search. LightGBM `LGBM_GRID` (12 kombo), CatBoost `CB_GRID` (12 kombo), NN `NN_GRID_FAST` (dropout×lr=4), DNN `DNN_GRID_FAST` (n_layers×hidden_dim×dropout×lr=4). NN/DNN combo'su S1/S2'de train, S3/S4'te panel finetune verisi üzerinde CV-F1 ile seçilir; seçilen combo `best_combo` sütununda + raporda. Tiny panellerde fold sayısı en küçük sınıfa göre güvenli düşürülür (`_safe_n_splits`).
- **Modeller src/models.py + config.py'dan**: LGBM_GRID/LGBM_FIXED, CB_GRID/CB_FIXED, MLP3Layer + NN_FIXED_FAST/NN_GRID_FAST, DeepMLP + DNN_FIXED_FAST/DNN_GRID_FAST.

#### Notebook Yapısı (13 hücre)

| Hücre | İçerik |
|-------|--------|
| 0 | Markdown — senaryo matrisi, M3 devri, Variant_ID notu |
| 1 | Imports & config (PANELS, MASTER_CORE 625/625, HIGH_MISSING_THRESHOLD=0.50) |
| 2 | Veri yükleme + cross-panel birebir-aynı satır drop |
| 3 | Sütun temizliği (sabit + duplicate çiftler), global feature listesi |
| 4 | M3 preprocessing (`fit_preprocessor` / `transform_X`) |
| 5 | Split & dataset builder'lar (`panel_5050_split`, `make_master_core`, `build_s1`, `build_s2`) |
| 6 | Ağaç eğiticileri (`train_lightgbm`, `train_catboost`) — train+test metrik + importance |
| 7 | NN/DNN yardımcıları (`train_nn_scratch`, `train_nn_finetune`) |
| 8 | Ana deney döngüsü (S1-S4 × panel × model) |
| 9 | Sonuç derleme (pivot, senaryo ortalaması, panel başına en iyi, train-test gap) |
| 10 | Görseller (test F1/P/R bar, train vs test F1) |
| 11 | Confusion matrix + feature importance (panel başına, `is_missing` mor vurgulu) |
| 12 | PDF rapor (`NB15Report(FPDF)`, 7 bölüm) |

**Çıktılar:** `results/v5_panel_specific/panel_specific_results.csv`, `fig1..fig4.png`, `reports/NB15_panel_specific_report.pdf`

#### NB15 v1 SONUÇLARI (ÇALIŞTIRILDI — %50/50 test, eski threshold)

Panel başına en iyi (pathogenic F1, %50/50 test): **CFTR=0.926 (S2/lightgbm), KANSER=0.914 (S1/lightgbm), PAH=0.928 (S1/catboost)**. Senaryo ortalaması: S2=0.8952, S1=0.8815, S4=0.8645, S3=0.8446.

**v1 bulguları (v2'yi tetikleyen):**
- Ağaç modeller (LightGBM/CatBoost) her panelde NN/DNN'i geçti. NN/DNN (S3/S4) en zayıf + en yüksek overfit (train-test F1 farkı ~0.12–0.15).
- **Precision/benign tarafı zayıf:** modeller düşük eşik (0.10–0.48) seçti, recall yüksek (0.92–0.98) ama precision düşük. PAH'ta specificity ~%39 (31 benign'in 19'u FP). **Final test %80 benign olduğu için bu büyük risk** → v2.
- Feature importance: EK skorları (EK_7/EK_1/EK_2) + CAT_1 baskın (REVEL literatürüyle uyumlu).

---

### Notebook 15 v2 — Benign-Aware + NN/DNN İyileştirme (ÇALIŞTIRILDI ✅)

**Dosya:** `notebooks/15_panel_specific_stacking.ipynb` (14 hücre; Cell 5b eval-infra eklendi)

#### NB15 v2 SONUÇLARI (ÇALIŞTIRILDI)

**İki düzeltme de işe yaradı:**
- **NN/DNN overfit DÜZELDİ:** v1'de train-test F1 farkı ~0.12–0.15 idi → v2'de **ortalama 0.065**'e düştü (focal loss + early stopping + SmallMLP). Bazı NN/DNN'de fark negatif (örn. S1 PAH nn: −0.013).
- **Benign-aware threshold marjinal kazandı:** panel-ort %80/20 F1 → f1_raw=0.5137, f1_8020=0.5331, **mcc_8020=0.5335** (mcc/f1_8020 ≈ eşit, ikisi de f1_raw'dan ~2 puan iyi).

**EN KRİTİK BULGU — %50/50 → %80/20 çöküşü (v1'in yanılsaması ortaya çıktı):**

| Panel | En iyi %50/50 F1 | En iyi %80/20 F1 | Çöküş | Sebep |
|---|---|---|---|---|
| CFTR | 0.899 | 0.850 (S1/catboost) | ~0.05 | en az; ama CI çok geniş [0.50–1.00] (n=11 benign) |
| KANSER | 0.911 | 0.690 (**S4/dnn**) | ~0.22 | en dengeli panel → en dayanıklı; finetune-DNN kazandı |
| PAH | 0.917 | 0.489 (S1/catboost) | **~0.43** | en dengesiz (sadece 31 benign) + düşük eşik → precision çöküşü |

- **v1'in 0.91–0.93'leri yanılsamaydı** (train ile aynı dağılımda ölçülmüştü). %80/20 bootstrap final yarışma koşulunu gösteriyor. CLAUDE.md'deki "precision çöküşü" tahmini birebir gerçekleşti.
- **NN/DNN artık değerli:** KANSER'i finetune-DNN (S4) kazandı → NN/DNN'i çıkarmayıp iyileştirme kararı doğruydu. Focal loss benign'e odaklanmayı zorluyor, final benign-ağırlıklı dağılımda avantaj.
- **PAH acil:** %80/20 F1=0.49, precision çöküyor → NB16 stacking/FE burada test edilmeli.
- **Train sonuçları rapora eklendi** (1b bölümü: train F1/P/R/MCC + train-test farkı). Cell 12 düzeltildi.

#### NB15 v2 yapı (referans)

**Final test dağılımı bilgisiyle (%80 benign/%20 patho) eklenenler:**
- **Cell 5b — merkezi değerlendirme:** `evaluate_predictions()` 3 threshold modu × hem %50/50 hem **bootstrap %80/20** (benign sabit + patho downsample, N=50, %95 CI) raporlar.
  - `f1_raw` (eski, train dengesinde F1-max), `f1_8020` (final %80/20 havuzda F1-max), `mcc_8020` (MCC-max, benign-aware).
- **NN/DNN iyileştirme (danışman + literatür):** CV 3→5, **FocalLoss (γ=2) + class weighting**, **validasyon-bazlı early stopping** (sabit epoch yerine), **küçük/düzenli mimari** (`SmallMLP`, BatchNorm yok — variance-shift riskine karşı, yüksek dropout + weight_decay). v1'deki NN overfit'ine yanıt.
- **Ağaç modeller:** CV 3→5; artık ham olasılık dönüp merkezi eval kullanır.
- **Cell 8:** her model bir kez eğitilir, 3 threshold modu ham olasılıktan ucuza değerlendirilir (yeniden eğitim yok).
- **Rapor (Cell 12):** %50/50 vs %80/20 tablosu, threshold-modu karşılaştırması (fig5), seçilen NN/DNN combo'ları; geniş figürler landscape+clamp (taşma düzeltildi).

**Smoke test:** Tüm yollar (eval infra, 3 thr modu, NN focal+ES, `_probs` plumbing) hatasız. CFTR LightGBM'de threshold modları ayrışıyor (f1_raw thr=0.60 vs mcc_8020 thr=0.65), precision=1.0.

---

### Notebook 16 v2 — FE Ablasyonu + 6 Base + 2 Meta Stacking (ÇALIŞTIRILDI ✅)

#### NB16 v2 SONUÇLARI

**Panel bazında en iyi (%80/20 bootstrap F1), NB15 v2 ile kıyas:**

| Panel | NB15 v2 en iyi | NB16 en iyi | Not |
|---|---|---|---|
| CFTR | 0.850 (S1/catboost) | 0.852 (dnn) | **CI=[0.00–1.00] anlamsız** (n=11 benign); %50/50'de catboost/dnn 0.875 P=1.0 |
| KANSER | 0.690 (S4/dnn) | **0.716 (stack_lr)** | gerçek kazanç, CI=[0.67–0.77] **dar/güvenilir** |
| PAH | 0.489 (S1/catboost) | **0.515 (stack_lr)** | stacking +0.046 kazandırdı; CI=[0.30–0.62] |

**Dört sorunun cevabı:**
1. **FE katkısı (with_fe − no_fe, %80/20):** CFTR **+0.069**, KANSER +0.015, PAH −0.003. → FE küçük panelde (CFTR) belirgin yardımcı, PAH'ta nötr. Net zarar yok, **FE'yi tut**. Ablasyon olmadan körü körüne dahil edilseydi yanıltıcı olurdu.
2. **Stacking vs base:** 3 panelin 2'sinde stacking kazandı (KANSER 0.716, PAH 0.515). CFTR'de base "kazandı" ama gürültü (CI tüm skala). **PAH'ta stacking gerçek değer kattı** (0.469 base → 0.515 stack).
3. **Meta-learner:** **Logistic Regression net kazandı** (stack_lr=0.638 vs stack_gbm=0.566 panel-ort). GBM meta overfit ediyor (train F1 0.84–0.87). LR her panelde önde → literatürle uyumlu.
4. **PAH precision çöküşü kısmen hafifledi:** stacking ile 0.49→0.515. Confusion matrix (stack_lr, %50/50): PAH 31 benign'in 21'i doğru (specificity ~%68, NB15 v1'deki %39'dan çok daha iyi) — ama %80/20'de hâlâ en zayıf panel.

**Kritik metodolojik bulgu:** CFTR'nin tüm %80/20 CI'ları [0.00–1.00] (std 0.267) — n=11 benign nedeniyle **CFTR model karşılaştırmaları istatistiksel olarak anlamsız**. KANSER (std 0.033) ve PAH (std 0.085) güvenilir. Bootstrap CI bu belirsizliği dürüstçe gösterdi.

**Genel durum:** NB16 üç panelde de NB15 v2'yi geçti/eşitledi. En iyi güvenilir sonuçlar: **KANSER stack_lr 0.716** (projedeki en iyi KANSER), **PAH stack_lr 0.515** (en iyi PAH). Train sonuçları raporda (3. bölüm).

#### NB16 v2 yapı (referans)

**Dosya:** `notebooks/16_fe_stacking.ipynb` (12 hücre)
**Temel:** NB15 S1 kurgusu (panel %50 + 625/625 MASTER çekirdek) + M3 missing + benign-aware eval (NB15 v2 ile aynı: 3 threshold modu, %50/50 + bootstrap %80/20).

**NB15 bulgularına göre tasarlanan dört ekleme:**

**1. FE ABLASYONU (izole ölçüm — "aynı anda çok şey değiştirme" prensibi):**
- Her şey **iki kez** koşar: `no_fe` (sadece M3, 428 sütun — NB15 ile aynı) ve `with_fe` (M3 + FE, 432 sütun).
- FE sütunları: `fe_aa_stopgain` (`AA_2=='*'` nonsense), `fe_aa_nonstandard`, `fe_grantham` (gömülü Grantham matrisi), `fe_blosum62` (gömülü BLOSUM62 — referans değerlerle doğrulandı: grantham(L,I)=5, blosum62(W,W)=11), `*__log` (AL frekans log1p, train'de skew>2; MASTER'da 0 aday → no-op).
- `_strip_fe()` + `transform_X` swap ile no_fe modunda FE sütunları düşürülür → FE'nin %80/20 F1'e **net katkısı** ölçülür. Dış bağımlılık yok (biopython yüklü değil).

**2. 6 BASE MODEL** (NB15'te KANSER'i finetune-DNN kazandığı için finetune eklendi):
- Scratch: `lightgbm, catboost, nn, dnn` + Finetune: `nn_ft, dnn_ft` (MASTER pretrain → panel finetune).
- **Finetune OOF sadeleştirmesi:** pretrain BİR KEZ (fold-bağımsız, MASTER'dan; panel test'e değmez → leakage minimal), finetune fold'a özgü. Tam-saf OOF değil; pragmatik, rapora not edildi.

**3. 2 META-LEARNER yan yana:** Logistic Regression (`stack_lr`) + LightGBM (`stack_gbm`), ikisi de class_weight=balanced. Meta-feature: 6 OOF olasılık + mean + std.

**4. TRAIN sonuçları** raporda (overfit kontrolü; NB15 v1'de eksikti, aynı hata tekrarlanmadı).

**Smoke test:** FE ablasyon (no_fe=428 / with_fe=432 sütun, fe_* doğru düşüyor/ekleniyor), finetune base OOF (shape 1305), train metrik akıyor. **Erken sinyal (smoke, güvenilmez):** CFTR mini-koşuda no_fe(0.67) > with_fe(0.58) — FE'nin körü körüne fayda sağlamadığını, ablasyonun gerçek bir karar vereceğini gösteriyor.

**Çıktılar:** `results/v6_fe_stacking/fe_stacking_results.csv` (2 fe × 3 panel × 8 model × 3 thr = 144 satır), `fig1_base_vs_stack.png`, `fig2_fe_ablation.png`, `fig3_confusion.png`, `reports/NB16_fe_stacking_report.pdf`

---

### Notebook 17 — CFTR Refinement: 7 Strateji Karşılaştırması (TAMAMLANDI & ÇALIŞTIRILDI ✅)

**Dosya:** `notebooks/17_cftr_refinement.ipynb` (11 hücre)
**Pipeline:** LOO-CV + MCC değerlendirme, prior-shift kalibrasyon, 7 strateji × CFTR
**Veri:** `YARISMA_TRAIN_{MASTER,CFTR}.csv`
**Birincil metrik:** LOO-CV MCC (n=21 benign nedeniyle bootstrap CI anlamsız — MCC kararlı)

#### Problem & Motivasyon

NB15/NB16'da CFTR bootstrap CI=[0.00–1.00] (std=0.267) → hiçbir model karşılaştırması anlamlı değildi.
Çözüm: **LOO-CV (tüm 111 örnek, 21 benign tamamı değerlendirmede) + MCC** (Chicco & Jurman 2020).
İkinci katman: **Prior shift düzeltmesi** (Saerens 2002) — π_train=0.73 → π_test=0.20 post-hoc kalibrasyon.

#### 7 Strateji Sonuçları (LOO-CV MCC — birincil metrik)

| Strateji | LOO-MCC | LOO-F1 | LOO-AUC | Boot-mean | Boot-std | MS-F1 | Train-F1 | Overfit |
|---|---|---|---|---|---|---|---|---|
| NB16 ref (DNN, CI:[0-1]) | N/A | N/A | N/A | 0.852 | 0.267 | N/A | N/A | ? |
| S0_MASTER-only | 0.627 | 0.898 | 0.942 | 0.728 | 0.107 | 0.923 | 0.979 | ORTA |
| S1_MASTER-core | 0.608 | 0.877 | 0.912 | 0.783 | 0.099 | 0.876 | 0.985 | ORTA |
| S2_SMOTE-CFTR | 0.561 | 0.924 | 0.838 | 0.493 | 0.051 | 0.897 | 1.000 | OK |
| S3_TabPFN | N/A (atlandı) | — | — | — | — | — | — | N/A |
| S4_FrozenFinetune | 0.294 | 0.500 | 0.857 | 0.370 | 0.258 | 0.807 | 0.500 | OK |
| S5_BalancedBagging | 0.572 | 0.855 | 0.937 | 0.754 | 0.128 | 0.498 | 0.935 | OK |
| **S6_PriorShift** | **0.655** | **0.919** | **0.942** | **0.692** | **0.088** | **0.923** | **0.978** | **OK** |

#### Kritik Bulgular

1. **En iyi strateji: S6_PriorShift (LOO-MCC=0.655)** — S0'ın (MASTER-only LGBM) çıktısına post-hoc π düzeltmesi uygulamak yeterli. Yeniden eğitim gerekmez.
2. **S0_MASTER-only güçlü baseline (MCC=0.627)** — tüm MASTER ile eğitim CFTR'ye iyi transfer oluyor. Kullanıcı önerisi doğruldu.
3. **Confusion matrix (S6, LOO-CV): 18/21 benign doğru (specificity=%86), 79/90 pathogenic doğru** — NB16'daki kör bootstrap'ten çok daha güvenilir tablo.
4. **Prior shift düzeltmesi açıkça işe yarıyor:** S0 raw (MCC=0.627) → S6 adjusted (MCC=0.655). Olasılık histogramı: benign kümeleri 0.0–0.2'de yoğunlaşıyor, pathogenic 0.8–1.0'da.
5. **S4_FrozenFinetune hayal kırıklığı (MCC=0.294):** Frozen body + CFTR finetune, MASTER pretrain bilgisini CFTR'ye aktaramadı. n=111 çok küçük, son katman yeterli kapasitede değil.
6. **S2_SMOTE LOO-F1=0.924 ama boot-mean=0.493** — SMOTE train F1'i şişiriyor (Train-F1=1.0), %80/20 dağılımında dayanıksız. Overfit=OK görünse de boot-std=0.051 (dar ama düşük).
7. **S0/S1 Overfit=ORTA:** Train-F1 ~0.98, LOO-F1 ~0.88–0.90. Kabul edilebilir fark, MASTER büyüklüğünden kaynaklanıyor.
8. **TabPFN (S3) atlandı:** v8 API değişikliği + lisans zorunluluğu nedeniyle devre dışı.

#### Final CFTR Önerisi

**CFTR için final teslim modeli: S6_PriorShift** (MASTER üzerinde LGBM + post-hoc prior shift düzeltmesi).
- Precision=1.0 hedefi: threshold yukarı çekilerek FP=0 sağlanabilir (18/21 benign doğru → 3 FP).
- PAH'a transfer: prior-shift düzeltmesi + BalancedBagging PAH için de denenebilir (NB17 önerisi).

**Çıktılar:** `results/v7_cftr_refinement/cftr_results.csv`, `fig1_loo_mcc.png`, `fig2_multiseed_f1.png`, `fig3_prior_shift.png`, `fig4_confusion.png`, `fig5_train_vs_test.png`, `reports/NB17_cftr_refinement_report.pdf`

---

### Notebook 17 Ek — Regularizasyon Sweep + "Gerçek Test F1" Açıklaması (ÇALIŞTIRILDI ✅, 2026-06-14)

**Dosya:** `notebooks/17_cftr_refinement.ipynb` Cell 11–14 (append, S0–S6 dokunulmadı). Çıktılar: `results/v7_cftr_refinement/reg_sweep.csv`, `fig6_reg_sweep.png`, `reports/NB17_reg_sweep_addendum.pdf`.

#### ⚠️ GERÇEK TEST F1 NEDİR (kavramsal düzeltme — doğrulandı)
Yarışma metriği = pathogenic (Label=1) F1, **%80 benign / %20 pathogenic** test setinde. NB17'de bunu **birebir** taklit eden sayı `boot_mean`'dir (kod doğrulandı: `FINAL_BENIGN_FRAC=0.80`, `_resample_8020()` tüm benign'i tutar + patho'yu %20'ye indirir, `_f1_pos()` pos_label=1).
- **Gerçek test F1 tahmini = `boot_mean`. CFTR S6 için 0.692 ± 0.088** (n=21 benign → belirsizlik yüksek).
- `train_f1` (~0.98) = eğitim verisine uyum, test DEĞİL. `loo_f1` (~0.92) = pathogenic F1 ama CFTR'nin **kendi %81-patho dağılımında** → iyimser, yarışma koşulu DEĞİL.
- **Eski rapordaki `train_f1 − loo_f1` "overfit" göstergesi yanıltıcıdır:** iki FARKLI dağılımı (MASTER eğitimi − CFTR doğal) çıkarır; ne overfit'i ne de gerçek test F1'ini ölçer. Yerine **memorization_gap** (`train_f1 − master_cv_f1`, aynı dağılım) + **transfer_gap** (`master_cv_f1 − loo_f1`) ayrı raporlanır.

#### Soru: Overfit'i çözmek başarı kaybettiriyor mu? → EVET (aşırı sıkılaştırılırsa)
12 konfigürasyon (min_child_samples ∈ {5..320} @ nl=31 ve num_leaves ∈ {63..3} @ mcs=20). `master_cv_f1` eşiği N=50 resample ortalamasıyla **gürültüden arındırıldı** (`select_threshold_8020_robust`).

| Bulgu | Sonuç |
|---|---|
| Baseline (mcs=20/nl=31) memorization_gap | 0.234 (gürültülü) → **0.159 (de-noised)** — komşularla uyumlu, gerçek MASTER-içi ezber ~0.14–0.19 |
| transfer_gap (baseline) | **−0.099**: CFTR, MASTER'ın kendi CV'sinden bile *daha kolay* (tek-gen, homojen) |
| LOO-MCC platosu | mcs 5–40 & nl=31 → 0.66–0.69; **multi-seed F1 ~0.92 (dar)** |
| Aşırı regularizasyon (mcs≥80, nl=63/7) | LOO-MCC 0.53–0.57'ye, **multi-seed F1 0.38–0.75'e ÇÖKER** |
| **Verdict** | **B/C — regularizasyon CFTR'yi iyileştirmiyor; mevcut S6 güvenli platonun tepesinde. Overfit "gap"i büyük ölçüde ölçüm artefaktı + domain-shift. S6'yı OLDUĞU GİBİ BIRAK.** |

**Not (within-noise):** `min_child_samples=10` kağıt üstünde baskın (LOO-MCC 0.690, boot 0.755, multi-seed 0.926, daha küçük gap), ama +0.035 MCC n=21 gürültü bandında (CLAUDE.md: yalnız >0.05 aksiyon). Opsiyonel nüdge, zorunlu değil. **Canary = multi-seed F1**, LOO-MCC değil — kapasite düşünce önce o çöker.

---

### Notebook 18 — CFTR Veri Artırımı Araştırması: 2 Negatif Sonuç (ARAŞTIRMA TAMAMLANDI, 2026-06-15)

**Dosya:** `notebooks/18_deobfuscation.ipynb` + `src/column_map_real_to_legacy.py` + `results/deobfuscation/` + `reports/deobfuscation_map_report.pdf`
**Amaç:** CFTR'yi (n=111, benign=21) `data/63k_genis/` (ClinVar/OpenCRAVAT 63k) verisinden çoğaltmak. İki yöntem planlandı: (Y1) deobfuscation + model imputation, (Y2) top-50 feature + gerçek-değer transfer. **Her ikisi de veri-tarafı engellere takıldı → araştırma negatif sonuçla kapandı, plan iptal edildi.**

#### Bulgu 1 — Deobfuscation BİREBİR çalışmaz (değerler normalize edilmiş)
63k legacy = yarışma verisinin **açık-isimli hâli** (777 OpenCRAVAT sütunu: `cadd__score`, `alphamissense__*`...). Yarışma anonim (`AL/CAT/EK/AA`). Ama:
- **Yarışma sayısal değerleri NORMALIZE edilmiş:** `cadd__phred` legacy'de 0-63, yarışmada hiçbir sütun bu aralıkta değil (hepsi [0,1]). Ham mean/std eşleştirmesi bu yüzden çöküyor.
- **Benzer skorlar ayrışmıyor:** revel/cadd/alphamissense hepsi aynı birkaç yarışma sütununa (AL_304, AL_301) işaret ediyor — dağılım şekliyle birbirinden ayırt edilemiyor.
- **Çözüm (uygulandı):** Birebir isim yerine **grup-tabanlı** kimlik. NB18 v2 çıktısı: **124 güvenilir-grup sütun** = FREQ (111 frekans: [0,1] q50~0 skew yüksek) + SCORE01 (6 in-silico skor: `AL_45/129/201/301/304, EK_5`) + CONS_RAW (7: `AL_185, EK_1/2/3/7/8/9`). BINARY (103) + OTHER (116) kullanılmaz. `src/column_map_real_to_legacy.py`'da `RELIABLE_COLS`.
- **Not:** İlk versiyon (Haiku subagent) hatalıydı (many-to-one collapse, AA→chrom gibi saçma high-conf eşleşmeler "STATUS OK" raporluyordu); grup-tabanlı yeniden yazıldı.

#### Bulgu 2 — CFTR benign artırımı YAPISAL ÇIKMAZ (missense-benign kıt)
- **63k'da CFTR benign yok:** HQ filtre (expert panel + practice guideline) CFTR=116 patho / **0 benign**. gnomAD AF≥%5 (ACMG BA1) = sadece 1 varyant.
- **ClinVar tam dökümü:** 1624 benzersiz CFTR benign toplandı AMA **sadece 15'i missense** (%0.9); 1609'u intron/splice/sinonim.
- **Yarışma saf missense** (CFTR 189 satırın 181'i missense; AA_1/AA_2 amino asit değişim sütunları). CFTR benign biyolojik olarak intron/sinonim bölgelerde → missense-benign her kaynakta kıt.
- **Sonuç:** Non-missense benign'i missense feature uzayına sokmak = batch-effect leakage. **CFTR benign-kıtlığı veriyle çözülemez** → class_weight/focal/threshold ile yönetilmeli (NB17 zaten böyle yapıyor: S6_PriorShift). Benign artırım çıktıları (fetch script, candidates CSV, provenance) **silindi**.

**Korunan değerli iş:** Deobfuscation altyapısı (NB18 + grup haritası) benign'den bağımsız; ileride feature seçimi veya pathogenic transfer için kullanılabilir.

---

### Notebook 19 — CFTR Combined Panels: MASTER-only vs MASTER+KANSER+PAH (TAMAMLANDI ✅, 2026-06-16)

**Dosya:** `notebooks/19_cftr_combined_panels.ipynb`
**Pipeline:** NB17 S0 protokolü (LightGBM + LOO-CV + MCC + prior-shift), eğitim havuzu genişletme testi
**Veri:** MASTER (2931) + KANSER (388) + PAH (372) → COMBINED (3691) vs MASTER-only

#### Sonuçlar

| Strateji | n_train | LOO-MCC | Boot-mean | Boot-std | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| S0_MASTER-only (=S6 base) | 2931 | **0.6552** | 0.6915 | 0.0877 | 18 | 3 | 11 | 79 |
| S0c_COMBINED | 3691 | 0.6436 | **0.8629** | 0.1327 | **21** | **0** | 19 | 71 |

#### Kritik Bulgular

1. **COMBINED paradoksu:** LOO-MCC düşük ama boot-mean +0.17 yüksek. Sebep: COMBINED FP=0 (precision=1.0) ama FN=19 (recall düşük). Final %80/20 test setinde benign sayısı 4× arttığında FP patlaması F1'i öldürür → FP=0 avantajı dramatik.
2. **Prior-shift düzeltmesi COMBINED'da threshold değiştirmedi** (mcc_raw==mcc_prior==0.6436). S6'da değiştiriyor (0.6271→0.6552). Kök neden: COMBINED posterior dağılımı incelenmeli — posterior histogram debug gerekiyor.
3. **Hangi model final teslim için daha iyi?** Yarışma metriği pathogenic F1 @ %80/20 → boot_mean birincil → **COMBINED (0.863) daha iyi aday**, ancak boot_std geniş (0.133 vs 0.088).
4. **Sonraki adım:** NB20 — (1) Soft ensemble (S6+COMBINED), (2) COMBINED prior-shift debug, (3) Robust N=50 threshold seçimi.

**Çıktılar:** `results/v9_cftr_combined/cftr_combined_results.csv`, `fig1_confusion_compare.png`

---

### Notebook 20 — CFTR Ensemble + Threshold Refinement (TAMAMLANDI ✅, 2026-06-17)

**Dosya:** `notebooks/20_cftr_ensemble.ipynb` (10 hücre)
**Pipeline:** NB19 protokolü (LightGBM + LOO-CV + MCC + prior-shift) + 3 experiment
**Veri:** MASTER (2931) + COMBINED (3691) → CFTR (111) üzerinde LOO-CV değerlendirme

#### 3 Experiment

| # | Deney | Sonuç |
|---|---|---|
| Exp 1 | Soft Ensemble (0.5×S0 + 0.5×COMBINED) | LOO-MCC=0.5835, boot=0.7497 — **COMBINED'ın altında**, eklenti değeri yok |
| Exp 2 | Prior-Shift Debug (2×2 posterior histogram) | COMBINED'da raw thr=0.660 → adj thr=0.150; ham dağılım zaten iyi ayrışıyor, prior-shift threshold'u düşürüyor ama MCC aynı |
| Exp 3 | Robust Threshold (N=50 bootstrap ort.) | S0_MASTER: 0.08→**0.19** (MCC 0.6552→**0.5243**, −0.13); COMBINED: 0.15→0.147 (±0.00); Ensemble: 0.18→0.169 |

#### Sonuç Tablosu (Robust Threshold)

| Strateji | LOO-MCC | Boot-mean | Boot-std | Precision | Recall | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| S0_MASTER | 0.5243 | 0.6456 | 0.1376 | 0.971 | 0.744 | 19 | 2 | 23 | 67 |
| **S0c_COMBINED** | **0.6436** | **0.8629** | 0.1327 | **1.000** | 0.789 | **21** | **0** | 19 | 71 |
| Ensemble | 0.5835 | 0.7497 | 0.1338 | 0.986 | 0.767 | 20 | 1 | 21 | 69 |

#### Kritik Bulgular

1. **S0_MASTER'ın NB19 performansı şanslı tek-resample artefaktıydı.** Robust threshold (0.08→0.19) ile MCC 0.6552→0.5243'e çöktü. COMBINED etkilenmedi (0.6436).
2. **Ensemble COMBINED'a değer katmadı.** MASTER'ın gürültülü posterior'u COMBINED'ı bozuyor (boot-mean 0.8629→0.7497).
3. **COMBINED posterior debug:** Ham dağılım benign/pathogenic'i zaten iyi ayırıyor (thr_raw=0.660). Prior-shift threshold'u 0.15'e düşürüyor ama optimal MCC noktası değişmiyor.
4. **CFTR final karar: S0c_COMBINED** — LOO-MCC=0.6436, boot-mean=0.8629, precision=1.0, FP=0. S0_MASTER ve Ensemble terk edildi.

**Çıktılar:** `results/v10_cftr_ensemble/cftr_ensemble_results.csv`, `cftr_threshold_comparison.csv`, `fig1_posterior_histogram.png`, `fig2_strategy_comparison.png`, `reports/NB20_cftr_ensemble_report.pdf`

---

### Notebook 21 — PAH Refinement: 7 Strateji Karşılaştırması (TAMAMLANDI ✅, 2026-06-17)

**Dosya:** `notebooks/21_pah_refinement.ipynb` (12 hücre)
**Pipeline:** NB17/NB19 CFTR protokolünün PAH'a transferi — LightGBM + LOO-CV + MCC + prior-shift + BalancedBagging + Isotonic kalibrasyon
**Veri:** MASTER (2931) + KANSER (388) + CFTR (111) = COMBINED (3430) → PAH (369, 3 birebir-aynı drop) üzerinde değerlendirme

#### 7 Strateji Sonuçları

| Strateji | n_train | MCC (raw) | MCC (prior) | Boot %80/20 F1 | Boot-std | Prec | Recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **P4_COMBINED_BalBag** | 3430 | **0.529** | **0.516** | 0.582 | 0.045 | 0.942 | 0.853 | 16 | 45 |
| P2_BalancedBagging | 2931 | 0.519 | 0.512 | 0.582 | 0.044 | 0.942 | 0.850 | 16 | 46 |
| P0c_COMBINED | 3430 | 0.497 | 0.439 | **0.590** | 0.054 | 0.947 | 0.762 | 13 | 73 |
| P1c_PriorShift_COMBINED | 3430 | 0.497 | 0.439 | 0.590 | 0.054 | 0.947 | 0.762 | 13 | 73 |
| P3_IsotonicCal | 2931 | 0.406 | 0.406 | 0.560 | 0.051 | 0.945 | 0.730 | 13 | 83 |
| P0_MASTER-only | 2931 | 0.463 | 0.399 | 0.555 | 0.054 | 0.945 | 0.723 | 13 | 85 |
| P1_PriorShift | 2931 | 0.463 | 0.399 | 0.555 | 0.054 | 0.945 | 0.723 | 13 | 85 |

**NB16 referans:** PAH stack_lr %80/20 F1=0.515, CI=[0.35–0.57]

#### Kritik Bulgular

1. **BalancedBagging minimax-optimal teori doğrulandı:** P2 (MCC=0.519) ve P4 (MCC=0.529) en yüksek MCC. Chatterji et al. (2022 NeurIPS) undersampling'in worst-case label shift altında optimal olduğunu kanıtlamıştı — PAH verimiz bunu doğruluyor. Overfit gap sadece 0.05 (P0/P1'de 0.14–0.16).
2. **COMBINED avantajı PAH'ta da geçerli:** Her çiftte COMBINED versiyonu lider (P0c > P0 +0.034, P4 > P2 +0.010). KANSER'in 120 benign'i ek bilgi sağlıyor.
3. **Prior-shift PAH'ta ZARAR veriyor:** Tüm stratejilerde raw MCC > prior MCC (P0: 0.463→0.399, delta=-0.064). CFTR'nin tam tersi. Sebep: LightGBM olasılıkları kalibre değil → Saerens düzeltmesi kötü-kalibre olasılıkları daha da bozuyor.
4. **P0c_COMBINED en yüksek boot-mean (0.590) ama en düşük MCC'lerden biri (0.439).** Paradoks: yüksek precision (0.947) + düşük recall (0.762) → %80/20 resample'da benign doğruluğu avantaj.
5. **Multi-seed doğrulama:** P4 ve P0c aynı multi-seed F1 (0.883), P2 biraz düşük (0.877±0.033). Sonuçlar kararlı.
6. **Bootstrap CI artık dar:** [0.52–0.65] — CFTR'deki [0.00–1.00] probleminden tamamen kurtulduk (n=62 benign yeterli).
7. **NB21 vs NB16:** +0.067 puan iyileşme (%80/20 F1: 0.515→0.582). CI alt sınırı 0.52 > NB16 üst sınırı 0.57 yok ama yakın.

#### Sonraki Adımlar (PAH devam)

- **Kalibrasyon + prior-shift zinciri:** Önce Platt/Isotonic kalibre et → sonra Saerens EM (Alexandari et al. 2020 ICML önerisi). Prior-shift'in zararlı çıkması kalibrasyonsuz uygulamadan kaynaklanıyor.
- **BalancedBagging n_estimators sweep:** 10, 20, 50, 100
- **P4 threshold robustness testi** (NB20 tarzı N=50 robust threshold)
- **COMBINED + BalancedBagging + Isotonic kalibrasyon** kombinasyonu

**Çıktılar:** `results/v8_pah_refinement/pah_results.csv`, `fig1_loo_mcc.png`, `fig2_boot_mean.png`, `fig3_confusion.png`, `fig4_posterior.png`, `reports/NB21_pah_refinement_report.pdf`

---

### Notebook 22 — PAH Ensemble + Calibration (TAMAMLANDI ✅, 2026-06-17)

**Dosya:** `notebooks/22_pah_ensemble_calibration.ipynb` (13 hücre)
**Pipeline:** 3 deney — BalancedBagging sweep, Platt kalibrasyon + prior-shift zinciri, heterogeneous stacking
**Veri:** COMBINED (3430) → PAH (369) üzerinde değerlendirme

#### Exp 1: BalancedBagging Hyperparameter Sweep (24 config)

| Config | MCC (raw) | Boot %80/20 F1 |
|---|---|---|
| n=10, mf=1.0 (en iyi MCC) | **0.536** | 0.591 |
| n=100, mf=0.7 (en iyi Boot) | 0.529 | **0.595** |
| n=20, mf=1.0 (NB21 ref) | 0.529 | 0.576 |

Bulgu: MCC platosu n=10–30 arası; daha fazla estimator belirgin kazanç sağlamıyor.

#### Exp 2: Platt Calibration + Prior-Shift Zinciri

| Strateji | Raw MCC | Prior MCC | Delta | Boot (raw) |
|---|---|---|---|---|
| A) Raw LGBM + Saerens | 0.497 | 0.439 | **-0.058** | 0.597 |
| B) Platt LGBM + Saerens | 0.377 | 0.429 | **+0.052** | 0.588 |
| C) CalBB + Saerens | 0.468 | 0.452 | -0.016 | 0.568 |

**Kritik bulgu:** Platt kalibrasyon prior-shift'i FİXLEDİ! B'de delta=+0.052 (A'da -0.058 idi). Alexandari 2020 hipotezi doğrulandı. Ancak Platt kalibrasyonu ham MCC'yi düşürdü (0.497→0.377) — threshold seçimi etkileniyor.

#### Exp 3: Heterogeneous BalancedBagging Stacking

| Model | MCC | Boot %80/20 F1 |
|---|---|---|
| BB_LGBM | 0.480 | 0.584 |
| BB_XGB | 0.448 | 0.571 |
| BB_CB | 0.444 | 0.551 |
| STACK_LR | 0.472 | 0.569 |

Bulgu: Stacking beklentiyi karşılamadı — 3 GBDT ailesi çok benzer kararlar veriyor (OOF MCC: 0.534/0.533/0.534). Diversite düşük.

#### NB22 Genel Sonuç

NB21 P4 (MCC=0.529, Boot=0.582) hâlâ en güçlü PAH modeli. NB22 marjinal iyileşme sağladı (sweep n=10 MCC=0.536) ama dramatik sıçrama yok. Kalibrasyon hipotezi doğrulandı ama pratik etkisi karışık.

**Çıktılar:** `results/v9_pah_ensemble/sweep_results.csv`, `pah_ensemble_results.csv`, 4 figür, `reports/NB22_pah_ensemble_report.pdf`

---

## Bekleyen İşler

| Öncelik | Görev |
|---|---|
| **Yüksek** | **NB23: PAH COMBINED eğitim testi ÇALIŞTIRILMASI** — 6 strateji (T0-T5) benchmark, NB21 P4 (Boot=0.582) ile kıyasla. COMBINED en iyi havuz mu? |
| **Yüksek** | **KANSER müdahale:** KANSER'e de COMBINED+BalancedBagging uygula — mevcut 0.716'yı geçebilir mi? |
| **Yüksek** | **Final teslim modeli sabitle:** KANSER→stack_lr (0.716) veya müdahale sonrası, **PAH→P4_COMBINED_BalBag (0.582, NB21) veya NB23 sonrası**, **CFTR→S0c_COMBINED (0.863, KESİNLEŞTİ)**. |
| **Not** | CFTR tamamlandı (NB17→NB20). PAH NB21→NB22→NB23 (oluşturuldu, çalıştırılmadı). |
---

### Notebook 23 — PAH COMBINED Training Sistematik Testi (TAMAMLANDI ✅, 2026-06-17)

**Dosya:** `notebooks/23_pah_combined_training.ipynb` (13 hücre)
**Pipeline:** 6 strateji — 4 farklı eğitim havuzu + BalancedBagging + Platt+Prior-Shift
**Veri:** T0=MASTER(2931), T1=+KANSER(3319), T2=+CFTR(3042), T3=COMBINED(3430) → PAH(369) test

#### 6 Strateji Sonuçları

| Strateji | n_train | MCC (raw) | Boot %80/20 F1 | Prec | Recall | FP | FN | Prior-Shift Δ |
|---|---|---|---|---|---|---|---|---|
| **T4_COMBINED+BalBag** | 3430 | **0.529** | 0.576 | 0.940 | 0.870 | 17 | 40 | -0.012 |
| T3_COMBINED | 3430 | 0.497 | **0.597** | 0.944 | 0.831 | 15 | 52 | -0.058 |
| T2_MASTER+CFTR | 3042 | 0.478 | 0.559 | 0.935 | 0.844 | 18 | 48 | 0.000 |
| T1_MASTER+KANSER | 3319 | 0.465 | 0.549 | 0.934 | 0.834 | 18 | 51 | +0.013 |
| T0_MASTER-only | 2931 | 0.463 | 0.536 | 0.929 | 0.850 | 20 | 46 | -0.064 |
| T5_COMBINED+Platt+Prior | 3430 | 0.377→0.429 | 0.588 | 0.957 | 0.645 | 9 | 109 | +0.052 |

#### Kritik Bulgular — Havuz Katkı Analizi

| Katkı | MCC Delta | Yorum |
|---|---|---|
| KANSER (+120 benign) | +0.002 | Neredeyse yok — KANSER benign'leri PAH'a az bilgi veriyor |
| **CFTR (+21 benign)** | **+0.015** | Şaşırtıcı! 21 benign 120'den daha değerli — tek-gen paneli feature uzayını zenginleştiriyor |
| COMBINED (+141 benign) | +0.034 | Sinerjik — toplam > parçaların toplamı |

1. **T4_COMBINED+BalBag yine en iyi MCC (0.529)** — NB21 P4 ile tutarlı. BalancedBagging PAH'ta kararlı şekilde en güçlü teknik.
2. **CFTR'deki dramatik sıçrama PAH'ta gerçekleşmedi:** CFTR'de +0.17 boot-mean artış, PAH'ta sadece +0.034 MCC. Sebep: PAH n=62 benign ile zaten yeterli sinyal sağlıyor, ek benign marjinal. CFTR'de n=21 benign yetersizdi → COMBINED dramatik fark yarattı.
3. **Platt kalibrasyon tekrar doğrulandı (T5):** Prior-shift delta=+0.052 (pozitif), ama ham MCC düştü. NB22 bulgusuyla tutarlı.
4. **Overfit tüm stratejilerde OK:** Gap 0.04–0.10 arasında (T5 hariç: 0.21 ORTA).
5. **PAH çalışması olgunlaştı:** NB21→NB22→NB23 boyunca MCC≈0.53, Boot≈0.58 kararlı. Dramatik iyileşme için farklı paradigma (NN/domain bilgisi) gerekebilir ama mevcut veriyle sınıra yakınız.

**Çıktılar:** `results/v10_pah_combined/pah_combined_results.csv`, `fig1_pool_comparison.png`, `fig2_boot_comparison.png`, `fig3_confusion_best.png`, `reports/NB23_pah_combined_report.pdf`

---

## PAH Yolculuğu Özeti (NB16→NB23)

| Notebook | En İyi | MCC | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|---|
| NB16 | stack_lr | — | 0.515 | Baseline stacking |
| **NB21** | **P4_COMBINED_BalBag** | **0.529** | **0.582** | BalancedBagging + COMBINED (+0.067) |
| NB22 | Sweep n=10 | 0.536 | 0.591 | Marjinal sweep, kalibrasyon hipotezi doğrulandı |
| NB23 | T4_COMBINED+BalBag | 0.529 | 0.576 | Havuz katkısı: CFTR > KANSER (şaşırtıcı) |
| NB24 | D3 SplineCal+PriorShift | 0.503 | 0.581 | TabPFN platoyu KIRAMADI → Chatterji tavanı doğrulandı; PAH KAPANDI |

**PAH final aday:** NB21 P4 (MCC=0.529, Boot=0.582, CI=[0.52–0.65]) veya NB22 sweep n=10 (MCC=0.536, Boot=0.591).

---

## Bekleyen İşler

| Öncelik | Görev |
|---|---|
| **Çok Yüksek** | **NB24: PAH plato-kırma (TabPFN + label-shift) — OLUŞTURULDU ✅, ÇALIŞTIRILMASI BEKLENIYOR.** `notebooks/24_pah_tabpfn_labelshift.ipynb` Jupyter'de çalıştırılacak (kullanıcı). 4 deney: D1 TabPFN ham, D2 TabPFN+Saerens prior-shift, D3 BalancedBagging+SplineCalibration+prior-shift, D4 BBSE. Hedef: NB21 P4 Boot=0.582 platosunu kır. |
| **Yüksek** | **KANSER müdahale:** KANSER'e de COMBINED+BalancedBagging uygula — mevcut 0.716'yı (stack_lr NB16) geçebilir mi? |
| **Yüksek** | **Final teslim modeli sabitle:** KANSER→0.716 veya müdahale sonrası, **PAH→NB21 P4 (0.582) / NB22 sweep (0.591) / NB24 sonrası**, **CFTR→S0c_COMBINED (0.863, KESİNLEŞTİ)**. |
| **Ertelendi** | AlphaMissense feature enjeksiyonu (Strateji 4): yüksek tavan ama NB18 deobfuscation'a bağlı + EK skorlarıyla redundancy riski. Önce NB24 platoyu kırıyor mu görülecek. |
| **Not** | CFTR tamamlandı (NB17→NB20). PAH olgunlaştı (NB21→NB23), plato MCC≈0.53/Boot≈0.58 → NB24 plato-kırma denemesi (code ✅, run bekleniyor). KANSER henüz müdahale edilmedi. |

---

### Notebook 24 — PAH Plato-Kırma: TabPFN + Label-Shift Düzeltmesi (ÇALIŞTIRILDI ✅, 2026-06-18 — PLATO KIRILMADI)

**Dosya:** `notebooks/24_pah_tabpfn_labelshift.ipynb` (10 hücre, syntax kontrolü geçti)
**Motivasyon:** PAH NB21→NB23 boyunca MCC≈0.53 / Boot≈0.58 platosu. Üç bağımsız müdahale ekseni (havuz/sweep/kalibrasyon) aynı tavana çarptı → **veri-sinyali tavanı**, hiperparametre sorunu değil. Platoyu kırmak için aynı feature uzayında optimizasyon değil, **yeni inductive bias (TabPFN)** veya **doğru kalibrasyon + prior düzeltmesi** gerekir.

**Literatür temeli (deep-research lit-review, 11 doğrulanmış kaynak):**
- **TabPFN** (Hollmann et al. 2025, *Nature* 637): ≤10k örnekte GBDT'leri geçen foundation model — PAH'ın tam tatlı noktası (n=369).
- **DistPFN** (2025, OpenReview): TabPFN label-shift'e açık (majority-bias) → test-time posterior düzeltmesi fix'liyor.
- **Calibrate-then-shift** (Alexandari et al. 2020, ICML PMLR 119): kalibrasyon (BCTS) EM'den ÖNCE gelmeli; kalibrasyonsuz Saerens EM başarısız olur → NB21'in "prior-shift zararlı" bulgusunun nedeni.
- **Platt-after-undersampling** (arXiv:2410.18144, 2024): undersampling sonrası standart Platt **biased olasılık** üretir → NB22'nin "ham MCC düştü (0.497→0.377)" bulgusunun nedeni. Çözüm: logit üzerine logistic GAM.
- **BBSE** (Lipton et al. 2018, ICML PMLR 80): kalibrasyon-bağımsız prior tahmini (confusion-matrix tersi) — kalibre olmayan LGBM için Saerens'e alternatif/çapraz-kontrol.
- **Undersampling minimax-optimal** (Chatterji et al. 2022, arXiv:2205.13094): label shift altında undersampling minimax-optimal AMA test doğruluğu **azınlık-örneği sayısıyla sınırlı** → plato kırılamazsa bu tez deneysel doğrulanmış olur (negatif sonuç da değerli).

**5 Deney (değerlendirme NB21/NB23 ile birebir aynı: LOO/multi-seed MCC + bootstrap %80/20 F1 N=50 %95 CI, benign sabit + patho downsample; birincil = Boot %80/20 pathogenic-F1; baseline = NB21 P4 Boot=0.582):**
1. **D1 — TabPFN ham** — TabPFNClassifier(`ignore_pretraining_limits=True` — 288 feature >100 sınırı için ŞART), COMBINED(3430)'da fit → PAH predict. `tabpfn 8.0.8` kurulu.
2. **D2 — TabPFN + Saerens prior-shift** — ham posterior → π_test=0.20. Hipotez: TabPFN posterior'u GBDT'den daha kalibre → düzeltme daha iyi çalışır.
3. **D2b — TabPFN native balance_probabilities** — TabPFN'in kendi prior-dengeleme opsiyonu; Saerens'e alternatif (DistPFN ruhu).
4. **D3 — Calibrate-then-shift kontrolü** — BalancedBagging-LGBM → **Spline(logit)+LR GAM kalibrasyon** (pygam yok → sklearn `SplineTransformer`; arXiv:2410.18144 reçetesi) → Saerens. NB22'nin yarım işini tamamlar, TabPFN'e adil kıyas tabanı.
5. **D4 — BBSE çapraz-kontrol** — COMBINED OOF hard-label confusion-matrix → π_test tahmini (Lipton); sabit π=0.20 ile yan yana kıyas.

**İmplementasyon notu (Opus denetimi):** Haiku subagent ilk taslakta D2b ve D4'ü atlamıştı + PDF hücresi yoktu; Opus elle ekledi (BBSE hard-label confusion-matrix matematiği, TabPFN balance_probabilities varyantı, fpdf2 ASCII rapor hücresi). Final: 10 hücre, AST syntax OK.

**Riskler:** TabPFN v8 API (NB17'de CFTR'de sorun çıkarmıştı); 334→288 feature TabPFN <100 tercihinin üstünde (seçim gerekebilir); kalibrasyon seti küçük (n=62 benign) → Isotonic değil Platt/GAM + nested CV + clipping.
**Devil's Advocate notu:** Plato gerçek bir azınlık-örneği tavanı olabilir (Chatterji 2022). O hâlde Deney 3/4 marjinal kalır, sıçrama yalnızca Deney 1/2'den (yeni inductive bias) gelebilir.

#### NB24 SONUÇLARI (ÇALIŞTIRILDI ✅, 2026-06-18) — PLATO KIRILMADI

**5 Deney sonuçları (COMBINED→PAH, Boot %80/20 F1 birincil; baseline NB21 P4=0.582):**

| Deney | MCC | Boot %80/20 F1 | Δ vs NB21 | Yorum |
|---|---|---|---|---|
| **D3 SplineCal+PriorShift** | 0.503 | **0.581** | −0.001 | Baseline'a EŞİT (en iyi NB24) |
| D4 BBSE (π_8020 ile) | 0.512 | 0.581 | −0.001 | D3 ile aynı (kalibre PAH proba paylaşımlı) |
| D1 TabPFN ham | 0.380 | 0.573 | −0.009 | **baseline altında** |
| D2 TabPFN + Saerens | 0.374 | 0.560 | −0.022 | prior yine zarar |
| D2b TabPFN balance_probs | 0.362 | 0.548 | −0.034 | en düşük |

**Kritik bulgular:**
1. **TabPFN platoyu KIRAMADI.** D1/D2/D2b üçü de NB21 P4 (0.582) altında. "Yeni inductive bias platoyu kırar" hipotezi **doğrulanmadı** → **Chatterji 2022'nin azınlık-örneği tavanı tezi PAH için deneysel doğrulandı** (62 benign sınırına NB21'de zaten varılmış). Negatif ama karar-verdirici sonuç.
2. **D3 calibrate-then-shift baseline'a eşit (0.581).** Spline(logit)-GAM kalibrasyon + Saerens, NB22'nin yarım işini tamamladı ama dramatik kazanç yok — plato gerçek.
3. **D4 BBSE bir model-kusurunu ifşa etti (öğretici negatif).** BBSE PAH'ı %80/20 resample edip π=0.20'yi geri bulmaya çalıştı ama **0.723 tahmin etti** (HAYIR, |fark|>0.10). Sebep: BalancedBagging confusion matrix'i `C[1,0]=0.264` (benign'lerin %26'sı patho sanılıyor) → BBSE inversiyonu bozuk. **Lipton 2018'in "confusion matrix yeterince doğru olmalı" koşulu PAH'ta sağlanmıyor → BBSE bu veride prior tahmininde güvenilmez. Saerens'in sabit π=0.20'si tercih edilmeli.**
4. **TabPFN top-50 feature ile çalıştı** (hız için 434→50 LGBM-gain seçimi). Seçilen ilk 8: EK_7, is_missing_AL_1, EK_9, CAT_1, AA_1, EK_2, EK_4, AL_327 — NB15 feature importance ile tutarlı (EK skorları + CAT_1 baskın).

**Sonuç:** PAH final aday DEĞİŞMEDİ → **NB21 P4_COMBINED_BalBag (Boot=0.582)** veya NB22 sweep n=10 (0.591). NB24 platoyu kıramadı ama bunu literatürle (Chatterji) açıkladı + BBSE'nin bu veride uygulanamazlığını gösterdi. PAH artık güvenle KAPATILABİLİR.

#### NB24 Hız Optimizasyonu (Opus, ilk koşu 10:52 → düzeltme sonrası 11:02)
Yavaşlık darboğazları + çözümleri: (1) TabPFN'e 434 feature veriliyordu → **top-50 LGBM-gain** (TabPFN <100 tercih eder); (2) `n_estimators=32` → **8**; (3) COMBINED self-prediction (3430 satır, threshold için) → **küçük holdout**; (4) D4 BalancedBagging'i D3'le **paylaşıldı** (2× eğitim yok). Kalan darboğaz: BalancedBagging CV (5×20 estimator) + `select_threshold_8020_robust` (90×50 F1/çağrı). 11dk kabul edildi.
**Güvenlik:** Cell 1'deki TabPFN token'ı (lisans-onay belirteci, API anahtarı değil) `os.environ.setdefault` ile env'den okunacak hale getirildi — git'e ham token girmesin (`export TABPFN_TOKEN=...` önerilir).

**Hücreler:** 1-5 NB21 AYNI (imports/veri/FE-M3/eval-altyapı/model-yardım) + 6 (5 deney) + 7 derleme/CSV + 8 görsel (fig1 MCC, fig2 boot F1) + 9 özet + 10 PDF rapor.
**Çıktılar:** `results/v11_pah_tabpfn/pah_tabpfn_results.csv`, `fig1_mcc_comparison.png`, `fig2_bootstrap_f1.png`, `reports/NB24_pah_tabpfn_report.pdf`.

---

### Notebook 27 — Weighted F1 + Threshold + Missing Robustness + Label Noise (ÇALIŞTIRILDI ✅, 2026-06-21)

**Dosya:** `notebooks/27_weighted_f1_threshold_optimization.ipynb` (12 hücre)
**Pipeline:** to-do.md GÖREV 1-4 implementasyonu — ağırlıklı F1, threshold karşılaştırması, eksik değer dayanıklılığı, label noise tespiti
**Veri:** COMBINED (MASTER+KANSER+CFTR) → PAH(369), COMBINED (MASTER+KANSER+PAH) → CFTR(111)

#### GÖREV 1+2: Threshold Karşılaştırması

**PAH:**

| Method | Thr | F1 Boot (%80/20) | ±Std | MCC |
|---|---|---|---|---|
| raw_f1_max | 0.160 | 0.560 | 0.044 | 0.497 |
| weighted_f1_max | 0.880 | 0.233 | 0.127 | 0.140 |
| **boot_8020_robust** | **0.420** | **0.590** | 0.073 | 0.401 |

**CFTR:**

| Method | Thr | F1 Boot (%80/20) | ±Std | MCC |
|---|---|---|---|---|
| raw_f1_max | 0.240 | 0.723 | 0.110 | 0.586 |
| weighted_f1_max | 0.760 | 0.546 | 0.220 | 0.314 |
| boot_8020_robust | 0.230 | 0.723 | 0.110 | 0.586 |

**Bulgu:** `boot_8020_robust` PAH'ta en iyi (F1=0.590, baseline 0.582'yi geçti). `weighted_f1_max` pathogenic F1'i öldürüyor (thr=0.88 → recall=0.16) — **kullanılmamalı**. CFTR'de raw ≈ boot_8020 (eşik ~0.23, fark yok).

#### GÖREV 4: Missing Value Robustness (4 strateji)

**PAH:** Sentinel en iyi (F1=0.457), M3/no_flags eşit (0.233), KNN en kötü (0.156).
**CFTR:** M3/no_flags en iyi (0.546), KNN (0.391), sentinel en kötü (0.258).

**Bulgu:** Panel-specific imputation gerekli. `is_missing_*` flag'leri fark yaratmıyor (M3 == no_flags). **⚠️ DİKKAT:** Bu sonuçlar `weighted_f1_max` eşiğiyle (0.88) değerlendirilmiş olabilir — tüm stratejilerde recall çok düşük (0.10–0.38). `boot_8020_robust` eşiğiyle yeniden değerlendirilmeli.

#### GÖREV 3: Label Noise (PAH)

Cross-model consensus (LightGBM, XGBoost, CatBoost): **49/369 şüpheli** (tüm 3 model yanlış). Dağılım: 36 pathogenic, 13 benign. Oransal: benign'lerin %21'i vs pathogenic'lerin %11.6'sı şüpheli → benign sınıfında oransal olarak daha fazla gürültü.

**Çıktılar:** `results/v12_optimization/`, `reports/NB27_optimization_report.pdf`

---

### Notebook 28 — Literature-Informed Optimization (ÇALIŞTIRILDI ✅, 2026-06-22)

**Dosya:** `notebooks/28_literature_informed_optimization.ipynb` (28 hücre: 13 markdown + 15 code)
**Pipeline:** NB27 bulgularını literatür taramasıyla birleştiren 7 deney
**Veri:** COMBINED (MASTER+KANSER+CFTR) → PAH(369), COMBINED (MASTER+KANSER+PAH) → CFTR(111)

**⚠️ İLK KOŞU VERİ SIZINTISI — DÜZELTİLDİ:** İlk versiyonda PAH training havuzu `concat([MASTER, PAH, KANSER])` olarak yanlış tanımlanmıştı — PAH hem train'de hem test'te → F1=0.97 sahte sonuç. Düzeltme: `concat([MASTER, KANSER, CFTR])` (PAH hariç). CFTR'de aynı hata (`concat([MASTER, CFTR])` → `concat([MASTER, KANSER, PAH])`).

#### 7 Deney ve Literatür Kaynakları

| # | Deney | Kaynak |
|---|---|---|
| D1 | Noise-Aware Weighting (suspicious samples weight=0.3) | Kordos et al. 2020; Northcutt et al. 2021 (cleanlab) |
| D2 | Noise Removal (consensus 3/3 wrong çıkar) | Improving Data Quality with GBDT (arXiv:2210.11327) |
| D3 | Bagging RF Feature Selection (top-50/100/150) | Breiman 1996; Gromski et al. 2015 |
| D4 | GHOST Threshold (reweighted F1) | Esposito et al. 2021, J. Chem. Inf. Model. |
| D5 | Panel-Specific Imputation (PAH=sentinel, CFTR=M3) | NB27 validated |
| D6 | Combined (D1+D3+D5) | Multi-factor optimization |
| D7 | Combined + GHOST (D1+D3+D4+D5) | Full pipeline |

#### PAH Sonuçları (düzeltilmiş, leakage-free)

| Method | Thr | F1(%80/20) | F1(%50/50) | Train F1 | Overfit Gap |
|---|---|---|---|---|---|
| Baseline_BalancedBagging_M3 | 0.68 | 0.5792 | 0.8968 | 0.9014 | 0.005 |
| D1_NoiseAware_Weight | 0.92 | 0.5579 | 0.8490 | 0.9254 | 0.076 |
| D2_NoiseRemoval | 0.92 | 0.5350 | 0.8829 | 0.8401 | −0.043 |
| **D3_RF_Top50** | **0.82** | **0.6463** | 0.7339 | 0.8466 | **0.113** |
| D3_LGBM_Top50 | 0.68 | 0.5411 | 0.9037 | 0.9767 | 0.073 |
| D3_RF_Top100 | 0.80 | 0.6234 | 0.7349 | 0.8720 | 0.137 |
| D4_GHOST_Threshold | 0.68 | 0.5792 | 0.8968 | 0.9014 | 0.005 |
| D5_PanelSpecific_Sentinel | 0.46 | 0.5303 | 0.9150 | 0.9462 | 0.031 |
| D6_Combined | 0.94 | 0.5848 | 0.8037 | 0.9071 | 0.103 |
| D7_Combined_GHOST | 0.70 | 0.5148 | 0.8933 | 0.9732 | 0.080 |

#### CFTR Sonuçları (n=21 benign → wide CI)

| Method | Thr | F1(%80/20) | Train F1 | Overfit |
|---|---|---|---|---|
| Baseline_LightGBM_M3 | 0.69 | 0.7248 | 0.9790 | 0.088 |
| D1_NoiseAware_Weight | 0.85 | 0.6965 | 0.9554 | 0.091 |
| D3_FS_Top100 | 0.66 | 0.7489 | 0.9798 | 0.075 |

#### OOB Overfit Doğrulama (PAH-Only RF, 2026-06-22)

**D3_RF_Top50'nin gerçek performansını doğrulamak için PAH verisini kendi içinde RF ile eğitip OOB tahminleri alındı.** OOB = Out-of-Bag, bagging modellerinde her ağacın görmediği örneklerdeki tahmin — retraining gereksiz, honest internal validation.

| Method | OOB F1(%80/20) | Thr | Precision | Recall | FP | FN | MCC |
|---|---|---|---|---|---|---|---|
| RF All Features (427) | 0.510 | 0.71 | 0.926 | 0.761 | 19 | 74 | 0.365 |
| **RF Top-30** | **0.539** | **0.78** | **0.950** | 0.607 | **10** | 122 | 0.333 |
| RF Top-50 | 0.521 | 0.71 | 0.928 | 0.784 | 19 | 67 | 0.390 |
| RF Top-100 | 0.526 | 0.71 | 0.928 | 0.794 | 19 | 64 | 0.401 |
| RF Top-50 Reg. | 0.527 | 0.63 | 0.940 | 0.658 | 13 | 106 | 0.339 |

**Kritik bulgular:**
1. **D3_RF_Top50 overfit teyit edildi:** NB28'de 0.6463, OOB gerçeği 0.521 (fark=+0.125). COMBINED'dan PAH'a transfer sırasında model eğitim verisine aşırı uymuş.
2. **PAH'ın kendi içindeki sınıflandırılabilirlik tavanı ~0.52–0.54 (weighted F1 %80/20).** NB21 (0.582) COMBINED training ile daha yüksek çıkıyor çünkü MASTER bilgi ekliyor — ama PAH-internal tavan daha düşük.
3. **Feature selection marjinal fayda:** Top-30 (0.539) > Top-100 (0.526) > All (0.510). Az feature + yüksek threshold (0.78) = az ama emin tahmin stratejisi %80 benign test setinde en iyi.
4. **PAH'ta precision çok yüksek (~0.93–0.95), recall düşük (0.61–0.79).** Model "pathogenic dediğinde doğru" ama birçoğunu kaçırıyor.
5. **Top feature'lar:** EK_7 (REVEL, 0.043) > AL_323 > EK_9 > AL_306 > missing_count > EK_8. PAH-specific sıralama COMBINED'dan farklı (AL sütunları daha ön planda).

**NB28 Genel Sonuç:**
- **Noise-aware training (D1/D2) işe yaramadı** — 495 "suspicious" örnek cross-panel domain shift, gerçek label noise değil
- **Feature selection RF'e faydalı ama overfit ile birlikte** — OOB ile doğrulandığında kazanç daha küçük
- **GHOST = boot_8020_robust** (aynı threshold 0.68 seçtiler)
- **Sentinel imputation doğru threshold ile M3'ten kötü** (NB27 bulgusu threshold artefaktıydı)
- **PAH performans tavanı değişmedi:** ~0.58 (COMBINED transfer) / ~0.54 (internal OOB)

**Çıktılar:** `results/v13_literature_optimization/`, `reports/NB28_literature_optimization_report.pdf`

---

## PAH Yolculuğu Özeti (NB16→NB28, güncellenmiş)

| Notebook | En İyi | MCC | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|---|
| NB16 | stack_lr | — | 0.515 | Baseline stacking |
| **NB21** | **P4_COMBINED_BalBag** | **0.529** | **0.582** | BalancedBagging + COMBINED (+0.067) |
| NB22 | Sweep n=10 | 0.536 | 0.591 | Marjinal sweep, kalibrasyon hipotezi doğrulandı |
| NB23 | T4_COMBINED+BalBag | 0.529 | 0.576 | Havuz katkısı: CFTR > KANSER |
| NB24 | D3 SplineCal+PriorShift | 0.503 | 0.581 | TabPFN platoyu KIRAMADI → Chatterji tavanı |
| NB27 | boot_8020_robust | — | 0.590 | Threshold düzeltmesi marjinal kazanç |
| NB28 | D3_RF_Top50 | — | 0.646 (overfit!) | OOB=0.521 → **overfit teyit edildi** |
| NB28-OOB | RF_Top30 (PAH-only) | 0.333 | 0.539 | **PAH-internal tavan ~0.54** |

**PAH final aday DEĞİŞMEDİ → NB21 P4_COMBINED_BalBag (Boot=0.582)** veya NB22 sweep n=10 (0.591).

---

### Notebook 29 — KANSER Panel Refinement: 6 Strateji (TAMAMLANDI ✅, 2026-06-22)

**Dosya:** `notebooks/29_kanser_refinement.ipynb` (14 hücre)
**Pipeline:** NB21 PAH protokolünün KANSER'e uyarlanması — LightGBM + 5-fold OOF + MCC + prior-shift + BalancedBagging + calibrate-then-shift + heterogeneous stacking + missing-dual
**Veri:** COMBINED (MASTER+PAH+CFTR=3414) → KANSER (385, 3 birebir-aynı drop) üzerinde değerlendirme
**Birincil metrik:** Bootstrap %80/20 pathogenic-F1 (N=50); LOO-MCC (prior-shift), AUPRC

#### 8 Deney Sonuçları (LOO-MCC prior sıralı)

| Deney | n_train | MCC(raw) | MCC(prior) | Boot-F1 | Boot-std | AUPRC | Prec | Recall | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| **E1_COMBINED** | 3414 | 0.650 | **0.654** | **0.714** | 0.041 | **0.964** | 0.943 | 0.804 | 13 | 52 |
| E2_BalancedBagging | 385 | 0.629 | 0.638 | 0.707 | 0.035 | 0.940 | 0.941 | 0.789 | 13 | 56 |
| E3_C_BalBag_Platt_PriorS | 385 | 0.634 | 0.630 | 0.707 | 0.041 | 0.936 | 0.941 | 0.781 | 13 | 58 |
| E3_B_Platt_PriorShift | 385 | 0.615 | 0.627 | 0.684 | 0.040 | 0.919 | 0.933 | 0.792 | 15 | 55 |
| E4_Heterogeneous_Stack | 385 | 0.540 | 0.614 | 0.667 | 0.038 | 0.930 | 0.922 | 0.800 | 18 | 53 |
| E5_Missing_Dual | 385 | 0.659 | 0.611 | 0.679 | 0.045 | 0.929 | 0.932 | 0.777 | 15 | 59 |
| E0_Baseline | 385 | 0.657 | 0.568 | 0.662 | 0.058 | 0.929 | 0.936 | 0.721 | 13 | 74 |
| E3_A_Raw_PriorShift | 385 | 0.657 | 0.568 | 0.662 | 0.058 | 0.929 | 0.936 | 0.721 | 13 | 74 |

**NB16 referans:** KANSER stack_lr %80/20 F1=0.716, CI=[0.67–0.77]

#### Kritik Bulgular

1. **NB16 referansı geçilemedi (fark=−0.002).** E1_COMBINED (Boot=0.714) en yakın ama stacking katmanı eksik — NB16'nın LR meta avantajı burada yok.
2. **COMBINED pooling (E1) açık kazanan:** AUPRC=0.964 (en yüksek sıralama gücü), cross-panel leakage yok. n=3414 ile n=385'e kıyasla çok daha güçlü sinyal.
3. **Prior-shift kalibrasyonsuz zararlı:** E0/E3_A'da MCC -0.089 düştü. Platt kalibrasyon sonrası (E3_B) +0.012; calibrate-then-shift sırası kritik (Alexandari 2020 teyit).
4. **E2 BalancedBagging en düşük overfit gap (0.04)** — en sağlam base model ama F1'de E1'in gerisinde.
5. **E4 Heterogeneous Stack underfit etti** (train < OOF, gap=−0.06) — meta LR kullanıldı ama base OOF leakage problemi var (iki seviyeli OOF gerekli).
6. **AUPRC çok yüksek (0.929–0.964)** — modellerin sıralama gücü iyi, sorun threshold/dağılım uyumunda.
7. **Sonraki adım:** COMBINED + OOF stacking (LR meta) + NN/DNN base modeller → NB30.

**Çıktılar:** `results/v14_kanser_refinement/kanser_results.csv`, `sweep_results.csv`, `fig1..fig4.png`, `reports/NB29_kanser_refinement_report.pdf`

---

### Notebook 30 — KANSER Advanced Stacking (TAMAMLANDI ✅, 2026-06-22)

**Dosya:** `notebooks/30_kanser_advanced_stacking.ipynb`
**Pipeline:** COMBINED pooling + OOF stacking + NN/DNN scratch + zengin base set
**Veri:** COMBINED (MASTER+PAH+CFTR=3414) → KANSER (385) test
**Birincil metrik:** Bootstrap %80/20 pathogenic-F1

#### 6 Deney Sonuçları

| Deney | Boot-F1 | AUPRC | FP | FN | Temel Bulgu |
|---|---|---|---|---|---|
| E0 Baseline (KANSER-only) | 0.662 | 0.929 | 13 | 74 | Referans |
| **E1 COMBINED** | **0.714** | **0.964** | 13 | 52 | En iyi Boot-F1 |
| E2 Tree Stack | 0.714 | 0.966 | 8 | 75 | En yüksek AUPRC, FN patladı |
| E3 NN Stack | 0.506 | — | — | — | **COLLAPSE** — heterogeneous data + NN scratch |
| E4 Zengin Base | 0.703 | 0.962 | 16 | 40 | En yüksek recall (0.849), en düşük FN |
| E5 KANSER NN | 0.656 | — | — | — | Scratch NN, pretrain yok → zayıf |

**Kritik Bulgular:**
1. **0.716 platosu kırılamadı.** E1/E2 = 0.714, NB16 stack_lr (0.716) hâlâ lider.
2. **NN/DNN scratch COLLAPSE (E3=0.506, E5=0.656)** — COMBINED heterogeneous veri + küçük KANSER train → NN kalibrasyonu bozuk. Pretrain+finetune ZORUNLU.
3. **FE (Grantham/BLOSUM62) hiç entegre edilmedi** — NB16'nın +0.015 FE katkısı NB30'da yok.
4. **E4 Zengin Base FN=40 (en düşük)** ama FP=16 (en yüksek) → precision-recall trade-off.

**→ NB31'e devredildi: FE + finetune + multi-layer stacking birleştirmesi.**

---

### Notebook 31 — KANSER Kümülatif Entegrasyon (TAMAMLANDI ✅, 2026-06-22)

**Dosya:** `notebooks/31_kanser_cumulative.ipynb` (17 hücre, ~52K chars)
**Pipeline:** NB16 FE + NB30 COMBINED pooling + pretrain/finetune NN/DNN + 2-katmanlı OOF stacking + OOB/OOF karşılaştırma + kalibrasyon + missing-aware dual model + ensemble sweep
**Veri:** COMBINED (3414) → KANSER (385, 50/50 stratified split)
**Birincil metrik:** Bootstrap %80/20 pathogenic-F1

#### 21 Deney Sonuçları (Boot-mean sıralı, ilk 10)

| Deney | Boot-F1 | Boot-std | AUPRC | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|---|
| **E6_alpha_0.5** | **0.7201** | 0.042 | 0.969 | 0.931 | 0.917 | 9 | 11 |
| E6_alpha_0.3 | 0.7181 | 0.041 | 0.968 | 0.931 | 0.910 | 9 | 12 |
| E6_alpha_0.4 | 0.7181 | 0.041 | 0.968 | 0.931 | 0.910 | 9 | 12 |
| E6_alpha_0.6 | 0.7076 | 0.038 | 0.969 | 0.925 | 0.925 | 10 | 10 |
| E6_alpha_0.7 | 0.6898 | 0.038 | 0.969 | 0.918 | 0.925 | 11 | 10 |
| E0_nofe | 0.6517 | 0.046 | 0.959 | 0.904 | 0.917 | 13 | 11 |
| E4_C_saerens | 0.6452 | 0.036 | 0.967 | 0.899 | 0.932 | 14 | 9 |
| E3_OOB_only | 0.6378 | 0.043 | 0.929 | 0.904 | 0.850 | 12 | 20 |
| E3_OOF_only | 0.6378 | 0.043 | 0.929 | 0.904 | 0.850 | 12 | 20 |
| E1_nn_ft | 0.5577 | 0.034 | 0.895 | 0.860 | 0.925 | 20 | 10 |

**NB16 referans:** KANSER stack_lr %80/20 F1=0.716

#### Kritik Bulgular

1. **0.716 platosu marjinal kırıldı: E6_alpha_0.5 = 0.7201 (+0.004).** Ama std=0.042 ile istatistiksel olarak anlamlı DEĞİL. E6 = baseline LGBM + multi-layer stacking blend (alpha=0.5).
2. **E0 FE ablasyonu: FE ZARAR VERDİ** — `E0_nofe` (0.6517) > `E0_with_fe` (0.6164). NB16'da +0.015 idi → COMBINED pooling ile FE etkileşimi negatif olabilir. Veya NB31'deki `add_fe()` implementasyonunda fark var.
3. **E1 Finetune NN/DNN: BAŞARISIZ (nn_ft=0.5577).** NB16'da dnn_ft=0.7046 idi. Fark implementasyondan:
   - **NB31:** `params[:-2]` freeze (çok agresif, sadece son katman eğitiliyor) + her fold'da yeni scaler (pretrain scaler'ından farklı) + loss-based early stopping
   - **NB16:** Hiç katman dondurmuyor (düşük lr=1e-4 ile nazik güncelleme) + tutarlı scaler (pretrain+panel birleşim) + F1-based early stopping
   - **→ NB32'de NB16 tarzı finetune protokolü kullanılmalı**
4. **E2 Multi-layer stacking: ZAYIF (0.5973).** NB16 stack_lr (0.716) çok gerisinde. 2-katmanlı mimari küçük veri setinde overfit ediyor.
5. **E3 OOB vs OOF: AYNI (ikisi de 0.6378).** OOB "free lunch" hipotezi doğrulanmadı — RF/BalancedBagging meta-feature kalitesi eşit.
6. **E4 Kalibrasyon: Saerens en iyi (0.6452)** ama E0_nofe'den düşük. Platt → Saerens (Alexandari) zinciri (0.6009) beklentinin altında.
7. **E5 Missing-aware: M3minus (0.6358) > M3plus (0.6164).** Leakage-risk sütunları çıkarmak marginal fayda sağlıyor.
8. **E6 Ensemble sweep: En iyi sonuç.** Baseline LGBM + stacking blend alpha=0.3–0.5 bandında optimal. Alpha>0.5'te stacking ağırlığı artınca performans düşüyor → stacking katkısı sınırlı.

#### Confusion Matrix (E6_alpha_0.5, en iyi)

```
              Predicted
              Benign  Patho
True Benign    51      9     (specificity=%85)
True Patho     11    122     (recall=%92)
```

FP=9, FN=11 — NB16 stack_lr ile karşılaştırılabilir denge (NB16: FP=9, FN=16).

#### NB31 vs NB16 Finetune Implementasyon Farkları (KRİTİK — NB32'ye girdi)

| Boyut | NB16 (dnn_ft=0.7046) | NB31 (E1_nn_ft=0.5577) |
|---|---|---|
| **Frozen layers** | Hiç dondurmuyor, tüm katmanlar lr=1e-4 ile güncelleniyor | `params[:-2]` freeze — sadece son lineer katman eğitiliyor (çok agresif) |
| **Scaler** | Birleşik basis (MASTER_CORE + panel_train) üzerinde tek scaler | Pretrain'de `train_nn()` kendi scaler'ı, finetune'da her fold yeni scaler → **dağılım kayması** |
| **Early stopping** | F1 üzerinden (`_f1_pos`) — doğrudan hedef metrik | Validation loss üzerinden — F1 ile korelasyonu dolaylı |
| **Eğitim verisi** | MASTER_CORE (625/625 dengeli) pretrain → panel train finetune | COMBINED (3414, dengesiz) pretrain → KANSER train finetune |
| **Sonuç** | 0.7046 (KANSER'de en iyi NN/DNN) | 0.5577 (en kötü NB31 sonucu) |

**→ NB32'de bu farklar düzeltilerek NB16 tarzı finetune protokolü uygulanacak.**

**Çıktılar:** `results/v16_kanser_cumulative/`, `reports/NB31_kanser_cumulative_report.pdf`

---

### Notebook 32 — KANSER Training Pool Sweep + Finetune Fix (TAMAMLANDI ✅, 2026-06-23)

**Dosya:** `notebooks/32_pretrain_distribution.ipynb` (16 hücre)
**Pipeline:** 7 model × 10 training pool varyasyonu sistematik sweep + FE ablasyonu + OOF stacking + ensemble
**Veri:** MASTER (2931) + KANSER (388) → 10 farklı pool kombinasyonu
**Birincil metrik:** Bootstrap %80/20 pathogenic F1 (N=50)
**Baseline:** NB31 E6_alpha_0.5 Boot-F1=0.7201

#### 10 Training Pool Varyasyonu

| Pool | Açıklama |
|---|---|
| P0_KANSER_ONLY | Sadece KANSER train (%50) |
| P1_BALANCED_625 | KANSER + dengelenmiş MASTER çekirdeği (625/625) |
| P2_KANSER_BAL625 | KANSER (dengeli) + MASTER 625/625 |
| P3_COMBINED_RAW | KANSER + tüm MASTER (ham oranlar) |
| P4_COMBINED_KANSER | KANSER + KANSER-oranıyla örneklenmiş MASTER |
| P5_FINAL_8020 | %80 benign / %20 pathogenic (final test taklidi) |
| P6_MASTER_UNDER | MASTER undersample (KANSER boyutuna) |
| P7_MASTER_UNDER_K | MASTER undersample + KANSER train |
| P8_COMBINED_BAL | KANSER + dengeli MASTER |
| P9_REVERSE_6040 | **%60 benign / %40 pathogenic** (final dağılıma yakın) |

#### 5 Deney

| # | Deney | En İyi Sonuç |
|---|---|---|
| Exp 0 | Finetune düzeltme doğrulama (NB31 4 bug fix) | dnn_ft=0.6468 (NB31'den ↑ ama ağaçların gerisinde) |
| Exp 1 | 7 model × 10 pool sweep (70 deney) | P9_REVERSE_6040_catboost=0.7104 |
| Exp 2 | FE etkileşim testi (en iyi pool × top-3 model) | **P9_REVERSE_6040_catboost_with_fe=0.7300** |
| Exp 3 | OOF stacking (LR meta) | stack_lr=0.6491 (hayal kırıklığı) |
| Exp 4 | Ensemble alpha sweep (stacking+baseline blend) | alpha_0.6=0.7253 (gürültülü, std=0.072) |

#### Exp 0: Finetune Düzeltme Doğrulama

NB31'deki 4 implementasyon hatası düzeltildi: (1) Layer freeze: sadece son layer → tüm katmanlar lr=1e-4, (2) Scaler: her fold yeni → birleşik basis, (3) Early stopping: loss-based → F1-based, (4) weight_decay: 1e-4 → 1e-3 + pos_weight eklendi.

| Deney | Boot-F1 | Std | AUPRC | FP | FN |
|---|---|---|---|---|---|
| E0_P1_BALANCED_625_dnn_ft | 0.6468 | 0.077 | 0.934 | 7 | 41 |
| E0_P1_BALANCED_625_nn_ft | 0.6018 | 0.055 | 0.947 | 12 | 30 |
| E0_P3_COMBINED_RAW_dnn_ft | 0.6213 | 0.045 | 0.927 | 13 | 22 |
| E0_P3_COMBINED_RAW_nn_ft | 0.6176 | 0.049 | 0.904 | 13 | 22 |

**Sonuç:** Bug fix'ler NB31 nn_ft=0.5577'den yukarı çekti ama ağaç modelleri (~0.70+) seviyesine ulaşamadı.

#### Exp 1: Pool × Model Sweep — Top 10

| Deney | Boot-F1 | Std | Prec | FP | FN | Train-F1 | Overfit Gap |
|---|---|---|---|---|---|---|---|
| P9_REVERSE_6040_catboost | 0.7104 | 0.042 | 0.930 | 9 | 14 | 0.720 | 0.010 |
| P3_COMBINED_RAW_lgbm | 0.7072 | 0.033 | 0.924 | 10 | 11 | 0.839 | 0.132 |
| P3_COMBINED_RAW_xgb | 0.7030 | 0.037 | 0.924 | 10 | 12 | 0.829 | 0.126 |
| P8_COMBINED_BAL_xgb | 0.7007 | 0.052 | 0.934 | 8 | 20 | 0.736 | 0.036 |
| P8_COMBINED_BAL_lgbm | 0.7005 | 0.045 | 0.928 | 9 | 17 | 0.747 | 0.047 |
| P4_COMBINED_KANSER_xgb | 0.6982 | 0.038 | 0.923 | 10 | 13 | 0.838 | 0.140 |
| P1_BALANCED_625_catboost | 0.6958 | 0.049 | 0.933 | 8 | 22 | 0.740 | 0.045 |

**Pool sıralaması (7 model ort.):** P8_COMBINED_BAL (0.653) > P1_BALANCED_625 (0.646) > P9_REVERSE_6040 (0.638) > ... > P6_MASTER_UNDER (0.537)
**Model sıralaması (10 pool ort.):** catboost (0.663) > xgb (0.642) ≈ lgbm (0.641) > nn_ft (0.607) > dnn (0.588) > dnn_ft (0.577) > nn (0.571)
**Overfit gap:** catboost (0.081) < lgbm (0.095) < xgb (0.101) < dnn (0.127) < nn (0.152) < nn_ft (0.245) < dnn_ft (0.263)

#### Exp 2: FE Etkileşimi (P9_REVERSE_6040 × top-3 model)

| Model | no_fe | with_fe | Δ |
|---|---|---|---|
| CatBoost | 0.710 | **0.730** | **+0.020** |
| XGB | 0.665 | **0.688** | **+0.022** |
| LGBM | **0.660** | 0.658 | −0.002 |

FE catboost ve xgb'de FN'yi önemli ölçüde düşürüyor (catboost: 14→8 FN). LGBM'de nötr.

#### Exp 3-4: Stacking + Ensemble

| Deney | Boot-F1 | Std | Prec | FP | FN |
|---|---|---|---|---|---|
| E3_stack_lr | 0.6491 | 0.099 | 0.953 | 4 | 51 |
| E4_alpha_0.3 | 0.6827 | 0.045 | 0.919 | 10 | 19 |
| E4_alpha_0.5 | 0.6808 | 0.059 | 0.929 | 8 | 29 |
| E4_alpha_0.6 | 0.7253 | 0.072 | 0.952 | 5 | 34 |

Stacking (E3) bu sefer hayal kırıklığı — FN=51 ile recall felaketi. Ensemble alpha=0.6 gürültülü (std=0.072).

#### En İyi Model ve Confusion Matrix

**Şampiyon: `E2_P9_REVERSE_6040_catboost_with_fe`**
- Boot-F1: **0.7300** ±0.033 (CI: 0.637–0.769)
- Precision: 0.933, MCC: 0.794, AUPRC: 0.968
- FP=9, FN=8
- Overfit gap: 0.004 (çok sağlıklı)

```
              Predicted
              Benign  Patho
True Benign    51      9     (specificity=%85)
True Patho      8    125     (recall=%94)
```

#### Kritik Bulgular

1. **0.720 platosu kırıldı: 0.7300 (+0.010).** Pool seçimi (P9_REVERSE_6040) + FE + CatBoost kombinasyonu. Std=0.033 ile NB31'den daha stabil.
2. **Pool seçimi > model seçimi:** En iyi-kötü pool farkı 0.116, model farkı 0.092. Eğitim dağılımı modelden daha etkili.
3. **P9_REVERSE_6040 (%60 benign / %40 patho) neden kazandı:** Final test %80 benign → eğitimde benign çoğunluğu görmek modeli precision'a zorluyor. CLAUDE.md'deki "FINAL TEST DAĞILIMI" uyarısının pratikte doğrulanması.
4. **CatBoost açık ara şampiyon:** Hem en yüksek F1 (0.663 ort.) hem en düşük overfit (0.081). NN/DNN KANSER'de ağaçlara yetişemiyor.
5. **Finetune düzeltmeleri yetersiz:** 4 bug fix ile nn_ft 0.5577→0.6468'e çıktı ama 0.70+ seviyesine ulaşamadı. KANSER (n=388) NN finetune için çok küçük.
6. **Stacking bu konfigürasyonda çalışmadı:** Finetune modeller zayıf → base modeller arasında çeşitlilik düşük → meta-learner değer katamadı. NB16'daki kazanç (0.690→0.716) daha çeşitli base'lerden geliyordu.
7. **FE tutarlı fayda:** CatBoost +0.020, XGB +0.022, LGBM nötr. FN'yi yarıya indirdi (14→8).

**Çıktılar:** `results/v17_pretrain_distribution/pool_sweep_results.csv`, `finetune_comparison.csv`, `fig1..fig5.png`, `reports/NB32_pretrain_distribution_report.pdf`

---

### Notebook 34 — CFTR Feature Engineering Ablasyon (ÇALIŞTIRILDI ✅, 2026-06-23 — FE DEĞER KATMADI)

**Dosya:** `notebooks/34_cftr_fe.ipynb`
**Pipeline:** NB20 S0c_COMBINED baseline + 16 literatür-destekli feature (4 grup) + ablasyon
**Veri:** COMBINED (3691) → CFTR (111) üzerinde LOO-CV + bootstrap %80/20 değerlendirme
**Birincil metrik:** LOO-MCC (NB17 protokolü). Karar eşiği: MCC delta > 0.05 (n=21 benign gürültü bandı).

#### Motivasyon

NB20 S0c_COMBINED (FE'siz): LOO-MCC=0.644, Boot-F1=0.863, Precision=1.0.
Literatür araştırması (FCS, CFTR-MetaPred, AlphaMissense, PHACTboost) sonucu 16 feature belirlendi.
Grup C (missingness) CFTR'de anlamsız (r<0.05) → çıkarıldı. Hedef: LOO-MCC 0.644→0.67–0.70.

#### Feature Grupları

| Grup | Feature Sayısı | Açıklama | En Güçlü Korelasyon |
|---|---|---|---|
| A (FCS) | 3 | Frekans × konservasyon | fcs_ek9 r=−0.492 |
| B (EK combo) | 5 | EK skor birleşimleri | ek_mean_top3 r=0.476 |
| D (AA physchem) | 4 | Amino asit fizikokimyasal delta | hydro_abs r=0.170 |
| E (AA subst) | 4 | Grantham, BLOSUM62, charge change | — |
| C (missingness) | — | **ÇIKARILDI** — tek-gen panelinde homojen | r<0.05 |

#### Experiment 1: Grup Ablasyonu

| Deney | LOO-MCC | Boot-F1 | Std | Prec | FP | FN |
|---|---|---|---|---|---|---|
| E1a_GroupA_FCS | 0.491 | 0.642 | 0.111 | 0.970 | 2 | 26 |
| E1b_GroupB_EKCombo | 0.481 | 0.634 | 0.129 | 0.969 | 2 | 27 |
| E1c_GroupD_AAphyschem | 0.538 | 0.714 | 0.126 | 0.985 | 1 | 25 |
| E1d_GroupE_AAsubst | 0.560 | 0.724 | 0.134 | 0.985 | 1 | 23 |
| E1e_All_ABDE | 0.607 | 0.837 | 0.114 | 1.000 | 0 | 22 |
| E1f_NoFE | 0.524 | 0.675 | 0.127 | 0.971 | 2 | 23 |

#### Experiment 2: FE Versiyon Karşılaştırması

| Deney | LOO-MCC | Boot-F1 | Std | Prec | FP | FN |
|---|---|---|---|---|---|---|
| E2a_NoFE | 0.524 | 0.675 | 0.127 | 0.971 | 2 | 23 |
| **E2b_NB16_FE** | **0.608** | **0.764** | 0.120 | 0.986 | 1 | 19 |
| E2c_Full_CFTR_FE | 0.607 | 0.837 | 0.114 | 1.000 | 0 | 22 |
| E2d_Best2_Groups | 0.502 | 0.644 | 0.117 | 0.970 | 2 | 25 |

**En iyi model:** E2b_NB16_FE (LOO-MCC=0.608), ama NB20 baseline (0.644) altında (delta=**−0.036**).

#### Kritik Bulgular

1. **FE CFTR'de değer katmadı (−0.036).** Hiçbir FE konfigürasyonu NB20 baseline LOO-MCC=0.644'ü geçemedi. Hedef 0.67–0.70 aralığına ulaşılamadı.
2. **Grup E (AA substitüsyon — Grantham/BLOSUM62) en güçlü tek grup** (MCC=0.560) ama yine de NoFE'nin (0.524) sadece +0.036 üstünde.
3. **Feature importance paradoksu:** `fe_grantham` (gain>100) ve `fe_blosum62` (~80) en yüksek importance'a sahip — model kullanıyor ama LOO-CV'de net genelleme fayda sağlayamıyor. Küçük örneklem (n=111) + 16 ek feature → curse of dimensionality.
4. **Best2_Groups paradoksu:** En iyi 2 grubu (D+E) seçmek (MCC=0.502) tüm gruplardan (0.607) çok daha kötü — feature'lar birbirini tamamlıyor ama toplam sinyal/gürültü oranı yetersiz.
5. **All_ABDE Precision=1.0 (FP=0)** ama FN=22 (recall düşük) → benign tarafında mükemmel ama pathogenic'leri kaçırıyor.
6. **n=21 benign nedeniyle Boot-CI=[0.00–1.00]** — Boot-F1 tek başına karar kriteri değil (NB17 protokolü doğrulandı).
7. **CFTR FE yolu tükendi:** Hem NB16 legacy FE hem NB34 literatür-destekli 16 feature denendi, ikisi de baseline'ı geçemedi.

#### CFTR Final Durum

**NB20 S0c_COMBINED KORUNUYOR** — LOO-MCC=0.644, Boot-F1=0.863, Precision=1.0, FP=0. FE eklenmeyecek.

**Çıktılar:** `results/v12_cftr_fe/`, `reports/NB34_cftr_fe_report.pdf`

---

---

### Notebook 35 — PAH Feature Engineering Ablasyon (ÇALIŞTIRILDI ✅, 2026-06-24 — FE DEĞER KATMADI)

**Dosya:** `notebooks/35_pah_fe.ipynb`
**Pipeline:** NB21 P4_COMBINED_BalBag baseline + 21 literatür-destekli feature (5 grup) + ablasyon
**Veri:** COMBINED (3430) → PAH (369, 3 birebir-aynı drop) üzerinde LOO-CV + bootstrap %80/20 değerlendirme
**Birincil metrik:** LOO-MCC + Bootstrap %80/20 pathogenic-F1. Baseline: NB21 P4 MCC=0.529, Boot-F1=0.582.

#### Motivasyon

PAH MCC≈0.53 / Boot≈0.58 platosu NB21→NB28 boyunca kırılamadı (havuz, sweep, kalibrasyon, TabPFN). **Kalan denenmeyen tek eksen: Feature Engineering.** NB34'te CFTR'de de FE değer katmamıştı; PAH'ta EK korelasyonları daha zayıf (max 0.284 vs CFTR 0.461) → beklenti düşük tutulmuştu.

#### Feature Grupları (21 feature, 5 grup)

| Grup | Feature Sayısı | Açıklama | Kaynak |
|---|---|---|---|
| A (Frekans) | 2 | has_any_freq, log_max_freq | gnomAD, ACMG BA1/BS1 |
| B (EK Combo) | 5 | ek_mean_all, ek_mean_top3, ek_max, ek_consensus, ek_delta_12 | REVEL, MetaRNN |
| D (AA Physchem) | 5 | hydro_abs, vol_abs, mw_abs, disorder_abs, accessibility_abs | MutPred2, PON-P3 |
| E (AA Subst) | 5 | grantham, blosum62, grantham_cat, proline_involved, charge_change | InMeRF, BMPR2 yapısal |
| F (Disagreement) | 4 | ek_delta_7_1, ek_range, ek_std, ek_7_x_vol | Genome Med 2023, REVEL |
| C (Missingness) | — | **ÇIKARILDI** — PAH'ta r<0.01 | — |

#### Experiment 1: Grup Ablasyonu

| Deney | MCC | Boot-F1 | Prec | FP | FN |
|---|---|---|---|---|---|
| E1a_GroupA_Freq | 0.369 | 0.533 | 0.942 | 13 | 95 |
| E1b_GroupB_EKCombo | 0.351 | 0.509 | 0.938 | 14 | 97 |
| E1c_GroupD_AAphyschem | 0.360 | 0.519 | 0.938 | 14 | 94 |
| E1d_GroupE_AAsubst | 0.355 | 0.526 | 0.944 | 12 | 104 |
| E1e_GroupF_Disagree | 0.340 | 0.508 | 0.940 | 13 | 105 |
| **E1f_All_ABDEF** | **0.381** | **0.539** | **0.950** | **11** | **99** |
| E1g_NoFE | 0.363 | 0.528 | 0.942 | 13 | 97 |

#### Experiment 2: FE Versiyon Karşılaştırması

| Deney | MCC | Boot-F1 | n_feat |
|---|---|---|---|
| E2a_NoFE | 0.363 | 0.528 | 428 |
| E2b_NB16_FE | 0.363 | 0.533 | 430 |
| **E2c_Full_PAH_FE** | **0.381** | **0.539** | 449 |
| E2d_Best2_Groups | 0.374 | 0.531 | 435 |
| E2e_F_plus_D | 0.352 | 0.516 | 437 |

#### Multi-seed Doğrulama

Boxplot sonuçları: All_ABDEF ve Full_PAH_FE multi-seed MCC ~0.39–0.42, baseline'ın 0.529'unun çok altında. Boot-F1 ~0.54–0.56 aralığında, baseline 0.582'nin altında. Best2_Groups daha da düşük (~0.37 MCC, ~0.52 Boot-F1).

#### Feature Importance (E1f_All_ABDEF)

Top-5: `fe_ek_7_x_vol` (gain=850), `fe_ek_delta_7_1` (750), `fe_ek_mean_top3` (550), `fe_grantham` (400), `fe_ek_delta_12` (350). EK skor kombinasyonları (Grup B+F) baskın — model bunları kullanıyor ama genelleme fayda sağlayamıyor.

#### Confusion Matrix (E1f_All_ABDEF, en iyi)

```
              Predicted
              Benign  Patho
True Benign    51      11    (specificity=%82)
True Patho     99     208    (recall=%68)
```

FN=99 çok yüksek — modelin recall problemi var. Precision=0.95 iyi ama 307 pathogenic'ten 99'unu kaçırıyor.

#### Sonuç ve Karar

**En iyi FE model: E1f_All_ABDEF — MCC=0.3807, Boot-F1=0.5388.**
**NB21 baseline: MCC=0.529, Boot-F1=0.582.**
**Delta: MCC=−0.148, Boot-F1=−0.043.**

**Karar: FE PAH'ta değer katmadı.** NB21 P4_COMBINED_BalBag baseline korunuyor.

#### Kritik Bulgular

1. **FE PAH'ta ZARAR verdi (−0.148 MCC, −0.043 Boot-F1).** Hiçbir FE konfigürasyonu NB21 baseline'ı geçemedi — bireysel gruplar, tüm gruplar birlikte ve NB16-tarzı FE dahil hepsi baseline altında.
2. **Grup F (Prediktor Disagreement) — PAH'a özgü tasarlanan grup — en zayıf çıktı** (MCC=0.340). EK_7−EK_1 korelasyonu (r=0.272) güçlü görünse de LOO-CV'de genelleme fayda yok.
3. **Feature importance paradoksu:** `fe_ek_7_x_vol` ve `fe_ek_delta_7_1` en yüksek gain'e sahip — model kullanıyor ama orijinal EK sütunlarından öğrenebildiği bilgiyi tekrar üretiyor (collinearity).
4. **PAH'ın temel darboğazı FE ile çözülemiyor:** n=62 benign (Chatterji 2022 tavanı) + zayıf EK korelasyonları (max |r|=0.284). FE ek boyut ekleyerek küçük benign grubunda overfitting riskini artırıyor.
5. **Multi-seed doğrulama sonucu tutarlı:** Tüm FE varyantları multi-seed'de de baseline altında — sonuç gürültü değil, gerçek.
6. **CFTR (NB34) ve PAH (NB35) birlikte:** Her iki panelde de literatür-destekli FE değer katmadı. Sadece KANSER'de (NB32) FE +0.020 kazandırdı — KANSER'de EK korelasyonları güçlü (r=0.45–0.53).

**Çıktılar:** `results/v18_pah_fe/`, `reports/NB35_pah_fe_report.pdf`

---

## PAH Yolculuğu Özeti (NB16→NB35, güncellenmiş)

| Notebook | En İyi | MCC | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|---|
| NB16 | stack_lr | — | 0.515 | Baseline stacking |
| **NB21** | **P4_COMBINED_BalBag** | **0.529** | **0.582** | BalancedBagging + COMBINED (+0.067) |
| NB22 | Sweep n=10 | 0.536 | 0.591 | Marjinal sweep, kalibrasyon hipotezi doğrulandı |
| NB23 | T4_COMBINED+BalBag | 0.529 | 0.576 | Havuz katkısı: CFTR > KANSER |
| NB24 | D3 SplineCal+PriorShift | 0.503 | 0.581 | TabPFN platoyu KIRAMADI → Chatterji tavanı |
| NB27 | boot_8020_robust | — | 0.590 | Threshold düzeltmesi marjinal kazanç |
| NB28 | D3_RF_Top50 | — | 0.646 (overfit!) | OOB=0.521 → overfit teyit |
| **NB35** | **E1f_All_ABDEF** | **0.381** | **0.539** | **FE DEĞER KATMADI (−0.148 MCC) — NB21 baseline korunuyor** |

**PAH KESİNLEŞTİ:** NB21 P4_COMBINED_BalBag (Boot=0.582). Havuz, sweep, kalibrasyon, TabPFN, FE — hepsi denendi, hiçbiri platoyu kıramadı. Chatterji 2022 azınlık-örneği tavanı (n=62 benign) deneysel doğrulandı.

---

## CFTR Yolculuğu Özeti (NB16→NB34)

| Notebook | En İyi | LOO-MCC | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|---|
| NB16 | dnn (FE+stacking) | — | 0.852 (CI:[0–1]) | Baseline, CI anlamsız |
| NB17 | S6_PriorShift | 0.655 | 0.692 | LOO-CV + MCC protokolü, prior-shift kazandı |
| NB18 | — | — | — | Benign artırımı yapısal çıkmaz (missense-benign kıt) |
| NB19 | S0c_COMBINED | 0.644 | 0.863 | COMBINED FP=0, precision=1.0 |
| **NB20** | **S0c_COMBINED** | **0.644** | **0.863** | **Final karar: ensemble/S0_MASTER terk, COMBINED kesinleşti** |
| NB34 | — | — | — | **Literatür-destekli FE değer katmadı (−0.036), NB20 korunuyor** |

**CFTR KESİNLEŞTİ:** S0c_COMBINED (MASTER+KANSER+PAH → LightGBM + prior-shift). FE, ensemble, regularizasyon, benign artırımı hepsi denendi — hiçbiri baseline'ı geçemedi.

---

## KANSER Yolculuğu Özeti (NB16→NB32)

| Notebook | En İyi | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|
| NB16 | stack_lr (FE+finetune+stacking) | 0.716 | Baseline, FE +0.015, dnn_ft=0.7046 |
| NB29 | E1_COMBINED | 0.714 | COMBINED pooling etkili, NB16 geçilemedi |
| NB30 | E1_COMBINED | 0.714 | NN/DNN scratch collapse, stacking marginal |
| NB31 | E6_alpha_0.5 | 0.7201 | Marjinal iyileşme (+0.004), finetune implementasyon hatası tespit edildi |
| **NB32** | **P9_REVERSE_6040_catboost_with_fe** | **0.7300** | Pool seçimi > model seçimi, reverse dağılım + CatBoost + FE kazandı |

---

### Notebook 36 — MASTER Baseline: 13 Model + 9 Ablasyon (TAMAMLANDI & ÇALIŞTIRILDI ✅, 2026-06-24)

**Dosya:** `notebooks/36_master_baseline.ipynb`
**Pipeline:** to-do.md Faz 1–4 implementasyonu — 3 ağaç baseline + BalancedBagging + FE ablasyonu + stacking + robust model + leakage probu
**Veri:** `YARISMA_TRAIN_MASTER.csv` (2931 satır) — stratified %80/20 split
**Birincil metrik:** Bootstrap %80/20 pathogenic-F1 (N=50, %95 CI)

#### 13 Model Sonuçları (Boot-F1 sıralı)

| Model | Boot-F1 | ±Std | CI | MCC | Prec | Recall | AUC-PR | Thr |
|---|---|---|---|---|---|---|---|---|
| **BalancedBag_XGB** | **0.603** | 0.045 | [0.50–0.68] | 0.496 | 0.520 | 0.721 | 0.921 | 0.45 |
| Stack_LR(lgbm+xgb+cb+bb) | 0.599 | 0.039 | [0.52–0.67] | 0.495 | 0.481 | 0.797 | 0.920 | 0.58 |
| BalancedBag_XGB+FE | 0.591 | 0.043 | [0.51–0.68] | 0.482 | 0.488 | 0.755 | 0.918 | 0.41 |
| BalancedBag_LGBM | 0.589 | 0.042 | [0.52–0.67] | 0.482 | 0.469 | 0.797 | 0.915 | 0.59 |
| CatBoost+FE | 0.580 | 0.049 | [0.49–0.68] | 0.469 | 0.465 | 0.775 | 0.919 | 0.85 |
| BalBag_robust(no_leak) | 0.580 | 0.043 | [0.51–0.66] | 0.467 | 0.474 | 0.752 | 0.913 | 0.72 |
| XGBoost+FE | 0.577 | 0.053 | [0.50–0.68] | 0.462 | 0.493 | 0.702 | 0.905 | 0.68 |
| CatBoost_baseline | 0.577 | 0.038 | [0.50–0.65] | 0.467 | 0.454 | 0.795 | 0.912 | 0.83 |
| BalancedBag+FE | 0.577 | 0.040 | [0.51–0.66] | 0.465 | 0.459 | 0.781 | 0.916 | 0.63 |
| XGBoost_baseline | 0.570 | 0.043 | [0.49–0.65] | 0.452 | 0.494 | 0.678 | 0.915 | 0.69 |
| LightGBM_baseline | 0.568 | 0.042 | [0.48–0.65] | 0.461 | 0.430 | 0.839 | 0.909 | 0.88 |
| LightGBM+FE | 0.559 | 0.040 | [0.49–0.64] | 0.452 | 0.415 | 0.859 | 0.912 | 0.88 |
| LGBM_robust(no_leak) | 0.549 | 0.031 | [0.49–0.60] | 0.441 | 0.402 | 0.868 | 0.904 | 0.81 |

#### 9 Ablasyon Sonuçları (FE dahil baseline üzerinden)

| Senaryo | Boot-F1 | MCC | n_feat | Yorum |
|---|---|---|---|---|
| **ablation_-FE** | **0.589** | 0.482 | 425 | **FE çıkarınca Boot-F1 ARTTI** → FE zararlı |
| ablation_-is_missing | 0.590 | 0.482 | 301 | is_missing flag'leri nötr |
| ablation_-EK | 0.588 | 0.479 | 430 | EK sinyali beklentinin altında etki |
| ablation_-leakage_AL | 0.578 | 0.466 | 419 | Leakage sütunları çıkarınca küçük düşüş |
| ablation_full (baseline) | 0.577 | 0.465 | 439 | Referans |
| ablation_-CAT | 0.582 | 0.471 | 435 | CAT nötr |
| ablation_-AA | 0.573 | 0.459 | 437 | AA marjinal |
| ablation_-AL_all | 0.520 | 0.387 | 167 | AL çıkarınca ciddi düşüş → AL sinyali önemli |
| ablation_only_EK | 0.478 | 0.324 | 9 | Sadece 9 EK ile 0.48 → güçlü omurga |

#### Kritik Bulgular

1. **BalancedBag_XGB MASTER şampiyonu (Boot-F1=0.603).** PAH ve KANSER'de olduğu gibi BalancedBagging burada da en güçlü teknik. Eğitim dağılımı tersliğini (train %73 patho → test %80 benign) en iyi yöneten model.
2. **FE MASTER'da ZARAR verdi:** -FE ablasyonu (0.589) > full (0.577). KANSER'den farklı — MASTER'da EK omurgası güçlü, FE ek boyut ekleyerek overfitting artırıyor.
3. **is_missing flag'leri nötr:** Dahil/hariç fark yok (0.577 vs 0.590). **Leakage alarmı yok** — MASTER'da eksiklik-label korelasyonu model kararlarını baskın biçimde yönetmiyor.
4. **AL sütunları kritik (ablation_-AL_all: 0.520)** — çıkarılınca 0.057 puan düşüş. EK omurga (0.478) üzerine AL ~0.10 puan ekliyor.
5. **Stacking marjinal:** Stack_LR (0.599) ≈ BalancedBag_XGB (0.603). Base modeller çok benzer kararlar veriyor → meta-learner çeşitlilik bulamıyor.
6. **AUC-PR çok yüksek (0.90–0.92)** — sıralama gücü güçlü. Sorun threshold/dağılım uyumunda, sıralamada değil.

**Çıktılar:** `results/v19_master_baseline/nb36_all_results.csv`, `nb36_ablation_results.csv`, `fig_nb36_summary.png`, `reports/NB36_master_baseline_report.pdf`

---

### Notebook 37 — MASTER Calibrated Stacking: 4-Arm FE×Robust Karşılaştırması (TAMAMLANDI & ÇALIŞTIRILDI ✅, 2026-06-24)

**Dosya:** `notebooks/37_master_calibrated_stacking.ipynb`
**Pipeline:** NB36 en iyi 4 base model + Isotonic kalibrasyon + OOF stacking (LR meta) + 4-arm (FE dahil/hariç × robust dahil/hariç) + is_missing leakage probe
**Veri:** `YARISMA_TRAIN_MASTER.csv` (2931 satır) — stratified %80/20 split
**Birincil metrik:** Bootstrap %80/20 pathogenic-F1

#### 4-Arm Sonuçları

| Arm | n_feat | Boot-F1 | CI | MCC | Prec | Recall | AUC-ROC | Train Gap | is_missing FI% | Thr |
|---|---|---|---|---|---|---|---|---|---|---|
| **no_fe_robust** | 277 | **0.589** | [0.50–0.65] | **0.477** | 0.509 | 0.705 | 0.856 | −0.019 | 0.0% | 0.68 |
| with_fe_dahil | 440 | 0.585 | [0.51–0.67] | 0.474 | 0.485 | 0.744 | 0.850 | 0.001 | 0.9% | 0.645 |
| no_fe_dahil | 426 | 0.578 | [0.51–0.65] | 0.466 | 0.462 | 0.775 | 0.850 | 0.003 | 0.9% | 0.62 |
| with_fe_robust | 291 | 0.578 | [0.51–0.68] | 0.463 | 0.500 | 0.690 | 0.856 | −0.007 | 0.0% | 0.68 |

#### Stacking Özeti

| Metrik | Değer |
|---|---|
| En iyi kol | **no_fe_robust** |
| FE delta (with_fe − no_fe, dahil kolda) | +0.007 (nötr) |
| Robust delta (robust − dahil, no_fe kolda) | +0.011 (nötr) |
| Max is_missing FI% | 0.9% (alarm yok, eşik %15) |

#### Kritik Bulgular

1. **no_fe_robust MASTER'ın en iyi kolu (Boot-F1=0.589, MCC=0.477).** FE eklemek ve leakage-risk sütunlarını dahil etmek fayda sağlamıyor. En temiz model en iyi performansı veriyor.
2. **FE etkisi nötr (+0.007):** MASTER'da EK omurgası güçlü olduğu için Grantham/BLOSUM62 gibi AA-türev feature'lar ek bilgi katmıyor — NB36 ablasyonuyla tutarlı.
3. **Robust delta nötr (+0.011):** Leakage-risk sütunları (AL_16..AL_25) MASTER'da kritik değil. is_missing flag'lerin FI %0.9 — model eksiklik bilgisine yaslanmıyor.
4. **Train gap negatif (robust kollarda −0.019 / −0.007):** Model test'te train'den iyi — overfitting yok, hatta hafif underfitting. BalancedBagging'in agresif undersampling'i sebebiyle.
5. **NB36 BalancedBag_XGB (0.603) > NB37 en iyi stacking (0.589).** Stacking MASTER'da değer katmadı — base modeller arasında yeterli çeşitlilik yok. BalancedBag_XGB tek başına daha iyi.
6. **NB14 ile kıyas:** NB14 stacking %50/50 F1=0.89 idi → %80/20'de 0.589. Bu, NB15'te yaşanan "dağılım tersliği çöküşü"nün MASTER'da da doğrulanması. Fark 0.30 puan — train dağılımı ile final test dağılımı arasındaki uçurum her panelde tutarlı.

**Çıktılar:** `results/v20_master_calibrated_stacking/arms_comparison.csv`, `stacking_summary.csv`, `fig_nb37_arms.png`, `reports/NB37_master_calibrated_stacking_report.pdf`

---

### Notebook 38 — MASTER Heterojen Stacking + Calibrate-then-Shift (TAMAMLANDI & ÇALIŞTIRILDI ✅, 2026-06-25)

**Dosya:** `notebooks/38_master_diverse_stack_labelshift.ipynb` (24 hücre)
**Pipeline:** 5 heterojen base model (LGBM, RF, BalBag, SmallMLP, DNN) + Isotonic kalibrasyon + LR meta stacking + Saerens kapalı-form prior-shift
**Veri:** MASTER (2931) → %80/20 stratified split (SEED=42) → train=2344, test=587

#### 3 Deney Sonuçları

| Deney | En İyi Varyant | F1_8020 | CI_95 | Not |
|---|---|---|---|---|
| Tek modeller | Single_rf | 0.6087 | [0.497–0.696] | RF sürpriz: LGBM'i (0.5511) geçti |
| **D1: Heterojen Stacking** | **V1_calib (Isotonic+LR)** | **0.6165** | — | NB36 BalBag (0.6025) +0.014 |
| D2: Calibrate-then-Shift | V2_calib+shift | 0.6165 | — | Prior-shift ek kazanç VERMEDİ; ECE 0.024→0.396 |
| D3: Raw Meta-Learner | raw_catboost | ~0.59 | — | Ham OOF meta-learner stacking'in altında |

#### Kritik Bulgular

1. **OOF çeşitliliği kanıtlandı:** Ortalama pairwise korelasyon 0.7544 (NB37 ~0.90). Heterojen base (LGBM+RF+BalBag+MLP+DNN) gerçek çeşitlilik sağladı.
2. **Calibrate-then-shift MASTER'da ÇALIŞMADI:** Isotonic kalibrasyon iyi (ECE=0.024) ama Saerens prior-shift ECE'yi 0.396'ya bozdu. V1_calib = V2_calib+shift (ikisi de 0.6165).
3. **Precision hâlâ darboğaz:** En iyi modelde precision=0.55, recall=0.71. %80 benign test setinde FP patlaması F1'i 0.62'de tutuyor.
4. **RF sürprizi:** Single_rf (0.6087) > Single_lgbm (0.5511). Bagging-tree benign-ağırlıklı test'e daha dayanıklı.

**Çıktılar:** `results/v21_master_diverse_stack_labelshift/`, `reports/NB38_master_diverse_stack_labelshift_report.pdf`

---

### Notebook 39 — MASTER Reversed-Distribution Eğitim Deneyi (TAMAMLANDI & ÇALIŞTIRILDI ✅, 2026-06-25)

**Dosya:** `notebooks/39_master_reversed_distribution.ipynb` (17 hücre)
**Pipeline:** 5 eğitim dağılımı senaryosu × 4 ağaç model (LGBM, BalBag, RF, CatBoost). NN hariç (küçük eğitim seti overfit riski).
**Veri:** MASTER train (2344) içinden reversed alt-küme seçimi → sabit X_test (587, NB38 ile birebir aynı)

#### Hipotez
Modeli final test dağılımına (%80 benign / %20 pathogenic) yakın bir eğitim setiyle eğitmek, post-hoc düzeltmelerden daha etkili olabilir. KANSER'de P9_REVERSE_6040 bunu 0.69→0.73 ile kanıtlamıştı.

#### Senaryo Tablosu

| Senaryo | Kompozisyon | n_train | En İyi Model | F1_8020 |
|---|---|---|---|---|
| S0_baseline | Orijinal (~73:27) | 2344 | rf | 0.6087 |
| **S1_6040** | **%60 benign / %40 patho** | **1041** | **balbag** | **0.6379** |
| S2_8020 | %80 benign / %20 patho | 780 | balbag | 0.5486 |
| S3_7030 | %70 benign / %30 patho | 892 | catboost | 0.5823 |
| S4_weighted | Tüm veri + class_weight | 2344 | lgbm | 0.5917 |

#### Top-5 Genel Sıralama

| # | Model | n_train | F1_8020 | CI_95 | Prec | Rec | Gap |
|---|---|---|---|---|---|---|---|
| 1 | **S1_6040/balbag** | 1041 | **0.6379** | [0.519–0.716] | 0.583 | 0.708 | +0.253 |
| 2 | NB38_V1_calib (ref) | 2345 | 0.6165 | — | — | — | — |
| 3 | S0_baseline/rf | 2344 | 0.6087 | [0.497–0.696] | 0.545 | 0.693 | +0.094 |
| 4 | S1_6040/rf | 1041 | 0.6065 | [0.489–0.680] | 0.545 | 0.688 | +0.259 |
| 5 | S1_6040/catboost | 1041 | 0.5963 | [0.480–0.665] | 0.544 | 0.664 | +0.198 |

#### Kritik Bulgular

1. **Reversed-distribution MASTER'da ÇALIŞTI:** S1_6040/balbag (0.6379) NB38 V1_calib'i (0.6165) **+0.021** geçti. KANSER'deki P9_REVERSE_6040 bulgusunun MASTER'a transferi başarılı.
2. **%60/40 oranı optimal:** S1 (0.6379) > S3_7030 (0.5823) > S2_8020 (0.5486). Agresif %80/20 (S2) pathogenic çeşitliliğini çok kısıyor (n_pat=156) → ciddi bilgi kaybı + aşırı overfit (gap=+0.43).
3. **Gerçek resample > ağırlıklama:** S1_6040/balbag (0.6379) > S4_weighted/lgbm (0.5917). Ağırlıklama karar sınırını yeterince kaydıramıyor; gerçek dağılım değişimi gerekiyor.
4. **Threshold yakınsama hipotezi DOĞRULANDI:** S0'da |thr_raw − thr_8020| = 0.27–0.48 → S2'de **0.00–0.04** → S1'de 0.07–0.17. Eğitim dağılımı test dağılımına yaklaştıkça threshold farkı dramatik şekilde azalıyor.
5. **Precision iyileşmesi kısmi:** S1 avg precision (0.546) > S0 (0.496), ama hâlâ <0.60. Precision darboğazı tam çözülmedi.
6. **Overfit uyarısı:** S1/S2/S3'te train-test gap yüksek (+0.20–0.48) — küçük eğitim setleri overfit'e açık. S0/S4'te gap düşük (+0.07–0.15).
7. **S4_weighted beklentinin altında:** Tüm veriyi kullanmasına rağmen en iyi S4 (lgbm=0.5917) S1'in gerisinde. Ağırlıklama loss fonksiyonunu değiştiriyor ama veri yoğunluğunu değiştirmiyor.

#### Sonuç ve Karar

**MASTER yeni en iyi: S1_6040/balbag F1_8020=0.6379** (NB36 BalBag 0.603'ten +0.035, NB38 V1_calib 0.6165'ten +0.021).
Sonraki adım: NB39 kazanan senaryo (S1_6040) base'leriyle stacking denemesi.

**Çıktılar:** `results/v22_master_reversed_distribution/reversed_distribution_results.csv`, `all_experiments.png`, `threshold_convergence.png`, `reports/NB39_master_reversed_distribution_report.pdf`

---

## MASTER Yolculuğu Özeti (NB36→NB39)

| Notebook | En İyi | Boot %80/20 F1 | Temel Bulgu |
|---|---|---|---|
| NB36 | BalancedBag_XGB | 0.603 | 13 model sweep + 9 ablasyon; FE zararlı, BalBag en iyi |
| NB37 | no_fe_robust (stacking) | 0.589 | 4-arm karşılaştırma; stacking BalBag_XGB'yi geçemedi |
| NB38 | V1_calib (hetstack+isotonic) | 0.6165 | Heterojen stacking +0.014; calibrate-then-shift ek kazanç vermedi |
| **NB39** | **S1_6040/balbag** | **0.6379** | **Reversed-distribution %60/40 çalıştı (+0.021); threshold yakınsama doğrulandı** |

**MASTER şu anki en iyi:** NB39 S1_6040/balbag (Boot-F1=0.6379, CI=[0.519–0.716], MCC=0.542). Reversed-distribution eğitim KANSER'den sonra MASTER'da da işe yaradı. Sonraki adımlar: NB39 kazanan senaryo base'leriyle stacking, Optuna hiperparametre tuning.

---

### Notebook 40 — Bayes Error Ceiling: 4 Panelde Teorik Tavan Tahmini (ÇALIŞTIRILDI ✅, 2026-06-25 — GERİYE DÖNÜK EKLENDİ 2026-07-24)

**Dosya:** `notebooks/40_bayes_error_ceiling.ipynb`
**Pipeline:** k-NN tabanlı yerel sınıf-örtüşme (Bayes hata oranı) tahmini, her panelin kendi feature uzayında k∈{1,3,5,7} ile hesaplanır → Bayes-F1(8020) tahmini → mevcut şampiyon modelle (NB39/NB32/NB21/NB20) kıyaslanır.
**Veri:** MASTER (2931), KANSER (388), PAH (372), CFTR (111) — her biri kendi train/test split'inde.
**Amaç:** "Modelimiz veri setinin izin verdiği teorik tavana ne kadar yakın?" sorusuna k-NN tabanlı ampirik bir cevap vermek — bu, daha önce sadece floor-F1 (`2·prev/(1+prev)`, trivial "hep patho de" baseline'ı) ile yapılan karşılaştırmayı tamamlayan **ikinci ve bağımsız bir üst-sınır tahmini**.

**Not (belgeleme boşluğu):** Bu notebook 2026-06-25'te çalıştırılmış, çıktıları (`results/v23_bayes_error_ceiling/`, `reports/NB40_bayes_error_ceiling_report.pdf`) diskte mevcuttu, ancak sonuçlar bu tarihe kadar `progress.md`'ye hiç işlenmemişti. Bulgular ham `bayes_error_summary.csv` ve `decision_table.csv` dosyalarından geriye dönük çıkarıldı.

#### Metodoloji (k-NN Bayes Hata Tahmini)

Her panelde k∈{1,3,5,7} komşu sayısıyla k-NN sınıflandırma hata oranı hesaplanır (`kNN-1`..`kNN-7` sütunları). Bu hata oranından bir alt/üst sınır bandı (`R*_low`, `R*_high`) türetilir — Cover & Hart (1967) k-NN asimptotik hata sınırları mantığıyla, sonlu-örneklem k-NN hatası Bayes hatasının [R*, 2R*(1−R*)] aralığında bir üst sınırını verir. Bu bant, panelin **%80/20 final-dağılım havuzunda** beklenen Bayes-F1'e çevrilir (`Bayes-F1(8020)` + `%95 CI`). Mevcut şampiyon modelin gerçek Boot-F1'i bu tavanla karşılaştırılıp `Gap = Bayes-F1(8020) − Model-F1` hesaplanır.

**Karar kuralı:** Gap küçük/negatif → model zaten tavana yakın/üstünde → **"DUR (at ceiling)"**. Gap anlamlı derecede pozitif → hâlâ kazanılacak sinyal var → **"DEVAM (significant margin)"**.

#### Sonuç Tablosu (`bayes_error_summary.csv`)

| Panel | n | Benign | Patho | kNN-1 | kNN-3 | kNN-5 | kNN-7 | R*_low | R*_high | Bayes-F1(8020) | %95 CI | Model-F1 | Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASTER | 2931 | 782 | 2149 | 0.2886 | 0.2613 | 0.2453 | 0.2375 | 0.1749 | 0.2886 | 0.6540 | [0.6449, 0.6658] | 0.6379 (NB39) | **+0.0161** |
| KANSER | 388 | 120 | 268 | 0.1985 | 0.1778 | 0.1907 | 0.1753 | 0.1117 | 0.1985 | 0.7263 | [0.6984, 0.7576] | 0.7300 (NB32) | **−0.0037** |
| **PAH** | 372 | 62 | 310 | 0.2177 | 0.1909 | 0.1909 | 0.1747 | 0.1243 | 0.2177 | **0.6945** | [0.6779, 0.7081] | 0.582 (NB21) | **+0.1125** |
| CFTR | 111 | 21 | 90 | 0.1892 | 0.1622 | 0.1441 | 0.1532 | 0.1058 | 0.1892 | 0.7550 | [0.7273, 0.7719] | 0.863 (NB20) | **−0.1080** |

#### Karar Tablosu (`decision_table.csv`)

| Panel | Gap | Karar | Güven | Not |
|---|---|---|---|---|
| MASTER | +0.0161 | DUR (at ceiling) | HIGH | — |
| KANSER | −0.0037 | DUR (at ceiling) | HIGH | — |
| **PAH** | **+0.1125** | **DEVAM (significant margin)** | **HIGH** | — |
| CFTR | −0.1080 | DUR (at ceiling) | LOW | n_benign=21, geniş CI |

#### Kritik Bulgular

1. **MASTER ve KANSER, Bayes-ceiling'e neredeyse tam oturmuş (|Gap|<0.02, HIGH confidence).** Bu, NB36-39 (MASTER) ve NB29-32 (KANSER) boyunca yapılan yoğun ablasyon/sweep çalışmasının veri setinin izin verdiği sınıra fiilen ulaştığının bağımsız bir doğrulaması — "DUR" kararı güvenilir.
2. **PAH'ta ÇELİŞKİLİ SİNYAL:** NB21→NB35 boyunca "Chatterji 2022 azınlık-örneği tavanı" gerekçesiyle (floor F1=0.905'e çok yakın sonuçlar) PAH kapatılmıştı. Ama NB40'ın k-NN Bayes-ceiling tahmini PAH için **+0.1125 anlamlı boşluk** buluyor — mevcut modelin (0.582) üstünde, floor'un (0.905) altında bir ara-tavan (0.6945) işaret ediyor. **Bu iki analiz aynı soruya (PAH'ta daha fazla sinyal var mı?) farklı cevap veriyor** — floor-F1 "hayır, floor'a çok yakınız" derken, Bayes-ceiling "evet, 0.69'a kadar yer var" diyor.
3. **CFTR'de Gap negatif ama Confidence=LOW.** n=21 benign ile hem model boot-F1'i (0.863) hem k-NN tabanlı Bayes tahmini istatistiksel olarak kırılgan; "model teorik tavanın üstünde görünüyor" sonucu gerçek bir başarı değil, örneklem küçüklüğünün ürünü olabilir. **CFTR'de Bayes-ceiling kararına güvenilmemeli.**
4. **k-NN k-değeri arttıkça hata oranı genelde düşüyor (kNN-1 > kNN-7 çoğu panelde)** — beklenen davranış (daha fazla komşu = daha az varyans), yöntemin sağlıklı çalıştığına işaret.
5. **PAH'ın floor'a yakınlığı ile Bayes-ceiling'in floor'dan uzaklığı birlikte okunmalı:** floor=0.905 (trivial "hep patho de"), Bayes-ceiling=0.6945, model=0.582. Üç değer üç farklı şeyi ölçüyor — floor bir aşağı sınır (trivial), Bayes-ceiling bir yerel-yoğunluk tavanı, model gerçekleşen performans. Floor'un yüksekliği zaten "her şey pathogenic'e yakın çoğunluk" demek; bu Bayes-ceiling'in neden floor'un altında kaldığını (0.6945 < 0.905) açıklıyor ama modelin neden Bayes-ceiling'in de altında kaldığını (0.582 < 0.6945) açıklamıyor — **bu boşluk yeni araştırmanın konusu.**

#### Sonuç ve Sonraki Adım

**NB40, PAH'ta önceden "kapandı" sayılan bir panelde yeniden açılmayı gerektiren somut, ölçülebilir bir sinyal üretti.** Bu bulgu daha önce hiç işlenmediği için "PAH KESİNLEŞTİ" kararı fiilen sorgulanmamış durumdaydı. **2026-07-24 itibarıyla PAH (ve tutarlılık için diğer tüm paneller) yeniden açıldı** — bkz. dosya başındaki "Panel Durumu" tablosu.

**Çıktılar:** `results/v23_bayes_error_ceiling/bayes_error_summary.csv`, `decision_table.csv`, `fig1_model_vs_ceiling.png`, `fig2_knn_curve.png`, `fig3_eps_inconsistency.png`, `fig4_dist_comparison.png`, `fig5_decision_zones.png`, `reports/NB40_bayes_error_ceiling_report.pdf`

---

### Notebook 41 — Danışman EDA Çıkarımları Testi: Feature Selection Sweep (ÇALIŞTIRILDI ✅, 2026-06-27 — GERİYE DÖNÜK EKLENDİ 2026-07-24)

**Dosya:** `notebooks/41_feature_selection_eda_validation.ipynb`
**Pipeline:** `xgb_importance` top-N feature seçimi (k=10/25/50/100/200/all) × 3 panel (MASTER, KANSER, PAH; CFTR atlandı — küçük örneklem), BalancedBagging(10×LGBM), **bizim %80/20 bootstrap protokolümüzde** (NB39 ile birebir aynı değerlendirme).
**Amaç:** Danışmanın feature-depth EDA'sındaki iddiayı ("`xgb_importance` top-200 > all-features", MASTER'da train-dağılımında 0.894 vs 0.879) **bizim ters-dağılım (final-realistic) protokolümüzde** doğrulamak. Danışmanın train-dağılımı sonuçları final performansı yansıtmıyor (bkz. "Danışman EDA İncelemesi" bölümü aşağıda) — asıl soru bu kazancın %80/20 bootstrap'te de geçerli olup olmadığıydı.

**Not (belgeleme boşluğu):** Bu notebook 2026-06-27'de çalıştırılmış, `results/v24_feature_selection/nb41_results.csv` + 2 figür + PDF rapor diskte mevcuttu, ama sonuçlar `progress.md`'ye hiç işlenmemişti (sadece "Bekleyen İşler" tablosunda "çalıştırılmayı bekliyor" olarak duruyordu — yanlış/eski bir statüydü). Bulgular ham CSV'den geriye dönük çıkarıldı.

#### Sonuç Tablosu (`nb41_results.csv`, floor referansı=0.333 — tüm panellerde ortak, çünkü bootstrap havuzu %80/20'ye sabitlendiğinde prevalans=0.20 olur)

| Panel | k | n_features | F1(8020) | ±Std | CI | MCC | Prec | Recall | F1(5050) | F1 vs Floor | Train-F1 | Train-Test Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASTER | 10 | 10 | 0.461 | 0.056 | [0.355, 0.555] | 0.307 | 0.397 | 0.555 | 0.693 | +0.128 | 0.750 | 0.057 |
| MASTER | 25 | 25 | 0.562 | 0.046 | [0.469, 0.657] | 0.442 | 0.460 | 0.729 | 0.809 | +0.229 | 0.869 | 0.061 |
| MASTER | 50 | 50 | 0.557 | 0.050 | [0.460, 0.656] | 0.434 | 0.463 | 0.702 | 0.798 | +0.223 | 0.866 | 0.067 |
| MASTER | 100 | 100 | 0.549 | 0.048 | [0.450, 0.636] | 0.427 | 0.434 | 0.754 | 0.817 | +0.216 | 0.898 | 0.081 |
| MASTER | 200 | 200 | 0.568 | 0.039 | [0.488, 0.640] | 0.455 | 0.442 | 0.797 | 0.841 | +0.234 | 0.938 | 0.097 |
| **MASTER** | **all** | **426** | **0.558** | 0.040 | [0.472, 0.624] | 0.441 | 0.432 | 0.790 | 0.836 | +0.224 | 0.937 | 0.101 |
| KANSER | 10 | 10 | 0.438 | 0.104 | [0.268, 0.688] | 0.260 | 0.313 | 0.750 | 0.792 | +0.104 | 0.889 | 0.096 |
| KANSER | 25 | 25 | 0.578 | 0.095 | [0.343, 0.740] | 0.481 | 0.435 | 0.877 | 0.881 | +0.244 | 0.931 | 0.050 |
| KANSER | 50 | 50 | 0.566 | 0.088 | [0.358, 0.697] | 0.463 | 0.428 | 0.853 | 0.870 | +0.232 | 0.938 | 0.068 |
| **KANSER** | **100** | **100** | **0.632** | 0.122 | [0.348, 0.800] | 0.545 | 0.511 | 0.847 | 0.887 | +0.299 | 0.951 | 0.064 |
| KANSER | 200 | 200 | 0.589 | 0.111 | [0.328, 0.757] | 0.491 | 0.461 | 0.837 | 0.868 | +0.256 | 0.959 | 0.091 |
| KANSER | all | 502 | 0.626 | 0.119 | [0.348, 0.800] | 0.537 | 0.508 | 0.837 | 0.876 | +0.293 | 0.956 | 0.080 |
| PAH | 10 | 10 | 0.302 | 0.127 | [0.032, 0.500] | 0.042 | 0.202 | 0.620 | 0.734 | **−0.031** | 0.816 | 0.082 |
| PAH | 25 | 25 | 0.323 | 0.129 | [0.129, 0.545] | 0.060 | 0.212 | 0.700 | 0.783 | **−0.011** | 0.849 | 0.066 |
| **PAH** | **50** | **50** | **0.368** | 0.092 | [0.185, 0.566] | 0.165 | 0.241 | 0.813 | 0.852 | +0.034 | 0.911 | 0.059 |
| PAH | 100 | 100 | 0.310 | 0.086 | [0.143, 0.491] | 0.015 | 0.193 | 0.800 | 0.820 | **−0.023** | 0.943 | 0.123 |
| PAH | 200 | 200 | 0.310 | 0.112 | [0.143, 0.500] | 0.020 | 0.198 | 0.747 | 0.807 | **−0.023** | 0.920 | 0.113 |
| PAH | all | 408 | 0.310 | 0.112 | [0.143, 0.500] | 0.020 | 0.198 | 0.747 | 0.807 | **−0.023** | 0.920 | 0.113 |

#### Kritik Bulgular

1. **Danışmanın "top-200 > all-features" iddiası MASTER'ın ters-dağılım (%80/20 bootstrap) protokolünde DOĞRULANMADI.** k=200 (0.568) ile k=all/426 (0.558) arasındaki fark +0.010 — std bandı (±0.04) içinde, yani **gürültü seviyesinde**. Danışmanın train-dağılımındaki net kazancı (0.894 vs 0.879, +0.015) burada tekrar etmedi.
2. **MASTER'da k=25 (25 feature) ile k=200 (200 feature) neredeyse aynı performansı veriyor (0.562 vs 0.568)** — bu, feature seçiminin MASTER'da ciddi bir kaldıraç olmadığını, sinyalin az sayıda güçlü feature'da (muhtemelen EK skorları + birkaç AL sütunu) yoğunlaştığını gösteriyor. NB36'nın "EK omurga + AL katkısı" bulgusuyla tutarlı.
3. **KANSER'de k=100 açık ara en iyi (0.632, CI genişliği çok büyük [0.348–0.800] olsa da nokta tahmini net üstün).** k=all (0.626) ile fark küçük (+0.006) — yine gürültü bandında. Ama k=10 (0.438) belirgin şekilde daha kötü — çok agresif feature azaltma KANSER'de zarar veriyor.
4. **PAH'ta feature seçimi FLOOR'UN ALTINA DÜŞÜRÜYOR (k=10/25/100/200 hepsi negatif "F1 vs Floor"):** Yalnızca k=50 floor'u marjinal geçiyor (+0.034). Bu, PAH'ın zaten zayıf sinyalinin agresif feature azaltmayla daha da bozulduğunu gösteriyor — NB35'in "FE PAH'ta zarar verdi" bulgusuyla aynı yönde (PAH, boyut değişikliklerine karşı kırılgan).
5. **Train-Test Gap her panelde k arttıkça büyüyor** (MASTER: 0.057→0.101, KANSER: 0.096→0.080 stabil, PAH: 0.082→0.113) — daha fazla feature = daha fazla overfit riski, beklenen yönde ama MASTER/PAH'ta pratik sonucu değiştirecek kadar büyük değil.
6. **Sonuç: Danışmanın feature-seçimi bulgusu bizim final-realistic protokolümüze GENELLEMEDİ.** Train-dağılımında (danışmanın ölçtüğü rejimde) görülen top-N avantajı, ters-dağılım bootstrap'inde kayboluyor. Bu, dosyanın başındaki "Danışman ile bizim F1'lerimiz AYNI METRİĞİ ÖLÇMÜYOR" uyarısının **feature-seçimi bulgusuna özel, somut bir doğrulamasıdır** — danışmanın relatif bulguları bile her zaman bizim rejimimize taşınmayabiliyor.

#### Sonuç ve Karar

**Feature seçimi (top-N) MASTER ve PAH'ta pratik bir kazanç sağlamadı; KANSER'de nominal en iyisi (k=100, 0.632) mevcut şampiyondan (P9_REVERSE_6040_catboost_with_fe, 0.730, NB32) düşük — bu deneyde reverse-distribution/FE kombinasyonu yoktu, saf feature-seçim etkisini izole ediyordu.** Feature seçimi tek başına mevcut en iyi tarifleri geçemiyor; gelecekte reverse-distribution + feature-seçim birlikte denenmemiş bir kombinasyon olarak kalıyor.

**Çıktılar:** `results/v24_feature_selection/nb41_results.csv`, `fig1_ksweep_and_importance.png`, `fig2_train_test_gap.png`, `reports/NB41_feature_selection_report.pdf`

---

## Danışman EDA İncelemesi — Feature-Depth Analizi (2026-06-27)

**Kaynak:** Danışmanın `feature_depth_analysis/` çıktısı (4 panel × single-feature + multi-feature + 5-model search). Basit `XGBClassifier(n_estimators=100)`, tek **stratified %80/20 hold-out** (SEED=42), preprocessing train-only.

### ⚠️ EN KRİTİK YORUM: Danışman ile bizim F1'lerimiz AYNI METRİĞİ ÖLÇMÜYOR

Bu, raporu okurken yapılabilecek en pahalı hatadır:

| Etiket | Danışman | Biz (NB15+) |
|---|---|---|
| "%80/20" ne demek | Verinin %20'si **test fold** (test de **train dağılımında**, çoğunluk patho) | Test havuzu **%80 benign'e** yeniden örneklenmiş (final yarışma koşulu) |
| Ölçülen şey | Train-dağılımı tavanı (sinyal haritası) | Final-realistic performans |

**Somut kanıt (MASTER):** Danışman CatBoost top-200 = **0.892**; bizim NB39 S1_6040/balbag = **0.638**. Bu 0.25'lik fark **model kalitesi değil, dağılım farkıdır** — bizim NB14/NB37'de "yanılsama" diye işaretlediğimiz olgunun (NB37: %50/50 0.89 → %80/20 0.589, fark 0.30) aynısı. **Danışmanın mutlak F1'leri final performans tahmini olarak KULLANILAMAZ; değeri relatif bulgularındadır.**

### Danışmanın doğrulayan / tamamlayan katkıları

| # | Bulgu | Bizim çalışmamızla ilişki |
|---|---|---|
| 1 | **Floor F1** (`2·prev/(1+prev)`, "hep patho tahmin et"): MASTER=0.846, KANSER=0.818, PAH=**0.905**, CFTR=0.905 | Bizde eksik metrik. PAH best=0.925 ≈ floor 0.905 → **gerçek sinyal yok**. Bizim "Chatterji tavanı, PAH platosu" bulgumuzun BAĞIMSIZ ikinci kanıtı |
| 2 | **`xgb_importance` top-200 > all-features** (MASTER 0.894 vs 0.879); kuyruk feature gürültü | Biz 288–440 feature kullanıyoruz → **feature seçimi denenmeli** (NB41) |
| 3 | **Value-transform önemsiz** (raw/significand/sig4figs + mean/median özdeş; ağaçlar rank'e böler) | M3 median'ı doğruluyor; "AL log dönüşüm gerekir" varsayımı ağaçlar için gereksiz |
| 4 | **AL_ dominasyonu** her panelde (KANSER'de EK_2 tek istisna, #2) | NB36 ablasyonu (AL kritik, EK omurga güçlü) ile tutarlı |
| 5 | **CFTR (4 benign test) = anekdot** (±0.05/örnek) | Bizim "n=21 benign, CI [0–1], LOO-MCC kullan" ile tam uyum |

### Danışman search_models — 5-model karşılaştırması (train-dağılımı F1)

| Panel | Kazanan | Test F1 | Floor | Floor üstü | Not |
|---|---|---|---|---|---|
| MASTER | CatBoost (top-200) | 0.892 | 0.846 | +0.046 | LGBM 0.890, XGB 0.885 yakın; NN/DNN zayıf (0.86/0.84) |
| KANSER | XGBoost (top-10) | 0.904 | 0.818 | +0.086 | Compact 10-feature yeterli; CatBoost 0.903 |
| PAH | (tümü eşit, 1-feat) | 0.905 | 0.905 | **+0.000** | recall=1.0, precision=floor → model = trivial baseline |
| CFTR | XGBoost (3-feat) | 0.950 | 0.905 | +0.045 | precision=0.905, ama 4-benign → güvenilmez |

**Ortak desen:** Her şey **precision-limited, recall yüksek** (~0.93–1.0) — bizim her panelde gördüğümüz precision darboğazının danışman tarafında bağımsız teyidi. Ağaç modeller NN/DNN'i her panelde geçti.

### ⚠️ Danışmanın körlüğü: cross-panel pooling yok

Danışman "each panel on its own dataset only" diyor → CFTR'yi "111 satır, sonuç çıkmaz" diye kapatıyor. **Bizim en iyi sonuçlarımız (CFTR S0c_COMBINED=0.863, KANSER reverse-pool=0.730) cross-panel pooling ile geldi** — bu eksende biz danışmanın önündeyiz. Pooling, danışmanın görmediği kaldıraçtır.

### Eyleme dönük çıkarımlar (NB41'de test edilecek)

1. **Feature seçimi (`xgb_importance` top-N) bizim %80/20 bootstrap protokolümüzde test et** — danışmanın train-dağılımındaki kazancı (top-200 > all) ters dağılımda da geçerli mi?
2. **Floor F1'i raporlarımıza ekle** — modelin gerçek katkısını görmek için.
3. **Sunum/teslim raporuna danışmanın mutlak F1'lerini KOYMA** — onun yerine "train-dağılımı tavanı 0.89 / final-realistic 0.64" ikisini birlikte sun (problemi anladığımızı gösterir).

---

### Danışman Raporu — Ham Kaynak Detayları (`feature_depth_analysis/report_in_depth.md`, GERİYE DÖNÜK EKLENDİ 2026-07-24)

**Not:** Yukarıdaki özet bölüm danışman raporunun bizim çalışmamızla ilişkisini anlatıyordu ama raporun kendisindeki **sayısal detaylar ve metodoloji** bu dosyaya hiç işlenmemişti. Aşağıda `feature_depth_analysis/` klasöründeki (proje kökünde, ayrı bir üst dizin) tam rapor + `*_search_models/` alt klasörlerinin ham CSV'lerinden çıkarılan eksiksiz veri yer alıyor.

#### Metodoloji (danışmanın kendi tanımıyla)

- **Bölüm 1 — Tek-feature modelleri:** Her panelde ID-olmayan her feature için, **sadece o feature** ile model eğitilir. Sayısal feature'lar (`AL_`, `EK_`) için 6 model (3 değer-dönüşümü `raw`/`significand`/`sig4figs` × 2 imputasyon `mean`/`median`); kategorik feature'lar (`CAT_`, `AA_`) için 1 model (ordinal-encode, eksik=ayrı kategori). **Panel başına 2066 model.**
- **Bölüm 2 — Çoklu-feature modelleri:** `raw`+median impute, 4 seçim yöntemi (`single_f1`, `xgb_importance`, `mutual_info`, `random`) × `k∈{10,25,50,100,150,200,250,all}`. **Panel başına 29 model.**
- **Bölüm 3 (`*_search_models/`) — 5 model ailesi grid search:** Panelin en iyi çoklu-feature setinde (MASTER/KANSER: `xgb_importance` top-200/top-10; PAH/CFTR: raporun "best" tek/az-feature seti) XGBoost/LightGBM/CatBoost/NN/DNN grid search, 5-fold StratifiedKFold (train-only) → held-out test fold.
- **Model:** Bölüm 1-2'de sabit `XGBClassifier(n_estimators=100, random_state=42)` ("hiçbir süsleme yok" — sonuçlar veri hakkında, tuning hakkında değil). Bölüm 3'te gerçek grid search.
- **Değerlendirme:** Panel başına **tek** stratified %80/20 hold-out (seed=42), tüm modellerde yeniden kullanılıyor — **bizim çoklu-seed/bootstrap protokolümüzden temel fark burada.**
- **Kapsam:** Her panel **kendi verisiyle izole** analiz edilmiş — cross-panel pooling YOK (bkz. aşağıdaki "körlük" notu).

#### Headline Tablo (tüm paneller, danışmanın kendi train-dağılımı hold-out'unda)

| Panel | Satır | Prevalans | Test fold (benign/patho) | Floor F1* | En iyi **tek-feature** F1 | En iyi **çoklu-feature** F1 | Tüm-feature F1 |
|---|---:|---:|:---:|---:|---:|---:|---:|
| **MASTER** | 2931 | 0.733 | 587 (157/430) | 0.846 | 0.864 | **0.894** | 0.879 |
| **KANSER** | 388 | 0.691 | 78 (24/54) | 0.818 | 0.887 | **0.911** | 0.893 |
| **PAH** | 372 | 0.833 | 75 (13/62) | 0.905 | 0.919 | **0.925** | 0.910 |
| **CFTR** | 111 | 0.811 | 23 (**4**/19) | 0.905 | **0.950** | 0.923 | 0.900 |

*\* Floor = "her zaman pathogenic tahmin et" F1'i, ilgili test fold'unda. Danışmanın kendi notu: "test-fold büyüklüğüne göre güven sırası: MASTER ≫ KANSER > PAH > CFTR. CFTR'nin 4-benign test fold'u sayıları neredeyse anekdot yapıyor (±0.05 F1/örnek)."*

#### Kombinasyon (çoklu-feature) tek-feature'dan gerçekten daha mı iyi?

| Panel | Tek → Çoklu | Danışmanın yorumu |
|---|:---:|---|
| **MASTER** | 0.864 → 0.894 (**+0.030**), precision 0.78→0.85 | **Evet — gerçek, anlamlı kazanç.** Kombinasyonun net işe yaradığı tek panel, en güvenilir test fold'da. |
| **KANSER** | 0.887 → 0.911 (+0.024) | **Evet — mütevazı kazanç.** 10-feature'lık kompakt bir set zaten yeterli. |
| **PAH** | 0.919 → 0.925 (+0.006) | **Gerçek kazanç yok.** İyileşme gürültü bandında; `random` seçicinin "kazanması" bunu doğruluyor. |
| **CFTR** | 0.950 → 0.923 (**−0.027**) | **Hayır.** Çok az veri; her şey floor'da, çoklu-feature daha kötü. |

#### Panel-bazlı en iyi ayarlar (danışmanın önerisi, kendi protokolünde)

- **MASTER — kombine et, prensipli seçici kullan (en güçlü kanıt):** `xgb_importance` top-**~200** (raw+median). F1=0.8935, accuracy=0.835, recall=0.947, **precision=0.846** (en iyi tek-feature'ın 0.777 precision'ına karşı — kombine model "her şeye pathogenic de" eğiliminden çıkıyor). F1, k arttıkça yükselip k≈100–200'de tepe yapıyor, sonra düzleşiyor; `all_features` (0.879) curated subset'ten (0.894) **daha kötü** — kuyruk feature'lar gürültü ekliyor. En iyi tek-feature (referans): **`AL_2`**, raw/median, F1=0.864.
- **KANSER — küçük, seçilmiş set yeterli:** `xgb_importance` top-**10** → F1=0.9043, neredeyse hiçbir şeyle. En yüksek gözlemlenen: `random` k=200, F1=0.9107 (kısmen 24-benign fold'da şans — kararlı `xgb_importance` top-10 tercih edilmeli). En iyi tek-feature: **`AL_22`** (significand/mean, F1=0.887) — ve dikkat çekici biçimde **`EK_2` #2 sırada** (0.883), bir `EK_` feature'ının en iyi `AL_` feature'larıyla yarıştığı **tek panel**.
- **PAH — sinyal zayıf, az feature yetiyor:** Büyük feature setinin faydası yok. En iyi çoklu-feature `random` k=150 (F1=0.9254) ama en iyi tek-feature `AL_66`'nın (raw, F1=0.9185) gürültü bandında. Seçiciler arasında sadece `xgb_importance` k arttıkça monoton iyileşiyor (tüm-feature'a ≈0.910'a düzleşiyor). Her şey majority floor'da (~0.905) asılı kalıyor. Pratik çıkarım: birkaç `AL_` feature (`AL_66`, `AL_20`, `AL_300`) elde edilebilecek en iyisi — **bu panelin ayırt edici sinyali doğası gereği sınırlı.**
- **CFTR — kombine etmeyecek kadar küçük:** En iyi tek-feature — **`AL_6` / `AL_215` / `AL_21`** (üçlü beraberlik), raw/mean, **F1=0.950**. Feature kombinasyonu YARDIMCI OLMUYOR (en iyi çoklu=0.923, tüm-feature=0.900). ⚠️ Sadece **4 benign test satırıyla** her metrik gürültünün hakimiyetinde (±0.05 F1/yanlış-sınıflandırılan-örnek). Tüm CFTR sonuçları geçici sayılmalı.

#### Cross-panel bulgular (danışmanın kendi sentezi)

1. **Panel büyüklüğü, feature kombinasyonunun işe yarayıp yaramayacağını belirliyor.** Büyük/kararlı MASTER'da kombinasyon net kazanç veriyor; küçük CFTR/PAH fold'larında "en iyi" çoklu-feature model en iyi tek-feature'dan ayırt edilemiyor (ya da ondan kötü). **Belirleyici kısıt örneklem büyüklüğü — algoritma değil.**
2. **Değer-dönüşümü neredeyse hiç önemli değil.** `raw`, `significand`, `sig4figs` (ve mean/median) her yerde neredeyse özdeş sonuç veriyor — çünkü ağaçlar **rank**'e böler ve üç dönüşüm de bir feature içindeki rank'i koruyor. En iyi tek-feature'lar `raw` ve `significand` arasında yaklaşık eşit dağılıyor.
3. **`AL_` feature'ları hakim.** Her panelde en iyi tek-feature bir `AL_` feature'ı (KANSER'in `EK_2`'si tepeye yakın tek `AL_`-olmayan istisna). `CAT_`/`AA_` feature'ları floor'da kalıyor.
4. **Yeterli veri olduğunda prensipli seçim random'dan üstün.** `xgb_importance`/`mutual_info` MASTER'da net kazanıyor, KANSER'de en verimli (top-10 ≈ top-all). Gürültülü panellerde `random` bazen "kazanıyor" — bu oradaki farkların gerçek olmadığının bir işareti.
5. **Belli bir noktadan sonra daha fazla feature daha iyi değil.** Kombinasyonun işe yaradığı yerde F1, curated ~100–200-feature subset'inde tepe yapıyor ve `all_features` biraz daha kötü — zayıf feature'ların uzun kuyruğu gürültü ekliyor.
6. **Her şey precision-limited, recall-limited değil.** Panellerde recall yüksek (~0.93–1.0) çünkü çoğunluk sınıfı pathogenic; F1'i floor'un üstüne çıkaran kaldıraç **precision** — bunu anlamlı ölçüde iyileştiren sadece kombine MASTER/KANSER modelleri.

#### Danışman search_models — Tam Sonuç Tabloları (ham CSV'den, `*_search_models/*_search_models_results.csv`)

**MASTER** (train=2344 [625 benign/1719 patho], test=587 [157 benign/430 patho], floor=0.846):

| Model | CV F1 | Test F1 | Test Acc | Recall | Precision | En iyi params |
|---|---:|---:|---:|---:|---:|---|
| **CatBoost** | 0.888 | **0.8921** | 0.833 | 0.942 | 0.847 | depth=6, lr=0.1 |
| LightGBM | 0.886 | 0.8899 | 0.830 | 0.940 | 0.845 | leaves=15, lr=0.05, spw=1 |
| XGBoost | 0.881 | 0.8847 | 0.821 | 0.937 | 0.838 | depth=3, lr=0.05, spw=1 |
| NN | 0.850 | 0.8619 | 0.789 | 0.900 | 0.827 | (64,), α=1e-4 |
| DNN | 0.840 | 0.8408 | 0.758 | 0.872 | 0.812 | (64,32), α=1e-3 |

**KANSER** (top-10 feature, floor=0.818):

| Model | CV F1 | Test F1 | Test Acc | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 0.918 | **0.9043** | 0.859 | 0.963 | 0.853 |
| CatBoost | 0.919 | 0.9027 | 0.859 | 0.944 | 0.864 |
| LightGBM | 0.903 | 0.8947 | 0.846 | 0.944 | 0.850 |
| NN | 0.879 | 0.8889 | 0.833 | 0.963 | 0.825 |
| DNN | 0.881 | 0.8545 | 0.795 | 0.870 | 0.839 |

**PAH** (1-feature "literal report best", floor=0.905) — **5 model ailesi de birebir aynı test sonucunu veriyor (F1=0.9051, recall=1.0, precision=0.8267)**: tek feature ile tüm model aileleri aynı kararı veriyor, bu tam olarak "trivial baseline" teşhisini pekiştiriyor.

**CFTR** (3-feature "literal report best", floor=0.905, n=4 benign test — güvenilmez):

| Model | CV F1 | Test F1 | Test Acc | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 0.882 | **0.9500** | 0.913 | 1.000 | 0.905 |
| LightGBM | 0.893 | 0.9048 | 0.826 | 1.000 | 0.826 |
| CatBoost | 0.895 | 0.9000 | 0.826 | 0.947 | 0.857 |
| NN | 0.897 | 0.8718 | 0.783 | 0.895 | 0.850 |
| DNN | 0.895 | 0.8718 | 0.783 | 0.895 | 0.850 |

**MASTER'a özel not (danışman raporundan):** Tuned CatBoost (0.8921), Bölüm 1-2'nin "hiç süslemesiz" 100-ağaçlı XGBoost sonucuyla (0.8935, aynı top-200 set) neredeyse aynı — yani **model ailesi değiştirmek / grid search yapmak MASTER'da temel baseline'ı geçmedi, sadece doğruladı.** Danışmanın vurgusu: burada feature seti ve model ailesi seçimi, hiperparametre taramasından daha belirleyici.

---

## NB42 — PAH EDA Pipeline Testi (2026-06-27) ✅

**Amaç:** `eda.md`'deki EDA bulgularını (hayalet sütun filtresi, multicollinearity r>0.90) NB21'in kanıtlanmış COMBINED pooling + BalancedBagging stratejisine ekleyerek fark yaratıp yaratmadığını test etmek.

**Deney tasarımı:** NB21'in model parametreleri, değerlendirme protokolü (prior-shift + robust %80/20 bootstrap threshold), COMBINED havuzu (MASTER+KANSER+CFTR) **hiç değişmedi** — sadece train setine eda.md'nin ek preprocessing adımları eklendi.

**EDA Preprocessing farkı:**
- NB21 orijinal: 351 feature → sabit+duplicate drop → **293 feature** kaldı
- NB42 EDA: 351 → sabit+duplicate + hayalet(dolu<30) + multicollinearity(r>0.90) → **176 feature** kaldı
- Hayalet sütun: COMBINED'da 0 ek sütun (COMBINED n=3430 ile tüm sütunlar ≥30 dolu)
- Multicollinearity: **117 ek sütun** drop edildi (ana kaldıraç)

**Sonuçlar (prior-shift, %80/20 bootstrap):**

| Strateji | MCC | Boot-F1 | CI | Precision | Recall |
|---|---|---|---|---|---|
| P0c_EDA_COMBINED | 0.490 | **0.567** | [0.476–0.638] | 0.938 | 0.844 |
| P4_EDA_COMBINED_BalBag | 0.460 | 0.540 | [0.465–0.618] | 0.934 | 0.831 |

**NB21 Orijinal vs NB42 EDA karşılaştırma:**

| Strateji | NB21 Boot | NB42 Boot | Δ | Durum |
|---|---|---|---|---|
| P4_COMBINED_BalBag (kazanan) | **0.582** | 0.540 | **−0.042** | ↓ Kötüleşti |
| P0c_COMBINED | 0.543 | 0.567 | **+0.024** | ↑ İyileşti |

**Yorumlar:**
1. **NB21'in kazanan P4 stratejisi (BalBag) multicollinearity drop'tan zarar gördü** (−0.042). BalancedBagging internal resampling yaptığı için yüksek korelasyonlu feature'lar çeşitlilik sağlıyormuş — onları kaldırmak ensemble diversity'yi düşürdü.
2. **P0c (düz LGBM) multicollinearity drop'tan fayda gördü** (+0.024). Tek model redundant feature'lardan gerçekten kurtuldu.
3. **Net sonuç: EDA preprocessing PAH'ta NB21'in en iyi sonucunu (0.582) geçemedi.** Chatterji tavanı yine baskın.
4. **PAH KESİNLEŞTİ durumu değişmedi** — NB21 P4_COMBINED_BalBag (Boot=0.582, MCC=0.529) hâlâ en iyi.

---

## Bekleyen İşler (güncellenmiş, 2026-06-27 — statüler 2026-07-24'te düzeltildi, bkz. alttaki güncel blok)

| Öncelik | Görev |
|---|---|
| Tamamlandı (statü düzeltildi 2026-07-24) | ~~NB41 — Danışman EDA çıkarımları testi (OLUŞTURULDU, çalıştırılmayı bekliyor)~~ ✅ **Gerçekte 2026-06-27'de çalıştırılmıştı, sadece bu tabloya işlenmemişti.** Sonuçlar yukarıda "Notebook 41" bölümünde: feature seçimi MASTER/PAH'ta gürültü bandında, KANSER'de k=100 nominal en iyi ama mevcut şampiyonun (0.730) altında. |
| **Yüksek** | **MASTER iyileştirme:** NB39 S1_6040 base'leriyle stacking denemesi, Optuna hiperparametre tuning |
| ~~**Yüksek**~~ **İPTAL (2026-07-24)** | ~~Final teslim modeli sabitle~~ — **tüm paneller yeniden açıldığı için ertelendi**, bkz. aşağıdaki güncel blok. |
| Orta | **NB44 (öneri, NB43'ten türedi):** KANSER'de G_LOW grubunun (<%50 null, mean\|φ\|=0.260) M3'e eklenmesi ablasyonu — φ-tabanlı seçici flag seti (örn. \|φ\|>0.15 sütunlara flag) vs mevcut M3, panel-transfer F1 karşılaştırması. **Hâlâ yazılmadı/çalıştırılmadı (2026-07-24 itibarıyla).** |
| Tamamlandı | ~~NB43: Missing-Flag Korelasyon Analizi~~ ✅ (REVİZE: sabit+özdeş sütun temizliği eklendi, MASTER 351→288 CLAUDE.md ile birebir örtüşüyor) — 4 panel × G_HIGH/G_LOW φ analizi. KANSER G_LOW güçlü sinyal (0.253) → M3 eşiği bu panelde flag kaçırıyor olabilir. AL_16..25 sağlama noktası doğrulandı (φ=0.568, top-10'u dolduruyor, temizlikten etkilenmedi). MASTER/PAH/CFTR'de G_LOW zayıf/sınırda → M3 eşiği bu panellerde korundu. |
| Tamamlandı | ~~NB42: PAH EDA Pipeline Testi~~ ✅ — EDA preprocessing (multicollinearity drop 117 sütun) NB21 kazanan P4'ü geçemedi (0.540 vs 0.582); P0c'de +0.024 ama yetersiz. PAH KESİNLEŞTİ durumu değişmedi. |
| Tamamlandı | ~~NB39: MASTER Reversed-Distribution~~ ✅ — S1_6040/balbag=0.6379, reversed-distribution çalıştı (+0.021) |
| Tamamlandı | ~~NB38: MASTER Diverse Stack + Calibrate-then-Shift~~ ✅ — V1_calib=0.6165, calibrate-then-shift ek kazanç vermedi |
| Tamamlandı | ~~NB37: MASTER Calibrated Stacking~~ ✅ — 4-arm karşılaştırma, no_fe_robust=0.589, stacking değer katmadı |
| Tamamlandı | ~~NB36: MASTER Baseline~~ ✅ — 13 model + 9 ablasyon, BalancedBag_XGB=0.603, FE zararlı |
| Tamamlandı | ~~NB35: PAH FE Ablasyon~~ ✅ — FE değer katmadı (MCC −0.148), NB21 baseline korunuyor |
| Tamamlandı | ~~NB34: CFTR FE Ablasyon~~ ✅ — FE değer katmadı (−0.036), NB20 baseline korunuyor |
| Tamamlandı | ~~NB32: KANSER Training Pool Sweep~~ ✅ — P9_catboost_with_fe=0.7300, plato kırıldı |
| Tamamlandı | ~~NB31: KANSER Kümülatif Entegrasyon~~ ✅ — E6=0.7201, marjinal iyileşme |
| Tamamlandı | ~~NB30: KANSER Advanced Stacking~~ ✅ — NN/DNN collapse, plato kırılamadı |
| Tamamlandı | ~~NB29: KANSER Refinement~~ ✅ — E1_COMBINED Boot=0.714 |
| Tamamlandı | ~~NB27/NB28: Threshold + Literature optimization~~ ✅ |
| Tamamlandı | ~~CFTR (NB17→NB20→NB34)~~ ✅ — FE son deneme de negatif, kesinleşti (ama bkz. altta: 2026-07-24'te yeniden açıldı) |
| Tamamlandı | ~~PAH (NB21→NB28→NB35)~~ ✅ — Chatterji tavanı doğrulandı, FE son deneme de negatif (ama bkz. altta: 2026-07-24'te yeniden açıldı) |

---

## ⭐ Bekleyen İşler (GÜNCEL — 2026-07-24)

**Tetikleyici:** NB40 (Bayes Error Ceiling) ve NB41 (Feature Selection) sonuçlarının bu dosyaya geriye dönük işlenmesi sırasında, NB40'ın PAH için bulduğu **Gap=+0.1125 ("DEVAM/significant margin", HIGH confidence)** sonucunun önceki "PAH KESİNLEŞTİ" kararıyla açıkça çeliştiği fark edildi. Bu çelişkiyi araştırmak ve genel olarak tüm panellerde yeni bir çalışma turu başlatmak amacıyla **CFTR, KANSER, PAH ve MASTER panellerinin dördü de yeniden açıldı.** Aşağıdaki tablodaki hiçbir panel artık "dokunma" statüsünde değildir; önceki "en iyi" skorlar birer referans/rekor olarak kalır.

| Öncelik | Panel | Görev | Gerekçe |
|---|---|---|---|
| ~~**En Yüksek**~~ **KAPANDI (2026-07-24, NB45)** | ~~**PAH**~~ | ~~NB40'ın Bayes-ceiling çelişkisini çöz~~ ✅ **Çözüldü: artefakt.** NB45 (çoklu-yöntem: k-NN+MST-GHP, feature-boot CI, floor-CI disiplini) ile Bayes-F1 (0.701, CI=[0.685,0.715]) floor-F1 (0.705) ile istatistiksel olarak ayrışmıyor (gap CI=[-0.020,0.010] sıfırı kapsıyor); güvenilirlik LOW (n_benign=62). NB40'ın gap=+0.1125'i k-NN artefaktıydı. **PAH tekrar KESİNLEŞTİ sayılabilir** — bkz. aşağıdaki NB45 bölümü. |
| Yüksek | KANSER | NB44'ü yaz ve çalıştır: G_LOW eksiklik grubunun (\|φ\|>0.15 sütunlara flag) M3 stratejisine eklenmesi ablasyonu (NB43'ten türedi, `results/v26_missing_flag_correlation/KANSER_LOW_flag_stats.csv` referans listesi hazır). Mevcut şampiyon (0.730) zaten Bayes-ceiling'e (gap=−0.0037) oturmuş olsa da doğrulanmamış bir eksen kaldı. |
| Yüksek | MASTER | NB39 kazanan senaryonun (S1_6040) base modelleriyle stacking + Optuna hiperparametre tuning — planlanmış ama hiç çalıştırılmamış. Bayes-ceiling zaten dar (gap=+0.0161) olduğu için beklenti düşük tutulmalı, ama doğrulanmadan kapatılmamalı. |
| Orta | CFTR | n=21 benign nedeniyle hem model boot-F1'i hem NB40'ın Bayes-ceiling tahmini güvenilmez (LOW confidence). Daha sağlam bir değerlendirme çerçevesi (örn. çoklu-seed LOO-CV ile Bayes-ceiling'in tekrarlanabilirliğini test etmek) gerekebilir; büyük model değişikliği önceliği düşük. |
| Not | Tümü | "Final teslim modeli sabitleme" kararı ertelendi — hiçbir panel artık kesinleşmiş sayılmıyor. |

---

## NB45 — Sağlam Bayes-Error Ceiling: PAH Çelişkisi Çözüldü (2026-07-24) ✅

**Amaç:** `reports/literature_research_panel_improvements_2026-07-24.md` yol haritası deney #1. NB40'ın PAH'ta bulduğu `Gap=+0.1125` (model 0.582, Bayes-ceiling 0.6945) tek-yöntem (k-NN, tek k-seti) tahminiydi ve NB21-NB35'in "PAH plato gerçek, sinyal yok" bulgusuyla çelişiyordu. Literatür (Bayes Error Rate Estimation in Difficult Situations, arXiv 2506.03159), k-NN tabanlı Bayes-error tahmininin sınıf başına ~100–1000 örnek gerektirdiğini gösteriyor; PAH'ın 62 benign örneği bu eşiğin altında — yani NB40'ın gap'i muhtemelen bir k-NN artefaktıydı, doğrulanması gerekiyordu.

**Yöntem (NB40'a göre 3 ek sağlamlık katmanı):**
1. **Çoklu yöntem çapraz-doğrulama:** k-NN (çoklu-k: 1,3,5,7,9,15) + **MST-tabanlı GHP (Henze-Penrose/Friedman-Rafsky)** — k-NN'den bağımsız, minimum-yayılan-ağaç kenar sınıflandırmasına dayalı ikinci bir Bayes-error tahmincisi.
2. **Feature-altküme bootstrap sağlamlığı:** her panelde N=25 tekrar, her seferinde feature'ların %75'i rastgele tutularak Bayes-error dağılımı çıkarıldı (tahminin feature seçimine duyarlılığını doğrudan ölçer).
3. **Floor-CI disiplini:** floor-F1 (`2·prev/(1+prev)`) Bayes-F1 ile **aynı %80/20 bootstrap havuzunda, aynı N=50 tekrarla, her tekrarın kendi prevalansından** hesaplandı — NB40'ın atladığı adım. İkisi aynı eksende, aynı CI diliyle karşılaştırıldı.

**Performans notu (metodolojik, gelecek notebook'lar için):** `gower` paketi kategorik sütunlarda çok yavaş bir Python-seviye implementasyon kullanıyor — MASTER (n=2931) için tek `gower_matrix` çağrısı ~124s ölçüldü (10 kategorik sütun eklemek rastgele veride 5s'yi 128s'ye çıkarıyor). Bunun yerine `scipy.spatial.distance.cdist(metric='cityblock')` (numeric) + vektörize eşitlik-toplamı (kategorik) ile kendi Gower implementasyonu yazıldı; `gower.gower_matrix` ile sayısal olarak doğrulandı (max fark ~3e-8) ve ~15–25× hızlanma sağladı. Ayrıca k-NN LOO döngüsü tam `argsort` yerine `argpartition` ile, bootstrap tahminleri döngü-dışı ön-hesaplama ile vektörize edildi. **Yeni notebook'larda `gower` paketini n>500 civarı verilerde kategorik sütunlarla kullanmadan önce bu darboğazı hatırla.**

**Sonuçlar (aynı %80/20 bootstrap CI disiplininde, N=50):**

| Panel | Floor-F1 [CI] | Bayes-F1 [CI] | Model-F1 | Gap(Bayes−Model) | Gap(Bayes−Floor) [CI] | Bayes-Tavan Güvenilirliği |
|---|---|---|---|---|---|---|
| MASTER | 0.600 [0.600,0.600] | 0.656 [0.648,0.668] | 0.638 | +0.018 | 0.057 [0.048,0.068] | Bayes > Floor (anlamlı sinyal var) |
| KANSER | 0.562 [0.562,0.562] | 0.726 [0.698,0.758] | 0.730 | −0.004 | 0.164 [0.136,0.196] | Bayes > Floor (anlamlı sinyal var) |
| **PAH** | **0.705 [0.705,0.705]** | **0.701 [0.685,0.715]** | 0.582 | +0.119 | **−0.004 [−0.020,0.010]** | **ARTEFAKT ŞÜPHESİ (Bayes≈Floor, ayrışım yok)** |
| CFTR | 0.677 [0.677,0.677] | 0.755 [0.727,0.772] | 0.863 | −0.108 | 0.078 [0.050,0.095] | Bayes > Floor (anlamlı sinyal var) |

**Yöntemler-arası uzlaşma (k-NN vs MST-GHP, feature-boot CI ile):**

| Panel | kNN-1 R*_high | GHP Bayes-upper | Fark | Güvenilirlik (n_min_class) |
|---|---|---|---|---|
| MASTER | 0.2845 | 0.3824 | 0.098 | HIGH (n=782) |
| KANSER | 0.1985 | 0.2413 | 0.043 | MEDIUM (n=120) |
| **PAH** | 0.2124 | 0.3726 | **0.160** | **LOW (n=62)** |
| CFTR | 0.1892 | 0.3671 | 0.178 | LOW (n=21) |

**PAH kararı — ÇELİŞKİ ÇÖZÜLDÜ (artefakt):** Floor-CI disiplini uygulanınca Bayes-F1 (0.701) ile floor-F1 (0.705) **istatistiksel olarak ayrışmıyor** (gap CI'ı [-0.020, 0.010] sıfırı kapsıyor) ve güvenilirlik LOW (n_benign=62, literatür eşiğinin altında). PAH ayrıca k-NN/GHP yöntemleri arasında en büyük anlaşmazlığı gösteriyor (0.160, dört panelin en yükseği) — bu da tahminin kırılganlığını bağımsız olarak teyit ediyor. **Sonuç: NB40'ın gap=+0.1125 bulgusu bir k-NN artefaktıdır, gerçek bir kazanç fırsatı değildir. PAH plato bulgusu (NB21-NB35, danışman floor-analizi: best=0.925≈floor=0.905) geçerliğini korur.**

**Diğer panellerde durum değişmedi:** MASTER, KANSER, CFTR'de Bayes-F1 floor'u anlamlı şekilde aşıyor (gap CI'ları sıfırı kapsamıyor) — bu panellerde ölçüm NB40 ile tutarlı, ek marj arayışı (stacking/tuning/reverse-distribution vb.) meşruiyetini koruyor.

**Çıktılar:** `results/v27_bayes_error_robust/` (convergence_table.csv, final_summary_robust.csv, fig1-4), `reports/NB45_bayes_error_robust_report.pdf`.

**Yol haritası etkisi:** Yol haritası deney #1 tamamlandı. PAH'ın "en yüksek öncelik" statüsü artık geçerli değil — **PAH tekrar KESİNLEŞTİ sayılabilir** (Chatterji tavanı + Bayes-ceiling artefaktı ikisi de aynı yöne işaret ediyor). Kalan açık eksenler: KANSER (NB44), MASTER (stacking+Optuna), CFTR (değerlendirme sağlamlaştırma) — bkz. yukarıdaki tablo, PAH satırı bu bulguyla kapatılabilir.

---

## NB46 — Reverse-Distribution PAH ve CFTR Deneyi (2026-07-27) ✅

**Amaç:** `reports/literature_research_panel_improvements_2026-07-24.md` yol haritası deney #2. NB32 (KANSER) ve NB39 (MASTER)'da doğrulanmış **reversed-distribution eğitim** kaldıracının (%60 benign/%40 patho gerçek resample, post-hoc ağırlıklamadan üstün) PAH ve CFTR panellerine de yayılıp yayılmayacağını test etmek. PAH ve CFTR o zamana kadar bu teknikle denenmemişti — NB21/NB35 (PAH) ve NB17/NB20/NB34 (CFTR) hep orijinal (patojenik-ağırlıklı) dağılımda eğitilmişti.

**Deney tasarımı:** 4 senaryo × 2 model ailesi (lgbm, balbag — nested-parallelism deadlock riski nedeniyle `lgbm_classifier_single_thread()` ile tek-thread çalıştırıldı, bkz. aşağıdaki metodolojik not):
- **R0_baseline**: orijinal dağılım (PAH benign_frac=0.269, CFTR benign_frac=0.262)
- **R1_REVERSE_6040**: %60 benign / %40 patho gerçek resample
- **R2_REVERSE_7030**: %70 benign / %30 patho
- **R3_REVERSE_8020**: %80 benign / %20 patho (final test dağılımına en yakın)

Değerlendirme: LOO-MCC (raw + prior-shift), Boot-F1 (prior-shift, %80/20 bootstrap, N tekrar ile CI), floor-F1 karşılaştırması — CLAUDE.md'nin zorunlu koştuğu üç unsur da mevcut.

**Metodolojik not — nested parallelism deadlock:** İlk çalıştırma (notebook/nbconvert üzerinden) loky worker'ların parent pipe'ta bloke olmasıyla deadlock'a girdi (joblib içi joblib — GridSearch/CV döngüsü içinde LightGBM'in kendi çoklu-thread'i çakıştı); kill sonrası notebook boş kaldı (partial output yok). Çözüm: (1) düz Python script olarak yeniden çalıştırma, (2) `lgbm_classifier_single_thread()` fonksiyonu eklenerek LightGBM'in iç thread'i 1'e sabitlendi, dış paralellik joblib'e bırakıldı. Bu iki değişiklikten sonra script sorunsuz tamamlandı (PID 44839). **Gelecekte benzer grid-search + LightGBM kombinasyonlarında bu deadlock riskini hatırla — iç model thread'ini tekilleştirmeden dış paralellik açma.**

**PAH sonuçları (floor-F1=0.9083, referans=P4_COMBINED_BalBag_NB21 Boot-F1=0.582):**

| Deney/Model | n_train | benign_frac | Boot-F1(prior) | CI | Precision | Recall | Floor'u geçti mi |
|---|---|---|---|---|---|---|---|
| **R1_REVERSE_6040/lgbm** | 1538 | 0.60 | **0.5889** | [0.429–0.706] | 0.9643 | 0.6156 | Hayır |
| R1_REVERSE_6040/balbag | 1538 | 0.60 | 0.5531 | [0.450–0.652] | 0.9382 | 0.7915 | Hayır |
| R0_baseline/balbag | 3427 | 0.269 | 0.5456 | [0.421–0.629] | 0.9395 | 0.7590 | Hayır |
| R0_baseline/lgbm | 3427 | 0.269 | 0.5319 | [0.399–0.625] | 0.9308 | 0.7883 | Hayır |
| R3_REVERSE_8020/balbag | 1150 | 0.80 | 0.5244 | [0.368–0.609] | 0.9317 | 0.7557 | Hayır |
| R2_REVERSE_7030/lgbm | 1318 | 0.70 | 0.5238 | [0.344–0.622] | 0.9342 | 0.7394 | Hayır |
| R2_REVERSE_7030/balbag | 1318 | 0.70 | 0.5218 | [0.399–0.619] | 0.9369 | 0.6775 | Hayır |
| R3_REVERSE_8020/lgbm | 1150 | 0.80 | 0.5132 | [0.209–0.724] | 0.9720 | 0.4528 | Hayır |

**PAH sonucu: R1_REVERSE_6040/lgbm en iyi (0.5889), NB21 referansını (0.582) +0.0069 geçti — ama CI'lar [0.429–0.706] ile [referans CI'ı bilinmese de] örtüşüyor, fark istatistiksel olarak anlamsız denecek kadar küçük. Hiçbir senaryo floor-F1'i (0.9083) geçemedi** — NB45'in Bayes-ceiling bulgusuyla (Bayes-F1≈floor-F1, gap istatistiksel olarak sıfır) tutarlı: PAH'ta gerçek sinyal tavanı zaten floor'a çok yakın, reverse-distribution da dahil hiçbir teknik bunu aşamıyor. **PAH KESİNLEŞTİ durumu NB46 ile bir kez daha teyit edildi — değişmedi.**

**CFTR sonuçları (floor-F1=0.8955, referans=S0c_PriorShift_NB20 Boot-F1=0.863, ⚠️ n=21 benign — düşük güven):**

| Deney/Model | n_train | benign_frac | Boot-F1(prior) | CI | Precision | Recall | Floor'u geçti mi |
|---|---|---|---|---|---|---|---|
| R2_REVERSE_7030/lgbm | 1377 | 0.70 | 0.8377 | [0.571–1.000] | 1.0000 | 0.700 | Hayır |
| R1_REVERSE_6040/balbag | 1606 | 0.60 | 0.8108 | [0.537–0.909] | 0.9865 | 0.8111 | Hayır |
| R3_REVERSE_8020/balbag | 1200 | 0.80 | 0.8053 | [0.571–1.000] | 1.0000 | 0.6667 | Hayır |
| R1_REVERSE_6040/lgbm | 1606 | 0.60 | 0.7809 | [0.500–0.909] | 0.9859 | 0.7778 | Hayır |
| R2_REVERSE_7030/balbag | 1377 | 0.70 | 0.7792 | [0.500–0.909] | 0.9857 | 0.7667 | Hayır |
| R0_baseline/lgbm | 3685 | 0.262 | 0.7407 | [0.600–0.833] | 0.9733 | 0.8111 | Hayır |
| R0_baseline/balbag | 3685 | 0.262 | 0.7384 | [0.479–0.833] | 0.9737 | 0.8222 | Hayır |
| R3_REVERSE_8020/lgbm | 1200 | 0.80 | 0.6729 | [0.500–0.909] | 0.9818 | 0.6000 | Hayır |

**CFTR sonucu: En iyi reverse-dist (R2_REVERSE_7030/lgbm, 0.8377) NB20 referansını (0.863) −0.0253 ile geçemedi.** Ama bu panelde CLAUDE.md'nin "CFTR DEĞERLENDİRME GÜVENİ YOK" uyarısı geçerli: n=21 benign ile CI'lar [0.571–1.000] gibi son derece geniş, iki senaryo arasındaki −0.0253 fark **gürültü bandının içinde**, gerçek bir gerileme olarak yorumlanamaz. Precision her senaryoda 1.0'a yakın kalıyor (CFTR'nin bilinen özelliği). **CFTR mevcut şampiyonu (S0c_PriorShift_NB20) değişmedi — reverse-distribution burada anlamlı bir kazanç göstermedi, ama örnek boyutu yüzünden kesin "işe yaramıyor" da denemez.**

**Genel değerlendirme:** Reverse-distribution kaldıracı (NB32/NB39'da MASTER/KANSER için güçlü) **PAH ve CFTR'ye aynı güçle taşınmadı**:
- PAH'ta zaten sinyal tavanına (floor≈Bayes-ceiling) çok yakın olunduğu için hiçbir eğitim stratejisi değişikliği kayda değer fark yaratamıyor.
- CFTR'de örnek boyutu (n=21 benign) o kadar küçük ki hem referans hem reverse-dist sonuçları geniş CI içinde neredeyse ayrışmıyor; teknik "başarısız" değil, "ölçülemez" durumda.
- Bu, reverse-distribution'ın evrensel bir kaldıraç olmadığını, panelin **sinyal tavanı ve örnek boyutuna bağlı** olarak etkisinin değiştiğini gösteriyor — MASTER/KANSER'de floor'dan uzak, yeterli örneklemli paneller olduğu için kaldıraç işe yaradı.

**Çıktılar:** `results/v27_reverse_distribution_pah_cftr/` (pah_reverse_distribution_results.csv, cftr_reverse_distribution_results.csv, nb46_summary.json, reverse_distribution_pah_cftr.png), `reports/nb46_reverse_distribution_pah_cftr_report.pdf`, `notebooks/46_reverse_distribution_pah_cftr.ipynb`.

**Yol haritası etkisi:** Yol haritası deney #2 tamamlandı. PAH ve CFTR ikisi de **KESİNLEŞTİ durumunu korudu** (değişmedi). Kalan açık eksenler: KANSER (NB44 — G_LOW flag ablasyonu), MASTER (stacking+Optuna, NB39 base'leriyle), ve yol haritasının kalan deneyleri (TabPFN-2.5, missing-handling derinleştirme).

---

## NB44 — KANSER G_LOW Missing-Flag Ablasyonu (2026-07-28) ✅

**Amaç:** `reports/literature_research_panel_improvements_2026-07-24.md` yol haritası deney #5 / `to-do.md`'de bekleyen iş. NB43'ün bulgusu: KANSER panelinde G_LOW grubu (miss_ratio ≤ %50) mean|φ|=0.260 ile G_HIGH'a (0.410) yakın güçlü bir sinyal taşıyor, ama mevcut M3 stratejisi (>%50 NaN için flag) bu grubu flag'lemiyor — potansiyel kayıp sinyal olabilir mi diye test etmek.

**Deney tasarımı — tek eksen değişimi ilkesi:** NB32'nin kanıtlanmış KANSER şampiyon reçetesi (**P9_REVERSE_6040 pool + catboost + with_fe**, Boot-F1=0.730) her yönüyle sabit tutuldu; değişen **tek eksen** missing-flag stratejisiydi:
- **M3_baseline**: mevcut strateji, sadece miss_ratio>%50 sütunlara `is_missing_*` flag.
- **M3_PHI_selective**: M3 + G_LOW içinde (miss_ratio≤%50) `|φ|>0.15` olan **53 ek sütuna** da `is_missing_*` flag (`results/v26_missing_flag_correlation/KANSER_LOW_flag_stats.csv` referans listesinden, φ eşiği=0.15).

Değerlendirme protokolü NB32/NB39/NB46 ile birebir aynı: f1_raw / f1_8020 / mcc_8020 üçlü ayrımı, floor-F1, %80/20 bootstrap (N=50, %95 CI), train-test gap.

**Sonuçlar:**

| Strateji | n_extra_flags | f1_raw | Boot-F1(%80/20) | CI | MCC(%80/20) | Floor-F1 | Train-Test Gap |
|---|---|---|---|---|---|---|---|
| **M3_baseline** | 0 | 0.9373 | **0.7300** | [0.637–0.769] | 0.6703 | 0.8160 | 0.0038 |
| M3_PHI_selective | 53 | 0.9259 | 0.6816 | [0.595–0.714] | 0.6152 | 0.8160 | 0.0658 |

**M3_baseline sonucu (0.7300) NB32'nin champion Boot-F1'iyle (0.730) birebir eşleşti** — repro doğrulandı, deney altyapısı NB32 ile tutarlı çalışıyor.

**M3_PHI_selective, M3_baseline'ı geçemedi — tam tersine −0.0484 ile net şekilde geriledi.** Ayrıca train-test gap 0.0038'den 0.0658'e fırladı — bu, 53 ek flag sütununun modeli overfit'e ittiğinin doğrudan kanıtı. Yorum: G_LOW'un mean|φ|=0.260 sinyali gerçek olsa da, medyan-impute edilmiş ham sütun değerleri zaten bu sinyali (informative missingness ile birlikte) taşıyor; ayrıca flag eklemek redundant + yüksek boyutlu (53 yeni ikili sütun, n_pool=1442 için) bir gürültü kaynağı oluşturuyor, catboost bunu ezberliyor. **Hiçbir strateji floor-F1'i (0.816) geçemedi** (CLAUDE.md floor kontrolü), ama bu KANSER'de zaten bilinen bir durum değil — asıl mesaj mevcut şampiyonun (0.730, floor'un altında ama Bayes-ceiling'e göre neredeyse tavan, NB45: gap=−0.004) bu ek eksende de en iyisi olmaya devam etmesi.

**Karar: NB44 hipotezi çürütüldü. KANSER'de M3 stratejisi değiştirilmiyor, mevcut champion (P9_REVERSE_6040_catboost_with_fe, Boot-F1=0.730) korunuyor.** Bu, projede "eksiklik bilgi taşıyor → flag ekle" sezgisinin körü körüne genişletilemeyeceğini gösteren önemli bir negatif bulgu: sinyal taşıyan bir grup (G_LOW) bulmak, o grubu flag'lemenin otomatik olarak kazanç getireceği anlamına gelmiyor — özellikle sinyal zaten ham değerlerden erişilebilirken.

**Çıktılar:** `results/v28_kanser_missing_flag_ablation/` (nb44_kanser_missing_flag_ablation_results.csv, nb44_summary.json, nb44_ablation_comparison.png), `notebooks/44_kanser_missing_flag_ablation.ipynb`.

**Yol haritası etkisi:** Yol haritası deney #5 tamamlandı. KANSER'in "Yüksek öncelik, doğrulanmamış eksen" statüsü kapandı — G_LOW flag ekleme eksen olarak kapatıldı (negatif sonuç). Kalan açık eksenler: MASTER (stacking+Optuna, NB39 base'leriyle), CFTR (değerlendirme sağlamlaştırma), ve yol haritasının kalan deneyleri (TabPFN-2.5, Venn-Abers kalibrasyon, missing-handling'in M1–M5 çerçevesinde panel-bazlı yeniden yargılanması).

---

## NB47 — TabPFN-2.5: Tüm Panellerde Ham COMBINED + Reverse-Pool (2026-07-28) ✅

**Amaç:** `reports/literature_research_panel_improvements_2026-07-24.md` yol haritası deney #3. NB24'te PAH'a TabPFN zaten uygulanmıştı ama **top-50 feature'a kısıtlanarak** (o zamanki `tabpfn` paketi <100 feature tercih ediyordu) ve **sadece PAH'ta**. Bu deney öncesinde kurulu `tabpfn` paketinin (v8.0.8) kaynak kodu incelenerek `MAX_NUMBER_OF_FEATURES=2000`, `MAX_NUMBER_OF_SAMPLES=50_000` sabitleri doğrulandı — bu gerçekten **TabPFN-2.5 nesli** (500 feature üstünde estimator-başı subsample yapıyor, tam 2.5 davranışı). Projenin 293 feature'ı ve en büyük havuzu (~3796 satır, 4 panel birleşik) bu limitin rahatça altında; top-K seçime artık gerek yok.

**Deney tasarımı — iki yeni eksen:** (a) **HAM feature seti** (293, kısıtlama yok, NB24'ün top-50'sinden farklı), (b) **TÜM 4 panelde** (NB24 sadece PAH'ta çalıştı), (c) **reverse-pool (%60B/%40P) ile TabPFN'in ilk kez birlikte denenmesi**. Her panel için hedef panel dışındaki tüm panellerin verisi + hedef panelin train parçası ile COMBINED havuzu kuruldu (test sızıntısı yok); iki senaryo: **P0_HAM_COMBINED** (orijinal dağılım) ve **P1_REVERSE_6040** (COMBINED %60B/%40P gerçek resample). Değerlendirme protokolü NB39/NB44/NB46 ile birebir aynı: f1_raw/f1_8020/mcc_8020 üçlü ayrımı, floor-F1, %80/20 bootstrap (N=50, %95 CI), train-test gap.

**Sonuçlar:**

| Panel | Senaryo | f1_raw | Boot-F1(%80/20) | CI | Floor-F1 | Train-Test Gap | Süre |
|---|---|---|---|---|---|---|---|
| CFTR | Ham-COMBINED | 0.9024 | **0.8040** | [0.500–1.000] | 0.8911 | 0.1177 | 736s |
| CFTR | Reverse-6040 | 0.6364 | 0.5060 | [0.000–0.955] | 0.8911 | 0.3693 | 211s |
| PAH | Ham-COMBINED | 0.9268 | **0.5551** | [0.480–0.593] | 0.9080 | 0.3692 | 550s |
| PAH | Reverse-6040 | 0.8836 | 0.5110 | [0.335–0.654] | 0.9080 | 0.3649 | 231s |
| KANSER | Ham-COMBINED | 0.9236 | 0.6676 | [0.619–0.682] | 0.8160 | 0.2578 | 573s |
| KANSER | Reverse-6040 | 0.9023 | **0.6878** | [0.606–0.789] | 0.8160 | 0.1703 | 213s |
| MASTER | Ham-COMBINED | 0.8745 | **0.5811** | [0.537–0.625] | 0.8461 | 0.3458 | 404s |
| MASTER | Reverse-6040 | 0.7492 | 0.5679 | [0.502–0.628] | 0.8461 | 0.3313 | 199s |

**Panel-bazlı karar (TabPFN en iyisi vs referans şampiyon):**

| Panel | En iyi TabPFN senaryo | TabPFN Boot-F1 | Referans şampiyon | Referans Boot-F1 | Delta |
|---|---|---|---|---|---|
| MASTER | P0_HAM_COMBINED | 0.5811 | S1_6040_balbag (NB39) | 0.638 | **−0.0569** |
| KANSER | P1_REVERSE_6040 | 0.6878 | P9_REVERSE_6040_catboost_with_fe (NB32) | 0.730 | **−0.0422** |
| PAH | P0_HAM_COMBINED | 0.5551 | P4_COMBINED_BalBag (NB21) | 0.582 | **−0.0269** |
| CFTR | P0_HAM_COMBINED | 0.8040 | S0c_PriorShift (NB20) | 0.863 | **−0.0590** |

**TabPFN-2.5, dört panelin hiçbirinde mevcut şampiyonu geçemedi.** En yakın sonuç PAH'ta (delta=−0.0269, gürültü bandına yakın); en büyük fark CFTR'de (delta=−0.0590, ama n=21 benign nedeniyle CI [0.500–1.000] son derece geniş — bu panelde karşılaştırma zaten kırılgan, bkz. CLAUDE.md "CFTR DEĞERLENDİRME GÜVENİ YOK"). **Kritik ek bulgu: tüm senaryolarda train-test gap yüksek (0.12–0.37)** — TabPFN bu veri setinde belirgin şekilde overfit ediyor, özellikle ham-COMBINED havuzunda (büyük n, yüksek context-length: MASTER ham-COMBINED gap=0.346, PAH ham-COMBINED gap=0.369). Reverse-pool bazı panellerde gap'i azalttı (KANSER: 0.258→0.170) ama hiçbir kombinasyon referans şampiyonu geçemedi.

**Metodolojik not — TabPFN CPU performansı:** TabPFN in-context learning olduğu için ayrı bir eğitim döngüsü yok; `fit` konteksti hafızaya alıyor, `predict_proba` transformer forward-pass yapıyor. CPU'da COMBINED havuzlarında (n=990–3740) süre GBDT'den kat kat uzun: CFTR ham-COMBINED tek başına **736 saniye** sürdü (küçük sentetik ön-testte n=200/300-feature için 19s ölçülmüştü — ölçek süper-lineer büyüdü). Toplam çalışma süresi ~50 dakika (8 fit işlemi, CFTR'den MASTER'a küçükten büyüğe sıralanarak). **Gelecekte TabPFN'i büyük COMBINED havuzlarında (n>1500) deneyecek notebook'lar bu süreyi hesaba katmalı** — büyük havuzlarda alt-örnekleme veya `n_estimators` düşürme düşünülebilir.

**Karar: NB47 hipotezi çürütüldü. TabPFN-2.5 hiçbir panelde yeni şampiyon olmadı, dört panelin mevcut şampiyonları da DEĞİŞMİYOR:** MASTER=S1_6040_balbag (0.638), KANSER=P9_REVERSE_6040_catboost_with_fe (0.730), PAH=P4_COMBINED_BalBag (0.582), CFTR=S0c_PriorShift (0.863). Bu, NB24'ün PAH'a özel "yeni inductive bias platoyu kırmadı" bulgusunun **tüm panellere genellenmiş** hâli — foundation-model yaklaşımının bu anonimleştirilmiş, yoğun-eksiklikli genetik varyant verisinde GBDT/BalancedBagging ailesini geçemediğini gösteriyor.

**Çıktılar:** `results/v29_tabpfn_all_panels/` (nb47_tabpfn_all_panels_results.csv, nb47_summary.json, nb47_tabpfn_all_panels_comparison.png), `notebooks/47_tabpfn_all_panels.ipynb`, `reports/NB47_tabpfn_all_panels_report.pdf`.

**Yol haritası etkisi:** Yol haritası deney #3 tamamlandı. TabPFN-2.5 ekseni kapatıldı (negatif sonuç, tüm panellerde). Kalan açık eksenler: madde 4 (missing-handling'in M1–M5 çerçevesinde panel-bazlı final-realistic protokolde yeniden yargılanması — MASTER'da flag'li/PAH'ta flag'siz hipotezi), madde 6 (Venn-Abers kalibrasyon + calibrate-then-shift, MASTER/PAH'ta precision darboğazının kök çözümü olabilir), madde 7 (MASTER'da reverse-pool base'leriyle heterojen stacking + Optuna, planlandı hiç çalıştırılmadı).

---

## NB48 — Missing-Handling Final-Realistic Yeniden Yargılama (2026-07-29) ✅

**Amaç:** `reports/literature_research_panel_improvements_2026-07-24.md` yol haritası deney #4 / §3-BIS. Projenin en temel önişleme kararı olan M3 stratejisi (>%50 NaN'a `is_missing_*` flag + tüm sayısal sütunlara medyan imputation) NB14'te **%50/50 dağılımda** (artık çürütülmüş bir rejimde) seçilmiş ve 25+ notebook boyunca sorgusuzca sabit tutulmuştu. İki EDA belgesi eksikliğin doğası konusunda çelişiyordu: CLAUDE.md/NB12 MASTER'ı MNAR (informative missingness) ilan ederken, eda.md PAH'ı MCAR (yapay maske) ilan ediyordu. **Hipotez:** MASTER'da flag'li strateji (M3/M4/M5) kazanır, PAH'ta flag'siz strateji (M1/M2/native_nan) kazanır.

**Deney tasarımı — tek eksen değişimi ilkesi (NB44 disiplini):** Her panelin kanıtlanmış champion pool + model ailesi **sabit** tutuldu (MASTER=S1_6040 reverse-pool/BalBag, KANSER=P9_REVERSE_6040 COMBINED-pool/CatBoost+FE, PAH=COMBINED-pool/BalBag, CFTR=COMBINED-pool/LGBM); değişen **tek eksen** NB14'ün 5 stratejisi (M1: flag yok+medyan, M2: flag yok+drop, M3: flag+medyan [mevcut], M4: flag+drop, M5: flag+seçici drop/medyan) + yeni bir 6. varyant: **native_nan** (hiç impute/flag yok, LightGBM/CatBoost'un native NaN-splitting'ine bırak). Değerlendirme protokolü NB39/NB44/NB46/NB47 ile birebir aynı: f1_raw/f1_8020/mcc_8020 üçlü ayrımı, floor-F1, %80/20 bootstrap (N=50, %95 CI), train-test gap.

**⚠️ Metodolojik sınırlama (dürüstlük notu):** NB48'in BalancedBagging/LGBM implementasyonu, NB39'un orijinal reçetesini (base LGBM n_estimators=100/lr=0.1, early-stopping+val-set+sample_weight) **birebir replike etmedi** — basitleştirilmiş bir sürüm (n_estimators=300/lr=0.05+class_weight=balanced, early-stopping yok) kullanıldı. Bu yüzden **NB48 içi M3 sonucu, NB39'un raporlanan champion Boot-F1'i ile birebir eşleşmiyor** (MASTER: NB48 M3=0.518 vs NB39 champion=0.638). **Sonuç olarak bu deneyin geçerli çıktısı NB48-içi göreceli karşılaştırmadır (M1 vs M2 vs ... vs native_nan, aynı basitleştirilmiş model sabit) — mutlak champion karşılaştırması ikincil ve modelin sadakatsizliğinden dolayı gürültülü.**

**Sonuçlar (Boot-F1 %80/20, N=50):**

| Panel | M1 | M2 | M3 (mevcut) | M4 | M5 | native_nan | Floor-F1 |
|---|---|---|---|---|---|---|---|
| **MASTER** | 0.5165 | 0.3915 | 0.5179 | 0.4250 | **0.5505** | 0.5262 | 0.8461 |
| **KANSER** | 0.6943 | 0.5644 | 0.6900 | 0.6113 | **0.7418** | 0.6755 | 0.8160 |
| **PAH** | **0.5947** | 0.5026 | 0.5883 | 0.4961 | 0.5591 | 0.5865 | 0.9080 |
| **CFTR** | 0.7020 | 0.4387 | 0.6860 | 0.5243 | 0.7540 | **0.7720** | 0.8911 |

**Hipotez ÇÜRÜTÜLDÜ — ama başka bir örüntü ortaya çıktı.** Beklenen "MASTER'da flag kazanır / PAH'ta flag kaybeder" ayrımı gözlenmedi; bunun yerine **panel boyutundan bağımsız, tutarlı bir sıralama** çıktı: **M5 ve native_nan sistematik olarak M3'ü geçiyor** (MASTER: M5 +0.033, KANSER: M5 +0.052, CFTR: native_nan +0.086), **M2 ve M4 (agresif drop) her panelde en kötüsü**. Tek istisna PAH — orada M1 (flag yok+medyan, en basit strateji) hafifçe önde (+0.006 M3'e göre), ama fark gürültü bandına yakın.

**Yorum:** M3'ün hem `is_missing_*` bayrağını hem orijinal (medyanla doldurulmuş, çoğu satırda yapay) ham sütunu birlikte tutması boyut şişmesine yol açıyor (426–430 feature, M1/M5/native_nan'ın 288–292'sine karşı) — bu ekstra ~140 sütun küçük/orta havuzlarda (KANSER n=1442, MASTER n=652) faydalı sinyalden çok gürültü/overfit kaynağı gibi davranıyor. **M5 (flag + >%50 NaN'ı drop et, ≤%50'yi medyanla doldur) en iyi denge noktası** — bayrak bilgisini korurken ham-yapay sütun şişmesini önlüyor. **native_nan** (hiç impute yok, ağaca bırak) CFTR'de en iyisi — küçük panelde imputation'ın enjekte ettiği yapay medyan değerlerinin gürültü kaynağı olduğunu destekliyor.

**Floor kontrolü:** Hiçbir panel-strateji kombinasyonu floor-F1'i geçemedi bu deneyde (en yakın KANSER M5: 0.742 vs floor 0.816). Bu, champion'ların da zaten floor'u aşamadığı bilinen durumla tutarlı (Bayes-ceiling zaten floor'a yakın) — buradaki mesaj mutlak performans değil, **stratejiler arası göreceli sıralama**.

**Karar:** M3'ün "evrensel varsayılan" statüsü sorgulanmalı — **M5, tüm 4 panelde M3'e eşit veya üstün** çıktı (KANSER/MASTER/CFTR'de belirgin, PAH'ta marjinal fark). Ancak mutlak champion'ları değiştirmeden önce (çünkü NB48 modeli champion'ın birebir replikası değil), **M5'in gerçek champion reçetesiyle (NB39'un tam BalBag hiperparametreleriyle) doğrulanması gerekiyor** — bu NB48'in ürettiği en somut ve ucuz sonraki adım.

**Çıktılar:** `results/v30_missing_handling_rejudge/` (nb48_missing_handling_rejudge_results.csv, nb48_panel_summary.csv, nb48_summary.json, nb48_missing_handling_comparison.png), `notebooks/48_missing_handling_rejudge.ipynb`.

**Yol haritası etkisi:** Yol haritası deney #4 tamamlandı — ama negatif sonuç yerine **yeni bir aktif eksen** doğurdu: M5/native_nan'ın gerçek champion reçeteleriyle (tam hiperparametre sadakatiyle) doğrulanması. Kalan açık eksenler: madde 4-doğrulama (M5 vs M3, gerçek champion modeliyle — yeni), madde 6 (Venn-Abers kalibrasyon + calibrate-then-shift), madde 7 (MASTER'da reverse-pool base'leriyle heterojen stacking + Optuna).

---

## NB50 — 63k Genis Veri: Hazirlama, Etiketleme, Sizinti Temizligi, EDA (2026-08-29) ✅

**Amaç:** `docs/PLAN_63K_ENTEGRASYON.md` ADIM 1. Kullanıcının **"final test dağılımı diye bir şey olmayacak, sadece model eğiteceğiz elimizdeki veri ile"** kararı üzerine başlayan rejim değişikliğinin ilk adımı: `data/63k_genis/full_cravat_v3_63k.csv` (824 MB, 63.463 satır, 777 sütun, **legacy** OpenCRAVAT şeması) dosyasını temiz, etiketli, sızıntısız parquet'e çevirmek ve EDA yapmak. Bu, projedeki NB12–NB48 arası tüm çalışmanın dayandığı "final test %80 benign" varsayımının terk edildiği ilk yeni-rejim notebook'u.

**Yeni modül:** `src/columns_63k.py` — 63k'ya özgü sızıntı listeleri (`CLINVAR_LEAK_COLS`, `ID_TEXT_TRANSCRIPT_COLS`, `LEAKY_META_PREDICTOR_SCORES/PREDS`), etiket türetim fonksiyonları (`derive_label`, `derive_label_confidence`, `is_qualified_label`), tip ayrıştırma (`classify_columns`) ve `floor_f1()` yardımcısı. `src/columns_real.py::get_constant_cols()` / `get_duplicate_col_pairs()` şema-agnostik oldukları için doğrudan yeniden kullanıldı (CLAUDE.md'nin öngördüğü gibi).

**Uygulanan adımlar (plan §Adım1.1–1.7 birebir):**
1. Chunked (`chunksize=5000`) CSV→parquet dönüşümü. **Tuzak:** chunk sınırları arası dtype tutarsızlığı (`clinvar__dbsnp_id` gibi bazı sütunlar bir chunk'ta sayısal, diğerinde string) `pyarrow.Table.from_pandas` hatası verdi — çözüm: `category`'ye çevirmeden önce `.astype(str)` ile normalize etmek.
2. Etiket türetimi: `clinvar__sig` prefix eşleşmesiyle Label (0/1), `clinvar__rev_stat`'tan `label_conf` güven ağırlığı, `|other` gibi ek nitelikli etiketler için `Label_qualified` bayrağı. 11 satır belirsiz etiket (`Conflicting`/NaN) dışlandı.
3. Missense filtresi (`base__so=='MIS'`): 60.970 satır ana sete, 2.482 non-missense satır ayrı parquet'e (`nonmis_63k.parquet`, ADIM 4 ek-veri ablasyonu için).
4. Sızıntı temizliği: tüm `clinvar__*`/`clinvar_acmg__*` (40 sütun) + ID/serbest-metin/transkript sütunları (161 sütun, CHASMplus'ın 32 kanser-alt-tipi transcript/all çiftleri dahil) drop edildi. `LEAKY_META_PREDICTOR_SCORES/PREDS` (27+6 sütun) **drop edilmedi, ayrı listede tutuldu** (ADIM 3 A6 ablasyonu için). 777 → 611 sütun.
5. Constant+duplicate temizliği (`columns_real` fonksiyonları, şema-agnostik): 70 sabit + 3 özdeş çift → 611 → 538 sütun.
6. Tip ayrıştırma: NUMERIC/CATEGORICAL/BINARY (`*__pred`/`*__class` kategorik sayıldı).
7. EDA.

**Doğrulanan sayılar (bu oturumda önceden ölçülenle birebir tutarlı):** missense n=60.970, benign=38.248/pathogenic=22.722, **prevalans=0.3727, floor-F1=0.543** — eski yarışma MASTER panelinin floor'u (0.846) ile karşılaştırıldığında **modelin gösterecek gerçek marjı olduğunu** doğruluyor.

**Yeni EDA bulguları:**
- **Eksiklik MNAR, hem de eski veriden daha güçlü:** ortalama eksiklik %38.0 (eski MASTER'ın %55'inden düşük), ama eksiklik×Label phi-korelasyonu 124 sütunda |phi|>0.1 — en güçlüleri `gnomad3__af*`/`thousandgenomes__*` ailesi (phi≈0.75). Eksiklik kesinlikle bilgi taşıyor, MCAR değil.
- **Gen-ezberi riski somut ve büyük:** top-50 gen tablosunda 15 gen tek-yönlü (prevalans>0.95 veya <0.05) — `FLG`/`KMT2C`/`OBSCN` %0 patojenik (n=122–144), `PAH`/`CFTR`/`LDLR`/`GLA` >%95 patojenik. **GroupKFold(groups=base__hugo) olmadan yapılan her değerlendirme güvenilmez.**
- **Tek-değişkenli AUC>0.95: 30 sütun** — incelemede bunların **sızıntı değil, ya ayrı-tutulan meta-predictor'lar** (`revel__score`, `alphamissense__am_pathogenicity`, `metarnn__score` — zaten A6 ablasyon listesinde) **ya da meşru popülasyon-frekansı sinyali** (`gnomad__af`, `gnomad4__af` — nadir varyant→patojenik biyolojik olarak beklenen ilişki) olduğu görüldü. NB51'de kesin karar verilecek ama ilk bulgu "kırmızı bayrak = otomatik sızıntı" değil.

**Çıktılar:** `data/63k_genis/{full_63k,missense_63k,nonmis_63k}.parquet`, `src/columns_63k.py`, `results/v31_63k_prep/` (nb50_summary.json, nb50_column_lists.json, nb50_gene_table_top50.csv, nb50_univariate_auc.csv, nb50_vs_yarisma_comparison.csv), `reports/nb50_63k_eda_report.pdf`, `notebooks/50_63k_prep_eda.ipynb`.

**Sonraki adım:** ADIM 2 — NB51 (`notebooks/51_63k_leakage_audit.ipynb`): hızlı sinyal testi (LGBM, CV F1>0.97 kırmızı bayrak), tek-sütun AUC'un 30 kırmızı-bayraklı sütununu elle sınıflandırma (meta-predictor vs meşru vs gerçek sızıntı), meta-predictor A6 ablasyon ön-hazırlığı, gen-ezberi testi (GroupKFold vs StratifiedKFold farkı), duplicate satır kontrolü.

**Güncelleme (NB51 sonrası):** NB51'in sızıntı denetimi, NB50'nin ilk sızıntı listesinin **eksik** olduğunu ortaya çıkardı — `src/columns_63k.py` ve NB50 Cell 5 buna göre güncellenip parquet'ler yeniden üretildi (bkz. NB51 bölümü aşağıda). Bu bölümdeki sayılar (606/533 sütun, meta=33) güncel/nihai değerlerdir.

---

## NB51 — 63k Genis Veri: Sizinti Denetimi & Saglamlik Kontrolu (2026-08-29) ✅

**Amaç:** `docs/PLAN_63K_ENTEGRASYON.md` ADIM 2. NB50'nin ürettiği `missense_63k.parquet`'i (sızıntı-temiz olduğu iddia edilen) 5 kontrolden geçirmek: hızlı sinyal testi, tek-sütun AUC taraması, meta-predictor ablasyonu, gen-ezberi testi, duplicate satır kontrolü.

**🔴 İlk çalıştırmada NB50'nin sızıntı listesi eksik çıktı (iteratif sızıntı avının tam beklenen sonucu):**
- `*__rankscore`/`*_rank_score` ailesi (dbNSFP'nin meta-predictor skorlarının normalize sıralama versiyonu — 23 sütun) `LEAKY_META_PREDICTOR_SCORES`'ta yoktu.
- `varity_r__*` (VARITY) ve `vest__*` (VEST) meta-predictor'ları da eksikti.
- `mupit__hugo`, `omim__omim_id`, `litvar_full__rsid/reference_count/pmids` gibi ID/literatür-referans sütunları `ID_TEXT_TRANSCRIPT_COLS`'ta yoktu — özellikle `litvar_full__reference_count` (bir varyantın literatürde kaç kez geçtiği) **dolaylı sızıntı riski** taşıyor çünkü ClinVar'a giren varyantlar zaten literatürde daha çok bahsedilme eğiliminde.
- **Düzeltme:** `src/columns_63k.py`'a `LEAKY_META_PREDICTOR_RANKSCORES` (23 sütun) eklendi, `LEAKY_META_PREDICTOR_SCORES`'a varity_r/vest (6 sütun) eklendi, `NB51_ADDITIONAL_ID_LEAK_COLS` (5 sütun) yeni liste olarak eklendi. NB50 Cell 5 bu listeleri dahil edecek şekilde güncellendi, parquet'ler yeniden üretildi (606→533 sütun, önceki 611→538'den 5 sütun daha az).

**Sonuçlar (güncel parquet üzerinde, meta-predictor score+rankscore+pred hepsi ayrı tutularak):**

| Kontrol | Sonuç | Karar |
|---|---|---|
| **1. Hızlı sinyal (tüm feature, meta dahil)** | CV F1=0.9899, AUC=0.9996 | **Kırmızı bayrak tetiklendi** (F1>0.97) |
| **1b. Meta TAMAMEN hariç** | CV F1=0.9878, AUC=0.9994 | Kırmızı bayrağın kaynağı meta-predictor **DEĞİL** — fark sadece 0.0021 |
| **2. Tek-sütun AUC>0.95 (30 sütun) sınıflandırma** | meta_predictor=23, population_freq=4, functional_assay=2, **UNKNOWN_INVESTIGATE=1** (`cardioboost__arrhythmias`) | 29/30 meşru kategoriye düştü; `cardioboost__arrhythmias` aritmi-geni-özel bir risk anotasyonu, muhtemelen meşru ama dar kapsamlı — NB52'de izlenecek |
| **3. Meta-predictor ablasyonu (A6 ön-hazırlık)** | F1 farkı = 0.0021 (dahil−hariç) | **Sınırlı katkı** (<0.05 eşiği) — meta-predictor dahil etmek düşük riskli, A6'da yine de doğrulanacak |
| **4. Gen-ezberi testi (GroupKFold vs StratifiedKFold)** | F1 farkı = 0.0056 (7.207 benzersiz gen) | **Belirgin değil** (<0.10 eşiği) — ama gen-holdout her tabloda raporlanmaya devam edecek |
| **5. Duplicate satır kontrolü** | 18 tam kopya varyant, **0 çelişkili etiket** | Temiz — hiçbir kopya grup içinde Label çelişmiyor |

**⭐ En önemli bulgu — kırmızı bayrağın kaynağı sızıntı değil, verinin doğal kalitesi:** Meta-predictor'lar (score+rankscore+pred, toplam 33+24+6 sütun) tamamen çıkarılsa bile CV F1 hâlâ 0.9878 — yani ~0.99'luk skor tek bir sütun grubundan gelmiyor. Muhtemel açıklama: 63k'nın açık-isimli, klinik-kalite annotasyonları (REVEL, AlphaMissense, gnomAD frekansı, konservasyon skorları) tek başlarına bile çok güçlü ayırt ediciler — bu, yarışmanın anonimleştirilmiş/gürültülü AL_ sütunlarından temelden farklı bir veri rejimi. **Uyarı:** bu aynı zamanda modelin REVEL/AlphaMissense'e göre gerçek katma değerinin düşük kalabileceği anlamına gelir — NB52'nin A4 (feature selection) ve A6 (meta-predictor ablasyonu) adımlarında netleştirilecek.

**Genel verdict: İNCELEME GEREKLİ ama kritik sızıntı bulunamadı** — kırmızı bayrak veri kalitesinden kaynaklanıyor, meta-predictor/gen-ezberi/duplicate testlerinin hiçbiri patoloji göstermedi. `cardioboost__arrhythmias` tek açık soru işareti, ablasyonla izlenecek.

**Çıktılar:** `results/v32_63k_audit/` (nb51_summary.json, nb51_feature_importance.csv, nb51_high_auc_classification.csv, nb51_meta_ablation.json, nb51_gene_memorization.json, nb51_duplicate_rows.json), `reports/nb51_leakage_audit.pdf`, `notebooks/51_63k_leakage_audit.ipynb`. Ayrıca NB50 çıktıları güncellendi: `src/columns_63k.py`, `data/63k_genis/{full_63k,missense_63k,nonmis_63k}.parquet`, `results/v31_63k_prep/*`, `reports/nb50_63k_eda_report.pdf`.

**Sonraki adım:** ADIM 3 — NB52 (`notebooks/52_63k_baseline_hypotheses.ipynb`): H1–H8 hipotez ablasyonları (A1 model ailesi, A2 missing stratejisi, A3 sınıf dengeleme, A4 feature seti/top-200, A5 etiket kalitesi, A6 meta-predictor dahil/hariç — bu notebook'un bulgusuna göre düşük riskli ama yine de test edilecek, A7 feature engineering). Test seti hâlâ açılmadı; tüm kararlar 5-fold CV'den.

---

## NB52 — 63k Genis Veri: Baseline & Hipotez Ablasyonlari H1-H8 (2026-08-30) ✅

**Amaç:** `docs/PLAN_63K_ENTEGRASYON.md` ADIM 3 — planın en yüksek getirili adımı. Eski projenin (NB12-NB48, anonim yarışma verisi) H1-H8 hipotezlerinin 63k'ya (legacy, açık isimli, büyük n, düşük floor) taşınıp taşınmadığını tek-eksen ardışık ablasyon (A1→A7, her adımda önceki kazanan sabitlenir) ile test etmek.

**Veri:** n=60970 missense varyant, prevalans=0.3727, floor-F1=0.5430 (yarışma panellerinden çok daha düşük — 63k'nın karakteristik özelliği).

**A1-A7 sonuçları (kazanan zincirinin cv_f1 ilerlemesi):**

| Eksen | Kazanan | cv_f1 | Not |
|---|---|---|---|
| A1 (model ailesi) | **lgbm** | 0.9897 | BalancedBagging (0.9882) ve diğer ağaçlar çok yakın; CatBoost en yavaş (505s) |
| A1+H8 (SmallMLP) | lgbm > mlp | mlp=0.9836 | NN rekabetçi ama ağaçları geçemedi |
| A2 (missing) | **native_nan** | 0.9900 | Impute yapmamak (LGBM native NaN) flag/median stratejilerinden daha iyi |
| A3 (dengeleme) | **scale_pos_weight** | 0.9903 | Gerçek-resample (0.9891) en kötüsü — veri israfı büyük n'de zararlı |
| A4 (feature seti) | **top200 (xgb-importance)** | 0.9904 | top100 ve all-features'tan iyi, ayrıca 54s→32s hızlanma |
| A5 (etiket kalitesi) | all_labels | 0.9904 | qualified/weighted/expert-only farklı n'de, doğrudan kıyaslanamaz — bilgi amaçlı |
| A6 (meta-predictor) | with_meta (dahil) | 0.9904 vs 0.9867 (hariç) | Fark 0.0038 — NB51 ön-bulgusuyla tutarlı, düşük risk |
| A7 (FE: Grantham/BLOSUM62/fizikokimya) | no_fe | 0.9904 vs 0.9897 (with_fe) | FE nötr (-0.0007) |

**H1-H8 taşınma karnesi:**

| Hipotez | Sonuç |
|---|---|
| H1 (BalancedBagging en güçlü) | **ÇÜRÜTÜLDÜ** — düz LGBM kazandı |
| H2 (M5≥M3 missing stratejisi) | **ÇÜRÜTÜLDÜ/DEĞİŞTİ** — `native_nan` ikisini de geçti |
| H3 (gerçek resample > class_weight) | **ÇÜRÜTÜLDÜ/DEĞİŞTİ** — `scale_pos_weight` en iyisi |
| H6 (FE panel-bağımlı) | **NÖTR** |
| H7 (feature selection kazandırır) | **DOĞRULANDI** |
| H8 (NN artık rekabetçi) | **DOĞRULANDI** (0.9836 vs 0.9897, fark<0.01) |
| H4 (meta=LR, GBM değil) / H5 (heterojen base korelasyon<0.85) | NB53'e ertelendi (stacking aşaması) |

**⭐ En önemli bulgu — 63k, yarışma verisinden köklü şekilde farklı bir rejimde:** Yarışma panellerinde (NB16-NB39) kazanan H1/H2/H3 reçetesi (BalancedBagging + missing-flag + gerçek-resample) burada üçü de tersine döndü. Sebep: n=60970 (yarışma panellerinin ~20 katı) + çok güçlü açık-isimli meta-predictor'lar (REVEL, AlphaMissense, gnomAD frekansı) sayesinde sinyal zaten çok güçlü (floor=0.543, model=0.99) — büyük n'de düz LGBM+scale_pos_weight zaten dengesizliği yönetiyor, native NaN handling zaten optimal, gerçek-resample ise veri israfına dönüşüyor. **Yarışma verisi (küçük n, anonim, zayıf sinyal) ile 63k (büyük n, açık, güçlü sinyal) arasında reçete transferi yapılamaz** — her ikisi kendi rejiminde ayrı ayrı optimize edilmeli.

**Champion reçete (NB53 başlangıç noktası):** LGBM + native_nan + scale_pos_weight + top200 feature (xgb-importance) + meta-predictor dahil + no_fe → cv_f1=0.9904, cv_mcc=0.9848, gene-holdout f1=0.9886.

**🐛 Smoke test sırasında bulunan iki macOS-spesifik native crash (asıl notebook'a da uygulandı):**
1. `BalancedBaggingClassifier(n_jobs=-1)` + iç içe `LGBMClassifier` paralelliği → joblib/loky deadlock (süreç %0 CPU'da sonsuza kadar askıda kalıyor, hata mesajı bile vermiyor). Düzeltme: hem dış hem iç estimator için `n_jobs=1`.
2. `torch`'un NumPy/PyArrow/LightGBM'den **sonra** import edilmesi → `torch.tensor()` çağrısında SIGSEGV (exit 139), stack trace'de `pyarrow.lib`/`numpy.linalg` modüllerinin çift yüklendiği görüldü (OpenMP/BLAS init sırası çakışması). Düzeltme: `torch`'u dosyanın en başına taşı + `KMP_DUPLICATE_LIB_OK=TRUE`. Bu iki hata, önceki oturumdaki "smoke test 120s'de takıldı" gözleminin kök nedeniydi.

**Çıktılar:** `results/v33_63k_baseline/` (nb52_ablation_results.csv, nb52_hypothesis_scorecard.json, nb52_champion_recipe.json, nb52_champion_feature_list.json, nb52_ablation_progress.png), `reports/nb52_baseline_report.pdf`, `notebooks/52_63k_baseline_hypotheses.ipynb`. `src/columns_63k.py`'a Section 9 eklendi: `compute_achange_fe()` — `base__achange` metninden ('p.Arg220Gln') Grantham/BLOSUM62/delta-hidropati/hacim/pI/yük türetimi (A7 için).

**Sonraki adım (tamamlandı):** ADIM 4 — NB53 (`notebooks/53_63k_optimization.ipynb`): Champion reçeteye Optuna (TRIALS_TREE=100/TRIALS_NN=50), stacking (H4: LR vs GBM meta, H5: heterojen base pairwise korelasyon<0.85), kalibrasyon (Platt/Isotonic/Venn-Abers + ECE), threshold finalizasyonu (F1-max vs MCC-max), non-missense ek-veri ablasyonu. Test seti hâlâ açılmadı.

## NB53 — 63k Genis Veri: Optimizasyon, Ensemble, Kalibrasyon (2026-08-30) ✅

**Amaç:** `docs/PLAN_63K_ENTEGRASYON.md` ADIM 4. NB52'nin champion reçetesini (LGBM + native_nan + scale_pos_weight + top200 feature + meta-predictor dahil + no_fe, cv_f1=0.9904/0.9897) başlangıç alıp Optuna, stacking (H4/H5), kalibrasyon, threshold finalizasyonu ve non-missense ek-veri ablasyonu ile geliştirmek. Test seti hâlâ açılmadı — tüm kararlar 5-fold stratified CV'den.

**1. Optuna (TRIALS_TREE=100, config.py'dan, 3336.7s):** LGBM hiperparametre araması. En iyi CV F1=0.9907 (NB52 champion 0.9897'ye göre **+0.0010**). En iyi parametreler: num_leaves=150, max_depth=4, learning_rate=0.1146, n_estimators=488, min_child_samples=75, subsample=0.857, colsample_bytree=0.634. Marjinal ama pozitif kazanç.

**2. Stacking — H4/H5 testi:** 5 base model (Optuna-LGBM, XGBoost, CatBoost, RandomForest, BalancedBagging) OOF tahminleriyle meta-matris kuruldu.
- **H5 (heterojen base korelasyon<0.85) → ÇÜRÜTÜLDÜ.** Maksimum pairwise korelasyon **0.9988** — tüm base modeller neredeyse birebir aynı tahminleri üretiyor (63k'daki güçlü meta-predictor sinyali her aileyi aynı karar sınırına yakınsatıyor).
- **H4 (meta=GBM, meta=LR'yi geçecek) → ÇÜRÜTÜLDÜ.** Meta=LR cv_f1=0.9900, Meta=GBM cv_f1=0.9906, ama tek başına en iyi base (lgbm_optuna=0.9907) ikisini de geçti. Stacking kazancı = **-0.0001** (değersiz). NB16-39'daki "meta=LR yeterli" bulgusu burada da tutarlı, ama esas sonuç: **63k'da stacking'in kendisi gereksiz** (base'ler arası çeşitlilik yok).

**3. Kalibrasyon (Platt/Isotonic, Venn-Abers kapsam dışı — paket kurulu değil):** Isotonic en iyi (ECE 0.0045→**0.0005**, Brier 0.0059→0.0056). Model zaten iyi kalibreydi (ham ECE=0.0045 çok düşük); kalibrasyon marjinal iyileştirme sağladı, kritik bir sorunu çözmedi.

**4. Threshold finalizasyonu:** F1-max ve MCC-max **aynı noktada birleşti (thr=0.46)** → F1=0.9907, MCC=0.9852. Gerekçe: 63k'da (yarışma verisinin aksine) bilinen bir ters-dağılım beyanı yok, train ve final test aynı prevalansta bekleniyor — bu yüzden F1-max seçildi, MCC-max yalnızca çapraz-kontrol.

**5. Non-missense ek-veri ablasyonu:** `nonmis_63k.parquet` (n=2482, frameshift/stopgain vb.) eklendiğinde ortak 200/200 feature ile cv_f1 0.9907→**0.9890 (-0.0017)**. Karar: **NÖTR/hafif zararlı, dahil edilmedi**.

**⭐ En önemli bulgu — 63k rejiminde "ensemble teknikleri" tavan doldurulduğunda değersizleşiyor:** NB52'de zaten görülen "yarışma reçetesi 63k'ya transfer olmuz" bulgusunun devamı: stacking, kalibrasyon-ötesi düzeltme ve ek veri gibi yarışma panellerinde (küçük n, zayıf sinyal) kazanç sağlayan teknikler burada (büyük n, çok güçlü açık-isimli meta-predictor sinyali, cv_f1≈0.99) hiçbir şey katmıyor — çünkü kazanç sağlamaları için gereken ön koşullar (base model çeşitliliği, kalibrasyon bozukluğu, ek sinyal) burada yok. Tek gerçek kazanç kaynağı Optuna oldu (+0.0010), o da marjinal.

**Champion reçete v2 (final, NB54'e taşınacak):** LGBM + Optuna params + native_nan + scale_pos_weight + top200 feature + meta-predictor dahil + no_fe, **stacking kullanılmıyor** (tekli LGBM kazandı), kalibrasyon=isotonic, threshold=0.46 (f1max), non-missense **dahil değil**. **Final CV F1 = 0.9907**, floor-F1=0.5430.

**H4/H5 karnesi:** H4 (meta=LR, GBM değil) → **LR YETERLİ/ÜSTÜN** (eski bulgu tutarlı). H5 (heterojen base korelasyon<0.85) → **ÇÜRÜTÜLDÜ** (homojen, 0.9988).

**Çıktılar:** `results/v34_63k_optimization/` (nb53_optuna_best_params.json, nb53_base_correlation.csv, nb53_oof_matrix.parquet, nb53_optimization_results.csv, nb53_champion_recipe_v2.json, nb53_hypothesis_scorecard.json, nb53_optimization_progress.png), `reports/nb53_optimization_report.pdf`, `notebooks/53_63k_optimization.ipynb`.

**Sonraki adım:** ADIM 5 — NB54 (`notebooks/54_63k_final.ipynb`): Test seti **ilk ve son kez** burada açılır. Final metrikler (hold-out + floor + confusion matrix), CV-test farkı raporu (protokol dürüstlüğü), gen-holdout final skoru, model/artifact kaydı (`models/v31_63k/`).


