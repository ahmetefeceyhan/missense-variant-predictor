# 63k Geniş Veri Entegrasyon Yol Planı (NB50+)

> **Kime:** Bu planı uygulayacak agent'a.
> **Tarih:** 2026-08-28
> **Durum:** Bu plan, kullanıcının şu kararından sonra yazılmıştır:
> **"Final test dağılımı diye bir şey olmayacak bundan sonra, sadece model eğiteceğiz elimizdeki veri ile."**

---

## 0. ÖNCE OKU — Rejim Değişikliği (en kritik bölüm)

Bu projede NB12–NB48 arası **tüm** çalışma, "yarışmanın final test seti ~%80 benign olacak"
varsayımı üzerine kuruluydu. **Bu varsayım artık geçersiz.** Sonuçları:

### ❌ ARTIK GEÇERSİZ (uygulama)

| Ne | Neden düştü |
|---|---|
| **%80/20 bootstrap değerlendirme** (birincil metrik idi) | Taklit edilecek bir final dağılım yok |
| `f1_8020` / `mcc_8020` threshold modları | Aynı sebep |
| "Reverse-distribution eğitim" (%60/40 benign resample) — *bir dağılım hilesi olarak* | Hedef dağılım yok. **Ama §2'ye bak: teknik olarak hâlâ faydalı olabilir, gerekçesi değişti** |
| Prior-shift / Saerens düzeltmesi | Kaydırılacak bir hedef prior yok |
| Anonim yarışma verisine köprü kurma (şema eşleme) | Teslim edilecek yarışma yok → köprü gereksiz |
| "Danışman F1'i ile bizim F1'imiz kıyaslanamaz" tartışması | İki rejim yok artık, tek rejim var |

### ✅ HÂLÂ GEÇERLİ VE ARTIK DAHA ÖNEMLİ

| Ne | Neden daha önemli oldu |
|---|---|
| **Floor-F1 referansı** (`2·prev/(1+prev)`) | Artık "final dağılım düşük skoru açıklıyor" mazereti YOK. Mutlak F1 doğrudan yargılanacak → floor'u geçmek zorunlu |
| **Sızıntı disiplini** (split önce, fit sonra) | 63k'da 777 sütun ve etiket `clinvar__sig`'den türetiliyor → sızıntı riski eski veriden çok daha yüksek |
| **Threshold tuning** (default 0.5 kullanma) | Değişmedi; sadece tek modda: eğitim/validasyon dağılımında F1-max (veya MCC-max) |
| **Train-test gap raporlama** | Değişmedi |
| **M5 ≳ M3 missing stratejisi** (NB48) | Doğrulanacak hipotez olarak taşınır |
| **Meta = Logistic Regression** (GBM meta değil) | Doğrulanacak hipotez olarak taşınır |
| **Heterojen base korelasyonu <0.85** (NB37/38) | Doğrulanacak hipotez olarak taşınır |
| **Constant + duplicate sütun temizliği** | Şema-agnostik, direkt taşınır |

### 🆕 YENİ REJİMİN DEĞERLENDİRME PROTOKOLÜ

**Bundan sonra tek protokol budur. Her notebook bunu kullanacak, yoksa kıyaslanamaz.**

1. **Split:** Stratified hold-out `TEST_SIZE=0.2`, `SEED=42`. Test seti **bir kez** ayrılır ve
   Adım 5'e kadar **açılmaz**.
2. **Model seçimi:** Train içinde **5-fold stratified CV** (mean ± std). Tüm ablasyon
   kararları CV'den verilir, test'ten değil.
3. **Metrikler (hepsi raporlanır):** F1(patho), MCC, AUC-PR, AUC-ROC, precision, recall,
   balanced accuracy, confusion matrix mutlak sayıları.
   **Birincil karar metriği: F1(patho) — ama daima MCC ile birlikte oku.**
4. **Floor-F1 zorunlu:** `2·prev/(1+prev)`, **kullanılan fold'un kendi prevalansından**.
   63k missense doğal prevalansı ≈ 0.373 → **floor ≈ 0.543**. Floor'u geçmeyen model değersizdir.
   Ayrıca bir de **majority-class accuracy** baseline'ı raporla.
5. **Threshold:** `metrics.optimize_threshold()` ile **train/CV üzerinde** F1-max seç
   (0.10–0.90 grid). MCC-max alternatifini de raporla. **Test'te threshold arama YAPMA.**
6. **Train metrikleri de raporlanır** (overfit gap = train_F1 − cv_F1).
7. **⚠️ Yeni ve zorunlu: gen-holdout kontrolü.** `GroupKFold(groups=base__hugo)` ile ikinci bir
   değerlendirme yap. 63k'da bazı genler tek yönlü (TTN 327/336 benign, NF1 258/273 patho) —
   rastgele split modelin **geni ezberlemesine** izin verir ve F1'i şişirir.
   **Rastgele-split F1 ile gen-holdout F1 arasındaki fark, modelin gerçek genelleme gücüdür.**
   Bu farkı her tabloda göster.

---

## 1. Yeni Veri: Ne Elimizde Var

`data/63k_genis/full_cravat_v3_63k.csv` — bu oturumda ölçüldü, doğrulanmış:

| Özellik | Değer |
|---|---|
| Satır | **63.463** |
| Sütun | **777** — ham OpenCRAVAT çıktısı, **açık isimli** (`cadd__score`, `revel__score`, `clinvar__sig`, ...) |
| Şema | **LEGACY** — `src/columns.py` evreni ile uyumlu, `src/columns_real.py` (AL/EK/CAT) ile DEĞİL |
| Etiket kaynağı | `clinvar__sig` |
| Etiket kalitesi | `clinvar__rev_stat`: %92.7 "multiple submitters, no conflicts", %7.2 "reviewed by expert panel" → **çok yüksek** |
| Varyant tipi | MIS 60.977 (%96,1), INT 680, SYN 632, MLO 566, CSS 208, diğer ~200 |
| Etiket dağılımı (tümü) | Benign 15.993 + Benign/Likely benign 13.883 + Likely benign 10.009 = **39.924** (%62,9)<br>Pathogenic 6.229 + Path/Likely path 11.746 + Likely path 5.468 = **23.528** (%37,1) |
| **Missense-only** | benign **38.248** / patho **22.722** → **prevalans 0.373**, **floor-F1 0.543** |
| Belirsiz | ~120 satır (`Conflicting`, `\|other`, `\|drug response`, NaN) → dışlanacak |
| Dosya boyutu | 824 MB → **parquet'e çevrilecek** |

### Eski veriyle kıyas — neden bu bir yükseltme

| | Eski (yarışma, anonim) | Yeni (63k, legacy) |
|---|---|---|
| n (MASTER) | 2.931 | **60.970** (missense) — ~21× |
| Feature | 351 ham (anonim, normalize) | 777 ham (**açık isimli, yorumlanabilir**) |
| Ortalama eksiklik | %55 | Adım 1'de ölçülecek (beklenti: çok daha düşük) |
| Prevalans | 0.73 (patho-ağır) | **0.373** (benign-ağır, dengeye yakın) |
| Floor-F1 | 0.846 (çok yüksek → marj yok) | **0.543** (düşük → **gerçek marj var**) |
| Yorumlanabilirlik | Yok (AL_137 nedir bilinmiyor) | **Tam** (SHAP anlamlı, biyolojik yorum mümkün) |

**En büyük kazanç, n değil floor'un düşmesi.** Eski panellerde floor 0.85–0.91 idi; "hep
pathogenic de" diyen trivial baseline neredeyse modeli yeniyordu — bu yüzden 25+ notebook
boyunca gerçek ilerleme ölçülemedi. 63k'da floor 0.543; modelin gösterecek yeri var.

### ⚠️ Yeni verinin tuzakları

1. **`clinvar__*` = hedef sızıntısı.** Etiketi `clinvar__sig`'den türetiyoruz. `clinvar__id`,
   `rev_stat`, `sig_conf`, `disease_names`, `hgvs`, `variant_type`, `allele_id`, `dbsnp_id`,
   **ve `clinvar__af_go_esp` / `af_exac` / `af_tgp`** — hepsi ClinVar kaydından gelir → **DROP**.
   `clinvar_acmg__*` de drop.
2. **Meta-predictor sızıntısı.** `src/columns.py::LEAKY_META_PREDICTOR_SCORES` — bazı skorlar
   (ör. ClinVar üzerinde eğitilmiş meta-predictor'lar) etiketi dolaylı taşır. Bu listeyi uygula
   ve **with/without ablasyonu yap** (Adım 3, A6).
3. **Gen-bazlı ezberleme.** TTN 336 satır → 327 benign; NF1 273 → 258 patho; PTEN 175 → 169 patho.
   Model geni tanıyıp etiketi tahmin edebilir. → **gen-holdout zorunlu** (§0.7).
4. **`base__hugo` feature olarak kullanılmamalı** (doğrudan gen ezberi). Gen-türevli özet
   istatistikler de (gen bazlı patojenite oranı) **sızıntıdır** — kullanma.
5. **Bellek.** 824 MB / 777 sütun → parquet + kategorik dtype şart.

---

## 2. Eski Bulgular Nasıl Korunur (üç katman)

**Kullanıcının açık talimatı:** *"eski bulgularımız ve yeni verisetimizi birleştirmeliyiz"*,
*"eski bulguları bırakıp bambaşka deneylere yönelme"*.

| Katman | İçerik | 63k'ya taşınma biçimi |
|---|---|---|
| **A — Metodolojik disiplin** | Split-önce-fit-sonra, constant/duplicate temizliği, threshold tuning, floor-F1 referansı, train-test gap, tek-eksen ablasyon ilkesi (NB44/NB48), hiperparametre sadakati | **Aynen korunur.** Sadece %80/20 bootstrap → 5-fold CV + gen-holdout ile değiştirildi. |
| **B — Modelleme bulguları** | Aşağıdaki H1–H8 | **Hipotez olarak taşınır, ablasyonla yeniden test edilir.** Sıfırdan arama yapma — bu liste arama uzayını ~10× daraltır. Asıl tasarruf budur. |
| **C — Sayısal sonuçlar** | MASTER 0.638, KANSER 0.730, PAH 0.582, CFTR 0.863 | **Arşivlenir, kıyaslanmaz.** Farklı veri + farklı protokol. `progress.md`'de "eski evren (arşiv)" başlığı altında kalır. |

### Taşınacak hipotezler (H1–H8) — Adım 3'te test edilecek

| # | Hipotez | Kaynak | 63k'da beklenti (yazılı tahmin, sonra doğrula/çürüt) |
|---|---|---|---|
| **H1** | BalancedBagging en güçlü tek model ailesi | NB21/NB32/NB36 | **Zayıflayabilir.** BalBag'in gücü sınıf dengesizliğini yönetmekti; 63k zaten dengeye yakın (0.373). Düz LGBM/CatBoost geçebilir. |
| **H2** | M5 (flag + >%50 NaN drop + ≤%50 medyan) ≳ M3 | NB48 | **Muhtemelen taşınır.** NB48'in gerekçesi boyut şişmesiydi, veri boyutundan bağımsız. |
| **H3** | Gerçek resample > class_weight | NB39 | **Gerekçesi düştü** (hedef dağılım yok) ama **teknik olarak hâlâ test et**: dengeli eğitim genel olarak F1'i iyileştirebilir. |
| **H4** | Stacking meta = Logistic Regression (GBM değil) | NB16/NB37 | **Değişebilir.** GBM meta'nın overfit sebebi küçük meta-matristi (n≈650); 63k'da n≈49k → GBM meta artık çalışabilir. **Mutlaka ikisini de dene.** |
| **H5** | Heterojen base'ler (pairwise kor. <0.85) gerekli | NB38 | Taşınır, doğrula. |
| **H6** | FE (Grantham/BLOSUM62/stopgain) panel-bağımlı, körü körüne açma | NB34/NB36 | 63k'da `ref_amino`/`alt_amino` türetilebilir → **ablasyonla test et** (with_fe vs no_fe). |
| **H7** | Feature selection (xgb-importance top-200) kazandırır | NB41 | **Güçlenebilir** — 777 sütunda kuyruk gürültüsü daha fazla. |
| **H8** | NN/DNN küçük havuzda ağaçların altında kalır | NB39 | **Muhtemelen tersine döner.** n 21× büyüdü; SmallMLP/DNN artık rekabetçi olmalı. Mutlaka dene. |

**Altın kural:** Bir hipotezi "eskiden işe yaramıştı" diye ablasyonsuz kullanma.
Ama sıfırdan da arama — H1–H8 senin başlangıç noktan.

---

## 3. Adım Adım Uygulama

Her adım bir notebook. Sırayla yap. Her adım sonunda `progress.md`'ye bölüm ekle
(**negatif sonuçlar dahil, hatta özellikle**).

---

### ADIM 1 — NB50: Veri Hazırlama, Etiketleme, Sızıntı Temizliği, EDA
`notebooks/50_63k_prep_eda.ipynb`

**Amaç:** 63k'yı temiz, etiketli, sızıntısız, hızlı okunur hale getir.

1. **Parquet dönüşümü.** Chunked oku (`chunksize=5000`), object sütunları `category`'ye çevir,
   yaz: `data/63k_genis/full_63k.parquet`. Bir kez yap; sonraki tüm notebook'lar bunu okur.

2. **Etiket türetimi.** `clinvar__sig`'den:
   - `Benign` / `Likely benign` / `Benign/Likely benign` ile **başlayanlar** → `Label=0`
   - `Pathogenic` / `Likely pathogenic` / `Pathogenic/Likely pathogenic` ile **başlayanlar** → `Label=1`
   - `Conflicting classifications`, `Uncertain`, NaN → **dışla** (`Label_excluded=True` işaretle)
   - `|other`, `|drug response`, `|risk factor` ekli olanlar: ana etiket prefix'ine göre sınıflandır
     ama `Label_qualified=True` işaretle → Adım 3 A5'te dışlama ablasyonu yapılacak.
   - **Güven ağırlığı sütunu** (`label_conf`): `reviewed by expert panel` / `practice guideline` → 1.0;
     `multiple submitters, no conflicts` → 0.9; `single submitter` → 0.6; `conflicting` → dışla.

3. **Varyant tipi filtresi.** Ana set: `base__so == 'MIS'` (n≈60.970).
   Non-missense (n≈2.490) ayrı parquet'e (`nonmis_63k.parquet`) — Adım 5'te ek-veri ablasyonu.

4. **🔴 SIZINTI TEMİZLİĞİ (en kritik adım).** Şunları drop et:
   - **Tüm `clinvar__*`** (AF'ler dahil: `af_go_esp`, `af_exac`, `af_tgp`)
   - **Tüm `clinvar_acmg__*`**
   - `src/columns.py::DIRECT_LEAK_COLS`
   - ID / serbest metin / transkript: `base__uid`, `base__all_mappings`, `base__note_variant`,
     `*__transcript`, `*__transcript_id`, `*__all`, `*__link`, `tagsampler__*`, `cedar__*_id`,
     `*__uniprot_id`, `*__p_vid`, `*__protein_variant`, `*__hgvs`
   - `base__hugo` → **feature değil, group anahtarı** olarak sakla (gen-holdout için)
   - `base__chrom`/`base__pos` → feature değil, split anahtarı olarak sakla
   - `LEAKY_META_PREDICTOR_SCORES` → **ayrı bir listede tut, drop etme henüz** (A6 ablasyonu)
   - **Temizlik sonrası kalan sütun sayısını raporla** (beklenti ~450–550)

5. **Constant + duplicate temizliği.** `columns_real.get_constant_cols()` ve
   `get_duplicate_col_pairs()` şema-agnostiktir, direkt çağır. Kaç sütun düştüğünü raporla.

6. **Tip ayrıştırma.** Kalan sütunları sınıflandır ve `src/columns_63k.py`'a yaz:
   `NUMERIC_COLS`, `CATEGORICAL_COLS`, `BINARY_COLS`, `PRED_STRING_COLS` (`*__pred`,
   `*__class` gibi — bunlar kategorik, sayısal değil), `AA_COLS` (`base__achange`'den türetilecek).

7. **EDA (zorunlu içerik):**
   - Sınıf dengesi (tümü + missense-only) + **floor-F1 = 0.543 açıkça yazılı**
   - **Eksiklik profili:** sütun bazlı NaN%, `Label` ile korelasyon.
     **Sor ve cevapla: 63k'da eksiklik MNAR mı, MCAR mı?** (Eski veride MNAR idi ve en büyük
     tuzaktı — burada durumun aynı olup olmadığını ölç. NB43'ün φ-korelasyon yöntemini kullan.)
   - **Gen bazlı tablo** (top-50): n, benign, patho, prevalans → gen-ezberi riskini görselleştir
   - Sütun bazlı tek değişkenli AUC (hangi annotator tek başına ne kadar güçlü) — top-30
   - Yarışma verisiyle karşılaştırma tablosu (n, prevalans, eksiklik, floor)

**Çıktılar:** `data/63k_genis/{full_63k,missense_63k,nonmis_63k}.parquet`,
`src/columns_63k.py`, `results/v31_63k_prep/`, `reports/nb50_63k_eda_report.pdf`.

**Bitiş kriteri:** Temiz parquet + `src/columns_63k.py` + EDA raporu hazır; sızıntı sütunları
listesi belgelenmiş; eksikliğin MNAR/MCAR yanıtı yazılmış.

---

### ADIM 2 — NB51: Sızıntı Denetimi & Sağlamlık Kontrolü
`notebooks/51_63k_leakage_audit.ipynb`

> **Bu adım atlanamaz.** 777 sütunlu bir veride etiketi `clinvar__sig`'den türetmek, sızıntıyı
> neredeyse davet eder. Eski projenin en pahalı hatası (`v1_leaked` klasörü) buydu.

1. **Hızlı sinyal testi.** Temizlenmiş feature seti ile basit bir LGBM eğit (default
   hiperparametre, 5-fold CV). **Kırmızı bayrak: CV F1 > 0.97 veya AUC > 0.995.**
   Eğer öyleyse feature importance top-30'a bak, sızan sütunu bul, Adım 1.4'e dön.
2. **Tek-sütun AUC taraması.** Her feature için tek başına AUC hesapla.
   **AUC > 0.95 olan her sütunu elle incele** — meşru güçlü predictor (REVEL, AlphaMissense,
   CADD gibi) mi, yoksa sızıntı mı?
3. **Meta-predictor ablasyonu (A6'nın ön hazırlığı).** `LEAKY_META_PREDICTOR_SCORES` ile ve
   olmadan CV F1 farkını ölç. Fark >0.05 ise bu sütunlar "modelin işini yapıyor" demektir —
   dahil edip etmeme **açıkça belgelenmiş bir karar** olmalı, sessiz varsayılan değil.
4. **Gen-ezberi testi.** `GroupKFold(groups=base__hugo)` vs `StratifiedKFold` F1 farkı.
   **Fark >0.10 ise model ciddi biçimde gen ezberliyor** — Adım 3'te bunu azaltacak
   önlemler (gen-bazlı feature'ları drop, regularizasyon) test edilmeli.
5. **Duplicate satır kontrolü.** `base__chrom+pos+ref+alt` üzerinden tam kopya var mı?
   Varsa drop; sayısını raporla.

**Çıktı:** `results/v32_63k_audit/`, `reports/nb51_leakage_audit.pdf`.
**Bitiş kriteri:** "Bu veri seti temizdir çünkü ..." diye yazılabilen, kanıtlı bir rapor.

---

### ADIM 3 — NB52: Baseline & Hipotez Ablasyonları (H1–H8)
`notebooks/52_63k_baseline_hypotheses.ipynb`

**Planın en yüksek getirili adımı.** Eski bulguların hangisinin taşındığını burada belirle.

**Protokol:** §0'daki yeni protokol. Test seti **açılmaz**; tüm kararlar 5-fold CV'den.
Her tabloda: `cv_f1 ± std`, `cv_mcc`, `cv_aucpr`, `train_f1`, `gap`, **`floor_f1`**,
`genegroup_f1` (gen-holdout).

**Ablasyon sırası — TEK EKSEN, ardışık.** Her adımda kazananı sabitle, sonrakine geç.
Kombinatoryal patlamaya girme (toplam ~35–40 fit hedefle).

| # | Eksen | Kollar | Test edilen hipotez |
|---|---|---|---|
| **A1** | Model ailesi | LGBM / XGBoost / CatBoost / RandomForest / BalancedBagging / SmallMLP(focal γ=2, ES) | **H1, H8** |
| **A2** | Missing stratejisi | M1 / M3 / M5 / native_nan | **H2** |
| **A3** | Sınıf dengeleme | yok / class_weight / scale_pos_weight / gerçek balanced resample | **H3** |
| **A4** | Feature seti | tümü / xgb-importance top-200 / top-100 | **H7** |
| **A5** | Etiket kalitesi | tüm etiketler / `Label_qualified` dışlanmış / `label_conf` sample_weight'li / yalnız expert-panel | **yeni eksen (63k'ya özgü)** |
| **A6** | Meta-predictor skorları | dahil / hariç | Adım 2.3'ün resmileştirilmesi |
| **A7** | Feature engineering | no_fe / with_fe (Grantham, BLOSUM62, Δhydropathy, Δvolume, Δcharge — `base__achange`'den) | **H6** |

**Not A7 için:** `src/features.py` legacy evren için yazılmış ve 63k **legacy şemadadır** —
yani `prepare_data_v3_no_insil()` içindeki FE adımları burada **kullanılabilir**. Önce sütun
uyumunu doğrula (777 sütun ⊃ eski 119 sütun mu?), uyuyorsa yeniden yazma, **yeniden kullan**.

**Çıktı:** `results/v33_63k_baseline/`, `reports/nb52_baseline_report.pdf`,
**`progress.md`'ye "H1–H8 taşınma tablosu"**: her hipotez için ✅ taşındı / ❌ çürüdü / ⚠️ kısmen + delta.

**Bitiş kriteri:** 63k champion reçetesi (tek model) + H1–H8 karnesi.

---

### ADIM 4 — NB53: Optimizasyon, Ensemble, Kalibrasyon
`notebooks/53_63k_optimization.ipynb`

Adım 3 net bir champion verdikten sonra.

1. **Optuna.** Champion reçeteye `TRIALS_TREE=100` (ağaç) / `TRIALS_NN=50` (NN) —
   `config.py`'dan al, kendi değer atama. Objective: **CV mean F1** (test'e dokunma).
2. **Stacking.**
   - Base'ler: A1'in en iyi 4–5'i, **pairwise korelasyon matrisi raporlanmış** (<0.85 hedef — **H5**).
   - Meta: **LR (L2, class_weight=balanced) VE GBM — ikisini de dene** (**H4**; 63k'da meta-matris
     n≈49k, GBM'in eski overfit gerekçesi düşmüş olabilir).
   - OOF üretimi: `src/stacking.py` mevcut, yeniden kullan.
3. **Kalibrasyon.** Platt / Isotonic / Venn-Abers. **ECE'yi önce-sonra raporla.**
   (Prior-shift YOK — hedef prior kalmadı.)
4. **Threshold finalizasyonu.** CV üzerinde F1-max ve MCC-max; ikisini de raporla, birini seç
   ve **seçimin gerekçesini yaz**.
5. **Ek veri ablasyonu.** Non-missense (n≈2.490) eğitime eklemek CV F1'i artırıyor mu? Ölç.

**Çıktı:** `results/v34_63k_optimization/`, `reports/nb53_optimization_report.pdf`,
`models/v31_63k/`.

---

### ADIM 5 — NB54: Final Değerlendirme & Yorumlama
`notebooks/54_63k_final.ipynb`

**Test seti ilk ve son kez burada açılır.**

1. **Final metrikler:** hold-out test üzerinde tam metrik seti + floor + confusion matrix.
   CV tahmini ile test sonucu arasındaki farkı raporla (protokol dürüstlüğü kanıtı).
2. **Gen-holdout final skoru.** Rastgele-split ile farkı → gerçek genelleme gücü.
3. **SHAP analizi.** 63k'nın büyük avantajı: **sütunlar açık isimli, SHAP yorumlanabilir.**
   Top-30 feature + biyolojik yorum. (Eski anonim veride bu imkânsızdı.)
   Beklenti: REVEL / AlphaMissense / CADD / konservasyon (phyloP, GERP) tepede olmalı —
   değilse şüphelen.
4. **Hata analizi.** FP ve FN'leri gen/varyant tipine göre grupla. Hangi genlerde model kötü?
5. **Model kaydı:** `models/v31_63k/` + `exported_models/` konvansiyonu; threshold, feature
   listesi ve imputer'ı modelle birlikte serialize et.
6. **`progress.md` + `to-do.md` güncelle.**

---

## 4. Uyulacak Kurallar (ihlal = deney geçersiz)

1. `SEED=42`, **`from config import SEED`** — kendi değer atama.
2. Path'ler `config.PROJECT_ROOT`'a göre **absolute**.
3. **Split ÖNCE, fit SONRA.** Imputer/scaler/encoder yalnız train'de `fit`.
4. **Her tabloda `floor_f1` sütunu**, ilgili fold'un kendi prevalansından hesaplanmış.
5. **Test seti Adım 5'e kadar açılmaz.** Ablasyon kararları CV'den.
6. **Her tabloda gen-holdout F1 sütunu.**
7. Train metrikleri + overfit gap her zaman raporlanır.
8. **63k skorları eski yarışma skorlarıyla asla aynı sütunda kıyaslanmaz** (farklı veri + protokol).
9. **Tek-eksen ablasyon ilkesi** (NB44/NB48 dersi): bir ablasyonda yalnız bir şey değişir; referans
   fit fonksiyonu **kelimesi kelimesine kopyalanır**, sonra tek değişken değiştirilir.
10. Notebook cell başlıkları: `# Cell N: Açıklama`.
11. Çıktılar `results/<deney>/`, `reports/<deney>_report.pdf`, `models/<versiyon>/`.
    **Kök dizine dosya yazma.**
12. Her adım sonrası `progress.md` güncellenir — **negatif sonuçlar dahil.**

---

## 5. Risk Kaydı

| Risk | Belirti | Aksiyon |
|---|---|---|
| **ClinVar sızıntısı** | CV F1 > 0.97, AUC > 0.995, importance'ta `clinvar__*` veya AF | Adım 2 zaten bunu yakalar; Adım 1.4'e dön |
| **Gen ezberleme** | GroupKFold F1, StratifiedKFold F1'den >0.10 düşük | `base__hugo` ve gen-türevli her şeyi drop; regularizasyonu artır; gen-holdout'u birincil metrik yap |
| **Meta-predictor "işi yapıyor"** | A6'da dahil/hariç farkı >0.05 | Karar ver ve **belgelendir** — sessiz varsayılan bırakma |
| **Bellek / süre** | 824 MB, 777 sütun, 63k satır | Parquet + kategorik dtype + `usecols`. Optuna'yı Adım 4'e kadar açma |
| **Eski bulguları körü körüne taşıma** | "BalBag kullandım çünkü eskiden iyiydi" | Adım 3 H1–H8 karnesi zorunlu |
| **Eski bulguları körü körüne atma** | Sıfırdan grid search | H1–H8 başlangıç noktan; arama uzayını daraltır |
| **Etiket gürültüsü** | `Likely benign`/`Likely pathogenic` ClinVar'da daha zayıf kanıtlı | A5 ablasyonu bunu ölçer |
| **Non-missense kontaminasyonu** | Ana sette INT/SYN karışmış | Adım 1.3 filtresini doğrula |

---

## 6. Özet — Tek Paragraf

63k verisi legacy OpenCRAVAT şemasında (777 açık isimli sütun), 63.463 satır, %96 missense,
%63 benign, çok yüksek kaliteli ClinVar etiketli. Kullanıcının kararıyla **final test dağılımı
varsayımı ortadan kalktı**: %80/20 bootstrap protokolü, prior-shift ve anonim-veriye köprü
kurma tamamen düşüyor; yerine standart stratified hold-out + 5-fold CV + **gen-holdout**
geliyor. Eski çalışma üç katmanda korunuyor: **metodolojik disiplin aynen taşınır**,
**modelleme bulguları H1–H8 hipotez listesi olarak yeniden test edilir** (arama uzayını 10×
daraltır — asıl tasarruf budur), **sayısal sonuçlar arşivlenir ama kıyaslanmaz**. En büyük
kazanç n'in 21× artması değil, floor-F1'in 0.846'dan **0.543'e düşmesi** — eski panellerde
trivial baseline modeli neredeyse yeniyordu ve gerçek ilerleme ölçülemiyordu; 63k'da modelin
gösterecek yeri var. İkinci büyük kazanç: sütunlar açık isimli olduğu için **SHAP artık
biyolojik olarak yorumlanabilir**. En büyük risk ise etiketin `clinvar__sig`'den türetilmesi
nedeniyle ClinVar sızıntısı ve tek yönlü genlerin (TTN, NF1, PTEN) ezberlenmesi — bu yüzden
Adım 2 (sızıntı denetimi) ve gen-holdout kontrolü atlanamaz.
