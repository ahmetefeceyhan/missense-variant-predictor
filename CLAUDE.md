# CLAUDE.md

Bu dosya, projeyle çalışan bir AI asistanı (Claude) için pusula niteliğindedir. Genel dosya yapısını veya `git log` çıktısını tekrarlamaz — bunlar zaten gözle görülür. Burada yalnızca **kod okuyarak hemen anlaşılamayan**, kolayca yanlış yapılabilecek kararlar ve sözleşmeler bulunur.

## Proje Bağlamı

**TEKNOFEST Sağlıkta Yapay Zeka Yarışması** — Genetik varyant patojenite tahmini (binary classification: 0 = benign, 1 = pathogenic).

Proje **iki ayrı veri evreni** içerir; karıştırılması projenin tamamen yanlış yere gitmesine neden olur:

| Evren                                 | Yer                                                           | Şema                                                                                | Durum                                                                                                                        |
| ------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Legacy (ClinVar/OpenCRAVAT)** | `data/raw/open_cravat_curation_v3.csv`                      | Açık sütun isimleri (`cadd__score`, `clinvar__sig`, `chasmplus__score` vs.) | NB02–NB11. Çoğunlukla bitmiş ablation / stacking çalışmaları.                                                        |
| **Yarışma (gerçek veri)**    | `data/real_data/YARISMA_TRAIN_{MASTER,CFTR,KANSER,PAH}.csv` | **Anonimleştirilmiş** (`AL_1..334`, `CAT_1..6`, `EK_1..9`, `AA_1/2`) | NB12–NB16 (baseline, ablation, stacking, panel-bazlı transfer, FE+stacking).**Asıl yarışma teslim hattı budur.** |

İkisi farklı feature space, farklı sütun listesi modülü, farklı leakage tuzakları kullanır. **Bir notebook'ta hangi evrenle çalıştığını her zaman önce belirle.**

## Modül Eşleştirmesi (KRİTİK)

| Modül                                  | Hangi Evren İçin                                                                                                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [src/columns.py](src/columns.py)           | **Legacy** — `DIRECT_LEAK_COLS`, `LEAKY_META_PREDICTOR_SCORES`, `RANKSCORE_COLS`, `GNOMAD_COLS`, `OHE_COLS_V3`, `LABEL_ENC_COLS`.                                        |
| [src/columns_real.py](src/columns_real.py) | **Yarışma** — `AL_COLS`, `CAT_COLS`, `EK_COLS`, `AA_COLS`, `AL_HIGH_MISSING_COLS`, `AL_MISSINGNESS_LEAKAGE_RISK`, `SUSPECTED_DUPLICATE_PAIRS`, MoE expert grupları. |
| [src/features.py](src/features.py)         | Legacy.`prepare_data_v3_no_insil()` / `prepare_data_v3()` 8 adımlı FE pipeline. Yarışma verisinde **direkt çağrılmaz** — sütun isimleri uyumsuz.                         |
| [src/models.py](src/models.py)             | Her iki evren.`grid_search_le_*` / `grid_search_catboost` yarışma için, `grid_search_*` legacy için.                                                                              |
| [src/metrics.py](src/metrics.py)           | Her iki evren.`optimize_threshold()` (F1-max, 0.10–0.90 grid) + `compute_all_metrics()` (9 metrik).                                                                                    |

Yarışma tarafında çalışırken `src/columns.py`'ı **import etme** — yanlış sütun isimleri arar ve sessizce yanlış davranır.

## Yarışma Verisi: Bilinmesi Şart Olan Detaylar

Bu detayların kaynağı [reports/DataAnalytics.pdf](reports/DataAnalytics.pdf) ve doğrulanmış EDA. Ezberlenmeli:

### Veri büyüklükleri ve dengesizlik

- MASTER: 2931 satır (pos=2149 / neg=782, **~%73 pozitif**)
- CFTR: 111 satır (pos=90 / neg=21) — alt-set, **tek başına model eğitilemez**
- KANSER: 388 satır (pos=268 / neg=120)
- PAH: 372 satır (pos=310 / neg=62)
- Üç hastalık alt-seti birbiriyle **hiç** Variant_ID kesişmez; her birinin ~%65–70'i MASTER'da da bulunur.

### ⚠️ CFTR DEĞERLENDİRME GÜVENİ YOK (n=21 benign — çalışmaya başlamadan oku)

- CFTR'nin **toplam benign sayısı 21**, %50/%50 split'te test'te ~11 benign kalır. %80/20 bootstrap değerlendirmede (NB15/NB16'nın standart final-realistic protokolü) bu, **F1 güven aralığını [0.00–1.00]'e** açar (std≈0.27). Yani **CFTR'de iki model arasındaki F1 farkı neredeyse her zaman gürültüdür** — "model A, B'yi geçti" demek istatistiksel olarak anlamsız.
- **Pratik:** CFTR'de tek bir %50/50 hold-out F1'ine veya tek bootstrap ortalamasına GÜVENME. (1) **Tekrarlı/iç içe split** (çok sayıda farklı seed) ile F1 dağılımı + CI raporla; nokta tahmin değil dağılım kıyasla. (2) Karar verirken precision (CFTR'de genelde 1.0) ve confusion matrix mutlak sayılarına bak. (3) CFTR benign'ini artırmak için MASTER'ın CFTR-benzeri benign'leri veya diğer panellerin benign'leriyle destekleme seçeneğini değerlendir — ama bu farklı-gen leakage riski taşır, dikkatli yap.
- Doğrulanmış (NB16): CFTR'de base DNN %80/20 F1=0.852, stacking=0.682 görünür ama **ikisi de CI=[0.00–1.00]** — bu fark gerçek değil. CFTR %50/50'de catboost/dnn F1≈0.875, precision=1.0.

### ⚠️ FINAL TEST DAĞILIMI (EN KRİTİK BİLGİ — train'in TERSİ)

- **Yarışmanın final test seti her panel için ~%80 benign / ~%20 pathogenic olacak.** Bu, eğitim verimizin (panellerde %73–83 pathogenic) **tam tersidir**. Metrik yine **pathogenic (Label=1) F1**.
- **Sonuç (zincirleme etki):** Eğitim çoğunluğu pathogenic olduğu için modeller doğal olarak düşük eşik seçip "şüphede pathogenic de" eğilimi gösterir (NB15'te seçilen eşikler 0.10–0.48). Final test'te benign çoğunluk olunca bu eşik **FP patlamasına** yol açar; pathogenic F1'in paydası küçük TP + büyük FP'den oluşur → **precision çöker, F1 düşer**. NB15'in 0.91–0.93 F1'leri final dağılımda **çok daha düşük** çıkma riski taşır.
- **Zorunlu pratik:** (1) Threshold'u **final dağılımı (%80/20) taklit eden bir validasyon/test havuzunda** seç, train'in dengesinde değil. (2) Modeli benign'i öğrenmeye zorla: `class_weight`/`scale_pos_weight`, undersampling, focal loss. (3) Değerlendirmeyi **hem %50/50 hem %80/20** dağılımda yan yana raporla; sadece %50/50 görmek yanıltır. (4) Salt pathogenic-F1-max threshold yerine **MCC-max veya precision-tabanlı** eşik de dene — yüksek recall / düşük precision tuzağına düşme.

### Eksiklik (en büyük modelleme zorluğu)

- MASTER ortalama eksiklik: %55; 165 sütun >%50 eksik; en kötü sütun (`CAT_6`) %97.7.
- Eksiklik **rastgele değil**: Label=1 satırlarda eksiklik oranı %59.9, Label=0'da %41.2. **Eksiklik bilgi taşıyor.**
- KANSER panelinde `AL_16..AL_25`'in eksiklik oranı Label=0'da ~%29, Label=1'de ~%86 — bu **dağılım kayması leakage riski**. `AL_MISSINGNESS_LEAKAGE_RISK` adıyla [columns_real.py](src/columns_real.py)'da ayrılmıştır. Yarışma test setinde aynı mekanizmanın olup olmadığı bilinmediği için bu sütunlarla ve `is_missing_*` bayraklarıyla iki ayrı model bulundurulmalı.
- Yüksek eksik blokları: `AL_1..AL_6` (%91+), `AL_27..AL_38` (%91+), `CAT_6` (%97.7) — `AL_HIGH_MISSING_COLS` olarak tanımlı.

### Sütun anatomisi (anonimleştirilmiş)

- `AL_1..AL_334`: 334 sayısal sütunun **103 tanesi binary 0/1 flag**, kalan 231 tanesi popülasyon frekansı/skor (geniş dinamik aralık, log dönüşüm gerekir).
- `CAT_1`: kaynak popülasyon kohortu (~30 değer: `gnomADe_EAS`, `gnomADg_NFE`, ...).
- `CAT_2`: AllofUs popülasyonu (7 değer).
- `CAT_3`, `CAT_4`, `CAT_5`: genotip (`C/C`, `T/T`, `G/G`, `A/A`, `./.`). **MASTER'da CAT_3 == CAT_5 birebir aynı** — biri drop edilmeli (bkz. `SUSPECTED_DUPLICATE_PAIRS`). **Ancak özdeşlik CAT_3/CAT_5 ile SINIRLI DEĞİL** (bkz. Sözleşmeler madde 4): MASTER'da büyük AL blokları (örn. `AL_101..AL_182` arası, `AL_191..AL_292` blokları, `AL_299..AL_332`) birbirinin birebir aynısıdır — çoğu sıfır-varyanslı (constant), bir kısmı (örn. `AL_200==AL_204`) varyanslı ama özdeş. `SUSPECTED_DUPLICATE_PAIRS` yalnızca CAT_3/CAT_5'i listeler; gerisini runtime'da `get_duplicate_col_pairs()` yakalar.
- `CAT_6`: genomik bölge bayrağı (`lcr`/`segdup`/`decoy&segdup`), %97.7 boş.
- `EK_1..EK_9`: yardımcı patojenite/konservasyon skorları; bazıları negatif değer alır (frekans değil, skor).
- `AA_1`, `AA_2`: tek harf amino asit (referans → değişen). Grantham/BLOSUM62 türetilebilir.
- `Variant_ID`: **feature olarak kullanılmaz**. MASTER içinde benzersizdir (2931/2931), ama **global primary key DEĞİLDİR**: aynı `Variant_ID` MASTER ile alt-panellerde görünse de bunlar **çoğunlukla farklı varyantlardır** (örtüşen çiftlerde ortalama 41–150 özellik hücresi farklı). Yani aynı ID'li satırları otomatik olarak "duplicate" sayıp droplama — birebir aynı oldukları doğrulanmadan asla (bkz. Sözleşmeler madde 2).

### Sözleşmeler ve tuzaklar

1. **Imputation TRAIN ayrılmadan ASLA yapılmaz.** PDF'de kırmızı uyarıyla belirtilmiş. Önce stratified split, sonra `fit` yalnızca train üzerinde, `transform` test/panel'e. Aynı kural OHE/LabelEncoder/scaler için de geçerli.
2. **Panel sızıntısı uyarısı (DÜZELTİLDİ)**: Aynı `Variant_ID` MASTER ve alt-panellerde bulunabilir, **ama bunlar genelde farklı varyanttır** — ID üzerinden kör filtreleme yanlıştır. Doğrulanmış EDA (NB15): örtüşen ID'lerin **yalnızca tüm özellik+label'ı birebir aynı** olanları gerçek kopyadır → **MASTER ile birebir aynı satır sayısı: KANSER=3, PAH=3, CFTR=0**. Cross-panel değerlendirmede sadece bu birebir-aynı satırları droparak panel test'ini temizle; salt ID eşleşmesiyle droplama. (Salt ID bazlı GroupKFold gereğinden fazla satır atar.)
3. **Sabit sütunlar (`nunique<=1`) train'den sonra tespit edilip drop**. Yardımcı: `columns_real.get_constant_cols(df)`. Doğrulanmış: MASTER'da **57 sabit sütun** (çoğu yüksek-eksik AL blokları).
4. **Birebir özdeş sütun çiftleri drop (KAPSAM DÜZELTİLDİ)**. Yardımcı: `columns_real.get_duplicate_col_pairs(df)`. Bu CAT_3/CAT_5'ten **çok daha geniştir**: MASTER'da **583 özdeş çift → 58 sütun drop adayı** (`CAT_5` + 61 AL sütununun çoğu). Constant + duplicate birlikte **toplam 63 sütun drop → ham 351 feature'dan 288 kalır** (NB15'te doğrulandı). `SUSPECTED_DUPLICATE_PAIRS` sabiti yalnızca CAT_3/CAT_5'i içerir; AL bloklarını **el ile listeleme**, runtime tespitine güven.
5. **Sınıf dengesizliği + dağılım tersliği** — accuracy yanıltır. Birincil metrikler **F1, MCC, AUC-PR**. `metrics.optimize_threshold()` F1-max threshold seçimi yapar; varsayılan 0.5 kullanma. **ÖNEMLİ:** `optimize_threshold()` train dengesinde (pathogenic-ağırlıklı) F1-max seçer — ama final test %80 benign olduğu için bu eşik orada **fazla agresif (çok düşük) kalır**. Final-realistic değerlendirmede eşiği %80/20 havuzda seç ve MCC-max alternatifini de değerlendir (bkz. FINAL TEST DAĞILIMI bölümü).
6. **Eksiklik göstergesi (`is_missing_*`)** dahil/hariç iki model yan yana eğitilip karşılaştırılmalı (PDF'in "en kritik kontrol" dediği şey budur). **Bulgu (NB14 panel-transfer):** en iyi strateji **M3** = `is_missing_*` flag (>%50 NaN sütunlar için) + tüm sayısal sütunlara medyan imputation (orijinal sütun korunur); panel ortalama F1=0.9210 ile flagsiz M1 (0.9165) ve drop-ağırlıklı M5 (0.9188) önünde. Yeni panel-bazlı deneylerde varsayılan olarak M3 kullan.
7. Yarışma verisinde sayısal sütunlara **gürültü eklenmediği gözlemlendi**, ancak risk hâlâ eksiklik örüntüsünün etiketle korelasyonudur.

## Doğrulanmış Değerlendirme Protokolü (yeni notebook'lar bunu kullanmalı — yoksa NB15/NB16 ile kıyaslanamaz)

- **Threshold 3 modda seç ve hepsini raporla:** `f1_raw` (eski, train dengesinde — sadece kıyas için), `f1_8020` ve `mcc_8020` (final %80/20 dağılıma yeniden-örneklenmiş train havuzunda; **benign-aware**). Bulgu: f1_8020/mcc_8020 ≈ eşit, ikisi de f1_raw'dan ~2 puan iyi → final için benign-aware kullan.
- **Test'i iki dağılımda raporla:** %50/50 (mevcut panel) + **bootstrap %80/20** (benign sabit + patho downsample, N=50, %95 CI). **Birincil metrik = %80/20 bootstrap pathogenic-F1.** Sadece %50/50 görmek yanıltır (NB15 v1 tuzağı).
- **Train metriklerini de raporla** (overfit kontrolü; train-test F1 farkı). NB15 v1 ve NB16 v1'de unutuldu — tekrarlama.

### ⭐ HER YENİ DENEYDE KESİN BULUNMASI GEREKENLER (danışman EDA'sından sonra zorunlu — 2026-06-27)

Bunlar artık opsiyonel değil; bu öğelerden biri eksikse deney NB39+ ile kıyaslanamaz ve raporu yanıltıcıdır:

1. **FLOOR F1 referansı.** Her panelin "hep pathogenic tahmin et" baseline'ı = `2·prev/(1+prev)`, **kullanılan test fold'unda** hesaplanır. Mutlak F1'i **tek başına raporlama — daima floor ile yan yana** göster. Danışman floor'ları (train-dağılımı): MASTER=0.846, KANSER=0.818, PAH=**0.905**, CFTR=0.905. ⚠️ %80/20 bootstrap havuzunda floor FARKLIDIR (prev=0.20 → floor≈0.333); **kendi test havuzunun prev'inden hesapla**, danışmanınkini kopyalama. Floor'u geçmeyen model = trivial baseline, değersiz.
2. **İki metriğin AYRIMI net olmalı.** Eğer bir notebook train-dağılımı (%50/50 veya stratified hold-out) F1 raporluyorsa, **bunu "tavan/sinyal" diye etiketle**; final-realistic kararı SADECE %80/20 bootstrap F1'den ver. İki sayıyı asla aynı sütunda karşılaştırma (danışmanın 0.89'u ile bizim 0.64'ümüz aynı şey değil).
3. **Reverse-pool varyantı denenmeli.** Yeni bir panel/model deneyinde en az bir **%60 benign/%40 patho gerçek-resample** kolu bulundur (NB32/NB39 kazanan reçete). Sadece orijinal dağılımda eğitip bırakma.
4. **FE kullanılacaksa ablasyonla (with_fe vs no_fe) kanıtla.** Varsayılan kapalı; MASTER'da zararlı olduğu kanıtlı. Körü körüne açma.

### Danışman Feature-Depth EDA'sından doğrulanan ek bulgular (`feature_depth_analysis/`)

- **Feature seçimi gerçek bir kaldıraç (test edilecek):** `xgb_importance` top-200 > all-features (MASTER train-dağılımında 0.894 vs 0.879). Kuyruk feature'lar gürültü ekler. Biz 288–440 feature kullanıyoruz → NB41'de %80/20 bootstrap'te test edilecek.
- **Value-transform ağaçlar için önemsiz:** raw/significand/sig4figs + mean/median neredeyse özdeş (ağaçlar rank'e böler). "AL frekans log dönüşümü gerekir" varsayımı **ağaç modeller için gereksiz** — M3 median yeterli. (Sadece NN/lineer modeller için log düşün.)
- **PAH'ta gerçek sinyal yok (bağımsız teyit):** danışman best=0.925 ≈ floor 0.905 → model trivial baseline'a eşit. Bizim "Chatterji tavanı / PAH platosu" bulgumuzun ikinci kanıtı. **PAH'ta yeni model arama — kapalı.**
- **Cross-panel pooling danışmanın görmediği kaldıraç:** danışman "her panel kendi verisiyle" çalıştı → CFTR'yi kapattı. Bizim en iyi CFTR/KANSER sonuçlarımız pooling'le geldi. Pooling'i koru.

### Doğrulanmış Modelleme Reçetesi (NB16→NB39 ÇALIŞTIRILDI — kanıtlanmış sonuçlar)

- **Çekirdek reçete:** **M3 missing** + **benign-aware threshold** + **SmallMLP (focal loss γ=2 + early stopping + yüksek dropout/weight_decay, BatchNorm YOK)** + **OOF stacking (meta = Logistic Regression, GBM meta DEĞİL — GBM overfit ediyor)** + **hafif FE (Grantham/BLOSUM62/stopgain, gömülü tablolar)**.
- **⭐ EN GÜÇLÜ KALDIRAÇ — reversed-distribution eğitim (NB32 KANSER, NB39 MASTER):** Modeli final test dağılımına yaklaştıran bir **gerçek resample** alt-kümesiyle eğitmek, post-hoc düzeltmelerden ve ağırlıklamadan daha iyi. **%60 benign / %40 patho oranı optimal** (S1_6040). Daha agresif %80/20 (S2) patho çeşitliliğini öldürür → overfit. **Gerçek resample > class_weight/scale_pos_weight** (NB39: S1_6040/balbag 0.638 > S4_weighted/lgbm 0.592). Yeni panel deneylerinde **mutlaka reverse-pool varyantı dene.**
- **Threshold yakınsama (NB39):** Eğitim dağılımı test dağılımına yaklaştıkça `|thr_raw − thr_8020|` 0.27–0.48'den 0.00–0.17'ye düşer. Reverse-distribution eğitim, threshold seçimini de kendiliğinden sağlamlaştırır.
- **BalancedBagging her panelde en güçlü tek teknik** (PAH NB21, KANSER, MASTER NB36 BalBag_XGB=0.603). Ters dağılımı en iyi yöneten model ailesi. RF de benign-ağırlıklı test'e dayanıklı (NB38: Single_rf 0.609 > Single_lgbm 0.551).
- **NN/DNN'i çıkarma — iyileştir:** v1'de overfit (train-test ~0.15); SmallMLP+focal+ES ile ~0.065'e düştü, **KANSER'i finetune-DNN kazandı** (NB16). AMA küçük reverse-pool'larda (n<1100) ağaç modeller NN'i geçer (NB39 NN hariç tutuldu).
- **FE körü körüne ekleme — ablasyonla ölç (panel-bağımlı, ZORUNLU):** with_fe vs no_fe ayrı koş. **MASTER'da FE ZARARLI** (NB36/NB37: −FE > full); **CFTR'de NB16'da +0.069 ama NB34'te net negatif**; KANSER +0.015. **Karar: FE'yi varsayılan açma — her panelde ayrı ablasyonla kanıtla.**
- **Stacking panel-bağımlı:** Küçük dengeli panelde (KANSER) gerçek kazanç; MASTER'da base modeller çok benzer (pairwise korelasyon ~0.90) → meta çeşitlilik bulamaz, BalBag_XGB tek başına stacking'i geçer (NB37). Heterojen base (LGBM+RF+BalBag+MLP+DNN, kor. 0.75) MASTER'da +0.014 verdi (NB38) ama reverse-distribution'ın (+0.021) altında.
- **Calibrate-then-shift (Saerens prior-shift) MASTER'da ÇALIŞMADI** (NB38: ECE 0.024→0.396 bozdu). CFTR'de işe yaradı (NB17 S6). Panel-bağımlı; körü körüne uygulama.
- **AUC-PR her panelde yüksek (0.90–0.92):** Sıralama gücü güçlü; **sorun threshold/dağılım uyumunda, sıralamada değil.** Bu yüzden çaba threshold + dağılım hizalamasına gitmeli.

### ⭐ ŞU ANKİ EN İYİ SONUÇLAR (%80/20 bootstrap pathogenic-F1) — GÜNCEL HEDEFLER

| Panel | En iyi model | Boot-F1 | Kaynak | Floor F1 (danışman) | Durum |
|---|---|---|---|---|---|
| **CFTR** | S0c_COMBINED (MASTER+KANSER+PAH → LGBM + prior-shift) | **0.863** (LOO-MCC=0.644, prec=1.0) | NB20 | 0.905 | **KESİNLEŞTİ** |
| **KANSER** | P9_REVERSE_6040_catboost_with_fe | **0.730** | NB32 | 0.818 | İyi; reverse-pool + FE kazandı |
| **MASTER** | S1_6040/balbag (reversed-distribution) | **0.638** (CI=[0.519–0.716], MCC=0.542) | NB39 | 0.846 | Aktif; stacking + Optuna açık |
| **PAH** | P4_COMBINED_BalBag | **0.582** (MCC=0.529) | NB21 | 0.905 | **KESİNLEŞTİ** (Chatterji tavanı) |

**Not:** Bu skorlar danışmanın train-dağılımı F1'leriyle (MASTER 0.89, KANSER 0.90, PAH 0.91, CFTR 0.95) **kıyaslanamaz** — danışmanınki test'i train dağılımında ölçüyor (sinyal tavanı), bizimki ters dağılımda (final-realistic). Aynı "%80/20" iki farklı şey (bkz. progress.md "Danışman EDA İncelemesi").

## Çalışma Kuralları (bu projede uygulanan)

- **SEED=42** her yerde sabit. Yeni bir notebook'ta da `from config import SEED` kullan; kendin değer atama.
- Her zaman için **eski bulguların** üzerine yenilerini ekle. Eski bulguları bırakıp bambaşka deneylere kullanıcı açık bir şekilde istememişse asla yönelme.
- **Optuna budget'ları**: ağaç modeller için `TRIALS_TREE=100`, NN için `TRIALS_NN=50`. Bunlar `config.py`'da.
- **Hold-out oranı**: `TEST_SIZE=0.2`, stratified.
- **Path konvansiyonu**: Tüm path'ler `config.PROJECT_ROOT`'a göre absolute. Notebook'lar farklı dizinden çalıştırılabildiği için relative path **kullanma**.
- **Notebook cell başlıkları**: `# Cell N: Açıklama` formatında (NB09–NB13 ile uyumlu kalsın).
- **Otomatik PDF rapor**: fpdf2 ile `AblationReport(FPDF)` veya benzeri sınıf üzerinden. NB10 ve NB13 referans pattern.
- **Çıktı yerleri**: `results/<deney_adi>/`, `reports/<deney_adi>_report.pdf`, `models/<versiyon>/`. Düz kök dizine dosya yazma.
- **`progress.md`**: her tamamlanan deneyden sonra burası güncellenir — proje ilerlemesinin tek doğruluk kaynağı.

## Dış Bağımlılıklar (Önemli Olanlar)

- `lightgbm>=4.0`, `xgboost>=2.0`, `catboost` (requirements.txt'te eksik olabilir — kullanmadan önce kontrol et).
- `torch>=2.0` + `torch-directml` — Windows GPU back-end. macOS'ta directml yüklenmez; sessizce CPU'ya düşmesini bekle.
- `optuna>=3.0`, `shap>=0.42`, `fpdf2>=2.7`.

## "Tipik Yeni Görev" Örüntüsü

1. **Önce evreni belirle** (legacy mi, yarışma mı). Yanlış evrende çalışmak en pahalı hata.
2. Uygun `columns*.py` modülünü import et.
3. Veri yükle, **önce split**, sonra fit/transform.
4. `compute_all_metrics()` + `optimize_threshold()` ile değerlendir, F1 odaklı.
5. Çıktıyı `results/<ad>/`'a, raporu `reports/<ad>_report.pdf`'e yaz.
6. `progress.md`'ye satır ekle.

## Sıkça Yapılan Hatalar (gözlemlenmiş)

- Yarışma CSV'sini legacy FE pipeline'ına vermek → KeyError (örn. `cadd__score`).
- Tüm veri üzerinde `SimpleImputer.fit_transform` yapıp sonra split etmek → silent leakage.
- Default 0.5 threshold ile F1 raporlamak → threshold-tuned modelden 2-5 puan düşük çıkar.
- CFTR alt-setinde tek başına model eğitmeye çalışmak (n=111, neg=21 yetersiz) — MASTER ile birleştirilmeli veya pretrain+fine-tune yapılmalı.
- Özdeş/sabit sütunları temizlememek → `CAT_3==CAT_5` yalnızca görünen ucu; aslında 583 özdeş çift + 57 sabit sütun var (toplam 63 drop). `get_duplicate_col_pairs()` + `get_constant_cols()` çağırmadan eğitime girmek boş kapasite + collinearity demektir.
- Aynı `Variant_ID`'li satırları "duplicate" sanıp droplama → çoğu farklı varyant; yalnızca tüm özellik+label birebir aynı olanlar (KANSER=3, PAH=3, CFTR=0) gerçek kopyadır.
- **Sadece %50/50 (train dengesi) test F1'ine bakıp sevinmek** → NB15 v1'de 0.91–0.93 görüldü, %80/20 final-realistic'te 0.49–0.85'e düştü. Final dağılım %80 benign; %50/50 F1 final performansı YANSITMAZ. Her zaman %80/20 bootstrap da raporla.
- **CFTR'de model A > model B demek** → n=21 benign, bootstrap CI [0.00–1.00]. CFTR fark karşılaştırmaları gürültü; tekrarlı split + precision/CM'ye bak (bkz. CFTR DEĞERLENDİRME GÜVENİ YOK).
- Stacking meta'sı olarak GBM/ağaç kullanmak → küçük meta-feature matrisinde overfit eder (NB16: stack_gbm train F1 0.84–0.87, test düşük). **Meta = Logistic Regression** (L2, class_weight=balanced).
- **Danışmanın feature-depth F1'lerini (MASTER 0.89 vb.) bizim %80/20 bootstrap F1'lerimizle (0.64) kıyaslamak** → danışman test'i train dağılımında ölçüyor (sinyal tavanı), biz ters dağılımda (final-realistic). Aynı "%80/20" iki farklı şey; fark model kalitesi değil dağılım. **Sunum/teslim raporuna danışmanın mutlak sayılarını koyma** — "tavan X / final-realistic Y" ikisini birlikte sun.
- **Floor F1 hesaplamadan mutlak F1'e sevinmek** → PAH'ta 0.92 görünür ama floor 0.905, yani model trivial. Her zaman floor ile yan yana raporla.
- **class_weight/scale_pos_weight ile yetinmek** → ağırlıklama karar sınırını yeterince kaydırmıyor (NB39: weighted 0.592 < reverse-resample 0.638). Gerçek %60/40 resample dene.
