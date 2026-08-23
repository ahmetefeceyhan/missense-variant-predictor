# Panel Bazlı Model Geliştirme — Eksiklik Analizi ve Literatür Taraması

**Tarih:** 2026-07-24
**Kapsam:** TEKNOFEST genetik varyant patojenite tahmini — 4 panel (MASTER, KANSER, PAH, CFTR)
**Amaç:** Her panelde mevcut performans boşluğunun hangi eksenlerden kaynaklandığını tespit etmek + bu boşlukları kapatabilecek teknikleri güncel literatürle desteklemek.
**Not:** Bu belge yalnızca **araştırma ve planlama** içindir. Kodlama içermez — deneyler ayrı bir oturumda yapılacaktır. Her öneri, projede **denenmemiş** bir kola veya denenmiş bir kolun **açık bir varyantına** karşılık gelir.

---

## 0. Yönetici Özeti (Executive Summary)

Projenin dört panelinde de tekrarlayan **tek bir yapısal darboğaz** var: **precision**. Her panelde recall yüksek (~0.93–1.0) ve AUC-PR yüksek (0.90–0.92), ama F1'i sınırlayan şey precision (0.43–0.58 bandı). Bu, modellerin **sıralama gücünün (ranking) yeterli, ancak karar eşiği/dağılım hizalamasının yetersiz** olduğu anlamına gelir. Dolayısıyla asıl kaldıraç yeni ve daha güçlü bir sınıflandırıcı bulmak **değil**; eşik seçimi, olasılık kalibrasyonu ve eğitim dağılımı hizalamasıdır.

**Panel bazlı en kritik boşluklar (öncelik sırası):**

| Panel | Mevcut en iyi (Boot-F1) | En kritik boşluk | Önerilen ilk hamle |
|---|---|---|---|
| **PAH** | 0.582 (NB21) | NB40 Bayes-ceiling çelişkisi (gap +0.1125) — gerçek mi artefakt mı? | Bayes-ceiling'i sağlam yöntemlerle yeniden hesapla (FeeBee/GHP), muhtemelen artefakt olduğunu doğrula |
| **KANSER** | 0.730 (NB32) | Reverse-pool + FE kazandı ama NN ve seçili-flag ekseni denenmedi | NB44 (G_LOW flag) + reverse-pool'a TabPFN-2.5 ekle |
| **MASTER** | 0.638 (NB39) | Reverse-distribution çalıştı ama stacking/Optuna/kalibrasyon kombinasyonu eksik | Reverse-pool base'leriyle heterojen stacking + Venn-Abers kalibrasyon |
| **CFTR** | 0.863 (NB20) | n=21 benign → tüm değerlendirme kırılgan | Değerlendirme çerçevesini sağlamlaştır (nested/repeated LOO), model değil |

**En güçlü genel kaldıraç (tüm panellerde denenmeli):** **Reverse-distribution (gerçek %60/40 resample) eğitiminin** henüz uygulanmadığı yerlere yayılması. Bu teknik KANSER (0.69→0.73) ve MASTER'da (0.62→0.638) bağımsız olarak kazandı, ama **PAH'a, CFTR'ye ve NN/TabPFN model ailelerine hiç uygulanmadı.** Literatür bunu "per-gene class balancing during training" olarak bağımsız doğruluyor (bkz. §3.4).

**En az yatırım yapılmış eksen (bakir alan):** **Missing-handling.** Projenin en temel önişleme kararı (M3 imputation+flag stratejisi) NB14'te **bir kez, çürütülmüş bir %50/50 rejiminde** seçildi ve 25 notebook boyunca sorgulanmadan donduruldu. Verinin %55'i boşken bu, en belirleyici karara en az yatırımın yapıldığı yer. İki EDA belgesi eksikliğin doğası konusunda çelişiyor (MASTER=informatif/MNAR, PAH=yapay maske/MCAR) — yani **tek-tip M3 en az bir panelde yanlış** ama bu hiç modele bağlanmadı. Ayrıntılı analiz ve deney tasarımları için bkz. **§3-BIS** (bu turda eklenen derinlemesine bölüm).

---

## 1. Metodoloji ve Değerlendirme Sözleşmesi Hatırlatması

Bu rapordaki tüm öneriler projenin **doğrulanmış değerlendirme protokolüne** bağlıdır; aksi belirtilmedikçe her yeni deney şunları içermelidir:

- **Birincil metrik:** Bootstrap %80/20 pathogenic-F1 (benign sabit + patho downsample, N=50, %95 CI). Bu, final yarışma dağılımını (%80 benign) taklit eder.
- **Floor F1 referansı:** `2·prev/(1+prev)`, kullanılan test fold'unun kendi prevalansından hesaplanır. Mutlak F1 asla tek başına raporlanmaz.
- **Danışman F1'leriyle kıyaslama yapılmaz:** Danışmanın train-dağılımı F1'leri (MASTER 0.89, PAH 0.91 vb.) "sinyal tavanı" ölçer; bizim ters-dağılım bootstrap F1'lerimiz (0.64, 0.58) "final-realistic" ölçer. **İkisi aynı ölçek değildir.**

⚠️ **Bu rapor bir ölçüm sözleşmesini de düzeltmeyi öneriyor** (bkz. §2.PAH ve §3.7): NB40'ın Bayes-ceiling'i, floor-F1 ile **aynı belirsizlik disiplinine tabi tutulmadığı** için PAH'ta yanıltıcı bir "boşluk" üretmiş olabilir.

---

## 2. Panel × Başlık Eksiklik Matrisi

Her panel için, kullanıcının istediği başlıklar (+ eklediklerim) altında **[Denenen]**, **[Boşluk]** ve **[Öneri]** ayrımıyla.

### Ortak Başlıklar (tüm panelleri etkiler)

#### A. Model Seçimi
- **[Denenen]** LightGBM, XGBoost, CatBoost, RandomForest, SVM, SmallMLP, DeepMLP, BalancedBagging. TabPFN v8 (NB24'te API/lisans + >100 feature sınırı nedeniyle **atlandı**).
- **[Boşluk]** **TabPFN-2.5 hiç denenmedi.** 2025'in sonunda çıkan sürüm 50.000 örnek + **2.000 feature**'a kadar destekliyor — projenin 288–440 feature'ı artık limit içinde (eski >100 sorunu ortadan kalktı). TabPFN v2 küçük tabular veride (n≈3k) tuned GBDT'leri geçiyor; PAH (n=369), KANSER (n=388), CFTR (n=111) tam "tatlı nokta".
- **[Öneri]** TabPFN-2.5'i her panelde COMBINED havuzuyla dene. In-context learning parametre güncellemesi gerektirmez → küçük panellerde overfit riski düşük. Fine-tuning varyantı (yandex-research/tabpfn-finetuning reçetesi) ikinci adım.

#### B. Ensemble / Eğitim Metodları
- **[Denenen]** OOF stacking (meta=LR ve GBM), heterojen base stacking (kor. 0.75), BalancedBagging, soft ensemble, alpha-blend. Bulgu: LR meta > GBM meta; BalBag her panelde en güçlü tek teknik; stacking panel-bağımlı (KANSER'de +, MASTER'da nötr).
- **[Boşluk 1]** **Base çeşitliliği hâlâ sınırlı.** MASTER'da base'ler ~0.90 korele (NB37); NB38'de heterojen base 0.75'e indi ama TabPFN gibi **farklı inductive bias**'lı bir base hiç eklenmedi. Gerçek çeşitlilik = düşük korelasyon = stacking kazancı.
- **[Boşluk 2]** **Cost-sensitive / class-balanced loss** GBDT'de sistematik denenmedi (sadece NN'de focal loss var). 2025 literatürü class-balanced focal loss'un GBDT'de imbalanced tabular'da kazandığını gösteriyor.
- **[Öneri]** (1) TabPFN base'i stacking'e ekle → korelasyonu düşür. (2) Reverse-distribution base'leriyle (NB39 S1_6040) stacking — kazanan senaryonun base'leriyle stacking hiç yapılmadı (progress.md'de "planlandı ama çalıştırılmadı").

#### C. Missing Flag Handling
- **[Denenen]** M1–M5 stratejileri (NB14). M3 (>%50 NaN için flag + median impute) varsayılan. NB43: KANSER'de G_LOW grubu (≤%50 null) mean|φ|=0.260 güçlü sinyal taşıyor ama M3 flag'lemiyor.
- **[Boşluk]** **NB44 hiç yazılmadı.** G_LOW içindeki |φ|>0.15 sütunlara seçici flag eklemek KANSER'de kayıp sinyali yakalayabilir. Referans liste hazır: `results/v26_missing_flag_correlation/KANSER_LOW_flag_stats.csv`.
- **[Öneri]** NB44'ü yaz (KANSER'de M3 vs M3+φ-seçici). Literatür "informative missingness" özelliklerinin klinik ML'de performansı artırdığını doğruluyor (bkz. §3.5) — eksiklik bir "veri problemi" değil, sinyal.

#### D. Eğitim Verisi Havuzu (Pooling)
- **[Denenen]** COMBINED (cross-panel pooling) her panelde kazandı: CFTR (+0.17 boot), KANSER (E1=0.714), PAH (+0.034 MCC). Naif `concat`. Birebir-aynı satır drop (KANSER=3, PAH=3, CFTR=0).
- **[Boşluk]** Pooling hâlâ **naif concat** — hiçbir domain-adaptation, örnek-ağırlıklı transfer veya multi-task mimari denenmedi. Havuz kompozisyonu manuel (T0–T5 denendi ama otomatik/ağırlıklı değil).
- **[Öneri]** (1) **Örnek-ağırlıklı pooling:** MASTER satırlarını hedef panele benzerliğe göre ağırlıklandır (importance weighting). (2) **Multi-task / shared-backbone** NN: bir gövde + panel-başı hafif başlık (bkz. §3.6). Danışmanın körlüğü tam da burada (cross-panel pooling görmedi) — bu bizim avantajımız, derinleştirilmeli.

#### E. Eğitim Verisi Dağılımı (Reverse-Distribution) ⭐
- **[Denenen]** KANSER P9_REVERSE_6040 (%60 benign/%40 patho) → 0.73; MASTER S1_6040 → 0.638. **%60/40 optimal** (%80/20 patho çeşitliliğini öldürüyor). Gerçek resample > class_weight.
- **[Boşluk]** **Sadece 2 panelde ve sadece ağaç modellerinde denendi.** PAH'a, CFTR'ye hiç uygulanmadı. NN/TabPFN ile hiç birleştirilmedi. Reverse-pool + FE + stacking üçlüsü sadece KANSER'de.
- **[Öneri]** Reverse-distribution'ı **evrensel bir ön-adım** olarak tüm panellere + tüm model ailelerine yay. Bu, raporun **en güçlü tek önerisi.** Literatür desteği güçlü (bkz. §3.4).

#### F. Feature Selection
- **[Denenen]** NB41: xgb_importance top-N (k=10..all). Bulgu: **top-N seçimi ters-dağılım protokolünde kazanç sağlamadı** (danışmanın train-dağılımı bulgusu genellemedi). KANSER'de k=100 nominal en iyi ama şampiyondan düşük.
- **[Boşluk]** Feature selection **reverse-distribution/FE ile birlikte hiç denenmedi** (NB41 saf FS etkisini izole etti). Ayrıca importance-tabanlı değil, **stability selection** veya **mutual-info** tabanlı seçim denenmedi.
- **[Öneri]** Düşük öncelik — NB41 net negatif sonuç verdi. Yalnızca reverse-pool + FS kombinasyonu bir kez denenebilir (izole değil, üretim reçetesinde).

#### G. Feature Engineering
- **[Denenen]** Grantham, BLOSUM62, conservation ürünleri, FCS, EK-combo. Bulgu panel-bağımlı: **MASTER'da FE ZARARLI** (EK omurga güçlü), **KANSER'de +0.020** (reverse-pool ile sinerjik), **CFTR/PAH'ta nötr/negatif**.
- **[Boşluk]** FE hâlâ **gömülü tablo (Grantham/BLOSUM) tabanlı** — modern **pre-trained embedding** (AlphaMissense, ESM, DNA-LM) enjeksiyonu hiç yapılmadı (NB18 deobfuscation'a bağlı, ertelendi).
- **[Öneri]** Düşük-orta öncelik. FE'yi körü körüne açma; ama KANSER'de reverse-pool+FE zaten kazanıyor → o kombinasyonu koru. Embedding enjeksiyonu uzun vadeli, riskli (bkz. §3.3 sınırları).

#### H. Olasılık Kalibrasyonu (eklediğim başlık) ⭐
- **[Denenen]** Isotonic, Platt, Saerens prior-shift, calibrate-then-shift (Alexandari). Bulgu: **panel-bağımlı** — CFTR'de prior-shift +, PAH/MASTER'da zararlı (LGBM olasılıkları kalibre değil). Platt kalibrasyon PAH'ta prior-shift'i fixledi ama ham MCC'yi düşürdü.
- **[Boşluk]** **Venn-Abers kalibrasyon hiç denenmedi.** Literatür Venn-Abers'in Platt ve Isotonic'i her metrikte geçtiğini gösteriyor (bkz. §3.2). Ayrıca **focal-loss GBDT'nin kalibre olmaması** için 2025 closed-form düzeltmesi denenmedi.
- **[Öneri]** Venn-Abers'i tüm panellerde prior-shift öncesi kalibratör olarak dene → "kalibrasyonsuz Saerens bozuyor" sorununun kök çözümü olabilir.

#### I. Eşik Seçimi / Karar Kuralı (eklediğim başlık)
- **[Denenen]** F1-max (raw), F1-max (%80/20), MCC-max, GHOST, robust N=50 bootstrap threshold. Bulgu: benign-aware (%80/20) eşik ~2 puan iyi; weighted_f1_max zararlı.
- **[Boşluk]** **İki-taraflı eşik (belirsizlik bölgesi)** denenmedi. Klinik varyant literatürü (REVEL/BayesDel) tek binary cut yerine gene-level 2-sided threshold öneriyor (bkz. §3.1).
- **[Öneri]** Panel-bazlı iki-taraflı eşik (yüksek-precision "kesin patho" + düşük-eşik "şüpheli") — özellikle final %80 benign dağılımında FP kontrolü için.

#### J. Değerlendirme Güvenilirliği (eklediğim başlık) ⭐
- **[Denenen]** Bootstrap %80/20 CI, LOO-CV (CFTR), multi-seed. NB40: k-NN Bayes-ceiling.
- **[Boşluk]** **NB40'ın Bayes-ceiling'i tek k-NN yöntemiyle, tek konfigürasyonla hesaplandı** ve floor-F1 ile aynı belirsizlik disiplinine tabi tutulmadı. Küçük panellerde (PAH n=62 benign, CFTR n=21) bu tahmin literatüre göre kırılgan (bkz. §3.7).
- **[Öneri]** Bayes-ceiling'i **çoklu yöntemle** (k-NN + GHP divergence + KDE) ve **çoklu seed/feature-altküme** ile yeniden hesapla; CI'ları floor ile aynı ölçekte raporla.

---

### 2.PAH — Öncelik: EN YÜKSEK (çelişki çözümü)

| Başlık | Durum | Kritik not |
|---|---|---|
| Model seçimi | BalBag şampiyon; TabPFN denenmedi | TabPFN-2.5 tek kalan yeni inductive bias |
| Ensemble | COMBINED+BalBag optimal; stacking düşük diversite (OOF kor. 0.53) | 3 GBDT ailesi çok benzer |
| Missing flag | G_LOW zayıf (φ=0.083) → M3 korunur | değişiklik gereksiz |
| Havuz | COMBINED kazandı; CFTR benign > KANSER benign (şaşırtıcı) | tek-gen paneli feature uzayı zenginleştiriyor |
| Dağılım | **reverse-distribution HİÇ denenmedi** | açık boşluk |
| Feature selection | NB41 floor'un altına düşürdü (kırılgan) | kapalı |
| FE | NB35: zararlı (−0.148 MCC) | kapalı |
| **Bayes-ceiling** | **NB40 gap=+0.1125 (DEVAM) — floor-analiziyle ÇELİŞİYOR** | **asıl araştırma konusu** |

**PAH'ın merkezi sorusu:** Model 0.582'de, floor 0.905'te (trivial "hep patho de"), Bayes-ceiling 0.6945'te. Üç değer üç farklı şey ölçüyor. NB40 "0.69'a kadar yer var" diyor; NB21–35 + danışman floor-analizi "sinyal yok, plato gerçek" diyor. **Literatür (§3.7) NB40'ın tahmininin PAH'ın 62 benign'i + yüksek feature sayısıyla istatistiksel olarak güvenilmez olduğunu güçlü şekilde ima ediyor** — yani çelişki muhtemelen k-NN artefaktı, gerçek bir kazanç fırsatı değil. Yine de **doğrulama şart**: reverse-distribution + TabPFN denenmeden "kapandı" denemez.

---

### 2.KANSER — Öncelik: YÜKSEK

En olgun panel (şampiyon 0.730, Bayes-ceiling'e gap=−0.0037 ile oturmuş). Kalan boşluklar:
- **NB44 (G_LOW seçici flag)** — tek doğrulanmamış eksen. |φ|>0.15 sütunlara flag.
- **TabPFN-2.5 + reverse-pool** — KANSER (n=388) TabPFN tatlı noktası; reverse-pool zaten kazandı, ikisinin birleşimi denenmedi.
- **EK_2 anomalisi:** Danışman EDA'sında KANSER, bir `EK_` feature'ının en iyi `AL_`'larla yarıştığı **tek panel** (EK_2, #2). KANSER'in EK yapısı özgün — hedefli FE burada anlamlı olabilir.

---

### 2.MASTER — Öncelik: YÜKSEK

En büyük panel (n=2931), en güvenilir ölçüm. Şampiyon 0.638 (reverse-dist), Bayes-ceiling gap=+0.0161 (dar ama pozitif).
- **Stacking + reverse-pool base'leri** — planlandı, hiç çalıştırılmadı. NB39 S1_6040 base'leriyle heterojen stacking.
- **Optuna tuning** — hiç yapılmadı. Ama danışman notu: "grid search MASTER'da baseline'ı geçmedi" → beklenti düşük tutulmalı.
- **Venn-Abers kalibrasyon** — precision darboğazı (0.55) MASTER'da en belirgin; kalibrasyon + iki-taraflı eşik en çok burada işe yarayabilir.

---

### 2.CFTR — Öncelik: ORTA (değerlendirme, model değil)

n=21 benign → tüm karşılaştırmalar CI=[0.00–1.00]. Şampiyon 0.863 ama Bayes-ceiling LOW confidence.
- **Model değiştirme önceliği DÜŞÜK** — asıl iş değerlendirme çerçevesini sağlamlaştırmak.
- Benign artırımı **yapısal çıkmaz** (NB18: missense-benign kıt, tekrar önerme).
- **Öneri:** Repeated/nested LOO-CV ile Bayes-ceiling'in tekrarlanabilirliğini test et. Precision=1.0 + confusion matrix mutlak sayıları F1'den daha güvenilir karar tabanı.

---

## 3. Literatür Sentezi (güncel kaynaklar, projeye bağlanmış)

### 3.1 Klinik Varyant Meta-Predictor'ları ve Eşik Stratejisi
REVEL (13 aracın ensemble'ı) ve BayesDel, diğer meta-predictor'ları (CADD, MetaSVM, Eigen) klinik varyant sınıflandırmasında geçiyor — **daha yüksek PPV (precision)**, karşılaştırılabilir NPV. **En kritik çıkarım:** Klinik literatür **gene-level, iki-taraflı eşik** öneriyor (tek binary cut değil) — "belirsizlik bölgesi" bırakarak. Bu, projedeki **panel-bazlı eşik seçimini doğruluyor** ve iki-taraflı eşik (§2.I) fikrinin klinik temeli.
> Bağlantı: Projedeki EK skorları (EK_7=REVEL literatürü, feature importance'ta baskın) bu meta-predictor ailesinden. Modelimiz zaten bir "meta-meta-predictor" — bunu bilerek EK skorlarına ağırlık vermek makul.
> Kaynak: [REVEL and BayesDel outperform other in silico meta-predictors (Sci Rep 2019)](https://www.nature.com/articles/s41598-019-49224-8)

### 3.2 Olasılık Kalibrasyonu — Venn-Abers Kaldıracı
Venn-Abers predictors, Platt scaling ve Isotonic regresyonu **neredeyse tüm metriklerde geçiyor** ve ham RF/GBDT olasılıklarından daha iyi kalibre. Ayrıca 2025 bulgusu: **focal loss ile eğitilen GBDT'nin çıktısı gerçek posterior olasılık DEĞİL** (focal loss proper scoring rule değil) → closed-form düzeltme öneriliyor.
> Bağlantı: Projede "kalibrasyonsuz Saerens prior-shift bozuyor" (NB21/38) tekrar tekrar görüldü. **Venn-Abers, prior-shift öncesi kök çözüm olabilir.** Ayrıca NN'de focal loss kullanılıyor → onun olasılıkları da kalibre değil, bu da benign-aware eşik seçimini etkiliyor olabilir.
> Kaynaklar: [Calibrating with Venn-Abers Predictors (SDM 2019)](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.4) · [Calibrated Risks with Focal Loss GBDT (Electronics 2025)](https://doi.org/10.3390/electronics14091838)

### 3.3 Disease-Specific Pathogenicity — Mimari ve Veri Stratejisi
DNA dil modeli (Nucleotide Transformer) + GNN ile **disease-specific** patojenite tahmini %85.6 balanced accuracy, %90.5 sensitivity. **Projeye doğrudan taşınabilir üç bulgu:**
1. **"Per-gene class balancing during training"** — her gen için benign/patho eşitleme, modelin gen-içi çarpık dağılımı sömürmesini engelliyor. **Bu, projedeki reverse-distribution/undersampling stratejisinin bağımsız literatür doğrulaması** (bkz. §3.4).
2. **Basit metin embedding'i (BioBERT) karmaşık DNA-LM ile eşit performans** — pahalı embedding şart değil; feature'ın rank bilgisi yeterli (danışmanın "value-transform önemsiz" bulgusuyla örtüşüyor).
3. **Küçük gen panellerinde örnek kıtlığı balancing ile kısmen çözülüyor** ama tam çözülmüyor — CFTR'nin yapısal sınırının literatür teyidi.
> Sınır: Embedding enjeksiyonu (AlphaMissense/ESM) projede yalnızca NB18 deobfuscation üzerinden mümkün ve o hat "değerler normalize, benign artırımı çıkmaz" ile kapandı. **Yüksek riskli, düşük öncelikli.**
> Kaynak: [Disease-Specific Prediction with DNA-LM and GNN (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12562010/)

### 3.4 Reverse-Distribution / Undersampling — Teorik Temel
Chatterji et al. (2022, projede zaten kullanılıyor) undersampling'in label-shift altında minimax-optimal olduğunu kanıtladı. 2024–2025 imbalanced tabular literatürü bunu genişletiyor: **gerçek resample > oversampling (SMOTE)** yüksek boyutta/küçük veride; **balanced underbagging** (BalancedBagging'in genel adı) 10 algoritma karşılaştırmasında güçlü. **En önemli:** disease-specific pathogenicity makalesi (§3.3) aynı fikri "per-gene training balancing" olarak bağımsız keşfetti.
> Bağlantı: Projedeki **%60/40 reverse-pool** keşfi (KANSER 0.73, MASTER 0.638) hem teorik (Chatterji) hem uygulamalı (per-gene balancing) literatürle destekleniyor. **Bu, tüm panellere yayılması için yeterli meşruiyet.**
> Kaynaklar: [Balanced Underbagged Ensemble (ISAF 2025)](https://onlinelibrary.wiley.com/doi/10.1002/isaf.70018) · [Class-balanced loss for GBDT (Neurocomputing 2025)](https://dl.acm.org/doi/10.1016/j.neucom.2025.129896) · [CLIMB: Class-imbalance Benchmark (arXiv 2505.17451)](https://arxiv.org/pdf/2505.17451)

### 3.5 Informative Missingness — M3 Stratejisinin Doğrulanması
Klinik ML literatürü: **eksiklik göstergesi (`is_missing_*`) özellikleri hastanın durumu hakkında bilgi taşır ve retrospektif görevlerde performansı artırır.** "Eksiklik informatifse (örn. ICU'da eksik vital), bir erken uyarı sinyali olarak ele al, sadece bir veri problemi olarak değil."
> Bağlantı: Projenin M3 stratejisi (eksiklik-label korelasyonu %59.9 vs %41.2) tam olarak bu prensibi uyguluyor. NB44 (KANSER G_LOW seçici flag) bu literatürle meşru — G_LOW'un φ=0.260 sinyali M3 eşiğinin altında kalıyor, yakalanmalı.
> Uyarı: Aynı literatür MNAR varsayımının kanıtlanmasının zor olduğunu, **eksiklik mekanizmasının test setinde de aynı olması gerektiğini** vurguluyor — bu, projedeki `AL_MISSINGNESS_LEAKAGE_RISK` (KANSER'de eksiklik-label kayması) endişesinin literatür karşılığı. İki model (flag'li/flag'siz) tutma kararı doğru.
> Kaynak: [Missingness Features in ML for Critical Care (PMC 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8701717/) · [Missing Data in EHR — Systematic Review 2010–2024 (PMC 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11615160/)

### 3.6 Cross-Panel Transfer / Multi-Task Learning
Multi-task literatürü: **shared backbone + domain-specific heads**, mixture-of-experts ve model merging küçük veride hesaplama maliyetini düşürürken performansı artırıyor. **Negatif transfer** ve dengesiz dağılımlar için domain-similarity kritik.
> Bağlantı: Projenin COMBINED pooling'i şu an naif concat. **Shared-backbone NN (bir gövde + 4 panel başlığı)** veya **importance-weighted pooling** (MASTER satırlarını hedef panele benzerliğe göre ağırlıklandır) denenmemiş kaldıraçlar. `columns_real.py`'da zaten tanımlı **MoE expert grupları** (EXPERT1–4) bu mimari için hazır iskele.
> Uyarı: Negatif transfer riski gerçek — PAH'ta KANSER benign'i CFTR benign'inden az fayda verdi (NB23). Benzerlik-tabanlı ağırlıklama bu riski azaltabilir.
> Kaynaklar: [Understanding Knowledge Transferability — Survey (arXiv 2507.03175)](https://arxiv.org/pdf/2507.03175) · [Multi-Task for Heterogeneous Block-Wise Missing Data (arXiv 2505.24413)](https://arxiv.org/pdf/2505.24413)

### 3.7 Bayes Error Estimation — PAH Çelişkisinin Çözümü ⭐
"Bayes Error Rate Estimation in Difficult Situations" (2025) tam olarak PAH sorununu adresliyor:
- **k-NN en iyi non-parametrik estimator** AMA **sınıf başına minimum ~1000 örnek** gerekiyor (%5 güven bandı için, basit 2D Gaussian'da). **Feature sayısı arttıkça bu dramatik büyüyor:** 4 feature'da 2500 örnek/sınıf.
- **Estimator'lar gerçek BER'den ayrı bir eğriyi takip ediyor** — örnek eklemekle azalmayan kalıcı bir bias var.
- **Yöntemler dengeli sınıf sayısı varsayıyor;** küçük azınlık sınıfında istatistiksel garantiler daha da bozuluyor.
- **Öneri:** Tek test durumu yerine geniş BER aralığı incele; nokta tahmin yerine güven bandı; percentile-tabanlı %95 bound.
> **PAH'a uygulama:** PAH'ta **62 benign** + **yüksek feature sayısı** (yüzlerce) var. Bu, makalenin "güvenilir k-NN BER için minimum" eşiğinin **çok altında.** Dolayısıyla NB40'ın PAH için verdiği **Bayes-F1=0.6945 ve gap=+0.1125 muhtemelen bir k-NN artefaktı** — gerçek bir kazanç fırsatı değil. Danışmanın floor-analizi (PAH best=0.925 ≈ floor 0.905 → sinyal yok) ve NB21–35'in plato bulgusu daha güvenilir. **FeeBee** çerçevesi (Bayes error estimator'ları gerçek verilerde değerlendiren benchmark) bunu doğrulamak için kullanılabilir.
> **Somut eylem:** NB40'ı yeniden çalıştır — (a) çoklu k, (b) çoklu feature-altküme, (c) GHP divergence + KDE ile çapraz-kontrol, (d) floor ile aynı CI disiplini. Eğer gap CI'ları floor belirsizliğiyle örtüşüyorsa çelişki çözülür (artefakt), PAH tekrar kapanır.
> Kaynaklar: [Bayes Error Rate Estimation in Difficult Situations (arXiv 2506.03159)](https://arxiv.org/html/2506.03159v3) · [FeeBee: Evaluating Bayes Error Estimators (arXiv 2108.13034)](https://arxiv.org/pdf/2108.13034)

### 3.8 TabPFN-2.5 ve Tabular Foundation Models
TabPFN v2 (2025 başı) küçük tabular veride (ort. n≈3k) tuned GBDT'leri geçti. **TabPFN-2.5** (2025 sonu): 50.000 örnek + **2.000 feature** desteği, TabArena benchmark'ında lider, AutoGluon 1.4'e eşit. TabICLv2 ve RealTabPFN-2.5 alternatifler.
> **Projeye kritik uygunluk:** NB24'te TabPFN v8 **>100 feature sınırı** nedeniyle sorun yaşamıştı. **TabPFN-2.5'te bu sınır 2000'e çıktı** → projenin 288–440 feature'ı artık rahatça sığıyor. Küçük paneller (PAH 369, KANSER 388, CFTR 111) TabPFN'in tam tatlı noktası. In-context learning parametre güncellemesi gerektirmediği için küçük veride overfit riski minimal.
> **Öneri sırası:** (1) TabPFN-2.5 ham, COMBINED havuzuyla, her panelde. (2) Reverse-pool ile birleştir. (3) Stacking base'i olarak ekle (diversite için). (4) Fine-tuning en son.
> Kaynaklar: [TabPFN-2.5 (arXiv 2511.08667)](https://arxiv.org/abs/2511.08667) · [On Finetuning Tabular Foundation Models (arXiv 2506.08982)](https://arxiv.org/html/2506.08982v2) · [Prior-Labs/TabPFN-v2-clf (HuggingFace)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)

### 3.9 Label-Shift Düzeltmesi — Yeni Yöntemler
Saerens EM (projede kullanılıyor) kalibrasyon varsayıyor (modern modellerde bozuk). BBSE (projede D4'te denendi) kalibrasyon-bağımsız ama confusion-matrix tersine bağlı. **2024–2025 yeni yöntem: GS-B³SE (Graph-Smoothed Bayesian BBSE)** — ilişkili etiketler arasında istatistiksel güç ödünç alıp Saerens post-processing'e besliyor, MLLS üzerine +2.3–4.9pp kazanç. **Calibrate-then-shift (Alexandari)** — kalibrasyon EM'den ÖNCE gelmeli (projede doğrulandı).
> Bağlantı: PAH/MASTER'da prior-shift zararlı çıkmasının kökü kalibrasyonsuzluk. **Venn-Abers (§3.2) + calibrate-then-shift zinciri** projede yarım kalan işi tamamlar. Binary problemde GS-B³SE marjinal (2 sınıf), ama Venn-Abers kalibrasyon net kazanç potansiyeli.
> Kaynaklar: [MLLS — Bias-Corrected Calibration (ICML 2020, arXiv 1901.06852)](https://arxiv.org/abs/1901.06852) · [Graph-Smoothed BBSE (arXiv 2505.16251)](https://arxiv.org/pdf/2505.16251) · [Label Shift Bayesian Approach (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Ye_Label_Shift_Estimation_for_Class-Imbalance_Problem_A_Bayesian_Approach_WACV_2024_paper.pdf)

### 3.10 SMOTE'un Sınırları — CFTR Bulgusunun Açıklaması
SMOTE ve varyantları (Borderline, ADASYN) **yüksek boyutta ve küçük veride** güvenilmez; sentetik örnekler azınlık manifoldunu bozabilir. SMOTE-Tomek/ENN gibi hibritler gürültü temizler ama küçük n'de yine kırılgan.
> Bağlantı: Projede **S2_SMOTE-CFTR** (NB17) tam bunu gösterdi — train F1=1.0'a şişti, boot-mean 0.493'e çöktü. **Literatür bu başarısızlığı öngörüyor.** Sonuç: küçük panellerde (CFTR/PAH) sentetik oversampling yerine **gerçek resample (undersample) + pooling** doğru yol. Zaten projenin vardığı sonuç — literatürle mühürlendi.
> Kaynak: [SMOTE variants comparative study (IIS 2025)](https://iacis.org/iis/2025/2_iis_2025_70-85.pdf)

---

## 3-BIS. DERİNLEMESİNE: Missing-Handling — Projenin En Az Yatırım Yapılan Ekseni ⭐

Kullanıcı doğru tespit etti: **eksik-veri işleme (imputation + flag stratejisi) hiçbir zaman kendi başına, final-realistic protokolde, ayrılmış bir deney olarak ele alınmadı.** Bu, projenin 25 notebook'luk emeğinin neredeyse tamamı model/ensemble/dağılım eksenlerine giderken, verinin **%55'inin boş** olduğu bir problemde en belirleyici karara yapılan yatırımın minimal kaldığı anlamına geliyor. Bu bölüm, neden burada gerçek bir fırsat olduğunu ve ne denenebileceğini ayrıntılandırır.

### 3-BIS.1 Dürüst Envanter — Ne Yapıldı, Ne Yapılmadı

**Yapılanlar (dağınık, sistematik değil):**
- **NB14 — M1–M5 taraması:** 5 strateji (flag yok/median, flag yok/drop, flag+median [M3], flag+drop, karma) MASTER'da denendi, M3 kazandı (panel-transfer ort. F1=0.9210). ⚠️ **Ama bu %50/50 dağılımda ölçüldü — yani şimdi "yanılsama" dediğimiz rejimde**, final-realistic %80/20 protokolü henüz yokken.
- **NB43 — φ-korelasyon analizi:** Her panelde eksiklik-label korelasyonu (φ) ölçüldü (KANSER G_LOW=0.260 güçlü, diğerleri zayıf). Bu bir **analiz**, bir modelleme deneyi değil — hiçbir model kararına bağlanmadı.
- **Dağınık dokunuşlar:** NB27 sentinel/KNN/M3 kıyası (şüpheli `weighted_f1_max` threshold'la kirlenmiş), NB36/37 "is_missing nötr" ablasyonu (yalnız MASTER, FI %0.9), NB31 E5 M3minus/M3plus.

**Yapılmayan (asıl boşluk):** **M3, NB14'te bir kez kazandı ve sonra NB15→NB39 boyunca 25 notebook boyunca sabit, sorgulanmamış bir varsayım olarak dondurularak taşındı.** Yani projedeki en temel önişleme kararı, artık **çürütülmüş bir değerlendirme rejiminde** (%50/50) seçilmiş ve bir daha final-realistic protokolde yargılanmamış.

### 3-BIS.2 En Kritik Bulgu — İki EDA Belgesi Eksikliğin Doğası Konusunda ÇELİŞİYOR

Bu, missing-handling'in kalbinde duran ve hiç çözülmemiş bir çelişki:

| Kaynak | Panel | İddia | Mekanizma |
|---|---|---|---|
| CLAUDE.md / NB12 | **MASTER** | "Eksiklik **bilgi taşıyor**": Label=1'de %59.9, Label=0'da %41.2 eksiklik | **MNAR** (informative missingness) → M3 flag gerekli |
| eda.md | **PAH** | "Eksiklik **yapay maske**, sinyal taşımıyor": Sınıf 0 %54.01, Sınıf 1 %54.35 (kusursuz homojen) | **MCAR** (komitenin rastgele dropout'u) → flag gürültü ekler |

**Sonuç:** Eksikliğin informatif olup olmadığı **panele göre değişiyor** — ama M3, dört panele de körlemesine aynı uygulanıyor. Bu, literatürdeki tam MNAR/MCAR ayrımıdır (bkz. §3.5): MASTER'da eksiklik erken-uyarı sinyali, PAH'ta ise saf gürültü. NB43 bu farkı φ ile **ölçtü** ama hiçbir modelleme kararına **bağlamadı.** Tek-tip M3 stratejisinin en az bir panelde (muhtemelen PAH) yanlış olduğu neredeyse kesin.

⚠️ **Metodolojik uyarı:** eda.md yalnızca **PAH** üzerine yazılmış ve "eksiklik yapay maske" iddiasını PAH'ın homojen oranından çıkarıyor. Bu iddianın MASTER'a genellenmesi **yanlış olur** — MASTER'da oranlar heterojen (%59.9 vs %41.2). Yani iki belge çelişmiyor bile olabilir; **farklı paneller farklı mekanizmalara sahip.** Asıl mesele budur: tek strateji tüm panellere uymaz.

### 3-BIS.3 Neden Burada Gerçek Fırsat Var — Üç Gerekçe

1. **M3'ün seçildiği zemin geçersiz.** M3 %50/50'de kazandı; sonra %50/50'nin final performansı yansıtmadığını öğrendik. **M3'ün %80/20 bootstrap'te de en iyi olduğu hiç doğrulanmadı.**
2. **Bu veri setinde missing = temizlik değil, ana sinyal kaynağı.** MASTER'da %55 hücre boş, 165 sütun >%50 eksik. Bu yoğunlukta, imputation/flag stratejisi modelin gördüğü sinyalin büyük kısmını belirler — bir yan-adım değil.
3. **Çelişki çözülmemiş.** İki EDA belgesinin işaret ettiği panel-bağımlılık hiç modele bağlanmadı; NB43 analiz düzeyinde kaldı.

### 3-BIS.4 Somut Denenebilir Deneyler (öncelik sıralı)

1. **M3'ü final-realistic protokolde panel-bazlı yeniden yargıla.** M1–M5'i (+ yeni varyantlar) %80/20 bootstrap'te, her panelde **ayrı** çalıştır. **Hipotez:** MASTER'da flag'li (MNAR) kazanır, PAH'ta flag'siz (MCAR) kazanır → tek evrensel M3 yanlış, panel-başı strateji doğru. Bu, en yüksek getirili ve doğrudan çelişkiyi çözen deney.
2. **φ-seçici flag (NB44'ün genellemesi).** NB43'ün φ'sini eyleme dök: sadece >%50 NaN'a değil, **|φ|>eşik olan her sütuna** flag ekle. KANSER G_LOW=0.260 sinyali M3'ün kaçırdığı yer; referans liste hazır (`results/v26_missing_flag_correlation/KANSER_LOW_flag_stats.csv`).
3. **Native NaN-handling vs imputation.** LightGBM/XGBoost/CatBoost NaN'ı doğal işler (ayrı dal). Şu an median'la doldurup **modelin kendi NaN-mantığını devre dışı bırakıyoruz.** "Impute etme, modele bırak" hiç kıyaslanmadı — ağaçlar için genelde daha iyi ve bu veri yoğunluğunda kritik olabilir.
4. **Missingness'i tek güçlü feature olarak.** Satır-başı eksiklik oranı / blok-başı eksiklik sayısı. NB28'de `missing_count` PAH top-6'da çıkmıştı ama izole test edilmedi. Klinik ML literatürü (§3.5) tam bunu öneriyor.
5. **Reverse-distribution × missing etkileşimi.** Eksiklik Label'la koreleyse (MASTER'da öyle), reverse-resample **eksiklik dağılımını da kaydırıyor.** Bu etkileşim hiç incelenmedi ve gizli bir leakage/kayma riski taşıyor — resample sonrası eksiklik-label korelasyonu train'de yapay şişebilir.

### 3-BIS.5 Dürüst Tavan Uyarısı

Bu eksenin **getirisi panel-bağımlı ve bazı panellerde sınırlı olacak:**
- **MASTER:** NB36/37 "is_missing nötr" buldu (FI %0.9). Büyük sıçrama beklenmemeli — ama panel-başı strateji yine de *net-negatif değil* doğrulaması olarak değerli.
- **KANSER:** φ güçlü (G_LOW=0.260) → **en umut verici panel.** NB44 + deney #1 burada birleşmeli.
- **CFTR:** Küçük panelde her karar oransal daha etkili; ama n=21 benign değerlendirmeyi kırılganlaştırıyor → dikkatli.
- **PAH:** Beklenti *pozitif kazanç değil, gürültü-azaltma.* eda.md'ye göre flag'i kaldırmak (MCAR) PAH'ta net-koruma sağlayabilir — mevcut M3 gereksiz gürültü ekliyor olabilir.

**Özet:** Missing-handling, projedeki **en düşük yatırım / en yüksek belirsizlik** ekseni. Büyük panelde tavan düşük olsa da, (a) çelişkiyi çözmek metodolojik olarak zorunlu, (b) KANSER'de somut kazanç potansiyeli var, (c) tek-tip M3'ün en az bir panelde yanlış olduğu neredeyse kesin. **Kayda değer katkı = tek evrensel M3'ü panel-bazlı, veri-mekanizmasına duyarlı (MNAR/MCAR-aware) bir stratejiyle değiştirmek.**

---

## 4. Önerilen Deney Yol Haritası (öncelik sıralı, kodlama sonraki oturumda)

| # | Deney | Panel(ler) | Beklenen kaldıraç | Risk |
|---|---|---|---|---|
| 1 | **NB40'ı sağlam Bayes-ceiling ile yeniden çalıştır** (çoklu k + GHP + KDE + floor-CI disiplini) | PAH (+tümü) | Çelişkiyi çöz; muhtemelen artefakt teyidi | Düşük |
| 2 | **Reverse-distribution'ı PAH ve CFTR'ye yay** (%60/40 gerçek resample) | PAH, CFTR | En güçlü kanıtlı teknik, 2 panelde denenmedi | Düşük |
| 3 | **TabPFN-2.5'i her panelde dene** (COMBINED + reverse-pool) | Tümü | Yeni inductive bias; küçük panel tatlı noktası | Orta (kurulum) |
| 4 | **Missing-handling'i final protokolde panel-bazlı yeniden yargıla** (M1–M5 + native-NaN, %80/20 bootstrap) | Tümü (özellikle KANSER/PAH) | Çürütülmüş rejimde seçilmiş M3'ü doğrula; MNAR/MCAR çelişkisini çöz — bakir eksen (bkz. §3-BIS) | Düşük |
| 5 | **NB44 (KANSER G_LOW φ-seçici flag) + φ-genellemesi** | KANSER | Doğrulanmamış tek eksen; referans liste hazır; en umut verici missing panel | Düşük |
| 6 | **Venn-Abers kalibrasyon + calibrate-then-shift** | MASTER, PAH | Prior-shift'in kök çözümü; precision iyileştirme | Orta |
| 7 | **Reverse-pool base'leriyle heterojen stacking** (+ TabPFN base) | MASTER | Diversite artışı; planlandı hiç yapılmadı | Orta |
| 8 | **İki-taraflı eşik / precision-öncelikli karar kuralı** | Tümü | FP kontrolü, final %80 benign'de precision | Düşük |
| 9 | **Importance-weighted pooling / shared-backbone MoE** | Tümü | Naif concat'i aşan transfer; MoE iskelesi hazır | Yüksek |

**Not (danışman uyumu):** Bu yol haritası danışmanın train-dağılımı F1'lerini hedef almıyor — hedef her zaman **%80/20 bootstrap pathogenic-F1** ve **floor'u anlamlı geçmek.** Deneyler NB39+ protokolüyle kıyaslanabilir olmalı (floor referansı + f1_raw/f1_8020/mcc_8020 üçlüsü + train-test gap zorunlu).

---

## 5. Kaynakça

**Label Shift / Kalibrasyon:**
- [Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation (Alexandari et al., ICML 2020)](https://arxiv.org/abs/1901.06852)
- [Graph-Smoothed Bayesian Black-Box Shift Estimator (2025)](https://arxiv.org/pdf/2505.16251)
- [Label Shift Estimation for Class-Imbalance: A Bayesian Approach (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Ye_Label_Shift_Estimation_for_Class-Imbalance_Problem_A_Bayesian_Approach_WACV_2024_paper.pdf)
- [Calibrating Probability Estimation Trees using Venn-Abers Predictors (SDM 2019)](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.4)
- [Estimating Calibrated Risks Using Focal Loss and GBDT (Electronics 2025)](https://doi.org/10.3390/electronics14091838)

**Genetik Varyant Patojenite:**
- [Disease-Specific Prediction with DNA Language Models and GNN (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12562010/)
- [REVEL and BayesDel outperform other in silico meta-predictors (Sci Rep 2019)](https://www.nature.com/articles/s41598-019-49224-8)
- [Disease-specific variant pathogenicity improves interpretation in cardiac conditions (Genet Med 2020)](https://www.nature.com/articles/s41436-020-00972-3)
- [Gene-specific ML for BRCA1/BRCA2 missense variants (Sci Rep 2023)](https://www.nature.com/articles/s41598-023-37698-6)
- [Precision in prediction: ML for breast cancer missense variants (Brief Bioinform 2025)](https://academic.oup.com/bib/article/26/6/bbaf611/8329262)

**İmbalanced / Tabular Learning:**
- [Class-balanced loss functions for GBDT (Neurocomputing 2025)](https://dl.acm.org/doi/10.1016/j.neucom.2025.129896)
- [Balanced Underbagged Ensemble (ISAF 2025)](https://onlinelibrary.wiley.com/doi/10.1002/isaf.70018)
- [CLIMB: Class-imbalanced Learning Benchmark on Tabular Data (arXiv 2505.17451)](https://arxiv.org/pdf/2505.17451)
- [SMOTE variants comparative study (IIS 2025)](https://iacis.org/iis/2025/2_iis_2025_70-85.pdf)

**Tabular Foundation Models:**
- [TabPFN-2.5 (arXiv 2511.08667)](https://arxiv.org/abs/2511.08667)
- [On Finetuning Tabular Foundation Models (arXiv 2506.08982)](https://arxiv.org/html/2506.08982v2)
- [Prior-Labs/TabPFN-v2-clf (HuggingFace)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)

**Bayes Error / Değerlendirme:**
- [Bayes Error Rate Estimation in Difficult Situations (arXiv 2506.03159)](https://arxiv.org/html/2506.03159v3)
- [FeeBee: Evaluating Bayes Error Estimators on Real-World Datasets (arXiv 2108.13034)](https://arxiv.org/pdf/2108.13034)

**Missing Data:**
- [Missingness Features in ML for Critical Care (PMC 2021)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8701717/)
- [Missing Data in EHR — Systematic Review 2010–2024 (PMC 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11615160/)

**Transfer / Multi-Task:**
- [Understanding Knowledge Transferability — Survey (arXiv 2507.03175)](https://arxiv.org/pdf/2507.03175)
- [Multi-Task Learning for Heterogeneous Block-Wise Missing Data (arXiv 2505.24413)](https://arxiv.org/pdf/2505.24413)

---

## 6. Sınırlamalar ve Dürüstlük Notu

- **Kaynak doğrulaması:** Yukarıdaki kaynaklar web araması + kısmi tam-metin çekimiyle toplandı. Birkaçının (özellikle çok yeni arXiv preprint'leri: GS-B³SE, TabICLv2) tam metni çekilmedi; başlık + özet düzeyinde doğrulandı. Bir deneye başlamadan önce ilgili kaynağın metodolojisi tam okunmalı.
- **Danışman F1 tuzağı:** Bu rapordaki hiçbir öneri danışmanın mutlak train-dağılımı F1'lerini (0.89–0.95) hedef almıyor. O sayılar sinyal tavanıdır; final performans değildir.
- **PAH çelişkisi kesin çözülmedi, ama yönlendirildi:** Literatür NB40'ın PAH Bayes-ceiling'inin artefakt olduğunu **güçlü şekilde ima ediyor** ama kanıtlamıyor — deney #1 (sağlam yeniden-hesap) bunu netleştirecek. "PAH kapandı" demeden önce reverse-distribution (#2) da denenmeli.
- **En yüksek getirili, en düşük riskli hamle** açık ara **reverse-distribution'ın yayılması (#2)** — kanıtlanmış, ucuz, iki panelde bakir.
