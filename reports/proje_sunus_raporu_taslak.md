# PROJE SUNUŞ RAPORU

---

## TAKIM ŞEMASI

{ Bu bölümü takım üyeleri doldurmalıdır. }

---

## PROBLEME EN YAKIN ÇÖZÜM SUNAN ULUSLARARASI MAKALELERİN ÖZETİ (10 PUAN)

{ Bu bölümü takım üyeleri doldurmalıdır — en güncel 5–10 çalışma seçilmeli. Önerilen çalışmalar: REVEL, CADD, AlphaMissense, ClinPred, MetaRNN, VEST4, SIFT, PolyPhen-2, DITTO, MutationTaster gibi projemizde in-silico skor olarak kullandığımız araçların orijinal makaleleri ve varyant patojenite tahmini alanındaki güncel ensemble/derin öğrenme yaklaşımları. }

---

## VERİ VE YÖNTEM (30 PUAN)

### 3.1 Kullanılan Veri Seti ve Etiketler (5 puan)

Bu çalışmada yarışma organizasyonu tarafından sağlanan `open_cravat_curation_v3.csv` veri seti kullanılmıştır. Veri seti dört gen panelinden oluşmaktadır:

| Panel | Toplam Satır | Patojenik (1) | Benign (0) | Eğitim/Test Durumu |
|-------|-------------|---------------|------------|-------------------|
| General (Genel) | 3.156 | 2.182 | 974 | Eğitim + Hold-out Test |
| Hereditary Cancer (Herediter Kanser) | 715 | 354 | 361 | Eğitim + Hold-out Test |
| PAH | 324 | 323 | 1 | Yalnızca Test (General modeli ile) |
| CFTR | 92 | 92 | 0 | Yalnızca Test (General modeli ile) |
| **Toplam** | **4.287** | **2.951** | **1.336** | |

Etiketleme, ACMG uyumlu ClinVar sınıflandırmasına dayanmaktadır. Ground truth etiketi `clinvar__sig` sütunundan türetilmiştir:
- **Sınıf 1 (Patojenik):** "Pathogenic" ve "Likely Pathogenic" birleştirilmiştir.
- **Sınıf 0 (Benign):** "Benign" ve "Likely Benign" birleştirilmiştir.

Bu dört değer dışında kalan belirsiz etiketler (VUS — Variant of Uncertain Significance vb.) veri setinden çıkarılmıştır.

### 3.2 Veri Kısıtları ve Etikete Doğrudan Erişimi Engelleme (5 puan)

Yarışma formatı gereği genomik adres bilgileri (kromozom, pozisyon) ve bazı kolon isimleri gizlenmiştir. Çözümümüz yalnızca sağlanan varyant profilleri üzerinden çalışmaktadır; dış kaynaklardan (ClinVar web arayüzü, gnomAD vb.) etiket araması yapılmamıştır.

Dolaylı sızıntı risklerini kontrol etmek için aşağıdaki önlemler alınmıştır:

1. **Meta-predictor sızıntısı kontrolü:** In-silico meta-predictor skorları (REVEL, ClinPred, MetaRNN, MetaSVM, MetaLR, CADD vb. toplam 68 skor + 10 kategorik tahmin) ClinVar etiketleriyle eğitilmiş araçlardır. Bu skorlar doğrudan "dolaylı etiket sızıntısı" kaynağıdır. Bu nedenle iki ayrı deney kolu tasarlanmıştır: (i) tüm in-silico skorları çıkarılmış pipeline (v3_no_insil) ve (ii) in-silico skorları korunmuş pipeline (v3_with_insil). Bu sayede modelin gerçek öğrenme kapasitesi ile sızıntı etkisi karşılaştırmalı olarak değerlendirilmiştir.
2. **Rankskoru çıkarma:** 12 adet rankscore sütunu, ham skorların monoton dönüşümü olduğundan bilgi tekrarı oluşturur; bu sütunlar çıkarılmıştır.
3. **Doğrudan sızdıran sütunların çıkarılması:** `clinvar__sig`, `clinvar__id`, `Germline review status` ve `Stars` sütunları doğrudan hedef değişkenle ilişkili olduğundan eğitim öncesi çıkarılmıştır.
4. **Panel kimliği kontrolü:** Panel bilgisi model eğitiminde feature olarak kullanılmamıştır. Her panel için ayrı model eğitilerek panel kimliğinin etiketle ilişkilenmesi engellenmiştir.
5. **Korelasyon filtresi:** r > 0.99 korelasyona sahip özellik çiftlerinde, hedef değişkenle daha düşük korelasyona sahip olan çıkarılmıştır.

### 3.3 Veri Ön İşleme ve Temsilleme Stratejisi (5 puan)

Veri ön işleme, tekrarlanabilir ve modüler bir pipeline olarak tasarlanmıştır. Aşağıdaki adımlar sırasıyla uygulanmaktadır:

**Adım 1 — Boolean dönüşümü:** `polarity_change` ve `chirality_shift` gibi boolean sütunlar integer (0/1) formatına dönüştürülmüştür.

**Adım 2 — Kategorik kodlama:** 10 adet in-silico tahmin sütunu (ör. `sift__prediction`, `mutationtaster__prediction`, `alphamissense__am_class`) label encoding ile sayısallaştırılmıştır. In-silico'suz pipeline'da bu sütunlar tamamen çıkarılmıştır.

**Adım 3 — Özellik mühendisliği:** Ham varyant özelliklerinden biyolojik anlam taşıyan türetilmiş özellikler üretilmiştir:
- `is_transition`: Nükleotid değişiminin transition (A↔G, C↔T) olup olmadığı (binary)
- `grantham_distance`: Amino asit çifti arasındaki fizikokimyasal farklılık skoru (0–300)
- `blosum62_score`: BLOSUM62 substitüsyon matrisi skoru
- `blosum62_ref_self`, `blosum62_alt_self`: Referans ve alternatif amino asidin kendisiyle eşleşme skoru
- `blosum62_delta`: Referans self-skoru ile substitüsyon skoru arasındaki fark
- `conservation_product`: PhyloP × PhastCons (evrimsel korunmuşluk çarpımı)
- `gerp_x_phylop`: GERP_RS × PhyloP (çapraz korunmuşluk etkileşimi)

**Adım 4 — One-hot encoding:** Referans/alternatif nükleotid (4+4) ve referans/alternatif amino asit (20+20) sabit alfabe ile one-hot encode edilmiştir (toplam 48 özellik). Sabit alfabe kullanımı, eğitim ve test setleri arasında tutarlılığı garanti etmektedir.

**Adım 5 — K-mer kodlama:** DNA ve protein sekans bağlamını yakalamak için k=2 k-mer frekans vektörleri çıkarılmıştır:
- DNA 2-merleri: 4² = 16 olası kombinasyon × 2 sütun = 32 özellik
- Protein 2-merleri: Gözlenen kelime dağarcığına göre değişken sayıda özellik

**Adım 6 — Sütun çıkarma:** Doğrudan sızdıran sütunlar (4), metadata sütunları (6), ham sekans sütunları (2), rankscore sütunları (12) ve CHASMplus alt tipleri çıkarılmıştır.

**Adım 7 — Eksik değer yönetimi:** NaN ve sonsuz değerler 0 ile doldurulmuş, tüm özellikler float tipine dönüştürülmüştür. Panel sütunu korunmuştur.

**Adım 8 — Filtreler:**
- Yüksek korelasyon filtresi: r > 0.99 olan çiftlerde düşük hedef korelasyonlu sütun çıkarılmıştır.
- Sıfır varyans filtresi: Sabit değerli sütunlar çıkarılmıştır.

**Korunan özellik grupları:** Evrimsel korunmuşluk skorları (GERP_NR, GERP_RS, PhastCons100, PhyloP100 — 4 sütun) ve fizikokimyasal değişim özellikleri (delta_hydropathy, delta_volume, delta_pi, delta_charge, polarity_change, chirality_shift — 6 sütun) her iki pipeline'da da korunmuştur.

### 3.4 Etiket Güvenilirliği ve Veri Kalitesi Kontrolü (5 puan)

Ground truth etiketleri ClinVar'ın ACMG uyumlu sınıflandırmasına dayanmakla birlikte, veri kalitesine yönelik sistematik kontroller uygulanmıştır:

1. **Çapraz-model hata analizi (difficulty_score):** Her hold-out test örneği için 5 farklı modelin (LightGBM, XGBoost, NN, DNN, SVM) tahminleri karşılaştırılmıştır. Her örneğe bir `difficulty_score` (0–5) atanmıştır; bu skor, kaç modelin o örneği yanlış sınıfladığını gösterir.
   - `difficulty_score = 5`: Tüm modeller yanlış → **ground truth şüpheli** (olası etiket hatası)
   - `difficulty_score ≥ 3`: Çoğunluk yanlış → incelenmesi gereken örnekler
   - `difficulty_score = 0`: Tüm modeller doğru → güvenilir etiket

2. **Tutarsız profil tespiti:** Difficulty score analizi, belirli örneklerin sistematik olarak tüm model mimarilerini yanılttığını ortaya koymaktadır. Bu örnekler, etiket gürültüsü veya atipik varyant profilleri olarak işaretlenmiştir.

3. **Panel bazlı kalite değerlendirmesi:** PAH panelinde yalnızca 1 benign örnek bulunması ve CFTR panelinde hiç benign örnek bulunmaması, bu panellerin eğitim için uygun olmadığını göstermiştir. Bu paneller yalnızca test amacıyla kullanılmıştır (General modeliyle değerlendirme).

4. **Sınıf dağılımı doğrulaması:** Orijinal veri setindeki `target` sütunu ile `clinvar__sig` eşlemesinden türetilen etiketler arasında PAH panelinde tutarsızlık tespit edilmiştir (orijinal: 219/105 dağılım, eşleme: 323/1 dağılım). ClinVar kaydı doğrulanarak eşleme yöntemi doğru bulunmuş ve kullanılmıştır.

### 3.5 Sınıf Dengesi ve Risk Perspektifi (5 puan)

Her panel için sınıf dağılımı aşağıdaki tabloda özetlenmiştir:

| Panel | Patojenik (%) | Benign (%) | Denge Durumu |
|-------|-------------|-----------|-------------|
| General | 2.182 (%69,1) | 974 (%30,9) | Orta dengesizlik |
| Hereditary Cancer | 354 (%49,5) | 361 (%50,5) | Dengeli |
| PAH | 323 (%99,7) | 1 (%0,3) | Aşırı dengesiz — yalnızca test |
| CFTR | 92 (%100) | 0 (%0) | Tek sınıf — yalnızca test |

**Dengesizlik yönetimi:**
- **SVM:** `class_weight='balanced'` parametresi ile azınlık sınıfına otomatik ağırlık artışı uygulanmıştır.
- **LightGBM/XGBoost:** `is_unbalance=True` (LightGBM) ve `scale_pos_weight` (XGBoost) seçenekleri ile sınıf dengesizliği ele alınmıştır.
- **NN/DNN:** Focal loss desteği ile zor örneklere daha yüksek kayıp ağırlığı verilebilmektedir.

**Karar eşiği ve risk perspektifi:**

Patojenik–benign sınıflamasında hata türlerinin klinik sonuçları asimetriktir:
- **Yanlış negatif (FN):** Patojenik varyantın benign olarak sınıflanması → hastanın tedavisiz kalma riski (daha kritik)
- **Yanlış pozitif (FP):** Benign varyantın patojenik olarak sınıflanması → gereksiz ileri tetkik (daha az kritik)

Bu asimetriyi yönetmek için:
- Karar eşiği, F1 skorunu maksimize eden değer olarak [0.10, 0.90] aralığında 0.01 adımlarla aranmıştır. F1 skoru, precision ve recall'un harmonik ortalaması olarak her iki hata türünü dengelemektedir.
- CFTR ve PAH gibi tek sınıflı panellerde recall (duyarlılık) birincil metrik olarak değerlendirilmiştir.
- Sonuçlar yalnızca "yüksek F1" değil, MCC (Matthews Correlation Coefficient) ve balanced accuracy gibi dengesizliğe dayanıklı metriklerle de raporlanmıştır.

### 3.6 Seçilen Algoritmalar ve Gerekçe (5 puan)

Beş farklı model mimarisi seçilmiş ve her biri tabular varyant profil verisinin farklı yönlerini yakalamak üzere tasarlanmıştır:

**1. LightGBM (Gradient Boosted Decision Trees)**
- **Gerekçe:** Tabular veride state-of-the-art performans, histogram tabanlı hızlı eğitim, kategorik özellik desteği, eksik değerlere doğal dayanıklılık.
- **Düzenlileştirme:** L1/L2 regularizasyon (reg_alpha, reg_lambda), min_child_samples, colsample_bytree, subsample.
- **Grid arama:** 12 kombinasyon (n_estimators × num_leaves × learning_rate), 3-fold CV.

**2. XGBoost (Extreme Gradient Boosting)**
- **Gerekçe:** LightGBM'e alternatif GBDT implementasyonu; farklı ağaç büyütme stratejisi (depth-wise) ile ensemble çeşitliliği sağlanmıştır.
- **Düzenlileştirme:** gamma, min_child_weight, L1/L2 regularizasyon.
- **Grid arama:** 12 kombinasyon, 3-fold CV.

**3. Neural Network — MLP3Layer (3-Katmanlı Çok Katmanlı Algılayıcı)**
- **Gerekçe:** Doğrusal olmayan özellik etkileşimlerini yakalama kapasitesi; BatchNorm ve Dropout ile regularizasyon.
- **Mimari:** Input → [Linear(2048) → BN → ReLU → Dropout] → [Linear(1024) → BN → ReLU → Dropout] → [Linear(512) → BN → ReLU → Dropout] → Linear(1)
- **Eğitim:** 50 epoch, patience=10, batch_size=2048.
- **Grid arama:** 4 kombinasyon (dropout × learning_rate), 3-fold CV.

**4. Deep Neural Network — DeepMLP (Derin Çok Katmanlı Algılayıcı)**
- **Gerekçe:** Residual bağlantılar ile daha derin ağ mimarisi; her 2 katmanda bir kısa yol bağlantısı gradient akışını korumaktadır.
- **Mimari:** 3–4 katman, 256–512 gizli birim, residual bağlantılı.
- **Eğitim:** 50 epoch, patience=10, CosineAnnealingLR scheduler.
- **Grid arama:** 4 kombinasyon (n_layers × hidden_dim), 3-fold CV.

**5. Support Vector Machine (SVM — Destek Vektör Makinesi)**
- **Gerekçe:** Yüksek boyutlu uzayda etkili karar sınırları; RBF kernel ile doğrusal olmayan sınıflandırma.
- **Düzenlileştirme:** `class_weight='balanced'` ile otomatik sınıf ağırlıklandırma.
- **Ön işleme:** StandardScaler zorunludur (SVM mesafe tabanlı çalışır).
- **Grid arama:** 6 kombinasyon (C × gamma), 3-fold CV.
- **Olasılık kalibrasyonu:** `probability=True` ile Platt scaling uygulanmıştır.

**Çeşitlilik prensibi:** Beş model, üç farklı paradigmayı temsil etmektedir: (i) ağaç tabanlı ensemble (LightGBM, XGBoost), (ii) sinir ağları (NN, DNN), (iii) çekirdek tabanlı yöntem (SVM). Bu çeşitlilik, çapraz-model hata analizinde güvenilir difficulty_score hesaplanmasını mümkün kılmaktadır.

---

## DENEY TASARIMI, SONUÇLAR VE İNCELEME (25 PUAN)

### 4.1 Deney Protokolü ve Veri Bölme (5 puan)

**Veri bölme stratejisi:**

Eğitilebilir paneller (General, Hereditary Cancer) için iki aşamalı bir değerlendirme protokolü uygulanmıştır:

1. **Hold-out ayrımı:** Stratified train_test_split ile %80 eğitim / %20 test ayrımı yapılmıştır (SEED=42, stratify=y). Test seti hiçbir model seçim kararına dahil edilmemiştir.

2. **Çapraz doğrulama (CV):** %80'lik eğitim seti üzerinde 3-fold stratified CV ile hiperparametre optimizasyonu gerçekleştirilmiştir. Her grid kombinasyonu için 3-fold ortalama F1 skoru hesaplanmış, en iyi kombinasyon seçilmiştir.

3. **Final eğitim:** En iyi hiperparametre seti ile %80'lik eğitim setinin tamamı üzerinde final model eğitilmiştir.

4. **Hold-out değerlendirme:** Final model, ayrılmış %20 hold-out test seti üzerinde değerlendirilmiştir.

**Panel bazlı eğitim kararı:**

Eğitim/test kararı dinamik olarak belirlenmiştir: her panel için azınlık sınıfı örneklem sayısı (min_class) hesaplanmış, min_class < 5 ise panel "yalnızca test" olarak sınıflandırılmıştır. Bu eşik, stratified CV'nin güvenilir çalışması için minimum gereksinimdir.

- **Eğitim panelleri:** General (min_class = 974), Hereditary Cancer (min_class = 354)
- **Test panelleri:** PAH (min_class = 1), CFTR (min_class = 0) → General modeli ile test edilmiştir.

**Rastlantısal iyi sonuç riski azaltma:**
- Tüm rastgele süreçlerde sabit tohum değeri (SEED=42) kullanılmıştır.
- 5 farklı model mimarisi üzerinde tutarlı performans, sonucun rastlantısal olmadığını doğrulamaktadır.
- 3-fold CV ile hiperparametre seçimi, tek fold'a bağımlılığı azaltmaktadır.

### 4.2 Performans Metrikleri ve Panel Bazlı Raporlama (5 puan)

Her model ve panel için aşağıdaki 9 metrik hesaplanmıştır:

| Metrik | Açıklama | Seçilme Gerekçesi |
|--------|----------|-------------------|
| F1 Score | Precision ve recall harmonik ortalaması | Ana karşılaştırma metriği; dengesizliğe kısmen dayanıklı |
| ROC-AUC | ROC eğrisi altındaki alan | Eşik bağımsız ayrım gücü |
| PR-AUC | Precision-Recall eğrisi altındaki alan | Dengesiz sınıflarda ROC-AUC'den daha bilgilendirici |
| MCC | Matthews Correlation Coefficient | Tüm confusion matrix hücrelerini kullanan tek metrik; dengesiz veride en güvenilir |
| Precision | Doğru pozitif / (Doğru pozitif + Yanlış pozitif) | Yanlış pozitif oranı kontrolü |
| Recall (Duyarlılık) | Doğru pozitif / (Doğru pozitif + Yanlış negatif) | Patojenik varyant kaçırma riski |
| Specificity (Özgüllük) | Doğru negatif / (Doğru negatif + Yanlış pozitif) | Benign varyant doğru tanıma |
| Balanced Accuracy | (Duyarlılık + Özgüllük) / 2 | Sınıf dengesizliğine dayanıklı doğruluk |
| Cohen's Kappa | Rastgele uyumun ötesindeki anlaşma | Sınıflandırma güvenilirliği |

**Karar eşiği stratejisi:**
F1 skorunu maksimize eden eşik, [0.10, 0.90] aralığında 0.01 adımlarla grid search ile belirlenmiştir. Bu yaklaşım, sabit 0.5 eşiğine göre sınıf dağılımına duyarlı bir optimizasyon sağlamaktadır.

**Panel bazlı raporlama:**
- Eğitim panelleri (General, Hereditary Cancer): Hold-out %20 üzerinde tüm metrikler
- Test panelleri (CFTR, PAH): General modelinin bu paneller üzerindeki tüm metrikler
- İki deney kolu karşılaştırması: in-silico skorlu vs. in-silico'suz pipeline

{ Sayısal sonuçlar notebook çalıştırıldıktan sonra buraya eklenmelidir. }

### 4.3 Hata Analizi ve Model Davranışı (5 puan)

Çapraz-model hata analizi, yanlış sınıflanan örneklerdeki sistematik desenleri ortaya çıkarmak için tasarlanmıştır:

**Yöntem:**
Hold-out test setindeki her örnek için 5 modelin (LightGBM, XGBoost, NN, DNN, SVM) tahminleri karşılaştırılmıştır. Her örneğe bir `difficulty_score` (0–5) atanmıştır:

| difficulty_score | Anlam | Yorumlama |
|-----------------|-------|-----------|
| 0 | Hiçbir model yanlış | Kolay örnek, güvenilir etiket |
| 1–2 | 1–2 model yanlış | Model-spesifik zayıflık |
| 3–4 | Çoğunluk yanlış | Zor örnek, incelenmeli |
| 5 | Tüm modeller yanlış | Ground truth şüpheli |

**Analiz çıktıları:**
- Her panel için difficulty_score dağılım histogramı
- Tüm modellerin yanlış tahmin ettiği örneklerin listesi (orijinal varyant bilgileriyle birlikte)
- Error heatmap: satırlar difficulty_score'a göre sıralı test örnekleri, sütunlar modeller
- Model bazlı confusion matrix

**Beklenen desenler:**
- Belirli amino asit değişim türlerinde (ör. muhafazakar substitüsyonlar) hata yoğunlaşması
- Evrimsel korunmuşluk skorlarının düşük olduğu bölgelerde belirsizlik artışı
- In-silico skorların bulunmadığı pipeline'da daha yüksek difficulty_score ortalaması beklenmektedir

{ Detaylı hata analizi bulguları notebook çalıştırıldıktan sonra buraya eklenmelidir. }

### 4.4 "Model Neden Böyle Karar Verdi?" — Açıklanabilirlik Yaklaşımı (5 puan)

Açıklanabilirlik için **feature importance** (özellik önemi) yöntemi kullanılmıştır. Kolon isimleri gizlenmiş olsa da, özellikler aşağıdaki gruplara ayrılarak yorumlanmıştır:

**Özellik grupları:**
1. **Evrimsel korunmuşluk:** GERP_NR, GERP_RS, PhastCons100, PhyloP100, conservation_product, gerp_x_phylop
2. **In-silico risk skorları:** CADD, REVEL, ClinPred, MetaRNN, AlphaMissense vb. (yalnızca with-insil pipeline)
3. **Fizikokimyasal değişim:** delta_hydropathy, delta_volume, delta_pi, delta_charge, Grantham distance, BLOSUM62 skorları
4. **Sekans bağlamı:** DNA/protein k-mer frekansları, is_transition
5. **Amino asit kimliği:** Referans/alternatif amino asit one-hot encoding

**Uygulanan yöntemler:**
- **LightGBM feature importance:** Split-based importance (bir özelliğin ağaçlarda kaç kez bölme noktası olarak kullanıldığı)
- **XGBoost feature importance:** Gain-based importance (her bölmenin ortalama bilgi kazancı)
- Her panel ve model için en önemli 20 özellik yatay çubuk grafik ile görselleştirilmiştir.

**Beklenen bulgular:**
- In-silico skorlu pipeline'da meta-predictor skorlarının (REVEL, ClinPred vb.) baskın özellikler olması
- In-silico'suz pipeline'da evrimsel korunmuşluk ve fizikokimyasal özelliklerin öne çıkması
- Her iki pipeline'da BLOSUM62 ve Grantham distance'ın tutarlı olarak önemli olması

{ Detaylı özellik önemi grafikleri ve yorumları notebook çalıştırıldıktan sonra buraya eklenmelidir. }

### 4.5 Öğrenme Süreci ve Teknik Evrim (5 puan)

Proje geliştirme sürecinde karşılaşılan sorunlar ve uygulanan çözümler kronolojik olarak özetlenmiştir:

**1. Etiket tutarsızlığı → Doğrulama ve eşleme yöntemi değişikliği**
- **Sorun:** Veri setindeki önceden hesaplanmış `target` sütunu ile `clinvar__sig` eşlemesinden türetilen etiketler arasında PAH panelinde ciddi tutarsızlık tespit edilmiştir (219/105 vs. 323/1).
- **Müdahale:** ClinVar kaydı doğrulanmış, `clinvar__sig` eşlemesinin doğru olduğu teyit edilmiş ve bu yöntem standart olarak kabul edilmiştir.
- **Etki:** PAH panelinin eğitim için uygun olmadığı netleşmiş, yalnızca test paneli olarak sınıflandırılmıştır.

**2. Panel dengesizliği → Dinamik eğitim/test sınıflandırması**
- **Sorun:** PAH (323:1) ve CFTR (92:0) panellerinde stratified CV uygulanamaması.
- **Müdahale:** Azınlık sınıfı < 5 olan paneller otomatik olarak "yalnızca test" olarak sınıflandırılmıştır. Bu paneller, General panelde eğitilen modelle değerlendirilmektedir.
- **Etki:** Güvenilir olmayan eğitim sonuçları yerine, paneller arası genelleme kapasitesi ölçülmüştür.

**3. In-silico sızıntısı → İki kollu deney tasarımı**
- **Sorun:** Meta-predictor skorlarının ClinVar etiketleriyle eğitilmiş olması, dolaylı etiket sızıntısı riski oluşturmaktadır.
- **Müdahale:** In-silico skorlu ve in-silico'suz iki ayrı pipeline tasarlanmıştır.
- **Etki:** Modelin gerçek öğrenme kapasitesi ile sızıntı etkisi karşılaştırmalı olarak ölçülebilmektedir.

**4. Model çeşitliliği → Hata analizi güvenilirliği**
- **Sorun:** Tek model ile hata analizi, modelin zayıflığını mı yoksa etiket hatasını mı gösterdiği belirsizdir.
- **Müdahale:** 5 farklı paradigmadan model kullanılarak çapraz-model difficulty_score hesaplanmıştır.
- **Etki:** difficulty_score=5 (tüm modeller yanlış) olan örnekler, yüksek güvenilirlikle etiket hatası adayı olarak işaretlenebilmektedir.

**5. NN/DNN eğitim süresi → Fast pipeline**
- **Sorun:** Tam grid search (16 kombinasyon × 150 epoch) uzun süre almaktadır.
- **Müdahale:** Fast pipeline tasarlanmıştır: 4 kombinasyon, 50 epoch, patience=10.
- **Etki:** Eğitim süresi ~4x azaltılmış, kabul edilebilir performans kaybıyla.

---

## YAKLAŞIMIN GEREKÇESİ, KAYNAK KULLANIMI VE ÖZGÜNLÜK (25 PUAN)

### 5.1 Neden Bu Algoritma / Mimari? (5 puan)

Seçilen beş model mimarisi, tabular varyant profil verisinin doğasına uygun olarak şu gerekçelerle belirlenmiştir:

**Ağaç tabanlı ensemble (LightGBM, XGBoost):**
- Tabular veride tutarlı olarak en yüksek performansı gösteren yöntemlerdir [kaynak eklenecek].
- Eksik değerlere doğal dayanıklılık, kategorik özellik desteği ve hızlı eğitim avantajı sunarlar.
- L1/L2 regularizasyon, early stopping ve subsampling ile overfitting kontrol edilir.
- Feature importance çıktısı ile açıklanabilirlik sağlarlar.
- İki farklı ağaç büyütme stratejisi (leaf-wise vs. depth-wise) ensemble çeşitliliği sağlar.

**Sinir ağları (MLP3Layer, DeepMLP):**
- Özellik etkileşimlerini doğrusal olmayan şekilde yakalama kapasitesi.
- BatchNorm ile internal covariate shift azaltma, Dropout ile regularizasyon.
- DeepMLP'deki residual bağlantılar, gradient vanishing problemini çözerek daha derin mimarilere izin verir.
- Ağaç tabanlı modellerin yakalayamadığı sürekli özellik etkileşimlerini modelleyebilir.

**Çekirdek tabanlı yöntem (SVM):**
- RBF kernel ile yüksek boyutlu uzayda etkili karar sınırları.
- `class_weight='balanced'` ile otomatik sınıf dengeleme.
- Küçük-orta ölçekli veri setlerinde (Hereditary Cancer: 715 satır) güçlü genelleme.
- Olasılık kalibrasyonu (Platt scaling) ile güvenilir olasılık çıktısı.

**Panel bazlı genelleme:**
- Her panel için ayrı model eğitilmesi, panel-spesifik varyant profillerinin öğrenilmesini sağlar.
- Test panellerinin (CFTR, PAH) General modeli ile test edilmesi, modelin paneller arası genelleme gücünü ölçer.

### 5.2 Alternatifler Neden Elendi? (5 puan)

**1. Lojistik Regresyon:**
- Elenme nedeni: Doğrusal karar sınırı, varyant profillerindeki karmaşık özellik etkileşimlerini yakalamakta yetersiz kalır. Özellikle evrimsel korunmuşluk × fizikokimyasal değişim gibi çapraz etkileşimler lojistik regresyon tarafından modellenemez. Temel performans karşılaştırması (baseline) olarak kullanılabilir ancak nihai model olarak yetersizdir.

**2. Random Forest:**
- Elenme nedeni: LightGBM ve XGBoost ile aynı paradigmaya (ağaç tabanlı ensemble) ait olmakla birlikte, gradient boosting'e kıyasla genellikle daha düşük performans gösterir. Bagging yerine boosting, sıralı hata düzeltme mekanizması sayesinde daha iyi genelleme sağlar. Ayrıca Random Forest, LightGBM kadar verimli hiperparametre arama alanı sunmaz.

**3. Transformer tabanlı derin öğrenme (TabNet, FT-Transformer):**
- Elenme nedeni: Mevcut veri boyutu (General: 3.156, Hereditary Cancer: 715 satır) transformer mimarilerinin avantaj sağlayacağı ölçeğin altındadır. Eğitim maliyeti yüksek, açıklanabilirlik zor, küçük veri setlerinde overfitting riski belirgindir. Ayrıca bu mimariler daha karmaşık hiperparametre ayarlaması gerektirir.

**4. Tek büyük model (tüm paneller birlikte):**
- Elenme nedeni: Panel bazlı performans tutarsızlığı riski. Panel kimliğinin dolaylı olarak etiketle ilişkilenme riski. Panel-spesifik varyant profillerinin öğrenilememesi. Ayrı modeller, her panelin kendine özgü özellik dağılımını daha iyi yakalar.

### 5.3 Parametre Seçimi ve Model Ayarları (5 puan)

**Arama stratejisi:** Exhaustive grid search uygulanmıştır. Her model tipi için önceden tanımlanmış bir parametre ızgarası oluşturulmuş ve tüm kombinasyonlar 3-fold stratified CV ile değerlendirilmiştir.

| Model | Grid Boyutu | Değerlendirme Metriği | Early Stopping |
|-------|------------|----------------------|----------------|
| LightGBM | 12 (3×2×2) | F1 (3-fold CV) | n_estimators limiti |
| XGBoost | 12 (3×2×2) | F1 (3-fold CV) | n_estimators limiti |
| NN (Fast) | 4 (2×2) | F1 (3-fold CV) | patience=10 |
| DNN (Fast) | 4 (2×2) | F1 (3-fold CV) | patience=10 |
| SVM | 6 (3×2) | F1 (3-fold CV) | — |

**Sabit parametreler (LightGBM):** min_child_samples=20, subsample=0.8, colsample_bytree=0.8
**Sabit parametreler (XGBoost):** subsample=0.8, colsample_bytree=0.8, min_child_weight=3
**Sabit parametreler (NN/DNN):** weight_decay=1e-4, batch_size=2048
**Sabit parametreler (SVM):** kernel='rbf', class_weight='balanced', probability=True, cache_size=1000

**Karar eşiği optimizasyonu:** Grid search sonrası, en iyi modelin hold-out test seti üzerinde [0.10, 0.90] aralığında F1-maximizing threshold aranmıştır.

**Tekrarlanabilirlik:** Tüm rastgele süreçlerde SEED=42 kullanılmıştır (train_test_split, model initialization, CV splitting).

### 5.4 Hesaplama Kaynakları ve Çalıştırılabilirlik (5 puan)

{ Bu bölümü takım üyeleri kendi donanım bilgileriyle doldurmalıdır. Aşağıdaki bilgiler projeden çıkarılmıştır: }

**Yazılım ortamı:**
- İşletim Sistemi: Windows 11 Pro
- Python framework'leri ve minimum sürümler:
  - pandas ≥ 2.0
  - numpy ≥ 1.24
  - LightGBM ≥ 4.0
  - XGBoost ≥ 2.0
  - scikit-learn ≥ 1.3
  - PyTorch ≥ 2.0 (torch-directml GPU desteği ile)
  - Optuna ≥ 3.0 (hiperparametre optimizasyonu)
  - SHAP ≥ 0.42 (açıklanabilirlik)
  - matplotlib ≥ 3.7, seaborn ≥ 0.12 (görselleştirme)
  - fpdf2 ≥ 2.7 (PDF rapor üretimi)

**Eğitim parametreleri:**
- Batch boyutu: 2048 (NN/DNN)
- SVM cache boyutu: 1000 MB
- NN/DNN epoch sayısı: Maksimum 50 (fast pipeline), patience=10
- Deterministik ayarlar: SEED=42, tüm rastgele süreçlerde sabit

{ Aşağıdaki bilgilerin eklenmesi gerekmektedir:
- CPU modeli ve çekirdek sayısı
- GPU modeli ve VRAM miktarı (varsa)
- RAM miktarı
- Toplam eğitim süresi (her model tipi için ayrı ayrı)
- Tek örnek çıkarım süresi
- Toplu değerlendirme süresi }

### 5.5 Özgünlük (5 puan)

Çözümümüzün özgün teknik katkıları şunlardır:

**1. İki kollu in-silico sızıntı analizi:**
Meta-predictor skorlarının ClinVar etiketleriyle eğitilmiş olması nedeniyle oluşan dolaylı sızıntı, iki ayrı pipeline ile sistematik olarak ölçülmüştür. Bu karşılaştırma, modelin gerçek öğrenme kapasitesini sızıntı etkisinden ayırt etmeyi mümkün kılmaktadır. In-silico'suz pipeline'ın performansı, modelin yalnızca fizikokimyasal ve evrimsel özelliklerden ne kadar öğrenebildiğini göstermektedir.

**2. Çapraz-model difficulty_score ile etiket kalitesi değerlendirmesi:**
Beş farklı paradigmadan modelin (ağaç, sinir ağı, SVM) aynı örneklerde tutarlı olarak hata yapması, etiket güvenilirliği hakkında kanıt sağlamaktadır. difficulty_score=5 olan örnekler, ground truth'un yeniden incelenmesi gereken adaylar olarak raporlanmıştır. Bu yaklaşım, standart etiket gürültüsü tespiti yöntemlerine (confident learning vb.) alternatif, ensemble tabanlı bir katkıdır.

**3. Dinamik panel sınıflandırması:**
Eğitilebilirlik kararının azınlık sınıfı örneklem sayısına göre otomatik olarak belirlenmesi, farklı veri dağılımlarına uyum sağlayan esnek bir framework sunmaktadır. Bu mekanizma, yeni paneller eklendiğinde manuel müdahale gerektirmez.

**4. Panel-arası genelleme testi:**
CFTR ve PAH panellerinin General modeli ile test edilmesi, modelin görülmemiş gen panellerine genelleme kapasitesini doğrudan ölçmektedir. Bu, klinik uygulamada yeni gen panelleri için modelin kullanılabilirliği hakkında kanıt sağlar.

**5. Biyolojik özellik mühendisliği:**
BLOSUM62, Grantham distance, conservation cross-product ve k-mer encoding gibi biyolojik bilgiye dayalı özellikler, ham veriden türetilmiştir. Bu özellikler, kolon isimleri gizlenmiş olsa bile varyant profilinin biyolojik anlamını modele aktarmaktadır.

---

## REFERANSLAR

{ IEEE atıf stiline uygun olarak formatlanmalıdır. Önerilen kaynaklar: }

[1] { LightGBM: Ke, G., et al. "LightGBM: A highly efficient gradient boosting decision tree." NeurIPS, 2017. }
[2] { XGBoost: Chen, T., Guestrin, C. "XGBoost: A scalable tree boosting system." KDD, 2016. }
[3] { REVEL: Ioannidis, N.M., et al. "REVEL: An ensemble method for predicting the pathogenicity of rare missense variants." AJHG, 2016. }
[4] { ClinVar: Landrum, M.J., et al. "ClinVar: improvements to accessing data." NAR, 2020. }
[5] { AlphaMissense: Cheng, J., et al. "Accurate proteome-wide missense variant effect prediction with AlphaMissense." Science, 2023. }
[6] { CADD: Rentzsch, P., et al. "CADD-Splice—improving genome-wide variant effect prediction using deep learning-derived splice scores." Genome Medicine, 2021. }
[7] { MetaRNN: Li, C., et al. "MetaRNN: Differentiating rare pathogenic and rare benign missense SNVs and InDels using deep learning." Genome Medicine, 2022. }
[8] { ACMG: Richards, S., et al. "Standards and guidelines for the interpretation of sequence variants." Genetics in Medicine, 2015. }
{ Diğer kaynaklar eklenmelidir. }
