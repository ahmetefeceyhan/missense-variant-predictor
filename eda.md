# CONTEXT: TEKNOFEST 2026 Sağlıkta Yapay Zeka - Varyant Analizi Veri Kümesi

## Metodolojik Keşifçi Veri Analizi (EDA) ve Pipeline Refaktör Kılavuzu

Bu doküman,** **`YARISMA_TRAIN_PAH.csv` veri kümesi üzerinde yapılan derinlemesine Keşifçi Veri Analizi (EDA) bulgularını, veri kümesindeki yapısal tuzakları ve** ** **Claude Code** 'un projeyi dönüştürürken uyması gereken tüm kısıtları ve implementasyon adımlarını içerir.

## 1. Veri Kümesinin Genel Anatomisi

* **Veri Boyutları:** 372 Satır (Gözlem), 353 Sütun (Özellik). Veri yapısı tipik bir** ****"High-Dimensional, Low-Sample Size" (Yüksek Boyutlu, Az Örneklemli)** problemdir. Modelin ezberleme (overfitting) eğilimi olağanüstü düzeyde yüksektir.
* **Sınıf Dengesi (Class Imbalance):** `Label` sütununda** ****310 tane "1" (Pozitif)** etiketine karşılık yalnızca** ****62 tane "0" (Negatif)** etiket bulunmaktadır. Dağılım oranı tam** ** **5:1** 'dir.
  * *Kritik Uyarı:* Standart Doğruluk (Accuracy) metriği aldatıcıdır. Ana optimizasyon odağı** ** **Recall (Duyarlılık)** ve** ****F1-Score** olmalıdır.
* **Genel Eksik Veri Oranı:** Verideki tüm hücrelerin** ****%54.30'u boş (NaN)** değerlerden oluşmaktadır.

## 2. Tespit Edilen Yapısal Komite Tuzakları (Adversarial Noise & Sızıntılar)

Yarışma komitesi, ham veriyi doğrudan modele besleyen yarışmacıları elemek amacıyla veri kümesinin içerisine 4 büyük yapısal "hinlik" yerleştirmiştir. Modellerin belirli bir doğruluk oranında tıkanmasının ana sebebi budur:

### A. Sıfır Varyanslı Sabit Sütun Tuzağı (90 Sütun)

* **Bulgu:** Verideki 353 sütunun** ****90 tanesinde varyans sıfırdır** (`AL_80`,** **`AL_101`,** **`AL_104`,** **`AL_110` vb.). Bu sütunlar satırlarda sadece** **`1.0` ve** **`NaN` değerlerini içerir, başka hiçbir varyasyon barındırmazlar.
* **Etki:** Ağaç tabanlı modeller (Random Forest, XGBoost vb.) her bölünmede özellikleri rastgele alt kümelerden seçtiği için, havuzun %25'ini kaplayan bu çöp sütunlar modelin gerçek sinyallere odaklanmasını engeller.

### B. "Hayalet Sinyal" (Ghost Feature) Tuzağı

* **Bulgu:** `AL_29`,** **`AL_30` ve** **`AL_2` gibi sütunlar sınıflar (0 ve 1) arasında devasa ortalama farkları barındırıyor gibi görünmektedir (Örn:** **`AL_29` ortalaması Sınıf 0 için 0.77, Sınıf 1 için 0.25). Ancak derin incelemede, bu sütunların 372 satırda** ****yalnızca 14 satırının dolu** olduğu, kalan 358 satırın** **`NaN` olduğu görülmüştür.
* **Etki:** Modeller bu sütunları** **`Feature Importance` hesaplamalarında en tepeye koyar ve sadece bu 14 satırlık gürültüyü ezberler. Test setinde bu sütunlar farklı geldiğinde model çöker.

### C. Birebir Kopya (İkiz) Sütunlar

* **Bulgu:** `CAT_3`,** **`CAT_4` ve** **`CAT_5` kategorik sütunları bağımsız özellikler gibi sunulmuş olsa da satır satır ve** **`NaN`pozisyonları bazında incelendiğinde** ****birebir aynı (kopya)** oldukları kanıtlanmıştır.

### D. Yapay Eksiklik Maskesi (Dropout Mask)

* **Bulgu:** Eksik verilerin (NaN) satır başına düşen oranı Sınıf 0 için** ** **%54.01** , Sınıf 1 için** ** **%54.35** 'tir. Bu kusursuz homojenlik, eksikliğin biyolojik bir süreçten değil, komitenin verinin üzerine rastgele uyguladığı bir** ****Eksiklik Maskesi** olmasından kaynaklanır. Eksik olma durumu kendi başına bir sınıfsal sinyal taşımamaktadır.

## 3. Kod Düzeyinde Veri Önişleme (Preprocessing) Pipeline Adımları

Claude Code, mevcut projedeki veri önişleme hattını tam olarak aşağıdaki** ** **metodolojik sıra ile refaktör etmelidir** :

### Adım 1: Sızıntı ve İkiz Sütunların Temizlenmesi

**Python**

```
import pandas as pd
import numpy as np

# Veri sızıntısı veya anlamsız ID içeren Variant_ID kaldırılmalı
if 'Variant_ID' in df.columns:
    df = df.drop(columns=['Variant_ID'])

# Birebir kopya olan kategorik sütunlar elenmeli
redundant_cats = [c for c in ['CAT_4', 'CAT_5'] if c in df.columns]
df = df.drop(columns=redundant_cats)
```

### Adım 2: Sıfır Varyans (Variance Threshold) Filtresi

**Python**

```
# NaN değerler dışarıda bırakıldığında sadece 1 benzersiz değer içeren sabit sütunlar temizlenir
constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1 and col != 'Label']
df = df.drop(columns=constant_cols)
```

### Adım 3: Hayalet Sütun (Düşük Doluluk) Filtresi

**Python**

```
# Dolu hücre sayısı kritik eşiğin (örn: 30 satır) altında kalan yalancı sinyaller elenir
min_non_null_required = 30
ghost_cols = [col for col in df.columns if df[col].notnull().sum() < min_non_null_required and col != 'Label']
df = df.drop(columns=ghost_cols)
```

### Adım 4: Çoklu Doğrusallık (Multicollinearity) Temizliği

**Python**

```
# Birbiriyle r > 0.90 üzerinde korelasyona sahip yüksek bağımlı sayısal sütunlar süzülür
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Label', errors='ignore').tolist()
corr_matrix = df[numeric_cols].corr().abs()

# Üst üçgen matris maskelenerek ikiz çiftlerden biri drop edilir
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
cols_to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.90)]

df = df.drop(columns=cols_to_drop_corr)
```

### Adım 5: Stratejik Eksik Veri Doldurma (Imputation)

* **Sayısal Özellikler:** Kalan sayısal özelliklerin ortalama mutlak çarpıklığı (Skewness)** ****2.84** seviyesindedir ve veride ciddi aykırı (outlier) değerler vardır. Bu yüzden kesinlikle** **`Mean` kullanılmamalı,** ****`Median`** imputation yapılmalıdır.
* **Kategorik Özellikler:** Boşlukları en çok tekrar eden sınıf (Mode) ile doldurmak genetik popülasyon dağılımını yapay olarak bozar. Bu nedenle** **`NaN` hücrelerine** ****`"Unknown"`** string sınıfı atanmalıdır.

**Python**

```
from sklearn.preprocessing import LabelEncoder

X = df.drop(columns=['Label'])
y = df['Label']

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].fillna('Unknown').astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    else:
        # Çarpık dağılım koruması
        X[col] = X[col].fillna(X[col].median())
```

## 4. Model Eğitim ve Düzenlileştirme (Regularization) Kısıtları

Komite gürültülerinden arındırılmış bu veri üzerinde eğitilecek modellerin ezberlemesini engellemek ve sınıf dengesizliğini yönetmek için şu parametre kısıtları zorunludur:

1. **Sınıf Ağırlıklandırma (Class Weight):** Kullanılan tüm algoritmalarda (Random Forest, XGBoost, LightGBM, CatBoost vb.) mutlaka dengesiz sınıf koruması aktif edilmelidir.
   * Örn:** **`class_weight='balanced'` veya** **`scale_pos_weight = (310 / 62)`
2. **Sığ Ağaç Yapıları (Strict Tree Pruning):** Boyut küçük ve veri yapay olarak maskelendiği için ağaçların derinleşmesine izin verilmemelidir.
   * `max_depth` parametresi kesinlikle** ****3** veya** ****4** ile sınırlandırılmalıdır.
   * `min_samples_leaf` parametresi** ****5** veya** ****10** olarak ayarlanmalıdır.
   * Eğitimde aşırı öğrenmeyi kesmek için erken durdurma (`early_stopping_rounds`) mekanizması kurgulanmalıdır.
