# -*- coding: utf-8 -*-
"""
Yarisma veri seti (data/real_data/) icin sutun siniflandirmasi.

Veri seti anonimlestirilmis -- sutun isimleri prefix tipine gore gruplanmis:
  - Variant_ID : varyant benzersiz kimligi (model girdisi DEGIL)
  - Label      : 0/1 hedef sinifi
  - AL_1..334  : sayisal anotasyon/ozellik (skor, frekans, flag karisik)
  - CAT_1..6   : kategorik (populasyon, genotip, bolge kalitesi)
  - EK_1..9    : ek sayisal skorlar (negatif degerli, etki/patojenite tipi)
  - AA_1, AA_2 : amino asit harfi (R, L, G, P, D, V vb.)

Dosyalar: YARISMA_TRAIN_{MASTER, CFTR, PAH, KANSER}.csv (4 dosya, 353 sutun)
Kaynak: reports/DataAnalytics.pdf
"""

# ============================================================================
# 1. ID / HEDEF SUTUNLARI (model girdisi olarak KULLANILMAZ)
# ============================================================================
ID_COL = 'Variant_ID'
TARGET_COL = 'Label'

# Tum panellerden cikarilacak (feature olmayan) sutunlar
NON_FEATURE_COLS = [ID_COL, TARGET_COL]


# ============================================================================
# 2. SAYISAL ANOTASYON SUTUNLARI (AL_*)
#    Toplam 334 adet. Cogu 0-1 araliginda (skor/frekans), bir kismi
#    ikili/dusuk kardinaliteli "flag" gibi davraniyor.
#    NOT: PDF'e gore flag/binary olanlar veya 0-1 dis ranged'lar
#    EDA ile tespit edilip cikarilabilir.
# ============================================================================
AL_COLS = [f'AL_{i}' for i in range(1, 335)]   # AL_1 .. AL_334


# ----------------------------------------------------------------------------
# 2a. ASIRI EKSIK AL BLOKLARI (%80-95+ eksik -- PDF'de isaretlenmis)
#     Bu sutunlar:
#       - ya dogrudan drop edilir
#       - ya da sadece "is_missing" bayragi olarak kullanilir
#     Ablation ile karar verilmeli.
# ----------------------------------------------------------------------------
AL_HIGH_MISSING_BLOCK_1 = [f'AL_{i}' for i in range(1, 7)]    # AL_1..AL_6
AL_HIGH_MISSING_BLOCK_2 = [f'AL_{i}' for i in range(27, 39)]  # AL_27..AL_38

AL_HIGH_MISSING_COLS = AL_HIGH_MISSING_BLOCK_1 + AL_HIGH_MISSING_BLOCK_2


# ----------------------------------------------------------------------------
# 2b. MISSINGNESS RISKI YUKSEK SUTUNLAR
#     KANSER panelinde AL_16..AL_25 icin Label=0'da ~%29 eksik,
#     Label=1'de ~%86 eksik -> eksiklik desenı sinifla korele.
#     Yarisma test setinde ayni mekanizma yoksa LEAKAGE RISKI.
#     Ablation icin ayri grup olarak tutuluyor.
# ----------------------------------------------------------------------------
AL_MISSINGNESS_LEAKAGE_RISK = [f'AL_{i}' for i in range(16, 26)]  # AL_16..AL_25


# ----------------------------------------------------------------------------
# 2c. "Guvenli" AL sutunlari (yukaridaki riskli/eksik bloklar haricindekiler)
#     Uzman 1 (ana anotasyon uzmanı) icin ana girdi.
# ----------------------------------------------------------------------------
_RISKY_AL = set(AL_HIGH_MISSING_COLS) | set(AL_MISSINGNESS_LEAKAGE_RISK)
AL_SAFE_COLS = [c for c in AL_COLS if c not in _RISKY_AL]


# ============================================================================
# 3. KATEGORIK SUTUNLAR (CAT_*)
# ============================================================================
# CAT_1: gnomAD benzeri populasyon kategorisi
#        (gnomADe_NFE, gnomADg_NFE, gnomADe_AFR, gnomADe_EAS)
# CAT_2: All of Us populasyon kategorisi
#        (AllofUs_EUR, AllofUs_AFR, AllofUs_AMR, AllofUs_EAS)
# CAT_3, CAT_4, CAT_5: Genotip/nukleotid formati (A/A, C/C, G/G, T/T, ./.)
#        DIKKAT: MASTER'da CAT_3 ile CAT_5 birebir AYNI gorunuyor!
# CAT_6: Genomik bolge kalitesi / tekrarli bolge bayragi
#        (segdup, lcr, decoy&segdup) -- COK SEYREK DOLU (%95+ eksik)

CAT_POPULATION_COLS = ['CAT_1', 'CAT_2']            # populasyon bilgileri
CAT_GENOTYPE_COLS   = ['CAT_3', 'CAT_4', 'CAT_5']   # genotip formati (3==5 olabilir!)
CAT_REGION_COLS     = ['CAT_6']                     # bolge kalitesi (cok seyrek)

CAT_COLS = CAT_POPULATION_COLS + CAT_GENOTYPE_COLS + CAT_REGION_COLS  # CAT_1..CAT_6

# Birebir ayni olma supheli ciftler -- EDA'da kontrol edilmeli, biri drop edilir
SUSPECTED_DUPLICATE_PAIRS = [
    ('CAT_3', 'CAT_5'),   # PDF'de MASTER icin birebir ayni isaretlenmis
]


# ============================================================================
# 4. EK SAYISAL SKORLAR (EK_*)
#    Bazilarinda negatif deger -> frekans degil, etki/patojenite/istatistiksel skor.
#    Uzman 3 (EK skor uzmani) icin ana girdi.
# ============================================================================
EK_COLS = [f'EK_{i}' for i in range(1, 10)]   # EK_1 .. EK_9


# ============================================================================
# 5. AMINO ASIT SUTUNLARI (AA_*)
#    Tek harfli amino asit kodu (R, L, G, P, D, V vb.)
#    Kategorik olarak ele alinir; label-encode veya one-hot edilir.
# ============================================================================
AA_COLS = ['AA_1', 'AA_2']

# Standart 20 amino asit alfabesi + bilinmeyen
AA_ALPHABET = list('ACDEFGHIKLMNPQRSTVWY')
AA_UNKNOWN_TOKEN = 'X'   # MISSING/UNKNOWN icin


# ============================================================================
# 6. TUM KATEGORIK SUTUNLAR (CatBoost cat_features icin)
# ============================================================================
ALL_CATEGORICAL_COLS = CAT_COLS + AA_COLS


# ============================================================================
# 7. MoE UZMANLARI ICIN OZELLIK GRUPLARI
#    (DataAnalytics.pdf MoE onerisine gore)
# ============================================================================
# Uzman 1 -- Ana anotasyon uzmani (LightGBM/CatBoost)
EXPERT1_FEATURES = AL_SAFE_COLS

# Uzman 2 -- Populasyon/kategorik uzmani (CatBoost)
EXPERT2_FEATURES = CAT_COLS + AA_COLS

# Uzman 3 -- EK skor uzmani (kucuk, duzenlilestirilmis model)
EXPERT3_FEATURES = EK_COLS

# Uzman 4 -- Eksiklik deseni uzmani (sadece missing-mask)
#   Calisma zamaninda is_missing_<col> sutunlari turetilir.
#   Burada hangi sutunlardan turetilecekleri tanimlanir.
EXPERT4_SOURCE_COLS = AL_COLS + CAT_COLS + EK_COLS + AA_COLS


# ============================================================================
# 8. BIRLESITIRILMIS LISTELER
# ============================================================================

# Tum potansiyel feature sutunlari (ID ve Label haric)
ALL_FEATURE_COLS = AL_COLS + CAT_COLS + EK_COLS + AA_COLS

# Yuksek riskli sutunlar (ablation deneylerinde drop adayi)
HIGH_RISK_DROP_CANDIDATES = (
    AL_HIGH_MISSING_COLS          # %80-95+ eksik
    + AL_MISSINGNESS_LEAKAGE_RISK # eksiklik-etiket korelasyonu olan
    + CAT_REGION_COLS             # CAT_6 cok seyrek dolu
)


# ============================================================================
# 9. YARDIMCI FONKSIYONLAR
# ============================================================================
def get_missing_mask_col_name(col: str) -> str:
    """Bir sutun icin missing-indicator sutun adini uretir."""
    return f'is_missing_{col}'


def get_constant_cols(df, tol: int = 1) -> list:
    """
    PDF madde 4: Her panelde tek degerli (sabit) sutunlar var, bilgi tasimaz.
    Bu fonksiyon nunique <= tol olan sutunlari dondurur.
    Train ayrildiktan SONRA cagrilmali.
    """
    return [c for c in df.columns if df[c].nunique(dropna=True) <= tol]


def get_duplicate_col_pairs(df) -> list:
    """
    PDF madde 5: MASTER'da birebir ayni sutun ciftleri var.
    Tum sutun ciftleri arasinda birebir ayni olanlari dondurur.
    Train ayrildiktan SONRA cagrilmali.
    """
    pairs = []
    cols = list(df.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            if df[c1].equals(df[c2]):
                pairs.append((c1, c2))
    return pairs


def get_high_missing_cols(df, threshold: float = 0.80) -> list:
    """
    Eksiklik orani >= threshold olan sutunlari dondurur.
    PDF'e gore %80+ eksik olanlar dikkatli ele alinmali (drop ya da sadece mask).
    """
    miss_ratio = df.isna().mean()
    return miss_ratio[miss_ratio >= threshold].index.tolist()


# ============================================================================
# 10. PANEL META BILGILERI
#     PDF'den: satir sayilari ve sinif dengesizligi
# ============================================================================
PANEL_INFO = {
    'MASTER': {'n_rows': 2931, 'pos': 782,  'neg': 2149, 'file': 'YARISMA_TRAIN_MASTER.csv'},
    'KANSER': {'n_rows': 388,  'pos': 268,  'neg': 120,  'file': 'YARISMA_TRAIN_KANSER.csv'},
    'PAH':    {'n_rows': 372,  'pos': 62,   'neg': 310,  'file': 'YARISMA_TRAIN_PAH.csv'},
    'CFTR':   {'n_rows': 111,  'pos': 21,   'neg': 90,   'file': 'YARISMA_TRAIN_CFTR.csv'},
}
