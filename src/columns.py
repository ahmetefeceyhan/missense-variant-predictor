# -*- coding: utf-8 -*-
"""
Sutun siniflandirma listeleri.
Veri sizintisi (data leakage) onleme icin kritik referans.
"""

# ============================================================================
# 1. DOGRUDAN SIZINTI SUTUNLARI (hedef degiskenin kopyasi/kaynagi)
# ============================================================================
DIRECT_LEAK_COLS = [
    'clinvar__sig',              # ClinVar klinik anlamlilik etiketi
    'clinvar__id',               # ClinVar varyant ID
    'Germline review status',    # Uzman panel degerlendirmesi
    'Stars',                     # ClinVar yildiz puani
]

# ============================================================================
# 2. META-PREDICTOR SKORLARI (ClinVar uzerinde egitilmis ML modelleri)
#    Dongusellik (circularity) yaratiyor -- TAMAMI kaldirilmali
# ============================================================================
LEAKY_META_PREDICTOR_SCORES = [
    # CADD
    'cadd__score', 'cadd__phred',
    # CHASMplus (ana + tum kanser alt tipleri)
    'chasmplus__score',
    # ClinPred
    'clinpred__score',
    # CScape
    'cscape__score', 'cscape_coding__score',
    # DITTO
    'ditto__score',
    # ESM-1b
    'esm1b__score',
    # FATHMM
    'fathmm_xf__score', 'fathmm_mkl__fathmm_mkl_coding_score',
    # FunSeq2
    'funseq2__score',
    # GenoCanyon
    'genocanyon__score',
    # GMVP
    'gmvp__score',
    # MetaLR / MetaRNN / MetaSVM (meta-ensemble tahminciler)
    'metalr__score', 'metarnn__score', 'metasvm__score',
    # MISTIC
    'mistic__score',
    # Mutation Assessor
    'mutation_assessor__score',
    # MutationTaster
    'mutationtaster__score',
    # MutPred2
    'mutpred2__score',
    # NCER
    'ncer__score',
    # PhD-SNPg
    'phdsnpg__score',
    # PROVEAN
    'provean__score',
    # REVEL (meta-predictor)
    'revel__score',
    # SIFT
    'sift__score',
    # VEST
    'vest__score',
    # AlphaMissense
    'alphamissense__am_pathogenicity',
    # BayesDel (meta-predictor)
    'bayesdel__bayesdel_addAF_score',
]

# Meta-predictor kategorik tahminleri
LEAKY_META_PREDICTOR_PREDS = [
    'esm1b__prediction',
    'metalr__pred',
    'metarnn__pred',
    'metasvm__pred',
    'mistic__pred',
    'mutationtaster__prediction',
    'phdsnpg__prediction',
    'provean__prediction',
    'sift__prediction',
    'alphamissense__am_class',
]

# ============================================================================
# 3. RANKSCORE SUTUNLARI (ham skorlarin monoton donusumleri)
# ============================================================================
RANKSCORE_COLS = [
    'clinpred__rankscore',
    'cscape_coding__rankscore',
    'esm1b__rankscore',
    'genocanyon__rankscore',
    'metalr__rankscore',
    'metasvm__rankscore',
    'mutation_assessor__rankscore',
    'mutationtaster__rankscore',
    'mutpred2__rankscore',
    'provean__rankscore',
    'revel__rankscore',
    'sift__rankscore',
]

# ============================================================================
# 4. gnomAD SUTUNLARI (kullanici karari: TAMAMI kaldirilacak)
# ============================================================================
GNOMAD_COLS = [
    'gnomad4__af',
    'gnomad4__an',
    'gnomad4__ac',
    'gnomad4__nhomalt',
]

# ============================================================================
# 5. METADATA / ID SUTUNLARI (her varyant icin benzersiz -- feature degil)
# ============================================================================
METADATA_COLS = [
    'base__chrom', 'base__pos',
    'base__cchange', 'base__achange',
    'alphamissense__protein_variant',
    'base__hugo',                        # Gen ismi (yuzlerce kategori, overfitting)
]

# Varyant degisim sutunlari -- one-hot encode edilecek, drop edilmeyecek
VARIANT_CHANGE_COLS = [
    'base__ref_base', 'base__alt_base',  # DNA bazı degisimi (A>G vb.)
    'ref_amino', 'alt_amino',            # Amino asit degisimi (V>A vb.)
]

# ============================================================================
# 6. GUVENLI BIYOLOJIK OZELLIKLER (sizinti icermeyen)
# ============================================================================
SAFE_CONSERVATION = [
    'gerp__gerp_nr',                    # GERP neutral rate
    'gerp__gerp_rs',                    # GERP rejected substitution
    'phastcons__phastcons100_vert',     # PhastCons vertebrate
    'phylop__phylop100_vert',           # PhyloP vertebrate
]

SAFE_PHYSICOCHEMICAL = [
    'delta_hydropathy',
    'delta_volume',
    'delta_pi',
    'delta_charge',
    'polarity_change',
    'chirality_shift',
]

SAFE_GENE = [
    # base__hugo v3'te METADATA_COLS'a tasinarak drop edildi (yuzlerce kategori → overfitting)
]

# ============================================================================
# 7. SEKANS SUTUNLARI
# ============================================================================
# Kullanilacak sekans sutunlari (sadece 11-mer'ler)
SEQUENCE_COLS = [
    'DNA_11mer_Ref', 'DNA_11mer_Alt',
    'Prot_11mer_Ref', 'Prot_11mer_Alt',
]

# Kullanilmayacak sekans sutunlari (tam uzunluk -- 11-mer yeterli)
DROP_SEQUENCE_COLS = [
    'Ref_Sequence', 'Alt_Sequence',
]

# ============================================================================
# 8. LABEL-ENCODE EDILECEK KATEGORIK TAHMIN SUTUNLARI (v3 modu)
#    Yarisma formatinda saglanan in-silico prediction sonuclari.
#    Integer'a cevrilerek feature olarak kullanilir.
# ============================================================================
LABEL_ENC_COLS = [
    'esm1b__prediction',          # Tolerated / Deleterious
    'metalr__pred',               # Tolerated / Damaging
    'metarnn__pred',              # Tolerated / Damaging
    'metasvm__pred',              # Tolerated / Damaging
    'mistic__pred',               # Benign / Deleterious
    'mutationtaster__prediction', # Damaging / Polymorphism / Automatic ...
    'phdsnpg__prediction',        # Pathogenic / Benign
    'provean__prediction',        # Damaging / Neutral
    'sift__prediction',           # Damaging / Tolerated
    'alphamissense__am_class',    # likely_benign / likely_pathogenic / ambiguous
]

# ============================================================================
# 9. ONE-HOT ENCODING SABITLERI (v3 modu — ayri encoding, sabit alfabe)
# ============================================================================
OHE_COLS_V3 = {
    'base__ref_base': list('ACGT'),
    'base__alt_base': list('ACGT'),
    'ref_amino':      list('ACDEFGHIKLMNPQRSTVWY'),
    'alt_amino':      list('ACDEFGHIKLMNPQRSTVWY'),
}

# K-mer alfabeleri
DNA_ALPHABET = list('ACGT')
PROT_ALPHABET = list('ACDEFGHIKLMNPQRSTVWY_')  # 20 AA + gap/pad

# ============================================================================
# BIRLESITIRILMIS LISTELER
# ============================================================================

# Temiz modda kaldirilacak tum sutunlar
ALL_LEAKY_COLS = (
    DIRECT_LEAK_COLS
    + LEAKY_META_PREDICTOR_SCORES
    + LEAKY_META_PREDICTOR_PREDS
    + RANKSCORE_COLS
    + GNOMAD_COLS
    + METADATA_COLS
    + DROP_SEQUENCE_COLS
)

# CHASMplus kanser alt tipi skorlari (dinamik tespit)
def get_chasmplus_subtypes(columns):
    """DataFrame sutunlarindan CHASMplus alt tip skorlarini tespit et."""
    return [c for c in columns
            if c.startswith('chasmplus_') and c.endswith('__score')
            and c != 'chasmplus__score']
