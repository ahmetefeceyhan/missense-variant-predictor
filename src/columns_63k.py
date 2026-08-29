# -*- coding: utf-8 -*-
"""
63k_genis (full_cravat_v3_63k.csv) icin sutun siniflandirma listeleri.
Sema legacy'dir (src/columns.py evreni ile ayni aile) ama sutun sayisi (777)
ve icerigi farklidir -- bu modul 63k'ya OZGUdur, columns.py'nin yerini almaz.

DIKKAT: Bu dosya calistirilarak degil, notebooks/50_63k_prep_eda.ipynb tarafindan
elle dogrulanan sutun listelerinin donduruldugu yerdir. Yeni bir CRAVAT surumu
gelirse (farklı annotator seti) bu listeler NB50'de yeniden kontrol edilmelidir.
"""

# ============================================================================
# 1. ETIKET KAYNAGI VE GRUP/SPLIT ANAHTARLARI (feature DEGIL)
# ============================================================================
LABEL_SOURCE_COL = 'clinvar__sig'
LABEL_REVSTAT_COL = 'clinvar__rev_stat'

GENE_GROUP_COL = 'base__hugo'          # feature olarak kullanilmaz, sadece GroupKFold anahtari
CHROM_COL = 'base__chrom'
POS_COL = 'base__pos'
REF_COL = 'base__ref_base'
ALT_COL = 'base__alt_base'
VARIANT_TYPE_COL = 'base__so'          # 'MIS' = missense filtresi icin
ACHANGE_COL = 'base__achange'          # p.Arg220Gln gibi -- FE icin (Grantham/BLOSUM62)

NON_FEATURE_COLS = [
    GENE_GROUP_COL, CHROM_COL, POS_COL, REF_COL, ALT_COL,
    VARIANT_TYPE_COL, ACHANGE_COL, LABEL_SOURCE_COL, LABEL_REVSTAT_COL,
]

# ============================================================================
# 2. DOGRUDAN SIZINTI: TUM clinvar__* VE clinvar_acmg__* (etiketin kendisi/kaynagi)
# ============================================================================
CLINVAR_LEAK_COLS = [
    'clinvar__sig', 'clinvar__disease_refs', 'clinvar__disease_refs_incl',
    'clinvar__disease_names', 'clinvar__clinvar_preferred_names', 'clinvar__hgvs',
    'clinvar__rev_stat', 'clinvar__id', 'clinvar__sig_conf', 'clinvar__sig_conf_incl',
    'clinvar__af_go_esp', 'clinvar__af_exac', 'clinvar__af_tgp',
    'clinvar__clinvar_allele_id', 'clinvar__variant_type',
    'clinvar__variant_type_sequence_ontology', 'clinvar__variant_clinical_sources',
    'clinvar__dbvar_id', 'clinvar__clinvar_gene_info', 'clinvar__gene_info',
    'clinvar__onc_disease_name', 'clinvar__onc_disease_name_incl',
    'clinvar__onc_disease_refs', 'clinvar__onc_disease_refs_incl',
    'clinvar__onc_classification', 'clinvar__onc_classification_type',
    'clinvar__onc_rev_stat', 'clinvar__onc_classification_conflicting',
    'clinvar__allele_origin', 'clinvar__dbsnp_id',
    'clinvar__somatic_disease_name', 'clinvar__somatic_disease_name_incl',
    'clinvar__somatic_refs', 'clinvar__somatic_refs_incl', 'clinvar__somatic_rev_stat',
    'clinvar__somatic_impact', 'clinvar__somatic_impact_incl',
    'clinvar__germline_or_somatic',
    'clinvar_acmg__ps1_id', 'clinvar_acmg__pm5_id',
]

# ============================================================================
# 3. ID / SERBEST METIN / TRANSKRIPT SUTUNLARI (feature degeri yok, sizinti riski var)
# ============================================================================
ID_TEXT_TRANSCRIPT_COLS = [
    'base__uid', 'base__note_variant', 'base__transcript', 'base__all_mappings',
    'base__cchange',  # cDNA notasyonu -- serbest metin, achange FE icin yeterli
    # transcript_id / *__transcript / *__all / *__link kaliplari (CRAVAT annotator ciktisi)
    'aloft__transcript', 'aloft__all',
    'alphamissense__transcript_id', 'alphamissense__uniprot_id',
    'alphamissense__protein_variant',
    'arrvars__link',
    'cancer_genome_interpreter__all',
    'chasmplus__transcript', 'chasmplus__all',
    'ditto__transcript', 'ditto__all',
    'encode_tfbs__all',
    'esm1b__transcript', 'esm1b__all',
    'eve__transcript',
    'funseq2__all',
    'gmvp__transcript',
    'grasp__all',
    'haploreg_afr__all', 'haploreg_amr__all', 'haploreg_asn__all', 'haploreg_eur__all',
    'interpro__all',
    'metarnn__transcript',
    'mirbase__transcript',
    'mupit__link',
    'mutation_assessor__transcript', 'mutation_assessor__all',
    'mutationtaster__transcript', 'mutationtaster__all',
    'mutpred1__transcript',
    'oncokb__all',
    'polyphen2__all',
    'provean__transcript', 'provean__all', 'provean__uniprot',
    'pseudogene__transcript',
    'revel__transcript', 'revel__all',
    'sift__transcript', 'sift__all',
    'swissprot_binding__all', 'swissprot_binding__uniprotkb',
    'swissprot_domains__all', 'swissprot_domains__uniprotkb',
    'swissprot_ptm__all', 'swissprot_ptm__uniprotkb',
    'ucscgenomebrowser__link',
    'vest__transcript', 'vest__all',
    'uniprot_domain__domain',
    'varity_r__p_vid',
    'cedar__epi_id', 'cedar__pubmed_id', 'cedar__epi_ref', 'cedar__epi_alt',
    'tagsampler__numsample', 'tagsampler__samples', 'tagsampler__tags',
]

# CHASMplus kanser-alt-tipi transcript/all sutunlari (32 kanser tipi x 2 sutun) -- pattern ile
_CHASMPLUS_CANCER_TYPES = [
    'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
    'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV',
    'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM',
    'UCEC', 'UCS', 'UVM',
]
for _ct in _CHASMPLUS_CANCER_TYPES:
    ID_TEXT_TRANSCRIPT_COLS.append(f'chasmplus_{_ct}__transcript')
    ID_TEXT_TRANSCRIPT_COLS.append(f'chasmplus_{_ct}__all')
del _ct

# ============================================================================
# 4. META-PREDICTOR SKORLARI (ClinVar uzerinde egitilmis/kismen ClinVar iceren)
#    Drop edilmez, ADIM 3 A6'da ayri ablasyon icin ayrilir.
# ============================================================================
LEAKY_META_PREDICTOR_SCORES = [
    'cadd__score', 'cadd__phred',
    'chasmplus__score',
    'clinpred__score',
    'cscape__score', 'cscape_coding__score',
    'ditto__score',
    'esm1b__score',
    'fathmm_xf__score', 'fathmm_mkl__fathmm_mkl_coding_score',
    'funseq2__score',
    'genocanyon__score',
    'gmvp__score',
    'metalr__score', 'metarnn__score', 'metasvm__score',
    'mistic__score',
    'mutation_assessor__score',
    'mutationtaster__score',
    'mutpred2__score',
    'ncer__score',
    'phdsnpg__score',
    'provean__score',
    'revel__score',
    'alphamissense__am_pathogenicity',
    'bayesdel__bayesdel_addAF_score', 'bayesdel__bayesdel_noAF_score',
]

LEAKY_META_PREDICTOR_PREDS = [
    'aloft__pred', 'metalr__pred', 'metarnn__pred', 'metasvm__pred', 'mistic__pred',
    'alphamissense__am_class',
]

# ============================================================================
# 5. PRED / CLASS STRING SUTUNLARI (kategorik, sayisal DEGIL)
# ============================================================================
PRED_STRING_SUFFIXES = ('__pred', '__class')


def get_pred_string_cols(columns):
    """Verilen sutun listesinden *__pred / *__class ile bitenleri dondurur."""
    return [c for c in columns if c.endswith(PRED_STRING_SUFFIXES)]


# ============================================================================
# 6. TIP AYRISTIRMA (NB50 Adim 6) -- calisma zamaninda df uzerinden hesaplanir
# ============================================================================
def classify_columns(df, exclude_cols=None):
    """
    Kalan (sizinti temizligi sonrasi) sutunlari NUMERIC / CATEGORICAL / BINARY
    olarak ayirir. PRED_STRING_SUFFIXES ile bitenler kategorik sayilir.

    Donen: dict(numeric=[...], categorical=[...], binary=[...])
    """
    exclude_cols = set(exclude_cols or [])
    numeric, categorical, binary = [], [], []

    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.endswith(PRED_STRING_SUFFIXES):
            categorical.append(col)
            continue

        series = df[col]
        if series.dtype == object or str(series.dtype) == 'category':
            categorical.append(col)
            continue

        nunique = series.dropna().nunique()
        if nunique <= 2:
            binary.append(col)
        else:
            numeric.append(col)

    return {'numeric': numeric, 'categorical': categorical, 'binary': binary}


# ============================================================================
# 7. ETIKET TURETIMI (NB50 Adim 2)
# ============================================================================
_BENIGN_PREFIXES = ('Benign/Likely benign', 'Benign', 'Likely benign')
_PATHOGENIC_PREFIXES = ('Pathogenic/Likely pathogenic', 'Pathogenic', 'Likely pathogenic')

_HIGH_CONF_REVSTAT = ('reviewed by expert panel', 'practice guideline')
_MED_CONF_REVSTAT = ('multiple submitters, no conflicts',)
_LOW_CONF_REVSTAT = ('single submitter', 'no assertion criteria provided')
_CONFLICT_REVSTAT_MARKERS = ('conflicting',)


def derive_label(sig_value):
    """
    clinvar__sig degerinden Label (0/1) turetir. Eslesmezse None doner
    (Label_excluded=True olarak isaretlenmeli).

    Not: 'Benign/Likely benign' `_BENIGN_PREFIXES` icinde `Benign`'den ONCE
    kontrol edilir -- aksi halde 'Benign/Likely benign' yanlislikla sadece
    'Benign' prefix'ine value uyar ama startswith('Benign') zaten True doner,
    bu yuzden sira onemli degil fakat 'Pathogenic/Likely pathogenic' icin
    startswith('Pathogenic') de True doner -- ayni mantik pathogenic tarafinda
    da simetrik calisir.
    """
    if not isinstance(sig_value, str):
        return None
    if sig_value.startswith(_BENIGN_PREFIXES):
        return 0
    if sig_value.startswith(_PATHOGENIC_PREFIXES):
        return 1
    return None


def derive_label_confidence(rev_stat_value):
    """clinvar__rev_stat -> label_conf agirligi. Conflicting/bilinmeyen -> None (disla)."""
    if not isinstance(rev_stat_value, str):
        return None
    val = rev_stat_value.lower()
    if any(marker in val for marker in _CONFLICT_REVSTAT_MARKERS):
        return None
    if any(marker in val for marker in _HIGH_CONF_REVSTAT):
        return 1.0
    if any(marker in val for marker in _MED_CONF_REVSTAT):
        return 0.9
    if any(marker in val for marker in _LOW_CONF_REVSTAT):
        return 0.6
    return 0.6


def is_qualified_label(sig_value):
    """'|other', '|drug response', '|risk factor' gibi ek nitelik tasiyan etiketleri isaretler."""
    if not isinstance(sig_value, str):
        return False
    return '|' in sig_value


# ============================================================================
# 8. FLOOR-F1 (her fold'un kendi prevalansindan hesaplanmali -- burada yardimci)
# ============================================================================
def floor_f1(prevalence):
    """Trivial 'hep pathogenic de' baseline F1'i: 2p/(1+p)."""
    return 2 * prevalence / (1 + prevalence)
