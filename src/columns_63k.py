# -*- coding: utf-8 -*-
"""
63k_genis (full_cravat_v3_63k.csv) icin sutun siniflandirma listeleri.
Sema legacy'dir (src/columns.py evreni ile ayni aile) ama sutun sayisi (777)
ve icerigi farklidir -- bu modul 63k'ya OZGUdur, columns.py'nin yerini almaz.

DIKKAT: Bu dosya calistirilarak degil, notebooks/50_63k_prep_eda.ipynb tarafindan
elle dogrulanan sutun listelerinin donduruldugu yerdir. Yeni bir CRAVAT surumu
gelirse (farklı annotator seti) bu listeler NB50'de yeniden kontrol edilmelidir.
"""
import re as _re

import numpy as np
import pandas as pd

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
    # VARITY / VEST -- dbNSFP ailesi, ClinVar uzerinde egitilmis diger meta-predictor'lar
    'varity_r__varity_r', 'varity_r__varity_er',
    'varity_r__varity_r_loo', 'varity_r__varity_er_loo',
    'vest__score', 'vest__pval',
]

# *__rankscore / *_rank_score -- dbNSFP'nin yukaridaki meta-predictor skorlarinin
# 0-1 normalize edilmis siralama versiyonu. Ayni sizinti riskini tasir (NB51'de
# CV F1'i 0.99'a tasiyan feature'larin buyuk kismi bu aileden cikti), bu yuzden
# skor listesine DEGIL ayri bir listeye konuyor -- filtrelemesi kolay olsun diye.
LEAKY_META_PREDICTOR_RANKSCORES = [
    'bayesdel__bayesdel_addAF_rankscore', 'bayesdel__bayesdel_noAF_rankscore',
    'clinpred__rankscore', 'cscape_coding__rankscore',
    'dann_coding__dann_rankscore', 'esm1b__rankscore', 'eve__rank_score',
    'fathmm_mkl__fathmm_mkl_coding_rankscore', 'fathmm_xf_coding__fathmm_xf_coding_rankscore',
    'fitcons__fitcons_coding_rankscore', 'genocanyon__rankscore',
    'gmvp__rank_score', 'lrt__lrt_converted_rankscore', 'metalr__rankscore',
    'metarnn__rank_score', 'metasvm__rankscore', 'mutation_assessor__rankscore',
    'mutationtaster__rankscore', 'mutpred1__mutpred_rankscore', 'mutpred2__rankscore',
    'primateai__primateai_rankscore', 'provean__rankscore', 'revel__rankscore',
    'sift__rankscore',
]

LEAKY_META_PREDICTOR_PREDS = [
    'aloft__pred', 'metalr__pred', 'metarnn__pred', 'metasvm__pred', 'mistic__pred',
    'alphamissense__am_class',
]

# ID sutunlari -- NB50'nin ID_TEXT_TRANSCRIPT_COLS listesinde gozden kacmis,
# NB51 feature-importance taramasinda yakalandi (mupit__hugo, omim__omim_id
# top-30da cikti). Literatur-referans sayisi da dolayli sizinti riski tasir:
# ClinVar'a giren varyantlar zaten literaturde daha cok bahsedilme egilimindedir.
NB51_ADDITIONAL_ID_LEAK_COLS = [
    'mupit__hugo',            # base__hugo'nun tekrari, ID/kategorik degil gen adi
    'omim__omim_id',          # OMIM kayit ID'si
    'litvar_full__rsid', 'litvar_full__reference_count', 'litvar_full__pmids',
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


# ============================================================================
# 9. base__achange FE (NB52 A7) -- 63k'da ref_amino/alt_amino onceden turetilmis
# gelmiyor (legacy CSV'nin aksine), bu yuzden 'p.Arg220Gln' formatindan sifirdan
# cikariyoruz. Grantham/BLOSUM62 tablolari icin src/features.py'daki GRANTHAM/
# BLOSUM62 sozlukleri (tek-harf AA kodlu) YENIDEN KULLANILIYOR.
# ============================================================================
_AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Gln': 'Q',
    'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V', 'Ter': '*', 'Sec': 'U', 'Pyl': 'O', 'Xaa': 'X',
}

_ACHANGE_RE = _re.compile(r'^p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|=|\*)$')

# Kyte-Doolittle hydropathy, molar volume (A^3), pI, net charge (fizyolojik pH)
_AA_HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
_AA_VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5, 'Q': 143.8,
    'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7, 'L': 166.7, 'K': 168.6,
    'M': 162.9, 'F': 189.9, 'P': 112.7, 'S': 89.0, 'T': 116.1, 'W': 227.8,
    'Y': 193.6, 'V': 140.0,
}
_AA_PI = {
    'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07, 'Q': 5.65,
    'E': 3.22, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'L': 5.98, 'K': 9.74,
    'M': 5.74, 'F': 5.48, 'P': 6.30, 'S': 5.68, 'T': 5.60, 'W': 5.89,
    'Y': 5.66, 'V': 5.96,
}
_AA_CHARGE = {  # fizyolojik pH'ta net yuk
    'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0,  # H notr yakinsa da genelde 0 alinir
}


def parse_achange(achange):
    """
    'p.Arg220Gln' -> ('R', 220, 'Q'). Stopgain ('p.Arg220*'), sessiz ('p.Arg220=')
    veya format-disi girdilerde (None, None, None) doner.
    """
    if not isinstance(achange, str):
        return (None, None, None)
    m = _ACHANGE_RE.match(achange.strip())
    if not m:
        return (None, None, None)
    ref3, pos, alt3 = m.groups()
    ref1 = _AA_3TO1.get(ref3)
    pos = int(pos)
    if alt3 == '=':
        alt1 = ref1
    elif alt3 == '*':
        alt1 = '*'
    else:
        alt1 = _AA_3TO1.get(alt3)
    return (ref1, pos, alt1)


def compute_achange_fe(achange_series, grantham_table, blosum62_table):
    """
    base__achange serisinden FE DataFrame'i uretir:
    ref_amino, alt_amino, aa_pos, grantham, blosum62,
    delta_hydropathy, delta_volume, delta_pi, delta_charge,
    is_stopgain, is_synonymous.

    grantham_table / blosum62_table: src.features.GRANTHAM / BLOSUM62 (tek-harf AA kodlu dict).
    """
    parsed = achange_series.astype(str).apply(parse_achange)
    ref_amino = parsed.apply(lambda t: t[0])
    aa_pos = parsed.apply(lambda t: t[1])
    alt_amino = parsed.apply(lambda t: t[2])

    is_stopgain = (alt_amino == '*').astype(int)
    is_synonymous = (ref_amino == alt_amino).astype(int)

    def _lookup(ref, alt, table):
        if ref is None or alt is None or ref == '*' or alt == '*':
            return np.nan
        return table.get((ref, alt), np.nan)

    grantham = pd.Series(
        [_lookup(r, a, grantham_table) for r, a in zip(ref_amino, alt_amino)],
        index=achange_series.index,
    )
    blosum62 = pd.Series(
        [_lookup(r, a, blosum62_table) for r, a in zip(ref_amino, alt_amino)],
        index=achange_series.index,
    )

    def _delta(prop_table):
        vals = []
        for r, a in zip(ref_amino, alt_amino):
            if r in prop_table and a in prop_table:
                vals.append(prop_table[a] - prop_table[r])
            else:
                vals.append(np.nan)
        return pd.Series(vals, index=achange_series.index)

    return pd.DataFrame({
        'ref_amino': ref_amino,
        'alt_amino': alt_amino,
        'aa_pos': aa_pos,
        'grantham': grantham,
        'blosum62': blosum62,
        'delta_hydropathy': _delta(_AA_HYDROPATHY),
        'delta_volume': _delta(_AA_VOLUME),
        'delta_pi': _delta(_AA_PI),
        'delta_charge': pd.Series(
            [_AA_CHARGE.get(a, 0) - _AA_CHARGE.get(r, 0) for r, a in zip(ref_amino, alt_amino)],
            index=achange_series.index,
        ),
        'is_stopgain': is_stopgain,
        'is_synonymous': is_synonymous,
    })
