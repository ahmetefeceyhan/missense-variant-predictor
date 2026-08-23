# -*- coding: utf-8 -*-
"""
Veri hazirlama ve ozellik muhendisligi (Feature Engineering) modulu.

4 mod desteklenir:
- no_fe:     Legacy baseline (sizintili -- meta-predictor skorlari korunur)
- with_fe:   Legacy FE (sizintili -- tool consensus korunur)
- clean:     Temiz baseline (tum meta-predictor/rankscore/gnomAD kaldirilir)
- clean_fe:  Temiz + gelismis FE (yonlu BLOSUM, PCA conservation, delta k-mer)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from itertools import product as itertools_product

from src.columns import (
    DIRECT_LEAK_COLS, LEAKY_META_PREDICTOR_SCORES, LEAKY_META_PREDICTOR_PREDS,
    RANKSCORE_COLS, GNOMAD_COLS, METADATA_COLS, SEQUENCE_COLS, DROP_SEQUENCE_COLS,
    VARIANT_CHANGE_COLS, ALL_LEAKY_COLS, get_chasmplus_subtypes,
    LABEL_ENC_COLS, OHE_COLS_V3, DNA_ALPHABET, PROT_ALPHABET,
)

# ============================================================================
# GRANTHAM DISTANCE LOOKUP TABLOSU
# ============================================================================
_GRANTHAM_RAW = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,('A','E'):107,
    ('A','G'):60,('A','H'):86,('A','I'):94,('A','L'):96,('A','K'):106,('A','M'):84,
    ('A','F'):113,('A','P'):27,('A','S'):99,('A','T'):58,('A','W'):148,('A','Y'):112,
    ('A','V'):64,('R','N'):86,('R','D'):96,('R','C'):180,('R','Q'):43,('R','E'):54,
    ('R','G'):125,('R','H'):29,('R','I'):97,('R','L'):102,('R','K'):26,('R','M'):91,
    ('R','F'):97,('R','P'):103,('R','S'):110,('R','T'):71,('R','W'):101,('R','Y'):77,
    ('R','V'):96,('N','D'):23,('N','C'):139,('N','Q'):46,('N','E'):42,('N','G'):80,
    ('N','H'):68,('N','I'):149,('N','L'):153,('N','K'):94,('N','M'):142,('N','F'):158,
    ('N','P'):91,('N','S'):46,('N','T'):65,('N','W'):174,('N','Y'):143,('N','V'):133,
    ('D','C'):154,('D','Q'):61,('D','E'):45,('D','G'):94,('D','H'):81,('D','I'):168,
    ('D','L'):172,('D','K'):101,('D','M'):160,('D','F'):177,('D','P'):108,('D','S'):65,
    ('D','T'):85,('D','W'):181,('D','Y'):160,('D','V'):152,('C','Q'):154,('C','E'):170,
    ('C','G'):159,('C','H'):174,('C','I'):198,('C','L'):198,('C','K'):202,('C','M'):196,
    ('C','F'):205,('C','P'):169,('C','S'):112,('C','T'):149,('C','W'):215,('C','Y'):194,
    ('C','V'):192,('Q','E'):29,('Q','G'):87,('Q','H'):24,('Q','I'):109,('Q','L'):113,
    ('Q','K'):53,('Q','M'):101,('Q','F'):116,('Q','P'):76,('Q','S'):68,('Q','T'):42,
    ('Q','W'):130,('Q','Y'):99,('Q','V'):96,('E','G'):98,('E','H'):40,('E','I'):134,
    ('E','L'):138,('E','K'):56,('E','M'):126,('E','F'):140,('E','P'):93,('E','S'):80,
    ('E','T'):65,('E','W'):152,('E','Y'):122,('E','V'):121,('G','H'):98,('G','I'):135,
    ('G','L'):138,('G','K'):127,('G','M'):127,('G','F'):153,('G','P'):42,('G','S'):56,
    ('G','T'):59,('G','W'):184,('G','Y'):147,('G','V'):109,('H','I'):94,('H','L'):99,
    ('H','K'):32,('H','M'):87,('H','F'):100,('H','P'):77,('H','S'):89,('H','T'):47,
    ('H','W'):115,('H','Y'):83,('H','V'):84,('I','L'):5,('I','K'):102,('I','M'):10,
    ('I','F'):21,('I','P'):95,('I','S'):142,('I','T'):89,('I','W'):61,('I','Y'):33,
    ('I','V'):29,('L','K'):107,('L','M'):15,('L','F'):22,('L','P'):98,('L','S'):145,
    ('L','T'):92,('L','W'):61,('L','Y'):36,('L','V'):32,('K','M'):95,('K','F'):102,
    ('K','P'):103,('K','S'):121,('K','T'):78,('K','W'):110,('K','Y'):85,('K','V'):97,
    ('M','F'):28,('M','P'):87,('M','S'):135,('M','T'):81,('M','W'):67,('M','Y'):36,
    ('M','V'):21,('F','P'):114,('F','S'):155,('F','T'):103,('F','W'):40,('F','Y'):22,
    ('F','V'):50,('P','S'):74,('P','T'):38,('P','W'):147,('P','Y'):110,('P','V'):68,
    ('S','T'):58,('S','W'):177,('S','Y'):144,('S','V'):124,('T','W'):128,('T','Y'):92,
    ('T','V'):69,('W','Y'):37,('W','V'):88,('Y','V'):55,
}
GRANTHAM = {}
for (a, b), v in _GRANTHAM_RAW.items():
    GRANTHAM[(a, b)] = v
    GRANTHAM[(b, a)] = v

# ============================================================================
# BLOSUM62 LOOKUP TABLOSU
# ============================================================================
_BLOSUM62_RAW = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,('A','Q'):-1,('A','E'):-1,
    ('A','G'):0,('A','H'):-2,('A','I'):-1,('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,
    ('A','P'):-1,('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):0,('R','D'):-2,('R','C'):-3,('R','Q'):1,('R','E'):0,('R','G'):-2,
    ('R','H'):0,('R','I'):-3,('R','L'):-2,('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,
    ('R','S'):-1,('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,('N','G'):0,('N','H'):1,
    ('N','I'):-3,('N','L'):-3,('N','K'):0,('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,
    ('N','T'):0,('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,('D','H'):-1,('D','I'):-3,
    ('D','L'):-4,('D','K'):-1,('D','M'):-3,('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,
    ('D','W'):-4,('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,('C','I'):-1,('C','L'):-1,
    ('C','K'):-3,('C','M'):-1,('C','F'):-2,('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,
    ('C','Y'):-2,('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-2,('Q','H'):0,('Q','I'):-3,('Q','L'):-2,('Q','K'):1,
    ('Q','M'):0,('Q','F'):-3,('Q','P'):-1,('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,
    ('Q','V'):-2,
    ('E','E'):5,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','L'):-3,('E','K'):1,('E','M'):-2,
    ('E','F'):-3,('E','P'):-1,('E','S'):0,('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,('G','M'):-3,('G','F'):-3,
    ('G','P'):-2,('G','S'):0,('G','T'):-2,('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,('H','F'):-1,('H','P'):-2,
    ('H','S'):-1,('H','T'):-2,('H','W'):-2,('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-3,('I','M'):1,('I','F'):0,('I','P'):-3,('I','S'):-2,
    ('I','T'):-1,('I','W'):-3,('I','Y'):-1,('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,('L','S'):-2,('L','T'):-1,
    ('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,('K','T'):-1,('K','W'):-3,
    ('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,('M','W'):-1,('M','Y'):-1,
    ('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,('P','V'):-2,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}
BLOSUM62 = {}
for (a, b), v in _BLOSUM62_RAW.items():
    BLOSUM62[(a, b)] = v
    BLOSUM62[(b, a)] = v


# ============================================================================
# YARDIMCI FONKSIYONLAR
# ============================================================================

def _encode_variant_changes(df):
    """
    Varyant degisim sutunlarini birlestirip one-hot encode et.
    base__ref_base + base__alt_base -> mutation_type (A>G vb.)
    ref_amino + alt_amino -> aa_change (V>A vb.)
    """
    new_cols = []

    # DNA baz degisimi
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        df['mutation_type'] = df['base__ref_base'].astype(str) + '>' + df['base__alt_base'].astype(str)
        mt_dummies = pd.get_dummies(df['mutation_type'], prefix='mut', dtype=int)
        new_cols.append(mt_dummies)
        df.drop(columns=['mutation_type', 'base__ref_base', 'base__alt_base'], inplace=True)
        print(f"  mutation_type: {mt_dummies.shape[1]} one-hot feature")

    # Amino asit degisimi
    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['aa_change'] = df['ref_amino'].astype(str) + '>' + df['alt_amino'].astype(str)
        aa_dummies = pd.get_dummies(df['aa_change'], prefix='aa', dtype=int)
        new_cols.append(aa_dummies)
        df.drop(columns=['aa_change', 'ref_amino', 'alt_amino'], inplace=True)
        print(f"  aa_change: {aa_dummies.shape[1]} one-hot feature")

    if new_cols:
        df = pd.concat([df] + new_cols, axis=1)

    return df


def _encode_kmer(df, seq_columns, max_features=200, min_df=3):
    """Sekans sutunlarini 2-mer frekanslarina donustur."""
    seq_features_list = []

    for col in seq_columns:
        if col in df.columns:
            vectorizer = CountVectorizer(
                analyzer='char', ngram_range=(2, 2),
                max_features=max_features, min_df=min_df
            )
            seq_matrix = vectorizer.fit_transform(df[col].fillna(''))
            feature_names = [f"{col}_{motif}" for motif in vectorizer.get_feature_names_out()]
            seq_df = pd.DataFrame(seq_matrix.toarray(), columns=feature_names, index=df.index)
            seq_features_list.append(seq_df)
            print(f"  {col}: {seq_df.shape[1]} k-mer feature")

    # Orijinal sekans metin sutunlarini dusur
    df.drop(columns=[c for c in seq_columns if c in df.columns], inplace=True)

    if seq_features_list:
        df = pd.concat([df] + seq_features_list, axis=1)

    return df


def _encode_kmer_with_delta(df, ref_alt_pairs, max_features=500, min_df=2):
    """
    Ref/Alt sekans ciftleri icin delta 2-mer feature'lari olustur.
    Delta = Ref 2-mer vektoru - Alt 2-mer vektoru (mutasyonun etkisini yakalar).
    """
    seq_features_list = []

    for ref_col, alt_col in ref_alt_pairs:
        if ref_col in df.columns and alt_col in df.columns:
            # Her iki sutunu birlikte fit et (ayni vocabulary)
            vectorizer = CountVectorizer(
                analyzer='char', ngram_range=(2, 2),
                max_features=max_features, min_df=min_df
            )
            all_seqs = pd.concat([df[ref_col].fillna(''), df[alt_col].fillna('')])
            vectorizer.fit(all_seqs)

            ref_matrix = vectorizer.transform(df[ref_col].fillna('')).toarray()
            alt_matrix = vectorizer.transform(df[alt_col].fillna('')).toarray()
            delta_matrix = ref_matrix - alt_matrix

            vocab = vectorizer.get_feature_names_out()
            base_name = ref_col.replace('Ref', '').replace('_Ref', '').replace('ref', '')

            # Ref features
            ref_names = [f"{ref_col}_{m}" for m in vocab]
            ref_df = pd.DataFrame(ref_matrix, columns=ref_names, index=df.index)
            seq_features_list.append(ref_df)

            # Alt features
            alt_names = [f"{alt_col}_{m}" for m in vocab]
            alt_df = pd.DataFrame(alt_matrix, columns=alt_names, index=df.index)
            seq_features_list.append(alt_df)

            # Delta features (mutasyonun k-mer etkisi)
            delta_names = [f"delta_{base_name}_{m}" for m in vocab]
            delta_df = pd.DataFrame(delta_matrix, columns=delta_names, index=df.index)
            seq_features_list.append(delta_df)

            print(f"  {ref_col}/{alt_col}: {len(vocab)} k-mer + {len(vocab)} delta")

    # Orijinal sekans sutunlarini dusur
    all_seq_cols = [c for pair in ref_alt_pairs for c in pair]
    df.drop(columns=[c for c in all_seq_cols if c in df.columns], inplace=True)

    if seq_features_list:
        df = pd.concat([df] + seq_features_list, axis=1)

    return df


def _filter_correlated(df, threshold=0.99):
    """r > threshold korelasyonlu kopya sutunlari filtrele (target'a daha az korele olani dusur)."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'target']

    if len(numeric_cols) <= 1:
        return df

    corr_matrix = df[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    target_corr = df[numeric_cols].corrwith(df['target']).abs()

    to_drop = set()
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if upper_tri.loc[idx, col] > threshold and idx not in to_drop and col not in to_drop:
                if target_corr.get(idx, 0) >= target_corr.get(col, 0):
                    to_drop.add(col)
                else:
                    to_drop.add(idx)

    if to_drop:
        df.drop(columns=list(to_drop), inplace=True)
        print(f"  r>{threshold} kopya filtreleme: {len(to_drop)} sutun dusuruldu")

    return df


def _filter_zero_variance(df):
    """Sifir varyanslı sutunlari kaldir."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'target']
    zero_var = [c for c in numeric_cols if df[c].std() == 0]
    if zero_var:
        df.drop(columns=zero_var, inplace=True)
        print(f"  Sifir varyans filtreleme: {len(zero_var)} sutun dusuruldu")
    return df


def _categorize_objects(df):
    """Object sutunlari category dtype'a cevir (Panel haric)."""
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Panel':
            df[col] = df[col].astype('category')
    return df


# ============================================================================
# MOD 1: NO-FE (Legacy -- sizintili baseline)
# ============================================================================

def prepare_data_no_fe(df):
    """
    Minimal veri hazirlama: sadece gerekli sutunlari dusur, sekans verilerini k-mer ile kodla.
    UYARI: Prediction ve rankscore sutunlari KORUNUR (sizintili baseline).
    """
    df = df.copy()

    # Bilgi sizintisi / metadata
    leak_cols = ['clinvar__sig', 'clinvar__id', 'Germline review status', 'Stars']
    df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True)

    # Yuksek kardinalite ID'ler
    id_cols = ['base__cchange', 'base__achange', 'base__chrom', 'base__pos',
               'base__ref_base', 'base__alt_base', 'ref_amino', 'alt_amino',
               'alphamissense__protein_variant']
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    # Kullanilmayacak sekans sutunlari (tam uzunluk -- 11-mer yeterli)
    df.drop(columns=[c for c in DROP_SEQUENCE_COLS if c in df.columns], inplace=True)

    # Boolean -> int
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Varyant degisim encoding (ref/alt -> one-hot)
    df = _encode_variant_changes(df)

    # K-mer kodlama (sadece 11-mer'ler, 2-mer)
    df = _encode_kmer(df, SEQUENCE_COLS, max_features=200, min_df=3)

    # Object -> category
    df = _categorize_objects(df)

    print(f"[No-FE] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# MOD 2: WITH-FE (Legacy -- kismi sizintili FE)
# ============================================================================

def prepare_data_with_fe(df):
    """
    Biyoinformatik feature engineering pipeline.
    UYARI: tool_consensus meta-predictor'lardan turetildigi icin circularity tasiyor.
    """
    df = df.copy()

    # Bilgi sizintisi
    leak_cols = ['clinvar__sig', 'clinvar__id', 'Germline review status', 'Stars']
    df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True)

    # Kullanilmayacak sekans sutunlari (tam uzunluk -- 11-mer yeterli)
    df.drop(columns=[c for c in DROP_SEQUENCE_COLS if c in df.columns], inplace=True)

    # Boolean -> int
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # CpG bolgesi
    if 'DNA_11mer_Ref' in df.columns:
        df['is_cpg_site'] = df['DNA_11mer_Ref'].apply(
            lambda s: int(s[5:7] == 'CG' or s[4:6] == 'CG') if isinstance(s, str) and len(s) >= 7 else 0)
        df['gc_content_11mer'] = df['DNA_11mer_Ref'].apply(
            lambda s: (s.count('G') + s.count('C')) / len(s) if isinstance(s, str) and len(s) > 0 else 0.5)

    # Transition vs Transversion
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    # Amino asit substitusyon ozellikleri
    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        df['grantham_category'] = pd.cut(
            df['grantham_distance'], bins=[-1, 50, 100, 150, 300],
            labels=[0, 1, 2, 3]).astype(float)

        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)

    # gnomAD turevleri
    if all(c in df.columns for c in ['gnomad4__ac', 'gnomad4__an']):
        df['gnomad4__af_computed'] = df['gnomad4__ac'] / df['gnomad4__an'].replace(0, np.nan)
    if 'gnomad4__af' in df.columns:
        df['log10_af'] = np.log10(df['gnomad4__af'].clip(lower=1e-8))
        df['gnomad4__is_absent'] = (df['gnomad4__af'] == 0).astype(int)
        df['af_bin'] = pd.cut(
            df['gnomad4__af'], bins=[-0.001, 0, 1e-5, 1e-3, 0.01, 1.0],
            labels=[0, 1, 2, 3, 4]).astype(float)
    if all(c in df.columns for c in ['gnomad4__nhomalt', 'gnomad4__ac']):
        df['homozygote_ratio'] = df['gnomad4__nhomalt'] / (df['gnomad4__ac'] / 2).replace(0, np.nan)

    # Conservation product
    if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
        df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']

    # CHASMplus: sadece ana skoru tut
    chasm_cols = get_chasmplus_subtypes(df.columns)
    if chasm_cols:
        df.drop(columns=chasm_cols, inplace=True)

    # Arac konsensus (UYARI: circularity tasiyor)
    pred_cols_map = {
        'esm1b__prediction': ['Damaging'], 'metalr__pred': ['Damaging'],
        'metarnn__pred': ['Damaging'], 'metasvm__pred': ['Damaging'],
        'mistic__pred': ['Pathogenic'], 'mutationtaster__prediction': ['Damaging'],
        'phdsnpg__prediction': ['Pathogenic'], 'provean__prediction': ['Damaging'],
        'sift__prediction': ['Damaging'],
        'alphamissense__am_class': ['likely_pathogenic', 'pathogenic'],
    }
    consensus_sum = pd.Series(0, index=df.index)
    tool_count = 0
    for col, labels in pred_cols_map.items():
        if col in df.columns:
            consensus_sum += df[col].isin(labels).astype(int)
            tool_count += 1
    df['tool_consensus_damaging'] = consensus_sum
    df['tool_consensus_ratio'] = consensus_sum / max(tool_count, 1)

    # Gereksiz sutunlari dusur
    seq_cols = SEQUENCE_COLS
    df.drop(columns=[c for c in seq_cols if c in df.columns], inplace=True)

    pred_drop = list(pred_cols_map.keys()) + ['alphamissense__protein_variant']
    df.drop(columns=[c for c in pred_drop if c in df.columns], inplace=True)

    rankscore_cols = [c for c in df.columns if '__rankscore' in c]
    df.drop(columns=rankscore_cols, inplace=True)

    id_cols = ['base__cchange', 'base__achange', 'base__chrom', 'base__pos']
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    # Varyant degisim encoding (ref/alt -> one-hot)
    df = _encode_variant_changes(df)

    # Filtreleme
    df = _filter_correlated(df, threshold=0.99)
    df = _filter_zero_variance(df)
    df = _categorize_objects(df)

    print(f"[With-FE] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# MOD 3: CLEAN (Temiz baseline -- tum sizinti kaldirilir)
# ============================================================================

def prepare_data_clean(df):
    """
    Temiz baseline: TUM meta-predictor skorlari, rankscore'lar, gnomAD kaldirilir.
    Sadece guvenli biyolojik ozellikler + iyilestirilmis k-mer korunur.
    """
    df = df.copy()

    # Tum sizintili ve gereksiz sutunlari toplu kaldir
    cols_to_drop = [c for c in ALL_LEAKY_COLS if c in df.columns]
    # CHASMplus alt tipleri de kaldir
    cols_to_drop += [c for c in get_chasmplus_subtypes(df.columns)]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  {len(cols_to_drop)} sizintili/gereksiz sutun kaldirildi")

    # Boolean -> int
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Varyant degisim encoding (ref/alt -> one-hot)
    df = _encode_variant_changes(df)

    # K-mer kodlama (2-mer, protein icin 500, min_df=2)
    df = _encode_kmer(df, SEQUENCE_COLS, max_features=500, min_df=2)

    # Object -> category
    df = _categorize_objects(df)

    print(f"[Clean] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# MOD 4: CLEAN-FE (Temiz + gelismis ozellik muhendisligi)
# Siralama: Once FE ozelliklerini cikar (ref_amino vb. lazim), sonra sizintili sutunlari drop et.
# ============================================================================

def prepare_data_clean_fe(df):
    """
    Temiz veri + gelismis biyoinformatik FE.
    Siralama: Once FE ozelliklerini cikar (ref_amino vb. lazim), sonra sizintili sutunlari drop et.
    """
    df = df.copy()

    # --- ADIM 1: Boolean -> int ---
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- ADIM 2: Sekans tabanli ozellikler (drop oncesi) ---
    if 'DNA_11mer_Ref' in df.columns:
        df['is_cpg_site'] = df['DNA_11mer_Ref'].apply(
            lambda s: int(s[5:7] == 'CG' or s[4:6] == 'CG') if isinstance(s, str) and len(s) >= 7 else 0)
        df['gc_content_11mer'] = df['DNA_11mer_Ref'].apply(
            lambda s: (s.count('G') + s.count('C')) / len(s) if isinstance(s, str) and len(s) > 0 else 0.5)

    # Transition vs Transversion
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    # --- ADIM 3: Amino asit substitusyon ozellikleri ---
    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        # Grantham Distance (simetrik -- biyokimyasal mesafe)
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        df['grantham_category'] = pd.cut(
            df['grantham_distance'], bins=[-1, 50, 100, 150, 300],
            labels=[0, 1, 2, 3]).astype(float)

        # BLOSUM62 substitusyon skoru
        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)

        # YONLU BLOSUM62: ref self-score ve delta
        df['blosum62_ref_self'] = df['ref_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        df['blosum62_alt_self'] = df['alt_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        # Delta: ne kadar kotu bir substitusyon (yuksek = daha kotu)
        df['blosum62_delta'] = df['blosum62_ref_self'] - df['blosum62_score']

    # --- ADIM 4: Evrimsel korunum ozellikleri ---
    conservation_cols = ['gerp__gerp_nr', 'gerp__gerp_rs',
                         'phastcons__phastcons100_vert', 'phylop__phylop100_vert']
    existing_cons = [c for c in conservation_cols if c in df.columns]

    if len(existing_cons) >= 2:
        # Conservation product
        if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
            df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']

        # GERP x PhyloP etkilesim
        if all(c in df.columns for c in ['gerp__gerp_rs', 'phylop__phylop100_vert']):
            df['gerp_x_phylop'] = df['gerp__gerp_rs'] * df['phylop__phylop100_vert']

        # PCA: 4 korunum skoru -> 2 temel bilesen
        if len(existing_cons) >= 3:
            cons_data = df[existing_cons].fillna(0)
            n_components = min(2, len(existing_cons))
            pca = PCA(n_components=n_components, random_state=42)
            pca_result = pca.fit_transform(cons_data)
            for i in range(n_components):
                df[f'conservation_pc{i+1}'] = pca_result[:, i]
            print(f"  Conservation PCA: {len(existing_cons)} skor -> {n_components} PC "
                  f"(varyans: {pca.explained_variance_ratio_.sum():.2%})")

    # --- ADIM 5: Delta k-mer (Ref - Alt farki) ---
    ref_alt_pairs = [
        ('DNA_11mer_Ref', 'DNA_11mer_Alt'),
        ('Prot_11mer_Ref', 'Prot_11mer_Alt'),
    ]
    # Sadece mevcut ciftleri isle
    valid_pairs = [(r, a) for r, a in ref_alt_pairs if r in df.columns and a in df.columns]
    if valid_pairs:
        df = _encode_kmer_with_delta(df, valid_pairs, max_features=500, min_df=2)

    # --- ADIM 6: Varyant degisim encoding (FE'den sonra, drop'tan once) ---
    df = _encode_variant_changes(df)

    # --- ADIM 7: Sizintili sutunlari SIMDI drop et ---
    cols_to_drop = [c for c in ALL_LEAKY_COLS if c in df.columns]
    cols_to_drop += get_chasmplus_subtypes(df.columns)
    # Kalan sekans sutunlarini da drop et (k-mer'e donusturulenler haric)
    remaining_seq = [c for c in SEQUENCE_COLS if c in df.columns]
    cols_to_drop += remaining_seq
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  {len(cols_to_drop)} sizintili/gereksiz sutun kaldirildi")

    # --- ADIM 8: Filtreleme ---
    df = _filter_correlated(df, threshold=0.99)
    df = _filter_zero_variance(df)
    df = _categorize_objects(df)

    print(f"[Clean-FE] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# MOD 5: FULL-FE (In-silico skorlar KORUNUR + gelismis FE)
# Yarismada in-silico skorlar verilecegi icin kaldirmiyoruz.
# ============================================================================

def prepare_data_full_fe(df):
    """
    Tam ozellik seti: In-silico meta-predictor skorlari + gelismis biyoinformatik FE.
    Sadece direct leak (clinvar_sig, Stars) ve gereksiz metadata kaldirilir.
    """
    df = df.copy()

    # Direct leak sutunlari kaldir (target'in kendisi)
    df.drop(columns=[c for c in DIRECT_LEAK_COLS if c in df.columns], inplace=True)

    # Kullanilmayacak uzun sekans sutunlari (Ref_Sequence, Alt_Sequence)
    df.drop(columns=[c for c in DROP_SEQUENCE_COLS if c in df.columns], inplace=True)

    # Boolean -> int
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- Sekans tabanli ozellikler ---
    if 'DNA_11mer_Ref' in df.columns:
        df['is_cpg_site'] = df['DNA_11mer_Ref'].apply(
            lambda s: int(s[5:7] == 'CG' or s[4:6] == 'CG') if isinstance(s, str) and len(s) >= 7 else 0)
        df['gc_content_11mer'] = df['DNA_11mer_Ref'].apply(
            lambda s: (s.count('G') + s.count('C')) / len(s) if isinstance(s, str) and len(s) > 0 else 0.5)

    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A','G'), ('G','A'), ('C','T'), ('T','C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    # --- Amino asit substitusyon ozellikleri ---
    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        df['grantham_category'] = pd.cut(
            df['grantham_distance'], bins=[-1, 50, 100, 150, 300],
            labels=[0, 1, 2, 3]).astype(float)

        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)

        df['blosum62_ref_self'] = df['ref_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        df['blosum62_alt_self'] = df['alt_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        df['blosum62_delta'] = df['blosum62_ref_self'] - df['blosum62_score']

    # --- gnomAD turevleri ---
    if all(c in df.columns for c in ['gnomad4__ac', 'gnomad4__an']):
        df['gnomad4__af_computed'] = df['gnomad4__ac'] / df['gnomad4__an'].replace(0, np.nan)
    if 'gnomad4__af' in df.columns:
        df['log10_af'] = np.log10(df['gnomad4__af'].clip(lower=1e-8))
        df['gnomad4__is_absent'] = (df['gnomad4__af'] == 0).astype(int)
        df['af_bin'] = pd.cut(
            df['gnomad4__af'], bins=[-0.001, 0, 1e-5, 1e-3, 0.01, 1.0],
            labels=[0, 1, 2, 3, 4]).astype(float)
    if all(c in df.columns for c in ['gnomad4__nhomalt', 'gnomad4__ac']):
        df['homozygote_ratio'] = df['gnomad4__nhomalt'] / (df['gnomad4__ac'] / 2).replace(0, np.nan)

    # --- Evrimsel korunum ozellikleri ---
    conservation_cols = ['gerp__gerp_nr', 'gerp__gerp_rs',
                         'phastcons__phastcons100_vert', 'phylop__phylop100_vert']
    existing_cons = [c for c in conservation_cols if c in df.columns]

    if len(existing_cons) >= 2:
        if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
            df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']
        if all(c in df.columns for c in ['gerp__gerp_rs', 'phylop__phylop100_vert']):
            df['gerp_x_phylop'] = df['gerp__gerp_rs'] * df['phylop__phylop100_vert']
        if len(existing_cons) >= 3:
            cons_data = df[existing_cons].fillna(0)
            n_components = min(2, len(existing_cons))
            pca = PCA(n_components=n_components, random_state=42)
            pca_result = pca.fit_transform(cons_data)
            for i in range(n_components):
                df[f'conservation_pc{i+1}'] = pca_result[:, i]
            print(f"  Conservation PCA: {len(existing_cons)} skor -> {n_components} PC "
                  f"(varyans: {pca.explained_variance_ratio_.sum():.2%})")

    # --- Delta k-mer ---
    ref_alt_pairs = [
        ('DNA_11mer_Ref', 'DNA_11mer_Alt'),
        ('Prot_11mer_Ref', 'Prot_11mer_Alt'),
    ]
    valid_pairs = [(r, a) for r, a in ref_alt_pairs if r in df.columns and a in df.columns]
    if valid_pairs:
        df = _encode_kmer_with_delta(df, valid_pairs, max_features=500, min_df=2)

    # --- Varyant degisim encoding ---
    df = _encode_variant_changes(df)

    # --- Gereksiz sutunlari drop et (in-silico skorlar KORUNUYOR) ---
    meta_drop = [c for c in METADATA_COLS if c in df.columns]
    meta_drop += get_chasmplus_subtypes(df.columns)
    meta_drop += [c for c in RANKSCORE_COLS if c in df.columns]
    meta_drop += [c for c in LEAKY_META_PREDICTOR_PREDS if c in df.columns]
    remaining_seq = [c for c in SEQUENCE_COLS if c in df.columns]
    meta_drop += remaining_seq
    df.drop(columns=meta_drop, inplace=True)
    print(f"  {len(meta_drop)} gereksiz sutun kaldirildi (in-silico skorlar KORUNDU)")

    # --- Filtreleme ---
    df = _filter_correlated(df, threshold=0.99)
    df = _filter_zero_variance(df)
    df = _categorize_objects(df)

    print(f"[Full-FE] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# V3 YARDIMCI FONKSIYONLAR
# ============================================================================

def _label_encode_predictions(df):
    """Kategorik prediction sutunlarini LabelEncoder ile integer'a cevir."""
    for col in LABEL_ENC_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"  LabelEncode {col}: {len(le.classes_)} sinif")
    return df


def _encode_ohe_separate(df):
    """
    Varyant sutunlarini AYRI one-hot encode et (sabit alfabe).
    base__ref_base → 4 binary, base__alt_base → 4 binary,
    ref_amino → 20 binary, alt_amino → 20 binary = 48 feature
    """
    ohe_parts = []
    for col, cats in OHE_COLS_V3.items():
        if col in df.columns:
            for cat in cats:
                col_name = f"{col}__{cat}"
                ohe_parts.append((df[col] == cat).astype(float).rename(col_name))
    if ohe_parts:
        ohe_df = pd.concat(ohe_parts, axis=1)
        # Kaynak sutunlari drop et
        src_cols = [c for c in OHE_COLS_V3.keys() if c in df.columns]
        df.drop(columns=src_cols, inplace=True)
        df = pd.concat([df, ohe_df], axis=1)
        print(f"  OHE (ayri alfabe): {ohe_df.shape[1]} feature")
    return df


def _build_kmer_vocab(sequences, k, alphabet):
    """
    K-mer vocabularyi olustur.
    DNA: tam kombinatorik (4^k). Protein: sadece gozlenen k-mer'ler.
    """
    if set(alphabet) <= set('ACGT'):
        # DNA: tam kombinatorik
        return [''.join(p) for p in itertools_product(alphabet, repeat=k)]
    # Protein: gozlenen vocab
    vocab = set()
    for seq in sequences.dropna():
        seq = str(seq).upper()
        for i in range(len(seq) - k + 1):
            vocab.add(seq[i:i + k])
    return sorted(vocab)


def _encode_kmer_col_manual(series, k, vocab):
    """Tek bir sekans sutununu k-mer frekanslarina donustur (manuel sayim)."""
    rows = []
    for seq in series:
        seq = str(seq).upper() if isinstance(seq, str) else ''
        counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
        rows.append({km: counts.get(km, 0) for km in vocab})
    return pd.DataFrame(rows, index=series.index)


def _encode_kmer_manual(df, k=2):
    """
    Tum 11-mer sekans sutunlarini manuel k-mer encoding ile donustur.
    DNA: tam kombinatorik vocab. Protein: gozlenen vocab.
    """
    dna_cols = ['DNA_11mer_Ref', 'DNA_11mer_Alt']
    prot_cols = ['Prot_11mer_Ref', 'Prot_11mer_Alt']
    parts = []

    for col, alphabet in [(c, DNA_ALPHABET) for c in dna_cols] + \
                          [(c, PROT_ALPHABET) for c in prot_cols]:
        if col not in df.columns:
            continue
        vocab = _build_kmer_vocab(df[col], k, alphabet)
        kmer_df = _encode_kmer_col_manual(df[col], k, vocab)
        kmer_df.columns = [f"{col}__{k}mer_{v}" for v in vocab]
        parts.append(kmer_df)
        print(f"  K-mer {col}: {len(vocab)} {k}-mer feature")

    # Orijinal sekans text sutunlarini drop et
    seq_drop = [c for c in dna_cols + prot_cols if c in df.columns]
    df.drop(columns=seq_drop, inplace=True)

    if parts:
        df = pd.concat([df] + parts, axis=1)

    return df


# ============================================================================
# MOD 6: V3 (Optimize edilmis pipeline — danisman yaklasimi + secilmis FE)
# ============================================================================

def prepare_data_v3(df):
    """
    Optimize edilmis pipeline:
    - In-silico skorlar KORUNUR
    - Prediction label'lari LABEL-ENCODE edilir (droplanmaz)
    - Ayri one-hot encoding (48 feature)
    - Manuel k-mer encoding (tam kombinatorik DNA, gozlenen protein)
    - Secilmis FE feature'lari (kanitlanmis faydalilari)
    - Korelasyon + sifir varyans filtresi
    """
    df = df.copy()

    # --- ADIM 1: Boolean → int ---
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- ADIM 2: Label-encode prediction sutunlari ---
    df = _label_encode_predictions(df)

    # --- ADIM 3: FE feature'lari (drop oncesi, ref_amino vb. lazim) ---

    # Transition vs Transversion
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    # Grantham Distance
    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        # BLOSUM62
        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)

        df['blosum62_ref_self'] = df['ref_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        df['blosum62_alt_self'] = df['alt_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)

        df['blosum62_delta'] = df['blosum62_ref_self'] - df['blosum62_score']

    # Conservation interaction terms
    if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
        df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']
    if all(c in df.columns for c in ['gerp__gerp_rs', 'phylop__phylop100_vert']):
        df['gerp_x_phylop'] = df['gerp__gerp_rs'] * df['phylop__phylop100_vert']

    # --- ADIM 4: One-hot encoding (ayri, sabit alfabe — 48 feature) ---
    df = _encode_ohe_separate(df)

    # --- ADIM 5: Manuel k-mer encoding ---
    df = _encode_kmer_manual(df, k=2)

    # --- ADIM 6: Gereksiz sutunlari drop et ---
    cols_to_drop = []
    # Direct leak
    cols_to_drop += [c for c in DIRECT_LEAK_COLS if c in df.columns]
    # Metadata (base__hugo dahil)
    cols_to_drop += [c for c in METADATA_COLS if c in df.columns]
    # Uzun sekans sutunlari
    cols_to_drop += [c for c in DROP_SEQUENCE_COLS if c in df.columns]
    # Rankscore'lar (ham skorlar yeterli)
    cols_to_drop += [c for c in RANKSCORE_COLS if c in df.columns]
    # CHASMplus subtypes
    cols_to_drop += get_chasmplus_subtypes(df.columns)
    # Kalan sekans text sutunlari (zaten k-mer'e donusturuldu)
    cols_to_drop += [c for c in SEQUENCE_COLS if c in df.columns]
    # Deduplicate
    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  {len(cols_to_drop)} gereksiz sutun kaldirildi")

    # --- ADIM 7: NaN → 0, inf → 0, float cast ---
    # Panel'i koru, diger object sutunlari drop et
    panel_backup = df['Panel'].copy() if 'Panel' in df.columns else None
    obj_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Panel']
    if obj_cols:
        print(f"  Uyari: object sutunlar droplaniyor: {obj_cols}")
        df.drop(columns=obj_cols, inplace=True)
    # Panel'i simdilik cikar, numeric islemlerden sonra geri ekle
    if 'Panel' in df.columns:
        df.drop(columns=['Panel'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df = df.astype(float)
    # Panel'i geri ekle
    if panel_backup is not None:
        df['Panel'] = panel_backup.values

    # --- ADIM 8: Korelasyon filtresi (r > 0.99) ---
    df = _filter_correlated(df, threshold=0.99)

    # --- ADIM 9: Sifir varyans filtresi ---
    df = _filter_zero_variance(df)

    print(f"[V3] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


def prepare_data_v3_no_pred(df):
    """
    V3 pipeline'inin prediction label'siz versiyonu.
    LABEL_ENC_COLS (esm1b__prediction, metalr__pred, vb.) DROPLANIYOR.
    Geri kalan her sey prepare_data_v3 ile ayni.
    """
    df = df.copy()

    # --- ADIM 1: Boolean → int ---
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- ADIM 2: Prediction sutunlarini DROP ET (label-encode YAPMA) ---
    pred_to_drop = [c for c in LABEL_ENC_COLS if c in df.columns]
    if pred_to_drop:
        df.drop(columns=pred_to_drop, inplace=True)
        print(f"  {len(pred_to_drop)} prediction sutunu droplandi: {pred_to_drop}")

    # --- ADIM 3: FE feature'lari ---
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df['blosum62_ref_self'] = df['ref_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)
        df['blosum62_alt_self'] = df['alt_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)
        df['blosum62_delta'] = df['blosum62_ref_self'] - df['blosum62_score']

    if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
        df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']
    if all(c in df.columns for c in ['gerp__gerp_rs', 'phylop__phylop100_vert']):
        df['gerp_x_phylop'] = df['gerp__gerp_rs'] * df['phylop__phylop100_vert']

    # --- ADIM 4: One-hot encoding ---
    df = _encode_ohe_separate(df)

    # --- ADIM 5: Manuel k-mer encoding ---
    df = _encode_kmer_manual(df, k=2)

    # --- ADIM 6: Gereksiz sutunlari drop et ---
    cols_to_drop = []
    cols_to_drop += [c for c in DIRECT_LEAK_COLS if c in df.columns]
    cols_to_drop += [c for c in METADATA_COLS if c in df.columns]
    cols_to_drop += [c for c in DROP_SEQUENCE_COLS if c in df.columns]
    cols_to_drop += [c for c in RANKSCORE_COLS if c in df.columns]
    cols_to_drop += get_chasmplus_subtypes(df.columns)
    cols_to_drop += [c for c in SEQUENCE_COLS if c in df.columns]
    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  {len(cols_to_drop)} gereksiz sutun kaldirildi")

    # --- ADIM 7: NaN → 0, inf → 0, float cast ---
    panel_backup = df['Panel'].copy() if 'Panel' in df.columns else None
    obj_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Panel']
    if obj_cols:
        print(f"  Uyari: object sutunlar droplaniyor: {obj_cols}")
        df.drop(columns=obj_cols, inplace=True)
    if 'Panel' in df.columns:
        df.drop(columns=['Panel'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df = df.astype(float)
    if panel_backup is not None:
        df['Panel'] = panel_backup.values

    # --- ADIM 8: Korelasyon filtresi (r > 0.99) ---
    df = _filter_correlated(df, threshold=0.99)

    # --- ADIM 9: Sifir varyans filtresi ---
    df = _filter_zero_variance(df)

    print(f"[V3-NoPred] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


def prepare_data_v3_no_insil(df):
    """
    V3 pipeline'inin in-silico skorlar OLMADAN versiyonu.
    LEAKY_META_PREDICTOR_SCORES + LEAKY_META_PREDICTOR_PREDS + CHASMplus subtypes
    TAMAMI DROPLANIYOR.
    Conservation skorlari (GERP, PhastCons, PhyloP) KORUNUYOR.
    """
    df = df.copy()

    # --- ADIM 1: Boolean → int ---
    for col in ['polarity_change', 'chirality_shift']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- ADIM 2: In-silico skorlari ve prediction'lari DROP ET ---
    insil_to_drop = []
    insil_to_drop += [c for c in LEAKY_META_PREDICTOR_SCORES if c in df.columns]
    insil_to_drop += [c for c in LEAKY_META_PREDICTOR_PREDS if c in df.columns]
    insil_to_drop += get_chasmplus_subtypes(df.columns)
    insil_to_drop = list(set(insil_to_drop))
    if insil_to_drop:
        df.drop(columns=insil_to_drop, inplace=True)
        print(f"  {len(insil_to_drop)} in-silico sutunu droplandi")

    # --- ADIM 3: FE feature'lari ---
    if all(c in df.columns for c in ['base__ref_base', 'base__alt_base']):
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        df['is_transition'] = df.apply(
            lambda r: int((r['base__ref_base'], r['base__alt_base']) in transitions), axis=1)

    if all(c in df.columns for c in ['ref_amino', 'alt_amino']):
        df['grantham_distance'] = df.apply(
            lambda r: GRANTHAM.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df.loc[df['ref_amino'] == df['alt_amino'], 'grantham_distance'] = 0

        df['blosum62_score'] = df.apply(
            lambda r: BLOSUM62.get((r['ref_amino'], r['alt_amino']), np.nan)
            if isinstance(r['ref_amino'], str) and isinstance(r['alt_amino'], str)
            and len(r['ref_amino']) == 1 and len(r['alt_amino']) == 1
            else np.nan, axis=1)
        df['blosum62_ref_self'] = df['ref_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)
        df['blosum62_alt_self'] = df['alt_amino'].apply(
            lambda aa: BLOSUM62.get((aa, aa), np.nan)
            if isinstance(aa, str) and len(aa) == 1 else np.nan)
        df['blosum62_delta'] = df['blosum62_ref_self'] - df['blosum62_score']

    if all(c in df.columns for c in ['phylop__phylop100_vert', 'phastcons__phastcons100_vert']):
        df['conservation_product'] = df['phylop__phylop100_vert'] * df['phastcons__phastcons100_vert']
    if all(c in df.columns for c in ['gerp__gerp_rs', 'phylop__phylop100_vert']):
        df['gerp_x_phylop'] = df['gerp__gerp_rs'] * df['phylop__phylop100_vert']

    # --- ADIM 4: One-hot encoding ---
    df = _encode_ohe_separate(df)

    # --- ADIM 5: Manuel k-mer encoding ---
    df = _encode_kmer_manual(df, k=2)

    # --- ADIM 6: Gereksiz sutunlari drop et ---
    cols_to_drop = []
    cols_to_drop += [c for c in DIRECT_LEAK_COLS if c in df.columns]
    cols_to_drop += [c for c in METADATA_COLS if c in df.columns]
    cols_to_drop += [c for c in DROP_SEQUENCE_COLS if c in df.columns]
    cols_to_drop += [c for c in RANKSCORE_COLS if c in df.columns]
    cols_to_drop += [c for c in SEQUENCE_COLS if c in df.columns]
    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"  {len(cols_to_drop)} gereksiz sutun kaldirildi")

    # --- ADIM 7: NaN → 0, inf → 0, float cast ---
    panel_backup = df['Panel'].copy() if 'Panel' in df.columns else None
    obj_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Panel']
    if obj_cols:
        print(f"  Uyari: object sutunlar droplaniyor: {obj_cols}")
        df.drop(columns=obj_cols, inplace=True)
    if 'Panel' in df.columns:
        df.drop(columns=['Panel'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df = df.astype(float)
    if panel_backup is not None:
        df['Panel'] = panel_backup.values

    # --- ADIM 8: Korelasyon filtresi (r > 0.99) ---
    df = _filter_correlated(df, threshold=0.99)

    # --- ADIM 9: Sifir varyans filtresi ---
    df = _filter_zero_variance(df)

    print(f"[V3-NoInSil] Veri boyutu: {df.shape[0]} x {df.shape[1]}")
    return df


# ============================================================================
# DISPATCHER: FE moduna gore fonksiyon sec
# ============================================================================

FE_PREPARERS = {
    'no_fe': prepare_data_no_fe,
    'with_fe': prepare_data_with_fe,
    'clean': prepare_data_clean,
    'clean_fe': prepare_data_clean_fe,
    'full_fe': prepare_data_full_fe,
    'v3': prepare_data_v3,
    'v3_no_pred': prepare_data_v3_no_pred,
    'v3_no_insil': prepare_data_v3_no_insil,
}


def prepare_data(df, fe_mode):
    """
    Verilen FE moduna gore veri hazirlama fonksiyonunu calistir.
    """
    if fe_mode not in FE_PREPARERS:
        raise ValueError(f"Bilinmeyen FE modu: {fe_mode}. Gecerli: {list(FE_PREPARERS.keys())}")
    return FE_PREPARERS[fe_mode](df)
