# -*- coding: utf-8 -*-
"""
YARISMA (anonim) -> Legacy OpenCRAVAT GRUP eslemesi.

Uretildi: notebooks/18_deobfuscation.ipynb (v2, grup-tabanli)

ONEMLI: Yarisma degerleri NORMALIZE edilmis oldugu icin BIREBIR isim eslestirmesi
guvenilmez (cadd phred legacy 0-63 / yarisma [0,1]; revel/cadd/alphamissense
dagilim sekliyle ayrismaz). Bu modul birebir isim DEGIL, GUVENILIR GRUP kimligi verir.

Gruplar:
  FREQ_COLS    : populasyon frekansi (gnomAD/AllofUs/1000G) -- [0,1], q50~0, skew yuksek
  SCORE01_COLS : in-silico patojenite skoru (rankscore) -- en bilgilendirici, az sayida
  CONS_RAW_COLS: ham olcekli konservasyon/etki skoru (negatif olabilir)
  BINARY_COLS  : 0/1 flag
  OTHER_COLS   : belirsiz -> Yontem 2'de KULLANILMAZ

LEGACY_*_CANDIDATES: Faz 2b'de OpenCRAVAT annotasyonunda secilecek aday annotator sutunlari.
"""

# --- Yarisma sutunlari, gruba gore ---
FREQ_COLS = ['AL_4', 'AL_6', 'AL_7', 'AL_8', 'AL_9', 'AL_10', 'AL_11', 'AL_12', 'AL_13', 'AL_14', 'AL_15', 'AL_16', 'AL_17', 'AL_18', 'AL_19', 'AL_20', 'AL_21', 'AL_22', 'AL_23', 'AL_24', 'AL_25', 'AL_26', 'AL_34', 'AL_38', 'AL_40', 'AL_43', 'AL_46', 'AL_49', 'AL_52', 'AL_55', 'AL_58', 'AL_61', 'AL_64', 'AL_67', 'AL_70', 'AL_73', 'AL_76', 'AL_79', 'AL_82', 'AL_85', 'AL_88', 'AL_91', 'AL_94', 'AL_97', 'AL_100', 'AL_103', 'AL_106', 'AL_109', 'AL_112', 'AL_115', 'AL_118', 'AL_121', 'AL_124', 'AL_127', 'AL_130', 'AL_133', 'AL_136', 'AL_139', 'AL_142', 'AL_145', 'AL_148', 'AL_151', 'AL_154', 'AL_157', 'AL_160', 'AL_163', 'AL_166', 'AL_169', 'AL_172', 'AL_175', 'AL_178', 'AL_181', 'AL_184', 'AL_186', 'AL_188', 'AL_190', 'AL_194', 'AL_199', 'AL_203', 'AL_207', 'AL_211', 'AL_215', 'AL_219', 'AL_224', 'AL_226', 'AL_230', 'AL_235', 'AL_239', 'AL_243', 'AL_247', 'AL_251', 'AL_255', 'AL_260', 'AL_262', 'AL_266', 'AL_271', 'AL_275', 'AL_279', 'AL_283', 'AL_287', 'AL_291', 'AL_296', 'AL_298', 'AL_302', 'AL_306', 'AL_311', 'AL_315', 'AL_319', 'AL_323', 'AL_327', 'AL_331']

SCORE01_COLS = ['AL_45', 'AL_129', 'AL_201', 'AL_301', 'AL_304', 'EK_5']

CONS_RAW_COLS = ['AL_185', 'EK_1', 'EK_2', 'EK_3', 'EK_7', 'EK_8', 'EK_9']

BINARY_COLS = ['AL_41', 'AL_42', 'AL_44', 'AL_54', 'AL_56', 'AL_59', 'AL_72', 'AL_74', 'AL_77', 'AL_78', 'AL_80', 'AL_83', 'AL_86', 'AL_89', 'AL_92', 'AL_93', 'AL_95', 'AL_99', 'AL_101', 'AL_102', 'AL_104', 'AL_107', 'AL_110', 'AL_111', 'AL_113', 'AL_116', 'AL_119', 'AL_120', 'AL_122', 'AL_125', 'AL_128', 'AL_131', 'AL_134', 'AL_135', 'AL_137', 'AL_138', 'AL_140', 'AL_143', 'AL_144', 'AL_146', 'AL_149', 'AL_152', 'AL_153', 'AL_155', 'AL_158', 'AL_161', 'AL_164', 'AL_167', 'AL_170', 'AL_173', 'AL_176', 'AL_177', 'AL_179', 'AL_182', 'AL_183', 'AL_191', 'AL_195', 'AL_197', 'AL_200', 'AL_204', 'AL_208', 'AL_212', 'AL_216', 'AL_220', 'AL_227', 'AL_231', 'AL_233', 'AL_236', 'AL_240', 'AL_244', 'AL_246', 'AL_248', 'AL_252', 'AL_253', 'AL_256', 'AL_258', 'AL_259', 'AL_263', 'AL_267', 'AL_269', 'AL_272', 'AL_276', 'AL_280', 'AL_282', 'AL_284', 'AL_288', 'AL_292', 'AL_295', 'AL_299', 'AL_303', 'AL_305', 'AL_307', 'AL_309', 'AL_312', 'AL_316', 'AL_320', 'AL_322', 'AL_324', 'AL_325', 'AL_326', 'AL_328', 'AL_332', 'AL_334']

OTHER_COLS = ['AL_1', 'AL_2', 'AL_3', 'AL_5', 'AL_27', 'AL_28', 'AL_29', 'AL_30', 'AL_31', 'AL_32', 'AL_33', 'AL_35', 'AL_36', 'AL_37', 'AL_39', 'AL_47', 'AL_48', 'AL_50', 'AL_51', 'AL_53', 'AL_57', 'AL_60', 'AL_62', 'AL_63', 'AL_65', 'AL_66', 'AL_68', 'AL_69', 'AL_71', 'AL_75', 'AL_81', 'AL_84', 'AL_87', 'AL_90', 'AL_96', 'AL_98', 'AL_105', 'AL_108', 'AL_114', 'AL_117', 'AL_123', 'AL_126', 'AL_132', 'AL_141', 'AL_147', 'AL_150', 'AL_156', 'AL_159', 'AL_162', 'AL_165', 'AL_168', 'AL_171', 'AL_174', 'AL_180', 'AL_187', 'AL_189', 'AL_192', 'AL_193', 'AL_196', 'AL_198', 'AL_202', 'AL_205', 'AL_206', 'AL_209', 'AL_210', 'AL_213', 'AL_214', 'AL_217', 'AL_218', 'AL_221', 'AL_222', 'AL_223', 'AL_225', 'AL_228', 'AL_229', 'AL_232', 'AL_234', 'AL_237', 'AL_238', 'AL_241', 'AL_242', 'AL_245', 'AL_249', 'AL_250', 'AL_254', 'AL_257', 'AL_261', 'AL_264', 'AL_265', 'AL_268', 'AL_270', 'AL_273', 'AL_274', 'AL_277', 'AL_278', 'AL_281', 'AL_285', 'AL_286', 'AL_289', 'AL_290', 'AL_293', 'AL_294', 'AL_297', 'AL_300', 'AL_308', 'AL_310', 'AL_313', 'AL_314', 'AL_317', 'AL_318', 'AL_321', 'AL_329', 'AL_330', 'AL_333', 'EK_4', 'EK_6']

# Yontem 2 icin guvenilir feature havuzu (OTHER haric)
RELIABLE_COLS = FREQ_COLS + SCORE01_COLS + CONS_RAW_COLS

# --- Legacy aday annotator sutunlari (Faz 2b annotasyon secimi icin) ---
LEGACY_FREQ_CANDIDATES = ['abraom__allele_freq', 'alfa__total_freq', 'alfa_asian__east_freq', 'alfa_asian__south_freq', 'alfa_asian__other_freq', 'alfa_asian__asian_freq', 'alfa_european__european_freq', 'alfa_latin_american__latin1_freq', 'alfa_latin_american__latin2_freq', 'alfa_other__other_freq', 'clinvar__af_go_esp', 'clinvar__af_exac', 'clinvar__af_tgp', 'gnomad__af', 'gnomad__af_afr', 'gnomad__af_amr', 'gnomad__af_asj', 'gnomad__af_eas', 'gnomad__af_fin', 'gnomad__af_nfe', 'gnomad__af_oth', 'gnomad__af_sas', 'gnomad3__af', 'gnomad3__af_afr', 'gnomad3__af_asj', 'gnomad3__af_eas', 'gnomad3__af_fin', 'gnomad3__af_lat', 'gnomad3__af_nfe', 'gnomad3__af_oth', 'gnomad3__af_sas', 'gnomad4__af', 'gnomad4__an', 'gnomad4__ac', 'gnomad4__af_afr', 'gnomad4__an_afr', 'gnomad4__ac_afr', 'gnomad4__af_ami', 'gnomad4__an_ami', 'gnomad4__ac_ami', 'gnomad4__af_amr', 'gnomad4__an_amr', 'gnomad4__ac_amr', 'gnomad4__af_asj', 'gnomad4__an_asj', 'gnomad4__ac_asj', 'gnomad4__af_eas', 'gnomad4__an_eas', 'gnomad4__ac_eas', 'gnomad4__af_fin', 'gnomad4__an_fin', 'gnomad4__ac_fin', 'gnomad4__af_mid', 'gnomad4__an_mid', 'gnomad4__ac_mid', 'gnomad4__af_nfe', 'gnomad4__an_nfe', 'gnomad4__ac_nfe', 'gnomad4__af_sas', 'gnomad4__an_sas', 'gnomad4__ac_sas', 'gnomad4__af_rem', 'gnomad4__an_rem', 'gnomad4__ac_rem', 'thousandgenomes__af', 'thousandgenomes__afr_af', 'thousandgenomes_african__acb_af']

LEGACY_SCORE01_CANDIDATES = ['bayesdel__bayesdel_addAF_rankscore', 'bayesdel__bayesdel_noAF_rankscore', 'clinpred__rankscore', 'cscape_coding__rankscore', 'dann_coding__dann_rankscore', 'esm1b__rankscore', 'eve__rank_score', 'fathmm_mkl__fathmm_mkl_coding_rankscore', 'fathmm_xf_coding__fathmm_xf_coding_rankscore', 'fitcons__fitcons_coding_rankscore', 'genocanyon__rankscore', 'gmvp__rank_score', 'lrt__lrt_converted_rankscore', 'metalr__rankscore', 'metarnn__rank_score', 'metasvm__rankscore', 'mutation_assessor__rankscore', 'mutationtaster__rankscore', 'mutpred1__mutpred_rankscore', 'mutpred2__rankscore', 'primateai__primateai_rankscore', 'provean__rankscore', 'revel__rankscore', 'sift__rankscore']

LEGACY_SCORE_CANDIDATES = ['alphamissense__am_pathogenicity', 'bayesdel__bayesdel_addAF_score', 'bayesdel__bayesdel_noAF_score', 'cadd__score', 'cadd__phred', 'cadd_exome__score', 'cadd_exome__phred', 'clinpred__score', 'dann__score', 'dann_coding__dann_coding_score', 'esm1b__score', 'eve__score', 'fathmm_mkl__fathmm_mkl_coding_score', 'fathmm_xf__score', 'fathmm_xf_coding__fathmm_xf_coding_score', 'metalr__score', 'metarnn__score', 'metasvm__score', 'mutationtaster__score', 'polyphen2__hdiv_rank', 'polyphen2__hvar_rank', 'primateai__primateai_score', 'provean__score', 'revel__score', 'sift__score', 'sift__med', 'sift__seqs', 'sift__multsite', 'vest__score', 'vest__pval']

LEGACY_CONS_CANDIDATES = ['ccr__pct', 'ccr__syn_density', 'ccr__cpg', 'ccr__cov_score', 'ccr__resid', 'ccr__resid_pct', 'fitcons__fitcons_coding_score', 'fitcons__fitcons_coding_pred', 'gerp__gerp_nr', 'gerp__gerp_rs', 'gerp__gerp_rs_rank', 'linsight__value', 'phastcons__phastcons100_vert', 'phastcons__phastcons100_vert_r', 'phastcons__phastcons470_mamm', 'phastcons__phastcons470_mamm_r', 'phastcons__phastcons17way_primate', 'phastcons__phastcons17way_primate_r', 'phylop__phylop100_vert', 'phylop__phylop100_vert_r', 'phylop__phylop470_mamm', 'phylop__phylop470_mamm_r', 'phylop__phylop17_primate', 'phylop__phylop17_primate_r', 'siphy__logodds', 'siphy__logodds_rank']
