#!/usr/bin/env python3
"""PAH Panel Gelistirme Raporu — Kapsamli Retrospektif PDF"""

import os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpdf import FPDF
from config import PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

FIGS = {
    "nb21_mcc": RESULTS / "v8_pah_refinement/fig1_loo_mcc.png",
    "nb21_boot": RESULTS / "v8_pah_refinement/fig2_boot_mean.png",
    "nb21_conf": RESULTS / "v8_pah_refinement/fig3_confusion.png",
    "nb21_post": RESULTS / "v8_pah_refinement/fig4_posterior.png",
    "nb22_sweep": RESULTS / "v9_pah_ensemble/fig1_sweep_heatmap.png",
    "nb22_calib": RESULTS / "v9_pah_ensemble/fig2_calibration_effect.png",
    "nb22_stack": RESULTS / "v9_pah_ensemble/fig3_stack_comparison.png",
    "nb22_conf": RESULTS / "v9_pah_ensemble/fig4_best_confusion.png",
    "nb23_pool": RESULTS / "v10_pah_combined/fig1_pool_comparison.png",
    "nb23_boot": RESULTS / "v10_pah_combined/fig2_boot_comparison.png",
    "nb23_conf": RESULTS / "v10_pah_combined/fig3_confusion_best.png",
    "nb24_mcc": RESULTS / "v11_pah_tabpfn/fig1_mcc_comparison.png",
    "nb24_boot": RESULTS / "v11_pah_tabpfn/fig2_bootstrap_f1.png",
    "nb26_boot_box": RESULTS / "v11_pah_final/fig1_multiseed_boot_boxplot.png",
    "nb26_mcc_box": RESULTS / "v11_pah_final/fig2_multiseed_mcc_boxplot.png",
    "nb26_scenario": RESULTS / "v11_pah_final/fig3_scenario_comparison.png",
    "nb26_scatter": RESULTS / "v11_pah_final/fig4_mcc_vs_boot_scatter.png",
    "nb26_fe": RESULTS / "v11_pah_final/fig5_fe_delta.png",
}


class PAHReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "PAH Panel Retrospektif Rapor", align="R")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(41, 65, 122)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {num}. {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 65, 122)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, txt):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, txt)
        self.ln(2)

    def bullet(self, txt):
        self.set_font("Helvetica", "", 9)
        x0 = self.l_margin
        self.set_x(x0)
        self.cell(5, 5, "-")
        self.multi_cell(self.w - self.r_margin - self.get_x(), 5, txt)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - 20) / len(headers)
            col_widths = [w] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(41, 65, 122)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(230, 236, 245)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5.5, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)

    def add_fig(self, key, caption, w=170):
        path = FIGS.get(key)
        if path and path.exists():
            if self.get_y() + 90 > self.h - 20:
                self.add_page()
            x = (self.w - w) / 2
            self.image(str(path), x=x, w=w)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(4)
        else:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, f"[Gorsel bulunamadi: {key}]", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    def add_two_figs(self, key1, cap1, key2, cap2, w=82):
        p1 = FIGS.get(key1)
        p2 = FIGS.get(key2)
        if self.get_y() + 75 > self.h - 20:
            self.add_page()
        x1 = 12
        x2 = 12 + w + 4
        if p1 and p1.exists():
            self.image(str(p1), x=x1, w=w)
        if p2 and p2.exists():
            self.image(str(p2), x=x2, y=self.get_y() - (w * 0.6 if p1 and p1.exists() else 0), w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.cell(w, 4, cap1, align="C")
        self.cell(4, 4, "")
        self.cell(w, 4, cap2, align="C")
        self.ln(6)


def build():
    pdf = PAHReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── KAPAK ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(41, 65, 122)
    pdf.cell(0, 12, "PAH Panel Gelistirme Raporu", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Kapsamli Retrospektif", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "TEKNOFEST Saglikta Yapay Zeka Yarismasi", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Genetik Varyant Patojenite Tahmini", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.cell(0, 7, "NB16 -> NB26 | 6 Notebook | 30+ Deney | 5 Paradigma", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 7, "Tarih: 19 Haziran 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── 1. PROBLEM TANIMI ──
    pdf.add_page()
    pdf.section_title(1, "Problem Tanimi")
    pdf.body_text(
        "PAH (Pulmoner Arteriyel Hipertansiyon) paneli, TEKNOFEST genetik varyant patojenite "
        "tahmini projesindeki en zorlu paneldir. 369 varyant icinden yalnizca 62'si benign sinifina "
        "aittir (307 pathogenic / 62 benign). Bu asiri dengesizlik, modelin benign ogrenmesini "
        "yapisal olarak sinirlar."
    )
    pdf.sub_title("Kritik Kisitlar")
    pdf.bullet("Veri: 369 varyant (307 pathogenic / 62 benign) -- asiri dengesiz")
    pdf.bullet("Final test seti: ~%80 benign / %20 pathogenic (egitimdeki dagilimin TAM TERSI)")
    pdf.bullet("Metrik: Pathogenic (Label=1) F1 @ %80/20 dagilim")
    pdf.bullet("Yapisan zorluk: Sadece 62 benign ornek -- modelin benign ogrenme kapasitesi sinirli")
    pdf.ln(3)
    pdf.body_text(
        "Bu ters dagilim, egitimde calisir gorunen modellerin final test setinde ciddi precision "
        "cokusu yasamasina neden olur. NB15 v1'de %50/50 test F1=0.928 gorulen PAH modeli, "
        "%80/20 bootstrap'te F1=0.489'a dustu -- ~0.44 puanlik cokus."
    )

    # ── 2. DENEY KRONOLOJISI ──
    pdf.add_page()
    pdf.section_title(2, "Deney Kronolojisi (NB16 -> NB26)")

    # NB16
    pdf.sub_title("NB16 -- Baseline Stacking (Baslangic Noktasi)")
    pdf.body_text(
        "Ilk PAH sonuclari NB16'da OOF stacking ile elde edildi. 6 base model (LightGBM, CatBoost, "
        "NN, DNN, NN_ft, DNN_ft) + 2 meta-learner (LR, GBM). Meta=LR net kazandi."
    )
    pdf.add_table(
        ["Model", "MCC", "Boot %80/20 F1", "CI", "Not"],
        [
            ["stack_lr", "--", "0.515", "[0.35-0.57]", "En iyi PAH baseline"],
            ["base catboost", "--", "0.489", "[0.30-0.62]", "Stacking +0.046 katti"],
        ],
        [30, 20, 30, 30, 80],
    )

    # NB21
    pdf.sub_title("NB21 -- PAH Refinement: 7 Strateji (ILK BUYUK SICRAMA)")
    pdf.body_text(
        "NB17'deki CFTR protokolu (LOO-CV + MCC + prior-shift + BalancedBagging) PAH'a transfer "
        "edildi. COMBINED = MASTER(2931) + KANSER(388) + CFTR(111) = 3430 ornek."
    )
    pdf.add_table(
        ["Strateji", "n_train", "MCC", "Boot F1", "Prec", "FP", "FN"],
        [
            ["P4_COMB_BalBag", "3430", "0.529", "0.582", "0.942", "16", "45"],
            ["P2_BalBag", "2931", "0.519", "0.582", "0.942", "16", "46"],
            ["P0c_COMBINED", "3430", "0.497", "0.590", "0.947", "13", "73"],
            ["P0_MASTER", "2931", "0.463", "0.555", "0.945", "13", "85"],
            ["P3_IsotonicCal", "2931", "0.406", "0.560", "0.945", "13", "83"],
        ],
        [30, 20, 20, 22, 20, 15, 15],
    )
    pdf.bullet("+0.067 puan iyilesme (0.515 -> 0.582) -- NB16'dan ciddi sicrama")
    pdf.bullet("BalancedBagging minimax-optimal (Chatterji 2022 NeurIPS) -- PAH verimizde dogrulandi")
    pdf.bullet("COMBINED her zaman MASTER-only'den iyi")
    pdf.bullet("Prior-shift PAH'ta ZARAR veriyor (CFTR'nin tam tersi!) -- kalibre olmayan olasiliklar")
    pdf.ln(2)

    pdf.add_two_figs("nb21_mcc", "Sekil 1: NB21 LOO-MCC karsilastirma", "nb21_boot", "Sekil 2: NB21 Bootstrap %80/20 F1")
    pdf.add_two_figs("nb21_conf", "Sekil 3: NB21 Confusion matrix", "nb21_post", "Sekil 4: NB21 Posterior dagilim")

    # NB22
    pdf.add_page()
    pdf.sub_title("NB22 -- Ensemble + Calibration (Marjinal Iyilesme)")
    pdf.body_text("Uc deney ekseni test edildi: BalancedBagging sweep, Platt kalibrasyon + prior-shift, heterogeneous stacking.")

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Exp 1: BalancedBagging Hyperparameter Sweep (24 config)", new_x="LMARGIN", new_y="NEXT")
    pdf.add_table(
        ["Config", "MCC", "Boot F1"],
        [
            ["n=10, mf=1.0 (en iyi MCC)", "0.536", "0.591"],
            ["n=100, mf=0.7 (en iyi Boot)", "0.529", "0.595"],
            ["n=20, mf=1.0 (NB21 ref)", "0.529", "0.576"],
        ],
        [65, 40, 40],
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Exp 2: Platt Calibration + Prior-Shift Zinciri", new_x="LMARGIN", new_y="NEXT")
    pdf.add_table(
        ["Strateji", "Raw MCC", "Prior MCC", "Delta"],
        [
            ["A) Raw LGBM + Saerens", "0.497", "0.439", "-0.058"],
            ["B) Platt + Saerens", "0.377", "0.429", "+0.052"],
            ["C) CalBB + Saerens", "0.468", "0.452", "-0.016"],
        ],
        [50, 30, 30, 30],
    )
    pdf.bullet("Platt kalibrasyon prior-shift'i FIXLEDI (delta: -0.058 -> +0.052) -- Alexandari 2020 dogrulandi")
    pdf.bullet("Ancak ham MCC dustugu icin pratik net kazanc yok")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Exp 3: Heterogeneous Stacking", new_x="LMARGIN", new_y="NEXT")
    pdf.bullet("3 GBDT ailesi (LGBM/XGB/CatBoost) cok benzer kararlar veriyor -> diversite dusuk")
    pdf.bullet("Stacking beklentiyi karsilamadi -- NB21 P4 hala lider")
    pdf.ln(2)

    pdf.add_two_figs("nb22_sweep", "Sekil 5: NB22 Sweep heatmap", "nb22_calib", "Sekil 6: NB22 Kalibrasyon etkisi")
    pdf.add_two_figs("nb22_stack", "Sekil 7: NB22 Stacking karsilastirma", "nb22_conf", "Sekil 8: NB22 En iyi confusion matrix")

    # NB23
    pdf.add_page()
    pdf.sub_title("NB23 -- COMBINED Havuz Katki Analizi")
    pdf.body_text(
        "Hangi panelin benign'leri PAH'a ne kadar katki sagliyor? 4 farkli egitim havuzu + "
        "BalancedBagging + Platt+Prior-Shift sistematik testi."
    )
    pdf.add_table(
        ["Strateji", "n_train", "MCC", "Boot F1", "Prior-Shift Delta"],
        [
            ["T4_COMB+BalBag", "3430", "0.529", "0.576", "-0.012"],
            ["T3_COMBINED", "3430", "0.497", "0.597", "-0.058"],
            ["T2_MASTER+CFTR", "3042", "0.478", "0.559", "0.000"],
            ["T1_MASTER+KANSER", "3319", "0.465", "0.549", "+0.013"],
            ["T0_MASTER-only", "2931", "0.463", "0.536", "-0.064"],
        ],
        [35, 22, 22, 25, 35],
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, "Havuz Katki Analizi:", new_x="LMARGIN", new_y="NEXT")
    pdf.add_table(
        ["Katki", "MCC Delta", "Yorum"],
        [
            ["KANSER (+120 benign)", "+0.002", "Neredeyse yok"],
            ["CFTR (+21 benign)", "+0.015", "21 benign, 120'den daha degerli!"],
            ["COMBINED (+141 benign)", "+0.034", "Sinerjik toplam"],
        ],
        [45, 30, 70],
    )
    pdf.bullet("Tek-gen paneli (CFTR) feature uzayini zenginlestiriyor; KANSER'in 120 benign'i az bilgi veriyor")
    pdf.bullet("PAH calismasi burada olgunlasti: MCC~0.53, Boot~0.58 kararli plato")
    pdf.ln(2)

    pdf.add_two_figs("nb23_pool", "Sekil 9: NB23 Havuz karsilastirma", "nb23_boot", "Sekil 10: NB23 Bootstrap karsilastirma")
    pdf.add_fig("nb23_conf", "Sekil 11: NB23 En iyi strateji confusion matrix", w=110)

    # NB24
    pdf.add_page()
    pdf.sub_title("NB24 -- Plato-Kirma Denemesi: TabPFN + Label-Shift (NEGATIF SONUC)")
    pdf.body_text(
        "Literatur destekli 5 deney ile platoyu kirma girisimi. TabPFN (Hollmann et al. 2025 Nature), "
        "Saerens EM prior-shift, SplineCalibration, BBSE (Lipton et al. 2018 ICML) test edildi."
    )
    pdf.add_table(
        ["Deney", "MCC", "Boot F1", "Delta vs NB21", "Yorum"],
        [
            ["D3 SplineCal+Prior", "0.503", "0.581", "-0.001", "Baseline'a ESIT"],
            ["D4 BBSE", "0.512", "0.581", "-0.001", "D3 ile ayni"],
            ["D1 TabPFN ham", "0.380", "0.573", "-0.009", "Baseline altinda"],
            ["D2 TabPFN+Saerens", "0.374", "0.560", "-0.022", "Prior yine zarar"],
            ["D2b TabPFN native", "0.362", "0.548", "-0.034", "En dusuk"],
        ],
        [33, 18, 20, 28, 50],
    )
    pdf.bullet("TabPFN (Nature 2025) platoyu KIRAMADI -- 'yeni inductive bias' hipotezi curutuldu")
    pdf.bullet("Chatterji 2022 azinlik-ornegi tavani tezi DENEYSEL DOGRULANDI")
    pdf.bullet("BBSE model kusurunu ifsa etti: confusion matrix yetersiz -> pi tahmini guvenilmez (0.723 vs gercek 0.20)")
    pdf.bullet("Calibrate-then-shift baseline'a esit -- plato gercek, kalibrasyon sorunu degil")
    pdf.ln(2)

    pdf.add_two_figs("nb24_mcc", "Sekil 12: NB24 MCC karsilastirma", "nb24_boot", "Sekil 13: NB24 Bootstrap F1 karsilastirma")

    # NB26
    pdf.add_page()
    pdf.sub_title("NB26 -- PAH Final: Multi-Seed + FE + HyperEnsemble (SON DENEY)")
    pdf.body_text(
        "10 farkli seed ile 6 senaryo test edildi -- en kapsamli ve en guvenilir PAH deneyi. "
        "HyperEnsemble (parSMURF-inspired) ve Feature Engineering etkileri olculdu."
    )
    pdf.add_table(
        ["Senaryo", "MCC mean+/-std", "Boot mean+/-std", "Best MCC", "Best Boot"],
        [
            ["S5_MASTER_PAH_FE", "0.532+/-0.024", "0.516+/-0.018", "0.578", "0.545"],
            ["S2_FE", "0.498+/-0.015", "0.518+/-0.013", "0.505", "0.530"],
            ["S6_Sweep_FE", "0.489+/-0.016", "0.526+/-0.015", "0.504", "0.548"],
            ["S1_Baseline", "0.472+/-0.016", "0.490+/-0.012", "0.491", "0.513"],
            ["S3_HyperEns", "0.458+/-0.021", "0.546+/-0.018", "0.496", "0.581"],
            ["S4_FE_HyperEns", "0.454+/-0.020", "0.544+/-0.017", "0.480", "0.572"],
        ],
        [35, 32, 32, 23, 23],
    )
    pdf.bullet("HyperEnsemble (parSMURF-inspired) platoyu kiramadi -- BalancedBagging'den kotu")
    pdf.bullet("Feature Engineering marjinal katki: S2_FE > S1_Baseline (+0.026 MCC)")
    pdf.bullet("10-seed validasyon sonuclari kararli: std < 0.025 -- plato gercek")
    pdf.bullet("Hicbir senaryo NB21 P4'un Boot=0.582'sini tutarli sekilde gecemedi")
    pdf.ln(2)

    pdf.add_two_figs("nb26_boot_box", "Sekil 14: NB26 Multi-seed Boot boxplot", "nb26_mcc_box", "Sekil 15: NB26 Multi-seed MCC boxplot")
    pdf.add_two_figs("nb26_scenario", "Sekil 16: NB26 Senaryo karsilastirma", "nb26_scatter", "Sekil 17: NB26 MCC vs Boot scatter")
    pdf.add_fig("nb26_fe", "Sekil 18: NB26 Feature Engineering delta etkisi", w=140)

    # ── 3. YOLCULUK OZETI ──
    pdf.add_page()
    pdf.section_title(3, "PAH Yolculugu Ozet Tablosu")
    pdf.add_table(
        ["Notebook", "En Iyi Model", "MCC", "Boot F1", "Temel Katki"],
        [
            ["NB16", "stack_lr", "--", "0.515", "Baseline stacking"],
            ["NB21", "P4_COMB_BalBag", "0.529", "0.582", "BalBag+COMBINED (+0.067)"],
            ["NB22", "Sweep n=10", "0.536", "0.591", "Marjinal sweep"],
            ["NB23", "T4_COMB+BalBag", "0.529", "0.576", "CFTR>KANSER analizi"],
            ["NB24", "D3 SplineCal", "0.503", "0.581", "TabPFN basarisiz"],
            ["NB26", "S5_MASTER_PAH", "0.532", "0.516", "HyperEns basarisiz"],
        ],
        [22, 32, 18, 20, 55],
    )

    # ── 4. BASARISIZ YAKLASIMLAR ──
    pdf.section_title(4, "Denenen ve Basarisiz Olan Yaklasimlar")
    pdf.add_table(
        ["Yaklasim", "Neden Basarisiz", "Referans"],
        [
            ["Prior-shift (kalibrasyonsuz)", "LGBM olasiliklari kalibre degil", "NB21"],
            ["Platt kalibrasyon", "Prior-shift fixliyor ama MCC dusuyor", "NB22"],
            ["Heterogeneous stacking", "3 GBDT benzer kararlar, dusuk diversite", "NB22"],
            ["TabPFN (Nature 2025)", "Azinlik-ornegi tavani, bias yetmez", "NB24"],
            ["BBSE (Lipton 2018)", "Confusion matrix yetersiz kalitede", "NB24"],
            ["HyperEnsemble (parSMURF)", "Deterministic part. BalBag'den kotu", "NB26"],
            ["Feature Engineering", "Anonimlestirilmis veride sinirli", "NB26"],
        ],
        [42, 65, 25],
    )

    # ── 5. BASARILI TEKNIKLER ──
    pdf.section_title(5, "Kanitlanmis Basarili Teknikler")
    pdf.add_table(
        ["Teknik", "Etki", "Kanit"],
        [
            ["BalancedBagging", "MCC 0.463->0.529 (+0.066)", "NB21, NB22, NB23 tutarli"],
            ["COMBINED havuz", "MCC +0.034 vs MASTER-only", "NB21, NB23"],
            ["Benign-aware threshold", "%80/20 F1'de ~2 puan kazanc", "NB15v2, NB21"],
            ["Multi-seed validasyon", "Kararlilik kaniti (std<0.025)", "NB26"],
        ],
        [42, 55, 50],
    )

    # ── 6. FINAL KARAR ──
    pdf.add_page()
    pdf.section_title(6, "Final Karar ve Sonuc")

    pdf.sub_title("PAH Final Aday Modelleri")
    pdf.add_table(
        ["Aday", "MCC", "Boot F1", "CI", "Guvenilirlik"],
        [
            ["NB22 Sweep n=10", "0.536", "0.591", "[0.52-0.65]", "Tek-seed (dikkat)"],
            ["NB21 P4_COMB_BalBag", "0.529", "0.582", "[0.52-0.65]", "Coklu dogrulama"],
        ],
        [38, 20, 22, 28, 40],
    )

    pdf.sub_title("PAH KAPANDI -- Gerekce")
    pdf.body_text(
        "6 notebook (NB16->NB26), 30+ deney, 5 farkli paradigma (stacking, BalancedBagging, "
        "kalibrasyon, TabPFN, HyperEnsemble) ayni MCC~0.53 / Boot~0.58 platosuna carpiyor."
    )
    pdf.body_text(
        "Chatterji 2022 azinlik-ornegi tavani tezi deneysel olarak dogrulandi: 62 benign ile bu "
        "feature uzayinda daha fazla iyilesme mumkun degil. Tavan ancak daha fazla benign veriyle "
        "(ki NB18 bunun biyolojik olarak kit oldugunu gosterdi) veya tamamen farkli feature "
        "uzayiyla kirilabilir."
    )

    pdf.sub_title("Literatur Dogrulamalari")
    pdf.bullet("Chatterji et al. 2022 (NeurIPS): Undersampling minimax-optimal -- PAH verisinde DOGRULANDI")
    pdf.bullet("Alexandari et al. 2020 (ICML): Calibrate-then-shift -- Platt prior-shift'i FIXLEDI")
    pdf.bullet("Lipton et al. 2018 (ICML): BBSE -- PAH'ta confusion matrix yetersiz, UYGULANAMAZ")
    pdf.bullet("Hollmann et al. 2025 (Nature): TabPFN -- azinlik tavani nedeniyle KIRAMADI")
    pdf.ln(3)

    pdf.sub_title("Kalan Acik Isler (PAH Disi)")
    pdf.add_table(
        ["Oncelik", "Gorev"],
        [
            ["Yuksek", "KANSER mudahale: COMBINED+BalancedBagging"],
            ["Yuksek", "Final teslim modeli sabitleme"],
            ["Tamamlandi", "CFTR: S0c_COMBINED (Boot=0.863)"],
            ["KAPANDI", "PAH: P4_COMB_BalBag (Boot=0.582)"],
        ],
        [30, 120],
    )

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        "Bu rapor, PAH panelindeki tum deneysel calismalarin retrospektif bir ozetidir. "
        "18 gorsel, 6 notebook'tan (NB16-NB26) derlenen sonuclari icerir. "
        "Rapor otomatik olarak fpdf2 ile uretilmistir."
    )

    out = REPORTS / "PAH_retrospective_report.pdf"
    pdf.output(str(out))
    print(f"Rapor olusturuldu: {out}")
    return out


if __name__ == "__main__":
    build()
