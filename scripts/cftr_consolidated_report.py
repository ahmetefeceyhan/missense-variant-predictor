"""
CFTR Consolidated Report -- NB17->NB20 full journey
Generates reports/CFTR_consolidated_report.pdf
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from fpdf import FPDF
from config import PROJECT_ROOT

ROOT = Path(PROJECT_ROOT)
REPORTS_DIR = ROOT / "reports"
V7 = ROOT / "results" / "v7_cftr_refinement"
V9 = ROOT / "results" / "v9_cftr_combined"
V10 = ROOT / "results" / "v10_cftr_ensemble"
OUT = REPORTS_DIR / "CFTR_consolidated_report.pdf"


class CFTRReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "CFTR Paneli Konsolide Rapor | TEKNOFEST 2025", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 102, 204)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")

    def section(self, title, level=1):
        if level == 1:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(0, 51, 102)
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(51, 51, 51)
        self.ln(3)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body(self, txt):
        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, txt)
        self.ln(1)

    def bullet(self, txt):
        self.set_font("Helvetica", "", 9)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, "  - " + txt)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(0, 102, 204)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_x(self.l_margin)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 8)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(230, 240, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5.5, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.set_x(self.l_margin)
        self.ln(2)

    def add_fig(self, path, w=170, caption=""):
        if os.path.exists(path):
            x = (210 - w) / 2
            self.image(path, x=x, w=w)
            if caption:
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 4, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(3)


pdf = CFTRReport()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ============================================================
# SAYFA 1 -- Baslik + Ozet
# ============================================================
pdf.add_page()
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(0, 51, 102)
pdf.ln(5)
pdf.cell(0, 12, "CFTR Paneli: Tam Calisma Raporu", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "NB17 -> NB18 -> NB19 -> NB20  |  2026-06-13 -- 2026-06-17", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)

pdf.section("Yonetici Ozeti")
pdf.body(
    "Bu rapor, TEKNOFEST Saglikta Yapay Zeka Yarismasi kapsaminda CFTR paneli icin "
    "yurutulen dort asamali model iyilestirme surecini ozetlemektedir. Calisma serisi "
    "NB17'deki 7-strateji karsilastirmasiyla baslamis, NB18'de veri artirimi cikmazi, "
    "NB19'da havuz genisletme kesfi ve NB20'de ensemble + threshold dogrulamasi ile "
    "tamamlanmistir."
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(0, 120, 0)
pdf.cell(0, 6, "FINAL KARAR: S0c_COMBINED (MASTER+KANSER+PAH, n=3691, LightGBM + prior-shift)", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(1)

pdf.table(
    ["Metrik", "Deger", "Aciklama"],
    [
        ["LOO-MCC", "0.6436", "111 ornek, 21 benign, LOO-CV (birincil)"],
        ["Boot-mean F1 @%80/20", "0.8629", "Final yarisma dagilimini simule eder"],
        ["Precision", "1.000", "Sifir false positive"],
        ["Recall", "0.789", "71/90 pathogenic dogru"],
        ["FP / FN", "0 / 19", "Benign tarafta mukemmel, 19 kacirilan patho"],
    ],
    [40, 30, 120],
)

pdf.section("CFTR Paneli -- Zorluklar")
pdf.bullet("Toplam 111 ornek, sadece 21 benign -- herhangi bir modelin guvenilir olarak degerlendirilmesi icin yetersiz bootstrap orneklemi (CI=[0.00-1.00]).")
pdf.bullet("Cozum: LOO-CV (tum 111 ornek degerlendirmede) + MCC (dengesizlige dayanikli metrik).")
pdf.bullet("Yarisma final testi ~%80 benign / %20 pathogenic -- egitimdeki %81 pathogenic oraninin tam tersi.")
pdf.bullet("CFTR tek-gen paneli, MASTER ise coklu-gen (2931 ornek) -- domain shift riski.")
pdf.bullet("Benign veri artirimi yapisal cikmaz: ClinVar'da CFTR benign=1624, ama sadece 15'i missense.")

# ============================================================
# SAYFA 2 -- NB17 Strateji Karsilastirmasi
# ============================================================
pdf.add_page()
pdf.section("1. NB17 -- 7 Strateji Karsilastirmasi (2026-06-13)")
pdf.body(
    "Ilk asama: CFTR icin 7 farkli modelleme stratejisini LOO-CV+MCC ile sistematik olarak "
    "karsilastirmak. Prior-shift kalibrasyonu (Saerens 2002) S0 uzerine uygulanarak S6 olusturuldu."
)

pdf.table(
    ["Strateji", "LOO-MCC", "Boot-mean", "Boot-std", "Train-F1", "Overfit"],
    [
        ["NB16 ref (DNN)", "N/A", "0.852", "0.267", "N/A", "?"],
        ["S0_MASTER-only", "0.627", "0.728", "0.107", "0.979", "ORTA"],
        ["S1_MASTER-core", "0.608", "0.783", "0.099", "0.985", "ORTA"],
        ["S2_SMOTE-CFTR", "0.561", "0.493", "0.051", "1.000", "OK"],
        ["S4_FrozenFinetune", "0.294", "0.370", "0.258", "0.500", "OK"],
        ["S5_BalancedBagging", "0.572", "0.754", "0.128", "0.935", "OK"],
        ["S6_PriorShift", "0.655", "0.692", "0.088", "0.978", "OK"],
    ],
    [38, 22, 25, 22, 22, 18],
)

pdf.section("NB17 Bulgulari", level=2)
pdf.bullet("S6_PriorShift (LOO-MCC=0.655) acik ara en iyi -- S0'in ham ciktisina post-hoc pi duzeltmesi yeterli.")
pdf.bullet("S4_FrozenFinetune hayal kirikligi (MCC=0.294): n=111 ile finetune basarisiz.")
pdf.bullet("S2_SMOTE train F1=1.0 ama boot-mean=0.493 -- sentetik veriyle overfit.")
pdf.bullet("S5_BalancedBagging LOO-MCC=0.572 ama multi-seed F1 varyans cok yuksek (0.498).")

if os.path.exists(V7 / "fig1_loo_mcc.png"):
    pdf.add_fig(str(V7 / "fig1_loo_mcc.png"), w=160, caption="Sekil 1: NB17 LOO-MCC strateji karsilastirmasi")

# ============================================================
# SAYFA 3 -- NB17 Prior-Shift + Confusion + Reg Sweep
# ============================================================
pdf.add_page()
pdf.section("1b. NB17 -- Prior-Shift Etkisi ve Confusion Matrix")
pdf.body(
    "Prior-shift duzeltmesi: odds_adj = odds * (pi_test/(1-pi_test)) / (pi_train/(1-pi_train)). "
    "pi_train=0.733, pi_test=0.20. Bu, modelin ciktisini final yarisma dagiliminina kalibre eder."
)

if os.path.exists(V7 / "fig3_prior_shift.png"):
    pdf.add_fig(str(V7 / "fig3_prior_shift.png"), w=140, caption="Sekil 2: S6 Prior-shift posterior dagilimi")

if os.path.exists(V7 / "fig4_confusion.png"):
    pdf.add_fig(str(V7 / "fig4_confusion.png"), w=140, caption="Sekil 3: S6 LOO-CV Confusion Matrix (TN=18, FP=3, FN=11, TP=79)")

pdf.add_page()
pdf.section("1c. NB17 Ek -- Regularizasyon Sweep (2026-06-14)")
pdf.body(
    "S6'nin overfit riski incelendi. 12 konfigurasyonda min_child_samples (5-320) ve "
    "num_leaves (3-63) taranarak memorization_gap, transfer_gap ve canary metrikleri olculdu."
)

pdf.table(
    ["Parametre", "Deger", "LOO-MCC", "Boot-mean", "Multi-seed F1", "Mem. Gap"],
    [
        ["min_child_samples", "5", "0.673", "0.748", "0.929", "0.156"],
        ["min_child_samples", "10", "0.690", "0.755", "0.926", "0.188"],
        ["min_child_samples", "20 (S6)", "0.655", "0.692", "0.923", "0.159"],
        ["min_child_samples", "40", "0.673", "0.743", "0.908", "0.138"],
        ["min_child_samples", "80", "0.534", "0.767", "0.581", "0.166"],
        ["min_child_samples", "160", "0.534", "0.799", "0.634", "0.050"],
        ["num_leaves", "63", "0.572", "0.744", "0.521", "0.150"],
        ["num_leaves", "15", "0.670", "0.861", "0.754", "0.116"],
        ["num_leaves", "7", "0.560", "0.718", "0.637", "0.101"],
        ["num_leaves", "3", "0.627", "0.732", "0.734", "0.042"],
    ],
    [32, 18, 22, 25, 28, 22],
)

pdf.section("Reg Sweep Karari", level=2)
pdf.bullet("S6 (mcs=20, nl=31) guvenli platonun icinde. Regularizasyonu arttirmak CFTR'ye zarar veriyor.")
pdf.bullet("Asiri sikistirma (mcs>=80) multi-seed F1'i 0.92'den 0.38-0.63'e indiriyor.")
pdf.bullet("Verdict: S6'yi degistirme -- mevcut parametreler optimalin yakininda.")

if os.path.exists(V7 / "fig6_reg_sweep.png"):
    pdf.add_fig(str(V7 / "fig6_reg_sweep.png"), w=170, caption="Sekil 4: Regularizasyon sweep -- LOO-MCC vs Multi-seed F1")

# ============================================================
# SAYFA -- NB18 Negatif Sonuclar
# ============================================================
pdf.add_page()
pdf.section("2. NB18 -- Veri Artirimi Arastirmasi: 2 Negatif Sonuc (2026-06-15)")
pdf.body(
    "CFTR'nin n=21 benign sinirlamasini asmak icin 63k ClinVar/OpenCRAVAT verisinden "
    "benign artirimi denendi. Iki yontem planlanmis, ikisi de yapisal engellere takilmistir."
)

pdf.section("Bulgu 1: Deobfuscation Birebir Calismaz", level=2)
pdf.bullet("Yarisma verisi anonimlestirilmis (AL/CAT/EK/AA) ve sayisal degerler NORMALIZE edilmis.")
pdf.bullet("Ham mean/std eslestirmesi cokuyor: cadd_phred legacy'de 0-63, yarismada [0,1].")
pdf.bullet("Cozum: Grup-tabanli kimlik (FREQ/SCORE01/CONS_RAW) -- 124 guvenilir sutun eslestirildi.")
pdf.bullet("Bu altyapi ileride pathogenic transfer icin kullanilabilir, ama benign artirimi icin yetersiz.")

pdf.section("Bulgu 2: CFTR Benign Artirimi Yapisal Cikmaz", level=2)
pdf.bullet("ClinVar'da CFTR benign = 1624, AMA sadece 15'i missense (%0.9).")
pdf.bullet("Yarisma verisi saf missense (181/189 CFTR ornegi). Non-missense benign uyumsuz.")
pdf.bullet("gnomAD AF >= %5 (ACMG BA1) ile sadece 1 CFTR varyant bulundu.")
pdf.bullet("KARAR: CFTR benign kitligi veriyle cozulemez -> class_weight/focal/threshold ile yonet.")

pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(180, 0, 0)
pdf.ln(2)
pdf.cell(0, 5, "NB18 Sonucu: Her iki yontem de yapisal cikmaz. Benign artirimi TERK EDILDI.", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)

# ============================================================
# SAYFA -- NB19 Combined Panels
# ============================================================
pdf.add_page()
pdf.section("3. NB19 -- COMBINED Egitim Havuzu Kesfi (2026-06-16)")
pdf.body(
    "Benign artirmak yerine, egitim havuzunu genisletme yaklasimi: MASTER (2931) + KANSER (388) + "
    "PAH (372) = COMBINED (3691) ile egitim, CFTR uzerinde LOO-CV test."
)

pdf.table(
    ["Strateji", "n_train", "LOO-MCC", "Boot-mean", "Boot-std", "TN", "FP", "FN", "TP"],
    [
        ["S0_MASTER-only", "2931", "0.6552", "0.6915", "0.0877", "18", "3", "11", "79"],
        ["S0c_COMBINED", "3691", "0.6436", "0.8629", "0.1327", "21", "0", "19", "71"],
    ],
    [30, 18, 20, 22, 20, 15, 15, 15, 15],
)

pdf.section("COMBINED Paradoksu", level=2)
pdf.bullet("LOO-MCC: MASTER kazaniyor (0.655 > 0.644) -- daha iyi recall (FN=11 vs 19).")
pdf.bullet("Boot-mean: COMBINED kazaniyor (0.863 >> 0.692) -- FP=0 avantaji %80/20'de dramatik.")
pdf.bullet("Final yarisma %80 benign -> FP patlamasi F1'i oldurur -> COMBINED'in FP=0 kritik avantaj.")
pdf.bullet("Prior-shift duzeltmesi COMBINED'da etkisiz (mcc_raw == mcc_prior = 0.6436).")
pdf.bullet("Sonraki adim: Bu paradoksu NB20'de ensemble ve histogram debug ile incelemek.")

if os.path.exists(V9 / "fig1_confusion_compare.png"):
    pdf.add_fig(str(V9 / "fig1_confusion_compare.png"), w=150, caption="Sekil 5: NB19 MASTER vs COMBINED confusion matrix karsilastirmasi")

# ============================================================
# SAYFA -- NB20 Ensemble + Threshold
# ============================================================
pdf.add_page()
pdf.section("4. NB20 -- Ensemble + Threshold Refinement (2026-06-17)")
pdf.body(
    "NB19 bulgularini test eden uc deney: (1) Soft ensemble (S0+COMBINED), "
    "(2) Prior-shift posterior histogram debug, (3) Robust N=50 bootstrap threshold."
)

pdf.section("Exp 1: Soft Ensemble", level=2)
pdf.body("p_ensemble = 0.5 * p_S0_adjusted + 0.5 * p_COMBINED_adjusted")

pdf.table(
    ["Strateji", "LOO-MCC", "Boot-mean", "Boot-std", "Precision", "Recall", "FP", "FN"],
    [
        ["S0_MASTER", "0.5243", "0.6456", "0.1376", "0.971", "0.744", "2", "23"],
        ["S0c_COMBINED", "0.6436", "0.8629", "0.1327", "1.000", "0.789", "0", "19"],
        ["Ensemble", "0.5835", "0.7497", "0.1338", "0.986", "0.767", "1", "21"],
    ],
    [30, 20, 22, 20, 22, 20, 14, 14],
)

pdf.bullet("Ensemble COMBINED'in ALTINDA kaldi -- MASTER'in gurultulu posterior'u COMBINED'i bozuyor.")
pdf.bullet("COMBINED zaten FP=0 ile calisiyor; MASTER'in 2 FP'sini karistirmak sadece zarar veriyor.")

pdf.section("Exp 2: Prior-Shift Debug", level=2)
pdf.body(
    "2x2 posterior histogrami: MASTER raw/adj ve COMBINED raw/adj. "
    "COMBINED'in ham posterior dagilimi benign/pathogenic'i zaten temiz ayiriyor (raw thr=0.660). "
    "Prior-shift threshold'u 0.150'ye dusurse de optimal MCC noktasi degismiyor -- "
    "modelin ic kalibrasyonu yeterli."
)

pdf.section("Exp 3: Robust Threshold (NB19 vs NB20)", level=2)

pdf.table(
    ["Strateji", "NB19 thr", "NB20 thr", "NB19 MCC", "NB20 MCC", "Fark"],
    [
        ["S0_MASTER", "0.080", "0.190", "0.6552", "0.5243", "-0.131"],
        ["S0c_COMBINED", "0.150", "0.147", "0.6436", "0.6436", "0.000"],
        ["Ensemble", "0.180", "0.169", "0.5961", "0.5835", "-0.013"],
    ],
    [30, 24, 24, 24, 24, 24],
)

pdf.bullet("S0_MASTER: Tek resample thr=0.08 (asiri agresif) -> robust thr=0.19 ile MCC -0.13 dusuyor.")
pdf.bullet("COMBINED: Threshold neredeyse ayni (0.150 vs 0.147) -- stabil model, resample'a duyarsiz.")
pdf.bullet("KRITIK: NB19'daki S0_MASTER MCC=0.6552 sansli tek-resample artefaktiydi.")

if os.path.exists(V10 / "fig2_strategy_comparison.png"):
    pdf.add_fig(str(V10 / "fig2_strategy_comparison.png"), w=170, caption="Sekil 6: NB20 Strateji karsilastirmasi (Robust Threshold)")

# ============================================================
# SAYFA -- Prior-shift histogram
# ============================================================
pdf.add_page()
pdf.section("4b. NB20 -- Posterior Histogram Analizi")
pdf.body(
    "Asagidaki 2x2 histogramda S0_MASTER ve S0c_COMBINED icin raw ve prior-shift adjusted "
    "posterior dagilimlari gorulmektedir. Kirmizi = benign (n=21), mavi = pathogenic (n=90)."
)

if os.path.exists(V10 / "fig1_posterior_histogram.png"):
    pdf.add_fig(str(V10 / "fig1_posterior_histogram.png"), w=180, caption="Sekil 7: Prior-shift posterior histogramlari (2x2)")

pdf.bullet("S0_MASTER raw: Dagilimlar genis ortusuyor (thr=0.650). Adj: yogunluk 0.0-0.2'ye sikisor (thr=0.080).")
pdf.bullet("S0c_COMBINED raw: Dagilimlar daha temiz ayrisiyor (thr=0.660). Adj: zaten iyi ayrismis (thr=0.150).")
pdf.bullet("COMBINED'da prior-shift etkisiz cunku ham daginim zaten iyi kalibre -- threshold oynamasi MCC'yi degistirmiyor.")

# ============================================================
# SAYFA -- Yolculuk Ozeti + Final
# ============================================================
pdf.add_page()
pdf.section("5. CFTR Yolculugu: Kronolojik Ozet")

journey = [
    ["NB15/16", "2026-06-10", "Bootstrap CI=[0.00-1.00]", "CFTR model karsilast. anlamsiz", "LOO-CV+MCC gerekli"],
    ["NB17", "2026-06-13", "7 strateji LOO-CV+MCC", "S6_PriorShift MCC=0.655", "Prior-shift calisiyor"],
    ["NB17-ek", "2026-06-14", "Reg sweep 12 config", "S6 guvenli plato", "Degistirme"],
    ["NB18", "2026-06-15", "Benign artirimi 2 yontem", "Yapisal cikmaz", "Veriyle cozulemez"],
    ["NB19", "2026-06-16", "COMBINED havuz genisletme", "boot=0.863 / FP=0", "Paradoks kesfedildi"],
    ["NB20", "2026-06-17", "Ensemble + robust thr", "COMBINED stabil", "FINAL: COMBINED"],
]

pdf.table(
    ["Notebook", "Tarih", "Deney", "Sonuc", "Karar"],
    journey,
    [18, 24, 42, 42, 42],
)

pdf.section("6. Final Karar ve Gerekce")
pdf.body(
    "CFTR panelinin final teslim modeli S0c_COMBINED olarak kesinlesmistir. Gerekce:"
)
pdf.bullet("LOO-MCC=0.6436: COMBINED, tum diger stratejiler arasinda MASTER-only (0.6552) haric en yuksek. Robust threshold ile MASTER 0.5243'e dustugunde COMBINED acik ara lider.")
pdf.bullet("Boot-mean=0.8629: Final yarisma dagilimini (%80 benign) simule eden en iyi tahmin. Rakiplerin hepsi 0.75'in altinda.")
pdf.bullet("Precision=1.0, FP=0: %80 benign test setinde precision cokmesi OLMAYACAK. Bu, diger tum stratejilerin (S0_MASTER dahil, FP=2-3) karsisinda kritik avantaj.")
pdf.bullet("Stabil threshold: Robust N=50 bootstrap ile threshold degismedi (0.150 vs 0.147). Tek-resample artefaktindan bagimsiz.")
pdf.bullet("Ensemble eklenti degeri yok: COMBINED'in FP=0 avantajini MASTER'in gurultusu bozuyor. Karisim zarar veriyor.")

pdf.section("7. Bilinen Sinirliliklar")
pdf.bullet("LOO-MCC=0.6436, hedeflenen >0.65'in hafifce altinda. n=21 benign ile MCC farki >0.05 olmadan anlamli ayrim yapilamaz.")
pdf.bullet("Boot-std=0.1327 (CI: [0.57, 1.00]) -- yuksek belirsizlik, n=21'in kacinilmaz sonucu.")
pdf.bullet("FN=19 (21/90 pathogenic kaciriliyor) -- recall 0.789. COMBINED konservatif tahmin yapiyor.")
pdf.bullet("Prior-shift COMBINED'da etkisiz -- post-hoc kalibrasyon ek fayda saglamamakta.")
pdf.bullet("CFTR benign artirimi yapisal olarak mumkun degil (NB18) -- bu sinirlik cozulemez.")

pdf.section("8. Panel Bazinda Genel Durum")
pdf.table(
    ["Panel", "Final Model", "Boot-mean F1", "LOO-MCC", "Durum"],
    [
        ["CFTR", "S0c_COMBINED", "0.863", "0.644", "KESINLESTI"],
        ["KANSER", "stack_lr (NB16)", "0.716", "--", "Kesinlesti"],
        ["PAH", "stack_lr (NB16)", "0.515", "--", "Mudahale gerekli"],
    ],
    [25, 40, 30, 25, 40],
)

pdf.body(
    "Sonraki oncelik: PAH paneli icin ozel mudahale (COMBINED yaklasimi + prior-shift). "
    "PAH n=62 benign ile CFTR'den (n=21) cok daha guvenilir CI'lar uretecektir."
)

pdf.set_font("Helvetica", "I", 8)
pdf.ln(5)
pdf.cell(0, 4, "Rapor otomatik olusturuldu: 2026-06-17 | NB17-NB20 verileri kullanildi", align="C")

# ============================================================
pdf.output(str(OUT))
print(f"Rapor olusturuldu: {OUT}")
print(f"Boyut: {os.path.getsize(OUT) / 1024:.1f} KB")
