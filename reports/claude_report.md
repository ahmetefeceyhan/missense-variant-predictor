Notebook 11 — Stacking General Panel için Yenilik Önerileri
Mevcut Durum Özeti
notebooks/11_stacking_general_panel.ipynb şu an çok temiz bir OOF stacking iskeletine sahip:

Baz modeller (4): LightGBM, XGBoost, DNN, SVM — hepsi 5-fold OOF ile eğitiliyor
İki meta tipi:
Standart (8 sütun: 4 model × [neg, pos])
Diversity-aware (19 sütun: 8 + disagreement, confidence, entropy, max_min_gap, 6 ikili diff_*)
En iyi sonuç: stacking_nn F1=0.9505, MCC=0.8347 (baz LightGBM 0.9454'ün +0.0051 üstünde)
Confusion matrix (Stacking NN): TP=422, FP=29, FN=15, TN=166 → kalan hatalar baskın olarak FP yönünde (29 vs 15 FN)
★ Insight ─────────────────────────────────────

ROC eğrilerine bakıldığında baz LightGBM (AUC=0.9715) tek başına meta modellerden (~0.966) daha yüksek; meta sadece F1'de eşik kalibrasyonu sayesinde kazanıyor. Bu, klasik bir "stacking platosu": çeşitlilik tükenmiş.
Korelasyon matrisinden görüyoruz ki LightGBM↔XGBoost meta-feature'ları r=0.97, DNN↔SVM r=0.87 ile birbirine çok yakın. Yani 8-vektör aslında ~3 bağımsız sinyal taşıyor — meta-model "ezbere" yapıyor.
Diversity feature importance'ında diff_xgboost_dnn ve diff_lightgbm_xgboost en üst sırada: yani modelin asıl sıkıştığı yer, modeller ÇELİŞTİĞİ örnekler. İşte yenilik buradan başlamalı.
─────────────────────────────────────────────────
Yenilik Yapılabilecek Beş Eksen
Aşağıdaki önerileri etki/efor sırasına göre sıraladım. Hepsi advisor_report.pdf'in tavsiyeleri ve mevcut hataların doğasıyla örtüşüyor.

1. Çeşitlilik Üreten Yeni Bir Baz Model: ESM-2 Δ-Embedding Tabanlı LightGBM (yüksek etki)
Sorun: Mevcut 4 baz model aynı 954-sütunluk X üzerinde eğitiliyor; Pearson 0.97 korelasyon doğal. Stacking'in teorik tavanı, baz modeller arası çeşitliliğin (diversity) ne kadar olduğuyla bellidir.

Öneri: advisor_report.pdf'in 3. sayfasındaki Δ-Embedding stratejisini bağımsız bir "5. baz model" olarak ekle:

Prot_11mer_Ref ve Prot_11mer_Alt sütunlarını (zaten ham veride var) ESM-2 small (ör. esm2_t6_8M_UR50D) modelinden geçir → her ikisi için 320-D mean-pooled vektör
ΔE = E_alt - E_ref (320-D) hesapla, PCA ile 32-D'ye indir
Bu 32-D'yi tek başına bir LightGBM ile OOF eğit → 5. meta-feature setini meta katmana ekle (1×10 standart vektör)
Neden çeşitlilik üretir: ESM-2 hiçbir ClinVar etiketini görmemiş; mevcut k-mer/OHE tamamen yüzeysel sayım, Δ-embedding ise 3D yapı + ko-evolüsyon çıkarımı taşıyor. Korelasyon doğal olarak düşük olacak.

Beklenen etki: Meta-NN'in F1'i 0.9505 → ~0.96+'ya çıkabilir; daha kritik olarak FP=29'un düşmesi beklenir çünkü FP'lerin çoğu "k-mer'in görmediği nadir motif" bölgesinde oluyor.

2. Focal Loss ile OOF Yeniden Eğitim (orta etki, düşük efor)
Sorun: Şu an LightGBM/XGBoost BCEWithLogitsLoss + class_weight=balanced ile eğitiliyor. advisor_report'un 7-9. sayfaları gösteriyor ki azınlık sınıfı yanlışları (FN) yine de var — Stacking NN'de FN=15.

Öneri: src/focal_loss.py zaten projede mevcut (SklearnMLP'de use_focal=True desteği de var). Notebook 11'de:

LightGBM için fobj=focal_loss_lgb (γ=2.0, α=0.25) ile alternatif bir OOF üret
Bu FocalLightGBM'i 5. model olarak değil, mevcut LightGBM'in yerine geçen bir varyant olarak dene; ya da ikisini birden tut → 5 baz model
Hangi durumda kullanılır: Klinik prioritede FN (kaçırılan patojenik) FP'den daha kritik. Focal γ artırılınca recall yükselir → FN=15→~10 hedefi makul.

3. Meta-Model Olarak Calibrated + Stack-of-Stacks (orta etki)
Sorun: Şu anki meta-NN (8 input → küçük MLP) bir lgbm_pos + xgboost_pos lineer kombinasyonundan çok az farklı çalışıyor (feature importance bunu gösteriyor: xgboost_pos ve svm_neg zirvede, dnn_pos/svm_pos neredeyse sıfır).

Öneri:

Meta katmanına Platt scaling / Isotonic calibration ekle: her baz modelin OOF olasılıklarını sigmoid/isotonic ile yeniden kalibre ettikten sonra meta'ya ver
İkinci tabaka olarak basit bir logistic regression stacker daha ekle (3 meta-model: lgbm, nn, lr) → bunların ortalamasını al (rank-averaging)
★ Insight ─────────────────────────────────────

Notebook 11'in feature importance grafiği (Sayfa 16) dnn_pos ve svm_pos puanlarının fiilen kullanılmadığını gösteriyor. NN baz modeli MCC=0.5948 ile diğerlerinden çok zayıf. Onu çıkarıp yerine DNN'i farklı feature subset üzerinde eğitmek (ör. yalnız non-sequence sütunlar) korelasyonu kıracaktır.
Calibration'ın değeri stacking yazınında çok bilinir: gradient boosting modelleri "aşırı güvenli" çıktı verir; meta-model bu güveni "biliyormuş gibi" kullanır. Isotonic ile düzelttiğinde meta-model daha doğru karar verir.
─────────────────────────────────────────────────
4. Hatalara-Odaklı OOF: Feature-Subspace Stacking (orta etki)
Sorun: Tüm baz modeller aynı 954 sütunu görüyor.

Öneri: Random Subspace Method'a benzer şekilde her baz modele farklı feature kümesi ver:

LightGBM_v1 → tüm 954
LightGBM_v2 → yalnız conservation + fizikokimyasal (~25 sütun) — bu kategori advisor_report'a göre "sızıntısız altın özellikler"
XGBoost → yalnız k-mer + OHE (~944 sütun)
DNN → yalnız non-sequence (sadece engineered features)
Bu, Notebook 09 ve 10'daki ablasyon çıktılarından feature_importance.csv türetip otomatik yapılabilir. Çeşitlilik (diff_*) artar, FP'ler düşer.

5. Test-Time Hata Kümeleme & Selective Prediction (düşük efor, yüksek pratik değer)
Sorun: Notebook 07'de "18 satırda tüm modeller yanlış" gözlemi vardı. Bunlar muhtemelen belirsiz veya yanlış etiketli ClinVar varyantları.

Öneri: Stacking'in çıkışına bir "abstain" katmanı ekle:

disagreement > threshold veya entropy > 0.6 olan örneklerde "VUS — manuel inceleme" etiketi ver
Geri kalanlarda F1'i yeniden hesapla → bu, "yüksek-güven F1" diye raporlanabilir, klinikte gerçek metrik budur
advisor_report'un son sayfasındaki "Klinik Aktarım Stratejisi" bölümü tam olarak bunu söylüyor. Sayısal kazanım sınırlı olabilir ama yarışma raporunda diferansiyel yaratır.

Önerim: Önce Hangisi?
Sırayla #2 → #1 → #3 yapmanı öneririm:

Önce Focal Loss (#2): En küçük değişiklik, mevcut iskelete tek bir fobj parametresi ekleyerek anında deneyebilirsin. Beklenen kazanım küçük ama 1-2 saatte ölçülür ve FN metriğini doğrudan iyileştirir.
Sonra ESM-2 Δ-Embedding 5. baz (#1): Asıl yenilik bu — yarışma raporunda ayırt edici fark yaratır, advisor_report'un en çok vurguladığı stratejik adım.
Son olarak calibration + stack-of-stacks (#3): Üstteki ikisi yerleşince meta katmanını rafine eder.
Hangisinden başlamak istersen söyle, doğrudan o değişikliğin kod implementasyonuna geçeyim — Notebook 11'in iskeleti zaten tüm bunları temiz şekilde kabul edecek modüler bir yapıda yazılmış.

