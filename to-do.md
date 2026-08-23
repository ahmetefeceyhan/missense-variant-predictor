# TEKNOFEST — Sonraki Adımlar

Son güncelleme: 2026-07-29
Durum: NB46 (PAH/CFTR reverse-distribution), NB44 (KANSER G_LOW flag ablasyonu), NB47 (TabPFN-2.5, tüm panellerde)
ve NB48 (missing-handling M1–M5+native_nan final-realistic yeniden yargılama) tamamlandı. Bulgular ve çıktı
yolları `progress.md`'de.

> Bu dosya yalnızca **sonraki çalışma oturumunda yapılacakları** içerir.
> Tamamlanmış deneyler ve bulgular `progress.md`'dedir.

---

## Panel durumları (2026-07-29 itibarıyla)

| Panel | Şampiyon | Boot-F1 | Durum |
|---|---|---|---|
| CFTR | S0c_PriorShift_NB20 | 0.863 | KESİNLEŞTİ (NB46 reverse-dist + NB47 TabPFN + NB48 missing hepsi geçemedi) |
| KANSER | P9_REVERSE_6040_catboost_with_fe (NB32) | 0.730 | KESİNLEŞTİ (NB44 φ-flag + NB47 TabPFN geriledi; NB48 M5 +0.012 ama basitleştirilmiş modelle — doğrulama gerekli) |
| PAH | P4_COMBINED_BalBag (NB21) | 0.582 | KESİNLEŞTİ (NB45 Bayes-ceiling artefaktı + NB46 reverse-dist + NB47 TabPFN + NB48 missing hepsi aynı yöne işaret ediyor) |
| **MASTER** | S1_6040/balbag (NB39) | 0.638 | **AÇIK — tek kalan aktif eksen (NB47 TabPFN geçemedi; NB48 M5 basitleştirilmiş modelle 0.550, gerçek champion hâlâ geçilmedi)** |

## Yol haritası ilerleme durumu (`reports/literature_research_panel_improvements_2026-07-24.md`)

| # | Deney | Durum |
|---|---|---|
| 1 | NB40'ı sağlam Bayes-ceiling ile yeniden çalıştır | ✅ NB45 |
| 2 | Reverse-distribution'ı PAH/CFTR'ye yay | ✅ NB46 |
| 3 | TabPFN-2.5'i her panelde dene | ✅ NB47 — negatif sonuç, hiçbir panelde şampiyonu geçemedi |
| 4 | Missing-handling'i final protokolde panel-bazlı yeniden yargıla | ✅ NB48 — hipotez çürütüldü, ama M5>M3 örüntüsü keşfedildi (bkz. aşağı) |
| 5 | NB44 (KANSER G_LOW φ-seçici flag) | ✅ NB44 — negatif sonuç |
| 6 | Venn-Abers kalibrasyon + calibrate-then-shift | ⬜ Açık |
| 7 | Reverse-pool base'leriyle heterojen stacking (MASTER) | ⬜ Açık |
| 8 | İki-taraflı eşik / precision-öncelikli karar kuralı | ⬜ Açık, düşük öncelik |
| 9 | Importance-weighted pooling / shared-backbone MoE | ⬜ Açık, yüksek risk |

## Öncelikli öneri (bir sonraki oturum): NB48'in Doğrulanması — M5 vs M3, Gerçek Champion Modeliyle

**NB48'in bulgusu:** M1–M5 + native_nan stratejileri 4 panelde, her panelin champion pool'u sabit tutularak
final-realistic protokolde (%80/20 bootstrap N=50) yeniden test edildi. Beklenen "MASTER'da flag kazanır / PAH'ta
flag kaybeder" ayrımı **gözlenmedi**. Bunun yerine tutarlı bir örüntü çıktı: **M5 (flag + >%50 NaN drop + ≤%50
medyan) ve native_nan (hiç impute/flag yok) sistematik olarak M3'ü (mevcut varsayılan) geçiyor** — MASTER: M5
0.5505 vs M3 0.5179 (+0.033), KANSER: M5 0.7418 vs M3 0.6900 (+0.052), CFTR: native_nan 0.7720 vs M3 0.6860
(+0.086). PAH'ta fark ters ama küçük (M1 hafif önde, +0.006).

**⚠️ Kritik sınırlama:** NB48'in BalancedBagging implementasyonu NB39'un orijinal champion reçetesini (early-stopping
+ val-set + sample_weight + farklı hiperparametreler) **birebir replike etmedi** — basitleştirilmiş bir sürüm
kullanıldı. Bu yüzden NB48-içi M3 sonucu bile NB39'un raporlanan 0.638'ine ulaşmıyor (NB48 M3=0.518). **Yani M5'in
üstünlüğü NB48'in kendi (basitleştirilmiş) modeli içinde geçerli, ama gerçek champion'ın tam hiperparametreleriyle
doğrulanmadı.**

**Denenecek (NB49 veya NB48 devamı):**
- Her panelin **tam, hiperparametre-sadık** champion reçetesini (NB39/NB32/NB21/NB20'den birebir kopyalanmış
  fit fonksiyonları) alıp, sadece missing-stratejisini M3'ten **M5'e** çevirerek yeniden çalıştır.
- Eğer M5 gerçek champion'da da net kazanç veriyorsa → **4 panelin de champion'ı M5'e güncellenmeli**, bu projenin
  en eski (25+ notebook, NB14'ten beri sorgulanmamış) varsayımını değiştiren önemli bir sonuç olur.
- Eğer kazanç kaybolursa (yani NB48'in kazancı basitleştirilmiş modelin bir artefaktıysa) → M3 mevcut champion'larda
  korunur, ama M5'in **düşük-sadakat modellerde daha sağlam olduğu** ikincil bir bulgu olarak not düşülür.

## Diğer açık eksenler (roadmap'ten, düşük öncelik)

- **MASTER reverse-pool base'leriyle heterojen stacking + Optuna** (madde 7): NB39'un S1_6040 pool'u ile LGBM+RF+
  BalBag stacking (meta=LR) hiç denenmedi; beklenti düşük (Bayes-ceiling gap dar, +0.0161) ama doğrulanmalı.
- **Venn-Abers kalibrasyon + calibrate-then-shift** (madde 6): MASTER/PAH'ta prior-shift'in kök çözümü olabilir,
  henüz denenmedi.
- **CFTR değerlendirme çerçevesi sağlamlaştırma**: n=21 benign nedeniyle nested/repeated LOO-CV gibi bir yaklaşım
  gerekebilir; model değişikliği değil, ölçüm güvenilirliği önceliği. NB48'de CFTR champion'ı (prior-shift içeren
  S0c) ile basit LGBM+COMBINED (prior-shift'siz) karşılaştırıldığı için CFTR'deki −0.091 delta'nın kaynağı
  missing-strateji değil, eksik prior-shift olabilir — bu karışıklık NB49'da ayrıştırılmalı.

## Metodolojik notlar

**NB47'den:** TabPFN CPU'da COMBINED havuzlarında (n>1500) çok yavaş (~10-12 dakika/senaryo, süper-lineer
ölçekleniyor). Gelecekte tekrar denenirse alt-örnekleme veya `n_estimators` düşürme ile başlat.

**NB48'den:** "Tek eksen değişimi" ilkesi (NB44'ten) uygulanırken model implementasyonunun **hiperparametre
düzeyinde birebir sadık** olması şart — aksi halde gözlenen fark missing-stratejisinden mi yoksa model
sadakatsizliğinden mi kaynaklandığı ayrışmaz. Yeni bir ablasyon yazarken önce referans notebook'un fit
fonksiyonunu **kelimesi kelimesine kopyala**, sonra tek değişkeni değiştir.
