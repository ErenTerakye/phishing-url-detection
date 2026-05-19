# 21 Mayıs Sunuş Outline

Hazırlık tarihi: 19.05.2026

## 1. Problem ve Motivasyon

Phishing URL'leri erken tespit etmek; recall odakli degerlendirme.

## 2. Veri Seti

PhiUSIIL: 235.795 URL, 54 sutun, 1=mesru ve 0=phishing.

## 3. Literatür Özeti

Prasad & Chandra, Vajrobol, Yoon, Rao, Taha ve Kytidou calismalari.

## 4. Leakage Problemi

URLSimilarityIndex, URLCharProb, TLDLegitimateProb, URLTitleMatchScore riskli ozellikler.

## 5. Deney Protokolü

Full, Vajrobol-5, RF Top-20, Leakage-Free ve en iyi 3 model icin CV.

## 6. Model Sonuçları

Vajrobol-5 LR accuracy: 99.63%; leakage-free en iyi model: XGBoost.

## 7. Leakage-Free Karşılaştırma

XGBoost leakage-free recall 99.99%, F1 99.99%.

## 8. Literatürden Farkımız

Yuksek skoru tekrar etmek yerine sızıntılı ozellik etkisini deneysel olarak ayirdik.

## 9. Sonuç ve Gelecek Çalışma

Dis test seti, zaman bazli ayrim, threshold tuning, aciklanabilirlik.

## 10 Dakikalık Video Akışı

0:00-1:00 problem ve veri seti; 1:00-3:00 literatür ve leakage; 3:00-6:30 deneyler ve metrikler; 6:30-8:30 leakage-free sonuçlar; 8:30-10:00 literatürden farkımız ve sonuç.
