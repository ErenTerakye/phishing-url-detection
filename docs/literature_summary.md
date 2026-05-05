# Literatür Özeti — Phishing URL Tespiti

## 1. Prasad & Chandra (2024)
**Başlık:** Incremental Learning for Phishing URL Detection Using PhiUSIIL Dataset  
**Yöntem:** BernoulliNB + Passive-Aggressive Classifier + SGD (artımlı/online learning)  
**Veri Seti:** PhiUSIIL (235.795 örnek)  
**Sonuçlar:** Accuracy %99.24, F1 %99.21  
**Öne Çıkan Nokta:** Büyük veri setlerinde batch training yerine online/incremental learning kullanarak bellek verimliliği sağlamıştır.

---

## 2. Vajrobol vd. (2024)
**Başlık:** Mutual Information-Based Feature Selection for Phishing URL Detection  
**Yöntem:** Mutual Information özellik seçimi + Logistic Regression  
**Veri Seti:** PhiUSIIL  
**Kullanılan 5 Özellik:** URLSimilarityIndex, LineOfCode, NoOfExternalRef, NoOfImage, NoOfSelfRef  
**Sonuçlar:** Accuracy **%99.97**, F1 %99.97  
**Öne Çıkan Nokta:** Sadece 5 özellik ile neredeyse mükemmel doğruluk elde edilmiştir. Bu, özellik seçiminin kritik önemini vurgular.

---

## 3. Yoon vd. (2024)
**Başlık:** Hybrid Deep Learning for Phishing Detection: CNN + Transformer + GCN  
**Yöntem:** CNN (karakter düzeyinde URL encoding) + Transformer (bağlam) + GCN (URL yapı grafı)  
**Veri Seti:** Common Crawl + PhishTank (özel derleme)  
**Sonuçlar:** Accuracy %98.12, F1 %97.89  
**Öne Çıkan Nokta:** URL'yi hem karakter dizisi hem de graf yapısı olarak modelleyerek zengin temsil elde edilmiştir. Derin öğrenme tabanlı en kapsamlı yaklaşımlardan biri.

---

## 4. Rao vd. (2025)
**Başlık:** Super Learner Ensemble for Phishing Website Detection  
**Yöntem:** Stacked ensemble (Super Learner): RF, XGB, LR, SVM meta-learner  
**Veri Seti:** PhishDump (özel, 50K+ örnek)  
**Sonuçlar:** Accuracy %98.93, ROC-AUC 0.994  
**Öne Çıkan Nokta:** Stacking ile bireysel modellerin üzerinde performans elde edilmiştir. Meta-öğrenici olarak Logistic Regression kullanılmıştır.

---

## 5. Taha vd. (2024)
**Başlık:** Comparative Analysis of ML Algorithms for Phishing Detection  
**Yöntem:** LR, DT, RF, AdaBoost, XGBoost (klasik ML karşılaştırması)  
**Veri Seti:** UCI Phishing Websites Dataset (11.055 örnek)  
**Sonuçlar:** En iyi: XGBoost %96.89, RF %96.54  
**Öne Çıkan Nokta:** Küçük veri setinde ensemble yöntemlerin üstünlüğü gösterilmiştir. Ancak veri seti PhiUSIIL'den çok daha küçüktür.

---

## Karşılaştırmalı Özet Tablosu

| Kaynak | Yıl | Yöntem | Veri Seti | Örnek Sayısı | Accuracy | F1 | Recall |
|--------|-----|--------|-----------|--------------|----------|----|--------|
| Prasad & Chandra | 2024 | Online NB+PAC+SGD | PhiUSIIL | 235.795 | %99.24 | %99.21 | — |
| Vajrobol vd. | 2024 | LR (5 özellik) | PhiUSIIL | 235.795 | **%99.97** | %99.97 | — |
| Yoon vd. | 2024 | CNN+Transformer+GCN | Özel | ~200K | %98.12 | %97.89 | — |
| Rao vd. | 2025 | Super Learner | PhishDump | ~50K | %98.93 | — | — |
| Taha vd. | 2024 | LR,DT,RF,XGB | UCI Phishing | 11.055 | %96.89 | — | — |
| **Bu Çalışma** | **2026** | **LR,DT,RF,XGB,SVM,KNN** | **PhiUSIIL** | **235.795** | **?** | **?** | **?** |

---

## Araştırma Boşlukları ve Bu Çalışmanın Katkısı

1. **Sistematik karşılaştırma eksikliği:** Vajrobol ve Prasad çalışmaları tek model/yaklaşıma odaklanmış; bu çalışma 6 farklı algoritmayı aynı ön işleme pipeline'ı ile karşılaştırmaktadır.
2. **Özellik seçimi etkisi:** Tam özellik seti vs. MI ile seçilmiş 5 özellik vs. RF ile seçilmiş Top-20 özelliğin model performansına etkisi sistematik olarak incelenmektedir.
3. **Leakage önleme:** Scaler'ın sadece train setine fit edilmesi ve tanımlayıcı sütunların çıkarılması titizlikle uygulanmaktadır.
4. **Recall odağı:** Literatürde çoğunlukla accuracy raporlanırken, bu çalışma false negative (kaçırılan phishing) maliyetini ön plana çıkarmaktadır.
