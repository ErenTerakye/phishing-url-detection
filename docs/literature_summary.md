# Literatür Özeti — Phishing URL Tespiti

## 1. Prasad & Chandra (2024)
**Başlık:** PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning  
**Dergi:** Computers & Security  
**DOI:** https://doi.org/10.1016/j.cose.2023.103545  
**Yöntem:** BernoulliNB + Passive-Aggressive Classifier + SGD (artımlı/online learning)  
**Veri Seti:** PhiUSIIL (235.795 örnek)  
**Sonuçlar:** Accuracy %99.24, F1 %99.21  
**Öne Çıkan Nokta:** Büyük veri setlerinde batch training yerine online/incremental learning kullanarak bellek verimliliği sağlamıştır. PhiUSIIL veri setini oluşturan ve tanıtan çalışmadır.

---

## 2. Vajrobol vd. (2024)
**Başlık:** Mutual information based logistic regression for phishing URL detection  
**Dergi:** Cyber Security and Applications  
**DOI:** https://doi.org/10.1016/j.csa.2024.100044  
**Yöntem:** Mutual Information özellik seçimi + Logistic Regression  
**Veri Seti:** PhiUSIIL  
**Kullanılan 5 Özellik:** URLSimilarityIndex, LineOfCode, NoOfExternalRef, NoOfImage, NoOfSelfRef  
**Sonuçlar:** Accuracy **%99.97**, F1 %99.97  
**Öne Çıkan Nokta:** Sadece 5 özellik ile neredeyse mükemmel doğruluk elde edilmiştir. Bu çalışmada Deney 2, bu 5 özellik kullanılarak Vajrobol vd.'nin bulguları replike edilmektedir.

---

## 3. Yoon vd. (2024)
**Başlık:** Phishing Webpage Detection via Multi-Modal Integration of HTML DOM Graphs and URL Features Based on Graph Convolutional and Transformer Networks  
**Dergi:** Electronics  
**DOI:** https://doi.org/10.3390/electronics13163344  
**Yöntem:** CNN (karakter düzeyinde URL encoding) + Transformer (bağlam) + GCN (URL yapı grafı)  
**Veri Seti:** Common Crawl + PhishTank (özel derleme)  
**Sonuçlar:** Accuracy %98.12, F1 %97.89  
**Öne Çıkan Nokta:** URL'yi hem karakter dizisi hem de graf yapısı olarak modelleyerek zengin temsil elde edilmiştir. Derin öğrenme tabanlı en kapsamlı yaklaşımlardan biri.

---

## 4. Rao vd. (2025)
**Başlık:** A hybrid super learner ensemble for phishing detection on mobile devices  
**Dergi:** Scientific Reports  
**DOI:** https://doi.org/10.1038/s41598-025-02009-8  
**Yöntem:** Stacked ensemble (Super Learner): RF, XGB, LR, SVM meta-learner  
**Veri Seti:** PhishDump (özel, 50K+ örnek)  
**Sonuçlar:** Accuracy %98.93, ROC-AUC 0.994  
**Öne Çıkan Nokta:** Stacking ile bireysel modellerin üzerinde performans elde edilmiştir. Meta-öğrenici olarak Logistic Regression kullanılmıştır.

---

## 5. Taha vd. (2024)
**Başlık:** A Machine Learning Algorithms for Detecting Phishing Websites: A Comparative Study  
**Dergi:** Iraqi Journal for Computer Science and Mathematics  
**DOI:** https://doi.org/10.52866/ijcsm.2024.05.03.015  
**Yöntem:** LR, DT, RF, AdaBoost, XGBoost (klasik ML karşılaştırması)  
**Veri Seti:** UCI Phishing Websites Dataset (11.055 örnek)  
**Sonuçlar:** En iyi: XGBoost %96.89, RF %96.54  
**Öne Çıkan Nokta:** Küçük veri setinde ensemble yöntemlerin üstünlüğü gösterilmiştir. Veri seti PhiUSIIL'den çok daha küçük olduğundan doğrudan karşılaştırma sınırlıdır.

---

## 6. Kytidou vd. (2025)
**Başlık:** Machine learning techniques for phishing detection: A review of methods, challenges, and future directions  
**Dergi:** Intelligent Decision Technologies  
**DOI:** https://doi.org/10.1177/18724981251366763  
**Yöntem:** Derleme/Survey makalesi  
**Öne Çıkan Nokta:** Phishing tespitinde kullanılan ML yöntemlerini kapsamlı biçimde ele almaktadır. Veri seti dengesizliği, kavram kayması (concept drift) ve kara kutu modellerin açıklanabilirliği başlıca açık problemler olarak vurgulanmaktadır.

---

## Karşılaştırmalı Özet Tablosu

| Kaynak | Yıl | Yöntem | Veri Seti | Örnek Sayısı | Accuracy | F1 | Recall |
|--------|-----|--------|-----------|--------------|----------|----|--------|
| Prasad & Chandra [1] | 2024 | Online NB+PAC+SGD | PhiUSIIL | 235.795 | %99.24 | %99.21 | — |
| Vajrobol vd. [2] | 2024 | LR (5 özellik) | PhiUSIIL | 235.795 | **%99.97** | %99.97 | — |
| Yoon vd. [3] | 2024 | CNN+Transformer+GCN | Özel | ~200K | %98.12 | %97.89 | — |
| Rao vd. [4] | 2025 | Super Learner | PhishDump | ~50K | %98.93 | — | — |
| Taha vd. [5] | 2024 | LR, DT, RF, XGB | UCI Phishing | 11.055 | %96.89 | — | — |
| Kytidou vd. [6] | 2025 | Survey | — | — | — | — | — |
| **Bu Çalışma** | **2026** | **LR, DT, RF, XGB, SVM, KNN** | **PhiUSIIL** | **235.795** | **?** | **?** | **?** |

---

## Metodolojik Değerlendirme: Etiket Sızıntısı (Data Leakage)

### Sorun: Neden %99–100 Doğruluk?

PhiUSIIL veri setinde dört özellik, veri seti **oluşturulurken** meşru URL listesi kullanılarak hesaplanmıştır. Bu özellikler, meşru sınıfı doğrudan tanımlayan istatistiklerden türetildiğinden modele **etiketin kendisini** vermektedir; bu durum *veri seti yapısından kaynaklanan etiket sızıntısı* (construction-time leakage) olarak tanımlanır.

| Sızıntılı Özellik | Sızıntı Mekanizması |
|-------------------|---------------------|
| `URLSimilarityIndex` | Veri seti toplanırken meşru URL'ler Tranco top-10M listesinden seçilmiştir. Bu özellik, aynı liste ile kosinüs benzerliği hesaplar → meşru URL'ler için çok yüksek, phishing için çok düşük çıkar; etiketi neredeyse deterministik olarak kodlar. |
| `URLCharProb` | Karakter frekansı istatistikleri aynı meşru URL korpusundan elde edilmiştir → sınıflar arasında yapısal fark kaçınılmazdır. |
| `TLDLegitimateProb` | TLD olasılığı meşru URL istatistiklerinden türetilmiştir → meşru siteler sistematik olarak yüksek, phishing siteleri sistematik olarak düşük skor alır. |
| `URLTitleMatchScore` | Meşru sitelerin başlık–URL eşleşmesi önceden hesaplanmış meta veriyle yapılmaktadır. |

### Örnekleme Yanlılığı (Sampling Bias)

Sızıntıya ek olarak, veri seti toplama yöntemi örnekleme yanlılığı da içermektedir:
- **Meşru URL'ler:** Tranco top-10M'den alınan köklü, uzun ömürlü siteler (google.com, wikipedia.org vb.)
- **Phishing URL'leri:** PhishTank'tan alınan kısa ömürlü, genellikle rastgele oluşturulmuş domainler

Bu iki grup, URL uzunluğu, domain yaşı ve karakter kalıpları açısından kolayca ayrıştırılabilir. Bu nedenle model, gerçek dünyada karşılaşılan **sofistike phishing saldırıları** (meşru siteleri taklit eden veya güvenilir TLD kullanan) için genellenemeyebilir.

### Düzeltme: Deney 5

Bu çalışmada söz konusu dört özellik `src/preprocessing.py`'deki `LEAKY_FEATURES` listesine alınmış ve `drop_leaky=True` parametresiyle pipeline'dan çıkarılmıştır. **Deney 5**, bu temiz özellik setiyle tüm modelleri yeniden değerlendirir ve gerçekçi bir performans tahmini sunar.

Vajrobol vd. [2]'nin orijinal Deney 2 replikasyonu (`URLSimilarityIndex` dahil) kasıtlı olarak korunmuştur; Vajrobol'un %99.97 bulgusunun neden kolayca elde edilebildiğini göstermek için bu karşılaştırma öğretici değer taşımaktadır.

### Literatürdeki Diğer Çalışmalarla Karşılaştırma

Prasad & Chandra [1] ve Vajrobol vd. [2] aynı veri setini kullandığından, bildirdikleri yüksek doğruluklar kısmen bu sızıntıyı yansıtıyor olabilir. Kytidou vd. [6]'nın derleme çalışmasında da vurgulandığı üzere, phishing tespiti literatüründe veri seti kalitesi ve leakage kontrolü önemli bir açık araştırma sorunudur.

---

## Araştırma Boşlukları ve Bu Çalışmanın Katkısı

1. **Sistematik karşılaştırma eksikliği:** Vajrobol ve Prasad çalışmaları tek model/yaklaşıma odaklanmış; bu çalışma 6 farklı algoritmayı aynı ön işleme pipeline'ı ile karşılaştırmaktadır.
2. **Özellik seçimi etkisi:** Tam özellik seti vs. MI ile seçilmiş 5 özellik [2] vs. RF ile seçilmiş Top-20 özelliğin model performansına etkisi sistematik olarak incelenmektedir.
3. **Leakage önleme:** Scaler'ın sadece train setine fit edilmesi ve tanımlayıcı sütunların çıkarılması titizlikle uygulanmaktadır.
4. **Recall odağı:** Literatürde çoğunlukla accuracy raporlanırken [1][2][5], bu çalışma false negative (kaçırılan phishing) maliyetini ön plana çıkarmaktadır.
