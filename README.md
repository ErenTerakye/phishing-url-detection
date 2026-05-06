# Phishing URL Tespiti — PhiUSIIL Veri Seti ile Karşılaştırmalı Makine Öğrenmesi Analizi

Eskişehir Osmangazi Üniversitesi, Bilgisayar Mühendisliği Bölümü  
Makine Öğrenmesi Dersi — Dönem Projesi (2025–2026 Bahar)

---

## Proje Özeti

Bu proje, UCI Machine Learning Repository'den alınan **PhiUSIIL Phishing URL Dataset** (235.795 örnek, 54 özellik) üzerinde çoklu makine öğrenmesi modellerini karşılaştırmaktadır. Hedef; Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM ve KNN algoritmalarını sistematik bir deney protokolüyle değerlendirmek ve literatürdeki çalışmalarla sonuçları karşılaştırmaktır.

---

## Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | [UCI ML Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) |
| Toplam örnek | 235.795 |
| Özellik sayısı | 54 (4 tanımlayıcı sütun çıkarıldıktan sonra 50 + encoded TLD) |
| Meşru URL | 134.850 (%57.2) |
| Phishing URL | 100.945 (%42.8) |
| Eksik değer | Yok |
| Hedef değişken | `label` (0=Meşru, 1=Phishing) |

---

## Kurulum

```bash
# Sanal ortam oluştur (önerilir)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

Veri setini indirip `data/` klasörüne koy:
```
data/PhiUSIIL_Phishing_URL_Dataset.csv
```

---

## Kullanım

Notebook'ları sırayla çalıştır:

| # | Notebook | İçerik |
|---|----------|--------|
| 1 | `notebooks/01_EDA.ipynb` | Keşifsel veri analizi, dağılımlar, korelasyon |
| 2 | `notebooks/02_preprocessing.ipynb` | Veri ön işleme, leakage önleme |
| 3 | `notebooks/03_feature_selection.ipynb` | RF önem skorları, MI, korelasyon analizi |
| 4 | `notebooks/04_model_training.ipynb` | Tüm deneylerin eğitimi |
| 5 | `notebooks/05_results_analysis.ipynb` | Karşılaştırmalı analiz, grafikler |

---

## Klasör Yapısı

```
phishing-url-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── data/               # CSV buraya konulur (repo'ya eklenmez)
├── notebooks/          # Sıralı Jupyter notebook'lar
├── src/                # Yeniden kullanılabilir Python modülleri
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   ├── figures/        # PNG grafikler (dpi=150)
│   └── metrics/        # CSV metrik tabloları
└── docs/
    └── literature_summary.md
```

---

## Deney Protokolü

| Deney | Özellik Seti | Amaç |
|-------|-------------|------|
| 1 | Tüm özellikler (50+TLD) | Baseline karşılaştırma |
| 2 | Vajrobol'un 5 özelliği | Literatür replikasyonu |
| 3 | RF Top-20 | Özellik seçiminin etkisi |
| 4 | 10-Fold CV | Genellenebilirlik kontrolü |

---

## Hedef Metrikler

| Metrik | Hedef |
|--------|-------|
| Accuracy | ≥ %97 |
| F1-Score | ≥ %96 |
| **Recall** | **≥ %97** ← Odak metrik |
| ROC-AUC | ≥ 0.98 |

> Recall odak metriktir çünkü False Negative (kaçırılan phishing) en kritik hatadır.

---

## Literatür Karşılaştırması

| Kaynak | Yöntem | Veri Seti | Accuracy |
|--------|--------|-----------|----------|
| Prasad & Chandra (2024) | BernoulliNB+PAC+SGD | PhiUSIIL | %99.24 |
| **Vajrobol vd. (2024)** | **LR (5 özellik)** | **PhiUSIIL** | **%99.97** |
| Yoon vd. (2024) | CNN+Transformer+GCN | Common Crawl | %98.12 |
| Rao vd. (2025) | Super Learner Ensemble | PhishDump | %98.93 |
| Taha vd. (2024) | LR, DT, RF, XGB | Phishing Websites | %96.89 |

---

## Kaynakça

[1] A. Prasad and S. Chandra, "PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning," *Computers & Security*, 2024. doi: [10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)

[2] V. Vajrobol, B. B. Gupta, and A. Gaurav, "Mutual information based logistic regression for phishing URL detection," *Cyber Security and Applications*, 2024. doi: [10.1016/j.csa.2024.100044](https://doi.org/10.1016/j.csa.2024.100044)

[3] J.-H. Yoon, S.-J. Buu, and H.-J. Kim, "Phishing Webpage Detection via Multi-Modal Integration of HTML DOM Graphs and URL Features Based on Graph Convolutional and Transformer Networks," *Electronics*, vol. 13, no. 16, 2024. doi: [10.3390/electronics13163344](https://doi.org/10.3390/electronics13163344)

[4] R. S. Rao, C. Kondaiah, A. R. Pais, and B. Lee, "A hybrid super learner ensemble for phishing detection on mobile devices," *Scientific Reports*, 2025. doi: [10.1038/s41598-025-02009-8](https://doi.org/10.1038/s41598-025-02009-8)

[5] M. A. Taha, H. D. A. Jabar, and W. K. Mohammed, "A Machine Learning Algorithms for Detecting Phishing Websites: A Comparative Study," *Iraqi Journal for Computer Science and Mathematics*, 2024. doi: [10.52866/ijcsm.2024.05.03.015](https://doi.org/10.52866/ijcsm.2024.05.03.015)

[6] E. Kytidou, T. Tsikriki, G. Drosatos, and K. Rantos, "Machine learning techniques for phishing detection: A review of methods, challenges, and future directions," *Intelligent Decision Technologies*, 2025. doi: [10.1177/18724981251366763](https://doi.org/10.1177/18724981251366763)

---

## Lisans

MIT License © 2026
