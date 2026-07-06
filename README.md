# Phishing URL Tespiti - PhiUSIIL Karşılaştırmalı ML Deneyleri

## Amaç

Bu proje, PhiUSIIL phishing URL veri seti üzerinde yalnızca en yüksek doğruluk değerini aramak yerine farklı makine öğrenmesi modellerini, farklı özellik seçimi stratejilerini ve başarı-maliyet-yorumlanabilirlik dengesini karşılaştırmalı olarak analiz eder.

## Veri Seti

- Veri seti: PhiUSIIL Phishing URL Dataset
- Beklenen dosya yolu: `data/PhiUSIIL_Phishing_URL_Dataset.csv`
- Yaklaşık boyut: 235.795 örnek, 54+ sütun
- Hedef değişken otomatik aranır: `label`, `Label`, `class`, `Class`, `status`, `Result`, `phishing`, `is_phishing`
- Ham metin/tanımlayıcı sütunlar, örneğin `URL`, `Domain`, `Title`, modelden çıkarılır.

## Kullanılan Modeller

- Logistic Regression
- Decision Tree
- Random Forest
- Linear SVM
- XGBoost, kurulu değilse otomatik atlanır
- GaussianNB
- SGDClassifier

## Özellik Seçimi Senaryoları

- `S1_FULL`: Tüm kullanılabilir özellikler
- `S2_VAJROBOL_5`: `URLSimilarityIndex`, `LineOfCode`, `NoOfExternalRef`, `NoOfImage`, `NoOfSelfRef`
- `S3_MI_TOP_K`: Mutual Information ile Top 5, 10, 15, 20
- `S4_TREE_IMPORTANCE_TOP_K`: Tree importance ile Top 10, 15, 20

Feature selection adımları pipeline içinde çalışır; selector test verisine fit edilmez.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

Hızlı doğrulama için 20.000 satırlık debug modu:

```bash
.venv/bin/python phishing_experiment.py --debug
```

Tüm veri seti ile final deneyleri:

```bash
.venv/bin/python phishing_experiment.py
```

Varsayılan özellik profili `url-only` profilidir. Bu profil yalnızca URL'den
üretilebilir lexical özellikleri kullanır ve HTML/içerik tabanlı özellikleri
dışarıda bırakır. Bu, gerçek dünyada URL metni geldiğinde hızlı karar veren bir
phishing URL tespit sistemi için daha dürüst bir senaryodur.

URL + HTML/içerik özelliklerinin tamamını karşılaştırma amacıyla çalıştırmak
için:

```bash
.venv/bin/python phishing_experiment.py --profile all
```

Varsayılan çalışma leakage-aware moddur. `URLSimilarityIndex`, `URLCharProb`,
`TLDLegitimateProb` ve `URLTitleMatchScore` gibi veri setinin oluşturulma
sürecinden gelen, IRL senaryoda etikete fazla yakın davranabilecek özellikler
dışarıda bırakılır.

Varsayılan split de domain bazlıdır: aynı `Domain` değeri train ve test
tarafında birlikte bulunmaz. Bu, rastgele satır bölmeye göre daha gerçekçi bir
genelleme testidir.

Literatürdeki yüksek skorları replikasyon amacıyla görmek için:

```bash
.venv/bin/python phishing_experiment.py --include-leaky
```

Bu mod final yorumunda gerçek dünya performansı gibi sunulmamalıdır.

Eski rastgele holdout protokolünü karşılaştırma amacıyla çalıştırmak için:

```bash
.venv/bin/python phishing_experiment.py --split random
```

Farklı veri yolu:

```bash
.venv/bin/python phishing_experiment.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv
```

## Üretilen Çıktılar

`outputs/tables/`

- `results_summary.csv`
- `selected_features.csv`
- `tree_feature_importance.csv`
- `best_model_metrics.txt`
- `report_comment_tr.txt`

`outputs/figures/`

- `model_comparison_f1.png`
- `model_comparison_accuracy.png`
- `model_comparison_recall.png`
- `training_time_comparison.png`
- `confusion_matrix_best_model.png`
- `feature_importance_top20.png`

`outputs/models/`

- `best_model.joblib`

## Kod Yapısı

```text
phishing-url-detection/
├── data/
├── notebooks/
├── src/
│   ├── data_utils.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   ├── models/
│   └── tables/
├── phishing_experiment.py
├── requirements.txt
└── README.md
```
