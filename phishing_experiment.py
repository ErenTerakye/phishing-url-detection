#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.data_utils import clean_data, find_target_column, load_data, split_data
from src.evaluation import evaluate_experiment_model
from src.models import build_models
from src.visualization import plot_confusion_matrix_best, plot_feature_importance, plot_results


RANDOM_STATE = 42
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 20_000
DATA_PATH = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")
OUTPUTS_DIR = Path("outputs")
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

VAJROBOL_5 = [
    "URLSimilarityIndex",
    "LineOfCode",
    "NoOfExternalRef",
    "NoOfImage",
    "NoOfSelfRef",
]

URL_ONLY_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "CharContinuationRate",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]


def ensure_output_dirs() -> None:
    for directory in (TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def maybe_sample_debug(df: pd.DataFrame, target_col: str, debug_mode: bool) -> pd.DataFrame:
    if not debug_mode or len(df) <= DEBUG_SAMPLE_SIZE:
        return df
    print(f"\nDEBUG_MODE aktif: veri {DEBUG_SAMPLE_SIZE:,} satira stratified ornekleniyor.")
    sample_df, _ = train_test_split(
        df,
        train_size=DEBUG_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )
    return sample_df.reset_index(drop=True)


def encode_target_if_needed(y: pd.Series, positive_label):
    """Encode non-numeric labels for estimators such as XGBoost."""
    if pd.api.types.is_numeric_dtype(y):
        return y.reset_index(drop=True), positive_label, None

    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y.astype(str)), name=y.name)
    positive_encoded = int(encoder.transform([str(positive_label)])[0])
    print(f"Hedef etiketleri encode edildi: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    return encoded, positive_encoded, encoder


def apply_feature_profile(X: pd.DataFrame, metadata, profile: str):
    if profile == "all":
        print("\nOzellik profili: all (URL + HTML/icerik ozellikleri, leakage kolonlari haric)")
        return X, metadata

    keep = [col for col in URL_ONLY_FEATURES if col in X.columns]
    missing = [col for col in URL_ONLY_FEATURES if col not in X.columns]
    if not keep:
        raise ValueError("URL-only profilinde kullanilabilir ozellik bulunamadi.")

    dropped = [col for col in X.columns if col not in keep]
    print("\nOzellik profili: url-only (yalnizca URL'den uretilebilir lexical ozellikler)")
    print(f"URL-only tutulan ozellik sayisi: {len(keep)}")
    if missing:
        print(f"URL-only profilde bulunamayan beklenen ozellikler: {missing}")
    print(f"HTML/icerik veya deploy edilemeyen ozellik olarak cikarilanlar: {dropped}")

    return X[keep].copy(), replace(
        metadata,
        numeric_columns=[col for col in metadata.numeric_columns if col in keep],
        categorical_columns=[col for col in metadata.categorical_columns if col in keep],
    )


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [name.split("__", 1)[-1] for name in names]
    except Exception:
        return [f"feature_{idx}" for idx in range(len(preprocessor.transformers_))]


def build_model_pipeline(model, preprocessor, feature_selector=None) -> Pipeline:
    steps = [("preprocess", clone(preprocessor))]
    if feature_selector is not None:
        steps.append(("feature_selection", clone(feature_selector)))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_set_name,
    model_name,
    feature_count,
    positive_label,
):
    """Project-level evaluation wrapper with the requested function name."""
    return evaluate_experiment_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_set_name,
        model_name,
        feature_count,
        positive_label,
    )


def build_tree_importance_scenarios(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_preprocessor: ColumnTransformer,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    fitted_preprocessor = clone(base_preprocessor)
    fitted_preprocessor.fit(X_train)
    feature_names = get_transformed_feature_names(fitted_preprocessor)
    X_train_transformed = fitted_preprocessor.transform(X_train)

    rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_transformed, y_train)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    rows = []
    scenarios = []
    transformed_count = len(feature_names)
    for k in (10, 15, 20):
        top_features = importance_df.head(min(k, transformed_count))["feature"].tolist()
        for rank, feature in enumerate(top_features, start=1):
            rows.append({"feature_set": f"S4_TREE_IMPORTANCE_TOP_{k}", "rank": rank, "feature": feature})

        selector = SelectFromModel(
            ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1),
            threshold=-np.inf,
            max_features=min(k, transformed_count),
        )
        scenarios.append(
            {
                "name": f"S4_TREE_IMPORTANCE_TOP_{k}",
                "columns": X_train.columns.tolist(),
                "selector": selector,
                "feature_count": min(k, transformed_count),
                "note": "ExtraTrees SelectFromModel pipeline icinde fit edilir",
            }
        )

    return scenarios, pd.DataFrame(rows), importance_df


def run_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metadata,
    positive_label,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    models = build_models(random_state=RANDOM_STATE)
    base_preprocessor = build_preprocessor(metadata.numeric_columns, metadata.categorical_columns)
    full_preprocessor = clone(base_preprocessor).fit(X_train)
    full_feature_count = len(get_transformed_feature_names(full_preprocessor))

    scenarios = [
        {
            "name": "S1_FULL",
            "columns": X_train.columns.tolist(),
            "selector": None,
            "feature_count": full_feature_count,
            "note": "Tum kullanilabilir ozellikler",
        }
    ]

    vajrobol_present = [col for col in VAJROBOL_5 if col in X_train.columns]
    vajrobol_missing = [col for col in VAJROBOL_5 if col not in X_train.columns]
    if vajrobol_present:
        scenarios.append(
            {
                "name": "S2_VAJROBOL_5",
                "columns": vajrobol_present,
                "selector": None,
                "feature_count": len(vajrobol_present),
                "note": f"Eksik ozellikler: {vajrobol_missing}" if vajrobol_missing else "Tam Vajrobol-5",
            }
        )
    else:
        warnings.warn("S2_VAJROBOL_5 atlandi: belirtilen ozellikler bulunamadi.", RuntimeWarning)

    for k in (5, 10, 15, 20):
        scenarios.append(
            {
                "name": f"S3_MI_TOP_{k}",
                "columns": X_train.columns.tolist(),
                "selector": SelectKBest(score_func=mutual_info_classif, k=min(k, full_feature_count)),
                "feature_count": min(k, full_feature_count),
                "note": "MI Top-K",
            }
        )

    tree_scenarios, selected_features_df, importance_df = build_tree_importance_scenarios(
        X_train, y_train, base_preprocessor
    )
    scenarios.extend(tree_scenarios)

    rows = []
    best = {"score": -1, "pipeline": None, "row": None, "y_pred": None}

    for scenario in scenarios:
        scenario_cols = scenario["columns"]
        scenario_preprocessor = build_preprocessor(
            [col for col in metadata.numeric_columns if col in scenario_cols],
            [col for col in metadata.categorical_columns if col in scenario_cols],
        )
        X_train_s = X_train[scenario_cols]
        X_test_s = X_test[scenario_cols]

        print(f"\n=== {scenario['name']} | ozellik sayisi: {scenario['feature_count']} ===")
        if scenario.get("note"):
            print(scenario["note"])

        for model_name, model in models.items():
            pipeline = build_model_pipeline(model, scenario_preprocessor, scenario["selector"])
            try:
                result, fitted_pipeline, y_pred = evaluate_model(
                    pipeline,
                    X_train_s,
                    X_test_s,
                    y_train,
                    y_test,
                    scenario["name"],
                    model_name,
                    scenario["feature_count"],
                    positive_label,
                )
                rows.append(result)
                print(
                    f"{model_name}: F1={result['F1-score']:.4f}, "
                    f"Recall={result['Recall']:.4f}, "
                    f"Accuracy={result['Accuracy']:.4f}, "
                    f"Train={result['Training Time (s)']:.2f}s"
                )

                if result["F1-score"] > best["score"]:
                    best = {
                        "score": result["F1-score"],
                        "pipeline": fitted_pipeline,
                        "row": result,
                        "y_pred": y_pred,
                        "columns": scenario_cols,
                    }
            except Exception as exc:
                warnings.warn(f"{scenario['name']} / {model_name} atlandi: {exc}", RuntimeWarning)
                rows.append(
                    {
                        "Feature Set": scenario["name"],
                        "Model": model_name,
                        "Accuracy": np.nan,
                        "Precision": np.nan,
                        "Recall": np.nan,
                        "F1-score": np.nan,
                        "ROC-AUC": np.nan,
                        "Training Time (s)": np.nan,
                        "Prediction Time (s)": np.nan,
                        "Feature Count": scenario["feature_count"],
                        "Confusion Matrix": None,
                        "Error": str(exc),
                    }
                )

    results_df = pd.DataFrame(rows)
    return results_df, best, selected_features_df, importance_df


def save_results(results_df: pd.DataFrame, selected_features_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    save_df = results_df.copy()
    save_df["Confusion Matrix"] = save_df["Confusion Matrix"].astype(str)
    save_df.to_csv(TABLES_DIR / "results_summary.csv", index=False)
    selected_features_df.to_csv(TABLES_DIR / "selected_features.csv", index=False)
    importance_df.to_csv(TABLES_DIR / "tree_feature_importance.csv", index=False)
    print(f"\nSonuc tablosu kaydedildi: {TABLES_DIR / 'results_summary.csv'}")


def generate_report_comment(results_df: pd.DataFrame) -> str:
    valid = results_df.dropna(subset=["Accuracy", "Recall", "F1-score", "Training Time (s)"]).copy()
    if valid.empty:
        return "Deneyler tamamlanamadi; rapor yorumu icin gecerli metrik bulunamadi."

    best_acc = valid.loc[valid["Accuracy"].idxmax()]
    best_recall = valid.loc[valid["Recall"].idxmax()]
    best_f1 = valid.loc[valid["F1-score"].idxmax()]
    fastest = valid.loc[valid["Training Time (s)"].idxmin()]

    full_best = valid[valid["Feature Set"] == "S1_FULL"]["F1-score"].max()
    compact = valid[valid["Feature Set"] != "S1_FULL"].sort_values("Feature Count").copy()
    compact_close = "hayir"
    compact_sentence = "Az ozellikli senaryolarda karsilastirma icin yeterli sonuc uretilemedi."
    if not compact.empty and pd.notna(full_best):
        compact["f1_gap"] = full_best - compact["F1-score"]
        close = compact[compact["f1_gap"] <= 0.01].sort_values(["Feature Count", "f1_gap"]).head(1)
        if not close.empty:
            compact_close = "evet"
            row = close.iloc[0]
            compact_sentence = (
                f"Az ozellikli {row['Feature Set']} senaryosu, tum ozellikli en iyi modele "
                f"{row['f1_gap']:.4f} F1 farki ile yakin basari vermistir."
            )
        else:
            row = compact.sort_values("f1_gap").iloc[0]
            compact_sentence = (
                f"Az ozellikli senaryolar tum ozellikli modele tam olarak yaklasamamistir; "
                f"en yakin sonuc {row['Feature Set']} ile {row['f1_gap']:.4f} F1 farkidir."
            )

    interpretable = valid[
        valid["Model"].isin(["Logistic Regression", "Decision Tree", "GaussianNB", "SGDClassifier"])
    ].sort_values(["F1-score", "Training Time (s)"], ascending=[False, True])
    recommendation = best_f1
    if not interpretable.empty and interpretable.iloc[0]["F1-score"] >= best_f1["F1-score"] - 0.01:
        recommendation = interpretable.iloc[0]

    text = (
        "Deneysel sonuclar, veri sizintisi riski tasiyan ozellikler disarida birakilip "
        "URL-only deploy profili ve domain bazli test ayrimi kullanildiginda "
        "yalnizca dogruluk degerine gore model secmenin yeterli olmadigini gostermektedir. "
        f"En yuksek accuracy {best_acc['Model']} modeli ile {best_acc['Feature Set']} senaryosunda "
        f"{best_acc['Accuracy']:.4f} olarak elde edilmistir. "
        f"En yuksek recall {best_recall['Model']} modeli ile {best_recall['Feature Set']} senaryosunda "
        f"{best_recall['Recall']:.4f}; en iyi F1-score ise {best_f1['Model']} modeli ile "
        f"{best_f1['Feature Set']} senaryosunda {best_f1['F1-score']:.4f} olmustur. "
        f"{compact_sentence} "
        f"En hizli egitim {fastest['Model']} modeli ile {fastest['Feature Set']} senaryosunda "
        f"{fastest['Training Time (s)']:.4f} saniyede gerceklesmistir. "
        f"Basari-maliyet-yorumlanabilirlik dengesi acisindan {recommendation['Model']} + "
        f"{recommendation['Feature Set']} kombinasyonu onerilebilir; bu secim yuksek basariyi daha dusuk karmasiklik "
        "ve raporlanabilir karar mantigi ile dengelemektedir. "
        "Bu calismada literaturdeki kaynaklardan farkli olarak yalnizca tek bir model veya tek bir ozellik secimi "
        "yontemi kullanilmamis; PhiUSIIL veri seti uzerinde farkli algoritmalar ve farkli ozellik kumeleri ayni "
        "deneysel cercevede karsilastirilmistir. Boylece phishing URL tespitinde en yuksek dogrulugun yaninda "
        "daha az ozellikle calisabilen, daha hizli ve daha yorumlanabilir modellerin uygulanabilirligi de "
        f"degerlendirilmistir. Az ozellikli model tum ozellikli modele yakin mi?: {compact_close}."
    )
    return text


def write_best_model_outputs(best: dict, y_test: pd.Series, positive_label, target_encoder) -> None:
    if best["pipeline"] is None:
        return

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "columns": best["columns"],
            "positive_label": positive_label,
            "target_encoder": target_encoder,
            "metrics": best["row"],
        },
        model_path,
    )

    labels = [positive_label] + [label for label in pd.Series(y_test).unique().tolist() if label != positive_label]
    plot_confusion_matrix_best(
        y_test,
        best["y_pred"],
        labels=labels,
        title=f"En Iyi Model Confusion Matrix: {best['row']['Model']} / {best['row']['Feature Set']}",
        output_path=FIGURES_DIR / "confusion_matrix_best_model.png",
    )

    with open(TABLES_DIR / "best_model_metrics.txt", "w", encoding="utf-8") as file:
        for key, value in best["row"].items():
            file.write(f"{key}: {value}\n")
        file.write(f"Saved model: {model_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhiUSIIL phishing URL final experiment runner")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="CSV veri seti yolu")
    parser.add_argument("--debug", action="store_true", help="20.000 satirlik hizli deneme modu")
    parser.add_argument(
        "--include-leaky",
        action="store_true",
        help="URLSimilarityIndex gibi veri sizintisi riski tasiyan ozellikleri dahil et; IRL yorum icin onerilmez.",
    )
    parser.add_argument(
        "--split",
        choices=["group", "random"],
        default="group",
        help="Varsayilan group: ayni Domain train ve testte birlikte bulunmaz. random eski holdout davranisidir.",
    )
    parser.add_argument(
        "--profile",
        choices=["url-only", "all"],
        default="url-only",
        help="Varsayilan url-only daha gercekci deploy senaryosudur. all URL+HTML ozelliklerini kullanir.",
    )
    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    ensure_output_dirs()

    df = load_data(args.data)
    target_col = find_target_column(df)
    df = maybe_sample_debug(df, target_col, args.debug or DEBUG_MODE)
    groups = None
    if args.split == "group":
        if "Domain" in df.columns:
            groups = df["Domain"].astype(str).reset_index(drop=True)
        else:
            warnings.warn("Domain sutunu bulunamadi; random stratified split kullanilacak.", RuntimeWarning)

    X, y, metadata = clean_data(df, target_col, drop_leaky_features=not args.include_leaky)
    X, metadata = apply_feature_profile(X, metadata, args.profile)
    y, positive_label, target_encoder = encode_target_if_needed(y, metadata.positive_label)
    X = X.reset_index(drop=True)

    X_train, X_test, y_train, y_test = split_data(X, y, groups=groups)
    results_df, best, selected_features_df, importance_df = run_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
        metadata,
        positive_label,
    )

    save_results(results_df, selected_features_df, importance_df)
    plot_results(results_df.dropna(subset=["Accuracy"]), FIGURES_DIR)
    plot_feature_importance(importance_df, FIGURES_DIR / "feature_importance_top20.png", top_n=20)
    write_best_model_outputs(best, y_test, positive_label, target_encoder)

    comment = generate_report_comment(results_df)
    with open(TABLES_DIR / "report_comment_tr.txt", "w", encoding="utf-8") as file:
        file.write(comment + "\n")

    elapsed = time.perf_counter() - start
    print("\nAkademik yorum:")
    print(comment)
    print(f"\nTamamlandi. Toplam sure: {elapsed:.2f}s")
    print(f"Ciktilar: {OUTPUTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
