#!/usr/bin/env python3
"""Run final leakage-aware phishing URL detection experiments.

Outputs are written under results/metrics and results/figures so the report,
presentation, and video all use the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_model, get_model_scores  # noqa: E402
from src.feature_selection import get_top5_vajrobol, rf_feature_importance  # noqa: E402
from src.models import get_baseline_models, train_model  # noqa: E402
from src.preprocessing import (  # noqa: E402
    IDENTIFIER_COLUMNS,
    LEGITIMATE_LABEL,
    LEAKY_FEATURES,
    PHISHING_LABEL,
    TARGET_COLUMN,
    drop_identifier_columns,
    drop_leaky_features,
    full_preprocessing_pipeline,
    load_data,
)


DATA_PATH = ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

LITERATURE_ROWS = [
    {
        "Kaynak": "Prasad & Chandra (2024)",
        "Yontem": "BernoulliNB + Passive-Aggressive + SGD",
        "Veri Seti": "PhiUSIIL",
        "Accuracy": 0.9924,
        "F1-Score": 0.9921,
        "Recall": np.nan,
        "Not": "PhiUSIIL veri setini ve incremental learning yaklasimini tanitir.",
    },
    {
        "Kaynak": "Vajrobol vd. (2024)",
        "Yontem": "Mutual information + Logistic Regression",
        "Veri Seti": "PhiUSIIL",
        "Accuracy": 0.9997,
        "F1-Score": 0.9997,
        "Recall": np.nan,
        "Not": "URLSimilarityIndex dahil 5 ozellik kullanir.",
    },
    {
        "Kaynak": "Yoon vd. (2024)",
        "Yontem": "CNN + Transformer + GCN",
        "Veri Seti": "Common Crawl + PhishTank",
        "Accuracy": 0.9812,
        "F1-Score": 0.9789,
        "Recall": np.nan,
        "Not": "URL/HTML/graf temsillerini birlikte kullanir.",
    },
    {
        "Kaynak": "Rao vd. (2025)",
        "Yontem": "Hybrid Super Learner ensemble",
        "Veri Seti": "PhishDump",
        "Accuracy": 0.9893,
        "F1-Score": np.nan,
        "Recall": np.nan,
        "Not": "Mobil cihaz senaryosunda stacking yaklasimi.",
    },
    {
        "Kaynak": "Taha vd. (2024)",
        "Yontem": "LR, DT, RF, AdaBoost, XGBoost",
        "Veri Seti": "UCI Phishing Websites",
        "Accuracy": 0.9689,
        "F1-Score": np.nan,
        "Recall": np.nan,
        "Not": "Kucuk veri setinde klasik ML karsilastirmasi.",
    },
]


def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = METRICS_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {path.relative_to(ROOT)}")
    return path


def plot_class_distribution(y: pd.Series) -> None:
    counts = y.value_counts().reindex([LEGITIMATE_LABEL, PHISHING_LABEL])
    labels = ["Legitimate (label=1)", "Phishing (label=0)"]
    colors = ["#2F80ED", "#EB5757"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n{count / total:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("PhiUSIIL Class Distribution")
    ax.set_ylabel("Number of URLs")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def compare_models_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["F1-Score", "Recall"], ascending=False).reset_index(drop=True)


def run_experiment(
    label: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\n=== {label} ===")
    results: dict[str, dict[str, float]] = {}
    trained: dict[str, object] = {}

    for name, model in get_baseline_models().items():
        print(f"[{name}]")
        model, elapsed = train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Train Time (s)"] = round(elapsed, 3)
        results[name] = metrics
        trained[name] = model
        print(
            "  "
            + " | ".join(
                f"{metric}={metrics[metric]:.4f}"
                for metric in ["Accuracy", "Recall", "F1-Score", "ROC-AUC"]
            )
        )

    table = compare_models_table(results)
    table.insert(0, "Experiment", label)
    return table, trained


def plot_metric_comparison(exp_tables: dict[str, pd.DataFrame]) -> None:
    rows = []
    for exp_name, table in exp_tables.items():
        tmp = table.copy()
        tmp["ExperimentShort"] = exp_name
        rows.append(tmp)
    all_df = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=(24, 5), sharey=True)
    for ax, metric in zip(axes, METRIC_COLS):
        sns.barplot(
            data=all_df,
            x="Model",
            y=metric,
            hue="ExperimentShort",
            ax=ax,
            palette="Set2",
        )
        ax.axhline(0.97, color="#B00020", linestyle="--", linewidth=0.9)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel(metric if metric == METRIC_COLS[0] else "")
        ax.set_ylim(0.85, 1.01)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Model and Experiment Metric Comparison", fontsize=14, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_metric_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_leakage_delta(full_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    full = full_df.set_index("Model")
    clean = clean_df.set_index("Model")
    delta = full[METRIC_COLS].sub(clean[METRIC_COLS]).reset_index()
    delta.insert(0, "Comparison", "Full - Leakage-Free")
    save_table(delta, "leakage_delta.csv")

    show_metrics = ["Accuracy", "Recall", "F1-Score"]
    plot_df = delta.melt(id_vars=["Model"], value_vars=show_metrics, var_name="Metric", value_name="Delta")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Model", y="Delta", hue="Metric", ax=ax, palette="Set1")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Performance Inflation Estimate: Full Features - Leakage-Free")
    ax.set_ylabel("Metric Delta")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leakage_delta.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return delta


def plot_confusion_for_best(clean_table: pd.DataFrame, trained: dict[str, object], X_test, y_test) -> str:
    best_name = clean_table.iloc[0]["Model"]
    y_pred = trained[best_name].predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[LEGITIMATE_LABEL, PHISHING_LABEL])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - Leakage-Free Best Model ({best_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_leakage_free_best.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return best_name


def plot_roc_curves(trained: dict[str, object], X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in trained.items():
        scores = get_model_scores(model, X_test, positive_label=PHISHING_LABEL)
        if scores is None:
            continue
        y_binary = (y_test == PHISHING_LABEL).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, scores)
        auc = roc_auc_score(y_binary, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    ax.set_title("ROC Curves - Leakage-Free Experiment")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves_leakage_free.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, filename: str) -> None:
    top = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], color="#2D9CDB")
    ax.set_title("Random Forest Top-20 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def leakage_feature_correlations(raw_df: pd.DataFrame) -> pd.DataFrame:
    available = [f for f in LEAKY_FEATURES if f in raw_df.columns]
    rows = []
    for feature in available:
        rows.append(
            {
                "Feature": feature,
                "Abs Pearson Corr With Label": abs(raw_df[feature].corr(raw_df[TARGET_COLUMN])),
            }
        )
    corr_df = pd.DataFrame(rows).sort_values("Abs Pearson Corr With Label", ascending=False)
    save_table(corr_df, "leaky_feature_correlations.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=corr_df, y="Feature", x="Abs Pearson Corr With Label", ax=ax, color="#EB5757")
    ax.set_title("Leaky Feature Correlation With Label")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leaky_feature_correlations.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return corr_df


def build_cv_pipeline(model, X_raw: pd.DataFrame) -> Pipeline:
    categorical = ["TLD"] if "TLD" in X_raw.columns else []
    numeric = [c for c in X_raw.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "tld",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline([("preprocess", preprocessor), ("model", clone(model))])


def run_cv_for_top_models(raw_df: pd.DataFrame, top_models: list[str]) -> pd.DataFrame:
    print("\n=== 10-fold CV on leakage-free top models ===")
    clean = drop_identifier_columns(raw_df.copy())
    clean = drop_leaky_features(clean)
    y = clean[TARGET_COLUMN]
    X = clean.drop(columns=[TARGET_COLUMN])
    base_models = get_baseline_models()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    def phishing_roc_auc(estimator, X_fold, y_fold):
        scores = get_model_scores(estimator, X_fold, positive_label=PHISHING_LABEL)
        return roc_auc_score((y_fold == PHISHING_LABEL).astype(int), scores)

    scoring = {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, pos_label=PHISHING_LABEL, zero_division=0),
        "Recall": make_scorer(recall_score, pos_label=PHISHING_LABEL, zero_division=0),
        "F1-Score": make_scorer(f1_score, pos_label=PHISHING_LABEL, zero_division=0),
        "ROC-AUC": phishing_roc_auc,
    }
    rows = []

    for name in top_models:
        print(f"[CV] {name}")
        pipeline = build_cv_pipeline(base_models[name], X)
        start = time.time()
        scores = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        elapsed = time.time() - start
        row = {"Model": name, "CV Time (s)": round(elapsed, 3)}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric} Mean"] = values.mean()
            row[f"{metric} Std"] = values.std()
        rows.append(row)

    return pd.DataFrame(rows)


def add_our_rows_to_literature(clean_table: pd.DataFrame, vajrobol_table: pd.DataFrame) -> pd.DataFrame:
    best = clean_table.iloc[0]
    lr_v = vajrobol_table[vajrobol_table["Model"] == "Logistic Regression"].iloc[0]
    rows = LITERATURE_ROWS + [
        {
            "Kaynak": "Bu Calisma (2026)",
            "Yontem": f"Leakage-free {best['Model']}",
            "Veri Seti": "PhiUSIIL",
            "Accuracy": best["Accuracy"],
            "F1-Score": best["F1-Score"],
            "Recall": best["Recall"],
            "Not": "Sizintili ozellikler cikarilarak degerlendirildi.",
        },
        {
            "Kaynak": "Bu Calisma (2026)",
            "Yontem": "Vajrobol-5 Logistic Regression replikasyonu",
            "Veri Seti": "PhiUSIIL",
            "Accuracy": lr_v["Accuracy"],
            "F1-Score": lr_v["F1-Score"],
            "Recall": lr_v["Recall"],
            "Not": "URLSimilarityIndex dahil literatur replikasyonu.",
        },
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--skip-cv", action="store_true", help="Skip 10-fold CV for faster iteration.")
    parser.add_argument("--cv-only", action="store_true", help="Only run 10-fold CV using existing leakage-free metrics.")
    parser.add_argument(
        "--cv-top-n",
        type=int,
        default=3,
        help="Number of leakage-free top models to cross-validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    raw_df = load_data(args.data)
    if args.cv_only:
        clean_metrics_path = METRICS_DIR / "experiment_4_leakage_free.csv"
        if not clean_metrics_path.exists():
            raise FileNotFoundError("Run holdout experiments before --cv-only.")
        clean_table = pd.read_csv(clean_metrics_path)
        top_models = clean_table["Model"].head(args.cv_top_n).tolist()
        cv_df = run_cv_for_top_models(raw_df, top_models)
        cv_path = save_table(cv_df, "cv_leakage_free_top3.csv")
        summary_path = METRICS_DIR / "experiment_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            summary.setdefault("metrics_files", {})["cv"] = str(cv_path.relative_to(ROOT))
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\nDone.")
        return

    plot_class_distribution(raw_df[TARGET_COLUMN])
    leakage_feature_correlations(raw_df)

    X_train, X_test, y_train, y_test, _, _ = full_preprocessing_pipeline(args.data, drop_leaky=False)
    X_train_clean, X_test_clean, y_train_clean, y_test_clean, _, _ = full_preprocessing_pipeline(
        args.data, drop_leaky=True
    )

    vajrobol_5 = [f for f in get_top5_vajrobol() if f in X_train.columns]
    if len(vajrobol_5) != 5:
        raise RuntimeError(f"Expected 5 Vajrobol features, found: {vajrobol_5}")

    rf_top20, rf_importance = rf_feature_importance(X_train, y_train, top_n=20)
    save_table(rf_importance, "rf_feature_importance_full.csv")
    plot_feature_importance(rf_importance, "feature_importance_rf_top20.png")

    exp1, trained_exp1 = run_experiment("Exp1_Full_Features", X_train, X_test, y_train, y_test)
    exp2, trained_exp2 = run_experiment("Exp2_Vajrobol_5", X_train[vajrobol_5], X_test[vajrobol_5], y_train, y_test)
    exp3, trained_exp3 = run_experiment("Exp3_RF_Top20", X_train[rf_top20], X_test[rf_top20], y_train, y_test)
    exp4, trained_exp4 = run_experiment(
        "Exp4_Leakage_Free", X_train_clean, X_test_clean, y_train_clean, y_test_clean
    )

    save_table(exp1, "experiment_1_full_features.csv")
    save_table(exp2, "experiment_2_vajrobol_5.csv")
    save_table(exp3, "experiment_3_rf_top20.csv")
    save_table(exp4, "experiment_4_leakage_free.csv")

    plot_metric_comparison(
        {
            "Full": exp1,
            "Vajrobol-5": exp2,
            "RF Top-20": exp3,
            "Leakage-Free": exp4,
        }
    )
    delta = plot_leakage_delta(exp1, exp4)
    best_model = plot_confusion_for_best(exp4, trained_exp4, X_test_clean, y_test_clean)
    plot_roc_curves(trained_exp4, X_test_clean, y_test_clean)

    literature = add_our_rows_to_literature(exp4, exp2)
    save_table(literature, "literature_comparison.csv")

    cv_path = None
    if not args.skip_cv:
        top_models = exp4["Model"].head(args.cv_top_n).tolist()
        cv_df = run_cv_for_top_models(raw_df, top_models)
        cv_path = save_table(cv_df, "cv_leakage_free_top3.csv")

    summary = {
        "data_path": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "n_rows": int(raw_df.shape[0]),
        "n_columns": int(raw_df.shape[1]),
        "class_counts": {str(k): int(v) for k, v in raw_df[TARGET_COLUMN].value_counts().sort_index().items()},
        "leaky_features_removed": LEAKY_FEATURES,
        "vajrobol_features": vajrobol_5,
        "rf_top20": rf_top20,
        "best_leakage_free_model": best_model,
        "metrics_files": {
            "exp1": "results/metrics/experiment_1_full_features.csv",
            "exp2": "results/metrics/experiment_2_vajrobol_5.csv",
            "exp3": "results/metrics/experiment_3_rf_top20.csv",
            "exp4": "results/metrics/experiment_4_leakage_free.csv",
            "delta": "results/metrics/leakage_delta.csv",
            "literature": "results/metrics/literature_comparison.csv",
            "cv": str(cv_path.relative_to(ROOT)) if cv_path else None,
        },
        "figure_files": sorted(str(p.relative_to(ROOT)) for p in FIGURES_DIR.glob("*.png")),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METRICS_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {(METRICS_DIR / 'experiment_summary.json').relative_to(ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
