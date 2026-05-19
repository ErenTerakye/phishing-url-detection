from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def _save_barplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    top_n: int | None = None,
) -> None:
    plot_df = df.copy()
    if top_n:
        plot_df = plot_df.sort_values(metric, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Model", y=metric, hue="Feature Set", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Feature Set", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_results(results_df: pd.DataFrame, figures_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _save_barplot(
        results_df,
        "F1-score",
        "Model Bazli F1-score Karsilastirmasi",
        figures_dir / "model_comparison_f1.png",
    )
    _save_barplot(
        results_df,
        "Accuracy",
        "Feature Set Bazli Accuracy Karsilastirmasi",
        figures_dir / "model_comparison_accuracy.png",
    )
    _save_barplot(
        results_df,
        "Recall",
        "Phishing Sinifi Recall Karsilastirmasi",
        figures_dir / "model_comparison_recall.png",
    )
    _save_barplot(
        results_df,
        "Training Time (s)",
        "Model Egitim Suresi Karsilastirmasi",
        figures_dir / "training_time_comparison.png",
        top_n=min(40, len(results_df)),
    )


def plot_confusion_matrix_best(
    y_true,
    y_pred,
    labels,
    title: str,
    output_path: str | Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[str(label) for label in labels],
        yticklabels=[str(label) for label in labels],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Tahmin Edilen Sinif")
    ax.set_ylabel("Gercek Sinif")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    if importance_df.empty:
        return
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature", color="#2D9CDB", ax=ax)
    ax.set_title(f"En Onemli {min(top_n, len(importance_df))} Ozellik")
    ax.set_xlabel("Random Forest Importance")
    ax.set_ylabel("Ozellik")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

