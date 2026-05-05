import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1-Score':  f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Legitimate', 'Phishing'],
        yticklabels=['Legitimate', 'Phishing'],
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, model in models_dict.items():
        if not hasattr(model, 'predict_proba'):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not expose feature importances.")
        return

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[idx][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top-{top_n} Feature Importances')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def compare_models_table(results_dict):
    rows = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].applymap(
        lambda x: round(x, 4) if x is not None else None
    )
    return df.sort_values('F1-Score', ascending=False)


def cross_validate_model(model, X, y, cv=10):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = {}
    for metric in metrics:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        results[metric] = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
        print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")
    return results
