import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

VAJROBOL_TOP5 = [
    'URLSimilarityIndex',   # NOTE: leaky — computed from the same legit URL corpus used for labeling
    'LineOfCode',
    'NoOfExternalRef',
    'NoOfImage',
    'NoOfSelfRef',
]

# Vajrobol-5 without the leaky URLSimilarityIndex feature
VAJROBOL_TOP4_CLEAN = [
    'LineOfCode',
    'NoOfExternalRef',
    'NoOfImage',
    'NoOfSelfRef',
]


def correlation_analysis(X, y=None, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Features with correlation > {threshold}: {to_drop}")
    return to_drop, corr_matrix


def rf_feature_importance(X_train, y_train, top_n=30):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_,
    }).sort_values('importance', ascending=False)
    top_features = importance_df.head(top_n)['feature'].tolist()
    print(f"Top-{top_n} features by RF importance: {top_features}")
    return top_features, importance_df


def mutual_info_selection(X_train, y_train, k=20):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    scores = pd.DataFrame({
        'feature': X_train.columns,
        'mi_score': selector.scores_,
    }).sort_values('mi_score', ascending=False)
    top_features = scores.head(k)['feature'].tolist()
    print(f"Top-{k} features by Mutual Information: {top_features}")
    return top_features, scores


def get_top5_vajrobol():
    return VAJROBOL_TOP5


def get_top4_vajrobol_clean():
    """Vajrobol-5 minus URLSimilarityIndex (the leaky feature)."""
    return VAJROBOL_TOP4_CLEAN
