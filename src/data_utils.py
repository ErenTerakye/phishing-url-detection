from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


TARGET_CANDIDATES = [
    "label",
    "Label",
    "class",
    "Class",
    "status",
    "Result",
    "phishing",
    "is_phishing",
]

RAW_TEXT_HINTS = ("url", "domain", "title", "filename", "file_name", "html", "content", "text")
KNOWN_LEAKY_FEATURES = [
    "URLSimilarityIndex",
    "URLCharProb",
    "TLDLegitimateProb",
    "URLTitleMatchScore",
]


@dataclass(frozen=True)
class DataMetadata:
    target_col: str
    positive_label: object
    dropped_text_columns: list[str]
    dropped_leaky_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    missing_values_before: int
    duplicate_rows: int


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and print a compact audit summary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Veri seti bulunamadi: {path}. CSV dosyasini bu yola koyun veya --data ile yeni yol verin."
        )

    df = pd.read_csv(path)
    print(f"\nVeri yuklendi: {path}")
    print(f"Boyut: {df.shape[0]:,} satir x {df.shape[1]:,} sutun")
    print("\nIlk 5 satir:")
    print(df.head())
    print("\nSutunlar:")
    print(list(df.columns))
    print("\nVeri tipleri:")
    print(df.dtypes)
    return df


def find_target_column(df: pd.DataFrame, candidates: Iterable[str] = TARGET_CANDIDATES) -> str:
    """Find the target column from common PhiUSIIL/phishing label names."""
    for col in candidates:
        if col in df.columns:
            return col

    lowered = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lowered:
            return lowered[col.lower()]

    raise ValueError(
        "Hedef degisken otomatik bulunamadi. Olası adlar: "
        + ", ".join(TARGET_CANDIDATES)
        + ". Lutfen CSV sutun adini kontrol edin."
    )


def infer_positive_label(y: pd.Series, target_col: str) -> object:
    """Infer the phishing class label for metric calculations."""
    values = [v for v in y.dropna().unique().tolist()]
    lower_map = {str(v).strip().lower(): v for v in values}

    for key in ("phishing", "phish", "malicious", "bad", "1", "true", "yes"):
        if key in lower_map:
            if key == "1" and target_col.lower() == "label" and 0 in values and 1 in values:
                # PhiUSIIL commonly uses label=0 for phishing and label=1 for legitimate.
                return 0
            return lower_map[key]

    if 0 in values and 1 in values and target_col.lower() in {"label", "result"}:
        return 0
    if 1 in values:
        return 1
    return values[0]


def _is_raw_text_column(series: pd.Series, col_name: str) -> bool:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    name = col_name.lower()
    if any(hint in name for hint in RAW_TEXT_HINTS):
        return True
    unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
    avg_len = series.dropna().astype(str).str.len().mean()
    return bool(unique_ratio > 0.5 or avg_len > 40)


def clean_data(
    df: pd.DataFrame,
    target_col: str,
    drop_leaky_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series, DataMetadata]:
    """Clean data without fitting transformations that could leak test information."""
    work = df.copy()
    work = work.replace([np.inf, -np.inf], np.nan)

    missing_values_before = int(work.isna().sum().sum())
    duplicate_rows = int(work.duplicated().sum())
    print(f"\nEksik deger sayisi: {missing_values_before:,}")
    print(f"Tekrarli satir sayisi: {duplicate_rows:,}")
    print("\nSinif dagilimi:")
    print(work[target_col].value_counts(dropna=False))

    work = work.dropna(subset=[target_col])
    y = work[target_col]
    X = work.drop(columns=[target_col])

    dropped_text_columns = [col for col in X.columns if _is_raw_text_column(X[col], col)]
    if dropped_text_columns:
        print(f"\nHam metin/tanimlayici olarak cikarilan sutunlar: {dropped_text_columns}")
        X = X.drop(columns=dropped_text_columns)

    dropped_leaky_columns = []
    if drop_leaky_features:
        dropped_leaky_columns = [col for col in KNOWN_LEAKY_FEATURES if col in X.columns]
        if dropped_leaky_columns:
            print(
                "\nVeri sizintisi riski nedeniyle cikarilan ozellikler: "
                f"{dropped_leaky_columns}"
            )
            X = X.drop(columns=dropped_leaky_columns)
    else:
        present = [col for col in KNOWN_LEAKY_FEATURES if col in X.columns]
        if present:
            print(
                "\nUYARI: Leakage riski tasiyan ozellikler deneyde tutuluyor: "
                f"{present}. Bu mod literatur replikasyonu icindir, IRL performans yorumu icin uygun degildir."
            )

    categorical_columns = [
        col
        for col in X.columns
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])
    ]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    metadata = DataMetadata(
        target_col=target_col,
        positive_label=infer_positive_label(y, target_col),
        dropped_text_columns=dropped_text_columns,
        dropped_leaky_columns=dropped_leaky_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        missing_values_before=missing_values_before,
        duplicate_rows=duplicate_rows,
    )

    print(f"Sayisal ozellik sayisi: {len(numeric_columns)}")
    print(f"Kategorik ozellik sayisi: {len(categorical_columns)}")
    print(f"Phishing pozitif sinif etiketi: {metadata.positive_label}")
    return X, y, metadata


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if groups is not None:
        n_splits = int(round(1 / test_size))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        overlap = set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
        print(f"\nSplit: StratifiedGroupKFold, group overlap: {len(overlap)}")
        print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test
