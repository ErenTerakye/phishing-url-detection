import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

IDENTIFIER_COLUMNS = ['FILENAME', 'URL', 'Domain', 'Title']
TARGET_COLUMN = 'label'


def load_data(filepath):
    df = pd.read_csv(filepath)
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' not found."
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Class distribution:\n{df[TARGET_COLUMN].value_counts()}")
    return df


def drop_identifier_columns(df):
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped identifier columns: {cols_to_drop}")
    return df


def encode_tld(df, fit=True, encoder=None):
    if 'TLD' not in df.columns:
        return df, encoder
    if fit:
        encoder = LabelEncoder()
        df = df.copy()
        df['TLD'] = encoder.fit_transform(df['TLD'].astype(str))
    else:
        df = df.copy()
        # Handle unseen labels gracefully
        known = set(encoder.classes_)
        df['TLD'] = df['TLD'].astype(str).apply(
            lambda x: x if x in known else encoder.classes_[0]
        )
        df['TLD'] = encoder.transform(df['TLD'])
    return df, encoder


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(filepath):
    df = load_data(filepath)
    df = drop_identifier_columns(df)
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    X, tld_encoder = encode_tld(X, fit=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, tld_encoder, scaler
