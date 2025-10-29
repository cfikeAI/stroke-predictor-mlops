# src/data_prep.py
"""
Data preprocessing and baseline statistics generator for TelemetryGuard
Dataset: Stroke Prediction (Kaggle)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
RAW_PATH = "data/raw/stroke.csv"
PROC_PATH = "data/processed"
BASELINE_PATH = "data/baselines"

os.makedirs(PROC_PATH, exist_ok=True)
os.makedirs(BASELINE_PATH, exist_ok=True)

def load_data(path=RAW_PATH):
    """Load the raw dataset."""
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame):
    """Clean missing values and drop irrelevant columns."""
    df = df.copy()
    # Drop ID column if present
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    # Fill BMI with median
    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    return df

def encode_categoricals(df: pd.DataFrame):
    #Encode categorical columns using LabelEncoder.
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def compute_baselines(df: pd.DataFrame):
    #Compute per-feature baseline statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    baselines = {}

    for col in numeric_cols:
        baselines[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    for col in cat_cols:
        freq = df[col].value_counts(normalize=True).to_dict()
        baselines[col] = {"category_freqs": freq}

    pd.DataFrame.from_dict(baselines, orient='index').to_json(
        os.path.join(BASELINE_PATH, "baseline_stats.json"),
        indent=2
    )
    print("Baseline statistics saved to data/baselines/baseline_stats.json")

def save_splits(X_train, X_test, y_train, y_test):
    #Save train/test splits
    X_train.to_csv(os.path.join(PROC_PATH, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROC_PATH, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROC_PATH, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROC_PATH, "y_test.csv"), index=False)
    print("Train/test splits saved to data/processed/")

def main():
    df = load_data()
    df = clean_data(df)
    df = encode_categoricals(df)

    # Separate features and target
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save processed data
    save_splits(X_train, X_test, y_train, y_test)
    compute_baselines(df)

if __name__ == "__main__":
    main()
