"""
preprocessing.py
────────────────
Handles loading, cleaning, scaling, and splitting the student dataset.
Returns ready-to-use train/test arrays and a fitted scaler object.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

FEATURE_COLS = ["attendance_pct", "study_hours", "assignment_score", "previous_marks"]
TARGET_COL   = "final_grade"

DEFAULT_MODEL_DIR = "models"
SCALER_PATH  = os.path.join(DEFAULT_MODEL_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(DEFAULT_MODEL_DIR, "imputer.pkl")


def load_and_preprocess(
    csv_path: str = "data/student_data.csv",
    test_size: float = 0.20,
    random_state: int = 42,
    save_artifacts: bool = True,
    model_dir: str = None,
):
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Load CSV
    2. Drop exact duplicate rows
    3. Median-impute missing feature values
    4. Scale features with StandardScaler
    5. 80/20 train-test split (stratified by letter_grade for balance)

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names                     : list[str]
    scaler                            : fitted StandardScaler
    df_clean                          : cleaned DataFrame (before scaling)
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[preprocessing] Loaded {len(df)} rows, {df.shape[1]} columns")

    # ── 2. Drop duplicates ────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocessing] Dropped {before - len(df)} duplicate rows")

    # ── 3. Impute missing values (median) ─────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    print(f"[preprocessing] Missing values imputed (strategy=median)")

    df_clean = df.copy()

    # ── 4. Feature / target split ─────────────────────────────────────────────
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # ── 5. Scale ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 6. Train / Test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    print(f"[preprocessing] Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 7. Persist scaler & imputer ───────────────────────────────────────────
    if save_artifacts:
        _model_dir    = model_dir if model_dir else DEFAULT_MODEL_DIR
        _scaler_path  = os.path.join(_model_dir, "scaler.pkl")
        _imputer_path = os.path.join(_model_dir, "imputer.pkl")
        os.makedirs(_model_dir, exist_ok=True)
        joblib.dump(scaler,  _scaler_path)
        joblib.dump(imputer, _imputer_path)
        print(f"[preprocessing] Scaler  → {_scaler_path}")
        print(f"[preprocessing] Imputer → {_imputer_path}")

    return X_train, X_test, y_train, y_test, FEATURE_COLS, scaler, df_clean


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feats, scaler, df = load_and_preprocess()
    print(f"\nFeatures : {feats}")
    print(f"X_train  : {X_train.shape}")
    print(f"X_test   : {X_test.shape}")
