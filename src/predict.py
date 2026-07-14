"""
predict.py
──────────
Loads the best saved model + scaler and exposes a simple predict() API.
Can also be run from the CLI for quick one-off predictions.
"""

import os
import sys
import argparse
import numpy as np
import joblib

# Resolve paths relative to this file's location (i.e., the src/ directory),
# then go one level up to reach the project root where models/ lives.
_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SRC_DIR)

MODEL_PATH   = os.path.join(_ROOT_DIR, "models", "best_model.pkl")
SCALER_PATH  = os.path.join(_ROOT_DIR, "models", "scaler.pkl")
IMPUTER_PATH = os.path.join(_ROOT_DIR, "models", "imputer.pkl")

FEATURE_NAMES = ["attendance_pct", "study_hours", "assignment_score", "previous_marks"]


def _load_artifacts():
    """Load model and scaler from disk (cached per-process)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Run main.py first to train the model."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def score_to_grade(score: float) -> str:
    """Map a numeric score (0-100) to a letter grade."""
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def predict(
    attendance_pct:   float,
    study_hours:      float,
    assignment_score: float,
    previous_marks:   float,
) -> dict:
    """
    Predict a student's final grade.

    Parameters
    ----------
    attendance_pct   : float  – attendance percentage (0–100)
    study_hours      : float  – daily study hours (0.5–10)
    assignment_score : float  – assignment score (0–100)
    previous_marks   : float  – previous semester marks (0–100)

    Returns
    -------
    dict with keys: predicted_marks (float), letter_grade (str)
    """
    model, scaler = _load_artifacts()

    X_raw = np.array([[attendance_pct, study_hours, assignment_score, previous_marks]])
    X_scaled = scaler.transform(X_raw)
    marks = float(np.clip(model.predict(X_scaled)[0], 0, 100))
    grade = score_to_grade(marks)

    return {
        "predicted_marks": round(marks, 2),
        "letter_grade"   : grade,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict student grade")
    parser.add_argument("--attendance",   type=float, default=90,  help="Attendance % (0-100)")
    parser.add_argument("--study_hours",  type=float, default=4,   help="Daily study hours")
    parser.add_argument("--assignment",   type=float, default=85,  help="Assignment score (0-100)")
    parser.add_argument("--prev_marks",   type=float, default=80,  help="Previous semester marks")
    args = parser.parse_args()

    result = predict(
        attendance_pct   = args.attendance,
        study_hours      = args.study_hours,
        assignment_score = args.assignment,
        previous_marks   = args.prev_marks,
    )

    print("\n" + "─" * 40)
    print("  PREDICTION RESULT")
    print("─" * 40)
    print(f"  Predicted Marks : {result['predicted_marks']}")
    print(f"  Letter Grade    : {result['letter_grade']}")
    print("─" * 40 + "\n")
