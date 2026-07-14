"""
data_generator.py
─────────────────
Generates a realistic synthetic student performance dataset (500 rows).
Features are correlated so the ML models learn meaningful patterns.
"""

import os
import numpy as np
import pandas as pd

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N_STUDENTS = 500


def generate_dataset(n: int = N_STUDENTS, save_path: str = "data/student_data.csv") -> pd.DataFrame:
    """
    Generate a synthetic student dataset and save it as CSV.

    Parameters
    ----------
    n         : number of student records to generate
    save_path : relative or absolute path for the CSV output

    Returns
    -------
    pd.DataFrame
    """
    # ── primary features ─────────────────────────────────────────────────────
    attendance_pct   = np.clip(np.random.normal(loc=75, scale=15, size=n), 40, 100)
    study_hours      = np.clip(np.random.normal(loc=4.5, scale=2.0, size=n), 0.5, 10)
    assignment_score = np.clip(np.random.normal(loc=70, scale=18, size=n), 0, 100)
    previous_marks   = np.clip(np.random.normal(loc=68, scale=16, size=n), 20, 100)

    # ── realistic target: weighted combination + noise ────────────────────────
    noise = np.random.normal(loc=0, scale=4, size=n)
    final_grade = (
        0.30 * previous_marks
        + 0.25 * attendance_pct
        + 0.25 * assignment_score
        + 0.20 * (study_hours * 10)   # scale hours to 0-100
        + noise
    )
    final_grade = np.clip(final_grade, 0, 100).round(2)

    # ── letter-grade mapping ──────────────────────────────────────────────────
    def to_letter(score: float) -> str:
        if score >= 85:  return "A"
        if score >= 70:  return "B"
        if score >= 55:  return "C"
        if score >= 40:  return "D"
        return "F"

    letter_grade = [to_letter(s) for s in final_grade]

    # ── assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "student_id"      : [f"STU{i+1:04d}" for i in range(n)],
        "attendance_pct"  : attendance_pct.round(2),
        "study_hours"     : study_hours.round(2),
        "assignment_score": assignment_score.round(2),
        "previous_marks"  : previous_marks.round(2),
        "final_grade"     : final_grade,
        "letter_grade"    : letter_grade,
    })

    # ── introduce ~3 % missing values for realism ─────────────────────────────
    feature_cols = ["attendance_pct", "study_hours", "assignment_score", "previous_marks"]
    for col in feature_cols:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_generator] Dataset saved → {save_path}  ({n} rows)")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"\nGrade distribution:\n{df['letter_grade'].value_counts()}")
