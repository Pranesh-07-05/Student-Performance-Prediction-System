"""
main.py
───────
CLI entry point — runs the complete Student Performance Prediction pipeline:

  1. Generate synthetic dataset (if not already present)
  2. Preprocess data
  3. Train & evaluate all four ML models
  4. Generate visualization charts
  5. Make a sample prediction

Usage
-----
  python main.py                          # full pipeline
  python main.py --no-generate            # skip dataset generation (use existing CSV)
  python main.py --predict-only           # skip training, predict with saved model
"""

import os
import sys
import argparse
import time

# ── ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_generator import generate_dataset
from preprocessing  import load_and_preprocess
from train_models   import train_and_evaluate
from evaluate       import build_comparison_table, print_report
from visualize      import generate_all_charts
from predict        import predict


def banner():
    print("\n" + "═" * 62)
    print("  🎓  STUDENT PERFORMANCE PREDICTION — ML PIPELINE")
    print("═" * 62)


def run_pipeline(args):
    banner()
    t0 = time.time()

    # ── Step 1: Generate dataset ───────────────────────────────────────────────
    csv_path = os.path.join("data", "student_data.csv")
    if args.generate and not os.path.exists(csv_path):
        print("\n[Step 1/5] Generating synthetic dataset …")
        generate_dataset(save_path=csv_path)
    elif os.path.exists(csv_path):
        print(f"\n[Step 1/5] Using existing dataset → {csv_path}")
    else:
        print("\n[Step 1/5] Generating synthetic dataset …")
        generate_dataset(save_path=csv_path)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n[Step 2/5] Preprocessing data …")
    X_train, X_test, y_train, y_test, feature_names, scaler, df_clean = \
        load_and_preprocess(csv_path=csv_path)

    # ── Step 3: Train & Evaluate ──────────────────────────────────────────────
    print("\n[Step 3/5] Training ML models …")
    results, best_name, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    comparison_df = build_comparison_table(results)
    print_report(comparison_df, best_name)

    # ── Step 4: Visualize ─────────────────────────────────────────────────────
    print("\n[Step 4/5] Generating visualization charts …")
    chart_paths = generate_all_charts(
        df_clean, results, feature_names, y_test, best_name, comparison_df
    )
    for label, path in chart_paths.items():
        print(f"  • {label:<22} → {path}")

    # ── Step 5: Sample Prediction ─────────────────────────────────────────────
    print("\n[Step 5/5] Sample prediction …")
    sample = dict(
        attendance_pct   = 90,
        study_hours      = 4,
        assignment_score = 85,
        previous_marks   = 80,
    )
    result = predict(**sample)

    print("\n┌─────────────────────────────────────────┐")
    print("│           SAMPLE PREDICTION RESULT      │")
    print("├─────────────────────────────────────────┤")
    for k, v in sample.items():
        print(f"│  {k:<25} : {v:<12}│")
    print("├─────────────────────────────────────────┤")
    print(f"│  Predicted Marks  : {result['predicted_marks']:<21}│")
    print(f"│  Letter Grade     : {result['letter_grade']:<21}│")
    print("└─────────────────────────────────────────┘")

    elapsed = time.time() - t0
    print(f"\n✅  Pipeline complete in {elapsed:.1f}s\n")
    print("  Run the interactive dashboard:")
    print("  ▶  streamlit run app.py\n")


def run_predict_only():
    banner()
    print("\n[predict] Loading saved model …")
    sample = dict(
        attendance_pct   = 90,
        study_hours      = 4,
        assignment_score = 85,
        previous_marks   = 80,
    )
    result = predict(**sample)
    print(f"  Predicted Marks : {result['predicted_marks']}")
    print(f"  Letter Grade    : {result['letter_grade']}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Performance Prediction Pipeline")
    parser.add_argument("--no-generate",  dest="generate",      action="store_false",
                        help="Skip dataset generation (use existing CSV)")
    parser.add_argument("--predict-only", dest="predict_only",   action="store_true",
                        help="Skip training; use saved model to predict")
    parser.set_defaults(generate=True, predict_only=False)
    args = parser.parse_args()

    if args.predict_only:
        run_predict_only()
    else:
        run_pipeline(args)
