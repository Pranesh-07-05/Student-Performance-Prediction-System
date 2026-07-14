"""
evaluate.py
───────────
Produces a clean comparison DataFrame of all model metrics
and prints a formatted report to stdout.
"""

import pandas as pd


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Convert results dict (from train_and_evaluate) into a tidy DataFrame.

    Parameters
    ----------
    results : dict – {model_name: {MAE, MSE, RMSE, R2, model, y_pred}}

    Returns
    -------
    pd.DataFrame with columns: Model, MAE, MSE, RMSE, R2
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "MAE"  : metrics["MAE"],
            "MSE"  : metrics["MSE"],
            "RMSE" : metrics["RMSE"],
            "R²"   : metrics["R2"],
        })

    df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    df.index += 1          # 1-based rank
    df.index.name = "Rank"
    return df


def print_report(comparison_df: pd.DataFrame, best_name: str) -> None:
    """Pretty-print the model comparison table."""
    print("\n" + "=" * 62)
    print("  MODEL COMPARISON REPORT")
    print("=" * 62)
    print(comparison_df.to_string())
    print("=" * 62)
    print(f"  🏆  Best Model → {best_name}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from preprocessing import load_and_preprocess
    from train_models  import train_and_evaluate

    X_train, X_test, y_train, y_test, *_ = load_and_preprocess()
    results, best_name, _ = train_and_evaluate(X_train, X_test, y_train, y_test)

    df = build_comparison_table(results)
    print_report(df, best_name)
