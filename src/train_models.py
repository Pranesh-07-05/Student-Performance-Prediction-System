"""
train_models.py
───────────────
Trains four scikit-learn regressors on the student dataset,
evaluates each, selects the best, and saves it with Joblib.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ALL_MODELS_PATH = os.path.join(MODEL_DIR, "all_models.pkl")


def build_models() -> dict:
    """Return a dict of model name → untrained estimator."""
    return {
        "Linear Regression"        : LinearRegression(),
        "Decision Tree"            : DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest"            : RandomForestRegressor(
                                         n_estimators=200,
                                         max_depth=None,
                                         random_state=42,
                                         n_jobs=-1
                                     ),
        "Gradient Boosting"        : GradientBoostingRegressor(
                                         n_estimators=200,
                                         learning_rate=0.1,
                                         max_depth=5,
                                         random_state=42
                                     ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple[dict, str, object]:
    """
    Train all models, evaluate on the test set, save the best.

    Returns
    -------
    results  : dict  – {model_name: {MAE, MSE, RMSE, R2, model}}
    best_name: str   – name of the best model (highest R²)
    best_model       – fitted estimator of the best model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    models  = build_models()
    results = {}

    print("\n" + "=" * 58)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 58)

    for name, model in models.items():
        print(f"\n▶  Training: {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results[name] = {
            "MAE"  : round(mae,  4),
            "MSE"  : round(mse,  4),
            "RMSE" : round(rmse, 4),
            "R2"   : round(r2,   4),
            "model": model,
            "y_pred": y_pred,
        }
        print(f"   MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")

    # ── pick best by R² ───────────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]

    print(f"\n✅  Best Model: {best_name}  (R² = {results[best_name]['R2']:.4f})")
    print("=" * 58)

    # ── persist ───────────────────────────────────────────────────────────────
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(results,    ALL_MODELS_PATH)
    print(f"[train_models] Best model saved → {BEST_MODEL_PATH}")

    return results, best_name, best_model


if __name__ == "__main__":
    # Quick smoke-test with random data
    from preprocessing import load_and_preprocess
    X_train, X_test, y_train, y_test, *_ = load_and_preprocess()
    train_and_evaluate(X_train, X_test, y_train, y_test)
