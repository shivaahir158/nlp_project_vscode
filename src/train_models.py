from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


FEAT_COLS = [
    "ttr", "mattr", "avg_word_len", "total_words",
    "avg_sent_len", "std_sent_len", "avg_dep_depth", "subord_ratio",
    "noun_ratio", "verb_ratio", "adj_ratio",
    "filler_rate", "repetition_rate", "cohort_rate",
    "first_person_rate", "pronoun_types",
    "flesch_reading_ease", "fk_grade", "gunning_fog",
]


def train_and_evaluate(df, output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

   # feat_cols = [c for c in FEAT_COLS if c in df.columns]
    feat_cols = [
    c for c in df.columns
    if c not in ["item_id", "birth_year", "recording_year", "age", "branch"]
    and pd.api.types.is_numeric_dtype(df[c])
]

    df_m = df[feat_cols + ["age"]].dropna()

    X = df_m[feat_cols].values
    y = df_m["age"].values

    print(f"Training on {len(df_m)} samples")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = {}

    best_name = None
    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        mae_scores = -cross_val_score(
            model, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )
        r2_scores = cross_val_score(
            model, X, y, cv=cv, scoring="r2"
        )

        results[name] = {
            "MAE": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "R2": r2_scores.mean(),
            "R2_std": r2_scores.std(),
        }

        print(f"{name:18s} MAE: {mae_scores.mean():.3f} | R2: {r2_scores.mean():.3f}")

        if mae_scores.mean() < best_mae:
            best_mae = mae_scores.mean()
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "model_results.csv")

    print(f"\nBest model: {best_name}")

    y_pred = cross_val_predict(best_model, X, y, cv=cv)

    mae_value = mean_absolute_error(y, y_pred)
    r2_value = r2_score(y, y_pred)
    within_5 = (np.abs(y_pred - y) <= 5).mean() * 100

    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(
        f"{best_name} Prediction\nMAE={mae_value:.3f}, R2={r2_value:.3f}, Within 5yr={within_5:.1f}%"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "pred.png", dpi=150)
    plt.close()

    print(f"\nFinal {best_name} MAE: {mae_value:.3f}")
    print(f"Final {best_name} R2: {r2_value:.3f}")
    print(f"Within 5 years: {within_5:.1f}%")

    return results_df