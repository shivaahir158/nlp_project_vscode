from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


# =========================
# FEATURE GROUPS
# =========================
TRADITIONAL_COLS = [
    "ttr", "mattr", "avg_word_len", "total_words",
    "avg_sent_len", "std_sent_len", "avg_dep_depth", "subord_ratio",
    "noun_ratio", "verb_ratio", "adj_ratio",
    "filler_rate", "repetition_rate", "cohort_rate",
    "first_person_rate", "pronoun_types",
    "flesch_reading_ease", "fk_grade", "gunning_fog",
]

DLATK_COLS = [
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "nrc_anger", "nrc_fear", "nrc_joy", "nrc_sadness",
    "nrc_trust", "nrc_anticipation",
    "first_person_word_rate", "function_word_rate",
]


# =========================
# MODELS
# =========================
def get_models():
    return {
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


# =========================
# CORE EVALUATION
# =========================
def evaluate_feature_set(df, feature_cols, setup_name, output_dir):

    # keep only valid numeric columns
    feature_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_m = df[feature_cols + ["age"]].dropna()

    X = df_m[feature_cols].values
    y = df_m["age"].values

    print("\n" + "=" * 60)
    print(f"ABLATION: {setup_name}")
    print("=" * 60)
    print(f"Training on {len(df_m)} samples")
    print(f"Number of features: {len(feature_cols)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()

    best_model = None
    best_name = None
    best_mae = float("inf")

    for name, model in models.items():
        mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")

        print(f"{name:18s} MAE: {mae.mean():.3f} | R2: {r2.mean():.3f}")

        if mae.mean() < best_mae:
            best_mae = mae.mean()
            best_model = model
            best_name = name

    # predictions
    y_pred = cross_val_predict(best_model, X, y, cv=cv)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")

    if isinstance(best_model, RandomForestRegressor):

        print("\nExtracting feature importance...")

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        rf.fit(X, y)

        importances = pd.Series(rf.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)

        print("\nTop 15 Important Features:")
        print(importances.head(15))

        importances.to_csv(output_dir / f"{safe_name}_feature_importance.csv")

        plt.figure(figsize=(8, 6))
        importances.head(15).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(f"Top Features - {setup_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_name}_feature_importance.png", dpi=150)
        plt.close()

    # =========================
    # ERROR ANALYSIS
    # =========================
    print("\nPerforming error analysis...")

    df_error = df_m.copy()
    df_error["pred_age"] = y_pred
    df_error["error"] = y_pred - y
    df_error["abs_error"] = np.abs(df_error["error"])

    print("\nTop 10 worst predictions:")
    print(
        df_error[["age", "pred_age", "abs_error"]]
        .sort_values("abs_error", ascending=False)
        .head(10)
    )

    error_corr = df_error[feature_cols].corrwith(df_error["abs_error"])
    error_corr = error_corr.dropna().sort_values(ascending=False)

    print("\nTop error-correlated features:")
    print(error_corr.head(10))

    error_corr.to_csv(output_dir / f"{safe_name}_error_correlation.csv")

    plt.figure(figsize=(8, 6))
    error_corr.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title(f"Error Analysis - {setup_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name}_error_analysis.png", dpi=150)
    plt.close()

    # =========================
    # FINAL METRICS
    # =========================
    final_mae = mean_absolute_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)
    within_5 = (np.abs(y_pred - y) <= 5).mean() * 100

    print(f"\nBest model: {best_name}")
    print(f"Final MAE: {final_mae:.3f}")
    print(f"Final R2: {final_r2:.3f}")
    print(f"Within 5 years: {within_5:.1f}%")

    # prediction plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(f"{setup_name} ({best_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name}_pred.png", dpi=150)
    plt.close()

    return {
        "Setup": setup_name,
        "Best_Model": best_name,
        "Samples": len(df_m),
        "Features": len(feature_cols),
        "MAE": final_mae,
        "R2": final_r2,
        "Within_5_Years": within_5,
    }


# =========================
# MAIN DRIVER
# =========================
def train_and_evaluate(df, output_dir="outputs"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bert_cols = [c for c in df.columns if c.startswith("bert_pc")]

    experiments = {
        "Traditional Only": TRADITIONAL_COLS,
        "Traditional + BERT": TRADITIONAL_COLS + bert_cols,
        "Traditional + BERT + DLATK": TRADITIONAL_COLS + bert_cols + DLATK_COLS,
    }

    summaries = []

    for name, cols in experiments.items():
        result = evaluate_feature_set(df, cols, name, output_dir)
        summaries.append(result)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(summary_df)

    return summary_df