from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    KFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Optional advanced models
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


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
    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )

    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.03,
            depth=4,
            loss_function="MAE",
            verbose=0,
            random_seed=42,
        )

    return models


# =========================
# CATBOOST HYPERPARAMETER TUNING
# =========================
def tune_catboost(X, y, cv):
    if CatBoostRegressor is None:
        print("\nCatBoost not installed. Skipping tuned CatBoost.")
        return None

    print("\nTuning CatBoost...")

    param_grid = {
        "iterations": [300, 500],
        "depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "l2_leaf_reg": [1, 3, 5],
    }

    base_model = CatBoostRegressor(
        loss_function="MAE",
        verbose=0,
        random_seed=42,
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("\nBest CatBoost parameters:")
    print(grid.best_params_)
    print(f"Best tuned CatBoost CV MAE: {-grid.best_score_:.3f}")

    return grid.best_estimator_


# =========================
# FEATURE IMPORTANCE
# =========================
def save_feature_importance(best_model, X, y, feature_cols, setup_name, output_dir):
    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")

    model_for_importance = clone(best_model)
    model_for_importance.fit(X, y)

    if isinstance(model_for_importance, Pipeline):
        final_model = model_for_importance.named_steps["model"]
    else:
        final_model = model_for_importance

    if not hasattr(final_model, "feature_importances_"):
        print("\nFeature importance not available for this model.")
        return

    print("\nExtracting feature importance...")

    importances = pd.Series(
        final_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

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
def save_error_analysis(df_m, y, y_pred, feature_cols, setup_name, output_dir):
    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")

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

    return df_error


# =========================
# AGE-GROUP ERROR ANALYSIS
# =========================
def save_age_group_error(df_error, setup_name, output_dir):
    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")

    bins = [0, 60, 70, 80, 90, 120]
    labels = ["<60", "60-69", "70-79", "80-89", "90+"]

    df_error["age_group"] = pd.cut(
        df_error["age"],
        bins=bins,
        labels=labels,
        right=False,
    )

    age_group_summary = (
        df_error.groupby("age_group", observed=True)
        .agg(
            samples=("age", "count"),
            mae=("abs_error", "mean"),
            mean_error=("error", "mean"),
            within_5=("abs_error", lambda x: (x <= 5).mean() * 100),
        )
        .reset_index()
    )

    print("\nAge-group error analysis:")
    print(age_group_summary)

    age_group_summary.to_csv(
        output_dir / f"{safe_name}_age_group_error.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.bar(age_group_summary["age_group"].astype(str), age_group_summary["mae"])
    plt.xlabel("Age Group")
    plt.ylabel("MAE")
    plt.title(f"MAE by Age Group - {setup_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name}_age_group_error.png", dpi=150)
    plt.close()


# =========================
# CORE EVALUATION
# =========================
def evaluate_feature_set(df, feature_cols, setup_name, output_dir):

    feature_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    df_m = df[feature_cols + ["age"]].dropna()

    # Keep X as DataFrame to avoid LightGBM feature-name warnings
    X = df_m[feature_cols]
    y = df_m["age"].values

    print("\n" + "=" * 60)
    print(f"ABLATION: {setup_name}")
    print("=" * 60)
    print(f"Training on {len(df_m)} samples")
    print(f"Number of features: {len(feature_cols)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = get_models()

    tuned_catboost = tune_catboost(X, y, cv)
    if tuned_catboost is not None:
        models["TunedCatBoost"] = tuned_catboost

    best_model = None
    best_name = None
    best_mae = float("inf")

    model_rows = []

    for name, model in models.items():
        mae_scores = -cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_absolute_error",
        )

        r2_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2",
        )

        print(
            f"{name:18s} "
            f"MAE: {mae_scores.mean():.3f} "
            f"| R2: {r2_scores.mean():.3f}"
        )

        model_rows.append({
            "Setup": setup_name,
            "Model": name,
            "MAE_mean": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "R2_mean": r2_scores.mean(),
            "R2_std": r2_scores.std(),
        })

        if mae_scores.mean() < best_mae:
            best_mae = mae_scores.mean()
            best_model = model
            best_name = name

    model_comparison = pd.DataFrame(model_rows)

    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")
    model_comparison.to_csv(
        output_dir / f"{safe_name}_model_comparison.csv",
        index=False,
    )

    y_pred = cross_val_predict(best_model, X, y, cv=cv)

    save_feature_importance(
        best_model,
        X,
        y,
        feature_cols,
        setup_name,
        output_dir,
    )

    df_error = save_error_analysis(
        df_m,
        y,
        y_pred,
        feature_cols,
        setup_name,
        output_dir,
    )

    save_age_group_error(
        df_error,
        setup_name,
        output_dir,
    )

    final_mae = mean_absolute_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)
    within_5 = (np.abs(y_pred - y) <= 5).mean() * 100

    print(f"\nBest model: {best_name}")
    print(f"Final MAE: {final_mae:.3f}")
    print(f"Final R2: {final_r2:.3f}")
    print(f"Within 5 years: {within_5:.1f}%")

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