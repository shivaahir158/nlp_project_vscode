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


def evaluate_feature_set(df, feature_cols, setup_name, output_dir):
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
            "Setup": setup_name,
            "Samples": len(df_m),
            "Features": len(feature_cols),
            "MAE": mae_scores.mean(),
            "MAE_std": mae_scores.std(),
            "R2": r2_scores.mean(),
            "R2_std": r2_scores.std(),
        }

        print(
            f"{name:18s} MAE: {mae_scores.mean():.3f} | "
            f"R2: {r2_scores.mean():.3f}"
        )

        if mae_scores.mean() < best_mae:
            best_mae = mae_scores.mean()
            best_name = name
            best_model = model

    y_pred = cross_val_predict(best_model, X, y, cv=cv)

    final_mae = mean_absolute_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)
    within_5 = (np.abs(y_pred - y) <= 5).mean() * 100

    print(f"\nBest model: {best_name}")
    print(f"Final MAE: {final_mae:.3f}")
    print(f"Final R2: {final_r2:.3f}")
    print(f"Within 5 years: {within_5:.1f}%")

    safe_name = setup_name.lower().replace(" ", "_").replace("+", "plus")

    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(
        f"{setup_name}\n{best_name}: MAE={final_mae:.3f}, R2={final_r2:.3f}"
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_name}_pred.png", dpi=150)
    plt.close()

    summary = {
        "Setup": setup_name,
        "Best_Model": best_name,
        "Samples": len(df_m),
        "Features": len(feature_cols),
        "Final_MAE": final_mae,
        "Final_R2": final_r2,
        "Within_5_Years": within_5,
    }

    return pd.DataFrame(results).T, summary


def train_and_evaluate(df, output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bert_cols = [c for c in df.columns if c.startswith("bert_pc")]

    traditional = TRADITIONAL_COLS

    traditional_plus_bert = TRADITIONAL_COLS + bert_cols

    traditional_plus_bert_dlatk = TRADITIONAL_COLS + bert_cols + DLATK_COLS

    experiments = {
        "Traditional Only": traditional,
        "Traditional + BERT": traditional_plus_bert,
        "Traditional + BERT + DLATK": traditional_plus_bert_dlatk,
    }

    all_results = []
    summaries = []

    for setup_name, cols in experiments.items():
        result_df, summary = evaluate_feature_set(
            df=df,
            feature_cols=cols,
            setup_name=setup_name,
            output_dir=output_dir,
        )

        all_results.append(result_df)
        summaries.append(summary)

    full_results = pd.concat(all_results)
    summary_df = pd.DataFrame(summaries)

    full_results.to_csv(output_dir / "ablation_all_model_results.csv", index=True)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL ABLATION SUMMARY")
    print("=" * 60)
    print(summary_df)

    return summary_df