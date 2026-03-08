"""
model.py
────────
Trains & compares Random Forest, Gradient Boosting (XGBoost-equivalent),
and SVR models. Saves the best model and produces visualisation plots.

Run:  python model.py
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_cleaning import load_and_clean, engineer_score, get_features

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── 1. PREPARE DATA ────────────────────────────────────────────────────────────
def prepare_data(filepath="chum.csv"):
    df = load_and_clean(filepath)
    df = engineer_score(df)
    features = get_features()
    X = df[features]
    y = df["performance_score"]
    return df, X, y, features


# ── 2. BUILD MODELS ────────────────────────────────────────────────────────────
def build_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=10)),
        ]),
    }


# ── 3. EVALUATE ────────────────────────────────────────────────────────────────
def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        results[name] = {
            "model"  : model,
            "y_pred" : y_pred,
            "R2"     : r2_score(y_test, y_pred),
            "MAE"    : mean_absolute_error(y_test, y_pred),
            "RMSE"   : np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV_R2"  : cv.mean(),
            "CV_Std" : cv.std(),
        }
        print(f"  {name:<22} R²={results[name]['R2']:.4f}  "
              f"MAE={results[name]['MAE']:.2f}  "
              f"CV_R²={results[name]['CV_R2']:.4f} ± {results[name]['CV_Std']:.4f}")
    return results


# ── 4. VISUALISATIONS ──────────────────────────────────────────────────────────
def plot_score_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("IPL Player Performance Score Distribution", fontsize=15, fontweight="bold")

    # Histogram
    axes[0].hist(df["performance_score"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Performance Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Scores")

    # Box-plot by recent years
    recent = df[df["Year"] >= 2018].copy()
    recent["Year"] = recent["Year"].astype(int)
    sns.boxplot(data=recent, x="Year", y="performance_score", ax=axes[1], palette="muted")
    axes[1].set_title("Score by Year (2018–2024)")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Performance Score")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/score_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/score_distribution.png")


def plot_model_comparison(results):
    names  = list(results.keys())
    r2s    = [results[n]["R2"]  for n in names]
    maes   = [results[n]["MAE"] for n in names]
    cv_r2s = [results[n]["CV_R2"] for n in names]
    cv_std = [results[n]["CV_Std"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, vals, title, ylabel in zip(
        axes,
        [r2s, maes, cv_r2s],
        ["Test R² Score", "Mean Absolute Error", "5-Fold CV R²"],
        ["R²", "MAE (points)", "CV R²"],
    ):
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Add error bars on CV chart
    axes[2].errorbar(names, cv_r2s, yerr=cv_std, fmt="none",
                     color="black", capsize=5, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/model_comparison.png")


def plot_actual_vs_predicted(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Actual vs Predicted Performance Score", fontsize=15, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res["y_pred"]
        ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="#4C72B0")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(f"{name}\nR²={res['R2']:.4f}  MAE={res['MAE']:.1f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/actual_vs_predicted.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/actual_vs_predicted.png")


def plot_feature_importance(best_model, features):
    # Works for tree-based models; skip for SVR pipeline
    estimator = (
        best_model.named_steps["svr"]
        if hasattr(best_model, "named_steps") else best_model
    )
    if not hasattr(estimator, "feature_importances_"):
        print("  Feature importance not available for SVR — skipping.")
        return

    importances = pd.Series(best_model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#4C72B0" if v < importances.median() else "#DD8452"
              for v in importances]
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Importances (Best Model)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"  Saved → {PLOT_DIR}/feature_importance.png")


# ── 5. MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  IPL Player Performance Score — Model Training")
    print("=" * 55)

    print("\n[1/5] Loading & cleaning data …")
    df, X, y, features = prepare_data()
    print(f"      {len(df)} records | {len(features)} features")

    print("\n[2/5] Splitting data …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[3/5] Training & evaluating models …")
    models  = build_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Pick best by R²
    best_name  = max(results, key=lambda n: results[n]["R2"])
    best_model = results[best_name]["model"]
    print(f"\n  🏆 Best model: {best_name} (R²={results[best_name]['R2']:.4f})")

    print("\n[4/5] Generating visualisations …")
    plot_score_distribution(df)
    plot_model_comparison(results)
    plot_actual_vs_predicted(results, y_test)
    plot_feature_importance(best_model, features)

    print("\n[5/5] Saving best model …")
    with open("best_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": features, "name": best_name}, f)
    print("  Saved → best_model.pkl")

    print("\n✅ Done! Check the 'plots/' folder for all charts.\n")


if __name__ == "__main__":
    main()