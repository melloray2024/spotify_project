from __future__ import annotations

from pathlib import Path
import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_PATH = DATA_DIR / "demo_spotify_tracks.csv"
DATASETS_CSV = DATA_DIR / "kaggle_spotify_datasets.csv"

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]
TARGET = "popularity"


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.read_csv(DEFAULT_DATA_PATH)
    df = pd.read_csv(path)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        summary["target_mean"] = round(float(df[TARGET].mean()), 2)
        summary["target_min"] = round(float(df[TARGET].min()), 2)
        summary["target_max"] = round(float(df[TARGET].max()), 2)
    else:
        summary["target_mean"] = None
        summary["target_min"] = None
        summary["target_max"] = None
    return summary


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature in FEATURES + [TARGET]:
        if feature not in df.columns:
            if feature == TARGET:
                df[feature] = np.nan
            else:
                df[feature] = np.nan
    return df


def train_model(df: pd.DataFrame, target: str, features: list[str], artifact_dir: Path | str) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_columns(df)

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    numeric = df[features + [target]].copy()

    X = numeric[features]
    y = numeric[target]

    # Drop rows without target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    if len(df) < 20 or len(X) < 20:
        return {
            "status": "insufficient_data",
            "message": "Not enough rows to train a reliable model.",
            "mae": None,
            "rmse": None,
            "r2": None,
            "baseline_mae": None,
            "baseline_rmse": None,
            "model_path": None,
            "feature_importance": [],
            "predictions": [],
            "actuals": [],
            "test_size": 0,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_pred = np.repeat(float(y_train.mean()), len(y_test))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = math.sqrt(mean_squared_error(y_test, baseline_pred))

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    model_path = artifact_dir / "linear_regression_model.joblib"
    joblib.dump(model, model_path)

    # coefficients after scaling
    coefs = model.named_steps["regressor"].coef_
    feature_importance = sorted(
        [{"feature": f, "coefficient": round(float(c), 4)} for f, c in zip(features, coefs)],
        key=lambda x: abs(x["coefficient"]),
        reverse=True,
    )

    return {
        "status": "trained",
        "message": "Model trained successfully.",
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "baseline_mae": round(float(baseline_mae), 3),
        "baseline_rmse": round(float(baseline_rmse), 3),
        "model_path": str(model_path),
        "feature_importance": feature_importance,
        "predictions": [round(float(x), 3) for x in pred[:30].tolist()],
        "actuals": [round(float(x), 3) for x in y_test.iloc[:30].tolist()],
        "test_size": int(len(y_test)),
    }


def make_plot_bundle(df: pd.DataFrame, model_result: dict, artifact_dir: Path | str) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = BASE_DIR / "app" / "static" / "generated"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Histogram
    if TARGET in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[TARGET].dropna(), bins=20)
        ax.set_title("Popularity Distribution")
        ax.set_xlabel("Popularity")
        ax.set_ylabel("Count")
        hist_path = plot_dir / "popularity_hist.png"
        fig.tight_layout()
        fig.savefig(hist_path, dpi=160)
        plt.close(fig)
        plots["histogram"] = str(hist_path.relative_to(BASE_DIR / "app" / "static"))

    # Correlation heatmap
    corr_cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    if len(corr_cols) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[corr_cols].corr(numeric_only=True)
        sns.heatmap(corr, annot=False, cmap="viridis", ax=ax)
        ax.set_title("Correlation Heatmap")
        heat_path = plot_dir / "correlation_heatmap.png"
        fig.tight_layout()
        fig.savefig(heat_path, dpi=160)
        plt.close(fig)
        plots["heatmap"] = str(heat_path.relative_to(BASE_DIR / "app" / "static"))

    # Actual vs predicted
    if model_result.get("actuals") and model_result.get("predictions"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(model_result["actuals"], model_result["predictions"])
        ax.plot([0, 100], [0, 100], linestyle="--")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Actual Popularity")
        ax.set_ylabel("Predicted Popularity")
        ax.set_title("Predicted vs Actual")
        scatter_path = plot_dir / "predicted_vs_actual.png"
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=160)
        plt.close(fig)
        plots["scatter"] = str(scatter_path.relative_to(BASE_DIR / "app" / "static"))

    return plots


def predict_popularity(values: dict) -> dict:
    model_path = ARTIFACTS_DIR / "linear_regression_model.joblib"
    if not model_path.exists():
        # Train quickly on default data if model is missing
        df = load_dataset(DEFAULT_DATA_PATH)
        result = train_model(df, TARGET, FEATURES, ARTIFACTS_DIR)
        if result.get("model_path"):
            model_path = Path(result["model_path"])

    model = joblib.load(model_path)
    frame = pd.DataFrame([values])
    prediction = float(model.predict(frame)[0])
    return {
        "prediction": round(max(0, min(100, prediction)), 2)
    }
