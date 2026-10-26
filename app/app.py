from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from pathlib import Path
import os
import pandas as pd

from app.utils import (
    load_dataset,
    summarize_dataset,
    train_model,
    make_plot_bundle,
    predict_popularity,
    FEATURES,
    DEFAULT_DATA_PATH,
    DATASETS_CSV,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "spotify-popularity-demo-secret")

def _get_dataset_path():
    csv_path = request.args.get("csv", "").strip()
    if csv_path:
        candidate = Path(csv_path)
        if candidate.exists():
            return candidate
    upload_name = request.args.get("upload", "").strip()
    if upload_name:
        candidate = BASE_DIR / "data" / "uploads" / upload_name
        if candidate.exists():
            return candidate
    return DEFAULT_DATA_PATH

@app.route("/")
def index():
    data_path = _get_dataset_path()
    df = load_dataset(data_path)
    summary = summarize_dataset(df)

    model_result = train_model(df, target="popularity", features=FEATURES, artifact_dir=ARTIFACTS_DIR)
    plots = make_plot_bundle(df, model_result, artifact_dir=ARTIFACTS_DIR)

    datasets = pd.read_csv(DATASETS_CSV).to_dict(orient="records")

    return render_template(
        "index.html",
        summary=summary,
        plots=plots,
        model=model_result,
        features=FEATURES,
        datasets=datasets,
        data_source=str(data_path.name),
    )

@app.route("/datasets")
def datasets():
    df = pd.read_csv(DATASETS_CSV)
    return render_template("datasets.html", datasets=df.to_dict(orient="records"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    upload_dir = BASE_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename:
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("upload"))
        save_path = upload_dir / file.filename
        file.save(save_path)
        flash("Dataset uploaded successfully.", "success")
        return redirect(url_for("index", upload=file.filename))

    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = {}
    for feature in FEATURES:
        try:
            payload[feature] = float(request.form.get(feature, 0))
        except ValueError:
            payload[feature] = 0.0

    prediction = predict_popularity(payload)
    return render_template("predict.html", prediction=prediction, values=payload)

@app.route("/api/summary")
def api_summary():
    data_path = _get_dataset_path()
    df = load_dataset(data_path)
    summary = summarize_dataset(df)
    return jsonify(summary)

@app.route("/api/datasets")
def api_datasets():
    df = pd.read_csv(DATASETS_CSV)
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
