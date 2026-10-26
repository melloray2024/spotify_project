"""
Optional helper: download Kaggle Spotify datasets after configuring Kaggle API credentials.

Usage:
    python scripts/download_kaggle_datasets.py --all
"""
from pathlib import Path
import argparse
import pandas as pd

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception:
    KaggleApi = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST = DATA_DIR / "kaggle_spotify_datasets.csv"

def main(download_all: bool = False):
    if KaggleApi is None:
        raise SystemExit("kaggle package is not installed. Install it and configure credentials first.")
    df = pd.read_csv(MANIFEST)
    api = KaggleApi()
    api.authenticate()
    target_dir = DATA_DIR / "kaggle_downloads"
    target_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        if not download_all:
            continue
        slug = row["url"].split("/datasets/")[-1] if "/datasets/" in row["url"] else row["url"].split("/competitions/")[-1]
        print(f"Downloading {row['title']} ...")
        # Kaggle API handles dataset or competition slugs differently
        if "/competitions/" in row["url"]:
            api.competition_download_files(slug, path=str(target_dir), quiet=False)
        else:
            api.dataset_download_files(slug, path=str(target_dir), quiet=False, unzip=True)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download all manifest entries")
    args = parser.parse_args()
    main(args.all)
