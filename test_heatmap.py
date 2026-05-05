import sys
sys.path.append('.')
from scripts.generate_clustering_report import generate_reports
import pandas as pd
from pathlib import Path
import json

res_df = pd.read_csv("results/step3/20260505_100212/esc50/clustering_results.csv")
with open("results/step0/simulated/esc50_inventory.json", "r") as f:
    rare_classes = [k for k, v in json.load(f)["per_class_clip_counts"].items() if v < 10]

data_dfs = {}
for rep in ["physics", "yamnet", "panns", "birdnet"]:
    path = Path("data/processed/esc50/embeddings_" + rep + ".parquet")
    if rep == "physics": path = Path("data/processed/esc50/physics_features.parquet")
    if path.exists():
        df = pd.read_parquet(path)
        manifest = pd.read_csv("data/processed/esc50/dataset_manifest.csv")
        data_dfs[rep] = df.merge(manifest[['clip_id', 'split']], on='clip_id', how='left')

generate_reports(res_df, "esc50", Path("results/step3/20260505_100212/esc50"), rare_classes, data_dfs, Path("results/step3/20260505_100212/esc50/assignments"))
