import os
import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import yaml
import tracemalloc
from datetime import datetime
import hashlib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from src.config import load_config
from src.utils import set_seeds
from src.clustering.wrappers import get_algorithm
from src.clustering.metrics import evaluate_clustering

warnings.filterwarnings("ignore")

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]

def normalize_representation(repr_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    dim_cols = [c for c in train_df.columns if "dim_" in c or "spectral_" in c or "power" in c or "hz" in c or "time" in c or "rate" in c or "count" in c or "snr" in c or "fraction" in c or "rms" in c]
    dim_cols = [c for c in dim_cols if c not in ["clip_id", "label", "split", "source_file"]]

    assert "label" not in dim_cols, "Label column leaked into feature matrix!"

    X_train = train_df[dim_cols].values
    X_test = test_df[dim_cols].values

    if repr_name == "physics":
        scaler = StandardScaler()
        if len(X_train) > 0:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_test = scaler.fit_transform(X_test)
    else:
        X_train = normalize(X_train, norm='l2', axis=1)
        X_test = normalize(X_test, norm='l2', axis=1)

    return X_train, X_test

def run_single_combination(algo_name, repr_name, params, train_df, test_df, rare_classes, memory_limit_mb, assignments_dir):
    set_seeds(42)

    X_train, X_test = normalize_representation(repr_name, train_df, test_df)
    y_test = test_df['label'].values

    algo = get_algorithm(algo_name)

    tracemalloc.start()
    start_time = time.time()

    try:
        y_pred = algo.fit_predict(X_train, X_test, params)
        success = True
        error_msg = ""
    except Exception as e:
        y_pred = np.full(len(y_test), -1)
        success = False
        error_msg = str(e)

    runtime = time.time() - start_time
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_memory_mb = peak_memory_bytes / 1048576.0

    if peak_memory_mb > memory_limit_mb:
        success = False
        error_msg = f"Memory limit exceeded: {peak_memory_mb:.2f} MB > {memory_limit_mb} MB"
        y_pred = np.full(len(y_test), -1)

    metrics = evaluate_clustering(y_test, y_pred, rare_classes)

    result_row = {
        "algorithm": algo_name,
        "representation": repr_name,
        "hyperparameters": json.dumps(params, sort_keys=True),
        "success": success,
        "error_msg": error_msg,
        "runtime_s": runtime,
        "peak_memory_mb": peak_memory_mb,
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "weighted_purity": metrics["weighted_purity"],
        "n_clusters": metrics["n_clusters"],
        "largest_cluster_fraction": metrics["largest_cluster_fraction"]
    }

    for rc in rare_classes:
        rc_metrics = metrics["rare_metrics"]["rare_class_metrics"].get(rc, {})
        result_row[f"rc_{rc}_survival"] = rc_metrics.get("survival_flag", False)
        result_row[f"rc_{rc}_purity"] = rc_metrics.get("purity", 0.0)
        result_row[f"rc_{rc}_noise_rate"] = rc_metrics.get("noise_rate", 1.0)
        result_row[f"rc_{rc}_recall"] = metrics["rare_metrics"]["per_class_recall"].get(rc, 0.0)

    hp_id = stable_hash(params)
    assignment_df = pd.DataFrame({
        "clip_id": test_df["clip_id"],
        "cluster_id": y_pred
    })
    assignment_df.to_parquet(assignments_dir / f"{algo_name}_{repr_name}_{hp_id}.parquet", index=False)

    return result_row

import itertools

def create_grid(params_dict):
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def process_dataset(dataset_name: str, base_config, n_jobs=-1):
    print(f"\nProcessing dataset: {dataset_name}")

    with open("configs/step3_clustering.yaml", "r") as f:
        step3_config = yaml.safe_load(f)

    memory_limit_mb = step3_config.get("memory_limit_mb", 2000)

    processed_dir = Path(base_config.paths.processed_dir) / dataset_name
    manifest_path = processed_dir / "dataset_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}, skipping.")
        return

    manifest = pd.read_csv(manifest_path)

    inv_path = Path(base_config.paths.results_dir) / "step0" / "simulated" / f"{dataset_name}_inventory.json"
    rare_classes = []
    if inv_path.exists():
        with open(inv_path, "r") as f:
            inv = json.load(f)
            counts = inv.get("per_class_clip_counts", {})
            total = sum(counts.values())
            rare_classes = [k for k, v in counts.items() if v < 10 or v / total < 0.01]

    if not rare_classes:
        print("Warning: No rare classes found in inventory. Falling back to global metric computation.")

    representations = ["physics", "yamnet", "panns", "birdnet"]
    algorithms = ["stream_kmeans", "denstream", "hdbscan", "umap_hdbscan"]

    data_dfs = {}
    for rep in representations:
        if rep == "physics":
            path = processed_dir / "physics_features.parquet"
        else:
            path = processed_dir / f"embeddings_{rep}.parquet"

        if not path.exists():
            print(f"Missing {path}, skipping representation {rep}")
            continue

        df = pd.read_parquet(path)
        df = df.merge(manifest[['clip_id', 'split']], on='clip_id', how='left')
        data_dfs[rep] = df

    if not data_dfs:
        print("No representations loaded. Exiting.")
        return

    runs = []
    for algo in algorithms:
        if algo not in step3_config["algorithms"]: continue

        grid = create_grid(step3_config["algorithms"][algo])
        for rep in data_dfs.keys():
            if algo == "hdbscan" and rep == "physics":
                pass

            for params in grid:
                runs.append((algo, rep, params))

    print(f"Total combinations to run: {len(runs)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_config.paths.results_dir) / "step3" / timestamp / dataset_name
    assignments_dir = out_dir / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    def process_run(run_tuple):
        algo, rep, params = run_tuple
        df = data_dfs[rep].copy()

        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy()

        if len(test_df) == 0:
            test_df = df.copy()
            train_df = df.copy()

        try:
            return run_single_combination(algo, rep, params, train_df, test_df, rare_classes, memory_limit_mb, assignments_dir)
        except Exception as e:
            print(f"Failed {algo} {rep}: {e}")
            return None

    print(f"Running combinations in parallel with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_run)(run_tuple) for run_tuple in tqdm(runs)
    )

    results = [r for r in results if r is not None]

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "clustering_results.csv", index=False)

    print(f"Results saved to {out_dir}")

    from scripts.generate_clustering_report import generate_reports
    generate_reports(res_df, dataset_name, out_dir, rare_classes, data_dfs, assignments_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    config = load_config()
    set_seeds(config.seed)

    datasets = ["esc50", "dcase2024_t5"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        process_dataset(ds, config, args.n_jobs)

if __name__ == "__main__":
    main()
