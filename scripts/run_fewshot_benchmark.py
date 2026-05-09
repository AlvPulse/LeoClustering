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
import pickle
import hashlib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from src.config import load_config
from src.utils import set_seeds
from src.fewshot.methods import FewShotFitter
from src.fewshot.metrics import evaluate_fewshot_metrics

warnings.filterwarnings("ignore")

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]

def normalize_representation(repr_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    dim_cols = [c for c in train_df.columns if "dim_" in c or "spectral_" in c or "power" in c or "hz" in c or "time" in c or "rate" in c or "count" in c or "snr" in c or "fraction" in c or "rms" in c]
    dim_cols = [c for c in dim_cols if c not in ["clip_id", "label", "split", "source_file"]]

    assert "label" not in dim_cols, "Label column leaked into feature matrix!"

    X_train = train_df[dim_cols].values
    X_test = test_df[dim_cols].values

    # We must handle NaNs that might exist from extraction failures or edge cases
    # Simple imputation: replace NaN with 0 before scaling/normalizing
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

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
        if len(X_test) > 0:
            X_test = normalize(X_test, norm='l2', axis=1)

    return X_train, X_test, train_df, test_df

def run_single_combination(target_class, n, method_name, repr_name, rep_idx, train_df, test_df, method_config, scorers_dir, scores_dir):
    set_seeds(42 + rep_idx)

    X_train, X_test, _, _ = normalize_representation(repr_name, train_df, test_df)

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    pos_train_mask = y_train == target_class
    pos_train_X = X_train[pos_train_mask]

    if len(pos_train_X) < n:
        return {"error": "N/A — insufficient labels"}

    idx = np.random.choice(len(pos_train_X), n, replace=False)
    exemplars = pos_train_X[idx]

    fitter = FewShotFitter(method_name, method_config)

    try:
        scorer = fitter.fit(exemplars, X_train, y_train, target_class)
    except Exception as e:
        return {"error": f"Fit failed: {str(e)}"}

    y_test_binary = (y_test == target_class).astype(int)

    if len(np.unique(y_test_binary)) < 2:
        return {"error": "Test set lacks positive or negative samples for this class"}

    try:
        scores = scorer.score(X_test)
    except Exception as e:
        return {"error": f"Scoring failed: {str(e)}"}

    try:
        metrics = evaluate_fewshot_metrics(y_test_binary, scores)
    except Exception as e:
        return {"error": f"Metrics failed: {str(e)}"}

    scorer_path = scorers_dir / f"{target_class}_{n}_{method_name}_{repr_name}_{rep_idx}.pkl"
    with open(scorer_path, "wb") as f:
        pickle.dump(scorer, f)

    score_df = pd.DataFrame({
        "clip_id": test_df["clip_id"],
        "score": scores,
        "label_is_target": y_test_binary
    })
    score_path = scores_dir / f"{target_class}_{n}_{method_name}_{repr_name}_{rep_idx}.parquet"
    score_df.to_parquet(score_path, index=False)

    return {
        "class": target_class,
        "n_exemplars": n,
        "method": method_name,
        "representation": repr_name,
        "repetition": rep_idx,
        "ap": metrics["average_precision"],
        "p_at_50": metrics["precision_at_recall_0.5"],
        "p_at_80": metrics["precision_at_recall_0.8"],
        "p_at_95": metrics["precision_at_recall_0.95"],
        "r_at_fpr_hour": metrics["recall_at_fpr_1_hour"] if metrics["recall_at_fpr_1_hour"] is not None else -1.0,
        "r_at_fpr_day": metrics["recall_at_fpr_1_day"] if metrics["recall_at_fpr_1_day"] is not None else -1.0,
        "negative_count_actual": getattr(scorer, "actual_negatives", -1)
    }

def process_dataset(dataset_name: str, base_config, n_jobs=-1):
    print(f"\nProcessing dataset: {dataset_name}")

    with open("configs/step4_fewshot.yaml", "r") as f:
        step4_config = yaml.safe_load(f)

    n_exemplars_grid = step4_config["fewshot"]["n_exemplars"]
    repetitions = step4_config["fewshot"]["repetitions"]
    methods = step4_config["fewshot"]["methods"]

    processed_dir = Path(base_config.paths.processed_dir) / dataset_name
    manifest_path = processed_dir / "dataset_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}, skipping.")
        return

    manifest = pd.read_csv(manifest_path)
    all_classes = manifest['label'].unique().tolist()

    representations = ["physics", "yamnet", "panns", "birdnet"]

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
        # Deduplicate before merge just in case
        df = df.drop_duplicates(subset=['clip_id'])
        manifest_sub = manifest[['clip_id', 'split', 'source_file']].drop_duplicates(subset=['clip_id'])
        df = df.merge(manifest_sub, on='clip_id', how='inner')
        data_dfs[rep] = df

    if not data_dfs:
        print("No representations loaded. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_config.paths.results_dir) / "step4" / timestamp / dataset_name
    scorers_dir = out_dir / "scorers"
    scores_dir = out_dir / "scores"
    scorers_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for cls in all_classes:
        for n in n_exemplars_grid:
            for method_name, method_config in methods.items():
                for rep in data_dfs.keys():
                    for rep_idx in range(repetitions):
                        runs.append((cls, n, method_name, rep, rep_idx, method_config))

    print(f"Total combinations to run: {len(runs)}")

    def process_run(run_tuple):
        cls, n, method_name, rep, rep_idx, method_config = run_tuple
        df = data_dfs[rep].copy()

        train_df = df[df['split'] == 'train'].copy()

        test_df = df[df['split'].isin(['test', 'val'])].copy()
        test_df = test_df.drop_duplicates(subset=['clip_id'])

        if len(test_df) == 0:
            test_df = df.copy()
            train_df = df.copy()

        try:
            return run_single_combination(cls, n, method_name, rep, rep_idx, train_df, test_df, method_config, scorers_dir, scores_dir)
        except Exception as e:
            return {"error": f"Fatal error: {str(e)}"}

    print(f"Running combinations in parallel with {n_jobs} workers...")
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(process_run)(run_tuple) for run_tuple in tqdm(runs)
    )

    flat_results = []
    for res, run_tuple in zip(raw_results, runs):
        cls, n, method_name, rep, rep_idx, _ = run_tuple
        if "error" in res:
            flat_results.append({
                "dataset": dataset_name, "class": cls, "n_exemplars": n,
                "method": method_name, "representation": rep, "repetition": rep_idx,
                "metric": "error", "value": res["error"]
            })
        else:
            for metric in ["ap", "p_at_50", "p_at_80", "p_at_95", "r_at_fpr_hour", "r_at_fpr_day"]:
                flat_results.append({
                    "dataset": dataset_name, "class": cls, "n_exemplars": n,
                    "method": method_name, "representation": rep, "repetition": rep_idx,
                    "metric": metric, "value": res[metric]
                })

    res_df = pd.DataFrame(flat_results)
    res_df.to_csv(out_dir / "fewshot_results.csv", index=False)

    print(f"Results saved to {out_dir}")

    from scripts.generate_fewshot_report import generate_reports
    generate_reports(res_df, dataset_name, out_dir, base_config)

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
