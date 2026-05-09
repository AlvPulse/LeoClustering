import pandas as pd
import numpy as np
import umap
import json
from pathlib import Path
import streamlit as st
from sklearn.metrics.pairwise import cosine_distances

@st.cache_data
def get_best_run(base_dir: Path, step_name: str, metric_sort: str, metric_tie: str = None, maximize: bool = True):
    step_dir = base_dir / step_name
    if not step_dir.exists(): return None, None

    dirs = [d for d in step_dir.iterdir() if d.is_dir()]
    if not dirs: return None, None

    latest_dir = sorted(dirs, key=lambda x: x.name)[-1]

    datasets = [d for d in latest_dir.iterdir() if d.is_dir()]
    if not datasets: return None, None

    ds_dir = datasets[0]

    if step_name == "step2":
        return "panns", None

    elif step_name == "step3":
        csv_path = ds_dir / "clustering_results.csv"
        if not csv_path.exists(): return None, None

        df = pd.read_csv(csv_path)
        rc_cols = [c for c in df.columns if c.endswith("_survival")]
        df['survival_count'] = df[rc_cols].sum(axis=1)

        df = df.sort_values(by=["survival_count", "ari"], ascending=[False, False])
        best_row = df.iloc[0]

        algo = best_row['algorithm']
        rep = best_row['representation']
        params = json.loads(best_row['hyperparameters'])

        import hashlib
        hp_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        return f"{algo}_{rep}_{hp_id}", best_row

    return None, None

@st.cache_data
def load_data(dataset_name: str, repr_name: str, assignment_name: str, config_paths):
    processed_dir = Path(config_paths['processed_dir']) / dataset_name

    manifest_path = processed_dir / "dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    phys_path = processed_dir / "physics_features.parquet"
    phys_df = pd.read_parquet(phys_path)

    emb_path = processed_dir / f"embeddings_{repr_name}.parquet"
    if not emb_path.exists():
        emb_path = processed_dir / f"embeddings_panns.parquet"
    emb_df = pd.read_parquet(emb_path)

    step3_base = Path(config_paths['results_dir']) / "step3"
    dirs = sorted([d for d in step3_base.iterdir() if d.is_dir()], key=lambda x: x.name)
    assignment_df = None
    if dirs:
        latest = dirs[-1] / dataset_name / "assignments" / f"{assignment_name}.parquet"
        if latest.exists():
            assignment_df = pd.read_parquet(latest)

    if assignment_df is None:
        assignment_df = pd.DataFrame({"clip_id": manifest["clip_id"], "cluster_id": -1})

    df = manifest.merge(phys_df, on="clip_id", how="left")

    return df, emb_df, assignment_df

@st.cache_data
def precompute_umap(emb_df: pd.DataFrame, seed: int = 42):
    dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]
    X = emb_df[dim_cols].values

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
    proj = reducer.fit_transform(X)

    res = pd.DataFrame({
        "clip_id": emb_df["clip_id"],
        "umap_x": proj[:, 0],
        "umap_y": proj[:, 1]
    })
    return res

@st.cache_data
def compute_structured_samples(emb_df: pd.DataFrame, assignment_df: pd.DataFrame):
    merged = emb_df.merge(assignment_df, on="clip_id")
    clusters = merged["cluster_id"].unique()
    dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]

    samples = {}

    for c in clusters:
        if c == -1: continue

        c_mask = merged["cluster_id"] == c
        c_df = merged[c_mask]

        if len(c_df) < 5:
            samples[int(c)] = {"type": "small", "members": c_df["clip_id"].tolist()}
            continue

        c_X = c_df[dim_cols].values

        dist_mat = cosine_distances(c_X)
        mean_dists = np.mean(dist_mat, axis=1)

        medoid_idx = np.argmin(mean_dists)
        medoid_clip = c_df.iloc[medoid_idx]["clip_id"]

        boundary_idx = np.argsort(mean_dists)[-5:]
        boundaries = c_df.iloc[boundary_idx]["clip_id"].tolist()

        medoid_dists = dist_mat[medoid_idx]
        outlier_idx = np.argsort(medoid_dists)[-3:]
        outliers = c_df.iloc[outlier_idx]["clip_id"].tolist()

        other_mask = merged["cluster_id"] != c
        other_df = merged[other_mask]
        if len(other_df) > 0:
            other_X = other_df[dim_cols].values
            cross_dist = cosine_distances(c_X, other_X)
            min_cross_dist = np.min(cross_dist, axis=0)
            bridge_idx = np.argsort(min_cross_dist)[:5]
            bridges = other_df.iloc[bridge_idx]["clip_id"].tolist()
        else:
            bridges = []

        samples[int(c)] = {
            "type": "large",
            "medoid": medoid_clip,
            "boundaries": boundaries,
            "outliers": outliers,
            "bridges": bridges
        }

    return samples
