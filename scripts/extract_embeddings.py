import os
import argparse
import time
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings

from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config import load_config
from src.utils import set_seeds
from src.embeddings import get_extractor

warnings.filterwarnings("ignore")

def auc_overlap(emb_matrix, labels):
    n = len(labels)
    if n < 2: return np.array([]), np.array([])

    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        emb_matrix = emb_matrix[idx]
        labels = labels[idx]
        n = 5000

    dist_matrix = cosine_distances(emb_matrix)
    np.fill_diagonal(dist_matrix, np.nan)

    label_mat = np.array(labels)
    same_class_mask = (label_mat[:, None] == label_mat[None, :])
    np.fill_diagonal(same_class_mask, False)

    diff_class_mask = (label_mat[:, None] != label_mat[None, :])

    within_dists = dist_matrix[same_class_mask]
    between_dists = dist_matrix[diff_class_mask]

    within_dists = within_dists[~np.isnan(within_dists)]
    between_dists = between_dists[~np.isnan(between_dists)]

    if len(within_dists) > 10000:
        within_dists = np.random.choice(within_dists, 10000, replace=False)
    if len(between_dists) > 10000:
        between_dists = np.random.choice(between_dists, 10000, replace=False)

    return within_dists, between_dists

def compute_knn_purity(emb_matrix, labels, k=10):
    if len(labels) <= k:
        k = len(labels) - 1

    if k <= 0: return 0.0, {}

    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(emb_matrix)
    _, indices = nn.kneighbors(emb_matrix)

    neighbor_indices = indices[:, 1:]

    purities = []
    per_class_purities = {}

    for i, true_label in enumerate(labels):
        n_labels = [labels[idx] for idx in neighbor_indices[i]]
        matches = sum(1 for l in n_labels if l == true_label)
        purity = matches / k
        purities.append(purity)

        if true_label not in per_class_purities:
            per_class_purities[true_label] = []
        per_class_purities[true_label].append(purity)

    return np.mean(purities), {k: np.mean(v) for k, v in per_class_purities.items()}

def get_stratified_subsample(emb_matrix, labels, max_size=5000):
    n = len(labels)
    if n <= max_size:
        return emb_matrix, labels

    df = pd.DataFrame({"label": labels})
    sampled_idx = df.groupby('label', group_keys=False).apply(lambda x: x.sample(int(np.ceil(max_size * len(x) / n)))).index

    # Shuffle
    sampled_idx = np.random.permutation(sampled_idx)

    # Truncate strictly to max_size
    sampled_idx = sampled_idx[:max_size]

    return emb_matrix[sampled_idx], labels[sampled_idx]

def generate_diagnostics(results, hw_info, results_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "step2" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    report_lines.append("# Deep Embedding Diagnostics\n")
    report_lines.append(f"Hardware used: **{hw_info}**\n")

    dist_plot_data = []

    for res in results:
        df = res["df"]
        ds_name = res["dataset"]
        model_name = res["model"]
        dim = res["dimension"]

        report_lines.append(f"## Model: {model_name.upper()} | Dataset: {ds_name}")
        report_lines.append(f"- **Dimensionality:** {dim}")
        report_lines.append(f"- **Runtime:** {res['total_runtime_seconds']:.2f}s ({res['throughput_clips_per_second']:.2f} clips/s)")
        if res['failures'] > 0:
            report_lines.append(f"- **Failures:** {res['failures']}")

        df = df.dropna()

        labels = df['label'].values
        emb_cols = [f"dim_{i}" for i in range(dim)]
        emb_matrix = df[emb_cols].values

        counts = pd.Series(labels).value_counts()
        rare_classes = counts[(counts < 10) | (counts / len(labels) < 0.01)].index.tolist()

        if len(labels) < 2:
            report_lines.append("- Not enough data for metrics.\n")
            continue

        # Stratified subsample for global silhouette
        emb_sub, labels_sub = get_stratified_subsample(emb_matrix, labels, max_size=5000)
        global_sil = silhouette_score(emb_sub, labels_sub, metric='cosine') if len(set(labels_sub)) > 1 else 0

        global_knn, class_knn = compute_knn_purity(emb_matrix, labels)
        within_dists, between_dists = auc_overlap(emb_matrix, labels)

        if len(within_dists) > 0 and len(between_dists) > 0:
            n_samples = min(len(within_dists), len(between_dists), 1000)
            wd = np.random.choice(within_dists, n_samples)
            bd = np.random.choice(between_dists, n_samples)
            auc_score = np.mean(wd[:, None] < bd[None, :])
        else:
            auc_score = 0.0

        dist_plot_data.append({
            "model": model_name,
            "dataset": ds_name,
            "within": within_dists,
            "between": between_dists
        })

        report_lines.append(f"### Global Metrics")
        report_lines.append(f"- Silhouette Score (Cosine): {global_sil:.4f}")
        report_lines.append(f"- 10-NN Class Purity: {global_knn:.4f}")
        report_lines.append(f"- Distance AUC (Within < Between): {auc_score:.4f}\n")

        if len(set(labels)) > 1:
            all_sils = silhouette_samples(emb_matrix, labels, metric='cosine')
        else:
            all_sils = np.zeros(len(labels))

        report_lines.append(f"### Per-Class Metrics")
        for cls in sorted(list(set(labels))):
            cls_mask = labels == cls
            cls_knn = class_knn.get(cls, 0.0)
            cls_sil = np.mean(all_sils[cls_mask]) if sum(cls_mask) > 1 else 0.0

            is_rare = "(RARE)" if cls in rare_classes else ""
            report_lines.append(f"- **{cls}** {is_rare} | Silhouette: {cls_sil:.4f} | 10-NN Purity: {cls_knn:.4f}")

        report_lines.append("\n---\n")

    report_lines.insert(2, "## Final Verdict\n**Which embedding has the best chance of supporting rare-class discovery?**\nBased on the rare-class silhouette scores and 10-NN purities, the optimal model is the one with the highest consistent positive values for rare classes. If all models have zero or negative silhouette for rare classes, then **NONE** of these embeddings meaningfully separate rare classes and this must be escalated before step 3.\n\n---\n")

    with open(out_dir / "embedding_diagnostics.md", "w") as f:
        f.write("\n".join(report_lines))

    n_plots = len(dist_plot_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots))
    if n_plots == 1: axes = [axes]

    for ax, data in zip(axes, dist_plot_data):
        if len(data["within"]) > 0:
            sns.kdeplot(data["within"], label="Within-class", fill=True, ax=ax)
        if len(data["between"]) > 0:
            sns.kdeplot(data["between"], label="Between-class", fill=True, ax=ax)
        ax.set_title(f"{data['model'].upper()} - {data['dataset']}")
        ax.set_xlabel("Cosine Distance")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "embedding_diagnostics.png")
    print(f"Diagnostics saved to {out_dir}")

def process_dataset(dataset_name: str, model_name: str, config):
    print(f"\n[{model_name}] Processing dataset: {dataset_name}")
    raw_dir = Path(config.paths.raw_dir)
    processed_dir = Path(config.paths.processed_dir) / dataset_name
    manifest_path = processed_dir / "dataset_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}, skipping.")
        return None

    manifest = pd.read_csv(manifest_path)

    extractor = get_extractor(model_name)
    dim = extractor.dimension

    embeddings = []
    clip_ids = []
    labels = []
    statuses = []

    start_total = time.time()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        clip_id = row['clip_id']
        source_file = row['source_file']
        audio_path = raw_dir / source_file

        clip_ids.append(clip_id)
        labels.append(row['label'])

        try:
            y, sr = sf.read(str(audio_path))
            emb = extractor.extract(y, sr)
            embeddings.append(emb)
            statuses.append("success")
        except Exception as e:
            statuses.append(f"error: {str(e)}")
            embeddings.append(np.full(dim, np.nan))

    total_time = time.time() - start_total

    df_out = pd.DataFrame({"clip_id": clip_ids, "label": labels})
    embeddings_np = np.array(embeddings)

    for i in range(dim):
        df_out[f"dim_{i}"] = embeddings_np[:, i]

    out_path = processed_dir / f"embeddings_{model_name}.parquet"
    df_out.to_parquet(out_path, index=False)

    failures = [s for s in statuses if s != "success"]

    return {
        "dataset": dataset_name,
        "model": model_name,
        "dimension": dim,
        "total_runtime_seconds": total_time,
        "throughput_clips_per_second": len(manifest) / total_time if total_time > 0 else 0,
        "failures": len(failures),
        "df": df_out
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["yamnet", "panns", "birdnet", "all"])
    args = parser.parse_args()

    config = load_config()
    set_seeds(config.seed)

    datasets = ["esc50", "dcase2024_t5"]
    models_to_run = ["yamnet", "panns", "birdnet"] if args.model == "all" else [args.model]

    import torch
    import tensorflow as tf
    hw_info = "GPU" if torch.cuda.is_available() or tf.config.list_physical_devices('GPU') else "CPU"

    all_results = []

    for ds in datasets:
        for m in models_to_run:
            res = process_dataset(ds, m, config)
            if res:
                all_results.append(res)

    if all_results:
        print(f"\nExtraction complete. Generating diagnostics...")
        generate_diagnostics(all_results, hw_info, Path(config.paths.results_dir))

if __name__ == "__main__":
    main()
