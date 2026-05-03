import os
import argparse
import time
import ast
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

from src.config import load_config
from src.utils import set_seeds
from src.physics_features.extractor import extract_all_features
import src.physics_features.spectral as spectral
import src.physics_features.temporal as temporal
import src.physics_features.harmonic as harmonic
import src.physics_features.quality as quality
import src.physics_features.bioacoustic as bioacoustic
import inspect

warnings.filterwarnings("ignore")

def extract_features_with_profiling(y: np.ndarray, sr: int, duration: float) -> dict:
    features = {}
    timings = {}

    t0 = time.time()
    extracted_spectral = spectral.compute_spectral_features(y, sr)
    features.update(extracted_spectral)
    timings.update({k: time.time() - t0 for k in extracted_spectral.keys()})

    t0 = time.time()
    extracted_temporal = temporal.compute_temporal_features(y, sr, duration)
    features.update(extracted_temporal)
    timings.update({k: time.time() - t0 for k in extracted_temporal.keys()})

    t0 = time.time()
    extracted_harmonic = harmonic.compute_harmonic_features(y, sr)
    features.update(extracted_harmonic)
    timings.update({k: time.time() - t0 for k in extracted_harmonic.keys()})

    t0 = time.time()
    extracted_quality = quality.compute_quality_features(y)
    features.update(extracted_quality)
    timings.update({k: time.time() - t0 for k in extracted_quality.keys()})

    t0 = time.time()
    extracted_bioacoustic = bioacoustic.compute_bioacoustic_features(y, sr, duration)
    features.update(extracted_bioacoustic)
    timings.update({k: time.time() - t0 for k in extracted_bioacoustic.keys()})

    return features, timings

def process_clip(row, raw_dir: Path, profile=False):
    start_time = time.time()
    clip_id = row['clip_id']
    source_file = row['source_file']
    audio_path = raw_dir / source_file

    features = {"clip_id": clip_id, "label": row["label"]}
    timings = {}

    try:
        y, sr = sf.read(str(audio_path))
        if profile:
            extracted, timings = extract_features_with_profiling(y, sr, row['duration_seconds'])
        else:
            extracted = extract_all_features(y, sr, row['duration_seconds'])
        features.update(extracted)
        status = "success"
    except Exception as e:
        status = f"error: {str(e)}"
        # Emit NaNs
        features.update({k: np.nan for k in extract_all_features(np.array([]), 22050, 0.0).keys()})

    end_time = time.time()

    return features, timings, end_time - start_time, status

def process_dataset(dataset_name: str, config, num_workers: int):
    print(f"\nProcessing dataset: {dataset_name}")
    raw_dir = Path(config.paths.raw_dir)
    processed_dir = Path(config.paths.processed_dir) / dataset_name
    manifest_path = processed_dir / "dataset_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}, skipping.")
        return None

    manifest = pd.read_csv(manifest_path)
    print(f"Extracting features for {len(manifest)} clips using {num_workers} workers...")

    start_total = time.time()
    # Only profile the first clip to calculate typical feature cost, to avoid saving massive timings dicts
    # but still execute parallel extraction for speed

    first_res = process_clip(manifest.iloc[0], raw_dir, profile=True)
    first_timings = first_res[1]

    results = Parallel(n_jobs=num_workers)(
        delayed(process_clip)(row, raw_dir, profile=False) for _, row in tqdm(manifest.iterrows(), total=len(manifest))
    )
    total_time = time.time() - start_total

    features_list = [r[0] for r in results]
    statuses = [r[3] for r in results]

    df_features = pd.DataFrame(features_list)
    out_path = processed_dir / "physics_features.parquet"
    df_features.to_parquet(out_path, index=False)

    # Process Timings
    median_time = np.median(list(first_timings.values()))
    slow_features = {k: v for k, v in first_timings.items() if v > 10 * median_time}

    throughput = len(manifest) / total_time
    failures = [s for s in statuses if s != "success"]

    diagnostics = {
        "dataset": dataset_name,
        "total_runtime_seconds": total_time,
        "throughput_clips_per_second": throughput,
        "median_feature_cost_seconds": median_time,
        "slow_features_gt_10x_median": slow_features,
        "failures": len(failures),
        "failure_reasons": list(set(failures)),
        "feature_timings": first_timings
    }

    return df_features, diagnostics

def parse_docstrings():
    docs = {}
    modules = [spectral, temporal, harmonic, quality, bioacoustic]
    for mod in modules:
        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if "compute_" in name:
                doc = inspect.getdoc(func)
                if doc:
                    lines = doc.split('\n')
                    current_feat = None
                    for line in lines:
                        line = line.strip()
                        if line.startswith(tuple(str(i)+'.' for i in range(1, 20))):
                            current_feat = line.split('.')[1].split(':')[0].strip()
                            if current_feat not in docs:
                                docs[current_feat] = {}
                        elif current_feat and line.startswith("- Physical meaning:"):
                            docs[current_feat]['meaning'] = line.replace("- Physical meaning:", "").strip()
                        elif current_feat and line.startswith("- Computation:"):
                            docs[current_feat]['computation'] = line.replace("- Computation:", "").strip()
                        elif current_feat and line.startswith("- Rationale:"):
                            docs[current_feat]['rationale'] = line.replace("- Rationale:", "").strip()
    return docs

def generate_reports(all_diagnostics, all_features_dict, results_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "step1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parquet copies
    for ds_name, df in all_features_dict.items():
        df.to_parquet(out_dir / f"{ds_name}_physics_features.parquet", index=False)

    # 2. Extraction diagnostics
    with open(out_dir / "extraction_diagnostics.md", "w") as f:
        f.write("# Extraction Diagnostics\n\n")
        for diag in all_diagnostics:
            f.write(f"## Dataset: {diag['dataset']}\n")
            f.write(f"- Total Runtime: {diag['total_runtime_seconds']:.2f} s\n")
            f.write(f"- Throughput: {diag['throughput_clips_per_second']:.2f} clips/s\n")
            f.write(f"- Failures: {diag['failures']}\n")
            if diag['slow_features_gt_10x_median']:
                f.write("### ⚠️ SLOW FEATURES ( > 10x median cost)\n")
                for k, v in diag['slow_features_gt_10x_median'].items():
                    f.write(f"- **{k}**: {v:.4f} s\n")
            else:
                f.write("\nNo features exceed 10x the median cost.\n")
            f.write("\n")

    # Combine datasets for unified reporting
    df_combined = pd.concat(all_features_dict.values(), ignore_index=True)
    features_to_plot = [c for c in df_combined.columns if c not in ["clip_id", "label"]]

    # Identify top 10 most populous classes + rare classes (<1% or <10 clips)
    class_counts = df_combined['label'].value_counts()
    top_10 = class_counts.nlargest(10).index.tolist()
    rare_classes = class_counts[(class_counts < 10) | (class_counts / len(df_combined) < 0.01)].index.tolist()
    target_classes = list(set(top_10 + rare_classes))

    df_filtered = df_combined[df_combined['label'].isin(target_classes)]

    # 3. Feature Dictionary
    docs = parse_docstrings()

    with open(out_dir / "feature_dictionary.md", "w") as f:
        f.write("# Feature Dictionary\n\n")
        for col in features_to_plot:
            nans = df_combined[col].isna().mean()
            f.write(f"## `{col}`\n")
            if col in docs:
                f.write(f"**Physical Meaning:** {docs[col].get('meaning', 'N/A')}\n\n")
                f.write(f"**Computation Summary:** {docs[col].get('computation', 'N/A')}\n\n")
                f.write(f"**Bioacoustic Rationale:** {docs[col].get('rationale', 'N/A')}\n\n")

            f.write(f"**Fraction NaNs:** {nans:.2%}\n\n")

            if not df_combined[col].isna().all():
                f.write(f"**Global Typical Range:** [{df_combined[col].min():.2f}, {df_combined[col].max():.2f}]\n\n")
                f.write(f"**Global Mean/Std:** {df_combined[col].mean():.2f} ± {df_combined[col].std():.2f}\n\n")
                f.write("**Per-Class Stats (Top 10 + Rare):**\n")
                for cls in target_classes:
                    cls_data = df_filtered[df_filtered['label'] == cls][col]
                    f.write(f"- `{cls}`: {cls_data.mean():.2f} ± {cls_data.std():.2f}\n")
            f.write("\n---\n\n")

    # 4. Feature distributions plot
    num_features = len(features_to_plot)
    cols = 4
    rows = (num_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(features_to_plot):
        sns.boxplot(data=df_filtered, x="label", y=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(out_dir / "feature_distributions.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    config = load_config()
    set_seeds(config.seed)

    datasets = ["esc50", "dcase2024_t5"]
    all_features_dict = {}
    all_diagnostics = []

    for ds in datasets:
        res = process_dataset(ds, config, args.num_workers)
        if res is not None:
            df_features, diagnostics = res
            all_features_dict[ds] = df_features
            all_diagnostics.append(diagnostics)

    generate_reports(all_diagnostics, all_features_dict, Path(config.paths.results_dir))
    print("Done! Reports generated.")

if __name__ == "__main__":
    main()
