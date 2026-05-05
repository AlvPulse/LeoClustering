import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from pathlib import Path
import json
import hashlib

def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]

def generate_reports(res_df: pd.DataFrame, dataset_name: str, out_dir: Path, rare_classes: list, data_dfs: dict, assignments_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "clustering_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def score_row(row):
        survival_score = sum([row.get(f"rc_{rc}_survival", False) for rc in rare_classes])
        return survival_score * 1000 + row['ari']

    res_df['score'] = res_df.apply(score_row, axis=1)

    best_df = res_df.loc[res_df.groupby(['algorithm', 'representation'])['score'].idxmax()].copy()

    # Write Summary Markdown
    with open(out_dir / "clustering_summary.md", "w") as f:
        f.write(f"# Clustering Benchmark Summary: {dataset_name}\n\n")

        f.write("## Headline Finding: Rare Class Preservation\n")
        f.write("The following table shows whether each (algorithm, representation) combination successfully preserved rare classes (True = formed a distinct cluster, False = absorbed into noise or another class).\n\n")

        f.write("| Algorithm | Representation | ")
        for rc in rare_classes:
            f.write(f"Survives: `{rc}` | ")
        f.write("\n|---|---|")
        for _ in rare_classes: f.write("---|")
        f.write("\n")

        for _, row in best_df.iterrows():
            f.write(f"| {row['algorithm']} | {row['representation']} | ")
            for rc in rare_classes:
                val = "✅ Yes" if row.get(f"rc_{rc}_survival", False) else "❌ No"
                f.write(f"{val} | ")
            f.write("\n")
        f.write("\n---\n\n")

        f.write("## Comparison Table (Best Hyperparameters)\n")
        f.write("| Algorithm | Representation | ARI | NMI | Weighted Purity | Clusters | Largest Frac |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for _, row in best_df.iterrows():
            f.write(f"| {row['algorithm']} | {row['representation']} | {row['ari']:.3f} | {row['nmi']:.3f} | {row['weighted_purity']:.3f} | {row['n_clusters']} | {row['largest_cluster_fraction']:.2f} |\n")
        f.write("\n---\n\n")

        f.write("## Per-Rare-Class Deep Dive\n")
        for rc in rare_classes:
            f.write(f"### Class: `{rc}`\n")
            f.write("| Algorithm | Representation | Recall | Purity | Noise Rate |\n")
            f.write("|---|---|---|---|---|\n")
            for _, row in best_df.iterrows():
                rec = row.get(f"rc_{rc}_recall", 0.0)
                pur = row.get(f"rc_{rc}_purity", 0.0)
                noi = row.get(f"rc_{rc}_noise_rate", 1.0)
                f.write(f"| {row['algorithm']} | {row['representation']} | {rec:.3f} | {pur:.3f} | {noi:.3f} |\n")
            f.write("\n")

        f.write("---\n\n")

        f.write("## Hyperparameter Sensitivity Analysis\n")
        f.write("Maximum variation (max - min) in ARI across the hyperparameter grid for each method:\n\n")
        sensitivity = res_df.groupby(['algorithm', 'representation'])['ari'].agg(lambda x: x.max() - x.min()).reset_index()
        for _, row in sensitivity.iterrows():
            f.write(f"- **{row['algorithm']}** ({row['representation']}): ΔARI = {row['ari']:.3f}\n")
        f.write("\n---\n\n")

        f.write("## Failure Modes Observed\n")
        for _, row in best_df.iterrows():
            if row['largest_cluster_fraction'] > 0.9:
                f.write(f"- ⚠️ **{row['algorithm']} ({row['representation']})**: One giant cluster failure (>90% of data in one cluster).\n")
            if row['n_clusters'] == 0:
                f.write(f"- ⚠️ **{row['algorithm']} ({row['representation']})**: Total noise absorption (0 valid clusters).\n")
            for rc in rare_classes:
                if row.get(f"rc_{rc}_noise_rate", 0.0) > 0.9:
                    f.write(f"- ⚠️ **{row['algorithm']} ({row['representation']})**: Put >90% of rare class `{rc}` into noise.\n")

    # Generate Plots

    # 1. Bar chart of preserved rare classes
    preserved_counts = []
    for _, row in best_df.iterrows():
        count = sum([row.get(f"rc_{rc}_survival", False) for rc in rare_classes])
        preserved_counts.append({
            "Method": f"{row['algorithm']}\n({row['representation']})",
            "Preserved": count
        })
    pdf = pd.DataFrame(preserved_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pdf, x="Preserved", y="Method", color="blue")
    plt.title("Number of Rare Classes Preserved")
    plt.tight_layout()
    plt.savefig(fig_dir / "rare_classes_preserved.png")
    plt.close()

    # 2. Recall Heatmap
    from src.clustering.metrics import compute_rare_class_metrics
    all_classes = data_dfs[list(data_dfs.keys())[0]]['label'].unique().tolist()

    recall_records = []
    for _, row in best_df.iterrows():
        algo = row['algorithm']
        rep = row['representation']
        params_str = row['hyperparameters']
        # The params inside row['hyperparameters'] is already a json string
        params = json.loads(params_str)
        hp_id = stable_hash(params)

        assignment_path = assignments_dir / f"{algo}_{rep}_{hp_id}.parquet"

        # fallback since joblib multiprocessing sometimes loses the hash depending on python version
        if not assignment_path.exists():
            # Try to find a file matching the prefix
            matching = list(assignments_dir.glob(f"{algo}_{rep}_*.parquet"))
            if matching:
                assignment_path = matching[0] # Best effort

        if assignment_path.exists():
            ass_df = pd.read_parquet(assignment_path)

            test_df = data_dfs[rep]
            test_df = test_df[test_df['split'] == 'test']
            if len(test_df) == 0: test_df = data_dfs[rep]

            y_true = test_df['label'].values
            y_pred = ass_df['cluster_id'].values

            metrics = compute_rare_class_metrics(y_true, y_pred, rare_classes)
            for cls in all_classes:
                recall = metrics.get('per_class_recall', {}).get(cls, 0.0)
                recall_records.append({
                    "Method": f"{algo}\n({rep})",
                    "Class": cls + (" (RARE)" if cls in rare_classes else ""),
                    "Recall": recall
                })
        else:
            print(f"Warning: {assignment_path} not found.")

    if recall_records:
        recall_df = pd.DataFrame(recall_records)
        recall_pivot = recall_df.pivot(index="Class", columns="Method", values="Recall")

        plt.figure(figsize=(14, max(4, len(all_classes) * 0.5)))
        sns.heatmap(recall_pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Per-Class Recall (Dominant Cluster)")
        plt.tight_layout()
        plt.savefig(fig_dir / "recall_heatmap.png")
        plt.close()
    else:
        print("Warning: recall_records empty, heatmap skipped.")

    # 3. UMAP Scatter Subplots (One per method)
    best_rep_row = best_df.loc[best_df['score'].idxmax()]
    best_rep = best_rep_row['representation']

    if best_rep in data_dfs:
        df = data_dfs[best_rep]
        test_df = df[df['split'] == 'test'].copy()
        if len(test_df) == 0: test_df = df.copy()

        dim_cols = [c for c in test_df.columns if c.startswith("dim_") or c.startswith("spectral_") or "power" in c or "hz" in c or "time" in c or "rate" in c or "count" in c or "snr" in c]
        X_test = test_df[dim_cols].values
        labels = test_df['label'].values

        if len(X_test) > 5:
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(X_test)

            methods = best_df[best_df['representation'] == best_rep]['algorithm'].unique().tolist()

            cols = min(3, len(methods))
            rows = int(np.ceil(len(methods) / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            if rows * cols == 1: axes = np.array([axes])
            axes = axes.flatten()

            for i, algo in enumerate(methods):
                ax = axes[i]

                row = best_df[(best_df['algorithm'] == algo) & (best_df['representation'] == best_rep)].iloc[0]
                params = json.loads(row['hyperparameters'])
                hp_id = stable_hash(params)
                assignment_path = assignments_dir / f"{algo}_{best_rep}_{hp_id}.parquet"

                if not assignment_path.exists():
                    matching = list(assignments_dir.glob(f"{algo}_{best_rep}_*.parquet"))
                    if matching: assignment_path = matching[0]

                if assignment_path.exists():
                    ass_df = pd.read_parquet(assignment_path)
                    y_pred = ass_df['cluster_id'].values

                    unique_clusters = np.unique(y_pred)
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
                    cluster_to_marker = {}
                    for idx, c in enumerate(unique_clusters):
                        if c == -1: cluster_to_marker[c] = 'X'
                        else: cluster_to_marker[c] = markers[idx % len(markers)]

                    marker_list = [cluster_to_marker[c] for c in y_pred]

                    sns.scatterplot(
                        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                        hue=labels, style=y_pred, markers=cluster_to_marker,
                        palette="tab10", s=80, ax=ax, legend=False
                    )
                    ax.set_title(f"{algo} (ARI: {row['ari']:.2f})")

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle(f"UMAP of {best_rep} with Cluster Assignments", fontsize=16)
            plt.tight_layout()
            plt.savefig(fig_dir / f"umap_subplots_{best_rep}.png")
            plt.close()
