import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

def generate_reports(res_df: pd.DataFrame, dataset_name: str, out_dir: Path, base_config):
    fig_dir = out_dir / "fewshot_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inv_path = Path(base_config.paths.results_dir) / "step0" / "simulated" / f"{dataset_name}_inventory.json"
    rare_classes = []
    if inv_path.exists():
        with open(inv_path, "r") as f:
            inv = json.load(f)
            counts = inv.get("per_class_clip_counts", {})
            total = sum(counts.values())
            rare_classes = [k for k, v in counts.items() if v < 10 or v / total < 0.01]

    df_valid = res_df[res_df['metric'] != 'error'].copy()

    # Check if df_valid is empty
    if len(df_valid) == 0:
        with open(out_dir / "fewshot_summary.md", "w") as f:
            f.write(f"# Few-Shot Detection Benchmark Summary: {dataset_name}\n\n")
            f.write("All runs resulted in errors. Insufficient data.\n")
        return

    df_valid['value'] = pd.to_numeric(df_valid['value'], errors='coerce')
    df_valid.loc[df_valid['value'] < 0, 'value'] = np.nan

    agg_df = df_valid.groupby(['class', 'n_exemplars', 'method', 'representation', 'metric'])['value'].agg(['mean', 'std']).reset_index()
    n_vals = sorted(agg_df['n_exemplars'].unique())

    # Write Summary Markdown
    with open(out_dir / "fewshot_summary.md", "w") as f:
        f.write(f"# Few-Shot Detection Benchmark Summary: {dataset_name}\n\n")

        f.write("## Headline Finding: Rare Class Detection\n")
        f.write("Best achievable Recall @ FPR=1/hour for each Rare Class across N exemplars. ")
        f.write("If values are NaN, it indicates insufficient labels or test sets too small to compute reliably.\n\n")

        f.write("| Rare Class | " + " | ".join([f"N={n}" for n in n_vals]) + " |\n")
        f.write("|---|" + "|".join(["---" for _ in n_vals]) + "|\n")

        for rc in rare_classes:
            f.write(f"| `{rc}` | ")
            rc_df = agg_df[(agg_df['class'] == rc) & (agg_df['metric'] == 'r_at_fpr_hour')]
            for n in n_vals:
                n_df = rc_df[rc_df['n_exemplars'] == n]
                if n_df.empty or n_df['mean'].isna().all():
                    f.write("N/A | ")
                else:
                    best = n_df.loc[n_df['mean'].idxmax()]
                    f.write(f"{best['mean']:.3f}±{best['std']:.3f} ({best['method']}/{best['representation']}) | ")
            f.write("\n")

        f.write("\n---\n\n")

        f.write("## Best Method / Representation Per Class\n")
        f.write("Evaluated based on max Average Precision (AP) at N=max.\n")
        f.write("| Class | Best Combination | Mean AP | Std AP |\n")
        f.write("|---|---|---|---|\n")

        max_n = max(n_vals) if n_vals else 0
        ap_df = agg_df[(agg_df['n_exemplars'] == max_n) & (agg_df['metric'] == 'ap')]
        for cls in sorted(agg_df['class'].unique()):
            cls_ap = ap_df[ap_df['class'] == cls]
            if not cls_ap.empty and not cls_ap['mean'].isna().all():
                best = cls_ap.loc[cls_ap['mean'].idxmax()]
                f.write(f"| {cls} | {best['method']} / {best['representation']} | {best['mean']:.3f} | {best['std']:.3f} |\n")
            else:
                f.write(f"| {cls} | N/A | N/A | N/A |\n")

        f.write("\n---\n\n")

        f.write("## Representation Comparison\n")
        f.write("Mean Average Precision across all classes and methods for N=10:\n")

        n10_ap = agg_df[(agg_df['n_exemplars'] == 10) & (agg_df['metric'] == 'ap')]
        if not n10_ap.empty:
            rep_ap = n10_ap.groupby('representation')['mean'].mean().sort_values(ascending=False)
            for rep, val in rep_ap.items():
                f.write(f"- **{rep}**: {val:.3f}\n")

        f.write("\n---\n\n")

        f.write("## Failure Modes\n")
        f.write("Classes where NO method reached > 0.5 Average Precision even at max N:\n")
        failed = []
        for cls in sorted(agg_df['class'].unique()):
            cls_ap = ap_df[ap_df['class'] == cls]
            if not cls_ap.empty:
                best_val = cls_ap['mean'].max()
                if best_val < 0.5:
                    failed.append(cls)
        if failed:
            for cls in failed:
                f.write(f"- ⚠️ **{cls}**\n")
        else:
            f.write("- None observed.\n")

    # Generate Learning Curves Figure (AP vs N by Repr)
    plot_df = agg_df[agg_df['metric'] == 'ap'].groupby(['n_exemplars', 'representation'])['mean'].mean().reset_index()

    if not plot_df.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=plot_df, x='n_exemplars', y='mean', hue='representation', marker='o')
        plt.title("Learning Curves: Average Precision vs N (by Representation)")
        plt.ylabel("Mean AP")
        plt.xlabel("N Exemplars")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / "learning_curves_repr.png")
        plt.close()

    # AP vs N by Method
    plot_df2 = agg_df[agg_df['metric'] == 'ap'].groupby(['n_exemplars', 'method'])['mean'].mean().reset_index()

    if not plot_df2.empty:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=plot_df2, x='n_exemplars', y='mean', hue='method', marker='o')
        plt.title("Learning Curves: Average Precision vs N (by Method)")
        plt.ylabel("Mean AP")
        plt.xlabel("N Exemplars")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / "learning_curves_method.png")
        plt.close()

    # Heatmap of best-achievable Recall @ FPR=1/hour
    hm_records = []
    for cls in sorted(agg_df['class'].unique()):
        cls_df = agg_df[(agg_df['class'] == cls) & (agg_df['metric'] == 'r_at_fpr_hour')]
        for n in n_vals:
            n_df = cls_df[cls_df['n_exemplars'] == n]
            best_val = n_df['mean'].max() if not n_df.empty and not n_df['mean'].isna().all() else np.nan
            hm_records.append({
                "Class": f"{cls}{' (RARE)' if cls in rare_classes else ''}",
                "N": n,
                "Recall_at_1_hour": best_val
            })

    if hm_records:
        hm_df = pd.DataFrame(hm_records)
        hm_pivot = hm_df.pivot(index="Class", columns="N", values="Recall_at_1_hour")

        plt.figure(figsize=(8, max(4, len(hm_pivot) * 0.5)))
        sns.heatmap(hm_pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Best Recall @ FPR=1/hour")
        plt.tight_layout()
        plt.savefig(fig_dir / "recall_fpr_heatmap.png")
        plt.close()

    # Score distribution for best method/N combinations per rare class
    scores_dir = out_dir / "scores"
    if scores_dir.exists():
        for rc in rare_classes:
            rc_ap = agg_df[(agg_df['class'] == rc) & (agg_df['metric'] == 'ap')]
            if not rc_ap.empty and not rc_ap['mean'].isna().all():
                best = rc_ap.loc[rc_ap['mean'].idxmax()]
                best_n = best['n_exemplars']
                best_meth = best['method']
                best_rep = best['representation']

                # Pick the first repetition
                score_file = scores_dir / f"{rc}_{best_n}_{best_meth}_{best_rep}_0.parquet"
                if score_file.exists():
                    sdf = pd.read_parquet(score_file)
                    plt.figure(figsize=(8, 4))
                    sns.histplot(data=sdf, x="score", hue="label_is_target", element="step", stat="density", common_norm=False)
                    plt.title(f"Score Distribution: {rc} (N={best_n}, {best_meth}, {best_rep})")
                    plt.tight_layout()
                    plt.savefig(fig_dir / f"score_dist_{rc}.png")
                    plt.close()
