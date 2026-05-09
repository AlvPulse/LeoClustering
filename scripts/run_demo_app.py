import streamlit as st
import yaml
from pathlib import Path
import json

from src.config import load_config
from src.demo.state import init_session_state, handle_plotly_click
from src.demo.data import get_best_run, load_data, precompute_umap, compute_structured_samples
from src.demo.labelling import LabelManager
from src.demo.components.panels import (
    render_projection_panel,
    render_feature_scatter_panel,
    render_cluster_nav_panel,
    render_clip_detail_panel,
    render_physics_panel
)

st.set_page_config(layout="wide", page_title="Audio Labeller Demo")

def main():
    st.title("Wildlife Monitoring Audio Labeller - Demo")
    st.error("**This is a benchmark demonstration, not a deployment.** It is read-only exploration without real-time federation, noise adaptation, or model retraining capabilities.")

    init_session_state()
    base_config = load_config()

    with open("configs/step6_demo.yaml", "r") as f:
        demo_config = yaml.safe_load(f)

    # Sidebar
    st.sidebar.title("Controls")
    dataset = st.sidebar.selectbox("Dataset", ["esc50", "dcase2024_t5"])

    # Mode toggle (Labelled vs unlabelled)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### View Mode")
    st.session_state.view_mode = st.sidebar.radio(
        "Select coloring mode:",
        ["Evaluation View", "Labeller View"],
        help="Evaluation shows ground truth colors. Labeller shows only clusters."
    )

    if st.session_state.view_mode == "Evaluation View":
        st.sidebar.info("🟦 **Evaluation Mode Active**\n\nColors = Ground Truth Labels")
    else:
        st.sidebar.warning("🟧 **Labeller Mode Active**\n\nColors = Blind Cluster Assignments")

    st.sidebar.markdown("---")

    # Auto-resolve best choices
    best_emb, _ = get_best_run(Path(base_config.paths.results_dir), "step2", "silhouette")
    best_clust, _ = get_best_run(Path(base_config.paths.results_dir), "step3", "survival_count")

    emb_choice = st.sidebar.text_input("Embedding", value=best_emb if best_emb else "panns")
    clust_choice = st.sidebar.text_input("Clustering", value=best_clust if best_clust else "hdbscan_panns_00000000")

    rare_filter = st.sidebar.checkbox("Show ONLY rare classes", value=False)

    if st.sidebar.button("Reset Selection"):
        st.session_state.selected_clip_id = None
        st.session_state.selected_cluster_id = None
        st.rerun()

    # Load Data
    with st.spinner("Loading representations..."):
        df, emb_df, assignment_df = load_data(dataset, emb_choice, clust_choice, base_config.paths.model_dump())

        # Determine rare classes
        inv_path = Path(base_config.paths.results_dir) / "step0" / "simulated" / f"{dataset}_inventory.json"
        rare_classes = []
        if inv_path.exists():
            with open(inv_path, "r") as f:
                inv = json.load(f)
                counts = inv.get("per_class_clip_counts", {})
                total = sum(counts.values())
                rare_classes = [k for k, v in counts.items() if v < 10 or v / total < 0.01]

    with st.spinner("Precomputing UMAP..."):
        umap_df = precompute_umap(emb_df)
        df_merged = df.merge(umap_df, on="clip_id").merge(assignment_df, on="clip_id")

    with st.spinner("Precomputing Structured Samples..."):
        samples = compute_structured_samples(emb_df, assignment_df)

    if rare_filter:
        df_merged = df_merged[df_merged["label"].isin(rare_classes)]

    # Labelling manager
    label_mgr = LabelManager(Path(base_config.paths.results_dir) / "step6" / "labels")

    # UI Layout
    col_proj, col_nav = st.columns([1, 1])

    with col_proj:
        proj_mode = st.radio("Projection Mode", ["UMAP View", "Feature Scatter"])
        if proj_mode == "UMAP View":
            event = render_projection_panel(df_merged, st.session_state.view_mode, rare_classes)
            handle_plotly_click(event)
        else:
            feat_list = demo_config["demo"]["default_features"]
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("X Axis", feat_list, index=0)
            with c2:
                y_col = st.selectbox("Y Axis", feat_list, index=1)

            event = render_feature_scatter_panel(df_merged, x_col, y_col, st.session_state.view_mode)
            handle_plotly_click(event)

    with col_nav:
        render_cluster_nav_panel(df_merged, samples)

    st.divider()

    col_det, col_phys = st.columns([1, 1])

    selected_clip = None
    if st.session_state.selected_clip_id is not None:
        sel_mask = df_merged["clip_id"] == st.session_state.selected_clip_id
        if sel_mask.any():
            selected_clip = df_merged[sel_mask]

    with col_det:
        render_clip_detail_panel(selected_clip, Path(base_config.paths.raw_dir), label_mgr)

    with col_phys:
        render_physics_panel(selected_clip, df_merged, demo_config["demo"]["default_features"])

if __name__ == "__main__":
    main()
