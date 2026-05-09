import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

def render_projection_panel(df_merged, view_mode, rare_classes):
    st.subheader("Projection Panel")

    # Subsample if needed
    max_pts = 5000
    if len(df_merged) > max_pts:
        plot_df = df_merged.sample(max_pts, random_state=42)
    else:
        plot_df = df_merged

    color_col = "label" if view_mode == "Evaluation View" else "cluster_id_str"

    if "cluster_id_str" not in plot_df.columns:
        plot_df["cluster_id_str"] = plot_df["cluster_id"].apply(lambda x: "Unclustered (noise)" if x == -1 else str(x))

    # Highlight rare classes via symbol
    plot_df["is_rare"] = plot_df["label"].apply(lambda x: "Rare" if x in rare_classes else "Common")

    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color=color_col,
        symbol="is_rare",
        hover_data=["clip_id", "label", "cluster_id"],
        custom_data=["clip_id"]
    )

    # Mark labelled points
    # if st.session_state.get('labels'): ...

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Use streamlit 1.35 native selection
    # If the user clicks, it returns a dict of selection data
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
    return event

def render_feature_scatter_panel(df_merged, x_col, y_col, view_mode):
    st.subheader(f"Scatter: {x_col} vs {y_col}")
    color_col = "label" if view_mode == "Evaluation View" else "cluster_id_str"
    if "cluster_id_str" not in df_merged.columns:
        df_merged["cluster_id_str"] = df_merged["cluster_id"].apply(lambda x: "Unclustered (noise)" if x == -1 else str(x))

    fig = px.scatter(
        df_merged,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=["clip_id", "label", "cluster_id"],
        custom_data=["clip_id"]
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
    return event

def render_cluster_nav_panel(df_merged, samples):
    st.subheader("Cluster Navigation")

    clusters = sorted(df_merged["cluster_id"].unique())
    clusters_str = ["Unclustered (noise)" if c == -1 else str(c) for c in clusters]

    sel_idx = 0
    if st.session_state.selected_cluster_id is not None:
        if st.session_state.selected_cluster_id in clusters:
            sel_idx = clusters.index(st.session_state.selected_cluster_id)

    sel_str = st.selectbox("Select Cluster", clusters_str, index=sel_idx)

    try:
        sel_c = -1 if sel_str == "Unclustered (noise)" else int(sel_str)
        st.session_state.selected_cluster_id = sel_c
    except:
        sel_c = -1

    if sel_c == -1:
        st.write("Noise points. No structured samples available.")
        return

    c_samples = samples.get(sel_c, {})
    if c_samples.get("type") == "small":
        st.write("Small cluster — all members shown.")
        for mem in c_samples["members"]:
            if st.button(f"Clip: {mem}", key=f"btn_{mem}"):
                st.session_state.selected_clip_id = mem
                st.rerun()
    elif c_samples.get("type") == "large":
        st.write("**Medoid**")
        if st.button(f"Clip: {c_samples['medoid']}", key=f"med_{c_samples['medoid']}"):
            st.session_state.selected_clip_id = c_samples['medoid']
            st.rerun()

        st.write("**Boundary Points**")
        for b in c_samples['boundaries']:
            if st.button(f"Clip: {b}", key=f"bound_{b}"):
                st.session_state.selected_clip_id = b
                st.rerun()

        st.write("**Outliers**")
        for o in c_samples['outliers']:
            if st.button(f"Clip: {o}", key=f"out_{o}"):
                st.session_state.selected_clip_id = o
                st.rerun()

@st.cache_data(max_entries=100)
def get_audio_and_spec(source_file, start_s, duration_s, raw_dir):
    full_path = raw_dir / source_file

    try:
        y, sr = sf.read(str(full_path))
    except Exception as e:
        return None, None, f"Audio load failed: {e}"

    # Trim to max 10 seconds for UI
    trim_note = ""
    if duration_s > 10.0:
        duration_s = 10.0
        trim_note = "(Trimmed to 10s for UI)"

    start_idx = int(start_s * sr)
    end_idx = start_idx + int(duration_s * sr)
    y_slice = y[start_idx:end_idx]

    # Generate spectrogram
    plt.figure(figsize=(6, 3))
    D = librosa.stft(y_slice, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='linear', cmap='magma', vmin=-80, vmax=0)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    return y_slice, sr, plt.gcf(), trim_note

def render_clip_detail_panel(selected_clip, raw_dir, label_manager):
    st.subheader("Clip Details & Labelling")
    if selected_clip is None:
        st.write("No clip selected.")
        return

    row = selected_clip.iloc[0]
    clip_id = row['clip_id']

    st.write(f"**ID:** `{clip_id}`")
    st.write(f"**Source:** `{row['source_file']}` @ {row['start_time_seconds']}s")

    y, sr, fig, trim_note = get_audio_and_spec(row['source_file'], row['start_time_seconds'], row['duration_seconds'], raw_dir)

    if y is not None:
        if trim_note: st.warning(trim_note)
        st.audio(y, sample_rate=sr)
        st.pyplot(fig)
    else:
        st.error(fig) # error msg

    # Labelling Card
    st.divider()
    st.markdown("### Labelling")

    current_label_data = label_manager.get_latest_labels().get(clip_id, {})
    current_assigned = current_label_data.get("label", "None")

    st.write(f"**Ground Truth:** `{row['label']}`")
    st.write(f"**Current Label:** `{current_assigned}`")

    new_lbl = st.text_input("Assign Label", value=current_assigned if current_assigned != "None" else "")
    conf = st.radio("Confidence", ["high", "medium", "low"])
    notes = st.text_area("Notes")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Label"):
            label_manager.append_action("demo_user", "label", clip_id, new_lbl, conf, notes, [])
            st.success("Saved!")
            st.rerun()
    with col2:
        if st.button("Mark Uncertain"):
            label_manager.append_action("demo_user", "uncertain", clip_id, "uncertain", conf, notes, [])
            st.success("Marked uncertain!")
            st.rerun()

def render_physics_panel(selected_clip, df_merged, features_list):
    st.subheader("Physics Features")
    if selected_clip is None:
        st.write("No clip selected.")
        return

    row = selected_clip.iloc[0]
    c_id = row['cluster_id']

    c_mask = df_merged['cluster_id'] == c_id
    c_df = df_merged[c_mask]

    for feat in features_list:
        if feat not in df_merged.columns: continue

        val = row[feat]

        fig, ax = plt.subplots(figsize=(5, 1.5))
        sns.kdeplot(df_merged[feat].dropna(), color='grey', fill=True, alpha=0.2, ax=ax)
        if len(c_df) > 1:
            sns.kdeplot(c_df[feat].dropna(), color='blue', fill=True, alpha=0.5, ax=ax)

        ax.axvline(val, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{feat}: {val:.2f}", fontsize=10)
        ax.set_yticks([])
        ax.set_ylabel("")

        st.pyplot(fig)
        plt.close(fig)
