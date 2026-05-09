import streamlit as st

def init_session_state():
    if "selected_clip_id" not in st.session_state:
        st.session_state.selected_clip_id = None

    if "selected_cluster_id" not in st.session_state:
        st.session_state.selected_cluster_id = None

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Evaluation View"

    if "feature_panels" not in st.session_state:
        st.session_state.feature_panels = 1

def handle_plotly_click(event_dict):
    """
    Parses native Streamlit 1.35+ Plotly selection events and updates st.session_state.
    """
    if event_dict and "selection" in event_dict:
        points = event_dict["selection"].get("points", [])
        if points:
            # We assume customdata contains clip_id
            clip_id = points[0].get("customdata", [None])[0]
            if clip_id and clip_id != st.session_state.selected_clip_id:
                st.session_state.selected_clip_id = clip_id
                st.rerun()
