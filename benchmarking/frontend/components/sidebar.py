"""Sidebar component."""

import streamlit as st


def render_sidebar():
    """Render the settings sidebar and return the results directory."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Directory selection
        results_dir = st.text_input(
            "Results Directory",
            value=".",
            help="Directory containing benchmark JSON results"
        )
        
        # Load results
        if st.button("🔄 Load Results", width="stretch"):
            st.session_state.results = None  # Clear cache to force reload
            st.rerun()
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        
        num_results = len(st.session_state.results) if st.session_state.results else 0
        st.info(f"**Loaded:** {num_results} result(s)")
        
        return results_dir