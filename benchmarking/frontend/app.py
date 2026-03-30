"""
Streamlit dashboard for Monte Carlo benchmarking results.

Modular design with components for overview, comparison, and detailed analysis.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from benchmarking.frontend.utils import load_json_results, results_to_dataframe
from benchmarking.frontend.components.sidebar import render_sidebar
from benchmarking.frontend.components.overview import render_overview
from benchmarking.frontend.components.compare import render_compare
from benchmarking.frontend.components.detailed import render_detailed
from benchmarking.frontend.components.ad_analysis import render_ad_analysis

# Page config
st.set_page_config(
    page_title="MC Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Monte Carlo Benchmarking Dashboard")
st.markdown("Visualization and analysis of MC+AD benchmark results")

# ============================================================================
# Sidebar
# ============================================================================

results_dir = render_sidebar()

# ============================================================================
# Main Dashboard
# ============================================================================

# Load results on button click via sidebar
if 'results' not in st.session_state:
    st.session_state.results = []

# Check if we need to load
if results_dir and st.session_state.results is None:
    st.session_state.results = load_json_results(results_dir)

if not st.session_state.results:
    st.warning("📂 No results loaded. Click 'Load Results' in the sidebar to get started.")
    st.info("""
    **How to use:**
    1. Ensure benchmark results are saved as JSON files (from `BenchmarkRunner.save_results()`)
    2. Set the results directory path in the sidebar
    3. Click 'Load Results'
    4. Explore visualizations below
    """)
else:
    # Convert to DataFrame
    df = results_to_dataframe(st.session_state.results)
    
    # Tabs
    tab_overview, tab_ad, tab_compare, tab_detailed = st.tabs([
        "📈 Overview",
        "🔬 AD Analysis",
        "🔍 Compare Runs",
        "📋 Detailed View"
    ])
    
    with tab_overview:
        render_overview(df)
    
    with tab_ad:
        render_ad_analysis(st.session_state.results)
    
    with tab_compare:
        render_compare(df)
    
    with tab_detailed:
        render_detailed(df, st.session_state.results)


st.divider()
st.caption("Monte Carlo Benchmarking Dashboard • Powered by Streamlit")