"""
Streamlit component for automatic differentiation metrics visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any


def render_ad_analysis(results: List[Dict[str, Any]]) -> None:
    """
    Render AD metrics analysis dashboard.
    
    Shows:
    - AD overhead ratio by mode (forward, reverse, none)
    - Gradient computation time breakdown
    - Memory overhead comparison
    - Wall-clock time stacked bar chart
    """
    st.header("Automatic Differentiation (AD) Analysis")
    
    # Filter results by ad_mode
    ad_modes = set(r.get("ad_mode", "none") for r in results)
    
    if "none" not in ad_modes or len(ad_modes) == 1:
        st.warning("⚠ No AD results found. Run mixed-mode experiment to enable this analysis.")
        return
    
    # Create metrics dataframe
    df_rows = []
    for result in results:
        ad_mode = result.get("ad_mode", "none")
        ad_metrics = result.get("ad_metrics", {})
        config = result.get("config", {})
        stats = result.get("statistics", {})
        
        df_rows.append({
            "Mode": ad_mode.capitalize(),
            "Engine": result.get("metadata", {}).get("engine_name", "Unknown"),
            "Mean Runtime (ms)": stats.get("mean_runtime", 0) * 1000,
            "Overhead Ratio": ad_metrics.get("ad_overhead_ratio", 1.0),
            "Gradient Time (ms)": ad_metrics.get("gradient_time_ms", 0),
            "Memory Peak (MB)": ad_metrics.get("memory_peak_mb", 0),
            "Accuracy Error": ad_metrics.get("ad_accuracy_error", 0),
        })
    
    df = pd.DataFrame(df_rows)
    
    if df.empty:
        st.info("No AD metrics available in current results.")
        return
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overhead Ratio", "Wall-Clock Times", "Memory & Accuracy", "Summary Table"]
    )
    
    with tab1:
        st.subheader("AD Overhead Ratio (Relative to Baseline)")
        
        # Filter to AD modes only
        df_ad = df[df["Mode"] != "None"].copy() if "None" in df["Mode"].values else df
        
        if not df_ad.empty:
            fig = px.bar(
                df_ad,
                x="Mode",
                y="Overhead Ratio",
                color="Mode",
                title="AD Overhead Comparison",
                labels={"Overhead Ratio": "Overhead Multiplier"},
                height=400
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Baseline (1x)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                fwd = df_ad[df_ad["Mode"] == "Forward"]["Overhead Ratio"].values
                if len(fwd) > 0:
                    st.metric("Forward Mode Overhead", f"{fwd[0]:.2f}x")
            with col2:
                rev = df_ad[df_ad["Mode"] == "Reverse"]["Overhead Ratio"].values
                if len(rev) > 0:
                    st.metric("Reverse Mode Overhead", f"{rev[0]:.2f}x")
            with col3:
                none = df_ad[df_ad["Mode"] == "None"]["Overhead Ratio"].values
                if len(none) > 0:
                    st.metric("Baseline Overhead", f"{none[0]:.2f}x")
    
    with tab2:
        st.subheader("Wall-Clock Time Comparison")
        
        # Stacked bar chart (execution + gradient time)
        df_times = df.copy()
        df_times["Simulation Time (ms)"] = (
            df_times["Mean Runtime (ms)"] - df_times["Gradient Time (ms)"]
        )
        df_times = df_times[["Mode", "Simulation Time (ms)", "Gradient Time (ms)"]]
        
        fig = go.Figure(
            data=[
                go.Bar(x=df_times["Mode"], y=df_times["Simulation Time (ms)"], 
                       name="Simulation", marker_color="lightblue"),
                go.Bar(x=df_times["Mode"], y=df_times["Gradient Time (ms)"], 
                       name="Gradient", marker_color="darkorange"),
            ]
        )
        fig.update_layout(
            barmode="stack",
            title="Time Breakdown: Simulation vs. Gradient Computation",
            xaxis_title="AD Mode",
            yaxis_title="Time (ms)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Memory & Numerical Accuracy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "Memory Peak (MB)" in df.columns and df["Memory Peak (MB)"].sum() > 0:
                fig = px.bar(
                    df,
                    x="Mode",
                    y="Memory Peak (MB)",
                    color="Mode",
                    title="Peak Memory Usage",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Memory profiling data not available. Enable memory tracking in run().")
        
        with col2:
            if "Accuracy Error" in df.columns and df["Accuracy Error"].sum() > 0:
                df_acc = df[df["Accuracy Error"] > 0].copy()
                if not df_acc.empty:
                    fig = px.bar(
                        df_acc,
                        x="Mode",
                        y="Accuracy Error",
                        color="Mode",
                        title="Gradient Accuracy (Error vs. Analytical)",
                        height=350,
                        labels={"Accuracy Error": "Relative Error"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Gradient accuracy data not available. Run validation in experiment.")
            else:
                st.info("Gradient accuracy data not available.")
    
    with tab4:
        st.subheader("AD Metrics Summary Table")
        
        # Display full metrics table
        display_df = df[[
            "Mode", "Mean Runtime (ms)", "Overhead Ratio", 
            "Gradient Time (ms)", "Memory Peak (MB)"
        ]].copy()
        
        st.dataframe(
            display_df.style.format({
                "Mean Runtime (ms)": "{:.3f}",
                "Overhead Ratio": "{:.2f}",
                "Gradient Time (ms)": "{:.2f}",
                "Memory Peak (MB)": "{:.2f}",
            }),
            use_container_width=True
        )
        
        # Key insights
        st.markdown("**Key Insights:**")
        
        if "Forward" in df["Mode"].values and "Reverse" in df["Mode"].values:
            fwd_oh = df[df["Mode"] == "Forward"]["Overhead Ratio"].values[0]
            rev_oh = df[df["Mode"] == "Reverse"]["Overhead Ratio"].values[0]
            
            if fwd_oh < rev_oh:
                st.success(f"✓ Forward mode is {rev_oh/fwd_oh:.2f}x faster than reverse mode")
            else:
                st.info(f"ℹ Reverse mode is {fwd_oh/rev_oh:.2f}x faster than forward mode")
        
        baseline_time = df[df["Mode"] == "None"]["Mean Runtime (ms)"].values
        if len(baseline_time) > 0:
            baseline = baseline_time[0]
            st.info(f"Baseline (no AD) runtime: {baseline:.3f} ms")
