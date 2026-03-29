"""Overview tab component."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_overview(df: pd.DataFrame):
    """Render the overview tab with summary statistics and visualizations."""
    st.header("Benchmark Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Runs",
            len(df),
            help="Number of benchmark runs loaded"
        )
    
    with col2:
        mean_price = df['Price'].mean()
        st.metric(
            "Mean Estimated Price",
            f"${mean_price:.4f}",
            help="Average option price across all runs"
        )
    
    with col3:
        mean_runtime = df['Mean Runtime (ms)'].mean()
        st.metric(
            "Mean Runtime",
            f"{mean_runtime:.2f} ms",
            help="Average runtime across all runs"
        )
    
    with col4:
        throughput = (df['M (paths)'].mean() / (df['Mean Runtime (ms)'].mean() / 1000)) / 1e6
        st.metric(
            "Throughput",
            f"{throughput:.2f}M paths/s",
            help="Average throughput (paths per second)"
        )
    
    st.divider()
    
    # Runtime distribution (only if multiple runs)
    if len(df) > 1:
        st.subheader("Runtime Statistics")
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df['Mean Runtime (ms)'],
            name="Mean Runtime",
            marker=dict(color='steelblue')
        ))
        fig.add_trace(go.Box(
            y=df['Std Dev (ms)'],
            name="Std Dev",
            marker=dict(color='orange')
        ))
        fig.update_layout(
            height=400,
            title="Runtime Distribution Across Runs",
            yaxis_title="Time (ms)"
        )
        st.plotly_chart(fig, width='stretch')
    
    # Price distribution
    st.subheader("Estimated Price Analysis")
    
    if len(df) > 1:
        fig = px.histogram(
            df,
            x='Price',
            nbins=min(10, max(3, len(df))),
            title="Distribution of Estimated Option Prices",
            labels={'Price': 'Option Price ($)'},
            marginal='box'
        )
    else:
        # Single result: show as a simple metric display
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=['Price'],
            y=[df['Price'].iloc[0]],
            mode='markers',
            marker=dict(size=20, color='steelblue'),
            text=[f"${df['Price'].iloc[0]:.6f}"],
            textposition="top center",
            hovertemplate="<b>Estimated Price</b><br>$%{y:.6f}<extra></extra>"
        ))
        fig.update_layout(
            height=300,
            title="Estimated Option Price",
            xaxis=dict(showticklabels=False),
            yaxis_title="Option Price ($)",
            showlegend=False
        )
    
    st.plotly_chart(fig, width='stretch')