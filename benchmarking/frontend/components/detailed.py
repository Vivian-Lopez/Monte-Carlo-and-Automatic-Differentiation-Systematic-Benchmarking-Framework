"""Detailed view tab component."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_detailed(df: pd.DataFrame, results: list):
    """Render the detailed view tab for single-run inspection."""
    st.header("Detailed Results")
    
    # Select a specific run
    selected_file = st.selectbox(
        "Select a result file to view:",
        options=df['Filename'].tolist(),
        index=0 if len(df) > 0 else None
    )
    
    if selected_file:
        # Find the matching result
        result = next(
            (r for r in results if r.get('_filename') == selected_file),
            None
        )
        
        if result:
            st.subheader(f"Run: {selected_file}")
            
            # Configuration
            with st.expander("📋 Configuration", expanded=True):
                config = result.get('config', {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**S0 (Stock):** ${config.get('S0', 'N/A')}")
                    st.write(f"**K (Strike):** ${config.get('K', 'N/A')}")
                    st.write(f"**R (Rate):** {config.get('r', 'N/A'):.2%}")
                
                with col2:
                    st.write(f"**σ (Volatility):** {config.get('sigma', 'N/A'):.2%}")
                    st.write(f"**T (Time):** {config.get('T', 'N/A')} years")
                    st.write(f"**M (Paths):** {config.get('M', 'N/A'):,}")
                
                with col3:
                    st.write(f"**N (Steps):** {config.get('N', 'N/A')}")
                    st.write(f"**Seed:** {config.get('seed', 'N/A')}")
                    st.write(f"**Config Hash:** `{result.get('config_hash', 'N/A')}`")
            
            # Results
            with st.expander("✅ Results", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Estimated Price",
                        f"${result.get('result', 'N/A'):.6f}"
                    )
                
                with col2:
                    st.metric(
                        "AD Mode",
                        result.get('ad_mode', 'none')
                    )
                
                with col3:
                    st.metric(
                        "Number of Runs",
                        len(result.get('runtimes', []))
                    )
            
            # Statistics
            with st.expander("📊 Statistics", expanded=True):
                stats = result.get('statistics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Mean Runtime",
                        f"{stats.get('mean_runtime', 0)*1000:.4f} ms"
                    )
                
                with col2:
                    st.metric(
                        "Std Dev",
                        f"{stats.get('std_runtime', 0)*1000:.4f} ms"
                    )
                
                with col3:
                    st.metric(
                        "Min Runtime",
                        f"{stats.get('min_runtime', 0)*1000:.4f} ms"
                    )
                
                with col4:
                    st.metric(
                        "Max Runtime",
                        f"{stats.get('max_runtime', 0)*1000:.4f} ms"
                    )
                
                # Individual run times
                st.write("**Individual Run Times (ms):**")
                runtimes_ms = [rt * 1000 for rt in result.get('runtimes', [])]
                st.write(", ".join([f"{rt:.4f}" for rt in runtimes_ms]))
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Run {i+1}" for i in range(len(runtimes_ms))],
                    y=runtimes_ms,
                    marker=dict(color='steelblue'),
                    name='Runtime'
                ))
                fig.add_hline(
                    y=stats.get('mean_runtime', 0)*1000,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Mean"
                )
                fig.update_layout(
                    height=400,
                    title="Individual Run Times",
                    xaxis_title="Run",
                    yaxis_title="Runtime (ms)"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Environment
            with st.expander("🖥️ Environment", expanded=False):
                metadata = result.get('metadata', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
                    st.write(f"**Python:** {metadata.get('python_version', 'N/A')}")
                    st.write(f"**Implementation:** {metadata.get('python_implementation', 'N/A')}")
                
                with col2:
                    st.write(f"**Platform:** {metadata.get('platform', 'N/A')}")
                    st.write(f"**Processor:** {metadata.get('processor', 'N/A')}")
                    st.write(f"**Framework:** {metadata.get('framework_version', 'N/A')}")
            
            # Raw JSON
            with st.expander("📝 Raw JSON", expanded=False):
                st.json(result)