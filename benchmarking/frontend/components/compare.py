"""Compare runs tab component."""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_compare(df: pd.DataFrame):
    """Render the compare tab for multi-run analysis."""
    st.header("Run Comparison")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ad_modes = df['AD Mode'].unique().tolist()
        selected_ad = st.multiselect(
            "AD Mode",
            ad_modes,
            default=ad_modes,
            help="Filter by AD mode"
        )
    
    with col2:
        configs = df['Config Hash'].unique().tolist()
        selected_configs = st.multiselect(
            "Config Hash",
            configs,
            default=configs,
            help="Filter by configuration"
        )
    
    with col3:
        path_counts = sorted(df['M (paths)'].unique().tolist())
        selected_paths = st.multiselect(
            "Path Count",
            path_counts,
            default=path_counts,
            help="Filter by number of MC paths"
        )
    
    # Apply filters
    filtered_df = df[
        (df['AD Mode'].isin(selected_ad)) &
        (df['Config Hash'].isin(selected_configs)) &
        (df['M (paths)'].isin(selected_paths))
    ]
    
    if len(filtered_df) == 0:
        st.warning("No results match the selected filters.")
        return
    
    st.divider()
    
    if len(filtered_df) == 1:
        st.info("💡 Load multiple results to enable comparison visualizations.")
        st.subheader("Single Result Summary")
        display_cols = [
            'Filename', 'Config Hash', 'AD Mode', 'M (paths)',
            'Price', 'Mean Runtime (ms)', 'Std Dev (ms)',
            'Timestamp'
        ]
        st.dataframe(
            filtered_df[display_cols],
            width='stretch',
            hide_index=True
        )
    else:
        # Multi-run comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='M (paths)',
                y='Mean Runtime (ms)',
                color='AD Mode',
                size='Std Dev (ms)',
                hover_data=['Config Hash', 'Std Dev (ms)', 'Price'],
                title='Runtime vs. Path Count',
                labels={
                    'M (paths)': 'Number of MC Paths',
                    'Mean Runtime (ms)': 'Runtime (ms)'
                }
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='M (paths)',
                y='Price',
                color='AD Mode',
                size='Mean Runtime (ms)',
                hover_data=['Config Hash', 'Mean Runtime (ms)'],
                title='Estimated Price vs. Path Count',
                labels={
                    'M (paths)': 'Number of MC Paths',
                    'Price': 'Option Price ($)'
                }
            )
            st.plotly_chart(fig, width='stretch')
        
        # Comparison table
        st.subheader("Detailed Comparison Table")
        display_cols = [
            'Filename', 'Config Hash', 'AD Mode', 'M (paths)',
            'Price', 'Mean Runtime (ms)', 'Std Dev (ms)',
            'Timestamp'
        ]
        st.dataframe(
            filtered_df[display_cols],
            width='stretch',
            hide_index=True
        )