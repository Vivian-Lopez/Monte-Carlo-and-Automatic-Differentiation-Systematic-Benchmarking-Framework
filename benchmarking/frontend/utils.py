"""Utility functions for dashboard."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st


def load_json_results(directory: str) -> List[Dict[str, Any]]:
    """Load all benchmark JSON results from a directory."""
    results = []
    if not os.path.exists(directory):
        return results
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data['_filename'] = filename
                    results.append(data)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")
    
    return results


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of result dicts to a pandas DataFrame for easier analysis."""
    data = []
    for r in results:
        try:
            config = r.get('config', {})
            stats = r.get('statistics', {})
            metadata = r.get('metadata', {})
            
            data.append({
                'Filename': r.get('_filename', 'unknown'),
                'Config Hash': r.get('config_hash', 'N/A'),
                'AD Mode': r.get('ad_mode', 'none'),
                'Price': r.get('result', float('nan')),
                'M (paths)': config.get('M', 0),
                'Mean Runtime (ms)': stats.get('mean_runtime', 0) * 1000,
                'Std Dev (ms)': stats.get('std_runtime', 0) * 1000,
                'Min Runtime (ms)': stats.get('min_runtime', 0) * 1000,
                'Max Runtime (ms)': stats.get('max_runtime', 0) * 1000,
                'S0': config.get('S0', 0),
                'K': config.get('K', 0),
                'Volatility': config.get('sigma', 0),
                'Rate': config.get('r', 0),
                'Timestamp': metadata.get('timestamp', 'N/A'),
                'Python Version': metadata.get('python_version', 'N/A'),
                'Platform': metadata.get('platform', 'N/A'),
            })
        except Exception as e:
            st.warning(f"Error processing result: {e}")
    
    return pd.DataFrame(data)