# visualization_toolkit.py
"""
Turbulance Visualization Toolkit
Generates publication-quality figures for hybrid symbolic-thermodynamic computing paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#0173B2',      # Blue
    'secondary': '#DE8F05',    # Orange
    'success': '#029E73',      # Green
    'warning': '#CC78BC',      # Purple
    'danger': '#CA9161',       # Brown
    'info': '#ECE133',         # Yellow
    'neutral': '#56B4E9',      # Light blue
    'dark': '#333333',         # Dark gray
    'light': '#CCCCCC'         # Light gray
}

class TurbulanceVisualizer:
    """Main visualization class for Turbulance execution outputs"""

    def __init__(self, output_dir: str = './figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_figure(self, fig, filename: str, formats: List[str] = ['pdf', 'png', 'svg']):
        """Save figure in multiple formats"""
        for fmt in formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
            print(f"Saved: {filepath}")
