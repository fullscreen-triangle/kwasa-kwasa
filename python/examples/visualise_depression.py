# visualize_depression_treatment.py
"""
Create comprehensive visualizations from parsed Turbulance files
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Import our visualization toolkit
import sys
sys.path.append('.')
from visualization_toolkit import TurbulanceVisualizer, COLORS

class DepressionTreatmentVisualizer:
    """Visualize depression treatment data"""

    def __init__(self, data_file: str = 'depression_treatment_parsed.json'):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.viz = TurbulanceVisualizer(output_dir='./figures')

    def create_overview_figure(self):
        """Create comprehensive overview figure"""
        fig = plt.figure(figsize=(20, 16))

        # Title
        fig.suptitle('Turbulance Depression Treatment: Complete System Overview',
                    fontsize=18, fontweight='bold', y=0.98)

        # Create 4 main panels (one for each file type)

        # Panel 1: .trb Protocol (top-left)
        ax1 = plt.subplot(2, 2, 1)
        self._plot_trb_panel(ax1)

        # Panel 2: .fs Flux State (top-right)
        ax2 = plt.subplot(2, 2, 2)
        self._plot_fs_panel(ax2)

        # Panel 3: .ghd Dependencies (bottom-left)
        ax3 = plt.subplot(2, 2, 3)
        self._plot_ghd_panel(ax3)

        # Panel 4: .hre Decision Log (bottom-right)
        ax4 = plt.subplot(2, 2, 4)
        self._plot_hre_panel(ax4)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        self.viz.save_figure(fig, 'depression_treatment_overview')
        plt.show()

    def _plot_trb_panel(self, ax):
        """Plot .trb protocol information"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('.trb - Protocol Specification',
                    fontsize=14, fontweight='bold',
                    color=COLORS['primary'], pad=10)

        trb_data = self.data.get('trb', {})

        # Background box
        box = FancyBboxPatch((0.5, 0.5), 9, 9,
                            boxstyle="round,pad=0.2",
                            edgecolor=COLORS['primary'],
                            facecolor=COLORS['primary'],
                            alpha=0.1,
                            linewidth=3)
        ax.add_patch(box)

        y_pos = 9

        # Documentation
        if 'documentation' in trb_data:
            ax.text(1, y_pos, 'Documentation:',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5
            for doc in trb_data['documentation'][:3]:
                ax.text(1.5, y_pos, f'• {doc[:60]}...',
                       fontsize=8, wrap=True)
                y_pos -= 0.4

        y_pos -= 0.5

        # Imports
        if 'imports' in trb_data:
            ax.text(1, y_pos, f'Imports ({len(trb_data["imports"])}):',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5
            for imp in trb_data['imports'][:6]:
                ax.text(1.5, y_pos, f'• {imp}',
                       fontsize=8, family='monospace')
                y_pos -= 0.35

        y_pos -= 0.5

        # Consciousness parameters
        if 'consciousness_params' in trb_data:
            params = trb_data['consciousness_params']
            ax.text(1, y_pos, 'Consciousness Parameters:',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5

            if params.get('h_plus_frequency_thz'):
                ax.text(1.5, y_pos, f'• H⁺ frequency: {params["h_plus_frequency_thz"]} THz',
                       fontsize=9, fontweight='bold', color=COLORS['secondary'])
                y_pos -= 0.35

            if params.get('theta_band_hz'):
                ax.text(1.5, y_pos, f'• θ-band: {params["theta_band_hz"][0]}-{params["theta_band_hz"][1]} Hz',
                       fontsize=9)
                y_pos -= 0.35

            if params.get('gamma_band_hz'):
                ax.text(1.5, y_pos, f'• γ-band: {params["gamma_band_hz"][0]}-{params["gamma_band_hz"][1]} Hz',
                       fontsize=9)

    def _plot_fs_panel(self, ax):
        """Plot .fs flux state information"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('.fs - Real-Time Consciousness Monitoring',
                    fontsize=14, fontweight='bold',
                    color=COLORS['success'], pad=10)

        fs_data = self.data.get('fs', {})

        # Background box
        box = FancyBboxPatch((0.5, 0.5), 9, 9,
                            boxstyle="round,pad=0.2",
                            edgecolor=COLORS['success'],
                            facecolor=COLORS['success'],
                            alpha=0.1,
                            linewidth=3)
        ax.add_patch(box)

        y_pos = 9

        # Consciousness metrics (most important)
        if 'consciousness_metrics' in fs_data:
            metrics = fs_data['consciousness_metrics']
            ax.text(5, y_pos, 'Consciousness Metrics',
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            y_pos -= 0.7

            for key, value in metrics.items():
                # Create bar
                bar_width = value * 6
                bar = FancyBboxPatch((2, y_pos - 0.15), bar_width, 0.25,
                                    boxstyle="round,pad=0.02",
                                    edgecolor=COLORS['success'],
                                    facecolor=COLORS['success'],
                                    alpha=0.6)
                ax.add_patch(bar)

                # Label
                ax.text(1, y_pos, f'{key}:',
                       fontsize=9, ha='right', va='center')
                ax.text(8.5, y_pos, f'{value:.2f}',
                       fontsize=9, fontweight='bold', ha='right', va='center',
                       family='monospace')
                y_pos -= 0.45

        y_pos -= 0.5

        # V8 modules status
        if 'v8_modules' in fs_data:
            modules = fs_data['v8_modules']
            ax.text(1, y_pos, f'V8 Modules ({len(modules)} active):',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5

            for module_name, module_data in list(modules.items())[:4]:
                status = module_data.get('status', 'UNKNOWN')
                color = COLORS['success'] if status == 'ACTIVE' else COLORS['danger']
                ax.text(1.5, y_pos, f'• {module_name}: {status}',
                       fontsize=8, color=color, fontweight='bold')
                y_pos -= 0.35

    def _plot_ghd_panel(self, ax):
        """Plot .ghd dependencies information"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('.ghd - Resource Dependencies',
                    fontsize=14, fontweight='bold',
                    color=COLORS['warning'], pad=10)

        ghd_data = self.data.get('ghd', {})

        # Background box
        box = FancyBboxPatch((0.5, 0.5), 9, 9,
                            boxstyle="round,pad=0.2",
                            edgecolor=COLORS['warning'],
                            facecolor=COLORS['warning'],
                            alpha=0.1,
                            linewidth=3)
        ax.add_patch(box)

        y_pos = 9

        # External databases
        if 'databases' in ghd_data:
            databases = ghd_data['databases']
            ax.text(1, y_pos, f'External Databases ({len(databases)}):',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5

            for db_name in list(databases.keys())[:6]:
                ax.text(1.5, y_pos, f'• {db_name}',
                       fontsize=8, family='monospace')
                y_pos -= 0.35

        y_pos -= 0.5

        # Computational resources
        if 'resources' in ghd_data:
            resources = ghd_data['resources']
            ax.text(1, y_pos, 'Computational Resources:',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5

            for key, value in resources.items():
                ax.text(1.5, y_pos, f'• {key}: {value}',
                       fontsize=9, fontweight='bold', color=COLORS['secondary'])
                y_pos -= 0.35

    def _plot_hre_panel(self, ax):
        """Plot .hre decision log information"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('.hre - Metacognitive Decision Log',
                    fontsize=14, fontweight='bold',
                    color=COLORS['danger'], pad=10)

        hre_data = self.data.get('hre', {})

        # Background box
        box = FancyBboxPatch((0.5, 0.5), 9, 9,
                            boxstyle="round,pad=0.2",
                            edgecolor=COLORS['danger'],
                            facecolor=COLORS['danger'],
                            alpha=0.1,
                            linewidth=3)
        ax.add_patch(box)

        y_pos = 9

        # Session info
        if 'session' in hre_data:
            ax.text(5, y_pos, hre_data['session'],
                   fontsize=9, ha='center', style='italic',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            y_pos -= 0.7

        # Scientific hypothesis
        if 'hypothesis' in hre_data:
            hypothesis = hre_data['hypothesis']
            ax.text(1, y_pos, 'Scientific Hypothesis:',
                   fontsize=10, fontweight='bold')
            y_pos -= 0.4

            # Wrap hypothesis text
            words = hypothesis.split()
            line = ""
            for word in words:
                if len(line + word) < 50:
                    line += word + " "
                else:
                    ax.text(1.5, y_pos, line, fontsize=8, style='italic')
                    y_pos -= 0.3
                    line = word + " "
            if line:
                ax.text(1.5, y_pos, line, fontsize=8, style='italic')
                y_pos -= 0.3

        y_pos -= 0.5

        # Decision phases
        if 'decisions' in hre_data:
            decisions = hre_data['decisions']
            ax.text(1, y_pos, f'Decision Phases ({len(decisions)}):',
                   fontsize=11, fontweight='bold')
            y_pos -= 0.5

            for i, decision in enumerate(decisions[:3]):
                ax.text(1.5, y_pos, f'{i+1}. {decision["phase"]}',
                       fontsize=8, fontweight='bold')
                y_pos -= 0.25
                ax.text(2, y_pos, f'   {decision["decision"][:45]}...',
                       fontsize=7, style='italic')
                y_pos -= 0.4

# Run the visualizer
if __name__ == '__main__':
    viz = DepressionTreatmentVisualizer()
    viz.create_overview_figure()
