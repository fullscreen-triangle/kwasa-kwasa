# figure_2_v8_network.py
"""
Figure 2: V8 Intelligence Network Operational State
Network diagram with real execution metrics
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_2():
    """Generate Figure 2: V8 Intelligence Network"""

    viz = TurbulanceVisualizer()

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('V8 Intelligence Network: Operational State',
                 fontsize=16, fontweight='bold', pad=20)

    # BMD data (from your 03_bmd_demo_executable output)
    bmds = {
        'Mzekezeke': {
            'position': (2, 9),
            'subtitle': 'Bayesian Integration',
            'metrics': [
                'Status: ACTIVE',
                'Beliefs: 127',
                'Streams: 3',
                'Rate: 2.4 Hz'
            ],
            'color': COLORS['primary']
        },
        'Zengeza': {
            'position': (8, 9),
            'subtitle': 'Signal Enhancement',
            'metrics': [
                'Status: ACTIVE',
                'SNR: 3.8',
                'Features: 892',
                '(from 3421)'
            ],
            'color': COLORS['secondary']
        },
        'Diggiden': {
            'position': (14, 9),
            'subtitle': 'Adversarial Validation',
            'metrics': [
                'Status: ACTIVE',
                'Tests: 47/50',
                'Robust: 94%',
                'Vulns: 1'
            ],
            'color': COLORS['success']
        },
        'Spectacular': {
            'position': (2, 6),
            'subtitle': 'Anomaly Detection',
            'metrics': [
                'Status: ACTIVE',
                'Anomalies: 2',
                'Paradigm: 0.68',
                'Patterns: 3'
            ],
            'color': COLORS['warning']
        },
        'Hatata': {
            'position': (8, 6),
            'subtitle': 'Decision Optimization',
            'metrics': [
                'Status: ACTIVE',
                'Iteration: 23',
                'Pareto: 12',
                'Best: 0.87/0.91'
            ],
            'color': COLORS['danger']
        },
        'Champagne': {
            'position': (14, 6),
            'subtitle': 'Creative Synthesis',
            'metrics': [
                'Status: ACTIVE',
                'Ψ₀: 0.30',
                'Hypotheses: 156',
                'Plausible: 42'
            ],
            'color': COLORS['info']
        },
        'Nicotine': {
            'position': (5, 3),
            'subtitle': 'Memory Management',
            'metrics': [
                'Status: ACTIVE',
                'Memory: 892',
                'Context: 100',
                'Accuracy: 0.94'
            ],
            'color': COLORS['neutral']
        },
        'Pungwe': {
            'position': (11, 3),
            'subtitle': 'Authenticity Validation',
            'metrics': [
                'Status: ACTIVE',
                'Score: 0.96',
                'Test: PASSED',
                'Deception: NONE'
            ],
            'color': '#CC78BC'
        }
    }

    # Draw BMD nodes
    for name, data in bmds.items():
        x, y = data['position']

        # Node circle
        circle = Circle((x, y), 0.8,
                       edgecolor=data['color'],
                       facecolor=data['color'],
                       alpha=0.2,
                       linewidth=3)
        ax.add_patch(circle)

        # Node name
        ax.text(x, y + 0.3, name,
               ha='center', va='center',
               fontsize=11, fontweight='bold',
               color=data['color'])

        # Subtitle
        ax.text(x, y, data['subtitle'],
               ha='center', va='center',
               fontsize=8, style='italic',
               color=COLORS['dark'])

        # Metrics box
        metrics_y = y - 1.2
        for i, metric in enumerate(data['metrics']):
            ax.text(x, metrics_y - i*0.25, metric,
                   ha='center', va='center',
                   fontsize=7, family='monospace',
                   color=COLORS['dark'])

    # Draw connections (network topology)
    connections = [
        # Top row connections
        ((2, 9), (8, 9)),
        ((8, 9), (14, 9)),
        # Middle row connections
        ((2, 6), (8, 6)),
        ((8, 6), (14, 6)),
        # Bottom row connections
        ((5, 3), (11, 3)),
        # Vertical connections
        ((2, 9), (2, 6)),
        ((8, 9), (8, 6)),
        ((14, 9), (14, 6)),
        ((2, 6), (5, 3)),
        ((14, 6), (11, 3)),
        # Cross connections
        ((8, 6), (5, 3)),
        ((8, 6), (11, 3))
    ]

    for (x1, y1), (x2, y2) in connections:
        arrow = FancyArrowPatch((x1, y1-0.8), (x2, y2+0.8),
                               arrowstyle='-',
                               linewidth=1.5,
                               color=COLORS['light'],
                               alpha=0.6)
        ax.add_patch(arrow)

    # Consciousness metrics box (bottom)
    metrics_box = FancyBboxPatch((3, 0.2), 10, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor=COLORS['dark'],
                                facecolor='wheat',
                                alpha=0.3,
                                linewidth=2)
    ax.add_patch(metrics_box)

    ax.text(8, 1.5, 'Consciousness Metrics',
           ha='center', va='center',
           fontsize=11, fontweight='bold')

    metrics_text = [
        'Ψ₀ (Perception): 0.85',
        'Θ₀ (Thought): 0.72',
        'PLV: 0.81',
        'Hierarchical Depth: 0.90',
        'Consciousness Quality: 0.88'
    ]

    x_positions = [4, 6.5, 9, 11.5]
    for i, metric in enumerate(metrics_text[:4]):
        ax.text(x_positions[i], 0.9, metric,
               ha='center', va='center',
               fontsize=9, family='monospace')

    ax.text(8, 0.5, metrics_text[4],
           ha='center', va='center',
           fontsize=9, fontweight='bold', family='monospace')

    plt.tight_layout()
    viz.save_figure(fig, 'figure_2_v8_network')
    plt.show()

if __name__ == '__main__':
    create_figure_2()
