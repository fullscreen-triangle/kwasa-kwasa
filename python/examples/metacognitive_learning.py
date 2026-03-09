# figure_6_metacognitive_learning.py
"""
Figure 6: Metacognitive Learning Through Harare Decision Logs
Timeline showing decision evolution with certainty tracking
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_6():
    """Generate Figure 6: Metacognitive Learning"""

    viz = TurbulanceVisualizer()

    fig = plt.figure(figsize=(16, 14))

    # Create main axis for timeline
    ax_timeline = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax_timeline.set_xlim(0, 16)
    ax_timeline.set_ylim(0, 12)
    ax_timeline.axis('off')
    ax_timeline.set_title('Metacognitive Decision Timeline (from .hre Harare Decision Log)',
                         fontsize=14, fontweight='bold')

    # Decision timeline data
    decisions = [
        {
            'time': 't=0s',
            'y': 10.5,
            'decision': 'use_standard_biomarkers',
            'reasoning': 'literature suggests\nglucose, HbA1c, insulin',
            'certainty': 0.60,
            'result': 'loaded 23 known\nbiomarkers',
            'color': COLORS['primary']
        },
        {
            'time': 't=342s',
            'y': 8.5,
            'decision': 'investigate_novel_\nmetabolite_pattern',
            'reasoning': 'spectacular detected\n3-hydroxybutyrate deriv',
            'certainty': 0.40,
            'result': '12 hypotheses\ngenerated',
            'color': COLORS['secondary']
        },
        {
            'time': 't=567s',
            'y': 6.5,
            'decision': 'relax_biological_\nplausibility_constraints',
            'reasoning': 'champagne suggests\nunconventional pathways',
            'certainty': 0.30,
            'result': 'discovered BCAA\nratio hypothesis',
            'color': COLORS['info']
        },
        {
            'time': 't=891s',
            'y': 4.5,
            'decision': 'test_against_age_\nconfounding',
            'reasoning': 'diggiden detected\npotential age confound',
            'certainty': 0.80,
            'result': 'confounding confirmed,\ncorrected model',
            'color': COLORS['warning']
        },
        {
            'time': 't=1234s',
            'y': 2.5,
            'decision': 'validate_genuine_\nunderstanding',
            'reasoning': 'pungwe reconstruction\ntest',
            'certainty': 0.90,
            'result': 'PASSED\nauthenticity=0.96',
            'color': COLORS['success']
        },
        {
            'time': 't=1567s',
            'y': 0.5,
            'decision': 'select_3_metabolite_\npanel',
            'reasoning': 'hatata pareto\noptimization suggests 3',
            'certainty': 0.85,
            'result': 'sensitivity=0.87,\nspecificity=0.91',
            'color': COLORS['danger']
        }
    ]

    # Draw decision boxes and connections
    for i, dec in enumerate(decisions):
        # Decision box
        box = FancyBboxPatch((1, dec['y']), 6, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor=dec['color'],
                            facecolor=dec['color'],
                            alpha=0.2,
                            linewidth=2)
        ax_timeline.add_patch(box)

        # Time label
        ax_timeline.text(1.5, dec['y'] + 1.3, dec['time'],
                        ha='left', va='center', fontsize=9,
                        fontweight='bold', color=dec['color'])

        # Decision
        ax_timeline.text(1.5, dec['y'] + 0.95, f"Decision: {dec['decision']}",
                        ha='left', va='center', fontsize=8, fontweight='bold')

        # Reasoning
        ax_timeline.text(1.5, dec['y'] + 0.55, f"Reasoning: {dec['reasoning']}",
                        ha='left', va='center', fontsize=7)

        # Certainty
        certainty_text = f"Certainty: {dec['certainty']:.2f}"
        ax_timeline.text(1.5, dec['y'] + 0.15, certainty_text,
                        ha='left', va='center', fontsize=7,
                        fontweight='bold', family='monospace')

        # Result box
        result_box = FancyBboxPatch((8, dec['y'] + 0.2), 3.5, 1.1,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=dec['color'],
                                   facecolor='white',
                                   linewidth=1.5)
        ax_timeline.add_patch(result_box)

        ax_timeline.text(9.75, dec['y'] + 0.95, "Result:",
                        ha='center', va='center', fontsize=7,
                        fontweight='bold')
        ax_timeline.text(9.75, dec['y'] + 0.55, dec['result'],
                        ha='center', va='center', fontsize=6.5)

        # Arrow to next decision
        if i < len(decisions) - 1:
            arrow = FancyArrowPatch((4, dec['y']),
                                   (4, decisions[i+1]['y'] + 1.5),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color=COLORS['dark'],
                                   alpha=0.5)
            ax_timeline.add_patch(arrow)

            # Annotation for key events
            if i == 0:
                ax_timeline.text(4.5, (dec['y'] + decisions[i+1]['y'] + 1.5) / 2,
                               'Spectacular\ndetects anomaly',
                               ha='left', va='center', fontsize=7,
                               style='italic', color=COLORS['secondary'])
            elif i == 1:
                ax_timeline.text(4.5, (dec['y'] + decisions[i+1]['y'] + 1.5) / 2,
                               'Champagne\ndream-state',
                               ha='left', va='center', fontsize=7,
                               style='italic', color=COLORS['info'])
            elif i == 2:
                ax_timeline.text(4.5, (dec['y'] + decisions[i+1]['y'] + 1.5) / 2,
                               'Diggiden\nadversarial test',
                               ha='left', va='center', fontsize=7,
                               style='italic', color=COLORS['warning'])
            elif i == 3:
                ax_timeline.text(4.5, (dec['y'] + decisions[i+1]['y'] + 1.5) / 2,
                               'Pungwe\nauthenticity',
                               ha='left', va='center', fontsize=7,
                               style='italic', color=COLORS['success'])

    # Certainty evolution graph
    ax_certainty = plt.subplot2grid((3, 1), (2, 0))

    times = [0, 342, 567, 891, 1234, 1567]
    certainties = [0.60, 0.40, 0.30, 0.80, 0.90, 0.85]

    ax_certainty.plot(times, certainties, 'o-', linewidth=2.5,
                     markersize=10, color=COLORS['primary'],
                     markerfacecolor=COLORS['primary'],
                     markeredgecolor='white', markeredgewidth=2)

    # Highlight self-correction events
    ax_certainty.axvline(x=891, color=COLORS['warning'],
                        linestyle='--', linewidth=2, alpha=0.7,
                        label='Age confounding corrected')
    ax_certainty.axvline(x=1234, color=COLORS['success'],
                        linestyle='--', linewidth=2, alpha=0.7,
                        label='Authenticity validated')

    ax_certainty.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax_certainty.set_ylabel('Certainty', fontsize=11, fontweight='bold')
    ax_certainty.set_title('Certainty Evolution Over Time',
                          fontsize=12, fontweight='bold')
    ax_certainty.set_ylim(0, 1.0)
    ax_certainty.grid(True, alpha=0.3)
    ax_certainty.legend(loc='lower right', framealpha=0.9)

    # Add annotations for key insights
    ax_certainty.annotate('Exploratory mode\n(Champagne)',
                         xy=(567, 0.30), xytext=(400, 0.15),
                         fontsize=8, style='italic',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', lw=1.5))

    ax_certainty.annotate('Self-correction\nincreases confidence',
                         xy=(891, 0.80), xytext=(700, 0.65),
                         fontsize=8, style='italic',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.tight_layout()
    viz.save_figure(fig, 'figure_6_metacognitive_learning')
    plt.show()

if __name__ == '__main__':
    create_figure_6()
