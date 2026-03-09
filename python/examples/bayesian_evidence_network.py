# figure_3_bayesian_integration.py
"""
Figure 3: Bayesian Evidence Integration in Clinical Diagnosis
Three-panel flow: Theory → Execution → Clinical Application
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_3():
    """Generate Figure 3: Bayesian Evidence Integration"""

    viz = TurbulanceVisualizer()

    fig = plt.figure(figsize=(18, 6))

    # Panel A: Mathematical Formulation
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('A. Mathematical Formulation',
                  fontsize=12, fontweight='bold', loc='left')

    # Prior
    ax1.text(5, 11, 'Prior: P(H) = 0.50',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.5))

    # Evidence FOR
    y_pos = 9.5
    ax1.text(5, y_pos, 'Evidence FOR:',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=COLORS['success'])

    evidence_for = [
        ('PHQ-9 = 18', 0.90),
        ('Anhedonia', 0.85),
        ('Sleep disturbance', 0.80),
        ('Duration > 2 weeks', 0.95)
    ]

    y_pos -= 0.7
    for evidence, certainty in evidence_for:
        # Bar
        bar_width = certainty * 4
        bar = FancyBboxPatch((5 - bar_width/2, y_pos - 0.2), bar_width, 0.3,
                            boxstyle="round,pad=0.02",
                            edgecolor=COLORS['success'],
                            facecolor=COLORS['success'],
                            alpha=0.6)
        ax1.add_patch(bar)

        # Text
        ax1.text(2, y_pos, f'• {evidence}',
                ha='left', va='center', fontsize=8)
        ax1.text(8, y_pos, f'{certainty:.2f}',
                ha='right', va='center', fontsize=8, fontweight='bold')
        y_pos -= 0.5

    # Evidence AGAINST
    y_pos -= 0.3
    ax1.text(5, y_pos, 'Evidence AGAINST:',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=COLORS['danger'])

    evidence_against = [
        ('Recent bereavement', 0.60),
        ('Thyroid dysfunction', 0.40)
    ]

    y_pos -= 0.7
    for evidence, certainty in evidence_against:
        # Bar
        bar_width = certainty * 4
        bar = FancyBboxPatch((5 - bar_width/2, y_pos - 0.2), bar_width, 0.3,
                            boxstyle="round,pad=0.02",
                            edgecolor=COLORS['danger'],
                            facecolor=COLORS['danger'],
                            alpha=0.6)
        ax1.add_patch(bar)

        # Text
        ax1.text(2, y_pos, f'• {evidence}',
                ha='left', va='center', fontsize=8)
        ax1.text(8, y_pos, f'{certainty:.2f}',
                ha='right', va='center', fontsize=8, fontweight='bold')
        y_pos -= 0.5

    # Likelihood ratios
    y_pos -= 0.5
    ax1.text(5, y_pos, 'Likelihood Ratios:',
            ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 0.4
    ax1.text(5, y_pos, 'LR_for = 2.45',
            ha='center', va='center', fontsize=8, family='monospace')
    y_pos -= 0.3
    ax1.text(5, y_pos, 'LR_against = 0.68',
            ha='center', va='center', fontsize=8, family='monospace')
    y_pos -= 0.3
    ax1.text(5, y_pos, 'LR_combined = 1.67',
            ha='center', va='center', fontsize=8, family='monospace',
            fontweight='bold')

    # Posterior
    y_pos -= 0.7
    ax1.text(5, y_pos, 'Posterior: P(H|E) = 0.73',
            ha='center', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['primary'], alpha=0.3))

    # Panel B: Execution Trace
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('B. Turbulance Execution Output',
                  fontsize=12, fontweight='bold', loc='left')

    # Execution output (from your actual output)
    output_lines = [
        ('Creating Resolution:', 'bold', COLORS['dark']),
        ('"patient has major depressive disorder"', 'italic', COLORS['primary']),
        ('', 'normal', COLORS['dark']),
        ('Evidence FOR:', 'bold', COLORS['success']),
        ('  ✓ "PHQ-9 score = 18" (0.90)', 'normal', COLORS['success']),
        ('  ✓ "anhedonia present" (0.85)', 'normal', COLORS['success']),
        ('  ✓ "sleep disturbance" (0.80)', 'normal', COLORS['success']),
        ('  ✓ "duration > 2 weeks" (0.95)', 'normal', COLORS['success']),
        ('', 'normal', COLORS['dark']),
        ('Evidence AGAINST:', 'bold', COLORS['danger']),
        ('  ✗ "recent bereavement" (0.60)', 'normal', COLORS['danger']),
        ('  ✗ "thyroid dysfunction" (0.40)', 'normal', COLORS['danger']),
        ('', 'normal', COLORS['dark']),
        ('Bayesian Update:', 'bold', COLORS['dark']),
        ('  Posterior certainty: 0.73', 'bold', COLORS['primary']),
        ('', 'normal', COLORS['dark']),
        ('Perturbation Validation:', 'bold', COLORS['dark']),
        ('  Robustness score: 0.87 (high)', 'bold', COLORS['secondary'])
    ]

    y_pos = 11.5
    for text, style, color in output_lines:
        weight = 'bold' if style == 'bold' else 'normal'
        fontstyle = 'italic' if style == 'italic' else 'normal'

        ax2.text(0.5, y_pos, text,
                ha='left', va='top', fontsize=8,
                color=color, fontweight=weight, fontstyle=fontstyle,
                family='monospace')
        y_pos -= 0.55

    # Panel C: Clinical Application
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 12)
    ax3.axis('off')
    ax3.set_title('C. Clinical Application',
                  fontsize=12, fontweight='bold', loc='left')

    # Diagnosis box
    diag_box = FancyBboxPatch((1, 10), 8, 1.5,
                             boxstyle="round,pad=0.1",
                             edgecolor=COLORS['primary'],
                             facecolor=COLORS['primary'],
                             alpha=0.2,
                             linewidth=2)
    ax3.add_patch(diag_box)
    ax3.text(5, 11, 'Diagnosis: Major Depressive Disorder',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(5, 10.4, 'Confidence: 73%',
            ha='center', va='center', fontsize=9)

    # Robustness analysis
    y_pos = 9
    ax3.text(5, y_pos, 'Robustness Analysis:',
            ha='center', va='center', fontsize=10, fontweight='bold')

    perturbations = [
        ('Remove PHQ-9', -0.05),
        ('Remove anhedonia', -0.03),
        ('Remove sleep', -0.02),
        ('Remove duration', -0.08)
    ]

    y_pos -= 0.8
    for evidence, delta in perturbations:
        # Delta bar
        bar_width = abs(delta) * 20
        bar_x = 5 - bar_width/2 if delta < 0 else 5
        bar_color = COLORS['danger'] if delta < 0 else COLORS['success']

        bar = FancyBboxPatch((bar_x, y_pos - 0.15), bar_width, 0.25,
                            boxstyle="round,pad=0.02",
                            edgecolor=bar_color,
                            facecolor=bar_color,
                            alpha=0.6)
        ax3.add_patch(bar)

        # Text
        ax3.text(1.5, y_pos, evidence,
                ha='left', va='center', fontsize=8)
        ax3.text(8.5, y_pos, f'Δ = {delta:+.2f}',
                ha='right', va='center', fontsize=8,
                fontweight='bold', family='monospace')
        y_pos -= 0.5

    # Most critical evidence
    y_pos -= 0.5
    ax3.text(5, y_pos, 'Most Critical Evidence:',
            ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 0.4
    ax3.text(5, y_pos, '1. Duration > 2 weeks (Δ = -8%)',
            ha='center', va='center', fontsize=8)
    y_pos -= 0.3
    ax3.text(5, y_pos, '2. PHQ-9 score (Δ = -5%)',
            ha='center', va='center', fontsize=8)

    # Recommendation
    y_pos -= 0.8
    rec_box = FancyBboxPatch((1, y_pos - 0.5), 8, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor=COLORS['success'],
                            facecolor=COLORS['success'],
                            alpha=0.2,
                            linewidth=2)
    ax3.add_patch(rec_box)
    ax3.text(5, y_pos, 'Recommendation:',
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax3.text(5, y_pos - 0.35, 'Proceed with treatment protocol',
            ha='center', va='center', fontsize=8)
    ax3.text(5, y_pos - 0.6, '(high confidence, robust evidence)',
            ha='center', va='center', fontsize=7, style='italic')

    plt.tight_layout()
    viz.save_figure(fig, 'figure_3_bayesian_integration')
    plt.show()

if __name__ == '__main__':
    create_figure_3()
