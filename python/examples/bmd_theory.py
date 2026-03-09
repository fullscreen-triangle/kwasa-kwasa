# figure_1_theory_implementation.py
"""
Figure 1: Information Catalyst Theory and Turbulance Implementation
Two-panel comparison showing mathematical formalization and executable code
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_1():
    """Generate Figure 1: Theory → Implementation"""

    viz = TurbulanceVisualizer()

    # Create figure with 2 panels
    fig = plt.figure(figsize=(14, 6))

    # Panel A: Mathematical Formalization
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('A. Information Catalyst Theory', fontsize=14, fontweight='bold', loc='left')

    # Ω_POT (top)
    omega_pot = FancyBboxPatch((1, 7.5), 8, 1.5,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['primary'],
                               facecolor=COLORS['primary'],
                               alpha=0.3,
                               linewidth=2)
    ax1.add_patch(omega_pot)
    ax1.text(5, 8.25, r'$\Omega_{POT}$ (Potential States)',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, 7.8, r'$|\Omega_{POT}| \sim 10^{44}$',
             ha='center', va='center', fontsize=10, style='italic')

    # Arrow down
    arrow1 = FancyArrowPatch((5, 7.3), (5, 6.2),
                            arrowstyle='->',
                            mutation_scale=30,
                            linewidth=3,
                            color=COLORS['dark'])
    ax1.add_patch(arrow1)

    # Information Catalyst (middle)
    catalyst = FancyBboxPatch((2, 4.8), 6, 1.2,
                             boxstyle="round,pad=0.1",
                             edgecolor=COLORS['secondary'],
                             facecolor=COLORS['secondary'],
                             alpha=0.3,
                             linewidth=2)
    ax1.add_patch(catalyst)
    ax1.text(5, 5.6, 'Information Catalyst C',
             ha='center', va='center', fontsize=11, fontweight='bold')
    ax1.text(5, 5.2, r'$C = I^m_{input} \circ I^m_{output}$',
             ha='center', va='center', fontsize=10)

    # Arrow down
    arrow2 = FancyArrowPatch((5, 4.6), (5, 3.5),
                            arrowstyle='->',
                            mutation_scale=30,
                            linewidth=3,
                            color=COLORS['dark'])
    ax1.add_patch(arrow2)

    # Ω_ACT (bottom)
    omega_act = FancyBboxPatch((3, 1.5), 4, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor=COLORS['success'],
                              facecolor=COLORS['success'],
                              alpha=0.3,
                              linewidth=2)
    ax1.add_patch(omega_act)
    ax1.text(5, 2.25, r'$\Omega_{ACT}$ (Actual States)',
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax1.text(5, 1.8, r'$|\Omega_{ACT}| \sim 10^{6}$',
             ha='center', va='center', fontsize=10, style='italic')

    # Order creation text
    ax1.text(5, 0.5, r'Order Creation: $|\Omega_{POT}|/|\Omega_{ACT}| \sim 10^{38}$',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Turbulance Execution
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('B. Turbulance Execution Output', fontsize=14, fontweight='bold', loc='left')

    # Execution output box
    output_box = FancyBboxPatch((0.5, 1), 9, 8,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['dark'],
                               facecolor='white',
                               linewidth=2)
    ax2.add_patch(output_box)

    # Execution output text (from your actual output)
    output_text = """
Creating Points...
✓ Point 1: "depression diagnosis"
   certainty: 0.80

✓ Point 2: "anxiety comorbidity"
   certainty: 0.65

Combining Points (AND operation)...
Result: certainty = min(0.80, 0.65) = 0.65

Combining Points (OR operation)...
Result: certainty = max(0.80, 0.65) = 0.80

Dependency tracking...
Point "treatment_decision" depends on:
  - "depression diagnosis" (0.80)
  - "anxiety comorbidity" (0.65)
Propagated certainty: 0.67
    """

    y_pos = 8.5
    for line in output_text.strip().split('\n'):
        if line.strip():
            if '✓' in line or 'Result:' in line:
                color = COLORS['success']
                weight = 'bold'
            elif 'certainty' in line.lower():
                color = COLORS['primary']
                weight = 'normal'
            else:
                color = COLORS['dark']
                weight = 'normal'

            ax2.text(1, y_pos, line.strip(),
                    ha='left', va='top', fontsize=9,
                    color=color, fontweight=weight,
                    family='monospace')
            y_pos -= 0.35

    # Highlight box
    highlight = FancyBboxPatch((0.7, 1.2), 8.6, 1.5,
                              boxstyle="round,pad=0.05",
                              edgecolor=COLORS['secondary'],
                              facecolor='none',
                              linewidth=2,
                              linestyle='--')
    ax2.add_patch(highlight)
    ax2.text(5, 1.5, 'Theory Implemented in Executable Code',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=COLORS['secondary'])

    plt.tight_layout()
    viz.save_figure(fig, 'figure_1_theory_implementation')
    plt.show()

if __name__ == '__main__':
    create_figure_1()
