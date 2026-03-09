# figure_4_consciousness_metrics.py
"""
Figure 4: Consciousness Metrics During Depression Treatment
Time-series plots showing evolution of consciousness parameters
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_4():
    """Generate Figure 4: Consciousness Metrics Over Time"""

    viz = TurbulanceVisualizer()

    # Generate synthetic time-series data based on your reported results
    # (Replace with actual .fs file data when available)
    weeks = np.linspace(0, 8, 100)

    # Perception flux (Ψ₀): 0.45 → 0.85
    psi_0 = 0.45 + 0.40 * (1 - np.exp(-weeks/2.5)) + np.random.normal(0, 0.02, 100)

    # Thought flux (Θ₀): 0.40 → 0.72
    theta_0 = 0.40 + 0.32 * (1 - np.exp(-weeks/3.0)) + np.random.normal(0, 0.02, 100)

    # Phase-locking value (PLV): 0.32 → 0.77
    plv = 0.32 + 0.45 * (1 - np.exp(-weeks/2.8)) + np.random.normal(0, 0.015, 100)

    # Consciousness quality: 0.45 → 0.88
    consciousness_quality = 0.45 + 0.43 * (1 - np.exp(-weeks/2.6)) + np.random.normal(0, 0.02, 100)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # Panel A: Perception Flux
    ax1 = axes[0]
    ax1.plot(weeks, psi_0, linewidth=2.5, color=COLORS['primary'], label='Ψ₀')
    ax1.axhline(y=0.45, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax1.axhline(y=0.85, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    ax1.fill_between(weeks, 0.45, psi_0, alpha=0.2, color=COLORS['primary'])
    ax1.set_ylabel('Perception Flux (Ψ₀)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_title('A. Perception Flux (External Sensory Processing)',
                  fontsize=12, fontweight='bold', loc='left')

    # Panel B: Thought Flux
    ax2 = axes[1]
    ax2.plot(weeks, theta_0, linewidth=2.5, color=COLORS['secondary'], label='Θ₀')
    ax2.axhline(y=0.40, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax2.axhline(y=0.72, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Target')
    ax2.fill_between(weeks, 0.40, theta_0, alpha=0.2, color=COLORS['secondary'])
    ax2.set_ylabel('Thought Flux (Θ₀)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0.3, 0.9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.set_title('B. Thought Flux (Internal Cognitive Processing)',
                  fontsize=12, fontweight='bold', loc='left')

    # Panel C: Phase-Locking Value
    ax3 = axes[2]
    ax3.plot(weeks, plv, linewidth=2.5, color=COLORS['success'], label='θ-band PLV')
    ax3.axhline(y=0.32, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Depressed baseline')
    ax3.axhline(y=0.77, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Post-intervention')
    ax3.fill_between(weeks, 0.32, plv, alpha=0.2, color=COLORS['success'])
    ax3.set_ylabel('Phase-Locking Value (PLV)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0.2, 0.9)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.set_title('C. Phase-Locking Value (θ-band, mPFC-Amygdala)',
                  fontsize=12, fontweight='bold', loc='left')

    # Add clinical outcome annotation
    ax3.annotate('Clinical outcome:\n65% symptom reduction (HDRS)',
                xy=(7, 0.75), xytext=(5, 0.55),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['dark']))

    # Panel D: Consciousness Quality
    ax4 = axes[3]
    ax4.plot(weeks, consciousness_quality, linewidth=2.5, color=COLORS['warning'], label='Consciousness Quality')
    ax4.axhline(y=0.45, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax4.axhline(y=0.88, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7, label='Post-intervention')
    ax4.fill_between(weeks, 0.45, consciousness_quality, alpha=0.2, color=COLORS['warning'])
    ax4.set_ylabel('Consciousness Quality', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (weeks)', fontsize=11, fontweight='bold')
    ax4.set_ylim(0.3, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.set_title('D. Integrated Consciousness Quality',
                  fontsize=12, fontweight='bold', loc='left')

    # Add formula annotation
    ax4.text(0.5, 0.35, r'$Q_{consciousness} = f(\Psi_0, \Theta_0, PLV, H_{depth})$',
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    viz.save_figure(fig, 'figure_4_consciousness_metrics')
    plt.show()

if __name__ == '__main__':
    create_figure_4()
