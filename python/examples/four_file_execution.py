# figure_5_four_file_system.py
"""
Figure 5: Turbulance Four-File Execution Architecture
Architecture diagram showing file interactions and safety mechanisms
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from visualization_toolkit import TurbulanceVisualizer, COLORS

def create_figure_5():
    """Generate Figure 5: Four-File Execution System"""
    
    viz = TurbulanceVisualizer()
    
    fig = plt.figure(figsize=(14, 16))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_title('Turbulance Four-File Execution Architecture', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # .trb file (top)
    trb_box = FancyBboxPatch((2, 13.5), 10, 2,
                            boxstyle="round,pad=0.15",
                            edgecolor=COLORS['primary'],
                            facecolor=COLORS['primary'],
                            alpha=0.2,
                            linewidth=3)
    ax.add_patch(trb_box)
    
    ax.text(3, 15.2, '.trb (Turbulance)', 
           ha='left', va='center', fontsize=12, fontweight='bold',
           color=COLORS['primary'])
    ax.text(3, 14.8, 'Protocol Specification', 
           ha='left', va='center', fontsize=10, style='italic')
    
    trb_content = [
        '• Defines semantic network (V8 BMDs)',
        '• Specifies experimental protocol',
        '• Declares Points, Resolutions, BMDs',
        '• Compilation targets (Python/R/SQL)'
    ]
    y_pos = 14.3
    for line in trb_content:
        ax.text(3.5, y_pos, line, ha='left', va='center', fontsize=8)
        y_pos -= 0.3
    
    # Arrow: compiles to
    arrow1 = FancyArrowPatch((7, 13.3), (7, 12.2),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=COLORS['dark'])
    ax.add_patch(arrow1)
    ax.text(8, 12.75, 'compiles to', ha='left', va='center', 
           fontsize=9, style='italic')
    
    # Polyglot execution (middle-top)
    poly_box = FancyBboxPatch((2, 10.5), 10, 1.5,
                             boxstyle="round,pad=0.15",
                             edgecolor=COLORS['secondary'],
                             facecolor=COLORS['secondary'],
                             alpha=0.2,
                             linewidth=3)
    ax.add_patch(poly_box)
    
    ax.text(7, 11.5, 'Python / R / SQL / Shell', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=COLORS['secondary'])
    ax.text(7, 11, '(Polyglot Execution)', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # Arrow: monitors
    arrow2 = FancyArrowPatch((7, 10.3), (7, 9.2),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=COLORS['dark'])
    ax.add_patch(arrow2)
    ax.text(8, 9.75, 'monitors', ha='left', va='center', 
           fontsize=9, style='italic')
    
    # .fs file (middle)
    fs_box = FancyBboxPatch((2, 7), 10, 2,
                           boxstyle="round,pad=0.15",
                           edgecolor=COLORS['success'],
                           facecolor=COLORS['success'],
                           alpha=0.2,
                           linewidth=3)
    ax.add_patch(fs_box)
    
    ax.text(3, 8.7, '.fs (Flux State)', 
           ha='left', va='center', fontsize=12, fontweight='bold',
           color=COLORS['success'])
    ax.text(3, 8.3, 'Real-Time Consciousness Monitoring', 
           ha='left', va='center', fontsize=10, style='italic')
    
    fs_content = [
        '• V8 module states (Mzekezeke, Zengeza, etc.)',
        '• Consciousness metrics (Ψ₀, Θ₀, PLV)',
        '• Thermodynamic state (S-entropy coordinates)',
        '• Updated in real-time during execution'
    ]
    y_pos = 7.8
    for line in fs_content:
        ax.text(3.5, y_pos, line, ha='left', va='center', fontsize=8)
        y_pos -= 0.3
    
    # Arrow: depends on
    arrow3 = FancyArrowPatch((7, 6.8), (7, 5.7),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=COLORS['dark'])
    ax.add_patch(arrow3)
    ax.text(8, 6.25, 'depends on', ha='left', va='center', 
           fontsize=9, style='italic')
    
    # .ghd file (middle-bottom)
    ghd_box = FancyBboxPatch((2, 3.5), 10, 2,
                            boxstyle="round,pad=0.15",
                            edgecolor=COLORS['warning'],
                            facecolor=COLORS['warning'],
                            alpha=0.2,
                            linewidth=3)
    ax.add_patch(ghd_box)
    
    ax.text(3, 5.2, '.ghd (Gerhard Dependencies)', 
           ha='left', va='center', fontsize=12, fontweight='bold',
           color=COLORS['warning'])
    ax.text(3, 4.8, 'Resource Network', 
           ha='left', va='center', fontsize=10, style='italic')
    
    ghd_content = [
        '• External databases (HMDB, KEGG, PubMed)',
        '• Computational resources (CPU, GPU, cloud)',
        '• Data dependencies (input/output files)',
        '• Software dependencies (packages, tools)'
    ]
    y_pos = 4.3
    for line in ghd_content:
        ax.text(3.5, y_pos, line, ha='left', va='center', fontsize=8)
        y_pos -= 0.3
    
    # Arrow: learns from
    arrow4 = FancyArrowPatch((7, 3.3), (7, 2.2),
                            arrowstyle='->', mutation_scale=30,
                            linewidth=3, color=COLORS['dark'])
    ax.add_patch(arrow4)
    ax.text(8, 2.75, 'learns from', ha='left', va='center', 
           fontsize=9, style='italic')
    
    # .hre file (bottom)
    hre_box = FancyBboxPatch((2, 0), 10, 2,
                            boxstyle="round,pad=0.15",
                            edgecolor=COLORS['danger'],
                            facecolor=COLORS['danger'],
                            alpha=0.2,
                            linewidth=3)
    ax.add_patch(hre_box)
    
    ax.text(3, 1.7, '.hre (Harare Decision Log)', 
           ha='left', va='center', fontsize=12, fontweight='bold',
           color=COLORS['danger'])
    ax.text(3, 1.3, 'Metacognitive Learning', 
           ha='left', va='center', fontsize=10, style='italic')
    
    hre_content = [
        '• Decision timeline (reasoning trace)',
        '• Metacognitive insights (lessons learned)',
        '• Self-corrections (errors detected/fixed)',
        '• Future protocols (improvement suggestions)'
    ]
    y_pos = 0.8
    for line in hre_content:
        ax.text(3.5, y_pos, line, ha='left', va='center', fontsize=8)
        y_pos -= 0.3
    
    plt.tight_layout()
    viz.save_figure(fig, 'figure_5_four_file_system')
    plt.show()

if __name__ == '__main__':
    create_figure_5()
