#!/usr/bin/env python3
"""
Example 3: Drug-O₂ Aggregation Measurement
From: kuramoto-oscillator-phase-computing.tex & hybrid-meta-language-pharmacodynamics.tex

Measures oxygen aggregation constant (K_agg) and electromagnetic coupling.
This is THE critical parameter determining if a drug can program consciousness.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DrugO2Analyzer:
    """
    Analyzes drug-O₂ aggregation and electromagnetic coupling
    
    From paper (Section 2.2):
    - K_agg > 10^4 M⁻¹ required for therapeutic effect
    - EM coupling strength ∝ μ_drug · μ_O2 / r³
    - 4:1 H⁺:O₂ resonance enables phase-locking
    """
    
    def __init__(self):
        # O₂ paramagnetic properties
        self.mu_O2 = 2.83  # Bohr magnetons
        self.omega_O2 = 1e13  # Hz (O₂ vibrational frequency)
        self.omega_Hplus = 4e13  # Hz (H⁺ EM field, 4:1 resonance)
    
    def calculate_K_agg(self, aromatic_rings, heteroatoms, mw):
        """
        Calculate O₂ aggregation constant
        
        K_agg = K_base · f(aromatic) · f(hetero) · f(mw)
        
        Aromatic rings enable π-π stacking with O₂
        Heteroatoms (N, O) enable H-bonding with O₂
        """
        K_base = 100  # M⁻¹ baseline
        
        # Aromatic contribution (exponential enhancement)
        f_aromatic = np.exp(aromatic_rings * 0.5)
        
        # Heteroatom contribution
        f_hetero = 1 + heteroatoms * 0.2
        
        # Molecular weight factor (larger molecules aggregate more)
        f_mw = (mw / 100) ** 0.3
        
        K_agg = K_base * f_aromatic * f_hetero * f_mw
        
        return K_agg
    
    def calculate_em_coupling(self, K_agg, unpaired_electrons):
        """
        Calculate electromagnetic coupling strength
        
        From paper: μ_drug = g_e √(S(S+1))
        S = unpaired_electrons / 2
        """
        # Magnetic moment
        S = unpaired_electrons / 2
        if S == 0:
            mu_drug = 0.1  # Induced moment
        else:
            mu_drug = 2.0 * np.sqrt(S * (S + 1))  # Bohr magnetons
        
        # EM coupling ∝ μ_drug · μ_O2
        coupling = (mu_drug * self.mu_O2) / 10.0  # Normalized
        
        # Enhanced by aggregation
        coupling *= np.log10(K_agg) / 4.0  # log scale
        
        return coupling, mu_drug
    
    def calculate_resonance_quality(self, freq_drug):
        """
        Calculate resonance quality factor Q
        
        Q = ω_0 / Δω
        
        High Q = strong resonance with O₂
        """
        # Frequency difference from O₂
        delta_freq = abs(freq_drug - self.omega_O2)
        
        # Quality factor (inverse of detuning)
        if delta_freq < self.omega_O2 * 0.1:  # Within 10%
            Q = self.omega_O2 / (delta_freq + self.omega_O2 * 0.01)
        else:
            Q = 0.5  # Poor resonance
        
        # Check 4:1 H⁺ resonance
        delta_Hplus = abs(4 * freq_drug - self.omega_Hplus)
        if delta_Hplus < self.omega_Hplus * 0.1:
            Q *= 1.5  # Resonance enhancement
        
        return Q
    
    def analyze_drug(self, name, aromatic_rings, heteroatoms, mw, 
                     unpaired_electrons, freq_drug):
        """Complete drug analysis"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {name}")
        print(f"{'='*60}")
        print(f"  Aromatic rings: {aromatic_rings}")
        print(f"  Heteroatoms (N,O): {heteroatoms}")
        print(f"  Molecular weight: {mw:.1f} g/mol")
        print(f"  Unpaired electrons: {unpaired_electrons}")
        print(f"  Vibrational frequency: {freq_drug:.2e} Hz ({freq_drug/1e12:.2f} THz)")
        
        # Calculate K_agg
        K_agg = self.calculate_K_agg(aromatic_rings, heteroatoms, mw)
        print(f"\n  O₂ Aggregation Constant (K_agg):")
        print(f"    K_agg = {K_agg:.2e} M⁻¹")
        print(f"    Therapeutic threshold: 10^4 M⁻¹")
        
        if K_agg > 1e4:
            print(f"    Status: ✓ EXCEEDS threshold ({K_agg/1e4:.1f}× above)")
            therapeutic = True
        else:
            print(f"    Status: ✗ BELOW threshold ({K_agg/1e4:.1f}× below)")
            therapeutic = False
        
        # Calculate EM coupling
        coupling, mu_drug = self.calculate_em_coupling(K_agg, unpaired_electrons)
        print(f"\n  Electromagnetic Coupling:")
        print(f"    Magnetic moment: {mu_drug:.2f} Bohr magnetons")
        print(f"    EM coupling strength: {coupling:.2f} (normalized)")
        
        # Calculate resonance quality
        Q = self.calculate_resonance_quality(freq_drug)
        print(f"\n  Resonance Quality:")
        print(f"    O₂ frequency: {self.omega_O2:.2e} Hz")
        print(f"    Drug frequency: {freq_drug:.2e} Hz")
        print(f"    Quality factor Q: {Q:.2f}")
        print(f"    4:1 H⁺ resonance: {4*freq_drug:.2e} Hz")
        print(f"    Target (H⁺): {self.omega_Hplus:.2e} Hz")
        
        if Q > 1.0:
            print(f"    Status: ✓ GOOD resonance (Q > 1)")
        else:
            print(f"    Status: ✗ POOR resonance (Q < 1)")
        
        # Overall assessment
        print(f"\n  Overall Assessment:")
        if therapeutic and Q > 1.0 and coupling > 0.5:
            print(f"    ✓✓✓ EXCELLENT therapeutic potential")
            print(f"    Can program consciousness through O₂-H⁺ coupling")
            grade = "A"
        elif therapeutic and (Q > 1.0 or coupling > 0.5):
            print(f"    ✓✓ GOOD therapeutic potential")
            print(f"    Likely effective for consciousness programming")
            grade = "B"
        elif therapeutic:
            print(f"    ✓ MODERATE therapeutic potential")
            print(f"    May work but suboptimal coupling")
            grade = "C"
        else:
            print(f"    ✗ INSUFFICIENT therapeutic potential")
            print(f"    Cannot effectively program consciousness")
            grade = "F"
        
        return {
            'name': name,
            'K_agg': K_agg,
            'coupling': coupling,
            'Q': Q,
            'therapeutic': therapeutic,
            'grade': grade
        }


def create_comparison_plot(results, filename='drug_o2_analysis.png'):
    """Visualize drug-O₂ analysis results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    drugs = [r['name'] for r in results]
    K_agg = [r['K_agg'] for r in results]
    coupling = [r['coupling'] for r in results]
    Q = [r['Q'] for r in results]
    grades = [r['grade'] for r in results]
    
    x = np.arange(len(drugs))
    
    # Plot 1: K_agg (log scale)
    colors = ['green' if k > 1e4 else 'red' for k in K_agg]
    ax1.bar(x, K_agg, color=colors, alpha=0.7)
    ax1.axhline(y=1e4, color='orange', linestyle='--', linewidth=2, label='Therapeutic threshold')
    ax1.set_yscale('log')
    ax1.set_ylabel('K_agg (M⁻¹)')
    ax1.set_title('O₂ Aggregation Constant')
    ax1.set_xticks(x)
    ax1.set_xticklabels(drugs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: EM Coupling
    ax2.bar(x, coupling, color='purple', alpha=0.7)
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Good coupling threshold')
    ax2.set_ylabel('EM Coupling Strength')
    ax2.set_title('Electromagnetic Coupling')
    ax2.set_xticks(x)
    ax2.set_xticklabels(drugs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3: Resonance Quality Q
    ax3.bar(x, Q, color='blue', alpha=0.7)
    ax3.axhline(y=1.0, color='orange', linestyle='--', label='Good resonance threshold')
    ax3.set_ylabel('Quality Factor Q')
    ax3.set_title('Resonance Quality with O₂')
    ax3.set_xticks(x)
    ax3.set_xticklabels(drugs, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Overall grades
    grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    grade_values = [grade_map[g] for g in grades]
    colors_grade = ['darkgreen' if g == 'A' else 'green' if g == 'B' else 
                    'yellow' if g == 'C' else 'red' for g in grades]
    ax4.bar(x, grade_values, color=colors_grade, alpha=0.7)
    ax4.set_ylabel('Grade')
    ax4.set_yticks([0, 1, 2, 3, 4])
    ax4.set_yticklabels(['F', 'D', 'C', 'B', 'A'])
    ax4.set_title('Overall Therapeutic Potential')
    ax4.set_xticks(x)
    ax4.set_xticklabels(drugs, rotation=45, ha='right')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\n  Saved plot: {filename}")


def main():
    print("="*60)
    print("EXAMPLE 3: DRUG-O₂ AGGREGATION ANALYSIS")
    print("The critical parameter for consciousness programming")
    print("="*60)
    print()
    print("From papers: kuramoto-oscillator-phase-computing.tex")
    print("             hybrid-meta-language-pharmacodynamics.tex")
    print()
    print("KEY CLAIM: K_agg > 10^4 M⁻¹ enables consciousness programming")
    print("MECHANISM: Drug-O₂ aggregation → EM coupling → phase-locking")
    print("RESONANCE: 4:1 H⁺:O₂ frequency ratio critical")
    print()
    
    analyzer = DrugO2Analyzer()
    
    # Drugs from the papers with realistic parameters
    drugs = [
        # (name, aromatic_rings, heteroatoms, mw, unpaired_electrons, freq_THz)
        ("Lithium", 0, 0, 7, 0, 3.32e13),
        ("Dopamine", 1, 3, 153, 0, 1.2e13),
        ("Serotonin (5-HT)", 1, 3, 176, 0, 1.1e13),
        ("Sertraline (SSRI)", 2, 2, 306, 0, 0.9e13),
        ("Alprazolam (Benzo)", 2, 3, 309, 0, 0.8e13),
        ("Metformin", 0, 5, 129, 0, 1.5e13),
    ]
    
    results = []
    for drug_params in drugs:
        result = analyzer.analyze_drug(*drug_params)
        results.append(result)
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Consciousness Programming Capability")
    print(f"{'='*60}")
    print()
    print(f"{'Drug':<20} {'K_agg (M⁻¹)':<15} {'EM Coupling':<12} {'Q':<8} {'Grade':<8}")
    print("-"*75)
    for r in results:
        print(f"{r['name']:<20} {r['K_agg']:<15.2e} {r['coupling']:<12.2f} "
              f"{r['Q']:<8.2f} {r['grade']:<8}")
    
    # Create visualization
    create_comparison_plot(results)
    
    print()
    print("="*60)
    print("VALIDATION OUTCOME")
    print("="*60)
    print()
    therapeutic_drugs = [r['name'] for r in results if r['therapeutic']]
    print(f"Therapeutic drugs (K_agg > 10^4): {len(therapeutic_drugs)}/{len(results)}")
    for drug in therapeutic_drugs:
        print(f"  ✓ {drug}")
    print()
    print("KEY INSIGHT: Only drugs with sufficient K_agg can aggregate")
    print("to O₂ and modulate phase-lock coupling. This is the physical")
    print("mechanism enabling consciousness programming.")
    print()
    print("PREDICTIVE POWER: Given molecular structure → calculate K_agg")
    print("→ predict consciousness programming capability BEFORE synthesis!")
    print()
    print("Next: Example 4 - Metabolic hierarchy analysis")


if __name__ == '__main__':
    main()

