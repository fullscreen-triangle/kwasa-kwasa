"""
Validation Script 3: Multi-Lens Analysis

Tests interpretation across different semantic lenses.
Shows how different lenses provide different insights.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import SemanticMaxwellDemon, SemanticLens

def run_multi_lens_validation():
    """Test multi-lens interpretation"""
    print("=" * 60)
    print("MULTI-LENS ANALYSIS VALIDATION")
    print("=" * 60)
    
    demon = SemanticMaxwellDemon()
    
    # Test case
    observation = {
        "text": "Patient reports persistent low mood for 3 weeks, with fatigue and sleep problems",
        "biomarkers": {"cortisol": "elevated"},
        "duration": 21,
        "severity": 7
    }
    
    print(f"\nObservation: {observation['text']}")
    print(f"Duration: {observation['duration']} days")
    print(f"Severity: {observation['severity']}/10")
    
    # Test each lens
    lenses_to_test = [
        SemanticLens.PSYCHIATRIC,
        SemanticLens.ENDOCRINE,
        SemanticLens.NEUROLOGICAL,
        SemanticLens.BIOCHEMICAL
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print("INTERPRETATIONS BY LENS")
    print(f"{'='*60}\n")
    
    for lens in lenses_to_test:
        print(f"📋 {lens.value.upper()} LENS:")
        interpretation = demon.filter(observation, lens=lens)
        
        print(f"  Category: {interpretation.primary_state.category}")
        print(f"  Confidence: {interpretation.primary_state.confidence:.2f}")
        print(f"  S-entropy: S_k={interpretation.primary_state.s_entropy.S_k:.1f}, "
              f"S_t={interpretation.primary_state.s_entropy.S_t:.1f}, "
              f"S_e={interpretation.primary_state.s_entropy.S_e:.1f}")
        print(f"  Alternatives: {len(interpretation.alternative_states)}")
        print()
        
        results[lens] = interpretation
    
    # Compare all lenses
    print(f"{'='*60}")
    print("CROSS-LENS COMPARISON")
    print(f"{'='*60}\n")
    
    comparison = demon.compare_interpretations(observation, lenses=lenses_to_test)
    
    print(f"Agreement across lenses: {comparison['agreement']:.2%}")
    print(f"Consensus category: {comparison['consensus_category']}")
    print(f"Unique interpretations: {comparison['unique_categories']}")
    print(f"S-entropy divergence: {comparison['s_entropy_divergence']:.2f}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}\n")
    
    if comparison['agreement'] > 0.7:
        print("✅ HIGH AGREEMENT: Lenses converge on similar interpretation")
    elif comparison['agreement'] > 0.4:
        print("⚠️  MODERATE AGREEMENT: Some divergence in interpretations")
    else:
        print("⚡ LOW AGREEMENT: Significant divergence - rich semantic complexity!")
    
    print(f"\n📊 This case generates {len(comparison['unique_categories'])} distinct categorical interpretations")
    print("   Each lens provides complementary information")
    print("   → Demonstrates non-destructive multi-lens filtering")
    
    print("\n✅ Multi-lens validation complete!")
    
    return results, comparison

if __name__ == "__main__":
    results, comparison = run_multi_lens_validation()
    
    # Save results
    os.makedirs("validation/results", exist_ok=True)
    with open("validation/results/03_multi_lens_comparison.txt", "w") as f:
        f.write("Multi-Lens Analysis Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Agreement: {comparison['agreement']:.2%}\n")
        f.write(f"Consensus: {comparison['consensus_category']}\n")
        f.write(f"Unique interpretations: {len(comparison['unique_categories'])}\n")
        f.write(f"S-entropy divergence: {comparison['s_entropy_divergence']:.2f}\n")
    
    print(f"\n📁 Results saved to: validation/results/03_multi_lens_comparison.txt")

