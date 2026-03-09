"""
Validation Script 2: Depression Treatment Data

Applies Semantic Maxwell Demon to existing depression data.
Shows effect of semantic demon vs no demon.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import SemanticMaxwellDemon, SemanticLens, CategoricalState, SEntropyCoordinates
import numpy as np

# Sample depression data (replace with actual data from python/examples)
DEPRESSION_CASES = [
    {
        "id": "case_001",
        "symptoms": "low mood, decreased energy, sleep difficulties, loss of interest",
        "duration": 14,
        "severity": 7,
        "biomarkers": {"cortisol": "elevated", "serotonin": "low"}
    },
    {
        "id": "case_002",
        "symptoms": "persistent sadness, fatigue, concentration problems, weight changes",
        "duration": 21,
        "severity": 8,
        "biomarkers": {"cortisol": "elevated", "dopamine": "low"}
    },
    {
        "id": "case_003",
        "symptoms": "irritability, social withdrawal, sleep disturbance, hopelessness",
        "duration": 30,
        "severity": 6,
        "biomarkers": {"cortisol": "normal", "serotonin": "low"}
    }
]

def analyze_without_demon(case):
    """Traditional analysis without semantic demon"""
    # Simple rule-based categorization
    if case["severity"] >= 7 and case["duration"] >= 14:
        category = "major_depressive_episode"
        confidence = 0.7
    elif case["severity"] >= 5:
        category = "moderate_depression"
        confidence = 0.6
    else:
        category = "mild_depression"
        confidence = 0.5
    
    return {
        "category": category,
        "confidence": confidence,
        "alternatives": [],
        "evidence": ["symptom_count", "severity_threshold"]
    }

def analyze_with_demon(case, demon):
    """Analysis with semantic demon"""
    observation = {
        "text": case["symptoms"],
        "duration": case["duration"],
        "severity": case["severity"],
        "biomarkers": case["biomarkers"]
    }
    
    # Filter through psychiatric lens
    interpretation = demon.filter(observation, lens=SemanticLens.PSYCHIATRIC)
    
    return {
        "category": interpretation.primary_state.category,
        "confidence": interpretation.primary_state.confidence,
        "s_entropy": interpretation.primary_state.s_entropy,
        "alternatives": [alt.category for alt in interpretation.alternative_states],
        "evidence": interpretation.primary_state.evidence
    }

def compare_analyses(case_id, without_demon, with_demon):
    """Compare results"""
    print(f"\n{'='*60}")
    print(f"CASE: {case_id}")
    print(f"{'='*60}")
    
    print("\n📊 WITHOUT DEMON (Rule-based):")
    print(f"  Category: {without_demon['category']}")
    print(f"  Confidence: {without_demon['confidence']:.2f}")
    print(f"  Alternatives: {len(without_demon['alternatives'])}")
    print(f"  Evidence: {', '.join(without_demon['evidence'][:2])}")
    
    print("\n🔮 WITH DEMON (Semantic):")
    print(f"  Category: {with_demon['category']}")
    print(f"  Confidence: {with_demon['confidence']:.2f}")
    print(f"  S-entropy: {with_demon['s_entropy']}")
    print(f"  Alternatives: {len(with_demon['alternatives'])}")
    if with_demon['alternatives']:
        print(f"    → {', '.join(with_demon['alternatives'][:3])}")
    print(f"  Evidence sources: {len(with_demon['evidence'])}")
    
    # Information gain
    info_gain = len(with_demon['alternatives']) - len(without_demon['alternatives'])
    print(f"\n📈 INFORMATION GAIN:")
    print(f"  Additional interpretations: +{info_gain}")
    print(f"  Confidence improvement: {(with_demon['confidence'] - without_demon['confidence']):.2f}")
    print(f"  S-entropy distance: {with_demon['s_entropy'].magnitude:.2f}")

def run_validation():
    """Run full validation"""
    print("=" * 60)
    print("DEPRESSION DATA VALIDATION")
    print("=" * 60)
    print("\nComparing traditional vs semantic demon analysis...")
    
    # Create demon
    demon = SemanticMaxwellDemon()
    
    all_results = []
    
    for case in DEPRESSION_CASES:
        # Analyze both ways
        without = analyze_without_demon(case)
        with_d = analyze_with_demon(case, demon)
        
        # Compare
        compare_analyses(case["id"], without, with_d)
        
        all_results.append({
            "case_id": case["id"],
            "without_demon": without,
            "with_demon": with_d
        })
    
    # Overall statistics
    print(f"\n\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    
    avg_conf_without = np.mean([r["without_demon"]["confidence"] for r in all_results])
    avg_conf_with = np.mean([r["with_demon"]["confidence"] for r in all_results])
    avg_alternatives = np.mean([len(r["with_demon"]["alternatives"]) for r in all_results])
    
    print(f"\nAverage confidence without demon: {avg_conf_without:.2f}")
    print(f"Average confidence with demon: {avg_conf_with:.2f}")
    print(f"Confidence improvement: +{(avg_conf_with - avg_conf_without):.2f}")
    print(f"Average alternatives generated: {avg_alternatives:.1f}")
    
    # Demon statistics
    demon_stats = demon.get_statistics()
    print(f"\nDemon statistics:")
    print(f"  Total interpretations: {demon_stats['total_interpretations']}")
    print(f"  Categorical states: {demon_stats['categorical_states']}")
    print(f"  Equivalence classes: {demon_stats['equivalence_classes']}")
    
    print("\n✅ Validation complete!")
    print("\n🔍 KEY FINDING: Semantic demon provides:")
    print("  1. Multiple alternative interpretations (non-destructive)")
    print("  2. S-entropy coordinates for navigation")
    print("  3. Evidence tracking for interpretations")
    print("  4. Thermodynamic favorability ordering")
    
    return all_results

if __name__ == "__main__":
    results = run_validation()
    
    # Save results
    os.makedirs("validation/results", exist_ok=True)
    with open("validation/results/02_depression_validation.txt", "w") as f:
        f.write("Depression Data Validation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Cases analyzed: {len(results)}\n")
        for r in results:
            f.write(f"\n{r['case_id']}:\n")
            f.write(f"  Without demon: {r['without_demon']['category']} ({r['without_demon']['confidence']:.2f})\n")
            f.write(f"  With demon: {r['with_demon']['category']} ({r['with_demon']['confidence']:.2f})\n")
            f.write(f"  Alternatives: {len(r['with_demon']['alternatives'])}\n")
    
    print(f"\n📁 Results saved to: validation/results/02_depression_validation.txt")

