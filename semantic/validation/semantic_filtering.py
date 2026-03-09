# validation_semantic_filtering.py
"""
Validate semantic filtering and categorical state detection
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.core.semantic_maxwell_demon import SemanticMaxwellDemon
from src.core.categorical_state import CategoricalState
from src.core.s_entropy import SEntropyCoordinates


def test_semantic_filtering():
    """Test semantic filtering capability"""

    print("=" * 70)
    print("TESTING SEMANTIC FILTERING")
    print("=" * 70)

    smd = SemanticMaxwellDemon()

    # Test cases: observations → filtered semantic states
    test_cases = [
        {
            'observations': [
                "Patient reports persistent sadness",
                "Sleep disturbances for 3 weeks",
                "Loss of interest in activities",
                "Difficulty concentrating"
            ],
            'context': {'domain': 'clinical', 'task': 'diagnosis'},
            'expected_category': 'depression'
        },
        {
            'observations': [
                "Elevated theta band power",
                "PLV: 0.32 in frontal regions",
                "Reduced gamma oscillations",
                "Phase-amplitude coupling disrupted"
            ],
            'context': {'domain': 'neuroscience', 'task': 'biomarker_detection'},
            'expected_category': 'neural_dysfunction'
        },
        {
            'observations': [
                "High entropy in semantic space",
                "Multiple competing interpretations",
                "Ambiguous categorical boundaries",
                "Requires deeper exploration"
            ],
            'context': {'domain': 'meta', 'task': 'semantic_analysis'},
            'expected_category': 'high_uncertainty'
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n[Test Case {i+1}]")
        print(f"Observations: {len(test_case['observations'])} items")
        print(f"Context: {test_case['context']}")

        try:
            # Semantic filtering
            filter_result = smd.filter(
                observations=test_case['observations'],
                context=test_case['context']
            )

            print(f"Filtered state: {filter_result.state}")
            print(f"Confidence: {filter_result.confidence:.2f}")

            # Get categorical state
            categorical_state = smd.get_categorical_state(
                observations=test_case['observations']
            )

            print(f"Categorical state: {categorical_state.category}")
            print(f"S-entropy: {categorical_state.s_entropy}")

            # Compute S-entropy coordinates
            s_coords = smd.compute_s_entropy(categorical_state)

            print(f"S-entropy coordinates: S_k={s_coords.S_k:.3f}, S_θ={s_coords.S_theta:.3f}")

            results.append({
                'filter_result': filter_result,
                'categorical_state': categorical_state,
                's_coords': s_coords,
                'success': True
            })

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'success': False, 'error': str(e)})

    # Summary
    success_count = sum(1 for r in results if r.get('success', False))
    success_rate = success_count / len(results)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Success rate: {success_rate:.1%} ({success_count}/{len(results)})")
    print(f"Semantic filtering validated: {success_rate >= 0.8}")

    return {
        'success_rate': success_rate,
        'results': results,
        'validated': success_rate >= 0.8
    }


if __name__ == '__main__':
    results = test_semantic_filtering()
