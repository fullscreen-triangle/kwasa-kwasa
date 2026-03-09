# validation_thought_injection.py
"""
Validate thought injection capability
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.bmd.thought_injection import ThoughtInjector
from src.bmd.bmd_state import BMDState


def test_thought_injection():
    """Test writing thoughts into user's mind"""

    print("=" * 70)
    print("TESTING THOUGHT INJECTION (THOUGHT WRITING)")
    print("=" * 70)

    injector = ThoughtInjector()

    # Test scenarios: semantic content → stimulus
    test_cases = [
        {
            'semantic_content': "Consider the relationship between theta oscillations and depression",
            'target_state': BMDState(
                state_name='contemplating',
                certainty=0.8,
                s_entropy_coords={'S_k': 0.5, 'S_theta': 0.3}
            ),
            'description': 'Inject contemplative thought about neural mechanisms'
        },
        {
            'semantic_content': "Novel biomarker discovered: BCAA ratio correlates with treatment response",
            'target_state': BMDState(
                state_name='discovering',
                certainty=0.9,
                s_entropy_coords={'S_k': 0.7, 'S_theta': 0.6}
            ),
            'description': 'Inject discovery of new biomarker'
        },
        {
            'semantic_content': "Age confounding detected in analysis - requires correction",
            'target_state': BMDState(
                state_name='correcting',
                certainty=0.85,
                s_entropy_coords={'S_k': 0.6, 'S_theta': 0.4}
            ),
            'description': 'Inject corrective thought about confounding'
        },
        {
            'semantic_content': "Multiple interpretations possible - explore alternative hypotheses",
            'target_state': BMDState(
                state_name='exploring',
                certainty=0.75,
                s_entropy_coords={'S_k': 0.4, 'S_theta': 0.7}
            ),
            'description': 'Inject exploratory thought about alternatives'
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n[Test Case {i+1}]")
        print(f"Description: {test_case['description']}")
        print(f"Semantic: {test_case['semantic_content'][:60]}...")
        print(f"Target state: {test_case['target_state'].state_name}")

        try:
            # Inject thought
            stimulus = injector.inject_thought(
                semantic_content=test_case['semantic_content'],
                target_state=test_case['target_state']
            )

            print(f"Generated stimulus type: {stimulus.type}")
            print(f"Stimulus intensity: {stimulus.intensity:.2f}")

            # Compute sufficiency
            sufficiency = injector.compute_sufficiency(
                stimulus=stimulus,
                target=test_case['target_state']
            )

            print(f"Sufficiency: {sufficiency:.2f}")

            # Validate
            sufficient = sufficiency >= 0.7
            appropriate_intensity = 0.3 <= stimulus.intensity <= 1.0

            results.append({
                'sufficient': sufficient,
                'appropriate_intensity': appropriate_intensity,
                'stimulus': stimulus,
                'sufficiency': sufficiency,
                'success': sufficient and appropriate_intensity
            })

            if sufficient:
                print("✓ Sufficient stimulus")
            else:
                print("✗ Insufficient stimulus")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'success': False, 'error': str(e)})

    # Summary
    success_count = sum(1 for r in results if r.get('success', False))
    accuracy = success_count / len(results)

    valid_results = [r for r in results if 'sufficiency' in r]
    if valid_results:
        avg_sufficiency = np.mean([r['sufficiency'] for r in valid_results])
    else:
        avg_sufficiency = 0.0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Success rate: {accuracy:.1%} ({success_count}/{len(results)})")
    print(f"Average sufficiency: {avg_sufficiency:.2f}")
    print(f"Thought injection validated: {accuracy >= 0.7 and avg_sufficiency >= 0.7}")

    return {
        'accuracy': accuracy,
        'avg_sufficiency': avg_sufficiency,
        'validated': accuracy >= 0.7 and avg_sufficiency >= 0.7,
        'results': results
    }


if __name__ == '__main__':
    results = test_thought_injection()
