# validation_bmd_detection.py
"""
Validate BMD state detection from behavior
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.bmd.state_detection import BMDStateDetector
from src.bmd.bmd_state import BMDState


def test_bmd_state_detection():
    """Test thought reading capability"""

    print("=" * 70)
    print("TESTING BMD STATE DETECTION (THOUGHT READING)")
    print("=" * 70)

    detector = BMDStateDetector()

    # Test scenarios: behavior → BMD state
    test_cases = [
        {
            'behavior_signals': {
                'mouse_trajectory': [(100, 200), (150, 250), (200, 300)],
                'keystroke_pattern': ['d', 'e', 'p', 'r', 'e'],
                'pause_duration': 2.3,
                'gaze_pattern': 'scanning',
                'interaction_velocity': 0.5
            },
            'expected_state': 'searching',
            'description': 'User searching for information'
        },
        {
            'behavior_signals': {
                'mouse_trajectory': [(500, 500)],  # Static
                'keystroke_pattern': [],
                'pause_duration': 5.1,
                'gaze_pattern': 'fixated',
                'interaction_velocity': 0.0
            },
            'expected_state': 'contemplating',
            'description': 'User contemplating/thinking'
        },
        {
            'behavior_signals': {
                'mouse_trajectory': [(x, y) for x, y in zip(range(0, 1000, 10), range(0, 1000, 10))],
                'keystroke_pattern': list("this is a complete thought"),
                'pause_duration': 0.2,
                'gaze_pattern': 'linear',
                'interaction_velocity': 0.9
            },
            'expected_state': 'expressing',
            'description': 'User expressing complete thought'
        },
        {
            'behavior_signals': {
                'mouse_trajectory': [(100, 100), (200, 200), (100, 100), (200, 200)],
                'keystroke_pattern': ['a', 'b', 'c', '<backspace>', '<backspace>'],
                'pause_duration': 1.5,
                'gaze_pattern': 'oscillating',
                'interaction_velocity': 0.3
            },
            'expected_state': 'uncertain',
            'description': 'User uncertain/revising'
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n[Test Case {i+1}]")
        print(f"Description: {test_case['description']}")
        print(f"Expected state: {test_case['expected_state']}")

        try:
            # Detect BMD state from behavior
            detected_state = detector.detect_from_behavior(
                behavior_signals=test_case['behavior_signals']
            )

            print(f"Detected state: {detected_state.state_name}")
            print(f"Certainty: {detected_state.certainty:.2f}")
            print(f"S-entropy coords: {detected_state.s_entropy_coords}")

            # Validate
            correct = detected_state.state_name == test_case['expected_state']
            high_certainty = detected_state.certainty >= 0.6

            results.append({
                'correct': correct,
                'high_certainty': high_certainty,
                'detected_state': detected_state,
                'success': correct and high_certainty
            })

            if correct:
                print("✓ Correct detection")
            else:
                print(f"✗ Incorrect (expected: {test_case['expected_state']})")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({'success': False, 'error': str(e)})

    # Summary
    success_count = sum(1 for r in results if r.get('success', False))
    accuracy = success_count / len(results)

    valid_results = [r for r in results if 'detected_state' in r]
    if valid_results:
        avg_certainty = np.mean([r['detected_state'].certainty for r in valid_results])
    else:
        avg_certainty = 0.0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Detection accuracy: {accuracy:.1%} ({success_count}/{len(results)})")
    print(f"Average certainty: {avg_certainty:.2f}")
    print(f"Thought reading validated: {accuracy >= 0.7}")

    return {
        'accuracy': accuracy,
        'avg_certainty': avg_certainty,
        'validated': accuracy >= 0.7,
        'results': results
    }


if __name__ == '__main__':
    results = test_bmd_state_detection()
