# validation_s_entropy_amplification.py
"""
Validate S-entropy distance amplification
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.stats import ttest_1samp
from src.core.semantic_maxwell_demon import SemanticMaxwellDemon
from src.core.s_entropy import SEntropyCoordinates
import Levenshtein


def test_s_entropy_amplification():
    """Test semantic distance amplification through S-entropy coordinates"""

    print("=" * 70)
    print("TESTING S-ENTROPY DISTANCE AMPLIFICATION")
    print("=" * 70)

    smd = SemanticMaxwellDemon()

    # Test pairs
    test_pairs = [
        # Similar concepts
        ("depression", "anxiety"),
        ("theta oscillation", "alpha oscillation"),
        ("biomarker", "indicator"),

        # Dissimilar concepts
        ("depression", "happiness"),
        ("theta", "gamma"),
        ("treatment", "disease"),

        # Clinical pairs
        ("PLV 0.32", "PLV 0.77"),
        ("frontal theta", "occipital gamma"),
        ("responder", "non-responder")
    ]

    amplification_factors = []

    for text1, text2 in test_pairs:
        # Base distance (character-level)
        base_dist = Levenshtein.distance(text1, text2)

        try:
            # Get categorical states
            state1 = smd.get_categorical_state([text1])
            state2 = smd.get_categorical_state([text2])

            # Compute S-entropy coordinates
            s_coords1 = smd.compute_s_entropy(state1)
            s_coords2 = smd.compute_s_entropy(state2)

            # S-entropy distance
            s_dist = np.sqrt(
                (s_coords1.S_k - s_coords2.S_k)**2 +
                (s_coords1.S_theta - s_coords2.S_theta)**2
            )

            # Amplification factor
            gamma = s_dist / base_dist if base_dist > 0 else 0
            amplification_factors.append(gamma)

            print(f"\n{text1} <-> {text2}")
            print(f"  Base distance: {base_dist:.2f}")
            print(f"  S-entropy distance: {s_dist:.3f}")
            print(f"  Amplification: {gamma:.1f}×")

        except Exception as e:
            print(f"\n{text1} <-> {text2}")
            print(f"  ERROR: {e}")

    if amplification_factors:
        mean_gamma = np.mean(amplification_factors)
        std_gamma = np.std(amplification_factors)

        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Mean amplification: {mean_gamma:.1f}×")
        print(f"Std deviation: {std_gamma:.1f}")
        print(f"Range: [{min(amplification_factors):.1f}, {max(amplification_factors):.1f}]")

        # Check if amplification is significant (>1)
        significant = mean_gamma > 1.0
        print(f"Significant amplification: {significant}")

        return {
            'mean': mean_gamma,
            'std': std_gamma,
            'validated': significant,
            'amplification_factors': amplification_factors
        }
    else:
        print("\n❌ No valid measurements")
        return None


if __name__ == '__main__':
    results = test_s_entropy_amplification()
