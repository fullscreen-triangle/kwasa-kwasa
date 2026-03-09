"""
Validation Script 1: Test Core Components

Tests that all core modules load and basic operations work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import (
    SEntropyCoordinates, SEntropyCalculator, CategoricalDistance,
    CategoricalState, SemanticLens, SemanticMaxwellDemon
)
from src.bmd import BMDState, BMDStateDetector, ThoughtInjector, BidirectionalDemon

def test_s_entropy():
    """Test S-entropy coordinate system"""
    print("=" * 60)
    print("TEST 1: S-Entropy Coordinates")
    print("=" * 60)
    
    # Create coordinates
    s1 = SEntropyCoordinates(S_k=5.0, S_t=3.0, S_e=4.0)
    s2 = SEntropyCoordinates(S_k=7.0, S_t=4.0, S_e=3.0)
    
    print(f"Coordinate 1: {s1}")
    print(f"Coordinate 2: {s2}")
    print(f"Magnitude 1: {s1.magnitude:.2f}")
    print(f"Distance between: {s1.distance_to(s2):.2f}")
    print(f"Cosine similarity: {CategoricalDistance.cosine_similarity(s1, s2):.2f}")
    
    print("✅ S-entropy coordinates working!\n")
    return True

def test_categorical_states():
    """Test categorical state representation"""
    print("=" * 60)
    print("TEST 2: Categorical States")
    print("=" * 60)
    
    # Create state
    state = CategoricalState(
        category="test_category",
        meaning="Test categorical state",
        confidence=0.8,
        s_entropy=SEntropyCoordinates(5.0, 3.0, 4.0),
        evidence=["evidence_1", "evidence_2"]
    )
    
    print(f"State: {state}")
    print(f"Category: {state.category}")
    print(f"Confidence: {state.confidence}")
    print(f"S-distance: {state.s_distance:.2f}")
    print(f"Is uncertain: {state.is_uncertain}")
    print(f"Is urgent: {state.is_urgent}")
    
    print("✅ Categorical states working!\n")
    return True

def test_semantic_demon():
    """Test Semantic Maxwell Demon"""
    print("=" * 60)
    print("TEST 3: Semantic Maxwell Demon")
    print("=" * 60)
    
    # Create demon
    demon = SemanticMaxwellDemon()
    
    # Test observation
    observation = {
        "text": "Patient shows symptoms of low mood and fatigue",
        "duration": 14,
        "severity": 7
    }
    
    print(f"Observation: {observation['text']}")
    print(f"Duration: {observation['duration']} days")
    print(f"Severity: {observation['severity']}/10\n")
    
    # Filter through psychiatric lens
    interpretation = demon.filter(observation, lens=SemanticLens.PSYCHIATRIC)
    
    print(f"Primary interpretation: {interpretation.primary_state.category}")
    print(f"Confidence: {interpretation.primary_state.confidence:.2f}")
    print(f"S-entropy: {interpretation.primary_state.s_entropy}")
    print(f"Alternatives: {len(interpretation.alternative_states)}")
    
    # Multi-lens comparison
    print("\n--- Testing multi-lens filtering ---")
    comparison = demon.compare_interpretations(observation)
    print(f"Agreement: {comparison['agreement']:.2f}")
    print(f"Unique interpretations: {len(comparison['unique_categories'])}")
    
    print("✅ Semantic Maxwell Demon working!\n")
    return True

def test_bmd_state():
    """Test BMD state representation"""
    print("=" * 60)
    print("TEST 4: BMD State Detection")
    print("=" * 60)
    
    # Create BMD state
    bmd_state = BMDState(
        s_entropy=SEntropyCoordinates(S_k=6.0, S_t=4.0, S_e=7.0),
        confidence=0.85
    )
    
    print(f"BMD State: {bmd_state}")
    print(f"Knowledge level (S_k): {bmd_state.knowledge_level:.2f}")
    print(f"Temporal state (S_t): {bmd_state.temporal_state:.2f}")
    print(f"Uncertainty (S_e): {bmd_state.uncertainty:.2f}")
    print(f"Query forming: {bmd_state.is_query_forming()}")
    print(f"Is uncertain: {bmd_state.is_uncertain()}")
    
    print("✅ BMD state representation working!\n")
    return True

def test_thought_injector():
    """Test thought injection"""
    print("=" * 60)
    print("TEST 5: Thought Injection")
    print("=" * 60)
    
    # Create injector
    injector = ThoughtInjector(user_id="test_user")
    
    # Create target thought
    target = CategoricalState(
        category="understanding",
        meaning="User should understand concept X",
        confidence=0.9,
        s_entropy=SEntropyCoordinates(4.0, 3.0, 2.0)
    )
    
    # Create user state
    user_state = BMDState(
        s_entropy=SEntropyCoordinates(3.0, 3.0, 5.0)
    )
    
    print(f"Target thought: {target.category}")
    print(f"User state: S_e={user_state.uncertainty:.1f} (uncertain)")
    
    # Inject thought
    stimulus = injector.inject(target, user_state)
    
    print(f"\nGenerated stimulus:")
    print(f"  Modality: {stimulus.modality.value}")
    print(f"  Intensity: {stimulus.intensity:.2f}")
    print(f"  Duration: {stimulus.duration:.2f}s")
    print(f"  Completion probability: {stimulus.completion_probability:.2f}")
    
    print("✅ Thought injection working!\n")
    return True

def test_bidirectional_interface():
    """Test bidirectional demon"""
    print("=" * 60)
    print("TEST 6: Bidirectional Interface")
    print("=" * 60)
    
    # Create interface
    bidirectional = BidirectionalDemon(user_id="test_user")
    
    # Start conversation
    bidirectional.start_conversation()
    print("Conversation started!")
    
    # Get statistics
    stats = bidirectional.get_statistics()
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Current state: {stats['detector_stats']['current_state']}")
    
    # Stop conversation
    bidirectional.stop_conversation()
    print("Conversation stopped!")
    
    print("✅ Bidirectional interface working!\n")
    return True

def run_all_tests():
    """Run all core component tests"""
    print("\n" + "=" * 60)
    print("SEMANTIC MAXWELL DEMON - CORE COMPONENT VALIDATION")
    print("=" * 60 + "\n")
    
    results = []
    
    try:
        results.append(("S-Entropy", test_s_entropy()))
    except Exception as e:
        print(f"❌ S-Entropy test failed: {e}\n")
        results.append(("S-Entropy", False))
    
    try:
        results.append(("Categorical States", test_categorical_states()))
    except Exception as e:
        print(f"❌ Categorical states test failed: {e}\n")
        results.append(("Categorical States", False))
    
    try:
        results.append(("Semantic Demon", test_semantic_demon()))
    except Exception as e:
        print(f"❌ Semantic demon test failed: {e}\n")
        results.append(("Semantic Demon", False))
    
    try:
        results.append(("BMD State", test_bmd_state()))
    except Exception as e:
        print(f"❌ BMD state test failed: {e}\n")
        results.append(("BMD State", False))
    
    try:
        results.append(("Thought Injection", test_thought_injector()))
    except Exception as e:
        print(f"❌ Thought injection test failed: {e}\n")
        results.append(("Thought Injection", False))
    
    try:
        results.append(("Bidirectional Interface", test_bidirectional_interface()))
    except Exception as e:
        print(f"❌ Bidirectional interface test failed: {e}\n")
        results.append(("Bidirectional Interface", False))
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All core components working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    # Save results
    os.makedirs("validation/results", exist_ok=True)
    with open("validation/results/01_core_test_results.txt", "w") as f:
        f.write("Core Component Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Status: {'PASSED' if success else 'FAILED'}\n")
    
    sys.exit(0 if success else 1)

