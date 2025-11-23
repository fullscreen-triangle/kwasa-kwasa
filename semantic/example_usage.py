"""
Example usage of Semantic Maxwell Demon package

Demonstrates both semantic filtering and bidirectional categorical telepathy.
"""

from src.core import SemanticMaxwellDemon, SemanticLens, SEntropyCoordinates
from src.bmd import BidirectionalDemon, BMDStateDetector, BehavioralSignal
from src.calibration import CalibrationEngine

def example_semantic_filtering():
    """Example: Basic semantic filtering"""
    print("=== Semantic Filtering Example ===\n")
    
    # Create demon
    demon = SemanticMaxwellDemon()
    
    # Observation to interpret
    observation = {
        "text": "Patient reports low mood, decreased energy, and sleep difficulties",
        "duration": 14,  # days
        "severity": 7  # out of 10
    }
    
    # Filter through psychiatric lens
    interpretation = demon.filter(observation, lens=SemanticLens.PSYCHIATRIC)
    
    print(f"Primary interpretation: {interpretation.primary_state.category}")
    print(f"Meaning: {interpretation.primary_state.meaning}")
    print(f"Confidence: {interpretation.primary_state.confidence:.2f}")
    print(f"S-entropy: {interpretation.primary_state.s_entropy}")
    print(f"Alternatives: {len(interpretation.alternative_states)}")
    
    # Compare across multiple lenses
    print("\n--- Multi-lens comparison ---")
    comparison = demon.compare_interpretations(observation)
    print(f"Agreement across lenses: {comparison['agreement']:.2f}")
    print(f"Consensus category: {comparison['consensus_category']}")
    print(f"Unique interpretations: {len(comparison['unique_categories'])}")


def example_bidirectional_telepathy():
    """Example: Bidirectional categorical communication"""
    print("\n\n=== Bidirectional Telepathy Example ===\n")
    
    # Create bidirectional interface
    user_id = "user_001"
    bidirectional = BidirectionalDemon(user_id=user_id)
    
    # Start conversation
    bidirectional.start_conversation()
    print("Conversation started!")
    
    # Simulate behavioral signals
    print("\n--- Processing behavioral signals ---")
    
    # User typing with pauses (forming a query)
    signals = [
        BehavioralSignal("keystroke", 1.0, "h"),
        BehavioralSignal("keystroke", 1.1, "o"),
        BehavioralSignal("keystroke", 1.2, "w"),
        BehavioralSignal("pause", 1.8, 0.6),  # 600ms pause
        BehavioralSignal("keystroke", 2.5, "d"),
    ]
    
    for signal in signals:
        response = bidirectional.process_signal(signal)
        if response:
            print(f"AI injecting thought: {response.modality.value}")
            print(f"  Completion probability: {response.completion_probability:.2f}")
    
    # Get statistics
    stats = bidirectional.get_statistics()
    print(f"\n--- Session statistics ---")
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Success rate: {stats['success_rate']:.2%}")


def example_user_calibration():
    """Example: User calibration"""
    print("\n\n=== User Calibration Example ===\n")
    
    engine = CalibrationEngine()
    user_id = "user_002"
    
    # Calibrate from interaction history
    history = [
        {
            "keystroke_timing": [0.1, 0.12, 0.15, 0.11],
            "pause_durations": [0.5, 0.3],
            "window_switches": 2,
            "cursor_movements": [(100, 200, 0.5), (150, 250, 0.6)],
            "action_type": "query",
            "action_description": "User asked about concept"
        }
    ] * 10  # Simulate 10 interactions
    
    print("Calibrating user model from history...")
    model = engine.calibrate_from_history(user_id, history)
    
    print(f"Model created: {model}")
    print(f"Total observations: {model.total_observations}")
    print(f"Success rate: {model.get_success_rate():.2%}")


def example_complete_workflow():
    """Example: Complete workflow combining all components"""
    print("\n\n=== Complete Workflow Example ===\n")
    
    # 1. Calibrate user
    print("1. Calibrating user...")
    engine = CalibrationEngine()
    user_model = engine.run_calibration("user_003", num_trials=20)
    print(f"   User model: {user_model}")
    
    # 2. Create bidirectional interface with calibrated model
    print("\n2. Creating bidirectional interface...")
    bidirectional = BidirectionalDemon(
        user_id="user_003",
        detector=BMDStateDetector()
    )
    bidirectional.start_conversation()
    print("   Interface ready!")
    
    # 3. Real-time interaction loop (simulated)
    print("\n3. Simulating interaction...")
    for i in range(5):
        signal = BehavioralSignal("keystroke", float(i), f"char_{i}")
        response = bidirectional.process_signal(signal)
        if response:
            print(f"   Step {i}: Injecting thought (P={response.completion_probability:.2f})")
    
    # 4. Get final stats
    stats = bidirectional.get_statistics()
    print(f"\n4. Final statistics:")
    print(f"   Interactions: {stats['total_interactions']}")
    print(f"   Success rate: {stats['success_rate']:.2%}")


if __name__ == "__main__":
    print("Semantic Maxwell Demon - Example Usage\n")
    print("=" * 60)
    
    # Run examples
    example_semantic_filtering()
    example_bidirectional_telepathy()
    example_user_calibration()
    example_complete_workflow()
    
    print("\n" + "=" * 60)
    print("\nAll examples completed successfully!")
    print("\nNext steps:")
    print("  1. Install package: pip install -e .")
    print("  2. Import: from semantic_maxwell_demon import *")
    print("  3. Build your application!")

