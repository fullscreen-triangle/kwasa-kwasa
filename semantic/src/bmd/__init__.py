"""
BMD Interface Modules - Bidirectional Categorical Communication

Enables thought-level communication between humans and AI through
BMD state detection and injection.
"""

from .bmd_state import BMDState, BMDStateVector
from .state_detection import BMDStateDetector, BehavioralSignal
from .thought_injection import ThoughtInjector, StimulusPattern
from .bidirectional_interface import BidirectionalDemon

__all__ = [
    "BMDState",
    "BMDStateVector",
    "BMDStateDetector",
    "BehavioralSignal",
    "ThoughtInjector",
    "StimulusPattern",
    "BidirectionalDemon",
]

