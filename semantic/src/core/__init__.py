"""
Core Semantic Maxwell Demon modules
"""

from .s_entropy import (
    SEntropyCoordinates,
    SEntropyCalculator,
    CategoricalDistance,
    interpolate_s_entropy
)
from .categorical_state import (
    CategoricalState,
    Interpretation,
    SemanticLens,
    CategoricalEquivalenceClass,
    CategoricalSpace
)
from .semantic_maxwell_demon import SemanticMaxwellDemon

__all__ = [
    "SEntropyCoordinates",
    "SEntropyCalculator",
    "CategoricalDistance",
    "interpolate_s_entropy",
    "CategoricalState",
    "Interpretation",
    "SemanticLens",
    "CategoricalEquivalenceClass",
    "CategoricalSpace",
    "SemanticMaxwellDemon",
]

