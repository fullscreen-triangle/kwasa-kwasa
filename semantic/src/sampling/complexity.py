"""Complexity analysis for semantic operations"""

import numpy as np
from typing import List
from ..core.categorical_state import CategoricalState

class ComplexityAnalyzer:
    """Analyze computational complexity of semantic operations"""

    @staticmethod
    def exhaustive_search_complexity(n: int) -> float:
        """Complexity of exhaustive search: O(n!)"""
        return float(np.math.factorial(min(n, 20)))  # Cap at 20 to avoid overflow

    @staticmethod
    def gravity_guided_complexity(n: int) -> float:
        """Complexity of gravity-guided navigation: O(log n)"""
        return np.log2(max(1, n))

    @staticmethod
    def compression_ratio(states: List[CategoricalState]) -> float:
        """Compute compression ratio achieved"""
        if not states:
            return 1.0

        # Original space size (all possible states)
        original_size = 10.0 ** 6  # Assume million possible states

        # Explored states
        explored_size = len(states)

        return original_size / max(1, explored_size)

    @staticmethod
    def speedup_factor(exhaustive_time: float, optimized_time: float) -> float:
        """Calculate speedup from optimization"""
        return exhaustive_time / max(optimized_time, 1e-10)

