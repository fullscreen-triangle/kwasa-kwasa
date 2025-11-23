"""Information density computation in semantic space"""

import numpy as np
from typing import List
from ..core.categorical_state import CategoricalState

class InformationDensity:
    """Compute information density in regions of semantic space"""

    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth

    def compute(self, states: List[CategoricalState], query_state: CategoricalState) -> float:
        """Compute information density around query state"""
        if not states:
            return 0.0

        # Kernel density estimation
        density = 0.0
        for state in states:
            distance = query_state.distance_to(state)
            kernel_value = np.exp(-(distance**2) / (2 * self.bandwidth**2))
            density += kernel_value * state.information_content

        return density / len(states)

    def find_high_density_regions(self, states: List[CategoricalState],
                                  threshold: float = 5.0) -> List[CategoricalState]:
        """Find states in high-density regions"""
        high_density = []

        for state in states:
            density = self.compute(states, state)
            if density > threshold:
                high_density.append(state)

        return high_density

