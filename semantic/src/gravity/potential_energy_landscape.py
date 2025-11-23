"""Potential energy landscape in semantic space"""

import numpy as np
from ..core.s_entropy import SEntropyCoordinates

class PotentialEnergyLandscape:
    """Semantic gravity field defined by potential energy"""

    def __init__(self, attractors: list = None):
        self.attractors = attractors or []

    def potential(self, position: SEntropyCoordinates) -> float:
        """Compute potential energy at position"""
        if not self.attractors:
            return position.magnitude

        # Sum of inverse distances to attractors
        total = 0.0
        for attractor in self.attractors:
            distance = position.distance_to(attractor)
            total += 1.0 / (1.0 + distance)

        return -total  # Negative for attractive potential

    def force(self, position: SEntropyCoordinates) -> SEntropyCoordinates:
        """Compute force (gradient of potential) at position"""
        epsilon = 0.01

        # Numerical gradient
        p = position
        f_k = -(self.potential(SEntropyCoordinates(p.S_k + epsilon, p.S_t, p.S_e)) -
                self.potential(p)) / epsilon
        f_t = -(self.potential(SEntropyCoordinates(p.S_k, p.S_t + epsilon, p.S_e)) -
                self.potential(p)) / epsilon
        f_e = -(self.potential(SEntropyCoordinates(p.S_k, p.S_t, p.S_e + epsilon)) -
                self.potential(p)) / epsilon

        return SEntropyCoordinates(f_k, f_t, f_e)

