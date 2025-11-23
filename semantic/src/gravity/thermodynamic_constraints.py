"""Thermodynamic constraints for semantic navigation"""

from ..core.s_entropy import SEntropyCoordinates

class ThermodynamicConstraint:
    """Constraints based on thermodynamic principles"""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def is_allowed(self, state: SEntropyCoordinates) -> bool:
        """Check if state satisfies thermodynamic constraints"""
        # Second law: entropy cannot decrease without work
        return state.S_e >= 0

    def boltzmann_probability(self, state: SEntropyCoordinates) -> float:
        """Boltzmann probability of state"""
        energy = state.magnitude
        return np.exp(-energy / (self.temperature + 1e-10))

    def transition_probability(self, from_state: SEntropyCoordinates,
                              to_state: SEntropyCoordinates) -> float:
        """Probability of transition between states"""
        delta_e = to_state.magnitude - from_state.magnitude

        if delta_e <= 0:
            return 1.0  # Spontaneous (downhill)
        else:
            return np.exp(-delta_e / self.temperature)  # Boltzmann factor

import numpy as np

