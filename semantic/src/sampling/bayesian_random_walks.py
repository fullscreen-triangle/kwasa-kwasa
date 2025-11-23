"""Bayesian random walk in semantic space"""

import numpy as np
from typing import Optional
from ..core.s_entropy import SEntropyCoordinates
from ..core.categorical_state import CategoricalState

class BayesianRandomWalk:
    """Constrained random walk guided by Bayesian inference"""

    def __init__(self, step_size: float = 0.1, temperature: float = 1.0):
        self.step_size = step_size
        self.temperature = temperature
        self.current_position: Optional[SEntropyCoordinates] = None

    def step(self, current: SEntropyCoordinates, target: SEntropyCoordinates) -> SEntropyCoordinates:
        """Take one Bayesian step toward target"""
        # Direction to target
        direction = SEntropyCoordinates(
            target.S_k - current.S_k,
            target.S_t - current.S_t,
            target.S_e - current.S_e
        ).normalize()

        # Add random exploration
        noise = np.random.randn(3) * self.temperature

        # Combined step
        new_s = SEntropyCoordinates(
            max(0, current.S_k + direction.S_k * self.step_size + noise[0]),
            max(0, current.S_t + direction.S_t * self.step_size + noise[1]),
            max(0, current.S_e + direction.S_e * self.step_size + noise[2])
        )

        self.current_position = new_s
        return new_s

    def walk(self, start: SEntropyCoordinates, target: SEntropyCoordinates,
             max_steps: int = 100) -> list:
        """Complete walk from start to target"""
        trajectory = [start]
        current = start

        for _ in range(max_steps):
            current = self.step(current, target)
            trajectory.append(current)

            if current.distance_to(target) < 0.5:
                break

        return trajectory

