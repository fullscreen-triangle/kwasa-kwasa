"""Convergence analysis for semantic navigation"""

import numpy as np
from typing import List
from ..core.s_entropy import SEntropyCoordinates

class ConvergenceAnalyzer:
    """Analyze convergence properties of semantic navigation"""

    @staticmethod
    def is_converging(trajectory: List[SEntropyCoordinates],
                     target: SEntropyCoordinates) -> bool:
        """Check if trajectory is converging to target"""
        if len(trajectory) < 3:
            return False

        # Check if distances are decreasing
        distances = [s.distance_to(target) for s in trajectory[-5:]]
        return all(distances[i] >= distances[i+1] for i in range(len(distances)-1))

    @staticmethod
    def estimate_steps_to_convergence(current: SEntropyCoordinates,
                                     target: SEntropyCoordinates,
                                     step_size: float = 0.1) -> int:
        """Estimate steps needed to reach target"""
        distance = current.distance_to(target)
        return int(np.ceil(distance / step_size))

    @staticmethod
    def convergence_rate(trajectory: List[SEntropyCoordinates],
                        target: SEntropyCoordinates) -> float:
        """Calculate convergence rate"""
        if len(trajectory) < 2:
            return 0.0

        distances = [s.distance_to(target) for s in trajectory]
        rates = [distances[i] - distances[i+1] for i in range(len(distances)-1)]

        return np.mean(rates) if rates else 0.0

