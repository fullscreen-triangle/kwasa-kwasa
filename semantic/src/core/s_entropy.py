"""
S-Entropy Coordinates: Sufficient Statistics for Categorical Navigation

The S-entropy coordinate system (S_k, S_t, S_e) provides a complete 
representation of categorical states that enables navigation without 
exhaustive exploration.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray


@dataclass
class SEntropyCoordinates:
    """
    Three-dimensional S-entropy coordinate system
    
    S_k: Knowledge entropy - information content/certainty
    S_t: Temporal entropy - time evolution/urgency  
    S_e: Evolution entropy - state transition/uncertainty
    """
    S_k: float  # Knowledge entropy [0, ∞)
    S_t: float  # Temporal entropy [0, ∞)
    S_e: float  # Evolution entropy [0, ∞)
    
    def __post_init__(self):
        """Validate entropy coordinates"""
        if self.S_k < 0 or self.S_t < 0 or self.S_e < 0:
            raise ValueError("S-entropy coordinates must be non-negative")
    
    @property
    def magnitude(self) -> float:
        """Total S-entropy distance from origin"""
        return np.sqrt(self.S_k**2 + self.S_t**2 + self.S_e**2)
    
    @property
    def vector(self) -> NDArray[np.float64]:
        """Return as numpy array"""
        return np.array([self.S_k, self.S_t, self.S_e])
    
    def distance_to(self, other: 'SEntropyCoordinates') -> float:
        """Categorical distance to another state"""
        return np.linalg.norm(self.vector - other.vector)
    
    def normalize(self) -> 'SEntropyCoordinates':
        """Return normalized unit vector in S-entropy space"""
        mag = self.magnitude
        if mag == 0:
            return SEntropyCoordinates(0.0, 0.0, 0.0)
        return SEntropyCoordinates(
            self.S_k / mag,
            self.S_t / mag,
            self.S_e / mag
        )
    
    def project_onto(self, direction: 'SEntropyCoordinates') -> float:
        """Project this coordinate onto a direction vector"""
        return np.dot(self.vector, direction.vector) / direction.magnitude
    
    def __add__(self, other: 'SEntropyCoordinates') -> 'SEntropyCoordinates':
        """Add two S-entropy coordinates"""
        return SEntropyCoordinates(
            self.S_k + other.S_k,
            self.S_t + other.S_t,
            self.S_e + other.S_e
        )
    
    def __sub__(self, other: 'SEntropyCoordinates') -> 'SEntropyCoordinates':
        """Subtract two S-entropy coordinates"""
        return SEntropyCoordinates(
            max(0, self.S_k - other.S_k),
            max(0, self.S_t - other.S_t),
            max(0, self.S_e - other.S_e)
        )
    
    def __mul__(self, scalar: float) -> 'SEntropyCoordinates':
        """Scalar multiplication"""
        return SEntropyCoordinates(
            self.S_k * scalar,
            self.S_t * scalar,
            self.S_e * scalar
        )
    
    def __repr__(self) -> str:
        return f"S(k={self.S_k:.3f}, t={self.S_t:.3f}, e={self.S_e:.3f})"


class SEntropyCalculator:
    """
    Calculate S-entropy coordinates from observations
    """
    
    @staticmethod
    def from_information_content(
        knowledge_bits: float,
        temporal_urgency: float,
        state_uncertainty: float
    ) -> SEntropyCoordinates:
        """
        Direct construction from measured quantities
        
        Args:
            knowledge_bits: Information content in bits
            temporal_urgency: Time-criticality [0,1]
            state_uncertainty: Probability uncertainty [0,1]
        """
        return SEntropyCoordinates(
            S_k=knowledge_bits,
            S_t=temporal_urgency * 10,  # Scale to reasonable range
            S_e=state_uncertainty * 10
        )
    
    @staticmethod
    def from_probability_distribution(
        prob_dist: NDArray[np.float64],
        time_weight: float = 1.0
    ) -> SEntropyCoordinates:
        """
        Calculate from probability distribution over states
        
        Args:
            prob_dist: Probability distribution over categorical states
            time_weight: Temporal weighting factor
        """
        # Shannon entropy for knowledge
        prob_dist = prob_dist / np.sum(prob_dist)  # Normalize
        S_k = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        
        # Temporal entropy from distribution variance
        S_t = time_weight * np.var(prob_dist)
        
        # Evolution entropy from entropy rate
        S_e = S_k * (1 - np.max(prob_dist))  # Higher when uncertain
        
        return SEntropyCoordinates(S_k, S_t, S_e)
    
    @staticmethod
    def from_trajectory(
        states: List[SEntropyCoordinates],
        window_size: int = 5
    ) -> SEntropyCoordinates:
        """
        Calculate instantaneous S-entropy from trajectory
        
        Uses recent state history to compute current coordinates
        """
        if len(states) == 0:
            return SEntropyCoordinates(0.0, 0.0, 0.0)
        
        recent = states[-window_size:]
        
        # Knowledge: Average recent S_k
        S_k = np.mean([s.S_k for s in recent])
        
        # Temporal: Rate of S_k change
        if len(recent) > 1:
            S_t = abs(recent[-1].S_k - recent[0].S_k) / len(recent)
        else:
            S_t = recent[-1].S_t
        
        # Evolution: Variance in trajectory
        S_e = np.std([s.magnitude for s in recent])
        
        return SEntropyCoordinates(S_k, S_t, S_e)


class CategoricalDistance:
    """
    Utilities for categorical distance computation
    """
    
    @staticmethod
    def euclidean(s1: SEntropyCoordinates, s2: SEntropyCoordinates) -> float:
        """Standard Euclidean distance in S-entropy space"""
        return s1.distance_to(s2)
    
    @staticmethod
    def manhattan(s1: SEntropyCoordinates, s2: SEntropyCoordinates) -> float:
        """Manhattan (L1) distance"""
        return (abs(s1.S_k - s2.S_k) + 
                abs(s1.S_t - s2.S_t) + 
                abs(s1.S_e - s2.S_e))
    
    @staticmethod
    def weighted(
        s1: SEntropyCoordinates, 
        s2: SEntropyCoordinates,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> float:
        """Weighted distance with per-dimension importance"""
        w_k, w_t, w_e = weights
        return np.sqrt(
            w_k * (s1.S_k - s2.S_k)**2 +
            w_t * (s1.S_t - s2.S_t)**2 +
            w_e * (s1.S_e - s2.S_e)**2
        )
    
    @staticmethod
    def cosine_similarity(s1: SEntropyCoordinates, s2: SEntropyCoordinates) -> float:
        """
        Cosine similarity in S-entropy space
        Returns: [0, 1] where 1 is identical direction
        """
        dot = np.dot(s1.vector, s2.vector)
        norms = s1.magnitude * s2.magnitude
        if norms == 0:
            return 0.0
        return dot / norms
    
    @staticmethod
    def are_equivalent(
        s1: SEntropyCoordinates,
        s2: SEntropyCoordinates,
        threshold: float = 0.1
    ) -> bool:
        """
        Check if two states are categorically equivalent
        
        Args:
            threshold: Maximum distance for equivalence
        """
        return s1.distance_to(s2) < threshold


def interpolate_s_entropy(
    s_start: SEntropyCoordinates,
    s_end: SEntropyCoordinates,
    alpha: float
) -> SEntropyCoordinates:
    """
    Linear interpolation between two S-entropy coordinates
    
    Args:
        s_start: Starting coordinate
        s_end: Ending coordinate
        alpha: Interpolation factor [0, 1]
    """
    return SEntropyCoordinates(
        S_k=s_start.S_k + alpha * (s_end.S_k - s_start.S_k),
        S_t=s_start.S_t + alpha * (s_end.S_t - s_start.S_t),
        S_e=s_start.S_e + alpha * (s_end.S_e - s_start.S_e)
    )

