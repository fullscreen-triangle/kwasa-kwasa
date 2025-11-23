"""Dual-strand complementary analysis"""

from typing import Tuple, List
import numpy as np

class DualStrandAnalyzer:
    """Analyze complementary information streams"""
    
    @staticmethod
    def analyze(strand1: List[float], strand2: List[float]) -> dict:
        """Analyze two complementary strands"""
        if len(strand1) != len(strand2):
            raise ValueError("Strands must have same length")
        
        # Correlation between strands
        correlation = np.corrcoef(strand1, strand2)[0, 1] if len(strand1) > 1 else 0.0
        
        # Complementarity (inverse correlation)
        complementarity = 1.0 - abs(correlation)
        
        # Information gain from dual analysis
        information_gain = complementarity * np.std(strand1 + strand2)
        
        return {
            "correlation": correlation,
            "complementarity": complementarity,
            "information_gain": information_gain
        }
    
    @staticmethod
    def extract_geometric_information(strand1: List, strand2: List) -> np.ndarray:
        """Extract geometric information from dual strands"""
        # Create 2D representation
        points = np.array(list(zip(strand1, strand2)))
        
        # Compute geometric properties
        centroid = np.mean(points, axis=0)
        dispersion = np.std(points, axis=0)
        
        return np.concatenate([centroid, dispersion])

