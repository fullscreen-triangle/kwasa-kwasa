"""Positional context encoding"""

import numpy as np

class PositionalEncoder:
    """Encode positional context for sequence information"""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, position: int) -> np.ndarray:
        """Sinusoidal positional encoding"""
        encoding = np.zeros(self.dim)

        for i in range(0, self.dim, 2):
            div_term = np.exp(i * -(np.log(10000.0) / self.dim))
            encoding[i] = np.sin(position * div_term)
            if i + 1 < self.dim:
                encoding[i + 1] = np.cos(position * div_term)

        return encoding

