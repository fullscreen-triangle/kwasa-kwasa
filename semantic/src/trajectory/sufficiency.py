"""Sufficiency calculation for thought injection"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from ..bmd.bmd_state import BMDState
from ..core.categorical_state import CategoricalState

@dataclass
class SufficientStimulus:
    """Minimal stimulus sufficient for completion"""
    stimulus: Any
    completion_probability: float
    information_content: float
    
    @property
    def is_sufficient(self) -> bool:
        return self.completion_probability > 0.7

class SufficiencyCalculator:
    """Calculate sufficient stimuli"""
    
    @staticmethod
    def compute(target: CategoricalState, current: BMDState) -> SufficientStimulus:
        """Compute sufficient stimulus"""
        distance = current.s_entropy.distance_to(target.s_entropy)
        prob = 1.0 / (1.0 + distance / 5.0)
        info = distance * 0.5
        
        return SufficientStimulus(
            stimulus={"text": target.meaning[:20]},
            completion_probability=prob,
            information_content=info
        )

