"""BMD Trajectory tracking and analysis"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from ..bmd.bmd_state import BMDState
from ..core.categorical_state import CategoricalState
from ..core.s_entropy import SEntropyCoordinates

@dataclass
class BMDTrajectory:
    """User's BMD state evolution over time"""
    timestamps: List[float]
    bmd_states: List[BMDState]
    semantic_states: List[CategoricalState]
    
    def predict_destination(self) -> Tuple[CategoricalState, float]:
        """Predict where trajectory is heading"""
        if len(self.bmd_states) < 3:
            return self.semantic_states[-1] if self.semantic_states else None, 0.0
        
        # Linear extrapolation
        recent = self.bmd_states[-5:]
        s_k_trend = np.polyfit(range(len(recent)), [s.s_entropy.S_k for s in recent], 1)[0]
        confidence = min(1.0, abs(s_k_trend) / 2.0)
        
        return self.semantic_states[-1] if self.semantic_states else None, confidence
    
    def is_query_forming(self) -> bool:
        """Check if query is forming (S_k increasing)"""
        if len(self.bmd_states) < 5:
            return False
        return self.bmd_states[-1].s_entropy.S_k > self.bmd_states[-5].s_entropy.S_k
    
    def uncertainty_level(self) -> float:
        """Current uncertainty (S_e)"""
        return self.bmd_states[-1].s_entropy.S_e if self.bmd_states else 5.0


class TrajectoryAnalyzer:
    """Analyze BMD trajectories"""
    
    @staticmethod
    def detect_pattern(trajectory: BMDTrajectory) -> str:
        """Detect trajectory pattern"""
        if len(trajectory.bmd_states) < 3:
            return "insufficient_data"
        
        states = trajectory.bmd_states[-10:]
        s_e_values = [s.s_entropy.S_e for s in states]
        
        if np.mean(s_e_values) > 7.0:
            return "confused"
        elif np.std(s_e_values) < 1.0 and np.mean(s_e_values) < 3.0:
            return "flow"
        elif s_e_values[-1] > s_e_values[0]:
            return "uncertainty_increasing"
        else:
            return "steady"

