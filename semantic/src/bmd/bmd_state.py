"""
BMD State Representation

Represents the biological Maxwell demon state of a user,
mapped to S-entropy coordinates.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
from numpy.typing import NDArray
import time

from ..core.s_entropy import SEntropyCoordinates


@dataclass
class BMDState:
    """
    Biological Maxwell Demon state
    
    Represents user's current cognitive/semantic state mapped to
    S-entropy coordinates for categorical navigation.
    """
    s_entropy: SEntropyCoordinates
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # Confidence in state measurement
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def knowledge_level(self) -> float:
        """S_k: Current knowledge/information state"""
        return self.s_entropy.S_k
    
    @property
    def temporal_state(self) -> float:
        """S_t: Temporal urgency/time-criticality"""
        return self.s_entropy.S_t
    
    @property
    def uncertainty(self) -> float:
        """S_e: Evolution/state uncertainty"""
        return self.s_entropy.S_e
    
    def is_query_forming(self) -> bool:
        """High S_k with high S_e suggests query formation"""
        return self.knowledge_level > 3.0 and self.uncertainty > 4.0
    
    def is_uncertain(self) -> bool:
        """High uncertainty state"""
        return self.uncertainty > 5.0
    
    def is_urgent(self) -> bool:
        """High temporal urgency"""
        return self.temporal_state > 5.0
    
    def distance_to(self, other: 'BMDState') -> float:
        """Categorical distance to another BMD state"""
        return self.s_entropy.distance_to(other.s_entropy)
    
    def __repr__(self) -> str:
        return f"BMDState(S_k={self.knowledge_level:.2f}, S_t={self.temporal_state:.2f}, S_e={self.uncertainty:.2f})"


@dataclass
class BMDStateVector:
    """
    Extended BMD state with additional behavioral features
    
    Includes both S-entropy and raw behavioral signals for
    richer state representation.
    """
    bmd_state: BMDState
    behavioral_features: Dict[str, float] = field(default_factory=dict)
    raw_signals: Dict[str, List[float]] = field(default_factory=dict)
    
    @classmethod
    def from_behaviors(
        cls,
        keystroke_timing: List[float],
        cursor_movements: List[tuple],
        window_switches: int,
        pause_durations: List[float],
        **other_signals
    ) -> 'BMDStateVector':
        """
        Construct BMD state vector from behavioral signals
        
        Args:
            keystroke_timing: Inter-keystroke intervals
            cursor_movements: (x, y, timestamp) tuples
            window_switches: Count of window/tab switches
            pause_durations: Lengths of typing pauses
            **other_signals: Additional behavioral signals
        """
        # Calculate S-entropy from behaviors
        s_k = cls._calculate_knowledge_entropy(keystroke_timing, pause_durations)
        s_t = cls._calculate_temporal_entropy(window_switches, keystroke_timing)
        s_e = cls._calculate_evolution_entropy(pause_durations, keystroke_timing)
        
        s_entropy = SEntropyCoordinates(s_k, s_t, s_e)
        bmd_state = BMDState(s_entropy=s_entropy, confidence=0.8)
        
        # Extract behavioral features
        features = {
            "avg_keystroke_interval": np.mean(keystroke_timing) if keystroke_timing else 0,
            "keystroke_variance": np.var(keystroke_timing) if keystroke_timing else 0,
            "pause_count": len(pause_durations),
            "avg_pause_duration": np.mean(pause_durations) if pause_durations else 0,
            "window_switch_rate": window_switches,
            "cursor_velocity": cls._calculate_cursor_velocity(cursor_movements),
        }
        
        raw_signals = {
            "keystroke_timing": keystroke_timing,
            "pause_durations": pause_durations,
            **other_signals
        }
        
        return cls(
            bmd_state=bmd_state,
            behavioral_features=features,
            raw_signals=raw_signals
        )
    
    @staticmethod
    def _calculate_knowledge_entropy(
        keystroke_timing: List[float],
        pause_durations: List[float]
    ) -> float:
        """
        S_k from typing patterns
        
        Steady typing = low S_k (executing known thought)
        Variable typing = high S_k (formulating new thought)
        """
        if not keystroke_timing:
            return 5.0
        
        variance = np.var(keystroke_timing)
        return min(10.0, variance * 10)  # Scale to reasonable range
    
    @staticmethod
    def _calculate_temporal_entropy(
        window_switches: int,
        keystroke_timing: List[float]
    ) -> float:
        """
        S_t from attention patterns
        
        Many switches = high S_t (urgency/searching)
        Focused = low S_t (steady state)
        """
        switch_contribution = min(5.0, window_switches * 0.5)
        
        if keystroke_timing:
            speed = 1.0 / (np.mean(keystroke_timing) + 0.01)
            speed_contribution = min(5.0, speed * 0.5)
        else:
            speed_contribution = 0
        
        return switch_contribution + speed_contribution
    
    @staticmethod
    def _calculate_evolution_entropy(
        pause_durations: List[float],
        keystroke_timing: List[float]
    ) -> float:
        """
        S_e from uncertainty indicators
        
        Long pauses = high S_e (uncertain/thinking)
        Erratic typing = high S_e (revising/uncertain)
        """
        pause_contribution = 0
        if pause_durations:
            avg_pause = np.mean(pause_durations)
            pause_contribution = min(7.0, avg_pause / 2.0)
        
        erratic_contribution = 0
        if len(keystroke_timing) > 3:
            # Calculate autocorrelation as measure of pattern
            ac = np.corrcoef(keystroke_timing[:-1], keystroke_timing[1:])[0, 1]
            erratic_contribution = (1 - abs(ac)) * 3.0  # High when random
        
        return pause_contribution + erratic_contribution
    
    @staticmethod
    def _calculate_cursor_velocity(cursor_movements: List[tuple]) -> float:
        """Calculate average cursor velocity"""
        if len(cursor_movements) < 2:
            return 0.0
        
        velocities = []
        for i in range(1, len(cursor_movements)):
            x1, y1, t1 = cursor_movements[i-1]
            x2, y2, t2 = cursor_movements[i]
            
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            dt = t2 - t1
            
            if dt > 0:
                velocities.append(distance / dt)
        
        return np.mean(velocities) if velocities else 0.0
    
    def to_feature_vector(self) -> NDArray[np.float64]:
        """
        Convert to numerical feature vector for ML models
        
        Returns: 9D vector: [S_k, S_t, S_e, + 6 behavioral features]
        """
        return np.array([
            self.bmd_state.s_entropy.S_k,
            self.bmd_state.s_entropy.S_t,
            self.bmd_state.s_entropy.S_e,
            self.behavioral_features.get("avg_keystroke_interval", 0),
            self.behavioral_features.get("keystroke_variance", 0),
            self.behavioral_features.get("pause_count", 0),
            self.behavioral_features.get("avg_pause_duration", 0),
            self.behavioral_features.get("window_switch_rate", 0),
            self.behavioral_features.get("cursor_velocity", 0),
        ])
    
    def __repr__(self) -> str:
        return f"BMDStateVector({self.bmd_state}, features={len(self.behavioral_features)})"

