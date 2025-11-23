"""
BMD State Detection

Detect user's BMD state from observable behavioral signals.
This is the "read" direction of categorical telepathy.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from collections import deque
import time

from .bmd_state import BMDState, BMDStateVector
from ..core.s_entropy import SEntropyCoordinates


@dataclass
class BehavioralSignal:
    """
    A single behavioral observation
    """
    signal_type: str  # "keystroke", "cursor", "window_switch", etc.
    timestamp: float
    value: Any  # Type depends on signal_type
    metadata: Dict[str, Any] = field(default_factory=dict)


class BMDStateDetector:
    """
    Detect user's BMD state from behavioral signals
    
    This is the "read" side of bidirectional communication:
    Observable behaviors → BMD state → Semantic interpretation
    """
    
    def __init__(
        self,
        window_size: int = 100,  # Number of signals to consider
        update_rate: float = 0.1,  # Update every 100ms
        smoothing_factor: float = 0.3  # Exponential smoothing
    ):
        """
        Initialize BMD state detector
        
        Args:
            window_size: Number of recent signals to consider
            update_rate: How often to update state (seconds)
            smoothing_factor: Alpha for exponential smoothing
        """
        self.window_size = window_size
        self.update_rate = update_rate
        self.smoothing_factor = smoothing_factor
        
        # Signal buffers
        self.keystroke_buffer: deque = deque(maxlen=window_size)
        self.cursor_buffer: deque = deque(maxlen=window_size)
        self.window_switch_buffer: deque = deque(maxlen=window_size)
        self.pause_buffer: deque = deque(maxlen=window_size)
        
        # Current state
        self.current_state: Optional[BMDState] = None
        self.last_update: float = 0.0
        
        # State history
        self.state_history: List[BMDState] = []
        self.max_history = 1000
    
    def observe_signal(self, signal: BehavioralSignal) -> Optional[BMDState]:
        """
        Observe a behavioral signal and potentially update BMD state
        
        Returns: Updated BMD state if update occurred, None otherwise
        """
        # Add to appropriate buffer
        if signal.signal_type == "keystroke":
            self.keystroke_buffer.append(signal)
        elif signal.signal_type == "cursor_move":
            self.cursor_buffer.append(signal)
        elif signal.signal_type == "window_switch":
            self.window_switch_buffer.append(signal)
        elif signal.signal_type == "pause":
            self.pause_buffer.append(signal)
        
        # Check if we should update state
        current_time = time.time()
        if current_time - self.last_update >= self.update_rate:
            return self.update_state()
        
        return None
    
    def update_state(self) -> BMDState:
        """
        Compute current BMD state from buffered signals
        """
        # Extract timing patterns
        keystroke_timings = self._extract_keystroke_timings()
        cursor_movements = self._extract_cursor_movements()
        window_switches = len(self.window_switch_buffer)
        pause_durations = [s.value for s in self.pause_buffer]
        
        # Create BMD state vector
        state_vector = BMDStateVector.from_behaviors(
            keystroke_timing=keystroke_timings,
            cursor_movements=cursor_movements,
            window_switches=window_switches,
            pause_durations=pause_durations
        )
        
        # Extract BMD state
        new_state = state_vector.bmd_state
        
        # Apply smoothing if we have previous state
        if self.current_state is not None:
            new_state = self._smooth_state(self.current_state, new_state)
        
        # Update current state
        self.current_state = new_state
        self.last_update = time.time()
        
        # Add to history
        self.state_history.append(new_state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        
        return new_state
    
    def get_current_state(self) -> Optional[BMDState]:
        """Get most recent BMD state"""
        return self.current_state
    
    def get_state_trajectory(self, n: int = 10) -> List[BMDState]:
        """Get last n BMD states"""
        return self.state_history[-n:]
    
    def _extract_keystroke_timings(self) -> List[float]:
        """Extract inter-keystroke intervals"""
        if len(self.keystroke_buffer) < 2:
            return []
        
        timings = []
        signals = list(self.keystroke_buffer)
        for i in range(1, len(signals)):
            interval = signals[i].timestamp - signals[i-1].timestamp
            timings.append(interval)
        
        return timings
    
    def _extract_cursor_movements(self) -> List[tuple]:
        """Extract cursor (x, y, timestamp) tuples"""
        movements = []
        for signal in self.cursor_buffer:
            if isinstance(signal.value, dict):
                x = signal.value.get('x', 0)
                y = signal.value.get('y', 0)
                movements.append((x, y, signal.timestamp))
        
        return movements
    
    def _smooth_state(
        self,
        old_state: BMDState,
        new_state: BMDState
    ) -> BMDState:
        """
        Apply exponential smoothing to state transition
        
        Prevents rapid oscillations in detected state.
        """
        alpha = self.smoothing_factor
        
        # Smooth S-entropy coordinates
        s_k = alpha * new_state.s_entropy.S_k + (1 - alpha) * old_state.s_entropy.S_k
        s_t = alpha * new_state.s_entropy.S_t + (1 - alpha) * old_state.s_entropy.S_t
        s_e = alpha * new_state.s_entropy.S_e + (1 - alpha) * old_state.s_entropy.S_e
        
        smoothed_entropy = SEntropyCoordinates(s_k, s_t, s_e)
        
        # Smooth confidence
        confidence = alpha * new_state.confidence + (1 - alpha) * old_state.confidence
        
        return BMDState(
            s_entropy=smoothed_entropy,
            confidence=confidence,
            metadata={
                "smoothed": True,
                "alpha": alpha
            }
        )
    
    def detect_query_formation(self, lookback: int = 5) -> tuple[bool, float]:
        """
        Detect if user is forming a query
        
        Returns: (is_forming, confidence)
        """
        if len(self.state_history) < lookback:
            return False, 0.0
        
        recent_states = self.state_history[-lookback:]
        
        # Query formation indicators:
        # 1. Increasing S_k (accumulating information/searching)
        # 2. High S_e (uncertainty about what to ask)
        # 3. Pauses (thinking about formulation)
        
        s_k_trend = self._calculate_trend([s.s_entropy.S_k for s in recent_states])
        s_e_level = np.mean([s.s_entropy.S_e for s in recent_states])
        
        is_forming = (s_k_trend > 0.5 and  # S_k increasing
                     s_e_level > 4.0)  # High uncertainty
        
        confidence = min(1.0, (s_k_trend + s_e_level / 10))
        
        return is_forming, confidence
    
    def detect_confusion(self, threshold: float = 6.0) -> bool:
        """
        Detect if user is confused/stuck
        
        High S_e + erratic behavior = confusion
        """
        if self.current_state is None:
            return False
        
        # High uncertainty + recent pauses
        high_uncertainty = self.current_state.s_entropy.S_e > threshold
        has_pauses = len(self.pause_buffer) > 3
        
        return high_uncertainty and has_pauses
    
    def detect_flow_state(self) -> bool:
        """
        Detect if user is in "flow" state
        
        Low S_e + steady behavior = flow
        """
        if len(self.state_history) < 10:
            return False
        
        recent_states = self.state_history[-10:]
        
        # Flow indicators:
        # 1. Low S_e (certainty)
        # 2. Steady state (low variance in S_entropy)
        # 3. Consistent typing rhythm
        
        avg_s_e = np.mean([s.s_entropy.S_e for s in recent_states])
        variance = np.var([s.s_entropy.magnitude for s in recent_states])
        
        return avg_s_e < 3.0 and variance < 1.0
    
    @staticmethod
    def _calculate_trend(values: List[float]) -> float:
        """
        Calculate trend in values (positive = increasing)
        
        Uses simple linear regression slope
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        return slope


class RealTimeMonitor:
    """
    Real-time BMD state monitoring system
    
    Continuously monitors behavioral signals and provides
    live BMD state updates.
    """
    
    def __init__(self, detector: BMDStateDetector):
        self.detector = detector
        self.callbacks: List[Callable[[BMDState], None]] = []
        self.is_monitoring = False
    
    def register_callback(self, callback: Callable[[BMDState], None]) -> None:
        """Register callback for state updates"""
        self.callbacks.append(callback)
    
    def on_state_change(self, new_state: BMDState) -> None:
        """Trigger all callbacks when state changes"""
        for callback in self.callbacks:
            try:
                callback(new_state)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        self.is_monitoring = True
        # In production, this would start a background thread
        # For now, just set flag
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.is_monitoring = False

