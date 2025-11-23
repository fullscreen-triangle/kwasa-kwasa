"""
Transfer Functions for BMD State Mapping

Forward: Observable behavior → BMD state
Inverse: Desired BMD state → Required stimulus

Author: Kundai Farai Sachikonye
"""

from typing import Callable, Any, Dict, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from ..bmd.bmd_state import BMDState
from ..core.s_entropy import SEntropyCoordinates


class TransferFunction(ABC):
    """Base class for transfer functions"""
    
    @abstractmethod
    def __call__(self, input_data: Any) -> Any:
        """Apply transfer function"""
        pass
    
    @abstractmethod
    def train(self, training_data: list) -> None:
        """Train function from data"""
        pass


class ForwardModel(TransferFunction):
    """
    Forward model: Behavior → BMD State
    
    Maps observable behaviors to internal BMD coordinates.
    """
    
    def __init__(self):
        self.is_trained = False
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
    
    def __call__(self, behavior_features: NDArray[np.float64]) -> BMDState:
        """
        Map behavior features to BMD state
        
        Args:
            behavior_features: Feature vector of behaviors
            
        Returns: Predicted BMD state
        """
        if not self.is_trained:
            # Use default mapping
            return self._default_mapping(behavior_features)
        
        # Linear transformation (in production, use neural network)
        s_vector = np.dot(self.weights, behavior_features) + self.bias
        
        # Clamp to valid range
        s_vector = np.maximum(0, s_vector)
        
        s_entropy = SEntropyCoordinates(
            S_k=float(s_vector[0]),
            S_t=float(s_vector[1]),
            S_e=float(s_vector[2])
        )
        
        return BMDState(s_entropy=s_entropy, confidence=0.8)
    
    def train(self, training_data: list) -> None:
        """
        Train forward model from data
        
        training_data: List of (behavior_features, true_bmd_state) tuples
        """
        if len(training_data) < 10:
            return  # Need minimum data
        
        # Extract features and targets
        X = np.array([features for features, _ in training_data])
        Y = np.array([[state.s_entropy.S_k, state.s_entropy.S_t, state.s_entropy.S_e] 
                      for _, state in training_data])
        
        # Simple linear regression (in production, use more sophisticated model)
        # Y = X @ W + b
        self.weights, _, _, _ = np.linalg.lstsq(
            np.column_stack([X, np.ones(len(X))]),
            Y,
            rcond=None
        )
        
        self.bias = self.weights[-1, :]
        self.weights = self.weights[:-1, :].T
        
        self.is_trained = True
    
    def _default_mapping(self, features: NDArray[np.float64]) -> BMDState:
        """Default mapping when untrained"""
        # Simple heuristic mapping
        s_k = np.clip(np.mean(features) * 5, 0, 10)
        s_t = np.clip(np.var(features) * 10, 0, 10)
        s_e = np.clip(np.std(features) * 10, 0, 10)
        
        return BMDState(
            s_entropy=SEntropyCoordinates(s_k, s_t, s_e),
            confidence=0.5
        )


class InverseModel(TransferFunction):
    """
    Inverse model: Desired BMD State → Required Stimulus
    
    Given target BMD state, compute what stimulus will achieve it.
    """
    
    def __init__(self):
        self.is_trained = False
        self.weights: Optional[NDArray[np.float64]] = None
        self.bias: Optional[NDArray[np.float64]] = None
    
    def __call__(self, target_state: BMDState) -> NDArray[np.float64]:
        """
        Compute stimulus features needed to reach target state
        
        Args:
            target_state: Desired BMD state
            
        Returns: Stimulus feature vector
        """
        # Extract S-entropy as vector
        s_vector = target_state.s_entropy.vector
        
        if not self.is_trained:
            # Use default inverse mapping
            return self._default_inverse(s_vector)
        
        # Apply inverse transformation
        stimulus_features = np.dot(self.weights, s_vector) + self.bias
        
        return stimulus_features
    
    def train(self, training_data: list) -> None:
        """
        Train inverse model from data
        
        training_data: List of (target_state, successful_stimulus) tuples
        """
        if len(training_data) < 10:
            return
        
        # Extract features and targets
        X = np.array([[state.s_entropy.S_k, state.s_entropy.S_t, state.s_entropy.S_e]
                      for state, _ in training_data])
        Y = np.array([stimulus for _, stimulus in training_data])
        
        # Linear regression
        self.weights, _, _, _ = np.linalg.lstsq(
            np.column_stack([X, np.ones(len(X))]),
            Y,
            rcond=None
        )
        
        self.bias = self.weights[-1, :]
        self.weights = self.weights[:-1, :].T
        
        self.is_trained = True
    
    def _default_inverse(self, s_vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Default inverse mapping when untrained"""
        # Simple heuristic: stimulus proportional to S-entropy
        return s_vector / 5.0  # Scale down to stimulus range

