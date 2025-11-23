"""
User BMD Model

Personalized model of how a specific user's BMD operates.
Different users complete stimuli differently - this captures those patterns.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
from numpy.typing import NDArray
import json
import time

from ..bmd.bmd_state import BMDState
from ..core.categorical_state import CategoricalState
from ..core.s_entropy import SEntropyCoordinates


@dataclass
class CompletionPattern:
    """
    A learned pattern: Stimulus → Completion
    
    Records what stimulus led to what completion for this user.
    """
    stimulus_features: NDArray[np.float64]  # Feature vector of stimulus
    completed_state: CategoricalState  # What user actually completed to
    success: bool  # Whether completion matched target
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Pattern({self.completed_state.category}, success={self.success})"


class UserBMDModel:
    """
    Personalized model of user's BMD completion dynamics
    
    Key insight: Same stimulus → different completions for different users
    This model captures each user's unique completion patterns.
    """
    
    def __init__(
        self,
        user_id: str,
        learning_rate: float = 0.1,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize user BMD model
        
        Args:
            user_id: Unique identifier for user
            learning_rate: How fast to adapt to new observations
            confidence_threshold: Minimum confidence for predictions
        """
        self.user_id = user_id
        self.learning_rate = learning_rate
        self.confidence_threshold = confidence_threshold
        
        # Learned completion patterns
        self.patterns: List[CompletionPattern] = []
        
        # Transfer functions (learned from patterns)
        self.forward_model: Optional[Callable] = None  # Behavior → BMD
        self.inverse_model: Optional[Callable] = None  # BMD → Stimulus
        
        # Temporal dynamics (time-of-day effects, etc.)
        self.temporal_patterns: Dict[str, Any] = {}
        
        # Context-specific patterns
        self.context_patterns: Dict[str, List[CompletionPattern]] = {}
        
        # Learning statistics
        self.total_observations = 0
        self.prediction_accuracy = 0.0
        self.last_updated = time.time()
        
        # Per-category success rates
        self.category_success_rates: Dict[str, float] = {}
    
    def predict_completion(
        self,
        stimulus_features: NDArray[np.float64],
        current_state: BMDState
    ) -> Tuple[CategoricalState, float]:
        """
        Predict what this user will complete given stimulus
        
        Args:
            stimulus_features: Feature vector of stimulus
            current_state: User's current BMD state
            
        Returns: (predicted_state, confidence)
        """
        if len(self.patterns) < 5:
            # Not enough data, return uncertain prediction
            return self._default_prediction(current_state), 0.3
        
        # Find similar past stimuli
        similar_patterns = self._find_similar_patterns(
            stimulus_features,
            current_state,
            k=5
        )
        
        if not similar_patterns:
            return self._default_prediction(current_state), 0.3
        
        # Weight by similarity and recency
        weighted_states = []
        total_weight = 0.0
        
        for pattern, similarity in similar_patterns:
            # Weight by similarity and success
            recency_factor = np.exp(-(time.time() - pattern.timestamp) / 86400)  # Decay over days
            weight = similarity * (1.0 if pattern.success else 0.5) * recency_factor
            
            weighted_states.append((pattern.completed_state, weight))
            total_weight += weight
        
        # Find most probable completion
        category_weights: Dict[str, float] = {}
        state_by_category: Dict[str, CategoricalState] = {}
        
        for state, weight in weighted_states:
            if state.category not in category_weights:
                category_weights[state.category] = 0.0
                state_by_category[state.category] = state
            category_weights[state.category] += weight
        
        # Normalize to get probabilities
        if total_weight > 0:
            category_probs = {cat: w / total_weight for cat, w in category_weights.items()}
        else:
            return self._default_prediction(current_state), 0.3
        
        # Return most probable category
        best_category = max(category_probs, key=category_probs.get)
        confidence = category_probs[best_category]
        
        return state_by_category[best_category], confidence
    
    def add_observation(
        self,
        stimulus_features: NDArray[np.float64],
        completed_state: CategoricalState,
        target_state: CategoricalState,
        current_bmd: BMDState,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add observed completion to model
        
        This is how the model learns from actual user completions.
        """
        # Check if completion matched target
        success = (completed_state.category == target_state.category or
                  completed_state.distance_to(target_state) < 1.0)
        
        # Create pattern
        pattern = CompletionPattern(
            stimulus_features=stimulus_features,
            completed_state=completed_state,
            success=success,
            timestamp=time.time(),
            context=context or {}
        )
        
        self.patterns.append(pattern)
        self.total_observations += 1
        
        # Update category success rates
        category = target_state.category
        if category not in self.category_success_rates:
            self.category_success_rates[category] = 0.0
        
        # Exponential moving average
        alpha = self.learning_rate
        current_rate = self.category_success_rates[category]
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        self.category_success_rates[category] = new_rate
        
        # Store in context-specific patterns if context provided
        if context and "context_type" in context:
            context_key = context["context_type"]
            if context_key not in self.context_patterns:
                self.context_patterns[context_key] = []
            self.context_patterns[context_key].append(pattern)
        
        # Update timestamp
        self.last_updated = time.time()
        
        # Retrain transfer functions periodically
        if self.total_observations % 50 == 0:
            self._update_transfer_functions()
    
    def _find_similar_patterns(
        self,
        stimulus_features: NDArray[np.float64],
        current_state: BMDState,
        k: int = 5
    ) -> List[Tuple[CompletionPattern, float]]:
        """
        Find k most similar past patterns
        
        Returns: List of (pattern, similarity) tuples
        """
        similarities = []
        
        for pattern in self.patterns:
            # Feature similarity
            feature_sim = self._cosine_similarity(
                stimulus_features,
                pattern.stimulus_features
            )
            
            # State similarity (if pattern has context with BMD state)
            state_sim = 1.0  # Default
            if "bmd_state" in pattern.context:
                past_state = pattern.context["bmd_state"]
                if isinstance(past_state, BMDState):
                    state_distance = current_state.distance_to(past_state)
                    state_sim = 1.0 / (1.0 + state_distance)
            
            # Combined similarity
            similarity = 0.7 * feature_sim + 0.3 * state_sim
            
            similarities.append((pattern, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    @staticmethod
    def _cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0
        
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _default_prediction(self, current_state: BMDState) -> CategoricalState:
        """Default prediction when insufficient data"""
        return CategoricalState(
            category="unknown",
            meaning="Insufficient data for prediction",
            confidence=0.3,
            s_entropy=current_state.s_entropy
        )
    
    def _update_transfer_functions(self) -> None:
        """
        Update transfer functions from accumulated patterns
        
        This would train ML models in production.
        For now, just placeholder.
        """
        # In production: Train neural network or other model
        # self.forward_model = train_forward_model(self.patterns)
        # self.inverse_model = train_inverse_model(self.patterns)
        pass
    
    def get_success_rate(self, category: Optional[str] = None) -> float:
        """
        Get success rate overall or for specific category
        
        Args:
            category: Specific category (or None for overall)
            
        Returns: Success rate [0, 1]
        """
        if category:
            return self.category_success_rates.get(category, 0.0)
        
        # Overall success rate
        if not self.patterns:
            return 0.0
        
        successes = sum(1 for p in self.patterns if p.success)
        return successes / len(self.patterns)
    
    def get_best_categories(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get categories this user is most successful with
        
        Returns: List of (category, success_rate) tuples
        """
        items = list(self.category_success_rates.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
    def get_temporal_pattern(self, time_of_day: str) -> Dict[str, Any]:
        """
        Get user's patterns for specific time of day
        
        Args:
            time_of_day: "morning", "afternoon", "evening", "night"
            
        Returns: Pattern statistics for that time
        """
        return self.temporal_patterns.get(time_of_day, {})
    
    def save(self, filepath: str) -> None:
        """Save model to file"""
        data = {
            "user_id": self.user_id,
            "learning_rate": self.learning_rate,
            "total_observations": self.total_observations,
            "prediction_accuracy": self.prediction_accuracy,
            "category_success_rates": self.category_success_rates,
            "last_updated": self.last_updated,
            # Patterns would need custom serialization
            "num_patterns": len(self.patterns)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'UserBMDModel':
        """Load model from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        model = cls(
            user_id=data["user_id"],
            learning_rate=data["learning_rate"]
        )
        
        model.total_observations = data["total_observations"]
        model.prediction_accuracy = data["prediction_accuracy"]
        model.category_success_rates = data["category_success_rates"]
        model.last_updated = data["last_updated"]
        
        return model
    
    def __repr__(self) -> str:
        return (f"UserBMDModel({self.user_id}, "
                f"obs={self.total_observations}, "
                f"acc={self.get_success_rate():.2f})")

