"""
Calibration Engine

Orchestrates user calibration process to build personalized BMD models.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np

from .user_model import UserBMDModel, CompletionPattern
from .transfer_functions import ForwardModel, InverseModel
from ..bmd.bmd_state import BMDState, BMDStateVector
from ..core.categorical_state import CategoricalState
from ..core.s_entropy import SEntropyCoordinates


@dataclass
class CalibrationSession:
    """A single calibration session"""
    user_id: str
    start_time: float
    end_time: Optional[float] = None
    trials: List[Dict[str, Any]] = None
    accuracy: float = 0.0
    
    def __post_init__(self):
        if self.trials is None:
            self.trials = []


class CalibrationEngine:
    """
    Engine for calibrating user-specific BMD models
    
    Runs interactive calibration sessions to learn how each user
    completes stimuli to thoughts.
    """
    
    def __init__(self):
        self.sessions: Dict[str, List[CalibrationSession]] = {}
        self.models: Dict[str, UserBMDModel] = {}
    
    def start_session(self, user_id: str) -> CalibrationSession:
        """Start new calibration session for user"""
        session = CalibrationSession(
            user_id=user_id,
            start_time=time.time(),
            trials=[]
        )
        
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        self.sessions[user_id].append(session)
        
        return session
    
    def run_calibration(
        self,
        user_id: str,
        num_trials: int = 50,
        trial_types: Optional[List[str]] = None
    ) -> UserBMDModel:
        """
        Run full calibration for user
        
        Args:
            user_id: User identifier
            num_trials: Number of calibration trials
            trial_types: Types of trials to run
            
        Returns: Calibrated UserBMDModel
        """
        # Create or get existing model
        if user_id not in self.models:
            self.models[user_id] = UserBMDModel(user_id)
        model = self.models[user_id]
        
        # Start session
        session = self.start_session(user_id)
        
        # Default trial types
        if trial_types is None:
            trial_types = [
                "text_completion",
                "color_association",
                "timing_pattern",
                "concept_mapping"
            ]
        
        # Run trials
        for i in range(num_trials):
            trial_type = trial_types[i % len(trial_types)]
            trial_result = self._run_trial(user_id, trial_type, model)
            session.trials.append(trial_result)
            
            # Add observation to model
            if trial_result["success"]:
                model.add_observation(
                    stimulus_features=trial_result["stimulus_features"],
                    completed_state=trial_result["completed_state"],
                    target_state=trial_result["target_state"],
                    current_bmd=trial_result["user_state"],
                    context={"trial_type": trial_type}
                )
        
        # End session
        session.end_time = time.time()
        session.accuracy = model.get_success_rate()
        
        return model
    
    def calibrate_from_history(
        self,
        user_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> UserBMDModel:
        """
        Calibrate model from historical interaction data
        
        Args:
            user_id: User identifier
            interaction_history: Past interactions with timestamps
            
        Returns: Calibrated model
        """
        # Create model
        model = UserBMDModel(user_id)
        
        # Process each interaction
        for interaction in interaction_history:
            # Extract behavior features
            behavior_features = self._extract_behavior_features(interaction)
            
            # Create BMD state vector
            state_vector = BMDStateVector.from_behaviors(
                keystroke_timing=interaction.get("keystroke_timing", []),
                cursor_movements=interaction.get("cursor_movements", []),
                window_switches=interaction.get("window_switches", 0),
                pause_durations=interaction.get("pause_durations", [])
            )
            
            # Extract what user actually did/thought
            completed_state = self._parse_user_action(interaction)
            
            # Extract what was intended (if available)
            target_state = interaction.get("target_state")
            if target_state is None:
                target_state = completed_state  # Assume completion was intended
            
            # Add to model
            model.add_observation(
                stimulus_features=behavior_features,
                completed_state=completed_state,
                target_state=target_state,
                current_bmd=state_vector.bmd_state,
                context=interaction.get("context", {})
            )
        
        self.models[user_id] = model
        return model
    
    def _run_trial(
        self,
        user_id: str,
        trial_type: str,
        model: UserBMDModel
    ) -> Dict[str, Any]:
        """
        Run single calibration trial
        
        In production, this would present stimulus and capture user response.
        For now, simulate trial.
        """
        # Generate stimulus based on trial type
        target_state = self._generate_target_state(trial_type)
        stimulus_features = self._generate_stimulus_features(trial_type)
        
        # Simulate user state
        user_state = BMDState(
            s_entropy=SEntropyCoordinates(
                S_k=np.random.uniform(2, 8),
                S_t=np.random.uniform(2, 8),
                S_e=np.random.uniform(2, 8)
            )
        )
        
        # Simulate user completion (in production, get actual user response)
        completed_state = self._simulate_completion(
            stimulus_features,
            target_state,
            user_state
        )
        
        # Check success
        success = (completed_state.category == target_state.category or
                  completed_state.distance_to(target_state) < 1.5)
        
        return {
            "trial_type": trial_type,
            "stimulus_features": stimulus_features,
            "target_state": target_state,
            "completed_state": completed_state,
            "user_state": user_state,
            "success": success,
            "timestamp": time.time()
        }
    
    def _generate_target_state(self, trial_type: str) -> CategoricalState:
        """Generate target state for trial"""
        categories = {
            "text_completion": ("concept_A", "Understanding of concept A"),
            "color_association": ("emotional_state", "Emotional association"),
            "timing_pattern": ("temporal_concept", "Timing-based understanding"),
            "concept_mapping": ("semantic_relation", "Conceptual relationship")
        }
        
        category, meaning = categories.get(trial_type, ("general", "General concept"))
        
        return CategoricalState(
            category=category,
            meaning=meaning,
            confidence=1.0,
            s_entropy=SEntropyCoordinates(
                S_k=np.random.uniform(3, 7),
                S_t=np.random.uniform(2, 6),
                S_e=np.random.uniform(2, 6)
            )
        )
    
    def _generate_stimulus_features(self, trial_type: str) -> np.ndarray:
        """Generate stimulus feature vector"""
        # Random 10D feature vector (in production, extract from actual stimulus)
        return np.random.randn(10)
    
    def _simulate_completion(
        self,
        stimulus_features: np.ndarray,
        target_state: CategoricalState,
        user_state: BMDState
    ) -> CategoricalState:
        """
        Simulate user completion (placeholder)
        
        In production, this would be actual user response.
        """
        # Simulate with some noise
        noise_level = user_state.uncertainty / 10.0
        
        if np.random.random() > noise_level:
            # Successful completion
            return target_state
        else:
            # Off-target completion
            return CategoricalState(
                category=f"{target_state.category}_variant",
                meaning=f"Similar to {target_state.meaning}",
                confidence=0.6,
                s_entropy=target_state.s_entropy
            )
    
    def _extract_behavior_features(self, interaction: Dict[str, Any]) -> np.ndarray:
        """Extract behavior feature vector from interaction"""
        # Extract key behavioral metrics
        features = [
            len(interaction.get("keystroke_timing", [])),
            np.mean(interaction.get("keystroke_timing", [1.0])),
            np.var(interaction.get("keystroke_timing", [0.0])),
            len(interaction.get("pause_durations", [])),
            np.mean(interaction.get("pause_durations", [1.0])),
            interaction.get("window_switches", 0),
            len(interaction.get("cursor_movements", [])),
            interaction.get("duration", 1.0),
            interaction.get("corrections", 0),
            interaction.get("hesitations", 0)
        ]
        
        return np.array(features)
    
    def _parse_user_action(self, interaction: Dict[str, Any]) -> CategoricalState:
        """Parse user's action/completion from interaction data"""
        # In production, parse actual user response
        # For now, create generic state
        return CategoricalState(
            category=interaction.get("action_type", "unknown"),
            meaning=interaction.get("action_description", "User action"),
            confidence=0.8,
            s_entropy=SEntropyCoordinates(5.0, 5.0, 5.0)
        )
    
    def get_model(self, user_id: str) -> Optional[UserBMDModel]:
        """Get calibrated model for user"""
        return self.models.get(user_id)
    
    def get_session_history(self, user_id: str) -> List[CalibrationSession]:
        """Get all calibration sessions for user"""
        return self.sessions.get(user_id, [])

