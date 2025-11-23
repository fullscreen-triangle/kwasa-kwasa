"""
Thought Injection

Inject semantic states as thoughts through sufficient stimuli.
This is the "write" direction of categorical telepathy.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

from .bmd_state import BMDState
from ..core.categorical_state import CategoricalState
from ..core.s_entropy import SEntropyCoordinates


class StimulusModality(Enum):
    """Types of stimuli that can be injected"""
    VISUAL_TEXT = "visual_text"  # Text on screen
    VISUAL_COLOR = "visual_color"  # Color changes
    VISUAL_PATTERN = "visual_pattern"  # Visual patterns
    AUDIO_TONE = "audio_tone"  # Audio frequencies
    AUDIO_SPEECH = "audio_speech"  # Spoken words
    TEMPORAL_RHYTHM = "temporal_rhythm"  # Timing patterns
    HAPTIC = "haptic"  # Touch/vibration
    CONTEXTUAL = "contextual"  # Environmental cues


@dataclass
class StimulusPattern:
    """
    A pattern of stimuli designed to induce a specific thought
    
    This is the "sufficient stimulus" - minimal information
    required for user's BMD to complete to target thought.
    """
    modality: StimulusModality
    content: Any  # Type depends on modality
    duration: float  # How long to present (seconds)
    intensity: float  # [0, 1] strength of stimulus
    timing: Optional[List[float]] = None  # Specific timing pattern
    target_state: Optional[CategoricalState] = None
    completion_probability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Stimulus({self.modality.value}, P={self.completion_probability:.2f})"


@dataclass
class InjectionResult:
    """Result of thought injection attempt"""
    stimulus: StimulusPattern
    success: bool
    actual_completion: Optional[CategoricalState] = None
    user_feedback: Optional[str] = None
    latency: float = 0.0  # Time from injection to completion


class ThoughtInjector:
    """
    Inject thoughts by generating sufficient stimuli
    
    This is the "write" side of bidirectional communication:
    Semantic state → Sufficient stimulus → User completes → Thought experience
    """
    
    def __init__(
        self,
        user_id: str,
        preferred_modalities: Optional[List[StimulusModality]] = None
    ):
        """
        Initialize thought injector
        
        Args:
            user_id: Identifier for user (personalization)
            preferred_modalities: User's preferred stimulus types
        """
        self.user_id = user_id
        self.preferred_modalities = preferred_modalities or [
            StimulusModality.VISUAL_TEXT,
            StimulusModality.TEMPORAL_RHYTHM,
            StimulusModality.VISUAL_COLOR
        ]
        
        # Injection history for learning
        self.injection_history: List[InjectionResult] = []
        
        # Per-user completion patterns (learned)
        self.completion_patterns: Dict[str, List[tuple]] = {}
    
    def inject(
        self,
        target_thought: CategoricalState,
        user_state: BMDState,
        modality: Optional[StimulusModality] = None
    ) -> StimulusPattern:
        """
        Generate and inject stimulus to induce target thought
        
        Args:
            target_thought: Desired semantic state
            user_state: User's current BMD state
            modality: Specific modality (or None for auto-select)
            
        Returns:
            Stimulus pattern to present to user
        """
        # Select modality if not specified
        if modality is None:
            modality = self._select_modality(target_thought, user_state)
        
        # Generate sufficient stimulus
        stimulus = self._generate_stimulus(
            target_thought=target_thought,
            user_state=user_state,
            modality=modality
        )
        
        return stimulus
    
    def inject_semantic_sequence(
        self,
        thought_sequence: List[CategoricalState],
        user_state: BMDState
    ) -> List[StimulusPattern]:
        """
        Inject sequence of thoughts (for complex ideas)
        
        Like building up to a conclusion through steps.
        """
        stimuli = []
        
        for i, thought in enumerate(thought_sequence):
            # Earlier thoughts can be stronger (foundation)
            # Later thoughts can be weaker (user completes pattern)
            intensity = 1.0 - (i / len(thought_sequence)) * 0.5
            
            stimulus = self.inject(thought, user_state)
            stimulus.intensity *= intensity
            stimuli.append(stimulus)
        
        return stimuli
    
    def record_completion(
        self,
        stimulus: StimulusPattern,
        actual_completion: CategoricalState,
        success: bool,
        latency: float
    ) -> None:
        """
        Record actual user completion for learning
        
        This is how the injector learns what works for this user.
        """
        result = InjectionResult(
            stimulus=stimulus,
            success=success,
            actual_completion=actual_completion,
            latency=latency
        )
        
        self.injection_history.append(result)
        
        # Update completion patterns
        pattern_key = f"{stimulus.modality.value}_{stimulus.target_state.category if stimulus.target_state else 'unknown'}"
        if pattern_key not in self.completion_patterns:
            self.completion_patterns[pattern_key] = []
        
        self.completion_patterns[pattern_key].append(
            (stimulus, actual_completion, success)
        )
    
    def _select_modality(
        self,
        target_thought: CategoricalState,
        user_state: BMDState
    ) -> StimulusModality:
        """
        Select best modality for this thought and user state
        
        Different thoughts work better with different modalities.
        Different user states are receptive to different stimuli.
        """
        # If user is in flow (low S_e), use subtle temporal rhythms
        if user_state.uncertainty < 3.0:
            return StimulusModality.TEMPORAL_RHYTHM
        
        # If user is confused (high S_e), use explicit visual text
        if user_state.uncertainty > 7.0:
            return StimulusModality.VISUAL_TEXT
        
        # If user is urgent (high S_t), use fast visual patterns
        if user_state.temporal_state > 7.0:
            return StimulusModality.VISUAL_PATTERN
        
        # Otherwise, use preferred modality
        return self.preferred_modalities[0]
    
    def _generate_stimulus(
        self,
        target_thought: CategoricalState,
        user_state: BMDState,
        modality: StimulusModality
    ) -> StimulusPattern:
        """
        Generate sufficient stimulus for target thought
        
        This is where the "sufficiency principle" is implemented.
        We generate minimal stimulus that will complete to target.
        """
        # Calculate how different target is from current state
        # Larger difference = need stronger stimulus
        if hasattr(user_state, 's_entropy'):
            distance = user_state.s_entropy.distance_to(target_thought.s_entropy)
            intensity = min(1.0, distance / 10.0)  # Scale intensity
        else:
            intensity = 0.5  # Default moderate intensity
        
        # Generate modality-specific content
        content = self._generate_content(target_thought, modality, user_state)
        
        # Calculate optimal duration
        duration = self._calculate_duration(target_thought, user_state, modality)
        
        # Generate timing pattern if using temporal modality
        timing = None
        if modality == StimulusModality.TEMPORAL_RHYTHM:
            timing = self._generate_temporal_pattern(target_thought)
        
        # Estimate completion probability
        completion_prob = self._estimate_completion_probability(
            target_thought, user_state, modality
        )
        
        return StimulusPattern(
            modality=modality,
            content=content,
            duration=duration,
            intensity=intensity,
            timing=timing,
            target_state=target_thought,
            completion_probability=completion_prob,
            metadata={
                "user_state": user_state,
                "distance_to_target": distance if 'distance' in locals() else None
            }
        )
    
    def _generate_content(
        self,
        target_thought: CategoricalState,
        modality: StimulusModality,
        user_state: BMDState
    ) -> Any:
        """Generate modality-specific content"""
        if modality == StimulusModality.VISUAL_TEXT:
            # Generate key words/phrases that point to thought
            return self._generate_text_prompt(target_thought)
        
        elif modality == StimulusModality.VISUAL_COLOR:
            # Generate color that maps to thought valence
            return self._generate_color_mapping(target_thought)
        
        elif modality == StimulusModality.AUDIO_TONE:
            # Generate frequency corresponding to S-entropy
            return self._generate_frequency(target_thought.s_entropy)
        
        elif modality == StimulusModality.TEMPORAL_RHYTHM:
            # Return placeholder - actual timing in timing field
            return "temporal_pattern"
        
        else:
            return {"thought": target_thought.meaning}
    
    def _generate_text_prompt(self, thought: CategoricalState) -> str:
        """
        Generate minimal text that points to thought
        
        Not the full thought - just sufficient keywords for completion.
        """
        # Extract key words from meaning
        words = thought.meaning.split()
        
        # Take 2-3 most informative words
        # (In production, use actual NLP/semantic analysis)
        key_words = words[:min(3, len(words))]
        
        return " ".join(key_words)
    
    def _generate_color_mapping(self, thought: CategoricalState) -> tuple:
        """
        Map thought valence to color
        
        Different semantic states → different colors
        User's brain associates colors with meanings
        """
        # Map S-entropy to HSV color space
        # High S_k → High saturation (rich information)
        # High S_t → Red hues (urgency)
        # High S_e → Cool hues (uncertainty)
        
        s_k = thought.s_entropy.S_k
        s_t = thought.s_entropy.S_t
        s_e = thought.s_entropy.S_e
        
        hue = int((s_t / 10.0) * 180)  # 0-180 (blue to red)
        saturation = int((s_k / 10.0) * 255)  # 0-255
        value = int((1.0 - s_e / 10.0) * 255)  # Lower for uncertainty
        
        # Convert HSV to RGB (simplified)
        return (hue, saturation, value)
    
    def _generate_frequency(self, s_entropy: SEntropyCoordinates) -> float:
        """Generate audio frequency from S-entropy"""
        # Map S-entropy magnitude to frequency range 200-2000 Hz
        magnitude = s_entropy.magnitude
        frequency = 200 + (magnitude / 20.0) * 1800
        return min(2000, max(200, frequency))
    
    def _generate_temporal_pattern(self, thought: CategoricalState) -> List[float]:
        """
        Generate temporal presentation pattern
        
        Different rhythms → different thoughts
        Brain is sensitive to timing patterns
        """
        # Use S-entropy to generate rhythm
        base_interval = 0.2  # 200ms base
        
        s_k = thought.s_entropy.S_k
        s_t = thought.s_entropy.S_t
        s_e = thought.s_entropy.S_e
        
        # Generate 5-pulse pattern
        pattern = [
            base_interval,
            base_interval * (1 + s_k / 10),
            base_interval * (1 + s_t / 10),
            base_interval * (1 + s_e / 10),
            base_interval * 2
        ]
        
        return pattern
    
    def _calculate_duration(
        self,
        thought: CategoricalState,
        user_state: BMDState,
        modality: StimulusModality
    ) -> float:
        """
        Calculate optimal presentation duration
        
        Complex thoughts need longer presentation
        Urgent states need shorter presentation
        """
        # Base duration by modality
        base_durations = {
            StimulusModality.VISUAL_TEXT: 1.0,
            StimulusModality.VISUAL_COLOR: 0.3,
            StimulusModality.AUDIO_TONE: 0.5,
            StimulusModality.TEMPORAL_RHYTHM: 2.0,
            StimulusModality.VISUAL_PATTERN: 0.8
        }
        
        base = base_durations.get(modality, 1.0)
        
        # Adjust for thought complexity (S_k)
        complexity_factor = 1.0 + (thought.s_entropy.S_k / 20.0)
        
        # Adjust for user urgency (S_t) - shorter if urgent
        urgency_factor = 1.0 / (1.0 + user_state.temporal_state / 10.0)
        
        return base * complexity_factor * urgency_factor
    
    def _estimate_completion_probability(
        self,
        target_thought: CategoricalState,
        user_state: BMDState,
        modality: StimulusModality
    ) -> float:
        """
        Estimate probability user will complete to target thought
        
        Based on:
        - Distance from current state to target
        - User's receptivity (uncertainty level)
        - Historical success rate with this modality
        """
        # Distance factor: closer = higher probability
        distance = user_state.s_entropy.distance_to(target_thought.s_entropy)
        distance_factor = 1.0 / (1.0 + distance / 5.0)
        
        # Receptivity factor: moderate S_e is most receptive
        optimal_s_e = 5.0
        s_e_diff = abs(user_state.uncertainty - optimal_s_e)
        receptivity_factor = 1.0 / (1.0 + s_e_diff / 5.0)
        
        # Historical success rate
        pattern_key = f"{modality.value}_{target_thought.category}"
        if pattern_key in self.completion_patterns:
            successes = sum(1 for _, _, success in self.completion_patterns[pattern_key] if success)
            total = len(self.completion_patterns[pattern_key])
            historical_factor = successes / total if total > 0 else 0.5
        else:
            historical_factor = 0.5  # Default unknown
        
        # Combine factors
        probability = distance_factor * receptivity_factor * historical_factor
        
        return min(1.0, probability)
    
    def get_success_rate(self) -> float:
        """Calculate overall injection success rate"""
        if not self.injection_history:
            return 0.0
        
        successes = sum(1 for r in self.injection_history if r.success)
        return successes / len(self.injection_history)
    
    def get_average_latency(self) -> float:
        """Calculate average completion latency"""
        if not self.injection_history:
            return 0.0
        
        return np.mean([r.latency for r in self.injection_history])

