"""
Categorical State Representation

Represents distinct semantic interpretations with S-entropy coordinates
and categorical equivalence classes.

Author: Kundai Farai Sachikonye
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import numpy as np

from .s_entropy import SEntropyCoordinates, CategoricalDistance


class SemanticLens(Enum):
    """Available semantic lenses (virtual preparation methods)"""
    PSYCHIATRIC = "psychiatric_dsm5"
    ENDOCRINE = "endocrine_metabolic"
    NEUROLOGICAL = "neurological_oscillatory"
    PSYCHOLOGICAL = "psychological_developmental"
    BIOCHEMICAL = "biochemical_molecular"
    CONTEXTUAL = "contextual_situational"
    PHYSICAL = "physical_measurement"
    SOCIAL = "social_behavioral"
    COGNITIVE = "cognitive_processing"
    LINGUISTIC = "linguistic_semantic"


@dataclass
class CategoricalState:
    """
    A categorical state represents a distinct semantic interpretation.
    
    Multiple observations may map to the same categorical state
    (categorical equivalence class).
    """
    category: str
    meaning: str
    confidence: float  # [0,1] - certainty about this categorization
    s_entropy: SEntropyCoordinates
    evidence: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    alternative_categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
    
    @property
    def s_distance(self) -> float:
        """Total S-entropy distance"""
        return self.s_entropy.magnitude
    
    @property
    def is_uncertain(self) -> bool:
        """High evolution entropy indicates uncertainty"""
        return self.s_entropy.S_e > 5.0
    
    @property
    def is_urgent(self) -> bool:
        """High temporal entropy indicates urgency"""
        return self.s_entropy.S_t > 5.0
    
    @property
    def information_content(self) -> float:
        """Knowledge entropy indicates information content"""
        return self.s_entropy.S_k
    
    def distance_to(self, other: 'CategoricalState') -> float:
        """Categorical distance to another state"""
        return self.s_entropy.distance_to(other.s_entropy)
    
    def is_equivalent_to(self, other: 'CategoricalState', threshold: float = 0.1) -> bool:
        """Check categorical equivalence"""
        return CategoricalDistance.are_equivalent(
            self.s_entropy, 
            other.s_entropy, 
            threshold
        )
    
    def add_evidence(self, evidence: str) -> None:
        """Add supporting evidence"""
        self.evidence.append(evidence)
    
    def add_implication(self, implication: str) -> None:
        """Add logical implication"""
        self.implications.append(implication)
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence with bounds checking"""
        self.confidence = max(0.0, min(1.0, new_confidence))
    
    def __repr__(self) -> str:
        return (f"CategoricalState('{self.category}', "
                f"S={self.s_distance:.2f}, conf={self.confidence:.2f})")


@dataclass
class Interpretation:
    """
    Result of filtering observation through a semantic lens.
    
    Key: Preserves alternative interpretations (non-destructive)
    """
    lens: SemanticLens
    primary_state: CategoricalState
    alternative_states: List[CategoricalState]
    raw_observation: Dict[str, Any]  # Original data (preserved)
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def rank_alternatives(self) -> List[CategoricalState]:
        """Rank all states by S-entropy (thermodynamic favorability)"""
        all_states = [self.primary_state] + self.alternative_states
        return sorted(all_states, key=lambda s: s.s_distance)
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get confidence distribution across all categorical states"""
        all_states = [self.primary_state] + self.alternative_states
        total_conf = sum(s.confidence for s in all_states)
        if total_conf == 0:
            return {}
        return {
            s.category: s.confidence / total_conf
            for s in all_states
        }
    
    def get_best_state(self) -> CategoricalState:
        """Return state with highest confidence"""
        all_states = [self.primary_state] + self.alternative_states
        return max(all_states, key=lambda s: s.confidence)
    
    def get_most_favorable(self) -> CategoricalState:
        """Return thermodynamically most favorable state (lowest S-entropy)"""
        return self.rank_alternatives()[0]
    
    def filter_by_confidence(self, threshold: float = 0.5) -> List[CategoricalState]:
        """Return states above confidence threshold"""
        all_states = [self.primary_state] + self.alternative_states
        return [s for s in all_states if s.confidence >= threshold]
    
    def __repr__(self) -> str:
        return (f"Interpretation({self.lens.value}, "
                f"primary={self.primary_state.category}, "
                f"alternatives={len(self.alternative_states)})")


class CategoricalEquivalenceClass:
    """
    Represents a class of categorically equivalent states
    
    All states in the class map to the same semantic interpretation
    despite potentially different surface features.
    """
    
    def __init__(self, representative: CategoricalState, threshold: float = 0.1):
        self.representative = representative
        self.threshold = threshold
        self.members: Set[str] = {representative.category}
        self.center = representative.s_entropy
    
    def contains(self, state: CategoricalState) -> bool:
        """Check if state belongs to this equivalence class"""
        return self.representative.is_equivalent_to(state, self.threshold)
    
    def add(self, state: CategoricalState) -> bool:
        """
        Add state to equivalence class if equivalent
        
        Returns: True if added, False if not equivalent
        """
        if self.contains(state):
            self.members.add(state.category)
            # Update center as centroid
            self._update_center(state.s_entropy)
            return True
        return False
    
    def _update_center(self, new_entropy: SEntropyCoordinates) -> None:
        """Update centroid of equivalence class"""
        alpha = 1.0 / len(self.members)
        self.center = SEntropyCoordinates(
            S_k=self.center.S_k * (1 - alpha) + new_entropy.S_k * alpha,
            S_t=self.center.S_t * (1 - alpha) + new_entropy.S_t * alpha,
            S_e=self.center.S_e * (1 - alpha) + new_entropy.S_e * alpha
        )
    
    def distance_to(self, state: CategoricalState) -> float:
        """Distance from state to class center"""
        return self.center.distance_to(state.s_entropy)
    
    def __len__(self) -> int:
        return len(self.members)
    
    def __repr__(self) -> str:
        return f"EquivalenceClass({self.representative.category}, members={len(self)})"


class CategoricalSpace:
    """
    Manage collection of categorical states and equivalence classes
    """
    
    def __init__(self, equivalence_threshold: float = 0.1):
        self.states: List[CategoricalState] = []
        self.equivalence_classes: List[CategoricalEquivalenceClass] = []
        self.equivalence_threshold = equivalence_threshold
    
    def add_state(self, state: CategoricalState) -> Optional[CategoricalEquivalenceClass]:
        """
        Add state to space and find/create equivalence class
        
        Returns: Equivalence class containing the state
        """
        self.states.append(state)
        
        # Try to add to existing equivalence class
        for eq_class in self.equivalence_classes:
            if eq_class.add(state):
                return eq_class
        
        # Create new equivalence class
        new_class = CategoricalEquivalenceClass(state, self.equivalence_threshold)
        self.equivalence_classes.append(new_class)
        return new_class
    
    def find_nearest(self, state: CategoricalState, k: int = 5) -> List[CategoricalState]:
        """Find k nearest states in categorical space"""
        if not self.states:
            return []
        
        distances = [(s, state.distance_to(s)) for s in self.states]
        distances.sort(key=lambda x: x[1])
        return [s for s, _ in distances[:k]]
    
    def find_equivalence_class(self, state: CategoricalState) -> Optional[CategoricalEquivalenceClass]:
        """Find equivalence class containing state"""
        for eq_class in self.equivalence_classes:
            if eq_class.contains(state):
                return eq_class
        return None
    
    def get_density(self, center: SEntropyCoordinates, radius: float = 1.0) -> int:
        """Count states within radius of center"""
        return sum(1 for s in self.states 
                  if center.distance_to(s.s_entropy) <= radius)
    
    def __len__(self) -> int:
        return len(self.states)

