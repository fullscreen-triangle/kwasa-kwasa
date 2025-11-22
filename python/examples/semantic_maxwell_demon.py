"""
Semantic Maxwell Demon: Virtual Instrument for Non-Committal Semantic Filtering

This implements a BMD as a virtual instrument that allows exploration of multiple
semantic "preparation paths" simultaneously without irreversible commitment,
analogous to how a spectrometer can measure all wavelengths non-destructively.

Key Innovation:
    In physical experiments, you must commit to one preparation method (e.g., EM vs fluorescence).
    The Semantic Demon allows you to explore ALL semantic interpretations simultaneously
    without destroying the observation or blocking alternative views.

Core Operation:
    iCat = ℑ_input ∘ ℑ_output

    ℑ_input:  Filter relevant semantic states from vast potential space
    ℑ_output: Channel toward thermodynamically optimal categorical states

Author: Kundai Farai Sachikonye
Date: November 22, 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json


class SemanticLens(Enum):
    """Available semantic lenses (virtual preparation methods)"""
    PSYCHIATRIC = "psychiatric_dsm5"
    ENDOCRINE = "endocrine_metabolic"
    NEUROLOGICAL = "neurological_oscillatory"
    PSYCHOLOGICAL = "psychological_developmental"
    BIOCHEMICAL = "biochemical_molecular"
    CONTEXTUAL = "contextual_situational"


@dataclass
class CategoricalState:
    """
    A categorical state represents a distinct semantic interpretation.

    Multiple observations may map to the same categorical state
    (categorical equivalence class).
    """
    category: str
    meaning: str
    confidence: float  # [0,1] - how certain about this categorization
    s_entropy: Tuple[float, float, float]  # (S_k, S_t, S_e)
    evidence: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    alternative_categories: List[str] = field(default_factory=list)

    @property
    def s_distance(self) -> float:
        """Total S-entropy distance"""
        return np.linalg.norm(self.s_entropy)

    def __repr__(self):
        return f"CategoricalState('{self.category}', S={self.s_distance:.2f}, conf={self.confidence:.2f})"


@dataclass
class Interpretation:
    """
    Result of filtering observation through a semantic lens.

    Key: Preserves alternative interpretations (non-destructive)
    """
    lens: SemanticLens
    primary_state: CategoricalState
    alternative_states: List[CategoricalState]
    raw_observation: Dict  # Original data (preserved)

    def rank_alternatives(self) -> List[CategoricalState]:
        """Rank all states by S-entropy (thermodynamic favorability)"""
        all_states = [self.primary_state] + self.alternative_states
        return sorted(all_states, key=lambda s: s.s_distance)

    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get confidence distribution across all categorical states"""
        all_states = [self.primary_state] + self.alternative_states
        total_conf = sum(s.confidence for s in all_states)
        return {
            s.category: s.confidence / total_conf
            for s in all_states
        }


class SemanticMaxwellDemon:
    """
    Virtual Instrument for Non-Committal Semantic Filtering

    Acts as a "semantic spectrometer" that can explore all interpretations
    simultaneously without destructive commitment to any single view.

    This implements the BMD dual-filter architecture:
        ℑ_input:  Semantic relevance filtering (vast → relevant)
        ℑ_output: Thermodynamic channeling (relevant → optimal)
    """

    def __init__(
        self,
        s_entropy_target: Tuple[float, float, float] = (0.001, 0.001, 2.3),
        confidence_threshold: float = 0.05
    ):
        self.s_entropy_target = np.array(s_entropy_target)
        self.confidence_threshold = confidence_threshold

        # Performance metrics
        self.catalysis_log = []

    def filter(
        self,
        observation: Dict,
        lens: SemanticLens,
        context: Optional[Dict] = None
    ) -> Interpretation:
        """
        Filter observation through semantic lens (non-destructive).

        This is the core virtual instrument operation:
        - Observation remains unchanged
        - Returns interpretation through specified lens
        - Preserves access to alternative interpretations

        Args:
            observation: Raw data (numbers, text, measurements)
            lens: Which semantic "preparation method" to apply
            context: Additional context for interpretation

        Returns:
            Interpretation with primary + alternative categorical states
        """
        context = context or {}

        # Stage 1: Generate potential categorical states (Ω^POT)
        potential_states = self._generate_categorical_space(observation, lens, context)

        print(f"\n{'='*70}")
        print(f"SEMANTIC DEMON: Virtual Instrument Filtering")
        print(f"{'='*70}")
        print(f"Lens: {lens.value}")
        print(f"Observation: {self._format_observation(observation)}")
        print(f"Potential categorical states: {len(potential_states)}")

        # Stage 2: Input filter (ℑ_input) - semantic relevance
        relevant_states = self._input_filter(potential_states, observation, context)
        print(f"After input filter (ℑ_input): {len(relevant_states)} relevant states")

        # Stage 3: Output filter (ℑ_output) - thermodynamic channeling
        channeled_states = self._output_filter(relevant_states, context)
        print(f"After output filter (ℑ_output): {len(channeled_states)} optimal states")

        # Stage 4: Select primary state (minimum S-entropy)
        primary = min(channeled_states, key=lambda s: s.s_distance)
        alternatives = [s for s in channeled_states if s != primary]

        print(f"\nSelected categorical state: {primary.category}")
        print(f"S-entropy: {primary.s_entropy} (distance: {primary.s_distance:.3f})")
        print(f"Confidence: {primary.confidence:.3f}")
        print(f"Alternative states preserved: {len(alternatives)}")
        print(f"{'='*70}\n")

        # Log catalysis event
        self.catalysis_log.append({
            'lens': lens.value,
            'potential_cardinality': len(potential_states),
            'actual_cardinality': len(channeled_states),
            'reduction_ratio': len(channeled_states) / len(potential_states),
            'selected_category': primary.category,
            's_distance': primary.s_distance
        })

        return Interpretation(
            lens=lens,
            primary_state=primary,
            alternative_states=alternatives,
            raw_observation=observation.copy()  # Preserve original
        )

    def filter_all(
        self,
        observation: Dict,
        lenses: List[SemanticLens],
        context: Optional[Dict] = None
    ) -> List[Interpretation]:
        """
        Apply multiple semantic lenses simultaneously (key capability!).

        This demonstrates the virtual instrument advantage:
        - In physical experiment: Must commit to ONE preparation method
        - With semantic demon: Explore ALL preparation methods simultaneously
        """
        print(f"\n{'='*70}")
        print(f"MULTI-LENS SEMANTIC FILTERING (Virtual Instrument)")
        print(f"{'='*70}")
        print(f"Applying {len(lenses)} semantic lenses simultaneously")
        print(f"This is IMPOSSIBLE in physical experiments (destructive commitment)")
        print(f"But TRIVIAL with Semantic Maxwell Demon (non-destructive)")
        print(f"{'='*70}\n")

        interpretations = []
        for lens in lenses:
            interpretation = self.filter(observation, lens, context)
            interpretations.append(interpretation)

        return interpretations

    def compare_interpretations(
        self,
        interpretations: List[Interpretation]
    ) -> Dict:
        """
        Compare interpretations from different lenses to find optimal view.

        Returns ranking by:
        1. S-entropy distance (thermodynamic favorability)
        2. Confidence (evidential support)
        3. Coherence with context
        """
        print(f"\n{'='*70}")
        print(f"COMPARING SEMANTIC INTERPRETATIONS")
        print(f"{'='*70}\n")

        comparisons = []
        for interp in interpretations:
            primary = interp.primary_state
            comparisons.append({
                'lens': interp.lens.value,
                'category': primary.category,
                'meaning': primary.meaning,
                's_distance': primary.s_distance,
                'confidence': primary.confidence,
                'score': primary.confidence / max(primary.s_distance, 0.1)  # Combined metric
            })

        # Sort by score (confidence / s_distance)
        comparisons.sort(key=lambda x: x['score'], reverse=True)

        print(f"{'Rank':<6}{'Lens':<25}{'Category':<30}{'S-dist':<10}{'Conf':<10}{'Score':<10}")
        print(f"{'-'*100}")
        for i, comp in enumerate(comparisons, 1):
            print(f"{i:<6}{comp['lens']:<25}{comp['category']:<30}"
                  f"{comp['s_distance']:<10.3f}{comp['confidence']:<10.3f}{comp['score']:<10.3f}")

        print(f"\n{'='*70}")
        print(f"OPTIMAL INTERPRETATION: {comparisons[0]['lens']} → {comparisons[0]['category']}")
        print(f"{'='*70}\n")

        return {
            'ranked_interpretations': comparisons,
            'optimal': comparisons[0],
            'catalysis_efficiency': self._calculate_catalysis_efficiency()
        }

    def measure_catalysis_effect(self) -> Dict:
        """
        Measure the information catalysis effect:
        How much state space reduction achieved?
        """
        if not self.catalysis_log:
            return {}

        total_potential = sum(log['potential_cardinality'] for log in self.catalysis_log)
        total_actual = sum(log['actual_cardinality'] for log in self.catalysis_log)

        reduction_ratio = total_actual / total_potential
        orders_magnitude = np.log10(total_potential / total_actual)

        return {
            'total_potential_states': total_potential,
            'total_actual_states': total_actual,
            'reduction_ratio': reduction_ratio,
            'orders_of_magnitude_reduced': orders_magnitude,
            'average_s_distance': np.mean([log['s_distance'] for log in self.catalysis_log]),
            'catalysis_events': len(self.catalysis_log)
        }

    # ========================================================================
    # Internal Methods: Implement BMD Dual-Filter Architecture
    # ========================================================================

    def _generate_categorical_space(
        self,
        observation: Dict,
        lens: SemanticLens,
        context: Dict
    ) -> List[CategoricalState]:
        """
        Generate potential categorical states (Ω^POT).

        This is the vast space of possible interpretations.
        """
        states = []

        # Different lenses generate different categorical spaces
        if lens == SemanticLens.PSYCHIATRIC:
            states.extend(self._psychiatric_categorical_space(observation, context))
        elif lens == SemanticLens.NEUROLOGICAL:
            states.extend(self._neurological_categorical_space(observation, context))
        elif lens == SemanticLens.ENDOCRINE:
            states.extend(self._endocrine_categorical_space(observation, context))
        elif lens == SemanticLens.PSYCHOLOGICAL:
            states.extend(self._psychological_categorical_space(observation, context))

        return states

    def _input_filter(
        self,
        potential_states: List[CategoricalState],
        observation: Dict,
        context: Dict
    ) -> List[CategoricalState]:
        """
        ℑ_input: Semantic relevance filtering.

        Reduces vast potential space to semantically relevant states.
        """
        # Filter by confidence threshold
        relevant = [s for s in potential_states if s.confidence > self.confidence_threshold]

        # Filter by context compatibility
        if 'domain' in context:
            relevant = [s for s in relevant if self._context_compatible(s, context)]

        return relevant

    def _output_filter(
        self,
        relevant_states: List[CategoricalState],
        context: Dict
    ) -> List[CategoricalState]:
        """
        ℑ_output: Thermodynamic channeling.

        Selects states that minimize S-entropy (thermodynamically optimal).
        """
        # Sort by S-entropy distance
        sorted_states = sorted(relevant_states, key=lambda s: s.s_distance)

        # Keep top candidates (within factor of 2 of minimum)
        min_distance = sorted_states[0].s_distance
        threshold = min_distance * 2.0

        channeled = [s for s in sorted_states if s.s_distance <= threshold]

        return channeled

    def _psychiatric_categorical_space(self, obs: Dict, ctx: Dict) -> List[CategoricalState]:
        """Generate psychiatric interpretations (DSM-5 lens)"""
        states = []

        # Major Depressive Disorder
        states.append(CategoricalState(
            category="Major Depressive Disorder",
            meaning="Clinical depression meeting DSM-5 criteria",
            confidence=0.82,
            s_entropy=(0.08, 0.05, 2.3),
            evidence=["Low mood >2 weeks", "Anhedonia", "Neurovegetative symptoms"],
            implications=["SSRI + psychotherapy", "Monitor suicidal ideation"],
            alternative_categories=["Dysthymia", "Adjustment Disorder"]
        ))

        # Bipolar Depression
        states.append(CategoricalState(
            category="Bipolar Disorder (Depressive Episode)",
            meaning="Bipolar depression, need history clarification",
            confidence=0.25,
            s_entropy=(0.52, 0.45, 3.8),
            evidence=["Current depressive symptoms"],
            implications=["Avoid SSRI monotherapy", "Mood stabilizer needed"],
            alternative_categories=["Major Depression", "Mixed Episode"]
        ))

        # Adjustment Disorder
        states.append(CategoricalState(
            category="Adjustment Disorder",
            meaning="Stress-related mood disturbance",
            confidence=0.35,
            s_entropy=(0.32, 0.28, 3.2),
            evidence=["Identifiable stressor possible"],
            implications=["Supportive therapy", "Time-limited"],
            alternative_categories=["Major Depression", "Normal Grief"]
        ))

        return states

    def _neurological_categorical_space(self, obs: Dict, ctx: Dict) -> List[CategoricalState]:
        """Generate neurological interpretations (oscillatory coherence lens)"""
        states = []

        # Get PLV if available
        plv = obs.get('plv', obs.get('phase_locking_value', 0.32))

        # Impaired Oscillatory Coherence
        if plv < 0.50:
            states.append(CategoricalState(
                category="Impaired Theta-Gamma Coupling",
                meaning="Reduced oscillatory phase coherence in emotion circuits",
                confidence=0.88,
                s_entropy=(0.05, 0.03, 2.1),
                evidence=[f"PLV={plv:.2f} (healthy>0.65)", "Theta band coupling weak"],
                implications=["Target phase coherence", "Enhance PFC-amygdala coupling"],
                alternative_categories=["Network Dysconnection", "Oscillatory Desynchrony"]
            ))
        else:
            states.append(CategoricalState(
                category="Restored Oscillatory Coherence",
                meaning="Normal phase-locking in emotion regulation circuits",
                confidence=0.91,
                s_entropy=(0.02, 0.01, 2.0),
                evidence=[f"PLV={plv:.2f} (approaching healthy)", "Improved coupling"],
                implications=["Coherence restoration successful", "Maintain treatment"],
                alternative_categories=["Partial Recovery", "Compensatory Synchrony"]
            ))

        return states

    def _endocrine_categorical_space(self, obs: Dict, ctx: Dict) -> List[CategoricalState]:
        """Generate endocrine interpretations (metabolic lens)"""
        states = []

        # Hypothyroidism
        states.append(CategoricalState(
            category="Possible Hypothyroidism",
            meaning="Thyroid dysfunction mimicking depression",
            confidence=0.35,
            s_entropy=(0.65, 0.58, 3.5),
            evidence=["Fatigue", "Cognitive slowing"],
            implications=["Check TSH, T4", "Rule out endocrine cause"],
            alternative_categories=["Primary Depression", "Subclinical Hypothyroidism"]
        ))

        return states

    def _psychological_categorical_space(self, obs: Dict, ctx: Dict) -> List[CategoricalState]:
        """Generate psychological interpretations (developmental lens)"""
        states = []

        # Normal Grief
        if ctx.get('recent_loss', False):
            states.append(CategoricalState(
                category="Normal Grief Reaction",
                meaning="Adaptive bereavement response",
                confidence=0.72,
                s_entropy=(0.18, 0.22, 2.8),
                evidence=["Recent loss", "Contextually appropriate"],
                implications=["Supportive therapy", "Monitor progression"],
                alternative_categories=["Complicated Grief", "Major Depression"]
            ))

        return states

    def _context_compatible(self, state: CategoricalState, context: Dict) -> bool:
        """Check if categorical state is compatible with context"""
        # Simple heuristic: states with lower S-entropy are preferred
        return state.s_distance < 4.0

    def _calculate_catalysis_efficiency(self) -> float:
        """Calculate overall information catalysis efficiency"""
        if not self.catalysis_log:
            return 0.0

        total_reduction = sum(
            np.log10(log['potential_cardinality'] / log['actual_cardinality'])
            for log in self.catalysis_log
        )

        return total_reduction / len(self.catalysis_log)

    def _format_observation(self, obs: Dict) -> str:
        """Format observation for display"""
        if 'plv' in obs:
            return f"PLV={obs['plv']:.2f}, symptoms={obs.get('symptoms', 'N/A')}"
        return str(obs)[:50] + "..."


# ========================================================================
# Comparison: WITH vs WITHOUT Semantic Demon
# ========================================================================

def compare_with_without_demon(observation: Dict, context: Dict):
    """
    Demonstrate the difference between:
    1. Traditional single-path commitment (WITHOUT demon)
    2. Multi-path exploration (WITH demon)
    """
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT Semantic Maxwell Demon")
    print("="*70)

    print("\n" + "-"*70)
    print("WITHOUT DEMON: Traditional Single-Path Commitment")
    print("-"*70)
    print("Problem: Must commit to ONE interpretation path irreversibly")
    print("Like choosing EM preparation: destroys sample for other methods\n")

    # Simulate traditional approach: commit to psychiatric view only
    print("Committed path: Psychiatric DSM-5")
    print("Result: Major Depressive Disorder (confidence: 0.82)")
    print("S-entropy: (0.08, 0.05, 2.3) → distance: 2.31")
    print("\nLimitations:")
    print("  ✗ Cannot explore neurological interpretation")
    print("  ✗ Cannot explore endocrine interpretation")
    print("  ✗ Cannot compare alternative views")
    print("  ✗ Locked into single semantic frame")
    print("  ✗ Potential optimal interpretation missed")

    print("\n" + "-"*70)
    print("WITH DEMON: Multi-Path Exploration (Virtual Instrument)")
    print("-"*70)
    print("Advantage: Explore ALL interpretation paths simultaneously")
    print("Like spectrometer: measure all wavelengths non-destructively\n")

    # Use demon to explore multiple lenses
    demon = SemanticMaxwellDemon()

    lenses = [
        SemanticLens.PSYCHIATRIC,
        SemanticLens.NEUROLOGICAL,
        SemanticLens.ENDOCRINE,
        SemanticLens.PSYCHOLOGICAL
    ]

    interpretations = demon.filter_all(observation, lenses, context)

    # Compare interpretations
    comparison = demon.compare_interpretations(interpretations)

    # Show catalysis effect
    catalysis = demon.measure_catalysis_effect()

    print(f"\n{'='*70}")
    print(f"INFORMATION CATALYSIS EFFECT")
    print(f"{'='*70}")
    print(f"State space reduction:")
    print(f"  Potential states (Ω^POT): {catalysis['total_potential_states']:,}")
    print(f"  Actual states (Ω^ACT): {catalysis['total_actual_states']:,}")
    print(f"  Reduction ratio: {catalysis['reduction_ratio']:.2e}")
    print(f"  Orders of magnitude: {catalysis['orders_of_magnitude_reduced']:.1f}x")
    print(f"  Average S-distance: {catalysis['average_s_distance']:.3f}")
    print(f"\nBenefits of Virtual Instrument:")
    print(f"  ✓ Explored {len(lenses)} semantic paths simultaneously")
    print(f"  ✓ Found optimal interpretation: {comparison['optimal']['category']}")
    print(f"  ✓ Preserved alternative views for consideration")
    print(f"  ✓ Non-destructive: can re-filter with new lenses anytime")
    print(f"  ✓ Thermodynamically optimized (minimum S-entropy)")
    print(f"{'='*70}\n")

    return {
        'interpretations': interpretations,
        'comparison': comparison,
        'catalysis': catalysis
    }


if __name__ == "__main__":
    # Demonstration
    print("\n" + "="*70)
    print("SEMANTIC MAXWELL DEMON: Virtual Instrument Demonstration")
    print("="*70)

    # Sample observation (like from your depression data)
    observation = {
        'plv': 0.77,
        'symptoms': 'low mood, anhedonia, fatigue',
        'duration_weeks': 6,
        'baseline_plv': 0.32
    }

    context = {
        'patient_age': 34,
        'treatment': 'SSRI',
        'recent_loss': False,
        'domain': 'clinical'
    }

    # Run comparison
    results = compare_with_without_demon(observation, context)

