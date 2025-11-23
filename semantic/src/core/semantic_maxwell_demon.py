"""
Semantic Maxwell Demon: Virtual Instrument for Non-Committal Semantic Filtering

This implements a BMD as a virtual instrument that allows exploration of multiple
semantic "preparation paths" simultaneously without irreversible commitment.

Core Operation:
    iCat = ℑ_input ∘ ℑ_output

    ℑ_input:  Filter relevant semantic states from vast potential space
    ℑ_output: Channel toward thermodynamically optimal categorical states

Author: Kundai Farai Sachikonye
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import numpy as np

from .categorical_state import (
    CategoricalState, Interpretation, SemanticLens,
    CategoricalSpace
)
from .s_entropy import SEntropyCoordinates, SEntropyCalculator


class SemanticMaxwellDemon:
    """
    Virtual Instrument for Non-Committal Semantic Filtering

    Acts as a "semantic spectrometer" that can explore all interpretations
    simultaneously without destructive commitment to any single view.

    This implements the BMD dual-filter architecture:
        ℑ_input:  Semantic relevance filtering (vast → relevant)
        ℑ_output: Thermodynamic optimization (relevant → favorable)
    """

    def __init__(
        self,
        available_lenses: Optional[List[SemanticLens]] = None,
        equivalence_threshold: float = 0.1,
        confidence_threshold: float = 0.3
    ):
        """
        Initialize Semantic Maxwell Demon

        Args:
            available_lenses: Semantic lenses available for filtering
            equivalence_threshold: Distance threshold for categorical equivalence
            confidence_threshold: Minimum confidence for state inclusion
        """
        self.available_lenses = available_lenses or list(SemanticLens)
        self.equivalence_threshold = equivalence_threshold
        self.confidence_threshold = confidence_threshold
        self.categorical_space = CategoricalSpace(equivalence_threshold)

        # History for learning
        self.interpretation_history: List[Interpretation] = []

    def filter(
        self,
        observation: Dict[str, Any],
        lens: Optional[SemanticLens] = None,
        preserve_alternatives: bool = True
    ) -> Interpretation:
        """
        Filter observation through semantic lens

        This is the core BMD operation: ℑ_input ∘ ℑ_output

        Args:
            observation: Raw data to interpret
            lens: Specific semantic lens (or None for multi-lens)
            preserve_alternatives: Keep alternative interpretations

        Returns:
            Interpretation with primary and alternative states
        """
        # If no specific lens, try all lenses and return best
        if lens is None:
            return self._multi_lens_filter(observation, preserve_alternatives)

        # ℑ_input: Generate potential categorical states
        potential_states = self._input_filter(observation, lens)

        # ℑ_output: Select thermodynamically favorable states
        primary, alternatives = self._output_filter(potential_states)

        # Create interpretation
        interpretation = Interpretation(
            lens=lens,
            primary_state=primary,
            alternative_states=alternatives if preserve_alternatives else [],
            raw_observation=observation
        )

        # Add to history
        self.interpretation_history.append(interpretation)

        # Update categorical space
        self.categorical_space.add_state(primary)
        for alt in alternatives:
            self.categorical_space.add_state(alt)

        return interpretation

    def filter_all(
        self,
        observation: Dict[str, Any]
    ) -> Dict[SemanticLens, Interpretation]:
        """
        Apply all available semantic lenses simultaneously

        This is the key non-committal capability: explore ALL interpretations
        without destroying the observation.

        Returns:
            Dictionary mapping each lens to its interpretation
        """
        results = {}
        for lens in self.available_lenses:
            results[lens] = self.filter(observation, lens, preserve_alternatives=True)
        return results

    def compare_interpretations(
        self,
        observation: Dict[str, Any],
        lenses: Optional[List[SemanticLens]] = None
    ) -> Dict[str, Any]:
        """
        Compare interpretations across different semantic lenses

        Returns:
            Analysis showing agreement, divergence, and confidence across lenses
        """
        lenses = lenses or self.available_lenses
        interpretations = {lens: self.filter(observation, lens) for lens in lenses}

        # Extract primary categories
        categories = [interp.primary_state.category for interp in interpretations.values()]

        # Calculate agreement
        unique_categories = set(categories)
        agreement = 1.0 - (len(unique_categories) - 1) / max(1, len(categories))

        # Calculate average confidence
        avg_confidence = np.mean([
            interp.primary_state.confidence
            for interp in interpretations.values()
        ])

        # Find consensus (most common category)
        from collections import Counter
        category_counts = Counter(categories)
        consensus = category_counts.most_common(1)[0][0] if category_counts else None

        # Calculate S-entropy divergence
        s_entropies = [interp.primary_state.s_entropy for interp in interpretations.values()]
        s_divergence = self._calculate_entropy_divergence(s_entropies)

        return {
            "interpretations": interpretations,
            "agreement": agreement,
            "average_confidence": avg_confidence,
            "consensus_category": consensus,
            "s_entropy_divergence": s_divergence,
            "unique_categories": list(unique_categories)
        }

    def _input_filter(
        self,
        observation: Dict[str, Any],
        lens: SemanticLens
    ) -> List[CategoricalState]:
        """
        ℑ_input: Generate potential categorical states from observation

        This is the "input filter" of the Maxwell Demon that selects
        relevant states from the vast potential space.
        """
        # Extract features from observation based on lens
        features = self._extract_features(observation, lens)

        # Generate categorical states
        states = []

        # Primary interpretation
        primary_category, primary_meaning = self._interpret_features(features, lens)
        s_entropy = self._calculate_s_entropy(features, lens)
        confidence = self._calculate_confidence(features, lens)

        primary_state = CategoricalState(
            category=primary_category,
            meaning=primary_meaning,
            confidence=confidence,
            s_entropy=s_entropy,
            evidence=[str(k) + "=" + str(v) for k, v in features.items()],
            metadata={"lens": lens.value}
        )
        states.append(primary_state)

        # Generate alternatives
        alternatives = self._generate_alternatives(features, lens, primary_state)
        states.extend(alternatives)

        return states

    def _output_filter(
        self,
        states: List[CategoricalState]
    ) -> Tuple[CategoricalState, List[CategoricalState]]:
        """
        ℑ_output: Select thermodynamically favorable states

        This is the "output filter" that channels toward states with
        lower S-entropy (thermodynamically favorable).
        """
        if not states:
            # Create default uncertain state
            default = CategoricalState(
                category="unknown",
                meaning="Unable to determine category",
                confidence=0.0,
                s_entropy=SEntropyCoordinates(10.0, 10.0, 10.0)
            )
            return default, []

        # Sort by thermodynamic favorability (lower S-entropy)
        # But also consider confidence
        def favorability_score(state: CategoricalState) -> float:
            # Balance S-entropy (lower better) with confidence (higher better)
            return state.confidence / (1.0 + state.s_distance)

        sorted_states = sorted(states, key=favorability_score, reverse=True)

        # Primary is most favorable
        primary = sorted_states[0]

        # Alternatives are other high-confidence states
        alternatives = [
            s for s in sorted_states[1:]
            if s.confidence >= self.confidence_threshold
        ]

        return primary, alternatives

    def _multi_lens_filter(
        self,
        observation: Dict[str, Any],
        preserve_alternatives: bool
    ) -> Interpretation:
        """
        Apply multiple lenses and return best interpretation
        """
        all_interpretations = self.filter_all(observation)

        # Find interpretation with highest primary confidence
        best_lens = max(
            all_interpretations.keys(),
            key=lambda l: all_interpretations[l].primary_state.confidence
        )

        return all_interpretations[best_lens]

    def _extract_features(
        self,
        observation: Dict[str, Any],
        lens: SemanticLens
    ) -> Dict[str, Any]:
        """
        Extract relevant features based on semantic lens

        Different lenses emphasize different aspects of the observation.
        """
        features = {}

        # Common features
        if "text" in observation:
            features["text_length"] = len(observation["text"])
            features["text_content"] = observation["text"]

        if "values" in observation:
            features["numeric_mean"] = np.mean(observation["values"])
            features["numeric_std"] = np.std(observation["values"])

        # Lens-specific features
        if lens == SemanticLens.PSYCHIATRIC:
            features["symptom_keywords"] = self._extract_psychiatric_keywords(observation)
        elif lens == SemanticLens.NEUROLOGICAL:
            features["neural_patterns"] = self._extract_neural_patterns(observation)
        elif lens == SemanticLens.BIOCHEMICAL:
            features["molecular_markers"] = self._extract_molecular_markers(observation)

        return features

    def _interpret_features(
        self,
        features: Dict[str, Any],
        lens: SemanticLens
    ) -> Tuple[str, str]:
        """
        Interpret features to generate category and meaning

        Returns: (category, meaning)
        """
        # This would be implemented with domain-specific logic
        # For now, provide generic interpretation

        category = f"{lens.value}_category"
        meaning = f"Interpretation through {lens.value} lens"

        # Add some basic logic based on features
        if "text_content" in features:
            text = features["text_content"]
            if "depression" in text.lower():
                category = "mood_disorder"
                meaning = "Indicators of depressive state"
            elif "anxiety" in text.lower():
                category = "anxiety_disorder"
                meaning = "Indicators of anxious state"

        return category, meaning

    def _calculate_s_entropy(
        self,
        features: Dict[str, Any],
        lens: SemanticLens
    ) -> SEntropyCoordinates:
        """
        Calculate S-entropy coordinates from features
        """
        # Knowledge entropy: Information content
        S_k = len(features) * 0.5  # More features = more information

        # Temporal entropy: Urgency/time-criticality
        S_t = features.get("urgency", 1.0)

        # Evolution entropy: Uncertainty in state
        if "confidence_raw" in features:
            S_e = (1.0 - features["confidence_raw"]) * 10
        else:
            S_e = 5.0  # Default moderate uncertainty

        return SEntropyCoordinates(S_k, S_t, S_e)

    def _calculate_confidence(
        self,
        features: Dict[str, Any],
        lens: SemanticLens
    ) -> float:
        """
        Calculate confidence in interpretation
        """
        # Base confidence on feature completeness
        required_features = 5
        available_features = len(features)

        confidence = min(1.0, available_features / required_features)

        # Adjust based on feature quality
        if "text_content" in features and len(features["text_content"]) > 100:
            confidence *= 1.2

        return min(1.0, confidence)

    def _generate_alternatives(
        self,
        features: Dict[str, Any],
        lens: SemanticLens,
        primary: CategoricalState
    ) -> List[CategoricalState]:
        """
        Generate alternative categorical interpretations
        """
        alternatives = []

        # Generate 2-3 alternatives with slightly different interpretations
        for i in range(2):
            alt_category = f"{primary.category}_variant_{i+1}"
            alt_meaning = f"Alternative interpretation: {primary.meaning}"

            # Slightly perturb S-entropy
            s_entropy_alt = SEntropyCoordinates(
                S_k=primary.s_entropy.S_k * (1.0 + 0.1 * (i + 1)),
                S_t=primary.s_entropy.S_t * (1.0 + 0.1 * (i + 1)),
                S_e=primary.s_entropy.S_e * (1.0 + 0.2 * (i + 1))
            )

            # Lower confidence for alternatives
            alt_confidence = primary.confidence * (0.8 - 0.1 * i)

            alt_state = CategoricalState(
                category=alt_category,
                meaning=alt_meaning,
                confidence=alt_confidence,
                s_entropy=s_entropy_alt,
                alternative_categories=[primary.category],
                metadata={"is_alternative": True}
            )

            alternatives.append(alt_state)

        return alternatives

    def _calculate_entropy_divergence(
        self,
        s_entropies: List[SEntropyCoordinates]
    ) -> float:
        """
        Calculate divergence in S-entropy across multiple interpretations
        """
        if len(s_entropies) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(s_entropies)):
            for j in range(i + 1, len(s_entropies)):
                distances.append(s_entropies[i].distance_to(s_entropies[j]))

        return np.mean(distances)

    def _extract_psychiatric_keywords(self, observation: Dict[str, Any]) -> List[str]:
        """Extract psychiatric symptom keywords"""
        keywords = ["depression", "anxiety", "mood", "sleep", "appetite", "energy"]
        text = observation.get("text", "").lower()
        return [kw for kw in keywords if kw in text]

    def _extract_neural_patterns(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract neurological patterns"""
        return {"pattern_type": "neural", "complexity": 1.0}

    def _extract_molecular_markers(self, observation: Dict[str, Any]) -> List[str]:
        """Extract biochemical markers"""
        return observation.get("markers", [])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about demon's operation
        """
        return {
            "total_interpretations": len(self.interpretation_history),
            "categorical_states": len(self.categorical_space),
            "equivalence_classes": len(self.categorical_space.equivalence_classes),
            "average_confidence": np.mean([
                interp.primary_state.confidence
                for interp in self.interpretation_history
            ]) if self.interpretation_history else 0.0
        }

