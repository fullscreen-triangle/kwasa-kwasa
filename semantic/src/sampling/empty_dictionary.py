"""Empty dictionary synthesis - Real-time meaning construction"""

from typing import Dict, Any, List
from ..core.categorical_state import CategoricalState
from ..core.s_entropy import SEntropyCoordinates

class EmptyDictionary:
    """Synthesize meanings without stored definitions"""

    def __init__(self):
        self.generated_meanings: Dict[str, CategoricalState] = {}

    def synthesize(self, concept: str, context: Dict[str, Any]) -> CategoricalState:
        """Generate meaning for concept in real-time"""
        # Bayesian inference from context
        s_k = len(context) * 0.5  # More context = more knowledge
        s_t = context.get("urgency", 3.0)
        s_e = 10.0 / (len(concept) + 1)  # Shorter concepts more uncertain

        state = CategoricalState(
            category=f"synthesized_{concept}",
            meaning=f"Real-time synthesis of '{concept}' from context",
            confidence=min(1.0, len(context) / 10.0),
            s_entropy=SEntropyCoordinates(s_k, s_t, s_e),
            evidence=[f"{k}={v}" for k, v in list(context.items())[:3]]
        )

        self.generated_meanings[concept] = state
        return state

    def refine(self, concept: str, new_evidence: List[str]) -> CategoricalState:
        """Refine synthesized meaning with new evidence"""
        if concept not in self.generated_meanings:
            return self.synthesize(concept, {"evidence": new_evidence})

        current = self.generated_meanings[concept]

        # Update with new evidence
        current.evidence.extend(new_evidence)
        current.confidence = min(1.0, current.confidence * 1.1)

        # Reduce uncertainty
        current.s_entropy.S_e *= 0.9

        return current

