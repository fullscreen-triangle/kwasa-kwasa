"""Query formation detection"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from ..bmd.bmd_state import BMDState
from ..core.categorical_state import CategoricalState

class QueryFormationState(Enum):
    """State of query formation"""
    NOT_FORMING = "not_forming"
    FORMING = "forming"
    READY = "ready"

@dataclass
class QueryDetector:
    """Detect when user is forming a query"""
    
    def detect(self, bmd_state: BMDState) -> tuple[QueryFormationState, float]:
        """Detect query formation state"""
        s_k, s_t, s_e = bmd_state.s_entropy.S_k, bmd_state.s_entropy.S_t, bmd_state.s_entropy.S_e
        
        if s_k > 5.0 and s_e > 6.0:
            return QueryFormationState.READY, 0.9
        elif s_k > 3.0 and s_e > 4.0:
            return QueryFormationState.FORMING, 0.6
        else:
            return QueryFormationState.NOT_FORMING, 0.3

