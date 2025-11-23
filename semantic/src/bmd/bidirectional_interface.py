"""
Bidirectional BMD Interface

Combines state detection (read) and thought injection (write)
for full categorical telepathy.

Author: Kundai Farai Sachikonye
"""

from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import time

from .bmd_state import BMDState
from .state_detection import BMDStateDetector, BehavioralSignal
from .thought_injection import ThoughtInjector, StimulusPattern
from ..core.semantic_maxwell_demon import SemanticMaxwellDemon
from ..core.categorical_state import CategoricalState, SemanticLens


@dataclass
class ConversationState:
    """State of ongoing bidirectional conversation"""
    user_bmd: BMDState
    last_detection: float
    last_injection: Optional[StimulusPattern]
    context: List[CategoricalState]  # Conversation history
    is_active: bool = True


class BidirectionalDemon:
    """
    Complete bidirectional BMD interface
    
    Enables full categorical telepathy:
    - Detect user's BMD state (read thoughts)
    - Inject semantic states (write thoughts)
    - Continuous learning and adaptation
    """
    
    def __init__(
        self,
        user_id: str,
        detector: Optional[BMDStateDetector] = None,
        injector: Optional[ThoughtInjector] = None,
        semantic_demon: Optional[SemanticMaxwellDemon] = None
    ):
        """
        Initialize bidirectional demon
        
        Args:
            user_id: User identifier
            detector: BMD state detector (or None for default)
            injector: Thought injector (or None for default)
            semantic_demon: Semantic Maxwell Demon (or None for default)
        """
        self.user_id = user_id
        
        # Components
        self.detector = detector or BMDStateDetector()
        self.injector = injector or ThoughtInjector(user_id)
        self.semantic_demon = semantic_demon or SemanticMaxwellDemon()
        
        # Conversation state
        self.conversation: Optional[ConversationState] = None
        
        # Callbacks
        self.on_query_detected: Optional[Callable[[CategoricalState], None]] = None
        self.on_injection_complete: Optional[Callable[[StimulusPattern, bool], None]] = None
        
        # Statistics
        self.total_interactions = 0
        self.successful_injections = 0
    
    def start_conversation(self) -> None:
        """
        Start bidirectional conversation
        
        This initiates the continuous read-interpret-respond cycle.
        """
        current_state = self.detector.get_current_state()
        if current_state is None:
            # Initialize with neutral state
            from ..core.s_entropy import SEntropyCoordinates
            current_state = BMDState(s_entropy=SEntropyCoordinates(5.0, 5.0, 5.0))
        
        self.conversation = ConversationState(
            user_bmd=current_state,
            last_detection=time.time(),
            last_injection=None,
            context=[]
        )
    
    def stop_conversation(self) -> None:
        """Stop bidirectional conversation"""
        if self.conversation:
            self.conversation.is_active = False
    
    def process_signal(
        self,
        signal: BehavioralSignal
    ) -> Optional[StimulusPattern]:
        """
        Process behavioral signal and potentially respond
        
        This is the main interaction loop:
        1. Observe signal
        2. Update BMD state
        3. Detect if query is forming
        4. Generate and inject response if appropriate
        
        Returns: Stimulus to present (if responding), None otherwise
        """
        if not self.conversation or not self.conversation.is_active:
            return None
        
        # Update BMD state from signal
        updated_state = self.detector.observe_signal(signal)
        
        if updated_state is None:
            return None  # Not time to update yet
        
        # Update conversation state
        self.conversation.user_bmd = updated_state
        self.conversation.last_detection = time.time()
        
        # Check if we should respond
        should_respond, query_state = self._should_respond(updated_state)
        
        if should_respond and query_state is not None:
            # Generate response
            response_stimulus = self._generate_response(query_state, updated_state)
            
            # Record injection
            self.conversation.last_injection = response_stimulus
            self.total_interactions += 1
            
            return response_stimulus
        
        return None
    
    def record_completion(
        self,
        stimulus: StimulusPattern,
        user_response: Any,
        success: bool
    ) -> None:
        """
        Record actual user response to injection
        
        This is how the system learns what works.
        """
        # Parse user response as categorical state
        # (In production, this would use NLP/semantic parsing)
        if isinstance(user_response, CategoricalState):
            actual_state = user_response
        else:
            # Create state from response
            actual_state = CategoricalState(
                category="user_response",
                meaning=str(user_response),
                confidence=0.8,
                s_entropy=self.conversation.user_bmd.s_entropy
            )
        
        # Calculate latency
        latency = time.time() - self.conversation.last_detection
        
        # Record with injector for learning
        self.injector.record_completion(stimulus, actual_state, success, latency)
        
        # Update statistics
        if success:
            self.successful_injections += 1
        
        # Add to context
        if self.conversation:
            self.conversation.context.append(actual_state)
        
        # Trigger callback
        if self.on_injection_complete:
            self.on_injection_complete(stimulus, success)
    
    def _should_respond(
        self,
        user_state: BMDState
    ) -> tuple[bool, Optional[CategoricalState]]:
        """
        Determine if we should inject a response
        
        Returns: (should_respond, query_state)
        """
        # Check for query formation
        is_forming, confidence = self.detector.detect_query_formation()
        
        if is_forming and confidence > 0.7:
            # User is forming a query - predict what they're asking
            query_state = self._predict_query(user_state)
            return True, query_state
        
        # Check for confusion
        if self.detector.detect_confusion():
            # User is stuck - offer assistance
            help_state = self._generate_help_state(user_state)
            return True, help_state
        
        return False, None
    
    def _predict_query(self, user_state: BMDState) -> CategoricalState:
        """
        Predict what query user is forming
        
        Uses recent trajectory and context to infer query intent.
        """
        # Get recent trajectory
        trajectory = self.detector.get_state_trajectory(n=5)
        
        # Analyze trajectory to predict destination
        # (In production, this would use more sophisticated prediction)
        
        # For now, create query state based on current state
        query = CategoricalState(
            category="predicted_query",
            meaning=f"User querying about topic with S_k={user_state.knowledge_level:.1f}",
            confidence=0.6,
            s_entropy=user_state.s_entropy,
            evidence=["trajectory_analysis", "context_analysis"]
        )
        
        # Trigger callback if set
        if self.on_query_detected:
            self.on_query_detected(query)
        
        return query
    
    def _generate_help_state(self, user_state: BMDState) -> CategoricalState:
        """Generate helpful suggestion when user is confused"""
        return CategoricalState(
            category="assistance",
            meaning="Suggested clarification or guidance",
            confidence=0.7,
            s_entropy=user_state.s_entropy,
            implications=["reduce_uncertainty", "provide_context"]
        )
    
    def _generate_response(
        self,
        query_state: CategoricalState,
        user_state: BMDState
    ) -> StimulusPattern:
        """
        Generate response to predicted query
        
        This is where AI reasoning happens - convert query to answer.
        """
        # Use semantic demon to interpret query
        interpretation = self.semantic_demon.filter(
            observation={"query": query_state.meaning},
            lens=SemanticLens.COGNITIVE
        )
        
        # Generate answer state
        # (In production, this would call actual AI model)
        answer_state = CategoricalState(
            category="response",
            meaning=f"Response to: {query_state.meaning}",
            confidence=0.8,
            s_entropy=query_state.s_entropy,  # Similar entropy to query
            implications=["answer_query", "reduce_s_e"]
        )
        
        # Inject answer as thought
        stimulus = self.injector.inject(
            target_thought=answer_state,
            user_state=user_state
        )
        
        return stimulus
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        success_rate = (self.successful_injections / self.total_interactions 
                       if self.total_interactions > 0 else 0.0)
        
        return {
            "total_interactions": self.total_interactions,
            "successful_injections": self.successful_injections,
            "success_rate": success_rate,
            "injector_stats": {
                "overall_success": self.injector.get_success_rate(),
                "average_latency": self.injector.get_average_latency()
            },
            "detector_stats": {
                "states_tracked": len(self.detector.state_history),
                "current_state": str(self.detector.get_current_state())
            }
        }
    
    def get_current_state(self) -> Optional[BMDState]:
        """Get user's current BMD state"""
        return self.conversation.user_bmd if self.conversation else None
    
    def get_conversation_context(self) -> List[CategoricalState]:
        """Get conversation history"""
        return self.conversation.context if self.conversation else []


# Convenience function for quick setup
def create_bidirectional_interface(user_id: str) -> BidirectionalDemon:
    """
    Create a complete bidirectional interface with default configuration
    
    Args:
        user_id: Identifier for user
        
    Returns:
        Configured BidirectionalDemon ready for conversation
    """
    return BidirectionalDemon(user_id=user_id)

