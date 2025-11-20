/// Consciousness Computing Primitives
/// 
/// This module implements the biological oscillatory computing framework
/// that bridges consciousness theory with computational reality.
/// 
/// Core mappings:
/// - H⁺ field entropy ≡ S_entropy (entropy navigation)
/// - Categorical state space ≡ S_knowledge (information deficit)
/// - Temporal scale hierarchy ≡ S_time (temporal distance)
/// - Oscillatory holes ≡ Information Catalysts (iCat)
/// - Categorical completion ≡ BMD frame selection
/// - Phase-locking ≡ Semantic coherence

pub mod types;
pub mod h_plus_field;
pub mod oxygen_categorical;
pub mod oscillatory_holes;
pub mod phase_locking;
pub mod s_entropy;
pub mod molecular_design;
pub mod temporal_scales;
pub mod thermodynamic_compiler;

pub use types::*;
pub use h_plus_field::*;
pub use oxygen_categorical::*;
pub use oscillatory_holes::*;
pub use phase_locking::*;
pub use s_entropy::*;
pub use molecular_design::*;
pub use temporal_scales::*;
pub use thermodynamic_compiler::*;

/// The fundamental consciousness substrate
#[derive(Debug, Clone)]
pub struct ConsciousnessState {
    /// H⁺ electric field state (reality substrate)
    pub h_plus_field: HydrogenFieldState,
    
    /// O₂ categorical clock state (computational timekeeping)
    pub oxygen_clock: OxygenCategoricalState,
    
    /// Active oscillatory holes (computation units)
    pub oscillatory_holes: Vec<OscillatoryHole>,
    
    /// Phase-locking matrix (emotional/cognitive state)
    pub phase_locks: PhaseLockingState,
    
    /// Current temporal scale
    pub temporal_scale: TemporalScale,
    
    /// S-entropy coordinates
    pub s_coordinates: SEntropyCoordinates,
}

impl ConsciousnessState {
    /// Create a new consciousness state
    pub fn new() -> Self {
        Self {
            h_plus_field: HydrogenFieldState::default(),
            oxygen_clock: OxygenCategoricalState::default(),
            oscillatory_holes: Vec::new(),
            phase_locks: PhaseLockingState::default(),
            temporal_scale: TemporalScale::T1Cellular,
            s_coordinates: SEntropyCoordinates::default(),
        }
    }
    
    /// Measure the overall consciousness coherence
    pub fn coherence(&self) -> f64 {
        let h_coherence = self.h_plus_field.coherence;
        let phase_coherence = self.phase_locks.global_coherence();
        let hole_stability = self.oscillatory_holes.iter()
            .map(|h| h.stability())
            .sum::<f64>() / self.oscillatory_holes.len().max(1) as f64;
        
        (h_coherence + phase_coherence + hole_stability) / 3.0
    }
    
    /// Calculate the emotional valence from H⁺ field aggregate state
    pub fn emotional_valence(&self) -> f64 {
        self.h_plus_field.aggregate_emotional_state()
    }
    
    /// Get current thought formation rate (holes per second)
    pub fn thought_rate(&self) -> f64 {
        self.oxygen_clock.completion_rate
    }
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_consciousness_state_creation() {
        let state = ConsciousnessState::new();
        assert!(state.coherence() >= 0.0 && state.coherence() <= 1.0);
    }
    
    #[test]
    fn test_emotional_valence_range() {
        let state = ConsciousnessState::new();
        let valence = state.emotional_valence();
        assert!(valence >= -1.0 && valence <= 1.0);
    }
}

