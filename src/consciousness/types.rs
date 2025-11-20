/// Core types for consciousness computing
/// 
/// These types represent the biological oscillatory substrate that underlies consciousness

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 3D vector for spatial coordinates
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}

/// H⁺ electric field state (reality substrate at ~40 THz)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrogenFieldState {
    /// Field oscillation frequency (typically ~40 THz)
    pub frequency: f64,
    
    /// Field coherence (0.0-1.0)
    pub coherence: f64,
    
    /// Field variance (σ²)
    pub variance: f64,
    
    /// Spatial extent of field
    pub spatial_extent: Vec3,
    
    /// Field strength at different spatial points
    pub field_map: HashMap<String, f64>,
}

impl HydrogenFieldState {
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency,
            coherence: 0.5,
            variance: 1.0,
            spatial_extent: Vec3::default(),
            field_map: HashMap::new(),
        }
    }
    
    /// Calculate emotional valence from field aggregate state
    /// Positive coherence with low variance = positive emotion
    /// Low coherence with high variance = negative emotion
    pub fn aggregate_emotional_state(&self) -> f64 {
        let coherence_factor = self.coherence;
        let variance_factor = 1.0 / (1.0 + self.variance);
        
        // Map to [-1, 1] range
        2.0 * (coherence_factor * variance_factor) - 1.0
    }
}

impl Default for HydrogenFieldState {
    fn default() -> Self {
        Self::new(40e12) // 40 THz default
    }
}

/// O₂ categorical state (computational timekeeping with 25,110 states)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxygenCategoricalState {
    /// Current quantum state (1-25,110)
    /// Derived from: spin × vibrational × rotational × electronic × nuclear
    pub quantum_state: u32,
    
    /// Categorical completion rate (Hz, typically ~10^13 Hz base)
    pub completion_rate: f64,
    
    /// Electron affinity for PCET events
    pub electron_affinity: f64,
    
    /// Decomposition of quantum state
    pub state_components: QuantumStateComponents,
}

impl OxygenCategoricalState {
    pub fn new(quantum_state: u32) -> Self {
        assert!(quantum_state > 0 && quantum_state <= 25110, 
                "Quantum state must be between 1 and 25,110");
        
        Self {
            quantum_state,
            completion_rate: 1e13, // 10^13 Hz base rate
            electron_affinity: 1.46, // eV
            state_components: QuantumStateComponents::from_state(quantum_state),
        }
    }
}

impl Default for OxygenCategoricalState {
    fn default() -> Self {
        Self::new(12605) // Middle state
    }
}

/// Decomposition of O₂ quantum state into components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateComponents {
    pub spin_state: u8,        // 1-3 (singlet, triplet)
    pub vibrational_state: u8, // 1-30
    pub rotational_state: u16, // 1-150
    pub electronic_state: u8,  // 1-10
    pub nuclear_state: u8,     // 1-5
}

impl QuantumStateComponents {
    /// Decode quantum state into components
    pub fn from_state(state: u32) -> Self {
        // Simplified decomposition (actual would be more complex)
        let spin = ((state - 1) % 3 + 1) as u8;
        let vibr = (((state - 1) / 3) % 30 + 1) as u8;
        let rot = (((state - 1) / 90) % 150 + 1) as u16;
        let elec = (((state - 1) / 13500) % 10 + 1) as u8;
        let nuc = (((state - 1) / 135000) % 5 + 1) as u8;
        
        Self {
            spin_state: spin,
            vibrational_state: vibr,
            rotational_state: rot,
            electronic_state: elec,
            nuclear_state: nuc,
        }
    }
}

/// Oscillatory hole (computation unit via variance minimization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryHole {
    /// Time of hole formation
    pub formation_time: f64,
    
    /// Duration of hole stability
    pub stability_duration: f64,
    
    /// Variance within hole (should be minimal)
    pub variance: f64,
    
    /// Proton-coupled electron transfer events
    pub pcet_events: Vec<ElectronTransfer>,
    
    /// Spatial location
    pub location: Vec3,
    
    /// Hole type (positive charge seeking electrons)
    pub hole_type: HoleType,
}

impl OscillatoryHole {
    pub fn new(formation_time: f64, location: Vec3) -> Self {
        Self {
            formation_time,
            stability_duration: 0.0,
            variance: 1.0,
            pcet_events: Vec::new(),
            location,
            hole_type: HoleType::PositiveCharge,
        }
    }
    
    /// Calculate stability metric
    pub fn stability(&self) -> f64 {
        // Stable = long duration + low variance + many PCET events
        let duration_factor = self.stability_duration.min(1.0);
        let variance_factor = 1.0 / (1.0 + self.variance);
        let pcet_factor = (self.pcet_events.len() as f64).min(10.0) / 10.0;
        
        (duration_factor + variance_factor + pcet_factor) / 3.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HoleType {
    PositiveCharge,  // H⁺ induced positive hole
    NegativeCharge,  // Electron excess
}

/// Electron transfer event (PCET - Proton-Coupled Electron Transfer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronTransfer {
    /// Time of transfer
    pub time: f64,
    
    /// Energy of transfer
    pub energy: f64,
    
    /// Donor location
    pub donor: Vec3,
    
    /// Acceptor location
    pub acceptor: Vec3,
    
    /// Coupled proton?
    pub proton_coupled: bool,
}

/// Phase-locking state (emotions and cognitive coupling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseLockingState {
    /// Phase-locking matrix between brain regions
    pub coherence_matrix: HashMap<String, HashMap<String, f64>>,
    
    /// Coupling strength
    pub coupling_strength: f64,
    
    /// Frequency band coherences
    pub frequency_bands: HashMap<FrequencyBand, f64>,
}

impl PhaseLockingState {
    pub fn new() -> Self {
        Self {
            coherence_matrix: HashMap::new(),
            coupling_strength: 0.5,
            frequency_bands: HashMap::new(),
        }
    }
    
    /// Calculate global coherence across all regions
    pub fn global_coherence(&self) -> f64 {
        if self.coherence_matrix.is_empty() {
            return 0.5;
        }
        
        let total: f64 = self.coherence_matrix.values()
            .flat_map(|inner| inner.values())
            .sum();
        let count = self.coherence_matrix.values()
            .map(|inner| inner.len())
            .sum::<usize>() as f64;
        
        if count > 0.0 {
            total / count
        } else {
            0.5
        }
    }
}

impl Default for PhaseLockingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Brain oscillation frequency bands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrequencyBand {
    Delta,      // 0.5-4 Hz
    Theta,      // 4-8 Hz
    Alpha,      // 8-13 Hz
    Beta,       // 13-30 Hz
    Gamma,      // 30-100 Hz
    ThetaGamma, // Cross-frequency coupling
}

impl FrequencyBand {
    pub fn frequency_range(&self) -> (f64, f64) {
        match self {
            FrequencyBand::Delta => (0.5, 4.0),
            FrequencyBand::Theta => (4.0, 8.0),
            FrequencyBand::Alpha => (8.0, 13.0),
            FrequencyBand::Beta => (13.0, 30.0),
            FrequencyBand::Gamma => (30.0, 100.0),
            FrequencyBand::ThetaGamma => (4.0, 100.0), // Coupling
        }
    }
}

/// Temporal scale hierarchy (T1-T5)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalScale {
    T1Cellular,       // 10^-1 to 10^1 hours
    T2Population,     // 10^1 to 10^2 hours (days)
    T3Tissue,         // 10^2 to 10^3 hours (weeks)
    T4Functional,     // 10^3 to 10^4 hours (months)
    T5Organismal,     // 10^4 to 10^6 hours (years)
}

impl TemporalScale {
    pub fn time_range_hours(&self) -> (f64, f64) {
        match self {
            TemporalScale::T1Cellular => (0.1, 10.0),
            TemporalScale::T2Population => (10.0, 100.0),
            TemporalScale::T3Tissue => (100.0, 1000.0),
            TemporalScale::T4Functional => (1000.0, 10000.0),
            TemporalScale::T5Organismal => (10000.0, 1000000.0),
        }
    }
    
    pub fn frequency_range_hz(&self) -> (f64, f64) {
        let (min_hours, max_hours) = self.time_range_hours();
        // Convert hours to Hz (1/seconds)
        (1.0 / (max_hours * 3600.0), 1.0 / (min_hours * 3600.0))
    }
}

/// S-entropy coordinates (tri-dimensional alignment space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyCoordinates {
    /// S_knowledge: Information deficit
    pub s_knowledge: f64,
    
    /// S_time: Temporal distance to solution
    pub s_time: f64,
    
    /// S_entropy: Entropy navigation distance
    pub s_entropy: f64,
}

impl SEntropyCoordinates {
    pub fn new(s_knowledge: f64, s_time: f64, s_entropy: f64) -> Self {
        Self { s_knowledge, s_time, s_entropy }
    }
    
    /// Calculate solution quality
    pub fn quality(&self) -> f64 {
        let epsilon = 0.001;
        1.0 / (self.s_knowledge + self.s_time + self.s_entropy + epsilon)
    }
    
    /// Calculate distance in S-space
    pub fn distance_to(&self, other: &SEntropyCoordinates) -> f64 {
        let dk = (self.s_knowledge - other.s_knowledge).powi(2);
        let dt = (self.s_time - other.s_time).powi(2);
        let de = (self.s_entropy - other.s_entropy).powi(2);
        (dk + dt + de).sqrt()
    }
}

impl Default for SEntropyCoordinates {
    fn default() -> Self {
        Self {
            s_knowledge: 1.0,
            s_time: 1.0,
            s_entropy: 1.0,
        }
    }
}

/// Molecular properties for oscillatory agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularProperties {
    /// SMILES chemical structure
    pub smiles: String,
    
    /// Oscillation frequency
    pub oscillation_frequency: f64,
    
    /// Coupling constant to biological oscillators
    pub coupling_constant: f64,
    
    /// Diffusion coefficient (cytoplasmic mobility)
    pub diffusion_coefficient: f64,
    
    /// Electromagnetic moment (in Bohr magnetons)
    pub electromagnetic_moment: f64,
    
    /// O₂ aggregation coefficient
    pub o2_aggregation: f64,
}

impl MolecularProperties {
    pub fn new(smiles: String) -> Self {
        Self {
            smiles,
            oscillation_frequency: 0.0,
            coupling_constant: 0.0,
            diffusion_coefficient: 0.0,
            electromagnetic_moment: 0.0,
            o2_aggregation: 0.0,
        }
    }
}

/// Propagation mode for molecular agents
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PropagationMode {
    CytoplasmicDiffusion,
    MembraneLateralDiffusion,
    ActiveTransport,
    ReceptorMediated,
}

impl PropagationMode {
    pub fn diffusion_rate(&self) -> f64 {
        match self {
            PropagationMode::CytoplasmicDiffusion => 1e-10, // m²/s
            PropagationMode::MembraneLateralDiffusion => 1e-12,
            PropagationMode::ActiveTransport => 1e-9,
            PropagationMode::ReceptorMediated => 1e-11,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_oxygen_quantum_states() {
        let state = OxygenCategoricalState::new(1);
        assert_eq!(state.quantum_state, 1);
        
        let state = OxygenCategoricalState::new(25110);
        assert_eq!(state.quantum_state, 25110);
    }
    
    #[test]
    #[should_panic]
    fn test_invalid_oxygen_state() {
        OxygenCategoricalState::new(25111);
    }
    
    #[test]
    fn test_s_entropy_quality() {
        let coords = SEntropyCoordinates::new(0.1, 0.1, 0.1);
        assert!(coords.quality() > 3.0); // Should be high quality
        
        let coords = SEntropyCoordinates::new(10.0, 10.0, 10.0);
        assert!(coords.quality() < 0.1); // Should be low quality
    }
    
    #[test]
    fn test_frequency_bands() {
        let theta = FrequencyBand::Theta;
        assert_eq!(theta.frequency_range(), (4.0, 8.0));
    }
}

