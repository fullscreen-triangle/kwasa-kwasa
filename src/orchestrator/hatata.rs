// Hatata - The Decision System
// Named after the Zimbabwean Shona word meaning "to think deeply about choices" - reflecting careful decision-making under uncertainty.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn};
use rand::Rng;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Utility function models for different decision-making scenarios
#[derive(Debug, Clone)]
pub enum UtilityModel {
    Linear {
        text_quality_weight: f64,
        atp_cost_weight: f64,
        time_weight: f64,
    },
    Quadratic {
        risk_aversion: f64,
        diminishing_returns: f64,
    },
    Exponential {
        risk_preference: f64,
        urgency_scaling: f64,
    },
    Logarithmic {
        satisfaction_base: f64,
        cost_sensitivity: f64,
    },
}

impl UtilityModel {
    pub fn compute_utility(
        &self,
        text_quality: f64,
        atp_consumed: f64,
        time_taken: f64,
        confidence_achieved: f64,
    ) -> f64 {
        match self {
            UtilityModel::Linear { text_quality_weight, atp_cost_weight, time_weight } => {
                text_quality_weight * text_quality 
                - atp_cost_weight * atp_consumed 
                - time_weight * time_taken
            }
            UtilityModel::Quadratic { risk_aversion, diminishing_returns } => {
                let quality_utility = text_quality - diminishing_returns * text_quality.powi(2);
                let risk_penalty = risk_aversion * (atp_consumed.powi(2) + time_taken.powi(2));
                quality_utility - risk_penalty
            }
            UtilityModel::Exponential { risk_preference, urgency_scaling } => {
                let quality_exp = (-risk_preference * (1.0 - text_quality)).exp();
                let urgency_factor = (-urgency_scaling * time_taken).exp();
                quality_exp * urgency_factor - atp_consumed * 0.1
            }
            UtilityModel::Logarithmic { satisfaction_base, cost_sensitivity } => {
                satisfaction_base * (text_quality + 0.1).ln() 
                - cost_sensitivity * (atp_consumed + time_taken)
            }
        }
    }
}

/// Text processing actions in the MDP
#[derive(Debug, Clone)]
pub enum TextProcessingAction {
    Transform {
        operation: String,
        intensity: f64,
        expected_atp_cost: f64,
    },
    Analyze {
        analysis_type: String,
        depth: f64,
        expected_atp_cost: f64,
    },
    ValidateContext {
        validation_rigor: f64,
        expected_atp_cost: f64,
    },
    SeekEvidence {
        evidence_type: String,
        search_breadth: f64,
        expected_atp_cost: f64,
    },
    Wait {
        duration: Duration,
        atp_regeneration: f64,
    },
    Terminate {
        confidence_threshold_met: bool,
    },
}

impl TextProcessingAction {
    pub fn expected_utility(&self, current_confidence: f64, utility_model: &UtilityModel) -> f64 {
        match self {
            TextProcessingAction::Transform { intensity, expected_atp_cost, .. } => {
                let expected_improvement = intensity * 0.2; // Heuristic
                utility_model.compute_utility(expected_improvement, *expected_atp_cost, 1.0, current_confidence)
            }
            TextProcessingAction::Analyze { depth, expected_atp_cost, .. } => {
                let expected_improvement = depth * 0.15;
                utility_model.compute_utility(expected_improvement, *expected_atp_cost, 0.8, current_confidence)
            }
            TextProcessingAction::ValidateContext { validation_rigor, expected_atp_cost } => {
                let expected_improvement = validation_rigor * 0.1;
                utility_model.compute_utility(expected_improvement, *expected_atp_cost, 0.5, current_confidence)
            }
            TextProcessingAction::SeekEvidence { search_breadth, expected_atp_cost, .. } => {
                let expected_improvement = search_breadth * 0.3;
                utility_model.compute_utility(expected_improvement, *expected_atp_cost, 1.5, current_confidence)
            }
            TextProcessingAction::Wait { atp_regeneration, .. } => {
                // Waiting has no immediate quality improvement but regenerates ATP
                utility_model.compute_utility(0.0, -*atp_regeneration, 1.0, current_confidence)
            }
            TextProcessingAction::Terminate { confidence_threshold_met } => {
                if *confidence_threshold_met {
                    utility_model.compute_utility(0.1, 0.0, 0.0, current_confidence)
                } else {
                    -10.0 // Penalty for premature termination
                }
            }
        }
    }

    pub fn get_atp_cost(&self) -> f64 {
        match self {
            TextProcessingAction::Transform { expected_atp_cost, .. } => *expected_atp_cost,
            TextProcessingAction::Analyze { expected_atp_cost, .. } => *expected_atp_cost,
            TextProcessingAction::ValidateContext { expected_atp_cost, .. } => *expected_atp_cost,
            TextProcessingAction::SeekEvidence { expected_atp_cost, .. } => *expected_atp_cost,
            TextProcessingAction::Wait { atp_regeneration, .. } => -*atp_regeneration,
            TextProcessingAction::Terminate { .. } => 0.0,
        }
    }
}

/// Text processing state in the MDP
#[derive(Debug, Clone)]
pub struct TextProcessingState {
    pub content_quality: f64,
    pub processing_confidence: f64,
    pub available_atp: f64,
    pub time_remaining: Option<f64>,
    pub context_coherence: f64,
    pub evidence_strength: f64,
}

impl TextProcessingState {
    pub fn new() -> Self {
        Self {
            content_quality: 0.5,
            processing_confidence: 0.5,
            available_atp: 100.0,
            time_remaining: Some(60.0), // 60 seconds
            context_coherence: 0.5,
            evidence_strength: 0.5,
        }
    }

    pub fn from_stream_data(data: &StreamData) -> Self {
        let mut state = Self::new();
        
        state.processing_confidence = data.confidence;
        
        // Extract additional state from metadata
        if let Some(atp) = data.metadata.get("available_atp") {
            if let Ok(atp_val) = atp.parse::<f64>() {
                state.available_atp = atp_val;
            }
        }
        
        if let Some(quality) = data.metadata.get("text_quality_composite") {
            if let Ok(quality_val) = quality.parse::<f64>() {
                state.content_quality = quality_val;
            }
        }
        
        if let Some(coherence) = data.metadata.get("semantic_coherence") {
            if let Ok(coherence_val) = coherence.parse::<f64>() {
                state.context_coherence = coherence_val;
            }
        }
        
        state
    }

    pub fn can_afford_action(&self, action: &TextProcessingAction) -> bool {
        self.available_atp >= action.get_atp_cost()
    }

    pub fn apply_action(&mut self, action: &TextProcessingAction) -> f64 {
        let atp_cost = action.get_atp_cost();
        self.available_atp -= atp_cost;
        
        match action {
            TextProcessingAction::Transform { intensity, .. } => {
                self.content_quality += intensity * 0.2;
                self.processing_confidence += intensity * 0.1;
            }
            TextProcessingAction::Analyze { depth, .. } => {
                self.evidence_strength += depth * 0.15;
                self.processing_confidence += depth * 0.05;
            }
            TextProcessingAction::ValidateContext { validation_rigor, .. } => {
                self.context_coherence += validation_rigor * 0.1;
            }
            TextProcessingAction::SeekEvidence { search_breadth, .. } => {
                self.evidence_strength += search_breadth * 0.3;
            }
            TextProcessingAction::Wait { atp_regeneration, .. } => {
                self.available_atp += atp_regeneration;
            }
            TextProcessingAction::Terminate { .. } => {
                // No state change for termination
            }
        }
        
        // Clamp values
        self.content_quality = self.content_quality.clamp(0.0, 1.0);
        self.processing_confidence = self.processing_confidence.clamp(0.0, 1.0);
        self.context_coherence = self.context_coherence.clamp(0.0, 1.0);
        self.evidence_strength = self.evidence_strength.clamp(0.0, 1.0);
        
        atp_cost
    }
}

/// Decision result from the MDP
#[derive(Debug)]
pub struct DecisionResult {
    pub chosen_action: TextProcessingAction,
    pub expected_utility: f64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Main Hatata decision system
pub struct HatataDecisionSystem {
    utility_model: UtilityModel,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl HatataDecisionSystem {
    pub fn new() -> Self {
        Self {
            utility_model: UtilityModel::Linear {
                text_quality_weight: 0.4,
                atp_cost_weight: 0.3,
                time_weight: 0.3,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn with_utility_model(mut self, model: UtilityModel) -> Self {
        self.utility_model = model;
        self
    }

    /// Generate possible actions for the current state
    fn generate_actions(&self, state: &TextProcessingState) -> Vec<TextProcessingAction> {
        let mut actions = Vec::new();
        
        // Transform actions
        if state.content_quality < 0.8 {
            actions.push(TextProcessingAction::Transform {
                operation: "enhance_quality".to_string(),
                intensity: 0.3,
                expected_atp_cost: 20.0,
            });
            actions.push(TextProcessingAction::Transform {
                operation: "deep_enhance".to_string(),
                intensity: 0.6,
                expected_atp_cost: 40.0,
            });
        }
        
        // Analysis actions
        if state.evidence_strength < 0.7 {
            actions.push(TextProcessingAction::Analyze {
                analysis_type: "evidence_gathering".to_string(),
                depth: 0.5,
                expected_atp_cost: 25.0,
            });
        }
        
        // Validation actions
        if state.context_coherence < 0.6 {
            actions.push(TextProcessingAction::ValidateContext {
                validation_rigor: 0.7,
                expected_atp_cost: 15.0,
            });
        }
        
        // Evidence seeking
        if state.evidence_strength < 0.8 {
            actions.push(TextProcessingAction::SeekEvidence {
                evidence_type: "supporting_evidence".to_string(),
                search_breadth: 0.4,
                expected_atp_cost: 30.0,
            });
        }
        
        // Wait action if low ATP
        if state.available_atp < 50.0 {
            actions.push(TextProcessingAction::Wait {
                duration: Duration::from_secs(5),
                atp_regeneration: 25.0,
            });
        }
        
        // Termination action
        let confidence_threshold = 0.8;
        actions.push(TextProcessingAction::Terminate {
            confidence_threshold_met: state.processing_confidence >= confidence_threshold,
        });
        
        actions
    }

    /// Choose the best action using utility maximization
    fn choose_action(&self, state: &TextProcessingState) -> DecisionResult {
        let actions = self.generate_actions(state);
        
        let mut best_action = None;
        let mut best_utility = f64::NEG_INFINITY;
        let mut best_reasoning = String::new();
        
        for action in actions {
            if !state.can_afford_action(&action) {
                continue;
            }
            
            let utility = action.expected_utility(state.processing_confidence, &self.utility_model);
            
            if utility > best_utility {
                best_utility = utility;
                best_reasoning = format!("Chose {:?} with utility {:.3}", action, utility);
                best_action = Some(action);
            }
        }
        
        let chosen_action = best_action.unwrap_or(TextProcessingAction::Terminate {
            confidence_threshold_met: false,
        });
        
        DecisionResult {
            chosen_action,
            expected_utility: best_utility,
            confidence: state.processing_confidence,
            reasoning: best_reasoning,
        }
    }

    /// Make optimal decision for text processing
    pub fn optimize_processing_decision(&self, state: &TextProcessingState) -> DecisionResult {
        let decision = self.choose_action(state);
        
        decision
    }
}

#[async_trait]
impl StreamProcessor for HatataDecisionSystem {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                // Add Hatata metadata
                data = data.with_metadata("hatata_processed", "true");
                
                if tx.send(data).await.is_err() {
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "HatataDecisionSystem"
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for HatataDecisionSystem {
    fn default() -> Self {
        Self::new()
    }
} 