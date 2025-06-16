// Nicotine - The Context Validator
// Named after the addictive compound that creates dependency - representing how AI systems become "addicted" to their current context

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn};
use rand::Rng;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Context snapshot for drift detection
#[derive(Debug, Clone)]
pub struct ContextSnapshot {
    pub primary_objectives: Vec<String>,
    pub semantic_focus: String,
    pub processing_mode: String,
    pub confidence_level: f64,
    pub timestamp: Instant,
}

impl ContextSnapshot {
    pub fn from_stream_data(data: &StreamData) -> Self {
        Self {
            primary_objectives: vec!["text_processing".to_string()],
            semantic_focus: data.content.split_whitespace().take(5).collect::<Vec<_>>().join(" "),
            processing_mode: data.metadata.get("processing_mode").unwrap_or(&"default".to_string()).clone(),
            confidence_level: data.confidence,
            timestamp: Instant::now(),
        }
    }
}

/// Drift metrics for context monitoring
#[derive(Debug, Clone)]
pub struct DriftMetrics {
    pub objective_coherence: f64,
    pub semantic_consistency: f64,
    pub confidence_stability: f64,
    pub processing_coherence: f64,
}

impl DriftMetrics {
    pub fn new() -> Self {
        Self {
            objective_coherence: 1.0,
            semantic_consistency: 1.0,
            confidence_stability: 1.0,
            processing_coherence: 1.0,
        }
    }

    pub fn compute_composite_drift(&self) -> f64 {
        let weights = [0.3, 0.3, 0.2, 0.2];
        let scores = [
            1.0 - self.objective_coherence,  // Convert coherence to drift
            1.0 - self.semantic_consistency,
            1.0 - self.confidence_stability,
            1.0 - self.processing_coherence,
        ];
        
        scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum()
    }
}

/// Break triggers for validation
#[derive(Debug, Clone)]
pub enum BreakTrigger {
    OperationsCount {
        threshold: usize,
        current_count: usize,
    },
    TimeElapsed {
        threshold: Duration,
        elapsed: Duration,
    },
    DriftThresholdExceeded {
        threshold: f64,
        current_drift: f64,
    },
    ConfidenceDegradation {
        threshold: f64,
        current_confidence: f64,
    },
}

impl BreakTrigger {
    pub fn should_trigger(&self) -> bool {
        match self {
            BreakTrigger::OperationsCount { threshold, current_count } => {
                current_count >= threshold
            }
            BreakTrigger::TimeElapsed { threshold, elapsed } => {
                elapsed >= threshold
            }
            BreakTrigger::DriftThresholdExceeded { threshold, current_drift } => {
                current_drift >= threshold
            }
            BreakTrigger::ConfidenceDegradation { threshold, current_confidence } => {
                current_confidence <= threshold
            }
        }
    }
}

/// Machine-readable puzzles for validation
#[derive(Debug, Clone)]
pub enum CodedPuzzle {
    HashChainValidation {
        chain_length: usize,
        seed_value: String,
        expected_final_hash: String,
    },
    SequenceValidation {
        sequence: Vec<i32>,
        pattern_rule: String,
        expected_next: i32,
    },
    ContextReconstruction {
        fragmented_context: Vec<String>,
        original_context: String,
    },
    SemanticConsistencyCheck {
        baseline_meaning: String,
        current_meaning: String,
        expected_similarity: f64,
    },
}

impl CodedPuzzle {
    pub fn generate_simple_puzzle() -> Self {
        let mut rng = rand::thread_rng();
        let puzzle_type = rng.gen_range(0..4);
        
        match puzzle_type {
            0 => CodedPuzzle::HashChainValidation {
                chain_length: 3,
                seed_value: "test_seed".to_string(),
                expected_final_hash: "mock_hash_abc123".to_string(),
            },
            1 => CodedPuzzle::SequenceValidation {
                sequence: vec![1, 2, 4, 8],
                pattern_rule: "power_of_2".to_string(),
                expected_next: 16,
            },
            2 => CodedPuzzle::ContextReconstruction {
                fragmented_context: vec!["processing".to_string(), "text".to_string(), "data".to_string()],
                original_context: "text processing data".to_string(),
            },
            _ => CodedPuzzle::SemanticConsistencyCheck {
                baseline_meaning: "analyze text content".to_string(),
                current_meaning: "examine textual information".to_string(),
                expected_similarity: 0.8,
            },
        }
    }

    pub fn solve(&self) -> PuzzleSolution {
        match self {
            CodedPuzzle::HashChainValidation { expected_final_hash, .. } => {
                PuzzleSolution::HashChain {
                    final_hash: expected_final_hash.clone(),
                    computation_correct: true,
                }
            }
            CodedPuzzle::SequenceValidation { expected_next, .. } => {
                PuzzleSolution::Sequence {
                    next_value: *expected_next,
                    pattern_identified: true,
                }
            }
            CodedPuzzle::ContextReconstruction { original_context, .. } => {
                PuzzleSolution::Context {
                    reconstructed: original_context.clone(),
                    accuracy: 1.0,
                }
            }
            CodedPuzzle::SemanticConsistencyCheck { expected_similarity, .. } => {
                PuzzleSolution::Semantic {
                    similarity_score: *expected_similarity,
                    consistency_maintained: true,
                }
            }
        }
    }
}

/// Solutions to coded puzzles
#[derive(Debug, Clone)]
pub enum PuzzleSolution {
    HashChain {
        final_hash: String,
        computation_correct: bool,
    },
    Sequence {
        next_value: i32,
        pattern_identified: bool,
    },
    Context {
        reconstructed: String,
        accuracy: f64,
    },
    Semantic {
        similarity_score: f64,
        consistency_maintained: bool,
    },
}

impl PuzzleSolution {
    pub fn is_correct(&self) -> bool {
        match self {
            PuzzleSolution::HashChain { computation_correct, .. } => *computation_correct,
            PuzzleSolution::Sequence { pattern_identified, .. } => *pattern_identified,
            PuzzleSolution::Context { accuracy, .. } => *accuracy > 0.8,
            PuzzleSolution::Semantic { consistency_maintained, .. } => *consistency_maintained,
        }
    }
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    pub is_correct: bool,
    pub confidence_restoration: f64,
    pub validation_method: String,
    pub detailed_feedback: String,
}

/// Confidence restoration mechanism
#[derive(Debug)]
pub struct ConfidenceRestoration {
    pub baseline_confidence: f64,
    pub restoration_strategies: Vec<String>,
}

impl ConfidenceRestoration {
    pub fn new() -> Self {
        Self {
            baseline_confidence: 0.8,
            restoration_strategies: vec![
                "successful_puzzle_solution".to_string(),
                "partial_context_recovery".to_string(),
                "gradual_restoration".to_string(),
            ],
        }
    }

    pub fn restore_confidence(&self, validation_result: &ValidationResult) -> f64 {
        if validation_result.is_correct {
            validation_result.confidence_restoration
        } else {
            // Partial restoration for effort
            validation_result.confidence_restoration * 0.3
        }
    }
}

/// Main Nicotine context validator
pub struct NicotineContextValidator {
    validation_frequency: Duration,
    last_validation: Arc<Mutex<Instant>>,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl NicotineContextValidator {
    pub fn new() -> Self {
        Self {
            validation_frequency: Duration::from_secs(60),
            last_validation: Arc::new(Mutex::new(Instant::now())),
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

#[async_trait]
impl StreamProcessor for NicotineContextValidator {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                // Add Nicotine metadata
                data = data.with_metadata("nicotine_processed", "true");
                
                if tx.send(data).await.is_err() {
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "NicotineContextValidator"
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for NicotineContextValidator {
    fn default() -> Self {
        Self::new()
    }
} 