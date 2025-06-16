use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn};

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Decay functions for temporal evidence
#[derive(Debug, Clone)]
pub enum DecayFunction {
    Exponential { lambda: f64 },
    Power { alpha: f64 },
    Logarithmic { base: f64 },
    Weibull { shape: f64, scale: f64 },
    Custom(fn(f64) -> f64),
}

impl DecayFunction {
    pub fn apply(&self, time_elapsed_secs: f64) -> f64 {
        match self {
            DecayFunction::Exponential { lambda } => (-lambda * time_elapsed_secs).exp(),
            DecayFunction::Power { alpha } => (time_elapsed_secs + 1.0).powf(-alpha),
            DecayFunction::Logarithmic { base } => 1.0 / (base * time_elapsed_secs + 1.0).ln(),
            DecayFunction::Weibull { shape, scale } => {
                (-((time_elapsed_secs / scale).powf(*shape))).exp()
            }
            DecayFunction::Custom(func) => func(time_elapsed_secs),
        }
    }
}

/// Multi-dimensional text assessment
#[derive(Debug, Clone)]
pub struct TextAssessment {
    pub semantic_coherence: f64,
    pub contextual_relevance: f64,
    pub temporal_validity: f64,
    pub source_credibility: f64,
    pub logical_consistency: f64,
    pub evidence_support: f64,
}

impl TextAssessment {
    pub fn new() -> Self {
        Self {
            semantic_coherence: 0.5,
            contextual_relevance: 0.5,
            temporal_validity: 0.5,
            source_credibility: 0.5,
            logical_consistency: 0.5,
            evidence_support: 0.5,
        }
    }

    pub fn composite_score(&self) -> f64 {
        let weights = [0.2, 0.15, 0.15, 0.2, 0.15, 0.15];
        let scores = [
            self.semantic_coherence,
            self.contextual_relevance,
            self.temporal_validity,
            self.source_credibility,
            self.logical_consistency,
            self.evidence_support,
        ];
        
        scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score.ln() * weight)
            .sum::<f64>()
            .exp()
    }

    pub fn identify_weak_dimensions(&self) -> Vec<String> {
        let mut weak_dimensions = Vec::new();
        let threshold = 0.6;
        
        if self.semantic_coherence < threshold {
            weak_dimensions.push("Semantic Coherence".to_string());
        }
        if self.contextual_relevance < threshold {
            weak_dimensions.push("Contextual Relevance".to_string());
        }
        if self.temporal_validity < threshold {
            weak_dimensions.push("Temporal Validity".to_string());
        }
        if self.source_credibility < threshold {
            weak_dimensions.push("Source Credibility".to_string());
        }
        if self.logical_consistency < threshold {
            weak_dimensions.push("Logical Consistency".to_string());
        }
        if self.evidence_support < threshold {
            weak_dimensions.push("Evidence Support".to_string());
        }
        
        weak_dimensions
    }
}

/// Evidence with temporal characteristics
#[derive(Debug, Clone)]
pub struct TemporalEvidence {
    pub id: String,
    pub content: String,
    pub credibility: f64,
    pub created_at: Instant,
    pub decay_function: DecayFunction,
    pub source: String,
    pub supporting_data: HashMap<String, String>,
}

impl TemporalEvidence {
    pub fn new(id: String, content: String, credibility: f64) -> Self {
        Self {
            id,
            content,
            credibility,
            created_at: Instant::now(),
            decay_function: DecayFunction::Exponential { lambda: 0.001 }, // Default 
            source: "unknown".to_string(),
            supporting_data: HashMap::new(),
        }
    }

    pub fn with_decay_function(mut self, decay_function: DecayFunction) -> Self {
        self.decay_function = decay_function;
        self
    }

    pub fn with_source(mut self, source: String) -> Self {
        self.source = source;
        self
    }

    pub fn current_strength(&self) -> f64 {
        let time_elapsed = self.created_at.elapsed().as_secs_f64();
        let decay_factor = self.decay_function.apply(time_elapsed);
        self.credibility * decay_factor
    }

    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Belief node in the Bayesian network
#[derive(Debug, Clone)]
pub struct BeliefNode {
    pub id: String,
    pub belief_strength: f64,
    pub uncertainty: f64,
    pub supporting_evidence: Vec<String>,
    pub contradicting_evidence: Vec<String>,
    pub last_updated: Instant,
}

impl BeliefNode {
    pub fn new(id: String, initial_strength: f64) -> Self {
        Self {
            id,
            belief_strength: initial_strength,
            uncertainty: 0.5, // Start with high uncertainty
            supporting_evidence: Vec::new(),
            contradicting_evidence: Vec::new(),
            last_updated: Instant::now(),
        }
    }

    pub fn update_with_evidence(&mut self, evidence: &TemporalEvidence, supports: bool) {
        let evidence_strength = evidence.current_strength();
        
        if supports {
            self.supporting_evidence.push(evidence.id.clone());
            // Bayesian update for supporting evidence
            self.belief_strength = (self.belief_strength + evidence_strength) / 2.0;
            self.uncertainty *= 0.9; // Reduce uncertainty
        } else {
            self.contradicting_evidence.push(evidence.id.clone());
            // Bayesian update for contradicting evidence
            self.belief_strength = (self.belief_strength - evidence_strength * 0.5).max(0.0);
            self.uncertainty *= 1.1; // Increase uncertainty due to contradiction
        }
        
        self.last_updated = Instant::now();
    }
}

/// ATP metabolism for belief updates
#[derive(Debug)]
pub struct ATPMetabolism {
    pub current_atp: f64,
    pub max_atp: f64,
    pub regeneration_rate: f64,
    pub last_regeneration: Instant,
}

impl ATPMetabolism {
    pub fn new(max_atp: f64, regeneration_rate: f64) -> Self {
        Self {
            current_atp: max_atp,
            max_atp,
            regeneration_rate,
            last_regeneration: Instant::now(),
        }
    }

    pub fn compute_belief_update_cost(&self, affected_nodes: usize, uncertainty_change: f64) -> f64 {
        let base_cost = 10.0;
        let complexity_cost = affected_nodes as f64 * 2.0;
        let uncertainty_cost = uncertainty_change.abs() * 5.0;
        
        base_cost + complexity_cost + uncertainty_cost
    }

    pub fn can_afford_operation(&mut self, cost: f64) -> bool {
        self.regenerate_atp();
        self.current_atp >= cost
    }

    pub fn consume_atp(&mut self, amount: f64) -> f64 {
        let consumed = amount.min(self.current_atp);
        self.current_atp -= consumed;
        consumed
    }

    pub fn regenerate_atp(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_regeneration).as_secs_f64();
        let regenerated = elapsed * self.regeneration_rate;
        
        self.current_atp = (self.current_atp + regenerated).min(self.max_atp);
        self.last_regeneration = now;
    }
}

/// Variational parameters for ELBO optimization
#[derive(Debug, Clone)]
pub struct VariationalParameters {
    pub mean: HashMap<String, f64>,
    pub variance: HashMap<String, f64>,
    pub learning_rate: f64,
}

impl VariationalParameters {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            mean: HashMap::new(),
            variance: HashMap::new(),
            learning_rate,
        }
    }

    pub fn update_with_gradient(&mut self, gradient: &HashMap<String, f64>) {
        for (param, grad) in gradient {
            let current_mean = self.mean.get(param).unwrap_or(&0.0);
            self.mean.insert(param.clone(), current_mean + self.learning_rate * grad);
        }
    }

    pub fn compute_entropy(&self) -> f64 {
        self.variance.values().map(|v| 0.5 * v.ln()).sum()
    }
}

/// Optimization result from belief network update
#[derive(Debug)]
pub struct OptimizationResult {
    pub improvement: f64,
    pub converged: bool,
    pub atp_cost: f64,
    pub new_elbo: f64,
    pub affected_beliefs: Vec<String>,
}

/// The main Mzekezeke Bayesian Engine
pub struct MzekezkeBayesianEngine {
    belief_network: Arc<Mutex<HashMap<String, BeliefNode>>>,
    evidence_store: Arc<Mutex<HashMap<String, TemporalEvidence>>>,
    variational_params: Arc<Mutex<VariationalParameters>>,
    atp_metabolism: Arc<Mutex<ATPMetabolism>>,
    optimization_target: Arc<Mutex<f64>>, // ELBO
    stats: Arc<Mutex<ProcessorStats>>,
}

impl MzekezkeBayesianEngine {
    pub fn new() -> Self {
        Self {
            belief_network: Arc::new(Mutex::new(HashMap::new())),
            evidence_store: Arc::new(Mutex::new(HashMap::new())),
            variational_params: Arc::new(Mutex::new(VariationalParameters::new(0.01))),
            atp_metabolism: Arc::new(Mutex::new(ATPMetabolism::new(1000.0, 5.0))),
            optimization_target: Arc::new(Mutex::new(0.0)),
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    /// Extract evidence from text content
    fn extract_evidence_from_text(&self, text: &str) -> Vec<TemporalEvidence> {
        let mut evidence = Vec::new();
        
        // Simple evidence extraction (would be more sophisticated in practice)
        let sentences: Vec<&str> = text.split('.').collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            if !sentence.trim().is_empty() {
                let evidence_id = format!("evidence_{}", i);
                let credibility = self.assess_sentence_credibility(sentence);
                
                let temporal_evidence = TemporalEvidence::new(
                    evidence_id,
                    sentence.trim().to_string(),
                    credibility,
                );
                
                evidence.push(temporal_evidence);
            }
        }
        
        evidence
    }

    /// Assess credibility of a sentence
    fn assess_sentence_credibility(&self, sentence: &str) -> f64 {
        // Simple heuristic-based credibility assessment
        let mut credibility = 0.5; // Start neutral
        
        // Increase credibility for certain indicators
        if sentence.contains("study") || sentence.contains("research") {
            credibility += 0.2;
        }
        if sentence.contains("data") || sentence.contains("evidence") {
            credibility += 0.1;
        }
        if sentence.contains("proven") || sentence.contains("demonstrated") {
            credibility += 0.15;
        }
        
        // Decrease credibility for uncertainty indicators
        if sentence.contains("might") || sentence.contains("could") {
            credibility -= 0.1;
        }
        if sentence.contains("possibly") || sentence.contains("perhaps") {
            credibility -= 0.05;
        }
        
        credibility.clamp(0.0, 1.0)
    }

    /// Assess text quality across multiple dimensions
    fn assess_text_quality(&self, text: &str) -> TextAssessment {
        let mut assessment = TextAssessment::new();
        
        // Simple heuristic assessments (would be more sophisticated in practice)
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split('.').count();
        
        // Semantic coherence - based on text structure
        assessment.semantic_coherence = if word_count > 50 && sentence_count > 3 {
            0.8
        } else {
            0.4
        };
        
        // Contextual relevance - would need context to assess properly
        assessment.contextual_relevance = 0.6; // Default
        
        // Temporal validity - newer is generally better
        assessment.temporal_validity = 0.9; // Assume recent
        
        // Source credibility - would need source metadata
        assessment.source_credibility = 0.7; // Default assumption
        
        // Logical consistency - simple check for contradictions
        assessment.logical_consistency = if text.contains("however") || text.contains("but") {
            0.6 // Potential contradictions
        } else {
            0.8
        };
        
        // Evidence support - look for supporting indicators
        assessment.evidence_support = if text.contains("because") || text.contains("therefore") {
            0.8
        } else {
            0.5
        };
        
        assessment
    }

    /// Compute Evidence Lower Bound (ELBO)
    fn compute_evidence_lower_bound(&self) -> f64 {
        let evidence_store = self.evidence_store.lock().unwrap();
        let variational_params = self.variational_params.lock().unwrap();
        
        // Expected log joint probability
        let mut expected_log_joint = 0.0;
        for evidence in evidence_store.values() {
            expected_log_joint += evidence.current_strength().ln();
        }
        
        // Entropy term
        let entropy_term = variational_params.compute_entropy();
        
        expected_log_joint + entropy_term
    }

    /// Optimize beliefs using variational inference
    fn optimize_beliefs(&self) -> OptimizationResult {
        let current_elbo = self.compute_evidence_lower_bound();
        
        let mut atp = self.atp_metabolism.lock().unwrap();
        let optimization_cost = 50.0; // Base cost for optimization
        
        if !atp.can_afford_operation(optimization_cost) {
            return OptimizationResult {
                improvement: 0.0,
                converged: false,
                atp_cost: 0.0,
                new_elbo: current_elbo,
                affected_beliefs: Vec::new(),
            };
        }
        
        // Simple gradient computation (would be more sophisticated)
        let mut gradient = HashMap::new();
        gradient.insert("global_belief".to_string(), 0.01);
        
        // Update variational parameters
        let mut variational_params = self.variational_params.lock().unwrap();
        variational_params.update_with_gradient(&gradient);
        drop(variational_params);
        
        let new_elbo = self.compute_evidence_lower_bound();
        let atp_consumed = atp.consume_atp(optimization_cost);
        
        // Update optimization target
        let mut target = self.optimization_target.lock().unwrap();
        *target = new_elbo;
        
        OptimizationResult {
            improvement: new_elbo - current_elbo,
            converged: (new_elbo - current_elbo).abs() < 1e-6,
            atp_cost: atp_consumed,
            new_elbo,
            affected_beliefs: vec!["global_belief".to_string()],
        }
    }

    /// Incorporate evidence and update beliefs
    fn incorporate_evidence(&self, evidence: Vec<TemporalEvidence>, assessment: TextAssessment) -> OptimizationResult {
        let mut evidence_store = self.evidence_store.lock().unwrap();
        let mut belief_network = self.belief_network.lock().unwrap();
        
        let mut affected_beliefs = Vec::new();
        
        // Store evidence
        for ev in evidence {
            let belief_id = format!("belief_{}", ev.id);
            
            // Create or update belief node
            let belief_node = belief_network.entry(belief_id.clone()).or_insert_with(|| {
                BeliefNode::new(belief_id.clone(), 0.5)
            });
            
            // Update belief with evidence (assume supporting for now)
            belief_node.update_with_evidence(&ev, true);
            affected_beliefs.push(belief_id);
            
            evidence_store.insert(ev.id.clone(), ev);
        }
        
        drop(evidence_store);
        drop(belief_network);
        
        // Optimize the belief network
        self.optimize_beliefs()
    }
}

#[async_trait]
impl StreamProcessor for MzekezkeBayesianEngine {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        let belief_network = self.belief_network.clone();
        let evidence_store = self.evidence_store.clone();
        let variational_params = self.variational_params.clone();
        let atp_metabolism = self.atp_metabolism.clone();
        let optimization_target = self.optimization_target.clone();
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                let start_time = Instant::now();
                
                debug!("Mzekezeke processing: {}", data.content);
                
                // Extract evidence from text
                let mzekezeke = MzekezkeBayesianEngine {
                    belief_network: belief_network.clone(),
                    evidence_store: evidence_store.clone(),
                    variational_params: variational_params.clone(),
                    atp_metabolism: atp_metabolism.clone(),
                    optimization_target: optimization_target.clone(),
                    stats: stats.clone(),
                };
                
                let evidence = mzekezeke.extract_evidence_from_text(&data.content);
                let assessment = mzekezeke.assess_text_quality(&data.content);
                
                // Incorporate evidence and update beliefs
                let optimization_result = mzekezeke.incorporate_evidence(evidence, assessment.clone());
                
                // Update metadata with Mzekezeke results
                data = data.with_metadata("mzekezeke_elbo", &optimization_result.new_elbo.to_string());
                data = data.with_metadata("mzekezeke_improvement", &optimization_result.improvement.to_string());
                data = data.with_metadata("mzekezeke_atp_cost", &optimization_result.atp_cost.to_string());
                data = data.with_metadata("text_quality_composite", &assessment.composite_score().to_string());
                data = data.with_metadata("semantic_coherence", &assessment.semantic_coherence.to_string());
                data = data.with_metadata("source_credibility", &assessment.source_credibility.to_string());
                
                // Adjust confidence based on belief strength and quality
                let quality_factor = assessment.composite_score();
                let elbo_factor = (optimization_result.new_elbo / 10.0).min(1.0).max(0.0);
                let new_confidence = (data.confidence + quality_factor + elbo_factor) / 3.0;
                data = data.with_confidence(new_confidence);
                
                // Update processing statistics
                {
                    let mut stats_guard = stats.lock().unwrap();
                    stats_guard.items_processed += 1;
                    let processing_time = start_time.elapsed().as_millis() as f64;
                    stats_guard.average_processing_time_ms = 
                        (stats_guard.average_processing_time_ms + processing_time) / 2.0;
                    stats_guard.last_processed = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                }
                
                info!("Mzekezeke processed with ELBO: {:.4}, Quality: {:.4}", 
                      optimization_result.new_elbo, quality_factor);
                
                if tx.send(data).await.is_err() {
                    warn!("Mzekezeke: Failed to send processed data");
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "MzekezkeBayesianEngine"
    }
    
    fn can_handle(&self, _data: &StreamData) -> bool {
        true // Can handle any text data
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for MzekezkeBayesianEngine {
    fn default() -> Self {
        Self::new()
    }
} 