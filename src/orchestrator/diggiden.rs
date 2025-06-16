// Diggiden - The Adversarial System
// Named after the Shona term meaning "to persistently dig" - constantly probing and excavating weaknesses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn, error};
use rand::Rng;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Types of attacks that Diggiden can perform
#[derive(Debug, Clone)]
pub enum AttackStrategy {
    ContradictionInjection {
        target_claims: Vec<String>,
        contradiction_strength: f64,
        stealth_level: StealthLevel,
    },
    TemporalManipulation {
        time_shift: Duration,
        decay_acceleration: f64,
        target_content: String,
    },
    SemanticSpoofing {
        original_meaning: String,
        spoofed_meaning: String,
        similarity_threshold: f64,
    },
    ContextHijacking {
        legitimate_context: String,
        malicious_context: String,
        transition_smoothness: f64,
    },
    PerturbationAttack {
        perturbation_magnitude: f64,
        target_components: Vec<String>,
    },
    BeliefPoisoning {
        poisoned_content: String,
        credibility_mask: f64,
        infection_vector: String,
    },
}

/// Stealth levels for attacks
#[derive(Debug, Clone)]
pub enum StealthLevel {
    Overt,          // Attacks are obvious for testing
    Subtle,         // Attacks mimic natural variations
    Invisible,      // Attacks are undetectable during operation
    Camouflaged,    // Attacks appear as beneficial operations
    Adaptive,       // Stealth level adapts based on detection risk
}

impl StealthLevel {
    pub fn detection_probability(&self) -> f64 {
        match self {
            StealthLevel::Overt => 1.0,
            StealthLevel::Subtle => 0.3,
            StealthLevel::Invisible => 0.05,
            StealthLevel::Camouflaged => 0.01,
            StealthLevel::Adaptive => 0.1, // Average for adaptive
        }
    }
}

/// Result of an attack attempt
#[derive(Debug, Clone)]
pub struct AttackResult {
    pub attack_type: String,
    pub succeeded: bool,
    pub impact_score: f64,
    pub side_effects: Vec<String>,
    pub detection_risk: f64,
    pub target_affected: String,
}

impl AttackResult {
    pub fn failed(reason: &str) -> Self {
        Self {
            attack_type: "Unknown".to_string(),
            succeeded: false,
            impact_score: 0.0,
            side_effects: vec![reason.to_string()],
            detection_risk: 0.0,
            target_affected: "None".to_string(),
        }
    }
}

/// Vulnerability scoring for different system components
#[derive(Debug, Clone)]
pub struct VulnerabilityScore {
    pub current_score: f64,        // 0.0 = invulnerable, 1.0 = completely vulnerable
    pub historical_max: f64,       // Worst vulnerability ever detected
    pub last_successful_attack: Option<Instant>,
    pub attack_success_rate: f64,  // Percentage of attacks that succeed
    pub vulnerability_trend: f64,  // Getting better (-) or worse (+) over time
}

impl VulnerabilityScore {
    pub fn new() -> Self {
        Self {
            current_score: 0.5, // Start with medium vulnerability
            historical_max: 0.5,
            last_successful_attack: None,
            attack_success_rate: 0.0,
            vulnerability_trend: 0.0,
        }
    }

    pub fn update_from_attack(&mut self, attack_succeeded: bool) {
        if attack_succeeded {
            self.current_score = (self.current_score + 0.1).min(1.0);
            self.last_successful_attack = Some(Instant::now());
            self.attack_success_rate = (self.attack_success_rate * 0.9) + (0.1 * 1.0);
            self.vulnerability_trend += 0.05;
        } else {
            self.current_score = (self.current_score - 0.05).max(0.0);
            self.attack_success_rate = (self.attack_success_rate * 0.9) + (0.1 * 0.0);
            self.vulnerability_trend -= 0.02;
        }
        
        self.historical_max = self.historical_max.max(self.current_score);
    }
}

/// Comprehensive vulnerability matrix
#[derive(Debug)]
pub struct VulnerabilityMatrix {
    pub belief_manipulation: VulnerabilityScore,
    pub context_exploitation: VulnerabilityScore,
    pub temporal_attacks: VulnerabilityScore,
    pub semantic_confusion: VulnerabilityScore,
    pub pipeline_bypass: VulnerabilityScore,
    pub confidence_inflation: VulnerabilityScore,
    pub evidence_poisoning: VulnerabilityScore,
    pub attention_diversion: VulnerabilityScore,
}

impl VulnerabilityMatrix {
    pub fn new() -> Self {
        Self {
            belief_manipulation: VulnerabilityScore::new(),
            context_exploitation: VulnerabilityScore::new(),
            temporal_attacks: VulnerabilityScore::new(),
            semantic_confusion: VulnerabilityScore::new(),
            pipeline_bypass: VulnerabilityScore::new(),
            confidence_inflation: VulnerabilityScore::new(),
            evidence_poisoning: VulnerabilityScore::new(),
            attention_diversion: VulnerabilityScore::new(),
        }
    }

    pub fn get_most_vulnerable_components(&self) -> Vec<(String, f64)> {
        let mut vulnerabilities = vec![
            ("BeliefManipulation".to_string(), self.belief_manipulation.current_score),
            ("ContextExploitation".to_string(), self.context_exploitation.current_score),
            ("TemporalAttacks".to_string(), self.temporal_attacks.current_score),
            ("SemanticConfusion".to_string(), self.semantic_confusion.current_score),
            ("PipelineBypass".to_string(), self.pipeline_bypass.current_score),
            ("ConfidenceInflation".to_string(), self.confidence_inflation.current_score),
            ("EvidencePoisoning".to_string(), self.evidence_poisoning.current_score),
            ("AttentionDiversion".to_string(), self.attention_diversion.current_score),
        ];
        
        vulnerabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vulnerabilities
    }

    pub fn update_vulnerability(&mut self, component: &str, attack_succeeded: bool) {
        match component {
            "BeliefManipulation" => self.belief_manipulation.update_from_attack(attack_succeeded),
            "ContextExploitation" => self.context_exploitation.update_from_attack(attack_succeeded),
            "TemporalAttacks" => self.temporal_attacks.update_from_attack(attack_succeeded),
            "SemanticConfusion" => self.semantic_confusion.update_from_attack(attack_succeeded),
            "PipelineBypass" => self.pipeline_bypass.update_from_attack(attack_succeeded),
            "ConfidenceInflation" => self.confidence_inflation.update_from_attack(attack_succeeded),
            "EvidencePoisoning" => self.evidence_poisoning.update_from_attack(attack_succeeded),
            "AttentionDiversion" => self.attention_diversion.update_from_attack(attack_succeeded),
            _ => warn!("Unknown vulnerability component: {}", component),
        }
    }
}

/// Adaptive attack engine that evolves strategies
#[derive(Debug)]
pub struct AdaptiveAttackEngine {
    strategy_success_rates: HashMap<String, f64>,
    target_weaknesses: HashMap<String, f64>,
    attack_history: Vec<(AttackStrategy, AttackResult)>,
    evolution_generation: u32,
}

impl AdaptiveAttackEngine {
    pub fn new() -> Self {
        Self {
            strategy_success_rates: HashMap::new(),
            target_weaknesses: HashMap::new(),
            attack_history: Vec::new(),
            evolution_generation: 0,
        }
    }

    /// Evolve attack strategies based on success/failure patterns
    pub fn evolve_attacks(&mut self) -> Vec<AttackStrategy> {
        self.evolution_generation += 1;
        let mut new_generation = Vec::new();
        
        // Get successful attacks from history
        let successful_attacks: Vec<_> = self.attack_history
            .iter()
            .filter(|(_, result)| result.succeeded)
            .map(|(strategy, _)| strategy)
            .collect();
        
        if successful_attacks.is_empty() {
            // Generate basic attacks if no successful history
            new_generation.push(AttackStrategy::ContradictionInjection {
                target_claims: vec!["basic_claim".to_string()],
                contradiction_strength: 0.5,
                stealth_level: StealthLevel::Subtle,
            });
            new_generation.push(AttackStrategy::SemanticSpoofing {
                original_meaning: "original".to_string(),
                spoofed_meaning: "spoofed".to_string(),
                similarity_threshold: 0.8,
            });
        } else {
            // Evolve from successful attacks
            for attack in successful_attacks.iter().take(3) {
                let evolved = self.mutate_attack(attack);
                new_generation.push(evolved);
            }
        }
        
        // Add novel attacks based on discovered patterns
        let novel_attacks = self.generate_novel_attacks();
        new_generation.extend(novel_attacks);
        
        debug!("Diggiden evolved {} attacks for generation {}", 
               new_generation.len(), self.evolution_generation);
        
        new_generation
    }

    /// Mutate an existing attack strategy
    fn mutate_attack(&self, attack: &AttackStrategy) -> AttackStrategy {
        let mut rng = rand::thread_rng();
        
        match attack {
            AttackStrategy::ContradictionInjection { target_claims, contradiction_strength, stealth_level } => {
                // Mutate strength and stealth
                let new_strength = (contradiction_strength + rng.gen_range(-0.2..0.2)).clamp(0.0, 1.0);
                let new_stealth = if rng.gen_bool(0.3) {
                    StealthLevel::Invisible
                } else {
                    stealth_level.clone()
                };
                
                AttackStrategy::ContradictionInjection {
                    target_claims: target_claims.clone(),
                    contradiction_strength: new_strength,
                    stealth_level: new_stealth,
                }
            }
            AttackStrategy::SemanticSpoofing { original_meaning, spoofed_meaning, similarity_threshold } => {
                let new_threshold = (similarity_threshold + rng.gen_range(-0.1..0.1)).clamp(0.5, 1.0);
                
                AttackStrategy::SemanticSpoofing {
                    original_meaning: original_meaning.clone(),
                    spoofed_meaning: format!("{}_mutated", spoofed_meaning),
                    similarity_threshold: new_threshold,
                }
            }
            // Add mutations for other attack types...
            _ => attack.clone(), // Default: no mutation
        }
    }

    /// Generate novel attack strategies
    fn generate_novel_attacks(&self) -> Vec<AttackStrategy> {
        let mut novel_attacks = Vec::new();
        let mut rng = rand::thread_rng();
        
        // Generate based on current target weaknesses
        for (target, weakness) in &self.target_weaknesses {
            if *weakness > 0.7 {
                // High vulnerability - generate targeted attack
                novel_attacks.push(AttackStrategy::PerturbationAttack {
                    perturbation_magnitude: weakness * rng.gen_range(0.5..1.0),
                    target_components: vec![target.clone()],
                });
            }
        }
        
        // Generate random experimental attacks
        if rng.gen_bool(0.3) {
            novel_attacks.push(AttackStrategy::ContextHijacking {
                legitimate_context: "legitimate_context".to_string(),
                malicious_context: "experimental_malicious_context".to_string(),
                transition_smoothness: rng.gen_range(0.3..0.9),
            });
        }
        
        novel_attacks
    }

    /// Record attack results for learning
    pub fn record_attack(&mut self, strategy: AttackStrategy, result: AttackResult) {
        let strategy_name = self.strategy_name(&strategy);
        
        // Update success rates
        let current_rate = self.strategy_success_rates.get(&strategy_name).unwrap_or(&0.0);
        let new_rate = if result.succeeded {
            (current_rate * 0.9) + (0.1 * 1.0)
        } else {
            (current_rate * 0.9) + (0.1 * 0.0)
        };
        self.strategy_success_rates.insert(strategy_name, new_rate);
        
        // Update target weaknesses
        if result.succeeded {
            let current_weakness = self.target_weaknesses.get(&result.target_affected).unwrap_or(&0.5);
            self.target_weaknesses.insert(
                result.target_affected.clone(), 
                (current_weakness + 0.1).min(1.0)
            );
        }
        
        // Store in history
        self.attack_history.push((strategy, result));
        
        // Keep history manageable
        if self.attack_history.len() > 100 {
            self.attack_history.remove(0);
        }
    }

    fn strategy_name(&self, strategy: &AttackStrategy) -> String {
        match strategy {
            AttackStrategy::ContradictionInjection { .. } => "ContradictionInjection".to_string(),
            AttackStrategy::TemporalManipulation { .. } => "TemporalManipulation".to_string(),
            AttackStrategy::SemanticSpoofing { .. } => "SemanticSpoofing".to_string(),
            AttackStrategy::ContextHijacking { .. } => "ContextHijacking".to_string(),
            AttackStrategy::PerturbationAttack { .. } => "PerturbationAttack".to_string(),
            AttackStrategy::BeliefPoisoning { .. } => "BeliefPoisoning".to_string(),
        }
    }
}

/// Main Diggiden adversarial system
pub struct DiggidenAdversarialSystem {
    vulnerability_matrix: Arc<Mutex<VulnerabilityMatrix>>,
    adaptive_engine: Arc<Mutex<AdaptiveAttackEngine>>,
    attack_frequency: Duration,
    last_attack: Arc<Mutex<Instant>>,
    stats: Arc<Mutex<ProcessorStats>>,
    enabled: Arc<Mutex<bool>>,
}

impl DiggidenAdversarialSystem {
    pub fn new() -> Self {
        Self {
            vulnerability_matrix: Arc::new(Mutex::new(VulnerabilityMatrix::new())),
            adaptive_engine: Arc::new(Mutex::new(AdaptiveAttackEngine::new())),
            attack_frequency: Duration::from_secs(30), // Attack every 30 seconds
            last_attack: Arc::new(Mutex::new(Instant::now())),
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
            enabled: Arc::new(Mutex::new(true)),
        }
    }

    pub fn with_attack_frequency(mut self, frequency: Duration) -> Self {
        self.attack_frequency = frequency;
        self
    }

    pub fn disable_attacks(&self) {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = false;
        info!("Diggiden attacks disabled");
    }

    pub fn enable_attacks(&self) {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = true;
        info!("Diggiden attacks enabled");
    }

    /// Execute a specific attack strategy
    fn execute_attack(&self, strategy: &AttackStrategy, data: &StreamData) -> AttackResult {
        match strategy {
            AttackStrategy::ContradictionInjection { target_claims, contradiction_strength, stealth_level } => {
                self.inject_contradictions(data, target_claims, *contradiction_strength, stealth_level)
            }
            AttackStrategy::TemporalManipulation { time_shift, decay_acceleration, target_content } => {
                self.manipulate_temporal_evidence(data, *time_shift, *decay_acceleration, target_content)
            }
            AttackStrategy::SemanticSpoofing { original_meaning, spoofed_meaning, similarity_threshold } => {
                self.spoof_semantic_meaning(data, original_meaning, spoofed_meaning, *similarity_threshold)
            }
            AttackStrategy::ContextHijacking { legitimate_context, malicious_context, transition_smoothness } => {
                self.hijack_context(data, legitimate_context, malicious_context, *transition_smoothness)
            }
            AttackStrategy::PerturbationAttack { perturbation_magnitude, target_components } => {
                self.perturb_processing(data, *perturbation_magnitude, target_components)
            }
            AttackStrategy::BeliefPoisoning { poisoned_content, credibility_mask, infection_vector } => {
                self.poison_beliefs(data, poisoned_content, *credibility_mask, infection_vector)
            }
        }
    }

    /// Inject contradictory information
    fn inject_contradictions(&self, data: &StreamData, _target_claims: &[String], 
                           contradiction_strength: f64, stealth_level: &StealthLevel) -> AttackResult {
        let success_probability = 0.3 * contradiction_strength;
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(success_probability);
        
        AttackResult {
            attack_type: "ContradictionInjection".to_string(),
            succeeded,
            impact_score: if succeeded { contradiction_strength } else { 0.0 },
            side_effects: if succeeded { 
                vec!["Belief inconsistency detected".to_string()] 
            } else { 
                vec!["Contradiction rejected".to_string()] 
            },
            detection_risk: stealth_level.detection_probability(),
            target_affected: "BeliefManipulation".to_string(),
        }
    }

    /// Manipulate temporal evidence decay
    fn manipulate_temporal_evidence(&self, _data: &StreamData, _time_shift: Duration, 
                                  decay_acceleration: f64, _target_content: &str) -> AttackResult {
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(0.2 * decay_acceleration);
        
        AttackResult {
            attack_type: "TemporalManipulation".to_string(),
            succeeded,
            impact_score: if succeeded { decay_acceleration } else { 0.0 },
            side_effects: if succeeded {
                vec!["Temporal consistency compromised".to_string()]
            } else {
                vec!["Temporal manipulation blocked".to_string()]
            },
            detection_risk: 0.15, // Temporal attacks are subtle
            target_affected: "TemporalAttacks".to_string(),
        }
    }

    /// Spoof semantic meaning
    fn spoof_semantic_meaning(&self, _data: &StreamData, _original: &str, _spoofed: &str, 
                            similarity_threshold: f64) -> AttackResult {
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(0.4 * similarity_threshold);
        
        AttackResult {
            attack_type: "SemanticSpoofing".to_string(),
            succeeded,
            impact_score: if succeeded { similarity_threshold } else { 0.0 },
            side_effects: if succeeded {
                vec!["Semantic confusion induced".to_string()]
            } else {
                vec!["Semantic spoofing detected".to_string()]
            },
            detection_risk: 1.0 - similarity_threshold, // Higher similarity = harder to detect
            target_affected: "SemanticConfusion".to_string(),
        }
    }

    /// Hijack processing context
    fn hijack_context(&self, _data: &StreamData, _legitimate: &str, _malicious: &str, 
                     transition_smoothness: f64) -> AttackResult {
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(0.25 * transition_smoothness);
        
        AttackResult {
            attack_type: "ContextHijacking".to_string(),
            succeeded,
            impact_score: if succeeded { transition_smoothness } else { 0.0 },
            side_effects: if succeeded {
                vec!["Context drift detected".to_string()]
            } else {
                vec!["Context hijack prevented".to_string()]
            },
            detection_risk: 1.0 - transition_smoothness,
            target_affected: "ContextExploitation".to_string(),
        }
    }

    /// Perturb processing pipeline
    fn perturb_processing(&self, _data: &StreamData, perturbation_magnitude: f64, 
                        _target_components: &[String]) -> AttackResult {
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(0.3 * perturbation_magnitude);
        
        AttackResult {
            attack_type: "PerturbationAttack".to_string(),
            succeeded,
            impact_score: if succeeded { perturbation_magnitude } else { 0.0 },
            side_effects: if succeeded {
                vec!["Processing pipeline disrupted".to_string()]
            } else {
                vec!["Perturbation filtered out".to_string()]
            },
            detection_risk: perturbation_magnitude * 0.5,
            target_affected: "PipelineBypass".to_string(),
        }
    }

    /// Poison belief network
    fn poison_beliefs(&self, _data: &StreamData, _poisoned_content: &str, 
                     credibility_mask: f64, _infection_vector: &str) -> AttackResult {
        let mut rng = rand::thread_rng();
        let succeeded = rng.gen_bool(0.2 * credibility_mask);
        
        AttackResult {
            attack_type: "BeliefPoisoning".to_string(),
            succeeded,
            impact_score: if succeeded { credibility_mask } else { 0.0 },
            side_effects: if succeeded {
                vec!["Belief network contaminated".to_string()]
            } else {
                vec!["Poisoning attempt blocked".to_string()]
            },
            detection_risk: 1.0 - credibility_mask,
            target_affected: "EvidencePoisoning".to_string(),
        }
    }

    /// Should we attack based on frequency?
    fn should_attack(&self) -> bool {
        let last_attack = self.last_attack.lock().unwrap();
        let enabled = self.enabled.lock().unwrap();
        *enabled && last_attack.elapsed() >= self.attack_frequency
    }
}

#[async_trait]
impl StreamProcessor for DiggidenAdversarialSystem {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        let vulnerability_matrix = self.vulnerability_matrix.clone();
        let adaptive_engine = self.adaptive_engine.clone();
        let last_attack = self.last_attack.clone();
        let stats = self.stats.clone();
        let attack_frequency = self.attack_frequency;
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                let start_time = Instant::now();
                
                debug!("Diggiden processing: {}", data.content);
                
                // Create Diggiden instance for this processing
                let diggiden = DiggidenAdversarialSystem {
                    vulnerability_matrix: vulnerability_matrix.clone(),
                    adaptive_engine: adaptive_engine.clone(),
                    attack_frequency,
                    last_attack: last_attack.clone(),
                    stats: stats.clone(),
                    enabled: Arc::new(Mutex::new(true)),
                };
                
                let mut attack_results = Vec::new();
                
                // Check if we should perform attacks
                if diggiden.should_attack() {
                    // Generate evolved attack strategies
                    let strategies = {
                        let mut engine = adaptive_engine.lock().unwrap();
                        engine.evolve_attacks()
                    };
                    
                    // Execute attacks
                    for strategy in strategies.iter().take(3) { // Limit to 3 attacks per cycle
                        let result = diggiden.execute_attack(strategy, &data);
                        
                        // Update vulnerability matrix
                        {
                            let mut matrix = vulnerability_matrix.lock().unwrap();
                            matrix.update_vulnerability(&result.target_affected, result.succeeded);
                        }
                        
                        // Record attack for learning
                        {
                            let mut engine = adaptive_engine.lock().unwrap();
                            engine.record_attack(strategy.clone(), result.clone());
                        }
                        
                        attack_results.push(result);
                    }
                    
                    // Update last attack time
                    {
                        let mut last = last_attack.lock().unwrap();
                        *last = Instant::now();
                    }
                }
                
                // Add attack results to metadata
                let total_attacks = attack_results.len();
                let successful_attacks = attack_results.iter().filter(|r| r.succeeded).count();
                let total_impact = attack_results.iter().map(|r| r.impact_score).sum::<f64>();
                
                data = data.with_metadata("diggiden_attacks_total", &total_attacks.to_string());
                data = data.with_metadata("diggiden_attacks_successful", &successful_attacks.to_string());
                data = data.with_metadata("diggiden_total_impact", &total_impact.to_string());
                
                // Get most vulnerable components
                let most_vulnerable = {
                    let matrix = vulnerability_matrix.lock().unwrap();
                    matrix.get_most_vulnerable_components()
                };
                
                if let Some((component, score)) = most_vulnerable.first() {
                    data = data.with_metadata("diggiden_most_vulnerable", component);
                    data = data.with_metadata("diggiden_vulnerability_score", &score.to_string());
                }
                
                // Adjust confidence based on vulnerability
                if total_impact > 0.0 {
                    let vulnerability_penalty = (total_impact * 0.2).min(0.5);
                    let new_confidence = (data.confidence - vulnerability_penalty).max(0.0);
                    data = data.with_confidence(new_confidence);
                    
                    warn!("Diggiden detected vulnerabilities, confidence reduced to {:.3}", new_confidence);
                }
                
                // Update processing statistics
                {
                    let mut stats_guard = stats.lock().unwrap();
                    stats_guard.items_processed += 1;
                    if total_impact > 0.0 {
                        stats_guard.errors += 1;
                    }
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
                
                info!("Diggiden processed: {} attacks, {} successful, impact: {:.3}", 
                      total_attacks, successful_attacks, total_impact);
                
                if tx.send(data).await.is_err() {
                    warn!("Diggiden: Failed to send processed data");
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "DiggidenAdversarialSystem"
    }
    
    fn can_handle(&self, _data: &StreamData) -> bool {
        true // Can attack any data
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for DiggidenAdversarialSystem {
    fn default() -> Self {
        Self::new()
    }
} 