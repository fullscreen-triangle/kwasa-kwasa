/// Debate Platform System for Kwasa-Kwasa
/// 
/// This module implements Resolutions as actual debate platforms where Points
/// are resolved through affirmations and contentions, with probabilistic
/// scoring based on evidence quality and logical consistency.

use std::collections::{HashMap, BTreeMap};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::probabilistic::{TextPoint, ResolutionResult, ResolutionStrategy};
use crate::turbulance::perturbation_validation::{ValidationResult, validate_resolution_quality, ValidationConfig};

/// A debate platform for resolving a specific Point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebatePlatform {
    /// Unique identifier for this platform
    pub id: Uuid,
    
    /// The point being debated
    pub point: TextPoint,
    
    /// Current resolution state
    pub resolution_state: ResolutionState,
    
    /// Affirmations supporting the point
    pub affirmations: Vec<Affirmation>,
    
    /// Contentions challenging the point
    pub contentions: Vec<Contention>,
    
    /// Evidence presented in the debate
    pub evidence: Vec<Evidence>,
    
    /// Current probabilistic score (0.0-1.0)
    pub current_score: f64,
    
    /// Confidence in the current score
    pub score_confidence: f64,
    
    /// Resolution strategy being used
    pub strategy: ResolutionStrategy,
    
    /// Platform configuration
    pub config: PlatformConfig,
    
    /// Debate history and metadata
    pub metadata: DebateMetadata,
    
    /// Participants in the debate
    pub participants: Vec<Participant>,
    
    /// Current debate status
    pub status: DebateStatus,
}

/// Current state of resolution on the platform
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ResolutionState {
    /// Debate is open for contributions
    Open,
    
    /// Gathering initial evidence
    GatheringEvidence,
    
    /// Active debate in progress
    ActiveDebate,
    
    /// Reaching preliminary consensus
    PreliminaryConsensus,
    
    /// Final resolution achieved
    Resolved,
    
    /// Debate is stalled or inconclusive
    Stalled,
    
    /// Requires external expert input
    RequiresExpertReview,
    
    /// Paused for additional research
    PausedForResearch,
}

/// An affirmation supporting the point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Affirmation {
    /// Unique identifier
    pub id: Uuid,
    
    /// Content of the affirmation
    pub content: String,
    
    /// Strength of support (0.0-1.0)
    pub strength: f64,
    
    /// Confidence in this affirmation
    pub confidence: f64,
    
    /// Supporting evidence IDs
    pub evidence_ids: Vec<Uuid>,
    
    /// Source of this affirmation
    pub source: String,
    
    /// Timestamp when added
    pub timestamp: DateTime<Utc>,
    
    /// Weight in the overall resolution
    pub weight: f64,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Validation results from perturbation testing
    pub validation: Option<ValidationResult>,
}

/// A contention challenging the point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contention {
    /// Unique identifier
    pub id: Uuid,
    
    /// Content of the contention
    pub content: String,
    
    /// Strength of challenge (0.0-1.0)
    pub strength: f64,
    
    /// Confidence in this contention
    pub confidence: f64,
    
    /// Supporting evidence IDs
    pub evidence_ids: Vec<Uuid>,
    
    /// Source of this contention
    pub source: String,
    
    /// Timestamp when added
    pub timestamp: DateTime<Utc>,
    
    /// Weight in the overall resolution
    pub weight: f64,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// What aspect of the point this challenges
    pub challenge_aspect: ChallengeAspect,
    
    /// Validation results from perturbation testing
    pub validation: Option<ValidationResult>,
}

/// Aspects of a point that can be challenged
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ChallengeAspect {
    /// Challenge the factual accuracy
    FactualAccuracy,
    
    /// Challenge the logical reasoning
    LogicalReasoning,
    
    /// Challenge the contextual relevance
    ContextualRelevance,
    
    /// Challenge the scope or generalizability
    Scope,
    
    /// Challenge the interpretation
    Interpretation,
    
    /// Challenge the evidence quality
    EvidenceQuality,
    
    /// Challenge the completeness
    Completeness,
    
    /// General challenge without specific focus
    General,
}

/// Evidence supporting affirmations or contentions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Evidence {
    /// Unique identifier
    pub id: Uuid,
    
    /// Content of the evidence
    pub content: String,
    
    /// Type of evidence
    pub evidence_type: EvidenceType,
    
    /// Quality score (0.0-1.0)
    pub quality: f64,
    
    /// Relevance to the debate (0.0-1.0)
    pub relevance: f64,
    
    /// Source information
    pub source: EvidenceSource,
    
    /// Timestamp when added
    pub timestamp: DateTime<Utc>,
    
    /// Verification status
    pub verification: VerificationStatus,
    
    /// Associated tags
    pub tags: Vec<String>,
    
    /// Reliability assessment
    pub reliability: f64,
}

/// Types of evidence
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EvidenceType {
    /// Empirical data or statistics
    Empirical,
    
    /// Expert testimony or opinion
    Expert,
    
    /// Historical precedent
    Historical,
    
    /// Logical argument
    Logical,
    
    /// Analogical reasoning
    Analogical,
    
    /// Citation from authoritative source
    Citation,
    
    /// Personal observation
    Observation,
    
    /// Experimental result
    Experimental,
    
    /// Theoretical analysis
    Theoretical,
}

/// Source information for evidence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceSource {
    /// Name or description of source
    pub name: String,
    
    /// Credibility score (0.0-1.0)
    pub credibility: f64,
    
    /// Source type
    pub source_type: SourceType,
    
    /// URL or reference if available
    pub reference: Option<String>,
    
    /// Date of source publication
    pub publication_date: Option<DateTime<Utc>>,
}

/// Types of evidence sources
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    /// Academic publication
    Academic,
    
    /// Government agency
    Government,
    
    /// News organization
    News,
    
    /// Expert individual
    Expert,
    
    /// Organization or institution
    Organization,
    
    /// Personal experience
    Personal,
    
    /// Online resource
    Online,
    
    /// Book or publication
    Publication,
    
    /// Unknown or unspecified
    Unknown,
}

/// Verification status of evidence
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VerificationStatus {
    /// Verified as accurate
    Verified,
    
    /// Partially verified
    PartiallyVerified,
    
    /// Not yet verified
    Unverified,
    
    /// Verification failed
    VerificationFailed,
    
    /// Cannot be verified
    Unverifiable,
    
    /// Under verification
    UnderVerification,
}

/// Participant in the debate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Participant {
    /// Unique identifier
    pub id: Uuid,
    
    /// Name or identifier of participant
    pub name: String,
    
    /// Role in the debate
    pub role: ParticipantRole,
    
    /// Credibility score (0.0-1.0)
    pub credibility: f64,
    
    /// Expertise areas
    pub expertise: Vec<String>,
    
    /// Contributions made
    pub contributions: Vec<Uuid>, // IDs of affirmations, contentions, evidence
    
    /// Bias indicators
    pub bias_indicators: Vec<BiasIndicator>,
}

/// Roles participants can take
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ParticipantRole {
    /// Supports the point
    Advocate,
    
    /// Challenges the point
    Challenger,
    
    /// Neutral evaluator
    Evaluator,
    
    /// Subject matter expert
    Expert,
    
    /// Fact-checker
    FactChecker,
    
    /// Moderator
    Moderator,
    
    /// Automated system
    System,
}

/// Indicators of potential bias
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BiasIndicator {
    /// Type of potential bias
    pub bias_type: BiasType,
    
    /// Strength of bias indication (0.0-1.0)
    pub strength: f64,
    
    /// Description of the bias
    pub description: String,
}

/// Types of bias
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BiasType {
    /// Confirmation bias
    Confirmation,
    
    /// Financial interest
    Financial,
    
    /// Personal relationship
    Personal,
    
    /// Professional interest
    Professional,
    
    /// Ideological bias
    Ideological,
    
    /// Cultural bias
    Cultural,
    
    /// Selection bias
    Selection,
    
    /// Recency bias
    Recency,
}

/// Current status of the debate
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DebateStatus {
    /// Just created, accepting initial inputs
    Created,
    
    /// Active with ongoing contributions
    Active,
    
    /// Temporarily paused
    Paused,
    
    /// Concluded with resolution
    Concluded,
    
    /// Archived for reference
    Archived,
    
    /// Cancelled or abandoned
    Cancelled,
    
    /// Waiting for expert input
    AwaitingExpert,
    
    /// Under review
    UnderReview,
}

/// Configuration for the debate platform
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlatformConfig {
    /// Minimum evidence quality threshold
    pub min_evidence_quality: f64,
    
    /// Minimum participant credibility
    pub min_participant_credibility: f64,
    
    /// Maximum number of affirmations
    pub max_affirmations: usize,
    
    /// Maximum number of contentions
    pub max_contentions: usize,
    
    /// Require evidence validation
    pub require_evidence_validation: bool,
    
    /// Minimum consensus threshold for resolution
    pub consensus_threshold: f64,
    
    /// Enable perturbation validation
    pub enable_perturbation_validation: bool,
    
    /// Bias detection sensitivity
    pub bias_detection_sensitivity: f64,
    
    /// Auto-resolution timeout (hours)
    pub auto_resolution_timeout: Option<u64>,
}

/// Metadata about the debate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebateMetadata {
    /// When the debate was created
    pub created_at: DateTime<Utc>,
    
    /// When last updated
    pub updated_at: DateTime<Utc>,
    
    /// When resolved (if applicable)
    pub resolved_at: Option<DateTime<Utc>>,
    
    /// Total time spent in debate
    pub total_debate_time: Option<chrono::Duration>,
    
    /// Number of iterations/rounds
    pub iterations: u32,
    
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Related debates
    pub related_debates: Vec<Uuid>,
}

/// Manager for debate platforms
pub struct DebatePlatformManager {
    /// Active platforms
    platforms: HashMap<Uuid, DebatePlatform>,
    
    /// Default configuration
    default_config: PlatformConfig,
    
    /// Performance statistics
    stats: ManagerStats,
}

/// Statistics for the debate platform manager
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ManagerStats {
    /// Total platforms created
    pub total_platforms: u64,
    
    /// Currently active platforms
    pub active_platforms: u64,
    
    /// Resolved platforms
    pub resolved_platforms: u64,
    
    /// Average resolution time (hours)
    pub avg_resolution_time: f64,
    
    /// Average score confidence
    pub avg_score_confidence: f64,
    
    /// Success rate of resolutions
    pub resolution_success_rate: f64,
}

impl DebatePlatform {
    /// Create a new debate platform for a point
    pub fn new(point: TextPoint, strategy: ResolutionStrategy, config: PlatformConfig) -> Self {
        let id = Uuid::new_v4();
        let now = Utc::now();
        
        Self {
            id,
            point,
            resolution_state: ResolutionState::Open,
            affirmations: Vec::new(),
            contentions: Vec::new(),
            evidence: Vec::new(),
            current_score: 0.5, // Start neutral
            score_confidence: 0.0,
            strategy,
            config,
            metadata: DebateMetadata {
                created_at: now,
                updated_at: now,
                resolved_at: None,
                total_debate_time: None,
                iterations: 0,
                quality_metrics: HashMap::new(),
                performance_metrics: HashMap::new(),
                tags: Vec::new(),
                related_debates: Vec::new(),
            },
            participants: Vec::new(),
            status: DebateStatus::Created,
        }
    }
    
    /// Add an affirmation to the debate
    pub async fn add_affirmation(&mut self, content: String, source: String, strength: f64, confidence: f64) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        
        // Validate the affirmation if enabled
        let validation = if self.config.enable_perturbation_validation {
            let affirmation_point = TextPoint::new(content.clone(), confidence);
            Some(validate_resolution_quality(&affirmation_point, &ResolutionResult::Certain(Value::String(content.clone())), None).await?)
        } else {
            None
        };
        
        let affirmation = Affirmation {
            id,
            content,
            strength: strength.max(0.0).min(1.0),
            confidence: confidence.max(0.0).min(1.0),
            evidence_ids: Vec::new(),
            source,
            timestamp: now,
            weight: 1.0, // Default weight
            tags: Vec::new(),
            validation,
        };
        
        self.affirmations.push(affirmation);
        self.update_score().await?;
        self.metadata.updated_at = now;
        
        Ok(id)
    }
    
    /// Add a contention to the debate
    pub async fn add_contention(&mut self, content: String, source: String, strength: f64, confidence: f64, challenge_aspect: ChallengeAspect) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        
        // Validate the contention if enabled
        let validation = if self.config.enable_perturbation_validation {
            let contention_point = TextPoint::new(content.clone(), confidence);
            Some(validate_resolution_quality(&contention_point, &ResolutionResult::Certain(Value::String(content.clone())), None).await?)
        } else {
            None
        };
        
        let contention = Contention {
            id,
            content,
            strength: strength.max(0.0).min(1.0),
            confidence: confidence.max(0.0).min(1.0),
            evidence_ids: Vec::new(),
            source,
            timestamp: now,
            weight: 1.0, // Default weight
            tags: Vec::new(),
            challenge_aspect,
            validation,
        };
        
        self.contentions.push(contention);
        self.update_score().await?;
        self.metadata.updated_at = now;
        
        Ok(id)
    }
    
    /// Add evidence to support an affirmation or contention
    pub fn add_evidence(&mut self, content: String, evidence_type: EvidenceType, quality: f64, relevance: f64, source: EvidenceSource) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let now = Utc::now();
        
        // Validate evidence quality
        if quality < self.config.min_evidence_quality {
            return Err(TurbulanceError::InvalidInput(format!("Evidence quality {} below minimum threshold {}", quality, self.config.min_evidence_quality)));
        }
        
        let evidence = Evidence {
            id,
            content,
            evidence_type,
            quality: quality.max(0.0).min(1.0),
            relevance: relevance.max(0.0).min(1.0),
            source,
            timestamp: now,
            verification: VerificationStatus::Unverified,
            tags: Vec::new(),
            reliability: quality * 0.8, // Conservative estimate
        };
        
        self.evidence.push(evidence);
        self.metadata.updated_at = now;
        
        Ok(id)
    }
    
    /// Link evidence to an affirmation
    pub fn link_evidence_to_affirmation(&mut self, affirmation_id: Uuid, evidence_id: Uuid) -> Result<()> {
        // Check that evidence exists
        if !self.evidence.iter().any(|e| e.id == evidence_id) {
            return Err(TurbulanceError::InvalidInput("Evidence not found".to_string()));
        }
        
        // Find and update affirmation
        if let Some(affirmation) = self.affirmations.iter_mut().find(|a| a.id == affirmation_id) {
            if !affirmation.evidence_ids.contains(&evidence_id) {
                affirmation.evidence_ids.push(evidence_id);
            }
            Ok(())
        } else {
            Err(TurbulanceError::InvalidInput("Affirmation not found".to_string()))
        }
    }
    
    /// Link evidence to a contention
    pub fn link_evidence_to_contention(&mut self, contention_id: Uuid, evidence_id: Uuid) -> Result<()> {
        // Check that evidence exists
        if !self.evidence.iter().any(|e| e.id == evidence_id) {
            return Err(TurbulanceError::InvalidInput("Evidence not found".to_string()));
        }
        
        // Find and update contention
        if let Some(contention) = self.contentions.iter_mut().find(|c| c.id == contention_id) {
            if !contention.evidence_ids.contains(&evidence_id) {
                contention.evidence_ids.push(evidence_id);
            }
            Ok(())
        } else {
            Err(TurbulanceError::InvalidInput("Contention not found".to_string()))
        }
    }
    
    /// Update the current probabilistic score based on affirmations and contentions
    pub async fn update_score(&mut self) -> Result<()> {
        let (score, confidence) = match self.strategy {
            ResolutionStrategy::MaximumLikelihood => self.calculate_maximum_likelihood_score(),
            ResolutionStrategy::BayesianWeighted => self.calculate_bayesian_score(),
            ResolutionStrategy::ConservativeMin => self.calculate_conservative_score(),
            ResolutionStrategy::ExploratoryMax => self.calculate_exploratory_score(),
            ResolutionStrategy::WeightedAggregate => self.calculate_weighted_aggregate_score(),
            ResolutionStrategy::FullDistribution => {
                // For now, use weighted aggregate as a single score
                self.calculate_weighted_aggregate_score()
            },
        };
        
        self.current_score = score;
        self.score_confidence = confidence;
        self.metadata.iterations += 1;
        
        // Check if we've reached resolution
        if confidence >= self.config.consensus_threshold {
            self.resolution_state = ResolutionState::Resolved;
            self.status = DebateStatus::Concluded;
            self.metadata.resolved_at = Some(Utc::now());
        }
        
        Ok(())
    }
    
    /// Calculate score using maximum likelihood
    fn calculate_maximum_likelihood_score(&self) -> (f64, f64) {
        let mut total_weight = 0.0;
        let mut weighted_score = 0.0;
        let mut total_confidence = 0.0;
        
        // Process affirmations (positive evidence)
        for affirmation in &self.affirmations {
            let evidence_boost = self.calculate_evidence_boost(&affirmation.evidence_ids);
            let effective_strength = affirmation.strength * (1.0 + evidence_boost);
            let weight = affirmation.weight * affirmation.confidence;
            
            weighted_score += effective_strength * weight;
            total_weight += weight;
            total_confidence += affirmation.confidence * weight;
        }
        
        // Process contentions (negative evidence)
        for contention in &self.contentions {
            let evidence_boost = self.calculate_evidence_boost(&contention.evidence_ids);
            let effective_strength = contention.strength * (1.0 + evidence_boost);
            let weight = contention.weight * contention.confidence;
            
            weighted_score -= effective_strength * weight; // Negative impact
            total_weight += weight;
            total_confidence += contention.confidence * weight;
        }
        
        if total_weight > 0.0 {
            let score = (weighted_score / total_weight + 1.0) / 2.0; // Normalize to 0-1
            let confidence = total_confidence / total_weight;
            (score.max(0.0).min(1.0), confidence.max(0.0).min(1.0))
        } else {
            (0.5, 0.0) // Neutral if no evidence
        }
    }
    
    /// Calculate score using Bayesian weighting
    fn calculate_bayesian_score(&self) -> (f64, f64) {
        // Start with prior (point's initial confidence)
        let mut posterior = self.point.confidence;
        let mut total_evidence_weight = 1.0; // Prior weight
        
        // Update with affirmations
        for affirmation in &self.affirmations {
            let evidence_quality = self.calculate_evidence_boost(&affirmation.evidence_ids);
            let likelihood = affirmation.strength * (1.0 + evidence_quality);
            let weight = affirmation.confidence;
            
            // Bayesian update
            posterior = (posterior * total_evidence_weight + likelihood * weight) / (total_evidence_weight + weight);
            total_evidence_weight += weight;
        }
        
        // Update with contentions
        for contention in &self.contentions {
            let evidence_quality = self.calculate_evidence_boost(&contention.evidence_ids);
            let likelihood = (1.0 - contention.strength) * (1.0 + evidence_quality);
            let weight = contention.confidence;
            
            // Bayesian update
            posterior = (posterior * total_evidence_weight + likelihood * weight) / (total_evidence_weight + weight);
            total_evidence_weight += weight;
        }
        
        let confidence = (total_evidence_weight - 1.0) / (total_evidence_weight + 1.0); // Confidence grows with evidence
        (posterior.max(0.0).min(1.0), confidence.max(0.0).min(1.0))
    }
    
    /// Calculate conservative (minimum) score
    fn calculate_conservative_score(&self) -> (f64, f64) {
        let (ml_score, ml_confidence) = self.calculate_maximum_likelihood_score();
        
        // Apply conservative penalty based on contention strength
        let max_contention_strength = self.contentions.iter()
            .map(|c| c.strength * c.confidence)
            .fold(0.0, f64::max);
            
        let conservative_score = ml_score * (1.0 - max_contention_strength * 0.5);
        (conservative_score.max(0.0).min(1.0), ml_confidence)
    }
    
    /// Calculate exploratory (maximum) score
    fn calculate_exploratory_score(&self) -> (f64, f64) {
        let (ml_score, ml_confidence) = self.calculate_maximum_likelihood_score();
        
        // Apply optimistic boost based on affirmation strength
        let max_affirmation_strength = self.affirmations.iter()
            .map(|a| a.strength * a.confidence)
            .fold(0.0, f64::max);
            
        let exploratory_score = ml_score + (1.0 - ml_score) * max_affirmation_strength * 0.3;
        (exploratory_score.max(0.0).min(1.0), ml_confidence)
    }
    
    /// Calculate weighted aggregate score
    fn calculate_weighted_aggregate_score(&self) -> (f64, f64) {
        // Use maximum likelihood as the base, but this could be enhanced
        self.calculate_maximum_likelihood_score()
    }
    
    /// Calculate evidence boost from linked evidence
    fn calculate_evidence_boost(&self, evidence_ids: &[Uuid]) -> f64 {
        let mut total_boost = 0.0;
        
        for evidence_id in evidence_ids {
            if let Some(evidence) = self.evidence.iter().find(|e| e.id == *evidence_id) {
                let boost = evidence.quality * evidence.relevance * evidence.reliability;
                total_boost += boost;
            }
        }
        
        // Diminishing returns for multiple evidence
        if evidence_ids.is_empty() {
            0.0
        } else {
            total_boost / (evidence_ids.len() as f64).sqrt()
        }
    }
    
    /// Get the current resolution result
    pub fn get_resolution(&self) -> ResolutionResult {
        match self.resolution_state {
            ResolutionState::Resolved => {
                ResolutionResult::Certain(Value::Number(self.current_score))
            },
            _ => {
                ResolutionResult::Uncertain {
                    possibilities: vec![
                        (Value::Number(self.current_score), self.score_confidence),
                        (Value::Number(1.0 - self.current_score), 1.0 - self.score_confidence),
                    ],
                    confidence_interval: (
                        (self.current_score - (1.0 - self.score_confidence) * 0.5).max(0.0),
                        (self.current_score + (1.0 - self.score_confidence) * 0.5).min(1.0)
                    ),
                    aggregated_confidence: self.score_confidence,
                }
            }
        }
    }
    
    /// Generate a summary of the debate
    pub fn generate_summary(&self) -> DebateSummary {
        DebateSummary {
            platform_id: self.id,
            point_content: self.point.content.clone(),
            current_score: self.current_score,
            score_confidence: self.score_confidence,
            resolution_state: self.resolution_state.clone(),
            total_affirmations: self.affirmations.len(),
            total_contentions: self.contentions.len(),
            total_evidence: self.evidence.len(),
            key_affirmations: self.affirmations.iter()
                .take(3)
                .map(|a| a.content.clone())
                .collect(),
            key_contentions: self.contentions.iter()
                .take(3)
                .map(|c| c.content.clone())
                .collect(),
            debate_duration: self.metadata.total_debate_time,
            resolution_quality: self.assess_resolution_quality(),
        }
    }
    
    /// Assess the quality of the current resolution
    fn assess_resolution_quality(&self) -> ResolutionQuality {
        let evidence_quality = if self.evidence.is_empty() {
            0.0
        } else {
            self.evidence.iter().map(|e| e.quality).sum::<f64>() / self.evidence.len() as f64
        };
        
        let balance_score = {
            let aff_strength: f64 = self.affirmations.iter().map(|a| a.strength).sum();
            let con_strength: f64 = self.contentions.iter().map(|c| c.strength).sum();
            let total = aff_strength + con_strength;
            if total > 0.0 {
                1.0 - ((aff_strength - con_strength).abs() / total)
            } else {
                0.0
            }
        };
        
        let overall_quality = (evidence_quality + balance_score + self.score_confidence) / 3.0;
        
        ResolutionQuality {
            overall_score: overall_quality,
            evidence_quality,
            balance_score,
            confidence_score: self.score_confidence,
            completeness_score: (self.affirmations.len() + self.contentions.len()) as f64 / 10.0, // Rough heuristic
        }
    }
}

/// Summary of a debate platform
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebateSummary {
    pub platform_id: Uuid,
    pub point_content: String,
    pub current_score: f64,
    pub score_confidence: f64,
    pub resolution_state: ResolutionState,
    pub total_affirmations: usize,
    pub total_contentions: usize,
    pub total_evidence: usize,
    pub key_affirmations: Vec<String>,
    pub key_contentions: Vec<String>,
    pub debate_duration: Option<chrono::Duration>,
    pub resolution_quality: ResolutionQuality,
}

/// Quality assessment of a resolution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolutionQuality {
    pub overall_score: f64,
    pub evidence_quality: f64,
    pub balance_score: f64,
    pub confidence_score: f64,
    pub completeness_score: f64,
}

impl DebatePlatformManager {
    /// Create a new debate platform manager
    pub fn new() -> Self {
        Self {
            platforms: HashMap::new(),
            default_config: PlatformConfig::default(),
            stats: ManagerStats::default(),
        }
    }
    
    /// Create a new debate platform
    pub fn create_platform(&mut self, point: TextPoint, strategy: ResolutionStrategy, config: Option<PlatformConfig>) -> Uuid {
        let config = config.unwrap_or_else(|| self.default_config.clone());
        let platform = DebatePlatform::new(point, strategy, config);
        let id = platform.id;
        
        self.platforms.insert(id, platform);
        self.stats.total_platforms += 1;
        self.stats.active_platforms += 1;
        
        id
    }
    
    /// Get a platform by ID
    pub fn get_platform(&self, id: &Uuid) -> Option<&DebatePlatform> {
        self.platforms.get(id)
    }
    
    /// Get a mutable platform by ID
    pub fn get_platform_mut(&mut self, id: &Uuid) -> Option<&mut DebatePlatform> {
        self.platforms.get_mut(id)
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> &ManagerStats {
        &self.stats
    }
    
    /// Get all platform summaries
    pub fn get_all_summaries(&self) -> Vec<DebateSummary> {
        self.platforms.values().map(|p| p.generate_summary()).collect()
    }
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            min_evidence_quality: 0.3,
            min_participant_credibility: 0.2,
            max_affirmations: 20,
            max_contentions: 20,
            require_evidence_validation: false,
            consensus_threshold: 0.8,
            enable_perturbation_validation: true,
            bias_detection_sensitivity: 0.7,
            auto_resolution_timeout: Some(72), // 72 hours
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_debate_platform_creation() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.7);
        let config = PlatformConfig::default();
        let platform = DebatePlatform::new(point, ResolutionStrategy::MaximumLikelihood, config);
        
        assert_eq!(platform.resolution_state, ResolutionState::Open);
        assert_eq!(platform.status, DebateStatus::Created);
        assert_eq!(platform.current_score, 0.5);
    }
    
    #[tokio::test]
    async fn test_add_affirmation() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.7);
        let config = PlatformConfig::default();
        let mut platform = DebatePlatform::new(point, ResolutionStrategy::MaximumLikelihood, config);
        
        let aff_id = platform.add_affirmation(
            "The solution demonstrates superior performance".to_string(),
            "Expert Analysis".to_string(),
            0.8,
            0.9
        ).await.unwrap();
        
        assert_eq!(platform.affirmations.len(), 1);
        assert_eq!(platform.affirmations[0].id, aff_id);
        assert!(platform.current_score > 0.5); // Should increase score
    }
    
    #[tokio::test]
    async fn test_add_contention() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.7);
        let config = PlatformConfig::default();
        let mut platform = DebatePlatform::new(point, ResolutionStrategy::MaximumLikelihood, config);
        
        let con_id = platform.add_contention(
            "The solution has significant limitations".to_string(),
            "Critical Review".to_string(),
            0.6,
            0.8,
            ChallengeAspect::Scope
        ).await.unwrap();
        
        assert_eq!(platform.contentions.len(), 1);
        assert_eq!(platform.contentions[0].id, con_id);
        assert!(platform.current_score < 0.5); // Should decrease score
    }
    
    #[tokio::test]
    async fn test_evidence_linking() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.7);
        let config = PlatformConfig::default();
        let mut platform = DebatePlatform::new(point, ResolutionStrategy::MaximumLikelihood, config);
        
        // Add affirmation
        let aff_id = platform.add_affirmation(
            "Strong performance data".to_string(),
            "Data Analysis".to_string(),
            0.8,
            0.9
        ).await.unwrap();
        
        // Add evidence
        let source = EvidenceSource {
            name: "Performance Study".to_string(),
            credibility: 0.9,
            source_type: SourceType::Academic,
            reference: None,
            publication_date: None,
        };
        
        let evidence_id = platform.add_evidence(
            "Benchmark results show 40% improvement".to_string(),
            EvidenceType::Empirical,
            0.8,
            0.9,
            source
        ).unwrap();
        
        // Link evidence to affirmation
        platform.link_evidence_to_affirmation(aff_id, evidence_id).unwrap();
        
        assert!(platform.affirmations[0].evidence_ids.contains(&evidence_id));
    }
    
    #[tokio::test]
    async fn test_platform_manager() {
        let mut manager = DebatePlatformManager::new();
        let point = TextPoint::new("Test point".to_string(), 0.6);
        
        let platform_id = manager.create_platform(point, ResolutionStrategy::BayesianWeighted, None);
        
        assert_eq!(manager.get_stats().total_platforms, 1);
        assert_eq!(manager.get_stats().active_platforms, 1);
        assert!(manager.get_platform(&platform_id).is_some());
    }
} 