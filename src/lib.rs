//! Kwasa-Kwasa: A comprehensive text processing and writing support system
//! 
//! This library provides:
//! - Advanced text unit processing and boundary detection
//! - Multi-domain analysis (text, genomics, chemistry, spectrometry)
//! - Intelligent orchestration with metacognitive and biomimetic processing
//! - Goal-oriented writing assistance with intervention systems
//! - Stream processing capabilities
//! - Comprehensive configuration management
//! 
//! # Quick Start
//! 
//! ```rust
//! use kwasa_kwasa::{KwasaSystem, Config};
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = Config::default();
//! let mut system = KwasaSystem::new(config).await?;
//! 
//! // Process text
//! let result = system.process_text("Your text here", None).await?;
//! 
//! // Set a writing goal
//! let goal_id = system.set_goal("Write a 1000-word essay about AI", None, Some(0.8)).await?;
//! 
//! // Get writing assistance
//! let suggestions = system.get_suggestions().await?;
//! # Ok(())
//! # }
//! ```

pub mod text_unit;
pub mod orchestrator;
pub mod genomic;
pub mod chemistry;
pub mod spectrometry;
pub mod turbulance;
pub mod knowledge;
pub mod cli;
pub mod utils;
pub mod pattern;
pub mod evidence;
pub mod error;
pub mod visualization;
pub mod external_apis;
pub mod image;
pub mod audio;
pub mod templates;

// **NEW IMPLEMENTATION MODULES**
pub mod fullscreen;      // Module 1: Network graph visualization (.fs files)
pub mod harare;          // Module 2: Meta orchestrator logging (.hre files)
pub mod space_computer;     // Module 3: Video processing (Moriarty + Vibrio + Space Computer + Morphine)
pub mod nebuchadnezzar;  // Module 4: Orchestrator-controlled AI assistant
pub mod trebuchet;       // Module 5: Microservices execution engine
pub mod sighthound;      // Module 6: Geospatial processing
pub mod zangalewa;       // Module 7: Code execution and intelligent error handling

// Re-export important modules
pub use orchestrator::Orchestrator;
pub use text_unit::{TextUnit, TextUnitId, TextUnitType, TextUnitRegistry, BoundaryDetectionOptions};
pub use error::{Error, Result, ErrorReporter};

// Import orchestrator types
pub use orchestrator::{
    Goal, GoalType, GoalMetrics, SuccessCriterion, Strategy, GoalStatus,
    Context,  WritingPattern,
    InterventionSystem, InterventionType, ActiveIntervention, InterventionOutcome,
    Config,
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::error::{Error, Result, ErrorReporter};
    pub use crate::{KwasaSystem, Config, Goal, Context, InterventionSystem};
}

use std::sync::Arc;
use tokio::sync::Mutex;
use log::info;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Main system that coordinates all Kwasa-Kwasa functionality
pub struct KwasaSystem {
    /// System configuration
    config: Config,
    
    /// The orchestrator that manages processing
    orchestrator: Arc<Mutex<Orchestrator>>,
    
    /// Text unit registry
    text_registry: Arc<Mutex<TextUnitRegistry>>,
    
    /// Current context
    context: Arc<Mutex<Context>>,
    
    /// Intervention system
    intervention_system: Arc<Mutex<InterventionSystem>>,
    
    /// Current goals
    goals: Arc<Mutex<Vec<Goal>>>,
    
    /// Knowledge database
    knowledge_db: Arc<Mutex<Result<knowledge::database::KnowledgeDatabase>>>,
    
    /// System state
    state: SystemState,
}

/// Current state of the system
#[derive(Debug, Clone, PartialEq)]
pub enum SystemState {
    /// System is initializing
    Initializing,
    
    /// System is ready for use
    Ready,
    
    /// System is actively processing
    Processing,
    
    /// System is paused
    Paused,
    
    /// System encountered an error
    Error(String),
    
    /// System is shutting down
    Shutdown,
}

/// Result type for system operations
pub type KwasaResult<T> = Result<T>;

/// Error types for the system
#[derive(Debug, thiserror::Error)]
pub enum KwasaError {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Processing error: {0}")]
    Processing(String),
    
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("System error: {0}")]
    System(String),
    
    #[error("Goal error: {0}")]
    Goal(String),
    
    #[error("Intervention error: {0}")]
    Intervention(String),
    
    #[error("Knowledge database error: {0}")]
    Knowledge(String),
}

/// Processing options for text analysis
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Enable boundary detection
    pub enable_boundaries: bool,
    
    /// Enable quality analysis
    pub enable_quality: bool,
    
    /// Enable semantic analysis
    pub enable_semantic: bool,
    
    /// Enable goal alignment checking
    pub enable_goal_alignment: bool,
    
    /// Enable intervention suggestions
    pub enable_interventions: bool,
    
    /// Maximum processing time (seconds)
    pub max_processing_time: u64,
    
    /// Domain-specific options
    pub domain_options: std::collections::HashMap<String, String>,
}

/// Result of text processing
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Processed text units
    pub units: Vec<TextUnit>,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Semantic analysis results
    pub semantic_analysis: SemanticAnalysis,
    
    /// Goal alignment scores
    pub goal_alignments: Vec<GoalAlignment>,
    
    /// Suggested interventions
    pub interventions: Vec<InterventionSuggestion>,
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Quality metrics for processed text
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Readability score (0.0-1.0)
    pub readability: f64,
    
    /// Coherence score (0.0-1.0)
    pub coherence: f64,
    
    /// Grammar score (0.0-1.0)
    pub grammar: f64,
    
    /// Style consistency (0.0-1.0)
    pub style: f64,
    
    /// Overall quality score (0.0-1.0)
    pub overall: f64,
}

/// Semantic analysis results
#[derive(Debug, Clone)]
pub struct SemanticAnalysis {
    /// Main topics identified
    pub topics: Vec<String>,
    
    /// Sentiment analysis (if applicable)
    pub sentiment: Option<Sentiment>,
    
    /// Key phrases extracted
    pub key_phrases: Vec<String>,
    
    /// Concept relationships
    pub concept_relations: Vec<ConceptRelation>,
}

/// Sentiment analysis result
#[derive(Debug, Clone)]
pub struct Sentiment {
    /// Sentiment polarity (-1.0 to 1.0)
    pub polarity: f64,
    
    /// Confidence in sentiment analysis (0.0-1.0)
    pub confidence: f64,
    
    /// Sentiment label
    pub label: String,
}

/// Relationship between concepts
#[derive(Debug, Clone)]
pub struct ConceptRelation {
    /// Source concept
    pub source: String,
    
    /// Target concept
    pub target: String,
    
    /// Relationship type
    pub relation_type: String,
    
    /// Strength of relationship (0.0-1.0)
    pub strength: f64,
}

/// Goal alignment result
#[derive(Debug, Clone)]
pub struct GoalAlignment {
    /// Goal ID
    pub goal_id: String,
    
    /// Alignment score (0.0-1.0)
    pub score: f64,
    
    /// Explanation of alignment
    pub explanation: String,
    
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Intervention suggestion
#[derive(Debug, Clone)]
pub struct InterventionSuggestion {
    /// Type of intervention
    pub intervention_type: InterventionType,
    
    /// Priority (0.0-1.0)
    pub priority: f64,
    
    /// Description
    pub description: String,
    
    /// Specific actions to take
    pub actions: Vec<String>,
}

/// Processing metadata
#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    /// Processing time (milliseconds)
    pub processing_time_ms: u64,
    
    /// Number of units processed
    pub units_processed: usize,
    
    /// Processing pipeline used
    pub pipeline: String,
    
    /// Warnings encountered
    pub warnings: Vec<String>,
    
    /// Additional metadata
    pub additional: std::collections::HashMap<String, String>,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            enable_boundaries: true,
            enable_quality: true,
            enable_semantic: true,
            enable_goal_alignment: true,
            enable_interventions: true,
            max_processing_time: 30, // 30 seconds
            domain_options: std::collections::HashMap::new(),
        }
    }
}

impl KwasaSystem {
    /// Create a new Kwasa-Kwasa system with the given configuration
    pub async fn new(config: Config) -> KwasaResult<Self> {
        info!("Initializing Kwasa-Kwasa system");
        
        // Validate configuration
        if let Err(errors) = config.validate() {
            return Err(KwasaError::Config(format!("Configuration validation failed: {:?}", errors)));
        }
        
        // Initialize components
        let temp_path = std::env::temp_dir().join("kwasa_system.db");
        let knowledge_db_result = knowledge::database::KnowledgeDatabase::new(temp_path);
        let knowledge_db = Arc::new(Mutex::new(knowledge_db_result));
        let default_goal = Goal::new("Default Goal", 0.5);
        let orchestrator = Arc::new(Mutex::new(Orchestrator::new(default_goal, knowledge_db.clone())));
        let text_registry = Arc::new(Mutex::new(TextUnitRegistry::new()));
        let context = Arc::new(Mutex::new(Context::new()));
        let intervention_system = Arc::new(Mutex::new(InterventionSystem::new()));
        let goals = Arc::new(Mutex::new(Vec::new()));
        
        info!("Kwasa-Kwasa system initialized successfully");
        
        Ok(Self {
            config,
            orchestrator,
            text_registry,
            context,
            intervention_system,
            goals,
            knowledge_db,
            state: SystemState::Ready,
        })
    }
    
    /// Create a system with default configuration
    pub async fn with_defaults() -> KwasaResult<Self> {
        Self::new(Config::default()).await
    }
    
    /// Process text with the given options
    pub async fn process_text(&mut self, text: &str, options: Option<ProcessingOptions>) -> KwasaResult<ProcessingResult> {
        let start_time = std::time::Instant::now();
        self.state = SystemState::Processing;
        
        let options = options.unwrap_or_default();
        
        info!("Processing text of {} characters", text.len());
        
        let mut units = Vec::new();
        let mut warnings = Vec::new();
        
        // Boundary detection
        if options.enable_boundaries {
            let mut registry = self.text_registry.lock().await;
            
            // Detect different boundary types
            let detection_options = BoundaryDetectionOptions::default();
            
            let paragraph_ids = text_unit::boundary::detect_paragraph_boundaries(text, &mut *registry, &detection_options);
            let sentence_ids = text_unit::boundary::detect_sentence_boundaries(text, &mut *registry, &detection_options);
            let word_ids = text_unit::boundary::detect_word_boundaries(text, &mut *registry, &detection_options);
            
            // Collect units
            for id in [paragraph_ids, sentence_ids, word_ids].concat() {
                if let Some(unit) = registry.get_unit(id) {
                    units.push(unit.clone());
                }
            }
        }
        
        // Quality analysis
        let quality_metrics = if options.enable_quality {
            self.analyze_quality(text).await?
        } else {
            QualityMetrics {
                readability: 0.5,
                coherence: 0.5,
                grammar: 0.5,
                style: 0.5,
                overall: 0.5,
            }
        };
        
        // Semantic analysis
        let semantic_analysis = if options.enable_semantic {
            self.analyze_semantics(text).await?
        } else {
            SemanticAnalysis {
                topics: Vec::new(),
                sentiment: None,
                key_phrases: Vec::new(),
                concept_relations: Vec::new(),
            }
        };
        
        // Goal alignment
        let goal_alignments = if options.enable_goal_alignment {
            self.check_goal_alignment(text).await?
        } else {
            Vec::new()
        };
        
        // Intervention suggestions
        let interventions = if options.enable_interventions {
            self.get_intervention_suggestions(text, &quality_metrics).await?
        } else {
            Vec::new()
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        let metadata = ProcessingMetadata {
            processing_time_ms: processing_time,
            units_processed: units.len(),
            pipeline: "default".to_string(),
            warnings,
            additional: std::collections::HashMap::new(),
        };
        
        self.state = SystemState::Ready;
        
        info!("Text processing completed in {}ms", processing_time);
        
        Ok(ProcessingResult {
            units,
            quality_metrics,
            semantic_analysis,
            goal_alignments,
            interventions,
            metadata,
        })
    }
    
    /// Set a writing goal
    pub async fn set_goal(&mut self, description: &str, goal_type: Option<GoalType>, priority: Option<f64>) -> KwasaResult<String> {
        let goal = Goal::with_type_and_keywords(
            description,
            goal_type.unwrap_or(GoalType::Personal),
            Vec::new(),
            priority.unwrap_or(0.5),
        );
        
        let goal_id = goal.id().to_string();
        
        {
            let mut goals = self.goals.lock().await;
            goals.push(goal.clone());
        }
        
        {
            let mut context = self.context.lock().await;
            context.update_goal_state(&goal);
        }
        
        info!("Goal set: {}", description);
        
        Ok(goal_id)
    }
    
    /// Get all current goals
    pub async fn get_goals(&self) -> KwasaResult<Vec<Goal>> {
        let goals = self.goals.lock().await;
        Ok(goals.clone())
    }
    
    /// Update goal progress
    pub async fn update_goal_progress(&mut self, goal_id: &str, progress: f64) -> KwasaResult<()> {
        let mut goals = self.goals.lock().await;
        
        if let Some(goal) = goals.iter_mut().find(|g| g.id() == goal_id) {
            goal.update_completion(progress);
            info!("Updated goal {} progress to {:.1}%", goal_id, progress * 100.0);
            Ok(())
        } else {
            Err(KwasaError::Goal(format!("Goal not found: {}", goal_id)))
        }
    }
    
    /// Get writing suggestions based on current context
    pub async fn get_suggestions(&self) -> KwasaResult<Vec<String>> {
        let context = self.context.lock().await;
        let goals = self.goals.lock().await;
        let intervention_system = self.intervention_system.lock().await;
        
        let current_goal = goals.first();
        let suggestions = intervention_system.get_suggestions(&*context, current_goal);
        
        Ok(suggestions)
    }
    
    /// Add knowledge to the knowledge database
    pub async fn add_knowledge(&mut self, concept: &str, description: &str, sources: Vec<String>) -> KwasaResult<()> {
        let knowledge_db_result = self.knowledge_db.lock().await;
        
        match knowledge_db_result.as_ref() {
            Ok(db) => {
                // For now, just log the knowledge addition
                // In a real implementation, we'd need a mutable reference
                info!("Would add knowledge concept: {} - {}", concept, description);
                Ok(())
            },
            Err(e) => Err(KwasaError::Knowledge(format!("Database not available: {}", e)))
        }
    }
    
    /// Search knowledge database
    pub async fn search_knowledge(&self, query: &str) -> KwasaResult<Vec<knowledge::KnowledgeResult>> {
        let knowledge_db_result = self.knowledge_db.lock().await;
        
        match knowledge_db_result.as_ref() {
            Ok(db) => {
                info!("Searching knowledge for: {}", query);
                // For now, return empty results
                // In a real implementation, we'd search the database
                Ok(Vec::new())
            },
            Err(e) => Err(KwasaError::Knowledge(format!("Database not available: {}", e)))
        }
    }
    
    /// Get system status
    pub fn get_status(&self) -> SystemState {
        self.state.clone()
    }
    
    /// Shutdown the system gracefully
    pub async fn shutdown(&mut self) -> KwasaResult<()> {
        info!("Shutting down Kwasa-Kwasa system");
        
        self.state = SystemState::Shutdown;
        
        // Save any pending state
        // Clean up resources
        // Stop background tasks
        
        info!("System shutdown completed");
        
        Ok(())
    }
    
    // Internal helper methods
    async fn analyze_quality(&self, text: &str) -> KwasaResult<QualityMetrics> {
        // Implement quality analysis
        // This is a placeholder implementation
        
        let word_count = text.split_whitespace().count();
        let sentence_count = text.matches(&['.', '!', '?'][..]).count();
        
        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };
        
        // Simple readability calculation (Flesch Reading Ease approximation)
        let readability = if sentence_count > 0 && word_count > 0 {
            let avg_word_length = text.chars().filter(|c| c.is_alphabetic()).count() as f64 / word_count as f64;
            let score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length);
            (score / 100.0).max(0.0).min(1.0)
        } else {
            0.5
        };
        
        Ok(QualityMetrics {
            readability,
            coherence: 0.7, // Placeholder
            grammar: 0.8,   // Placeholder
            style: 0.6,     // Placeholder
            overall: (readability + 0.7 + 0.8 + 0.6) / 4.0,
        })
    }
    
    async fn analyze_semantics(&self, text: &str) -> KwasaResult<SemanticAnalysis> {
        // Implement semantic analysis
        // This is a placeholder implementation
        
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut topics = Vec::new();
        let mut key_phrases = Vec::new();
        
        // Simple keyword extraction
        for word in &words {
            let clean_word = word.trim_matches(&['.', ',', '!', '?', ';', ':'][..]).to_lowercase();
            if clean_word.len() > 4 && !is_stop_word(&clean_word) {
                if !topics.contains(&clean_word) {
                    topics.push(clean_word.clone());
                }
            }
        }
        
        // Limit to top 5 topics
        topics.truncate(5);
        
        // Extract some key phrases (simplified)
        for window in words.windows(2) {
            let phrase = window.join(" ").to_lowercase();
            if phrase.len() > 6 && !key_phrases.contains(&phrase) {
                key_phrases.push(phrase);
            }
        }
        key_phrases.truncate(5);
        
        Ok(SemanticAnalysis {
            topics,
            sentiment: Some(Sentiment {
                polarity: 0.1, // Slightly positive placeholder
                confidence: 0.6,
                label: "neutral".to_string(),
            }),
            key_phrases,
            concept_relations: Vec::new(), // Placeholder
        })
    }
    
    async fn check_goal_alignment(&self, text: &str) -> KwasaResult<Vec<GoalAlignment>> {
        let goals = self.goals.lock().await;
        let mut alignments = Vec::new();
        
        for goal in goals.iter() {
            let score = goal.evaluate_alignment(text);
            
            alignments.push(GoalAlignment {
                goal_id: goal.id().to_string(),
                score,
                explanation: if score > 0.7 {
                    "Text aligns well with goal".to_string()
                } else if score > 0.4 {
                    "Text partially aligns with goal".to_string()
                } else {
                    "Text has low alignment with goal".to_string()
                },
                suggestions: if score < 0.5 {
                    vec!["Consider adding more relevant keywords".to_string()]
                } else {
                    Vec::new()
                },
            });
        }
        
        Ok(alignments)
    }
    
    async fn get_intervention_suggestions(&self, text: &str, quality: &QualityMetrics) -> KwasaResult<Vec<InterventionSuggestion>> {
        let mut suggestions = Vec::new();
        
        // Check for quality issues
        if quality.readability < 0.5 {
            suggestions.push(InterventionSuggestion {
                intervention_type: InterventionType::QualityDegradation,
                priority: 0.8,
                description: "Text readability could be improved".to_string(),
                actions: vec![
                    "Try shorter sentences".to_string(),
                    "Use simpler vocabulary".to_string(),
                ],
            });
        }
        
        if quality.coherence < 0.6 {
            suggestions.push(InterventionSuggestion {
                intervention_type: InterventionType::Confusion,
                priority: 0.7,
                description: "Text coherence could be better".to_string(),
                actions: vec![
                    "Add transition phrases".to_string(),
                    "Reorganize content for better flow".to_string(),
                ],
            });
        }
        
        // Check for writer's block indicators
        if text.len() < 50 {
            suggestions.push(InterventionSuggestion {
                intervention_type: InterventionType::WritersBlock,
                priority: 0.9,
                description: "Very short text might indicate writer's block".to_string(),
                actions: vec![
                    "Try free writing for 5 minutes".to_string(),
                    "Use a writing prompt".to_string(),
                ],
            });
        }
        
        Ok(suggestions)
    }
}

/// Check if a word is a stop word (simplified implementation)
fn is_stop_word(word: &str) -> bool {
    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "from", "of", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "must", "that", "this", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    ];
    
    STOP_WORDS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_creation() {
        let system = KwasaSystem::with_defaults().await;
        assert!(system.is_ok());
        
        let system = system.unwrap();
        assert_eq!(system.get_status(), SystemState::Ready);
    }
    
    #[tokio::test]
    async fn test_text_processing() {
        let mut system = KwasaSystem::with_defaults().await.unwrap();
        
        let text = "This is a test sentence for processing. It contains multiple sentences and should be analyzed properly.";
        let result = system.process_text(text, None).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        
        assert!(!result.units.is_empty());
        assert!(result.quality_metrics.overall > 0.0);
    }
    
    #[tokio::test]
    async fn test_goal_management() {
        let mut system = KwasaSystem::with_defaults().await.unwrap();
        
        let goal_id = system.set_goal("Write a test essay", None, Some(0.8)).await.unwrap();
        
        let goals = system.get_goals().await.unwrap();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].id(), goal_id);
        
        system.update_goal_progress(&goal_id, 0.5).await.unwrap();
        
        let goals = system.get_goals().await.unwrap();
        assert_eq!(goals[0].completion(), 0.5);
    }
    
    #[tokio::test]
    async fn test_knowledge_management() {
        let mut system = KwasaSystem::with_defaults().await.unwrap();
        
        system.add_knowledge(
            "artificial intelligence",
            "The simulation of human intelligence processes by machines",
            vec!["Encyclopedia".to_string()]
        ).await.unwrap();
        
        let results = system.search_knowledge("artificial").await.unwrap();
        assert!(!results.is_empty());
    }
}

/// The main Kwasa-Kwasa framework that orchestrates all modules
#[derive(Debug)]
pub struct KwasaKwasa {
    pub id: Uuid,
    pub config: KwasaKwasaConfig,
    
    // Core semantic engines
    pub turbulance_engine: TurbulanceEngine,
    pub orchestrator: Orchestrator,
    
    // Implementation modules
    pub fullscreen: fullscreen::FullscreenSystem,
    pub harare: harare::HarareOrchestrator,
    pub spectacular: space_computer::SpectacularSystem,
    pub nebuchadnezzar: nebuchadnezzar::NebuchadnezzarAssistant,
    pub trebuchet: trebuchet::TrebuchetSystem,
    pub sighthound: sighthound::SighthoundSystem,
    pub zangalewa: zangalewa::ZangalewaSystem,
    
    // State management
    pub session_id: Uuid,
    pub workspace_path: PathBuf,
    pub active_contexts: HashMap<String, serde_json::Value>,
}

/// Configuration for the complete Kwasa-Kwasa system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KwasaKwasaConfig {
    pub workspace_path: PathBuf,
    pub enable_all_modules: bool,
    
    // Module-specific configurations
    pub fullscreen_config: Option<fullscreen::FullscreenConfig>,
    pub harare_config: harare::OrchestratorConfig,
    pub spectacular_config: space_computer::SpectacularConfig,
    pub nebuchadnezzar_config: nebuchadnezzar::AssistantConfig,
    pub trebuchet_config: trebuchet::TrebuchetConfig,
    pub sighthound_config: sighthound::SighthoundConfig,
    pub zangalewa_config: zangalewa::ZangalewaConfig,
    
    // Core engine configurations
    pub turbulance_config: TurbulanceConfig,
    pub orchestrator_config: OrchestratorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulanceConfig {
    pub semantic_processing: bool,
    pub cross_modal_analysis: bool,
    pub proposition_reasoning: bool,
    pub motion_tracking: bool,
    pub evidence_integration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub enable_metacognition: bool,
    pub learning_enabled: bool,
    pub auto_optimization: bool,
    pub parallel_execution: bool,
    pub resource_management: bool,
}

impl KwasaKwasa {
    /// Create a new Kwasa-Kwasa system with all modules
    pub async fn new(config: KwasaKwasaConfig) -> Result<Self> {
        let id = Uuid::new_v4();
        let session_id = Uuid::new_v4();
        
        // Initialize core engines
        let turbulance_engine = TurbulanceEngine::new(config.turbulance_config.clone()).await?;
        let orchestrator = Orchestrator::new(config.orchestrator_config.clone()).await?;
        
        // Initialize implementation modules
        let fullscreen = fullscreen::FullscreenSystem::new(config.workspace_path.clone());
        
        let harare = harare::HarareOrchestrator::new(config.harare_config.clone()).await?;
        
        let spectacular = space_computer::SpectacularSystem::new(config.spectacular_config.clone()).await?;
        
        let nebuchadnezzar = nebuchadnezzar::NebuchadnezzarAssistant::new(
            config.nebuchadnezzar_config.clone()
        ).await?;
        
        let trebuchet = trebuchet::TrebuchetSystem::new(config.trebuchet_config.clone()).await?;
        
        let sighthound = sighthound::SighthoundSystem::new(config.sighthound_config.clone()).await?;
        
        let zangalewa = zangalewa::ZangalewaSystem::new(config.zangalewa_config.clone()).await?;
        
        Ok(Self {
            id,
            config: config.clone(),
            turbulance_engine,
            orchestrator,
            fullscreen,
            harare,
            spectacular,
            nebuchadnezzar,
            trebuchet,
            sighthound,
            zangalewa,
            session_id,
            workspace_path: config.workspace_path,
            active_contexts: HashMap::new(),
        })
    }

    /// Start the complete Kwasa-Kwasa system
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting Kwasa-Kwasa framework: {}", self.id);
        
        // Start core orchestrator first
        self.harare.start().await?;
        
        // Start all implementation modules
        if self.config.enable_all_modules {
            // Discover project structure with Fullscreen
            self.fullscreen.discover_project_structure().await?;
            
            // Start microservices with Trebuchet
            self.trebuchet.start().await?;
            
            // Initialize other modules as needed
        }
        
        tracing::info!("Kwasa-Kwasa framework started successfully");
        Ok(())
    }

    /// Process Turbulance code through the complete system
    pub async fn process_turbulance_code(&mut self, code: &str) -> Result<ProcessingResult> {
        tracing::info!("Processing Turbulance code through Kwasa-Kwasa system");
        
        // Let Harare orchestrate the execution
        let orchestrator_result = self.harare.process_turbulance_code(code).await?;
        
        // The orchestrator will coordinate with other modules as needed
        // based on the semantic analysis of the code
        
        Ok(ProcessingResult {
            session_id: self.session_id,
            orchestrator_result,
            module_results: HashMap::new(),
            execution_time_ms: 0, // Would be measured
            quality_score: 0.95,
        })
    }

    /// Execute code with Zangalewa
    pub async fn execute_code(&mut self, request: zangalewa::CodeExecutionRequest) -> Result<zangalewa::ExecutionSession> {
        self.zangalewa.execute_code(request).await
    }

    /// Process video with Spectacular
    pub async fn process_video(&mut self, task: space_computer::VideoTask) -> Result<space_computer::ProcessingResults> {
        self.spectacular.process_video(task).await
    }

    /// Process geospatial data with Sighthound
    pub async fn process_geospatial_data(&mut self, request: sighthound::GeospatialProcessingRequest) -> Result<sighthound::ProcessingResults> {
        self.sighthound.process_geospatial_data(request).await
    }

    /// Execute service request through Trebuchet
    pub async fn execute_service_request(&self, request: trebuchet::ServiceExecutionRequest) -> Result<trebuchet::ServiceExecutionResponse> {
        self.trebuchet.execute_service_request(request).await
    }

    /// Invoke AI assistant through Nebuchadnezzar
    pub async fn invoke_ai_assistant(&mut self, request: nebuchadnezzar::InvocationRequest) -> Result<nebuchadnezzar::AssistantResponse> {
        self.nebuchadnezzar.process_invocation(request).await
    }

    /// Generate network graph with Fullscreen
    pub async fn generate_network_graph(&self, output_path: &std::path::Path) -> Result<()> {
        self.fullscreen.generate_visualization(output_path).await
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> SystemStatus {
        SystemStatus {
            framework_id: self.id,
            session_id: self.session_id,
            uptime_seconds: 0, // Would be calculated
            modules_status: ModulesStatus {
                fullscreen_active: true,
                harare_active: true,
                spectacular_active: true,
                nebuchadnezzar_active: true,
                trebuchet_active: true,
                sighthound_active: true,
                zangalewa_active: true,
            },
            performance_metrics: PerformanceMetrics {
                total_requests_processed: 0, // Would be tracked
                average_response_time_ms: 0.0, // Would be calculated
                error_rate_percent: 0.0, // Would be calculated
                resource_utilization_percent: 0.0, // Would be measured
            },
            active_contexts: self.active_contexts.len(),
            workspace_path: self.workspace_path.clone(),
        }
    }

    /// Process multi-modal semantic operations
    pub async fn process_semantic_operation(&mut self, operation: SemanticOperation) -> Result<SemanticResult> {
        match operation {
            SemanticOperation::TextAnalysis { text, analysis_type } => {
                // Use Turbulance engine for text processing
                let text_unit = text_unit::TextUnit::new(&text);
                let result = self.turbulance_engine.process_text_unit(text_unit).await?;
                Ok(SemanticResult::TextResult(result))
            }
            SemanticOperation::VideoAnalysis { video_path, analysis_options } => {
                // Use Spectacular for video processing
                let task = space_computer::VideoTask {
                    id: Uuid::new_v4(),
                    input_path: video_path,
                    output_path: self.workspace_path.join("video_output"),
                    task_type: space_computer::VideoTaskType::PoseAnalysis,
                    processing_config: space_computer::ProcessingConfig {
                        enable_pose_detection: true,
                        enable_biomechanical_analysis: true,
                        enable_motion_tracking: true,
                        enable_annotation_ai: true,
                        enable_3d_visualization: true,
                        enable_streaming: false,
                        custom_parameters: HashMap::new(),
                    },
                    metadata: space_computer::VideoMetadata {
                        duration_seconds: 0.0,
                        frame_count: 0,
                        frame_rate: 0.0,
                        resolution: (0, 0),
                        file_size_bytes: 0,
                        codec: "unknown".to_string(),
                        creation_time: std::time::SystemTime::now(),
                        subject_info: None,
                    },
                    status: space_computer::TaskStatus::Queued,
                    progress: 0.0,
                    results: None,
                };
                
                let result = self.spectacular.process_video(task).await?;
                Ok(SemanticResult::VideoResult(result))
            }
            SemanticOperation::GeospatialAnalysis { data_sources, pipeline } => {
                // Use Sighthound for geospatial processing
                let request = sighthound::GeospatialProcessingRequest {
                    data_sources,
                    pipeline,
                    output_requirements: Vec::new(),
                    quality_requirements: sighthound::QualityRequirements {
                        min_accuracy_m: 5.0,
                        min_completeness_percent: 90.0,
                        min_consistency_score: 0.8,
                    },
                };
                
                let result = self.sighthound.process_geospatial_data(request).await?;
                Ok(SemanticResult::GeospatialResult(result))
            }
            SemanticOperation::CodeExecution { code_files, execution_options } => {
                // Use Zangalewa for code execution
                let request = zangalewa::CodeExecutionRequest {
                    project_path: self.workspace_path.clone(),
                    language: "rust".to_string(), // Would be detected
                    execution_type: zangalewa::ExecutionType::ScriptExecution,
                    code_files,
                    execution_options,
                    analysis_options: zangalewa::AnalysisOptions {
                        static_analysis: true,
                        dynamic_analysis: true,
                        dependency_analysis: true,
                        performance_analysis: true,
                        security_analysis: true,
                        quality_analysis: true,
                    },
                };
                
                let result = self.zangalewa.execute_code(request).await?;
                Ok(SemanticResult::CodeResult(result))
            }
        }
    }

    /// Create a default configuration for Kwasa-Kwasa
    pub fn default_config(workspace_path: PathBuf) -> KwasaKwasaConfig {
        KwasaKwasaConfig {
            workspace_path: workspace_path.clone(),
            enable_all_modules: true,
            fullscreen_config: None,
            harare_config: harare::OrchestratorConfig {
                project_root: workspace_path.clone(),
                log_level: harare::LogLevel::Info,
                auto_recovery: true,
                parallel_execution: true,
                max_retry_attempts: 3,
                decision_timeout_ms: 5000,
                metacognitive_interval_ms: 1000,
                hre_output_path: workspace_path.join("logs"),
            },
            spectacular_config: space_computer::SpectacularConfig {
                processing_mode: space_computer::ProcessingMode::Hybrid,
                video_formats: vec!["mp4".to_string(), "avi".to_string(), "mov".to_string()],
                output_formats: vec!["json".to_string(), "csv".to_string(), "video".to_string()],
                quality_settings: space_computer::QualitySettings {
                    input_resolution: (1920, 1080),
                    output_resolution: (1920, 1080),
                    frame_rate: 30.0,
                    compression_quality: 0.85,
                    pose_detection_accuracy: 0.9,
                    biomechanical_precision: 0.95,
                },
                performance_settings: space_computer::PerformanceSettings {
                    max_concurrent_videos: 4,
                    gpu_acceleration: true,
                    cpu_thread_count: 8,
                    memory_limit_mb: 8192,
                    cache_size_mb: 2048,
                    parallel_processing: true,
                },
                ai_models: space_computer::AIModelConfig {
                    pose_detection_model: "mediapipe_blazepose".to_string(),
                    object_tracking_model: "deepsort".to_string(),
                    biomechanical_analysis_model: "custom_biomech".to_string(),
                    motion_prediction_model: "lstm_motion".to_string(),
                    annotation_ai_model: "gpt4_vision".to_string(),
                },
                streaming_config: space_computer::StreamingConfig {
                    enable_streaming: false,
                    stream_protocols: vec!["rtmp".to_string(), "webrtc".to_string()],
                    buffer_size_seconds: 5.0,
                    adaptive_bitrate: true,
                    low_latency_mode: false,
                },
            },
            nebuchadnezzar_config: create_default_nebuchadnezzar_config(),
            trebuchet_config: create_default_trebuchet_config(),
            sighthound_config: create_default_sighthound_config(),
            zangalewa_config: create_default_zangalewa_config(),
            turbulance_config: TurbulanceConfig {
                semantic_processing: true,
                cross_modal_analysis: true,
                proposition_reasoning: true,
                motion_tracking: true,
                evidence_integration: true,
            },
            orchestrator_config: OrchestratorConfig {
                enable_metacognition: true,
                learning_enabled: true,
                auto_optimization: true,
                parallel_execution: true,
                resource_management: true,
            },
        }
    }
}

/// Multi-modal semantic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticOperation {
    TextAnalysis {
        text: String,
        analysis_type: String,
    },
    VideoAnalysis {
        video_path: PathBuf,
        analysis_options: HashMap<String, serde_json::Value>,
    },
    GeospatialAnalysis {
        data_sources: Vec<sighthound::DataSource>,
        pipeline: sighthound::ProcessingPipeline,
    },
    CodeExecution {
        code_files: Vec<zangalewa::CodeFile>,
        execution_options: zangalewa::ExecutionOptions,
    },
}

/// Results from semantic operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SemanticResult {
    TextResult(serde_json::Value),
    VideoResult(space_computer::ProcessingResults),
    GeospatialResult(sighthound::ProcessingResults),
    CodeResult(zangalewa::ExecutionSession),
}

/// Overall processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub session_id: Uuid,
    pub orchestrator_result: serde_json::Value,
    pub module_results: HashMap<String, serde_json::Value>,
    pub execution_time_ms: u64,
    pub quality_score: f64,
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub framework_id: Uuid,
    pub session_id: Uuid,
    pub uptime_seconds: u64,
    pub modules_status: ModulesStatus,
    pub performance_metrics: PerformanceMetrics,
    pub active_contexts: usize,
    pub workspace_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModulesStatus {
    pub fullscreen_active: bool,
    pub harare_active: bool,
    pub spectacular_active: bool,
    pub nebuchadnezzar_active: bool,
    pub trebuchet_active: bool,
    pub sighthound_active: bool,
    pub zangalewa_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_requests_processed: u64,
    pub average_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub resource_utilization_percent: f64,
}

// Helper functions for creating default configurations
fn create_default_nebuchadnezzar_config() -> nebuchadnezzar::AssistantConfig {
    nebuchadnezzar::AssistantConfig {
        model_config: nebuchadnezzar::ModelConfig {
            primary_model: "gpt-4".to_string(),
            fallback_models: vec!["gpt-3.5-turbo".to_string()],
            model_parameters: HashMap::new(),
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        },
        invocation_settings: nebuchadnezzar::InvocationSettings {
            trigger_keywords: vec!["help".to_string(), "fix".to_string(), "explain".to_string()],
            trigger_patterns: vec!["error".to_string(), "exception".to_string()],
            auto_invocation_contexts: vec!["compilation_error".to_string()],
            manual_invocation_key: "F1".to_string(),
            context_sensitivity: 0.8,
            response_delay_ms: 100,
        },
        context_window_size: 8192,
        response_formatting: nebuchadnezzar::ResponseFormatting {
            output_format: nebuchadnezzar::OutputFormat::CodeBlock,
            code_syntax_highlighting: true,
            include_explanations: true,
            include_alternatives: true,
            confidence_indicators: true,
            orchestrator_metadata: true,
        },
        orchestrator_constraints: nebuchadnezzar::OrchestratorConstraints {
            allowed_operations: vec!["code_analysis".to_string(), "error_fixing".to_string()],
            forbidden_operations: vec!["file_deletion".to_string(), "system_access".to_string()],
            resource_limits: nebuchadnezzar::ResourceLimits {
                max_execution_time_ms: 30000,
                max_memory_usage_mb: 1024,
                max_file_operations: 10,
                max_network_requests: 5,
                max_concurrent_invocations: 3,
            },
            scope_restrictions: vec!["project_directory".to_string()],
            approval_required_actions: vec!["major_refactoring".to_string()],
        },
        learning_settings: nebuchadnezzar::LearningSettings {
            enable_continuous_learning: true,
            user_preference_tracking: true,
            code_pattern_learning: true,
            domain_specific_adaptation: true,
            feedback_integration: true,
        },
    }
}

fn create_default_trebuchet_config() -> trebuchet::TrebuchetConfig {
    trebuchet::TrebuchetConfig {
        deployment_config: trebuchet::DeploymentConfig {
            auto_scaling: true,
            min_instances: 1,
            max_instances: 10,
            deployment_strategy: trebuchet::DeploymentStrategy::RollingUpdate,
            rollback_enabled: true,
            health_check_interval_ms: 30000,
            startup_timeout_ms: 60000,
        },
        performance_config: trebuchet::PerformanceConfig {
            target_latency_ms: 100,
            max_throughput_rps: 1000,
            cpu_optimization: true,
            memory_optimization: true,
            cache_optimization: true,
            connection_pooling: true,
        },
        networking_config: trebuchet::NetworkingConfig {
            service_mesh_enabled: true,
            load_balancing_algorithm: trebuchet::LoadBalancingAlgorithm::PerformanceBased,
            circuit_breaker_enabled: true,
            retry_policy: trebuchet::RetryPolicy {
                max_retries: 3,
                backoff_strategy: trebuchet::BackoffStrategy::Exponential,
                retry_conditions: vec!["timeout".to_string(), "server_error".to_string()],
            },
            timeout_policy: trebuchet::TimeoutPolicy {
                connection_timeout_ms: 5000,
                request_timeout_ms: 30000,
                idle_timeout_ms: 60000,
            },
        },
        security_config: trebuchet::SecurityConfig {
            authentication_enabled: true,
            authorization_enabled: true,
            encryption_in_transit: true,
            encryption_at_rest: true,
            rate_limiting: trebuchet::RateLimitConfig {
                requests_per_second: 100,
                burst_capacity: 200,
                rate_limit_algorithm: trebuchet::RateLimitAlgorithm::TokenBucket,
            },
            firewall_rules: Vec::new(),
        },
        monitoring_config: trebuchet::MonitoringConfig {
            metrics_collection_interval_ms: 1000,
            log_level: trebuchet::LogLevel::Info,
            distributed_tracing: true,
            performance_profiling: true,
            anomaly_detection: true,
            alerting_config: trebuchet::AlertingConfig {
                alert_channels: Vec::new(),
                alert_thresholds: HashMap::new(),
                escalation_policies: Vec::new(),
            },
        },
        resource_limits: trebuchet::GlobalResourceLimits {
            max_total_cpu_cores: 16,
            max_total_memory_gb: 32,
            max_total_disk_gb: 100,
            max_network_bandwidth_gbps: 1.0,
            max_concurrent_connections: 10000,
        },
    }
}

fn create_default_sighthound_config() -> sighthound::SighthoundConfig {
    sighthound::SighthoundConfig {
        data_sources: sighthound::DataSourceConfig {
            supported_formats: vec!["gpx".to_string(), "kml".to_string(), "tcx".to_string(), "fit".to_string()],
            quality_thresholds: sighthound::QualityThresholds {
                min_accuracy_meters: 5.0,
                max_speed_mps: 100.0,
                max_acceleration_mps2: 50.0,
                min_satellite_count: 4,
                max_hdop: 10.0,
            },
            interpolation_settings: sighthound::InterpolationSettings {
                method: sighthound::InterpolationMethod::Kalman,
                max_gap_seconds: 30,
                smoothing_factor: 0.5,
                temporal_resolution_ms: 1000,
            },
            fusion_weights: HashMap::new(),
        },
        processing_config: sighthound::ProcessingConfig {
            enable_filtering: true,
            enable_smoothing: true,
            enable_outlier_detection: true,
            enable_path_optimization: true,
            enable_altitude_correction: true,
            coordinate_system: sighthound::CoordinateSystem::WGS84,
        },
        accuracy_config: sighthound::AccuracyConfig {
            target_horizontal_accuracy_m: 2.0,
            target_vertical_accuracy_m: 5.0,
            confidence_level: 0.95,
            error_modeling: sighthound::ErrorModelingConfig {
                model_systematic_errors: true,
                model_random_errors: true,
                model_multipath_effects: true,
                model_atmospheric_delays: true,
            },
        },
        output_config: sighthound::OutputConfig {
            output_formats: vec![sighthound::OutputFormat::GeoJson, sighthound::OutputFormat::Czml],
            precision_digits: 6,
            include_metadata: true,
            include_quality_indicators: true,
            compression_enabled: true,
        },
        performance_config: sighthound::PerformanceConfig {
            parallel_processing: true,
            max_worker_threads: 4,
            memory_limit_mb: 4096,
            enable_gpu_acceleration: false,
            cache_size_mb: 1024,
        },
        satellite_config: sighthound::SatelliteConfig {
            tle_sources: vec!["https://celestrak.com/NORAD/elements/active.txt".to_string()],
            update_interval_hours: 24,
            prediction_horizon_days: 7,
            orbital_models: vec![sighthound::OrbitalModel::SGP4],
        },
    }
}

fn create_default_zangalewa_config() -> zangalewa::ZangalewaConfig {
    zangalewa::ZangalewaConfig {
        execution_config: zangalewa::ExecutionConfig {
            max_execution_time_ms: 60000,
            max_memory_mb: 2048,
            max_cpu_cores: 4,
            sandboxing_enabled: true,
            network_access_allowed: false,
            file_system_access: zangalewa::FileSystemAccess::ProjectDirectory,
            environment_isolation: true,
        },
        analysis_config: zangalewa::AnalysisConfig {
            static_analysis_enabled: true,
            dynamic_analysis_enabled: true,
            semantic_analysis_depth: zangalewa::AnalysisDepth::Deep,
            code_quality_checks: true,
            dependency_analysis: true,
            security_analysis: true,
            performance_analysis: true,
        },
        error_handling_config: zangalewa::ErrorHandlingConfig {
            auto_recovery_enabled: true,
            max_auto_fix_attempts: 3,
            learning_from_errors: true,
            error_pattern_detection: true,
            intelligent_suggestions: true,
            context_aware_fixes: true,
        },
        security_config: zangalewa::SecurityConfig {
            vulnerability_scanning: true,
            code_injection_detection: true,
            dependency_security_check: true,
            secrets_detection: true,
            permission_analysis: true,
            sandbox_escape_detection: true,
        },
        performance_config: zangalewa::PerformanceConfig {
            profiling_enabled: true,
            memory_tracking: true,
            cpu_profiling: true,
            io_monitoring: true,
            network_monitoring: true,
            bottleneck_detection: true,
        },
        language_configs: HashMap::new(),
    }
} 