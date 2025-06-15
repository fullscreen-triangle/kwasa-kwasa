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