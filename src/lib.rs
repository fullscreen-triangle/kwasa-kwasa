//! Kwasa-Kwasa: A metacognitive text processing framework with Turbulance DSL language
//! 
//! This library provides:
//! - Turbulance DSL language implementation (lexer, parser, interpreter)
//! - Advanced text unit processing and boundary detection
//! - Intelligent orchestration with metacognitive processing
//! - Goal-oriented writing assistance with intervention systems
//! - Stream processing capabilities
//! - Comprehensive configuration management
//! - Optional integration with specialized external modules
//! 
//! # Quick Start
//! 
//! ```rust
//! use kwasa_kwasa::{KwasaFramework, FrameworkConfig};
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = FrameworkConfig::default();
//! let mut framework = KwasaFramework::new(config).await?;
//! 
//! // Process text with Turbulance DSL
//! let result = framework.process_turbulance_code("funxn analyze(text): return process(text)").await?;
//! 
//! // Process text with framework
//! let result = framework.process_text("Your text here", None).await?;
//! 
//! // Set a writing goal
//! let goal_id = framework.set_goal("Write a 1000-word essay about AI", None, Some(0.8)).await?;
//! 
//! // Get writing assistance
//! let suggestions = framework.get_suggestions().await?;
//! # Ok(())
//! # }
//! ```

// Core Framework Modules
pub mod turbulance;      // DSL language implementation
pub mod text_unit;       // Text processing framework
pub mod orchestrator;    // Core orchestration framework
pub mod knowledge;       // Knowledge management
pub mod cli;             // Command line interface
pub mod utils;           // Framework utilities
pub mod pattern;         // Pattern processing
pub mod evidence;        // Evidence framework
pub mod error;           // Error handling
pub mod templates;       // Framework templates
pub mod external_apis;   // External API integration
pub mod visualization;   // Basic visualization
pub mod wasm;            // WebAssembly support

// External Module Integrations (conditionally compiled)
#[cfg(feature = "meta-orchestrator")]
pub use kwasa_meta_orchestrator as meta_orchestrator;

#[cfg(feature = "cheminformatics")]
pub use kwasa_cheminformatics as chemistry;

#[cfg(feature = "systems-biology")]
pub use kwasa_systems_biology as genomic;

#[cfg(feature = "spectrometry")]
pub use kwasa_spectrometry as spectrometry;

#[cfg(feature = "multimedia")]
pub use kwasa_multimedia as multimedia;

#[cfg(feature = "specialized-modules")]
pub use kwasa_specialized_modules as specialized;

// Re-export core framework components
pub use orchestrator::Orchestrator;
pub use text_unit::{TextUnit, TextUnitId, TextUnitType, TextUnitRegistry, BoundaryDetectionOptions};
pub use error::{Error, Result, ErrorReporter};
pub use turbulance::{run as run_turbulance, run_with_context, validate as validate_turbulance};

// Import orchestrator types
pub use orchestrator::{
    Goal, GoalType, GoalMetrics, SuccessCriterion, Strategy, GoalStatus,
    Context, WritingPattern,
    InterventionSystem, InterventionType, ActiveIntervention, InterventionOutcome,
    Config as OrchestratorConfig,
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::error::{Error, Result, ErrorReporter};
    pub use crate::{KwasaFramework, FrameworkConfig, Goal, Context, InterventionSystem};
    pub use crate::turbulance::{run as run_turbulance, validate as validate_turbulance};
}

use std::sync::Arc;
use tokio::sync::Mutex;
use log::info;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Main framework that coordinates all Kwasa-Kwasa functionality
pub struct KwasaFramework {
    /// Framework configuration
    config: FrameworkConfig,
    
    /// The core orchestrator that manages processing
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
    
    /// Framework state
    state: FrameworkState,
    
    /// Session ID
    session_id: Uuid,
    
    /// External module integrations
    #[cfg(feature = "meta-orchestrator")]
    meta_orchestrator: Option<Arc<Mutex<kwasa_meta_orchestrator::MetaOrchestrator>>>,
    
    #[cfg(feature = "cheminformatics")]
    chemistry_engine: Option<Arc<Mutex<kwasa_cheminformatics::ChemistryEngine>>>,
    
    #[cfg(feature = "systems-biology")]
    biology_engine: Option<Arc<Mutex<kwasa_systems_biology::BiologyEngine>>>,
    
    #[cfg(feature = "spectrometry")]
    spectrometry_engine: Option<Arc<Mutex<kwasa_spectrometry::SpectrometryEngine>>>,
    
    #[cfg(feature = "multimedia")]
    multimedia_engine: Option<Arc<Mutex<kwasa_multimedia::MultimediaEngine>>>,
    
    #[cfg(feature = "specialized-modules")]
    specialized_modules: Option<Arc<Mutex<kwasa_specialized_modules::SpecializedModules>>>,
}

/// Framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    /// Workspace path
    pub workspace_path: PathBuf,
    
    /// Enable framework core features
    pub enable_core_features: bool,
    
    /// Core orchestrator configuration
    pub orchestrator_config: OrchestratorConfig,
    
    /// Turbulance DSL configuration
    pub turbulance_config: TurbulanceConfig,
    
    /// External module configurations
    #[cfg(feature = "meta-orchestrator")]
    pub meta_orchestrator_config: Option<kwasa_meta_orchestrator::Config>,
    
    #[cfg(feature = "cheminformatics")]
    pub chemistry_config: Option<kwasa_cheminformatics::Config>,
    
    #[cfg(feature = "systems-biology")]
    pub biology_config: Option<kwasa_systems_biology::Config>,
    
    #[cfg(feature = "spectrometry")]
    pub spectrometry_config: Option<kwasa_spectrometry::Config>,
    
    #[cfg(feature = "multimedia")]
    pub multimedia_config: Option<kwasa_multimedia::Config>,
    
    #[cfg(feature = "specialized-modules")]
    pub specialized_config: Option<kwasa_specialized_modules::Config>,
}

/// Turbulance DSL configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulanceConfig {
    /// Enable semantic processing
    pub semantic_processing: bool,
    
    /// Enable cross-modal analysis
    pub cross_modal_analysis: bool,
    
    /// Enable proposition reasoning
    pub proposition_reasoning: bool,
    
    /// Enable evidence integration
    pub evidence_integration: bool,
    
    /// Maximum execution time for Turbulance code (seconds)
    pub max_execution_time: u64,
}

/// Current state of the framework
#[derive(Debug, Clone, PartialEq)]
pub enum FrameworkState {
    /// Framework is initializing
    Initializing,
    
    /// Framework is ready for use
    Ready,
    
    /// Framework is actively processing
    Processing,
    
    /// Framework is paused
    Paused,
    
    /// Framework encountered an error
    Error(String),
    
    /// Framework is shutting down
    Shutdown,
}

/// Result type for framework operations
pub type FrameworkResult<T> = Result<T>;

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
    pub domain_options: HashMap<String, String>,
}

/// Result of text processing
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Session ID
    pub session_id: Uuid,
    
    /// Processed text units
    pub units: Vec<TextUnit>,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Semantic analysis
    pub semantic_analysis: SemanticAnalysis,
    
    /// Goal alignment scores
    pub goal_alignments: Vec<GoalAlignment>,
    
    /// Suggested interventions
    pub interventions: Vec<InterventionSuggestion>,
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
    
    /// External module results (if enabled)
    pub external_results: HashMap<String, serde_json::Value>,
}

/// Quality metrics
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

/// Sentiment information
#[derive(Debug, Clone)]
pub struct Sentiment {
    /// Sentiment polarity (-1.0 to 1.0)
    pub polarity: f64,
    
    /// Confidence in sentiment analysis (0.0-1.0)
    pub confidence: f64,
    
    /// Sentiment label
    pub label: String,
}

/// Concept relationship
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

/// Goal alignment information
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
    pub additional: HashMap<String, String>,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            enable_boundaries: true,
            enable_quality: true,
            enable_semantic: true,
            enable_goal_alignment: true,
            enable_interventions: true,
            max_processing_time: 30,
            domain_options: HashMap::new(),
        }
    }
}

impl Default for TurbulanceConfig {
    fn default() -> Self {
        Self {
            semantic_processing: true,
            cross_modal_analysis: false,
            proposition_reasoning: true,
            evidence_integration: true,
            max_execution_time: 30,
        }
    }
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            workspace_path: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            enable_core_features: true,
            orchestrator_config: OrchestratorConfig::default(),
            turbulance_config: TurbulanceConfig::default(),
            
            #[cfg(feature = "meta-orchestrator")]
            meta_orchestrator_config: None,
            
            #[cfg(feature = "cheminformatics")]
            chemistry_config: None,
            
            #[cfg(feature = "systems-biology")]
            biology_config: None,
            
            #[cfg(feature = "spectrometry")]
            spectrometry_config: None,
            
            #[cfg(feature = "multimedia")]
            multimedia_config: None,
            
            #[cfg(feature = "specialized-modules")]
            specialized_config: None,
        }
    }
}

impl KwasaFramework {
    /// Create a new framework instance
    pub async fn new(config: FrameworkConfig) -> FrameworkResult<Self> {
        let session_id = Uuid::new_v4();
        
        info!("Initializing Kwasa-Kwasa Framework with session ID: {}", session_id);
        
        // Initialize knowledge database
        let knowledge_db = Arc::new(Mutex::new(
            knowledge::database::KnowledgeDatabase::new(&config.workspace_path.join("knowledge.db"))
                .await.map_err(|e| anyhow::anyhow!("Failed to initialize knowledge database: {}", e))
        ));
        
        // Create default goal
        let default_goal = Goal::new("General text processing", GoalType::Quality, 0.7);
        
        // Initialize orchestrator
        let orchestrator = Arc::new(Mutex::new(
            Orchestrator::new(default_goal, knowledge_db.clone())
        ));
        
        // Initialize other components
        let text_registry = Arc::new(Mutex::new(TextUnitRegistry::new()));
        let context = Arc::new(Mutex::new(Context::new()));
        let intervention_system = Arc::new(Mutex::new(InterventionSystem::new()));
        let goals = Arc::new(Mutex::new(Vec::new()));
        
        let mut framework = Self {
            config,
            orchestrator,
            text_registry,
            context,
            intervention_system,
            goals,
            knowledge_db,
            state: FrameworkState::Initializing,
            session_id,
            
            #[cfg(feature = "meta-orchestrator")]
            meta_orchestrator: None,
            
            #[cfg(feature = "cheminformatics")]
            chemistry_engine: None,
            
            #[cfg(feature = "systems-biology")]
            biology_engine: None,
            
            #[cfg(feature = "spectrometry")]
            spectrometry_engine: None,
            
            #[cfg(feature = "multimedia")]
            multimedia_engine: None,
            
            #[cfg(feature = "specialized-modules")]
            specialized_modules: None,
        };
        
        // Initialize external modules if configured
        framework.initialize_external_modules().await?;
        
        framework.state = FrameworkState::Ready;
        
        info!("Kwasa-Kwasa Framework initialized successfully");
        
        Ok(framework)
    }
    
    /// Initialize external modules based on configuration
    async fn initialize_external_modules(&mut self) -> FrameworkResult<()> {
        #[cfg(feature = "meta-orchestrator")]
        if let Some(config) = &self.config.meta_orchestrator_config {
            self.meta_orchestrator = Some(Arc::new(Mutex::new(
                kwasa_meta_orchestrator::MetaOrchestrator::new(config.clone()).await?
            )));
            info!("Meta-orchestrator module initialized");
        }
        
        #[cfg(feature = "cheminformatics")]
        if let Some(config) = &self.config.chemistry_config {
            self.chemistry_engine = Some(Arc::new(Mutex::new(
                kwasa_cheminformatics::ChemistryEngine::new(config.clone()).await?
            )));
            info!("Cheminformatics module initialized");
        }
        
        #[cfg(feature = "systems-biology")]
        if let Some(config) = &self.config.biology_config {
            self.biology_engine = Some(Arc::new(Mutex::new(
                kwasa_systems_biology::BiologyEngine::new(config.clone()).await?
            )));
            info!("Systems biology module initialized");
        }
        
        #[cfg(feature = "spectrometry")]
        if let Some(config) = &self.config.spectrometry_config {
            self.spectrometry_engine = Some(Arc::new(Mutex::new(
                kwasa_spectrometry::SpectrometryEngine::new(config.clone()).await?
            )));
            info!("Spectrometry module initialized");
        }
        
        #[cfg(feature = "multimedia")]
        if let Some(config) = &self.config.multimedia_config {
            self.multimedia_engine = Some(Arc::new(Mutex::new(
                kwasa_multimedia::MultimediaEngine::new(config.clone()).await?
            )));
            info!("Multimedia module initialized");
        }
        
        #[cfg(feature = "specialized-modules")]
        if let Some(config) = &self.config.specialized_config {
            self.specialized_modules = Some(Arc::new(Mutex::new(
                kwasa_specialized_modules::SpecializedModules::new(config.clone()).await?
            )));
            info!("Specialized modules initialized");
        }
        
        Ok(())
    }
    
    /// Create a framework with default configuration
    pub async fn with_defaults() -> FrameworkResult<Self> {
        Self::new(FrameworkConfig::default()).await
    }
    
    /// Process Turbulance DSL code
    pub async fn process_turbulance_code(&mut self, code: &str) -> FrameworkResult<String> {
        self.state = FrameworkState::Processing;
        
        let start_time = std::time::Instant::now();
        
        // Create a context for execution
        let mut context = turbulance::Context::new();
        
        // Run the Turbulance code
        let result = turbulance::run_with_context(code, &mut context)
            .map_err(|e| anyhow::anyhow!("Turbulance execution error: {}", e))?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        info!("Turbulance code processed in {}ms", processing_time);
        
        self.state = FrameworkState::Ready;
        
        Ok(result)
    }
    
    /// Process text with the framework
    pub async fn process_text(&mut self, text: &str, options: Option<ProcessingOptions>) -> FrameworkResult<ProcessingResult> {
        self.state = FrameworkState::Processing;
        
        let start_time = std::time::Instant::now();
        let options = options.unwrap_or_default();
        
        // Create text units
        let mut units = Vec::new();
        let text_unit = TextUnit::new(text.to_string(), TextUnitType::Paragraph, HashMap::new());
        units.push(text_unit);
        
        // Analyze quality
        let quality_metrics = self.analyze_quality(text).await?;
        
        // Perform semantic analysis
        let semantic_analysis = self.analyze_semantics(text).await?;
        
        // Check goal alignment
        let goal_alignments = self.check_goal_alignment(text).await?;
        
        // Get intervention suggestions
        let interventions = self.get_intervention_suggestions(text, &quality_metrics).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Collect external module results
        let mut external_results = HashMap::new();
        
        #[cfg(feature = "cheminformatics")]
        if let Some(engine) = &self.chemistry_engine {
            if let Ok(result) = engine.lock().await.analyze_text(text).await {
                external_results.insert("chemistry".to_string(), serde_json::to_value(result)?);
            }
        }
        
        #[cfg(feature = "systems-biology")]
        if let Some(engine) = &self.biology_engine {
            if let Ok(result) = engine.lock().await.analyze_text(text).await {
                external_results.insert("biology".to_string(), serde_json::to_value(result)?);
            }
        }
        
        let result = ProcessingResult {
            session_id: self.session_id,
            units,
            quality_metrics,
            semantic_analysis,
            goal_alignments,
            interventions,
            metadata: ProcessingMetadata {
                processing_time_ms: processing_time,
                units_processed: 1,
                pipeline: "default".to_string(),
                warnings: Vec::new(),
                additional: HashMap::new(),
            },
            external_results,
        };
        
        self.state = FrameworkState::Ready;
        
        Ok(result)
    }
    
    /// Set a goal for the framework
    pub async fn set_goal(&mut self, description: &str, goal_type: Option<GoalType>, priority: Option<f64>) -> FrameworkResult<String> {
        let goal = Goal::new(
            description,
            goal_type.unwrap_or(GoalType::Quality),
            priority.unwrap_or(0.7)
        );
        
        let goal_id = goal.id().to_string();
        
        // Set goal in orchestrator
        self.orchestrator.lock().await.set_goal(goal.clone());
        
        // Add to goals list
        self.goals.lock().await.push(goal);
        
        Ok(goal_id)
    }
    
    /// Get current goals
    pub async fn get_goals(&self) -> FrameworkResult<Vec<Goal>> {
        Ok(self.goals.lock().await.clone())
    }
    
    /// Get writing suggestions
    pub async fn get_suggestions(&self) -> FrameworkResult<Vec<String>> {
        // This would integrate with the intervention system
        // For now, return placeholder suggestions
        Ok(vec![
            "Consider improving readability by shortening sentences".to_string(),
            "Add more concrete examples to support your points".to_string(),
            "Ensure logical flow between paragraphs".to_string(),
        ])
    }
    
    /// Get the current framework state
    pub fn get_state(&self) -> FrameworkState {
        self.state.clone()
    }
    
    /// Get the session ID
    pub fn session_id(&self) -> Uuid {
        self.session_id
    }
    
    /// Shutdown the framework
    pub async fn shutdown(&mut self) -> FrameworkResult<()> {
        self.state = FrameworkState::Shutdown;
        
        // Shutdown external modules
        #[cfg(feature = "meta-orchestrator")]
        if let Some(orchestrator) = &self.meta_orchestrator {
            orchestrator.lock().await.shutdown().await?;
        }
        
        info!("Kwasa-Kwasa Framework shutdown complete");
        
        Ok(())
    }
    
    // Private helper methods
    async fn analyze_quality(&self, text: &str) -> FrameworkResult<QualityMetrics> {
        // Placeholder quality analysis
        // In a real implementation, this would use NLP libraries
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split('.').count();
        let avg_sentence_length = if sentence_count > 0 { 
            word_count as f64 / sentence_count as f64 
        } else { 
            0.0 
        };
        
        let readability = if avg_sentence_length < 20.0 { 0.8 } else { 0.6 };
        let coherence = 0.7; // Placeholder
        let grammar = 0.85; // Placeholder
        let style = 0.75; // Placeholder
        let overall = (readability + coherence + grammar + style) / 4.0;
        
        Ok(QualityMetrics {
            readability,
            coherence,
            grammar,
            style,
            overall,
        })
    }
    
    async fn analyze_semantics(&self, text: &str) -> FrameworkResult<SemanticAnalysis> {
        // Placeholder semantic analysis
        let words: Vec<&str> = text.split_whitespace().collect();
        let topics = vec!["general".to_string()]; // Placeholder
        let key_phrases = words.iter()
            .filter(|&&word| word.len() > 5)
            .take(5)
            .map(|&word| word.to_string())
            .collect();
        
        Ok(SemanticAnalysis {
            topics,
            sentiment: Some(Sentiment {
                polarity: 0.1,
                confidence: 0.6,
                label: "neutral".to_string(),
            }),
            key_phrases,
            concept_relations: Vec::new(),
        })
    }
    
    async fn check_goal_alignment(&self, text: &str) -> FrameworkResult<Vec<GoalAlignment>> {
        let goals = self.goals.lock().await;
        let mut alignments = Vec::new();
        
        for goal in goals.iter() {
            // Placeholder alignment calculation
            let score = 0.7; // Would be calculated based on goal criteria
            
            alignments.push(GoalAlignment {
                goal_id: goal.id().to_string(),
                score,
                explanation: format!("Text alignment with goal: {}", goal.description()),
                suggestions: vec!["Consider adding more specific examples".to_string()],
            });
        }
        
        Ok(alignments)
    }
    
    async fn get_intervention_suggestions(&self, text: &str, quality: &QualityMetrics) -> FrameworkResult<Vec<InterventionSuggestion>> {
        let mut suggestions = Vec::new();
        
        if quality.readability < 0.7 {
            suggestions.push(InterventionSuggestion {
                intervention_type: InterventionType::StyleImprovement,
                priority: 0.8,
                description: "Improve readability by simplifying sentences".to_string(),
                actions: vec![
                    "Break down complex sentences".to_string(),
                    "Use simpler vocabulary where appropriate".to_string(),
                ],
            });
        }
        
        if quality.coherence < 0.7 {
            suggestions.push(InterventionSuggestion {
                intervention_type: InterventionType::StructureImprovement,
                priority: 0.7,
                description: "Improve text coherence and flow".to_string(),
                actions: vec![
                    "Add transition sentences between paragraphs".to_string(),
                    "Ensure logical progression of ideas".to_string(),
                ],
            });
        }
        
        Ok(suggestions)
    }
}

// Compatibility re-exports for existing code
pub use KwasaFramework as KwasaSystem;
pub use FrameworkConfig as Config; 