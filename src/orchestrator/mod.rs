// Metacognitive Orchestrator Module
// Implements a streaming-based concurrent processing architecture

pub mod stream;
pub mod metacognitive;
pub mod biomimetic;
pub mod types;
pub mod examples;
pub mod goal;
pub mod context;
pub mod intervention;
pub mod config;

// Seven Intelligence Modules for Metacognitive Orchestrator
pub mod mzekezeke;     // Bayesian Learning Engine
pub mod diggiden;      // Adversarial System
pub mod hatata;        // Decision System
pub mod spectacular;   // Extraordinary Handler
pub mod nicotine;      // Context Validator
pub mod zengeza;       // Intelligent Noise Reduction Engine
pub mod diadochi;      // Multi-Domain LLM Orchestration Framework
pub mod integrated;    // Integrated Intelligence Orchestrator
pub mod intelligence_examples; // Examples and usage demonstrations

// Revolutionary Tres Commas Engine & V8 Metabolism Pipeline
pub mod tres_commas;   // Trinity-Based Cognitive Architecture
pub mod v8_metabolism; // V8 Biological Truth Processing Pipeline
pub mod clothesline;   // Comprehension Validator & Context Gatekeeper
pub mod pungwe;        // ATP Synthase & Understanding Gap Detector

// Re-export main components
pub use metacognitive::MetacognitiveOrchestrator;
pub use stream::{StreamProcessor, StreamPipeline};
pub use types::StreamData;
pub use examples::{create_default_orchestrator, create_custom_orchestrator};

// Re-export intelligence modules
pub use mzekezeke::MzekezkeBayesianEngine;
pub use diggiden::DiggidenAdversarialSystem;
pub use hatata::HatataDecisionSystem;
pub use spectacular::SpectacularHandler;
pub use nicotine::NicotineContextValidator;
pub use zengeza::ZengezaNoiseReduction;
pub use diadochi::DiadochiFramework;
pub use integrated::IntegratedMetacognitiveOrchestrator;
pub use intelligence_examples::run_all_intelligence_examples;

// Re-export revolutionary Tres Commas Engine components
pub use tres_commas::{TresCommasEngine, ConsciousnessLayer, CognitiveProcess, RespirationState, demonstrate_tres_commas};

// Re-export main types for easy access
pub use goal::{Goal, GoalType, GoalMetrics, SuccessCriterion, Strategy, GoalStatus};
pub use intervention::{InterventionSystem, InterventionType, ActiveIntervention, InterventionOutcome};
pub use context::{Context, WritingPattern};
pub use config::KwasaConfig as Config;
pub use stream::{StreamPipeline as ProcessingPipeline};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use log::{info, warn, debug};
use crate::turbulance::ast::Value;
use crate::knowledge::KnowledgeDatabase;
use crate::text_unit::{TextUnit, TextUnitRegistry, TextUnitId, TextUnitType};
use intervention::Intervention;

/// Represents the Metacognitive Orchestrator that manages the writing process
pub struct Orchestrator {
    /// The current goal the user is working towards
    goal: Goal,
    
    /// The current context of the writing
    context: Arc<Mutex<Context>>,
    
    /// The text units registry
    units: Arc<Mutex<TextUnitRegistry>>,
    
    /// Connection to the knowledge database
    knowledge_db: Arc<Mutex<Result<KnowledgeDatabase, crate::error::Error>>>,
    
    /// Available interventions
    interventions: Vec<Box<dyn Intervention>>,
    
    /// Variables storage
    variables: HashMap<String, Value>,
    
    /// Tracking state
    is_tracking: bool,
    
    /// Current processing configuration
    config: Config,
    
    /// Processing pipelines
    pipelines: HashMap<String, StreamPipeline>,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(goal: Goal, knowledge_db: Arc<Mutex<Result<KnowledgeDatabase, crate::error::Error>>>) -> Self {
        let context = Arc::new(Mutex::new(Context::new()));
        let units = Arc::new(Mutex::new(TextUnitRegistry::new()));
        
        // Built-in interventions
        let interventions: Vec<Box<dyn Intervention>> = vec![
            Box::new(intervention::ReadabilityIntervention::new()),
            Box::new(intervention::CoherenceIntervention::new()),
            Box::new(intervention::ResearchIntervention::new()),
        ];
        
        Self {
            goal,
            context,
            units,
            knowledge_db,
            interventions,
            variables: HashMap::new(),
            is_tracking: false,
            config: Config::default(),
            pipelines: HashMap::new(),
        }
    }
    
    /// Start tracking the writing process
    pub fn start_tracking(&mut self) {
        info!("Orchestrator started tracking");
        self.is_tracking = true;
    }
    
    /// Stop tracking the writing process
    pub fn stop_tracking(&mut self) {
        info!("Orchestrator stopped tracking");
        self.is_tracking = false;
    }
    
    /// Set a new goal
    pub fn set_goal(&mut self, goal: Goal) {
        info!("Setting new goal: {}", goal.description());
        self.goal = goal;
        self.context.lock().unwrap().update_goal_state(&self.goal);
    }
    
    /// Get the current goal
    pub fn goal(&self) -> &Goal {
        &self.goal
    }
    
    /// Add a text unit to the orchestrator
    pub fn add_text_unit(&self, unit: TextUnit) -> TextUnitId {
        let mut units = self.units.lock().unwrap();
        let id = units.add_unit(unit);
        
        // Update the context with the new unit
        drop(units); // Release the lock
        self.update_context_with_unit(id);
        
        id
    }
    
    /// Get a text unit by ID
    pub fn get_text_unit(&self, id: TextUnitId) -> Option<TextUnit> {
        let units = self.units.lock().unwrap();
        units.get_unit(id).cloned()
    }
    
    /// Update the context with a text unit
    fn update_context_with_unit(&self, unit_id: TextUnitId) {
        let units = self.units.lock().unwrap();
        
        if let Some(unit) = units.get_unit(unit_id) {
            // Add relevant keywords from the unit to the context
            let keywords = extract_keywords(&unit.content);
            for keyword in keywords {
                self.context.lock().unwrap().add_keyword(keyword);
            }
            
            // Check if unit relates to goal
            if self.goal.is_relevant(&unit.content) {
                debug!("Text unit {} is relevant to the goal", unit_id);
            }
        }
    }
    
    /// Process text with interventions
    pub fn process_text(&self, text: &str) -> String {
        if !self.is_tracking {
            return text.to_string();
        }
        
        let mut result = text.to_string();
        
        // Add the text as a unit
        let unit = TextUnit::new(
            text.to_string(),
            0,
            text.len(),
            TextUnitType::Paragraph,
            0,
        );
        
        let unit_id = self.add_text_unit(unit);
        
        // Let each intervention process the text
        for intervention in &self.interventions {
            if intervention.should_intervene(&self.goal, &self.context.lock().unwrap()) {
                match intervention.process_text(text, &self.goal, &self.context.lock().unwrap()) {
                    Ok(processed) => {
                        if processed != text {
                            info!("Intervention {} modified the text", intervention.name());
                            result = processed;
                        }
                    },
                    Err(e) => {
                        warn!("Intervention {} failed: {}", intervention.name(), e);
                    }
                }
            }
        }
        
        result
    }
    
    /// Add a variable
    pub fn set_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }
    
    /// Get a variable
    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }
    
    /// Perform research on a topic
    pub fn research(&self, topic: &str) -> Vec<String> {
        info!("Researching topic: {}", topic);
        
        // Add to context
        self.context.lock().unwrap().add_research_term(topic);
        
        // Query knowledge DB
        let db_result = self.knowledge_db.lock().unwrap();
        match db_result.as_ref() {
            Ok(db) => match db.search(topic) {
                Ok(results) => results,
                Err(_) => Vec::new(),
            },
            Err(_) => Vec::new(),
        }
    }
    
    /// Check if the text aligns with the goal
    pub fn evaluate_alignment(&self, text: &str) -> f64 {
        self.goal.evaluate_alignment(text)
    }
    
    /// Register a new intervention
    pub fn register_intervention(&mut self, intervention: Box<dyn Intervention>) {
        info!("Registering new intervention: {}", intervention.name());
        self.interventions.push(intervention);
    }
    
    /// Get current configuration
    pub fn config(&self) -> &Config {
        &self.config
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: Config) {
        self.config = config;
    }
    
    /// Add a processing pipeline
    pub fn add_pipeline(&mut self, name: String, pipeline: StreamPipeline) {
        self.pipelines.insert(name, pipeline);
    }
    
    /// Get a processing pipeline
    pub fn get_pipeline(&self, name: &str) -> Option<&StreamPipeline> {
        self.pipelines.get(name)
    }
    
    /// Get all pipeline names
    pub fn pipeline_names(&self) -> Vec<&String> {
        self.pipelines.keys().collect()
    }
}

impl Default for Orchestrator {
    fn default() -> Self {
        let temp_path = std::env::temp_dir().join("kwasa_default.db");
        let db_result = crate::knowledge::KnowledgeDatabase::new(&temp_path);
        Self::new(Goal::new("Default Goal", 0.5), Arc::new(Mutex::new(db_result)))
    }
}

/// Extract potential keywords from text
fn extract_keywords(text: &str) -> Vec<String> {
    // This is a very simple implementation
    // In a real system, we'd use NLP techniques
    
    let stop_words = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "from", "of", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "must", "that", "this", "these", "those",
    ];
    
    text.split_whitespace()
        .map(|word| word.to_lowercase())
        .filter(|word| {
            word.len() > 3 && !stop_words.contains(&word.as_str())
        })
        .map(|word| {
            // Remove punctuation
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::tests::create_test_db;
    
    #[test]
    fn test_orchestrator_basic() {
        let db = Arc::new(Mutex::new(Ok(create_test_db())));
        let goal = Goal::new("Write about machine learning", 0.5);
        
        let orchestrator = Orchestrator::new(goal, db);
        
        assert_eq!(orchestrator.goal().description(), "Write about machine learning");
        assert!(!orchestrator.is_tracking);
    }
    
    #[test]
    fn test_extract_keywords() {
        let text = "The quick brown fox jumps over the lazy dog";
        let keywords = extract_keywords(text);
        
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"brown".to_string()));
        assert!(keywords.contains(&"jumps".to_string()));
        assert!(keywords.contains(&"lazy".to_string()));
        
        // Should not contain stop words
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"over".to_string()));
    }
    
    #[test]
    fn test_orchestrator_variables() {
        let db = Arc::new(Mutex::new(Ok(create_test_db())));
        let goal = Goal::new("Test variables", 0.5);
        
        let mut orchestrator = Orchestrator::new(goal, db);
        
        // Set variables
        orchestrator.set_variable("test", Value::String("value".to_string()));
        orchestrator.set_variable("number", Value::Number(42.0));
        
        // Get variables
        assert_eq!(
            orchestrator.get_variable("test"),
            Some(&Value::String("value".to_string()))
        );
        
        assert_eq!(
            orchestrator.get_variable("number"),
            Some(&Value::Number(42.0))
        );
        
        // Non-existent variable
        assert_eq!(orchestrator.get_variable("nonexistent"), None);
    }
} 