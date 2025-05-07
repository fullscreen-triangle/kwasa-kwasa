use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::turbulance::interpreter::Interpreter;
use crate::turbulance::parser::Parser;
use crate::turbulance::lexer::Lexer;
use crate::text_unit::{TextUnit, TextUnitType};
use crate::orchestrator::Orchestrator;
use crate::orchestrator::goal::Goal;
use crate::knowledge::KnowledgeDatabase;
use std::sync::{Arc, Mutex};
use log::info;

/// Configuration for the Kwasa-Kwasa WebAssembly module
#[wasm_bindgen]
pub struct KwasaConfig {
    /// Whether to enable debug mode
    debug: bool,
    
    /// The goal description for orchestration
    goal_description: String,
    
    /// The relevance threshold for the goal
    relevance_threshold: f64,
}

#[wasm_bindgen]
impl KwasaConfig {
    /// Create a new configuration with default values
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            debug: false,
            goal_description: "General writing assistance".to_string(),
            relevance_threshold: 0.3,
        }
    }
    
    /// Enable or disable debug mode
    #[wasm_bindgen]
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    
    /// Set the goal description
    #[wasm_bindgen]
    pub fn set_goal(&mut self, goal: String) {
        self.goal_description = goal;
    }
    
    /// Set the relevance threshold
    #[wasm_bindgen]
    pub fn set_relevance_threshold(&mut self, threshold: f64) {
        self.relevance_threshold = threshold;
    }
}

/// Results from executing Turbulance code
#[derive(Serialize, Deserialize)]
pub struct ExecutionResult {
    /// The result value as a string
    pub result: String,
    
    /// Any error message that occurred
    pub error: Option<String>,
    
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// Interface for the Kwasa-Kwasa framework in WebAssembly
#[wasm_bindgen]
pub struct KwasaWasm {
    /// The Turbulance interpreter
    interpreter: Interpreter,
    
    /// The metacognitive orchestrator
    orchestrator: Orchestrator,
    
    /// Configuration
    config: KwasaConfig,
}

#[wasm_bindgen]
impl KwasaWasm {
    /// Create a new instance of the Kwasa-Kwasa framework
    #[wasm_bindgen(constructor)]
    pub fn new(config: KwasaConfig) -> Self {
        // Initialize console logging for wasm
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        
        let interpreter = Interpreter::new();
        
        // Create knowledge database
        let knowledge_db = Arc::new(Mutex::new(KnowledgeDatabase::new_in_memory().unwrap()));
        
        // Create goal
        let goal = Goal::new(&config.goal_description, config.relevance_threshold);
        
        // Create orchestrator
        let orchestrator = Orchestrator::new(goal, knowledge_db);
        
        info!("KwasaWasm initialized with goal: {}", config.goal_description);
        
        Self {
            interpreter,
            orchestrator,
            config,
        }
    }
    
    /// Execute Turbulance code
    #[wasm_bindgen]
    pub fn execute_code(&mut self, code: String) -> JsValue {
        let start_time = web_sys::window()
            .expect("no global window")
            .performance()
            .expect("no performance object")
            .now();
        
        // Parse and execute the code
        let mut lexer = Lexer::new(&code);
        let tokens = lexer.tokenize();
        
        match tokens {
            Ok(tokens) => {
                let mut parser = Parser::new(tokens);
                match parser.parse() {
                    Ok(ast) => {
                        match self.interpreter.evaluate(&ast) {
                            Ok(value) => {
                                let end_time = web_sys::window()
                                    .expect("no global window")
                                    .performance()
                                    .expect("no performance object")
                                    .now();
                                
                                let result = ExecutionResult {
                                    result: value.to_string(),
                                    error: None,
                                    execution_time_ms: end_time - start_time,
                                };
                                
                                return serde_wasm_bindgen::to_value(&result).unwrap();
                            },
                            Err(e) => {
                                let result = ExecutionResult {
                                    result: String::new(),
                                    error: Some(format!("Runtime error: {}", e)),
                                    execution_time_ms: 0.0,
                                };
                                
                                return serde_wasm_bindgen::to_value(&result).unwrap();
                            }
                        }
                    },
                    Err(e) => {
                        let result = ExecutionResult {
                            result: String::new(),
                            error: Some(format!("Parse error: {}", e)),
                            execution_time_ms: 0.0,
                        };
                        
                        return serde_wasm_bindgen::to_value(&result).unwrap();
                    }
                }
            },
            Err(e) => {
                let result = ExecutionResult {
                    result: String::new(),
                    error: Some(format!("Lexical error: {}", e)),
                    execution_time_ms: 0.0,
                };
                
                return serde_wasm_bindgen::to_value(&result).unwrap();
            }
        }
    }
    
    /// Process text with the metacognitive orchestrator
    #[wasm_bindgen]
    pub fn process_text(&self, text: String) -> String {
        self.orchestrator.process_text(&text)
    }
    
    /// Set a new goal for the orchestrator
    #[wasm_bindgen]
    pub fn set_goal(&mut self, goal_description: String, relevance_threshold: f64) {
        let goal = Goal::new(&goal_description, relevance_threshold);
        self.orchestrator.set_goal(goal);
        self.config.goal_description = goal_description;
        self.config.relevance_threshold = relevance_threshold;
    }
    
    /// Evaluate how aligned a text is with the current goal
    #[wasm_bindgen]
    pub fn evaluate_alignment(&self, text: String) -> f64 {
        self.orchestrator.evaluate_alignment(&text)
    }
    
    /// Divide text into units of the specified type
    #[wasm_bindgen]
    pub fn divide_text(&self, text: String, unit_type: String) -> JsValue {
        let unit_type = match unit_type.to_lowercase().as_str() {
            "character" => TextUnitType::Character,
            "word" => TextUnitType::Word,
            "sentence" => TextUnitType::Sentence,
            "paragraph" => TextUnitType::Paragraph,
            "section" => TextUnitType::Section,
            "document" => TextUnitType::Document,
            _ => TextUnitType::Paragraph, // Default
        };
        
        // Create a text unit for the document
        let document_unit = TextUnit::new(
            text,
            0,
            0, // Length will be calculated within new()
            TextUnitType::Document,
            0,
        );
        
        // Divide into the requested units
        let units = document_unit.divide(unit_type);
        
        // Convert to JS-friendly format
        let result: Vec<String> = units.into_iter()
            .map(|unit| unit.content)
            .collect();
            
        serde_wasm_bindgen::to_value(&result).unwrap()
    }
    
    /// Perform research on a topic
    #[wasm_bindgen]
    pub fn research(&self, topic: String) -> JsValue {
        let results = self.orchestrator.research(&topic);
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}

// Export a function to create the module
#[wasm_bindgen]
pub fn init_kwasa_wasm() -> KwasaWasm {
    let config = KwasaConfig::new();
    KwasaWasm::new(config)
} 