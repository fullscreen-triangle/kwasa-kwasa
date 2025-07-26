//! Kwasa-Kwasa: A metacognitive text processing framework with Turbulance DSL language
//!
//! This library provides:
//! - Turbulance DSL language implementation (lexer, parser, interpreter)
//! - Advanced text unit processing and boundary detection
//! - Intelligent orchestration with metacognitive processing
//! - Goal-oriented writing assistance with intervention systems
//! - Stream processing capabilities
//! - Comprehensive configuration management
//! - V8 Intelligence Network for biological quantum computing
//! - Four-File System for multi-modal processing
//! - Semantic Engine for memory contamination and cognitive frame selection
//! - Domain extensions for quantum, neural, molecular, and fuzzy systems
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
//! // Execute Turbulance code
//! let result = framework.process_turbulance_code("
//!     funxn quantum_demo():
//!         var qubits = quantum_init(4)
//!         var gate_result = quantum_gate(\"hadamard\", [0, 1], [])
//!         var measurements = quantum_measure([0, 1, 2, 3])
//!         return measurements
//! ").await?;
//!
//! println!(\"Quantum processing result: {:?}\", result);
//! # Ok(())
//! # }
//! ```

pub mod ai_enhancement;
pub mod audio;
pub mod chemistry;
pub mod cli;
pub mod error;
pub mod evidence;
pub mod external_apis;
pub mod fullscreen;
pub mod genomic;
pub mod harare;
pub mod image;
pub mod knowledge;
pub mod lib;
pub mod nebuchadnezzar;
pub mod orchestrator;
pub mod pattern;
pub mod sighthound;
pub mod space_computer;
pub mod spectrometry;
pub mod templates;
pub mod text_unit;
pub mod trebuchet;
pub mod turbulance;
pub mod s_entropy;
pub mod utils;
pub mod visualization;
pub mod wasm;
pub mod zangalewa;

// Re-export core types and functions
pub use error::{Error, Result};
pub use turbulance::{run, validate, run_with_context, TurbulanceError, VERSION};
pub use turbulance::interpreter::{Interpreter, Value, NativeFunction};
pub use turbulance::v8_intelligence::{V8IntelligenceNetwork, ProcessingInput, ProcessingOutput};
pub use turbulance::four_file_system::{FourFileSystem, FileType, ProcessedFile, SystemState, ExecutionResult};
pub use turbulance::semantic_engine::{SemanticEngine, CognitiveFrame, UserProfile, ContaminationSequence};
pub use turbulance::domain_extensions;

use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

/// Main framework configuration
#[derive(Debug, Clone)]
pub struct FrameworkConfig {
    pub enable_v8_intelligence: bool,
    pub enable_four_file_system: bool,
    pub enable_semantic_engine: bool,
    pub enable_quantum_interface: bool,
    pub enable_neural_networks: bool,
    pub enable_molecular_assembly: bool,
    pub enable_fuzzy_logic: bool,
    pub enable_vpos_interface: bool,
    pub quantum_coherence_threshold: f64,
    pub neural_learning_rate: f64,
    pub molecular_stability_threshold: f64,
    pub fuzzy_inference_method: String,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            enable_v8_intelligence: true,
            enable_four_file_system: true,
            enable_semantic_engine: true,
            enable_quantum_interface: true,
            enable_neural_networks: true,
            enable_molecular_assembly: true,
            enable_fuzzy_logic: true,
            enable_vpos_interface: true,
            quantum_coherence_threshold: 0.85,
            neural_learning_rate: 0.01,
            molecular_stability_threshold: 0.9,
            fuzzy_inference_method: "mamdani".to_string(),
        }
    }
}

/// Main Kwasa-Kwasa framework
pub struct KwasaFramework {
    config: FrameworkConfig,
    interpreter: Interpreter,
    v8_network: Option<Arc<RwLock<turbulance::v8_intelligence::V8IntelligenceNetwork>>>,
    four_file_system: Option<Arc<RwLock<turbulance::four_file_system::FourFileSystem>>>,
    semantic_engine: Option<Arc<RwLock<turbulance::semantic_engine::SemanticEngine>>>,
}

impl KwasaFramework {
    /// Create a new framework instance
    pub async fn new(config: FrameworkConfig) -> Result<Self> {
        let mut interpreter = Interpreter::new();
        
        // Register standard library functions
        let stdlib_functions = turbulance::stdlib_functions();
        interpreter.register_stdlib_functions(stdlib_functions);
        
        // Initialize subsystems based on configuration
        let v8_network = if config.enable_v8_intelligence {
            Some(Arc::new(RwLock::new(turbulance::v8_intelligence::V8IntelligenceNetwork::new())))
        } else {
            None
        };
        
        let four_file_system = if config.enable_four_file_system {
            Some(Arc::new(RwLock::new(turbulance::four_file_system::FourFileSystem::new())))
        } else {
            None
        };
        
        let semantic_engine = if config.enable_semantic_engine {
            Some(Arc::new(RwLock::new(turbulance::semantic_engine::SemanticEngine::new())))
        } else {
            None
        };
        
        // Register domain extensions
        if let Err(e) = turbulance::domain_extensions::register_domain_extensions(&mut interpreter) {
            eprintln!("Failed to register domain extensions: {}", e);
        }
        
        // Register VPOS interface if enabled
        if config.enable_vpos_interface {
            if let Err(e) = turbulance::domain_extensions::register_vpos_interface(&mut interpreter) {
                eprintln!("Failed to register VPOS interface: {}", e);
            }
        }

        Ok(Self {
            config,
            interpreter,
            v8_network,
            four_file_system,
            semantic_engine,
        })
    }
    
    /// Process Turbulance code through the framework
    pub async fn process_turbulance_code(&mut self, code: &str) -> Result<Value> {
        // Tokenize and parse the code
        let mut lexer = turbulance::lexer::Lexer::new(code);
        let tokens = lexer.tokenize();
        
        let mut parser = turbulance::parser::Parser::new(tokens);
        let ast = parser.parse()
            .map_err(|e| Error::runtime(format!("Parse error: {}", e)))?;
        
        // Execute with the interpreter
        let result = self.interpreter.execute(&ast)
            .map_err(|e| Error::runtime(format!("Execution error: {}", e)))?;
        
        Ok(result)
    }
    
    /// Process with V8 Intelligence Network
    pub async fn process_with_v8_intelligence(
        &self,
        input: ProcessingInput,
    ) -> Result<ProcessingOutput> {
        if let Some(ref v8_network) = self.v8_network {
            let network = v8_network.read().await;
            network.process_network(input).await
                .map_err(|e| Error::runtime(format!("V8 Intelligence processing error: {}", e)))
        } else {
            Err(Error::runtime("V8 Intelligence Network not enabled".to_string()))
        }
    }
    
    /// Process file through Four-File System
    pub async fn process_four_file_system(
        &mut self,
        file_path: std::path::PathBuf,
    ) -> Result<turbulance::four_file_system::ProcessedFile> {
        if let Some(ref four_file_system) = self.four_file_system {
            let mut system = four_file_system.write().await;
            system.process_file(file_path).await
                .map_err(|e| Error::runtime(format!("Four-File System processing error: {}", e)))
        } else {
            Err(Error::runtime("Four-File System not enabled".to_string()))
        }
    }
    
    /// Execute integrated processing across all systems
    pub async fn execute_integrated_processing(&mut self) -> Result<ExecutionResult> {
        if let Some(ref four_file_system) = self.four_file_system {
            let mut system = four_file_system.write().await;
            system.execute_integrated_processing().await
                .map_err(|e| Error::runtime(format!("Integrated processing error: {}", e)))
        } else {
            Err(Error::runtime("Four-File System not enabled".to_string()))
        }
    }
    
    /// Contaminate memory network with semantic engine
    pub async fn contaminate_memory_network(
        &self,
        target_concept: String,
        themes: Vec<String>,
        user_id: &str,
    ) -> Result<turbulance::semantic_engine::ContaminationMetrics> {
        if let Some(ref semantic_engine) = self.semantic_engine {
            let engine = semantic_engine.read().await;
            engine.contaminate_memory_network(target_concept, themes, user_id).await
                .map_err(|e| Error::runtime(format!("Memory contamination error: {}", e)))
        } else {
            Err(Error::runtime("Semantic Engine not enabled".to_string()))
        }
    }
    
    /// Get framework configuration
    pub fn get_config(&self) -> &FrameworkConfig {
        &self.config
    }
    
    /// Get system status across all components
    pub async fn get_system_status(&self) -> SystemStatus {
        let mut status = SystemStatus::default();
        
        // Get V8 Intelligence Network status
        if let Some(ref v8_network) = self.v8_network {
            let network = v8_network.read().await;
            let network_status = network.get_network_status().await;
            status.v8_intelligence_active = !network_status.active_modules.is_empty();
            status.v8_processing_capacity = network_status.processing_capacity;
            status.v8_coherence_level = network_status.coherence_level;
        }
        
        // Get Four-File System status
        if let Some(ref four_file_system) = self.four_file_system {
            let system = four_file_system.read().await;
            let system_state = system.get_system_state();
            status.four_file_quantum_coherence = system_state.quantum_coherence;
            status.four_file_neural_activation = system_state.neural_activation;
            status.four_file_molecular_stability = system_state.molecular_stability;
            status.four_file_fuzzy_consistency = system_state.fuzzy_consistency;
        }
        
        status.turbulance_interpreter_ready = true;
        status.framework_operational = true;
        
        status
    }
}

/// System status information
#[derive(Debug, Clone, Default)]
pub struct SystemStatus {
    pub framework_operational: bool,
    pub turbulance_interpreter_ready: bool,
    pub v8_intelligence_active: bool,
    pub v8_processing_capacity: f64,
    pub v8_coherence_level: f64,
    pub four_file_quantum_coherence: f64,
    pub four_file_neural_activation: f64,
    pub four_file_molecular_stability: f64,
    pub four_file_fuzzy_consistency: f64,
    pub semantic_engine_active: bool,
    pub vpos_interface_active: bool,
}

// Include generated code
mod generated {
    pub(crate) mod prelude {
        pub use super::super::turbulance::interpreter::{NativeFunction, Value};
        pub use super::super::turbulance::{Result, TokenKind, TurbulanceError};
    }
    pub(crate) use prelude::*;

    // For now, provide stub implementations for generated functions
    // In a real build, these would be generated by build.rs
    
    pub fn keywords_table() -> std::collections::HashMap<&'static str, crate::turbulance::TokenKind> {
        let mut keywords = std::collections::HashMap::new();
        keywords.insert("funxn", crate::turbulance::TokenKind::FunctionDecl);
        keywords.insert("within", crate::turbulance::TokenKind::Within);
        keywords.insert("given", crate::turbulance::TokenKind::Given);
        keywords.insert("project", crate::turbulance::TokenKind::ProjectDecl);
        keywords.insert("ensure", crate::turbulance::TokenKind::Ensure);
        keywords.insert("return", crate::turbulance::TokenKind::Return);
        keywords.insert("proposition", crate::turbulance::TokenKind::Proposition);
        keywords.insert("var", crate::turbulance::TokenKind::Var);
        keywords.insert("if", crate::turbulance::TokenKind::If);
        keywords.insert("else", crate::turbulance::TokenKind::Else);
        keywords.insert("true", crate::turbulance::TokenKind::True);
        keywords.insert("false", crate::turbulance::TokenKind::False);
        keywords
    }
    
    pub fn operator_precedence() -> std::collections::HashMap<crate::turbulance::TokenKind, u8> {
        let mut precedence = std::collections::HashMap::new();
        precedence.insert(crate::turbulance::TokenKind::Plus, 10);
        precedence.insert(crate::turbulance::TokenKind::Minus, 10);
        precedence.insert(crate::turbulance::TokenKind::Multiply, 20);
        precedence.insert(crate::turbulance::TokenKind::Divide, 20);
        precedence.insert(crate::turbulance::TokenKind::Equal, 5);
        precedence.insert(crate::turbulance::TokenKind::NotEqual, 5);
        precedence.insert(crate::turbulance::TokenKind::LessThan, 5);
        precedence.insert(crate::turbulance::TokenKind::GreaterThan, 5);
        precedence.insert(crate::turbulance::TokenKind::LessThanEqual, 5);
        precedence.insert(crate::turbulance::TokenKind::GreaterThanEqual, 5);
        precedence
    }
}

// Re-export generated types
pub use generated::*;

// Core functionality for running Turbulance code
pub async fn run_async(source: &str) -> Result<()> {
    let config = FrameworkConfig::default();
    let mut framework = KwasaFramework::new(config).await?;
    framework.process_turbulance_code(source).await?;
    Ok(())
}

/// Version of the Kwasa-Kwasa framework
pub const FRAMEWORK_VERSION: &str = "0.1.0";

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_framework_creation() {
        let config = FrameworkConfig::default();
        let framework = KwasaFramework::new(config).await;
        assert!(framework.is_ok());
    }

    #[tokio::test]
    async fn test_turbulance_execution() {
        let config = FrameworkConfig::default();
        let mut framework = KwasaFramework::new(config).await.unwrap();
        
        let result = framework.process_turbulance_code("return 42").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_system_status() {
        let config = FrameworkConfig::default();
        let framework = KwasaFramework::new(config).await.unwrap();
        
        let status = framework.get_system_status().await;
        assert!(status.framework_operational);
        assert!(status.turbulance_interpreter_ready);
    }
}
