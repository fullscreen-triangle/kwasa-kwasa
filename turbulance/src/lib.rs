//! # Turbulance - Universal Scientific Experiment DSL
//!
//! Turbulance is a domain-specific language designed for scientific research and experimentation.
//! It provides a declarative syntax for hypothesis testing, data analysis, and research workflows.
//!
//! ## Features
//!
//! - **Scientific Syntax**: Purpose-built language constructs for research
//! - **Multi-domain Support**: Audio, chemistry, genomics, spectrometry, and more
//! - **Probabilistic Operations**: Built-in uncertainty and confidence modeling
//! - **Research Integration**: Connect to scientific databases and literature
//! - **Cross-platform**: Desktop, web (WASM), and embedded systems
//!
//! ## Quick Start
//!
//! ```rust
//! use turbulance::{Engine, Script};
//!
//! let mut engine = Engine::new();
//! let script = Script::from_source(r#"
//!     proposition TestHypothesis:
//!         motion Hypothesis("Test hypothesis")
//!         given true: return "success"
//! "#)?;
//! let result = engine.execute(&script)?;
//! # Ok::<(), turbulance::error::TurbulanceError>(())
//! ```
//!
//! ## Scientific Domains
//!
//! - **Audio Processing**: Spectral analysis, neural models, stem separation
//! - **Chemistry**: Molecular analysis, reaction prediction, fingerprinting
//! - **Genomics**: Sequence alignment, variant calling, phylogenetic analysis
//! - **Spectrometry**: Mass spec, NMR, IR, UV-Vis analysis
//! - **Image Processing**: Computer vision, medical imaging, reconstruction
//! - **Knowledge Management**: Literature search, citation analysis
//! - **Statistics**: Hypothesis testing, regression, machine learning

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]

// Core language modules
pub mod error;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod interpreter;
pub mod engine;
pub mod script;
pub mod context;

// Scientific domain modules - ALL functionality from kwasa-kwasa
pub mod audio;
pub mod chemistry;
pub mod genomic;
pub mod spectrometry;
pub mod image;
pub mod knowledge;
pub mod visualization;
pub mod external_apis;
pub mod text_unit;
pub mod orchestrator;
pub mod pattern;
pub mod space_computer;
pub mod fullscreen;
pub mod trebuchet;
pub mod harare;
pub mod utils;

// Re-export main public API
pub use engine::{Engine, EngineStats};
pub use script::{Script, ValidationResult, ScriptMetadata};
pub use context::{Context, ContextSettings, DataSource, DataSourceType};
pub use interpreter::Value;
pub use error::{TurbulanceError, Result};
pub use ast::{Node, TextUnit};

// Re-export scientific functionality
pub use audio::*;
pub use chemistry::*;
pub use genomic::*;
pub use spectrometry::*;
pub use image::*;
pub use knowledge::*;
pub use visualization::*;
pub use external_apis::*;
pub use text_unit::*;
pub use orchestrator::*;
pub use pattern::*;
pub use space_computer::*;
pub use fullscreen::*;
pub use trebuchet::*;
pub use harare::*;

/// Validate Turbulance source code
pub fn validate(source: &str) -> Result<ValidationResult> {
    let script = Script::from_source(source)?;
    script.validate()
}

/// Execute Turbulance source code
pub fn execute(source: &str) -> Result<Value> {
    let mut engine = Engine::new();
    engine.execute_source(source)
}

/// Execute Turbulance source with custom context
pub fn execute_with_context(source: &str, context: Context) -> Result<Value> {
    let mut engine = Engine::with_context(context);
    engine.execute_source(source)
}

/// Create a new engine instance
pub fn new_engine() -> Engine {
    Engine::new()
}

/// Create a new context instance
pub fn new_context() -> Context {
    Context::new()
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_execution() {
        let result = execute("42").unwrap();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_scientific_function() {
        let result = execute("abs(-5)").unwrap();
        assert_eq!(result, Value::Number(5.0));
    }

    #[test]
    fn test_validation() {
        let result = validate("42").unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_proposition() {
        let source = r#"
            proposition Test:
                motion Hypothesis("test")
                given true: return "success"
        "#;
        let result = execute(source).unwrap();
        assert_eq!(result, Value::String("success".to_string()));
    }
} 