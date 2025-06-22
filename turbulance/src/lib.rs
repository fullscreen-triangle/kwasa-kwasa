//! # Turbulance - Universal Scientific Experiment DSL
//!
//! Turbulance is a domain-specific language designed for formalizing scientific methods and experiments.
//! It provides semantic computation capabilities where textual inputs are processed as structured 
//! semantic units, enabling scientists to express experimental procedures, hypotheses, and data 
//! transformations in a programmatic yet natural way.
//!
//! ## Core Philosophy
//!
//! The language operates on the principle that scientific meaning can be preserved through 
//! computational transformation. Text is decomposed into meaningful units that can undergo 
//! mathematical operations while preserving semantic content.
//!
//! ## Basic Usage
//!
//! ```rust
//! use turbulance::{Engine, Script};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = Engine::new();
//! let script = Script::from_source(r#"
//!     funxn analyze_data(dataset):
//!         item cleaned = dataset / noise
//!         item patterns = extract_patterns(cleaned)
//!         return patterns + statistical_summary(cleaned)
//! "#)?;
//!
//! let result = engine.execute(&script)?;
//! println!("Result: {}", result);
//! # Ok(())
//! # }
//! ```
//!
//! ## Scientific Constructs
//!
//! Turbulance provides specialized constructs for scientific workflows:
//!
//! - **Propositions**: Formalize hypotheses and scientific claims
//! - **Motions**: Define procedural steps in experiments
//! - **Semantic Operations**: Apply meaningful transformations to data
//! - **Evidence Integration**: Combine multiple data sources semantically
//!
//! ## Features
//!
//! - **Lightweight**: Minimal dependencies, fast execution
//! - **Cross-platform**: Works on desktop, web (WASM), and embedded systems
//! - **Extensible**: Plugin system for domain-specific extensions
//! - **Scientific**: Built-in statistical and analytical functions

#![deny(missing_docs)]
#![warn(rust_2018_idioms)]

// Core language modules
pub mod lexer;
pub mod parser;
pub mod ast;
pub mod interpreter;
pub mod error;

// Language features
pub mod proposition;
pub mod text_unit;
pub mod semantic_ops;

// Standard library
pub mod stdlib;

// Runtime and execution
pub mod engine;
pub mod script;
pub mod context;

// Optional features
#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "scientific-stdlib")]
pub mod scientific;

// Re-exports for convenience
pub use engine::Engine;
pub use script::Script;
pub use context::Context;
pub use error::{TurbulanceError, Result};
pub use interpreter::Value;

/// Version of the Turbulance language
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parse and validate Turbulance source code without executing it
pub fn validate(source: &str) -> Result<bool> {
    let tokens = lexer::tokenize(source)?;
    let ast = parser::parse(tokens)?;
    Ok(ast.is_valid())
}

/// Quick execution of Turbulance source code with default settings
pub fn execute(source: &str) -> Result<Value> {
    let engine = Engine::new();
    let script = Script::from_source(source)?;
    engine.execute(&script)
}

/// Create a new execution context with default settings
pub fn new_context() -> Context {
    Context::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_simple_execution() {
        let result = execute("return 42").unwrap();
        match result {
            Value::Number(n) => assert_eq!(n, 42.0),
            _ => panic!("Expected number"),
        }
    }

    #[test]
    fn test_validation() {
        assert!(validate("funxn test(): return 42").unwrap());
        assert!(!validate("invalid syntax @#$%").unwrap());
    }
} 