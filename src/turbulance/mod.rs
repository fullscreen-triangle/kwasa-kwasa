pub mod lexer;
pub mod parser;
pub mod ast;
pub mod interpreter;
pub mod stdlib;
pub mod proposition;
pub mod datastructures;
pub mod domain_extensions;
pub mod context;

pub use lexer::{Lexer, Token, TokenKind};
pub use proposition::{Proposition, Motion};
pub use datastructures::{TextGraph, ConceptChain, IdeaHierarchy, ArgMap};
pub use context::Context;
pub use context::Value;
pub use context::Function;

// Include generated code
mod generated {
    pub(crate) mod prelude {
        // Common imports for generated code
        use std::collections::HashMap;
        use crate::turbulance::{TokenKind, Result, TurbulanceError};
        use crate::turbulance::interpreter::{Value, NativeFunction};

        // Re-export common types
        pub use std::collections::HashMap;
        pub use crate::turbulance::{TokenKind, Result, TurbulanceError};
        pub use crate::turbulance::interpreter::{Value, NativeFunction};
    }
    pub(crate) use prelude::*;
    
    include!(concat!(env!("OUT_DIR"), "/generated/parser_tables.rs"));
    include!(concat!(env!("OUT_DIR"), "/generated/stdlib_bindings.rs"));
    include!(concat!(env!("OUT_DIR"), "/generated/token_definitions.rs"));
}

// Re-export generated types
pub use generated::*;

/// Error types for the Turbulance language
#[derive(Debug, thiserror::Error)]
pub enum TurbulanceError {
    #[error("Lexical error at position {position}: {message}")]
    LexicalError { position: usize, message: String },

    #[error("Syntax error at position {position}: {message}")]
    SyntaxError { position: usize, message: String },

    #[error("Semantic error: {message}")]
    SemanticError { message: String },

    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for Turbulance operations
pub type Result<T> = std::result::Result<T, TurbulanceError>;

/// Parse and run a Turbulance script
pub fn run(source: &str) -> Result<()> {
    // Initialize language components using generated code
    let keyword_table = keywords_table();
    let operator_precedence = operator_precedence();
    let stdlib = stdlib_functions();
    
    // Tokenize the source code
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    
    // Parse tokens into AST
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse()?;
    
    // Execute program with standard library
    let mut interpreter = interpreter::Interpreter::new();
    interpreter.register_stdlib_functions(stdlib);
    
    // Register domain extensions
    domain_extensions::register_domain_extensions(&mut interpreter)
        .map_err(|e| TurbulanceError::RuntimeError { message: e.to_string() })?;
    
    let _ = interpreter.execute(&ast)?;
    
    Ok(())
}

/// Version of the Turbulance language
pub const VERSION: &str = "0.1.0";

/// Check if the given source code is syntactically valid
pub fn validate(source: &str) -> Result<bool> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    
    let mut parser = parser::Parser::new(tokens);
    match parser.parse() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

// Serialize AST to JSON
pub fn serialize_ast(program: &ast::Program) -> String {
    let serializable = SerializableAst::from(program);
    serde_json::to_string_pretty(&serializable).unwrap_or_else(|_| "{}".to_string())
}

/// Run Turbulance code with a specific context
pub fn run_with_context(source: &str, context: &mut Context) -> Result<String> {
    let tokens = lexer::lex(source)?;
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse()?;
    let result = interpreter::interpret(ast, context)?;
    Ok(result)
}

impl TurbulanceError {
    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::LexicalError { .. } | Self::SyntaxError { .. } => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_run_placeholder() {
        let result = run("funxn test(): return 42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_placeholder() {
        let result = validate("funxn test(): return 42");
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}
