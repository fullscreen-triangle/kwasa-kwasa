pub mod lexer;
pub mod parser;
pub mod ast;
pub mod interpreter;
pub mod stdlib;
pub mod proposition;
pub mod datastructures;

pub use lexer::{Lexer, Token, TokenKind};
pub use proposition::{Proposition, Motion};
pub use datastructures::{TextGraph, ConceptChain, IdeaHierarchy, ArgMap};

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
    // This is a placeholder for now - we'll implement the full execution pipeline
    // once we have the parser and interpreter
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse()?;
    
    let mut interpreter = interpreter::Interpreter::new();
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
