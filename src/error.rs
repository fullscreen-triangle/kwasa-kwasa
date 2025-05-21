use std::fmt;
use std::io;
use std::result;
use thiserror::Error;

/// Result type for Kwasa-kwasa operations
pub type Result<T> = result::Result<T, Error>;

/// Comprehensive error types for the Kwasa-kwasa framework
#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Parsing error: {message} at line {line}:{column}")]
    Parse {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("Lexical error: {message} at position {position}")]
    Lexical {
        message: String,
        position: usize,
    },

    #[error("Syntax error: {message} at position {position}")]
    Syntax {
        message: String,
        position: usize,
    },

    #[error("Semantic error: {0}")]
    Semantic(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Genomic error: {0}")]
    Genomic(String),

    #[error("Spectrometry error: {0}")]
    Spectrometry(String),

    #[error("Chemistry error: {0}")]
    Chemistry(String),

    #[error("Pattern recognition error: {0}")]
    Pattern(String),

    #[error("Evidence error: {0}")]
    Evidence(String),

    #[error("Knowledge error: {0}")]
    Knowledge(String),

    #[error("Orchestration error: {0}")]
    Orchestration(String),

    #[error("Text unit error: {0}")]
    TextUnit(String),

    #[error("CLI error: {0}")]
    Cli(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("WASM error: {0}")]
    Wasm(String),

    #[error("{0}")]
    Custom(String),
}

/// Error reporter for detailed error reporting
pub struct ErrorReporter {
    source: Option<String>,
    errors: Vec<Error>,
    source_lines: Option<Vec<String>>,
}

impl ErrorReporter {
    pub fn new() -> Self {
        Self {
            source: None,
            errors: Vec::new(),
            source_lines: None,
        }
    }

    pub fn with_source(mut self, source: String) -> Self {
        self.source_lines = Some(source.lines().map(String::from).collect());
        self.source = Some(source);
        self
    }

    pub fn add_error(&mut self, error: Error) {
        self.errors.push(error);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn report(&self) -> String {
        if self.errors.is_empty() {
            return "No errors".to_string();
        }

        let mut report = format!("{} error(s) found:\n", self.errors.len());

        for (i, error) in self.errors.iter().enumerate() {
            report.push_str(&format!("{}. {}\n", i + 1, error));
            
            // Add source context for position-based errors
            if let Some(ref lines) = self.source_lines {
                if let Error::Parse { line, column, .. } = error {
                    if *line > 0 && *line <= lines.len() {
                        // Show the line containing the error
                        let line_idx = line - 1;
                        report.push_str(&format!("   → {}: {}\n", line, lines[line_idx]));
                        
                        // Show pointer to the exact position
                        if *column <= lines[line_idx].len() {
                            let mut pointer = " ".repeat(*column + 5);
                            pointer.push('^');
                            report.push_str(&format!("   {}\n", pointer));
                        }
                        
                        // Show surrounding context (1 line before and after if available)
                        if line_idx > 0 {
                            report.push_str(&format!("     {}: {}\n", line_idx, lines[line_idx - 1]));
                        }
                        if line_idx + 1 < lines.len() {
                            report.push_str(&format!("     {}: {}\n", line_idx + 2, lines[line_idx + 1]));
                        }
                    }
                }
            }
        }

        report
    }
    
    /// Try to recover from errors with a best-effort approach
    pub fn recover(&self) -> bool {
        // Only attempt recovery for non-critical errors
        self.errors.iter().all(|e| match e {
            Error::Parse { .. } | Error::Lexical { .. } | Error::Syntax { .. } => false,
            _ => true,
        })
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
} 