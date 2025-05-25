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

    /// Visualization errors
    #[error("Visualization error: {0}")]
    Visualization(String),

    /// External API errors
    #[error("External API error: {0}")]
    ExternalApi(String),
}

impl Error {
    /// Create a parsing error
    pub fn parse(message: &str, line: usize, column: usize) -> Self {
        Self::Parse {
            message: message.to_string(),
            line,
            column,
        }
    }
    
    /// Create a lexical error
    pub fn lexical(message: &str, position: usize) -> Self {
        Self::Lexical {
            message: message.to_string(),
            position,
        }
    }
    
    /// Create a syntax error
    pub fn syntax(message: &str, position: usize) -> Self {
        Self::Syntax {
            message: message.to_string(),
            position,
        }
    }
    
    /// Create a semantic error
    pub fn semantic(message: &str) -> Self {
        Self::Semantic(message.to_string())
    }
    
    /// Create a runtime error
    pub fn runtime(message: &str) -> Self {
        Self::Runtime(message.to_string())
    }
    
    /// Create a pattern recognition error
    pub fn pattern(message: &str) -> Self {
        Self::Pattern(message.to_string())
    }
    
    /// Create a knowledge error
    pub fn knowledge(message: &str) -> Self {
        Self::Knowledge(message.to_string())
    }
    
    /// Create an evidence error
    pub fn evidence(message: &str) -> Self {
        Self::Evidence(message.to_string())
    }
    
    /// Create a CLI error
    pub fn cli(message: impl Into<String>) -> Self {
        Self::Cli(message.into())
    }
    
    /// Create a text unit error
    pub fn text_unit(message: &str) -> Self {
        Self::TextUnit(message.to_string())
    }
    
    /// Create a visualization error
    pub fn visualization(msg: impl Into<String>) -> Self {
        Self::Visualization(msg.into())
    }
    
    /// Create an external API error
    pub fn external_api(msg: impl Into<String>) -> Self {
        Self::ExternalApi(msg.into())
    }
    
    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Parse { .. } | Self::Lexical { .. } | Self::Syntax { .. } => false,
            _ => true,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Custom(format!("JSON serialization error: {}", err))
    }
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
        // Only attempt recovery if all errors are recoverable
        self.errors.iter().all(|e| e.is_recoverable())
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new()
    }
} 