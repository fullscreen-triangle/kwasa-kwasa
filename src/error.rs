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
    pub fn pattern(message: impl Into<String>) -> Self {
        Self::Pattern(message.into())
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

impl From<rustyline::error::ReadlineError> for Error {
    fn from(err: rustyline::error::ReadlineError) -> Self {
        Self::Cli(format!("Readline error: {}", err))
    }
}

impl From<rusqlite::Error> for Error {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database(format!("SQLite error: {}", err))
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

#[derive(Debug, Clone)]
pub enum KwasaError {
    /// Configuration and setup errors
    ConfigError(String),
    
    /// Environment management errors
    EnvironmentError(String),
    
    /// File system and I/O errors
    IoError(String),
    
    /// Parsing and syntax errors
    ParseError(String),
    
    /// Runtime execution errors
    RuntimeError(String),
    
    /// Scientific validation errors
    ValidationError(String),
    
    /// Network and external API errors
    NetworkError(String),
    
    /// Database and storage errors
    DatabaseError(String),
    
    /// Authentication and authorization errors
    AuthError(String),
    
    /// Resource exhaustion errors
    ResourceError(String),
}

impl fmt::Display for KwasaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KwasaError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            KwasaError::EnvironmentError(msg) => write!(f, "Environment error: {}", msg),
            KwasaError::IoError(msg) => write!(f, "I/O error: {}", msg),
            KwasaError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            KwasaError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            KwasaError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            KwasaError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            KwasaError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            KwasaError::AuthError(msg) => write!(f, "Authentication error: {}", msg),
            KwasaError::ResourceError(msg) => write!(f, "Resource error: {}", msg),
        }
    }
}

impl std::error::Error for KwasaError {}

// Convenience conversion functions
impl From<std::io::Error> for KwasaError {
    fn from(error: std::io::Error) -> Self {
        KwasaError::IoError(error.to_string())
    }
}

impl From<serde_json::Error> for KwasaError {
    fn from(error: serde_json::Error) -> Self {
        KwasaError::ConfigError(format!("JSON error: {}", error))
    }
}

impl From<toml::de::Error> for KwasaError {
    fn from(error: toml::de::Error) -> Self {
        KwasaError::ConfigError(format!("TOML parse error: {}", error))
    }
}

impl From<toml::ser::Error> for KwasaError {
    fn from(error: toml::ser::Error) -> Self {
        KwasaError::ConfigError(format!("TOML serialize error: {}", error))
    }
}

/// Turbulance-specific error types for the DSL
#[derive(Debug, Clone)]
pub enum TurbulanceError {
    /// Lexical analysis errors
    LexError { message: String, line: usize, column: usize },
    
    /// Parsing errors
    ParseError { message: String, line: usize, column: usize },
    
    /// Type checking errors
    TypeError { message: String, expected: String, found: String },
    
    /// Runtime execution errors
    RuntimeError { message: String, context: String },
    
    /// Scientific validation errors
    ValidationError { message: String, validation_type: String },
    
    /// Undefined variable or function errors
    UndefinedError { name: String, context: String },
    
    /// Import and module errors
    ImportError { module: String, message: String },
    
    /// Memory and resource errors
    ResourceError { resource: String, message: String },
}

impl fmt::Display for TurbulanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TurbulanceError::LexError { message, line, column } => {
                write!(f, "Lexical error at line {}, column {}: {}", line, column, message)
            },
            TurbulanceError::ParseError { message, line, column } => {
                write!(f, "Parse error at line {}, column {}: {}", line, column, message)
            },
            TurbulanceError::TypeError { message, expected, found } => {
                write!(f, "Type error: {}. Expected {}, found {}", message, expected, found)
            },
            TurbulanceError::RuntimeError { message, context } => {
                write!(f, "Runtime error in {}: {}", context, message)
            },
            TurbulanceError::ValidationError { message, validation_type } => {
                write!(f, "Validation error ({}): {}", validation_type, message)
            },
            TurbulanceError::UndefinedError { name, context } => {
                write!(f, "Undefined {} in {}", name, context)
            },
            TurbulanceError::ImportError { module, message } => {
                write!(f, "Import error in module '{}': {}", module, message)
            },
            TurbulanceError::ResourceError { resource, message } => {
                write!(f, "Resource error ({}): {}", resource, message)
            },
        }
    }
}

impl std::error::Error for TurbulanceError {}

/// Result type for Kwasa operations
pub type KwasaResult<T> = Result<T, KwasaError>;

/// Result type for Turbulance operations
pub type TurbulanceResult<T> = Result<T, TurbulanceError>;

/// Error context for better error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
    pub source_line: Option<String>,
}

impl ErrorContext {
    pub fn new(file: Option<String>, line: usize, column: usize, source_line: Option<String>) -> Self {
        Self { file, line, column, source_line }
    }
    
    pub fn format_error(&self, error: &TurbulanceError) -> String {
        let mut output = String::new();
        
        if let Some(file) = &self.file {
            output.push_str(&format!("File: {}\n", file));
        }
        
        output.push_str(&format!("Error: {}\n", error));
        
        if let Some(source) = &self.source_line {
            output.push_str(&format!("  {}\n", source));
            output.push_str(&format!("  {}^\n", " ".repeat(self.column.saturating_sub(1))));
        }
        
        output
    }
}

/// Macro for creating validation errors
#[macro_export]
macro_rules! validation_error {
    ($validation_type:expr, $message:expr) => {
        TurbulanceError::ValidationError {
            message: $message.to_string(),
            validation_type: $validation_type.to_string(),
        }
    };
}

/// Macro for creating runtime errors with context
#[macro_export]
macro_rules! runtime_error {
    ($context:expr, $message:expr) => {
        TurbulanceError::RuntimeError {
            message: $message.to_string(),
            context: $context.to_string(),
        }
    };
} 