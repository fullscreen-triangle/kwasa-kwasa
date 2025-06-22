//! Error types and handling for the Turbulance language

use std::fmt;

/// Result type for Turbulance operations
pub type Result<T> = std::result::Result<T, TurbulanceError>;

/// Comprehensive error types for the Turbulance language
#[derive(Debug, Clone, thiserror::Error)]
pub enum TurbulanceError {
    /// Lexical analysis errors
    #[error("Lexical error at position {position}: {message}")]
    LexicalError {
        /// Position in source where error occurred
        position: usize,
        /// Error message
        message: String,
    },

    /// Syntax parsing errors
    #[error("Syntax error at line {line}, column {column}: {message}")]
    SyntaxError {
        /// Line number where error occurred
        line: usize,
        /// Column number where error occurred
        column: usize,
        /// Error message
        message: String,
    },

    /// Semantic analysis errors
    #[error("Semantic error: {message}")]
    SemanticError {
        /// Error message
        message: String,
    },

    /// Runtime execution errors
    #[error("Runtime error: {message}")]
    RuntimeError {
        /// Error message
        message: String,
    },

    /// Type-related errors
    #[error("Type error: {message}")]
    TypeError {
        /// Error message
        message: String,
    },

    /// Variable/function not found errors
    #[error("Name error: '{name}' is not defined")]
    NameError {
        /// Name that was not found
        name: String,
    },

    /// Invalid function arguments
    #[error("Argument error: {message}")]
    ArgumentError {
        /// Error message
        message: String,
    },

    /// I/O related errors
    #[error("IO error: {message}")]
    IoError {
        /// Error message
        message: String,
    },

    /// Invalid input format
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Processing timeout
    #[error("Processing timeout exceeded ({timeout_ms}ms)")]
    ProcessingTimeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },

    /// Scientific computation errors
    #[error("Scientific computation error: {message}")]
    ScientificError {
        /// Error message
        message: String,
    },

    /// Text processing errors
    #[error("Text processing error: {message}")]
    TextProcessingError {
        /// Error message
        message: String,
    },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
    },
}

impl TurbulanceError {
    /// Create a new lexical error
    pub fn lexical<S: Into<String>>(position: usize, message: S) -> Self {
        Self::LexicalError {
            position,
            message: message.into(),
        }
    }

    /// Create a new syntax error
    pub fn syntax<S: Into<String>>(line: usize, column: usize, message: S) -> Self {
        Self::SyntaxError {
            line,
            column,
            message: message.into(),
        }
    }

    /// Create a new semantic error
    pub fn semantic<S: Into<String>>(message: S) -> Self {
        Self::SemanticError {
            message: message.into(),
        }
    }

    /// Create a new runtime error
    pub fn runtime<S: Into<String>>(message: S) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }

    /// Create a new type error
    pub fn type_error<S: Into<String>>(message: S) -> Self {
        Self::TypeError {
            message: message.into(),
        }
    }

    /// Create a new name error
    pub fn name_error<S: Into<String>>(name: S) -> Self {
        Self::NameError {
            name: name.into(),
        }
    }

    /// Create a new argument error
    pub fn argument_error<S: Into<String>>(message: S) -> Self {
        Self::ArgumentError {
            message: message.into(),
        }
    }

    /// Create a new scientific error
    pub fn scientific<S: Into<String>>(message: S) -> Self {
        Self::ScientificError {
            message: message.into(),
        }
    }

    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::LexicalError { .. } | Self::SyntaxError { .. } => false,
            Self::ProcessingTimeout { .. } => false,
            _ => true,
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::LexicalError { .. } => ErrorCategory::Lexical,
            Self::SyntaxError { .. } => ErrorCategory::Syntax,
            Self::SemanticError { .. } => ErrorCategory::Semantic,
            Self::RuntimeError { .. } => ErrorCategory::Runtime,
            Self::TypeError { .. } => ErrorCategory::Type,
            Self::NameError { .. } => ErrorCategory::Name,
            Self::ArgumentError { .. } => ErrorCategory::Argument,
            Self::IoError { .. } => ErrorCategory::Io,
            Self::InvalidInput(_) => ErrorCategory::Input,
            Self::ProcessingTimeout { .. } => ErrorCategory::Timeout,
            Self::ScientificError { .. } => ErrorCategory::Scientific,
            Self::TextProcessingError { .. } => ErrorCategory::TextProcessing,
            Self::ConfigError { .. } => ErrorCategory::Config,
        }
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            Self::LexicalError { message, .. } => {
                format!("There's a problem with the text formatting: {}", message)
            }
            Self::SyntaxError { message, .. } => {
                format!("The syntax is incorrect: {}", message)
            }
            Self::SemanticError { message } => {
                format!("The meaning is unclear: {}", message)
            }
            Self::RuntimeError { message } => {
                format!("An error occurred while running: {}", message)
            }
            Self::TypeError { message } => {
                format!("There's a type mismatch: {}", message)
            }
            Self::NameError { name } => {
                format!("'{}' is not recognized. Check spelling or define it first.", name)
            }
            Self::ArgumentError { message } => {
                format!("Function argument issue: {}", message)
            }
            Self::ScientificError { message } => {
                format!("Scientific computation failed: {}", message)
            }
            _ => self.to_string(),
        }
    }
}

/// Error categories for better error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Lexical analysis errors
    Lexical,
    /// Syntax parsing errors
    Syntax,
    /// Semantic analysis errors
    Semantic,
    /// Runtime execution errors
    Runtime,
    /// Type-related errors
    Type,
    /// Name resolution errors
    Name,
    /// Function argument errors
    Argument,
    /// I/O errors
    Io,
    /// Input validation errors
    Input,
    /// Timeout errors
    Timeout,
    /// Scientific computation errors
    Scientific,
    /// Text processing errors
    TextProcessing,
    /// Configuration errors
    Config,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::Lexical => write!(f, "lexical"),
            ErrorCategory::Syntax => write!(f, "syntax"),
            ErrorCategory::Semantic => write!(f, "semantic"),
            ErrorCategory::Runtime => write!(f, "runtime"),
            ErrorCategory::Type => write!(f, "type"),
            ErrorCategory::Name => write!(f, "name"),
            ErrorCategory::Argument => write!(f, "argument"),
            ErrorCategory::Io => write!(f, "io"),
            ErrorCategory::Input => write!(f, "input"),
            ErrorCategory::Timeout => write!(f, "timeout"),
            ErrorCategory::Scientific => write!(f, "scientific"),
            ErrorCategory::TextProcessing => write!(f, "text-processing"),
            ErrorCategory::Config => write!(f, "config"),
        }
    }
}

// Standard library error conversions
impl From<std::io::Error> for TurbulanceError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError {
            message: error.to_string(),
        }
    }
}

impl From<serde_json::Error> for TurbulanceError {
    fn from(error: serde_json::Error) -> Self {
        Self::IoError {
            message: format!("JSON error: {}", error),
        }
    }
}

impl From<serde_yaml::Error> for TurbulanceError {
    fn from(error: serde_yaml::Error) -> Self {
        Self::IoError {
            message: format!("YAML error: {}", error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = TurbulanceError::runtime("test error");
        assert!(matches!(err, TurbulanceError::RuntimeError { .. }));
        assert_eq!(err.category(), ErrorCategory::Runtime);
    }

    #[test]
    fn test_user_message() {
        let err = TurbulanceError::name_error("undefined_var");
        let msg = err.user_message();
        assert!(msg.contains("undefined_var"));
        assert!(msg.contains("not recognized"));
    }

    #[test]
    fn test_recoverable() {
        let runtime_err = TurbulanceError::runtime("test");
        let syntax_err = TurbulanceError::syntax(1, 1, "test");
        
        assert!(runtime_err.is_recoverable());
        assert!(!syntax_err.is_recoverable());
    }
} 