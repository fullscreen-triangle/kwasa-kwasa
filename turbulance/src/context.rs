//! Execution context for Turbulance

use crate::interpreter::Value;
use std::collections::HashMap;
use std::path::PathBuf;

/// Execution context configuration
#[derive(Debug, Clone)]
pub struct Context {
    /// Working directory for file operations
    pub working_directory: PathBuf,
    /// Global variables and constants
    pub globals: HashMap<String, Value>,
    /// Configuration settings
    pub settings: ContextSettings,
    /// Data sources configuration
    pub data_sources: Vec<DataSource>,
}

impl Context {
    /// Create a new default context
    pub fn new() -> Self {
        Self {
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            globals: HashMap::new(),
            settings: ContextSettings::default(),
            data_sources: Vec::new(),
        }
    }

    /// Create a context with custom working directory
    pub fn with_working_dir<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            working_directory: path.into(),
            globals: HashMap::new(),
            settings: ContextSettings::default(),
            data_sources: Vec::new(),
        }
    }

    /// Set a global variable
    pub fn set_global<K: Into<String>>(&mut self, key: K, value: Value) {
        self.globals.insert(key.into(), value);
    }

    /// Get a global variable
    pub fn get_global(&self, key: &str) -> Option<&Value> {
        self.globals.get(key)
    }

    /// Add a data source
    pub fn add_data_source(&mut self, source: DataSource) {
        self.data_sources.push(source);
    }

    /// Get all data sources
    pub fn data_sources(&self) -> &[DataSource] {
        &self.data_sources
    }

    /// Update settings
    pub fn with_settings(mut self, settings: ContextSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Get a mutable reference to settings
    pub fn settings_mut(&mut self) -> &mut ContextSettings {
        &mut self.settings
    }

    /// Get settings
    pub fn settings(&self) -> &ContextSettings {
        &self.settings
    }

    /// Resolve a file path relative to the working directory
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        if std::path::Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.working_directory.join(path)
        }
    }

    /// Get context information as a summary
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str(&format!("Working Directory: {}\n", self.working_directory.display()));
        summary.push_str(&format!("Global Variables: {}\n", self.globals.len()));
        summary.push_str(&format!("Data Sources: {}\n", self.data_sources.len()));
        
        if self.settings.debug_mode {
            summary.push_str("Debug Mode: Enabled\n");
        }
        
        if self.settings.strict_mode {
            summary.push_str("Strict Mode: Enabled\n");
        }
        
        summary
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Context configuration settings
#[derive(Debug, Clone)]
pub struct ContextSettings {
    /// Enable debug mode for verbose output
    pub debug_mode: bool,
    /// Enable strict mode for stricter validation
    pub strict_mode: bool,
    /// Confidence threshold for probabilistic operations
    pub confidence_threshold: f64,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
    /// Enable experimental features
    pub experimental_features: bool,
    /// Scientific notation precision
    pub scientific_precision: usize,
    /// Text processing settings
    pub text_processing: TextProcessingSettings,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            debug_mode: false,
            strict_mode: false,
            confidence_threshold: 0.8,
            max_recursion_depth: 1000,
            experimental_features: false,
            scientific_precision: 6,
            text_processing: TextProcessingSettings::default(),
        }
    }
}

/// Text processing specific settings
#[derive(Debug, Clone)]
pub struct TextProcessingSettings {
    /// Default language for text operations
    pub default_language: String,
    /// Enable semantic understanding
    pub semantic_understanding: bool,
    /// Minimum confidence for text operations
    pub min_confidence: f64,
    /// Maximum text length for processing
    pub max_text_length: usize,
}

impl Default for TextProcessingSettings {
    fn default() -> Self {
        Self {
            default_language: "en".to_string(),
            semantic_understanding: true,
            min_confidence: 0.7,
            max_text_length: 100_000,
        }
    }
}

/// Data source configuration
#[derive(Debug, Clone)]
pub struct DataSource {
    /// Unique identifier for the data source
    pub id: String,
    /// Type of data source (local, web, database, etc.)
    pub source_type: DataSourceType,
    /// Connection parameters
    pub parameters: HashMap<String, String>,
    /// Whether this source is currently active
    pub active: bool,
}

impl DataSource {
    /// Create a new data source
    pub fn new<S: Into<String>>(id: S, source_type: DataSourceType) -> Self {
        Self {
            id: id.into(),
            source_type,
            parameters: HashMap::new(),
            active: true,
        }
    }

    /// Add a parameter
    pub fn with_parameter<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }

    /// Set active status
    pub fn set_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&str> {
        self.parameters.get(key).map(|s| s.as_str())
    }
}

/// Types of data sources
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSourceType {
    /// Local file system
    Local,
    /// Web-based source (HTTP/HTTPS)
    Web,
    /// Database connection
    Database,
    /// Scientific database (PubMed, arXiv, etc.)
    Scientific,
    /// Cloud storage
    Cloud,
    /// Memory/cache
    Memory,
    /// Custom/plugin-based source
    Custom(String),
}

impl std::fmt::Display for DataSourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSourceType::Local => write!(f, "local"),
            DataSourceType::Web => write!(f, "web"),
            DataSourceType::Database => write!(f, "database"),
            DataSourceType::Scientific => write!(f, "scientific"),
            DataSourceType::Cloud => write!(f, "cloud"),
            DataSourceType::Memory => write!(f, "memory"),
            DataSourceType::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let context = Context::new();
        assert!(context.working_directory.exists() || context.working_directory == PathBuf::from("."));
        assert!(context.globals.is_empty());
        assert!(context.data_sources.is_empty());
    }

    #[test]
    fn test_context_with_working_dir() {
        let context = Context::with_working_dir("/tmp");
        assert_eq!(context.working_directory, PathBuf::from("/tmp"));
    }

    #[test]
    fn test_global_variables() {
        let mut context = Context::new();
        
        context.set_global("test_var", Value::Number(42.0));
        assert_eq!(context.get_global("test_var"), Some(&Value::Number(42.0)));
        assert_eq!(context.get_global("nonexistent"), None);
    }

    #[test]
    fn test_data_sources() {
        let mut context = Context::new();
        
        let source = DataSource::new("test_db", DataSourceType::Database)
            .with_parameter("host", "localhost")
            .with_parameter("port", "5432");
        
        context.add_data_source(source);
        
        assert_eq!(context.data_sources().len(), 1);
        assert_eq!(context.data_sources()[0].id, "test_db");
        assert_eq!(context.data_sources()[0].source_type, DataSourceType::Database);
        assert_eq!(context.data_sources()[0].get_parameter("host"), Some("localhost"));
    }

    #[test]
    fn test_path_resolution() {
        let context = Context::with_working_dir("/home/user");
        
        // Relative path
        let relative = context.resolve_path("data.csv");
        assert_eq!(relative, PathBuf::from("/home/user/data.csv"));
        
        // Absolute path
        let absolute = context.resolve_path("/tmp/data.csv");
        assert_eq!(absolute, PathBuf::from("/tmp/data.csv"));
    }

    #[test]
    fn test_context_settings() {
        let mut settings = ContextSettings::default();
        assert!(!settings.debug_mode);
        assert_eq!(settings.confidence_threshold, 0.8);
        
        settings.debug_mode = true;
        settings.confidence_threshold = 0.9;
        
        let context = Context::new().with_settings(settings);
        assert!(context.settings().debug_mode);
        assert_eq!(context.settings().confidence_threshold, 0.9);
    }

    #[test]
    fn test_data_source_types() {
        assert_eq!(DataSourceType::Local.to_string(), "local");
        assert_eq!(DataSourceType::Database.to_string(), "database");
        assert_eq!(DataSourceType::Custom("test".to_string()).to_string(), "custom:test");
    }

    #[test]
    fn test_context_summary() {
        let mut context = Context::new();
        context.set_global("test", Value::String("value".to_string()));
        context.add_data_source(DataSource::new("test", DataSourceType::Local));
        
        let summary = context.summary();
        assert!(summary.contains("Working Directory"));
        assert!(summary.contains("Global Variables: 1"));
        assert!(summary.contains("Data Sources: 1"));
    }
} 