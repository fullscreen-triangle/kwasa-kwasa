use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use dirs;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

/// Configuration for the Kwasa-Kwasa CLI
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CliConfig {
    /// REPL configuration
    pub repl: ReplConfig,
    /// Output formatting preferences
    pub output: OutputConfig,
    /// Editor preferences
    pub editor: EditorConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Custom user settings
    pub custom: HashMap<String, String>,
}

/// REPL-specific configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReplConfig {
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
    /// Enable auto-completion
    pub auto_completion: bool,
    /// History file size limit
    pub history_size: usize,
    /// Prompt format
    pub prompt: String,
    /// Enable vi mode
    pub vi_mode: bool,
    /// Auto-save session
    pub auto_save_session: bool,
}

/// Output formatting configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputConfig {
    /// Use colored output
    pub colored: bool,
    /// Verbosity level (0-3)
    pub verbosity: u8,
    /// Show timestamps
    pub timestamps: bool,
    /// Pretty print JSON/structured output
    pub pretty_print: bool,
    /// Output format preference
    pub format: OutputFormat,
}

/// Editor preferences
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EditorConfig {
    /// Default editor command
    pub editor_command: String,
    /// Tab width for indentation
    pub tab_width: u8,
    /// Use spaces instead of tabs
    pub use_spaces: bool,
    /// Auto-indent
    pub auto_indent: bool,
}

/// Performance settings
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub parallel_processing: bool,
    /// Number of threads to use (0 = auto)
    pub thread_count: usize,
    /// Memory limit in MB (0 = unlimited)
    pub memory_limit: usize,
    /// Timeout for operations in seconds
    pub timeout: u64,
}

/// Output format options
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OutputFormat {
    Text,
    Json,
    Yaml,
    Table,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            repl: ReplConfig::default(),
            output: OutputConfig::default(),
            editor: EditorConfig::default(),
            performance: PerformanceConfig::default(),
            custom: HashMap::new(),
        }
    }
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            syntax_highlighting: true,
            auto_completion: true,
            history_size: 10000,
            prompt: "turbulance> ".to_string(),
            vi_mode: false,
            auto_save_session: true,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            colored: true,
            verbosity: 1,
            timestamps: false,
            pretty_print: true,
            format: OutputFormat::Text,
        }
    }
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            editor_command: std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string()),
            tab_width: 4,
            use_spaces: true,
            auto_indent: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel_processing: true,
            thread_count: 0, // Auto-detect
            memory_limit: 0, // Unlimited
            timeout: 300,    // 5 minutes
        }
    }
}

impl CliConfig {
    /// Load configuration from the default location
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path();
        
        if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
            
            let config: CliConfig = toml::from_str(&content)
                .with_context(|| "Failed to parse config file")?;
            
            Ok(config)
        } else {
            // Create default config
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }
    
    /// Save configuration to the default location
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        
        // Create config directory if it doesn't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {}", parent.display()))?;
        }
        
        let content = toml::to_string_pretty(self)
            .with_context(|| "Failed to serialize config")?;
        
        fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;
        
        Ok(())
    }
    
    /// Get the default configuration file path
    pub fn config_path() -> PathBuf {
        if let Some(config_dir) = dirs::config_dir() {
            config_dir.join("kwasa-kwasa").join("config.toml")
        } else {
            PathBuf::from(".kwasa-kwasa-config.toml")
        }
    }
    
    /// Get a custom setting value
    pub fn get_custom(&self, key: &str) -> Option<&String> {
        self.custom.get(key)
    }
    
    /// Set a custom setting value
    pub fn set_custom(&mut self, key: String, value: String) {
        self.custom.insert(key, value);
    }
    
    /// Update REPL configuration
    pub fn update_repl<F>(&mut self, f: F) where F: FnOnce(&mut ReplConfig) {
        f(&mut self.repl);
    }
    
    /// Update output configuration
    pub fn update_output<F>(&mut self, f: F) where F: FnOnce(&mut OutputConfig) {
        f(&mut self.output);
    }
    
    /// Update editor configuration
    pub fn update_editor<F>(&mut self, f: F) where F: FnOnce(&mut EditorConfig) {
        f(&mut self.editor);
    }
    
    /// Update performance configuration
    pub fn update_performance<F>(&mut self, f: F) where F: FnOnce(&mut PerformanceConfig) {
        f(&mut self.performance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_default_config() {
        let config = CliConfig::default();
        assert!(config.repl.syntax_highlighting);
        assert!(config.output.colored);
        assert_eq!(config.repl.history_size, 10000);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = CliConfig::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: CliConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.repl.syntax_highlighting, deserialized.repl.syntax_highlighting);
        assert_eq!(config.output.verbosity, deserialized.output.verbosity);
    }
    
    #[test]
    fn test_custom_settings() {
        let mut config = CliConfig::default();
        config.set_custom("theme".to_string(), "dark".to_string());
        
        assert_eq!(config.get_custom("theme"), Some(&"dark".to_string()));
        assert_eq!(config.get_custom("nonexistent"), None);
    }
}

