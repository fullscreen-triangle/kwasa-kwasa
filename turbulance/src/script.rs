//! Turbulance script representation

use crate::ast::Node;
use crate::lexer::tokenize;
use crate::parser::parse;
use crate::error::Result;
use std::path::Path;

/// Represents a parsed Turbulance script
#[derive(Debug, Clone)]
pub struct Script {
    /// The parsed AST
    ast: Node,
    /// Original source code
    source: String,
    /// Optional file path
    file_path: Option<String>,
}

impl Script {
    /// Create a script from source code
    pub fn from_source(source: &str) -> Result<Self> {
        let tokens = tokenize(source)?;
        let ast = parse(tokens)?;
        
        Ok(Self {
            ast,
            source: source.to_string(),
            file_path: None,
        })
    }

    /// Create a script from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path)
            .map_err(|e| crate::error::TurbulanceError::IoError { 
                message: format!("Failed to read file {}: {}", path.display(), e) 
            })?;
        
        let tokens = tokenize(&source)?;
        let ast = parse(tokens)?;
        
        Ok(Self {
            ast,
            source,
            file_path: Some(path.to_string_lossy().to_string()),
        })
    }

    /// Create a script from an existing AST
    pub fn from_ast(ast: Node, source: String) -> Self {
        Self {
            ast,
            source,
            file_path: None,
        }
    }

    /// Get the AST
    pub fn ast(&self) -> &Node {
        &self.ast
    }

    /// Get the source code
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the file path (if loaded from file)
    pub fn file_path(&self) -> Option<&str> {
        self.file_path.as_deref()
    }

    /// Validate the script syntax and semantics
    pub fn validate(&self) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();
        
        // Basic AST validation
        if !self.ast.is_valid() {
            result.add_error("Invalid AST structure".to_string());
        }
        
        // Semantic validation would go here
        // For now, we'll just check basic structure
        if self.source.trim().is_empty() {
            result.add_warning("Empty script".to_string());
        }
        
        Ok(result)
    }

    /// Get metadata about the script
    pub fn metadata(&self) -> ScriptMetadata {
        let lines = self.source.lines().count();
        let chars = self.source.chars().count();
        let bytes = self.source.len();
        
        // Count different types of nodes (simplified)
        let functions = self.count_nodes_of_type("function");
        let propositions = self.count_nodes_of_type("proposition");
        
        ScriptMetadata {
            lines,
            characters: chars,
            bytes,
            functions,
            propositions,
            file_path: self.file_path.clone(),
        }
    }

    /// Count nodes of a specific type (simplified implementation)
    fn count_nodes_of_type(&self, _node_type: &str) -> usize {
        // This would need a proper AST traversal implementation
        // For now, return 0
        0
    }

    /// Get a pretty-printed representation of the AST
    pub fn ast_debug(&self) -> String {
        format!("{:#?}", self.ast)
    }

    /// Get a summary of the script
    pub fn summary(&self) -> String {
        let metadata = self.metadata();
        let mut summary = String::new();
        
        if let Some(path) = &self.file_path {
            summary.push_str(&format!("File: {}\n", path));
        }
        
        summary.push_str(&format!("Lines: {}\n", metadata.lines));
        summary.push_str(&format!("Characters: {}\n", metadata.characters));
        
        if metadata.functions > 0 {
            summary.push_str(&format!("Functions: {}\n", metadata.functions));
        }
        
        if metadata.propositions > 0 {
            summary.push_str(&format!("Propositions: {}\n", metadata.propositions));
        }
        
        summary
    }
}

/// Script validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get all errors
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Get all warnings
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Get a summary of the validation result
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        
        if self.errors.is_empty() && self.warnings.is_empty() {
            summary.push_str("✓ Validation passed with no issues");
        } else {
            if !self.errors.is_empty() {
                summary.push_str(&format!("✗ {} error(s):\n", self.errors.len()));
                for (i, error) in self.errors.iter().enumerate() {
                    summary.push_str(&format!("  {}. {}\n", i + 1, error));
                }
            }
            
            if !self.warnings.is_empty() {
                summary.push_str(&format!("⚠ {} warning(s):\n", self.warnings.len()));
                for (i, warning) in self.warnings.iter().enumerate() {
                    summary.push_str(&format!("  {}. {}\n", i + 1, warning));
                }
            }
        }
        
        summary
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a script
#[derive(Debug, Clone)]
pub struct ScriptMetadata {
    /// Number of lines
    pub lines: usize,
    /// Number of characters
    pub characters: usize,
    /// Number of bytes
    pub bytes: usize,
    /// Number of function definitions
    pub functions: usize,
    /// Number of proposition definitions
    pub propositions: usize,
    /// File path (if loaded from file)
    pub file_path: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_from_source() {
        let script = Script::from_source("42").unwrap();
        assert_eq!(script.source(), "42");
        assert!(script.file_path().is_none());
    }

    #[test]
    fn test_script_validation() {
        let script = Script::from_source("42").unwrap();
        let result = script.validate().unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_script_metadata() {
        let script = Script::from_source("funxn test(): return 42").unwrap();
        let metadata = script.metadata();
        assert_eq!(metadata.lines, 1);
        assert!(metadata.characters > 0);
    }

    #[test]
    fn test_empty_script_warning() {
        let script = Script::from_source("").unwrap();
        let result = script.validate().unwrap();
        assert!(result.is_valid()); // No errors
        assert!(!result.warnings().is_empty()); // But has warnings
    }

    #[test]
    fn test_complex_script() {
        let source = r#"
            proposition TestHypothesis:
                motion Hypothesis("Test hypothesis")
                
                given true:
                    return "success"
            
            funxn analyze():
                item data = load_data("test.csv")
                return data
        "#;
        
        let script = Script::from_source(source).unwrap();
        assert!(script.source().contains("proposition"));
        assert!(script.source().contains("funxn"));
        
        let metadata = script.metadata();
        assert!(metadata.lines > 1);
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.is_valid());
        
        result.add_error("Test error".to_string());
        assert!(!result.is_valid());
        assert_eq!(result.errors().len(), 1);
        
        result.add_warning("Test warning".to_string());
        assert_eq!(result.warnings().len(), 1);
        
        let summary = result.summary();
        assert!(summary.contains("error"));
        assert!(summary.contains("warning"));
    }
} 