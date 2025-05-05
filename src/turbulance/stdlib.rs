use std::collections::HashMap;
use crate::turbulance::ast::Value;
use crate::turbulance::TurbulanceError;
use crate::text_unit::boundary::TextUnit;

/// Type alias for standard library functions
pub type StdLibFn = fn(Vec<Value>) -> Result<Value, TurbulanceError>;

/// Represents the standard library for Turbulance
pub struct StdLib {
    /// Map of function names to implementations
    functions: HashMap<String, StdLibFn>,
}

impl StdLib {
    /// Create a new standard library with all built-in functions
    pub fn new() -> Self {
        let mut stdlib = Self {
            functions: HashMap::new(),
        };
        
        // Register all standard library functions
        stdlib.register_all();
        
        stdlib
    }
    
    /// Register all standard library functions
    fn register_all(&mut self) {
        // Text analysis functions
        self.register("readability_score", readability_score);
        self.register("contains", contains);
        self.register("extract_patterns", extract_patterns);
        
        // Text transformation functions
        self.register("research_context", research_context);
        self.register("ensure_explanation_follows", ensure_explanation_follows);
        self.register("simplify_sentences", simplify_sentences);
        self.register("replace_jargon", replace_jargon);
        
        // Utility functions
        self.register("print", print);
        self.register("len", len);
        self.register("typeof", typeof);
    }
    
    /// Register a standard library function
    fn register(&mut self, name: &str, func: StdLibFn) {
        self.functions.insert(name.to_string(), func);
    }
    
    /// Call a standard library function
    pub fn call(&self, name: &str, args: Vec<Value>) -> Result<Value, TurbulanceError> {
        match self.functions.get(name) {
            Some(func) => func(args),
            None => Err(TurbulanceError::RuntimeError {
                message: format!("Unknown function: {}", name),
            }),
        }
    }
    
    /// Check if a function exists in the standard library
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
    
    /// Get a list of all available functions
    pub fn list_functions(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

// ========== Text Analysis Functions ==========

/// Calculate readability score for a text
fn readability_score(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement readability scoring
    // This is a stub implementation
    
    // Check that we have at least one argument
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "readability_score requires a text argument".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "readability_score expects a string or text unit".to_string(),
            });
        }
    };
    
    // Placeholder implementation
    // In a real implementation, this would use proper readability metrics
    let words = text.split_whitespace().count();
    let sentences = text.split(&['.', '!', '?']).filter(|s| !s.trim().is_empty()).count();
    
    if sentences == 0 {
        return Ok(Value::Number(0.0));
    }
    
    // Simple average words per sentence as placeholder
    let avg_words_per_sentence = words as f64 / sentences as f64;
    let score = 100.0 - (avg_words_per_sentence * 5.0).min(100.0);
    
    Ok(Value::Number(score))
}

/// Check if text contains a pattern
fn contains(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement pattern matching
    // This is a stub implementation
    
    // Check that we have at least two arguments
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contains requires text and pattern arguments".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "contains expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Extract the pattern argument
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "contains expects a string pattern as second argument".to_string(),
            });
        }
    };
    
    // Simple string contains check as placeholder
    // In a real implementation, this would use more sophisticated pattern matching
    Ok(Value::Bool(text.contains(pattern)))
}

/// Extract patterns from text
fn extract_patterns(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement pattern extraction
    // This is a stub implementation
    
    Err(TurbulanceError::RuntimeError {
        message: "extract_patterns not yet implemented".to_string(),
    })
}

// ========== Text Transformation Functions ==========

/// Research context for a domain
fn research_context(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement research context
    // This is a stub implementation
    
    Err(TurbulanceError::RuntimeError {
        message: "research_context not yet implemented".to_string(),
    })
}

/// Ensure explanation follows a term
fn ensure_explanation_follows(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement ensure explanation follows
    // This is a stub implementation
    
    Err(TurbulanceError::RuntimeError {
        message: "ensure_explanation_follows not yet implemented".to_string(),
    })
}

/// Simplify sentences in text
fn simplify_sentences(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement sentence simplification
    // This is a stub implementation
    
    Err(TurbulanceError::RuntimeError {
        message: "simplify_sentences not yet implemented".to_string(),
    })
}

/// Replace jargon with simpler terms
fn replace_jargon(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement jargon replacement
    // This is a stub implementation
    
    Err(TurbulanceError::RuntimeError {
        message: "replace_jargon not yet implemented".to_string(),
    })
}

// ========== Utility Functions ==========

/// Print a value to the console
fn print(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    for arg in args {
        println!("{}", arg);
    }
    
    Ok(Value::None)
}

/// Get the length of a value
fn len(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "len requires an argument".to_string(),
        });
    }
    
    let length = match &args[0] {
        Value::String(s) => s.len(),
        Value::TextUnit(tu) => tu.content.len(),
        Value::List(l) => l.len(),
        Value::Map(m) => m.len(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "len can only be called on strings, text units, lists, or maps".to_string(),
            });
        }
    };
    
    Ok(Value::Number(length as f64))
}

/// Get the type of a value
fn typeof(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "typeof requires an argument".to_string(),
        });
    }
    
    let type_name = match &args[0] {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::List(_) => "list",
        Value::Map(_) => "map",
        Value::Function(_) => "function",
        Value::TextUnit(_) => "text_unit",
        Value::None => "none",
    };
    
    Ok(Value::String(type_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdlib_functions_exist() {
        let stdlib = StdLib::new();
        
        assert!(stdlib.has_function("readability_score"));
        assert!(stdlib.has_function("contains"));
        assert!(stdlib.has_function("print"));
        assert!(stdlib.has_function("len"));
        assert!(stdlib.has_function("typeof"));
        
        assert!(!stdlib.has_function("nonexistent_function"));
    }
    
    #[test]
    fn test_contains_function() {
        let stdlib = StdLib::new();
        
        let text = Value::String("Hello, world!".to_string());
        let pattern = Value::String("world".to_string());
        
        let result = stdlib.call("contains", vec![text.clone(), pattern]).unwrap();
        assert_eq!(result, Value::Bool(true));
        
        let wrong_pattern = Value::String("universe".to_string());
        let result = stdlib.call("contains", vec![text, wrong_pattern]).unwrap();
        assert_eq!(result, Value::Bool(false));
    }
    
    #[test]
    fn test_typeof_function() {
        let stdlib = StdLib::new();
        
        let string_val = Value::String("test".to_string());
        let result = stdlib.call("typeof", vec![string_val]).unwrap();
        assert_eq!(result, Value::String("string".to_string()));
        
        let number_val = Value::Number(42.0);
        let result = stdlib.call("typeof", vec![number_val]).unwrap();
        assert_eq!(result, Value::String("number".to_string()));
        
        let bool_val = Value::Bool(true);
        let result = stdlib.call("typeof", vec![bool_val]).unwrap();
        assert_eq!(result, Value::String("boolean".to_string()));
    }
}
