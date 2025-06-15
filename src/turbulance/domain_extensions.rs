//! Domain Extensions for Turbulance
//! 
//! This module integrates all domain extensions with the Turbulance language.

use crate::turbulance::interpreter::{Interpreter, Value, ObjectRef, Object};
use crate::turbulance::TurbulanceError;
use crate::turbulance::context::Context;
use crate::error::{Error, Result};
use crate::turbulance::ast::Node;
use std::collections::HashMap;
use std::sync::Arc;

/// Register all domain extensions with the interpreter
pub fn register_domain_extensions(interpreter: &mut Interpreter) -> std::result::Result<(), TurbulanceError> {
    register_genomic_extensions(interpreter)?;
    register_spectrometry_extensions(interpreter)?;
    register_chemistry_extensions(interpreter)?;
    register_pattern_extensions(interpreter)?;
    Ok(())
}

/// Register genomic extensions with the interpreter
fn register_genomic_extensions(interpreter: &mut Interpreter) -> std::result::Result<(), TurbulanceError> {
    let genomic_module = create_module("genomic");
    
    // Register placeholder genomic constructors
    genomic_module.borrow_mut().set(
        "NucleotideSequence", 
        Value::Object(HashMap::new())
    );
    
    interpreter.set_global("genomic", Value::Module(genomic_module));
    Ok(())
}

/// Register spectrometry extensions with the interpreter
fn register_spectrometry_extensions(interpreter: &mut Interpreter) -> std::result::Result<(), TurbulanceError> {
    let spectrometry_module = create_module("spectrometry");
    interpreter.set_global("spectrometry", Value::Module(spectrometry_module));
    Ok(())
}

/// Register chemistry extensions with the interpreter
fn register_chemistry_extensions(interpreter: &mut Interpreter) -> std::result::Result<(), TurbulanceError> {
    let chemistry_module = create_module("chemistry");
    interpreter.set_global("chemistry", Value::Module(chemistry_module));
    Ok(())
}

/// Register pattern extensions with the interpreter
fn register_pattern_extensions(interpreter: &mut Interpreter) -> std::result::Result<(), TurbulanceError> {
    let pattern_module = create_module("pattern");
    interpreter.set_global("pattern", Value::Module(pattern_module));
    Ok(())
}

// Helper function to create a new module
fn create_module(name: &str) -> ObjectRef {
    let module = ObjectRef::new(Module {
        name: name.to_string(),
        fields: HashMap::new(),
    });
    module
}

/// Module struct for domain modules
#[derive(Debug)]
struct Module {
    name: String,
    fields: HashMap<String, Value>,
}

impl Module {
    fn set(&mut self, name: &str, value: Value) {
        self.fields.insert(name.to_string(), value);
    }
}

impl Object for Module {
    fn get(&self, name: &str) -> Option<Value> {
        self.fields.get(name).cloned()
    }
    
    fn set(&mut self, name: &str, value: Value) {
        self.fields.insert(name.to_string(), value);
    }
}



/// Implements advanced domain-specific language features for the Turbulance language
pub struct DomainExtensions {
    // ... existing code ...
    
    /// Domain-specific keyword registry for custom syntax
    pub domain_keywords: HashMap<String, DomainKeywordHandler>,
    
    /// Custom type definitions for domain-specific types
    pub domain_types: HashMap<String, DomainTypeDefinition>,
}

/// Represents a domain-specific type with validation and operations
#[derive(Clone)]
pub struct DomainTypeDefinition {
    /// Name of the domain type
    pub name: String,
    
    /// Validation function for checking if a value is of this type
    pub validator: Arc<dyn Fn(&Value) -> bool + Send + Sync>,
    
    /// Type-specific operations
    pub operations: HashMap<String, DomainOperation>,
    
    /// Documentation for the type
    pub documentation: String,
}

/// Represents a domain-specific operation on a type
#[derive(Clone)]
pub struct DomainOperation {
    /// Name of the operation
    pub name: String,
    
    /// Function implementing the operation
    pub implementation: Arc<dyn Fn(&mut Context, &[Value]) -> Result<Value> + Send + Sync>,
    
    /// Parameter information
    pub parameters: Vec<ParameterInfo>,
    
    /// Documentation for the operation
    pub documentation: String,
}

/// Information about operation parameters
#[derive(Clone)]
pub struct ParameterInfo {
    /// Name of the parameter
    pub name: String,
    
    /// Whether the parameter is optional
    pub optional: bool,
    
    /// Expected type of the parameter
    pub expected_type: String,
    
    /// Default value for optional parameters
    pub default_value: Option<Value>,
}

/// Function type for handling domain-specific keywords
pub type DomainKeywordHandler = Arc<dyn Fn(&mut Context, &[Node]) -> Result<Value> + Send + Sync>;

impl DomainExtensions {
    /// Create a new DomainExtensions instance
    pub fn new() -> Self {
        Self {
            domain_keywords: HashMap::new(),
            domain_types: HashMap::new(),
        }
    }
    
    /// Registers a new domain-specific type
    pub fn register_domain_type(&mut self, type_def: DomainTypeDefinition) {
        self.domain_types.insert(type_def.name.clone(), type_def);
    }
    
    /// Registers a new domain-specific keyword
    pub fn register_domain_keyword(&mut self, keyword: String, handler: DomainKeywordHandler) {
        self.domain_keywords.insert(keyword, handler);
    }
    
    /// Validates if a value matches a domain-specific type
    pub fn validate_domain_type(&self, type_name: &str, value: &Value) -> bool {
        if let Some(type_def) = self.domain_types.get(type_name) {
            return (type_def.validator)(value);
        }
        false
    }
    
    /// Executes a domain-specific operation on a value (simplified)
    pub fn execute_domain_operation(
        &self,
        type_name: &str,
        operation: &str,
        _value: &Value,
        _args: &[Value]
    ) -> Result<Value> {
        // Simplified implementation
        match (type_name, operation) {
            ("string", "transform") => Ok(Value::String("transformed".to_string())),
            _ => Err(Error::runtime(&format!("Operation '{}' not found for domain type '{}'", operation, type_name)))
        }
    }
} 