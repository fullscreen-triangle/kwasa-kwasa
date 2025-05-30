//! Domain Extensions for Turbulance
//! 
//! This module integrates all domain extensions with the Turbulance language.

use crate::genomic::prelude::*;
use crate::spectrometry::prelude::*;
use crate::chemistry::prelude::*;
use crate::pattern::prelude::*;
use crate::turbulance::interpreter::{Interpreter, Value, RuntimeError, ObjectRef, Function};
use std::collections::HashMap;
use std::sync::Arc;
use crate::error::{Error, Result};
use crate::turbulance::ast::Node;
use crate::turbulance::context::Context;

/// Register all domain extensions with the interpreter
pub fn register_domain_extensions(interpreter: &mut Interpreter) -> Result<(), RuntimeError> {
    register_genomic_extensions(interpreter)?;
    register_spectrometry_extensions(interpreter)?;
    register_chemistry_extensions(interpreter)?;
    register_pattern_extensions(interpreter)?;
    Ok(())
}

/// Register genomic extensions with the interpreter
fn register_genomic_extensions(interpreter: &mut Interpreter) -> Result<(), RuntimeError> {
    let genomic_module = create_module("genomic");
    
    // Register NucleotideSequence constructor
    genomic_module.borrow_mut().set(
        "NucleotideSequence", 
        Value::Object(ObjectRef::new(NucleotideSequenceConstructor {}))
    );
    
    // Register NucleotideOperations constructor
    genomic_module.borrow_mut().set(
        "NucleotideOperations", 
        Value::Object(ObjectRef::new(NucleotideOperationsConstructor {}))
    );
    
    interpreter.set_global("genomic", Value::Module(genomic_module));
    Ok(())
}

/// Register spectrometry extensions with the interpreter
fn register_spectrometry_extensions(interpreter: &mut Interpreter) -> Result<(), RuntimeError> {
    let spectrometry_module = create_module("spectrometry");
    
    // Register MassSpectrum constructor
    spectrometry_module.borrow_mut().set(
        "MassSpectrum", 
        Value::Object(ObjectRef::new(MassSpectrumConstructor {}))
    );
    
    // Register SpectrumOperations constructor
    spectrometry_module.borrow_mut().set(
        "SpectrumOperations", 
        Value::Object(ObjectRef::new(SpectrumOperationsConstructor {}))
    );
    
    interpreter.set_global("spectrometry", Value::Module(spectrometry_module));
    Ok(())
}

/// Register chemistry extensions with the interpreter
fn register_chemistry_extensions(interpreter: &mut Interpreter) -> Result<(), RuntimeError> {
    let chemistry_module = create_module("chemistry");
    
    // Register Molecule constructor
    chemistry_module.borrow_mut().set(
        "Molecule", 
        Value::Object(ObjectRef::new(MoleculeConstructor {}))
    );
    
    // Register MoleculeOperations constructor
    chemistry_module.borrow_mut().set(
        "MoleculeOperations", 
        Value::Object(ObjectRef::new(MoleculeOperationsConstructor {}))
    );
    
    interpreter.set_global("chemistry", Value::Module(chemistry_module));
    Ok(())
}

/// Register pattern extensions with the interpreter
fn register_pattern_extensions(interpreter: &mut Interpreter) -> Result<(), RuntimeError> {
    let pattern_module = create_module("pattern");
    
    // Register PatternAnalyzer constructor
    pattern_module.borrow_mut().set(
        "PatternAnalyzer", 
        Value::Object(ObjectRef::new(PatternAnalyzerConstructor {}))
    );
    
    // Register OrthographicAnalyzer constructor
    pattern_module.borrow_mut().set(
        "OrthographicAnalyzer", 
        Value::Object(ObjectRef::new(OrthographicAnalyzerConstructor {}))
    );
    
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
    
    fn get(&self, name: &str) -> Option<Value> {
        self.fields.get(name).cloned()
    }
}

// Constructor types for each domain

/// NucleotideSequence constructor
#[derive(Debug)]
struct NucleotideSequenceConstructor;

/// NucleotideOperations constructor
#[derive(Debug)]
struct NucleotideOperationsConstructor;

/// MassSpectrum constructor
#[derive(Debug)]
struct MassSpectrumConstructor;

/// SpectrumOperations constructor
#[derive(Debug)]
struct SpectrumOperationsConstructor;

/// Molecule constructor
#[derive(Debug)]
struct MoleculeConstructor;

/// MoleculeOperations constructor
#[derive(Debug)]
struct MoleculeOperationsConstructor;

/// PatternAnalyzer constructor
#[derive(Debug)]
struct PatternAnalyzerConstructor;

/// OrthographicAnalyzer constructor
#[derive(Debug)]
struct OrthographicAnalyzerConstructor;

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
    // ... existing code ...
    
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
    
    /// Executes a domain-specific operation on a value
    pub fn execute_domain_operation(
        &self,
        ctx: &mut Context,
        type_name: &str,
        operation: &str,
        value: &Value,
        args: &[Value]
    ) -> Result<Value> {
        if let Some(type_def) = self.domain_types.get(type_name) {
            if let Some(op) = type_def.operations.get(operation) {
                let mut all_args = vec![value.clone()];
                all_args.extend_from_slice(args);
                return (op.implementation)(ctx, &all_args);
            }
            return Err(Error::turbulance(format!("Operation '{}' not found for domain type '{}'", operation, type_name)));
        }
        Err(Error::turbulance(format!("Domain type '{}' not registered", type_name)))
    }
    
    /// Handles domain-specific keyword usage in the AST
    pub fn handle_domain_keyword(&self, ctx: &mut Context, keyword: &str, args: &[Node]) -> Result<Value> {
        if let Some(handler) = self.domain_keywords.get(keyword) {
            return handler(ctx, args);
        }
        Err(Error::turbulance(format!("Domain keyword '{}' not registered", keyword)))
    }
} 