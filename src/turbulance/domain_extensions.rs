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