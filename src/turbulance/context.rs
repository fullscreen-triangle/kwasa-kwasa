use std::collections::HashMap;
use crate::error::{Error, Result};

/// Execution context for Turbulance scripts
pub struct Context {
    /// Variables in the current scope
    variables: HashMap<String, Value>,
    /// Function definitions
    functions: HashMap<String, Function>,
    /// Parent context in the chain
    parent: Option<Box<Context>>,
}

/// A value in the Turbulance language
#[derive(Debug, Clone)]
pub enum Value {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Array value
    Array(Vec<Value>),
    /// Object value
    Object(HashMap<String, Value>),
    /// Function reference
    Function(String),
}

/// A function definition
#[derive(Debug, Clone)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Parameter names
    pub parameters: Vec<String>,
    /// Function body
    pub body: String,
    /// Is this a native function?
    pub native: bool,
}

impl Context {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: None,
        }
    }
    
    /// Create a child context
    pub fn new_child(&self) -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }
    
    /// Set a variable in the current scope
    pub fn set_variable(&mut self, name: &str, value: Value) {
        self.variables.insert(name.to_string(), value);
    }
    
    /// Get a variable from this context or parent contexts
    pub fn get_variable(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.variables.get(name) {
            return Some(value.clone());
        }
        
        // Check parent context
        if let Some(parent) = &self.parent {
            return parent.get_variable(name);
        }
        
        None
    }
    
    /// Define a function
    pub fn define_function(&mut self, function: Function) {
        self.functions.insert(function.name.clone(), function);
    }
    
    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<Function> {
        if let Some(function) = self.functions.get(name) {
            return Some(function.clone());
        }
        
        // Check parent context
        if let Some(parent) = &self.parent {
            return parent.get_function(name);
        }
        
        None
    }
    
    /// Reset this context
    pub fn reset(&mut self) {
        self.variables.clear();
        self.functions.clear();
        self.parent = None;
    }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
            functions: self.functions.clone(),
            parent: self.parent.clone(),
        }
    }
} 