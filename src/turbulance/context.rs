use std::collections::HashMap;
use std::time::{Instant, Duration};
use crate::error::{Error, Result, ErrorReporter};

/// Execution context for Turbulance scripts
pub struct Context {
    /// Variables in the current scope
    variables: HashMap<String, Value>,
    /// Function definitions
    functions: HashMap<String, Function>,
    /// Parent context in the chain
    parent: Option<Box<Context>>,
    /// Execution state
    execution_state: ExecutionState,
    /// Error reporter
    error_reporter: ErrorReporter,
}

/// Tracks the execution state of a Turbulance script
#[derive(Debug, Clone)]
pub struct ExecutionState {
    /// Start time of execution
    pub start_time: Option<Instant>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Current execution depth (for recursive calls)
    pub depth: usize,
    /// Maximum allowed execution depth
    pub max_depth: usize,
    /// Whether execution is in a recovery mode
    pub recovery_mode: bool,
    /// Execution path (function calls)
    pub call_stack: Vec<String>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self {
            start_time: None,
            duration: None,
            depth: 0,
            max_depth: 100, // Default max recursion depth
            recovery_mode: false,
            call_stack: Vec::new(),
            metrics: HashMap::new(),
        }
    }
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
            execution_state: ExecutionState::default(),
            error_reporter: ErrorReporter::new(),
        }
    }
    
    /// Create a child context
    pub fn new_child(&self) -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: Some(Box::new(self.clone())),
            execution_state: self.execution_state.clone(),
            error_reporter: ErrorReporter::new(),
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
        self.execution_state = ExecutionState::default();
        self.error_reporter = ErrorReporter::new();
    }
    
    /// Begin execution tracking
    pub fn begin_execution(&mut self) {
        self.execution_state.start_time = Some(Instant::now());
        self.execution_state.call_stack.clear();
        self.execution_state.depth = 0;
        self.execution_state.recovery_mode = false;
    }
    
    /// End execution tracking
    pub fn end_execution(&mut self) {
        if let Some(start) = self.execution_state.start_time {
            self.execution_state.duration = Some(start.elapsed());
        }
    }
    
    /// Enter a function call
    pub fn enter_function(&mut self, function_name: &str) -> Result<()> {
        self.execution_state.depth += 1;
        
        // Check for stack overflow
        if self.execution_state.depth > self.execution_state.max_depth {
            return Err(Error::Runtime(format!(
                "Stack overflow: exceeded maximum recursion depth of {}",
                self.execution_state.max_depth
            )));
        }
        
        self.execution_state.call_stack.push(function_name.to_string());
        Ok(())
    }
    
    /// Exit a function call
    pub fn exit_function(&mut self) {
        if !self.execution_state.call_stack.is_empty() {
            self.execution_state.call_stack.pop();
        }
        if self.execution_state.depth > 0 {
            self.execution_state.depth -= 1;
        }
    }
    
    /// Record a performance metric
    pub fn record_metric(&mut self, name: &str, value: f64) {
        self.execution_state.metrics.insert(name.to_string(), value);
    }
    
    /// Get the current call stack as a string
    pub fn get_call_stack(&self) -> String {
        self.execution_state.call_stack.join(" -> ")
    }
    
    /// Get execution performance report
    pub fn get_performance_report(&self) -> String {
        let mut report = String::new();
        
        if let Some(duration) = self.execution_state.duration {
            report.push_str(&format!("Execution time: {:?}\n", duration));
        }
        
        if !self.execution_state.metrics.is_empty() {
            report.push_str("Performance metrics:\n");
            for (name, value) in &self.execution_state.metrics {
                report.push_str(&format!("  {}: {}\n", name, value));
            }
        }
        
        report
    }
    
    /// Add an error to the error reporter
    pub fn add_error(&mut self, error: Error) {
        self.error_reporter.add_error(error);
    }
    
    /// Get the error reporter
    pub fn error_reporter(&self) -> &ErrorReporter {
        &self.error_reporter
    }
    
    /// Get a mutable reference to the error reporter
    pub fn error_reporter_mut(&mut self) -> &mut ErrorReporter {
        &mut self.error_reporter
    }
    
    /// Enter recovery mode for graceful error handling
    pub fn enter_recovery_mode(&mut self) {
        self.execution_state.recovery_mode = true;
    }
    
    /// Exit recovery mode
    pub fn exit_recovery_mode(&mut self) {
        self.execution_state.recovery_mode = false;
    }
    
    /// Check if in recovery mode
    pub fn is_in_recovery_mode(&self) -> bool {
        self.execution_state.recovery_mode
    }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Self {
            variables: self.variables.clone(),
            functions: self.functions.clone(),
            parent: self.parent.clone(),
            execution_state: self.execution_state.clone(),
            error_reporter: ErrorReporter::new(), // Don't propagate errors in cloned contexts
        }
    }
} 