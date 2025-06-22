//! Turbulance execution engine

use crate::interpreter::{Interpreter, Value};
use crate::script::Script;
use crate::context::Context;
use crate::error::Result;
use std::time::{Duration, Instant};

/// Main execution engine for Turbulance
pub struct Engine {
    interpreter: Interpreter,
    context: Context,
    timeout: Option<Duration>,
}

impl Engine {
    /// Create a new engine with default settings
    pub fn new() -> Self {
        Self {
            interpreter: Interpreter::new(),
            context: Context::new(),
            timeout: Some(Duration::from_secs(30)), // 30 second default timeout
        }
    }

    /// Create an engine with custom context
    pub fn with_context(context: Context) -> Self {
        Self {
            interpreter: Interpreter::new(),
            context,
            timeout: Some(Duration::from_secs(30)),
        }
    }

    /// Set execution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Remove execution timeout
    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    /// Execute a script
    pub fn execute(&mut self, script: &Script) -> Result<Value> {
        let start_time = Instant::now();
        
        // Check timeout before execution
        if let Some(timeout) = self.timeout {
            if start_time.elapsed() > timeout {
                return Err(crate::error::TurbulanceError::ProcessingTimeout { 
                    timeout_ms: timeout.as_millis() as u64 
                });
            }
        }

        // Execute the script's AST
        self.interpreter.execute(script.ast())
    }

    /// Execute source code directly
    pub fn execute_source(&mut self, source: &str) -> Result<Value> {
        let script = Script::from_source(source)?;
        self.execute(&script)
    }

    /// Get the current context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get a mutable reference to the context
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Set a global variable in the execution environment
    pub fn set_global<S: Into<String>>(&mut self, name: S, value: Value) {
        // For now, this would need to be implemented in the interpreter
        // self.interpreter.set_global(name.into(), value);
    }

    /// Get a global variable from the execution environment
    pub fn get_global(&self, name: &str) -> Option<Value> {
        // For now, this would need to be implemented in the interpreter
        // self.interpreter.get_global(name)
        None
    }

    /// Reset the engine state
    pub fn reset(&mut self) {
        self.interpreter = Interpreter::new();
        self.context = Context::new();
    }

    /// Get execution statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            // These would be collected during execution
            executed_statements: 0,
            function_calls: 0,
            memory_usage: 0,
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine execution statistics
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Number of statements executed
    pub executed_statements: usize,
    /// Number of function calls made
    pub function_calls: usize,
    /// Approximate memory usage in bytes
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = Engine::new();
        assert!(engine.timeout.is_some());
    }

    #[test]
    fn test_engine_with_timeout() {
        let engine = Engine::new().with_timeout(Duration::from_secs(60));
        assert_eq!(engine.timeout.unwrap().as_secs(), 60);
    }

    #[test]
    fn test_engine_without_timeout() {
        let engine = Engine::new().without_timeout();
        assert!(engine.timeout.is_none());
    }

    #[test]
    fn test_execute_simple() {
        let mut engine = Engine::new();
        let result = engine.execute_source("42").unwrap();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_execute_arithmetic() {
        let mut engine = Engine::new();
        let result = engine.execute_source("2 + 3 * 4").unwrap();
        assert_eq!(result, Value::Number(14.0));
    }

    #[test]
    fn test_execute_assignment() {
        let mut engine = Engine::new();
        let result = engine.execute_source("item x = 42").unwrap();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_reset() {
        let mut engine = Engine::new();
        engine.execute_source("item x = 42").unwrap();
        engine.reset();
        // After reset, x should not be defined
        // (This test would need variable lookup to be implemented)
    }
} 