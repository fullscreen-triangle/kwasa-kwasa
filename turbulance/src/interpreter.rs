//! Interpreter for executing Turbulance code

use crate::ast::{Node, BinaryOp, UnaryOp, TextOp, TextUnit};
use crate::error::{TurbulanceError, Result};
use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

/// Runtime values in Turbulance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Numeric value
    Number(f64),
    /// String value
    String(String),
    /// Boolean value
    Boolean(bool),
    /// Null value
    Null,
    /// Array of values
    Array(Vec<Value>),
    /// Object with string keys
    Object(HashMap<String, Value>),
    /// Text unit for semantic operations
    TextUnit(TextUnit),
    /// Function (represented as name for now)
    Function(String),
}

impl Value {
    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Null => false,
            Value::Number(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(arr) => !arr.is_empty(),
            Value::Object(obj) => !obj.is_empty(),
            Value::TextUnit(unit) => !unit.content.is_empty(),
            Value::Function(_) => true,
        }
    }

    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Number(_) => "Number",
            Value::String(_) => "String",
            Value::Boolean(_) => "Boolean",
            Value::Null => "Null",
            Value::Array(_) => "Array",
            Value::Object(_) => "Object",
            Value::TextUnit(_) => "TextUnit",
            Value::Function(_) => "Function",
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Boolean(b) => b.to_string(),
            Value::Null => "null".to_string(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string()))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::TextUnit(unit) => unit.content.clone(),
            Value::Function(name) => format!("function {}", name),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Execution environment
#[derive(Debug, Clone)]
pub struct Environment {
    variables: HashMap<String, Value>,
    parent: Option<Box<Environment>>,
}

impl Environment {
    /// Create a new empty environment
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            parent: None,
        }
    }

    /// Create a new environment with a parent
    pub fn with_parent(parent: Environment) -> Self {
        Self {
            variables: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Define a variable
    pub fn define(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    /// Get a variable value
    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.variables.get(name) {
            Some(value.clone())
        } else if let Some(parent) = &self.parent {
            parent.get(name)
        } else {
            None
        }
    }

    /// Set a variable value (in the environment where it's defined)
    pub fn set(&mut self, name: &str, value: Value) -> Result<()> {
        if self.variables.contains_key(name) {
            self.variables.insert(name.to_string(), value);
            Ok(())
        } else if let Some(parent) = &mut self.parent {
            parent.set(name, value)
        } else {
            Err(TurbulanceError::name_error(name))
        }
    }
}

/// Interpreter for executing Turbulance AST
pub struct Interpreter {
    environment: Environment,
    globals: HashMap<String, Value>,
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        let mut interpreter = Self {
            environment: Environment::new(),
            globals: HashMap::new(),
        };
        
        interpreter.setup_globals();
        interpreter
    }

    /// Setup global functions and constants
    fn setup_globals(&mut self) {
        // Mathematical constants
        self.globals.insert("PI".to_string(), Value::Number(std::f64::consts::PI));
        self.globals.insert("E".to_string(), Value::Number(std::f64::consts::E));
        
        // Built-in functions (represented as function names for now)
        self.globals.insert("abs".to_string(), Value::Function("abs".to_string()));
        self.globals.insert("sqrt".to_string(), Value::Function("sqrt".to_string()));
        self.globals.insert("sin".to_string(), Value::Function("sin".to_string()));
        self.globals.insert("cos".to_string(), Value::Function("cos".to_string()));
        self.globals.insert("tan".to_string(), Value::Function("tan".to_string()));
        self.globals.insert("log".to_string(), Value::Function("log".to_string()));
        
        // Text processing functions
        self.globals.insert("understand_text".to_string(), Value::Function("understand_text".to_string()));
        self.globals.insert("load_data".to_string(), Value::Function("load_data".to_string()));
        self.globals.insert("train_model".to_string(), Value::Function("train_model".to_string()));
        self.globals.insert("assess_quality".to_string(), Value::Function("assess_quality".to_string()));
        
        // Scientific functions
        self.globals.insert("pearson".to_string(), Value::Function("pearson".to_string()));
        self.globals.insert("t_test".to_string(), Value::Function("t_test".to_string()));
        self.globals.insert("cohen_d".to_string(), Value::Function("cohen_d".to_string()));
        self.globals.insert("cross_validate".to_string(), Value::Function("cross_validate".to_string()));
        
        // Copy globals to environment
        for (name, value) in &self.globals {
            self.environment.define(name.clone(), value.clone());
        }
    }

    /// Execute a node
    pub fn execute(&mut self, node: &Node) -> Result<Value> {
        self.evaluate(node)
    }

    /// Evaluate a node and return its value
    fn evaluate(&mut self, node: &Node) -> Result<Value> {
        match node {
            Node::Number { value, .. } => Ok(Value::Number(*value)),
            
            Node::String { value, .. } => Ok(Value::String(value.clone())),
            
            Node::Boolean { value, .. } => Ok(Value::Boolean(*value)),
            
            Node::Null { .. } => Ok(Value::Null),
            
            Node::Identifier { name, .. } => {
                self.environment.get(name)
                    .ok_or_else(|| TurbulanceError::name_error(name))
            }
            
            Node::Array { elements, .. } => {
                let mut values = Vec::new();
                for element in elements {
                    values.push(self.evaluate(element)?);
                }
                Ok(Value::Array(values))
            }
            
            Node::Object { fields, .. } => {
                let mut object = HashMap::new();
                for (key, value_node) in fields {
                    let value = self.evaluate(value_node)?;
                    object.insert(key.clone(), value);
                }
                Ok(Value::Object(object))
            }
            
            Node::BinaryOp { left, operator, right, .. } => {
                self.evaluate_binary_op(left, operator, right)
            }
            
            Node::UnaryOp { operator, operand, .. } => {
                self.evaluate_unary_op(operator, operand)
            }
            
            Node::Call { callee, arguments, .. } => {
                self.evaluate_call(callee, arguments)
            }
            
            Node::Member { object, property, .. } => {
                self.evaluate_member_access(object, property)
            }
            
            Node::Assignment { target, value, .. } => {
                self.evaluate_assignment(target, value)
            }
            
            Node::Block { statements, .. } => {
                self.evaluate_block(statements)
            }
            
            Node::Program { statements, .. } => {
                self.evaluate_block(statements)
            }
            
            Node::ExpressionStatement { expression, .. } => {
                self.evaluate(expression)
            }
            
            Node::Return { value, .. } => {
                if let Some(val) = value {
                    self.evaluate(val)
                } else {
                    Ok(Value::Null)
                }
            }
            
            Node::FunctionDecl { name, .. } => {
                // For now, just store function as a value
                let func_value = Value::Function(name.clone());
                self.environment.define(name.clone(), func_value.clone());
                Ok(func_value)
            }
            
            Node::ProjectDecl { name, body, .. } => {
                // Execute project body
                self.evaluate(body)
            }
            
            Node::Proposition { body, .. } => {
                // Execute proposition body
                self.evaluate(body)
            }
            
            Node::Given { condition, then_branch, else_branch, .. } => {
                let condition_value = self.evaluate(condition)?;
                if condition_value.is_truthy() {
                    self.evaluate(then_branch)
                } else if let Some(else_node) = else_branch {
                    self.evaluate(else_node)
                } else {
                    Ok(Value::Null)
                }
            }
            
            Node::Within { target: _, body, .. } => {
                // For now, just execute the body
                self.evaluate(body)
            }
            
            Node::Considering { items: _, body, .. } => {
                // For now, just execute the body
                self.evaluate(body)
            }
            
            Node::Ensure { condition, .. } => {
                let value = self.evaluate(condition)?;
                if !value.is_truthy() {
                    return Err(TurbulanceError::runtime("Ensure condition failed"));
                }
                Ok(value)
            }
            
            Node::Research { query, .. } => {
                // For now, just return the query as a string
                let query_value = self.evaluate(query)?;
                Ok(Value::String(format!("Research: {}", query_value.to_string())))
            }
            
            Node::TextOperation { operation, target, arguments, .. } => {
                self.evaluate_text_operation(operation, target, arguments)
            }
            
            _ => {
                Err(TurbulanceError::runtime("Unsupported node type"))
            }
        }
    }

    /// Evaluate binary operations
    fn evaluate_binary_op(&mut self, left: &Node, op: &BinaryOp, right: &Node) -> Result<Value> {
        let left_val = self.evaluate(left)?;
        let right_val = self.evaluate(right)?;
        
        match op {
            BinaryOp::Add => self.add_values(left_val, right_val),
            BinaryOp::Subtract => self.subtract_values(left_val, right_val),
            BinaryOp::Multiply => self.multiply_values(left_val, right_val),
            BinaryOp::Divide => self.divide_values(left_val, right_val),
            BinaryOp::Equal => Ok(Value::Boolean(self.values_equal(&left_val, &right_val))),
            BinaryOp::NotEqual => Ok(Value::Boolean(!self.values_equal(&left_val, &right_val))),
            BinaryOp::LessThan => self.compare_values(&left_val, &right_val, |a, b| a < b),
            BinaryOp::GreaterThan => self.compare_values(&left_val, &right_val, |a, b| a > b),
            BinaryOp::LessThanEqual => self.compare_values(&left_val, &right_val, |a, b| a <= b),
            BinaryOp::GreaterThanEqual => self.compare_values(&left_val, &right_val, |a, b| a >= b),
            BinaryOp::And => Ok(Value::Boolean(left_val.is_truthy() && right_val.is_truthy())),
            BinaryOp::Or => Ok(Value::Boolean(left_val.is_truthy() || right_val.is_truthy())),
            BinaryOp::Pipe => self.pipe_values(left_val, right_val),
            _ => Err(TurbulanceError::runtime(&format!("Unsupported binary operator: {:?}", op))),
        }
    }

    /// Evaluate unary operations
    fn evaluate_unary_op(&mut self, op: &UnaryOp, operand: &Node) -> Result<Value> {
        let value = self.evaluate(operand)?;
        
        match op {
            UnaryOp::Negate => match value {
                Value::Number(n) => Ok(Value::Number(-n)),
                _ => Err(TurbulanceError::type_error("Cannot negate non-numeric value")),
            },
            UnaryOp::Not => Ok(Value::Boolean(!value.is_truthy())),
            UnaryOp::Plus => match value {
                Value::Number(n) => Ok(Value::Number(n)),
                _ => Err(TurbulanceError::type_error("Cannot apply unary plus to non-numeric value")),
            },
        }
    }

    /// Evaluate function calls
    fn evaluate_call(&mut self, callee: &Node, arguments: &Vec<Node>) -> Result<Value> {
        let func = self.evaluate(callee)?;
        let mut arg_values = Vec::new();
        
        for arg in arguments {
            arg_values.push(self.evaluate(arg)?);
        }
        
        match func {
            Value::Function(name) => self.call_builtin_function(&name, arg_values),
            _ => Err(TurbulanceError::type_error("Cannot call non-function value")),
        }
    }

    /// Call built-in functions
    fn call_builtin_function(&mut self, name: &str, args: Vec<Value>) -> Result<Value> {
        match name {
            "abs" => {
                if args.len() != 1 {
                    return Err(TurbulanceError::argument_error("abs expects 1 argument"));
                }
                match &args[0] {
                    Value::Number(n) => Ok(Value::Number(n.abs())),
                    _ => Err(TurbulanceError::type_error("abs expects a number")),
                }
            }
            
            "sqrt" => {
                if args.len() != 1 {
                    return Err(TurbulanceError::argument_error("sqrt expects 1 argument"));
                }
                match &args[0] {
                    Value::Number(n) => Ok(Value::Number(n.sqrt())),
                    _ => Err(TurbulanceError::type_error("sqrt expects a number")),
                }
            }
            
            "understand_text" => {
                if args.len() != 1 {
                    return Err(TurbulanceError::argument_error("understand_text expects 1 argument"));
                }
                match &args[0] {
                    Value::String(s) => {
                        let text_unit = TextUnit::new(s.clone()).with_confidence(0.85);
                        Ok(Value::TextUnit(text_unit))
                    }
                    _ => Err(TurbulanceError::type_error("understand_text expects a string")),
                }
            }
            
            "load_data" => {
                if args.len() != 1 {
                    return Err(TurbulanceError::argument_error("load_data expects 1 argument"));
                }
                // Mock data loading - return sample data
                Ok(Value::Array(vec![
                    Value::Number(1.0),
                    Value::Number(2.0),
                    Value::Number(3.0),
                    Value::Number(4.0),
                    Value::Number(5.0),
                ]))
            }
            
            "pearson" => {
                if args.len() != 2 {
                    return Err(TurbulanceError::argument_error("pearson expects 2 arguments"));
                }
                // Mock correlation calculation
                Ok(Value::Number(0.85))
            }
            
            _ => {
                // For unknown functions, return a mock result
                Ok(Value::String(format!("Result of {}", name)))
            }
        }
    }

    /// Evaluate member access
    fn evaluate_member_access(&mut self, object: &Node, property: &str) -> Result<Value> {
        let obj_value = self.evaluate(object)?;
        
        match obj_value {
            Value::Object(obj) => {
                obj.get(property)
                    .cloned()
                    .ok_or_else(|| TurbulanceError::runtime(&format!("Property '{}' not found", property)))
            }
            Value::TextUnit(unit) => {
                match property {
                    "content" => Ok(Value::String(unit.content)),
                    "confidence" => Ok(Value::Number(unit.confidence)),
                    _ => Err(TurbulanceError::runtime(&format!("TextUnit has no property '{}'", property))),
                }
            }
            _ => Err(TurbulanceError::type_error("Cannot access property of non-object value")),
        }
    }

    /// Evaluate assignment
    fn evaluate_assignment(&mut self, target: &Node, value: &Node) -> Result<Value> {
        let val = self.evaluate(value)?;
        
        if let Node::Identifier { name, .. } = target {
            self.environment.define(name.clone(), val.clone());
            Ok(val)
        } else {
            Err(TurbulanceError::runtime("Invalid assignment target"))
        }
    }

    /// Evaluate a block of statements
    fn evaluate_block(&mut self, statements: &[Node]) -> Result<Value> {
        let mut result = Value::Null;
        
        for statement in statements {
            result = self.evaluate(statement)?;
            
            // Handle early returns
            if let Node::Return { .. } = statement {
                break;
            }
        }
        
        Ok(result)
    }

    /// Evaluate text operations
    fn evaluate_text_operation(&mut self, _op: &TextOp, target: &Node, _args: &[Node]) -> Result<Value> {
        let target_val = self.evaluate(target)?;
        
        // For now, just return the target as a text unit
        match target_val {
            Value::String(s) => {
                let text_unit = TextUnit::new(s).with_confidence(0.9);
                Ok(Value::TextUnit(text_unit))
            }
            Value::TextUnit(unit) => Ok(Value::TextUnit(unit)),
            _ => Err(TurbulanceError::type_error("Text operations require string or TextUnit")),
        }
    }

    // Helper methods for value operations

    fn add_values(&self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(a + &b)),
            (Value::Array(mut a), Value::Array(b)) => {
                a.extend(b);
                Ok(Value::Array(a))
            }
            _ => Err(TurbulanceError::type_error("Cannot add these value types")),
        }
    }

    fn subtract_values(&self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
            _ => Err(TurbulanceError::type_error("Cannot subtract these value types")),
        }
    }

    fn multiply_values(&self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
            (Value::String(s), Value::Number(n)) => {
                if n >= 0.0 && n.fract() == 0.0 {
                    Ok(Value::String(s.repeat(n as usize)))
                } else {
                    Err(TurbulanceError::runtime("String repetition requires non-negative integer"))
                }
            }
            _ => Err(TurbulanceError::type_error("Cannot multiply these value types")),
        }
    }

    fn divide_values(&self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(a), Value::Number(b)) => {
                if b == 0.0 {
                    Err(TurbulanceError::runtime("Division by zero"))
                } else {
                    Ok(Value::Number(a / b))
                }
            }
            _ => Err(TurbulanceError::type_error("Cannot divide these value types")),
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }

    fn compare_values<F>(&self, a: &Value, b: &Value, op: F) -> Result<Value>
    where
        F: Fn(f64, f64) -> bool,
    {
        match (a, b) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(op(*a, *b))),
            _ => Err(TurbulanceError::type_error("Cannot compare non-numeric values")),
        }
    }

    fn pipe_values(&self, left: Value, _right: Value) -> Result<Value> {
        // For now, just return the left value
        Ok(left)
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;

    fn interpret_source(source: &str) -> Result<Value> {
        let tokens = tokenize(source)?;
        let ast = parse(tokens)?;
        let mut interpreter = Interpreter::new();
        interpreter.execute(&ast)
    }

    #[test]
    fn test_number_literal() {
        let result = interpret_source("42").unwrap();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_string_literal() {
        let result = interpret_source(r#""hello""#).unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[test]
    fn test_arithmetic() {
        let result = interpret_source("2 + 3 * 4").unwrap();
        assert_eq!(result, Value::Number(14.0));
    }

    #[test]
    fn test_assignment() {
        let result = interpret_source("item x = 42").unwrap();
        assert_eq!(result, Value::Number(42.0));
    }

    #[test]
    fn test_function_call() {
        let result = interpret_source("abs(-5)").unwrap();
        assert_eq!(result, Value::Number(5.0));
    }

    #[test]
    fn test_given_statement() {
        let result = interpret_source(r#"
            given true:
                return "success"
            alternatively:
                return "failure"
        "#).unwrap();
        assert_eq!(result, Value::String("success".to_string()));
    }

    #[test]
    fn test_understand_text() {
        let result = interpret_source(r#"understand_text("test data")"#).unwrap();
        if let Value::TextUnit(unit) = result {
            assert_eq!(unit.content, "test data");
            assert_eq!(unit.confidence, 0.85);
        } else {
            panic!("Expected TextUnit");
        }
    }
} 