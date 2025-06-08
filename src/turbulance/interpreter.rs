use std::collections::HashMap;
use crate::turbulance::ast::{Node, BinaryOp, UnaryOp, TextOp};
use crate::turbulance::TurbulanceError;
use crate::text_unit::boundary::TextUnit;
use crate::turbulance::stdlib::StdLib;
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;
use crate::turbulance::context::{Context, Value as ContextValue};

// Define Result type for Turbulance operations
type Result<T> = std::result::Result<T, TurbulanceError>;

// Define Statement as an alias for Node
type Statement = Node;

/// Native function type that can be called from Turbulance
pub type NativeFunction = Box<dyn Fn(Vec<Value>) -> Result<Value> + Send + Sync>;

/// Value types in the Turbulance language
#[derive(Clone, Debug)]
pub enum Value {
    Number(f64),
    String(String),
    Boolean(bool),
    Function(Function),
    NativeFunction(NativeFunction),
    Array(Vec<Value>),
    Object(std::collections::HashMap<String, Value>),
    Module(ObjectRef),
    TextUnit(TextUnit),
    List(Vec<Value>),
    Map(std::collections::HashMap<String, Value>),
    Null,
}

// Adding implementations for Value comparison
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => {
                // Handle NaN cases: NaN == NaN is true for our purposes
                if a.is_nan() && b.is_nan() {
                    true
                } else {
                    a == b
                }
            },
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::NativeFunction(_), Value::NativeFunction(_)) => {
                // NativeFunctions are compared by pointer equality (they're Box<dyn Fn>)
                // For our purposes, we'll consider them never equal unless they're the same instance
                false
            },
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Object(a), Value::Object(b)) => a == b,
            (Value::Module(_), Value::Module(_)) => {
                // Module references are compared by identity
                false
            },
            (Value::TextUnit(a), Value::TextUnit(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::Null, Value::Null) => true,
            _ => false, // Different variants are never equal
        }
    }
}

// Note: We implement Eq for Value even though it contains f64
// We treat NaN as equal to NaN for our use case, which is consistent with our PartialEq impl
impl Eq for Value {}

// Implement Hash for Value
impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Number(n) => {
                // Convert float to bits for hashing
                let bits = n.to_bits();
                bits.hash(state);
            },
            Value::String(s) => s.hash(state),
            Value::Boolean(b) => b.hash(state),
            Value::Function(f) => f.hash(state),
            Value::NativeFunction(_) => {
                // Function pointers aren't hashable directly
                // Use type ID as a stand-in
                std::any::TypeId::of::<NativeFunction>().hash(state);
            },
            Value::Array(arr) => {
                // Hash each element
                for v in arr {
                    v.hash(state);
                }
            },
            Value::Object(map) => {
                // Hash each key-value pair
                for (k, v) in map {
                    k.hash(state);
                    v.hash(state);
                }
            },
            Value::Module(_) => {
                // Module references aren't hashable directly
                // Use type ID as a stand-in
                std::any::TypeId::of::<ObjectRef>().hash(state);
            },
            Value::TextUnit(tu) => {
                tu.content.hash(state);
            },
            Value::List(list) => {
                // Hash each element
                for v in list {
                    v.hash(state);
                }
            },
            Value::Map(map) => {
                // Hash each key-value pair
                for (k, v) in map {
                    k.hash(state);
                    v.hash(state);
                }
            },
            Value::Null => 0.hash(state),
        }
    }
}

// Add Hash implementation for Function
impl std::hash::Hash for Function {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.params.hash(state);
        // We can't hash the Statement directly, so use a proxy value
        self.body.to_string().hash(state);
    }
}

// Add PartialEq implementation for Function
impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.params == other.params && 
        self.body.to_string() == other.body.to_string()
    }
}

// Add Eq implementation for Function
impl Eq for Function {}

#[derive(Clone, Debug)]
pub struct Function {
    params: Vec<String>,
    body: Box<Statement>,
    closure: Environment,
}

#[derive(Clone, Debug)]
struct Environment {
    values: HashMap<String, Value>,
    enclosing: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    fn new() -> Self {
        Environment {
            values: HashMap::new(),
            enclosing: None,
        }
    }
    
    fn with_enclosing(enclosing: Rc<RefCell<Environment>>) -> Self {
        Environment {
            values: HashMap::new(),
            enclosing: Some(enclosing),
        }
    }
    
    fn define(&mut self, name: String, value: Value) {
        self.values.insert(name, value);
    }
    
    fn get(&self, name: &str) -> Option<Value> {
        match self.values.get(name) {
            Some(value) => Some(value.clone()),
            None => {
                if let Some(enclosing) = &self.enclosing {
                    enclosing.borrow().get(name)
                } else {
                    None
                }
            }
        }
    }
    
    fn assign(&mut self, name: &str, value: Value) -> Result<()> {
        if self.values.contains_key(name) {
            self.values.insert(name.to_string(), value);
            Ok(())
        } else if let Some(enclosing) = &self.enclosing {
            enclosing.borrow_mut().assign(name, value)
        } else {
            Err(TurbulanceError::RuntimeError { 
                message: format!("Undefined variable '{}'.", name) 
            })
        }
    }
}

/// Represents the runtime environment for executing Turbulance code
pub struct Interpreter {
    /// Global symbol table for storing variables
    globals: HashMap<String, Value>,
    
    /// Stack of local scopes for nested execution contexts
    scopes: Vec<HashMap<String, Value>>,
    
    /// Standard library functions
    stdlib: StdLib,
    
    environment: Rc<RefCell<Environment>>,
    global_environment: Rc<RefCell<Environment>>,
    stdlib_loaded: bool,
}

impl Interpreter {
    /// Create a new interpreter instance
    pub fn new() -> Self {
        let global_env = Rc::new(RefCell::new(Environment::new()));
        Self {
            globals: HashMap::new(),
            scopes: vec![HashMap::new()], // Start with one scope (global)
            stdlib: StdLib::new(),
            environment: Rc::clone(&global_env),
            global_environment: global_env,
            stdlib_loaded: false,
        }
    }
    
    /// Register standard library functions
    pub fn register_stdlib_functions(&mut self, functions: HashMap<&'static str, NativeFunction>) {
        if self.stdlib_loaded {
            return;
        }
        
        let mut env = self.global_environment.borrow_mut();
        for (name, func) in functions {
            env.define(name.to_string(), Value::NativeFunction(func));
        }
        
        self.stdlib_loaded = true;
    }
    
    /// Execute a full program node
    pub fn execute(&mut self, node: &Node) -> Result<Value> {
        self.evaluate(node)
    }
    
    /// Evaluate a node and return its value
    fn evaluate(&mut self, node: &Node) -> Result<Value> {
        match node {
            // Literals
            Node::StringLiteral(value, _) => Ok(Value::String(value.clone())),
            Node::NumberLiteral(value, _) => Ok(Value::Number(*value)),
            Node::BoolLiteral(value, _) => Ok(Value::Boolean(*value)),
            
            // Variables
            Node::Identifier(name, span) => {
                self.lookup_variable(name)
                    .ok_or_else(|| TurbulanceError::RuntimeError { 
                        message: format!("Undefined variable '{}'", name) 
                    })
            },
            
            // Expressions
            Node::BinaryExpr { left, operator, right, span } => {
                self.evaluate_binary_expr(left, operator, right)
            },
            
            Node::UnaryExpr { operator, operand, span } => {
                self.evaluate_unary_expr(operator, operand)
            },
            
            Node::FunctionCall { function, arguments, span } => {
                self.evaluate_function_call(function, arguments)
            },
            
            // Control flow
            Node::IfExpr { condition, then_branch, else_branch, span } => {
                self.evaluate_if_expr(condition, then_branch, else_branch)
            },
            
            // Blocks and statements
            Node::Block { statements, span } => {
                self.evaluate_block(statements)
            },
            
            Node::Assignment { target, value, span } => {
                self.evaluate_assignment(target, value)
            },
            
            Node::ReturnStmt { value, span } => {
                match value {
                    Some(expr) => self.evaluate(expr),
                    None => Ok(Value::Null),
                }
            },
            
            // Turbulance-specific operations
            Node::WithinBlock { target, body, span } => {
                self.evaluate_within_block(target, body)
            },
            
            Node::GivenBlock { condition, body, span } => {
                self.evaluate_given_block(condition, body)
            },
            
            Node::EnsureStmt { condition, span } => {
                self.evaluate_ensure_stmt(condition)
            },
            
            Node::ResearchStmt { query, span } => {
                self.evaluate_research_stmt(query)
            },
            
            Node::TextOperation { operation, target, arguments, span } => {
                self.evaluate_text_operation(operation, target, arguments)
            },
            
            // Function and project declarations
            Node::FunctionDecl { name, parameters, body, span } => {
                self.evaluate_function_decl(name, parameters, body)
            },
            
            Node::ProjectDecl { name, attributes, body, span } => {
                self.evaluate_project_decl(name, attributes, body)
            },
            
            Node::SourcesDecl { sources, span } => {
                self.evaluate_sources_decl(sources)
            },
            
            // Error and other nodes
            Node::Error(message, _) => {
                Err(TurbulanceError::RuntimeError { message: message.clone() })
            },
            
            // Placeholder for any other node types that might be added later
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Unimplemented node type: {:?}", node) 
            }),
        }
    }
    
    /// Look up a variable in the current scopes
    fn lookup_variable(&self, name: &str) -> Option<Value> {
        // Check local scopes from innermost outward
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value.clone());
            }
        }
        
        // Check globals last
        self.globals.get(name).cloned()
    }
    
    /// Define a variable in the current (innermost) scope
    fn define_variable(&mut self, name: String, value: Value) {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.insert(name, value);
        }
    }
    
    /// Enter a new scope
    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    /// Exit the current scope
    fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }
    
    // Implementation of evaluation methods (placeholder stubs for now)
    
    fn evaluate_binary_expr(&mut self, left: &Node, operator: &BinaryOp, right: &Node) -> Result<Value> {
        let left_val = self.evaluate(left)?;
        let right_val = self.evaluate(right)?;
        
        match operator {
            BinaryOp::Add => self.evaluate_add(left_val, right_val),
            BinaryOp::Subtract => self.evaluate_subtract(left_val, right_val),
            BinaryOp::Multiply => self.evaluate_multiply(left_val, right_val),
            BinaryOp::Divide => self.evaluate_divide(left_val, right_val),
            BinaryOp::Equal => Ok(Value::Boolean(left_val == right_val)),
            BinaryOp::NotEqual => Ok(Value::Boolean(left_val != right_val)),
            BinaryOp::LessThan => self.evaluate_less_than(left_val, right_val),
            BinaryOp::GreaterThan => self.evaluate_greater_than(left_val, right_val),
            BinaryOp::LessThanEqual => self.evaluate_less_than_equal(left_val, right_val),
            BinaryOp::GreaterThanEqual => self.evaluate_greater_than_equal(left_val, right_val),
            BinaryOp::And => self.evaluate_and(left_val, right_val),
            BinaryOp::Or => self.evaluate_or(left_val, right_val),
            BinaryOp::Pipe => self.evaluate_pipe(left_val, right_val),
            BinaryOp::PipeForward => self.evaluate_pipe_forward(left_val, right_val),
            BinaryOp::Arrow => self.evaluate_arrow(left_val, right_val),
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Unsupported binary operator: {:?}", operator) 
            }),
        }
    }
    
    fn evaluate_unary_expr(&mut self, operator: &UnaryOp, operand: &Node) -> Result<Value> {
        let operand_value = self.evaluate(operand)?;
        
        match operator {
            UnaryOp::Negate => self.evaluate_negate(operand_value),
            UnaryOp::Not => self.evaluate_not(operand_value),
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Unsupported unary operator: {:?}", operator) 
            }),
        }
    }
    
    fn evaluate_negate(&mut self, operand: Value) -> Result<Value> {
        match operand {
            Value::Number(n) => Ok(Value::Number(-n)),
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot negate non-numeric value: {:?}", operand) 
            }),
        }
    }
    
    fn evaluate_not(&mut self, operand: Value) -> Result<Value> {
        match operand {
            Value::Boolean(b) => Ok(Value::Boolean(!b)),
            _ => {
                // Non-boolean values are coerced using is_truthy
                Ok(Value::Boolean(!self.is_truthy(&operand)))
            }
        }
    }
    
    fn evaluate_function_call(&mut self, function: &Node, arguments: &[Node]) -> Result<Value> {
        // Evaluate the function expression
        let function_value = self.evaluate(function)?;
        
        // Evaluate the arguments
        let mut evaluated_args = Vec::with_capacity(arguments.len());
        for arg in arguments {
            evaluated_args.push(self.evaluate(arg)?);
        }
        
        match function_value {
            Value::Function(func) => {
                // Check argument count
                if evaluated_args.len() != func.params.len() {
                    return Err(TurbulanceError::RuntimeError { 
                        message: format!(
                            "Expected {} arguments but got {}.", 
                            func.params.len(), 
                            evaluated_args.len()
                        ) 
                    });
                }
                
                // Create new environment with function's closure as parent
                let closure_env = Rc::new(RefCell::new(func.closure.clone()));
                let mut env = Environment::with_enclosing(closure_env);
                
                // Define parameters in the new environment
                for (param, value) in func.params.iter().zip(evaluated_args.iter()) {
                    env.define(param.clone(), value.clone());
                }
                
                // Execute the function body with the new environment
                let previous_env = self.environment.clone();
                self.environment = Rc::new(RefCell::new(env));
                
                let result = self.evaluate(&func.body)?;
                
                // Restore the previous environment
                self.environment = previous_env;
                
                Ok(result)
            },
            
            Value::NativeFunction(f) => {
                // Call native function directly
                f(evaluated_args)
            },
            
            Value::String(name) => {
                // Try to call a standard library function by name
                if let Some(stdlib) = self.get_stdlib() {
                    if stdlib.has_function(&name) {
                        return stdlib.call(&name, evaluated_args);
                    }
                }
                
                Err(TurbulanceError::RuntimeError {
                    message: format!("String '{}' is not a function", name)
                })
            },
            
            _ => Err(TurbulanceError::RuntimeError {
                message: format!("Cannot call non-function value: {:?}", function_value)
            })
        }
    }
    
    fn evaluate_if_expr(&mut self, condition: &Node, then_branch: &Node, else_branch: &Option<Box<Node>>) -> Result<Value> {
        // Evaluate the condition
        let condition_value = self.evaluate(condition)?;
        
        // Check if the condition is truthy
        if self.is_truthy(&condition_value) {
            // Execute the then branch
            self.evaluate(then_branch)
        } else if let Some(else_branch) = else_branch {
            // Execute the else branch if it exists
            self.evaluate(else_branch)
        } else {
            // No else branch, return None
            Ok(Value::Null)
        }
    }
    
    fn evaluate_block(&mut self, statements: &[Node]) -> Result<Value> {
        // Enter a new scope for the block
        self.enter_scope();
        
        // Evaluate each statement in the block
        let mut result = Value::Null;
        for statement in statements {
            match statement {
                // Handle early returns within the block
                Node::ReturnStmt { value, span } => {
                    result = match value {
                        Some(expr) => self.evaluate(expr)?,
                        None => Value::Null,
                    };
                    
                    // Exit the scope before returning
                    self.exit_scope();
                    return Ok(result);
                },
                // For normal statements, just evaluate and continue
                _ => {
                    result = self.evaluate(statement)?;
                }
            }
        }
        
        // Exit the scope before returning
        self.exit_scope();
        Ok(result)
    }
    
    fn evaluate_assignment(&mut self, target: &Node, value: &Node) -> Result<Value> {
        // Evaluate the right-hand side first
        let value = self.evaluate(value)?;
        
        // Handle different types of assignment targets
        match target {
            Node::Identifier(name, _) => {
                // Simple variable assignment
                self.define_variable(name.clone(), value.clone());
                Ok(value)
            },
            
            Node::MemberAccess { object, property, span } => {
                // Object property assignment (e.g., obj.prop = value)
                let mut object_value = self.evaluate(object)?;
                
                match (&mut object_value, property) {
                    (Value::Map(map), Node::Identifier(key, _)) => {
                        // Map property assignment
                        map.insert(key.clone(), value.clone());
                        Ok(value)
                    },
                    
                    (Value::TextUnit(text_unit), Node::Identifier(key, _)) => {
                        // TextUnit metadata assignment
                        text_unit.metadata.insert(key.clone(), value.clone());
                        Ok(value)
                    },
                    
                    _ => Err(TurbulanceError::RuntimeError {
                        message: format!("Invalid assignment target: cannot assign to property of {:?}", object_value)
                    })
                }
            },
            
            Node::IndexAccess { object, index, span } => {
                // Array/list index assignment (e.g., arr[0] = value)
                let mut object_value = self.evaluate(object)?;
                let index_value = self.evaluate(index)?;
                
                match (&mut object_value, &index_value) {
                    (Value::List(list), Value::Number(i)) => {
                        let i = *i as usize;
                        if i < list.len() {
                            list[i] = value.clone();
                            Ok(value)
                        } else {
                            Err(TurbulanceError::RuntimeError {
                                message: format!("Index out of bounds: {} for list of length {}", i, list.len())
                            })
                        }
                    },
                    
                    (Value::Map(map), Value::String(key)) => {
                        // Map indexing with string keys
                        map.insert(key.clone(), value.clone());
                        Ok(value)
                    },
                    
                    _ => Err(TurbulanceError::RuntimeError {
                        message: format!("Invalid assignment target: cannot index into {:?} with {:?}", object_value, index_value)
                    })
                }
            },
            
            _ => Err(TurbulanceError::RuntimeError {
                message: format!("Invalid assignment target: {:?}", target)
            })
        }
    }
    
    fn evaluate_within_block(&mut self, target: &Node, body: &Node) -> Result<Value> {
        // Evaluate the target expression (should result in a TextUnit)
        let target_value = self.evaluate(target)?;
        
        // The target must be a TextUnit
        let text_unit = match target_value {
            Value::TextUnit(unit) => unit,
            Value::String(text) => TextUnit::new(text),
            _ => return Err(TurbulanceError::RuntimeError {
                message: format!("'within' target must be a TextUnit or String, got: {:?}", target_value)
            })
        };
        
        // Create a special scope for the within block
        self.enter_scope();
        
        // Bind the text unit to a special variable 'this'
        self.define_variable("this".to_string(), Value::TextUnit(text_unit.clone()));
        
        // Execute the body
        let result = self.evaluate(body)?;
        
        // Check if 'this' was modified during execution
        let final_this = self.lookup_variable("this").unwrap_or(Value::TextUnit(text_unit));
        
        // Exit the within scope
        self.exit_scope();
        
        // If the body evaluates to a value other than None, return that
        // Otherwise return the (possibly modified) text unit
        if matches!(result, Value::Null) {
            Ok(final_this)
        } else {
            Ok(result)
        }
    }
    
    fn evaluate_given_block(&mut self, condition: &Node, body: &Node) -> Result<Value> {
        // Evaluate the condition
        let condition_value = self.evaluate(condition)?;
        
        // Check if the condition is truthy
        if self.is_truthy(&condition_value) {
            // Execute the body if the condition is true
            self.evaluate(body)
        } else {
            // Skip the body if the condition is false
            Ok(Value::Null)
        }
    }
    
    fn evaluate_ensure_stmt(&mut self, condition: &Node) -> Result<Value> {
        // Evaluate the condition
        let condition_value = self.evaluate(condition)?;
        
        // Check if the condition is truthy
        if self.is_truthy(&condition_value) {
            // If the condition is true, return None (continue execution)
            Ok(Value::Null)
        } else {
            // If the condition is false, raise a runtime error
            match condition {
                Node::StringLiteral(message, _) => {
                    Err(TurbulanceError::RuntimeError {
                        message: format!("Ensure failed: {}", message)
                    })
                },
                _ => {
                    Err(TurbulanceError::RuntimeError {
                        message: "Ensure statement failed".to_string()
                    })
                }
            }
        }
    }
    
    fn evaluate_research_stmt(&mut self, query: &Node) -> Result<Value> {
        // Evaluate the query
        let query_value = self.evaluate(query)?;
        
        // Extract the query text
        let query_text = match query_value {
            Value::String(text) => text,
            _ => return Err(TurbulanceError::RuntimeError {
                message: format!("Research query must be a string, got: {:?}", query_value)
            })
        };
        
        // This is a placeholder for actual research functionality
        // In a complete implementation, this would connect to a knowledge database
        // or an external research API
        let research_result = format!("Research results for: {}", query_text);
        
        Ok(Value::TextUnit(TextUnit::new(research_result)))
    }
    
    fn evaluate_text_operation(&mut self, operation: &TextOp, target: &Node, arguments: &[Node]) -> Result<Value> {
        // Evaluate the target
        let target_value = self.evaluate(target)?;
        
        // Ensure the target is a TextUnit or String
        let text_unit = match &target_value {
            Value::TextUnit(unit) => unit.clone(),
            Value::String(text) => TextUnit::new(text.clone()),
            _ => return Err(TurbulanceError::RuntimeError {
                message: format!("Text operation target must be a TextUnit or String, got: {:?}", target_value)
            })
        };
        
        // Evaluate the arguments
        let mut evaluated_args = Vec::with_capacity(arguments.len());
        for arg in arguments {
            evaluated_args.push(self.evaluate(arg)?);
        }
        
        // Apply the operation
        match operation {
            TextOp::Simplify => self.apply_simplify(text_unit, &evaluated_args),
            TextOp::Expand => self.apply_expand(text_unit, &evaluated_args),
            TextOp::Formalize => self.apply_formalize(text_unit, &evaluated_args),
            TextOp::Informalize => self.apply_informalize(text_unit, &evaluated_args),
            TextOp::Rewrite => self.apply_rewrite(text_unit, &evaluated_args),
            TextOp::Translate => self.apply_translate(text_unit, &evaluated_args),
            TextOp::Extract => self.apply_extract(text_unit, &evaluated_args),
            TextOp::Summarize => self.apply_summarize(text_unit, &evaluated_args),
            _ => Err(TurbulanceError::RuntimeError {
                message: format!("Unsupported text operation: {:?}", operation)
            })
        }
    }
    
    // Helper for checking if a value is "truthy"
    fn is_truthy(&self, value: &Value) -> bool {
        match value {
            Value::Boolean(b) => *b,
            Value::Number(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::Array(l) => !l.is_empty(),
            Value::Object(m) => !m.is_empty(),
            Value::TextUnit(tu) => !tu.content.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Function(_) => true,
            Value::NativeFunction(_) => true,
            Value::Module(_) => true,
            Value::Null => false,
        }
    }
    
    // Text operation implementations
    fn apply_simplify(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Implement text simplification logic
        let simplified = text_unit.content
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" "); // Remove extra whitespace
        
        let simplified = simplified
            .replace("very ", "")
            .replace("really ", "")
            .replace("quite ", "")
            .replace("extremely ", "")
            .replace("in order to", "to")
            .replace("due to the fact that", "because")
            .replace("it is important to note that", "")
            .replace("it should be mentioned that", "");
        
        let mut result_unit = text_unit;
        result_unit.content = simplified;
        result_unit.metadata.insert("operation".to_string(), Value::String("simplified".to_string()));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_expand(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Implement text expansion logic
        let mut expanded = text_unit.content.clone();
        
        // Add elaborative phrases
        expanded = expanded.replace(".", ". This is significant because it demonstrates the underlying principles.");
        expanded = expanded.replace("!", "! This point cannot be overstated in its importance.");
        expanded = expanded.replace("?", "? This question raises several important considerations.");
        
        // Add a concluding paragraph
        expanded.push_str("\n\nIn conclusion, these points collectively illustrate the complexity and nuance of the subject matter, requiring careful consideration of multiple perspectives and factors.");
        
        let mut result_unit = text_unit;
        result_unit.content = expanded;
        result_unit.metadata.insert("operation".to_string(), Value::String("expanded".to_string()));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_formalize(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Implement text formalization logic
        let formalized = text_unit.content
            .replace("I think", "It is proposed that")
            .replace("I believe", "It is hypothesized that")
            .replace("we", "one")
            .replace("you", "one")
            .replace("can't", "cannot")
            .replace("won't", "will not")
            .replace("don't", "do not")
            .replace("isn't", "is not")
            .replace("aren't", "are not")
            .replace("wasn't", "was not")
            .replace("weren't", "were not")
            .replace("haven't", "have not")
            .replace("hasn't", "has not")
            .replace("hadn't", "had not")
            .replace("wouldn't", "would not")
            .replace("shouldn't", "should not")
            .replace("couldn't", "could not")
            .replace("mustn't", "must not");
        
        let mut result_unit = text_unit;
        result_unit.content = formalized;
        result_unit.metadata.insert("operation".to_string(), Value::String("formalized".to_string()));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_informalize(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Implement text informalization logic
        let informalized = text_unit.content
            .replace("It is proposed that", "I think")
            .replace("It is hypothesized that", "I believe")
            .replace("one must", "you should")
            .replace("one should", "you should")
            .replace("one can", "you can")
            .replace("one may", "you might")
            .replace("cannot", "can't")
            .replace("will not", "won't")
            .replace("do not", "don't")
            .replace("is not", "isn't")
            .replace("are not", "aren't")
            .replace("was not", "wasn't")
            .replace("were not", "weren't")
            .replace("have not", "haven't")
            .replace("has not", "hasn't")
            .replace("had not", "hadn't")
            .replace("would not", "wouldn't")
            .replace("should not", "shouldn't")
            .replace("could not", "couldn't")
            .replace("must not", "mustn't");
        
        let mut result_unit = text_unit;
        result_unit.content = informalized;
        result_unit.metadata.insert("operation".to_string(), Value::String("informalized".to_string()));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_rewrite(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Check for style argument
        let style = if !args.is_empty() {
            match &args[0] {
                Value::String(s) => s.clone(),
                _ => "default".to_string()
            }
        } else {
            "default".to_string()
        };
        
        // Implement style-specific rewriting
        let rewritten = match style.as_str() {
            "academic" => {
                format!("In the context of scholarly discourse, {}. This analysis provides significant insights into the underlying mechanisms and theoretical frameworks that govern such phenomena.", text_unit.content)
            },
            "conversational" => {
                format!("So here's the thing: {}. Pretty interesting stuff, right? It really makes you think about how all of this fits together.", text_unit.content)
            },
            "technical" => {
                format!("System analysis indicates: {}. Implementation parameters must be configured according to specification requirements and operational constraints.", text_unit.content)
            },
            "creative" => {
                format!("Imagine, if you will, a world where {}. The implications ripple outward like stones cast into still water, each creating ever-widening circles of possibility.", text_unit.content)
            },
            _ => text_unit.content.clone()
        };
        
        let mut result_unit = text_unit;
        result_unit.content = rewritten;
        result_unit.metadata.insert("operation".to_string(), Value::String("rewritten".to_string()));
        result_unit.metadata.insert("style".to_string(), Value::String(style));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_translate(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Check for language argument
        if args.is_empty() {
            return Err(TurbulanceError::RuntimeError {
                message: "translate operation requires a target language".to_string()
            });
        }
        
        let language = match &args[0] {
            Value::String(s) => s.clone(),
            _ => return Err(TurbulanceError::RuntimeError {
                message: "translate operation requires a string language argument".to_string()
            })
        };
        
        // Mock translation (in real implementation, this would call a translation service)
        let translated = match language.as_str() {
            "spanish" | "es" => {
                format!("[Traducido al español] {}", text_unit.content)
            },
            "french" | "fr" => {
                format!("[Traduit en français] {}", text_unit.content)
            },
            "german" | "de" => {
                format!("[Ins Deutsche übersetzt] {}", text_unit.content)
            },
            "japanese" | "ja" => {
                format!("[日本語に翻訳] {}", text_unit.content)
            },
            _ => {
                format!("[Translated to {}] {}", language, text_unit.content)
            }
        };
        
        let mut result_unit = text_unit;
        result_unit.content = translated;
        result_unit.metadata.insert("operation".to_string(), Value::String("translated".to_string()));
        result_unit.metadata.insert("target_language".to_string(), Value::String(language));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn apply_extract(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Check for pattern argument
        if args.is_empty() {
            return Err(TurbulanceError::RuntimeError {
                message: "extract operation requires a pattern".to_string()
            });
        }
        
        let pattern = match &args[0] {
            Value::String(s) => s.clone(),
            _ => return Err(TurbulanceError::RuntimeError {
                message: "extract operation requires a string pattern argument".to_string()
            })
        };
        
        // Implement pattern extraction logic
        let extracted_parts: Vec<String> = text_unit.content
            .split_whitespace()
            .filter(|word| word.to_lowercase().contains(&pattern.to_lowercase()))
            .map(|word| word.to_string())
            .collect();
        
        if !extracted_parts.is_empty() {
            let extracted_content = extracted_parts.join(" ");
            let mut result_unit = text_unit;
            result_unit.content = extracted_content;
            result_unit.metadata.insert("operation".to_string(), Value::String("extracted".to_string()));
            result_unit.metadata.insert("pattern".to_string(), Value::String(pattern));
            
            Ok(Value::TextUnit(result_unit))
        } else {
            // Return a list of all sentences containing the pattern
            let sentences: Vec<String> = text_unit.content
                .split(&['.', '!', '?'][..])
                .filter(|sentence| sentence.to_lowercase().contains(&pattern.to_lowercase()))
                .map(|sentence| sentence.trim().to_string())
                .filter(|sentence| !sentence.is_empty())
                .collect();
            
            let result: Vec<Value> = sentences.into_iter()
                .map(|sentence| Value::TextUnit(TextUnit::new(sentence)))
                .collect();
            
            Ok(Value::List(result))
        }
    }
    
    fn apply_summarize(&self, text_unit: TextUnit, args: &[Value]) -> Result<Value> {
        // Check for length argument (optional)
        let target_length = if !args.is_empty() {
            match &args[0] {
                Value::Number(n) => *n as usize,
                _ => 100 // default word count
            }
        } else {
            100 // default word count
        };
        
        // Implement summarization logic
        let sentences: Vec<&str> = text_unit.content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
        
        let summary = if sentences.len() <= 3 {
            // If already short, just return the first sentence
            sentences.get(0).unwrap_or(&"").trim().to_string()
        } else {
            // Take first and last sentences, plus one from the middle
            let first = sentences.get(0).unwrap_or(&"").trim();
            let middle = sentences.get(sentences.len() / 2).unwrap_or(&"").trim();
            let last = sentences.get(sentences.len() - 1).unwrap_or(&"").trim();
            
            format!("{}. {}. {}.", first, middle, last)
        };
        
        // Trim to target length if specified
        let words: Vec<&str> = summary.split_whitespace().collect();
        let final_summary = if words.len() > target_length {
            words[..target_length].join(" ") + "..."
        } else {
            summary
        };
        
        let mut result_unit = text_unit;
        result_unit.content = final_summary;
        result_unit.metadata.insert("operation".to_string(), Value::String("summarized".to_string()));
        result_unit.metadata.insert("target_length".to_string(), Value::Number(target_length as f64));
        
        Ok(Value::TextUnit(result_unit))
    }
    
    fn evaluate_function_decl(&mut self, name: &str, parameters: &[crate::turbulance::ast::Parameter], body: &Node) -> Result<Value> {
        // Create a function value
        let func = crate::turbulance::ast::Function {
            name: name.to_string(),
            parameters: parameters.to_vec(),
            body: Box::new(body.clone()),
            closure: self.capture_current_scope(),
        };
        
        // Store the function in the current scope
        let function_value = Value::Function(func);
        self.define_variable(name.to_string(), function_value.clone());
        
        // Return the function value
        Ok(function_value)
    }
    
    fn evaluate_project_decl(&mut self, name: &str, attributes: &HashMap<String, Node>, body: &Node) -> Result<Value> {
        // Create a map to store the evaluated attributes
        let mut project_attributes = HashMap::new();
        
        // Evaluate each attribute
        for (key, value_node) in attributes {
            let value = self.evaluate(value_node)?;
            project_attributes.insert(key.clone(), value);
        }
        
        // Create a new scope for the project
        self.enter_scope();
        
        // Add project attributes to the scope
        for (key, value) in &project_attributes {
            self.define_variable(key.clone(), value.clone());
        }
        
        // Add a special variable for the project name
        self.define_variable("_project_name".to_string(), Value::String(name.to_string()));
        
        // Execute the project body
        let result = self.evaluate(body)?;
        
        // Exit the project scope
        self.exit_scope();
        
        // Return the project as a map
        Ok(Value::Map(project_attributes))
    }
    
    fn evaluate_sources_decl(&mut self, sources: &[crate::turbulance::ast::Source]) -> Result<Value> {
        // Create a list to store the evaluated sources
        let mut sources_list = Vec::new();
        
        // Evaluate each source
        for source in sources {
            let mut source_map = HashMap::new();
            
            // Add the source path
            source_map.insert("path".to_string(), Value::String(source.path.clone()));
            
            // Add the source type if present
            if let Some(ref source_type) = source.source_type {
                source_map.insert("type".to_string(), Value::String(source_type.clone()));
            }
            
            // Add the source to the list
            sources_list.push(Value::Map(source_map));
        }
        
        // Store the sources in a global variable
        self.define_variable("_sources".to_string(), Value::List(sources_list.clone()));
        
        // Return the sources list
        Ok(Value::List(sources_list))
    }
    
    // Helper methods for binary operations
    
    fn evaluate_add(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            // Number + Number = Number
            (Value::Number(l), Value::Number(r)) => Ok(Value::Number(l + r)),
            
            // String + String = Concatenated String
            (Value::String(l), Value::String(r)) => Ok(Value::String(l + &r)),
            
            // TextUnit + TextUnit = Combined TextUnit with proper connectives
            (Value::TextUnit(l), Value::TextUnit(r)) => {
                let combined_content = format!("{} {}", l.content, r.content);
                let mut combined_metadata = l.metadata.clone();
                
                // Merge metadata
                for (key, value) in r.metadata {
                    combined_metadata.insert(key, value);
                }
                
                Ok(Value::TextUnit(TextUnit::with_metadata(combined_content, combined_metadata)))
            },
            
            // TextUnit + String = TextUnit with appended string
            (Value::TextUnit(mut l), Value::String(r)) => {
                l.content = format!("{} {}", l.content, r);
                Ok(Value::TextUnit(l))
            },
            
            // String + TextUnit = TextUnit with prepended string
            (Value::String(l), Value::TextUnit(mut r)) => {
                r.content = format!("{} {}", l, r.content);
                Ok(Value::TextUnit(r))
            },
            
            // TextUnit + TextUnit = Combined TextUnit with proper connectives
            (Value::TextUnit(l), Value::TextUnit(r)) => {
                let combined_content = format!("{} {}", l.content, r.content);
                let mut combined_metadata = l.metadata.clone();
                
                // Merge metadata
                for (key, value) in r.metadata {
                    combined_metadata.insert(key, value);
                }
                
                Ok(Value::TextUnit(TextUnit::with_metadata(combined_content, combined_metadata)))
            },
            
            // TextUnit + String = TextUnit with appended string
            (Value::TextUnit(mut l), Value::String(r)) => {
                l.content = format!("{} {}", l.content, r);
                Ok(Value::TextUnit(l))
            },
            
            // String + TextUnit = TextUnit with prepended string
            (Value::String(l), Value::TextUnit(mut r)) => {
                r.content = format!("{} {}", l, r.content);
                Ok(Value::TextUnit(r))
            },

            // Arrays can be concatenated
            (Value::Array(mut l), Value::Array(r)) => {
                l.extend(r);
                Ok(Value::Array(l))
            },
            
            // Lists can be concatenated
            (Value::List(mut l), Value::List(r)) => {
                l.extend(r);
                Ok(Value::List(l))
            },
            
            // Type mismatch
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot add values of different types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_subtract(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            // Number - Number = Number
            (Value::Number(l), Value::Number(r)) => Ok(Value::Number(l - r)),
            
            // TextUnit - String = TextUnit with occurrences of the string removed
            (Value::TextUnit(mut l), Value::String(r)) => {
                l.content = l.content.replace(&r, "").replace("  ", " ").trim().to_string();
                Ok(Value::TextUnit(l))
            },
            
            // Subtracting elements from an array
            (Value::Array(l), Value::Array(r)) => {
                let result: Vec<Value> = l.into_iter()
                    .filter(|item| !r.contains(item))
                    .collect();
                Ok(Value::Array(result))
            },
            
            // Subtracting elements from a list
            (Value::List(l), Value::List(r)) => {
                let result: Vec<Value> = l.into_iter()
                    .filter(|item| !r.contains(item))
                    .collect();
                Ok(Value::List(result))
            },
            
            // Type mismatch
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot subtract values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_multiply(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            // Number * Number = Number
            (Value::Number(l), Value::Number(r)) => Ok(Value::Number(l * r)),
            
            // String * Number = Repeated String
            (Value::String(s), Value::Number(n)) => {
                if n.is_sign_negative() || n.fract() != 0.0 {
                    return Err(TurbulanceError::RuntimeError { 
                        message: "String multiplication requires a non-negative integer".to_string() 
                    });
                }
                
                let repeated = s.repeat(n as usize);
                Ok(Value::String(repeated))
            },
            
            // Number * String = Repeated String (commutative)
            (Value::Number(n), Value::String(s)) => {
                self.evaluate_multiply(Value::String(s), Value::Number(n))
            },
            
            // TextUnit * TextUnit = Merged text units with proper transitions
            (Value::TextUnit(l), Value::TextUnit(r)) => {
                // This is a placeholder for a more sophisticated merge
                // In a complete implementation, this would analyze the texts and
                // create appropriate transitions between them
                let combined_content = format!("{}.\n\nFurthermore, {}.", l.content, r.content);
                let mut combined_metadata = l.metadata.clone();
                
                // Merge metadata
                for (key, value) in r.metadata {
                    combined_metadata.insert(key, value);
                }
                
                Ok(Value::TextUnit(TextUnit::with_metadata(combined_content, combined_metadata)))
            },
            
            // Type mismatch
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot multiply values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_divide(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            // Number / Number = Number
            (Value::Number(l), Value::Number(r)) => {
                if r == 0.0 {
                    return Err(TurbulanceError::RuntimeError { 
                        message: "Division by zero".to_string() 
                    });
                }
                Ok(Value::Number(l / r))
            },
            
            // TextUnit / String = List of TextUnits split by the string
            (Value::TextUnit(l), Value::String(delimiter)) => {
                let parts: Vec<String> = l.content.split(&delimiter).map(|s| s.trim().to_string()).collect();
                let result: Vec<Value> = parts.into_iter()
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let unit = TextUnit::new(s);
                        Value::TextUnit(unit)
                    })
                    .collect();
                
                Ok(Value::List(result))
            },
            
            // TextUnit / TextUnit = Division by semantic boundaries
            // This is a placeholder for more sophisticated semantic division
            (Value::TextUnit(l), Value::TextUnit(r)) => {
                // In a complete implementation, this would use r as a pattern to divide l
                // For now, we'll just split by paragraphs
                let parts: Vec<String> = l.content.split("\n\n").map(|s| s.trim().to_string()).collect();
                let result: Vec<Value> = parts.into_iter()
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let unit = TextUnit::new(s);
                        Value::TextUnit(unit)
                    })
                    .collect();
                
                Ok(Value::List(result))
            },
            
            // Type mismatch
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot divide values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_less_than(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(l), Value::Number(r)) => Ok(Value::Boolean(l < r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l < r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot compare values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_greater_than(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(l), Value::Number(r)) => Ok(Value::Boolean(l > r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l > r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot compare values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_less_than_equal(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(l), Value::Number(r)) => Ok(Value::Boolean(l <= r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l <= r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot compare values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_greater_than_equal(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Number(l), Value::Number(r)) => Ok(Value::Boolean(l >= r)),
            (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l >= r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Cannot compare values of these types: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_and(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l && r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Logical AND requires boolean operands, got: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_or(&mut self, left: Value, right: Value) -> Result<Value> {
        match (left, right) {
            (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l || r)),
            (l, r) => Err(TurbulanceError::RuntimeError { 
                message: format!("Logical OR requires boolean operands, got: {:?} and {:?}", l, r) 
            }),
        }
    }
    
    fn evaluate_pipe(&mut self, left: Value, right: Value) -> Result<Value> {
        // The pipe operator (|) applies the right function to the left value
        // For example: "text" | simplify_function
        match right {
            Value::Function(f) => {
                // Create a new environment for function execution
                let previous_env = self.environment.clone();
                self.environment = Rc::new(RefCell::new(f.closure.clone()));
                
                // Define parameters (assuming single parameter for pipe)
                if f.params.len() != 1 {
                    return Err(TurbulanceError::RuntimeError { 
                        message: format!("Pipe function must have exactly 1 parameter, got {}", f.params.len()) 
                    });
                }
                
                self.environment.borrow_mut().define(f.params[0].clone(), left);
                
                // Execute function body
                let result = self.evaluate(&f.body)?;
                
                // Restore previous environment
                self.environment = previous_env;
                
                Ok(result)
            },
            Value::NativeFunction(f) => {
                // Call native function with left as argument
                f(vec![left])
            },
            Value::String(function_name) => {
                // Try to call a built-in text operation by name
                match function_name.as_str() {
                    "simplify" => {
                        match left {
                            Value::TextUnit(tu) => self.apply_simplify(tu, &[]),
                            Value::String(s) => self.apply_simplify(TextUnit::new(s), &[]),
                            _ => Err(TurbulanceError::RuntimeError {
                                message: "Simplify operation requires TextUnit or String".to_string()
                            })
                        }
                    },
                    "expand" => {
                        match left {
                            Value::TextUnit(tu) => self.apply_expand(tu, &[]),
                            Value::String(s) => self.apply_expand(TextUnit::new(s), &[]),
                            _ => Err(TurbulanceError::RuntimeError {
                                message: "Expand operation requires TextUnit or String".to_string()
                            })
                        }
                    },
                    "formalize" => {
                        match left {
                            Value::TextUnit(tu) => self.apply_formalize(tu, &[]),
                            Value::String(s) => self.apply_formalize(TextUnit::new(s), &[]),
                            _ => Err(TurbulanceError::RuntimeError {
                                message: "Formalize operation requires TextUnit or String".to_string()
                            })
                        }
                    },
                    "summarize" => {
                        match left {
                            Value::TextUnit(tu) => self.apply_summarize(tu, &[]),
                            Value::String(s) => self.apply_summarize(TextUnit::new(s), &[]),
                            _ => Err(TurbulanceError::RuntimeError {
                                message: "Summarize operation requires TextUnit or String".to_string()
                            })
                        }
                    },
                    _ => Err(TurbulanceError::RuntimeError {
                        message: format!("Unknown text operation: {}", function_name)
                    })
                }
            },
            _ => Err(TurbulanceError::RuntimeError { 
                message: "Pipe operator requires a function, native function, or operation name as its right operand".to_string() 
            }),
        }
    }
    
    fn evaluate_pipe_forward(&mut self, left: Value, right: Value) -> Result<Value> {
        // The forward pipe operator (|>) is specialized for chaining text operations
        // It preserves the TextUnit structure through the pipeline
        match (left, right) {
            (Value::TextUnit(tu), Value::String(operation)) => {
                // Apply the named operation to the TextUnit
                match operation.as_str() {
                    "simplify" => self.apply_simplify(tu, &[]),
                    "expand" => self.apply_expand(tu, &[]),
                    "formalize" => self.apply_formalize(tu, &[]),
                    "informalize" => self.apply_informalize(tu, &[]),
                    "summarize" => self.apply_summarize(tu, &[]),
                    _ => Err(TurbulanceError::RuntimeError {
                        message: format!("Unknown text operation: {}", operation)
                    })
                }
            },
            (Value::String(s), Value::String(operation)) => {
                // Convert string to TextUnit and apply operation
                let tu = TextUnit::new(s);
                match operation.as_str() {
                    "simplify" => self.apply_simplify(tu, &[]),
                    "expand" => self.apply_expand(tu, &[]),
                    "formalize" => self.apply_formalize(tu, &[]),
                    "informalize" => self.apply_informalize(tu, &[]),
                    "summarize" => self.apply_summarize(tu, &[]),
                    _ => Err(TurbulanceError::RuntimeError {
                        message: format!("Unknown text operation: {}", operation)
                    })
                }
            },
            (left_val, Value::Function(f)) => {
                // Chain with a function
                self.evaluate_pipe(left_val, Value::Function(f))
            },
            _ => Err(TurbulanceError::RuntimeError {
                message: "Pipe forward operator requires compatible operands".to_string()
            })
        }
    }
    
    fn evaluate_arrow(&mut self, left: Value, right: Value) -> Result<Value> {
        // The arrow operator (=>) creates a transformation mapping
        // It can be used to assign transformation results or create mappings
        match right {
            Value::String(variable_name) => {
                // Assign the left value to the named variable
                self.define_variable(variable_name, left.clone());
                Ok(left)
            },
            Value::Function(f) => {
                // Apply the function to the left value (similar to pipe)
                self.evaluate_pipe(left, Value::Function(f))
            },
            _ => Err(TurbulanceError::RuntimeError {
                message: "Arrow operator requires a variable name or function as its right operand".to_string()
            })
        }
    }
    
    // Helper function to call a user-defined function
    fn execute_call(&mut self, callee: &str, arguments: &[Value]) -> Result<Value> {
        // Get the function from the environment
        let func = match self.environment.borrow().get(callee) {
            Some(Value::Function(f)) => f,
            Some(Value::NativeFunction(f)) => {
                // Execute native function
                let mut args = Vec::new();
                for arg in arguments {
                    args.push(arg.clone());
                }
                return f(args);
            },
            _ => return Err(TurbulanceError::RuntimeError { 
                message: format!("'{}' is not a function.", callee) 
            }),
        };
        
        // Check argument count
        if arguments.len() != func.params.len() {
            return Err(TurbulanceError::RuntimeError { 
                message: format!(
                    "Expected {} arguments but got {}.", 
                    func.params.len(), 
                    arguments.len()
                ) 
            });
        }
        
        // Create new environment with function's closure as parent
        let closure_env = Rc::new(RefCell::new(func.closure.clone()));
        let mut env = Environment::with_enclosing(closure_env);
        
        // Define parameters in the new environment
        for (param, value) in func.params.iter().zip(arguments.iter()) {
            env.define(param.clone(), value.clone());
        }
        
        // Execute the function body with the new environment
        let previous_env = self.environment.clone();
        self.environment = Rc::new(RefCell::new(env));
        
        let result = self.evaluate(&func.body)?;
        
        // Restore the previous environment
        self.environment = previous_env;
        
        Ok(result)
    }
    
    // Capture the current scope for closures
    fn capture_current_scope(&self) -> Environment {
        if self.scopes.is_empty() {
            return Environment::new();
        }
        
        // Create a new map to hold the captured variables
        let mut captured = HashMap::new();
        
        // Capture all variables from all scopes
        for scope in &self.scopes {
            for (name, value) in scope {
                captured.insert(name.clone(), value.clone());
            }
        }
        
        Environment {
            values: captured,
            enclosing: None,
        }
    }
    
    // Get access to the standard library
    fn get_stdlib(&self) -> Option<&StdLib> {
        Some(&self.stdlib)
    }
    
    /// Set a global variable
    pub fn set_global(&mut self, name: &str, value: Value) {
        let mut env = self.global_environment.borrow_mut();
        env.define(name.to_string(), value);
    }
}

/// Reference to a shared object (using reference counting)
#[derive(Clone)]
pub struct ObjectRef(Rc<RefCell<dyn Object>>);

impl ObjectRef {
    /// Create a new object reference
    pub fn new<T: Object + 'static>(obj: T) -> Self {
        ObjectRef(Rc::new(RefCell::new(obj)))
    }
    
    /// Borrow the object
    pub fn borrow(&self) -> std::cell::Ref<dyn Object> {
        self.0.borrow()
    }
    
    /// Borrow the object mutably
    pub fn borrow_mut(&self) -> std::cell::RefMut<dyn Object> {
        self.0.borrow_mut()
    }
}

impl fmt::Debug for ObjectRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectRef")
    }
}

/// Object trait for custom objects
pub trait Object: fmt::Debug {
    /// Get a property from the object
    fn get(&self, _name: &str) -> Option<Value> {
        None
    }
    
    /// Set a property on the object
    fn set(&mut self, _name: &str, _value: Value) {
        // Default implementation does nothing
    }
    
    /// Call a method on the object
    fn call_method(&self, _name: &str, _args: &[Value]) -> Result<Value> {
        Err(TurbulanceError::RuntimeError { 
            message: format!("Object doesn't support method calls") 
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::ast::{Node, BinaryOp, UnaryOp, TextOp, Position, Span};
    
    #[test]
    fn test_interpreter_literals() {
        let mut interpreter = Interpreter::new();
        
        // Create a dummy span for testing
        let span = Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0));
        
        // Test string literal
        let string_node = Node::StringLiteral("hello".to_string(), span);
        let result = interpreter.execute(&string_node);
        assert!(result.is_ok());
        if let Ok(Value::String(s)) = result {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected string value");
        }
        
        // Test number literal
        let number_node = Node::NumberLiteral(42.0, span);
        let result = interpreter.execute(&number_node);
        assert!(result.is_ok());
        if let Ok(Value::Number(n)) = result {
            assert_eq!(n, 42.0);
        } else {
            panic!("Expected number value");
        }
        
        // Test boolean literal
        let bool_node = Node::BoolLiteral(true, span);
        let result = interpreter.execute(&bool_node);
        assert!(result.is_ok());
        if let Ok(Value::Boolean(b)) = result {
            assert_eq!(b, true);
        } else {
            panic!("Expected boolean value");
        }
    }
    
    #[test]
    fn test_binary_expressions() {
        let mut interpreter = Interpreter::new();
        let span = Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0));
        
        // Test addition
        let left = Box::new(Node::NumberLiteral(5.0, span));
        let right = Box::new(Node::NumberLiteral(3.0, span));
        let add_node = Node::BinaryExpr {
            left,
            operator: BinaryOp::Add,
            right,
            span,
        };
        
        let result = interpreter.execute(&add_node);
        assert!(result.is_ok());
        if let Ok(Value::Number(n)) = result {
            assert_eq!(n, 8.0);
        } else {
            panic!("Expected number value");
        }
    }
    
    #[test]
    fn test_string_concatenation() {
        let mut interpreter = Interpreter::new();
        let span = Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0));
        
        // Test string concatenation
        let left = Box::new(Node::StringLiteral("Hello, ".to_string(), span));
        let right = Box::new(Node::StringLiteral("world!".to_string(), span));
        let add_node = Node::BinaryExpr {
            left,
            operator: BinaryOp::Add,
            right,
            span,
        };
        
        let result = interpreter.execute(&add_node);
        assert!(result.is_ok());
        if let Ok(Value::String(s)) = result {
            assert_eq!(s, "Hello, world!");
        } else {
            panic!("Expected string value");
        }
    }
    
    #[test]
    fn test_variable_assignment_and_lookup() {
        let mut interpreter = Interpreter::new();
        let span = Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0));
        
        // Create variable assignment: x = 42
        let target = Node::Identifier("x".to_string(), span);
        let value = Node::NumberLiteral(42.0, span);
        let assignment = Node::Assignment {
            target: Box::new(target.clone()),
            value: Box::new(value),
            span,
        };
        
        // Execute the assignment
        let result = interpreter.execute(&assignment);
        assert!(result.is_ok());
        
        // Check that we can look up the variable
        let result = interpreter.execute(&target);
        assert!(result.is_ok());
        if let Ok(Value::Number(n)) = result {
            assert_eq!(n, 42.0);
        } else {
            panic!("Expected number value");
        }
    }
    
    #[test]
    fn test_block_evaluation() {
        let mut interpreter = Interpreter::new();
        let span = Span::new(Position::new(0, 0, 0), Position::new(0, 0, 0));
        
        // Create a block with assignment and return
        let statements = vec![
            Node::Assignment {
                target: Box::new(Node::Identifier("x".to_string(), span)),
                value: Box::new(Node::NumberLiteral(10.0, span)),
                span,
            },
            Node::Assignment {
                target: Box::new(Node::Identifier("y".to_string(), span)),
                value: Box::new(Node::NumberLiteral(20.0, span)),
                span,
            },
            Node::BinaryExpr {
                left: Box::new(Node::Identifier("x".to_string(), span)),
                operator: BinaryOp::Add,
                right: Box::new(Node::Identifier("y".to_string(), span)),
                span,
            },
        ];
        
        let block = Node::Block {
            statements,
            span,
        };
        
        // Execute the block
        let result = interpreter.execute(&block);
        assert!(result.is_ok());
        if let Ok(Value::Number(n)) = result {
            assert_eq!(n, 30.0);
        } else {
            panic!("Expected number value");
        }
    }
}


