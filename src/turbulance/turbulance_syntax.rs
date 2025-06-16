/// Turbulance Language Syntax for Hybrid Processing
/// 
/// This module implements the specialized Turbulance syntax for hybrid processing
/// that combines deterministic and probabilistic operations within the same
/// control flow constructs.

use std::collections::HashMap;
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::hybrid_processing::{HybridProcessor, ProbabilisticFloor, HybridConfig, HybridResult};
use crate::turbulance::probabilistic::TextPoint;

/// Turbulance hybrid function processor
pub struct TurbulanceProcessor {
    /// Underlying hybrid processor
    processor: HybridProcessor,
    
    /// Function definitions
    functions: HashMap<String, TurbulanceFunction>,
    
    /// Global variables
    variables: HashMap<String, Value>,
}

/// A Turbulance function definition
#[derive(Clone, Debug)]
pub struct TurbulanceFunction {
    /// Function name
    pub name: String,
    
    /// Parameters
    pub parameters: Vec<String>,
    
    /// Function body operations
    pub operations: Vec<TurbulanceOperation>,
    
    /// Return type
    pub return_type: TurbulanceType,
}

/// Types in Turbulance
#[derive(Clone, Debug, PartialEq)]
pub enum TurbulanceType {
    /// Single point
    Point,
    
    /// Collection of points
    Points,
    
    /// Confirmed/determined point
    Entity,
    
    /// Probabilistic floor (iterable dict of points)
    Floor,
    
    /// Text corpus
    Corpus,
    
    /// Generic value
    Value,
}

/// Operations in Turbulance functions
#[derive(Clone, Debug)]
pub enum TurbulanceOperation {
    /// Basic cycle: `cycle item over floor: resolve item`
    Cycle {
        item_var: String,
        floor_var: String,
        body: Box<TurbulanceOperation>,
    },
    
    /// Probabilistic drift: `drift text in corpus: resolution analyze text`
    Drift {
        text_var: String,
        corpus_var: String,
        body: Box<TurbulanceOperation>,
    },
    
    /// Streaming flow: `flow line on floor: resolution parse line`
    Flow {
        line_var: String,
        floor_var: String,
        body: Box<TurbulanceOperation>,
    },
    
    /// Roll until settled: `roll until settled: resolution guess next`
    RollUntilSettled {
        body: Box<TurbulanceOperation>,
    },
    
    /// Considering loop with probabilistic conditions
    Considering {
        item_var: String,
        collection_var: String,
        condition: Box<TurbulanceCondition>,
        body: Box<TurbulanceOperation>,
    },
    
    /// Resolution operation
    Resolution {
        operation: String,
        target: String,
    },
    
    /// Variable assignment
    Assignment {
        var_name: String,
        value: Value,
    },
    
    /// Block of operations
    Block(Vec<TurbulanceOperation>),
    
    /// Return statement
    Return(Value),
}

/// Conditions in Turbulance (can be probabilistic)
#[derive(Clone, Debug)]
pub enum TurbulanceCondition {
    /// Binary condition
    Binary {
        left: String,
        operator: String,
        right: Value,
    },
    
    /// Probabilistic condition based on resolution confidence
    ProbabilisticConfidence {
        resolution_var: String,
        threshold: f64,
        operator: ConfidenceOperator,
    },
    
    /// Complex condition with multiple clauses
    Complex {
        conditions: Vec<TurbulanceCondition>,
        logic: LogicOperator,
    },
}

/// Operators for confidence-based conditions
#[derive(Clone, Debug, PartialEq)]
pub enum ConfidenceOperator {
    GreaterThan,
    LessThan,
    Within,
    Outside,
}

/// Logic operators for combining conditions
#[derive(Clone, Debug, PartialEq)]
pub enum LogicOperator {
    And,
    Or,
    Xor,
}

impl TurbulanceProcessor {
    /// Create a new Turbulance processor
    pub fn new() -> Self {
        Self {
            processor: HybridProcessor::new(HybridConfig::default()),
            functions: HashMap::new(),
            variables: HashMap::new(),
        }
    }
    
    /// Register a Turbulance function
    pub fn register_function(&mut self, function: TurbulanceFunction) {
        self.functions.insert(function.name.clone(), function);
    }
    
    /// Execute a Turbulance function
    pub async fn execute_function(&mut self, function_name: &str, args: Vec<Value>) -> Result<Value> {
        let function = self.functions.get(function_name)
            .ok_or_else(|| TurbulanceError::RuntimeError { 
                message: format!("Function '{}' not found", function_name) 
            })?
            .clone();
        
        // Bind parameters
        for (i, param) in function.parameters.iter().enumerate() {
            if let Some(arg) = args.get(i) {
                self.variables.insert(param.clone(), arg.clone());
            }
        }
        
        // Execute function body
        for operation in &function.operations {
            let result = self.execute_operation(operation).await?;
            if let Value::String(s) = &result {
                if s.starts_with("return:") {
                    return Ok(Value::String(s.strip_prefix("return:").unwrap().to_string()));
                }
            }
        }
        
        Ok(Value::String("function completed".to_string()))
    }
    
    /// Execute a single operation
    pub fn execute_operation<'a>(&'a mut self, operation: &'a TurbulanceOperation) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + 'a>> {
        Box::pin(async move {
        match operation {
            TurbulanceOperation::Cycle { item_var, floor_var, body } => {
                self.execute_cycle(item_var, floor_var, body).await
            },
            
            TurbulanceOperation::Drift { text_var, corpus_var, body } => {
                self.execute_drift(text_var, corpus_var, body).await
            },
            
            TurbulanceOperation::Flow { line_var, floor_var, body } => {
                self.execute_flow(line_var, floor_var, body).await
            },
            
            TurbulanceOperation::RollUntilSettled { body } => {
                self.execute_roll_until_settled(body).await
            },
            
            TurbulanceOperation::Considering { item_var, collection_var, condition, body } => {
                self.execute_considering(item_var, collection_var, condition, body).await
            },
            
            TurbulanceOperation::Resolution { operation, target } => {
                self.execute_resolution(operation, target).await
            },
            
            TurbulanceOperation::Assignment { var_name, value } => {
                self.variables.insert(var_name.clone(), value.clone());
                Ok(value.clone())
            },
            
            TurbulanceOperation::Block(operations) => {
                let mut last_result = Value::String("block completed".to_string());
                for op in operations {
                    last_result = self.execute_operation(op).await?;
                }
                Ok(last_result)
            },
            
            TurbulanceOperation::Return(value) => {
                Ok(Value::String(format!("return:{}", self.value_to_string(value))))
            },
        }
    }
    
    /// Execute cycle operation
    async fn execute_cycle(&mut self, item_var: &str, floor_var: &str, body: &TurbulanceOperation) -> Result<Value> {
        // Get floor from variables
        let floor = self.get_floor(floor_var)?;
        
        let mut results = Vec::new();
        
        // Iterate over floor items
        for (key, point, weight) in floor.probabilistic_iter() {
            // Set current item variable
            self.variables.insert(item_var.to_string(), Value::String(point.content.clone()));
            self.variables.insert("current_weight".to_string(), Value::Number(weight));
            
            // Execute body
            let result = self.execute_operation(body).await?;
            results.push(result);
        }
        
        Ok(Value::String(format!("cycle completed: {} items processed", results.len())))
    }
    
    /// Execute drift operation
    async fn execute_drift(&mut self, text_var: &str, corpus_var: &str, body: &TurbulanceOperation) -> Result<Value> {
        // Get corpus from variables
        let corpus = self.get_string_variable(corpus_var)?;
        
        // Process corpus through drift
        let drift_results = self.processor.drift(&corpus).await?;
        
        let mut results = Vec::new();
        for drift_result in drift_results {
            // Set current text variable
            self.variables.insert(text_var.to_string(), Value::String(drift_result.input.clone()));
            self.variables.insert("current_confidence".to_string(), Value::Number(drift_result.confidence));
            self.variables.insert("current_mode".to_string(), Value::String(drift_result.mode.clone()));
            
            // Execute body
            let result = self.execute_operation(body).await?;
            results.push(result);
        }
        
        Ok(Value::String(format!("drift completed: {} texts processed", results.len())))
    }
    
    /// Execute flow operation
    async fn execute_flow(&mut self, line_var: &str, floor_var: &str, body: &TurbulanceOperation) -> Result<Value> {
        // Get lines from floor variable (treating it as text lines)
        let text = self.get_string_variable(floor_var)?;
        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        
        // Process through flow
        let flow_results = self.processor.flow(&lines).await?;
        
        let mut results = Vec::new();
        for flow_result in flow_results {
            // Set current line variable
            self.variables.insert(line_var.to_string(), Value::String(flow_result.input.clone()));
            self.variables.insert("current_settled".to_string(), Value::Bool(flow_result.settled));
            
            // Execute body
            let result = self.execute_operation(body).await?;
            results.push(result);
        }
        
        Ok(Value::String(format!("flow completed: {} lines processed", results.len())))
    }
    
    /// Execute roll until settled
    async fn execute_roll_until_settled(&mut self, body: &TurbulanceOperation) -> Result<Value> {
        // Get current point (assumes it's been set)
        let current_point_text = self.get_string_variable("current_point")?;
        let point = TextPoint::new(current_point_text, 0.5); // Default confidence
        
        // Roll until settled
        let roll_result = self.processor.roll_until_settled(&point).await?;
        
        // Set result variables
        self.variables.insert("final_confidence".to_string(), Value::Number(roll_result.confidence));
        self.variables.insert("iterations".to_string(), Value::Number(roll_result.iterations as f64));
        self.variables.insert("settled".to_string(), Value::Bool(roll_result.settled));
        
        // Execute body
        self.execute_operation(body).await
    }
    
    /// Execute considering loop with probabilistic conditions
    async fn execute_considering(&mut self, 
                                item_var: &str, 
                                collection_var: &str, 
                                condition: &TurbulanceCondition, 
                                body: &TurbulanceOperation) -> Result<Value> {
        
        // Get collection (treating as sentences)
        let collection_text = self.get_string_variable(collection_var)?;
        let sentences: Vec<&str> = collection_text.split(&['.', '!', '?']).collect();
        
        let mut results = Vec::new();
        
        for sentence in sentences {
            if sentence.trim().is_empty() {
                continue;
            }
            
            // Set current item
            self.variables.insert(item_var.to_string(), Value::String(sentence.trim().to_string()));
            
            // Check condition
            if self.evaluate_condition(condition).await? {
                // Execute body
                let result = self.execute_operation(body).await?;
                results.push(result);
            }
        }
        
        Ok(Value::String(format!("considering completed: {} items processed", results.len())))
    }
    
    /// Execute resolution operation
    async fn execute_resolution(&mut self, operation: &str, target: &str) -> Result<Value> {
        match operation {
            "analyze" => {
                let text = self.get_string_variable(target)?;
                // Simulate analysis
                let confidence = if text.contains("certain") || text.contains("definite") { 0.9 }
                    else if text.contains("uncertain") || text.contains("maybe") { 0.3 }
                    else { 0.6 };
                
                self.variables.insert("resolution_confidence".to_string(), Value::Number(confidence));
                Ok(Value::String(format!("analyzed: {}", target)))
            },
            
            "parse" => {
                let text = self.get_string_variable(target)?;
                // Simulate parsing
                let point = TextPoint::new(text, 0.7);
                self.variables.insert("parsed_point".to_string(), Value::String(point.content));
                Ok(Value::String(format!("parsed: {}", target)))
            },
            
            "guess" => {
                // Simulate guess based on current context
                let confidence = self.variables.get("current_confidence")
                    .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
                    .unwrap_or(0.5);
                
                let new_confidence = (confidence + 0.1).min(1.0);
                self.variables.insert("guessed_confidence".to_string(), Value::Number(new_confidence));
                Ok(Value::String("guess completed".to_string()))
            },
            
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Unknown resolution operation: {}", operation) 
            }),
        }
    }
    
    /// Evaluate a condition
    async fn evaluate_condition(&mut self, condition: &TurbulanceCondition) -> Result<bool> {
        match condition {
            TurbulanceCondition::Binary { left, operator, right } => {
                let left_value = self.get_variable_value(left)?;
                self.evaluate_binary_condition(&left_value, operator, right)
            },
            
            TurbulanceCondition::ProbabilisticConfidence { resolution_var, threshold, operator } => {
                let confidence = self.get_number_variable(resolution_var)?;
                self.evaluate_confidence_condition(confidence, *threshold, operator)
            },
            
            TurbulanceCondition::Complex { conditions, logic } => {
                self.evaluate_complex_condition(conditions, logic).await
            },
        }
    }
    
    /// Evaluate binary condition
    fn evaluate_binary_condition(&self, left: &Value, operator: &str, right: &Value) -> Result<bool> {
        match operator {
            "contains" => {
                if let (Value::String(left_str), Value::String(right_str)) = (left, right) {
                    Ok(left_str.contains(right_str))
                } else {
                    Ok(false)
                }
            },
            "==" => Ok(left == right),
            "!=" => Ok(left != right),
            _ => Err(TurbulanceError::RuntimeError { 
                message: format!("Unknown binary operator: {}", operator) 
            }),
        }
    }
    
    /// Evaluate confidence-based condition
    fn evaluate_confidence_condition(&self, confidence: f64, threshold: f64, operator: &ConfidenceOperator) -> Result<bool> {
        match operator {
            ConfidenceOperator::GreaterThan => Ok(confidence > threshold),
            ConfidenceOperator::LessThan => Ok(confidence < threshold),
            ConfidenceOperator::Within => Ok((confidence - threshold).abs() < 0.1),
            ConfidenceOperator::Outside => Ok((confidence - threshold).abs() > 0.1),
        }
    }
    
    /// Evaluate complex condition
    async fn evaluate_complex_condition(&mut self, conditions: &[TurbulanceCondition], logic: &LogicOperator) -> Result<bool> {
        let mut results = Vec::new();
        for condition in conditions {
            results.push(self.evaluate_condition(condition).await?);
        }
        
        match logic {
            LogicOperator::And => Ok(results.iter().all(|&x| x)),
            LogicOperator::Or => Ok(results.iter().any(|&x| x)),
            LogicOperator::Xor => Ok(results.iter().filter(|&&x| x).count() == 1),
        }
    }
    
    /// Helper methods for variable access
    fn get_floor(&self, var_name: &str) -> Result<ProbabilisticFloor> {
        // For demonstration, create a simple floor
        // In a real implementation, this would retrieve from variables
        let mut floor = ProbabilisticFloor::new(0.5);
        floor.add_point("demo1".to_string(), TextPoint::new("Demo point 1".to_string(), 0.8), 1.0);
        floor.add_point("demo2".to_string(), TextPoint::new("Demo point 2".to_string(), 0.6), 0.8);
        Ok(floor)
    }
    
    fn get_string_variable(&self, var_name: &str) -> Result<String> {
        match self.variables.get(var_name) {
            Some(Value::String(s)) => Ok(s.clone()),
            Some(_) => Err(TurbulanceError::RuntimeError { 
                message: format!("Variable '{}' is not a string", var_name) 
            }),
            None => Err(TurbulanceError::RuntimeError { 
                message: format!("Variable '{}' not found", var_name) 
            }),
        }
    }
    
    fn get_number_variable(&self, var_name: &str) -> Result<f64> {
        match self.variables.get(var_name) {
            Some(Value::Number(n)) => Ok(*n),
            Some(_) => Err(TurbulanceError::RuntimeError { 
                message: format!("Variable '{}' is not a number", var_name) 
            }),
            None => Err(TurbulanceError::RuntimeError { 
                message: format!("Variable '{}' not found", var_name) 
            }),
        }
    }
    
    fn get_variable_value(&self, var_name: &str) -> Result<Value> {
        self.variables.get(var_name)
            .cloned()
            .ok_or_else(|| TurbulanceError::RuntimeError { 
                message: format!("Variable '{}' not found", var_name) 
            })
    }
    
    fn value_to_string(&self, value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => "unknown".to_string(),
        }
    }
}

/// Demonstrate the Turbulance syntax for hybrid processing
pub async fn demonstrate_turbulance_syntax() -> Result<()> {
    println!("🎵 Turbulance Syntax Demonstration 🎵\n");
    
    let mut processor = TurbulanceProcessor::new();
    
    // Set up variables
    processor.variables.insert("paragraph".to_string(), 
        Value::String("The solution demonstrates superior performance. Critics question its scalability. The evidence shows mixed results. Final assessment requires more data.".to_string()));
    
    // Example 1: Basic cycle operation
    println!("1. 🔄 Basic Cycle Operation:");
    println!("   Turbulance: cycle item over floor: resolve item");
    
    let cycle_operation = TurbulanceOperation::Cycle {
        item_var: "item".to_string(),
        floor_var: "demo_floor".to_string(),
        body: Box::new(TurbulanceOperation::Resolution {
            operation: "analyze".to_string(),
            target: "item".to_string(),
        }),
    };
    
    let result = processor.execute_operation(&cycle_operation).await?;
    println!("   Result: {}\n", processor.value_to_string(&result));
    
    // Example 2: Probabilistic drift
    println!("2. 🌊 Probabilistic Drift:");
    println!("   Turbulance: drift text in corpus: resolution analyze text");
    
    let drift_operation = TurbulanceOperation::Drift {
        text_var: "text".to_string(),
        corpus_var: "paragraph".to_string(),
        body: Box::new(TurbulanceOperation::Resolution {
            operation: "analyze".to_string(),
            target: "text".to_string(),
        }),
    };
    
    let result = processor.execute_operation(&drift_operation).await?;
    println!("   Result: {}\n", processor.value_to_string(&result));
    
    // Example 3: Considering loop with probabilistic condition
    println!("3. 🔀 Considering Loop with Probabilistic Processing:");
    println!("   Turbulance:");
    println!("   considering sentence in paragraph:");
    println!("       given resolution is within x percentage, continue or");
    println!("       either change affirmations and contentions till resolved");
    
    let considering_operation = TurbulanceOperation::Considering {
        item_var: "sentence".to_string(),
        collection_var: "paragraph".to_string(),
        condition: Box::new(TurbulanceCondition::Binary {
            left: "sentence".to_string(),
            operator: "contains".to_string(),
            right: Value::String("question".to_string()),
        }),
        body: Box::new(TurbulanceOperation::Block(vec![
            TurbulanceOperation::Resolution {
                operation: "analyze".to_string(),
                target: "sentence".to_string(),
            },
            TurbulanceOperation::Assignment {
                var_name: "current_point".to_string(),
                value: Value::String("Current sentence needs analysis".to_string()),
            },
            TurbulanceOperation::RollUntilSettled {
                body: Box::new(TurbulanceOperation::Resolution {
                    operation: "guess".to_string(),
                    target: "current_point".to_string(),
                }),
            },
        ])),
    };
    
    let result = processor.execute_operation(&considering_operation).await?;
    println!("   Result: {}\n", processor.value_to_string(&result));
    
    // Example 4: Complete hybrid function
    println!("4. 🎯 Complete Hybrid Function:");
    println!("   funxn hybrid_analysis(paragraph):");
    println!("       considering sentence in paragraph:");
    println!("           given sentence contains points, probabilistic operations");
    println!("           given resolution is within x percentage, continue or");
    println!("           either change affirmations and contentions till resolved");
    
    let hybrid_function = TurbulanceFunction {
        name: "hybrid_analysis".to_string(),
        parameters: vec!["paragraph".to_string()],
        operations: vec![
            TurbulanceOperation::Considering {
                item_var: "sentence".to_string(),
                collection_var: "paragraph".to_string(),
                condition: Box::new(TurbulanceCondition::ProbabilisticConfidence {
                    resolution_var: "resolution_confidence".to_string(),
                    threshold: 0.7,
                    operator: ConfidenceOperator::LessThan,
                }),
                body: Box::new(TurbulanceOperation::Block(vec![
                    TurbulanceOperation::Resolution {
                        operation: "analyze".to_string(),
                        target: "sentence".to_string(),
                    },
                    TurbulanceOperation::RollUntilSettled {
                        body: Box::new(TurbulanceOperation::Resolution {
                            operation: "guess".to_string(),
                            target: "sentence".to_string(),
                        }),
                    },
                ])),
            },
            TurbulanceOperation::Return(Value::String("hybrid analysis completed".to_string())),
        ],
        return_type: TurbulanceType::Value,
    };
    
    processor.register_function(hybrid_function);
    
    let args = vec![Value::String("The solution demonstrates superior performance. Critics question its scalability.".to_string())];
    let result = processor.execute_function("hybrid_analysis", args).await?;
    println!("   Function result: {}\n", processor.value_to_string(&result));
    
    println!("🎵 Turbulance syntax demonstration complete! 🎵");
    println!("This showcases:");
    println!("  • cycle - basic deterministic iteration");
    println!("  • drift - probabilistic corpus processing");
    println!("  • flow - streaming line processing");
    println!("  • roll until settled - iterative resolution");
    println!("  • considering - hybrid loops with probabilistic conditions");
    println!("  • Seamless switching between deterministic and probabilistic modes");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_turbulance_processor() {
        let processor = TurbulanceProcessor::new();
        assert!(processor.functions.is_empty());
        assert!(processor.variables.is_empty());
    }
    
    #[tokio::test]
    async fn test_cycle_operation() {
        let mut processor = TurbulanceProcessor::new();
        
        let operation = TurbulanceOperation::Cycle {
            item_var: "item".to_string(),
            floor_var: "test_floor".to_string(),
            body: Box::new(TurbulanceOperation::Resolution {
                operation: "analyze".to_string(),
                target: "item".to_string(),
            }),
        };
        
        let result = processor.execute_operation(&operation).await.unwrap();
        assert!(processor.value_to_string(&result).contains("cycle completed"));
    }
    
    #[tokio::test]
    async fn test_turbulance_syntax_demonstration() {
        let result = demonstrate_turbulance_syntax().await;
        assert!(result.is_ok());
    }
} 