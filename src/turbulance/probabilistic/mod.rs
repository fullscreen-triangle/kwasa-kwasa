/// Probabilistic text processing module for Kwasa-Kwasa
/// 
/// This module implements "Points" and "Resolution Functions" that handle
/// the inherent uncertainty and partial truths in natural language processing.

use std::collections::HashMap;
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use uuid::Uuid;
use rand::prelude::*;
use rand_distr::{Normal, Uniform, Beta, Gamma};

/// A Point represents text with inherent uncertainty and multiple interpretations
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TextPoint {
    /// The primary textual content
    pub content: String,
    
    /// Overall confidence in this point (0.0 to 1.0)
    pub confidence: f64,
    
    /// Multiple possible interpretations of this text
    pub interpretations: Vec<TextInterpretation>,
    
    /// Context dependencies that affect meaning
    pub context_dependencies: HashMap<String, f64>,
    
    /// Semantic bounds for the meaning space
    pub semantic_bounds: (f64, f64),
    
    /// Metadata about this point
    pub metadata: HashMap<String, Value>,
}

/// Different ways this text could be interpreted
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TextInterpretation {
    /// The interpreted meaning
    pub meaning: String,
    
    /// Probability of this interpretation being correct
    pub probability: f64,
    
    /// Context that supports this interpretation
    pub context: HashMap<String, Value>,
    
    /// Evidence supporting this interpretation
    pub evidence: Vec<String>,
    
    /// Linguistic features that led to this interpretation
    pub features: Vec<LinguisticFeature>,
}

/// Linguistic features that influence interpretation
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum LinguisticFeature {
    Ambiguity { words: Vec<String>, interpretations: Vec<String> },
    Metaphor { literal: String, figurative: String },
    Context { domain: String, specificity: f64 },
    Sentiment { polarity: f64, subjectivity: f64 },
    Pragmatics { implied_meaning: String, confidence: f64 },
}

/// Result of a resolution function - handles uncertainty in results
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ResolutionResult {
    /// Certain result with high confidence
    Certain(Value),
    
    /// Uncertain result with multiple possibilities
    Uncertain {
        possibilities: Vec<(Value, f64)>, // (value, probability)
        confidence_interval: (f64, f64),
        aggregated_confidence: f64,
    },
    
    /// Result that depends on context
    Contextual {
        base_result: Value,
        context_variants: HashMap<String, (Value, f64)>,
        resolution_strategy: ResolutionStrategy,
    },
    
    /// Fuzzy result for inherently vague operations
    Fuzzy {
        membership_function: Vec<(f64, f64)>, // (value, membership_degree)
        central_tendency: f64,
        spread: f64,
    },
}

/// Strategy for resolving uncertainty in results
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ResolutionStrategy {
    /// Return the most probable interpretation
    MaximumLikelihood,
    
    /// Weight results by prior beliefs (Bayesian)
    BayesianWeighted,
    
    /// Choose the safest/most conservative interpretation
    ConservativeMin,
    
    /// Choose the most informative/exploratory interpretation
    ExploratoryMax,
    
    /// Aggregate all interpretations with weights
    WeightedAggregate,
    
    /// Return all possibilities with their probabilities
    FullDistribution,
}

/// A Resolution Function applies probabilistic operations to TextPoints
pub trait ResolutionFunction {
    /// The name of this resolution function
    fn name(&self) -> &str;
    
    /// Apply this function to a TextPoint
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> Result<ResolutionResult>;
    
    /// Get the uncertainty introduced by this function
    fn uncertainty_factor(&self) -> f64;
    
    /// Whether this function can handle the given point type
    fn can_handle(&self, point: &TextPoint) -> bool;
}

/// Context for resolution operations
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolutionContext {
    /// Domain context (e.g., "scientific", "informal", "legal")
    pub domain: Option<String>,
    
    /// Cultural/linguistic context
    pub culture: Option<String>,
    
    /// Temporal context
    pub time_period: Option<String>,
    
    /// Purpose/goal of the analysis
    pub purpose: Option<String>,
    
    /// Additional context parameters
    pub parameters: HashMap<String, Value>,
    
    /// Strategy for handling uncertainty
    pub resolution_strategy: ResolutionStrategy,
}

impl TextPoint {
    /// Create a new TextPoint from content and confidence
    pub fn new(content: String, confidence: f64) -> Self {
        Self {
            content,
            confidence: confidence.max(0.0).min(1.0),
            interpretations: Vec::new(),
            context_dependencies: HashMap::new(),
            semantic_bounds: (0.0, 1.0),
            metadata: HashMap::new(),
        }
    }
    
    /// Add an interpretation to this point
    pub fn add_interpretation(&mut self, interpretation: TextInterpretation) {
        self.interpretations.push(interpretation);
        // Recompute confidence based on interpretations
        self.recompute_confidence();
    }
    
    /// Get the most likely interpretation
    pub fn primary_interpretation(&self) -> Option<&TextInterpretation> {
        self.interpretations.iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
    }
    
    /// Calculate entropy of interpretations (how ambiguous is this point)
    pub fn interpretation_entropy(&self) -> f64 {
        if self.interpretations.is_empty() {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for interp in &self.interpretations {
            if interp.probability > 0.0 {
                entropy -= interp.probability * interp.probability.log2();
            }
        }
        
        entropy
    }
    
    /// Merge with another TextPoint (for combining evidence)
    pub fn merge_with(&mut self, other: &TextPoint) -> Result<()> {
        // Combine interpretations
        for other_interp in &other.interpretations {
            // Check if we already have a similar interpretation
            let existing = self.interpretations.iter_mut()
                .find(|interp| interp.meaning == other_interp.meaning);
                
            match existing {
                Some(existing_interp) => {
                    // Update probability using Bayesian combination
                    let combined_prob = (existing_interp.probability + other_interp.probability) / 2.0;
                    existing_interp.probability = combined_prob;
                    
                    // Merge evidence
                    for evidence in &other_interp.evidence {
                        if !existing_interp.evidence.contains(evidence) {
                            existing_interp.evidence.push(evidence.clone());
                        }
                    }
                },
                None => {
                    // Add new interpretation
                    self.interpretations.push(other_interp.clone());
                }
            }
        }
        
        // Update confidence
        self.confidence = (self.confidence + other.confidence) / 2.0;
        self.recompute_confidence();
        
        Ok(())
    }
    
    /// Recompute confidence based on interpretation consistency
    fn recompute_confidence(&mut self) {
        if self.interpretations.is_empty() {
            return;
        }
        
        // Calculate agreement between interpretations
        let entropy = self.interpretation_entropy();
        let max_entropy = (self.interpretations.len() as f64).log2();
        
        // Higher entropy means more uncertainty, lower confidence
        let entropy_factor = if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        };
        
        // Adjust confidence based on interpretation consistency
        self.confidence = (self.confidence + entropy_factor) / 2.0;
    }
}

/// Probabilistic length function - demonstrates the concept
pub struct ProbabilisticLength;

impl ResolutionFunction for ProbabilisticLength {
    fn name(&self) -> &str {
        "probabilistic_len"
    }
    
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> Result<ResolutionResult> {
        let mut possibilities = Vec::new();
        let base_confidence = point.confidence;
        
        // Character count (most certain)
        let char_count = point.content.chars().count() as f64;
        possibilities.push((Value::Number(char_count), base_confidence * 0.95));
        
        // Word count (fairly certain)
        let word_count = point.content.split_whitespace().count() as f64;
        let word_result = Value::Object({
            let mut map = HashMap::new();
            map.insert("unit".to_string(), Value::String("words".to_string()));
            map.insert("count".to_string(), Value::Number(word_count));
            map
        });
        possibilities.push((word_result, base_confidence * 0.85));
        
        // Semantic units (less certain - depends on interpretation)
        let semantic_count = if point.interpretations.is_empty() {
            1.0 // Default to one semantic unit
        } else {
            // Count distinct meanings
            point.interpretations.len() as f64
        };
        
        let semantic_result = Value::Object({
            let mut map = HashMap::new();
            map.insert("unit".to_string(), Value::String("semantic_units".to_string()));
            map.insert("count".to_string(), Value::Number(semantic_count));
            map
        });
        possibilities.push((semantic_result, base_confidence * 0.6));
        
        // Contextual length (depends on domain)
        if let Some(domain) = &context.domain {
            let contextual_length = match domain.as_str() {
                "twitter" => {
                    // Twitter context: categorize by tweet length norms
                    if char_count <= 50.0 { 0.3 } // short
                    else if char_count <= 140.0 { 0.6 } // medium  
                    else { 0.9 } // long
                },
                "academic" => {
                    // Academic context: different norms
                    if word_count <= 10.0 { 0.2 } // very short
                    else if word_count <= 100.0 { 0.5 } // normal
                    else { 0.8 } // long
                },
                _ => 0.5, // neutral
            };
            
            let contextual_result = Value::Object({
                let mut map = HashMap::new();
                map.insert("unit".to_string(), Value::String("contextual".to_string()));
                map.insert("relative_length".to_string(), Value::Number(contextual_length));
                map.insert("domain".to_string(), Value::String(domain.clone()));
                map
            });
            possibilities.push((contextual_result, base_confidence * 0.7));
        }
        
        // Calculate confidence interval
        let confidences: Vec<f64> = possibilities.iter().map(|(_, conf)| *conf).collect();
        let min_conf = confidences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_conf = confidences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Aggregate confidence (average weighted by individual confidences)
        let total_weight: f64 = confidences.iter().sum();
        let aggregated_confidence = if total_weight > 0.0 {
            total_weight / possibilities.len() as f64
        } else {
            0.0
        };
        
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (min_conf, max_conf),
            aggregated_confidence,
        })
    }
    
    fn uncertainty_factor(&self) -> f64 {
        0.2 // Length has some inherent uncertainty in text
    }
    
    fn can_handle(&self, _point: &TextPoint) -> bool {
        true // Can handle any text point
    }
}

/// Manager for resolution functions
pub struct ResolutionManager {
    functions: HashMap<String, Box<dyn ResolutionFunction>>,
    default_context: ResolutionContext,
}

impl ResolutionManager {
    pub fn new() -> Self {
        let mut manager = Self {
            functions: HashMap::new(),
            default_context: ResolutionContext {
                domain: None,
                culture: None,
                time_period: None,
                purpose: None,
                parameters: HashMap::new(),
                resolution_strategy: ResolutionStrategy::MaximumLikelihood,
            },
        };
        
        // Register default resolution functions
        manager.register_function(Box::new(ProbabilisticLength));
        
        manager
    }
    
    pub fn register_function(&mut self, function: Box<dyn ResolutionFunction>) {
        self.functions.insert(function.name().to_string(), function);
    }
    
    pub fn resolve(&self, function_name: &str, point: &TextPoint, context: Option<&ResolutionContext>) -> Result<ResolutionResult> {
        let function = self.functions.get(function_name)
            .ok_or_else(|| TurbulanceError::RuntimeError {
                message: format!("Unknown resolution function: {}", function_name),
            })?;
            
        let ctx = context.unwrap_or(&self.default_context);
        
        if !function.can_handle(point) {
            return Err(TurbulanceError::RuntimeError {
                message: format!("Function '{}' cannot handle this text point", function_name),
            });
        }
        
        function.resolve(point, ctx)
    }
    
    pub fn available_functions(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }
}

/// Create a TextPoint from a string value (convenience function)
pub fn point(content: &str, confidence: f64) -> TextPoint {
    TextPoint::new(content.to_string(), confidence)
}

/// Create a resolution context with the given domain
pub fn context(domain: &str) -> ResolutionContext {
    ResolutionContext {
        domain: Some(domain.to_string()),
        culture: None,
        time_period: None,
        purpose: None,
        parameters: HashMap::new(),
        resolution_strategy: ResolutionStrategy::MaximumLikelihood,
    }
}

impl Default for ResolutionContext {
    fn default() -> Self {
        Self {
            domain: None,
            culture: None,
            time_period: None,
            purpose: None,
            parameters: HashMap::new(),
            resolution_strategy: ResolutionStrategy::MaximumLikelihood,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_point_creation() {
        let point = point("Hello world", 0.9);
        assert_eq!(point.content, "Hello world");
        assert_eq!(point.confidence, 0.9);
        assert!(point.interpretations.is_empty());
    }
    
    #[test]
    fn test_probabilistic_length() {
        let text_point = point("Hello world", 0.9);
        let manager = ResolutionManager::new();
        let context = context("twitter");
        
        let result = manager.resolve("probabilistic_len", &text_point, Some(&context)).unwrap();
        
        match result {
            ResolutionResult::Uncertain { possibilities, .. } => {
                assert!(!possibilities.is_empty());
                // Should have multiple ways to measure length
                assert!(possibilities.len() >= 3);
            },
            _ => panic!("Expected uncertain result"),
        }
    }
    
    #[test]
    fn test_interpretation_entropy() {
        let mut point = point("bank", 0.8);
        
        // Add multiple interpretations for ambiguous word "bank"
        point.add_interpretation(TextInterpretation {
            meaning: "financial institution".to_string(),
            probability: 0.6,
            context: HashMap::new(),
            evidence: vec!["context suggests money".to_string()],
            features: vec![],
        });
        
        point.add_interpretation(TextInterpretation {
            meaning: "river bank".to_string(),
            probability: 0.4,
            context: HashMap::new(),
            evidence: vec!["context suggests water".to_string()],
            features: vec![],
        });
        
        let entropy = point.interpretation_entropy();
        assert!(entropy > 0.0); // Should have some entropy due to ambiguity
        assert!(entropy < 1.0); // But not maximum entropy since probabilities differ
    }
} 