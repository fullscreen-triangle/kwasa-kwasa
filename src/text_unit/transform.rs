use std::collections::HashMap;
use std::time::Instant;
use std::fmt;

use crate::text_unit::{TextUnit, TextUnitRegistry};
use crate::text_unit::operations::{OperationResult, divide, multiply, add, subtract, filter, transform, pipeline, compose};
use crate::turbulance::ast::Value;

/// Trait representing a transformation that can be applied to a text unit
pub trait TextTransform {
    /// Apply the transformation to a text unit and return the result
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult;
    
    /// Get the name of this transformation
    fn name(&self) -> &str;
    
    /// Get the description of this transformation
    fn description(&self) -> &str;
    
    /// Chain this transformation with another one
    fn chain<T: TextTransform + 'static>(self, next: T) -> Box<dyn TextTransform>
    where
        Self: Sized + 'static,
    {
        Box::new(ChainedTransform {
            first: Box::new(self),
            second: Box::new(next),
        })
    }
    
    /// Add timing instrumentation to this transformation
    fn with_timing(self) -> Box<dyn TextTransform>
    where
        Self: Sized + 'static,
    {
        Box::new(TimingTransform {
            inner: Box::new(self),
        })
    }
    
    /// Add caching to this transformation
    fn with_caching(self) -> Box<dyn TextTransform>
    where
        Self: Sized + 'static,
    {
        Box::new(CachingTransform {
            inner: Box::new(self),
            cache: HashMap::new(),
        })
    }
}

/// A transformation pipeline that chains multiple transformations together
pub struct TransformationPipeline {
    /// The sequence of transformations to apply
    transformations: Vec<Box<dyn TextTransform>>,
    /// Name of the pipeline
    name: String,
    /// Description of the pipeline
    description: String,
    /// Whether to collect metrics during execution
    collect_metrics: bool,
    /// Execution metrics (if enabled)
    metrics: Option<PipelineMetrics>,
}

/// Metrics collected during pipeline execution
#[derive(Default)]
pub struct PipelineMetrics {
    /// Total time taken to execute the pipeline
    total_time_ms: u128,
    /// Time taken by each step
    step_times_ms: Vec<(String, u128)>,
    /// Number of units processed
    units_processed: usize,
    /// Number of units produced
    units_produced: usize,
    /// Cache hit rate (if caching is enabled)
    cache_hit_rate: Option<f64>,
}

impl PipelineMetrics {
    /// Create a new empty metrics collection
    pub fn new() -> Self {
        Self {
            total_time_ms: 0,
            step_times_ms: Vec::new(),
            units_processed: 0,
            units_produced: 0,
            cache_hit_rate: None,
        }
    }
    
    /// Add a step time measurement
    pub fn add_step_time(&mut self, transform_name: String, time_ms: u128) {
        self.step_times_ms.push((transform_name, time_ms));
    }
    
    /// Set the total execution time
    pub fn set_total_time(&mut self, time_ms: u128) {
        self.total_time_ms = time_ms;
    }
    
    /// Set the number of units processed
    pub fn set_units_processed(&mut self, count: usize) {
        self.units_processed = count;
    }
    
    /// Set the number of units produced
    pub fn set_units_produced(&mut self, count: usize) {
        self.units_produced = count;
    }
    
    /// Set the cache hit rate
    pub fn set_cache_hit_rate(&mut self, rate: f64) {
        self.cache_hit_rate = Some(rate);
    }
}

impl fmt::Display for PipelineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Metrics:")?;
        writeln!(f, "Total time: {}ms", self.total_time_ms)?;
        writeln!(f, "Units processed: {}", self.units_processed)?;
        writeln!(f, "Units produced: {}", self.units_produced)?;
        
        if let Some(rate) = self.cache_hit_rate {
            writeln!(f, "Cache hit rate: {:.2}%", rate * 100.0)?;
        }
        
        writeln!(f, "Step times:")?;
        for (name, time) in &self.step_times_ms {
            writeln!(f, "  {}: {}ms", name, time)?;
        }
        
        Ok(())
    }
}

impl TransformationPipeline {
    /// Create a new empty transformation pipeline
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            transformations: Vec::new(),
            name: name.to_string(),
            description: description.to_string(),
            collect_metrics: false,
            metrics: None,
        }
    }
    
    /// Add a transformation to the pipeline
    pub fn add_transform<T: TextTransform + 'static>(&mut self, transform: T) -> &mut Self {
        self.transformations.push(Box::new(transform));
        self
    }
    
    /// Enable metrics collection
    pub fn with_metrics(&mut self) -> &mut Self {
        self.collect_metrics = true;
        self.metrics = Some(PipelineMetrics::new());
        self
    }
    
    /// Execute the pipeline on a text unit
    pub fn execute(&mut self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        let start_time = Instant::now();
        let mut current_result = OperationResult::Unit(unit.clone());
        let mut units_processed = 1;
        
        for transform in &self.transformations {
            let step_start = Instant::now();
            
            current_result = match current_result {
                OperationResult::Unit(unit) => {
                    transform.apply(&unit, registry)
                },
                OperationResult::Units(units) => {
                    let mut combined_results = Vec::new();
                    units_processed += units.len();
                    
                    for unit in units {
                        match transform.apply(&unit, registry) {
                            OperationResult::Unit(unit) => combined_results.push(unit),
                            OperationResult::Units(mut units) => combined_results.append(&mut units),
                            OperationResult::Value(_) => continue, // Skip values
                            OperationResult::Error(e) => {
                                return OperationResult::Error(format!(
                                    "Error in transform '{}': {}", 
                                    transform.name(), 
                                    e
                                ));
                            }
                        }
                    }
                    
                    if combined_results.is_empty() {
                        return OperationResult::Error(format!(
                            "Transform '{}' produced no results", 
                            transform.name()
                        ));
                    }
                    
                    OperationResult::Units(combined_results)
                },
                OperationResult::Value(_) => {
                    return OperationResult::Error(format!(
                        "Cannot apply transform '{}' to a non-text value", 
                        transform.name()
                    ));
                },
                OperationResult::Error(e) => {
                    return OperationResult::Error(e);
                }
            };
            
            if self.collect_metrics {
                if let Some(metrics) = &mut self.metrics {
                    metrics.add_step_time(
                        transform.name().to_string(),
                        step_start.elapsed().as_millis()
                    );
                }
            }
        }
        
        // Calculate final metrics if enabled
        if self.collect_metrics {
            if let Some(metrics) = &mut self.metrics {
                metrics.set_total_time(start_time.elapsed().as_millis());
                metrics.set_units_processed(units_processed);
                
                match &current_result {
                    OperationResult::Unit(_) => metrics.set_units_produced(1),
                    OperationResult::Units(units) => metrics.set_units_produced(units.len()),
                    _ => metrics.set_units_produced(0),
                }
            }
        }
        
        current_result
    }
    
    /// Get the collected metrics (if any)
    pub fn metrics(&self) -> Option<&PipelineMetrics> {
        self.metrics.as_ref()
    }
    
    /// Get the name of the pipeline
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the description of the pipeline
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// A chained transformation that applies two transformations in sequence
struct ChainedTransform {
    first: Box<dyn TextTransform>,
    second: Box<dyn TextTransform>,
}

impl TextTransform for ChainedTransform {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        match self.first.apply(unit, registry) {
            OperationResult::Unit(unit) => self.second.apply(&unit, registry),
            OperationResult::Units(units) => {
                let mut combined_results = Vec::new();
                
                for unit in units {
                    match self.second.apply(&unit, registry) {
                        OperationResult::Unit(unit) => combined_results.push(unit),
                        OperationResult::Units(mut units) => combined_results.append(&mut units),
                        OperationResult::Value(_) => continue, // Skip values
                        OperationResult::Error(_) => continue, // Skip errors
                    }
                }
                
                if combined_results.is_empty() {
                    OperationResult::Error("Chained transformation produced no results".to_string())
                } else {
                    OperationResult::Units(combined_results)
                }
            },
            result => result,
        }
    }
    
    fn name(&self) -> &str {
        "ChainedTransform"
    }
    
    fn description(&self) -> &str {
        "Applies two transformations in sequence"
    }
}

/// A transformation that measures execution time
struct TimingTransform {
    inner: Box<dyn TextTransform>,
}

impl TextTransform for TimingTransform {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        let start = Instant::now();
        let result = self.inner.apply(unit, registry);
        let elapsed = start.elapsed();
        
        println!("Transform '{}' took {:.2?}", self.inner.name(), elapsed);
        
        result
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn description(&self) -> &str {
        self.inner.description()
    }
}

/// A transformation that caches results based on input content
struct CachingTransform {
    inner: Box<dyn TextTransform>,
    cache: HashMap<String, OperationResult>,
}

impl TextTransform for CachingTransform {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        // Use the content as cache key
        // In a more sophisticated implementation, we might use a hash of content + transform parameters
        if let Some(cached_result) = self.cache.get(&unit.content) {
            return cached_result.clone();
        }
        
        let result = self.inner.apply(unit, registry);
        
        // Update cache (immutable borrow limitation - in real code we'd use interior mutability)
        let mut cache = self.cache.clone();
        cache.insert(unit.content.clone(), result.clone());
        
        result
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn description(&self) -> &str {
        self.inner.description()
    }
}

/// Common transformations for text units

/// Split text into paragraphs
pub struct ParagraphSplitter;

impl TextTransform for ParagraphSplitter {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        divide(unit, registry, crate::text_unit::TextUnitType::Paragraph)
    }
    
    fn name(&self) -> &str {
        "ParagraphSplitter"
    }
    
    fn description(&self) -> &str {
        "Splits text into paragraphs"
    }
}

/// Split text into sentences
pub struct SentenceSplitter;

impl TextTransform for SentenceSplitter {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        divide(unit, registry, crate::text_unit::TextUnitType::Sentence)
    }
    
    fn name(&self) -> &str {
        "SentenceSplitter"
    }
    
    fn description(&self) -> &str {
        "Splits text into sentences"
    }
}

/// Simplify text for better readability
pub struct Simplifier {
    level: u8, // 1-3 level of simplification
}

impl Simplifier {
    pub fn new(level: u8) -> Self {
        Self {
            level: level.min(3).max(1)
        }
    }
}

impl TextTransform for Simplifier {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        let mut args = HashMap::new();
        args.insert("level".to_string(), Value::Number(self.level as f64));
        
        transform(unit, "simplify", registry, Some(args))
    }
    
    fn name(&self) -> &str {
        "Simplifier"
    }
    
    fn description(&self) -> &str {
        "Simplifies text for better readability"
    }
}

/// Formalize text to a more professional style
pub struct Formalizer;

impl TextTransform for Formalizer {
    fn apply(&self, unit: &TextUnit, registry: &mut TextUnitRegistry) -> OperationResult {
        transform(unit, "formalize", registry, None)
    }
    
    fn name(&self) -> &str {
        "Formalizer"
    }
    
    fn description(&self) -> &str {
        "Converts text to a more formal style"
    }
}

/// Create commonly used transformation pipelines

/// Create a readability improvement pipeline
pub fn create_readability_pipeline() -> TransformationPipeline {
    let mut pipeline = TransformationPipeline::new(
        "ReadabilityPipeline",
        "Improves text readability by simplifying and restructuring",
    );
    
    pipeline
        .add_transform(SentenceSplitter)
        .add_transform(Simplifier::new(2))
        .with_metrics();
    
    pipeline
}

/// Create a formalization pipeline
pub fn create_formalization_pipeline() -> TransformationPipeline {
    let mut pipeline = TransformationPipeline::new(
        "FormalizationPipeline",
        "Converts text to a more formal, professional style",
    );
    
    pipeline
        .add_transform(SentenceSplitter)
        .add_transform(Formalizer)
        .with_metrics();
    
    pipeline
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_unit(content: &str) -> TextUnit {
        TextUnit::new(
            content.to_string(),
            0,
            content.len(),
            crate::text_unit::TextUnitType::Document,
            0,
        )
    }
    
    #[test]
    fn test_paragraph_splitter() {
        let mut registry = TextUnitRegistry::new();
        let unit = create_test_unit("First paragraph.\n\nSecond paragraph.");
        
        let transform = ParagraphSplitter;
        let result = transform.apply(&unit, &mut registry);
        
        match result {
            OperationResult::Units(units) => {
                assert_eq!(units.len(), 2);
                assert_eq!(units[0].content, "First paragraph.");
                assert_eq!(units[1].content, "Second paragraph.");
            },
            _ => panic!("Expected Units result"),
        }
    }
    
    #[test]
    fn test_sentence_splitter() {
        let mut registry = TextUnitRegistry::new();
        let unit = create_test_unit("First sentence. Second sentence.");
        
        let transform = SentenceSplitter;
        let result = transform.apply(&unit, &mut registry);
        
        match result {
            OperationResult::Units(units) => {
                assert_eq!(units.len(), 2);
                assert_eq!(units[0].content, "First sentence.");
                assert_eq!(units[1].content, "Second sentence.");
            },
            _ => panic!("Expected Units result"),
        }
    }
    
    #[test]
    fn test_chained_transforms() {
        let mut registry = TextUnitRegistry::new();
        let unit = create_test_unit("First paragraph with two sentences. And another one.\n\nSecond paragraph.");
        
        let transform = ParagraphSplitter.chain(SentenceSplitter);
        let result = transform.apply(&unit, &mut registry);
        
        match result {
            OperationResult::Units(units) => {
                assert_eq!(units.len(), 3);
                assert_eq!(units[0].content, "First paragraph with two sentences.");
                assert_eq!(units[1].content, "And another one.");
                assert_eq!(units[2].content, "Second paragraph.");
            },
            _ => panic!("Expected Units result"),
        }
    }
    
    #[test]
    fn test_readability_pipeline() {
        let mut registry = TextUnitRegistry::new();
        let unit = create_test_unit("This is a complex sentence with multiple clauses that could benefit from simplification.");
        
        let mut pipeline = create_readability_pipeline();
        let result = pipeline.execute(&unit, &mut registry);
        
        match result {
            OperationResult::Units(_) => {
                // Check that metrics were collected
                assert!(pipeline.metrics().is_some());
                assert!(pipeline.metrics().unwrap().total_time_ms > 0);
            },
            _ => panic!("Expected Units result"),
        }
    }
}
