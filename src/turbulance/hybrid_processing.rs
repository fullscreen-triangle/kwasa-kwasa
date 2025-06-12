/// Hybrid Processing Module for Kwasa-Kwasa
/// 
/// This module implements hybrid control flow that combines deterministic and 
/// probabilistic operations within the same loop constructs, allowing for 
/// adaptive processing based on uncertainty levels and resolution confidence.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::probabilistic::{TextPoint, ResolutionResult, ResolutionManager, ResolutionContext};
use crate::turbulance::debate_platform::{DebatePlatform, DebatePlatformManager};
use crate::turbulance::streaming::{TextStream, StreamConfig};

/// A probabilistic floor - an iterable collection of points with uncertainty
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbabilisticFloor {
    /// Points with their probability weights
    pub points: HashMap<String, (TextPoint, f64)>,
    
    /// Total probability mass
    pub total_mass: f64,
    
    /// Uncertainty threshold for probabilistic operations
    pub uncertainty_threshold: f64,
    
    /// Processing metadata
    pub metadata: FloorMetadata,
}

/// Metadata for probabilistic floor
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FloorMetadata {
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    
    /// Processing statistics
    pub stats: FloorStats,
    
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
}

/// Statistics for probabilistic floor operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FloorStats {
    /// Total cycles performed
    pub cycles_performed: u64,
    
    /// Probabilistic drifts executed
    pub drifts_executed: u64,
    
    /// Streaming flows processed
    pub flows_processed: u64,
    
    /// Roll operations until settlement
    pub rolls_until_settled: u64,
    
    /// Average resolution time
    pub avg_resolution_time_ms: f64,
    
    /// Success rate of settlements
    pub settlement_success_rate: f64,
}

/// Hybrid loop processor that can switch between deterministic and probabilistic modes
pub struct HybridProcessor {
    /// Resolution manager for probabilistic operations
    resolution_manager: ResolutionManager,
    
    /// Debate platform manager for iterative resolution
    debate_manager: DebatePlatformManager,
    
    /// Text stream for flow operations
    text_stream: TextStream,
    
    /// Current processing context
    context: HybridContext,
    
    /// Configuration
    config: HybridConfig,
}

/// Context for hybrid processing
#[derive(Clone, Debug)]
pub struct HybridContext {
    /// Current processing mode
    pub mode: ProcessingMode,
    
    /// Active probabilistic floor
    pub floor: Option<ProbabilisticFloor>,
    
    /// Resolution context
    pub resolution_context: ResolutionContext,
    
    /// Loop state variables
    pub loop_state: LoopState,
    
    /// Accumulated results
    pub results: Vec<HybridResult>,
}

/// Processing modes for hybrid operations
#[derive(Clone, Debug, PartialEq)]
pub enum ProcessingMode {
    /// Basic deterministic cycling
    Cycle,
    
    /// Probabilistic drifting through corpus
    Drift,
    
    /// Streaming flow processing
    Flow,
    
    /// Iterative rolling until settlement
    RollUntilSettled,
    
    /// Hybrid mode - switches based on uncertainty
    Hybrid,
}

/// State of hybrid loops
#[derive(Clone, Debug, Default)]
pub struct LoopState {
    /// Current iteration count
    pub iteration: u64,
    
    /// Accumulated confidence
    pub accumulated_confidence: f64,
    
    /// Settlement threshold
    pub settlement_threshold: f64,
    
    /// Maximum iterations before forced settlement
    pub max_iterations: u64,
    
    /// Current probabilistic weight
    pub current_weight: f64,
    
    /// Processing variables
    pub variables: HashMap<String, Value>,
}

/// Configuration for hybrid processing
#[derive(Clone, Debug)]
pub struct HybridConfig {
    /// Threshold for switching to probabilistic mode
    pub probabilistic_threshold: f64,
    
    /// Settlement confidence threshold
    pub settlement_threshold: f64,
    
    /// Maximum iterations for roll operations
    pub max_roll_iterations: u64,
    
    /// Enable adaptive loop behavior
    pub enable_adaptive_loops: bool,
    
    /// Probability density resolution
    pub density_resolution: usize,
    
    /// Stream processing buffer size
    pub stream_buffer_size: usize,
}

/// Result of hybrid processing operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridResult {
    /// Operation type performed
    pub operation: String,
    
    /// Input content
    pub input: String,
    
    /// Resolution result
    pub resolution: ResolutionResult,
    
    /// Processing mode used
    pub mode: String,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Number of iterations required
    pub iterations: u64,
    
    /// Settlement achieved
    pub settled: bool,
    
    /// Processing metadata
    pub metadata: HashMap<String, Value>,
}

impl HybridProcessor {
    /// Create a new hybrid processor
    pub fn new(config: HybridConfig) -> Self {
        Self {
            resolution_manager: ResolutionManager::new(),
            debate_manager: DebatePlatformManager::new(),
            text_stream: TextStream::new(StreamConfig {
                sentence_buffer_size: config.stream_buffer_size,
                ..StreamConfig::default()
            }),
            context: HybridContext {
                mode: ProcessingMode::Hybrid,
                floor: None,
                resolution_context: ResolutionContext::default(),
                loop_state: LoopState {
                    settlement_threshold: config.settlement_threshold,
                    max_iterations: config.max_roll_iterations,
                    ..LoopState::default()
                },
                results: Vec::new(),
            },
            config,
        }
    }
    
    /// Execute a cycle operation - basic iteration over floor items
    pub async fn cycle<F>(&mut self, floor: &ProbabilisticFloor, mut operation: F) -> Result<Vec<HybridResult>>
    where
        F: FnMut(&TextPoint, f64) -> Result<ResolutionResult>,
    {
        let mut results = Vec::new();
        self.context.mode = ProcessingMode::Cycle;
        
        for (key, (point, weight)) in &floor.points {
            self.context.loop_state.iteration += 1;
            
            // Basic deterministic operation
            let resolution = operation(point, *weight)?;
            
            let result = HybridResult {
                operation: "cycle".to_string(),
                input: point.content.clone(),
                resolution,
                mode: "deterministic".to_string(),
                confidence: point.confidence,
                iterations: 1,
                settled: true,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("key".to_string(), Value::String(key.clone()));
                    meta.insert("weight".to_string(), Value::Number(*weight));
                    meta
                },
            };
            
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Execute a drift operation - probabilistic processing over corpus
    pub async fn drift(&mut self, text_corpus: &str) -> Result<Vec<HybridResult>> {
        let mut results = Vec::new();
        self.context.mode = ProcessingMode::Drift;
        
        // Process text through stream to extract points
        let stream_result = self.text_stream.process_chunk(text_corpus).await?;
        
        for point in stream_result.points {
            // Determine if we should use probabilistic processing
            let use_probabilistic = point.confidence < self.config.probabilistic_threshold;
            
            let resolution = if use_probabilistic {
                // Probabilistic resolution with uncertainty handling
                self.probabilistic_resolve(&point).await?
            } else {
                // Deterministic resolution
                ResolutionResult::Certain(Value::String(point.content.clone()))
            };
            
            let result = HybridResult {
                operation: "drift".to_string(),
                input: point.content.clone(),
                resolution,
                mode: if use_probabilistic { "probabilistic" } else { "deterministic" }.to_string(),
                confidence: point.confidence,
                iterations: 1,
                settled: point.confidence > self.config.settlement_threshold,
                metadata: HashMap::new(),
            };
            
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Execute a flow operation - streaming processing over lines
    pub async fn flow(&mut self, text_lines: &[String]) -> Result<Vec<HybridResult>> {
        let mut results = Vec::new();
        self.context.mode = ProcessingMode::Flow;
        
        for line in text_lines {
            if line.trim().is_empty() {
                continue;
            }
            
            // Process line through streaming
            let stream_result = self.text_stream.process_chunk(line).await?;
            
            for point in stream_result.points {
                let resolution = self.adaptive_resolve(&point).await?;
                
                let result = HybridResult {
                    operation: "flow".to_string(),
                    input: point.content.clone(),
                    resolution,
                    mode: "streaming".to_string(),
                    confidence: point.confidence,
                    iterations: 1,
                    settled: true,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("line_number".to_string(), Value::Number(results.len() as f64 + 1.0));
                        meta
                    },
                };
                
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// Execute roll until settled - iterative resolution until convergence
    pub async fn roll_until_settled(&mut self, point: &TextPoint) -> Result<HybridResult> {
        self.context.mode = ProcessingMode::RollUntilSettled;
        self.context.loop_state.iteration = 0;
        self.context.loop_state.accumulated_confidence = point.confidence;
        
        let mut current_point = point.clone();
        let mut final_resolution = ResolutionResult::Uncertain {
            possibilities: vec![(Value::String(point.content.clone()), point.confidence)],
            confidence_interval: (0.0, 1.0),
            aggregated_confidence: point.confidence,
        };
        
        // Create debate platform for iterative resolution
        let platform_id = self.debate_manager.create_platform(
            current_point.clone(),
            crate::turbulance::probabilistic::ResolutionStrategy::BayesianWeighted,
            None
        );
        
        while self.context.loop_state.iteration < self.context.loop_state.max_iterations {
            self.context.loop_state.iteration += 1;
            
            // Get current platform state
            if let Some(platform) = self.debate_manager.get_platform_mut(&platform_id) {
                // Add evidence or contentions based on current confidence
                if current_point.confidence < self.config.settlement_threshold {
                    // Low confidence - add contention to trigger more analysis
                    platform.add_contention(
                        format!("Iteration {} - requires further analysis", self.context.loop_state.iteration),
                        "Iterative Processor".to_string(),
                        1.0 - current_point.confidence,
                        0.8,
                        crate::turbulance::debate_platform::ChallengeAspect::EvidenceQuality
                    ).await?;
                } else {
                    // Higher confidence - add affirmation
                    platform.add_affirmation(
                        format!("Iteration {} - evidence supports resolution", self.context.loop_state.iteration),
                        "Iterative Processor".to_string(),
                        current_point.confidence,
                        0.9
                    ).await?;
                }
                
                // Update platform score
                platform.update_score().await?;
                
                // Get updated resolution
                final_resolution = platform.get_resolution();
                
                // Update accumulated confidence
                self.context.loop_state.accumulated_confidence = platform.score_confidence;
                
                // Check for settlement
                if platform.score_confidence >= self.context.loop_state.settlement_threshold {
                    break;
                }
                
                // Update current point for next iteration
                current_point.confidence = platform.score_confidence;
            }
        }
        
        let settled = self.context.loop_state.accumulated_confidence >= self.context.loop_state.settlement_threshold;
        
        Ok(HybridResult {
            operation: "roll_until_settled".to_string(),
            input: point.content.clone(),
            resolution: final_resolution,
            mode: "iterative".to_string(),
            confidence: self.context.loop_state.accumulated_confidence,
            iterations: self.context.loop_state.iteration,
            settled,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("platform_id".to_string(), Value::String(platform_id.to_string()));
                meta.insert("final_confidence".to_string(), Value::Number(self.context.loop_state.accumulated_confidence));
                meta
            },
        })
    }
    
    /// Hybrid function with conditional probabilistic processing
    pub async fn hybrid_function<F>(&mut self, 
                                   paragraph: &str, 
                                   confidence_threshold: f64,
                                   mut sentence_processor: F) -> Result<Vec<HybridResult>>
    where
        F: FnMut(&str) -> Result<bool>,
    {
        let mut results = Vec::new();
        
        // Split paragraph into sentences
        let sentences: Vec<&str> = paragraph.split(&['.', '!', '?']).collect();
        
        for sentence in sentences {
            if sentence.trim().is_empty() {
                continue;
            }
            
            // Regular deterministic processing first
            let should_continue = sentence_processor(sentence)?;
            if !should_continue {
                continue;
            }
            
            // Extract points from sentence
            let stream_result = self.text_stream.process_chunk(sentence).await?;
            
            for point in stream_result.points {
                // Check given resolution is within threshold for probabilistic processing
                if point.confidence < confidence_threshold {
                    // Switch to probabilistic mode - iterative resolution loop
                    let mut current_confidence = point.confidence;
                    let mut iteration = 0;
                    
                    // Create debate platform for this uncertain point
                    let platform_id = self.debate_manager.create_platform(
                        point.clone(),
                        crate::turbulance::probabilistic::ResolutionStrategy::BayesianWeighted,
                        None
                    );
                    
                    // Loop until resolved or max iterations
                    while current_confidence < confidence_threshold && iteration < self.config.max_roll_iterations {
                        iteration += 1;
                        
                        if let Some(platform) = self.debate_manager.get_platform_mut(&platform_id) {
                            // Add affirmations and contentions based on analysis
                            if iteration % 2 == 1 {
                                // Odd iterations - add supporting evidence
                                platform.add_affirmation(
                                    format!("Analysis iteration {} supports this interpretation", iteration),
                                    "Hybrid Processor".to_string(),
                                    current_confidence + 0.1,
                                    0.8
                                ).await?;
                            } else {
                                // Even iterations - add challenges for robustness
                                platform.add_contention(
                                    format!("Critical review {} identifies potential issues", iteration),
                                    "Hybrid Processor".to_string(),
                                    0.3,
                                    0.7,
                                    crate::turbulance::debate_platform::ChallengeAspect::LogicalReasoning
                                ).await?;
                            }
                            
                            platform.update_score().await?;
                            current_confidence = platform.score_confidence;
                        }
                    }
                    
                    let final_resolution = if let Some(platform) = self.debate_manager.get_platform(&platform_id) {
                        platform.get_resolution()
                    } else {
                        ResolutionResult::Uncertain {
                            possibilities: vec![(Value::String(point.content.clone()), current_confidence)],
                            confidence_interval: (current_confidence * 0.9, current_confidence * 1.1),
                            aggregated_confidence: current_confidence,
                        }
                    };
                    
                    results.push(HybridResult {
                        operation: "hybrid_function".to_string(),
                        input: point.content.clone(),
                        resolution: final_resolution,
                        mode: "hybrid_probabilistic".to_string(),
                        confidence: current_confidence,
                        iterations: iteration,
                        settled: current_confidence >= confidence_threshold,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("original_confidence".to_string(), Value::Number(point.confidence));
                            meta.insert("threshold".to_string(), Value::Number(confidence_threshold));
                            meta.insert("probabilistic_loops".to_string(), Value::Number(iteration as f64));
                            meta
                        },
                    });
                } else {
                    // High confidence - use deterministic processing
                    results.push(HybridResult {
                        operation: "hybrid_function".to_string(),
                        input: point.content.clone(),
                        resolution: ResolutionResult::Certain(Value::String(point.content.clone())),
                        mode: "deterministic".to_string(),
                        confidence: point.confidence,
                        iterations: 1,
                        settled: true,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("skipped_probabilistic".to_string(), Value::Bool(true));
                            meta
                        },
                    });
                }
            }
        }
        
        Ok(results)
    }
    
    /// Probabilistic resolution with uncertainty handling
    async fn probabilistic_resolve(&mut self, point: &TextPoint) -> Result<ResolutionResult> {
        // Create multiple resolution possibilities based on uncertainty
        let base_confidence = point.confidence;
        
        let possibilities = vec![
            (Value::String(format!("High confidence: {}", point.content)), base_confidence * 1.2),
            (Value::String(format!("Medium confidence: {}", point.content)), base_confidence),
            (Value::String(format!("Low confidence: {}", point.content)), base_confidence * 0.8),
        ];
        
        Ok(ResolutionResult::Uncertain {
            possibilities,
            confidence_interval: (base_confidence * 0.7, base_confidence * 1.3),
            aggregated_confidence: base_confidence,
        })
    }
    
    /// Adaptive resolution based on context
    async fn adaptive_resolve(&mut self, point: &TextPoint) -> Result<ResolutionResult> {
        if point.confidence > 0.8 {
            Ok(ResolutionResult::Certain(Value::String(point.content.clone())))
        } else if point.confidence > 0.5 {
            self.probabilistic_resolve(point).await
        } else {
            // Very uncertain - needs iterative resolution
            let result = self.roll_until_settled(point).await?;
            Ok(result.resolution)
        }
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> &FloorStats {
        if let Some(floor) = &self.context.floor {
            &floor.metadata.stats
        } else {
            &FloorStats::default()
        }
    }
}

impl ProbabilisticFloor {
    /// Create a new probabilistic floor
    pub fn new(uncertainty_threshold: f64) -> Self {
        let now = chrono::Utc::now();
        Self {
            points: HashMap::new(),
            total_mass: 0.0,
            uncertainty_threshold,
            metadata: FloorMetadata {
                created_at: now,
                updated_at: now,
                stats: FloorStats::default(),
                quality_metrics: HashMap::new(),
            },
        }
    }
    
    /// Add a point to the floor
    pub fn add_point(&mut self, key: String, point: TextPoint, weight: f64) {
        self.points.insert(key, (point, weight));
        self.total_mass += weight;
        self.metadata.updated_at = chrono::Utc::now();
    }
    
    /// Get probabilistic iterator over points
    pub fn probabilistic_iter(&self) -> impl Iterator<Item = (&String, &TextPoint, f64)> {
        self.points.iter().map(|(k, (p, w))| (k, p, *w / self.total_mass))
    }
    
    /// Sample a point based on probability weights
    pub fn sample_point(&self) -> Option<(&String, &TextPoint)> {
        if self.points.is_empty() {
            return None;
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let target = rng.gen::<f64>() * self.total_mass;
        
        let mut current = 0.0;
        for (key, (point, weight)) in &self.points {
            current += weight;
            if current >= target {
                return Some((key, point));
            }
        }
        
        // Fallback to first item
        self.points.iter().next().map(|(k, (p, _))| (k, p))
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            probabilistic_threshold: 0.7,
            settlement_threshold: 0.8,
            max_roll_iterations: 10,
            enable_adaptive_loops: true,
            density_resolution: 100,
            stream_buffer_size: 10,
        }
    }
}

/// Demonstrate hybrid processing with the examples from the user
pub async fn demonstrate_hybrid_processing() -> Result<()> {
    println!("🔄 Hybrid Processing Demonstration 🔄\n");
    
    let mut processor = HybridProcessor::new(HybridConfig::default());
    
    // Create a probabilistic floor with some points
    let mut floor = ProbabilisticFloor::new(0.7);
    floor.add_point("point1".to_string(), TextPoint::new("The algorithm works well".to_string(), 0.9), 1.0);
    floor.add_point("point2".to_string(), TextPoint::new("Results are questionable".to_string(), 0.4), 0.8);
    floor.add_point("point3".to_string(), TextPoint::new("Further analysis needed".to_string(), 0.3), 0.6);
    
    println!("1. 🔄 Cycle Operation (Basic Iteration):");
    let cycle_results = processor.cycle(&floor, |point, weight| {
        println!("   Processing: '{}' (weight: {:.2})", point.content, weight);
        Ok(ResolutionResult::Certain(Value::String(format!("Processed: {}", point.content))))
    }).await?;
    
    println!("   Results: {} items processed\n", cycle_results.len());
    
    println!("2. 🌊 Drift Operation (Probabilistic Corpus Processing):");
    let corpus = "The quantum algorithm shows promise. However, scalability remains uncertain. More research is definitely needed.";
    let drift_results = processor.drift(corpus).await?;
    
    for result in &drift_results {
        println!("   {} mode: '{}' (confidence: {:.2})", result.mode, result.input, result.confidence);
    }
    println!();
    
    println!("3. 🏃 Flow Operation (Streaming Line Processing):");
    let lines = vec![
        "First line of analysis".to_string(),
        "Second line with uncertainty".to_string(),
        "Third line confirms findings".to_string(),
    ];
    let flow_results = processor.flow(&lines).await?;
    
    for result in &flow_results {
        println!("   Line processed: '{}' (settled: {})", result.input, result.settled);
    }
    println!();
    
    println!("4. 🎲 Roll Until Settled (Iterative Resolution):");
    let uncertain_point = TextPoint::new("This claim needs thorough verification".to_string(), 0.3);
    let roll_result = processor.roll_until_settled(&uncertain_point).await?;
    
    println!("   Original confidence: {:.2}", uncertain_point.confidence);
    println!("   Final confidence: {:.2}", roll_result.confidence);
    println!("   Iterations required: {}", roll_result.iterations);
    println!("   Settled: {}\n", roll_result.settled);
    
    println!("5. 🔀 Hybrid Function (Conditional Probabilistic Processing):");
    let paragraph = "The solution demonstrates superior performance. Critics question its scalability. The evidence shows mixed results. Final assessment requires more data.";
    
    let hybrid_results = processor.hybrid_function(paragraph, 0.6, |sentence| {
        println!("   Regular processing: '{}'", sentence);
        Ok(true) // Continue processing
    }).await?;
    
    for result in &hybrid_results {
        println!("   {} processing: '{}' (iterations: {}, settled: {})", 
                 result.mode, result.input, result.iterations, result.settled);
    }
    
    println!("\n🎵 Hybrid processing complete! 🎵");
    println!("This demonstrates:");
    println!("  • Seamless switching between deterministic and probabilistic modes");
    println!("  • Probabilistic looping based on uncertainty levels"); 
    println!("  • Iterative resolution until settlement");
    println!("  • Adaptive control flow within the same function");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_probabilistic_floor() {
        let mut floor = ProbabilisticFloor::new(0.5);
        let point = TextPoint::new("Test point".to_string(), 0.8);
        
        floor.add_point("test".to_string(), point, 1.0);
        assert_eq!(floor.points.len(), 1);
        assert_eq!(floor.total_mass, 1.0);
    }
    
    #[tokio::test]
    async fn test_cycle_operation() {
        let mut processor = HybridProcessor::new(HybridConfig::default());
        let mut floor = ProbabilisticFloor::new(0.5);
        
        floor.add_point("p1".to_string(), TextPoint::new("Point 1".to_string(), 0.9), 1.0);
        floor.add_point("p2".to_string(), TextPoint::new("Point 2".to_string(), 0.7), 0.8);
        
        let results = processor.cycle(&floor, |point, _weight| {
            Ok(ResolutionResult::Certain(Value::String(point.content.clone())))
        }).await.unwrap();
        
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.settled));
    }
    
    #[tokio::test]
    async fn test_roll_until_settled() {
        let mut processor = HybridProcessor::new(HybridConfig::default());
        let point = TextPoint::new("Uncertain claim".to_string(), 0.2);
        
        let result = processor.roll_until_settled(&point).await.unwrap();
        
        assert!(result.iterations > 1);
        assert!(result.confidence >= point.confidence); // Should improve
    }
    
    #[tokio::test]
    async fn test_hybrid_function() {
        let mut processor = HybridProcessor::new(HybridConfig::default());
        let paragraph = "Clear statement. Uncertain claim. Definitive conclusion.";
        
        let results = processor.hybrid_function(paragraph, 0.6, |_| Ok(true)).await.unwrap();
        
        assert!(!results.is_empty());
        // Should have a mix of deterministic and probabilistic results
    }
    
    #[tokio::test]
    async fn test_demonstration() {
        let result = demonstrate_hybrid_processing().await;
        assert!(result.is_ok());
    }
} 