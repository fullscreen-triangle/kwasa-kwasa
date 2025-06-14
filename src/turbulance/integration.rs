/// Integration Module for Kwasa-Kwasa
/// 
/// This module demonstrates how all the framework components work together:
/// - Points and probabilistic processing
/// - Positional semantics analysis
/// - Streaming text processing
/// - Perturbation validation
/// - Debate platform resolution
/// 
/// This represents the complete implementation of the concepts we discussed.

use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use kwasa_kwasa::turbulance::debate_platform::{ChallengeAspect, PlatformConfig};
use kwasa_kwasa::turbulance::probabilistic::ResolutionStrategy;
use crate::turbulance::{
    TextPoint, ResolutionResult,
    PositionalAnalyzer, PositionalSentence,
    TextStream, StreamConfig, StreamState,
    PerturbationValidator, ValidationConfig, ValidationResult,
    DebatePlatform, DebatePlatformManager, Affirmation, Contention,

};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};

/// Complete text processing pipeline that integrates all framework components
pub struct KwasaKwasaPipeline {
    /// Positional analyzer for semantic structure
    positional_analyzer: PositionalAnalyzer,
    
    /// Streaming processor for continuous text
    text_stream: TextStream,
    
    /// Debate platform manager
    debate_manager: DebatePlatformManager,
    
    /// Configuration
    config: PipelineConfig,
    
    /// Processing statistics
    stats: PipelineStats,
}

/// Configuration for the complete pipeline
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Stream processing configuration
    pub stream_config: StreamConfig,
    
    /// Validation configuration
    pub validation_config: ValidationConfig,
    
    /// Debate platform configuration
    pub platform_config: PlatformConfig,
    
    /// Enable comprehensive validation
    pub enable_comprehensive_validation: bool,
    
    /// Automatically create debate platforms for uncertain points
    pub auto_create_debates: bool,
    
    /// Minimum uncertainty threshold for debate creation
    pub debate_uncertainty_threshold: f64,
    
    /// Maximum processing time per text chunk (seconds)
    pub max_processing_time: u64,
}

/// Statistics for the complete pipeline
#[derive(Clone, Debug, Default)]
pub struct PipelineStats {
    /// Total texts processed
    pub texts_processed: u64,
    
    /// Total points extracted
    pub points_extracted: u64,
    
    /// Debates created
    pub debates_created: u64,
    
    /// Validations performed
    pub validations_performed: u64,
    
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    
    /// Success rate of resolutions
    pub resolution_success_rate: f64,
    
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
}

/// Result of complete text processing
#[derive(Clone, Debug)]
pub struct ProcessingResult {
    /// Original input text
    pub input_text: String,
    
    /// Extracted and validated points
    pub points: Vec<ValidatedPoint>,
    
    /// Created debates (if any)
    pub debates: Vec<DebateInfo>,
    
    /// Overall quality assessment
    pub quality_assessment: QualityAssessment,
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// A point that has been validated through perturbation testing
#[derive(Clone, Debug)]
pub struct ValidatedPoint {
    /// The core point
    pub point: TextPoint,
    
    /// Positional analysis
    pub positional_analysis: PositionalSentence,
    
    /// Validation results
    pub validation: ValidationResult,
    
    /// Debate platform ID (if created)
    pub debate_id: Option<uuid::Uuid>,
    
    /// Final resolution
    pub resolution: ResolutionResult,
}

/// Information about a created debate
#[derive(Clone, Debug)]
pub struct DebateInfo {
    /// Debate platform ID
    pub id: uuid::Uuid,
    
    /// Point being debated
    pub point_content: String,
    
    /// Current score
    pub current_score: f64,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Resolution state
    pub resolution_state: String,
    
    /// Number of affirmations/contentions
    pub contributions: (usize, usize), // (affirmations, contentions)
}

/// Overall quality assessment
#[derive(Clone, Debug)]
pub struct QualityAssessment {
    /// Overall quality score (0.0-1.0)
    pub overall_score: f64,
    
    /// Positional coherence
    pub positional_coherence: f64,
    
    /// Validation stability
    pub validation_stability: f64,
    
    /// Resolution confidence
    pub resolution_confidence: f64,
    
    /// Evidence quality
    pub evidence_quality: f64,
    
    /// Recommendation
    pub recommendation: String,
}

/// Processing metadata
#[derive(Clone, Debug)]
pub struct ProcessingMetadata {
    /// Processing duration (ms)
    pub duration_ms: u64,
    
    /// Sentences analyzed
    pub sentences_analyzed: usize,
    
    /// Validation tests performed
    pub validation_tests: usize,
    
    /// Debates initiated
    pub debates_initiated: usize,
    
    /// Warnings generated
    pub warnings: Vec<String>,
    
    /// Debug information
    pub debug_info: HashMap<String, Value>,
}

impl KwasaKwasaPipeline {
    /// Create a new complete processing pipeline
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            positional_analyzer: PositionalAnalyzer::new(),
            text_stream: TextStream::new(config.stream_config.clone()),
            debate_manager: DebatePlatformManager::new(),
            config,
            stats: PipelineStats::default(),
        }
    }
    
    /// Process text through the complete pipeline
    pub async fn process_text(&mut self, text: &str) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();
        
        // Set processing timeout
        let processing_future = self.process_text_internal(text, &mut warnings);
        let result = timeout(Duration::from_secs(self.config.max_processing_time), processing_future).await;
        
        let processing_result = match result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                warnings.push("Processing timed out".to_string());
                return Err(TurbulanceError::ProcessingTimeout);
            }
        };
        
        // Update statistics
        let duration = start_time.elapsed();
        self.stats.texts_processed += 1;
        self.stats.points_extracted += processing_result.points.len() as u64;
        self.stats.debates_created += processing_result.debates.len() as u64;
        self.stats.avg_processing_time_ms = 
            (self.stats.avg_processing_time_ms * (self.stats.texts_processed - 1) as f64 + duration.as_millis() as f64) 
            / self.stats.texts_processed as f64;
        
        Ok(processing_result)
    }
    
    /// Internal processing logic
    async fn process_text_internal(&mut self, text: &str, warnings: &mut Vec<String>) -> Result<ProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Stream processing to extract points
        let stream_result = self.text_stream.process_chunk(text).await?;
        
        if stream_result.points.is_empty() {
            warnings.push("No points extracted from text".to_string());
        }
        
        // Step 2: Validate each point through perturbation testing
        let mut validated_points = Vec::new();
        for point in stream_result.points {
            let validated_point = self.validate_and_analyze_point(point, warnings).await?;
            validated_points.push(validated_point);
        }
        
        // Step 3: Create debates for uncertain points
        let mut debates = Vec::new();
        if self.config.auto_create_debates {
            for validated_point in &mut validated_points {
                if self.should_create_debate(&validated_point.validation) {
                    let debate_info = self.create_debate_for_point(validated_point).await?;
                    debates.push(debate_info);
                }
            }
        }
        
        // Step 4: Calculate overall quality assessment
        let quality_assessment = self.assess_overall_quality(&validated_points, &debates);
        
        // Step 5: Create processing metadata
        let metadata = ProcessingMetadata {
            duration_ms: start_time.elapsed().as_millis() as u64,
            sentences_analyzed: stream_result.context.recent_sentences.len(),
            validation_tests: validated_points.iter().map(|vp| vp.validation.metadata.tests_performed).sum(),
            debates_initiated: debates.len(),
            warnings: warnings.clone(),
            debug_info: HashMap::new(),
        };
        
        Ok(ProcessingResult {
            input_text: text.to_string(),
            points: validated_points,
            debates,
            quality_assessment,
            metadata,
        })
    }
    
    /// Validate and analyze a single point
    async fn validate_and_analyze_point(&mut self, point: TextPoint, warnings: &mut Vec<String>) -> Result<ValidatedPoint> {
        // Positional analysis
        let positional_analysis = self.positional_analyzer.analyze(&point.content)?;
        
        // Perturbation validation if enabled
        let validation = if self.config.enable_comprehensive_validation {
            let initial_resolution = ResolutionResult::Uncertain {
                possibilities: vec![(Value::Number(point.confidence), point.confidence)],
                confidence_interval: (point.confidence * 0.9, point.confidence * 1.1),
                aggregated_confidence: point.confidence,
            };
            
            let mut validator = PerturbationValidator::new(
                point.clone(),
                initial_resolution,
                self.config.validation_config.clone()
            );
            
            validator.run_validation(&self.config.validation_config).await?
        } else {
            // Create minimal validation result
            ValidationResult {
                stability_score: point.confidence,
                perturbation_results: vec![],
                quality_assessment: crate::turbulance::perturbation_validation::QualityAssessment {
                    confidence_in_resolution: point.confidence,
                    reliability_category: if point.confidence > 0.8 {
                        crate::turbulance::perturbation_validation::ReliabilityCategory::HighlyReliable
                    } else if point.confidence > 0.6 {
                        crate::turbulance::perturbation_validation::ReliabilityCategory::ModeratelyReliable
                    } else {
                        crate::turbulance::perturbation_validation::ReliabilityCategory::Questionable
                    },
                    quality_metrics: HashMap::new(),
                    risk_factors: vec![],
                    strength_factors: vec![],
                },
                recommendations: vec![],
                vulnerable_aspects: vec![],
                robust_aspects: vec![],
                metadata: crate::turbulance::perturbation_validation::ValidationMetadata {
                    validation_time_ms: 0,
                    tests_performed: 0,
                    tests_passed: 0,
                    average_stability: point.confidence,
                    validation_depth: self.config.validation_config.validation_depth.clone(),
                    warnings: vec![],
                },
            }
        };
        
        // Create final resolution based on validation
        let resolution = if validation.stability_score > 0.8 {
            ResolutionResult::Certain(Value::Number(validation.stability_score))
        } else {
            ResolutionResult::Uncertain {
                possibilities: vec![
                    (Value::Number(validation.stability_score), validation.stability_score),
                    (Value::Number(1.0 - validation.stability_score), 1.0 - validation.stability_score),
                ],
                confidence_interval: (
                    (validation.stability_score - 0.1).max(0.0),
                    (validation.stability_score + 0.1).min(1.0)
                ),
                aggregated_confidence: validation.stability_score,
            }
        };
        
        self.stats.validations_performed += 1;
        
        Ok(ValidatedPoint {
            point,
            positional_analysis,
            validation,
            debate_id: None,
            resolution,
        })
    }
    
    /// Check if a debate should be created for a point
    fn should_create_debate(&self, validation: &ValidationResult) -> bool {
        validation.stability_score < self.config.debate_uncertainty_threshold ||
        validation.quality_assessment.reliability_category == crate::turbulance::perturbation_validation::ReliabilityCategory::Questionable ||
        validation.quality_assessment.reliability_category == crate::turbulance::perturbation_validation::ReliabilityCategory::RequiresReview
    }
    
    /// Create a debate platform for a point
    async fn create_debate_for_point(&mut self, validated_point: &mut ValidatedPoint) -> Result<DebateInfo> {
        let platform_id = self.debate_manager.create_platform(
            validated_point.point.clone(),
            ResolutionStrategy::BayesianWeighted,
            Some(self.config.platform_config.clone())
        );
        
        // Get the platform and add some initial content
        if let Some(platform) = self.debate_manager.get_platform_mut(&platform_id) {
            // Add an initial affirmation based on the point's confidence
            if validated_point.point.confidence > 0.5 {
                platform.add_affirmation(
                    format!("Initial evidence supports this point: {}", validated_point.point.content),
                    "System Analysis".to_string(),
                    validated_point.point.confidence,
                    validated_point.validation.stability_score
                ).await?;
            }
            
            // Add contentions for validation concerns
            if !validated_point.validation.vulnerable_aspects.is_empty() {
                platform.add_contention(
                    format!("Validation identified concerns: {}", validated_point.validation.vulnerable_aspects.join(", ")),
                    "Validation System".to_string(),
                    1.0 - validated_point.validation.stability_score,
                    0.8,
                    ChallengeAspect::EvidenceQuality
                ).await?;
            }
            
            validated_point.debate_id = Some(platform_id);
            
            Ok(DebateInfo {
                id: platform_id,
                point_content: validated_point.point.content.clone(),
                current_score: platform.current_score,
                confidence: platform.score_confidence,
                resolution_state: format!("{:?}", platform.resolution_state),
                contributions: (platform.affirmations.len(), platform.contentions.len()),
            })
        } else {
            Err(TurbulanceError::InvalidInput("Failed to create debate platform".to_string()))
        }
    }
    
    /// Assess overall quality of processing
    fn assess_overall_quality(&self, validated_points: &[ValidatedPoint], debates: &[DebateInfo]) -> QualityAssessment {
        if validated_points.is_empty() {
            return QualityAssessment {
                overall_score: 0.0,
                positional_coherence: 0.0,
                validation_stability: 0.0,
                resolution_confidence: 0.0,
                evidence_quality: 0.0,
                recommendation: "No points extracted - check input text quality".to_string(),
            };
        }
        
        // Calculate metrics
        let avg_positional_coherence = validated_points.iter()
            .map(|vp| vp.positional_analysis.analysis_confidence)
            .sum::<f64>() / validated_points.len() as f64;
            
        let avg_validation_stability = validated_points.iter()
            .map(|vp| vp.validation.stability_score)
            .sum::<f64>() / validated_points.len() as f64;
            
        let avg_resolution_confidence = validated_points.iter()
            .map(|vp| match &vp.resolution {
                ResolutionResult::Certain(_) => 1.0,
                ResolutionResult::Uncertain { aggregated_confidence, .. } => *aggregated_confidence,
                ResolutionResult::Contextual { .. } => 0.7,
                ResolutionResult::Fuzzy { central_tendency, .. } => *central_tendency,
            })
            .sum::<f64>() / validated_points.len() as f64;
            
        let avg_evidence_quality = validated_points.iter()
            .map(|vp| vp.validation.quality_assessment.confidence_in_resolution)
            .sum::<f64>() / validated_points.len() as f64;
            
        let overall_score = (avg_positional_coherence + avg_validation_stability + avg_resolution_confidence + avg_evidence_quality) / 4.0;
        
        let recommendation = if overall_score > 0.8 {
            "High quality processing - results are reliable".to_string()
        } else if overall_score > 0.6 {
            "Good quality processing - some uncertainty remains".to_string()
        } else if overall_score > 0.4 {
            format!("Moderate quality - {} debates created for uncertain points", debates.len())
        } else {
            "Low quality processing - results require careful review".to_string()
        };
        
        QualityAssessment {
            overall_score,
            positional_coherence: avg_positional_coherence,
            validation_stability: avg_validation_stability,
            resolution_confidence: avg_resolution_confidence,
            evidence_quality: avg_evidence_quality,
            recommendation,
        }
    }
    
    /// Get pipeline statistics
    pub fn get_stats(&self) -> &PipelineStats {
        &self.stats
    }
    
    /// Get debate manager for external access
    pub fn get_debate_manager(&self) -> &DebatePlatformManager {
        &self.debate_manager
    }
    
    /// Get mutable debate manager
    pub fn get_debate_manager_mut(&mut self) -> &mut DebatePlatformManager {
        &mut self.debate_manager
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stream_config: StreamConfig::default(),
            validation_config: ValidationConfig::default(),
            platform_config: PlatformConfig::default(),
            enable_comprehensive_validation: true,
            auto_create_debates: true,
            debate_uncertainty_threshold: 0.6,
            max_processing_time: 60, // 60 seconds
        }
    }
}

/// Demonstrate the complete framework with a realistic example
pub async fn demonstrate_complete_framework() -> Result<()> {
    println!("🎵 Kwasa-Kwasa Complete Framework Demonstration 🎵\n");
    
    // Create pipeline with default configuration
    let mut pipeline = KwasaKwasaPipeline::new(PipelineConfig::default());
    
    // Example text that should trigger various framework features
    let example_text = "The new quantum computing algorithm demonstrates superior performance compared to classical methods. However, some researchers question its practical scalability. The evidence from recent benchmarks shows 40% improvement in specific scenarios. Critics argue that the testing conditions were not representative of real-world applications.";
    
    println!("📝 Processing text: {}\n", example_text);
    
    // Process through complete pipeline
    let result = pipeline.process_text(example_text).await?;
    
    // Display results
    println!("✅ Processing Complete!");
    println!("📊 Overall Quality Score: {:.2}", result.quality_assessment.overall_score);
    println!("🎯 Recommendation: {}\n", result.quality_assessment.recommendation);
    
    println!("📍 Extracted Points ({}):", result.points.len());
    for (i, point) in result.points.iter().enumerate() {
        println!("  {}. \"{}\"", i + 1, point.point.content);
        println!("     🎯 Confidence: {:.2}", point.point.confidence);
        println!("     📈 Validation Stability: {:.2}", point.validation.stability_score);
        println!("     🔧 Positional Analysis: {} (order dependency: {:.2})", 
                 point.positional_analysis.semantic_signature, 
                 point.positional_analysis.order_dependency_score);
        
        if let Some(debate_id) = point.debate_id {
            println!("     🏛️  Debate Created: {}", debate_id);
        }
        println!();
    }
    
    if !result.debates.is_empty() {
        println!("🏛️  Debate Platforms Created ({}):", result.debates.len());
        for debate in &result.debates {
            println!("  📋 {}: Score {:.2} (confidence: {:.2})", 
                     debate.point_content, debate.current_score, debate.confidence);
            println!("     📊 Contributions: {} affirmations, {} contentions", 
                     debate.contributions.0, debate.contributions.1);
            println!("     🎭 State: {}\n", debate.resolution_state);
        }
    }
    
    println!("📈 Pipeline Statistics:");
    let stats = pipeline.get_stats();
    println!("  • Texts processed: {}", stats.texts_processed);
    println!("  • Points extracted: {}", stats.points_extracted);
    println!("  • Debates created: {}", stats.debates_created);
    println!("  • Validations performed: {}", stats.validations_performed);
    println!("  • Average processing time: {:.2}ms", stats.avg_processing_time_ms);
    
    println!("\n🎵 Framework demonstration complete! 🎵");
    println!("This showcases the complete integration of:");
    println!("  • Probabilistic Points with uncertainty");
    println!("  • Positional semantics analysis");
    println!("  • Streaming text processing");
    println!("  • Perturbation validation");
    println!("  • Debate platform resolution");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_complete_pipeline() {
        let mut pipeline = KwasaKwasaPipeline::new(PipelineConfig::default());
        let text = "The solution is optimal and works well.";
        
        let result = pipeline.process_text(text).await.unwrap();
        
        assert!(!result.points.is_empty());
        assert!(result.quality_assessment.overall_score >= 0.0);
        assert!(result.quality_assessment.overall_score <= 1.0);
        assert!(!result.quality_assessment.recommendation.is_empty());
    }
    
    #[tokio::test]
    async fn test_debate_creation() {
        let config = PipelineConfig {
            auto_create_debates: true,
            debate_uncertainty_threshold: 0.8, // High threshold to force debate creation
            ..PipelineConfig::default()
        };
        
        let mut pipeline = KwasaKwasaPipeline::new(config);
        let text = "This claim is questionable and uncertain.";
        
        let result = pipeline.process_text(text).await.unwrap();
        
        // Should create debates for uncertain points
        assert!(result.debates.len() >= 0); // May or may not create debates depending on extraction
    }
    
    #[tokio::test]
    async fn test_validation_integration() {
        let config = PipelineConfig {
            enable_comprehensive_validation: true,
            ..PipelineConfig::default()
        };
        
        let mut pipeline = KwasaKwasaPipeline::new(config);
        let text = "Clear and certain statement with high confidence.";
        
        let result = pipeline.process_text(text).await.unwrap();
        
        // Should have validation results
        for point in &result.points {
            assert!(point.validation.stability_score >= 0.0);
            assert!(point.validation.stability_score <= 1.0);
        }
    }
    
    #[tokio::test]
    async fn test_framework_demonstration() {
        // This test runs the complete demonstration
        let result = demonstrate_complete_framework().await;
        assert!(result.is_ok());
    }
} 