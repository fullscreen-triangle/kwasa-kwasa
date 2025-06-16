//! Kwasa-Kwasa Image Processing Framework
//! 
//! This module implements the revolutionary image processing capabilities that mirror
//! the kwasa-kwasa philosophy: understanding through reconstruction and metacognitive orchestration.
//! 
//! ## Core Philosophy
//! 
//! Just as kwasa-kwasa proves text understanding through semantic reconstruction,
//! this framework proves image understanding through visual reconstruction.
//! 
//! "The best way to know if an AI has truly analyzed an image is if it can perfectly reconstruct it."
//! 
//! ## Architecture Overview
//! 
//! ```
//! Kwasa-Kwasa Image Framework
//! ├── Helicopter Engine (Understanding through Reconstruction)
//! │   ├── Autonomous Reconstruction
//! │   ├── Segment-Aware Processing  
//! │   ├── Zengeza Noise Detection
//! │   ├── Hatata MDP Validation
//! │   └── Nicotine Context Maintenance
//! ├── Pakati Engine (Regional Generation)
//! │   ├── Regional Prompting
//! │   ├── Reference Understanding
//! │   ├── Progressive Refinement
//! │   └── Template System
//! └── Metacognitive Orchestrator
//!     ├── Tres Commas Integration
//!     ├── Visual Propositions & Motions
//!     ├── Image Units & Boundaries
//!     └── Cross-Modal Reasoning
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::orchestrator::tres_commas::TresCommasEngine;

pub mod helicopter;
pub mod pakati;
pub mod units;
pub mod propositions;
pub mod reconstruction;
pub mod generation;
pub mod understanding;
pub mod analysis;
pub mod types;

/// Re-exports for convenient access
pub mod prelude {
    pub use super::{
        ImageFramework, ImageUnit, ImageProcessor, 
        HelicopterEngine, PakatiEngine, ImageOrchestrator,
        VisualProposition, VisualMotion, ReconstructionResult,
        GenerationResult, UnderstandingLevel, ImageAnalysis
    };
    pub use super::helicopter::*;
    pub use super::pakati::*;
    pub use super::units::*;
    pub use super::propositions::*;
    pub use super::reconstruction::*;
    pub use super::generation::*;
    pub use super::understanding::*;
    pub use super::analysis::*;
    pub use super::types::*;
}

/// Main image processing framework that integrates Helicopter and Pakati
#[derive(Debug, Clone)]
pub struct ImageFramework {
    /// Helicopter engine for understanding through reconstruction
    pub helicopter: HelicopterEngine,
    /// Pakati engine for regional generation
    pub pakati: PakatiEngine,
    /// Metacognitive orchestrator
    pub orchestrator: ImageOrchestrator,
    /// Configuration
    pub config: ImageFrameworkConfig,
}

/// Core image processing trait
pub trait ImageProcessor {
    /// Process an image with the kwasa-kwasa approach
    fn process(&mut self, input: &ImageInput) -> Result<ImageProcessingResult>;
    
    /// Understand an image through reconstruction
    fn understand(&mut self, image_path: &Path) -> Result<UnderstandingResult>;
    
    /// Generate new content with regional control
    fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResult>;
    
    /// Analyze cross-modal relationships between text and image
    fn cross_modal_analysis(&mut self, text: &str, image_path: &Path) -> Result<CrossModalResult>;
}

/// Image units - parallel to text units in kwasa-kwasa
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUnit {
    /// Unique identifier
    pub id: String,
    /// Type of image unit
    pub unit_type: ImageUnitType,
    /// Spatial boundaries
    pub boundaries: ImageBoundaries,
    /// Visual content description
    pub description: String,
    /// Confidence in understanding this unit
    pub confidence: f64,
    /// Reconstruction quality if available
    pub reconstruction_quality: Option<f64>,
    /// Semantic metadata
    pub metadata: HashMap<String, String>,
}

/// Types of image units
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImageUnitType {
    /// Complete image
    Image,
    /// Major visual region
    Region,
    /// Object or entity
    Object,
    /// Texture or pattern
    Texture,
    /// Edge or boundary
    Edge,
    /// Color region
    ColorRegion,
    /// Semantic segment
    SemanticSegment,
    /// Pixel cluster
    PixelCluster,
}

/// Spatial boundaries for image units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBoundaries {
    /// X coordinate
    pub x: u32,
    /// Y coordinate  
    pub y: u32,
    /// Width
    pub width: u32,
    /// Height
    pub height: u32,
    /// Optional polygon vertices for complex shapes
    pub polygon: Option<Vec<(u32, u32)>>,
    /// Confidence in boundary detection
    pub boundary_confidence: f64,
}

/// Visual propositions - parallel to text propositions in kwasa-kwasa
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualProposition {
    /// Proposition name
    pub name: String,
    /// Description of what this proposition tests
    pub description: String,
    /// Visual motions within this proposition
    pub motions: Vec<VisualMotion>,
    /// Overall support score
    pub support_score: f64,
    /// Evidence from image analysis
    pub evidence: Vec<VisualEvidence>,
}

/// Visual motions - specific testable claims about images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualMotion {
    /// Motion identifier
    pub id: String,
    /// Claim being tested
    pub claim: String,
    /// Confidence in this motion
    pub confidence: f64,
    /// Supporting evidence
    pub support_evidence: Vec<VisualEvidence>,
    /// Contradicting evidence
    pub contra_evidence: Vec<VisualEvidence>,
    /// Requirements for this motion to be true
    pub requirements: Vec<VisualRequirement>,
}

/// Evidence for visual claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualEvidence {
    /// Evidence type
    pub evidence_type: VisualEvidenceType,
    /// Strength of evidence (0.0 to 1.0)
    pub strength: f64,
    /// Description of evidence
    pub description: String,
    /// Source region in image
    pub source_region: Option<ImageBoundaries>,
    /// Reconstruction quality if applicable
    pub reconstruction_quality: Option<f64>,
}

/// Types of visual evidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisualEvidenceType {
    /// Successful reconstruction
    ReconstructionSuccess,
    /// Failed reconstruction
    ReconstructionFailure,
    /// Color analysis
    ColorAnalysis,
    /// Texture analysis
    TextureAnalysis,
    /// Edge detection
    EdgeDetection,
    /// Object recognition
    ObjectRecognition,
    /// Semantic segmentation
    SemanticSegmentation,
    /// Noise detection
    NoiseDetection,
    /// Statistical analysis
    StatisticalAnalysis,
}

/// Requirements for visual motions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualRequirement {
    /// Requirement description
    pub description: String,
    /// Metric to measure
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
}

/// Comparison operators for requirements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    EqualTo,
    GreaterOrEqual,
    LessOrEqual,
    NotEqual,
}

/// Main orchestrator for image processing
#[derive(Debug, Clone)]
pub struct ImageOrchestrator {
    /// Tres Commas engine integration
    pub tres_commas: Option<TresCommasEngine>,
    /// Current processing context
    pub context: ImageProcessingContext,
    /// Active propositions
    pub propositions: Vec<VisualProposition>,
    /// Processing goals
    pub goals: Vec<ImageProcessingGoal>,
    /// Performance metrics
    pub metrics: ImageProcessingMetrics,
}

/// Processing context for images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingContext {
    /// Domain context (medical, natural, artistic, etc.)
    pub domain: String,
    /// Processing goals
    pub objectives: Vec<String>,
    /// Quality thresholds
    pub quality_thresholds: HashMap<String, f64>,
    /// Time constraints
    pub time_budget: Option<u64>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

/// Resource constraints for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
    /// Maximum processing time in seconds
    pub max_time_seconds: u32,
    /// GPU availability
    pub gpu_available: bool,
    /// API rate limits
    pub api_rate_limits: HashMap<String, u32>,
}

/// Processing goals for images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingGoal {
    /// Goal identifier
    pub id: String,
    /// Goal description
    pub description: String,
    /// Success criteria
    pub success_criteria: Vec<SuccessCriterion>,
    /// Priority (0.0 to 1.0)
    pub priority: f64,
    /// Current progress
    pub progress: f64,
}

/// Success criteria for goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    /// Criterion description
    pub description: String,
    /// Metric to measure
    pub metric: String,
    /// Target value
    pub target_value: f64,
    /// Current value
    pub current_value: f64,
    /// Whether this criterion is met
    pub is_met: bool,
}

/// Performance metrics for image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingMetrics {
    /// Total processing time
    pub total_time_ms: u64,
    /// Reconstruction accuracy
    pub reconstruction_accuracy: f64,
    /// Understanding confidence
    pub understanding_confidence: f64,
    /// Generation quality
    pub generation_quality: f64,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// API calls made
    pub api_calls: u32,
    /// Success rate
    pub success_rate: f64,
}

/// Configuration for the image framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFrameworkConfig {
    /// Helicopter engine configuration
    pub helicopter_config: HelicopterConfig,
    /// Pakati engine configuration
    pub pakati_config: PakatiConfig,
    /// Orchestrator configuration
    pub orchestrator_config: OrchestratorConfig,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// API configurations
    pub api_configs: HashMap<String, ApiConfig>,
}

/// Configuration for Helicopter engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelicopterConfig {
    /// Patch size for reconstruction
    pub patch_size: u32,
    /// Context size around patches
    pub context_size: u32,
    /// Maximum reconstruction iterations
    pub max_iterations: u32,
    /// Target reconstruction quality
    pub target_quality: f64,
    /// Enable segment-aware processing
    pub segment_aware: bool,
    /// Enable noise detection
    pub noise_detection: bool,
    /// Enable context validation
    pub context_validation: bool,
}

/// Configuration for Pakati engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PakatiConfig {
    /// Default image dimensions
    pub default_width: u32,
    pub default_height: u32,
    /// Maximum refinement passes
    pub max_refinement_passes: u32,
    /// Target generation quality
    pub target_generation_quality: f64,
    /// Enable reference understanding
    pub reference_understanding: bool,
    /// Enable progressive refinement
    pub progressive_refinement: bool,
    /// Template storage path
    pub template_storage_path: Option<PathBuf>,
}

/// Configuration for orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Enable tres commas integration
    pub enable_tres_commas: bool,
    /// Maximum concurrent operations
    pub max_concurrent_operations: u32,
    /// Default processing strategy
    pub default_strategy: ProcessingStrategy,
    /// Enable cross-modal analysis
    pub enable_cross_modal: bool,
    /// Metacognitive validation interval
    pub validation_interval: u32,
}

/// Processing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingStrategy {
    /// Speed-optimized processing
    SpeedOptimized,
    /// Quality-optimized processing
    QualityOptimized,
    /// Balanced approach
    Balanced,
    /// Research-grade analysis
    ResearchGrade,
    /// Custom strategy with parameters
    Custom(HashMap<String, f64>),
}

/// Quality thresholds for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum reconstruction quality
    pub min_reconstruction_quality: f64,
    /// Minimum understanding confidence
    pub min_understanding_confidence: f64,
    /// Minimum generation quality
    pub min_generation_quality: f64,
    /// Minimum cross-modal alignment
    pub min_cross_modal_alignment: f64,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API key
    pub api_key: String,
    /// Base URL
    pub base_url: String,
    /// Rate limit per minute
    pub rate_limit: u32,
    /// Timeout in seconds
    pub timeout_seconds: u32,
    /// Model preferences
    pub preferred_models: Vec<String>,
}

/// Input for image processing
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// Image file path or data
    pub image_source: ImageSource,
    /// Processing instructions
    pub instructions: ProcessingInstructions,
    /// Context information
    pub context: Option<String>,
    /// Reference images if any
    pub references: Vec<ReferenceImage>,
}

/// Source of image data
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// File path
    FilePath(PathBuf),
    /// Raw image data
    RawData(Vec<u8>),
    /// URL to download
    Url(String),
    /// Base64 encoded data
    Base64(String),
}

/// Instructions for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInstructions {
    /// Type of processing requested
    pub operation_type: OperationType,
    /// Specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
    /// Time constraints
    pub time_constraints: Option<TimeConstraints>,
}

/// Types of operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperationType {
    /// Understand image through reconstruction
    Understand,
    /// Generate new content
    Generate,
    /// Analyze existing content
    Analyze,
    /// Cross-modal text-image analysis
    CrossModal,
    /// Regional editing
    RegionalEdit,
    /// Reference-guided refinement
    ReferenceRefinement,
}

/// Quality requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum overall quality
    pub min_quality: f64,
    /// Specific metric requirements
    pub metric_requirements: HashMap<String, f64>,
    /// Validation strategy
    pub validation_strategy: ValidationStrategy,
}

/// Validation strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// Fast validation
    Fast,
    /// Thorough validation
    Thorough,
    /// Cross-validation
    CrossValidation,
    /// Human-in-the-loop
    HumanInLoop,
}

/// Time constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConstraints {
    /// Maximum total time in seconds
    pub max_total_seconds: u32,
    /// Intermediate deadlines
    pub intermediate_deadlines: Vec<TimeDeadline>,
}

/// Time deadline for intermediate steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeDeadline {
    /// Step description
    pub step: String,
    /// Deadline in seconds from start
    pub deadline_seconds: u32,
}

/// Reference image for guidance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceImage {
    /// Image source
    pub source: String,
    /// Description of what to use from this reference
    pub description: String,
    /// Aspect to focus on
    pub aspect: ReferenceAspect,
    /// Weight in decision making
    pub weight: f64,
}

/// Aspects of reference images to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReferenceAspect {
    /// Color palette and distribution
    Color,
    /// Texture and surface details
    Texture,
    /// Composition and layout
    Composition,
    /// Lighting and mood
    Lighting,
    /// Overall style
    Style,
    /// Specific objects or elements
    Objects,
    /// General similarity
    General,
}

/// Result of image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingResult {
    /// Success status
    pub success: bool,
    /// Result data
    pub result: ProcessingResultData,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Performance metrics
    pub performance_metrics: ImageProcessingMetrics,
    /// Insights and recommendations
    pub insights: Vec<ProcessingInsight>,
}

/// Different types of processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingResultData {
    /// Understanding result
    Understanding(UnderstandingResult),
    /// Generation result
    Generation(GenerationResult),
    /// Analysis result
    Analysis(AnalysisResult),
    /// Cross-modal result
    CrossModal(CrossModalResult),
}

/// Result of understanding through reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingResult {
    /// Overall understanding level
    pub understanding_level: UnderstandingLevel,
    /// Reconstruction quality
    pub reconstruction_quality: f64,
    /// Confidence in understanding
    pub confidence: f64,
    /// Identified visual elements
    pub visual_elements: Vec<VisualElement>,
    /// Failed reconstruction areas
    pub failure_regions: Vec<ImageBoundaries>,
    /// Insights about the image
    pub insights: Vec<UnderstandingInsight>,
}

/// Levels of understanding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnderstandingLevel {
    /// No understanding achieved
    None,
    /// Basic structure understood
    Basic,
    /// Moderate understanding
    Moderate,
    /// Good understanding
    Good,
    /// Excellent understanding
    Excellent,
    /// Perfect understanding
    Perfect,
}

/// Visual elements identified in images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualElement {
    /// Element type
    pub element_type: VisualElementType,
    /// Boundaries in image
    pub boundaries: ImageBoundaries,
    /// Confidence in detection
    pub confidence: f64,
    /// Properties of this element
    pub properties: HashMap<String, f64>,
    /// Description
    pub description: String,
}

/// Types of visual elements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisualElementType {
    /// Object or entity
    Object,
    /// Person or human figure
    Person,
    /// Animal
    Animal,
    /// Vegetation
    Vegetation,
    /// Architecture
    Architecture,
    /// Vehicle
    Vehicle,
    /// Text or writing
    Text,
    /// Pattern or texture
    Pattern,
    /// Background
    Background,
    /// Unknown element
    Unknown,
}

/// Insight from understanding process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingInsight {
    /// Insight type
    pub insight_type: InsightType,
    /// Confidence in insight
    pub confidence: f64,
    /// Description
    pub description: String,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Types of insights
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsightType {
    /// Quality assessment
    Quality,
    /// Content analysis
    Content,
    /// Technical observation
    Technical,
    /// Artistic evaluation
    Artistic,
    /// Recommendation
    Recommendation,
    /// Warning or concern
    Warning,
}

/// Result of generation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated image path
    pub image_path: Option<PathBuf>,
    /// Generation quality
    pub quality: f64,
    /// Alignment with request
    pub alignment_score: f64,
    /// Number of refinement passes
    pub refinement_passes: u32,
    /// Generation parameters used
    pub parameters_used: HashMap<String, serde_json::Value>,
    /// Template information if saved
    pub template_info: Option<TemplateInfo>,
}

/// Template information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateInfo {
    /// Template name
    pub name: String,
    /// Template path
    pub path: PathBuf,
    /// Reusable configuration
    pub config: serde_json::Value,
}

/// Result of analysis operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Analysis type performed
    pub analysis_type: AnalysisType,
    /// Findings from analysis
    pub findings: Vec<AnalysisFinding>,
    /// Statistical measures
    pub statistics: HashMap<String, f64>,
    /// Visualizations generated
    pub visualizations: Vec<String>,
}

/// Types of analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnalysisType {
    /// Color analysis
    Color,
    /// Texture analysis
    Texture,
    /// Composition analysis
    Composition,
    /// Object detection
    ObjectDetection,
    /// Semantic segmentation
    SemanticSegmentation,
    /// Quality assessment
    Quality,
    /// Noise analysis
    Noise,
}

/// Finding from analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisFinding {
    /// Finding description
    pub description: String,
    /// Confidence in finding
    pub confidence: f64,
    /// Quantitative value if applicable
    pub value: Option<f64>,
    /// Region where finding applies
    pub region: Option<ImageBoundaries>,
}

/// Result of cross-modal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalResult {
    /// Alignment between text and image
    pub alignment_score: f64,
    /// Text elements that match image
    pub matching_elements: Vec<CrossModalMatch>,
    /// Discrepancies found
    pub discrepancies: Vec<CrossModalDiscrepancy>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

/// Match between text and image elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalMatch {
    /// Text portion
    pub text_element: String,
    /// Corresponding visual element
    pub visual_element: VisualElement,
    /// Strength of match
    pub match_strength: f64,
}

/// Discrepancy between text and image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalDiscrepancy {
    /// Type of discrepancy
    pub discrepancy_type: DiscrepancyType,
    /// Description
    pub description: String,
    /// Severity
    pub severity: f64,
    /// Suggested resolution
    pub suggested_resolution: String,
}

/// Types of cross-modal discrepancies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscrepancyType {
    /// Text describes something not in image
    MissingVisual,
    /// Image shows something not in text
    ExtraVisual,
    /// Contradictory information
    Contradiction,
    /// Different emphasis or focus
    EmphasisMismatch,
    /// Style inconsistency
    StyleMismatch,
}

/// Insight from processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInsight {
    /// Insight category
    pub category: InsightCategory,
    /// Insight description
    pub description: String,
    /// Confidence in insight
    pub confidence: f64,
    /// Actionable recommendations
    pub recommendations: Vec<String>,
}

/// Categories of processing insights
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InsightCategory {
    /// Performance optimization
    Performance,
    /// Quality improvement
    Quality,
    /// Technical recommendation
    Technical,
    /// Creative suggestion
    Creative,
    /// Resource optimization
    Resource,
    /// Process improvement
    Process,
}

impl ImageFramework {
    /// Create a new image framework instance
    pub fn new(config: ImageFrameworkConfig) -> Result<Self> {
        let helicopter = HelicopterEngine::new(config.helicopter_config.clone())?;
        let pakati = PakatiEngine::new(config.pakati_config.clone())?;
        let orchestrator = ImageOrchestrator::new(config.orchestrator_config.clone())?;

        Ok(Self {
            helicopter,
            pakati,
            orchestrator,
            config,
        })
    }

    /// Process an image using the integrated approach
    pub fn process(&mut self, input: &ImageInput) -> Result<ImageProcessingResult> {
        // Use orchestrator to coordinate processing
        self.orchestrator.coordinate_processing(input, &mut self.helicopter, &mut self.pakati)
    }

    /// Understand an image through reconstruction (Helicopter approach)
    pub fn understand(&mut self, image_path: &Path) -> Result<UnderstandingResult> {
        self.helicopter.understand_through_reconstruction(image_path)
    }

    /// Generate content with regional control (Pakati approach)
    pub fn generate(&mut self, request: &GenerationRequest) -> Result<GenerationResult> {
        self.pakati.generate_with_regions(request)
    }

    /// Perform cross-modal analysis between text and image
    pub fn cross_modal_analysis(&mut self, text: &str, image_path: &Path) -> Result<CrossModalResult> {
        self.orchestrator.cross_modal_analysis(text, image_path, &mut self.helicopter)
    }
}

impl Default for ImageFrameworkConfig {
    fn default() -> Self {
        Self {
            helicopter_config: HelicopterConfig::default(),
            pakati_config: PakatiConfig::default(),
            orchestrator_config: OrchestratorConfig::default(),
            quality_thresholds: QualityThresholds::default(),
            api_configs: HashMap::new(),
        }
    }
}

impl Default for HelicopterConfig {
    fn default() -> Self {
        Self {
            patch_size: 32,
            context_size: 96,
            max_iterations: 50,
            target_quality: 0.85,
            segment_aware: true,
            noise_detection: true,
            context_validation: true,
        }
    }
}

impl Default for PakatiConfig {
    fn default() -> Self {
        Self {
            default_width: 1024,
            default_height: 1024,
            max_refinement_passes: 8,
            target_generation_quality: 0.9,
            reference_understanding: true,
            progressive_refinement: true,
            template_storage_path: None,
        }
    }
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            enable_tres_commas: true,
            max_concurrent_operations: 4,
            default_strategy: ProcessingStrategy::Balanced,
            enable_cross_modal: true,
            validation_interval: 10,
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_reconstruction_quality: 0.8,
            min_understanding_confidence: 0.7,
            min_generation_quality: 0.85,
            min_cross_modal_alignment: 0.75,
        }
    }
}

/// Placeholder for GenerationRequest - will be defined in generation module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Request details will be defined in generation module
    pub placeholder: String,
} 