use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod moriarty;    // Biomechanical analysis
pub mod vibrio;      // Computer vision pipeline  
pub mod space_computer; // 3D visualization and AI analysis
pub mod morphine;    // Streaming and real-time processing
pub mod integration; // Cross-module integration
pub mod pipeline;    // Processing pipeline orchestration

/// The main Spectacular video processing system
pub struct SpectacularSystem {
    pub config: SpectacularConfig,
    pub moriarty: Arc<RwLock<moriarty::MoriartyEngine>>,
    pub vibrio: Arc<RwLock<vibrio::VibrioProcessor>>,
    pub space_computer: Arc<RwLock<space_computer::SpaceComputerPlatform>>,
    pub morphine: Arc<RwLock<morphine::MorphineStreamer>>,
    pub pipeline: pipeline::ProcessingPipeline,
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectacularConfig {
    pub processing_mode: ProcessingMode,
    pub video_formats: Vec<String>,
    pub output_formats: Vec<String>,
    pub quality_settings: QualitySettings,
    pub performance_settings: PerformanceSettings,
    pub ai_models: AIModelConfig,
    pub streaming_config: StreamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingMode {
    RealTime,
    Batch,
    Interactive,
    Streaming,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    pub input_resolution: (u32, u32),
    pub output_resolution: (u32, u32),
    pub frame_rate: f64,
    pub compression_quality: f64,
    pub pose_detection_accuracy: f64,
    pub biomechanical_precision: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    pub max_concurrent_videos: u32,
    pub gpu_acceleration: bool,
    pub cpu_thread_count: u32,
    pub memory_limit_mb: u64,
    pub cache_size_mb: u64,
    pub parallel_processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIModelConfig {
    pub pose_detection_model: String,
    pub object_tracking_model: String,
    pub biomechanical_analysis_model: String,
    pub motion_prediction_model: String,
    pub annotation_ai_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub enable_streaming: bool,
    pub stream_protocols: Vec<String>,
    pub buffer_size_seconds: f64,
    pub adaptive_bitrate: bool,
    pub low_latency_mode: bool,
}

/// Video processing task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoTask {
    pub id: Uuid,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub task_type: VideoTaskType,
    pub processing_config: ProcessingConfig,
    pub metadata: VideoMetadata,
    pub status: TaskStatus,
    pub progress: f64,
    pub results: Option<ProcessingResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoTaskType {
    PoseAnalysis,
    BiomechanicalAnalysis,
    MotionTracking,
    PerformanceAnalysis,
    AnnotationGeneration,
    RealTimeStreaming,
    InteractiveAnalysis,
    CrossModalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub enable_pose_detection: bool,
    pub enable_biomechanical_analysis: bool,
    pub enable_motion_tracking: bool,
    pub enable_annotation_ai: bool,
    pub enable_3d_visualization: bool,
    pub enable_streaming: bool,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub duration_seconds: f64,
    pub frame_count: u64,
    pub frame_rate: f64,
    pub resolution: (u32, u32),
    pub file_size_bytes: u64,
    pub codec: String,
    pub creation_time: std::time::SystemTime,
    pub subject_info: Option<SubjectInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectInfo {
    pub subject_id: String,
    pub sport: String,
    pub skill_level: String,
    pub anthropometric_data: Option<AnthropometricData>,
    pub performance_goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropometricData {
    pub height_cm: f64,
    pub weight_kg: f64,
    pub limb_lengths: HashMap<String, f64>,
    pub body_fat_percent: Option<f64>,
    pub age: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Processing,
    Completed,
    Failed,
    Cancelled,
    Paused,
}

/// Results from video processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResults {
    pub pose_data: Option<PoseAnalysisResults>,
    pub biomechanical_data: Option<BiomechanicalResults>,
    pub motion_data: Option<MotionTrackingResults>,
    pub annotation_data: Option<AnnotationResults>,
    pub visualization_data: Option<VisualizationResults>,
    pub performance_metrics: ProcessingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseAnalysisResults {
    pub pose_sequences: Vec<PoseFrame>,
    pub confidence_scores: Vec<f64>,
    pub skeleton_data: SkeletonData,
    pub key_poses: Vec<KeyPose>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseFrame {
    pub frame_number: u64,
    pub timestamp_ms: u64,
    pub joints: HashMap<String, Joint3D>,
    pub confidence: f64,
    pub bounding_box: BoundingBox,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub confidence: f64,
    pub visibility: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonData {
    pub joint_connections: Vec<(String, String)>,
    pub bone_lengths: HashMap<String, f64>,
    pub joint_angles: HashMap<String, f64>,
    pub symmetry_analysis: SymmetryAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryAnalysis {
    pub left_right_symmetry: f64,
    pub temporal_symmetry: f64,
    pub asymmetry_points: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPose {
    pub frame_number: u64,
    pub pose_type: String,
    pub significance_score: f64,
    pub description: String,
    pub biomechanical_insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomechanicalResults {
    pub joint_angles: Vec<JointAngleSequence>,
    pub velocities: Vec<VelocityData>,
    pub forces: Vec<ForceData>,
    pub energy_analysis: EnergyAnalysis,
    pub efficiency_metrics: EfficiencyMetrics,
    pub injury_risk_assessment: InjuryRiskAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointAngleSequence {
    pub joint_name: String,
    pub angles: Vec<f64>,
    pub timestamps: Vec<u64>,
    pub range_of_motion: f64,
    pub peak_angles: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityData {
    pub joint_name: String,
    pub linear_velocity: Vec<(f64, f64, f64)>,
    pub angular_velocity: Vec<f64>,
    pub peak_velocity: f64,
    pub acceleration: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceData {
    pub force_type: String,
    pub magnitude: Vec<f64>,
    pub direction: Vec<(f64, f64, f64)>,
    pub application_points: Vec<String>,
    pub ground_reaction_forces: Option<Vec<(f64, f64, f64)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyAnalysis {
    pub kinetic_energy: Vec<f64>,
    pub potential_energy: Vec<f64>,
    pub total_energy: Vec<f64>,
    pub energy_transfer_efficiency: f64,
    pub power_output: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub movement_efficiency: f64,
    pub energy_efficiency: f64,
    pub technique_score: f64,
    pub optimization_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjuryRiskAssessment {
    pub overall_risk_score: f64,
    pub risk_factors: Vec<RiskFactor>,
    pub high_risk_movements: Vec<String>,
    pub prevention_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub factor_name: String,
    pub risk_level: f64,
    pub description: String,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionTrackingResults {
    pub tracked_objects: Vec<TrackedObject>,
    pub motion_patterns: Vec<MotionPattern>,
    pub optical_flow_data: OpticalFlowData,
    pub speed_analysis: SpeedAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedObject {
    pub object_id: u32,
    pub object_type: String,
    pub trajectory: Vec<(f64, f64, u64)>, // x, y, timestamp
    pub confidence_sequence: Vec<f64>,
    pub bounding_boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPattern {
    pub pattern_name: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase_relationships: HashMap<String, f64>,
    pub regularity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticalFlowData {
    pub flow_vectors: Vec<Vec<(f64, f64)>>, // Per frame
    pub motion_magnitude: Vec<f64>,
    pub motion_direction: Vec<f64>,
    pub flow_quality: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedAnalysis {
    pub instantaneous_speeds: Vec<f64>,
    pub average_speed: f64,
    pub peak_speed: f64,
    pub acceleration_phases: Vec<AccelerationPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationPhase {
    pub start_frame: u64,
    pub end_frame: u64,
    pub acceleration: f64,
    pub phase_type: String, // "acceleration", "deceleration", "constant"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationResults {
    pub expert_annotations: Vec<ExpertAnnotation>,
    pub ai_generated_insights: Vec<AIInsight>,
    pub annotation_models: Vec<AnnotationModel>,
    pub cross_video_patterns: Vec<CrossVideoPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertAnnotation {
    pub expert_id: String,
    pub timestamp_ms: u64,
    pub annotation_type: String,
    pub content: String,
    pub confidence: f64,
    pub validation_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f64,
    pub supporting_data: serde_json::Value,
    pub actionable_recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationModel {
    pub model_id: String,
    pub model_type: String,
    pub accuracy_score: f64,
    pub application_domain: String,
    pub usage_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossVideoPattern {
    pub pattern_id: String,
    pub pattern_description: String,
    pub occurrence_frequency: f64,
    pub affected_videos: Vec<String>,
    pub insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationResults {
    pub rendered_frames: Vec<RenderedFrame>,
    pub interactive_models: Vec<InteractiveModel>,
    pub annotation_overlays: Vec<AnnotationOverlay>,
    pub comparative_visualizations: Vec<ComparativeVisualization>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderedFrame {
    pub frame_number: u64,
    pub render_type: String,
    pub output_path: PathBuf,
    pub render_time_ms: u64,
    pub quality_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveModel {
    pub model_id: String,
    pub model_type: String,
    pub interaction_capabilities: Vec<String>,
    pub web_url: Option<String>,
    pub embedding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationOverlay {
    pub overlay_type: String,
    pub timestamp_ms: u64,
    pub overlay_data: serde_json::Value,
    pub visibility_settings: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeVisualization {
    pub comparison_type: String,
    pub compared_videos: Vec<String>,
    pub visualization_path: PathBuf,
    pub insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    pub total_processing_time_ms: u64,
    pub frames_processed_per_second: f64,
    pub memory_usage_peak_mb: u64,
    pub gpu_utilization_percent: Option<f64>,
    pub quality_scores: HashMap<String, f64>,
    pub error_count: u32,
    pub warnings: Vec<String>,
}

impl SpectacularSystem {
    /// Create a new Spectacular video processing system
    pub async fn new(config: SpectacularConfig) -> Result<Self> {
        let session_id = Uuid::new_v4();
        
        // Initialize all sub-systems
        let moriarty = Arc::new(RwLock::new(
            moriarty::MoriartyEngine::new(config.clone()).await?
        ));
        
        let vibrio = Arc::new(RwLock::new(
            vibrio::VibrioProcessor::new(config.clone()).await?
        ));
        
        let space_computer = Arc::new(RwLock::new(
            space_computer::SpaceComputerPlatform::new(config.clone()).await?
        ));
        
        let morphine = Arc::new(RwLock::new(
            morphine::MorphineStreamer::new(config.clone()).await?
        ));
        
        let pipeline = pipeline::ProcessingPipeline::new(
            moriarty.clone(),
            vibrio.clone(), 
            space_computer.clone(),
            morphine.clone(),
        ).await?;
        
        Ok(Self {
            config,
            moriarty,
            vibrio,
            space_computer,
            morphine,
            pipeline,
            session_id,
        })
    }

    /// Process a video with full Spectacular analysis
    pub async fn process_video(&mut self, task: VideoTask) -> Result<ProcessingResults> {
        tracing::info!("Starting Spectacular video processing for task: {}", task.id);
        
        // Route task through the processing pipeline
        let results = self.pipeline.process_video(task).await?;
        
        tracing::info!("Spectacular video processing completed successfully");
        Ok(results)
    }

    /// Process video in real-time streaming mode
    pub async fn process_stream(&mut self, stream_config: StreamProcessingConfig) -> Result<()> {
        let morphine = self.morphine.read().await;
        morphine.start_stream_processing(stream_config).await
    }

    /// Generate interactive 3D visualization
    pub async fn generate_interactive_visualization(&self, video_path: &Path) -> Result<InteractiveModel> {
        let space_computer = self.space_computer.read().await;
        space_computer.generate_interactive_model(video_path).await
    }

    /// Apply expert annotation model to video
    pub async fn apply_annotation_model(&self, video_path: &Path, model_id: &str) -> Result<AnnotationResults> {
        let space_computer = self.space_computer.read().await;
        space_computer.apply_annotation_model(video_path, model_id).await
    }

    /// Get system health and performance metrics
    pub async fn get_system_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            moriarty_status: self.moriarty.read().await.get_health_status().await,
            vibrio_status: self.vibrio.read().await.get_health_status().await,
            space_computer_status: self.space_computer.read().await.get_health_status().await,
            morphine_status: self.morphine.read().await.get_health_status().await,
            overall_performance: self.calculate_overall_performance().await,
            resource_usage: self.get_resource_usage().await,
        }
    }

    async fn calculate_overall_performance(&self) -> f64 {
        // Calculate overall system performance score
        0.85 // Placeholder
    }

    async fn get_resource_usage(&self) -> ResourceUsage {
        ResourceUsage {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            gpu_usage_percent: None,
            disk_io_mb_per_sec: 0.0,
            network_io_mb_per_sec: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingConfig {
    pub input_stream_url: String,
    pub output_stream_url: Option<String>,
    pub processing_options: ProcessingConfig,
    pub latency_target_ms: u64,
    pub quality_target: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub moriarty_status: ComponentHealth,
    pub vibrio_status: ComponentHealth,
    pub space_computer_status: ComponentHealth,
    pub morphine_status: ComponentHealth,
    pub overall_performance: f64,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub uptime_seconds: u64,
    pub error_count: u32,
    pub performance_score: f64,
    pub last_health_check: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub gpu_usage_percent: Option<f64>,
    pub disk_io_mb_per_sec: f64,
    pub network_io_mb_per_sec: f64,
} 