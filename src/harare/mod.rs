use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use anyhow::{Result, Context};
use uuid::Uuid;
use tracing::{info, warn, error, debug, trace};

pub mod orchestrator;
pub mod logging;
pub mod decision_engine;
pub mod context_manager;
pub mod execution_planner;
pub mod hre_parser;
pub mod metacognitive_loop;

/// The main Harare orchestrator system
pub struct HarareOrchestrator {
    pub id: Uuid,
    pub config: OrchestratorConfig,
    pub context: Arc<RwLock<GlobalContext>>,
    pub decision_engine: decision_engine::DecisionEngine,
    pub execution_planner: execution_planner::ExecutionPlanner,
    pub logger: logging::HarareLogger,
    pub event_stream: mpsc::UnboundedSender<OrchestratorEvent>,
    pub module_registry: Arc<RwLock<ModuleRegistry>>,
    pub metacognitive_state: Arc<RwLock<MetacognitiveState>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub project_root: PathBuf,
    pub log_level: LogLevel,
    pub auto_recovery: bool,
    pub parallel_execution: bool,
    pub max_retry_attempts: u32,
    pub decision_timeout_ms: u64,
    pub metacognitive_interval_ms: u64,
    pub hre_output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Global execution context maintained by Harare
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalContext {
    pub current_task: Option<Task>,
    pub active_modules: Vec<ModuleInfo>,
    pub execution_history: VecDeque<ExecutionRecord>,
    pub environment_state: HashMap<String, String>,
    pub resource_usage: ResourceMetrics,
    pub user_intent: Option<UserIntent>,
    pub semantic_workspace: SemanticWorkspace,
}

/// Task representation in Harare
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub name: String,
    pub task_type: TaskType,
    pub priority: Priority,
    pub dependencies: Vec<Uuid>,
    pub target_modules: Vec<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub deadline: Option<SystemTime>,
    pub created_at: SystemTime,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Analysis,
    Processing,
    Generation,
    Orchestration,
    Monitoring,
    Recovery,
    Learning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Planning,
    Executing,
    Completed,
    Failed,
    Cancelled,
    Retrying,
}

/// Module registry for tracking all system modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleRegistry {
    pub modules: HashMap<String, ModuleInfo>,
    pub dependencies: HashMap<String, Vec<String>>,
    pub health_status: HashMap<String, ModuleHealth>,
    pub performance_metrics: HashMap<String, PerformanceMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub module_type: ModuleType,
    pub version: String,
    pub capabilities: Vec<Capability>,
    pub resource_requirements: ResourceRequirements,
    pub interface_endpoints: Vec<InterfaceEndpoint>,
    pub last_health_check: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleType {
    Fullscreen,
    Spectacular,
    Nebuchadnezzar,
    Trebuchet,
    Sighthound,
    Zangalewa,
    TurbulanceCore,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
    pub complexity_score: f64,
    pub latency_estimate_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub disk_mb: u64,
    pub gpu_memory_mb: Option<u64>,
    pub network_bandwidth_mbps: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceEndpoint {
    pub name: String,
    pub endpoint_type: EndpointType,
    pub address: String,
    pub protocol: String,
    pub authentication_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EndpointType {
    Rest,
    Grpc,
    WebSocket,
    MessageQueue,
    DirectCall,
}

/// Health status of modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleHealth {
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub error_count: u32,
    pub response_time_ms: u64,
    pub uptime_percent: f64,
    pub issues: Vec<HealthIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    pub severity: IssueSeverity,
    pub description: String,
    pub first_observed: SystemTime,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Performance metrics for modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_ops_per_sec: f64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub error_rate_percent: f64,
    pub resource_utilization: HashMap<String, f64>,
    pub last_updated: SystemTime,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub disk_usage_mb: u64,
    pub network_in_mbps: f64,
    pub network_out_mbps: f64,
    pub gpu_usage_percent: Option<f64>,
    pub timestamp: SystemTime,
}

/// User intent understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserIntent {
    pub intent_type: IntentType,
    pub confidence: f64,
    pub parameters: HashMap<String, serde_json::Value>,
    pub context_requirements: Vec<String>,
    pub expected_outputs: Vec<String>,
    pub timeline: Option<Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentType {
    Analysis,
    Generation,
    Transformation,
    Query,
    Configuration,
    Debugging,
    Learning,
    Exploration,
}

/// Semantic workspace for maintaining semantic state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticWorkspace {
    pub active_items: HashMap<String, SemanticItem>,
    pub relationships: Vec<SemanticRelationship>,
    pub context_stack: VecDeque<SemanticContext>,
    pub attention_focus: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticItem {
    pub id: String,
    pub item_type: String,
    pub content: serde_json::Value,
    pub metadata: HashMap<String, String>,
    pub semantic_weight: f64,
    pub last_accessed: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelationship {
    pub from_item: String,
    pub to_item: String,
    pub relationship_type: String,
    pub strength: f64,
    pub bidirectional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    pub name: String,
    pub active_items: Vec<String>,
    pub goals: Vec<String>,
    pub constraints: Vec<String>,
    pub established_at: SystemTime,
}

/// Execution record for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub task_id: Uuid,
    pub module_sequence: Vec<String>,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: TaskStatus,
    pub resource_usage: ResourceMetrics,
    pub outputs: HashMap<String, serde_json::Value>,
    pub errors: Vec<ExecutionError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub module_name: String,
    pub error_type: String,
    pub message: String,
    pub timestamp: SystemTime,
    pub recoverable: bool,
}

/// Metacognitive state for self-awareness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveState {
    pub learning_objectives: Vec<LearningObjective>,
    pub performance_patterns: Vec<PerformancePattern>,
    pub adaptation_history: Vec<AdaptationRecord>,
    pub self_assessment: SelfAssessment,
    pub knowledge_gaps: Vec<KnowledgeGap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningObjective {
    pub objective: String,
    pub target_improvement: f64,
    pub measurement_metric: String,
    pub deadline: Option<SystemTime>,
    pub progress: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePattern {
    pub pattern_name: String,
    pub conditions: Vec<String>,
    pub observed_behavior: String,
    pub frequency: u32,
    pub impact_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationRecord {
    pub adaptation_type: String,
    pub trigger_conditions: Vec<String>,
    pub changes_made: Vec<String>,
    pub outcome_improvement: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAssessment {
    pub overall_performance: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub confidence_level: f64,
    pub last_assessment: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub domain: String,
    pub gap_description: String,
    pub impact_severity: f64,
    pub learning_priority: Priority,
    pub suggested_actions: Vec<String>,
}

/// Events emitted by the orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrchestratorEvent {
    TaskCreated(Task),
    TaskStarted(Uuid),
    TaskCompleted(Uuid, TaskResult),
    TaskFailed(Uuid, String),
    ModuleHealthChanged(String, HealthStatus),
    ResourceThresholdExceeded(String, f64),
    DecisionMade(Decision),
    AdaptationTriggered(String),
    UserIntentDetected(UserIntent),
    SystemAlert(AlertLevel, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub outputs: HashMap<String, serde_json::Value>,
    pub execution_time_ms: u64,
    pub resources_used: ResourceMetrics,
    pub quality_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub decision_id: Uuid,
    pub decision_type: String,
    pub rationale: String,
    pub confidence: f64,
    pub alternatives_considered: Vec<String>,
    pub expected_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

impl HarareOrchestrator {
    /// Create a new Harare orchestrator
    pub async fn new(config: OrchestratorConfig) -> Result<Self> {
        let (event_sender, _event_receiver) = mpsc::unbounded_channel();
        
        let context = Arc::new(RwLock::new(GlobalContext {
            current_task: None,
            active_modules: Vec::new(),
            execution_history: VecDeque::new(),
            environment_state: HashMap::new(),
            resource_usage: ResourceMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0,
                disk_usage_mb: 0,
                network_in_mbps: 0.0,
                network_out_mbps: 0.0,
                gpu_usage_percent: None,
                timestamp: SystemTime::now(),
            },
            user_intent: None,
            semantic_workspace: SemanticWorkspace {
                active_items: HashMap::new(),
                relationships: Vec::new(),
                context_stack: VecDeque::new(),
                attention_focus: None,
            },
        }));
        
        let logger = logging::HarareLogger::new(&config.hre_output_path).await?;
        let decision_engine = decision_engine::DecisionEngine::new();
        let execution_planner = execution_planner::ExecutionPlanner::new();
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            context,
            decision_engine,
            execution_planner,
            logger,
            event_stream: event_sender,
            module_registry: Arc::new(RwLock::new(ModuleRegistry {
                modules: HashMap::new(),
                dependencies: HashMap::new(),
                health_status: HashMap::new(),
                performance_metrics: HashMap::new(),
            })),
            metacognitive_state: Arc::new(RwLock::new(MetacognitiveState {
                learning_objectives: Vec::new(),
                performance_patterns: Vec::new(),
                adaptation_history: Vec::new(),
                self_assessment: SelfAssessment {
                    overall_performance: 0.5,
                    strengths: Vec::new(),
                    weaknesses: Vec::new(),
                    confidence_level: 0.5,
                    last_assessment: SystemTime::now(),
                },
                knowledge_gaps: Vec::new(),
            })),
        })
    }

    /// Start the orchestrator's main execution loop
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting Harare orchestrator with ID: {}", self.id);
        
        // Initialize all modules
        self.initialize_modules().await?;
        
        // Start health monitoring
        self.start_health_monitoring().await;
        
        // Start metacognitive loop
        self.start_metacognitive_loop().await;
        
        // Start resource monitoring
        self.start_resource_monitoring().await;
        
        info!("Harare orchestrator started successfully");
        Ok(())
    }

    /// Initialize and register all system modules
    async fn initialize_modules(&mut self) -> Result<()> {
        let modules_to_register = vec![
            ("fullscreen", ModuleType::Fullscreen),
            ("spectacular", ModuleType::Spectacular),
            ("nebuchadnezzar", ModuleType::Nebuchadnezzar),
            ("trebuchet", ModuleType::Trebuchet),
            ("sighthound", ModuleType::Sighthound),
            ("zangalewa", ModuleType::Zangalewa),
            ("turbulance_core", ModuleType::TurbulanceCore),
        ];
        
        let mut registry = self.module_registry.write().await;
        
        for (name, module_type) in modules_to_register {
            let module_info = ModuleInfo {
                name: name.to_string(),
                module_type,
                version: "0.1.0".to_string(),
                capabilities: self.get_module_capabilities(name).await,
                resource_requirements: self.get_module_resource_requirements(name).await,
                interface_endpoints: self.get_module_endpoints(name).await,
                last_health_check: SystemTime::now(),
            };
            
            registry.modules.insert(name.to_string(), module_info);
            registry.health_status.insert(name.to_string(), ModuleHealth {
                status: HealthStatus::Unknown,
                last_check: SystemTime::now(),
                error_count: 0,
                response_time_ms: 0,
                uptime_percent: 100.0,
                issues: Vec::new(),
            });
        }
        
        Ok(())
    }

    /// Get capabilities for a specific module
    async fn get_module_capabilities(&self, module_name: &str) -> Vec<Capability> {
        match module_name {
            "fullscreen" => vec![
                Capability {
                    name: "network_graph_generation".to_string(),
                    input_types: vec!["module_list".to_string()],
                    output_types: vec!["svg_graph".to_string(), "interactive_html".to_string()],
                    complexity_score: 0.6,
                    latency_estimate_ms: 500,
                }
            ],
            "spectacular" => vec![
                Capability {
                    name: "video_analysis".to_string(),
                    input_types: vec!["video_file".to_string()],
                    output_types: vec!["biomechanical_data".to_string(), "pose_analysis".to_string()],
                    complexity_score: 0.9,
                    latency_estimate_ms: 5000,
                }
            ],
            // Add more module capabilities...
            _ => Vec::new(),
        }
    }

    /// Get resource requirements for a specific module
    async fn get_module_resource_requirements(&self, module_name: &str) -> ResourceRequirements {
        match module_name {
            "spectacular" => ResourceRequirements {
                cpu_cores: 4,
                memory_mb: 2048,
                disk_mb: 1024,
                gpu_memory_mb: Some(1024),
                network_bandwidth_mbps: Some(100),
            },
            "trebuchet" => ResourceRequirements {
                cpu_cores: 2,
                memory_mb: 1024,
                disk_mb: 512,
                gpu_memory_mb: None,
                network_bandwidth_mbps: Some(50),
            },
            _ => ResourceRequirements {
                cpu_cores: 1,
                memory_mb: 512,
                disk_mb: 256,
                gpu_memory_mb: None,
                network_bandwidth_mbps: None,
            },
        }
    }

    /// Get interface endpoints for a specific module
    async fn get_module_endpoints(&self, module_name: &str) -> Vec<InterfaceEndpoint> {
        match module_name {
            "trebuchet" => vec![
                InterfaceEndpoint {
                    name: "microservice_api".to_string(),
                    endpoint_type: EndpointType::Grpc,
                    address: "localhost:50051".to_string(),
                    protocol: "grpc".to_string(),
                    authentication_required: true,
                }
            ],
            "nebuchadnezzar" => vec![
                InterfaceEndpoint {
                    name: "ai_assistant_api".to_string(),
                    endpoint_type: EndpointType::Rest,
                    address: "localhost:8080".to_string(),
                    protocol: "http".to_string(),
                    authentication_required: false,
                }
            ],
            _ => Vec::new(),
        }
    }

    /// Process a user's Turbulance code and orchestrate execution
    pub async fn process_turbulance_code(&mut self, code: &str) -> Result<serde_json::Value> {
        info!("Processing Turbulance code");
        
        // Parse user intent from the code
        let intent = self.parse_user_intent(code).await?;
        self.context.write().await.user_intent = Some(intent.clone());
        
        // Create execution plan
        let execution_plan = self.execution_planner.create_plan(&intent, &self.module_registry).await?;
        
        // Log the orchestration decision
        self.logger.log_decision(&format!("Created execution plan for intent: {:?}", intent.intent_type)).await?;
        
        // Execute the plan
        let result = self.execute_plan(execution_plan).await?;
        
        Ok(result)
    }

    /// Parse user intent from Turbulance code
    async fn parse_user_intent(&self, code: &str) -> Result<UserIntent> {
        // Simplified intent parsing - in reality this would be much more sophisticated
        let intent_type = if code.contains("analyze") {
            IntentType::Analysis
        } else if code.contains("generate") || code.contains("create") {
            IntentType::Generation
        } else if code.contains("transform") {
            IntentType::Transformation
        } else {
            IntentType::Query
        };
        
        Ok(UserIntent {
            intent_type,
            confidence: 0.8,
            parameters: HashMap::new(),
            context_requirements: Vec::new(),
            expected_outputs: Vec::new(),
            timeline: None,
        })
    }

    /// Execute an execution plan
    async fn execute_plan(&mut self, plan: execution_planner::ExecutionPlan) -> Result<serde_json::Value> {
        let task_id = Uuid::new_v4();
        let start_time = SystemTime::now();
        
        // Log execution start
        self.logger.log_execution_start(task_id, &plan).await?;
        
        // Execute steps in sequence or parallel based on plan
        let mut results = HashMap::new();
        
        for step in plan.steps {
            match self.execute_step(&step).await {
                Ok(result) => {
                    results.insert(step.module_name.clone(), result);
                }
                Err(e) => {
                    error!("Step execution failed for module {}: {}", step.module_name, e);
                    self.logger.log_execution_error(task_id, &step.module_name, &e.to_string()).await?;
                    
                    if !step.optional {
                        return Err(e);
                    }
                }
            }
        }
        
        // Log execution completion
        let end_time = SystemTime::now();
        let execution_time = end_time.duration_since(start_time)?;
        
        self.logger.log_execution_completion(task_id, execution_time, &results).await?;
        
        Ok(serde_json::json!(results))
    }

    /// Execute a single step
    async fn execute_step(&self, step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        debug!("Executing step for module: {}", step.module_name);
        
        // This would route to the appropriate module implementation
        match step.module_name.as_str() {
            "fullscreen" => self.execute_fullscreen_step(step).await,
            "spectacular" => self.execute_spectacular_step(step).await,
            "nebuchadnezzar" => self.execute_nebuchadnezzar_step(step).await,
            "trebuchet" => self.execute_trebuchet_step(step).await,
            "sighthound" => self.execute_sighthound_step(step).await,
            "zangalewa" => self.execute_zangalewa_step(step).await,
            _ => Err(anyhow::anyhow!("Unknown module: {}", step.module_name)),
        }
    }

    /// Execute fullscreen module step
    async fn execute_fullscreen_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual fullscreen module
        Ok(serde_json::json!({"status": "completed", "module": "fullscreen"}))
    }

    /// Execute spectacular module step
    async fn execute_spectacular_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual spectacular module
        Ok(serde_json::json!({"status": "completed", "module": "spectacular"}))
    }

    /// Execute nebuchadnezzar module step
    async fn execute_nebuchadnezzar_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual nebuchadnezzar module
        Ok(serde_json::json!({"status": "completed", "module": "nebuchadnezzar"}))
    }

    /// Execute trebuchet module step
    async fn execute_trebuchet_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual trebuchet module
        Ok(serde_json::json!({"status": "completed", "module": "trebuchet"}))
    }

    /// Execute sighthound module step
    async fn execute_sighthound_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual sighthound module
        Ok(serde_json::json!({"status": "completed", "module": "sighthound"}))
    }

    /// Execute zangalewa module step
    async fn execute_zangalewa_step(&self, _step: &execution_planner::ExecutionStep) -> Result<serde_json::Value> {
        // TODO: Integrate with actual zangalewa module
        Ok(serde_json::json!({"status": "completed", "module": "zangalewa"}))
    }

    /// Start health monitoring for all modules
    async fn start_health_monitoring(&self) {
        // TODO: Implement health monitoring loop
    }

    /// Start metacognitive loop
    async fn start_metacognitive_loop(&self) {
        // TODO: Implement metacognitive loop
    }

    /// Start resource monitoring
    async fn start_resource_monitoring(&self) {
        // TODO: Implement resource monitoring
    }
} 