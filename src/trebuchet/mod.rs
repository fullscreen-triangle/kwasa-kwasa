use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use tokio::sync::{RwLock, mpsc};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use tonic::{transport::Server, Request, Response, Status};

pub mod microservice_manager;
pub mod service_registry;
pub mod load_balancer;
pub mod deployment_engine;
pub mod monitoring;
pub mod resource_manager;
pub mod communication;

/// The main Trebuchet microservices system
pub struct TrebuchetSystem {
    pub id: Uuid,
    pub config: TrebuchetConfig,
    pub service_registry: Arc<RwLock<service_registry::ServiceRegistry>>,
    pub microservice_manager: Arc<RwLock<microservice_manager::MicroserviceManager>>,
    pub load_balancer: Arc<RwLock<load_balancer::LoadBalancer>>,
    pub deployment_engine: Arc<RwLock<deployment_engine::DeploymentEngine>>,
    pub monitoring: Arc<RwLock<monitoring::MonitoringSystem>>,
    pub resource_manager: Arc<RwLock<resource_manager::ResourceManager>>,
    pub communication_hub: Arc<RwLock<communication::CommunicationHub>>,
    pub active_services: Arc<RwLock<HashMap<String, ServiceInstance>>>,
    pub orchestrator_commands: mpsc::UnboundedReceiver<OrchestratorCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrebuchetConfig {
    pub deployment_config: DeploymentConfig,
    pub performance_config: PerformanceConfig,
    pub networking_config: NetworkingConfig,
    pub security_config: SecurityConfig,
    pub monitoring_config: MonitoringConfig,
    pub resource_limits: GlobalResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub auto_scaling: bool,
    pub min_instances: u32,
    pub max_instances: u32,
    pub deployment_strategy: DeploymentStrategy,
    pub rollback_enabled: bool,
    pub health_check_interval_ms: u64,
    pub startup_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    BlueGreen,
    RollingUpdate,
    Canary,
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub target_latency_ms: u64,
    pub max_throughput_rps: u64,
    pub cpu_optimization: bool,
    pub memory_optimization: bool,
    pub cache_optimization: bool,
    pub connection_pooling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkingConfig {
    pub service_mesh_enabled: bool,
    pub load_balancing_algorithm: LoadBalancingAlgorithm,
    pub circuit_breaker_enabled: bool,
    pub retry_policy: RetryPolicy,
    pub timeout_policy: TimeoutPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    IpHash,
    Random,
    PerformanceBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub backoff_strategy: BackoffStrategy,
    pub retry_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Linear,
    Exponential,
    Fixed,
    Jittered,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutPolicy {
    pub connection_timeout_ms: u64,
    pub request_timeout_ms: u64,
    pub idle_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub authentication_enabled: bool,
    pub authorization_enabled: bool,
    pub encryption_in_transit: bool,
    pub encryption_at_rest: bool,
    pub rate_limiting: RateLimitConfig,
    pub firewall_rules: Vec<FirewallRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: u64,
    pub burst_capacity: u64,
    pub rate_limit_algorithm: RateLimitAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    TokenBucket,
    LeakyBucket,
    FixedWindow,
    SlidingWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    pub rule_id: String,
    pub source_ip_range: String,
    pub destination_port: u16,
    pub protocol: String,
    pub action: FirewallAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FirewallAction {
    Allow,
    Deny,
    Log,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_collection_interval_ms: u64,
    pub log_level: LogLevel,
    pub distributed_tracing: bool,
    pub performance_profiling: bool,
    pub anomaly_detection: bool,
    pub alerting_config: AlertingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub alert_channels: Vec<AlertChannel>,
    pub alert_thresholds: HashMap<String, f64>,
    pub escalation_policies: Vec<EscalationPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertChannel {
    pub channel_type: String,
    pub endpoint: String,
    pub severity_filter: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub policy_name: String,
    pub escalation_steps: Vec<EscalationStep>,
    pub timeout_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationStep {
    pub step_number: u32,
    pub delay_minutes: u64,
    pub notification_targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalResourceLimits {
    pub max_total_cpu_cores: u32,
    pub max_total_memory_gb: u64,
    pub max_total_disk_gb: u64,
    pub max_network_bandwidth_gbps: f64,
    pub max_concurrent_connections: u64,
}

/// Service instance running in Trebuchet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInstance {
    pub instance_id: Uuid,
    pub service_name: String,
    pub service_type: ServiceType,
    pub version: String,
    pub status: ServiceStatus,
    pub endpoint: ServiceEndpoint,
    pub resource_allocation: ResourceAllocation,
    pub health_metrics: HealthMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub deployment_time: SystemTime,
    pub last_health_check: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceType {
    AudioProcessing,    // Heihachi
    NlpProcessing,      // Gospel
    ModelManager,       // Purpose  
    DataIntegration,    // Combine
    VideoProcessing,    // Spectacular
    GeospatialProcessing, // Sighthound
    CodeExecution,      // Zangalewa
    CustomService,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
    Upgrading,
    Scaling,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub host: String,
    pub port: u16,
    pub protocol: Protocol,
    pub path: Option<String>,
    pub health_check_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Protocol {
    Http,
    Https,
    Grpc,
    WebSocket,
    Tcp,
    Udp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub disk_mb: u64,
    pub network_bandwidth_mbps: f64,
    pub gpu_memory_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub is_healthy: bool,
    pub response_time_ms: u64,
    pub error_rate_percent: f64,
    pub uptime_seconds: u64,
    pub last_error: Option<String>,
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub requests_per_second: f64,
    pub average_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub active_connections: u64,
    pub throughput_mbps: f64,
}

/// Commands from the Harare orchestrator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorCommand {
    pub command_id: Uuid,
    pub command_type: CommandType,
    pub target_service: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub priority: CommandPriority,
    pub deadline: Option<SystemTime>,
    pub constraints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    DeployService,
    UpdateService,
    ScaleService,
    StopService,
    RestartService,
    ConfigureService,
    HealthCheck,
    CollectMetrics,
    ExecuteTask,
    OptimizePerformance,
    ApplySecurityPatch,
    BackupData,
    RestoreData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandPriority {
    Critical,
    High,
    Normal,
    Low,
    Background,
}

/// Service execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceExecutionRequest {
    pub request_id: Uuid,
    pub service_name: String,
    pub operation: String,
    pub input_data: serde_json::Value,
    pub execution_context: ExecutionContext,
    pub resource_requirements: ResourceRequirements,
    pub performance_requirements: PerformanceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub user_id: Option<String>,
    pub session_id: Option<Uuid>,
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub environment: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_cpu_cores: f64,
    pub min_memory_mb: u64,
    pub max_execution_time_ms: u64,
    pub requires_gpu: bool,
    pub network_access_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub max_latency_ms: u64,
    pub min_throughput_rps: f64,
    pub reliability_target: f64,
    pub consistency_level: ConsistencyLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Session,
    Bounded,
}

/// Service execution response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceExecutionResponse {
    pub request_id: Uuid,
    pub execution_id: Uuid,
    pub status: ExecutionStatus,
    pub result: Option<serde_json::Value>,
    pub error: Option<ExecutionError>,
    pub metrics: ExecutionMetrics,
    pub trace_data: TraceData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
    ResourceLimited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub error_code: String,
    pub error_message: String,
    pub error_details: HashMap<String, serde_json::Value>,
    pub stack_trace: Option<String>,
    pub retry_recommended: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub execution_time_ms: u64,
    pub cpu_time_ms: u64,
    pub memory_peak_mb: u64,
    pub network_bytes_in: u64,
    pub network_bytes_out: u64,
    pub disk_reads_mb: u64,
    pub disk_writes_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceData {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub tags: HashMap<String, String>,
    pub logs: Vec<TraceLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLog {
    pub timestamp: SystemTime,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
}

impl TrebuchetSystem {
    /// Create a new Trebuchet microservices system
    pub async fn new(config: TrebuchetConfig) -> Result<Self> {
        let id = Uuid::new_v4();
        let (orchestrator_sender, orchestrator_receiver) = mpsc::unbounded_channel();
        
        // Initialize all subsystems
        let service_registry = Arc::new(RwLock::new(
            service_registry::ServiceRegistry::new().await?
        ));
        
        let microservice_manager = Arc::new(RwLock::new(
            microservice_manager::MicroserviceManager::new(config.clone()).await?
        ));
        
        let load_balancer = Arc::new(RwLock::new(
            load_balancer::LoadBalancer::new(config.networking_config.clone()).await?
        ));
        
        let deployment_engine = Arc::new(RwLock::new(
            deployment_engine::DeploymentEngine::new(config.deployment_config.clone()).await?
        ));
        
        let monitoring = Arc::new(RwLock::new(
            monitoring::MonitoringSystem::new(config.monitoring_config.clone()).await?
        ));
        
        let resource_manager = Arc::new(RwLock::new(
            resource_manager::ResourceManager::new(config.resource_limits.clone()).await?
        ));
        
        let communication_hub = Arc::new(RwLock::new(
            communication::CommunicationHub::new().await?
        ));
        
        Ok(Self {
            id,
            config,
            service_registry,
            microservice_manager,
            load_balancer,
            deployment_engine,
            monitoring,
            resource_manager,
            communication_hub,
            active_services: Arc::new(RwLock::new(HashMap::new())),
            orchestrator_commands: orchestrator_receiver,
        })
    }

    /// Start the Trebuchet system
    pub async fn start(&mut self) -> Result<()> {
        tracing::info!("Starting Trebuchet microservices system: {}", self.id);
        
        // Start all subsystems
        self.start_service_registry().await?;
        self.start_microservice_manager().await?;
        self.start_load_balancer().await?;
        self.start_monitoring().await?;
        self.start_resource_manager().await?;
        self.start_communication_hub().await?;
        
        // Start orchestrator command processing loop
        self.start_orchestrator_command_loop().await?;
        
        // Deploy default microservices
        self.deploy_default_services().await?;
        
        tracing::info!("Trebuchet system started successfully");
        Ok(())
    }

    /// Process orchestrator command
    pub async fn process_orchestrator_command(&mut self, command: OrchestratorCommand) -> Result<serde_json::Value> {
        tracing::info!("Processing orchestrator command: {:?}", command.command_type);
        
        let result = match command.command_type {
            CommandType::DeployService => self.deploy_service(&command).await?,
            CommandType::UpdateService => self.update_service(&command).await?,
            CommandType::ScaleService => self.scale_service(&command).await?,
            CommandType::StopService => self.stop_service(&command).await?,
            CommandType::RestartService => self.restart_service(&command).await?,
            CommandType::ConfigureService => self.configure_service(&command).await?,
            CommandType::HealthCheck => self.perform_health_check(&command).await?,
            CommandType::CollectMetrics => self.collect_metrics(&command).await?,
            CommandType::ExecuteTask => self.execute_task(&command).await?,
            CommandType::OptimizePerformance => self.optimize_performance(&command).await?,
            CommandType::ApplySecurityPatch => self.apply_security_patch(&command).await?,
            CommandType::BackupData => self.backup_data(&command).await?,
            CommandType::RestoreData => self.restore_data(&command).await?,
        };
        
        Ok(result)
    }

    /// Execute a service request
    pub async fn execute_service_request(&self, request: ServiceExecutionRequest) -> Result<ServiceExecutionResponse> {
        let execution_id = Uuid::new_v4();
        let start_time = SystemTime::now();
        
        tracing::info!("Executing service request: {} on service: {}", request.request_id, request.service_name);
        
        // Validate resource requirements
        self.resource_manager.read().await.validate_resource_requirements(&request.resource_requirements).await?;
        
        // Route request to appropriate service
        let service_instance = self.select_service_instance(&request.service_name, &request.performance_requirements).await?;
        
        // Execute the request
        let result = self.execute_on_service_instance(&service_instance, &request).await?;
        
        // Collect execution metrics
        let end_time = SystemTime::now();
        let execution_time = end_time.duration_since(start_time).unwrap_or(Duration::ZERO);
        
        let response = ServiceExecutionResponse {
            request_id: request.request_id,
            execution_id,
            status: ExecutionStatus::Completed,
            result: Some(result),
            error: None,
            metrics: ExecutionMetrics {
                execution_time_ms: execution_time.as_millis() as u64,
                cpu_time_ms: 0, // Would be measured
                memory_peak_mb: 0, // Would be measured
                network_bytes_in: 0, // Would be measured
                network_bytes_out: 0, // Would be measured
                disk_reads_mb: 0, // Would be measured
                disk_writes_mb: 0, // Would be measured
            },
            trace_data: TraceData {
                trace_id: request.execution_context.trace_id,
                span_id: execution_id.to_string(),
                parent_span_id: request.execution_context.parent_span_id,
                operation_name: request.operation,
                start_time,
                end_time: Some(end_time),
                tags: HashMap::new(),
                logs: Vec::new(),
            },
        };
        
        Ok(response)
    }

    /// Get system status and metrics
    pub async fn get_system_status(&self) -> SystemStatus {
        let active_services = self.active_services.read().await;
        let total_services = active_services.len();
        let healthy_services = active_services.values()
            .filter(|s| s.health_metrics.is_healthy)
            .count();
        
        SystemStatus {
            system_id: self.id,
            uptime_seconds: 0, // Would be calculated
            total_services,
            healthy_services,
            total_requests_processed: 0, // Would be tracked
            average_response_time_ms: 0.0, // Would be calculated
            resource_utilization: ResourceUtilization {
                cpu_usage_percent: 0.0, // Would be measured
                memory_usage_percent: 0.0, // Would be measured
                disk_usage_percent: 0.0, // Would be measured
                network_usage_percent: 0.0, // Would be measured
            },
            performance_metrics: SystemPerformanceMetrics {
                throughput_rps: 0.0, // Would be calculated
                error_rate_percent: 0.0, // Would be calculated
                p95_latency_ms: 0.0, // Would be calculated
                p99_latency_ms: 0.0, // Would be calculated
            },
            alerts: Vec::new(), // Would include active alerts
        }
    }

    // Implementation of private methods...
    
    async fn start_service_registry(&self) -> Result<()> {
        // Initialize service registry
        Ok(())
    }

    async fn start_microservice_manager(&self) -> Result<()> {
        // Initialize microservice manager
        Ok(())
    }

    async fn start_load_balancer(&self) -> Result<()> {
        // Initialize load balancer
        Ok(())
    }

    async fn start_monitoring(&self) -> Result<()> {
        // Initialize monitoring system
        Ok(())
    }

    async fn start_resource_manager(&self) -> Result<()> {
        // Initialize resource manager
        Ok(())
    }

    async fn start_communication_hub(&self) -> Result<()> {
        // Initialize communication hub
        Ok(())
    }

    async fn start_orchestrator_command_loop(&self) -> Result<()> {
        // Start command processing loop
        Ok(())
    }

    async fn deploy_default_services(&self) -> Result<()> {
        // Deploy default microservices
        Ok(())
    }

    async fn deploy_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "deployed"}))
    }

    async fn update_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "updated"}))
    }

    async fn scale_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "scaled"}))
    }

    async fn stop_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "stopped"}))
    }

    async fn restart_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "restarted"}))
    }

    async fn configure_service(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "configured"}))
    }

    async fn perform_health_check(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "healthy"}))
    }

    async fn collect_metrics(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"metrics": "collected"}))
    }

    async fn execute_task(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "executed"}))
    }

    async fn optimize_performance(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "optimized"}))
    }

    async fn apply_security_patch(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "patched"}))
    }

    async fn backup_data(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "backed_up"}))
    }

    async fn restore_data(&self, _command: &OrchestratorCommand) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"status": "restored"}))
    }

    async fn select_service_instance(&self, service_name: &str, _requirements: &PerformanceRequirements) -> Result<ServiceInstance> {
        let active_services = self.active_services.read().await;
        
        // Find the best service instance for the request
        for service in active_services.values() {
            if service.service_name == service_name && service.health_metrics.is_healthy {
                return Ok(service.clone());
            }
        }
        
        Err(anyhow::anyhow!("No healthy service instance found for: {}", service_name))
    }

    async fn execute_on_service_instance(&self, _service: &ServiceInstance, _request: &ServiceExecutionRequest) -> Result<serde_json::Value> {
        // Execute request on specific service instance
        Ok(serde_json::json!({"result": "executed"}))
    }
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub system_id: Uuid,
    pub uptime_seconds: u64,
    pub total_services: usize,
    pub healthy_services: usize,
    pub total_requests_processed: u64,
    pub average_response_time_ms: f64,
    pub resource_utilization: ResourceUtilization,
    pub performance_metrics: SystemPerformanceMetrics,
    pub alerts: Vec<SystemAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    pub throughput_rps: f64,
    pub error_rate_percent: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAlert {
    pub alert_id: Uuid,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub timestamp: SystemTime,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
} 