use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::time::{SystemTime, Duration};
use tree_sitter::{Parser, Language, Tree, Node};

pub mod execution_engine;
pub mod error_analyzer;
pub mod codebase_analyzer;
pub mod intelligent_fixes;
pub mod language_support;
pub mod dependency_manager;
pub mod testing_framework;
pub mod performance_profiler;
pub mod security_scanner;

/// The main Zangalewa code execution and analysis system
pub struct ZangalewaSystem {
    pub id: Uuid,
    pub config: ZangalewaConfig,
    pub execution_engine: Arc<RwLock<execution_engine::ExecutionEngine>>,
    pub error_analyzer: Arc<RwLock<error_analyzer::ErrorAnalyzer>>,
    pub codebase_analyzer: Arc<RwLock<codebase_analyzer::CodebaseAnalyzer>>,
    pub intelligent_fixes: Arc<RwLock<intelligent_fixes::IntelligentFixEngine>>,
    pub language_support: Arc<RwLock<language_support::LanguageSupport>>,
    pub dependency_manager: Arc<RwLock<dependency_manager::DependencyManager>>,
    pub testing_framework: Arc<RwLock<testing_framework::TestingFramework>>,
    pub performance_profiler: Arc<RwLock<performance_profiler::PerformanceProfiler>>,
    pub security_scanner: Arc<RwLock<security_scanner::SecurityScanner>>,
    pub active_executions: Arc<RwLock<HashMap<Uuid, ExecutionSession>>>,
    pub project_contexts: Arc<RwLock<HashMap<PathBuf, ProjectContext>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZangalewaConfig {
    pub execution_config: ExecutionConfig,
    pub analysis_config: AnalysisConfig,
    pub error_handling_config: ErrorHandlingConfig,
    pub security_config: SecurityConfig,
    pub performance_config: PerformanceConfig,
    pub language_configs: HashMap<String, LanguageConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub max_execution_time_ms: u64,
    pub max_memory_mb: u64,
    pub max_cpu_cores: u32,
    pub sandboxing_enabled: bool,
    pub network_access_allowed: bool,
    pub file_system_access: FileSystemAccess,
    pub environment_isolation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileSystemAccess {
    None,
    ReadOnly,
    ProjectDirectory,
    Full,
    Custom(Vec<PathBuf>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub static_analysis_enabled: bool,
    pub dynamic_analysis_enabled: bool,
    pub semantic_analysis_depth: AnalysisDepth,
    pub code_quality_checks: bool,
    pub dependency_analysis: bool,
    pub security_analysis: bool,
    pub performance_analysis: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Surface,
    Moderate,
    Deep,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    pub auto_recovery_enabled: bool,
    pub max_auto_fix_attempts: u32,
    pub learning_from_errors: bool,
    pub error_pattern_detection: bool,
    pub intelligent_suggestions: bool,
    pub context_aware_fixes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub vulnerability_scanning: bool,
    pub code_injection_detection: bool,
    pub dependency_security_check: bool,
    pub secrets_detection: bool,
    pub permission_analysis: bool,
    pub sandbox_escape_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub profiling_enabled: bool,
    pub memory_tracking: bool,
    pub cpu_profiling: bool,
    pub io_monitoring: bool,
    pub network_monitoring: bool,
    pub bottleneck_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub language_name: String,
    pub interpreter_path: Option<PathBuf>,
    pub compiler_path: Option<PathBuf>,
    pub package_manager: Option<String>,
    pub syntax_parser: SyntaxParserConfig,
    pub execution_flags: Vec<String>,
    pub environment_setup: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxParserConfig {
    pub tree_sitter_language: String,
    pub ast_analysis_enabled: bool,
    pub semantic_highlighting: bool,
    pub error_recovery: bool,
}

/// Code execution session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSession {
    pub session_id: Uuid,
    pub project_path: PathBuf,
    pub language: String,
    pub execution_type: ExecutionType,
    pub start_time: SystemTime,
    pub status: ExecutionStatus,
    pub code_files: Vec<CodeFile>,
    pub execution_results: Option<ExecutionResults>,
    pub error_analysis: Option<ErrorAnalysisResults>,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub security_findings: Option<SecurityFindings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionType {
    ScriptExecution,
    UnitTesting,
    Integration,
    Debugging,
    Profiling,
    Analysis,
    Compilation,
    Deployment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Queued,
    Preparing,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
    SecurityBlocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    pub file_path: PathBuf,
    pub language: String,
    pub content: String,
    pub syntax_tree: Option<SyntaxTree>,
    pub dependencies: Vec<Dependency>,
    pub complexity_metrics: Option<ComplexityMetrics>,
    pub quality_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxTree {
    pub root_node: SyntaxNode,
    pub parse_errors: Vec<ParseError>,
    pub language_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxNode {
    pub node_type: String,
    pub start_position: Position,
    pub end_position: Position,
    pub children: Vec<SyntaxNode>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub column: u32,
    pub byte_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub error_type: String,
    pub message: String,
    pub position: Position,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: Option<String>,
    pub dependency_type: DependencyType,
    pub source: DependencySource,
    pub is_dev_dependency: bool,
    pub security_issues: Vec<SecurityIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Runtime,
    BuildTime,
    Test,
    Optional,
    Peer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencySource {
    PackageManager,
    Git,
    Local,
    Url,
    Registry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityIssue {
    pub issue_id: String,
    pub severity: SecuritySeverity,
    pub description: String,
    pub cve_id: Option<String>,
    pub affected_versions: Vec<String>,
    pub fixed_versions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: u32,
    pub cognitive_complexity: u32,
    pub lines_of_code: u32,
    pub maintainability_index: f64,
    pub technical_debt_minutes: u32,
    pub code_coverage: Option<f64>,
}

/// Execution results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResults {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub execution_time_ms: u64,
    pub resource_usage: ResourceUsage,
    pub test_results: Option<TestResults>,
    pub compilation_artifacts: Vec<CompilationArtifact>,
    pub runtime_errors: Vec<RuntimeError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub max_memory_mb: u64,
    pub cpu_time_ms: u64,
    pub wall_time_ms: u64,
    pub disk_reads_mb: u64,
    pub disk_writes_mb: u64,
    pub network_bytes_in: u64,
    pub network_bytes_out: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub total_tests: u32,
    pub passed_tests: u32,
    pub failed_tests: u32,
    pub skipped_tests: u32,
    pub test_cases: Vec<TestCase>,
    pub coverage_report: Option<CoverageReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub test_name: String,
    pub status: TestStatus,
    pub execution_time_ms: u64,
    pub error_message: Option<String>,
    pub assertions: Vec<Assertion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    pub assertion_type: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub line_coverage_percent: f64,
    pub branch_coverage_percent: f64,
    pub function_coverage_percent: f64,
    pub covered_lines: Vec<u32>,
    pub uncovered_lines: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationArtifact {
    pub artifact_type: ArtifactType,
    pub file_path: PathBuf,
    pub size_bytes: u64,
    pub checksum: String,
    pub optimization_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    Executable,
    Library,
    Object,
    Archive,
    Documentation,
    Debug,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeError {
    pub error_type: String,
    pub message: String,
    pub stack_trace: Vec<StackFrame>,
    pub line_number: Option<u32>,
    pub file_path: Option<PathBuf>,
    pub severity: ErrorSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    pub function_name: String,
    pub file_path: PathBuf,
    pub line_number: u32,
    pub column_number: Option<u32>,
}

/// Error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysisResults {
    pub error_classification: ErrorClassification,
    pub root_cause_analysis: RootCauseAnalysis,
    pub suggested_fixes: Vec<SuggestedFix>,
    pub similar_errors: Vec<SimilarError>,
    pub prevention_recommendations: Vec<PreventionRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorClassification {
    pub primary_category: ErrorCategory,
    pub subcategories: Vec<String>,
    pub confidence: f64,
    pub error_patterns: Vec<ErrorPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCategory {
    SyntaxError,
    TypeError,
    LogicError,
    RuntimeError,
    PerformanceIssue,
    SecurityVulnerability,
    DependencyIssue,
    ConfigurationError,
    EnvironmentIssue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern_name: String,
    pub pattern_description: String,
    pub frequency: u32,
    pub typical_causes: Vec<String>,
    pub typical_solutions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCauseAnalysis {
    pub primary_cause: String,
    pub contributing_factors: Vec<ContributingFactor>,
    pub causal_chain: Vec<CausalLink>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub factor_type: String,
    pub description: String,
    pub impact_level: ImpactLevel,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImpactLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalLink {
    pub from_event: String,
    pub to_event: String,
    pub relationship_type: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedFix {
    pub fix_id: Uuid,
    pub fix_type: FixType,
    pub description: String,
    pub code_changes: Vec<CodeChange>,
    pub confidence: f64,
    pub estimated_effort: EffortEstimate,
    pub side_effects: Vec<SideEffect>,
    pub success_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixType {
    AutomaticFix,
    SemiAutomaticFix,
    ManualFix,
    RefactoringRequired,
    ArchitecturalChange,
    ConfigurationChange,
    DependencyUpdate,
    EnvironmentFix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    pub file_path: PathBuf,
    pub change_type: ChangeType,
    pub start_line: u32,
    pub end_line: u32,
    pub original_code: String,
    pub new_code: String,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Replace,
    Insert,
    Delete,
    Move,
    Rename,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortEstimate {
    pub time_minutes: u32,
    pub difficulty: Difficulty,
    pub risk_level: RiskLevel,
    pub required_skills: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Difficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffect {
    pub effect_type: String,
    pub description: String,
    pub likelihood: f64,
    pub severity: ImpactLevel,
    pub mitigation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarError {
    pub error_description: String,
    pub similarity_score: f64,
    pub context_similarity: f64,
    pub resolution_success_rate: f64,
    pub common_fixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreventionRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub implementation_steps: Vec<String>,
    pub prevention_effectiveness: f64,
    pub maintenance_required: bool,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_profile: ExecutionProfile,
    pub memory_profile: MemoryProfile,
    pub io_profile: IoProfile,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub total_time_ms: u64,
    pub function_times: HashMap<String, u64>,
    pub hot_paths: Vec<HotPath>,
    pub call_graph: CallGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPath {
    pub path_description: String,
    pub execution_count: u64,
    pub total_time_ms: u64,
    pub average_time_ms: f64,
    pub percentage_of_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraph {
    pub nodes: Vec<CallGraphNode>,
    pub edges: Vec<CallGraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphNode {
    pub function_name: String,
    pub file_path: PathBuf,
    pub line_number: u32,
    pub execution_count: u64,
    pub total_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphEdge {
    pub caller: String,
    pub callee: String,
    pub call_count: u64,
    pub total_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub peak_memory_mb: u64,
    pub memory_timeline: Vec<MemorySnapshot>,
    pub allocation_hotspots: Vec<AllocationHotspot>,
    pub memory_leaks: Vec<MemoryLeak>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp_ms: u64,
    pub total_memory_mb: u64,
    pub heap_memory_mb: u64,
    pub stack_memory_mb: u64,
    pub active_objects: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationHotspot {
    pub location: String,
    pub allocation_count: u64,
    pub total_bytes: u64,
    pub average_allocation_size: f64,
    pub percentage_of_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    pub leak_type: String,
    pub estimated_leak_rate_mb_per_sec: f64,
    pub source_location: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoProfile {
    pub total_io_time_ms: u64,
    pub disk_operations: Vec<DiskOperation>,
    pub network_operations: Vec<NetworkOperation>,
    pub io_bottlenecks: Vec<IoBottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskOperation {
    pub operation_type: String,
    pub file_path: PathBuf,
    pub bytes_transferred: u64,
    pub duration_ms: u64,
    pub iops: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOperation {
    pub operation_type: String,
    pub endpoint: String,
    pub bytes_transferred: u64,
    pub duration_ms: u64,
    pub throughput_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoBottleneck {
    pub bottleneck_type: String,
    pub description: String,
    pub impact_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub bottleneck_id: String,
    pub bottleneck_type: BottleneckType,
    pub location: String,
    pub impact_score: f64,
    pub description: String,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    MemoryBound,
    IoBound,
    NetworkBound,
    AlgorithmicComplexity,
    ResourceContention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_id: String,
    pub optimization_type: OptimizationType,
    pub description: String,
    pub expected_improvement: ExpectedImprovement,
    pub implementation_complexity: Difficulty,
    pub code_changes: Vec<CodeChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    AlgorithmOptimization,
    DataStructureOptimization,
    MemoryOptimization,
    IoOptimization,
    CachingStrategy,
    ParallelizationOpportunity,
    CompilerOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedImprovement {
    pub performance_gain_percent: f64,
    pub memory_reduction_percent: f64,
    pub latency_reduction_percent: f64,
    pub throughput_increase_percent: f64,
}

/// Security findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityFindings {
    pub vulnerabilities: Vec<SecurityVulnerability>,
    pub security_score: f64,
    pub compliance_status: ComplianceStatus,
    pub remediation_plan: RemediationPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVulnerability {
    pub vulnerability_id: String,
    pub vulnerability_type: VulnerabilityType,
    pub severity: SecuritySeverity,
    pub description: String,
    pub location: VulnerabilityLocation,
    pub cwe_id: Option<String>,
    pub cvss_score: Option<f64>,
    pub remediation: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilityType {
    InjectionAttack,
    CrossSiteScripting,
    InsecureDeserialization,
    BrokenAuthentication,
    SensitiveDataExposure,
    XmlExternalEntities,
    BrokenAccessControl,
    SecurityMisconfiguration,
    KnownVulnerableComponents,
    InsufficientLogging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityLocation {
    pub file_path: PathBuf,
    pub line_number: u32,
    pub function_name: Option<String>,
    pub code_snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub overall_compliance_score: f64,
    pub standards_compliance: HashMap<String, f64>,
    pub failed_checks: Vec<ComplianceCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    pub check_id: String,
    pub standard: String,
    pub description: String,
    pub status: CheckStatus,
    pub remediation_steps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckStatus {
    Passed,
    Failed,
    Warning,
    NotApplicable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationPlan {
    pub high_priority_actions: Vec<RemediationAction>,
    pub medium_priority_actions: Vec<RemediationAction>,
    pub low_priority_actions: Vec<RemediationAction>,
    pub estimated_total_effort_hours: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemediationAction {
    pub action_id: String,
    pub description: String,
    pub vulnerability_ids: Vec<String>,
    pub estimated_effort_hours: u32,
    pub risk_reduction: f64,
    pub implementation_steps: Vec<String>,
}

/// Project context for maintaining state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectContext {
    pub project_path: PathBuf,
    pub project_type: ProjectType,
    pub languages: Vec<String>,
    pub dependencies: HashMap<String, Vec<Dependency>>,
    pub build_configuration: BuildConfiguration,
    pub test_configuration: TestConfiguration,
    pub quality_metrics: QualityMetrics,
    pub recent_errors: VecDeque<ErrorRecord>,
    pub learning_data: LearningData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProjectType {
    Library,
    Application,
    WebService,
    MobileApp,
    DesktopApp,
    Script,
    DataPipeline,
    MachineLearning,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfiguration {
    pub build_system: String,
    pub build_file_path: PathBuf,
    pub build_targets: Vec<String>,
    pub environment_variables: HashMap<String, String>,
    pub build_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    pub test_framework: String,
    pub test_directories: Vec<PathBuf>,
    pub test_patterns: Vec<String>,
    pub coverage_enabled: bool,
    pub test_environment: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub overall_quality_score: f64,
    pub maintainability_score: f64,
    pub reliability_score: f64,
    pub security_score: f64,
    pub performance_score: f64,
    pub test_coverage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub timestamp: SystemTime,
    pub error_type: String,
    pub error_message: String,
    pub file_path: PathBuf,
    pub resolution_attempted: bool,
    pub resolution_successful: bool,
    pub learning_extracted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningData {
    pub common_error_patterns: HashMap<String, u32>,
    pub successful_fix_patterns: HashMap<String, u32>,
    pub user_preferences: UserPreferences,
    pub project_specific_rules: Vec<ProjectRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub preferred_fix_types: Vec<FixType>,
    pub auto_fix_confidence_threshold: f64,
    pub preferred_coding_style: HashMap<String, String>,
    pub notification_preferences: NotificationPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub error_notifications: bool,
    pub fix_suggestions: bool,
    pub performance_warnings: bool,
    pub security_alerts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectRule {
    pub rule_id: String,
    pub rule_type: RuleType,
    pub condition: String,
    pub action: String,
    pub confidence: f64,
    pub usage_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleType {
    ErrorPattern,
    FixPattern,
    QualityRule,
    PerformanceRule,
    SecurityRule,
}

impl ZangalewaSystem {
    /// Create a new Zangalewa system
    pub async fn new(config: ZangalewaConfig) -> Result<Self> {
        let id = Uuid::new_v4();
        
        // Initialize all subsystems
        let execution_engine = Arc::new(RwLock::new(
            execution_engine::ExecutionEngine::new(config.execution_config.clone()).await?
        ));
        
        let error_analyzer = Arc::new(RwLock::new(
            error_analyzer::ErrorAnalyzer::new(config.error_handling_config.clone()).await?
        ));
        
        let codebase_analyzer = Arc::new(RwLock::new(
            codebase_analyzer::CodebaseAnalyzer::new(config.analysis_config.clone()).await?
        ));
        
        let intelligent_fixes = Arc::new(RwLock::new(
            intelligent_fixes::IntelligentFixEngine::new().await?
        ));
        
        let language_support = Arc::new(RwLock::new(
            language_support::LanguageSupport::new(config.language_configs.clone()).await?
        ));
        
        let dependency_manager = Arc::new(RwLock::new(
            dependency_manager::DependencyManager::new().await?
        ));
        
        let testing_framework = Arc::new(RwLock::new(
            testing_framework::TestingFramework::new().await?
        ));
        
        let performance_profiler = Arc::new(RwLock::new(
            performance_profiler::PerformanceProfiler::new(config.performance_config.clone()).await?
        ));
        
        let security_scanner = Arc::new(RwLock::new(
            security_scanner::SecurityScanner::new(config.security_config.clone()).await?
        ));
        
        Ok(Self {
            id,
            config,
            execution_engine,
            error_analyzer,
            codebase_analyzer,
            intelligent_fixes,
            language_support,
            dependency_manager,
            testing_framework,
            performance_profiler,
            security_scanner,
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            project_contexts: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Execute code with comprehensive analysis
    pub async fn execute_code(&mut self, request: CodeExecutionRequest) -> Result<ExecutionSession> {
        let session_id = Uuid::new_v4();
        let start_time = SystemTime::now();
        
        tracing::info!("Starting Zangalewa code execution session: {}", session_id);
        
        // Create execution session
        let mut session = ExecutionSession {
            session_id,
            project_path: request.project_path.clone(),
            language: request.language.clone(),
            execution_type: request.execution_type.clone(),
            start_time,
            status: ExecutionStatus::Preparing,
            code_files: request.code_files.clone(),
            execution_results: None,
            error_analysis: None,
            performance_metrics: None,
            security_findings: None,
        };
        
        // Add to active executions
        self.active_executions.write().await.insert(session_id, session.clone());
        
        // Analyze codebase first
        session.status = ExecutionStatus::Running;
        let codebase_analysis = self.analyze_codebase(&request).await?;
        
        // Execute the code
        let execution_results = self.execution_engine.write().await
            .execute_code(&request).await?;
        
        // Analyze any errors
        let error_analysis = if execution_results.exit_code != 0 || !execution_results.runtime_errors.is_empty() {
            Some(self.analyze_errors(&execution_results, &codebase_analysis).await?)
        } else {
            None
        };
        
        // Profile performance if enabled
        let performance_metrics = if self.config.performance_config.profiling_enabled {
            Some(self.performance_profiler.read().await.profile_execution(&execution_results).await?)
        } else {
            None
        };
        
        // Scan for security issues
        let security_findings = if self.config.security_config.vulnerability_scanning {
            Some(self.security_scanner.read().await.scan_code(&request.code_files).await?)
        } else {
            None
        };
        
        // Update session with results
        session.status = ExecutionStatus::Completed;
        session.execution_results = Some(execution_results);
        session.error_analysis = error_analysis;
        session.performance_metrics = performance_metrics;
        session.security_findings = security_findings;
        
        // Update active executions
        self.active_executions.write().await.insert(session_id, session.clone());
        
        // Learn from the execution
        self.learn_from_execution(&session).await?;
        
        tracing::info!("Zangalewa code execution completed successfully");
        Ok(session)
    }

    /// Analyze codebase for patterns and issues
    pub async fn analyze_codebase(&self, request: &CodeExecutionRequest) -> Result<CodebaseAnalysis> {
        self.codebase_analyzer.read().await.analyze_codebase(request).await
    }

    /// Analyze errors and provide intelligent fixes
    pub async fn analyze_errors(&self, execution_results: &ExecutionResults, codebase_analysis: &CodebaseAnalysis) -> Result<ErrorAnalysisResults> {
        self.error_analyzer.read().await.analyze_errors(execution_results, codebase_analysis).await
    }

    /// Apply intelligent fixes to code
    pub async fn apply_intelligent_fixes(&self, session_id: Uuid, fix_ids: Vec<Uuid>) -> Result<FixApplicationResults> {
        let session = {
            let sessions = self.active_executions.read().await;
            sessions.get(&session_id).cloned()
                .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?
        };
        
        if let Some(error_analysis) = &session.error_analysis {
            self.intelligent_fixes.read().await
                .apply_fixes(&error_analysis.suggested_fixes, fix_ids).await
        } else {
            Err(anyhow::anyhow!("No error analysis available for session: {}", session_id))
        }
    }

    /// Get project context
    pub async fn get_project_context(&self, project_path: &Path) -> Option<ProjectContext> {
        let contexts = self.project_contexts.read().await;
        contexts.get(project_path).cloned()
    }

    /// Update project context with learning data
    pub async fn update_project_context(&self, project_path: PathBuf, context: ProjectContext) {
        let mut contexts = self.project_contexts.write().await;
        contexts.insert(project_path, context);
    }

    /// Learn from execution for future improvements
    async fn learn_from_execution(&self, session: &ExecutionSession) -> Result<()> {
        // Extract learning patterns from the execution
        // Update project context
        // Improve error detection and fixing capabilities
        Ok(())
    }

    /// Get system health and metrics
    pub async fn get_system_health(&self) -> SystemHealth {
        let active_executions = self.active_executions.read().await;
        let project_contexts = self.project_contexts.read().await;
        
        SystemHealth {
            system_id: self.id,
            active_sessions: active_executions.len(),
            total_projects: project_contexts.len(),
            total_executions_completed: 0, // Would be tracked
            success_rate_percent: 0.0, // Would be calculated
            average_execution_time_ms: 0.0, // Would be calculated
            error_resolution_rate_percent: 0.0, // Would be calculated
            learning_progress_score: 0.0, // Would be calculated
        }
    }
}

/// Request for code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionRequest {
    pub project_path: PathBuf,
    pub language: String,
    pub execution_type: ExecutionType,
    pub code_files: Vec<CodeFile>,
    pub execution_options: ExecutionOptions,
    pub analysis_options: AnalysisOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOptions {
    pub timeout_ms: Option<u64>,
    pub memory_limit_mb: Option<u64>,
    pub enable_profiling: bool,
    pub enable_security_scan: bool,
    pub auto_fix_errors: bool,
    pub test_coverage_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisOptions {
    pub static_analysis: bool,
    pub dynamic_analysis: bool,
    pub dependency_analysis: bool,
    pub performance_analysis: bool,
    pub security_analysis: bool,
    pub quality_analysis: bool,
}

/// Codebase analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseAnalysis {
    pub project_structure: ProjectStructure,
    pub code_quality: CodeQuality,
    pub dependencies: DependencyAnalysis,
    pub patterns: Vec<CodePattern>,
    pub technical_debt: TechnicalDebtAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStructure {
    pub total_files: u32,
    pub lines_of_code: u32,
    pub languages: HashMap<String, u32>,
    pub module_dependencies: Vec<ModuleDependency>,
    pub architecture_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDependency {
    pub from_module: String,
    pub to_module: String,
    pub dependency_type: String,
    pub coupling_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQuality {
    pub overall_score: f64,
    pub maintainability_index: f64,
    pub cyclomatic_complexity: f64,
    pub code_duplication_percent: f64,
    pub test_coverage_percent: f64,
    pub documentation_coverage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyAnalysis {
    pub total_dependencies: u32,
    pub outdated_dependencies: Vec<OutdatedDependency>,
    pub security_vulnerabilities: Vec<SecurityIssue>,
    pub dependency_tree_depth: u32,
    pub circular_dependencies: Vec<CircularDependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutdatedDependency {
    pub name: String,
    pub current_version: String,
    pub latest_version: String,
    pub security_issues: Vec<SecurityIssue>,
    pub breaking_changes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependency {
    pub dependency_cycle: Vec<String>,
    pub severity: ImpactLevel,
    pub suggested_resolution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePattern {
    pub pattern_name: String,
    pub pattern_type: PatternType,
    pub occurrences: u32,
    pub quality_impact: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    DesignPattern,
    AntiPattern,
    CodeSmell,
    SecurityPattern,
    PerformancePattern,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalDebtAnalysis {
    pub total_debt_minutes: u32,
    pub debt_by_category: HashMap<String, u32>,
    pub high_debt_files: Vec<DebtFile>,
    pub debt_trend: DebtTrend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebtFile {
    pub file_path: PathBuf,
    pub debt_minutes: u32,
    pub debt_ratio: f64,
    pub primary_issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebtTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Fix application results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixApplicationResults {
    pub applied_fixes: Vec<AppliedFix>,
    pub failed_fixes: Vec<FailedFix>,
    pub overall_success_rate: f64,
    pub code_changes_made: Vec<CodeChange>,
    pub verification_results: Option<VerificationResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedFix {
    pub fix_id: Uuid,
    pub fix_description: String,
    pub application_time_ms: u64,
    pub success: bool,
    pub side_effects_observed: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedFix {
    pub fix_id: Uuid,
    pub fix_description: String,
    pub failure_reason: String,
    pub error_details: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResults {
    pub compilation_successful: bool,
    pub tests_passed: bool,
    pub no_new_errors: bool,
    pub performance_maintained: bool,
    pub security_maintained: bool,
}

/// System health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub system_id: Uuid,
    pub active_sessions: usize,
    pub total_projects: usize,
    pub total_executions_completed: u64,
    pub success_rate_percent: f64,
    pub average_execution_time_ms: f64,
    pub error_resolution_rate_percent: f64,
    pub learning_progress_score: f64,
} 