use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Main configuration for the Kwasa-Kwasa system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KwasaConfig {
    /// Core system settings
    pub system: SystemConfig,
    
    /// Processing configuration
    pub processing: ProcessingConfig,
    
    /// User interface settings
    pub ui: UIConfig,
    
    /// Storage and persistence settings
    pub storage: StorageConfig,
    
    /// Performance and resource settings
    pub performance: PerformanceConfig,
    
    /// Domain-specific configurations
    pub domains: DomainConfigs,
    
    /// Feature flags
    pub features: FeatureFlags,
    
    /// Debugging and development settings
    pub debug: DebugConfig,
}

/// Core system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System name and version
    pub name: String,
    pub version: String,
    
    /// Installation directory
    pub install_dir: PathBuf,
    
    /// Configuration directory
    pub config_dir: PathBuf,
    
    /// Data directory
    pub data_dir: PathBuf,
    
    /// Log directory
    pub log_dir: PathBuf,
    
    /// Temporary directory
    pub temp_dir: PathBuf,
    
    /// Default language/locale
    pub locale: String,
    
    /// Timezone
    pub timezone: String,
    
    /// Auto-save interval (seconds)
    pub auto_save_interval: u64,
    
    /// Session timeout (seconds)
    pub session_timeout: u64,
}

/// Processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Text processing settings
    pub text: TextProcessingConfig,
    
    /// Genomic processing settings
    pub genomic: GenomicProcessingConfig,
    
    /// Stream processing settings
    pub stream: StreamProcessingConfig,
    
    /// Analysis configuration
    pub analysis: AnalysisConfig,
    
    /// Orchestrator settings
    pub orchestrator: OrchestratorConfig,
}

/// Text processing specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    /// Maximum text unit size (characters)
    pub max_unit_size: usize,
    
    /// Boundary detection settings
    pub boundary_detection: BoundaryDetectionConfig,
    
    /// Language processing settings
    pub language: LanguageConfig,
    
    /// Quality assessment settings
    pub quality: QualityConfig,
    
    /// Tokenization settings
    pub tokenization: TokenizationConfig,
}

/// Boundary detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDetectionConfig {
    /// Enable document boundary detection
    pub enable_document: bool,
    
    /// Enable section boundary detection
    pub enable_sections: bool,
    
    /// Enable paragraph boundary detection
    pub enable_paragraphs: bool,
    
    /// Enable sentence boundary detection
    pub enable_sentences: bool,
    
    /// Enable word boundary detection
    pub enable_words: bool,
    
    /// Enable character boundary detection
    pub enable_characters: bool,
    
    /// Enable semantic boundary detection
    pub enable_semantic: bool,
    
    /// Minimum semantic unit size
    pub min_semantic_size: usize,
    
    /// Custom boundary patterns
    pub custom_patterns: Vec<String>,
}

/// Language processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    /// Default language
    pub default_language: String,
    
    /// Supported languages
    pub supported_languages: Vec<String>,
    
    /// Auto-detect language
    pub auto_detect: bool,
    
    /// Stemming configuration
    pub stemming: StemmingConfig,
    
    /// Stopword lists
    pub stopwords: StopwordConfig,
    
    /// N-gram configuration
    pub ngrams: NgramConfig,
}

/// Stemming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StemmingConfig {
    /// Enable stemming
    pub enabled: bool,
    
    /// Stemming algorithm
    pub algorithm: String,
    
    /// Language-specific stemming rules
    pub language_rules: HashMap<String, String>,
}

/// Stopword configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopwordConfig {
    /// Enable stopword filtering
    pub enabled: bool,
    
    /// Use built-in stopword lists
    pub use_builtin: bool,
    
    /// Custom stopword lists by language
    pub custom_lists: HashMap<String, Vec<String>>,
    
    /// Additional stopwords to add
    pub additional_words: Vec<String>,
}

/// N-gram configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramConfig {
    /// Enable n-gram extraction
    pub enabled: bool,
    
    /// Minimum n-gram size
    pub min_size: usize,
    
    /// Maximum n-gram size
    pub max_size: usize,
    
    /// Minimum frequency threshold
    pub min_frequency: usize,
}

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Enable readability assessment
    pub enable_readability: bool,
    
    /// Enable coherence assessment
    pub enable_coherence: bool,
    
    /// Enable grammar checking
    pub enable_grammar: bool,
    
    /// Enable style analysis
    pub enable_style: bool,
    
    /// Quality thresholds
    pub thresholds: QualityThresholds,
}

/// Quality assessment thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum readability score (0.0-1.0)
    pub min_readability: f64,
    
    /// Minimum coherence score (0.0-1.0)
    pub min_coherence: f64,
    
    /// Maximum grammar error rate (0.0-1.0)
    pub max_grammar_errors: f64,
    
    /// Style consistency threshold (0.0-1.0)
    pub style_consistency: f64,
}

/// Tokenization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationConfig {
    /// Tokenization strategy
    pub strategy: String,
    
    /// Preserve whitespace
    pub preserve_whitespace: bool,
    
    /// Handle contractions
    pub handle_contractions: bool,
    
    /// Split on punctuation
    pub split_punctuation: bool,
    
    /// Custom tokenization patterns
    pub custom_patterns: Vec<String>,
}

/// Genomic processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicProcessingConfig {
    /// Enable genomic analysis
    pub enabled: bool,
    
    /// Sequence processing settings
    pub sequence: SequenceConfig,
    
    /// Motif analysis settings
    pub motif: MotifConfig,
    
    /// High-throughput processing settings
    pub high_throughput: HighThroughputConfig,
}

/// Sequence processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceConfig {
    /// Maximum sequence length
    pub max_length: usize,
    
    /// Minimum sequence length
    pub min_length: usize,
    
    /// Quality score threshold
    pub quality_threshold: f64,
    
    /// Enable sequence validation
    pub validate_sequences: bool,
    
    /// Allowed nucleotides
    pub allowed_nucleotides: Vec<char>,
}

/// Motif analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifConfig {
    /// Minimum motif length
    pub min_length: usize,
    
    /// Maximum motif length
    pub max_length: usize,
    
    /// Significance threshold
    pub significance_threshold: f64,
    
    /// Enable PWM (Position Weight Matrix) analysis
    pub enable_pwm: bool,
    
    /// Conservation threshold
    pub conservation_threshold: f64,
}

/// High-throughput processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighThroughputConfig {
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Memory limit for batches (MB)
    pub memory_limit_mb: usize,
    
    /// Enable caching
    pub enable_caching: bool,
}

/// Stream processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessingConfig {
    /// Default buffer size
    pub buffer_size: usize,
    
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
    
    /// Stream timeout (milliseconds)
    pub stream_timeout_ms: u64,
    
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    
    /// Retry configuration
    pub retry: RetryConfig,
    
    /// Metrics collection
    pub metrics: MetricsConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Enable retries
    pub enabled: bool,
    
    /// Maximum retry attempts
    pub max_attempts: u32,
    
    /// Initial retry delay (milliseconds)
    pub initial_delay_ms: u64,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum retry delay (milliseconds)
    pub max_delay_ms: u64,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics reporting interval (seconds)
    pub reporting_interval: u64,
    
    /// Metrics storage backend
    pub storage_backend: String,
    
    /// Enable performance monitoring
    pub enable_performance: bool,
    
    /// Enable error tracking
    pub enable_error_tracking: bool,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Statistical analysis settings
    pub statistics: StatisticsConfig,
    
    /// Pattern recognition settings
    pub patterns: PatternConfig,
    
    /// Similarity analysis settings
    pub similarity: SimilarityConfig,
    
    /// Clustering settings
    pub clustering: ClusteringConfig,
}

/// Statistical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsConfig {
    /// Enable statistical analysis
    pub enabled: bool,
    
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    
    /// Sample size requirements
    pub min_sample_size: usize,
    
    /// Enable correlation analysis
    pub enable_correlation: bool,
    
    /// Enable distribution analysis
    pub enable_distribution: bool,
}

/// Pattern recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Enable pattern recognition
    pub enabled: bool,
    
    /// Pattern matching algorithms
    pub algorithms: Vec<String>,
    
    /// Minimum pattern frequency
    pub min_frequency: usize,
    
    /// Pattern significance threshold
    pub significance_threshold: f64,
    
    /// Enable fuzzy matching
    pub enable_fuzzy: bool,
}

/// Similarity analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Default similarity metric
    pub default_metric: String,
    
    /// Available similarity metrics
    pub available_metrics: Vec<String>,
    
    /// Similarity threshold
    pub threshold: f64,
    
    /// Enable semantic similarity
    pub enable_semantic: bool,
    
    /// Enable structural similarity
    pub enable_structural: bool,
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Default clustering algorithm
    pub default_algorithm: String,
    
    /// Available clustering algorithms
    pub available_algorithms: Vec<String>,
    
    /// Default number of clusters
    pub default_clusters: usize,
    
    /// Enable hierarchical clustering
    pub enable_hierarchical: bool,
    
    /// Distance metric for clustering
    pub distance_metric: String,
}

/// Orchestrator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// Metacognitive processing settings
    pub metacognitive: MetacognitiveConfig,
    
    /// Biomimetic processing settings
    pub biomimetic: BiomimeticConfig,
    
    /// Goal management settings
    pub goals: GoalConfig,
    
    /// Context management settings
    pub context: ContextConfig,
    
    /// Intervention settings
    pub intervention: InterventionConfig,
}

/// Metacognitive processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveConfig {
    /// Enable metacognitive processing
    pub enabled: bool,
    
    /// Number of processing layers
    pub num_layers: usize,
    
    /// Layer processing timeouts (milliseconds)
    pub layer_timeouts: Vec<u64>,
    
    /// Enable self-reflection
    pub enable_reflection: bool,
    
    /// Reflection interval (seconds)
    pub reflection_interval: u64,
}

/// Biomimetic processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiomimeticConfig {
    /// Enable biomimetic processing
    pub enabled: bool,
    
    /// Enable memory consolidation
    pub enable_memory_consolidation: bool,
    
    /// Enable dreaming module
    pub enable_dreaming: bool,
    
    /// Attention span settings
    pub attention: AttentionConfig,
    
    /// Learning rate settings
    pub learning: LearningConfig,
}

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Base attention span (seconds)
    pub base_span: u64,
    
    /// Attention decay rate
    pub decay_rate: f64,
    
    /// Factors that influence attention
    pub influence_factors: Vec<String>,
    
    /// Enable attention switching
    pub enable_switching: bool,
}

/// Learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Base learning rate
    pub base_rate: f64,
    
    /// Learning rate decay
    pub rate_decay: f64,
    
    /// Enable adaptive learning
    pub enable_adaptive: bool,
    
    /// Memory consolidation interval (hours)
    pub consolidation_interval: u64,
}

/// Goal management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalConfig {
    /// Enable goal tracking
    pub enabled: bool,
    
    /// Default goal timeout (hours)
    pub default_timeout: u64,
    
    /// Progress update interval (minutes)
    pub progress_interval: u64,
    
    /// Enable automatic sub-goal creation
    pub auto_subgoals: bool,
    
    /// Goal priority weights
    pub priority_weights: HashMap<String, f64>,
}

/// Context management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    /// Maximum context history size
    pub max_history_size: usize,
    
    /// Context persistence interval (minutes)
    pub persistence_interval: u64,
    
    /// Enable context prediction
    pub enable_prediction: bool,
    
    /// Context similarity threshold
    pub similarity_threshold: f64,
    
    /// Enable context sharing
    pub enable_sharing: bool,
}

/// Intervention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionConfig {
    /// Enable interventions
    pub enabled: bool,
    
    /// Default intervention timeout (minutes)
    pub default_timeout: u64,
    
    /// Maximum interventions per hour
    pub max_per_hour: u32,
    
    /// Intervention sensitivity (0.0-1.0)
    pub sensitivity: f64,
    
    /// Enable adaptive interventions
    pub enable_adaptive: bool,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// Interface theme
    pub theme: String,
    
    /// Font settings
    pub font: FontConfig,
    
    /// Layout settings
    pub layout: LayoutConfig,
    
    /// Accessibility settings
    pub accessibility: AccessibilityConfig,
    
    /// Notification settings
    pub notifications: NotificationConfig,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    /// Default font family
    pub family: String,
    
    /// Default font size
    pub size: u32,
    
    /// Font weight
    pub weight: String,
    
    /// Line height
    pub line_height: f64,
    
    /// Letter spacing
    pub letter_spacing: f64,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Default window width
    pub window_width: u32,
    
    /// Default window height
    pub window_height: u32,
    
    /// Panel configurations
    pub panels: HashMap<String, PanelConfig>,
    
    /// Enable responsive layout
    pub responsive: bool,
}

/// Panel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelConfig {
    /// Panel visibility
    pub visible: bool,
    
    /// Panel width (percentage or pixels)
    pub width: String,
    
    /// Panel height (percentage or pixels)
    pub height: String,
    
    /// Panel position
    pub position: String,
    
    /// Panel docking
    pub dockable: bool,
}

/// Accessibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessibilityConfig {
    /// Enable screen reader support
    pub screen_reader: bool,
    
    /// High contrast mode
    pub high_contrast: bool,
    
    /// Large text mode
    pub large_text: bool,
    
    /// Keyboard navigation
    pub keyboard_nav: bool,
    
    /// Voice commands
    pub voice_commands: bool,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Enable notifications
    pub enabled: bool,
    
    /// Notification types to show
    pub types: Vec<String>,
    
    /// Notification timeout (seconds)
    pub timeout: u64,
    
    /// Sound settings
    pub sound: SoundConfig,
    
    /// Visual settings
    pub visual: VisualNotificationConfig,
}

/// Sound configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundConfig {
    /// Enable sound notifications
    pub enabled: bool,
    
    /// Volume level (0.0-1.0)
    pub volume: f64,
    
    /// Sound theme
    pub theme: String,
    
    /// Custom sound files
    pub custom_sounds: HashMap<String, PathBuf>,
}

/// Visual notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualNotificationConfig {
    /// Enable visual notifications
    pub enabled: bool,
    
    /// Animation duration (milliseconds)
    pub animation_duration: u64,
    
    /// Notification position
    pub position: String,
    
    /// Maximum notifications to show
    pub max_notifications: u32,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// File storage configuration
    pub files: FileStorageConfig,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Backup configuration
    pub backup: BackupConfig,
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database type
    pub db_type: String,
    
    /// Connection string
    pub connection_string: String,
    
    /// Connection pool size
    pub pool_size: u32,
    
    /// Connection timeout (seconds)
    pub connection_timeout: u64,
    
    /// Query timeout (seconds)
    pub query_timeout: u64,
    
    /// Enable migrations
    pub enable_migrations: bool,
}

/// File storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStorageConfig {
    /// Root storage directory
    pub root_dir: PathBuf,
    
    /// File naming convention
    pub naming_convention: String,
    
    /// Maximum file size (MB)
    pub max_file_size: usize,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Compression algorithm
    pub compression_algorithm: String,
    
    /// Enable encryption
    pub enable_encryption: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache type
    pub cache_type: String,
    
    /// Maximum cache size (MB)
    pub max_size_mb: usize,
    
    /// Cache TTL (seconds)
    pub ttl_seconds: u64,
    
    /// Enable LRU eviction
    pub enable_lru: bool,
    
    /// Cache warming strategies
    pub warming_strategies: Vec<String>,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    
    /// Backup interval (hours)
    pub interval_hours: u64,
    
    /// Backup retention (days)
    pub retention_days: u64,
    
    /// Backup directory
    pub backup_dir: PathBuf,
    
    /// Backup compression
    pub compression: bool,
    
    /// Remote backup settings
    pub remote: Option<RemoteBackupConfig>,
}

/// Remote backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteBackupConfig {
    /// Remote backup provider
    pub provider: String,
    
    /// Connection settings
    pub connection: HashMap<String, String>,
    
    /// Encryption settings
    pub encryption: HashMap<String, String>,
    
    /// Synchronization interval (hours)
    pub sync_interval: u64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Memory settings
    pub memory: MemoryConfig,
    
    /// CPU settings
    pub cpu: CpuConfig,
    
    /// I/O settings
    pub io: IoConfig,
    
    /// Network settings
    pub network: NetworkConfig,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum heap size (MB)
    pub max_heap_mb: usize,
    
    /// Garbage collection settings
    pub gc_settings: HashMap<String, String>,
    
    /// Memory pool configurations
    pub pools: HashMap<String, PoolConfig>,
    
    /// Enable memory monitoring
    pub enable_monitoring: bool,
}

/// Pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    
    /// Maximum pool size
    pub max_size: usize,
    
    /// Pool growth strategy
    pub growth_strategy: String,
    
    /// Pool cleanup interval (seconds)
    pub cleanup_interval: u64,
}

/// CPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Thread pool configurations
    pub thread_pools: HashMap<String, ThreadPoolConfig>,
    
    /// CPU affinity settings
    pub affinity: Option<Vec<usize>>,
    
    /// Enable NUMA optimization
    pub enable_numa: bool,
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Core pool size
    pub core_size: usize,
    
    /// Maximum pool size
    pub max_size: usize,
    
    /// Keep alive time (seconds)
    pub keep_alive: u64,
    
    /// Queue capacity
    pub queue_capacity: usize,
    
    /// Thread naming pattern
    pub name_pattern: String,
}

/// I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    /// Buffer sizes
    pub buffer_sizes: HashMap<String, usize>,
    
    /// I/O timeout (milliseconds)
    pub timeout_ms: u64,
    
    /// Enable async I/O
    pub enable_async: bool,
    
    /// File system settings
    pub filesystem: FilesystemConfig,
}

/// Filesystem configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemConfig {
    /// Enable file watching
    pub enable_watching: bool,
    
    /// File watch patterns
    pub watch_patterns: Vec<String>,
    
    /// Temporary file cleanup interval (minutes)
    pub temp_cleanup_interval: u64,
    
    /// Enable file caching
    pub enable_caching: bool,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Connection timeout (seconds)
    pub connection_timeout: u64,
    
    /// Read timeout (seconds)
    pub read_timeout: u64,
    
    /// Write timeout (seconds)
    pub write_timeout: u64,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
    
    /// Enable keep-alive
    pub enable_keep_alive: bool,
    
    /// Proxy settings
    pub proxy: Option<ProxyConfig>,
}

/// Proxy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    /// Proxy type
    pub proxy_type: String,
    
    /// Proxy host
    pub host: String,
    
    /// Proxy port
    pub port: u16,
    
    /// Authentication
    pub auth: Option<ProxyAuth>,
}

/// Proxy authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyAuth {
    /// Username
    pub username: String,
    
    /// Password
    pub password: String,
}

/// Domain-specific configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfigs {
    /// Text analysis domain
    pub text_analysis: HashMap<String, String>,
    
    /// Genomics domain
    pub genomics: HashMap<String, String>,
    
    /// Chemistry domain
    pub chemistry: HashMap<String, String>,
    
    /// Spectrometry domain
    pub spectrometry: HashMap<String, String>,
    
    /// Custom domains
    pub custom: HashMap<String, HashMap<String, String>>,
}

/// Feature flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Experimental features
    pub experimental: HashMap<String, bool>,
    
    /// Beta features
    pub beta: HashMap<String, bool>,
    
    /// Development features
    pub development: HashMap<String, bool>,
    
    /// A/B testing features
    pub ab_testing: HashMap<String, String>,
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable debug mode
    pub enabled: bool,
    
    /// Log level
    pub log_level: String,
    
    /// Log output destinations
    pub log_outputs: Vec<String>,
    
    /// Enable profiling
    pub enable_profiling: bool,
    
    /// Profiling configuration
    pub profiling: ProfilingConfig,
    
    /// Enable tracing
    pub enable_tracing: bool,
    
    /// Tracing configuration
    pub tracing: TracingConfig,
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Profiling output directory
    pub output_dir: PathBuf,
    
    /// Profiling interval (seconds)
    pub interval: u64,
    
    /// Enable CPU profiling
    pub enable_cpu: bool,
    
    /// Enable memory profiling
    pub enable_memory: bool,
    
    /// Enable I/O profiling
    pub enable_io: bool,
}

/// Tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Tracing output directory
    pub output_dir: PathBuf,
    
    /// Trace buffer size
    pub buffer_size: usize,
    
    /// Enable distributed tracing
    pub enable_distributed: bool,
    
    /// Tracing endpoint
    pub endpoint: Option<String>,
    
    /// Sampling rate (0.0-1.0)
    pub sampling_rate: f64,
}

impl Default for KwasaConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            processing: ProcessingConfig::default(),
            ui: UIConfig::default(),
            storage: StorageConfig::default(),
            performance: PerformanceConfig::default(),
            domains: DomainConfigs::default(),
            features: FeatureFlags::default(),
            debug: DebugConfig::default(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            name: "Kwasa-Kwasa".to_string(),
            version: "0.1.0".to_string(),
            install_dir: PathBuf::from("/opt/kwasa-kwasa"),
            config_dir: PathBuf::from("~/.config/kwasa-kwasa"),
            data_dir: PathBuf::from("~/.local/share/kwasa-kwasa"),
            log_dir: PathBuf::from("~/.local/share/kwasa-kwasa/logs"),
            temp_dir: PathBuf::from("/tmp/kwasa-kwasa"),
            locale: "en_US.UTF-8".to_string(),
            timezone: "UTC".to_string(),
            auto_save_interval: 300, // 5 minutes
            session_timeout: 3600,   // 1 hour
        }
    }
}

// Implement Default for all other config structs...
impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            text: TextProcessingConfig::default(),
            genomic: GenomicProcessingConfig::default(),
            stream: StreamProcessingConfig::default(),
            analysis: AnalysisConfig::default(),
            orchestrator: OrchestratorConfig::default(),
        }
    }
}

impl Default for TextProcessingConfig {
    fn default() -> Self {
        Self {
            max_unit_size: 10000, // 10KB
            boundary_detection: BoundaryDetectionConfig::default(),
            language: LanguageConfig::default(),
            quality: QualityConfig::default(),
            tokenization: TokenizationConfig::default(),
        }
    }
}

impl Default for BoundaryDetectionConfig {
    fn default() -> Self {
        Self {
            enable_document: true,
            enable_sections: true,
            enable_paragraphs: true,
            enable_sentences: true,
            enable_words: true,
            enable_characters: false,
            enable_semantic: true,
            min_semantic_size: 100,
            custom_patterns: Vec::new(),
        }
    }
}

impl Default for LanguageConfig {
    fn default() -> Self {
        Self {
            default_language: "en".to_string(),
            supported_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
            auto_detect: true,
            stemming: StemmingConfig::default(),
            stopwords: StopwordConfig::default(),
            ngrams: NgramConfig::default(),
        }
    }
}

// Continue with other Default implementations...
// (I'll provide a few key ones to demonstrate the pattern)

impl Default for GenomicProcessingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sequence: SequenceConfig::default(),
            motif: MotifConfig::default(),
            high_throughput: HighThroughputConfig::default(),
        }
    }
}

impl Default for StreamProcessingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            max_concurrent_streams: 100,
            stream_timeout_ms: 30000, // 30 seconds
            enable_backpressure: true,
            retry: RetryConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            memory: MemoryConfig::default(),
            cpu: CpuConfig::default(),
            io: IoConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_heap_mb: 2048, // 2GB
            gc_settings: HashMap::new(),
            pools: HashMap::new(),
            enable_monitoring: true,
        }
    }
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            thread_pools: HashMap::new(),
            affinity: None,
            enable_numa: false,
        }
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            theme: "default".to_string(),
            font: FontConfig::default(),
            layout: LayoutConfig::default(),
            accessibility: AccessibilityConfig::default(),
            notifications: NotificationConfig::default(),
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            files: FileStorageConfig::default(),
            cache: CacheConfig::default(),
            backup: BackupConfig::default(),
        }
    }
}

impl Default for DomainConfigs {
    fn default() -> Self {
        Self {
            text_analysis: HashMap::new(),
            genomics: HashMap::new(),
            chemistry: HashMap::new(),
            spectrometry: HashMap::new(),
            custom: HashMap::new(),
        }
    }
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            experimental: HashMap::new(),
            beta: HashMap::new(),
            development: HashMap::new(),
            ab_testing: HashMap::new(),
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            log_level: "info".to_string(),
            log_outputs: vec!["console".to_string(), "file".to_string()],
            enable_profiling: false,
            profiling: ProfilingConfig::default(),
            enable_tracing: false,
            tracing: TracingConfig::default(),
        }
    }
}

// Add more Default implementations as needed...

impl KwasaConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: KwasaConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Merge with another configuration
    pub fn merge_with(&mut self, other: &KwasaConfig) {
        // Implement configuration merging logic
        // This would selectively override values from other config
        // For now, a simple placeholder
        if other.debug.enabled {
            self.debug.enabled = true;
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        
        // Validate system paths
        if !self.system.install_dir.exists() {
            errors.push("Install directory does not exist".to_string());
        }
        
        // Validate memory settings
        if self.performance.memory.max_heap_mb < 128 {
            errors.push("Maximum heap size too small (minimum 128MB)".to_string());
        }
        
        // Validate thread counts
        if self.performance.cpu.worker_threads == 0 {
            errors.push("Worker thread count cannot be zero".to_string());
        }
        
        // Add more validation rules as needed
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    /// Get configuration as environment variables
    pub fn as_env_vars(&self) -> HashMap<String, String> {
        let mut env_vars = HashMap::new();
        
        env_vars.insert("KWASA_LOG_LEVEL".to_string(), self.debug.log_level.clone());
        env_vars.insert("KWASA_WORKER_THREADS".to_string(), self.performance.cpu.worker_threads.to_string());
        env_vars.insert("KWASA_MAX_HEAP_MB".to_string(), self.performance.memory.max_heap_mb.to_string());
        
        // Add more environment variable mappings as needed
        
        env_vars
    }
}

// Add more Default implementations for missing structs...

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enable_readability: true,
            enable_coherence: true,
            enable_grammar: true,
            enable_style: true,
            thresholds: QualityThresholds::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_readability: 0.6,
            min_coherence: 0.7,
            max_grammar_errors: 0.05,
            style_consistency: 0.8,
        }
    }
}

impl Default for TokenizationConfig {
    fn default() -> Self {
        Self {
            strategy: "word_boundary".to_string(),
            preserve_whitespace: false,
            handle_contractions: true,
            split_punctuation: true,
            custom_patterns: Vec::new(),
        }
    }
}

impl Default for StemmingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "porter".to_string(),
            language_rules: HashMap::new(),
        }
    }
}

impl Default for StopwordConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_builtin: true,
            custom_lists: HashMap::new(),
            additional_words: Vec::new(),
        }
    }
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_size: 1,
            max_size: 3,
            min_frequency: 2,
        }
    }
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            max_length: 10000,
            min_length: 10,
            quality_threshold: 0.8,
            validate_sequences: true,
            allowed_nucleotides: vec!['A', 'T', 'G', 'C', 'N'],
        }
    }
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            min_length: 6,
            max_length: 20,
            significance_threshold: 0.05,
            enable_pwm: true,
            conservation_threshold: 0.8,
        }
    }
}

impl Default for HighThroughputConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            worker_threads: num_cpus::get(),
            batch_size: 1000,
            memory_limit_mb: 512,
            enable_caching: true,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30000,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reporting_interval: 60,
            storage_backend: "memory".to_string(),
            enable_performance: true,
            enable_error_tracking: true,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            statistics: StatisticsConfig::default(),
            patterns: PatternConfig::default(),
            similarity: SimilarityConfig::default(),
            clustering: ClusteringConfig::default(),
        }
    }
}

impl Default for StatisticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            confidence_level: 0.95,
            min_sample_size: 30,
            enable_correlation: true,
            enable_distribution: true,
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec!["regex".to_string(), "fuzzy".to_string()],
            min_frequency: 3,
            significance_threshold: 0.05,
            enable_fuzzy: true,
        }
    }
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            default_metric: "cosine".to_string(),
            available_metrics: vec!["cosine".to_string(), "jaccard".to_string(), "euclidean".to_string()],
            threshold: 0.7,
            enable_semantic: true,
            enable_structural: true,
        }
    }
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            default_algorithm: "kmeans".to_string(),
            available_algorithms: vec!["kmeans".to_string(), "hierarchical".to_string(), "dbscan".to_string()],
            default_clusters: 5,
            enable_hierarchical: true,
            distance_metric: "euclidean".to_string(),
        }
    }
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            metacognitive: MetacognitiveConfig::default(),
            biomimetic: BiomimeticConfig::default(),
            goals: GoalConfig::default(),
            context: ContextConfig::default(),
            intervention: InterventionConfig::default(),
        }
    }
}

impl Default for MetacognitiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            num_layers: 3,
            layer_timeouts: vec![1000, 2000, 5000], // milliseconds
            enable_reflection: true,
            reflection_interval: 300, // 5 minutes
        }
    }
}

impl Default for BiomimeticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_memory_consolidation: true,
            enable_dreaming: false,
            attention: AttentionConfig::default(),
            learning: LearningConfig::default(),
        }
    }
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            base_span: 1800, // 30 minutes
            decay_rate: 0.1,
            influence_factors: vec!["relevance".to_string(), "novelty".to_string(), "importance".to_string()],
            enable_switching: true,
        }
    }
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            base_rate: 0.01,
            rate_decay: 0.99,
            enable_adaptive: true,
            consolidation_interval: 6, // 6 hours
        }
    }
}

impl Default for GoalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_timeout: 24, // 24 hours
            progress_interval: 15, // 15 minutes
            auto_subgoals: true,
            priority_weights: HashMap::new(),
        }
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            persistence_interval: 30, // 30 minutes
            enable_prediction: true,
            similarity_threshold: 0.7,
            enable_sharing: false,
        }
    }
}

impl Default for InterventionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_timeout: 10, // 10 minutes
            max_per_hour: 5,
            sensitivity: 0.6,
            enable_adaptive: true,
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Inter".to_string(),
            size: 14,
            weight: "normal".to_string(),
            line_height: 1.4,
            letter_spacing: 0.0,
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            window_width: 1200,
            window_height: 800,
            panels: HashMap::new(),
            responsive: true,
        }
    }
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            screen_reader: false,
            high_contrast: false,
            large_text: false,
            keyboard_nav: true,
            voice_commands: false,
        }
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            types: vec!["info".to_string(), "warning".to_string(), "error".to_string()],
            timeout: 5,
            sound: SoundConfig::default(),
            visual: VisualNotificationConfig::default(),
        }
    }
}

impl Default for SoundConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            volume: 0.5,
            theme: "default".to_string(),
            custom_sounds: HashMap::new(),
        }
    }
}

impl Default for VisualNotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            animation_duration: 300,
            position: "top-right".to_string(),
            max_notifications: 5,
        }
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_type: "sqlite".to_string(),
            connection_string: "kwasa_kwasa.db".to_string(),
            pool_size: 10,
            connection_timeout: 30,
            query_timeout: 60,
            enable_migrations: true,
        }
    }
}

impl Default for FileStorageConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("./data"),
            naming_convention: "timestamp".to_string(),
            max_file_size: 100, // 100MB
            enable_compression: true,
            compression_algorithm: "gzip".to_string(),
            enable_encryption: false,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_type: "memory".to_string(),
            max_size_mb: 256,
            ttl_seconds: 3600, // 1 hour
            enable_lru: true,
            warming_strategies: vec!["preload".to_string()],
        }
    }
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_hours: 24,
            retention_days: 30,
            backup_dir: PathBuf::from("./backups"),
            compression: true,
            remote: None,
        }
    }
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            buffer_sizes: HashMap::new(),
            timeout_ms: 5000,
            enable_async: true,
            filesystem: FilesystemConfig::default(),
        }
    }
}

impl Default for FilesystemConfig {
    fn default() -> Self {
        Self {
            enable_watching: true,
            watch_patterns: vec!["*.toml".to_string(), "*.txt".to_string()],
            temp_cleanup_interval: 60,
            enable_caching: true,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            connection_timeout: 30,
            read_timeout: 60,
            write_timeout: 60,
            max_connections: 100,
            enable_keep_alive: true,
            proxy: None,
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./profile"),
            interval: 60,
            enable_cpu: true,
            enable_memory: true,
            enable_io: false,
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./traces"),
            buffer_size: 1024,
            enable_distributed: false,
            endpoint: None,
            sampling_rate: 0.1,
        }
    }
} 