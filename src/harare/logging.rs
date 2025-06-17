use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter, BufRead, BufReader};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use super::{Task, ResourceMetrics, execution_planner::ExecutionPlan};

/// The main Harare logger that manages .hre files
pub struct HarareLogger {
    current_session_id: Uuid,
    log_file_path: PathBuf,
    writer: Option<BufWriter<File>>,
    session_start: SystemTime,
    entry_count: u64,
    buffered_entries: Vec<HreEntry>,
    flush_interval_ms: u64,
}

/// HRE log entry - the fundamental unit of Harare logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HreEntry {
    pub entry_id: Uuid,
    pub session_id: Uuid,
    pub timestamp: SystemTime,
    pub entry_type: HreEntryType,
    pub content: HreContent,
    pub metadata: HashMap<String, String>,
    pub context_snapshot: Option<ContextSnapshot>,
}

/// Types of HRE log entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HreEntryType {
    SessionStart,
    SessionEnd,
    DecisionMade,
    TaskCreated,
    TaskStarted,
    TaskCompleted,
    TaskFailed,
    ModuleInvoked,
    ModuleCompleted,
    ModuleError,
    UserInteraction,
    SystemAlert,
    ResourceThreshold,
    LearningEvent,
    AdaptationTrigger,
    ContextChange,
    ErrorRecovery,
    PerformanceMetric,
}

/// Content of HRE log entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HreContent {
    SessionInfo {
        orchestrator_id: Uuid,
        project_root: String,
        configuration: HashMap<String, serde_json::Value>,
    },
    Decision {
        decision_id: Uuid,
        decision_type: String,
        rationale: String,
        confidence: f64,
        alternatives: Vec<String>,
        expected_outcome: String,
    },
    TaskEvent {
        task_id: Uuid,
        task_name: String,
        task_type: String,
        details: HashMap<String, serde_json::Value>,
    },
    ModuleEvent {
        module_name: String,
        operation: String,
        input_summary: String,
        output_summary: Option<String>,
        execution_time_ms: Option<u64>,
        resource_usage: Option<ResourceMetrics>,
    },
    UserEvent {
        user_input: String,
        intent_detected: Option<String>,
        confidence: Option<f64>,
        response_generated: Option<String>,
    },
    SystemEvent {
        event_type: String,
        severity: String,
        description: String,
        affected_modules: Vec<String>,
        resolution: Option<String>,
    },
    LearningEvent {
        learning_type: String,
        objective: String,
        progress_delta: f64,
        insights_gained: Vec<String>,
        adaptations_made: Vec<String>,
    },
    MetricUpdate {
        metric_name: String,
        metric_value: f64,
        previous_value: Option<f64>,
        trend: String,
        threshold_info: Option<ThresholdInfo>,
    },
}

/// Context snapshot for understanding state at log time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub active_tasks: Vec<String>,
    pub module_states: HashMap<String, String>,
    pub resource_usage: ResourceMetrics,
    pub user_context: Option<String>,
    pub semantic_focus: Option<String>,
    pub attention_metrics: AttentionMetrics,
}

/// Attention and focus metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMetrics {
    pub primary_focus: Option<String>,
    pub focus_duration_ms: u64,
    pub context_switches: u32,
    pub cognitive_load: f64,
    pub distraction_events: u32,
}

/// Threshold information for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdInfo {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub current_status: String,
    pub time_in_current_status_ms: u64,
}

impl HarareLogger {
    /// Create a new Harare logger
    pub async fn new(log_directory: &Path) -> Result<Self> {
        let session_id = Uuid::new_v4();
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let log_file_path = log_directory.join(format!("harare_session_{}_{}.hre", timestamp, session_id));
        
        // Ensure log directory exists
        if !log_directory.exists() {
            std::fs::create_dir_all(log_directory)
                .with_context(|| format!("Failed to create log directory: {:?}", log_directory))?;
        }
        
        let mut logger = Self {
            current_session_id: session_id,
            log_file_path,
            writer: None,
            session_start: SystemTime::now(),
            entry_count: 0,
            buffered_entries: Vec::new(),
            flush_interval_ms: 1000, // Flush every second
        };
        
        // Initialize the log file and write session start
        logger.initialize_log_file().await?;
        logger.log_session_start().await?;
        
        Ok(logger)
    }

    /// Initialize the log file with header information
    async fn initialize_log_file(&mut self) -> Result<()> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.log_file_path)
            .with_context(|| format!("Failed to open log file: {:?}", self.log_file_path))?;
        
        self.writer = Some(BufWriter::new(file));
        
        // Write HRE file header
        self.write_hre_header().await?;
        
        Ok(())
    }

    /// Write HRE file header with format information
    async fn write_hre_header(&mut self) -> Result<()> {
        let header = format!(
            "# Harare Orchestrator Log (HRE)\n\
             # Session ID: {}\n\
             # Created: {}\n\
             # Format Version: 1.0\n\
             # Encoding: UTF-8\n\n",
            self.current_session_id,
            DateTime::<Utc>::from(self.session_start).to_rfc3339()
        );
        
        if let Some(writer) = &mut self.writer {
            writer.write_all(header.as_bytes())?;
            writer.flush()?;
        }
        
        Ok(())
    }

    /// Log session start
    pub async fn log_session_start(&mut self) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: self.session_start,
            entry_type: HreEntryType::SessionStart,
            content: HreContent::SessionInfo {
                orchestrator_id: self.current_session_id,
                project_root: std::env::current_dir()?.to_string_lossy().to_string(),
                configuration: HashMap::new(),
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("version".to_string(), "0.1.0".to_string());
                meta.insert("rust_version".to_string(), env!("RUSTC_VERSION").to_string());
                meta
            },
            context_snapshot: Some(ContextSnapshot {
                active_tasks: Vec::new(),
                module_states: HashMap::new(),
                resource_usage: ResourceMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0,
                    disk_usage_mb: 0,
                    network_in_mbps: 0.0,
                    network_out_mbps: 0.0,
                    gpu_usage_percent: None,
                    timestamp: SystemTime::now(),
                },
                user_context: None,
                semantic_focus: None,
                attention_metrics: AttentionMetrics {
                    primary_focus: None,
                    focus_duration_ms: 0,
                    context_switches: 0,
                    cognitive_load: 0.0,
                    distraction_events: 0,
                },
            }),
        };
        
        self.write_entry(entry).await
    }

    /// Log a decision made by the orchestrator
    pub async fn log_decision(&mut self, rationale: &str) -> Result<()> {
        let decision_id = Uuid::new_v4();
        
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::DecisionMade,
            content: HreContent::Decision {
                decision_id,
                decision_type: "orchestration".to_string(),
                rationale: rationale.to_string(),
                confidence: 0.8, // This would be calculated in real implementation
                alternatives: Vec::new(),
                expected_outcome: "successful_execution".to_string(),
            },
            metadata: HashMap::new(),
            context_snapshot: None, // Would include full context in real implementation
        };
        
        self.write_entry(entry).await
    }

    /// Log execution start
    pub async fn log_execution_start(&mut self, task_id: Uuid, plan: &ExecutionPlan) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::TaskStarted,
            content: HreContent::TaskEvent {
                task_id,
                task_name: plan.name.clone(),
                task_type: "execution".to_string(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("step_count".to_string(), serde_json::json!(plan.steps.len()));
                    details.insert("parallel_execution".to_string(), serde_json::json!(plan.parallel_execution));
                    details
                },
            },
            metadata: HashMap::new(),
            context_snapshot: None,
        };
        
        self.write_entry(entry).await
    }

    /// Log execution error
    pub async fn log_execution_error(&mut self, task_id: Uuid, module_name: &str, error: &str) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::ModuleError,
            content: HreContent::ModuleEvent {
                module_name: module_name.to_string(),
                operation: "execution".to_string(),
                input_summary: "N/A".to_string(),
                output_summary: None,
                execution_time_ms: None,
                resource_usage: None,
            },
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("task_id".to_string(), task_id.to_string());
                meta.insert("error".to_string(), error.to_string());
                meta.insert("severity".to_string(), "error".to_string());
                meta
            },
            context_snapshot: None,
        };
        
        self.write_entry(entry).await
    }

    /// Log execution completion
    pub async fn log_execution_completion(
        &mut self, 
        task_id: Uuid, 
        duration: Duration,
        results: &HashMap<String, serde_json::Value>
    ) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::TaskCompleted,
            content: HreContent::TaskEvent {
                task_id,
                task_name: "execution_plan".to_string(),
                task_type: "completion".to_string(),
                details: {
                    let mut details = HashMap::new();
                    details.insert("duration_ms".to_string(), serde_json::json!(duration.as_millis()));
                    details.insert("result_count".to_string(), serde_json::json!(results.len()));
                    details.insert("success".to_string(), serde_json::json!(true));
                    details
                },
            },
            metadata: HashMap::new(),
            context_snapshot: None,
        };
        
        self.write_entry(entry).await
    }

    /// Log user interaction
    pub async fn log_user_interaction(&mut self, user_input: &str, response: Option<&str>) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::UserInteraction,
            content: HreContent::UserEvent {
                user_input: user_input.to_string(),
                intent_detected: None, // Would be filled by intent analysis
                confidence: None,
                response_generated: response.map(|s| s.to_string()),
            },
            metadata: HashMap::new(),
            context_snapshot: None,
        };
        
        self.write_entry(entry).await
    }

    /// Log learning event
    pub async fn log_learning_event(&mut self, learning_type: &str, insights: Vec<String>) -> Result<()> {
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::LearningEvent,
            content: HreContent::LearningEvent {
                learning_type: learning_type.to_string(),
                objective: "continuous_improvement".to_string(),
                progress_delta: 0.1, // Would be calculated
                insights_gained: insights,
                adaptations_made: Vec::new(),
            },
            metadata: HashMap::new(),
            context_snapshot: None,
        };
        
        self.write_entry(entry).await
    }

    /// Write an entry to the log file
    async fn write_entry(&mut self, entry: HreEntry) -> Result<()> {
        self.entry_count += 1;
        
        // Add to buffer
        self.buffered_entries.push(entry.clone());
        
        // Serialize entry
        let serialized = self.serialize_entry(&entry)?;
        
        // Write to file
        if let Some(writer) = &mut self.writer {
            writeln!(writer, "{}", serialized)?;
            
            // Flush if buffer is full or on important events
            if self.buffered_entries.len() >= 10 || 
               matches!(entry.entry_type, HreEntryType::TaskCompleted | HreEntryType::SessionEnd) {
                writer.flush()?;
                self.buffered_entries.clear();
            }
        }
        
        Ok(())
    }

    /// Serialize an HRE entry to the file format
    fn serialize_entry(&self, entry: &HreEntry) -> Result<String> {
        // Use a compact JSON format for HRE files
        let json = serde_json::to_string(entry)?;
        Ok(format!("[{}] {}", self.entry_count, json))
    }

    /// Read and parse an HRE file
    pub async fn read_hre_file(file_path: &Path) -> Result<Vec<HreEntry>> {
        let file = File::open(file_path)
            .with_context(|| format!("Failed to open HRE file: {:?}", file_path))?;
        
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        
        for line in reader.lines() {
            let line = line?;
            
            // Skip header lines and empty lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            
            // Parse entry line
            if let Some(json_start) = line.find('{') {
                let json_part = &line[json_start..];
                match serde_json::from_str::<HreEntry>(json_part) {
                    Ok(entry) => entries.push(entry),
                    Err(e) => {
                        eprintln!("Failed to parse HRE entry: {}", e);
                        continue;
                    }
                }
            }
        }
        
        Ok(entries)
    }

    /// Analyze patterns in HRE logs
    pub async fn analyze_session_patterns(entries: &[HreEntry]) -> Result<SessionAnalysis> {
        let mut decision_count = 0;
        let mut task_count = 0;
        let mut error_count = 0;
        let mut module_usage = HashMap::new();
        let mut execution_times = Vec::new();
        
        let session_start = entries.first().map(|e| e.timestamp).unwrap_or(SystemTime::now());
        let session_end = entries.last().map(|e| e.timestamp).unwrap_or(SystemTime::now());
        
        for entry in entries {
            match entry.entry_type {
                HreEntryType::DecisionMade => decision_count += 1,
                HreEntryType::TaskCompleted => task_count += 1,
                HreEntryType::ModuleError => error_count += 1,
                HreEntryType::ModuleCompleted => {
                    if let HreContent::ModuleEvent { module_name, execution_time_ms, .. } = &entry.content {
                        *module_usage.entry(module_name.clone()).or_insert(0) += 1;
                        if let Some(time) = execution_time_ms {
                            execution_times.push(*time);
                        }
                    }
                }
                _ => {}
            }
        }
        
        let session_duration = session_end.duration_since(session_start).unwrap_or(Duration::ZERO);
        let avg_execution_time = if execution_times.is_empty() {
            0.0
        } else {
            execution_times.iter().sum::<u64>() as f64 / execution_times.len() as f64
        };
        
        Ok(SessionAnalysis {
            session_duration,
            total_entries: entries.len(),
            decision_count,
            task_count,
            error_count,
            error_rate: if task_count > 0 { error_count as f64 / task_count as f64 } else { 0.0 },
            module_usage,
            average_execution_time_ms: avg_execution_time,
            session_efficiency: calculate_efficiency_score(task_count, error_count, &execution_times),
        })
    }

    /// Flush all buffered entries and close the log file
    pub async fn close(&mut self) -> Result<()> {
        // Log session end
        let entry = HreEntry {
            entry_id: Uuid::new_v4(),
            session_id: self.current_session_id,
            timestamp: SystemTime::now(),
            entry_type: HreEntryType::SessionEnd,
            content: HreContent::SessionInfo {
                orchestrator_id: self.current_session_id,
                project_root: "".to_string(),
                configuration: {
                    let mut config = HashMap::new();
                    config.insert("total_entries".to_string(), serde_json::json!(self.entry_count));
                    config.insert("session_duration_ms".to_string(), 
                        serde_json::json!(SystemTime::now().duration_since(self.session_start)?.as_millis()));
                    config
                },
            },
            metadata: HashMap::new(),
            context_snapshot: None,
        };
        
        self.write_entry(entry).await?;
        
        // Flush and close
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        
        Ok(())
    }
}

/// Analysis results for an HRE session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAnalysis {
    pub session_duration: Duration,
    pub total_entries: usize,
    pub decision_count: u32,
    pub task_count: u32,
    pub error_count: u32,
    pub error_rate: f64,
    pub module_usage: HashMap<String, u32>,
    pub average_execution_time_ms: f64,
    pub session_efficiency: f64,
}

/// Calculate efficiency score for a session
fn calculate_efficiency_score(task_count: u32, error_count: u32, execution_times: &[u64]) -> f64 {
    if task_count == 0 {
        return 0.0;
    }
    
    let success_rate = 1.0 - (error_count as f64 / task_count as f64);
    let speed_factor = if execution_times.is_empty() {
        1.0
    } else {
        let avg_time = execution_times.iter().sum::<u64>() as f64 / execution_times.len() as f64;
        (1000.0 / (avg_time + 1.0)).min(1.0) // Normalize based on 1 second baseline
    };
    
    (success_rate * 0.7 + speed_factor * 0.3).max(0.0).min(1.0)
} 