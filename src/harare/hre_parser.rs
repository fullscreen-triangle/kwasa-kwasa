use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HreFile {
    pub path: std::path::PathBuf,
    pub metadata: HreMetadata,
    pub orchestration_rules: Vec<OrchestrationRule>,
    pub logging_config: LoggingConfig,
    pub performance_targets: PerformanceTargets,
    pub resource_limits: ResourceLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HreMetadata {
    pub version: String,
    pub created_at: std::time::SystemTime,
    pub author: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationRule {
    pub rule_id: String,
    pub condition: String,
    pub action: String,
    pub priority: u32,
    pub enabled: bool,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub log_level: String,
    pub output_format: String,
    pub rotation_policy: RotationPolicy,
    pub filters: Vec<LogFilter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicy {
    pub max_size_mb: u64,
    pub max_files: u32,
    pub rotation_interval: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFilter {
    pub module_pattern: String,
    pub level_threshold: String,
    pub include_traces: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub max_response_time_ms: u64,
    pub min_throughput_ops_per_sec: f64,
    pub max_error_rate_percent: f64,
    pub target_availability_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_cpu_percent: f64,
    pub max_memory_mb: u64,
    pub max_disk_io_mb_per_sec: f64,
    pub max_network_mb_per_sec: f64,
    pub max_concurrent_tasks: u32,
}

pub struct HreParser;

impl HreParser {
    pub async fn parse_hre_file(path: &Path) -> Result<HreFile> {
        let content = tokio::fs::read_to_string(path).await
            .with_context(|| format!("Failed to read HRE file: {:?}", path))?;
        
        Self::parse_hre_content(&content, path.to_owned()).await
    }

    pub async fn parse_hre_content(content: &str, path: std::path::PathBuf) -> Result<HreFile> {
        // Simple HRE format parser - in practice this would be more sophisticated
        let lines: Vec<&str> = content.lines().collect();
        
        let metadata = Self::parse_metadata(&lines)?;
        let orchestration_rules = Self::parse_orchestration_rules(&lines)?;
        let logging_config = Self::parse_logging_config(&lines)?;
        let performance_targets = Self::parse_performance_targets(&lines)?;
        let resource_limits = Self::parse_resource_limits(&lines)?;

        Ok(HreFile {
            path,
            metadata,
            orchestration_rules,
            logging_config,
            performance_targets,
            resource_limits,
        })
    }

    fn parse_metadata(lines: &[&str]) -> Result<HreMetadata> {
        let mut metadata = HreMetadata {
            version: "1.0".to_string(),
            created_at: std::time::SystemTime::now(),
            author: "unknown".to_string(),
            description: "".to_string(),
            tags: Vec::new(),
        };

        for line in lines {
            if line.starts_with("# VERSION:") {
                metadata.version = line.replace("# VERSION:", "").trim().to_string();
            } else if line.starts_with("# AUTHOR:") {
                metadata.author = line.replace("# AUTHOR:", "").trim().to_string();
            } else if line.starts_with("# DESCRIPTION:") {
                metadata.description = line.replace("# DESCRIPTION:", "").trim().to_string();
            } else if line.starts_with("# TAGS:") {
                metadata.tags = line.replace("# TAGS:", "")
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
            }
        }

        Ok(metadata)
    }

    fn parse_orchestration_rules(lines: &[&str]) -> Result<Vec<OrchestrationRule>> {
        let mut rules = Vec::new();
        let mut in_rules_section = false;

        for line in lines {
            if line.trim() == "[ORCHESTRATION_RULES]" {
                in_rules_section = true;
                continue;
            } else if line.starts_with('[') && line.ends_with(']') {
                in_rules_section = false;
                continue;
            }

            if in_rules_section && !line.trim().is_empty() && !line.starts_with('#') {
                if let Ok(rule) = Self::parse_rule_line(line) {
                    rules.push(rule);
                }
            }
        }

        Ok(rules)
    }

    fn parse_rule_line(line: &str) -> Result<OrchestrationRule> {
        let parts: Vec<&str> = line.split(" -> ").collect();
        if parts.len() != 2 {
            return Err(anyhow::anyhow!("Invalid rule format"));
        }

        Ok(OrchestrationRule {
            rule_id: uuid::Uuid::new_v4().to_string(),
            condition: parts[0].trim().to_string(),
            action: parts[1].trim().to_string(),
            priority: 1,
            enabled: true,
            parameters: HashMap::new(),
        })
    }

    fn parse_logging_config(lines: &[&str]) -> Result<LoggingConfig> {
        let mut config = LoggingConfig {
            log_level: "INFO".to_string(),
            output_format: "JSON".to_string(),
            rotation_policy: RotationPolicy {
                max_size_mb: 100,
                max_files: 10,
                rotation_interval: "daily".to_string(),
            },
            filters: Vec::new(),
        };

        let mut in_logging_section = false;
        for line in lines {
            if line.trim() == "[LOGGING]" {
                in_logging_section = true;
                continue;
            } else if line.starts_with('[') && line.ends_with(']') {
                in_logging_section = false;
                continue;
            }

            if in_logging_section && line.contains('=') {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim();
                    let value = parts[1].trim();
                    
                    match key {
                        "level" => config.log_level = value.to_string(),
                        "format" => config.output_format = value.to_string(),
                        _ => {}
                    }
                }
            }
        }

        Ok(config)
    }

    fn parse_performance_targets(_lines: &[&str]) -> Result<PerformanceTargets> {
        Ok(PerformanceTargets {
            max_response_time_ms: 1000,
            min_throughput_ops_per_sec: 100.0,
            max_error_rate_percent: 1.0,
            target_availability_percent: 99.9,
        })
    }

    fn parse_resource_limits(_lines: &[&str]) -> Result<ResourceLimits> {
        Ok(ResourceLimits {
            max_cpu_percent: 80.0,
            max_memory_mb: 4096,
            max_disk_io_mb_per_sec: 100.0,
            max_network_mb_per_sec: 50.0,
            max_concurrent_tasks: 10,
        })
    }
} 