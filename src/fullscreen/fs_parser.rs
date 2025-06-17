use std::collections::HashMap;
use std::path::PathBuf;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use toml;

use super::{
    FullscreenFile, ModuleNode, ModuleEdge, ModuleType, ConnectionType, 
    DataFlowType, LayoutHints, LayoutType
};

/// Configuration structure that matches .fs file format
#[derive(Debug, Clone, Deserialize)]
struct FsFileConfig {
    module: ModuleConfig,
    layout: Option<LayoutConfig>,
    connections: Option<ConnectionsConfig>,
    metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModuleConfig {
    name: String,
    r#type: String,
    path: String,
    complexity: Option<f64>,
    dependencies: Option<Vec<String>>,
    semantic_weight: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct LayoutConfig {
    r#type: String,
    clustering: Option<Vec<Vec<String>>>,
    positions: Option<HashMap<String, (f64, f64)>>,
    zoom: Option<f64>,
    focus: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ConnectionsConfig {
    dependency: Option<Vec<DependencyConfig>>,
    data_flow: Option<Vec<DataFlowConfig>>,
    control: Option<Vec<ControlConfig>>,
}

#[derive(Debug, Clone, Deserialize)]
struct DependencyConfig {
    target: String,
    r#type: String,
    weight: Option<f64>,
    bidirectional: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct DataFlowConfig {
    from: String,
    to: String,
    flow_type: String,
    weight: Option<f64>,
    latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct ControlConfig {
    target: String,
    control_type: String,
    weight: Option<f64>,
}

/// Parse the content of a .fs file
pub async fn parse_fs_content(content: &str, file_path: PathBuf) -> Result<FullscreenFile> {
    // First try to parse as TOML
    let config: FsFileConfig = toml::from_str(content)
        .with_context(|| format!("Failed to parse .fs file as TOML: {:?}", file_path))?;
    
    // Convert to our internal representation
    let module_id = Uuid::new_v4();
    let module = convert_module_config(config.module, module_id)?;
    let layout_hints = convert_layout_config(config.layout);
    let connections = convert_connections_config(config.connections, module_id).await?;
    
    Ok(FullscreenFile {
        path: file_path,
        modules: vec![module],
        connections,
        layout_hints,
        metadata: config.metadata.unwrap_or_default(),
    })
}

/// Convert module configuration to internal representation
fn convert_module_config(config: ModuleConfig, id: Uuid) -> Result<ModuleNode> {
    let module_type = match config.r#type.to_lowercase().as_str() {
        "core" => ModuleType::Core,
        "processing" => ModuleType::Processing,
        "orchestrator" => ModuleType::Orchestrator,
        "interface" => ModuleType::Interface,
        "storage" => ModuleType::Storage,
        "analysis" => ModuleType::Analysis,
        "external" => ModuleType::External,
        _ => return Err(anyhow::anyhow!("Unknown module type: {}", config.r#type)),
    };
    
    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "fs_file".to_string());
    
    Ok(ModuleNode {
        id,
        name: config.name,
        module_type,
        path: PathBuf::from(config.path),
        dependencies: config.dependencies.unwrap_or_default(),
        semantic_weight: config.semantic_weight.unwrap_or(0.5),
        complexity_score: config.complexity.unwrap_or(0.5),
        metadata,
    })
}

/// Convert layout configuration to internal representation
fn convert_layout_config(config: Option<LayoutConfig>) -> LayoutHints {
    if let Some(config) = config {
        let layout_type = match config.r#type.to_lowercase().as_str() {
            "force" => LayoutType::Force,
            "hierarchical" => LayoutType::Hierarchical,
            "circular" => LayoutType::Circular,
            "grid" => LayoutType::Grid,
            "custom" => LayoutType::Custom,
            _ => LayoutType::Force,
        };
        
        // Convert clustering from string IDs to UUIDs (simplified for now)
        let clustering = config.clustering.unwrap_or_default()
            .into_iter()
            .map(|cluster| {
                cluster.into_iter()
                    .map(|_| Uuid::new_v4()) // In real implementation, we'd resolve these
                    .collect()
            })
            .collect();
        
        // Convert positions (simplified)
        let positions = config.positions.unwrap_or_default()
            .into_iter()
            .map(|(_, pos)| (Uuid::new_v4(), pos))
            .collect();
        
        LayoutHints {
            preferred_layout: layout_type,
            clustering,
            positions,
            zoom_level: config.zoom.unwrap_or(1.0),
            focus_node: None, // TODO: resolve focus node ID
        }
    } else {
        LayoutHints {
            preferred_layout: LayoutType::Force,
            clustering: Vec::new(),
            positions: HashMap::new(),
            zoom_level: 1.0,
            focus_node: None,
        }
    }
}

/// Convert connections configuration to internal representation
async fn convert_connections_config(
    config: Option<ConnectionsConfig>, 
    module_id: Uuid
) -> Result<Vec<ModuleEdge>> {
    let mut edges = Vec::new();
    
    if let Some(config) = config {
        // Convert dependency connections
        if let Some(dependencies) = config.dependency {
            for dep in dependencies {
                let connection_type = match dep.r#type.to_lowercase().as_str() {
                    "semantic" | "semanticdependency" => ConnectionType::SemanticDependency,
                    "data" | "dataflow" => ConnectionType::DataFlow,
                    "control" => ConnectionType::Control,
                    "feedback" => ConnectionType::Feedback,
                    "coordination" => ConnectionType::Coordination,
                    "inheritance" => ConnectionType::Inheritance,
                    "composition" => ConnectionType::Composition,
                    _ => ConnectionType::SemanticDependency,
                };
                
                let edge = ModuleEdge {
                    from: module_id,
                    to: Uuid::new_v4(), // TODO: resolve target module ID
                    connection_type,
                    weight: dep.weight.unwrap_or(1.0),
                    data_flow: if dep.bidirectional.unwrap_or(false) {
                        DataFlowType::Bidirectional
                    } else {
                        DataFlowType::Unidirectional
                    },
                    latency_ms: None,
                };
                
                edges.push(edge);
            }
        }
        
        // Convert data flow connections
        if let Some(data_flows) = config.data_flow {
            for flow in data_flows {
                let data_flow_type = match flow.flow_type.to_lowercase().as_str() {
                    "unidirectional" => DataFlowType::Unidirectional,
                    "bidirectional" => DataFlowType::Bidirectional,
                    "broadcast" => DataFlowType::Broadcast,
                    "aggregation" => DataFlowType::Aggregation,
                    "transform" => DataFlowType::Transform,
                    _ => DataFlowType::Unidirectional,
                };
                
                let edge = ModuleEdge {
                    from: Uuid::new_v4(), // TODO: resolve from module ID
                    to: Uuid::new_v4(),   // TODO: resolve to module ID
                    connection_type: ConnectionType::DataFlow,
                    weight: flow.weight.unwrap_or(1.0),
                    data_flow: data_flow_type,
                    latency_ms: flow.latency_ms,
                };
                
                edges.push(edge);
            }
        }
        
        // Convert control connections
        if let Some(controls) = config.control {
            for control in controls {
                let edge = ModuleEdge {
                    from: module_id,
                    to: Uuid::new_v4(), // TODO: resolve target module ID
                    connection_type: ConnectionType::Control,
                    weight: control.weight.unwrap_or(1.0),
                    data_flow: DataFlowType::Unidirectional,
                    latency_ms: None,
                };
                
                edges.push(edge);
            }
        }
    }
    
    Ok(edges)
}

/// Advanced parser for complex .fs file formats
pub struct AdvancedFsParser {
    module_registry: HashMap<String, Uuid>,
    path_registry: HashMap<PathBuf, Uuid>,
}

impl AdvancedFsParser {
    pub fn new() -> Self {
        Self {
            module_registry: HashMap::new(),
            path_registry: HashMap::new(),
        }
    }
    
    /// Parse multiple .fs files and resolve cross-references
    pub async fn parse_multiple_fs_files(
        &mut self, 
        fs_files: Vec<PathBuf>
    ) -> Result<Vec<FullscreenFile>> {
        let mut parsed_files = Vec::new();
        
        // First pass: parse all files and build registry
        for file_path in &fs_files {
            let content = std::fs::read_to_string(file_path)?;
            let mut fs_file = parse_fs_content(&content, file_path.clone()).await?;
            
            // Register modules
            for module in &fs_file.modules {
                self.module_registry.insert(module.name.clone(), module.id);
                self.path_registry.insert(module.path.clone(), module.id);
            }
            
            parsed_files.push(fs_file);
        }
        
        // Second pass: resolve cross-references
        for fs_file in &mut parsed_files {
            self.resolve_cross_references(fs_file).await?;
        }
        
        Ok(parsed_files)
    }
    
    /// Resolve cross-references between modules
    async fn resolve_cross_references(&self, fs_file: &mut FullscreenFile) -> Result<()> {
        for connection in &mut fs_file.connections {
            // This is where we'd resolve module names to UUIDs
            // For now, this is a placeholder
        }
        
        Ok(())
    }
}

/// Validate .fs file syntax and semantics
pub fn validate_fs_file(content: &str) -> Result<Vec<ValidationError>> {
    let mut errors = Vec::new();
    
    // Try to parse as TOML first
    match toml::from_str::<FsFileConfig>(content) {
        Ok(config) => {
            // Validate module configuration
            if config.module.name.is_empty() {
                errors.push(ValidationError::new(
                    ValidationErrorType::EmptyModuleName,
                    "Module name cannot be empty".to_string(),
                ));
            }
            
            // Validate module type
            let valid_types = ["core", "processing", "orchestrator", "interface", "storage", "analysis", "external"];
            if !valid_types.contains(&config.module.r#type.to_lowercase().as_str()) {
                errors.push(ValidationError::new(
                    ValidationErrorType::InvalidModuleType,
                    format!("Invalid module type: {}", config.module.r#type),
                ));
            }
            
            // Validate complexity score
            if let Some(complexity) = config.module.complexity {
                if complexity < 0.0 || complexity > 1.0 {
                    errors.push(ValidationError::new(
                        ValidationErrorType::InvalidComplexityScore,
                        "Complexity score must be between 0.0 and 1.0".to_string(),
                    ));
                }
            }
        }
        Err(e) => {
            errors.push(ValidationError::new(
                ValidationErrorType::SyntaxError,
                format!("TOML syntax error: {}", e),
            ));
        }
    }
    
    Ok(errors)
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
}

impl ValidationError {
    pub fn new(error_type: ValidationErrorType, message: String) -> Self {
        Self { error_type, message }
    }
}

#[derive(Debug, Clone)]
pub enum ValidationErrorType {
    SyntaxError,
    EmptyModuleName,
    InvalidModuleType,
    InvalidComplexityScore,
    MissingDependency,
    CircularDependency,
} 