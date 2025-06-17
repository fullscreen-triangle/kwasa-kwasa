use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use petgraph::{Graph, Directed, graph::NodeIndex};
use petgraph::algo::{dijkstra, connected_components};
use petgraph::visit::EdgeRef;
use anyhow::{Result, Context};
use uuid::Uuid;

pub mod graph_engine;
pub mod visualization;
pub mod layout;
pub mod analysis;
pub mod fs_parser;
pub mod interactive;

/// Core types for the Fullscreen network graph system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleNode {
    pub id: Uuid,
    pub name: String,
    pub module_type: ModuleType,
    pub path: PathBuf,
    pub dependencies: Vec<String>,
    pub semantic_weight: f64,
    pub complexity_score: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleType {
    Core,
    Processing,
    Orchestrator,
    Interface,
    Storage,
    Analysis,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleEdge {
    pub from: Uuid,
    pub to: Uuid,
    pub connection_type: ConnectionType,
    pub weight: f64,
    pub data_flow: DataFlowType,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    SemanticDependency,
    DataFlow,
    Control,
    Feedback,
    Coordination,
    Inheritance,
    Composition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFlowType {
    Unidirectional,
    Bidirectional,
    Broadcast,
    Aggregation,
    Transform,
}

/// The main Fullscreen system that manages network graphs
pub struct FullscreenSystem {
    graph: Graph<ModuleNode, ModuleEdge, Directed>,
    node_index_map: HashMap<Uuid, NodeIndex>,
    project_root: PathBuf,
    fs_files: HashMap<PathBuf, FullscreenFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullscreenFile {
    pub path: PathBuf,
    pub modules: Vec<ModuleNode>,
    pub connections: Vec<ModuleEdge>,
    pub layout_hints: LayoutHints,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutHints {
    pub preferred_layout: LayoutType,
    pub clustering: Vec<Vec<Uuid>>,
    pub positions: HashMap<Uuid, (f64, f64)>,
    pub zoom_level: f64,
    pub focus_node: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    Force,
    Hierarchical,
    Circular,
    Grid,
    Custom,
}

impl FullscreenSystem {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            graph: Graph::new(),
            node_index_map: HashMap::new(),
            project_root,
            fs_files: HashMap::new(),
        }
    }

    /// Discover and analyze all .fs files in the project
    pub async fn discover_project_structure(&mut self) -> Result<()> {
        let fs_files = self.find_fs_files(&self.project_root.clone())?;
        
        for fs_file_path in fs_files {
            let fs_file = self.parse_fs_file(&fs_file_path).await?;
            self.integrate_fs_file(fs_file)?;
        }
        
        self.analyze_cross_module_dependencies().await?;
        self.calculate_semantic_weights()?;
        
        Ok(())
    }

    /// Find all .fs files in the project directory
    fn find_fs_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut fs_files = Vec::new();
        
        fn visit_dir(dir: &Path, fs_files: &mut Vec<PathBuf>) -> Result<()> {
            if dir.is_dir() {
                for entry in fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();
                    
                    if path.is_dir() {
                        visit_dir(&path, fs_files)?;
                    } else if path.extension().and_then(|s| s.to_str()) == Some("fs") {
                        fs_files.push(path);
                    }
                }
            }
            Ok(())
        }
        
        visit_dir(dir, &mut fs_files)?;
        Ok(fs_files)
    }

    /// Parse a .fs file and extract module information
    async fn parse_fs_file(&self, path: &Path) -> Result<FullscreenFile> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read .fs file: {:?}", path))?;
        
        fs_parser::parse_fs_content(&content, path.to_owned()).await
    }

    /// Integrate a parsed .fs file into the system graph
    fn integrate_fs_file(&mut self, fs_file: FullscreenFile) -> Result<()> {
        // Add nodes to the graph
        for module in &fs_file.modules {
            let node_index = self.graph.add_node(module.clone());
            self.node_index_map.insert(module.id, node_index);
        }
        
        // Add edges to the graph
        for connection in &fs_file.connections {
            if let (Some(&from_idx), Some(&to_idx)) = 
                (self.node_index_map.get(&connection.from), 
                 self.node_index_map.get(&connection.to)) {
                self.graph.add_edge(from_idx, to_idx, connection.clone());
            }
        }
        
        self.fs_files.insert(fs_file.path.clone(), fs_file);
        Ok(())
    }

    /// Analyze dependencies across modules to build the complete graph
    async fn analyze_cross_module_dependencies(&mut self) -> Result<()> {
        // Analyze actual code dependencies
        for (_, fs_file) in &self.fs_files {
            for module in &fs_file.modules {
                let dependencies = self.analyze_module_dependencies(&module.path).await?;
                
                for dep_path in dependencies {
                    if let Some(dep_module) = self.find_module_by_path(&dep_path) {
                        let edge = ModuleEdge {
                            from: module.id,
                            to: dep_module.id,
                            connection_type: ConnectionType::SemanticDependency,
                            weight: 1.0,
                            data_flow: DataFlowType::Unidirectional,
                            latency_ms: None,
                        };
                        
                        if let (Some(&from_idx), Some(&to_idx)) = 
                            (self.node_index_map.get(&module.id), 
                             self.node_index_map.get(&dep_module.id)) {
                            self.graph.add_edge(from_idx, to_idx, edge);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Analyze dependencies for a specific module
    async fn analyze_module_dependencies(&self, module_path: &Path) -> Result<Vec<PathBuf>> {
        let mut dependencies = Vec::new();
        
        // Check for Rust module dependencies
        if module_path.extension().and_then(|s| s.to_str()) == Some("rs") {
            dependencies.extend(self.analyze_rust_dependencies(module_path).await?);
        }
        
        // Check for Turbulance dependencies (.tb files)
        if module_path.extension().and_then(|s| s.to_str()) == Some("tb") {
            dependencies.extend(self.analyze_turbulance_dependencies(module_path).await?);
        }
        
        // Check for Gerhard dependencies (.ghd files)
        if module_path.extension().and_then(|s| s.to_str()) == Some("ghd") {
            dependencies.extend(self.analyze_gerhard_dependencies(module_path).await?);
        }
        
        Ok(dependencies)
    }

    /// Analyze Rust module dependencies
    async fn analyze_rust_dependencies(&self, _path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: Implement AST parsing for Rust dependencies
        Ok(Vec::new())
    }

    /// Analyze Turbulance script dependencies
    async fn analyze_turbulance_dependencies(&self, _path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: Implement Turbulance syntax parsing for dependencies
        Ok(Vec::new())
    }

    /// Analyze Gerhard file dependencies
    async fn analyze_gerhard_dependencies(&self, _path: &Path) -> Result<Vec<PathBuf>> {
        // TODO: Implement .ghd file parsing for external dependencies
        Ok(Vec::new())
    }

    /// Find a module by its file path
    fn find_module_by_path(&self, path: &Path) -> Option<&ModuleNode> {
        for fs_file in self.fs_files.values() {
            for module in &fs_file.modules {
                if module.path == path {
                    return Some(module);
                }
            }
        }
        None
    }

    /// Calculate semantic weights for all modules based on their relationships
    fn calculate_semantic_weights(&mut self) -> Result<()> {
        let node_count = self.graph.node_count();
        
        for node_idx in self.graph.node_indices() {
            let node = &mut self.graph[node_idx];
            
            // Calculate weight based on:
            // 1. Number of incoming connections
            // 2. Number of outgoing connections
            // 3. Module type importance
            // 4. Complexity score
            
            let incoming = self.graph.edges_directed(node_idx, petgraph::Incoming).count();
            let outgoing = self.graph.edges_directed(node_idx, petgraph::Outgoing).count();
            
            let type_weight = match node.module_type {
                ModuleType::Core => 1.0,
                ModuleType::Orchestrator => 0.9,
                ModuleType::Processing => 0.7,
                ModuleType::Interface => 0.6,
                ModuleType::Analysis => 0.5,
                ModuleType::Storage => 0.4,
                ModuleType::External => 0.3,
            };
            
            let connection_weight = (incoming + outgoing) as f64 / node_count as f64;
            
            node.semantic_weight = (type_weight + connection_weight + node.complexity_score) / 3.0;
        }
        
        Ok(())
    }

    /// Generate a visual representation of the network graph
    pub async fn generate_visualization(&self, output_path: &Path) -> Result<()> {
        visualization::generate_svg_graph(self, output_path).await
    }

    /// Export the graph to various formats
    pub async fn export_graph(&self, format: ExportFormat, output_path: &Path) -> Result<()> {
        match format {
            ExportFormat::Dot => self.export_dot(output_path).await,
            ExportFormat::Json => self.export_json(output_path).await,
            ExportFormat::Svg => self.generate_visualization(output_path).await,
            ExportFormat::Interactive => self.export_interactive_html(output_path).await,
        }
    }

    async fn export_dot(&self, output_path: &Path) -> Result<()> {
        visualization::export_dot_format(self, output_path).await
    }

    async fn export_json(&self, output_path: &Path) -> Result<()> {
        let graph_data = serde_json::to_string_pretty(&self.fs_files)?;
        fs::write(output_path, graph_data)?;
        Ok(())
    }

    async fn export_interactive_html(&self, output_path: &Path) -> Result<()> {
        interactive::generate_interactive_visualization(self, output_path).await
    }

    /// Get graph analytics
    pub fn get_analytics(&self) -> GraphAnalytics {
        analysis::compute_graph_analytics(&self.graph)
    }

    /// Find paths between modules
    pub fn find_semantic_path(&self, from: Uuid, to: Uuid) -> Option<Vec<Uuid>> {
        if let (Some(&from_idx), Some(&to_idx)) = 
            (self.node_index_map.get(&from), self.node_index_map.get(&to)) {
            
            let paths = dijkstra(&self.graph, from_idx, Some(to_idx), |e| e.weight() as i32);
            
            if paths.contains_key(&to_idx) {
                // Reconstruct path (simplified for now)
                Some(vec![from, to])
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub enum ExportFormat {
    Dot,
    Json,
    Svg,
    Interactive,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphAnalytics {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub connected_components: usize,
    pub average_degree: f64,
    pub density: f64,
    pub most_connected_modules: Vec<String>,
    pub critical_paths: Vec<Vec<String>>,
}

/// Create a default .fs file for a module
pub fn create_default_fs_file(module_name: &str, module_path: &Path) -> Result<String> {
    let template = format!(r#"
# Fullscreen Network Graph for {module_name}
# This file defines the module relationships and semantic connections

[module]
name = "{module_name}"
type = "Processing"
path = "{path}"
complexity = 0.5

[layout]
type = "Force"
clustering = []
zoom = 1.0

[connections]
# Define connections to other modules
# Example:
# [[connections.dependency]]
# target = "turbulance::core"
# type = "SemanticDependency"
# weight = 1.0

[metadata]
description = "Generated .fs file for {module_name}"
created = "{timestamp}"
"#, 
        module_name = module_name,
        path = module_path.display(),
        timestamp = chrono::Utc::now().to_rfc3339()
    );
    
    Ok(template)
} 