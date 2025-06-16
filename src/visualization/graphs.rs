//! Graph visualization components

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::pattern::prelude::{Pattern, MetaNode, MetaEdge, MetaNodeType, MetaEdgeType};
use super::{
    Visualization, VisualizationType, VisualizationData, VisualizationConfig,
    NetworkData, NetworkNode, NetworkEdge, NetworkLayout, ColorScheme
};

/// Graph-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Layout algorithm to use
    pub layout: NetworkLayout,
    /// Node styling configuration
    pub node_config: NodeConfig,
    /// Edge styling configuration
    pub edge_config: EdgeConfig,
    /// Force-directed layout parameters
    pub force_config: ForceConfig,
    /// Whether to show labels
    pub show_labels: bool,
    /// Whether to show tooltips on hover
    pub show_tooltips: bool,
    /// Animation settings
    pub animation: AnimationConfig,
    /// Clustering settings
    pub clustering: ClusteringConfig,
}

/// Node styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Default node size
    pub default_size: f64,
    /// Size range for dynamic sizing
    pub size_range: (f64, f64),
    /// Default node color
    pub default_color: String,
    /// Color mapping for node types
    pub type_colors: HashMap<String, String>,
    /// Node shape
    pub shape: NodeShape,
    /// Border width
    pub border_width: f32,
    /// Border color
    pub border_color: String,
}

/// Edge styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Default edge width
    pub default_width: f32,
    /// Width range for dynamic sizing
    pub width_range: (f32, f32),
    /// Default edge color
    pub default_color: String,
    /// Color mapping for edge types
    pub type_colors: HashMap<String, String>,
    /// Edge style
    pub style: EdgeStyle,
    /// Arrow size for directed edges
    pub arrow_size: f32,
    /// Whether to show edge labels
    pub show_labels: bool,
}

/// Force-directed layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceConfig {
    /// Repulsion strength between nodes
    pub repulsion_strength: f64,
    /// Attraction strength of edges
    pub attraction_strength: f64,
    /// Gravity towards center
    pub gravity: f64,
    /// Damping factor
    pub damping: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Animation duration in milliseconds
    pub duration: u32,
    /// Animation easing function
    pub easing: String,
    /// Whether to animate layout changes
    pub animate_layout: bool,
    /// Whether to animate data changes
    pub animate_data: bool,
}

/// Clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Whether to enable clustering
    pub enabled: bool,
    /// Clustering algorithm
    pub algorithm: ClusteringAlgorithm,
    /// Minimum cluster size
    pub min_cluster_size: usize,
    /// Cluster colors
    pub cluster_colors: Vec<String>,
}

/// Node shapes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeShape {
    Circle,
    Square,
    Triangle,
    Diamond,
    Hexagon,
    Star,
}

/// Edge styles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeStyle {
    Solid,
    Dashed,
    Dotted,
    Curved,
    Straight,
}

/// Clustering algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    /// Community detection based on modularity
    Modularity,
    /// Label propagation
    LabelPropagation,
    /// Hierarchical clustering
    Hierarchical,
    /// K-means clustering
    KMeans,
}

/// Network visualization struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkVisualization {
    /// Unique identifier
    pub id: String,
    /// Title
    pub title: String,
    /// Graph configuration
    pub config: GraphConfig,
    /// Network data
    pub network_data: NetworkData,
    /// Computed clusters
    pub clusters: Vec<NodeCluster>,
    /// Graph metrics
    pub metrics: GraphMetrics,
}

/// Node cluster information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCluster {
    /// Cluster ID
    pub id: String,
    /// Nodes in this cluster
    pub nodes: Vec<String>,
    /// Cluster center (computed)
    pub center: (f64, f64),
    /// Cluster color
    pub color: String,
    /// Cluster density
    pub density: f64,
}

/// Graph metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Graph density
    pub density: f64,
    /// Average degree
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Connected components
    pub connected_components: usize,
    /// Diameter
    pub diameter: Option<usize>,
    /// Node centrality measures
    pub centrality: HashMap<String, NodeCentrality>,
}

/// Centrality measures for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCentrality {
    /// Degree centrality
    pub degree: f64,
    /// Betweenness centrality
    pub betweenness: f64,
    /// Closeness centrality
    pub closeness: f64,
    /// Eigenvector centrality
    pub eigenvector: f64,
    /// PageRank
    pub pagerank: f64,
}

/// Enhanced network node with additional visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualNetworkNode {
    /// Base network node
    pub base: NetworkNode,
    /// Computed position
    pub position: (f64, f64),
    /// Velocity (for force-directed layouts)
    pub velocity: (f64, f64),
    /// Forces acting on this node
    pub forces: (f64, f64),
    /// Node importance score
    pub importance: f64,
    /// Cluster membership
    pub cluster_id: Option<String>,
    /// Visibility state
    pub visible: bool,
    /// Selection state
    pub selected: bool,
}

/// Enhanced network edge with additional visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualNetworkEdge {
    /// Base network edge
    pub base: NetworkEdge,
    /// Computed control points for curved edges
    pub control_points: Vec<(f64, f64)>,
    /// Edge length
    pub length: f64,
    /// Visibility state
    pub visible: bool,
    /// Selection state
    pub selected: bool,
}

/// Graph builder for creating network visualizations
pub struct GraphBuilder {
    config: GraphConfig,
    nodes: Vec<VisualNetworkNode>,
    edges: Vec<VisualNetworkEdge>,
    title: String,
    width: u32,
    height: u32,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            config: GraphConfig::default(),
            nodes: Vec::new(),
            edges: Vec::new(),
            title: "Network Graph".to_string(),
            width: 800,
            height: 600,
        }
    }

    /// Set graph configuration
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = config;
        self
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Add a node
    pub fn add_node(mut self, node: VisualNetworkNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add an edge
    pub fn add_edge(mut self, edge: VisualNetworkEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Build network from patterns
    pub fn from_patterns(mut self, patterns: &[Pattern]) -> Self {
        for (i, pattern) in patterns.iter().enumerate() {
            let node = VisualNetworkNode {
                base: NetworkNode {
                    id: pattern.id.clone(),
                    label: pattern.name.clone(),
                    x: None,
                    y: None,
                    size: 10.0 + (pattern.confidence * 20.0),
                    color: Some(self.pattern_color(pattern)),
                    node_type: Some(pattern.pattern_type.clone()),
                },
                position: (0.0, 0.0), // Will be computed during layout
                velocity: (0.0, 0.0),
                forces: (0.0, 0.0),
                importance: pattern.significance,
                cluster_id: None,
                visible: true,
                selected: false,
            };
            self.nodes.push(node);

            // Create edges based on pattern relationships
            for (j, other_pattern) in patterns.iter().enumerate() {
                if i != j && pattern.is_similar_to(other_pattern) {
                    let edge = VisualNetworkEdge {
                        base: NetworkEdge {
                            source: pattern.id.clone(),
                            target: other_pattern.id.clone(),
                            weight: self.calculate_pattern_similarity(pattern, other_pattern),
                            label: Some("similar".to_string()),
                            color: Some("#94a3b8".to_string()),
                            edge_type: Some("similarity".to_string()),
                        },
                        control_points: Vec::new(),
                        length: 0.0,
                        visible: true,
                        selected: false,
                    };
                    self.edges.push(edge);
                }
            }
        }
        self
    }

    /// Build network from metacognitive nodes and edges
    pub fn from_metacognitive(
        mut self, 
        meta_nodes: &[MetaNode], 
        meta_edges: &[MetaEdge]
    ) -> Self {
        // Add nodes
        for meta_node in meta_nodes {
            let node = VisualNetworkNode {
                base: NetworkNode {
                    id: meta_node.id.clone(),
                    label: meta_node.content.clone(),
                    x: None,
                    y: None,
                    size: 8.0 + (meta_node.confidence * 25.0),
                    color: Some(self.meta_node_color(&meta_node.node_type)),
                    node_type: Some(format!("{:?}", meta_node.node_type)),
                },
                position: (0.0, 0.0),
                velocity: (0.0, 0.0),
                forces: (0.0, 0.0),
                importance: meta_node.confidence,
                cluster_id: None,
                visible: true,
                selected: false,
            };
            self.nodes.push(node);
        }

        // Add edges
        for meta_edge in meta_edges {
            let edge = VisualNetworkEdge {
                base: NetworkEdge {
                    source: meta_edge.source.clone(),
                    target: meta_edge.target.clone(),
                    weight: meta_edge.strength,
                    label: Some(format!("{:?}", meta_edge.edge_type)),
                    color: Some(self.meta_edge_color(&meta_edge.edge_type)),
                    edge_type: Some(format!("{:?}", meta_edge.edge_type)),
                },
                control_points: Vec::new(),
                length: 0.0,
                visible: true,
                selected: false,
            };
            self.edges.push(edge);
        }

        self
    }

    /// Build the visualization
    pub fn build(mut self) -> Result<Visualization> {
        // Apply layout algorithm
        self.apply_layout()?;
        
        // Calculate metrics
        let metrics = self.calculate_metrics();
        
        // Detect clusters if enabled
        let clusters = if self.config.clustering.enabled {
            self.detect_clusters()?
        } else {
            Vec::new()
        };

        // Convert to network data
        let network_data = NetworkData {
            nodes: self.nodes.iter().map(|vn| vn.base.clone()).collect(),
            edges: self.edges.iter().map(|ve| ve.base.clone()).collect(),
            layout: self.config.layout.clone(),
        };

        let network_viz = NetworkVisualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: self.title.clone(),
            config: self.config.clone(),
            network_data,
            clusters,
            metrics,
        };

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: self.title,
            description: "Interactive network graph visualization".to_string(),
            visualization_type: VisualizationType::NetworkGraph,
            data: VisualizationData::Custom(serde_json::to_value(network_viz)?),
            config: VisualizationConfig {
                width: self.width,
                height: self.height,
                color_scheme: ColorScheme::Default,
                show_legend: true,
                show_grid: false,
                font_size: 12,
                style_options: HashMap::new(),
            },
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    // Helper methods

    fn apply_layout(&mut self) -> Result<()> {
        match self.config.layout {
            NetworkLayout::ForceDirected => self.apply_force_directed_layout(),
            NetworkLayout::Circular => self.apply_circular_layout(),
            NetworkLayout::Grid => self.apply_grid_layout(),
            NetworkLayout::Hierarchical => self.apply_hierarchical_layout(),
            NetworkLayout::Fixed => Ok(()), // Use existing positions
        }
    }

    fn apply_force_directed_layout(&mut self) -> Result<()> {
        let center_x = self.width as f64 / 2.0;
        let center_y = self.height as f64 / 2.0;

        // Initialize random positions
        for (i, node) in self.nodes.iter_mut().enumerate() {
            if node.base.x.is_none() || node.base.y.is_none() {
                node.position = (
                    center_x + (i as f64 * 50.0) % (self.width as f64) - center_x,
                    center_y + (i as f64 * 37.0) % (self.height as f64) - center_y,
                );
            } else {
                node.position = (node.base.x.unwrap(), node.base.y.unwrap());
            }
        }

        // Run force-directed simulation
        for _ in 0..self.config.force_config.max_iterations {
            self.calculate_forces();
            self.update_positions();
            
            if self.check_convergence() {
                break;
            }
        }

        Ok(())
    }

    fn calculate_forces(&mut self) {
        let repulsion = self.config.force_config.repulsion_strength;
        let attraction = self.config.force_config.attraction_strength;
        let gravity = self.config.force_config.gravity;
        let center_x = self.width as f64 / 2.0;
        let center_y = self.height as f64 / 2.0;

        // Reset forces
        for node in &mut self.nodes {
            node.forces = (0.0, 0.0);
        }

        // Calculate repulsion forces between all nodes
        for i in 0..self.nodes.len() {
            for j in (i + 1)..self.nodes.len() {
                let dx = self.nodes[j].position.0 - self.nodes[i].position.0;
                let dy = self.nodes[j].position.1 - self.nodes[i].position.1;
                let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                
                let force = repulsion / distance;
                let fx = force * dx / distance;
                let fy = force * dy / distance;
                
                self.nodes[i].forces.0 -= fx;
                self.nodes[i].forces.1 -= fy;
                self.nodes[j].forces.0 += fx;
                self.nodes[j].forces.1 += fy;
            }
        }

        // Calculate attraction forces from edges
        for edge in &self.edges {
            if let (Some(source_idx), Some(target_idx)) = (
                self.nodes.iter().position(|n| n.base.id == edge.base.source),
                self.nodes.iter().position(|n| n.base.id == edge.base.target),
            ) {
                let dx = self.nodes[target_idx].position.0 - self.nodes[source_idx].position.0;
                let dy = self.nodes[target_idx].position.1 - self.nodes[source_idx].position.1;
                let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                
                let force = attraction * edge.base.weight * distance;
                let fx = force * dx / distance;
                let fy = force * dy / distance;
                
                self.nodes[source_idx].forces.0 += fx;
                self.nodes[source_idx].forces.1 += fy;
                self.nodes[target_idx].forces.0 -= fx;
                self.nodes[target_idx].forces.1 -= fy;
            }
        }

        // Apply gravity towards center
        for node in &mut self.nodes {
            let dx = center_x - node.position.0;
            let dy = center_y - node.position.1;
            node.forces.0 += gravity * dx;
            node.forces.1 += gravity * dy;
        }
    }

    fn update_positions(&mut self) {
        let damping = self.config.force_config.damping;
        
        for node in &mut self.nodes {
            node.velocity.0 = (node.velocity.0 + node.forces.0) * damping;
            node.velocity.1 = (node.velocity.1 + node.forces.1) * damping;
            
            node.position.0 += node.velocity.0;
            node.position.1 += node.velocity.1;
            
            // Keep nodes within bounds
            node.position.0 = node.position.0.max(50.0).min(self.width as f64 - 50.0);
            node.position.1 = node.position.1.max(50.0).min(self.height as f64 - 50.0);
        }
    }

    fn check_convergence(&self) -> bool {
        let threshold = self.config.force_config.convergence_threshold;
        let total_energy: f64 = self.nodes.iter()
            .map(|node| node.velocity.0 * node.velocity.0 + node.velocity.1 * node.velocity.1)
            .sum();
        
        total_energy < threshold
    }

    fn apply_circular_layout(&mut self) -> Result<()> {
        let center_x = self.width as f64 / 2.0;
        let center_y = self.height as f64 / 2.0;
        let radius = (self.width.min(self.height) as f64 / 2.0) * 0.8;
        let node_count = self.nodes.len(); // Store length before borrowing mutably
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / node_count as f64;
            node.position = (
                center_x + radius * angle.cos(),
                center_y + radius * angle.sin(),
            );
        }
        
        Ok(())
    }

    fn apply_grid_layout(&mut self) -> Result<()> {
        let cols = (self.nodes.len() as f64).sqrt().ceil() as usize;
        let cell_width = self.width as f64 / cols as f64;
        let cell_height = self.height as f64 / ((self.nodes.len() + cols - 1) / cols) as f64;
        
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let row = i / cols;
            let col = i % cols;
            node.position = (
                (col as f64 + 0.5) * cell_width,
                (row as f64 + 0.5) * cell_height,
            );
        }
        
        Ok(())
    }

    fn apply_hierarchical_layout(&mut self) -> Result<()> {
        // Simple tree layout - in practice, this would be more sophisticated
        let levels = self.calculate_node_levels();
        let max_level = levels.values().max().copied().unwrap_or(0);
        
        for level in 0..=max_level {
            let nodes_at_level: Vec<_> = levels.iter()
                .filter(|(_, &l)| l == level)
                .map(|(id, _)| id)
                .collect();
            
            let y = (level as f64 + 0.5) * self.height as f64 / (max_level + 1) as f64;
            
            for (i, node_id) in nodes_at_level.iter().enumerate() {
                if let Some(node) = self.nodes.iter_mut().find(|n| &n.base.id == *node_id) {
                    let x = (i as f64 + 0.5) * self.width as f64 / nodes_at_level.len() as f64;
                    node.position = (x, y);
                }
            }
        }
        
        Ok(())
    }

    fn calculate_node_levels(&self) -> HashMap<String, usize> {
        let mut levels = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        
        // Simple BFS to assign levels
        let mut queue = std::collections::VecDeque::new();
        
        // Start with nodes that have no incoming edges
        for node in &self.nodes {
            let has_incoming = self.edges.iter().any(|e| e.base.target == node.base.id);
            if !has_incoming {
                queue.push_back((node.base.id.clone(), 0));
            }
        }
        
        while let Some((node_id, level)) = queue.pop_front() {
            if visited.insert(node_id.clone()) {
                levels.insert(node_id.clone(), level);
                
                // Add children to queue
                for edge in &self.edges {
                    if edge.base.source == node_id && !visited.contains(&edge.base.target) {
                        queue.push_back((edge.base.target.clone(), level + 1));
                    }
                }
            }
        }
        
        levels
    }

    fn calculate_metrics(&self) -> GraphMetrics {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();
        
        let density = if node_count > 1 {
            2.0 * edge_count as f64 / (node_count * (node_count - 1)) as f64
        } else {
            0.0
        };
        
        let average_degree = if node_count > 0 {
            2.0 * edge_count as f64 / node_count as f64
        } else {
            0.0
        };
        
        // Calculate centrality measures (simplified)
        let mut centrality = HashMap::new();
        for node in &self.nodes {
            let degree = self.edges.iter()
                .filter(|e| e.base.source == node.base.id || e.base.target == node.base.id)
                .count() as f64;
            
            centrality.insert(node.base.id.clone(), NodeCentrality {
                degree: degree / (node_count - 1) as f64,
                betweenness: 0.0, // Would need proper calculation
                closeness: 0.0,   // Would need proper calculation
                eigenvector: 0.0, // Would need proper calculation
                pagerank: 1.0 / node_count as f64, // Simplified
            });
        }
        
        GraphMetrics {
            node_count,
            edge_count,
            density,
            average_degree,
            clustering_coefficient: 0.0, // Would need proper calculation
            connected_components: 1,     // Simplified
            diameter: None,              // Would need proper calculation
            centrality,
        }
    }

    fn detect_clusters(&self) -> Result<Vec<NodeCluster>> {
        // Simple clustering based on connectivity
        // In practice, this would use more sophisticated algorithms
        Ok(Vec::new())
    }

    fn pattern_color(&self, pattern: &Pattern) -> String {
        match pattern.pattern_type.as_str() {
            "linguistic" => "#3b82f6".to_string(),  // Blue
            "semantic" => "#10b981".to_string(),    // Green
            "syntactic" => "#f59e0b".to_string(),   // Amber
            "phonetic" => "#8b5cf6".to_string(),    // Violet
            _ => "#6b7280".to_string(),             // Gray
        }
    }

    fn meta_node_color(&self, node_type: &MetaNodeType) -> String {
        match node_type {
            MetaNodeType::Concept => "#3b82f6".to_string(),      // Blue
            MetaNodeType::Relationship => "#10b981".to_string(), // Green
            MetaNodeType::Inference => "#f59e0b".to_string(),    // Amber
            MetaNodeType::Hypothesis => "#8b5cf6".to_string(),   // Violet
            MetaNodeType::Pattern => "#ef4444".to_string(),      // Red
            MetaNodeType::Reflection => "#6b7280".to_string(),   // Gray
        }
    }

    fn meta_edge_color(&self, edge_type: &MetaEdgeType) -> String {
        match edge_type {
            MetaEdgeType::Causes => "#ef4444".to_string(),       // Red
            MetaEdgeType::PartOf => "#3b82f6".to_string(),       // Blue
            MetaEdgeType::InstanceOf => "#10b981".to_string(),   // Green
            MetaEdgeType::IsA => "#f59e0b".to_string(),          // Amber
            MetaEdgeType::Opposes => "#8b5cf6".to_string(),      // Violet
            MetaEdgeType::SimilarTo => "#06b6d4".to_string(),    // Cyan
            MetaEdgeType::Follows => "#84cc16".to_string(),      // Lime
            MetaEdgeType::ReflectsOn => "#6b7280".to_string(),   // Gray
        }
    }

    fn calculate_pattern_similarity(&self, p1: &Pattern, p2: &Pattern) -> f64 {
        // Simple similarity based on confidence and common elements
        let confidence_similarity = 1.0 - (p1.confidence - p2.confidence).abs();
        let element_overlap = p1.elements.iter()
            .filter(|e| p2.elements.contains(e))
            .count() as f64 / p1.elements.len().max(p2.elements.len()) as f64;
        
        (confidence_similarity + element_overlap) / 2.0
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            layout: NetworkLayout::ForceDirected,
            node_config: NodeConfig::default(),
            edge_config: EdgeConfig::default(),
            force_config: ForceConfig::default(),
            show_labels: true,
            show_tooltips: true,
            animation: AnimationConfig::default(),
            clustering: ClusteringConfig::default(),
        }
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            default_size: 10.0,
            size_range: (5.0, 30.0),
            default_color: "#3b82f6".to_string(),
            type_colors: HashMap::new(),
            shape: NodeShape::Circle,
            border_width: 1.0,
            border_color: "#ffffff".to_string(),
        }
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            default_width: 2.0,
            width_range: (1.0, 5.0),
            default_color: "#6b7280".to_string(),
            type_colors: HashMap::new(),
            style: EdgeStyle::Solid,
            arrow_size: 8.0,
            show_labels: false,
        }
    }
}

impl Default for ForceConfig {
    fn default() -> Self {
        Self {
            repulsion_strength: 100.0,
            attraction_strength: 0.1,
            gravity: 0.01,
            damping: 0.9,
            max_iterations: 1000,
            convergence_threshold: 0.01,
        }
    }
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            duration: 1000,
            easing: "ease-in-out".to_string(),
            animate_layout: true,
            animate_data: true,
        }
    }
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: ClusteringAlgorithm::Modularity,
            min_cluster_size: 3,
            cluster_colors: vec![
                "#ef4444".to_string(), // Red
                "#3b82f6".to_string(), // Blue
                "#10b981".to_string(), // Green
                "#f59e0b".to_string(), // Amber
                "#8b5cf6".to_string(), // Violet
                "#06b6d4".to_string(), // Cyan
            ],
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
} 