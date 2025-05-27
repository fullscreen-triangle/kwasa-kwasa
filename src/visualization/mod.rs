//! Visualization Components for Kwasa-Kwasa Framework
//! 
//! This module provides comprehensive visualization capabilities for text analysis,
//! pattern recognition, scientific data, and metacognitive reasoning results.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::text_unit::TextUnit;
use crate::pattern::{Pattern, MetaNode, MetaEdge};

pub mod charts;
pub mod graphs;
pub mod scientific;
pub mod text;

/// Re-exports for easy access
pub mod prelude {
    pub use super::{
        Visualization, VisualizationType, ChartConfig, GraphConfig,
        TextVisualization, ScientificVisualization, NetworkVisualization,
        VisualizationRenderer, HtmlRenderer, SvgRenderer, JsonRenderer
    };
    pub use super::charts::*;
    pub use super::graphs::*;
    pub use super::scientific::*;
    pub use super::text::*;
}

/// Main visualization structure that can represent various types of visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Visualization {
    /// Unique identifier for the visualization
    pub id: String,
    /// Title of the visualization
    pub title: String,
    /// Description of what this visualization shows
    pub description: String,
    /// Type of visualization
    pub visualization_type: VisualizationType,
    /// Data for the visualization
    pub data: VisualizationData,
    /// Configuration for rendering
    pub config: VisualizationConfig,
    /// Metadata about the visualization
    pub metadata: HashMap<String, String>,
}

/// Types of visualizations supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisualizationType {
    /// Line chart
    LineChart,
    /// Bar chart
    BarChart,
    /// Scatter plot
    ScatterPlot,
    /// Histogram
    Histogram,
    /// Heatmap
    Heatmap,
    /// Network graph
    NetworkGraph,
    /// Tree diagram
    TreeDiagram,
    /// Text visualization (word cloud, text structure, etc.)
    TextVisualization,
    /// Scientific data visualization (genomic, spectral, etc.)
    ScientificVisualization,
    /// Dashboard with multiple visualizations
    Dashboard,
}

/// Data for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationData {
    /// Time series data
    TimeSeries(Vec<TimeSeriesPoint>),
    /// Categorical data
    Categorical(Vec<CategoryData>),
    /// Network data
    Network(NetworkData),
    /// Text analysis data
    Text(TextAnalysisData),
    /// Scientific data
    Scientific(ScientificData),
    /// Custom data structure
    Custom(serde_json::Value),
}

/// Configuration for visualization rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Color scheme to use
    pub color_scheme: ColorScheme,
    /// Whether to show legend
    pub show_legend: bool,
    /// Whether to show grid
    pub show_grid: bool,
    /// Font size
    pub font_size: u32,
    /// Additional styling options
    pub style_options: HashMap<String, String>,
}

/// Color schemes for visualizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default blue theme
    Default,
    /// Scientific publication theme
    Scientific,
    /// High contrast for accessibility
    HighContrast,
    /// Colorblind-friendly palette
    ColorblindFriendly,
    /// Custom color palette
    Custom(Vec<String>),
}

/// Point in a time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    /// X-axis value (typically time)
    pub x: f64,
    /// Y-axis value
    pub y: f64,
    /// Optional label
    pub label: Option<String>,
}

/// Categorical data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryData {
    /// Category name
    pub category: String,
    /// Value for this category
    pub value: f64,
    /// Optional color override
    pub color: Option<String>,
}

/// Network visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkData {
    /// Nodes in the network
    pub nodes: Vec<NetworkNode>,
    /// Edges connecting nodes
    pub edges: Vec<NetworkEdge>,
    /// Layout algorithm to use
    pub layout: NetworkLayout,
}

/// Node in a network visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    /// Unique identifier
    pub id: String,
    /// Display label
    pub label: String,
    /// X position (if using fixed layout)
    pub x: Option<f64>,
    /// Y position (if using fixed layout)
    pub y: Option<f64>,
    /// Size of the node
    pub size: f64,
    /// Color of the node
    pub color: Option<String>,
    /// Node type for styling
    pub node_type: Option<String>,
}

/// Edge in a network visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge weight
    pub weight: f64,
    /// Edge label
    pub label: Option<String>,
    /// Edge color
    pub color: Option<String>,
    /// Edge type for styling
    pub edge_type: Option<String>,
}

/// Network layout algorithms
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NetworkLayout {
    /// Force-directed layout
    ForceDirected,
    /// Hierarchical layout
    Hierarchical,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
    /// Fixed positions (use node x,y coordinates)
    Fixed,
}

/// Text analysis visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextAnalysisData {
    /// Word frequency data
    pub word_frequencies: HashMap<String, u32>,
    /// Sentence structure data
    pub sentence_structures: Vec<SentenceStructure>,
    /// Reading difficulty progression
    pub difficulty_progression: Vec<f64>,
    /// Topic distribution
    pub topic_distribution: HashMap<String, f64>,
}

/// Structure information for a sentence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceStructure {
    /// Sentence text
    pub text: String,
    /// Length in words
    pub word_count: usize,
    /// Complexity score
    pub complexity: f64,
    /// Position in document
    pub position: usize,
}

/// Scientific data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificData {
    /// Genomic data
    pub genomic: Option<GenomicVisualizationData>,
    /// Spectral data
    pub spectral: Option<SpectralVisualizationData>,
    /// Chemical structure data
    pub chemical: Option<ChemicalVisualizationData>,
}

/// Genomic visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicVisualizationData {
    /// Sequence coverage data
    pub coverage: Vec<f64>,
    /// Variant positions and types
    pub variants: Vec<VariantVisualization>,
    /// Gene annotations
    pub genes: Vec<GeneVisualization>,
}

/// Variant for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantVisualization {
    /// Position
    pub position: usize,
    /// Variant type
    pub variant_type: String,
    /// Quality score
    pub quality: f64,
}

/// Gene for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneVisualization {
    /// Gene name
    pub name: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Strand
    pub strand: i8,
    /// Expression level (if available)
    pub expression: Option<f64>,
}

/// Spectral visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralVisualizationData {
    /// Mass-to-charge ratios
    pub mz_values: Vec<f64>,
    /// Intensity values
    pub intensities: Vec<f64>,
    /// Peak annotations
    pub peak_annotations: Vec<PeakAnnotation>,
}

/// Peak annotation for spectral data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakAnnotation {
    /// M/z value
    pub mz: f64,
    /// Intensity
    pub intensity: f64,
    /// Annotation text
    pub annotation: String,
}

/// Chemical structure visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalVisualizationData {
    /// SMILES representation
    pub smiles: String,
    /// Atom coordinates (if available)
    pub atom_coordinates: Vec<(f64, f64)>,
    /// Bond information
    pub bonds: Vec<ChemicalBond>,
}

/// Chemical bond for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalBond {
    /// First atom index
    pub atom1: usize,
    /// Second atom index
    pub atom2: usize,
    /// Bond type (1=single, 2=double, 3=triple)
    pub bond_type: u8,
}

/// Trait for rendering visualizations
pub trait VisualizationRenderer {
    /// Render visualization to string format
    fn render(&self, visualization: &Visualization) -> Result<String>;
    
    /// Get the MIME type for this renderer
    fn mime_type(&self) -> &'static str;
    
    /// Get file extension for this renderer
    fn file_extension(&self) -> &'static str;
}

/// HTML renderer for web display
pub struct HtmlRenderer {
    /// Include interactive features
    pub interactive: bool,
    /// JavaScript libraries to include
    pub js_libraries: Vec<String>,
}

impl HtmlRenderer {
    /// Create a new HTML renderer
    pub fn new() -> Self {
        Self {
            interactive: true,
            js_libraries: vec!["d3".to_string(), "plotly".to_string()],
        }
    }
}

impl VisualizationRenderer for HtmlRenderer {
    fn render(&self, visualization: &Visualization) -> Result<String> {
        let mut html = String::new();
        
        // HTML header
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", visualization.title));
        
        // Include JavaScript libraries
        for lib in &self.js_libraries {
            match lib.as_str() {
                "d3" => html.push_str("<script src=\"https://d3js.org/d3.v7.min.js\"></script>\n"),
                "plotly" => html.push_str("<script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>\n"),
                _ => {}
            }
        }
        
        html.push_str("</head>\n<body>\n");
        
        // Visualization container
        html.push_str(&format!(
            "<div id=\"viz-{}\" style=\"width:{}px; height:{}px;\"></div>\n",
            visualization.id, visualization.config.width, visualization.config.height
        ));
        
        // Generate visualization code based on type
        html.push_str("<script>\n");
        match visualization.visualization_type {
            VisualizationType::LineChart => {
                html.push_str(&self.render_line_chart(visualization)?);
            }
            VisualizationType::NetworkGraph => {
                html.push_str(&self.render_network_graph(visualization)?);
            }
            // Add other visualization types...
            _ => {
                html.push_str("console.log('Visualization type not yet implemented');");
            }
        }
        html.push_str("</script>\n");
        
        html.push_str("</body>\n</html>");
        
        Ok(html)
    }
    
    fn mime_type(&self) -> &'static str {
        "text/html"
    }
    
    fn file_extension(&self) -> &'static str {
        "html"
    }
}

impl HtmlRenderer {
    fn render_line_chart(&self, visualization: &Visualization) -> Result<String> {
        if let VisualizationData::TimeSeries(points) = &visualization.data {
            let data_json = serde_json::to_string(points)
                .map_err(|e| Error::visualization(format!("Failed to serialize data: {}", e)))?;
            
            Ok(format!(
                r#"
                var data = {};
                var trace = {{
                    x: data.map(d => d.x),
                    y: data.map(d => d.y),
                    type: 'scatter',
                    mode: 'lines+markers'
                }};
                var layout = {{
                    title: '{}',
                    xaxis: {{ title: 'X Axis' }},
                    yaxis: {{ title: 'Y Axis' }}
                }};
                Plotly.newPlot('viz-{}', [trace], layout);
                "#,
                data_json, visualization.title, visualization.id
            ))
        } else {
            Err(Error::visualization("Invalid data type for line chart"))
        }
    }
    
    fn render_network_graph(&self, visualization: &Visualization) -> Result<String> {
        if let VisualizationData::Network(network) = &visualization.data {
            let nodes_json = serde_json::to_string(&network.nodes)
                .map_err(|e| Error::visualization(format!("Failed to serialize nodes: {}", e)))?;
            let edges_json = serde_json::to_string(&network.edges)
                .map_err(|e| Error::visualization(format!("Failed to serialize edges: {}", e)))?;
            
            Ok(format!(
                r#"
                var nodes = {};
                var edges = {};
                var svg = d3.select('#viz-{}').append('svg')
                    .attr('width', {})
                    .attr('height', {});
                
                var simulation = d3.forceSimulation(nodes)
                    .force('link', d3.forceLink(edges).id(d => d.id))
                    .force('charge', d3.forceManyBody())
                    .force('center', d3.forceCenter({} / 2, {} / 2));
                
                var link = svg.append('g')
                    .selectAll('line')
                    .data(edges)
                    .enter().append('line')
                    .attr('stroke', '#999')
                    .attr('stroke-opacity', 0.6);
                
                var node = svg.append('g')
                    .selectAll('circle')
                    .data(nodes)
                    .enter().append('circle')
                    .attr('r', d => d.size || 5)
                    .attr('fill', d => d.color || '#69b3a2');
                
                simulation.on('tick', () => {{
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    node
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                }});
                "#,
                nodes_json, edges_json, visualization.id,
                visualization.config.width, visualization.config.height,
                visualization.config.width, visualization.config.height
            ))
        } else {
            Err(Error::visualization("Invalid data type for network graph"))
        }
    }
}

/// SVG renderer for scalable vector graphics
pub struct SvgRenderer;

impl SvgRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl VisualizationRenderer for SvgRenderer {
    fn render(&self, visualization: &Visualization) -> Result<String> {
        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            visualization.config.width, visualization.config.height
        );
        
        // Add title
        svg.push_str(&format!(
            r#"<text x="10" y="20" font-size="16" font-weight="bold">{}</text>"#,
            visualization.title
        ));
        
        // Generate visualization based on type
        match visualization.visualization_type {
            VisualizationType::BarChart => {
                svg.push_str(&self.render_bar_chart(visualization)?);
            }
            _ => {
                svg.push_str(r#"<text x="10" y="50">Visualization type not implemented</text>"#);
            }
        }
        
        svg.push_str("</svg>");
        Ok(svg)
    }
    
    fn mime_type(&self) -> &'static str {
        "image/svg+xml"
    }
    
    fn file_extension(&self) -> &'static str {
        "svg"
    }
}

impl SvgRenderer {
    fn render_bar_chart(&self, visualization: &Visualization) -> Result<String> {
        if let VisualizationData::Categorical(categories) = &visualization.data {
            let mut bars = String::new();
            let bar_width = 40;
            let max_value = categories.iter().map(|c| c.value).fold(0.0, f64::max);
            let scale = (visualization.config.height - 100) as f64 / max_value;
            
            for (i, category) in categories.iter().enumerate() {
                let x = 50 + i * (bar_width + 10);
                let height = (category.value * scale) as u32;
                let y = visualization.config.height - height - 50;
                
                bars.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}", fill="#69b3a2"/>"#,
                    x, y, bar_width, height
                ));
                
                bars.push_str(&format!(
                    r#"<text x="{}" y="{}" font-size="12" text-anchor="middle">{}</text>"#,
                    x + bar_width / 2, visualization.config.height - 30, category.category
                ));
            }
            
            Ok(bars)
        } else {
            Err(Error::visualization("Invalid data type for bar chart"))
        }
    }
}

/// JSON renderer for data export
pub struct JsonRenderer;

impl JsonRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl VisualizationRenderer for JsonRenderer {
    fn render(&self, visualization: &Visualization) -> Result<String> {
        serde_json::to_string_pretty(visualization)
            .map_err(|e| Error::visualization(format!("Failed to serialize visualization: {}", e)))
    }
    
    fn mime_type(&self) -> &'static str {
        "application/json"
    }
    
    fn file_extension(&self) -> &'static str {
        "json"
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            color_scheme: ColorScheme::Default,
            show_legend: true,
            show_grid: true,
            font_size: 12,
            style_options: HashMap::new(),
        }
    }
}

impl Default for HtmlRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SvgRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for JsonRenderer {
    fn default() -> Self {
        Self::new()
    }
} 