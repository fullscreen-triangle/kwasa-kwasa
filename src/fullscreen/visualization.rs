use std::collections::HashMap;
use std::path::Path;
use std::fs;
use anyhow::Result;
use petgraph::visit::EdgeRef;
use uuid::Uuid;

use super::{FullscreenSystem, ModuleType, ConnectionType, DataFlowType};

/// Generate SVG visualization of the network graph
pub async fn generate_svg_graph(system: &FullscreenSystem, output_path: &Path) -> Result<()> {
    let svg_content = build_svg_content(system).await?;
    fs::write(output_path, svg_content)?;
    Ok(())
}

/// Build the SVG content for the graph
async fn build_svg_content(system: &FullscreenSystem) -> Result<String> {
    let mut svg = String::new();
    
    // SVG header
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="800" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .module-core { fill: #ff6b6b; stroke: #d63031; }
      .module-processing { fill: #4ecdc4; stroke: #00b894; }
      .module-orchestrator { fill: #ffe66d; stroke: #fdcb6e; }
      .module-interface { fill: #a8e6cf; stroke: #00b894; }
      .module-storage { fill: #ffd93d; stroke: #f39c12; }
      .module-analysis { fill: #6c5ce7; stroke: #5f3dc4; }
      .module-external { fill: #fd79a8; stroke: #e84393; }
      
      .edge-semantic { stroke: #2d3436; stroke-width: 2; }
      .edge-dataflow { stroke: #0984e3; stroke-width: 2; }
      .edge-control { stroke: #e17055; stroke-width: 2; }
      .edge-feedback { stroke: #00b894; stroke-width: 2; stroke-dasharray: 5,5; }
      
      .module-text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
      .edge-label { font-family: Arial, sans-serif; font-size: 10px; fill: #636e72; }
    </style>
    
    <!-- Arrow markers for directed edges -->
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2d3436" />
    </marker>
    
    <marker id="arrowhead-blue" markerWidth="10" markerHeight="7" 
            refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#0984e3" />
    </marker>
    
    <marker id="arrowhead-red" markerWidth="10" markerHeight="7" 
            refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#e17055" />
    </marker>
  </defs>
"#);

    // Calculate layout positions using force-directed algorithm
    let positions = calculate_force_layout(system).await?;
    
    // Draw edges first (so they appear behind nodes)
    svg.push_str("  <!-- Edges -->\n");
    for edge_ref in system.graph.edge_references() {
        let edge = edge_ref.weight();
        let source_idx = edge_ref.source();
        let target_idx = edge_ref.target();
        
        if let (Some(source_node), Some(target_node)) = 
            (system.graph.node_weight(source_idx), system.graph.node_weight(target_idx)) {
            
            if let (Some(&source_pos), Some(&target_pos)) = 
                (positions.get(&source_node.id), positions.get(&target_node.id)) {
                
                let edge_svg = generate_edge_svg(edge, source_pos, target_pos);
                svg.push_str(&edge_svg);
            }
        }
    }
    
    // Draw nodes
    svg.push_str("  <!-- Nodes -->\n");
    for node_idx in system.graph.node_indices() {
        if let Some(node) = system.graph.node_weight(node_idx) {
            if let Some(&pos) = positions.get(&node.id) {
                let node_svg = generate_node_svg(node, pos);
                svg.push_str(&node_svg);
            }
        }
    }
    
    // Add legend
    svg.push_str(&generate_legend());
    
    // SVG footer
    svg.push_str("</svg>");
    
    Ok(svg)
}

/// Calculate force-directed layout positions for nodes
async fn calculate_force_layout(system: &FullscreenSystem) -> Result<HashMap<Uuid, (f64, f64)>> {
    let mut positions = HashMap::new();
    let node_count = system.graph.node_count() as f64;
    
    // Initial random positions
    for (i, node_idx) in system.graph.node_indices().enumerate() {
        if let Some(node) = system.graph.node_weight(node_idx) {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / node_count;
            let radius = 200.0 + node.semantic_weight * 100.0;
            
            let x = 600.0 + radius * angle.cos();
            let y = 400.0 + radius * angle.sin();
            
            positions.insert(node.id, (x, y));
        }
    }
    
    // Apply force-directed layout algorithm
    for _iteration in 0..100 {
        let mut forces: HashMap<Uuid, (f64, f64)> = HashMap::new();
        
        // Repulsive forces between all nodes
        for node_idx_a in system.graph.node_indices() {
            if let Some(node_a) = system.graph.node_weight(node_idx_a) {
                if let Some(&pos_a) = positions.get(&node_a.id) {
                    let mut total_force = (0.0, 0.0);
                    
                    for node_idx_b in system.graph.node_indices() {
                        if node_idx_a != node_idx_b {
                            if let Some(node_b) = system.graph.node_weight(node_idx_b) {
                                if let Some(&pos_b) = positions.get(&node_b.id) {
                                    let dx = pos_a.0 - pos_b.0;
                                    let dy = pos_a.1 - pos_b.1;
                                    let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                                    
                                    let force_magnitude = 1000.0 / (distance * distance);
                                    let force_x = (dx / distance) * force_magnitude;
                                    let force_y = (dy / distance) * force_magnitude;
                                    
                                    total_force.0 += force_x;
                                    total_force.1 += force_y;
                                }
                            }
                        }
                    }
                    
                    forces.insert(node_a.id, total_force);
                }
            }
        }
        
        // Attractive forces between connected nodes
        for edge_ref in system.graph.edge_references() {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();
            let edge_weight = edge_ref.weight().weight;
            
            if let (Some(source_node), Some(target_node)) = 
                (system.graph.node_weight(source_idx), system.graph.node_weight(target_idx)) {
                
                if let (Some(&pos_source), Some(&pos_target)) = 
                    (positions.get(&source_node.id), positions.get(&target_node.id)) {
                    
                    let dx = pos_target.0 - pos_source.0;
                    let dy = pos_target.1 - pos_source.1;
                    let distance = (dx * dx + dy * dy).sqrt().max(1.0);
                    
                    let force_magnitude = distance * edge_weight * 0.1;
                    let force_x = (dx / distance) * force_magnitude;
                    let force_y = (dy / distance) * force_magnitude;
                    
                    // Apply force to source node (towards target)
                    if let Some(force) = forces.get_mut(&source_node.id) {
                        force.0 += force_x;
                        force.1 += force_y;
                    }
                    
                    // Apply opposite force to target node
                    if let Some(force) = forces.get_mut(&target_node.id) {
                        force.0 -= force_x;
                        force.1 -= force_y;
                    }
                }
            }
        }
        
        // Update positions based on forces
        for (node_id, force) in forces {
            if let Some(pos) = positions.get_mut(&node_id) {
                let damping = 0.1;
                pos.0 += force.0 * damping;
                pos.1 += force.1 * damping;
                
                // Keep nodes within bounds
                pos.0 = pos.0.max(50.0).min(1150.0);
                pos.1 = pos.1.max(50.0).min(750.0);
            }
        }
    }
    
    Ok(positions)
}

/// Generate SVG for a single node
fn generate_node_svg(node: &super::ModuleNode, pos: (f64, f64)) -> String {
    let class = match node.module_type {
        ModuleType::Core => "module-core",
        ModuleType::Processing => "module-processing",
        ModuleType::Orchestrator => "module-orchestrator",
        ModuleType::Interface => "module-interface",
        ModuleType::Storage => "module-storage",
        ModuleType::Analysis => "module-analysis",
        ModuleType::External => "module-external",
    };
    
    let radius = 20.0 + node.semantic_weight * 15.0;
    
    format!(r#"  <g class="module-node">
    <circle cx="{}" cy="{}" r="{}" class="{}" opacity="0.8"/>
    <text x="{}" y="{}" class="module-text">{}</text>
    <title>{} ({})</title>
  </g>
"#, 
        pos.0, pos.1, radius, class,
        pos.0, pos.1 + 5.0, truncate_name(&node.name, 12),
        node.name, format!("{:?}", node.module_type)
    )
}

/// Generate SVG for a single edge
fn generate_edge_svg(
    edge: &super::ModuleEdge, 
    source_pos: (f64, f64), 
    target_pos: (f64, f64)
) -> String {
    let (class, marker) = match edge.connection_type {
        ConnectionType::SemanticDependency => ("edge-semantic", "arrowhead"),
        ConnectionType::DataFlow => ("edge-dataflow", "arrowhead-blue"),
        ConnectionType::Control => ("edge-control", "arrowhead-red"),
        ConnectionType::Feedback => ("edge-feedback", "arrowhead"),
        ConnectionType::Coordination => ("edge-semantic", "arrowhead"),
        ConnectionType::Inheritance => ("edge-semantic", "arrowhead"),
        ConnectionType::Composition => ("edge-semantic", "arrowhead"),
    };
    
    let line_style = match edge.data_flow {
        DataFlowType::Bidirectional => "stroke-dasharray: none;",
        DataFlowType::Broadcast => "stroke-dasharray: 3,3;",
        DataFlowType::Aggregation => "stroke-dasharray: 7,3;",
        DataFlowType::Transform => "stroke-dasharray: 2,2;",
        _ => "stroke-dasharray: none;",
    };
    
    format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" 
           class="{}" style="{}" marker-end="url(#{})"/>
"#,
        source_pos.0, source_pos.1, target_pos.0, target_pos.1,
        class, line_style, marker
    )
}

/// Generate legend for the visualization
fn generate_legend() -> String {
    r#"  <!-- Legend -->
  <g class="legend" transform="translate(50, 50)">
    <rect x="0" y="0" width="200" height="250" fill="white" stroke="#ddd" opacity="0.9"/>
    <text x="10" y="20" font-weight="bold">Module Types</text>
    
    <circle cx="20" cy="40" r="8" class="module-core"/>
    <text x="35" y="45" class="module-text" text-anchor="start">Core</text>
    
    <circle cx="20" cy="60" r="8" class="module-processing"/>
    <text x="35" y="65" class="module-text" text-anchor="start">Processing</text>
    
    <circle cx="20" cy="80" r="8" class="module-orchestrator"/>
    <text x="35" y="85" class="module-text" text-anchor="start">Orchestrator</text>
    
    <circle cx="20" cy="100" r="8" class="module-interface"/>
    <text x="35" y="105" class="module-text" text-anchor="start">Interface</text>
    
    <text x="10" y="140" font-weight="bold">Connection Types</text>
    
    <line x1="10" y1="155" x2="40" y2="155" class="edge-semantic" marker-end="url(#arrowhead)"/>
    <text x="45" y="160" class="edge-label" text-anchor="start">Semantic</text>
    
    <line x1="10" y1="175" x2="40" y2="175" class="edge-dataflow" marker-end="url(#arrowhead-blue)"/>
    <text x="45" y="180" class="edge-label" text-anchor="start">Data Flow</text>
    
    <line x1="10" y1="195" x2="40" y2="195" class="edge-control" marker-end="url(#arrowhead-red)"/>
    <text x="45" y="200" class="edge-label" text-anchor="start">Control</text>
    
    <line x1="10" y1="215" x2="40" y2="215" class="edge-feedback" marker-end="url(#arrowhead)"/>
    <text x="45" y="220" class="edge-label" text-anchor="start">Feedback</text>
  </g>
"#.to_string()
}

/// Export graph in DOT format for Graphviz
pub async fn export_dot_format(system: &FullscreenSystem, output_path: &Path) -> Result<()> {
    let mut dot_content = String::new();
    
    dot_content.push_str("digraph kwasa_kwasa {\n");
    dot_content.push_str("  rankdir=TB;\n");
    dot_content.push_str("  node [shape=circle, style=filled];\n");
    dot_content.push_str("  edge [fontsize=10];\n\n");
    
    // Add nodes
    for node_idx in system.graph.node_indices() {
        if let Some(node) = system.graph.node_weight(node_idx) {
            let color = match node.module_type {
                ModuleType::Core => "red",
                ModuleType::Processing => "lightblue",
                ModuleType::Orchestrator => "yellow",
                ModuleType::Interface => "lightgreen",
                ModuleType::Storage => "orange",
                ModuleType::Analysis => "purple",
                ModuleType::External => "pink",
            };
            
            dot_content.push_str(&format!(
                "  \"{}\" [label=\"{}\" fillcolor={} tooltip=\"{}\"];\n",
                node.id, node.name, color, node.path.display()
            ));
        }
    }
    
    dot_content.push_str("\n");
    
    // Add edges
    for edge_ref in system.graph.edge_references() {
        let edge = edge_ref.weight();
        let source_idx = edge_ref.source();
        let target_idx = edge_ref.target();
        
        if let (Some(source_node), Some(target_node)) = 
            (system.graph.node_weight(source_idx), system.graph.node_weight(target_idx)) {
            
            let edge_style = match edge.connection_type {
                ConnectionType::DataFlow => "color=blue",
                ConnectionType::Control => "color=red",
                ConnectionType::Feedback => "style=dashed",
                _ => "color=black",
            };
            
            dot_content.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{:.1}\" {}];\n",
                source_node.id, target_node.id, edge.weight, edge_style
            ));
        }
    }
    
    dot_content.push_str("}\n");
    
    fs::write(output_path, dot_content)?;
    Ok(())
}

/// Truncate a name to fit in the visualization
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("{}...", &name[..max_len-3])
    }
} 