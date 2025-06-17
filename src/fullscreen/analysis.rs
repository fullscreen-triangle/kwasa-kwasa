use std::collections::HashMap;
use petgraph::Graph;
use petgraph::algo::{dijkstra, connected_components};
use anyhow::Result;
use uuid::Uuid;
use super::{ModuleNode, ModuleEdge, GraphAnalytics};

pub struct GraphAnalyzer;

impl GraphAnalyzer {
    pub fn analyze_graph(graph: &Graph<ModuleNode, ModuleEdge, petgraph::Directed>) -> GraphAnalytics {
        let total_nodes = graph.node_count();
        let total_edges = graph.edge_count();
        
        let connected_components = connected_components(graph);
        
        let average_degree = if total_nodes > 0 {
            (total_edges * 2) as f64 / total_nodes as f64
        } else {
            0.0
        };
        
        let density = if total_nodes > 1 {
            total_edges as f64 / (total_nodes * (total_nodes - 1) / 2) as f64
        } else {
            0.0
        };
        
        let most_connected_modules = Self::find_most_connected_modules(graph);
        let critical_paths = Self::find_critical_paths(graph);
        
        GraphAnalytics {
            total_nodes,
            total_edges,
            connected_components,
            average_degree,
            density,
            most_connected_modules,
            critical_paths,
        }
    }
    
    fn find_most_connected_modules(graph: &Graph<ModuleNode, ModuleEdge, petgraph::Directed>) -> Vec<String> {
        let mut connections: Vec<(String, usize)> = graph
            .node_indices()
            .map(|idx| {
                let node = &graph[idx];
                let degree = graph.edges(idx).count() + graph.edges_directed(idx, petgraph::Incoming).count();
                (node.name.clone(), degree)
            })
            .collect();
        
        connections.sort_by(|a, b| b.1.cmp(&a.1));
        connections.into_iter().take(5).map(|(name, _)| name).collect()
    }
    
    fn find_critical_paths(graph: &Graph<ModuleNode, ModuleEdge, petgraph::Directed>) -> Vec<Vec<String>> {
        // Simplified critical path finding - in a real implementation, this would be more sophisticated
        let mut paths = Vec::new();
        
        for node_idx in graph.node_indices() {
            let node = &graph[node_idx];
            let distances = dijkstra(graph, node_idx, None, |edge| edge.weight as i32);
            
            if distances.len() > 3 {
                let mut path: Vec<String> = distances
                    .keys()
                    .take(4)
                    .map(|&idx| graph[idx].name.clone())
                    .collect();
                
                if !path.is_empty() {
                    paths.push(path);
                }
            }
        }
        
        paths.into_iter().take(3).collect()
    }
    
    pub fn find_semantic_path(
        graph: &Graph<ModuleNode, ModuleEdge, petgraph::Directed>,
        node_map: &HashMap<Uuid, petgraph::graph::NodeIndex>,
        from: Uuid,
        to: Uuid,
    ) -> Option<Vec<Uuid>> {
        if let (Some(&from_idx), Some(&to_idx)) = (node_map.get(&from), node_map.get(&to)) {
            let distances = dijkstra(graph, from_idx, Some(to_idx), |edge| edge.weight as i32);
            
            if distances.contains_key(&to_idx) {
                // Reconstruct path - simplified implementation
                Some(vec![from, to])
            } else {
                None
            }
        } else {
            None
        }
    }
} 