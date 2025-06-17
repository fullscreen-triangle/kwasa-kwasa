use std::collections::HashMap;
use petgraph::{Graph, Directed, graph::NodeIndex};
use anyhow::Result;
use uuid::Uuid;
use super::{ModuleNode, ModuleEdge};

pub struct GraphEngine {
    graph: Graph<ModuleNode, ModuleEdge, Directed>,
    node_index_map: HashMap<Uuid, NodeIndex>,
}

impl GraphEngine {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_index_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: ModuleNode) -> NodeIndex {
        let node_index = self.graph.add_node(node.clone());
        self.node_index_map.insert(node.id, node_index);
        node_index
    }

    pub fn add_edge(&mut self, from: Uuid, to: Uuid, edge: ModuleEdge) -> Result<()> {
        if let (Some(&from_idx), Some(&to_idx)) = 
            (self.node_index_map.get(&from), self.node_index_map.get(&to)) {
            self.graph.add_edge(from_idx, to_idx, edge);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Node not found"))
        }
    }

    pub fn get_graph(&self) -> &Graph<ModuleNode, ModuleEdge, Directed> {
        &self.graph
    }

    pub fn get_node_index(&self, id: &Uuid) -> Option<NodeIndex> {
        self.node_index_map.get(id).copied()
    }
} 