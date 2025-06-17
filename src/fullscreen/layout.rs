use std::collections::HashMap;
use anyhow::Result;
use uuid::Uuid;
use super::LayoutType;

pub struct LayoutManager {
    layout_type: LayoutType,
    positions: HashMap<Uuid, (f64, f64)>,
}

impl LayoutManager {
    pub fn new(layout_type: LayoutType) -> Self {
        Self {
            layout_type,
            positions: HashMap::new(),
        }
    }

    pub fn calculate_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        match self.layout_type {
            LayoutType::Force => self.calculate_force_layout(nodes),
            LayoutType::Hierarchical => self.calculate_hierarchical_layout(nodes),
            LayoutType::Circular => self.calculate_circular_layout(nodes),
            LayoutType::Grid => self.calculate_grid_layout(nodes),
            LayoutType::Custom => self.calculate_custom_layout(nodes),
        }
    }

    fn calculate_force_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        // Simple force-directed layout implementation
        for (i, &node_id) in nodes.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / nodes.len() as f64;
            let radius = 100.0;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            self.positions.insert(node_id, (x, y));
        }
        Ok(self.positions.clone())
    }

    fn calculate_hierarchical_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        // Simple hierarchical layout
        for (i, &node_id) in nodes.iter().enumerate() {
            let x = (i % 5) as f64 * 100.0;
            let y = (i / 5) as f64 * 100.0;
            self.positions.insert(node_id, (x, y));
        }
        Ok(self.positions.clone())
    }

    fn calculate_circular_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        // Circular layout
        for (i, &node_id) in nodes.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / nodes.len() as f64;
            let radius = 150.0;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            self.positions.insert(node_id, (x, y));
        }
        Ok(self.positions.clone())
    }

    fn calculate_grid_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        // Grid layout
        let grid_size = (nodes.len() as f64).sqrt().ceil() as usize;
        for (i, &node_id) in nodes.iter().enumerate() {
            let x = (i % grid_size) as f64 * 80.0;
            let y = (i / grid_size) as f64 * 80.0;
            self.positions.insert(node_id, (x, y));
        }
        Ok(self.positions.clone())
    }

    fn calculate_custom_layout(&mut self, nodes: &[Uuid]) -> Result<HashMap<Uuid, (f64, f64)>> {
        // Custom layout - use current positions or defaults
        for (i, &node_id) in nodes.iter().enumerate() {
            if !self.positions.contains_key(&node_id) {
                self.positions.insert(node_id, (i as f64 * 50.0, i as f64 * 50.0));
            }
        }
        Ok(self.positions.clone())
    }
} 