use std::collections::HashMap;
use crate::text_unit::{TextUnit, TextUnitType};

/// Text unit registry for managing units in a document
#[derive(Debug, Default)]
pub struct TextUnitRegistry {
    units: HashMap<usize, TextUnit>,
    next_id: usize,
}

impl TextUnitRegistry {
    /// Create a new text unit registry
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
            next_id: 0,
        }
    }
    
    /// Add a text unit to the registry
    pub fn add_unit(&mut self, mut unit: TextUnit) -> usize {
        let id = self.next_id;
        unit.id = id;
        self.units.insert(id, unit);
        self.next_id += 1;
        id
    }
    
    /// Get a text unit by ID
    pub fn get_unit(&self, id: usize) -> Option<&TextUnit> {
        self.units.get(&id)
    }
    
    /// Get a mutable reference to a text unit by ID
    pub fn get_unit_mut(&mut self, id: usize) -> Option<&mut TextUnit> {
        self.units.get_mut(&id)
    }
    
    /// Remove a text unit from the registry
    pub fn remove_unit(&mut self, id: usize) -> Option<TextUnit> {
        self.units.remove(&id)
    }
    
    /// Get the number of units in the registry
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }
    
    /// Get all units in the registry
    pub fn all_units(&self) -> Vec<&TextUnit> {
        self.units.values().collect()
    }
    
    /// Get all units of a specific type
    pub fn units_of_type(&self, unit_type: TextUnitType) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.unit_type == unit_type)
            .collect()
    }
    
    /// Get all root units (units without a parent)
    pub fn root_units(&self) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.parent_id.is_none())
            .collect()
    }
    
    /// Get all children of a unit
    pub fn children_of(&self, id: usize) -> Vec<&TextUnit> {
        if let Some(parent) = self.get_unit(id) {
            parent.children.iter()
                .filter_map(|&child_id| self.get_unit(child_id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Set up a parent-child relationship between units
    pub fn set_parent_child(&mut self, parent_id: usize, child_id: usize) -> bool {
        // Check that both units exist
        if !self.units.contains_key(&parent_id) || !self.units.contains_key(&child_id) {
            return false;
        }
        
        // Add child to parent
        if let Some(parent) = self.units.get_mut(&parent_id) {
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
        }
        
        // Set parent for child
        if let Some(child) = self.units.get_mut(&child_id) {
            child.parent_id = Some(parent_id);
        }
        
        true
    }
    
    /// Get the next available ID
    pub fn next_available_id(&self) -> usize {
        self.next_id
    }
} 