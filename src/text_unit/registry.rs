use super::types::{TextUnit, TextUnitId, TextUnitType, BoundaryType};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Registry for managing text units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextUnitRegistry {
    /// Map of unit ID to text unit
    units: HashMap<TextUnitId, TextUnit>,
    
    /// Statistics about the registry
    stats: RegistryStats,
}

/// Statistics about the text unit registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    /// Total number of units
    pub total_units: usize,
    
    /// Units by type
    pub units_by_type: HashMap<String, usize>,
    
    /// Total content length
    pub total_content_length: usize,
    
    /// Average unit size
    pub average_unit_size: f64,
}

/// Options for boundary detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDetectionOptions {
    /// Minimum unit size for detection
    pub min_unit_size: usize,
    
    /// Maximum unit size for detection
    pub max_unit_size: usize,
    
    /// Confidence threshold for boundaries
    pub confidence_threshold: f64,
    
    /// Enable semantic boundary detection
    pub enable_semantic: bool,
    
    /// Enable structural boundary detection
    pub enable_structural: bool,
    
    /// Custom patterns for boundary detection
    pub custom_patterns: Vec<String>,
    
    /// Language-specific settings
    pub language: String,
}

impl Default for BoundaryDetectionOptions {
    fn default() -> Self {
        Self {
            min_unit_size: 1,
            max_unit_size: 10000,
            confidence_threshold: 0.7,
            enable_semantic: true,
            enable_structural: true,
            custom_patterns: Vec::new(),
            language: "en".to_string(),
        }
    }
}

impl TextUnitRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            units: HashMap::new(),
            stats: RegistryStats {
                total_units: 0,
                units_by_type: HashMap::new(),
                total_content_length: 0,
                average_unit_size: 0.0,
            },
        }
    }
    
    /// Add a text unit to the registry
    pub fn add_unit(&mut self, unit: TextUnit) -> TextUnitId {
        let id = unit.id();
        
        // Update statistics
        self.stats.total_units += 1;
        self.stats.total_content_length += unit.content.len();
        
        let type_name = unit.unit_type.to_string();
        *self.stats.units_by_type.entry(type_name).or_insert(0) += 1;
        
        self.stats.average_unit_size = 
            self.stats.total_content_length as f64 / self.stats.total_units as f64;
        
        self.units.insert(id, unit);
        id
    }
    
    /// Get a text unit by ID
    pub fn get_unit(&self, id: TextUnitId) -> Option<&TextUnit> {
        self.units.get(&id)
    }
    
    /// Get a mutable reference to a text unit by ID
    pub fn get_unit_mut(&mut self, id: TextUnitId) -> Option<&mut TextUnit> {
        self.units.get_mut(&id)
    }
    
    /// Remove a text unit from the registry
    pub fn remove_unit(&mut self, id: TextUnitId) -> Option<TextUnit> {
        if let Some(unit) = self.units.remove(&id) {
            // Update statistics
            self.stats.total_units -= 1;
            self.stats.total_content_length -= unit.content.len();
            
            let type_name = unit.unit_type.to_string();
            if let Some(count) = self.stats.units_by_type.get_mut(&type_name) {
                *count -= 1;
                if *count == 0 {
                    self.stats.units_by_type.remove(&type_name);
                }
            }
            
            if self.stats.total_units > 0 {
                self.stats.average_unit_size = 
                    self.stats.total_content_length as f64 / self.stats.total_units as f64;
            } else {
                self.stats.average_unit_size = 0.0;
            }
            
            Some(unit)
        } else {
            None
        }
    }
    
    /// Get all units of a specific type
    pub fn get_units_by_type(&self, unit_type: &TextUnitType) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| &unit.unit_type == unit_type)
            .collect()
    }
    
    /// Get all units that contain the given position
    pub fn get_units_at_position(&self, position: usize) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.contains_position(position))
            .collect()
    }
    
    /// Get all units that overlap with the given range
    pub fn get_units_in_range(&self, start: usize, end: usize) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.overlaps_with(start, end))
            .collect()
    }
    
    /// Get all root units (units with no parent)
    pub fn get_root_units(&self) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.parent.is_none())
            .collect()
    }
    
    /// Get all child units of a given unit
    pub fn get_children(&self, parent_id: TextUnitId) -> Vec<&TextUnit> {
        if let Some(parent) = self.get_unit(parent_id) {
            parent.children.iter()
                .filter_map(|&child_id| self.get_unit(child_id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get all descendant units of a given unit (recursive)
    pub fn get_descendants(&self, parent_id: TextUnitId) -> Vec<&TextUnit> {
        let mut descendants = Vec::new();
        let mut to_visit = vec![parent_id];
        
        while let Some(current_id) = to_visit.pop() {
            let children = self.get_children(current_id);
            for child in &children {
                descendants.push(*child);
                to_visit.push(child.id());
            }
        }
        
        descendants
    }
    
    /// Establish parent-child relationship between units
    pub fn set_parent_child(&mut self, parent_id: TextUnitId, child_id: TextUnitId) -> bool {
        // Check if both units exist
        if !self.units.contains_key(&parent_id) || !self.units.contains_key(&child_id) {
            return false;
        }
        
        // Add child to parent's children list
        if let Some(parent) = self.units.get_mut(&parent_id) {
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
        }
        
        // Set parent for child
        if let Some(child) = self.units.get_mut(&child_id) {
            child.parent = Some(parent_id);
        }
        
        true
    }
    
    /// Remove parent-child relationship
    pub fn remove_parent_child(&mut self, parent_id: TextUnitId, child_id: TextUnitId) -> bool {
        // Remove child from parent's children list
        if let Some(parent) = self.units.get_mut(&parent_id) {
            parent.children.retain(|&id| id != child_id);
        }
        
        // Remove parent from child
        if let Some(child) = self.units.get_mut(&child_id) {
            if child.parent == Some(parent_id) {
                child.parent = None;
                return true;
            }
        }
        
        false
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> &RegistryStats {
        &self.stats
    }
    
    /// Get the total number of units
    pub fn len(&self) -> usize {
        self.units.len()
    }
    
    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }
    
    /// Clear all units from the registry
    pub fn clear(&mut self) {
        self.units.clear();
        self.stats = RegistryStats {
            total_units: 0,
            units_by_type: HashMap::new(),
            total_content_length: 0,
            average_unit_size: 0.0,
        };
    }
    
    /// Get all unit IDs
    pub fn get_all_ids(&self) -> Vec<TextUnitId> {
        self.units.keys().copied().collect()
    }
    
    /// Get all units as a vector
    pub fn get_all_units(&self) -> Vec<&TextUnit> {
        self.units.values().collect()
    }
    
    /// Find units matching a predicate
    pub fn find_units<F>(&self, predicate: F) -> Vec<&TextUnit>
    where
        F: Fn(&TextUnit) -> bool,
    {
        self.units.values().filter(|unit| predicate(unit)).collect()
    }
    
    /// Search units by content (simple text search)
    pub fn search_content(&self, query: &str) -> Vec<&TextUnit> {
        let query_lower = query.to_lowercase();
        self.units.values()
            .filter(|unit| unit.content.to_lowercase().contains(&query_lower))
            .collect()
    }
    
    /// Get units with quality scores above a threshold
    pub fn get_high_quality_units(&self, threshold: f64) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| {
                unit.quality_score.map_or(false, |score| score >= threshold)
            })
            .collect()
    }
    
    /// Get units with specific semantic tags
    pub fn get_units_with_tag(&self, tag: &str) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.semantic_tags.contains(&tag.to_string()))
            .collect()
    }
    
    /// Validate the integrity of the registry
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        for (id, unit) in &self.units {
            // Check if unit ID matches map key
            if unit.id() != *id {
                issues.push(format!("Unit ID mismatch: map key {} vs unit ID {}", id, unit.id()));
            }
            
            // Check parent references
            if let Some(parent_id) = unit.parent {
                if !self.units.contains_key(&parent_id) {
                    issues.push(format!("Unit {} has invalid parent reference {}", id, parent_id));
                }
            }
            
            // Check child references
            for &child_id in &unit.children {
                if !self.units.contains_key(&child_id) {
                    issues.push(format!("Unit {} has invalid child reference {}", id, child_id));
                } else if let Some(child) = self.units.get(&child_id) {
                    if child.parent != Some(*id) {
                        issues.push(format!("Parent-child relationship inconsistency: {} -> {}", id, child_id));
                    }
                }
            }
            
            // Check position consistency
            if unit.start_pos >= unit.end_pos {
                issues.push(format!("Unit {} has invalid position range: {}-{}", id, unit.start_pos, unit.end_pos));
            }
        }
        
        issues
    }
}

impl Default for TextUnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_unit::types::TextUnitType;
    
    #[test]
    fn test_registry_creation() {
        let registry = TextUnitRegistry::new();
        
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
        assert_eq!(registry.get_stats().total_units, 0);
    }
    
    #[test]
    fn test_add_and_get_unit() {
        let mut registry = TextUnitRegistry::new();
        
        let unit = TextUnit::new(
            "Test content".to_string(),
            0,
            12,
            TextUnitType::Paragraph,
            0, // hierarchy_level
        );
        let unit_id = unit.id();
        
        let added_id = registry.add_unit(unit);
        assert_eq!(added_id, unit_id);
        
        let retrieved = registry.get_unit(unit_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "Test content");
        
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get_stats().total_units, 1);
    }
    
    #[test]
    fn test_parent_child_relationships() {
        let mut registry = TextUnitRegistry::new();
        
        let parent = TextUnit::new(
            "Parent content".to_string(),
            0,
            100,
            TextUnitType::Document,
            0, // hierarchy_level
        );
        let child = TextUnit::new(
            "Child content".to_string(),
            0,
            50,
            TextUnitType::Paragraph,
            1, // hierarchy_level
        );
        
        let parent_id = registry.add_unit(parent);
        let child_id = registry.add_unit(child);
        
        // Establish relationship
        assert!(registry.set_parent_child(parent_id, child_id));
        
        // Check relationship
        let parent_unit = registry.get_unit(parent_id).unwrap();
        let child_unit = registry.get_unit(child_id).unwrap();
        
        assert!(parent_unit.children.contains(&child_id));
        assert_eq!(child_unit.parent, Some(parent_id));
        
        // Test getting children
        let children = registry.get_children(parent_id);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id(), child_id);
    }
    
    #[test]
    fn test_search_functions() {
        let mut registry = TextUnitRegistry::new();
        
        let unit1 = TextUnit::new(
            "This is a test sentence".to_string(),
            0,
            23,
            TextUnitType::Sentence,
            0, // hierarchy_level
        );
        let unit2 = TextUnit::new(
            "Another example paragraph".to_string(),
            24,
            49,
            TextUnitType::Paragraph,
            0, // hierarchy_level
        );
        
        registry.add_unit(unit1);
        registry.add_unit(unit2);
        
        // Test content search
        let results = registry.search_content("test");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "This is a test sentence");
        
        // Test search by type
        let sentences = registry.get_units_by_type(&TextUnitType::Sentence);
        assert_eq!(sentences.len(), 1);
        
        let paragraphs = registry.get_units_by_type(&TextUnitType::Paragraph);
        assert_eq!(paragraphs.len(), 1);
    }
    
    #[test]
    fn test_position_queries() {
        let mut registry = TextUnitRegistry::new();
        
        let unit = TextUnit::new(
            "Hello world".to_string(),
            10,
            21,
            TextUnitType::Sentence,
            0, // hierarchy_level
        );
        
        registry.add_unit(unit);
        
        // Test position containment
        let units_at_15 = registry.get_units_at_position(15);
        assert_eq!(units_at_15.len(), 1);
        
        let units_at_5 = registry.get_units_at_position(5);
        assert_eq!(units_at_5.len(), 0);
        
        // Test range overlap
        let units_in_range = registry.get_units_in_range(5, 15);
        assert_eq!(units_in_range.len(), 1);
        
        let units_in_range2 = registry.get_units_in_range(25, 30);
        assert_eq!(units_in_range2.len(), 0);
    }
    
    #[test]
    fn test_registry_validation() {
        let mut registry = TextUnitRegistry::new();
        
        let unit = TextUnit::new(
            "Valid unit".to_string(),
            0,
            10,
            TextUnitType::Paragraph,
            0, // hierarchy_level
        );
        
        registry.add_unit(unit);
        
        let issues = registry.validate();
        assert!(issues.is_empty(), "Registry should be valid: {:?}", issues);
    }
} 