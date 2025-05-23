pub mod boundary;
pub mod operations;
pub mod registry;
pub mod hierarchy;
pub mod utils;
pub mod transform;
pub mod advanced_processing;

// Re-export common types for convenience
pub use boundary::{UnitBoundary, BoundaryType};
pub use operations::{Operation, FilterPredicate};
pub use hierarchy::{HierarchyNode, HierarchyNodeType};
pub use transform::{TransformationPipeline, PipelineStage, TransformationMetrics};
pub use advanced_processing::{AdvancedTextProcessor, SemanticAnalysis, StyleAnalysis, ReadabilityMetrics};
pub use registry::TextUnitRegistry;

use std::collections::HashMap;
use std::fmt;

use crate::turbulance::ast::{Value, TextUnit as AstTextUnit};

/// Represents a text unit, which is a block of text with metadata and boundaries
#[derive(Debug, Clone, PartialEq)]
pub struct TextUnit {
    /// The content of the text unit
    pub content: String,
    
    /// Metadata about the text unit
    pub metadata: HashMap<String, Value>,
    
    /// Start position in the original document
    pub start: usize,
    
    /// End position in the original document
    pub end: usize,
    
    /// Unit type (paragraph, sentence, section, etc.)
    pub unit_type: TextUnitType,
    
    /// Parent unit ID (if part of a hierarchy)
    pub parent_id: Option<usize>,
    
    /// Children unit IDs (if has nested units)
    pub children: Vec<usize>,
    
    /// Unique identifier for this text unit
    pub id: usize,
}

/// Types of text units
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextUnitType {
    /// Full document
    Document,
    /// Section of document
    Section,
    /// Paragraph
    Paragraph,
    /// Sentence
    Sentence,
    /// Single word
    Word,
    /// Character
    Character,
    /// Custom type with ID
    Custom(usize),
}

impl TextUnit {
    /// Create a new text unit
    pub fn new(
        content: String, 
        start: usize, 
        end: usize, 
        unit_type: TextUnitType,
        id: usize
    ) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
            start,
            end,
            unit_type,
            parent_id: None,
            children: Vec::new(),
            id,
        }
    }
    
    /// Create a new text unit with metadata
    pub fn with_metadata(
        content: String, 
        metadata: HashMap<String, Value>, 
        start: usize, 
        end: usize, 
        unit_type: TextUnitType,
        id: usize
    ) -> Self {
        Self {
            content,
            metadata,
            start,
            end,
            unit_type,
            parent_id: None,
            children: Vec::new(),
            id,
        }
    }
    
    /// Add a child unit to this unit
    pub fn add_child(&mut self, child_id: usize) {
        self.children.push(child_id);
    }
    
    /// Set the parent unit for this unit
    pub fn set_parent(&mut self, parent_id: usize) {
        self.parent_id = Some(parent_id);
    }
    
    /// Add metadata to this unit
    pub fn add_metadata(&mut self, key: &str, value: Value) {
        self.metadata.insert(key.to_string(), value);
    }
    
    /// Get the length of the unit's content
    pub fn len(&self) -> usize {
        self.content.len()
    }
    
    /// Check if the unit is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
    
    /// Convert to an AST TextUnit
    pub fn to_ast_unit(&self) -> AstTextUnit {
        AstTextUnit {
            content: self.content.clone(),
            metadata: self.metadata.clone(),
        }
    }
    
    /// Create a new text unit from a string with a specific type
    pub fn from_string_with_type(content: String, unit_type: TextUnitType) -> Self {
        Self::new(content, 0, 0, unit_type, 0)
    }
    
    /// Calculate the text unit's complexity
    pub fn complexity(&self) -> f64 {
        // A simple complexity measure based on word count and average word length
        // We'll enhance this with more sophisticated metrics in the future
        
        let words: Vec<&str> = self.content.split_whitespace().collect();
        let word_count = words.len();
        
        if word_count == 0 {
            return 0.0;
        }
        
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / word_count as f64;
        
        // Complexity formula: normalize to a 0-1 scale
        // Higher values = more complex
        (word_count as f64 * 0.01).min(1.0) * (avg_word_length / 10.0).min(1.0)
    }
    
    /// Calculate the readability score of the text unit
    pub fn readability_score(&self) -> f64 {
        // Simple readability score based on average sentence length
        // and average word length (similar to Flesch Reading Ease)
        // We'll implement more sophisticated metrics in the future
        
        let sentences: Vec<&str> = self.content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
            
        let words: Vec<&str> = self.content.split_whitespace().collect();
        
        let sentence_count = sentences.len();
        let word_count = words.len();
        
        if sentence_count == 0 || word_count == 0 {
            return 100.0; // Perfect score for empty/simple text
        }
        
        let avg_sentence_length = word_count as f64 / sentence_count as f64;
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / word_count as f64;
        
        // Higher score = more readable
        // Scale is 0-100
        100.0 - (0.39 * avg_sentence_length + 11.8 * avg_word_length - 15.59)
    }
}

impl fmt::Display for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}-{}): '{}'", 
            self.unit_type, 
            self.start, 
            self.end, 
            if self.content.len() > 30 {
                format!("{}...", &self.content[0..30])
            } else {
                self.content.clone()
            }
        )
    }
}

impl fmt::Display for TextUnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextUnitType::Character => write!(f, "Character"),
            TextUnitType::Word => write!(f, "Word"),
            TextUnitType::Sentence => write!(f, "Sentence"),
            TextUnitType::Paragraph => write!(f, "Paragraph"),
            TextUnitType::Section => write!(f, "Section"),
            TextUnitType::Document => write!(f, "Document"),
            TextUnitType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

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
    
    /// Get the root units (units with no parent)
    pub fn root_units(&self) -> Vec<&TextUnit> {
        self.units.values()
            .filter(|unit| unit.parent_id.is_none())
            .collect()
    }
    
    /// Get the child units of a given unit
    pub fn children_of(&self, id: usize) -> Vec<&TextUnit> {
        if let Some(parent) = self.get_unit(id) {
            parent.children.iter()
                .filter_map(|&child_id| self.get_unit(child_id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Create a parent-child relationship between units
    pub fn set_parent_child(&mut self, parent_id: usize, child_id: usize) -> bool {
        // Check if both units exist
        if !self.units.contains_key(&parent_id) || !self.units.contains_key(&child_id) {
            return false;
        }
        
        // Add child to parent's children list
        if let Some(parent) = self.units.get_mut(&parent_id) {
            parent.add_child(child_id);
        }
        
        // Set parent for child
        if let Some(child) = self.units.get_mut(&child_id) {
            child.set_parent(parent_id);
        }
        
        true
    }
    
    /// Get the next available ID
    pub fn next_available_id(&self) -> usize {
        self.next_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_unit_creation() {
        let unit = TextUnit::new(
            "This is a test paragraph.".to_string(),
            0,
            25,
            TextUnitType::Paragraph,
            1
        );
        
        assert_eq!(unit.content, "This is a test paragraph.");
        assert_eq!(unit.start, 0);
        assert_eq!(unit.end, 25);
        assert_eq!(unit.unit_type, TextUnitType::Paragraph);
        assert_eq!(unit.id, 1);
        assert!(unit.parent_id.is_none());
        assert!(unit.children.is_empty());
        assert!(unit.metadata.is_empty());
    }
    
    #[test]
    fn test_text_unit_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), Value::String("en".to_string()));
        metadata.insert("sentiment".to_string(), Value::Number(0.75));
        
        let unit = TextUnit::with_metadata(
            "This is a test with metadata.".to_string(),
            metadata.clone(),
            0,
            30,
            TextUnitType::Paragraph,
            2
        );
        
        assert_eq!(unit.content, "This is a test with metadata.");
        assert_eq!(unit.metadata.len(), 2);
        
        match unit.metadata.get("language") {
            Some(Value::String(lang)) => assert_eq!(lang, "en"),
            _ => panic!("Expected language metadata"),
        }
    }
    
    #[test]
    fn test_text_unit_registry() {
        let mut registry = TextUnitRegistry::new();
        
        // Add some units
        let doc_id = registry.add_unit(TextUnit::new(
            "Full document".to_string(),
            0,
            100,
            TextUnitType::Document,
            0
        ));
        
        let para1_id = registry.add_unit(TextUnit::new(
            "Paragraph 1".to_string(),
            0,
            30,
            TextUnitType::Paragraph,
            0
        ));
        
        let para2_id = registry.add_unit(TextUnit::new(
            "Paragraph 2".to_string(),
            31,
            60,
            TextUnitType::Paragraph,
            0
        ));
        
        // Set up parent-child relationships
        assert!(registry.set_parent_child(doc_id, para1_id));
        assert!(registry.set_parent_child(doc_id, para2_id));
        
        // Check relationships
        let doc = registry.get_unit(doc_id).unwrap();
        assert_eq!(doc.children.len(), 2);
        assert!(doc.children.contains(&para1_id));
        assert!(doc.children.contains(&para2_id));
        
        let para1 = registry.get_unit(para1_id).unwrap();
        assert_eq!(para1.parent_id, Some(doc_id));
        
        let para2 = registry.get_unit(para2_id).unwrap();
        assert_eq!(para2.parent_id, Some(doc_id));
        
        // Check fetching by type
        let paragraphs = registry.units_of_type(TextUnitType::Paragraph);
        assert_eq!(paragraphs.len(), 2);
        
        // Check fetching children
        let children = registry.children_of(doc_id);
        assert_eq!(children.len(), 2);
        
        // Check root units
        let roots = registry.root_units();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, doc_id);
    }
    
    #[test]
    fn test_complexity_and_readability() {
        let simple_unit = TextUnit::new(
            "This is a simple test. It has short words.".to_string(),
            0,
            42,
            TextUnitType::Paragraph,
            1
        );
        
        let complex_unit = TextUnit::new(
            "The implementation of metacognitive orchestration within the framework necessitates sophisticated algorithms for contextual awareness and semantic understanding of multifaceted textual structures.".to_string(),
            0,
            200,
            TextUnitType::Paragraph,
            2
        );
        
        // Simple text should have lower complexity
        assert!(simple_unit.complexity() < complex_unit.complexity());
        
        // Simple text should have higher readability score
        assert!(simple_unit.readability_score() > complex_unit.readability_score());
    }
}
