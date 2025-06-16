use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for text units
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextUnitId(u64);

impl TextUnitId {
    /// Create a new unique ID
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
    
    /// Get the raw ID value
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for TextUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextUnit({})", self.0)
    }
}

/// Types of text units that can be processed
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextUnitType {
    /// A complete document
    Document,
    /// A section within a document
    Section,
    /// A paragraph
    Paragraph,
    /// A sentence
    Sentence,
    /// A phrase or clause
    Phrase,
    /// A single word
    Word,
    /// A character or character sequence
    Character,
    /// Custom domain-specific unit
    Custom(String),
    /// Genomic sequence
    Sequence,
    /// Chemical formula
    Formula,
    /// Spectral data
    Spectrum,
    /// Code block
    Code,
    /// Mathematical expression
    Math,
    /// Table or structured data
    Table,
    /// List item
    ListItem,
    /// Header or title
    Header,
    /// Footer
    Footer,
    /// Citation or reference
    Citation,
}

impl fmt::Display for TextUnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextUnitType::Document => write!(f, "Document"),
            TextUnitType::Section => write!(f, "Section"),
            TextUnitType::Paragraph => write!(f, "Paragraph"),
            TextUnitType::Sentence => write!(f, "Sentence"),
            TextUnitType::Phrase => write!(f, "Phrase"),
            TextUnitType::Word => write!(f, "Word"),
            TextUnitType::Character => write!(f, "Character"),
            TextUnitType::Custom(name) => write!(f, "Custom({})", name),
            TextUnitType::Sequence => write!(f, "Sequence"),
            TextUnitType::Formula => write!(f, "Formula"),
            TextUnitType::Spectrum => write!(f, "Spectrum"),
            TextUnitType::Code => write!(f, "Code"),
            TextUnitType::Math => write!(f, "Math"),
            TextUnitType::Table => write!(f, "Table"),
            TextUnitType::ListItem => write!(f, "ListItem"),
            TextUnitType::Header => write!(f, "Header"),
            TextUnitType::Footer => write!(f, "Footer"),
            TextUnitType::Citation => write!(f, "Citation"),
        }
    }
}

/// Types of boundaries between text units
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryType {
    /// Hard boundary (definitive break)
    Hard,
    /// Soft boundary (suggested break)
    Soft,
    /// Semantic boundary (meaning-based)
    Semantic,
    /// Structural boundary (formatting-based)
    Structural,
    /// Domain-specific boundary
    Domain(String),
}

/// Information about a boundary between text units
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boundary {
    /// Position in the text
    pub position: usize,
    
    /// Type of boundary
    pub boundary_type: BoundaryType,
    
    /// Confidence in this boundary (0.0-1.0)
    pub confidence: f64,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// A unit of text with associated metadata and relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextUnit {
    /// Unique identifier
    pub id: TextUnitId,
    
    /// The actual text content
    pub content: String,
    
    /// Type of text unit
    pub unit_type: TextUnitType,
    
    /// Start position in the original text
    pub start_pos: usize,
    
    /// End position in the original text
    pub end_pos: usize,
    
    /// Metadata associated with this unit
    pub metadata: HashMap<String, String>,
    
    /// Child units (if this is a hierarchical unit)
    pub children: Vec<TextUnitId>,
    
    /// Parent unit (if this is part of a larger unit)
    pub parent: Option<TextUnitId>,
    
    /// Boundaries within this unit
    pub boundaries: Vec<Boundary>,
    
    /// Quality score of this text unit (0.0-1.0)
    pub quality_score: Option<f64>,
    
    /// Semantic tags associated with this unit
    pub semantic_tags: Vec<String>,
    
    /// Timestamp when this unit was created
    pub created_at: u64,
}

impl TextUnit {
    /// Create a new text unit
    pub fn new(
        content: String,
        start_pos: usize,
        end_pos: usize,
        unit_type: TextUnitType,
        hierarchy_level: usize,
    ) -> Self {
        Self {
            id: TextUnitId::new(),
            content,
            unit_type,
            start_pos,
            end_pos,
            metadata: HashMap::new(),
            children: Vec::new(),
            parent: None,
            boundaries: Vec::new(),
            quality_score: None,
            semantic_tags: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

impl TextUnit {
    /// Get the ID of this text unit
    pub fn id(&self) -> TextUnitId {
        self.id
    }
    
    /// Check if this unit contains the given position
    pub fn contains_position(&self, pos: usize) -> bool {
        pos >= self.start_pos && pos < self.end_pos
    }
    
    /// Check if this unit overlaps with the given range
    pub fn overlaps_with(&self, start: usize, end: usize) -> bool {
        !(self.end_pos <= start || self.start_pos >= end)
    }
    
    /// Get the parent units in order from immediate parent to root
    pub fn get_ancestry(&self, registry: &super::TextUnitRegistry) -> Vec<TextUnitId> {
        let mut ancestry = Vec::new();
        let mut current_parent = self.parent;
        
        while let Some(parent_id) = current_parent {
            ancestry.push(parent_id);
            if let Some(parent_unit) = registry.get_unit(parent_id) {
                current_parent = parent_unit.parent;
            } else {
                break;
            }
        }
        
        ancestry
    }
    
    /// Check if this unit is an ancestor of the given unit
    pub fn is_ancestor_of(&self, other: &TextUnit, registry: &super::TextUnitRegistry) -> bool {
        other.get_ancestry(registry).contains(&self.id)
    }
    
    /// Check if this unit is a descendant of the given unit
    pub fn is_descendant_of(&self, other: &TextUnit, registry: &super::TextUnitRegistry) -> bool {
        other.is_ancestor_of(self, registry)
    }
    
    /// Calculate a simple complexity score for this text unit (0.0-1.0)
    pub fn complexity(&self) -> f64 {
        // Simple complexity calculation based on sentence and word structure
        let sentences: Vec<&str> = self.content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
            
        let words: Vec<&str> = self.content.split_whitespace().collect();
        
        if sentences.is_empty() || words.is_empty() {
            return 0.0;
        }
        
        let avg_sentence_length = words.len() as f64 / sentences.len() as f64;
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / words.len() as f64;
        
        // Normalize complexity score (higher values = more complex)
        let complexity = ((avg_sentence_length - 10.0).abs() / 20.0 + 
                         (avg_word_length - 4.0).abs() / 10.0) / 2.0;
        complexity.max(0.0).min(1.0)
    }
    
    /// Calculate readability score for this text unit (0.0-1.0, higher = more readable)
    pub fn readability_score(&self) -> f64 {
        // Simplified Flesch Reading Ease calculation
        let sentences: Vec<&str> = self.content
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
            
        let words: Vec<&str> = self.content.split_whitespace().collect();
        
        if sentences.is_empty() || words.is_empty() {
            return 0.5; // Neutral score for empty content
        }
        
        let avg_sentence_length = words.len() as f64 / sentences.len() as f64;
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / words.len() as f64;
        
        // Simplified Flesch Reading Ease formula
        let raw_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length);
        
        // Convert to 0-1 scale where 1 is most readable
        (raw_score / 100.0).max(0.0).min(1.0)
    }
}

impl fmt::Display for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let content_preview = if self.content.len() > 50 {
            format!("{}...", &self.content[..47])
        } else {
            self.content.clone()
        };
        
        write!(
            f,
            "{} [{}:{}] '{}'",
            self.unit_type,
            self.start_pos,
            self.end_pos,
            content_preview
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_unit_id_generation() {
        let id1 = TextUnitId::new();
        let id2 = TextUnitId::new();
        
        assert_ne!(id1, id2);
        assert!(id1.as_u64() > 0);
        assert!(id2.as_u64() > 0);
    }
    
    #[test]
    fn test_text_unit_type_display() {
        assert_eq!(TextUnitType::Paragraph.to_string(), "Paragraph");
        assert_eq!(TextUnitType::Custom("test".to_string()).to_string(), "Custom(test)");
    }
    
    #[test]
    fn test_boundary_type() {
        let boundary = Boundary {
            position: 10,
            boundary_type: BoundaryType::Hard,
            confidence: 0.9,
            metadata: HashMap::new(),
        };
        
        assert_eq!(boundary.position, 10);
        assert_eq!(boundary.boundary_type, BoundaryType::Hard);
        assert_eq!(boundary.confidence, 0.9);
    }
} 