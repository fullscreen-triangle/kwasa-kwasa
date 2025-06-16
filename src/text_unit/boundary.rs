use super::types::{TextUnit, TextUnitId, TextUnitType, BoundaryType, Boundary};
use super::registry::{TextUnitRegistry, BoundaryDetectionOptions};
use std::collections::HashMap;

/// Detect paragraph boundaries in text
pub fn detect_paragraph_boundaries(
    text: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<TextUnitId> {
    let mut unit_ids = Vec::new();
    
    // Split by double newlines (common paragraph separator)
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut current_pos = 0;
    
    for paragraph in paragraphs {
        if paragraph.trim().is_empty() {
            current_pos += paragraph.len() + 2; // +2 for \n\n
            continue;
        }
        
        let start_pos = current_pos;
        let end_pos = current_pos + paragraph.len();
        
        if end_pos - start_pos >= options.min_unit_size && end_pos - start_pos <= options.max_unit_size {
            let mut unit = TextUnit::new(
                paragraph.trim().to_string(),
                start_pos,
                end_pos,
                TextUnitType::Paragraph,
                0, // hierarchy_level
            );
            
            // Add boundary information
            if options.enable_structural {
                let boundary = Boundary {
                    position: start_pos,
                    boundary_type: BoundaryType::Structural,
                    confidence: 0.9,
                    metadata: HashMap::new(),
                };
                unit.boundaries.push(boundary);
            }
            
            let unit_id = registry.add_unit(unit);
            unit_ids.push(unit_id);
        }
        
        current_pos = end_pos + 2; // +2 for \n\n separator
    }
    
    unit_ids
}

/// Detect sentence boundaries in text
pub fn detect_sentence_boundaries(
    text: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<TextUnitId> {
    let mut unit_ids = Vec::new();
    
    // Simple sentence detection using common punctuation
    let sentence_endings = ['.', '!', '?'];
    let mut sentence_start = 0;
    
    let chars: Vec<char> = text.chars().collect();
    
    for (i, &ch) in chars.iter().enumerate() {
        if sentence_endings.contains(&ch) {
            // Check if this is likely the end of a sentence
            let is_sentence_end = if i + 1 < chars.len() {
                let next_char = chars[i + 1];
                next_char.is_whitespace() || next_char == '\n'
            } else {
                true // End of text
            };
            
            if is_sentence_end {
                let sentence_text = chars[sentence_start..=i].iter().collect::<String>();
                let trimmed = sentence_text.trim();
                
                if !trimmed.is_empty() && 
                   trimmed.len() >= options.min_unit_size && 
                   trimmed.len() <= options.max_unit_size {
                    
                    let mut unit = TextUnit::new(
                        trimmed.to_string(),
                        sentence_start,
                        i + 1,
                        TextUnitType::Sentence,
                        0, // hierarchy_level
                    );
                    
                    // Add boundary information
                    if options.enable_structural {
                        let boundary = Boundary {
                            position: sentence_start,
                            boundary_type: BoundaryType::Structural,
                            confidence: 0.8,
                            metadata: HashMap::new(),
                        };
                        unit.boundaries.push(boundary);
                    }
                    
                    let unit_id = registry.add_unit(unit);
                    unit_ids.push(unit_id);
                }
                
                // Find start of next sentence (skip whitespace)
                sentence_start = i + 1;
                while sentence_start < chars.len() && chars[sentence_start].is_whitespace() {
                    sentence_start += 1;
                }
            }
        }
    }
    
    // Handle any remaining text as a sentence
    if sentence_start < chars.len() {
        let remaining_text = chars[sentence_start..].iter().collect::<String>();
        let trimmed = remaining_text.trim();
        
        if !trimmed.is_empty() && 
           trimmed.len() >= options.min_unit_size && 
           trimmed.len() <= options.max_unit_size {
            
            let unit = TextUnit::new(
                trimmed.to_string(),
                sentence_start,
                chars.len(),
                TextUnitType::Sentence,
                0, // hierarchy_level
            );
            
            let unit_id = registry.add_unit(unit);
            unit_ids.push(unit_id);
        }
    }
    
    unit_ids
}

/// Detect word boundaries in text
pub fn detect_word_boundaries(
    text: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<TextUnitId> {
    let mut unit_ids = Vec::new();
    
    // Split by whitespace and punctuation
    let mut word_start = 0;
    let mut in_word = false;
    
    let chars: Vec<char> = text.chars().collect();
    
    for (i, &ch) in chars.iter().enumerate() {
        if ch.is_alphabetic() || ch.is_numeric() {
            if !in_word {
                word_start = i;
                in_word = true;
            }
        } else {
            if in_word {
                let word_text = chars[word_start..i].iter().collect::<String>();
                
                if word_text.len() >= options.min_unit_size && word_text.len() <= options.max_unit_size {
                    let mut unit = TextUnit::new(
                        word_text,
                        word_start,
                        i,
                        TextUnitType::Word,
                        0, // hierarchy_level
                    );
                    
                    // Add boundary information
                    if options.enable_structural {
                        let boundary = Boundary {
                            position: word_start,
                            boundary_type: BoundaryType::Structural,
                            confidence: 0.95,
                            metadata: HashMap::new(),
                        };
                        unit.boundaries.push(boundary);
                    }
                    
                    let unit_id = registry.add_unit(unit);
                    unit_ids.push(unit_id);
                }
                
                in_word = false;
            }
        }
    }
    
    // Handle word at end of text
    if in_word {
        let word_text = chars[word_start..].iter().collect::<String>();
        
        if word_text.len() >= options.min_unit_size && word_text.len() <= options.max_unit_size {
            let unit = TextUnit::new(
                word_text,
                word_start,
                chars.len(),
                TextUnitType::Word,
                0, // hierarchy_level
            );
            
            let unit_id = registry.add_unit(unit);
            unit_ids.push(unit_id);
        }
    }
    
    unit_ids
}

/// Legacy boundary types for compatibility
#[derive(Debug, Clone, PartialEq)]
pub enum UnitBoundary {
    /// Start boundary
    Start,
    /// End boundary  
    End,
    /// Section boundary
    Section,
    /// Paragraph boundary
    Paragraph,
    /// Sentence boundary
    Sentence,
    /// Word boundary
    Word,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_paragraph_detection() {
        let mut registry = TextUnitRegistry::new();
        let options = BoundaryDetectionOptions::default();
        
        let text = "This is the first paragraph.\n\nThis is the second paragraph.\n\nAnd this is the third.";
        
        let unit_ids = detect_paragraph_boundaries(text, &mut registry, &options);
        
        assert_eq!(unit_ids.len(), 3);
        
        let first_unit = registry.get_unit(unit_ids[0]).unwrap();
        assert_eq!(first_unit.content, "This is the first paragraph.");
        assert_eq!(first_unit.unit_type, TextUnitType::Paragraph);
    }
    
    #[test]
    fn test_sentence_detection() {
        let mut registry = TextUnitRegistry::new();
        let options = BoundaryDetectionOptions::default();
        
        let text = "This is sentence one. This is sentence two! Is this sentence three?";
        
        let unit_ids = detect_sentence_boundaries(text, &mut registry, &options);
        
        assert_eq!(unit_ids.len(), 3);
        
        let first_unit = registry.get_unit(unit_ids[0]).unwrap();
        assert_eq!(first_unit.content, "This is sentence one.");
        assert_eq!(first_unit.unit_type, TextUnitType::Sentence);
    }
    
    #[test]
    fn test_word_detection() {
        let mut registry = TextUnitRegistry::new();
        let options = BoundaryDetectionOptions::default();
        
        let text = "Hello world, this is a test!";
        
        let unit_ids = detect_word_boundaries(text, &mut registry, &options);
        
        assert!(unit_ids.len() >= 5); // At least Hello, world, this, test
        
        let first_unit = registry.get_unit(unit_ids[0]).unwrap();
        assert_eq!(first_unit.unit_type, TextUnitType::Word);
    }
} 