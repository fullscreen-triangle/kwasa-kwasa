use crate::text_unit::{TextUnit, TextUnitType, TextUnitRegistry};
use crate::turbulance::ast::Value;
use std::collections::HashMap;

/// The result of a text operation
#[derive(Debug, Clone)]
pub enum OperationResult {
    /// A single text unit was produced
    Unit(TextUnit),
    
    /// Multiple text units were produced
    Units(Vec<TextUnit>),
    
    /// A value was produced (not a text unit)
    Value(Value),
    
    /// The operation failed
    Error(String),
}

impl OperationResult {
    /// Convert the result to a text unit
    pub fn to_unit(self, registry: &mut TextUnitRegistry) -> Result<usize, String> {
        match self {
            OperationResult::Unit(unit) => Ok(registry.add_unit(unit)),
            OperationResult::Units(units) => {
                if units.is_empty() {
                    return Err("No units produced".to_string());
                }
                
                // Combine all units into one
                let combined = units.iter()
                    .map(|u| u.content.clone())
                    .collect::<Vec<String>>()
                    .join(" ");
                
                let start = units.first().unwrap().start;
                let end = units.last().unwrap().end;
                
                let unit = TextUnit::new(
                    combined,
                    start,
                    end,
                    TextUnitType::Custom(1), // Merged unit
                    registry.next_available_id(),
                );
                
                Ok(registry.add_unit(unit))
            }
            OperationResult::Value(Value::TextUnit(ast_unit)) => {
                let unit = TextUnit::with_metadata(
                    ast_unit.content,
                    ast_unit.metadata,
                    0, // Position unknown
                    0, // Position unknown
                    TextUnitType::Custom(0),
                    registry.next_available_id(),
                );
                
                Ok(registry.add_unit(unit))
            }
            OperationResult::Value(_) => Err("Result is not a text unit".to_string()),
            OperationResult::Error(e) => Err(e),
        }
    }
    
    /// Convert the result to a value
    pub fn to_value(self) -> Result<Value, String> {
        match self {
            OperationResult::Unit(unit) => Ok(Value::TextUnit(unit.to_ast_unit())),
            OperationResult::Units(units) => {
                let ast_units = units.into_iter()
                    .map(|u| Value::TextUnit(u.to_ast_unit()))
                    .collect();
                
                Ok(Value::List(ast_units))
            }
            OperationResult::Value(value) => Ok(value),
            OperationResult::Error(e) => Err(e),
        }
    }
}

/// Divide a text unit into smaller units
pub fn divide(
    unit: &TextUnit,
    registry: &mut TextUnitRegistry,
    division_type: TextUnitType,
) -> OperationResult {
    // If the unit is already smaller than or equal to the target division type,
    // we can't divide it further
    if unit_type_level(unit.unit_type) <= unit_type_level(division_type) {
        return OperationResult::Error(format!(
            "Cannot divide a {:?} into {:?}s", 
            unit.unit_type, 
            division_type
        ));
    }
    
    // Get child units of the target type if they already exist
    let existing_children = registry.children_of(unit.id)
        .into_iter()
        .filter(|child| child.unit_type == division_type)
        .map(|child| child.clone())
        .collect::<Vec<TextUnit>>();
    
    if !existing_children.is_empty() {
        return OperationResult::Units(existing_children);
    }
    
    // Otherwise, perform text boundary detection based on the division type
    let mut result_units = Vec::new();
    
    match division_type {
        TextUnitType::Paragraph => {
            // Split by paragraph breaks
            let paragraphs = unit.content.split("\n\n")
                .filter(|p| !p.trim().is_empty())
                .collect::<Vec<&str>>();
            
            let mut offset = unit.start;
            
            for paragraph in paragraphs {
                let para_unit = TextUnit::new(
                    paragraph.to_string(),
                    offset,
                    offset + paragraph.len(),
                    TextUnitType::Paragraph,
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(para_unit.clone());
                registry.set_parent_child(unit.id, id);
                
                result_units.push(para_unit);
                offset += paragraph.len() + 2; // +2 for "\n\n"
            }
        }
        TextUnitType::Sentence => {
            // Simple sentence splitting (can be improved)
            let sentence_endings = ['.', '!', '?'];
            let mut sentences = Vec::new();
            let mut current = String::new();
            let mut offset = unit.start;
            
            for c in unit.content.chars() {
                current.push(c);
                
                if sentence_endings.contains(&c) && !current.trim().is_empty() {
                    let sentence_unit = TextUnit::new(
                        current.clone(),
                        offset,
                        offset + current.len(),
                        TextUnitType::Sentence,
                        registry.next_available_id(),
                    );
                    
                    let id = registry.add_unit(sentence_unit.clone());
                    registry.set_parent_child(unit.id, id);
                    
                    sentences.push(sentence_unit);
                    offset += current.len();
                    current.clear();
                }
            }
            
            // Add the last part if not empty
            if !current.trim().is_empty() {
                let sentence_unit = TextUnit::new(
                    current,
                    offset,
                    offset + current.len(),
                    TextUnitType::Sentence,
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(sentence_unit.clone());
                registry.set_parent_child(unit.id, id);
                
                sentences.push(sentence_unit);
            }
            
            result_units = sentences;
        }
        TextUnitType::Word => {
            // Split into words
            let words = unit.content.split_whitespace()
                .collect::<Vec<&str>>();
            
            let mut offset = unit.start;
            
            for word in words {
                let word_unit = TextUnit::new(
                    word.to_string(),
                    offset,
                    offset + word.len(),
                    TextUnitType::Word,
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(word_unit.clone());
                registry.set_parent_child(unit.id, id);
                
                result_units.push(word_unit);
                offset += word.len() + 1; // +1 for space
            }
        }
        _ => {
            return OperationResult::Error(format!(
                "Division into {:?} not implemented", 
                division_type
            ));
        }
    }
    
    OperationResult::Units(result_units)
}

/// Combine multiple text units into a larger unit
pub fn multiply(
    units: &[&TextUnit],
    registry: &mut TextUnitRegistry,
    result_type: TextUnitType,
) -> OperationResult {
    if units.is_empty() {
        return OperationResult::Error("Cannot multiply empty list of units".to_string());
    }
    
    // Sort units by their position
    let mut sorted_units = units.to_vec();
    sorted_units.sort_by_key(|u| u.start);
    
    // Check that all units are of the same type
    let first_type = sorted_units[0].unit_type;
    if !sorted_units.iter().all(|u| u.unit_type == first_type) {
        return OperationResult::Error("Cannot multiply units of different types".to_string());
    }
    
    // Check that target type is larger than source type
    if unit_type_level(first_type) >= unit_type_level(result_type) {
        return OperationResult::Error(format!(
            "Cannot multiply {:?}s into a {:?}", 
            first_type, 
            result_type
        ));
    }
    
    // Combine contents based on result type
    let mut combined = String::new();
    let mut metadata = HashMap::new();
    
    match result_type {
        TextUnitType::Paragraph => {
            // Combine sentences into paragraph
            for (i, unit) in sorted_units.iter().enumerate() {
                if i > 0 {
                    combined.push(' ');
                }
                combined.push_str(&unit.content);
            }
        }
        TextUnitType::Sentence => {
            // Combine words into sentence
            for (i, unit) in sorted_units.iter().enumerate() {
                if i > 0 {
                    combined.push(' ');
                }
                combined.push_str(&unit.content);
            }
            combined.push('.');
        }
        TextUnitType::Section => {
            // Combine paragraphs into section
            for (i, unit) in sorted_units.iter().enumerate() {
                if i > 0 {
                    combined.push_str("\n\n");
                }
                combined.push_str(&unit.content);
            }
        }
        TextUnitType::Document => {
            // Combine sections into document
            for (i, unit) in sorted_units.iter().enumerate() {
                if i > 0 {
                    combined.push_str("\n\n");
                }
                combined.push_str(&unit.content);
            }
        }
        _ => {
            return OperationResult::Error(format!(
                "Multiplication into {:?} not implemented", 
                result_type
            ));
        }
    }
    
    // Combine metadata from all units
    for unit in sorted_units {
        for (key, value) in &unit.metadata {
            metadata.insert(key.clone(), value.clone());
        }
    }
    
    // Create the resulting unit
    let start = sorted_units.first().unwrap().start;
    let end = sorted_units.last().unwrap().end;
    
    let result_unit = TextUnit::with_metadata(
        combined,
        metadata,
        start,
        end,
        result_type,
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Add two text units (concatenation)
pub fn add(
    unit1: &TextUnit,
    unit2: &TextUnit,
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    // Determines which unit comes first
    let (first, second) = if unit1.start <= unit2.start {
        (unit1, unit2)
    } else {
        (unit2, unit1)
    };
    
    // Combine the content
    let mut combined = first.content.clone();
    
    // Add appropriate separator based on unit type
    match first.unit_type {
        TextUnitType::Word => combined.push(' '),
        TextUnitType::Sentence => combined.push(' '),
        TextUnitType::Paragraph => combined.push_str("\n\n"),
        TextUnitType::Section => combined.push_str("\n\n"),
        _ => combined.push('\n'),
    }
    
    combined.push_str(&second.content);
    
    // Combine metadata
    let mut metadata = first.metadata.clone();
    for (key, value) in &second.metadata {
        metadata.insert(key.clone(), value.clone());
    }
    
    // Create the resulting unit
    let result_unit = TextUnit::with_metadata(
        combined,
        metadata,
        first.start,
        second.end,
        first.unit_type, // Keep the same type
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Subtract one text unit from another (remove content)
pub fn subtract(
    from: &TextUnit,
    what: &TextUnit,
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    // Check if the unit to be removed is contained in the source
    if !from.content.contains(&what.content) {
        return OperationResult::Error("Cannot subtract: content not found".to_string());
    }
    
    // Remove the content
    let new_content = from.content.replace(&what.content, "");
    
    // Clean up double spaces, etc.
    let new_content = new_content
        .replace("  ", " ")
        .replace("\n\n\n", "\n\n")
        .trim()
        .to_string();
    
    // Create the resulting unit
    let result_unit = TextUnit::with_metadata(
        new_content,
        from.metadata.clone(),
        from.start,
        // End position is approximate since we removed content
        from.start + new_content.len(),
        from.unit_type,
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Filter a text unit based on a predicate
pub fn filter(
    unit: &TextUnit,
    predicate: &str,
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    // This is a simple implementation for basic filtering
    // In the future, we can use a more sophisticated approach
    
    match predicate {
        "long_sentences" => {
            // Divide into sentences and keep only long ones
            if let OperationResult::Units(sentences) = divide(unit, registry, TextUnitType::Sentence) {
                let long_sentences = sentences.into_iter()
                    .filter(|s| s.content.split_whitespace().count() > 10)
                    .collect::<Vec<TextUnit>>();
                
                if long_sentences.is_empty() {
                    return OperationResult::Error("No long sentences found".to_string());
                }
                
                OperationResult::Units(long_sentences)
            } else {
                OperationResult::Error("Failed to divide into sentences".to_string())
            }
        }
        "short_sentences" => {
            // Divide into sentences and keep only short ones
            if let OperationResult::Units(sentences) = divide(unit, registry, TextUnitType::Sentence) {
                let short_sentences = sentences.into_iter()
                    .filter(|s| s.content.split_whitespace().count() <= 10)
                    .collect::<Vec<TextUnit>>();
                
                if short_sentences.is_empty() {
                    return OperationResult::Error("No short sentences found".to_string());
                }
                
                OperationResult::Units(short_sentences)
            } else {
                OperationResult::Error("Failed to divide into sentences".to_string())
            }
        }
        "complex" => {
            // Filter for complex content (high complexity score)
            if let OperationResult::Units(paragraphs) = divide(unit, registry, TextUnitType::Paragraph) {
                let complex_paragraphs = paragraphs.into_iter()
                    .filter(|p| p.complexity() > 0.6)
                    .collect::<Vec<TextUnit>>();
                
                if complex_paragraphs.is_empty() {
                    return OperationResult::Error("No complex paragraphs found".to_string());
                }
                
                OperationResult::Units(complex_paragraphs)
            } else {
                OperationResult::Error("Failed to divide into paragraphs".to_string())
            }
        }
        "simple" => {
            // Filter for simple content (low complexity score)
            if let OperationResult::Units(paragraphs) = divide(unit, registry, TextUnitType::Paragraph) {
                let simple_paragraphs = paragraphs.into_iter()
                    .filter(|p| p.complexity() <= 0.6)
                    .collect::<Vec<TextUnit>>();
                
                if simple_paragraphs.is_empty() {
                    return OperationResult::Error("No simple paragraphs found".to_string());
                }
                
                OperationResult::Units(simple_paragraphs)
            } else {
                OperationResult::Error("Failed to divide into paragraphs".to_string())
            }
        }
        _ => {
            // Try to filter by content containing the predicate
            if let OperationResult::Units(paragraphs) = divide(unit, registry, TextUnitType::Paragraph) {
                let filtered = paragraphs.into_iter()
                    .filter(|p| p.content.to_lowercase().contains(&predicate.to_lowercase()))
                    .collect::<Vec<TextUnit>>();
                
                if filtered.is_empty() {
                    return OperationResult::Error(format!("No content matching '{}'", predicate));
                }
                
                OperationResult::Units(filtered)
            } else {
                OperationResult::Error("Failed to divide into paragraphs".to_string())
            }
        }
    }
}

/// Get the hierarchy level of a unit type
fn unit_type_level(unit_type: TextUnitType) -> usize {
    match unit_type {
        TextUnitType::Character => 1,
        TextUnitType::Word => 2,
        TextUnitType::Sentence => 3,
        TextUnitType::Paragraph => 4,
        TextUnitType::Section => 5,
        TextUnitType::Document => 6,
        TextUnitType::Custom(level) => level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_divide_paragraph_to_sentences() {
        let mut registry = TextUnitRegistry::new();
        
        let paragraph = TextUnit::new(
            "This is the first sentence. This is the second! And this is the third?".to_string(),
            0,
            72,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let para_id = registry.add_unit(paragraph.clone());
        
        let result = divide(&paragraph, &mut registry, TextUnitType::Sentence);
        
        match result {
            OperationResult::Units(sentences) => {
                assert_eq!(sentences.len(), 3);
                
                // Check that the sentences were added to the registry
                assert_eq!(registry.units_of_type(TextUnitType::Sentence).len(), 3);
                
                // Check that parent-child relationships were established
                assert_eq!(registry.children_of(para_id).len(), 3);
            },
            _ => panic!("Expected Units result"),
        }
    }
    
    #[test]
    fn test_multiply_sentences_to_paragraph() {
        let mut registry = TextUnitRegistry::new();
        
        let sentence1 = TextUnit::new(
            "This is the first sentence".to_string(),
            0,
            26,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence2 = TextUnit::new(
            "This is the second sentence".to_string(),
            27,
            54,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let s1_id = registry.add_unit(sentence1.clone());
        let s2_id = registry.add_unit(sentence2.clone());
        
        let s1 = registry.get_unit(s1_id).unwrap();
        let s2 = registry.get_unit(s2_id).unwrap();
        
        let result = multiply(&[s1, s2], &mut registry, TextUnitType::Paragraph);
        
        match result {
            OperationResult::Unit(paragraph) => {
                assert_eq!(paragraph.unit_type, TextUnitType::Paragraph);
                assert_eq!(paragraph.content, "This is the first sentence This is the second sentence");
            },
            _ => panic!("Expected Unit result"),
        }
    }
    
    #[test]
    fn test_add_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let para1 = TextUnit::new(
            "First paragraph.".to_string(),
            0,
            16,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let para2 = TextUnit::new(
            "Second paragraph.".to_string(),
            17,
            34,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let p1_id = registry.add_unit(para1.clone());
        let p2_id = registry.add_unit(para2.clone());
        
        let p1 = registry.get_unit(p1_id).unwrap();
        let p2 = registry.get_unit(p2_id).unwrap();
        
        let result = add(p1, p2, &mut registry);
        
        match result {
            OperationResult::Unit(combined) => {
                assert_eq!(combined.unit_type, TextUnitType::Paragraph);
                assert_eq!(combined.content, "First paragraph.\n\nSecond paragraph.");
            },
            _ => panic!("Expected Unit result"),
        }
    }
    
    #[test]
    fn test_subtract_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let text = TextUnit::new(
            "This sentence contains unwanted text.".to_string(),
            0,
            36,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let remove = TextUnit::new(
            "unwanted".to_string(),
            19,
            27,
            TextUnitType::Word,
            registry.next_available_id(),
        );
        
        let text_id = registry.add_unit(text.clone());
        let remove_id = registry.add_unit(remove.clone());
        
        let text_unit = registry.get_unit(text_id).unwrap();
        let remove_unit = registry.get_unit(remove_id).unwrap();
        
        let result = subtract(text_unit, remove_unit, &mut registry);
        
        match result {
            OperationResult::Unit(result_unit) => {
                assert_eq!(result_unit.content, "This sentence contains text.");
            },
            _ => panic!("Expected Unit result"),
        }
    }
    
    #[test]
    fn test_filter_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let text = TextUnit::new(
            "This is a short sentence. This sentence is much longer and contains more words which increases its overall complexity. Another short one. This is yet another example of a longer sentence with more words and detail.".to_string(),
            0,
            200,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let text_id = registry.add_unit(text.clone());
        let text_unit = registry.get_unit(text_id).unwrap();
        
        // Filter for long sentences
        let result = filter(text_unit, "long_sentences", &mut registry);
        
        match result {
            OperationResult::Units(long_sentences) => {
                assert_eq!(long_sentences.len(), 2);
                assert!(long_sentences[0].content.contains("much longer"));
                assert!(long_sentences[1].content.contains("yet another example"));
            },
            _ => panic!("Expected Units result"),
        }
        
        // Filter for short sentences
        let result = filter(text_unit, "short_sentences", &mut registry);
        
        match result {
            OperationResult::Units(short_sentences) => {
                assert_eq!(short_sentences.len(), 2);
                assert!(short_sentences[0].content.contains("short sentence"));
                assert!(short_sentences[1].content.contains("Another short one"));
            },
            _ => panic!("Expected Units result"),
        }
    }
}
