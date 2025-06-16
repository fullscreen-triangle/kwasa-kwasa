use crate::text_unit::{TextUnit, TextUnitType, TextUnitRegistry};
use crate::text_unit::types::TextUnitId;
use crate::turbulance::ast::Value;
use std::collections::HashMap;
use regex;
use rand;

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
    pub fn to_unit(self, registry: &mut TextUnitRegistry) -> Result<TextUnitId, String> {
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
                
                let start = units.first().unwrap().start_pos;
                let end = units.last().unwrap().end_pos;
                
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
                    let current_len = current.len();
                    let sentence_unit = TextUnit::new(
                        current,
                        offset,
                        offset + current_len,
                        TextUnitType::Sentence,
                        registry.next_available_id(),
                    );
                    
                    let id = registry.add_unit(sentence_unit.clone());
                    registry.set_parent_child(unit.id, id);
                    
                    sentences.push(sentence_unit);
                    offset += current_len;
                    current.clear();
                }
            }
            
            // Add the last part if not empty
            if !current.trim().is_empty() {
                let current_len = current.len();
                let sentence_unit = TextUnit::new(
                    current,
                    offset,
                    offset + current_len,
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
    for unit in &sorted_units {
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
        new_content.clone(),
        from.metadata.clone(),
        from.start,
        // End position is approximate since we removed content
        from.start + new_content.len(),
        from.unit_type,
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Filter a text unit based on a predicate expression
pub fn filter(
    unit: &TextUnit,
    predicate: &str,
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    // First, we need to divide the unit into smaller units to apply the filter
    let division_type = match unit.unit_type {
        TextUnitType::Document => TextUnitType::Section,
        TextUnitType::Section => TextUnitType::Paragraph,
        TextUnitType::Paragraph => TextUnitType::Sentence,
        TextUnitType::Sentence => TextUnitType::Word,
        _ => return OperationResult::Error("Cannot filter unit of this type".to_string()),
    };
    
    // Divide into smaller units
    let divided = match divide(unit, registry, division_type) {
        OperationResult::Units(units) => units,
        OperationResult::Error(e) => return OperationResult::Error(e),
        _ => return OperationResult::Error("Unexpected result from division".to_string()),
    };
    
    // Parse the predicate
    let mut filtered_units = Vec::new();
    
    // Supported predicates:
    // - "contains(text)" - unit contains the specified text
    // - "length > N" - unit length is greater than N
    // - "length < N" - unit length is less than N
    // - "readability > N" - readability score is greater than N
    // - "readability < N" - readability score is less than N
    // - "index < N" - position in parent is less than N
    // - "index > N" - position in parent is greater than N
    // - "matches(regex)" - unit matches the regex pattern
    
    if predicate.starts_with("contains(") && predicate.ends_with(")") {
        let search_text = &predicate[9..predicate.len()-1];
        for unit in divided {
            if unit.content.to_lowercase().contains(&search_text.to_lowercase()) {
                filtered_units.push(unit);
            }
        }
    } else if predicate.starts_with("matches(") && predicate.ends_with(")") {
        let pattern = &predicate[8..predicate.len()-1];
        match regex::Regex::new(pattern) {
            Ok(re) => {
                for unit in divided {
                    if re.is_match(&unit.content) {
                        filtered_units.push(unit);
                    }
                }
            },
            Err(_) => return OperationResult::Error("Invalid regex pattern".to_string()),
        }
    } else if predicate.starts_with("length > ") {
        if let Ok(threshold) = predicate[9..].parse::<usize>() {
            for unit in divided {
                if unit.content.len() > threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid length threshold".to_string());
        }
    } else if predicate.starts_with("length < ") {
        if let Ok(threshold) = predicate[9..].parse::<usize>() {
            for unit in divided {
                if unit.content.len() < threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid length threshold".to_string());
        }
    } else if predicate.starts_with("readability > ") {
        if let Ok(threshold) = predicate[14..].parse::<f64>() {
            for unit in divided {
                if unit.readability_score() > threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid readability threshold".to_string());
        }
    } else if predicate.starts_with("readability < ") {
        if let Ok(threshold) = predicate[14..].parse::<f64>() {
            for unit in divided {
                if unit.readability_score() < threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid readability threshold".to_string());
        }
    } else if predicate.starts_with("index < ") {
        if let Ok(threshold) = predicate[8..].parse::<usize>() {
            for (i, unit) in divided.into_iter().enumerate() {
                if i < threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid index threshold".to_string());
        }
    } else if predicate.starts_with("index > ") {
        if let Ok(threshold) = predicate[8..].parse::<usize>() {
            for (i, unit) in divided.into_iter().enumerate() {
                if i > threshold {
                    filtered_units.push(unit);
                }
            }
        } else {
            return OperationResult::Error("Invalid index threshold".to_string());
        }
    } else if predicate == "first" {
        if !divided.is_empty() {
            filtered_units.push(divided[0].clone());
        }
    } else if predicate == "last" {
        if !divided.is_empty() {
            filtered_units.push(divided[divided.len() - 1].clone());
        }
    } else {
        return OperationResult::Error(format!("Unsupported predicate: {}", predicate));
    }
    
    if filtered_units.is_empty() {
        OperationResult::Error("No units matched the filter".to_string())
    } else {
        OperationResult::Units(filtered_units)
    }
}

/// Transform a text unit using a specified transformation function
pub fn transform(
    unit: &TextUnit,
    transformation: &str,
    registry: &mut TextUnitRegistry,
    args: Option<HashMap<String, Value>>,
) -> OperationResult {
    let args = args.unwrap_or_default();
    
    // Transformations:
    // - simplify - simplify the text for better readability
    // - formalize - make the text more formal
    // - expand - add more detail and explanation
    // - summarize - create a shorter summary
    // - normalize - standardize formatting and style
    // - capitalize - convert first letter of sentences to uppercase
    // - lowercase - convert all text to lowercase
    // - uppercase - convert all text to uppercase
    
    let transformed = match transformation {
        "simplify" => {
            // Simplify: replace complex words, shorten sentences
            let level = if let Some(Value::String(level)) = args.get("level") {
                level
            } else {
                "medium"
            };
            
            let mut content = unit.content.clone();
            
            // Example simplification (would be more sophisticated in real implementation)
            match level {
                "high" => {
                    // Replace complex words with simpler alternatives
                    content = content.replace("utilize", "use")
                        .replace("demonstrate", "show")
                        .replace("sufficient", "enough")
                        .replace("implement", "use")
                        .replace("comprehensive", "complete")
                        .replace("subsequently", "later")
                        .replace("additionally", "also");
                    
                    // Shorten sentences (very basic algorithm)
                    let sentences: Vec<&str> = content
                        .split(&['.', '!', '?'][..])
                        .filter(|s| !s.trim().is_empty())
                        .collect();
                    
                    content = sentences.into_iter()
                        .map(|s| {
                            if s.split_whitespace().count() > 20 {
                                // Crude simplification - take first part of sentence
                                let words: Vec<&str> = s.split_whitespace().take(15).collect();
                                words.join(" ") + "..."
                            } else {
                                s.to_string()
                            }
                        })
                        .collect::<Vec<String>>()
                        .join(". ");
                    
                    content.push('.');
                },
                "medium" => {
                    // Replace some complex words
                    content = content.replace("utilize", "use")
                        .replace("demonstrate", "show")
                        .replace("sufficient", "enough");
                },
                "low" => {
                    // Minimal simplification
                    content = content.replace("utilize", "use");
                },
                _ => {
                    return OperationResult::Error(format!("Unsupported simplification level: {}", level));
                }
            }
            
            content
        },
        "formalize" => {
            // Make text more formal
            let mut content = unit.content.clone();
            
            // Remove contractions
            content = content.replace("don't", "do not")
                .replace("won't", "will not")
                .replace("can't", "cannot")
                .replace("I'm", "I am")
                .replace("you're", "you are")
                .replace("they're", "they are")
                .replace("we're", "we are")
                .replace("isn't", "is not")
                .replace("aren't", "are not")
                .replace("hasn't", "has not")
                .replace("haven't", "have not")
                .replace("wasn't", "was not")
                .replace("weren't", "were not");
            
            // Replace informal words with more formal alternatives
            content = content.replace("lots of", "many")
                .replace("a lot of", "many")
                .replace("kind of", "somewhat")
                .replace("sort of", "somewhat")
                .replace("kids", "children")
                .replace("totally", "completely")
                .replace("awesome", "excellent")
                .replace("cool", "impressive")
                .replace("huge", "significant")
                .replace("big", "substantial")
                .replace("get", "obtain")
                .replace("show", "demonstrate");
            
            content
        },
        "expand" => {
            // Expand the text with more details
            // In a real implementation, this would use more sophisticated NLP
            let mut content = unit.content.clone();
            
            // Add explanatory phrases
            content = content.replace(". ", ". In other words, ")
                .replace("? ", "? To clarify, ")
                .replace("! ", "! Indeed, ");
            
            // Add qualifiers to strengthen statements
            let words: Vec<&str> = content.split_whitespace().collect();
            let with_qualifiers = words.iter()
                .map(|&word| {
                    if word.len() > 5 && !word.contains('.') && !word.contains(',') && rand::random::<f32>() > 0.8 {
                        format!("{} (which is important)", word)
                    } else {
                        word.to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join(" ");
            
            with_qualifiers
        },
        "summarize" => {
            // Create a summary (simplified implementation)
            let content = unit.content.clone();
            
            // Extract key sentences (very basic approach)
            let sentences: Vec<&str> = content
                .split(&['.', '!', '?'][..])
                .filter(|s| !s.trim().is_empty())
                .collect();
            
            if sentences.len() <= 2 {
                // Already short, just return it
                content
            } else {
                // Take first and last sentence
                let mut summary = sentences.first().unwrap().to_string();
                summary.push('.');
                
                if sentences.len() > 3 {
                    summary.push_str(" [...] ");
                    summary.push_str(sentences.last().unwrap());
                    summary.push('.');
                }
                
                summary
            }
        },
        "normalize" => {
            // Standardize formatting
            let content = unit.content.clone();
            
            // Remove extra whitespace
            let mut normalized = content.split_whitespace().collect::<Vec<&str>>().join(" ");
            
            // Ensure proper sentence capitalization
            let sentences: Vec<&str> = normalized
                .split(&['.', '!', '?'][..])
                .filter(|s| !s.trim().is_empty())
                .collect();
            
            normalized = sentences.into_iter()
                .map(|s| {
                    let s = s.trim();
                    if s.is_empty() {
                        s.to_string()
                    } else {
                        let mut chars = s.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                        }
                    }
                })
                .collect::<Vec<String>>()
                .join(". ");
            
            normalized.push('.');
            normalized
        },
        "capitalize" => {
            // Capitalize the first letter of each sentence
            let content = unit.content.clone();
            
            let sentences: Vec<&str> = content
                .split(&['.', '!', '?'][..])
                .filter(|s| !s.trim().is_empty())
                .collect();
            
            let capitalized = sentences.into_iter()
                .map(|s| {
                    let s = s.trim();
                    if s.is_empty() {
                        s.to_string()
                    } else {
                        let mut chars = s.chars();
                        match chars.next() {
                            None => String::new(),
                            Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                        }
                    }
                })
                .collect::<Vec<String>>()
                .join(". ");
            
            capitalized
        },
        "lowercase" => {
            // Convert to lowercase
            unit.content.to_lowercase()
        },
        "uppercase" => {
            // Convert to uppercase
            unit.content.to_uppercase()
        },
        _ => {
            return OperationResult::Error(format!("Unsupported transformation: {}", transformation));
        }
    };
    
    // Create a new unit with the transformed content
    let result_unit = TextUnit::with_metadata(
        transformed.clone(),
        unit.metadata.clone(),
        unit.start,
        unit.start + transformed.len(),
        unit.unit_type,
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Apply a pipeline of operations to a text unit
pub fn pipeline(
    unit: &TextUnit,
    operations: &[(&str, Option<HashMap<String, Value>>)],
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    let mut current_result = OperationResult::Unit(unit.clone());
    
    for (op, args) in operations {
        current_result = match current_result {
            OperationResult::Unit(unit) => {
                match *op {
                    "filter" => {
                        if let Some(args) = args {
                            if let Some(Value::String(predicate)) = args.get("predicate") {
                                filter(&unit, predicate, registry)
                            } else {
                                OperationResult::Error("Missing predicate for filter".to_string())
                            }
                        } else {
                            OperationResult::Error("Missing arguments for filter".to_string())
                        }
                    },
                    "transform" => {
                        if let Some(args) = args {
                            if let Some(Value::String(transformation)) = args.get("type") {
                                transform(&unit, transformation, registry, Some(args.clone()))
                            } else {
                                OperationResult::Error("Missing transformation type".to_string())
                            }
                        } else {
                            OperationResult::Error("Missing arguments for transform".to_string())
                        }
                    },
                    "divide" => {
                        if let Some(args) = args {
                            if let Some(Value::String(type_str)) = args.get("type") {
                                let division_type = match type_str.as_str() {
                                    "paragraph" => TextUnitType::Paragraph,
                                    "sentence" => TextUnitType::Sentence,
                                    "word" => TextUnitType::Word,
                                    "section" => TextUnitType::Section,
                                    _ => return OperationResult::Error(format!("Unknown division type: {}", type_str)),
                                };
                                divide(&unit, registry, division_type)
                            } else {
                                OperationResult::Error("Missing division type".to_string())
                            }
                        } else {
                            OperationResult::Error("Missing arguments for divide".to_string())
                        }
                    },
                    _ => OperationResult::Error(format!("Unsupported operation: {}", op)),
                }
            },
            OperationResult::Units(units) => {
                // For operations that work on multiple units, we can combine results
                let mut combined_results = Vec::new();
                
                for u in units {
                    let result = match *op {
                        "filter" => {
                            if let Some(args) = args {
                                if let Some(Value::String(predicate)) = args.get("predicate") {
                                    filter(&u, predicate, registry)
                                } else {
                                    continue; // Skip invalid filters
                                }
                            } else {
                                continue; // Skip invalid filters
                            }
                        },
                        "transform" => {
                            if let Some(args) = args {
                                if let Some(Value::String(transformation)) = args.get("type") {
                                    transform(&u, transformation, registry, Some(args.clone()))
                                } else {
                                    continue; // Skip invalid transforms
                                }
                            } else {
                                continue; // Skip invalid transforms
                            }
                        },
                        _ => continue, // Skip unsupported operations
                    };
                    
                    match result {
                        OperationResult::Unit(unit) => combined_results.push(unit),
                        OperationResult::Units(mut units) => combined_results.append(&mut units),
                        _ => {} // Ignore errors or values
                    }
                }
                
                if combined_results.is_empty() {
                    OperationResult::Error("Pipeline produced no results".to_string())
                } else {
                    OperationResult::Units(combined_results)
                }
            },
            _ => {
                return OperationResult::Error("Previous operation didn't produce a valid unit".to_string());
            }
        };
    }
    
    current_result
}

/// Compose text units with intelligent transitions
pub fn compose(
    units: &[&TextUnit],
    style: &str,
    registry: &mut TextUnitRegistry,
) -> OperationResult {
    if units.len() < 2 {
        return OperationResult::Error("Composition requires at least two units".to_string());
    }
    
    let mut composed = String::new();
    let mut current_offset = 0;
    
    // First, ensure units are sorted by position
    let mut sorted_units = units.to_vec();
    sorted_units.sort_by_key(|u| u.start);
    
    // Generate appropriate transitions based on the style
    for i in 0..sorted_units.len() {
        if i > 0 {
            // Add transition between units
            let transition = match style {
                "sequential" => {
                    match sorted_units[0].unit_type {
                        TextUnitType::Sentence => " Then, ",
                        TextUnitType::Paragraph => "\n\nNext, ",
                        _ => "\n\nFollowing this, ",
                    }
                },
                "causal" => {
                    match sorted_units[0].unit_type {
                        TextUnitType::Sentence => " Therefore, ",
                        TextUnitType::Paragraph => "\n\nAs a result, ",
                        _ => "\n\nConsequently, ",
                    }
                },
                "contrastive" => {
                    match sorted_units[0].unit_type {
                        TextUnitType::Sentence => " However, ",
                        TextUnitType::Paragraph => "\n\nIn contrast, ",
                        _ => "\n\nOn the other hand, ",
                    }
                },
                "elaborative" => {
                    match sorted_units[0].unit_type {
                        TextUnitType::Sentence => " Furthermore, ",
                        TextUnitType::Paragraph => "\n\nMoreover, ",
                        _ => "\n\nTo elaborate, ",
                    }
                },
                "summarizing" => {
                    if i == sorted_units.len() - 1 {
                        match sorted_units[0].unit_type {
                            TextUnitType::Sentence => " In summary, ",
                            TextUnitType::Paragraph => "\n\nIn conclusion, ",
                            _ => "\n\nTo summarize, ",
                        }
                    } else {
                        match sorted_units[0].unit_type {
                            TextUnitType::Sentence => " Additionally, ",
                            TextUnitType::Paragraph => "\n\nFurthermore, ",
                            _ => "\n\nAlso, ",
                        }
                    }
                },
                _ => {
                    match sorted_units[0].unit_type {
                        TextUnitType::Sentence => " ",
                        TextUnitType::Paragraph => "\n\n",
                        _ => "\n\n",
                    }
                },
            };
            
            composed.push_str(transition);
            current_offset += transition.len();
        }
        
        composed.push_str(&sorted_units[i].content);
        current_offset += sorted_units[i].content.len();
    }
    
    // Create a new unit with the composed content
    let result_unit = TextUnit::new(
        composed.clone(),
        sorted_units[0].start,
        sorted_units[0].start + composed.len(),
        sorted_units[0].unit_type,
        registry.next_available_id(),
    );
    
    OperationResult::Unit(result_unit)
}

/// Get the numeric level of a unit type for hierarchy comparisons
fn unit_type_level(unit_type: TextUnitType) -> usize {
    match unit_type {
        TextUnitType::Character => 1,
        TextUnitType::Word => 2,
        TextUnitType::Phrase => 2,
        TextUnitType::Sentence => 3,
        TextUnitType::Paragraph => 4,
        TextUnitType::Section => 5,
        TextUnitType::Document => 6,
        TextUnitType::Custom(_) => 0, // Custom types are lowest level
        TextUnitType::Sequence => 3,
        TextUnitType::Formula => 3,
        TextUnitType::Spectrum => 3,
        TextUnitType::Code => 3,
        TextUnitType::Math => 3,
        TextUnitType::Table => 4,
        TextUnitType::ListItem => 3,
        TextUnitType::Header => 5,
        TextUnitType::Footer => 5,
        TextUnitType::Citation => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_unit::{TextUnit, TextUnitType, TextUnitRegistry};
    
    #[test]
    fn test_divide_paragraph_to_sentences() {
        let mut registry = TextUnitRegistry::new();
        
        let paragraph = TextUnit::new(
            "This is sentence one. This is sentence two! This is sentence three?".to_string(),
            0,
            67,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let paragraph_id = registry.add_unit(paragraph.clone());
        
        let result = divide(
            registry.get_unit(paragraph_id).unwrap(),
            &mut registry,
            TextUnitType::Sentence,
        );
        
        match result {
            OperationResult::Units(sentences) => {
                assert_eq!(sentences.len(), 3);
                assert_eq!(sentences[0].content, "This is sentence one.");
                assert_eq!(sentences[1].content, "This is sentence two!");
                assert_eq!(sentences[2].content, "This is sentence three?");
            },
            _ => panic!("Expected Units result from divide operation"),
        }
    }
    
    #[test]
    fn test_multiply_sentences_to_paragraph() {
        let mut registry = TextUnitRegistry::new();
        
        let sentence1 = TextUnit::new(
            "This is sentence one.".to_string(),
            0,
            21,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence2 = TextUnit::new(
            "This is sentence two.".to_string(),
            22,
            43,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence1_id = registry.add_unit(sentence1.clone());
        let sentence2_id = registry.add_unit(sentence2.clone());
        
        let result = multiply(
            &[
                registry.get_unit(sentence1_id).unwrap(),
                registry.get_unit(sentence2_id).unwrap(),
            ],
            &mut registry,
            TextUnitType::Paragraph,
        );
        
        match result {
            OperationResult::Unit(paragraph) => {
                assert_eq!(paragraph.unit_type, TextUnitType::Paragraph);
                assert_eq!(paragraph.content, "This is sentence one. This is sentence two.");
            },
            _ => panic!("Expected Unit result from multiply operation"),
        }
    }
    
    #[test]
    fn test_add_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let unit1 = TextUnit::new(
            "First part.".to_string(),
            0,
            11,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let unit2 = TextUnit::new(
            "Second part.".to_string(),
            12,
            24,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let unit1_id = registry.add_unit(unit1.clone());
        let unit2_id = registry.add_unit(unit2.clone());
        
        let result = add(
            registry.get_unit(unit1_id).unwrap(),
            registry.get_unit(unit2_id).unwrap(),
            &mut registry,
        );
        
        match result {
            OperationResult::Unit(combined) => {
                assert_eq!(combined.unit_type, TextUnitType::Paragraph);
                assert_eq!(combined.content, "First part.\n\nSecond part.");
            },
            _ => panic!("Expected Unit result from add operation"),
        }
    }
    
    #[test]
    fn test_subtract_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let unit1 = TextUnit::new(
            "This text contains some removable content.".to_string(),
            0,
            41,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let unit2 = TextUnit::new(
            "removable content".to_string(),
            19,
            36,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let unit1_id = registry.add_unit(unit1.clone());
        let unit2_id = registry.add_unit(unit2.clone());
        
        let result = subtract(
            registry.get_unit(unit1_id).unwrap(),
            registry.get_unit(unit2_id).unwrap(),
            &mut registry,
        );
        
        match result {
            OperationResult::Unit(subtracted) => {
                assert_eq!(subtracted.unit_type, TextUnitType::Sentence);
                assert_eq!(subtracted.content, "This text contains some.");
            },
            _ => panic!("Expected Unit result from subtract operation"),
        }
    }
    
    #[test]
    fn test_filter_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let paragraph = TextUnit::new(
            "This is a long sentence with many words. This is a short one.".to_string(),
            0,
            62,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let paragraph_id = registry.add_unit(paragraph.clone());
        
        // Test length filter
        let result = filter(
            registry.get_unit(paragraph_id).unwrap(),
            "length > 20",
            &mut registry,
        );
        
        match result {
            OperationResult::Units(filtered) => {
                assert_eq!(filtered.len(), 1);
                assert_eq!(filtered[0].content, "This is a long sentence with many words.");
            },
            _ => panic!("Expected Units result from filter operation"),
        }
        
        // Test contains filter
        let result = filter(
            registry.get_unit(paragraph_id).unwrap(),
            "contains(short)",
            &mut registry,
        );
        
        match result {
            OperationResult::Units(filtered) => {
                assert_eq!(filtered.len(), 1);
                assert_eq!(filtered[0].content, "This is a short one.");
            },
            _ => panic!("Expected Units result from filter operation"),
        }
    }
    
    #[test]
    fn test_transform_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let sentence = TextUnit::new(
            "this sentence should be capitalized.".to_string(),
            0,
            35,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence_id = registry.add_unit(sentence.clone());
        
        let result = transform(
            registry.get_unit(sentence_id).unwrap(),
            "capitalize",
            &mut registry,
            None,
        );
        
        match result {
            OperationResult::Unit(transformed) => {
                assert_eq!(transformed.content, "This sentence should be capitalized.");
            },
            _ => panic!("Expected Unit result from transform operation"),
        }
    }
    
    #[test]
    fn test_pipeline_operations() {
        let mut registry = TextUnitRegistry::new();
        
        let paragraph = TextUnit::new(
            "This is a LONG sentence with many words. This is a short one.".to_string(),
            0,
            62,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let paragraph_id = registry.add_unit(paragraph.clone());
        
        // Create a pipeline: divide -> filter -> transform
        let mut filter_args = HashMap::new();
        filter_args.insert("predicate".to_string(), Value::String("contains(LONG)".to_string()));
        
        let mut transform_args = HashMap::new();
        transform_args.insert("type".to_string(), Value::String("lowercase".to_string()));
        
        let operations = vec![
            ("filter", Some(filter_args)),
            ("transform", Some(transform_args)),
        ];
        
        let result = pipeline(
            registry.get_unit(paragraph_id).unwrap(),
            &operations,
            &mut registry,
        );
        
        match result {
            OperationResult::Unit(transformed) => {
                assert_eq!(transformed.content, "this is a long sentence with many words.");
            },
            _ => panic!("Expected Unit result from pipeline operation"),
        }
    }
    
    #[test]
    fn test_compose_text_units() {
        let mut registry = TextUnitRegistry::new();
        
        let sentence1 = TextUnit::new(
            "This is the first point".to_string(),
            0,
            24,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence2 = TextUnit::new(
            "This is the second point".to_string(),
            25,
            50,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let sentence1_id = registry.add_unit(sentence1.clone());
        let sentence2_id = registry.add_unit(sentence2.clone());
        
        let result = compose(
            &[
                registry.get_unit(sentence1_id).unwrap(),
                registry.get_unit(sentence2_id).unwrap(),
            ],
            "contrastive",
            &mut registry,
        );
        
        match result {
            OperationResult::Unit(composed) => {
                assert_eq!(composed.content, "This is the first point However, This is the second point");
            },
            _ => panic!("Expected Unit result from compose operation"),
        }
    }
}
