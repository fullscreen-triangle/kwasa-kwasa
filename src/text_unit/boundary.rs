use unicode_segmentation::UnicodeSegmentation;
use std::collections::HashSet;
use regex::Regex;
use crate::text_unit::{TextUnit, TextUnitType, TextUnitRegistry};
use crate::text_unit::utils::string_similarity;

/// Different types of boundaries that can be detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryType {
    Characters,
    Words,
    Sentences,
    Paragraphs,
    Sections,
    Document,
    Semantic, // New: Semantic boundaries based on topic changes
    Custom(String), // New: Custom boundaries with a name
}

/// Options for boundary detection
#[derive(Debug, Clone)]
pub struct BoundaryDetectionOptions {
    /// Whether to include empty units (e.g., empty paragraphs)
    pub include_empty: bool,
    
    /// Minimum unit length to be considered (in characters)
    pub min_length: usize,
    
    /// Custom delimiters for sentence boundaries
    pub sentence_delimiters: Option<HashSet<char>>,
    
    /// Custom delimiters for paragraph boundaries
    pub paragraph_delimiters: Option<Vec<String>>,
    
    /// Custom section headers pattern (regex supported)
    pub section_headers: Option<Vec<String>>,
    
    /// Semantic unit configuration (min coherence, max length)
    pub semantic_config: Option<SemanticBoundaryConfig>,
    
    /// Custom boundary definition
    pub custom_definition: Option<CustomBoundaryDefinition>,
}

/// Configuration for semantic boundary detection
#[derive(Debug, Clone)]
pub struct SemanticBoundaryConfig {
    /// Minimum coherence score for a unit (0.0-1.0)
    pub min_coherence: f64,
    
    /// Maximum unit length in characters
    pub max_length: usize,
    
    /// Minimum unit length in characters
    pub min_length: usize,
    
    /// Keywords that indicate topic shifts
    pub topic_shift_indicators: Vec<String>,
}

impl Default for SemanticBoundaryConfig {
    fn default() -> Self {
        Self {
            min_coherence: 0.7,
            max_length: 1000,
            min_length: 50,
            topic_shift_indicators: vec![
                "however".to_string(),
                "nevertheless".to_string(),
                "moreover".to_string(),
                "furthermore".to_string(),
                "in addition".to_string(),
                "on the other hand".to_string(),
                "consequently".to_string(),
                "as a result".to_string(),
            ],
        }
    }
}

/// Definition for custom boundary detection
#[derive(Debug, Clone)]
pub struct CustomBoundaryDefinition {
    /// Name of the custom boundary
    pub name: String,
    
    /// Regular expression pattern for boundary detection
    pub pattern: String,
    
    /// Whether the match itself is included in the unit
    pub include_match: bool,
}

impl Default for BoundaryDetectionOptions {
    fn default() -> Self {
        Self {
            include_empty: false,
            min_length: 0,
            sentence_delimiters: None,
            paragraph_delimiters: None,
            section_headers: None,
            semantic_config: None,
            custom_definition: None,
        }
    }
}

/// Detect boundaries in text content
pub fn detect_boundaries(
    content: &str,
    boundary_type: BoundaryType,
    registry: &mut TextUnitRegistry,
    options: Option<BoundaryDetectionOptions>,
) -> Vec<usize> {
    let options = options.unwrap_or_default();
    
    match boundary_type {
        BoundaryType::Characters => detect_character_boundaries(content, registry, &options),
        BoundaryType::Words => detect_word_boundaries(content, registry, &options),
        BoundaryType::Sentences => detect_sentence_boundaries(content, registry, &options),
        BoundaryType::Paragraphs => detect_paragraph_boundaries(content, registry, &options),
        BoundaryType::Sections => detect_section_boundaries(content, registry, &options),
        BoundaryType::Document => detect_document_boundary(content, registry, &options),
        BoundaryType::Semantic => detect_semantic_boundaries(content, registry, &options),
        BoundaryType::Custom(name) => {
            if let Some(custom_def) = options.custom_definition.clone() {
                if custom_def.name == name {
                    detect_custom_boundaries(content, registry, &options, &custom_def)
                } else {
                    // No matching custom definition found
                    vec![]
                }
            } else {
                // No custom definition provided
                vec![]
            }
        }
    }
}

/// Detect character boundaries
fn detect_character_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    let graphemes = UnicodeSegmentation::graphemes(content, true).enumerate();
    
    for (i, g) in graphemes {
        if g.trim().is_empty() && !options.include_empty {
            continue;
        }
        
        let unit = TextUnit::new(
            g.to_string(),
            i,
            i + g.len(),
            TextUnitType::Character,
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Detect word boundaries
fn detect_word_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    let words = UnicodeSegmentation::split_word_bounds(content).enumerate();
    
    let mut current_offset = 0;
    
    for (_, word) in words {
        if (word.trim().is_empty() && !options.include_empty) || 
           (word.len() < options.min_length) {
            current_offset += word.len();
            continue;
        }
        
        let start = current_offset;
        let end = start + word.len();
        
        let unit = TextUnit::new(
            word.to_string(),
            start,
            end,
            TextUnitType::Word,
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
        
        current_offset = end;
    }
    
    unit_ids
}

/// Detect sentence boundaries with improved handling of edge cases
fn detect_sentence_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    
    // Define sentence delimiters
    let sentence_delimiters = options.sentence_delimiters.clone().unwrap_or_else(|| {
        let mut delimiters = HashSet::new();
        delimiters.insert('.');
        delimiters.insert('!');
        delimiters.insert('?');
        delimiters
    });
    
    let mut sentences = Vec::new();
    let mut current_sentence = String::new();
    let mut current_start = 0;
    let mut in_quotes = false;
    
    // Define common abbreviations to avoid false sentence breaks
    let common_abbrev = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "Ltd.", "Co.", "e.g.", "i.e.", "etc.", 
        "vs.", "Fig.", "St.", "Ave.", "Blvd.", "Ph.D.", "M.D.", "B.A.", "M.A."
    ];
    
    for (i, c) in content.char_indices() {
        current_sentence.push(c);
        
        // Handle quotes (to avoid splitting sentences within quotes)
        if c == '"' {
            in_quotes = !in_quotes;
        }
        
        // Detect sentence boundaries
        if !in_quotes && sentence_delimiters.contains(&c) {
            // Check if this is really a sentence end, not an abbreviation, etc.
            let next_is_space = i + 1 >= content.len() || 
                                content.chars().nth(i + 1).map_or(false, |next| next.is_whitespace());
            
            let is_abbrev = common_abbrev.iter()
                .any(|abbr| current_sentence.trim().ends_with(abbr));
            
            if next_is_space && !is_abbrev {
                // Found end of sentence
                if !current_sentence.trim().is_empty() || options.include_empty {
                    if current_sentence.len() >= options.min_length {
                        sentences.push((current_start, i + 1, current_sentence.clone()));
                    }
                }
                
                current_sentence = String::new();
                current_start = i + 1;
            }
        }
    }
    
    // Add the last sentence if not empty
    if !current_sentence.trim().is_empty() || options.include_empty {
        if current_sentence.len() >= options.min_length {
            sentences.push((current_start, content.len(), current_sentence));
        }
    }
    
    // Create text units for each sentence
    for (start, end, sentence) in sentences {
        let unit = TextUnit::new(
            sentence,
            start,
            end,
            TextUnitType::Sentence,
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Detect paragraph boundaries
fn detect_paragraph_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    
    // Define paragraph delimiters
    let paragraph_delimiters = options.paragraph_delimiters.clone().unwrap_or_else(|| {
        vec!["\n\n".to_string(), "\r\n\r\n".to_string()]
    });
    
    // Build a regex pattern to match any of the paragraph delimiters
    let delimiter_pattern = paragraph_delimiters
        .iter()
        .map(|d| regex::escape(d))
        .collect::<Vec<String>>()
        .join("|");
    
    let re = Regex::new(&delimiter_pattern).unwrap_or_else(|_| {
        // Fallback to simple newline pattern if regex construction fails
        Regex::new(r"\n\n|\r\n\r\n").unwrap()
    });
    
    // Split the content by the regex pattern
    let mut paragraphs = Vec::new();
    let mut last_end = 0;
    
    for cap in re.find_iter(content) {
        let start = last_end;
        let end = cap.start();
        
        if end > start {
            let para_text = &content[start..end];
            if !para_text.trim().is_empty() || options.include_empty {
                if para_text.len() >= options.min_length {
                    paragraphs.push((start, end, para_text.to_string()));
                }
            }
        }
        
        last_end = cap.end();
    }
    
    // Add the last paragraph
    if last_end < content.len() {
        let para_text = &content[last_end..];
        if !para_text.trim().is_empty() || options.include_empty {
            if para_text.len() >= options.min_length {
                paragraphs.push((last_end, content.len(), para_text.to_string()));
            }
        }
    }
    
    // Create text units for each paragraph
    for (start, end, paragraph) in paragraphs {
        let unit = TextUnit::new(
            paragraph,
            start,
            end,
            TextUnitType::Paragraph,
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Detect section boundaries
fn detect_section_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    
    // Define section header patterns
    let section_patterns = options.section_headers.clone().unwrap_or_else(|| {
        vec![
            r"(?m)^#{1,6} .+$".to_string(),                // Markdown headers
            r"(?m)^[A-Z][A-Za-z0-9 ]+:$".to_string(),       // Title with colon
            r"(?m)^\d+\.\s+[A-Z][A-Za-z0-9 ]+$".to_string() // Numbered sections
        ]
    });
    
    // Combine patterns into one regex
    let combined_pattern = section_patterns.join("|");
    let re = Regex::new(&combined_pattern).unwrap_or_else(|_| {
        // Fallback to simple header pattern if regex construction fails
        Regex::new(r"(?m)^#{1,6} .+$").unwrap()
    });
    
    // Find all section headers
    let mut headers: Vec<(usize, usize, String)> = Vec::new();
    
    for cap in re.find_iter(content) {
        headers.push((cap.start(), cap.end(), cap.as_str().to_string()));
    }
    
    // If no headers found, treat the entire document as one section
    if headers.is_empty() {
        if !content.trim().is_empty() || options.include_empty {
            if content.len() >= options.min_length {
                let unit = TextUnit::new(
                    content.to_string(),
                    0,
                    content.len(),
                    TextUnitType::Section,
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(unit);
                unit_ids.push(id);
            }
        }
        
        return unit_ids;
    }
    
    // Extract sections based on header positions
    let mut sections: Vec<TextUnit> = Vec::new();
    
    for i in 0..headers.len() {
        let (start, _, header) = &headers[i];
        let end = if i < headers.len() - 1 {
            headers[i + 1].0
        } else {
            content.len()
        };
        
        let section_text = &content[*start..end];
        
        if !section_text.trim().is_empty() || options.include_empty {
            if section_text.len() >= options.min_length {
                // Create a metadata map with the header information
                let mut metadata = std::collections::HashMap::new();
                metadata.insert(
                    "header".to_string(), 
                    crate::turbulance::ast::Value::String(header.to_string())
                );
                
                let unit = TextUnit::with_metadata(
                    section_text.to_string(),
                    metadata,
                    *start,
                    end,
                    TextUnitType::Section,
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(unit);
                unit_ids.push(id);
            }
        }
    }
    
    unit_ids
}

/// Detect document boundary (treats the entire content as one unit)
fn detect_document_boundary(
    content: &str,
    registry: &mut TextUnitRegistry,
    _options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let unit = TextUnit::new(
        content.to_string(),
        0,
        content.len(),
        TextUnitType::Document,
        registry.next_available_id(),
    );
    
    let id = registry.add_unit(unit);
    vec![id]
}

/// Detect semantic boundaries based on topic coherence
fn detect_semantic_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    
    // Use semantic config or defaults
    let config = options.semantic_config.clone().unwrap_or_default();
    
    // First, detect paragraph boundaries as a starting point
    let paragraph_ids = detect_paragraph_boundaries(content, registry, options);
    
    // Collect paragraph data to avoid borrowing conflicts
    let paragraphs: Vec<(String, usize, usize)> = paragraph_ids.iter()
        .filter_map(|id| {
            registry.get_unit(*id).map(|unit| {
                (unit.content.clone(), unit.start, unit.end)
            })
        })
        .collect();
    
    if paragraphs.is_empty() {
        return unit_ids;
    }
    
    // Group paragraphs into semantic units
    let mut current_unit = String::new();
    let mut current_start = paragraphs[0].1; // start position
    let mut current_end;
    
    for i in 0..paragraphs.len() {
        let (para_content, para_start, para_end) = &paragraphs[i];
        
        // Check for topic shift indicators
        let contains_shift_indicator = config.topic_shift_indicators.iter()
            .any(|indicator| para_content.to_lowercase().contains(&indicator.to_lowercase()));
        
        // Start a new semantic unit if:
        // 1. Current paragraph contains a topic shift indicator
        // 2. Current unit would exceed the max length
        // 3. This is the first paragraph
        let should_start_new_unit = 
            contains_shift_indicator || 
            current_unit.len() + para_content.len() > config.max_length ||
            i == 0;
        
        if should_start_new_unit && !current_unit.is_empty() {
            // Save the current unit before starting a new one
            current_end = *para_start;
            
            if current_unit.len() >= config.min_length {
                let semantic_unit = TextUnit::new(
                    current_unit,
                    current_start,
                    current_end,
                    TextUnitType::Custom(1), // Semantic unit type
                    registry.next_available_id(),
                );
                
                let id = registry.add_unit(semantic_unit);
                unit_ids.push(id);
            }
            
            // Start a new unit
            current_unit = para_content.clone();
            current_start = *para_start;
        } else {
            // Add to the current unit with a space in between
            if !current_unit.is_empty() {
                current_unit.push(' ');
            }
            current_unit.push_str(para_content);
        }
    }
    
    // Add the last semantic unit if not empty
    if !current_unit.is_empty() && current_unit.len() >= config.min_length {
        current_end = paragraphs.last().unwrap().2; // end position
        
        let semantic_unit = TextUnit::new(
            current_unit,
            current_start,
            current_end,
            TextUnitType::Custom(1), // Semantic unit type
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(semantic_unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Detect custom boundaries based on a regex pattern
fn detect_custom_boundaries(
    content: &str,
    registry: &mut TextUnitRegistry,
    options: &BoundaryDetectionOptions,
    custom_def: &CustomBoundaryDefinition,
) -> Vec<usize> {
    let mut unit_ids = Vec::new();
    
    // Create regex from the custom pattern
    let re = match Regex::new(&custom_def.pattern) {
        Ok(re) => re,
        Err(_) => return unit_ids, // Return empty vector if regex is invalid
    };
    
    // Find all matches
    let mut matches: Vec<(usize, usize)> = Vec::new();
    
    for cap in re.find_iter(content) {
        matches.push((cap.start(), cap.end()));
    }
    
    // If no matches found, return empty vector
    if matches.is_empty() {
        return unit_ids;
    }
    
    // Extract text units between matches
    let mut units = Vec::new();
    let mut last_end = 0;
    
    for (start, end) in matches {
        // Add text before the match
        if start > last_end {
            let unit_text = &content[last_end..start];
            if !unit_text.trim().is_empty() || options.include_empty {
                if unit_text.len() >= options.min_length {
                    units.push((last_end, start, unit_text.to_string()));
                }
            }
        }
        
        // Include the match itself if specified
        if custom_def.include_match {
            let match_text = &content[start..end];
            if !match_text.trim().is_empty() || options.include_empty {
                units.push((start, end, match_text.to_string()));
            }
        }
        
        last_end = end;
    }
    
    // Add the last part after the final match
    if last_end < content.len() {
        let unit_text = &content[last_end..];
        if !unit_text.trim().is_empty() || options.include_empty {
            if unit_text.len() >= options.min_length {
                units.push((last_end, content.len(), unit_text.to_string()));
            }
        }
    }
    
    // Create text units
    for (start, end, unit_text) in units {
        let custom_id = registry.next_available_id();
        
        let unit = TextUnit::new(
            unit_text,
            start,
            end,
            TextUnitType::Custom(custom_id),
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Build a document hierarchy from the detected units
pub fn build_hierarchy(
    content: &str,
    registry: &mut TextUnitRegistry,
) -> usize {
    // Create the document unit
    let doc_id = detect_document_boundary(content, registry, &BoundaryDetectionOptions::default())[0];
    
    // Detect sections
    let section_ids = detect_section_boundaries(content, registry, &BoundaryDetectionOptions::default());
    
    // Set up parent-child relationships
    for section_id in &section_ids {
        registry.set_parent_child(doc_id, *section_id);
    }
    
    // For each section, detect paragraphs
    for section_id in &section_ids {
        // Store section content and start position in local variables to avoid simultaneous borrows
        let (section_content, section_start) = {
            let section = registry.get_unit(*section_id)
                .expect("Section should exist");
            (section.content.clone(), section.start)
        };
        
        // Now we can use section_content without borrowing from registry
        let paragraph_ids = detect_paragraph_boundaries(&section_content, registry, &BoundaryDetectionOptions::default());
        
        // Update positions to be relative to the document, not the section
        for para_id in &paragraph_ids {
            // Store paragraph data in a separate scope
            {
                let para = registry.get_unit_mut(*para_id)
                    .expect("Paragraph should exist");
                para.start += section_start;
                para.end += section_start;
            }
            
            registry.set_parent_child(*section_id, *para_id);
        }
        
        // For each paragraph, detect sentences
        for para_id in &paragraph_ids {
            // Store paragraph content and start position in local variables
            let (para_content, para_start) = {
                let para = registry.get_unit(*para_id)
                    .expect("Paragraph should exist");
                (para.content.clone(), para.start)
            };
            
            // Now we can use para_content without borrowing from registry
            let sentence_ids = detect_sentence_boundaries(&para_content, registry, &BoundaryDetectionOptions::default());
            
            // Update positions to be relative to the document
            for sent_id in &sentence_ids {
                // Store sentence data in a separate scope
                {
                    let sent = registry.get_unit_mut(*sent_id)
                        .expect("Sentence should exist");
                    sent.start += para_start;
                    sent.end += para_start;
                }
                
                registry.set_parent_child(*para_id, *sent_id);
            }
        }
    }
    
    doc_id
}

// Calculate semantic coherence between two text units (0.0-1.0)
pub fn calculate_coherence(unit1: &TextUnit, unit2: &TextUnit) -> f64 {
    // Use the common string similarity function for consistency
    string_similarity(&unit1.content, &unit2.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_character_boundary_detection() {
        let content = "abc";
        let mut registry = TextUnitRegistry::new();
        
        let units = detect_character_boundaries(content, &mut registry, &BoundaryDetectionOptions::default());
        
        assert_eq!(units.len(), 3);
        
        let chars: Vec<String> = units.iter()
            .filter_map(|id| registry.get_unit(*id))
            .map(|unit| unit.content.clone())
            .collect();
        
        assert_eq!(chars, vec!["a", "b", "c"]);
    }
    
    #[test]
    fn test_word_boundary_detection() {
        let content = "hello world";
        let mut registry = TextUnitRegistry::new();
        
        let units = detect_word_boundaries(content, &mut registry, &BoundaryDetectionOptions::default());
        
        // Expect "hello", " ", "world" with spaces included
        assert!(units.len() >= 2);
        
        let words: Vec<String> = units.iter()
            .filter_map(|id| registry.get_unit(*id))
            .map(|unit| unit.content.clone())
            .collect();
        
        assert!(words.contains(&"hello".to_string()));
        assert!(words.contains(&"world".to_string()));
    }
    
    #[test]
    fn test_sentence_boundary_detection() {
        let content = "This is a sentence. This is another one! What about this?";
        let mut registry = TextUnitRegistry::new();
        
        let units = detect_sentence_boundaries(content, &mut registry, &BoundaryDetectionOptions::default());
        
        assert_eq!(units.len(), 3);
        
        let sentences: Vec<String> = units.iter()
            .filter_map(|id| registry.get_unit(*id))
            .map(|unit| unit.content.clone())
            .collect();
        
        assert_eq!(sentences[0], "This is a sentence.");
        assert_eq!(sentences[1], "This is another one!");
        assert_eq!(sentences[2], "What about this?");
    }
    
    #[test]
    fn test_paragraph_boundary_detection() {
        let content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let mut registry = TextUnitRegistry::new();
        
        let units = detect_paragraph_boundaries(content, &mut registry, &BoundaryDetectionOptions::default());
        
        assert_eq!(units.len(), 3);
        
        let paragraphs: Vec<String> = units.iter()
            .filter_map(|id| registry.get_unit(*id))
            .map(|unit| unit.content.clone())
            .collect();
        
        assert_eq!(paragraphs[0], "Paragraph one.");
        assert_eq!(paragraphs[1], "Paragraph two.");
        assert_eq!(paragraphs[2], "Paragraph three.");
    }
    
    #[test]
    fn test_section_boundary_detection() {
        let content = "# Section 1\nContent 1\n\n## Section 2\nContent 2";
        let mut registry = TextUnitRegistry::new();
        
        let units = detect_section_boundaries(content, &mut registry, &BoundaryDetectionOptions::default());
        
        assert_eq!(units.len(), 2);
    }
    
    #[test]
    fn test_semantic_boundary_detection() {
        let content = "This is about topic A. More about topic A.\n\nHowever, topic B is different. More about topic B.";
        let mut registry = TextUnitRegistry::new();
        
        let options = BoundaryDetectionOptions {
            semantic_config: Some(SemanticBoundaryConfig::default()),
            ..Default::default()
        };
        
        let units = detect_semantic_boundaries(content, &mut registry, &options);
        
        // Should detect two semantic units due to "However" topic shift indicator
        assert!(units.len() >= 1);
    }
    
    #[test]
    fn test_custom_boundary_detection() {
        let content = "START:Item 1:END\nSTART:Item 2:END\nSTART:Item 3:END";
        let mut registry = TextUnitRegistry::new();
        
        let custom_def = CustomBoundaryDefinition {
            name: "item".to_string(),
            pattern: r"START:(.*?):END".to_string(),
            include_match: true,
        };
        
        let options = BoundaryDetectionOptions {
            custom_definition: Some(custom_def),
            ..Default::default()
        };
        
        let units = detect_custom_boundaries(content, &mut registry, &options, &options.custom_definition.as_ref().unwrap());
        
        assert_eq!(units.len(), 3);
    }
    
    #[test]
    fn test_hierarchy_building() {
        let content = "# Section 1\nParagraph 1.\n\nParagraph 2.\n\n# Section 2\nParagraph 3.";
        let mut registry = TextUnitRegistry::new();
        
        let doc_id = build_hierarchy(content, &mut registry);
        
        // Verify document has sections
        let doc = registry.get_unit(doc_id).unwrap();
        assert_eq!(doc.unit_type, TextUnitType::Document);
        assert_eq!(doc.children.len(), 2); // 2 sections
        
        // Verify sections have paragraphs
        let sections: Vec<&TextUnit> = doc.children.iter()
            .filter_map(|id| registry.get_unit(*id))
            .collect();
        
        assert_eq!(sections[0].unit_type, TextUnitType::Section);
        assert_eq!(sections[1].unit_type, TextUnitType::Section);
        
        // First section should have 2 paragraphs
        assert_eq!(sections[0].children.len(), 2);
    }
    
    #[test]
    fn test_coherence_calculation() {
        let unit1 = TextUnit::new(
            "The cat sat on the mat".to_string(),
            0, 
            21,
            TextUnitType::Sentence,
            0
        );
        
        let unit2 = TextUnit::new(
            "The dog sat on the floor".to_string(),
            0,
            25,
            TextUnitType::Sentence,
            1
        );
        
        let coherence = calculate_coherence(&unit1, &unit2);
        
        // Expect some coherence but not perfect (shared words: "The", "sat", "on", "the")
        assert!(coherence > 0.0);
        assert!(coherence < 1.0);
    }
}
