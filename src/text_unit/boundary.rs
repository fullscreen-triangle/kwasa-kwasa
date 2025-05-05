use unicode_segmentation::UnicodeSegmentation;
use std::collections::HashSet;
use crate::text_unit::{TextUnit, TextUnitType, TextUnitRegistry};

/// Different types of boundaries that can be detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryType {
    Characters,
    Words,
    Sentences,
    Paragraphs,
    Sections,
    Document,
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
    
    /// Custom section headers pattern (regex not implemented yet)
    pub section_headers: Option<Vec<String>>,
}

impl Default for BoundaryDetectionOptions {
    fn default() -> Self {
        Self {
            include_empty: false,
            min_length: 0,
            sentence_delimiters: None,
            paragraph_delimiters: None,
            section_headers: None,
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
    
    for (i, word) in words {
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

/// Detect sentence boundaries
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
    
    for (i, c) in content.char_indices() {
        current_sentence.push(c);
        
        // Handle quotes (to avoid splitting sentences within quotes)
        if c == '"' {
            in_quotes = !in_quotes;
        }
        
        // Detect sentence boundaries
        if !in_quotes && sentence_delimiters.contains(&c) {
            // Check if this is really a sentence end, not an abbreviation, etc.
            // (basic heuristic, can be improved)
            let is_sentence_end = i + 1 >= content.len() || 
                                 content.chars().nth(i + 1).map_or(false, |next| next.is_whitespace());
            
            if is_sentence_end {
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
    
    // Split content into paragraphs
    let mut paragraphs = Vec::new();
    let mut current_paragraph = String::new();
    let mut current_start = 0;
    let mut i = 0;
    
    while i < content.len() {
        let mut delimiter_matched = false;
        
        for delimiter in &paragraph_delimiters {
            if i + delimiter.len() <= content.len() && 
               &content[i..i + delimiter.len()] == delimiter {
                // Found a paragraph delimiter
                if !current_paragraph.trim().is_empty() || options.include_empty {
                    if current_paragraph.len() >= options.min_length {
                        paragraphs.push((current_start, i, current_paragraph.clone()));
                    }
                }
                
                current_paragraph = String::new();
                current_start = i + delimiter.len();
                i += delimiter.len();
                delimiter_matched = true;
                break;
            }
        }
        
        if !delimiter_matched {
            if let Some(c) = content.chars().nth(i) {
                current_paragraph.push(c);
                i += c.len_utf8();
            } else {
                break;
            }
        }
    }
    
    // Add the last paragraph if not empty
    if !current_paragraph.trim().is_empty() || options.include_empty {
        if current_paragraph.len() >= options.min_length {
            paragraphs.push((current_start, content.len(), current_paragraph));
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
    
    // Define section headers (default: Markdown-style headers)
    let section_headers = options.section_headers.clone().unwrap_or_else(|| {
        vec![
            "# ".to_string(),    // H1
            "## ".to_string(),   // H2
            "### ".to_string(),  // H3
            "#### ".to_string(), // H4
        ]
    });
    
    // Split content into lines
    let lines: Vec<&str> = content.lines().collect();
    
    // Find section boundaries
    let mut sections = Vec::new();
    let mut current_section = String::new();
    let mut current_start = 0;
    let mut line_start = 0;
    
    for (i, line) in lines.iter().enumerate() {
        let is_header = section_headers.iter()
            .any(|header| line.trim().starts_with(header));
        
        if is_header && i > 0 {
            // Found a new section, add the previous one if not empty
            if !current_section.trim().is_empty() || options.include_empty {
                if current_section.len() >= options.min_length {
                    sections.push((current_start, line_start, current_section.clone()));
                }
            }
            
            current_section = String::new();
            current_start = line_start;
        }
        
        // Add line to current section
        if !current_section.is_empty() {
            current_section.push('\n');
        }
        current_section.push_str(line);
        
        // Update line_start for the next line
        line_start += line.len() + 1; // +1 for the newline
    }
    
    // Add the last section if not empty
    if !current_section.trim().is_empty() || options.include_empty {
        if current_section.len() >= options.min_length {
            sections.push((current_start, content.len(), current_section));
        }
    }
    
    // Create text units for each section
    for (start, end, section) in sections {
        let unit = TextUnit::new(
            section,
            start,
            end,
            TextUnitType::Section,
            registry.next_available_id(),
        );
        
        let id = registry.add_unit(unit);
        unit_ids.push(id);
    }
    
    unit_ids
}

/// Detect document boundary (the entire content as one unit)
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

/// Build a hierarchical structure of text units
pub fn build_hierarchy(
    content: &str,
    registry: &mut TextUnitRegistry,
) -> usize {
    // First detect all boundaries
    let document_ids = detect_document_boundary(content, registry, &BoundaryDetectionOptions::default());
    let document_id = document_ids[0];
    
    let section_ids = detect_section_boundaries(content, registry, &BoundaryDetectionOptions::default());
    let paragraph_ids = detect_paragraph_boundaries(content, registry, &BoundaryDetectionOptions::default());
    let sentence_ids = detect_sentence_boundaries(content, registry, &BoundaryDetectionOptions::default());
    
    // Build the hierarchy: Document -> Sections -> Paragraphs -> Sentences
    
    // Link sections to document
    for &section_id in &section_ids {
        registry.set_parent_child(document_id, section_id);
    }
    
    // Link paragraphs to sections or document
    for &paragraph_id in &paragraph_ids {
        let paragraph = registry.get_unit(paragraph_id).unwrap().clone();
        
        // Find the section that contains this paragraph
        let containing_section = section_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .find(|section| {
                paragraph.start >= section.start && paragraph.end <= section.end
            });
        
        if let Some(section) = containing_section {
            registry.set_parent_child(section.id, paragraph_id);
        } else {
            // If no containing section, link directly to document
            registry.set_parent_child(document_id, paragraph_id);
        }
    }
    
    // Link sentences to paragraphs
    for &sentence_id in &sentence_ids {
        let sentence = registry.get_unit(sentence_id).unwrap().clone();
        
        // Find the paragraph that contains this sentence
        let containing_paragraph = paragraph_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .find(|paragraph| {
                sentence.start >= paragraph.start && sentence.end <= paragraph.end
            });
        
        if let Some(paragraph) = containing_paragraph {
            registry.set_parent_child(paragraph.id, sentence_id);
        }
    }
    
    document_id
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_character_boundary_detection() {
        let content = "Hello";
        let mut registry = TextUnitRegistry::new();
        
        let character_ids = detect_character_boundaries(
            content,
            &mut registry,
            &BoundaryDetectionOptions::default(),
        );
        
        assert_eq!(character_ids.len(), 5); // "H", "e", "l", "l", "o"
        
        let characters: Vec<&str> = character_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .map(|unit| unit.content.as_str())
            .collect();
        
        assert_eq!(characters, vec!["H", "e", "l", "l", "o"]);
    }
    
    #[test]
    fn test_word_boundary_detection() {
        let content = "This is a test.";
        let mut registry = TextUnitRegistry::new();
        
        let word_ids = detect_word_boundaries(
            content,
            &mut registry,
            &BoundaryDetectionOptions::default(),
        );
        
        // Expected: "This", " ", "is", " ", "a", " ", "test", "."
        assert_eq!(word_ids.len(), 8);
        
        let words: Vec<&str> = word_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .map(|unit| unit.content.as_str())
            .collect();
        
        assert_eq!(words, vec!["This", " ", "is", " ", "a", " ", "test", "."]);
    }
    
    #[test]
    fn test_sentence_boundary_detection() {
        let content = "This is the first sentence. This is the second! And this is the third?";
        let mut registry = TextUnitRegistry::new();
        
        let sentence_ids = detect_sentence_boundaries(
            content,
            &mut registry,
            &BoundaryDetectionOptions::default(),
        );
        
        assert_eq!(sentence_ids.len(), 3);
        
        let sentences: Vec<&str> = sentence_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .map(|unit| unit.content.as_str())
            .collect();
        
        assert_eq!(sentences, vec![
            "This is the first sentence.", 
            "This is the second!", 
            "And this is the third?"
        ]);
    }
    
    #[test]
    fn test_paragraph_boundary_detection() {
        let content = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.";
        let mut registry = TextUnitRegistry::new();
        
        let paragraph_ids = detect_paragraph_boundaries(
            content,
            &mut registry,
            &BoundaryDetectionOptions::default(),
        );
        
        assert_eq!(paragraph_ids.len(), 3);
        
        let paragraphs: Vec<&str> = paragraph_ids.iter()
            .filter_map(|&id| registry.get_unit(id))
            .map(|unit| unit.content.as_str())
            .collect();
        
        assert_eq!(paragraphs, vec![
            "This is paragraph one.",
            "This is paragraph two.",
            "This is paragraph three."
        ]);
    }
    
    #[test]
    fn test_section_boundary_detection() {
        let content = "# Section 1\nThis is content in section 1.\n\n## Subsection 1.1\nThis is subsection content.\n\n# Section 2\nThis is content in section 2.";
        let mut registry = TextUnitRegistry::new();
        
        let section_ids = detect_section_boundaries(
            content,
            &mut registry,
            &BoundaryDetectionOptions::default(),
        );
        
        // Should detect 3 sections: Section 1, Subsection 1.1, Section 2
        assert_eq!(section_ids.len(), 3);
    }
    
    #[test]
    fn test_hierarchy_building() {
        let content = "# Section 1\n\nThis is paragraph one.\nThis continues paragraph one.\n\nThis is paragraph two.\n\n# Section 2\n\nThis is paragraph three.";
        let mut registry = TextUnitRegistry::new();
        
        let document_id = build_hierarchy(content, &mut registry);
        
        // Get the document
        let document = registry.get_unit(document_id).unwrap();
        assert_eq!(document.unit_type, TextUnitType::Document);
        
        // Document should have 2 sections as children
        let sections = registry.children_of(document_id);
        assert_eq!(sections.len(), 2);
        assert!(sections.iter().all(|s| s.unit_type == TextUnitType::Section));
        
        // First section should have 2 paragraphs
        let first_section = sections[0];
        let first_section_paragraphs = registry.children_of(first_section.id);
        assert_eq!(first_section_paragraphs.len(), 2);
        
        // Second section should have 1 paragraph
        let second_section = sections[1];
        let second_section_paragraphs = registry.children_of(second_section.id);
        assert_eq!(second_section_paragraphs.len(), 1);
    }
}
