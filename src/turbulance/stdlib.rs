use std::collections::HashMap;
use crate::turbulance::ast::Value;
use crate::turbulance::TurbulanceError;
use crate::text_unit::boundary::TextUnit;
use regex;

/// Type alias for standard library functions
pub type StdLibFn = fn(Vec<Value>) -> Result<Value, TurbulanceError>;

/// Represents the standard library for Turbulance
pub struct StdLib {
    /// Map of function names to implementations
    functions: HashMap<String, StdLibFn>,
}

impl StdLib {
    /// Create a new standard library with all built-in functions
    pub fn new() -> Self {
        let mut stdlib = Self {
            functions: HashMap::new(),
        };
        
        // Register all standard library functions
        stdlib.register_all();
        
        stdlib
    }
    
    /// Register all standard library functions
    fn register_all(&mut self) {
        // Text analysis functions
        self.register("readability_score", readability_score);
        self.register("contains", contains);
        self.register("extract_patterns", extract_patterns);
        
        // Text transformation functions
        self.register("research_context", research_context);
        self.register("ensure_explanation_follows", ensure_explanation_follows);
        self.register("simplify_sentences", simplify_sentences);
        self.register("replace_jargon", replace_jargon);
        
        // Utility functions
        self.register("print", print);
        self.register("len", len);
        self.register("typeof", typeof);
    }
    
    /// Register a standard library function
    fn register(&mut self, name: &str, func: StdLibFn) {
        self.functions.insert(name.to_string(), func);
    }
    
    /// Call a standard library function
    pub fn call(&self, name: &str, args: Vec<Value>) -> Result<Value, TurbulanceError> {
        match self.functions.get(name) {
            Some(func) => func(args),
            None => Err(TurbulanceError::RuntimeError {
                message: format!("Unknown function: {}", name),
            }),
        }
    }
    
    /// Check if a function exists in the standard library
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
    
    /// Get a list of all available functions
    pub fn list_functions(&self) -> Vec<String> {
        self.functions.keys().cloned().collect()
    }
}

// ========== Text Analysis Functions ==========

/// Calculate readability score for a text
fn readability_score(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement readability scoring
    // This is a stub implementation
    
    // Check that we have at least one argument
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "readability_score requires a text argument".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "readability_score expects a string or text unit".to_string(),
            });
        }
    };
    
    // Placeholder implementation
    // In a real implementation, this would use proper readability metrics
    let words = text.split_whitespace().count();
    let sentences = text.split(&['.', '!', '?']).filter(|s| !s.trim().is_empty()).count();
    
    if sentences == 0 {
        return Ok(Value::Number(0.0));
    }
    
    // Simple average words per sentence as placeholder
    let avg_words_per_sentence = words as f64 / sentences as f64;
    let score = 100.0 - (avg_words_per_sentence * 5.0).min(100.0);
    
    Ok(Value::Number(score))
}

/// Check if text contains a pattern
fn contains(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // TODO: Implement pattern matching
    // This is a stub implementation
    
    // Check that we have at least two arguments
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contains requires text and pattern arguments".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "contains expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Extract the pattern argument
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "contains expects a string pattern as second argument".to_string(),
            });
        }
    };
    
    // Simple string contains check as placeholder
    // In a real implementation, this would use more sophisticated pattern matching
    Ok(Value::Bool(text.contains(pattern)))
}

/// Extract patterns from text
fn extract_patterns(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // Check that we have at least two arguments
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "extract_patterns requires text and pattern arguments".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "extract_patterns expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Extract the pattern argument
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "extract_patterns expects a string pattern as second argument".to_string(),
            });
        }
    };
    
    // Create a regex from the pattern
    let regex = match regex::Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => {
            return Err(TurbulanceError::RuntimeError {
                message: format!("Invalid regex pattern: {}", e),
            });
        }
    };
    
    // Find all matches
    let mut matches = Vec::new();
    for capture in regex.captures_iter(text) {
        if let Some(m) = capture.get(0) {
            matches.push(Value::String(m.as_str().to_string()));
        }
    }
    
    Ok(Value::List(matches))
}

// ========== Text Transformation Functions ==========

/// Research context for a domain
fn research_context(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // Check that we have at least one argument
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "research_context requires a domain argument".to_string(),
        });
    }
    
    // Extract the domain argument
    let domain = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "research_context expects a string domain as first argument".to_string(),
            });
        }
    };
    
    // Optional depth parameter
    let depth = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => *n as usize,
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "research_context expects a number for depth as second argument".to_string(),
                });
            }
        }
    } else {
        1 // Default depth
    };
    
    // Create a map of domain knowledge (this would connect to a real knowledge base in production)
    let mut context = HashMap::new();
    
    // Add some basic context based on the domain (simulated knowledge)
    match domain.to_lowercase().as_str() {
        "medicine" => {
            context.insert("field".to_string(), Value::String("Healthcare and medical science".to_string()));
            context.insert("key_concepts".to_string(), Value::List(vec![
                Value::String("diagnosis".to_string()),
                Value::String("treatment".to_string()),
                Value::String("prognosis".to_string()),
                Value::String("etiology".to_string()),
            ]));
            if depth > 1 {
                context.insert("common_abbreviations".to_string(), Value::Map({
                    let mut map = HashMap::new();
                    map.insert("BP".to_string(), Value::String("Blood Pressure".to_string()));
                    map.insert("HR".to_string(), Value::String("Heart Rate".to_string()));
                    map.insert("Rx".to_string(), Value::String("Prescription".to_string()));
                    map
                }));
            }
        },
        "computer_science" => {
            context.insert("field".to_string(), Value::String("Computing and software engineering".to_string()));
            context.insert("key_concepts".to_string(), Value::List(vec![
                Value::String("algorithms".to_string()),
                Value::String("data structures".to_string()),
                Value::String("programming languages".to_string()),
                Value::String("software design".to_string()),
            ]));
            if depth > 1 {
                context.insert("common_abbreviations".to_string(), Value::Map({
                    let mut map = HashMap::new();
                    map.insert("OOP".to_string(), Value::String("Object-Oriented Programming".to_string()));
                    map.insert("API".to_string(), Value::String("Application Programming Interface".to_string()));
                    map.insert("SQL".to_string(), Value::String("Structured Query Language".to_string()));
                    map
                }));
            }
        },
        "linguistics" => {
            context.insert("field".to_string(), Value::String("Study of language".to_string()));
            context.insert("key_concepts".to_string(), Value::List(vec![
                Value::String("phonology".to_string()),
                Value::String("morphology".to_string()),
                Value::String("syntax".to_string()),
                Value::String("semantics".to_string()),
            ]));
            if depth > 1 {
                context.insert("common_abbreviations".to_string(), Value::Map({
                    let mut map = HashMap::new();
                    map.insert("NLP".to_string(), Value::String("Natural Language Processing".to_string()));
                    map.insert("SLA".to_string(), Value::String("Second Language Acquisition".to_string()));
                    map.insert("IPA".to_string(), Value::String("International Phonetic Alphabet".to_string()));
                    map
                }));
            }
        },
        _ => {
            // Generic context for unrecognized domains
            context.insert("field".to_string(), Value::String(format!("Study of {}", domain)));
            context.insert("note".to_string(), Value::String("Detailed context not available for this domain.".to_string()));
        }
    }
    
    Ok(Value::Map(context))
}

/// Ensure explanation follows a term
fn ensure_explanation_follows(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // Check that we have at least two arguments
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows requires text and term arguments".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        Value::TextUnit(tu) => tu.content.clone(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "ensure_explanation_follows expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Extract the term argument
    let term = match &args[1] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "ensure_explanation_follows expects a string term as second argument".to_string(),
            });
        }
    };
    
    // Optional explanation to use if not found
    let default_explanation = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.clone()),
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "ensure_explanation_follows expects a string for explanation as third argument".to_string(),
                });
            }
        }
    } else {
        None
    };
    
    // Check if term is followed by an explanation
    // We'll consider an explanation to be present if there is:
    // 1. A colon or dash after the term
    // 2. A definition in parentheses
    // 3. Text beginning with "which is", "that is", etc.
    
    // Create a regex to find the term and see if it's followed by an explanation
    let term_pattern = format!(r"{}((\s*[:–—-]\s*)|(\s+\()|(\s+which\s+is)|(\s+that\s+is)|(\s+is\s+defined\s+as))", regex::escape(term));
    let regex = match regex::Regex::new(&term_pattern) {
        Ok(r) => r,
        Err(e) => {
            return Err(TurbulanceError::RuntimeError {
                message: format!("Invalid regex pattern: {}", e),
            });
        }
    };
    
    if regex.is_match(&text) {
        // Explanation exists
        Ok(Value::String(text))
    } else {
        // No explanation found, add one if provided
        if let Some(explanation) = default_explanation {
            // Find the last occurrence of the term
            if let Some(pos) = text.to_lowercase().rfind(&term.to_lowercase()) {
                let end_pos = pos + term.len();
                let mut result = text.clone();
                let insert_text = format!(" ({})", explanation);
                result.insert_str(end_pos, &insert_text);
                Ok(Value::String(result))
            } else {
                // Term not found at all
                Ok(Value::String(text))
            }
        } else {
            // No explanation found and none provided
            Err(TurbulanceError::RuntimeError {
                message: format!("Term '{}' is not followed by an explanation", term),
            })
        }
    }
}

/// Simplify sentences in text
fn simplify_sentences(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // Check that we have at least one argument
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "simplify_sentences requires a text argument".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        Value::TextUnit(tu) => tu.content.clone(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "simplify_sentences expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Optional level of simplification (1-3, where 3 is most aggressive)
    let level = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => {
                let level = *n as usize;
                if level < 1 || level > 3 {
                    return Err(TurbulanceError::RuntimeError {
                        message: "simplify_sentences level must be between 1 and 3".to_string(),
                    });
                }
                level
            }
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "simplify_sentences expects a number for level as second argument".to_string(),
                });
            }
        }
    } else {
        1 // Default level
    };
    
    // Split text into sentences
    let sentence_endings = ['.', '!', '?'];
    let mut sentences: Vec<String> = Vec::new();
    let mut current_sentence = String::new();
    
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        current_sentence.push(c);
        
        if sentence_endings.contains(&c) {
            // Check for abbreviations like "Dr.", "Ms.", etc.
            let is_abbreviation = if current_sentence.len() >= 3 {
                let last_word_start = current_sentence
                    .trim_end_matches(|c: char| !c.is_alphanumeric())
                    .rfind(|c: char| c.is_whitespace())
                    .map(|pos| pos + 1)
                    .unwrap_or(0);
                
                let last_word = &current_sentence[last_word_start..];
                matches!(last_word, "Dr." | "Mr." | "Mrs." | "Ms." | "vs." | "etc." | "i.e." | "e.g.")
            } else {
                false
            };
            
            // Skip if this is an abbreviation
            if is_abbreviation {
                continue;
            }
            
            // Check for end of sentence (followed by space or end of text)
            let is_end_of_sentence = chars.peek().map_or(true, |next_char| next_char.is_whitespace());
            
            if is_end_of_sentence {
                sentences.push(current_sentence.trim().to_string());
                current_sentence.clear();
            }
        }
    }
    
    // If there's anything left, add it as a sentence
    if !current_sentence.is_empty() {
        sentences.push(current_sentence.trim().to_string());
    }
    
    // Apply simplification techniques based on level
    let simplified_sentences: Vec<String> = sentences.into_iter().map(|sentence| {
        let mut simplified = sentence.clone();
        
        // Level 1: Basic simplification
        // - Remove excessive adverbs
        simplified = remove_excessive_adverbs(&simplified);
        
        if level >= 2 {
            // Level 2: Moderate simplification
            // - Simplify complex phrases
            simplified = simplify_complex_phrases(&simplified);
            // - Reduce nested clauses
            simplified = reduce_nested_clauses(&simplified);
        }
        
        if level >= 3 {
            // Level 3: Advanced simplification
            // - Split long sentences
            simplified = split_long_sentences(&simplified);
            // - Favor active voice
            simplified = favor_active_voice(&simplified);
        }
        
        simplified
    }).collect();
    
    // Join simplified sentences
    let result = simplified_sentences.join(" ");
    
    Ok(Value::String(result))
}

/// Helper function to remove excessive adverbs
fn remove_excessive_adverbs(text: &str) -> String {
    // List of common unnecessary adverbs
    let adverbs = [
        "very", "really", "quite", "simply", "just", "completely", "totally",
        "absolutely", "actually", "basically", "literally", "definitely",
    ];
    
    let mut result = text.to_string();
    for adverb in adverbs.iter() {
        // Match adverb with space boundaries
        let pattern = format!(r"\s{}\s", adverb);
        let re = regex::Regex::new(&pattern).unwrap();
        result = re.replace_all(&result, " ").to_string();
    }
    
    result
}

/// Helper function to simplify complex phrases
fn simplify_complex_phrases(text: &str) -> String {
    let replacements = [
        ("in order to", "to"),
        ("due to the fact that", "because"),
        ("for the purpose of", "for"),
        ("in the event that", "if"),
        ("prior to", "before"),
        ("subsequent to", "after"),
        ("at this point in time", "now"),
        ("with regard to", "about"),
        ("in the absence of", "without"),
        ("in the vicinity of", "near"),
    ];
    
    let mut result = text.to_string();
    for (complex, simple) in replacements.iter() {
        let pattern = regex::Regex::new(&regex::escape(complex)).unwrap();
        result = pattern.replace_all(&result, *simple).to_string();
    }
    
    result
}

/// Helper function to reduce nested clauses
fn reduce_nested_clauses(text: &str) -> String {
    // This is a simplified approach - a full implementation would use a proper parser
    let mut result = text.to_string();
    
    // Remove text in parentheses if it's not essential
    let re = regex::Regex::new(r"\([^)]*\)").unwrap();
    result = re.replace_all(&result, "").to_string();
    
    // Replace commas with periods to split nested clauses
    // This is very simplistic and would need refinement
    let re = regex::Regex::new(r", which | that | who ").unwrap();
    result = re.replace_all(&result, ". ").to_string();
    
    result
}

/// Helper function to split long sentences
fn split_long_sentences(text: &str) -> String {
    // If the sentence is already short, return it as is
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 15 {
        return text.to_string();
    }
    
    // Look for conjunction points to split the sentence
    let conjunctions = [", and ", ", but ", ", or ", "; ", ": "];
    
    for conjunction in conjunctions.iter() {
        if text.contains(conjunction) {
            let parts: Vec<&str> = text.split(conjunction).collect();
            if parts.len() >= 2 {
                let mut result = String::new();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 {
                        // Capitalize the first letter of the new sentence
                        let mut chars = part.chars();
                        if let Some(first_char) = chars.next() {
                            result.push_str(". ");
                            result.push(first_char.to_uppercase().next().unwrap_or(first_char));
                            result.push_str(chars.as_str());
                        }
                    } else {
                        result.push_str(part);
                    }
                }
                return result;
            }
        }
    }
    
    // If no good split point was found, return the original
    text.to_string()
}

/// Helper function to favor active voice
fn favor_active_voice(text: &str) -> String {
    // This is a simplified approach - a full implementation would use a proper parser
    let passive_patterns = [
        (r"is being ([a-z]+ed)", "someone is $1ing"),
        (r"are being ([a-z]+ed)", "someone is $1ing"),
        (r"was being ([a-z]+ed)", "someone was $1ing"),
        (r"were being ([a-z]+ed)", "someone was $1ing"),
        (r"has been ([a-z]+ed)", "someone has $1ed"),
        (r"have been ([a-z]+ed)", "someone has $1ed"),
        (r"had been ([a-z]+ed)", "someone had $1ed"),
        (r"will be ([a-z]+ed)", "someone will $1"),
        (r"is ([a-z]+ed)", "someone $1s"),
        (r"are ([a-z]+ed)", "someone $1s"),
        (r"was ([a-z]+ed)", "someone $1ed"),
        (r"were ([a-z]+ed)", "someone $1ed"),
    ];
    
    let mut result = text.to_string();
    for (pattern, replacement) in passive_patterns.iter() {
        let re = regex::Regex::new(pattern).unwrap();
        result = re.replace_all(&result, *replacement).to_string();
    }
    
    result
}

/// Replace jargon with simpler terms
fn replace_jargon(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    // Check that we have at least one argument
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "replace_jargon requires a text argument".to_string(),
        });
    }
    
    // Extract the text argument
    let text = match &args[0] {
        Value::String(s) => s.clone(),
        Value::TextUnit(tu) => tu.content.clone(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "replace_jargon expects a string or text unit as first argument".to_string(),
            });
        }
    };
    
    // Optional domain parameter
    let domain = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => Some(s.clone()),
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "replace_jargon expects a string domain as second argument".to_string(),
                });
            }
        }
    } else {
        None
    };
    
    // Optional custom jargon map
    let custom_map = if args.len() > 2 {
        match &args[2] {
            Value::Map(m) => {
                let mut result = HashMap::new();
                for (k, v) in m {
                    match v {
                        Value::String(s) => {
                            result.insert(k.clone(), s.clone());
                        }
                        _ => {
                            return Err(TurbulanceError::RuntimeError {
                                message: "replace_jargon custom map values must be strings".to_string(),
                            });
                        }
                    }
                }
                Some(result)
            }
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "replace_jargon expects a map for custom replacements as third argument".to_string(),
                });
            }
        }
    } else {
        None
    };
    
    // Create jargon replacement dictionary
    let mut jargon_map = HashMap::new();
    
    // Add domain-specific jargon if a domain was specified
    if let Some(domain_str) = domain {
        match domain_str.to_lowercase().as_str() {
            "tech" | "technology" | "computer_science" => {
                jargon_map.extend([
                    ("utilize".to_string(), "use".to_string()),
                    ("implementation".to_string(), "setup".to_string()),
                    ("functionality".to_string(), "feature".to_string()),
                    ("leverage".to_string(), "use".to_string()),
                    ("optimize".to_string(), "improve".to_string()),
                    ("interface".to_string(), "connect".to_string()),
                    ("bandwidth".to_string(), "capacity".to_string()),
                    ("robust".to_string(), "strong".to_string()),
                    ("paradigm".to_string(), "model".to_string()),
                    ("scalable".to_string(), "adaptable".to_string()),
                    ("cloud-based".to_string(), "online".to_string()),
                    ("mission-critical".to_string(), "essential".to_string()),
                    ("bleeding-edge".to_string(), "newest".to_string()),
                    ("synergize".to_string(), "combine".to_string()),
                ]);
            },
            "business" | "corporate" => {
                jargon_map.extend([
                    ("synergy".to_string(), "cooperation".to_string()),
                    ("actionable".to_string(), "practical".to_string()),
                    ("deliverable".to_string(), "result".to_string()),
                    ("incentivize".to_string(), "motivate".to_string()),
                    ("leverage".to_string(), "use".to_string()),
                    ("core competency".to_string(), "strength".to_string()),
                    ("streamline".to_string(), "simplify".to_string()),
                    ("paradigm shift".to_string(), "major change".to_string()),
                    ("low-hanging fruit".to_string(), "easy wins".to_string()),
                    ("value-added".to_string(), "useful".to_string()),
                    ("best practices".to_string(), "proven methods".to_string()),
                    ("touch base".to_string(), "talk".to_string()),
                    ("game-changer".to_string(), "important development".to_string()),
                    ("deep dive".to_string(), "detailed look".to_string()),
                ]);
            },
            "medicine" | "healthcare" => {
                jargon_map.extend([
                    ("acute".to_string(), "sudden".to_string()),
                    ("chronic".to_string(), "long-lasting".to_string()),
                    ("etiology".to_string(), "cause".to_string()),
                    ("idiopathic".to_string(), "unknown cause".to_string()),
                    ("prognosis".to_string(), "outlook".to_string()),
                    ("contraindication".to_string(), "reason not to use".to_string()),
                    ("comorbidity".to_string(), "related condition".to_string()),
                    ("sequela".to_string(), "aftereffect".to_string()),
                    ("hypotensive".to_string(), "low blood pressure".to_string()),
                    ("hypertensive".to_string(), "high blood pressure".to_string()),
                    ("pathology".to_string(), "disease".to_string()),
                    ("prodromal".to_string(), "early symptom".to_string()),
                    ("asymptomatic".to_string(), "without symptoms".to_string()),
                    ("nosocomial".to_string(), "hospital-acquired".to_string()),
                ]);
            },
            _ => {
                // Generic jargon for unrecognized domains
                jargon_map.extend([
                    ("utilize".to_string(), "use".to_string()),
                    ("commence".to_string(), "start".to_string()),
                    ("terminate".to_string(), "end".to_string()),
                    ("facilitate".to_string(), "help".to_string()),
                    ("implement".to_string(), "do".to_string()),
                    ("regarding".to_string(), "about".to_string()),
                    ("endeavor".to_string(), "try".to_string()),
                    ("subsequently".to_string(), "later".to_string()),
                    ("additional".to_string(), "more".to_string()),
                    ("approximately".to_string(), "about".to_string()),
                    ("sufficient".to_string(), "enough".to_string()),
                    ("initiate".to_string(), "start".to_string()),
                    ("prior to".to_string(), "before".to_string()),
                    ("ascertain".to_string(), "find out".to_string()),
                ]);
            }
        }
    } else {
        // General jargon replacements
        jargon_map.extend([
            ("utilize".to_string(), "use".to_string()),
            ("facilitate".to_string(), "help".to_string()),
            ("leverage".to_string(), "use".to_string()),
            ("synergy".to_string(), "cooperation".to_string()),
            ("paradigm".to_string(), "model".to_string()),
            ("robust".to_string(), "strong".to_string()),
            ("optimize".to_string(), "improve".to_string()),
            ("implementation".to_string(), "setup".to_string()),
            ("functionality".to_string(), "feature".to_string()),
            ("streamline".to_string(), "simplify".to_string()),
        ]);
    }
    
    // Add custom jargon if provided
    if let Some(custom) = custom_map {
        jargon_map.extend(custom);
    }
    
    // Replace jargon in the text
    let mut result = text.clone();
    
    // Sort keys by length (descending) to replace longer phrases first
    let mut keys: Vec<&String> = jargon_map.keys().collect();
    keys.sort_by(|a, b| b.len().cmp(&a.len()));
    
    for key in keys {
        let replacement = &jargon_map[key];
        
        // Create regex patterns for variations of the term
        let word_boundary_pattern = format!(r"\b{}\b", regex::escape(key));
        let capitalized_key = key.chars().next().map_or(String::new(), |c| {
            let mut s = String::new();
            s.push(c.to_uppercase().next().unwrap_or(c));
            s.push_str(&key[c.len_utf8()..]);
            s
        });
        let capitalized_pattern = format!(r"\b{}\b", regex::escape(&capitalized_key));
        
        // Replace with word boundaries to avoid partial word matches
        let word_re = regex::Regex::new(&word_boundary_pattern).unwrap();
        result = word_re.replace_all(&result, replacement).to_string();
        
        // Replace capitalized versions
        let cap_re = regex::Regex::new(&capitalized_pattern).unwrap();
        let capitalized_replacement = replacement.chars().next().map_or(String::new(), |c| {
            let mut s = String::new();
            s.push(c.to_uppercase().next().unwrap_or(c));
            s.push_str(&replacement[c.len_utf8()..]);
            s
        });
        result = cap_re.replace_all(&result, capitalized_replacement).to_string();
    }
    
    Ok(Value::String(result))
}

// ========== Utility Functions ==========

/// Print a value to the console
fn print(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    for arg in args {
        println!("{}", arg);
    }
    
    Ok(Value::None)
}

/// Get the length of a value
fn len(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "len requires an argument".to_string(),
        });
    }
    
    let length = match &args[0] {
        Value::String(s) => s.len(),
        Value::TextUnit(tu) => tu.content.len(),
        Value::List(l) => l.len(),
        Value::Map(m) => m.len(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "len can only be called on strings, text units, lists, or maps".to_string(),
            });
        }
    };
    
    Ok(Value::Number(length as f64))
}

/// Get the type of a value
fn typeof(args: Vec<Value>) -> Result<Value, TurbulanceError> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "typeof requires an argument".to_string(),
        });
    }
    
    let type_name = match &args[0] {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Bool(_) => "boolean",
        Value::List(_) => "list",
        Value::Map(_) => "map",
        Value::Function(_) => "function",
        Value::TextUnit(_) => "text_unit",
        Value::None => "none",
    };
    
    Ok(Value::String(type_name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdlib_functions_exist() {
        let stdlib = StdLib::new();
        
        assert!(stdlib.has_function("readability_score"));
        assert!(stdlib.has_function("contains"));
        assert!(stdlib.has_function("print"));
        assert!(stdlib.has_function("len"));
        assert!(stdlib.has_function("typeof"));
        
        assert!(!stdlib.has_function("nonexistent_function"));
    }
    
    #[test]
    fn test_contains_function() {
        let stdlib = StdLib::new();
        
        let text = Value::String("Hello, world!".to_string());
        let pattern = Value::String("world".to_string());
        
        let result = stdlib.call("contains", vec![text.clone(), pattern]).unwrap();
        assert_eq!(result, Value::Bool(true));
        
        let wrong_pattern = Value::String("universe".to_string());
        let result = stdlib.call("contains", vec![text, wrong_pattern]).unwrap();
        assert_eq!(result, Value::Bool(false));
    }
    
    #[test]
    fn test_typeof_function() {
        let stdlib = StdLib::new();
        
        let string_val = Value::String("test".to_string());
        let result = stdlib.call("typeof", vec![string_val]).unwrap();
        assert_eq!(result, Value::String("string".to_string()));
        
        let number_val = Value::Number(42.0);
        let result = stdlib.call("typeof", vec![number_val]).unwrap();
        assert_eq!(result, Value::String("number".to_string()));
        
        let bool_val = Value::Bool(true);
        let result = stdlib.call("typeof", vec![bool_val]).unwrap();
        assert_eq!(result, Value::String("boolean".to_string()));
    }
}
