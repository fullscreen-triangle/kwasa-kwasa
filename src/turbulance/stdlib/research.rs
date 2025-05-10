use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};

/// Retrieves contextual information about a topic
pub fn research_context(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "research_context() requires at least one argument (topic)".to_string(),
        });
    }

    // Extract topic from argument
    let topic = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "research_context() first argument must be a string".to_string(),
            });
        }
    };
    
    // Get depth parameter if provided (default to "medium")
    let depth = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "medium",
        }
    } else {
        "medium"
    };
    
    // Simple research context implementation
    // In a real implementation, this would connect to a knowledge database or external API
    let context = match topic.to_lowercase().as_str() {
        "turbulance" => {
            "Turbulance is a domain-specific language designed for text manipulation with semantic awareness. \
            It includes specialized syntax for context-aware text operations, knowledge integration, and metacognitive processing."
        },
        "metacognition" => {
            "Metacognition refers to the awareness and understanding of one's own thought processes. \
            In the context of text processing, metacognitive systems can observe, regulate, and optimize how text is transformed and understood."
        },
        "text processing" => {
            "Text processing encompasses various computational methods for manipulating and analyzing text data. \
            This includes tokenization, parsing, transformation, and semantic analysis of textual content."
        },
        _ => {
            // Generic response for unknown topics
            "No specific information found on this topic. Consider researching through other channels or refining your query."
        }
    };
    
    // Adjust depth of context
    let detailed_context = match depth {
        "shallow" => {
            // Return just a brief summary
            let first_sentence = context.split('.').next().unwrap_or(context);
            first_sentence.to_string() + "."
        },
        "medium" => {
            // Return the full context
            context.to_string()
        },
        "deep" => {
            // Return extended context with references
            format!(
                "{}\n\nAdditional resources for {}:\n- Research papers: https://example.com/papers/{}\n- Books: https://example.com/books/{}\n- Communities: https://example.com/communities/{}", 
                context,
                topic,
                topic.to_lowercase().replace(' ', "-"),
                topic.to_lowercase().replace(' ', "-"),
                topic.to_lowercase().replace(' ', "-")
            )
        },
        _ => context.to_string(),
    };
    
    Ok(Value::String(detailed_context))
}

/// Verifies factual claims in text
pub fn fact_check(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "fact_check() requires at least one argument (statement)".to_string(),
        });
    }

    // Extract statement from argument
    let statement = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "fact_check() first argument must be a string".to_string(),
            });
        }
    };
    
    // Simple fact checking implementation
    // In a real implementation, this would connect to a knowledge database or external API
    
    // Example claims that we'll hardcode responses for
    let (is_verified, confidence, explanation) = match statement.to_lowercase().as_str() {
        s if s.contains("earth is flat") => {
            (false, 0.99, "The Earth is an oblate spheroid, as confirmed by countless scientific observations, satellite imagery, and circumnavigation.")
        },
        s if s.contains("water boils at 100 degrees celsius") => {
            (true, 0.9, "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atmosphere). This temperature changes with altitude and pressure.")
        },
        s if s.contains("turbulance is a programming language") => {
            (true, 0.95, "Turbulance is a domain-specific programming language designed for text processing with semantic awareness.")
        },
        _ => {
            // Generic response for unknown statements
            (false, 0.0, "This statement could not be verified with the available knowledge base.")
        }
    };
    
    // Create a map to return verification details
    let mut result_map = std::collections::HashMap::new();
    result_map.insert("verified".to_string(), Value::Boolean(is_verified));
    result_map.insert("confidence".to_string(), Value::Number(confidence));
    result_map.insert("explanation".to_string(), Value::String(explanation.to_string()));
    result_map.insert("statement".to_string(), Value::String(statement.clone()));
    
    Ok(Value::Object(result_map))
}

/// Ensures that technical terms are followed by explanations
pub fn ensure_explanation_follows(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows() requires at least one argument (text or term)".to_string(),
        });
    }

    // First argument can be either text to analyze or a specific term to look for
    let first_arg = &args[0];
    
    // If only one argument, assume it's the term to look for
    // and the text is the current context
    if args.len() == 1 {
        match first_arg {
            Value::String(term) => {
                // This would check the current context in a full implementation
                // For this example, we'll return a constructed explanation
                let explanation = format!(
                    "The term \"{}\" requires an explanation. Consider adding a definition or clarification after this term.", 
                    term
                );
                return Ok(Value::String(explanation));
            },
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "ensure_explanation_follows() argument must be a string (term)".to_string(),
                });
            }
        }
    }
    
    // If two arguments, first is text, second is term
    if args.len() >= 2 {
        let text = match first_arg {
            Value::String(s) => s,
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "ensure_explanation_follows() first argument must be a string (text)".to_string(),
                });
            }
        };
        
        let term = match &args[1] {
            Value::String(s) => s,
            _ => {
                return Err(TurbulanceError::RuntimeError {
                    message: "ensure_explanation_follows() second argument must be a string (term)".to_string(),
                });
            }
        };
        
        // Check if the term is in the text
        if !text.contains(term) {
            return Ok(Value::Boolean(true)); // Term not found, so no explanation needed
        }
        
        // Find the term position in the text
        let term_pos = text.find(term).unwrap();
        let after_term = &text[term_pos + term.len()..];
        
        // Check if an explanation follows the term
        // Simple heuristic: look for phrases that typically introduce explanations
        let explanation_markers = [
            "is", "refers to", "means", "defined as", "which is", "that is",
            ":", ";", "-", "–", "(", "["
        ];
        
        let has_explanation = explanation_markers.iter().any(|marker| {
            // Look for marker within a reasonable distance after the term
            let search_range = std::cmp::min(after_term.len(), 50);
            let search_text = &after_term[..search_range];
            search_text.contains(marker)
        });
        
        if has_explanation {
            // Term is followed by an explanation
            Ok(Value::Boolean(true))
        } else {
            // Term lacks an explanation
            // In interactive mode, we would suggest an explanation
            let suggested_text = suggest_explanation_for_term(term);
            
            // Create a map to return details
            let mut result_map = std::collections::HashMap::new();
            result_map.insert("has_explanation".to_string(), Value::Boolean(false));
            result_map.insert("term".to_string(), Value::String(term.clone()));
            result_map.insert("suggested_explanation".to_string(), Value::String(suggested_text));
            
            Ok(Value::Object(result_map))
        }
    } else {
        // Shouldn't reach here due to earlier checks
        Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows() invalid arguments".to_string(),
        })
    }
}

/// Generates a suggested explanation for a technical term
fn suggest_explanation_for_term(term: &str) -> String {
    // In a real implementation, this would connect to a knowledge database
    // For this example, we'll hardcode some explanations
    match term.to_lowercase().as_str() {
        "turbulance" => {
            "Turbulance is a specialized programming language designed for text operations with semantic awareness."
        },
        "metacognition" => {
            "Metacognition refers to the awareness and understanding of one's own thought processes."
        },
        "proposition" => {
            "In Turbulance, a Proposition is a container for related semantic units called Motions."
        },
        "motion" => {
            "In Turbulance, a Motion represents a piece of an idea with semantic meaning, contained within a Proposition."
        },
        _ => {
            // Generic suggestion for unknown terms
            "Consider adding a brief definition or explanation of this term for readers who may be unfamiliar with it."
        }
    }.to_string()
} 