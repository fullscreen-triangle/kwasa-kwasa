use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};

/// Simplifies complex sentences in text
pub fn simplify_sentences(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "simplify_sentences() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "simplify_sentences() first argument must be a string".to_string(),
            });
        }
    };
    
    // Get level parameter if provided (default to "moderate")
    let level = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "moderate",
        }
    } else {
        "moderate"
    };
    
    // Simple sentence simplification implementation
    // In a real implementation, this would use more sophisticated NLP
    
    // Split into sentences
    let sentences: Vec<&str> = text.split(['.', '!', '?']).filter(|s| !s.trim().is_empty()).collect();
    
    // Apply simplification to each sentence based on level
    let simplified_sentences: Vec<String> = sentences.iter().map(|sentence| {
        let trimmed = sentence.trim();
        
        // Skip sentences that are already simple
        if trimmed.split_whitespace().count() < 8 {
            return trimmed.to_string() + ".";
        }
        
        match level {
            "light" => {
                // Light simplification: Just remove some filler words
                let filler_words = ["actually", "basically", "definitely", "literally", "very", "really"];
                let mut simplified = trimmed.to_string();
                for word in filler_words.iter() {
                    simplified = simplified.replace(&format!(" {} ", word), " ");
                }
                simplified + "."
            },
            "moderate" => {
                // Moderate simplification: Break into shorter sentences
                if trimmed.contains(", ") {
                    // Break at commas for moderate simplification
                    let parts: Vec<&str> = trimmed.split(", ").collect();
                    if parts.len() > 1 {
                        let first_part = capitalize_first_letter(parts[0]);
                        let second_part = capitalize_first_letter(&parts[1..].join(", "));
                        format!("{}. {}.", first_part, second_part)
                    } else {
                        trimmed.to_string() + "."
                    }
                } else {
                    // If no commas, try to break at conjunctions
                    for conjunction in &[" and ", " but ", " or ", " so ", " for ", " yet ", " nor "] {
                        if trimmed.contains(conjunction) {
                            let parts: Vec<&str> = trimmed.split(conjunction).collect();
                            if parts.len() > 1 {
                                let first_part = capitalize_first_letter(parts[0]);
                                let second_part = capitalize_first_letter(parts[1]);
                                return format!("{}. {}.", first_part, second_part);
                            }
                        }
                    }
                    trimmed.to_string() + "."
                }
            },
            "strong" => {
                // Strong simplification: Break into very short sentences, replace complex words
                let complex_words = [
                    ("utilize", "use"),
                    ("implement", "use"),
                    ("sufficient", "enough"),
                    ("requirements", "needs"),
                    ("approximately", "about"),
                    ("demonstrate", "show"),
                    ("modification", "change"),
                    ("endeavor", "try"),
                    ("assistance", "help"),
                    ("numerous", "many"),
                ];
                
                // Replace complex words
                let mut simplified = trimmed.to_string();
                for (complex, simple) in complex_words.iter() {
                    simplified = simplified.replace(complex, simple);
                }
                
                // Break at any punctuation or conjunction
                let break_points = [",", ";", ":", " and ", " but ", " or ", " so ", " because "];
                let mut parts = Vec::new();
                let mut current = simplified.clone();
                
                for point in break_points.iter() {
                    if current.contains(point) {
                        let split_parts: Vec<&str> = current.split(point).collect();
                        parts.push(split_parts[0].trim().to_string());
                        current = split_parts[1..].join(point);
                    }
                }
                
                if !current.is_empty() {
                    parts.push(current.trim().to_string());
                }
                
                // Format into short sentences
                parts.iter()
                    .filter(|p| !p.is_empty())
                    .map(|p| capitalize_first_letter(p) + ".")
                    .collect::<Vec<String>>()
                    .join(" ")
            },
            _ => trimmed.to_string() + ".",
        }
    }).collect();
    
    // Join sentences into paragraph(s)
    let simplified_text = simplified_sentences.join(" ");
    
    Ok(Value::String(simplified_text))
}

/// Replaces jargon in text with plain language alternatives
pub fn replace_jargon(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "replace_jargon() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "replace_jargon() first argument must be a string".to_string(),
            });
        }
    };
    
    // Get domain parameter if provided (default to "general")
    let domain = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "general",
        }
    } else {
        "general"
    };
    
    // Define jargon replacements for different domains
    let replacements = match domain {
        "tech" => [
            ("utilize", "use"),
            ("functionality", "feature"),
            ("implementation", "code"),
            ("leverage", "use"),
            ("paradigm", "approach"),
            ("architecture", "design"),
            ("interface", "connect"),
            ("infrastructure", "system"),
            ("mission-critical", "important"),
            ("robust", "strong"),
        ],
        "academic" => [
            ("paradigm", "model"),
            ("methodology", "method"),
            ("conceptualization", "idea"),
            ("subsequently", "later"),
            ("implementation", "use"),
            ("endeavor", "try"),
            ("elucidate", "explain"),
            ("ameliorate", "improve"),
            ("utilize", "use"),
            ("cognizant", "aware"),
        ],
        "business" => [
            ("synergy", "cooperation"),
            ("leverage", "use"),
            ("incentivize", "motivate"),
            ("optimization", "improvement"),
            ("paradigm shift", "change"),
            ("actionable", "usable"),
            ("deliverable", "product"),
            ("bandwidth", "capacity"),
            ("core competency", "strength"),
            ("utilize", "use"),
        ],
        "legal" => [
            ("aforementioned", "previously mentioned"),
            ("pursuant to", "under"),
            ("heretofore", "until now"),
            ("hereinafter", "later"),
            ("in accordance with", "by"),
            ("notwithstanding", "despite"),
            ("shall", "will"),
            ("whereas", "because"),
            ("herein", "here"),
            ("said", "the"),
        ],
        // Default to general replacements
        _ => [
            ("utilize", "use"),
            ("implement", "use"),
            ("endeavor", "try"),
            ("leverage", "use"),
            ("optimize", "improve"),
            ("facilitate", "help"),
            ("subsequently", "later"),
            ("commence", "start"),
            ("terminate", "end"),
            ("prior to", "before"),
        ],
    };
    
    // Apply replacements
    let mut result = text.to_string();
    for (jargon, plain) in replacements.iter() {
        // Case-insensitive replacement
        let capitalized_jargon = capitalize_first_letter(jargon);
        let capitalized_plain = capitalize_first_letter(plain);
        
        // Replace capitalized version
        result = result.replace(&capitalized_jargon, &capitalized_plain);
        
        // Replace lowercase version
        result = result.replace(jargon, plain);
    }
    
    Ok(Value::String(result))
}

/// Makes text more formal
pub fn formalize(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "formalize() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "formalize() first argument must be a string".to_string(),
            });
        }
    };
    
    // Define informal -> formal replacements
    let replacements = [
        ("I'm", "I am"),
        ("you're", "you are"),
        ("we're", "we are"),
        ("they're", "they are"),
        ("it's", "it is"),
        ("that's", "that is"),
        ("there's", "there is"),
        ("here's", "here is"),
        ("who's", "who is"),
        ("what's", "what is"),
        ("where's", "where is"),
        ("when's", "when is"),
        ("why's", "why is"),
        ("how's", "how is"),
        ("I've", "I have"),
        ("you've", "you have"),
        ("we've", "we have"),
        ("they've", "they have"),
        ("I'll", "I will"),
        ("you'll", "you will"),
        ("we'll", "we will"),
        ("they'll", "they will"),
        ("I'd", "I would"),
        ("you'd", "you would"),
        ("we'd", "we would"),
        ("they'd", "they would"),
        ("can't", "cannot"),
        ("won't", "will not"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("didn't", "did not"),
        ("shouldn't", "should not"),
        ("wouldn't", "would not"),
        ("couldn't", "could not"),
        ("might've", "might have"),
        ("must've", "must have"),
        ("would've", "would have"),
        ("could've", "could have"),
        ("should've", "should have"),
        ("wanna", "want to"),
        ("gonna", "going to"),
        ("gotta", "got to"),
        ("sorta", "sort of"),
        ("kinda", "kind of"),
        ("yeah", "yes"),
        ("yep", "yes"),
        ("nope", "no"),
        ("ok", "okay"),
        ("cause", "because"),
        ("cuz", "because"),
        ("bout", "about"),
        ("u", "you"),
        ("ur", "your"),
        ("r", "are"),
        ("c", "see"),
        ("b", "be"),
        ("2", "to"),
        ("4", "for"),
        ("8", "ate"),
        ("btw", "by the way"),
        ("lol", "[laughs]"),
        ("asap", "as soon as possible"),
        ("fyi", "for your information"),
        ("thru", "through"),
        ("thx", "thanks"),
        ("imo", "in my opinion"),
        ("idk", "I do not know"),
        ("omg", "oh my goodness"),
        ("bff", "best friend"),
        ("lmk", "let me know"),
    ];
    
    // Apply replacements
    let mut result = text.to_string();
    for (informal, formal) in replacements.iter() {
        // Convert all forms
        let capitalized_informal = capitalize_first_letter(informal);
        let capitalized_formal = capitalize_first_letter(formal);
        let uppercase_informal = informal.to_uppercase();
        let uppercase_formal = formal.to_uppercase();
        
        // Replace all variants
        result = result.replace(&capitalized_informal, &capitalized_formal);
        result = result.replace(informal, formal);
        result = result.replace(&uppercase_informal, &uppercase_formal);
    }
    
    // Handle first-person pronouns
    result = result.replace(" I ", " one ");
    result = result.replace("I ", "One ");
    result = result.replace(" my ", " one's ");
    result = result.replace("My ", "One's ");
    result = result.replace(" me ", " oneself ");
    result = result.replace("Me ", "Oneself ");
    
    // Handle second-person pronouns (convert to third-person or passive voice)
    result = result.replace(" you ", " one ");
    result = result.replace("You ", "One ");
    result = result.replace(" your ", " one's ");
    result = result.replace("Your ", "One's ");
    
    // Remove multiple exclamation marks and replace with period
    result = replace_regex(&result, r"!+", ".");
    
    // Replace multiple question marks with a single one
    result = replace_regex(&result, r"\?+", "?");
    
    // Convert incomplete sentences to complete ones (very simple heuristic)
    result = result.replace(". ", ". ");
    
    Ok(Value::String(result))
}

// Helper functions

/// Capitalizes the first letter of a string
fn capitalize_first_letter(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let capitalized = first.to_uppercase().collect::<String>();
            capitalized + chars.as_str()
        }
    }
}

/// Simple regex-like replacement (placeholder for actual regex)
fn replace_regex(text: &str, pattern: &str, replacement: &str) -> String {
    // In a real implementation, this would use the regex crate
    // For simplicity, we'll just handle a few common patterns
    match pattern {
        r"!+" => text.replace("!!", ".").replace("!", "."),
        r"\?+" => text.replace("??", "?").replace("???", "?"),
        _ => text.to_string(),
    }
} 