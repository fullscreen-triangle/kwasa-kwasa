use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;

/// Simplify complex sentences into more readable forms
pub fn simplify_sentences(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "simplify_sentences requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "simplify_sentences requires a string or TextUnit".to_string(),
        }),
    };

    let mut simplified = text.clone();

    // Replace complex constructions with simpler ones
    let complex_replacements = [
        ("in order to", "to"),
        ("due to the fact that", "because"),
        ("it is important to note that", "note that"),
        ("it should be mentioned that", ""),
        ("with regard to", "about"),
        ("in relation to", "about"),
        ("for the purpose of", "to"),
        ("in the event that", "if"),
        ("prior to", "before"),
        ("subsequent to", "after"),
        ("in addition to", "and"),
        ("nevertheless", "however"),
        ("consequently", "so"),
        ("furthermore", "also"),
        ("moreover", "also"),
        ("notwithstanding", "despite"),
        ("accordingly", "so"),
        ("therefore", "so"),
        ("thus", "so"),
        ("hence", "so"),
        ("as a result", "so"),
        ("in conclusion", "finally"),
        ("to summarize", "in summary"),
    ];

    for (complex, simple) in &complex_replacements {
        simplified = simplified.replace(complex, simple);
    }

    // Remove redundant intensifiers
    let redundant_words = [
        " very ", " really ", " quite ", " rather ", " somewhat ", " fairly ",
        " pretty ", " extremely ", " incredibly ", " absolutely ", " totally ",
        " completely ", " entirely ", " perfectly ", " exactly ", " precisely ",
    ];

    for word in &redundant_words {
        simplified = simplified.replace(word, " ");
    }

    // Clean up multiple spaces
    while simplified.contains("  ") {
        simplified = simplified.replace("  ", " ");
    }

    simplified = simplified.trim().to_string();

    Ok(Value::String(simplified))
}

/// Replace technical jargon with simpler terms
pub fn replace_jargon(args: Vec<Value>) -> Result<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "replace_jargon requires 1-2 arguments (text, optional domain)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "replace_jargon first argument must be a string or TextUnit".to_string(),
        }),
    };

    let domain = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "general",
        }
    } else {
        "general"
    };

    let mut simplified = text.clone();

    // General jargon replacements
    let general_jargon = [
        ("utilize", "use"),
        ("functionality", "function"),
        ("implementation", "use"),
        ("optimization", "improvement"),
        ("methodology", "method"),
        ("specification", "details"),
        ("configuration", "setup"),
        ("infrastructure", "system"),
        ("architecture", "design"),
        ("paradigm", "model"),
        ("algorithm", "method"),
        ("protocol", "rules"),
        ("interface", "connection"),
        ("validate", "check"),
        ("facilitate", "help"),
        ("demonstrate", "show"),
        ("indicate", "show"),
        ("establish", "set up"),
        ("terminate", "end"),
        ("commence", "start"),
        ("initiate", "start"),
        ("construct", "build"),
        ("acquire", "get"),
        ("obtain", "get"),
        ("eliminate", "remove"),
        ("enhance", "improve"),
        ("modify", "change"),
        ("transform", "change"),
        ("examine", "look at"),
        ("analyze", "study"),
        ("evaluate", "judge"),
        ("determine", "find"),
        ("substantial", "large"),
        ("significant", "important"),
        ("numerous", "many"),
        ("adequate", "enough"),
        ("appropriate", "right"),
        ("beneficial", "helpful"),
        ("detrimental", "harmful"),
        ("essential", "needed"),
        ("necessary", "needed"),
    ];

    // Apply general replacements
    for (jargon, simple) in &general_jargon {
        simplified = apply_case_aware_replacement(&simplified, jargon, simple);
    }

    // Domain-specific replacements
    let domain_specific = get_domain_specific_replacements(domain);
    for (jargon, simple) in &domain_specific {
        simplified = apply_case_aware_replacement(&simplified, jargon, simple);
    }

    Ok(Value::String(simplified))
}

/// Transform text to be more formal
pub fn formalize(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "formalize requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "formalize requires a string or TextUnit".to_string(),
        }),
    };

    let mut formalized = text.clone();

    // Expand contractions
    let contractions = [
        ("can't", "cannot"),
        ("won't", "will not"),
        ("don't", "do not"),
        ("doesn't", "does not"),
        ("didn't", "did not"),
        ("isn't", "is not"),
        ("aren't", "are not"),
        ("wasn't", "was not"),
        ("weren't", "were not"),
        ("haven't", "have not"),
        ("hasn't", "has not"),
        ("hadn't", "had not"),
        ("wouldn't", "would not"),
        ("shouldn't", "should not"),
        ("couldn't", "could not"),
        ("mustn't", "must not"),
        ("I'm", "I am"),
        ("you're", "you are"),
        ("he's", "he is"),
        ("she's", "she is"),
        ("it's", "it is"),
        ("we're", "we are"),
        ("they're", "they are"),
        ("I've", "I have"),
        ("you've", "you have"),
        ("we've", "we have"),
        ("they've", "they have"),
        ("I'll", "I will"),
        ("you'll", "you will"),
        ("we'll", "we will"),
        ("they'll", "they will"),
    ];

    for (contraction, expansion) in &contractions {
        formalized = apply_case_aware_replacement(&formalized, contraction, expansion);
    }

    // Replace informal expressions with formal ones
    let informal_to_formal = [
        ("I think", "It is proposed that"),
        ("I believe", "It is suggested that"),
        ("I feel", "It appears that"),
        ("you know", "as is understood"),
        ("kind of", "somewhat"),
        ("sort of", "somewhat"),
        ("a lot of", "numerous"),
        ("lots of", "numerous"),
        ("pretty good", "satisfactory"),
        ("really good", "excellent"),
        ("get", "obtain"),
        ("give", "provide"),
        ("take", "accept"),
        ("make", "create"),
        ("show", "demonstrate"),
        ("tell", "inform"),
        ("ask", "inquire"),
        ("help", "assist"),
        ("try", "attempt"),
        ("use", "utilize"),
        ("need", "require"),
        ("want", "desire"),
        ("big", "substantial"),
        ("small", "minimal"),
        ("fast", "rapid"),
        ("slow", "gradual"),
        ("easy", "straightforward"),
        ("hard", "challenging"),
        ("start", "commence"),
        ("stop", "cease"),
        ("end", "conclude"),
        ("buy", "purchase"),
        ("sell", "market"),
        ("build", "construct"),
        ("fix", "repair"),
        ("change", "modify"),
    ];

    for (informal, formal) in &informal_to_formal {
        formalized = apply_case_aware_replacement(&formalized, informal, formal);
    }

    Ok(Value::String(formalized))
}

/// Transform text to be more informal/casual
pub fn informalize(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "informalize requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "informalize requires a string or TextUnit".to_string(),
        }),
    };

    let mut informal = text.clone();

    // Create contractions
    let formal_to_contractions = [
        ("cannot", "can't"),
        ("will not", "won't"),
        ("do not", "don't"),
        ("does not", "doesn't"),
        ("did not", "didn't"),
        ("is not", "isn't"),
        ("are not", "aren't"),
        ("was not", "wasn't"),
        ("were not", "weren't"),
        ("have not", "haven't"),
        ("has not", "hasn't"),
        ("had not", "hadn't"),
        ("would not", "wouldn't"),
        ("should not", "shouldn't"),
        ("could not", "couldn't"),
        ("I am", "I'm"),
        ("you are", "you're"),
        ("he is", "he's"),
        ("she is", "she's"),
        ("it is", "it's"),
        ("we are", "we're"),
        ("they are", "they're"),
        ("I have", "I've"),
        ("you have", "you've"),
        ("we have", "we've"),
        ("they have", "they've"),
        ("I will", "I'll"),
        ("you will", "you'll"),
        ("we will", "we'll"),
        ("they will", "they'll"),
    ];

    for (formal, contraction) in &formal_to_contractions {
        informal = apply_case_aware_replacement(&informal, formal, contraction);
    }

    // Replace formal expressions with informal ones
    let formal_to_informal = [
        ("It is proposed that", "I think"),
        ("It is suggested that", "I believe"),
        ("It appears that", "I feel"),
        ("as is understood", "you know"),
        ("somewhat", "kind of"),
        ("numerous", "lots of"),
        ("satisfactory", "pretty good"),
        ("excellent", "really good"),
        ("obtain", "get"),
        ("provide", "give"),
        ("accept", "take"),
        ("create", "make"),
        ("demonstrate", "show"),
        ("inform", "tell"),
        ("inquire", "ask"),
        ("assist", "help"),
        ("attempt", "try"),
        ("utilize", "use"),
        ("require", "need"),
        ("desire", "want"),
        ("substantial", "big"),
        ("minimal", "small"),
        ("rapid", "fast"),
        ("gradual", "slow"),
        ("straightforward", "easy"),
        ("challenging", "hard"),
        ("commence", "start"),
        ("cease", "stop"),
        ("conclude", "end"),
        ("purchase", "buy"),
        ("construct", "build"),
        ("repair", "fix"),
        ("modify", "change"),
    ];

    for (formal, informal_word) in &formal_to_informal {
        informal = apply_case_aware_replacement(&informal, formal, informal_word);
    }

    Ok(Value::String(informal))
}

/// Transform text for a specific audience (academic, business, casual, etc.)
pub fn transform_for_audience(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "transform_for_audience requires exactly 2 arguments (text, audience)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "transform_for_audience first argument must be a string or TextUnit".to_string(),
        }),
    };

    let audience = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "transform_for_audience second argument must be a string".to_string(),
        }),
    };

    let transformed = match audience {
        "academic" => transform_to_academic_style(text),
        "business" => transform_to_business_style(text),
        "casual" => transform_to_casual_style(text),
        "technical" => transform_to_technical_style(text),
        "legal" => transform_to_legal_style(text),
        _ => return Err(TurbulanceError::RuntimeError {
            message: format!("Unknown audience type: {}", audience),
        }),
    };

    Ok(Value::String(transformed))
}

/// Remove or replace profanity and inappropriate language
pub fn clean_language(args: Vec<Value>) -> Result<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "clean_language requires 1-2 arguments (text, optional replacement)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "clean_language first argument must be a string or TextUnit".to_string(),
        }),
    };

    let replacement = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => "[removed]".to_string(),
        }
    } else {
        "[removed]".to_string()
    };

    // Basic profanity filter (in a real implementation, this would be more comprehensive)
    let profanity_list = [
        "damn", "hell", "crap", "stupid", "idiot", "moron", "dumb"
    ];

    let mut cleaned = text.clone();
    for profanity in &profanity_list {
        cleaned = apply_case_aware_replacement(&cleaned, profanity, &replacement);
    }

    Ok(Value::String(cleaned))
}

// Helper functions

fn apply_case_aware_replacement(text: &str, from: &str, to: &str) -> String {
    let mut result = text.to_string();
    
    // Replace exact match
    result = result.replace(from, to);
    
    // Replace capitalized version
    if let Some(first_char) = from.chars().next() {
        if first_char.is_lowercase() {
            let capitalized_from = first_char.to_uppercase().collect::<String>() + &from[1..];
            let capitalized_to = if let Some(first_to) = to.chars().next() {
                first_to.to_uppercase().collect::<String>() + &to[1..]
            } else {
                to.to_string()
            };
            result = result.replace(&capitalized_from, &capitalized_to);
        }
    }
    
    // Replace uppercase version
    let uppercase_from = from.to_uppercase();
    let uppercase_to = to.to_uppercase();
    result = result.replace(&uppercase_from, &uppercase_to);
    
    result
}

fn get_domain_specific_replacements(domain: &str) -> Vec<(&'static str, &'static str)> {
    match domain {
        "medical" => vec![
            ("myocardial infarction", "heart attack"),
            ("cerebrovascular accident", "stroke"),
            ("hypertension", "high blood pressure"),
            ("hypotension", "low blood pressure"),
            ("tachycardia", "fast heart rate"),
            ("bradycardia", "slow heart rate"),
            ("pneumonia", "lung infection"),
            ("gastroenteritis", "stomach bug"),
            ("dermatitis", "skin inflammation"),
            ("therapeutic", "healing"),
            ("prophylactic", "preventive"),
            ("diagnosis", "finding"),
            ("prognosis", "outlook"),
            ("symptom", "sign"),
        ],
        "legal" => vec![
            ("plaintiff", "person suing"),
            ("defendant", "person being sued"),
            ("litigation", "lawsuit"),
            ("jurisdiction", "court authority"),
            ("statute", "law"),
            ("precedent", "past case rule"),
            ("liability", "responsibility"),
            ("damages", "payment for harm"),
            ("testimony", "statement under oath"),
            ("verdict", "decision"),
            ("settlement", "agreement to end case"),
        ],
        "financial" => vec![
            ("portfolio", "investments"),
            ("diversification", "spreading risk"),
            ("liquidity", "easy to sell"),
            ("volatility", "price swings"),
            ("equity", "ownership"),
            ("liability", "debt"),
            ("asset", "valuable item"),
            ("leverage", "borrowed money"),
            ("dividend", "profit share"),
            ("yield", "return rate"),
        ],
        "technical" => vec![
            ("algorithm", "step-by-step process"),
            ("bandwidth", "data capacity"),
            ("cache", "temporary storage"),
            ("debugging", "finding errors"),
            ("encryption", "data protection"),
            ("firewall", "security barrier"),
            ("malware", "harmful software"),
            ("protocol", "communication rules"),
            ("bandwidth", "connection speed"),
            ("latency", "delay time"),
        ],
        _ => vec![],
    }
}

fn transform_to_academic_style(text: &str) -> String {
    let mut academic = text.to_string();
    
    // Add academic hedging
    academic = academic.replace("This shows", "This evidence suggests");
    academic = academic.replace("This proves", "This research demonstrates");
    academic = academic.replace("Obviously", "It is evident that");
    academic = academic.replace("Clearly", "The data indicates that");
    
    // Add scholarly language
    academic = academic.replace("because", "due to the fact that");
    academic = academic.replace("but", "however");
    academic = academic.replace("also", "furthermore");
    academic = academic.replace("so", "consequently");
    
    academic
}

fn transform_to_business_style(text: &str) -> String {
    let mut business = text.to_string();
    
    // Add business terminology
    business = business.replace("use", "leverage");
    business = business.replace("improve", "optimize");
    business = business.replace("work together", "synergize");
    business = business.replace("plan", "strategy");
    business = business.replace("goal", "objective");
    business = business.replace("result", "deliverable");
    
    business
}

fn transform_to_casual_style(text: &str) -> String {
    let mut casual = text.to_string();
    
    // Make more conversational
    casual = casual.replace("It is important to note", "Just so you know");
    casual = casual.replace("Furthermore", "Also");
    casual = casual.replace("However", "But");
    casual = casual.replace("Therefore", "So");
    casual = casual.replace("In conclusion", "Bottom line");
    
    casual
}

fn transform_to_technical_style(text: &str) -> String {
    let mut technical = text.to_string();
    
    // Add technical precision
    technical = technical.replace("use", "implement");
    technical = technical.replace("make", "generate");
    technical = technical.replace("change", "modify");
    technical = technical.replace("connect", "interface");
    technical = technical.replace("setup", "configuration");
    
    technical
}

fn transform_to_legal_style(text: &str) -> String {
    let mut legal = text.to_string();
    
    // Add legal formality
    legal = legal.replace("will", "shall");
    legal = legal.replace("if", "in the event that");
    legal = legal.replace("before", "prior to");
    legal = legal.replace("after", "subsequent to");
    legal = legal.replace("about", "with respect to");
    legal = legal.replace("because", "whereas");
    
    legal
} 