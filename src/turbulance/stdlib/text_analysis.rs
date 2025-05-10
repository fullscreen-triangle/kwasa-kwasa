use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};

/// Calculate readability score for text
pub fn readability_score(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "readability_score() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "readability_score() first argument must be a string".to_string(),
            });
        }
    };

    // Calculate Flesch-Kincaid readability score
    let score = calculate_flesch_kincaid_score(text);
    
    Ok(Value::Number(score))
}

/// Analyzes sentiment of text
pub fn sentiment_analysis(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "sentiment_analysis() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "sentiment_analysis() first argument must be a string".to_string(),
            });
        }
    };

    // Simple sentiment analysis implementation
    // This is a placeholder - would be replaced with a proper NLP approach
    let positive_words = ["good", "great", "excellent", "happy", "joy", "love", "best"];
    let negative_words = ["bad", "terrible", "awful", "sad", "hate", "worst", "poor"];
    
    let lowercase_text = text.to_lowercase();
    let words: Vec<&str> = lowercase_text.split_whitespace().collect();
    
    let positive_count = words.iter().filter(|w| positive_words.contains(w)).count();
    let negative_count = words.iter().filter(|w| negative_words.contains(w)).count();
    
    let sentiment_score = if words.is_empty() {
        0.0
    } else {
        (positive_count as f64 - negative_count as f64) / words.len() as f64
    };
    
    // Create a map to return sentiment details
    let mut result_map = std::collections::HashMap::new();
    result_map.insert("score".to_string(), Value::Number(sentiment_score));
    result_map.insert("positive_count".to_string(), Value::Number(positive_count as f64));
    result_map.insert("negative_count".to_string(), Value::Number(negative_count as f64));
    result_map.insert("total_words".to_string(), Value::Number(words.len() as f64));
    
    Ok(Value::Object(result_map))
}

/// Extracts keywords from text
pub fn extract_keywords(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "extract_keywords() requires at least one argument (text)".to_string(),
        });
    }

    // Extract text from argument
    let text = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "extract_keywords() first argument must be a string".to_string(),
            });
        }
    };
    
    // Get count parameter if provided
    let count = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => *n as usize,
            _ => 10, // Default to 10
        }
    } else {
        10
    };
    
    // Simple keyword extraction implementation
    // This is a placeholder - would be replaced with a proper NLP approach
    let stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
                     "by", "about", "as", "of", "from", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall",
                     "should", "can", "could", "may", "might", "must", "that", "this", "these", 
                     "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", 
                     "us", "them"];
    
    let lowercase_text = text.to_lowercase();
    let words: Vec<&str> = lowercase_text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !stopwords.contains(w))
        .collect();
    
    // Count word frequencies
    let mut word_counts = std::collections::HashMap::new();
    for word in words {
        *word_counts.entry(word).or_insert(0) += 1;
    }
    
    // Sort by frequency
    let mut word_counts_vec: Vec<(&&str, &i32)> = word_counts.iter().collect();
    word_counts_vec.sort_by(|a, b| b.1.cmp(a.1));
    
    // Take the top N keywords
    let keywords: Vec<Value> = word_counts_vec
        .iter()
        .take(count)
        .map(|(word, count)| {
            let mut kw_map = std::collections::HashMap::new();
            kw_map.insert("word".to_string(), Value::String((*word).to_string()));
            kw_map.insert("count".to_string(), Value::Number(**count as f64));
            Value::Object(kw_map)
        })
        .collect();
    
    Ok(Value::Array(keywords))
}

// Helper functions

/// Calculate Flesch-Kincaid readability score
/// Returns a score from 0-100, where higher scores indicate easier readability
fn calculate_flesch_kincaid_score(text: &str) -> f64 {
    // Simple implementation of Flesch-Kincaid readability score
    // Formula: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    
    // Count sentences (simple heuristic)
    let sentence_count = text.split(['.', '!', '?']).filter(|s| !s.trim().is_empty()).count();
    
    // Count words
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();
    
    // Estimate syllable count (very simple heuristic)
    let syllable_count = words.iter().map(|word| estimate_syllables(word)).sum::<usize>();
    
    // Prevent division by zero
    if word_count == 0 || sentence_count == 0 {
        return 0.0;
    }
    
    // Calculate score
    let score = 206.835 - 1.015 * (word_count as f64 / sentence_count as f64) - 
                84.6 * (syllable_count as f64 / word_count as f64);
    
    // Clamp score to 0-100 range
    score.max(0.0).min(100.0)
}

/// Estimate syllable count for a word (very simple heuristic)
fn estimate_syllables(word: &str) -> usize {
    let word = word.to_lowercase();
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    
    let mut syllable_count = 0;
    let mut prev_is_vowel = false;
    
    for c in word.chars() {
        let is_vowel = vowels.contains(&c);
        if is_vowel && !prev_is_vowel {
            syllable_count += 1;
        }
        prev_is_vowel = is_vowel;
    }
    
    // Handle silent e at end of word
    if word.ends_with('e') && syllable_count > 1 {
        syllable_count -= 1;
    }
    
    // Every word has at least one syllable
    syllable_count.max(1)
} 