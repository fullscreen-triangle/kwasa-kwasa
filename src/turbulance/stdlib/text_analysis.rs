use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;
use regex::Regex;

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

/// Calculate n-gram probability for a sequence in text
pub fn ngram_probability(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "ngram_probability() requires text, sequence, and optional n".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let sequence = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be sequence string".to_string(),
        }),
    };

    let n = if args.len() > 2 {
        match &args[2] {
            Value::Number(num) => *num as usize,
            _ => 3,
        }
    } else {
        3
    };

    let probability = calculate_ngram_probability(text, sequence, n);
    Ok(Value::Number(probability))
}

/// Calculate conditional probability of sequence given condition
pub fn conditional_probability(args: Vec<Value>) -> Result<Value> {
    if args.len() < 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "conditional_probability() requires text, sequence, and condition".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let sequence = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be sequence string".to_string(),
        }),
    };

    let condition = match &args[2] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Third argument must be condition string".to_string(),
        }),
    };

    let probability = calculate_conditional_probability(text, sequence, condition);
    Ok(Value::Number(probability))
}

/// Map positional distribution of pattern across text
pub fn positional_distribution(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "positional_distribution() requires text and pattern".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be pattern string".to_string(),
        }),
    };

    let distribution = calculate_positional_distribution(text, pattern);
    let result: Vec<Value> = distribution.into_iter()
        .map(|(pos, count)| {
            let mut map = HashMap::new();
            map.insert("position".to_string(), Value::Number(pos as f64));
            map.insert("count".to_string(), Value::Number(count as f64));
            Value::Object(map)
        })
        .collect();

    Ok(Value::Array(result))
}

/// Calculate information entropy within sliding windows
pub fn entropy_measure(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "entropy_measure() requires at least text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let window_size = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => *n as usize,
            _ => 50,
        }
    } else {
        50
    };

    let entropy = calculate_entropy_measure(text, window_size);
    Ok(Value::Number(entropy))
}

/// Test statistical significance of sequence compared to baseline
pub fn sequence_significance(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "sequence_significance() requires text and sequence".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let sequence = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be sequence string".to_string(),
        }),
    };

    let significance = calculate_sequence_significance(text, sequence);
    Ok(Value::Number(significance))
}

/// Generate transition probability matrix for text elements
pub fn markov_transition(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "markov_transition() requires at least text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let order = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => *n as usize,
            _ => 1,
        }
    } else {
        1
    };

    let transitions = calculate_markov_transitions(text, order);
    let mut result_map = HashMap::new();
    
    for ((from, to), prob) in transitions {
        let key = format!("{}->{}", from, to);
        result_map.insert(key, Value::Number(prob));
    }
    
    Ok(Value::Object(result_map))
}

/// Analyze token frequency distribution against Zipf's law
pub fn zipf_analysis(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "zipf_analysis() requires text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let analysis = perform_zipf_analysis(text);
    Ok(Value::Object(analysis))
}

/// Measure information distribution across structural units
pub fn positional_entropy(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "positional_entropy() requires text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let unit = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "paragraph",
        }
    } else {
        "paragraph"
    };

    let entropy = calculate_positional_entropy(text, unit);
    Ok(Value::Number(entropy))
}

/// Evaluate how distinctive a sequence is in different contexts
pub fn contextual_uniqueness(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contextual_uniqueness() requires text and sequence".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let sequence = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be sequence string".to_string(),
        }),
    };

    let uniqueness = calculate_contextual_uniqueness(text, sequence);
    Ok(Value::Number(uniqueness))
}

/// Check if text contains specific patterns
pub fn contains(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contains() requires text and pattern".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be pattern string".to_string(),
        }),
    };

    // Check for case-insensitive pattern matching
    let case_sensitive = if args.len() > 2 {
        match &args[2] {
            Value::Boolean(b) => *b,
            _ => false, // Default to case-insensitive
        }
    } else {
        false
    };

    let contains_pattern = if case_sensitive {
        text.contains(pattern)
    } else {
        text.to_lowercase().contains(&pattern.to_lowercase())
    };

    Ok(Value::Boolean(contains_pattern))
}

/// Extract patterns from text using regex
pub fn extract_patterns(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "extract_patterns() requires text and pattern".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be pattern string".to_string(),
        }),
    };

    let matches = extract_regex_patterns(text, pattern)?;
    let result: Vec<Value> = matches.into_iter()
        .map(|m| Value::String(m))
        .collect();

    Ok(Value::Array(result))
}

// Helper functions for statistical analysis

fn calculate_ngram_probability(text: &str, sequence: &str, n: usize) -> f64 {
    let chars: Vec<char> = text.chars().collect();
    let seq_chars: Vec<char> = sequence.chars().collect();
    
    if seq_chars.len() != n || chars.len() < n {
        return 0.0;
    }
    
    let mut total_ngrams = 0;
    let mut matching_ngrams = 0;
    
    for i in 0..=chars.len() - n {
        total_ngrams += 1;
        let ngram: Vec<char> = chars[i..i + n].to_vec();
        if ngram == seq_chars {
            matching_ngrams += 1;
        }
    }
    
    if total_ngrams == 0 {
        0.0
    } else {
        matching_ngrams as f64 / total_ngrams as f64
    }
}

fn calculate_conditional_probability(text: &str, sequence: &str, condition: &str) -> f64 {
    let condition_count = text.matches(condition).count();
    if condition_count == 0 {
        return 0.0;
    }
    
    let joint_pattern = format!("{}{}", condition, sequence);
    let joint_count = text.matches(&joint_pattern).count();
    
    joint_count as f64 / condition_count as f64
}

fn calculate_positional_distribution(text: &str, pattern: &str) -> Vec<(usize, usize)> {
    let mut distribution = HashMap::new();
    let text_len = text.len();
    let pattern_len = pattern.len();
    
    if pattern_len == 0 || text_len == 0 {
        return Vec::new();
    }
    
    // Divide text into 10 sections for distribution analysis
    let section_size = text_len / 10;
    
    let mut start = 0;
    while let Some(pos) = text[start..].find(pattern) {
        let absolute_pos = start + pos;
        let section = if section_size > 0 {
            absolute_pos / section_size
        } else {
            0
        }.min(9);
        
        *distribution.entry(section).or_insert(0) += 1;
        start = absolute_pos + pattern_len;
    }
    
    let mut result: Vec<(usize, usize)> = distribution.into_iter().collect();
    result.sort_by_key(|(pos, _)| *pos);
    result
}

fn calculate_entropy_measure(text: &str, window_size: usize) -> f64 {
    if text.len() < window_size {
        return 0.0;
    }
    
    let mut total_entropy = 0.0;
    let mut window_count = 0;
    
    let chars: Vec<char> = text.chars().collect();
    
    for i in 0..=chars.len() - window_size {
        let window: Vec<char> = chars[i..i + window_size].to_vec();
        let entropy = calculate_shannon_entropy(&window);
        total_entropy += entropy;
        window_count += 1;
    }
    
    if window_count == 0 {
        0.0
    } else {
        total_entropy / window_count as f64
    }
}

fn calculate_shannon_entropy(data: &[char]) -> f64 {
    let mut char_counts = HashMap::new();
    
    for &ch in data {
        *char_counts.entry(ch).or_insert(0) += 1;
    }
    
    let total_chars = data.len() as f64;
    let mut entropy = 0.0;
    
    for count in char_counts.values() {
        let probability = *count as f64 / total_chars;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }
    
    entropy
}

fn calculate_sequence_significance(text: &str, sequence: &str) -> f64 {
    let observed_count = text.matches(sequence).count();
    let text_len = text.len();
    let seq_len = sequence.len();
    
    if seq_len == 0 || text_len < seq_len {
        return 0.0;
    }
    
    // Calculate expected frequency based on character frequency
    let char_counts: HashMap<char, usize> = text.chars().fold(HashMap::new(), |mut acc, c| {
        *acc.entry(c).or_insert(0) += 1;
        acc
    });
    
    let expected_prob: f64 = sequence.chars()
        .map(|c| char_counts.get(&c).unwrap_or(&0))
        .map(|&count| count as f64 / text_len as f64)
        .product();
    
    let possible_positions = text_len - seq_len + 1;
    let expected_count = expected_prob * possible_positions as f64;
    
    if expected_count == 0.0 {
        0.0
    } else {
        observed_count as f64 / expected_count
    }
}

fn calculate_markov_transitions(text: &str, order: usize) -> HashMap<(String, String), f64> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut transitions: HashMap<(String, String), i32> = HashMap::new();
    let mut from_counts: HashMap<String, i32> = HashMap::new();
    
    if words.len() <= order {
        return HashMap::new();
    }
    
    for i in 0..words.len() - order {
        let from_state = words[i..i + order].join(" ");
        let to_state = words[i + order].to_string();
        
        *transitions.entry((from_state.clone(), to_state)).or_insert(0i32) += 1i32;
        *from_counts.entry(from_state).or_insert(0i32) += 1i32;
    }
    
    // Convert counts to probabilities
    let mut probabilities = HashMap::new();
    for ((from, to), count) in transitions {
        let total_from = from_counts[&from];
        probabilities.insert((from, to), count as f64 / total_from as f64);
    }
    
    probabilities
}

fn perform_zipf_analysis(text: &str) -> HashMap<String, Value> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut word_counts = HashMap::new();
    
    for word in words {
        let word = word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string();
        if !word.is_empty() {
            *word_counts.entry(word).or_insert(0) += 1;
        }
    }
    
    let mut sorted_counts: Vec<(String, usize)> = word_counts.into_iter().collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));
    
    let mut result = HashMap::new();
    result.insert("total_unique_words".to_string(), Value::Number(sorted_counts.len() as f64));
    
    if !sorted_counts.is_empty() {
        result.insert("most_frequent_word".to_string(), Value::String(sorted_counts[0].0.clone()));
        result.insert("most_frequent_count".to_string(), Value::Number(sorted_counts[0].1 as f64));
        
        // Calculate Zipf coefficient
        let zipf_coefficient = if sorted_counts.len() > 1 {
            let first_freq = sorted_counts[0].1 as f64;
            let second_freq = sorted_counts[1].1 as f64;
            if second_freq > 0.0 {
                first_freq / second_freq
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        result.insert("zipf_coefficient".to_string(), Value::Number(zipf_coefficient));
    }
    
    result
}

fn calculate_positional_entropy(text: &str, unit: &str) -> f64 {
    let units: Vec<&str> = match unit {
        "sentence" => text.split(&['.', '!', '?']).collect(),
        "paragraph" => text.split("\n\n").collect(),
        "word" => text.split_whitespace().collect(),
        _ => text.split("\n\n").collect(),
    };
    
    if units.is_empty() {
        return 0.0;
    }
    
    let entropies: Vec<f64> = units.iter()
        .map(|unit| {
            let chars: Vec<char> = unit.chars().collect();
            calculate_shannon_entropy(&chars)
        })
        .collect();
    
    entropies.iter().sum::<f64>() / entropies.len() as f64
}

fn calculate_contextual_uniqueness(text: &str, sequence: &str) -> f64 {
    let contexts = extract_contexts(text, sequence, 10);
    
    if contexts.is_empty() {
        return 0.0;
    }
    
    let unique_contexts: std::collections::HashSet<String> = contexts.into_iter().collect();
    unique_contexts.len() as f64 / text.matches(sequence).count().max(1) as f64
}

fn extract_contexts(text: &str, sequence: &str, context_window: usize) -> Vec<String> {
    let mut contexts = Vec::new();
    let mut start = 0;
    
    while let Some(pos) = text[start..].find(sequence) {
        let absolute_pos = start + pos;
        let context_start = absolute_pos.saturating_sub(context_window);
        let context_end = (absolute_pos + sequence.len() + context_window).min(text.len());
        
        let context = text[context_start..context_end].to_string();
        contexts.push(context);
        
        start = absolute_pos + sequence.len();
    }
    
    contexts
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

fn extract_regex_patterns(text: &str, pattern: &str) -> Result<Vec<String>> {
    let re = Regex::new(pattern)
        .map_err(|e| TurbulanceError::RuntimeError {
            message: format!("Invalid regex pattern: {}", e),
        })?;
    let matches: Vec<_> = re.find_iter(text).map(|m| m.as_str().to_string()).collect();
    Ok(matches)
} 