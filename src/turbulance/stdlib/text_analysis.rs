use crate::turbulance::interpreter::Value;
use crate::turbulance::Result;
use crate::turbulance::TurbulanceError;
use std::collections::HashMap;
use std::rc::Rc;

/// Calculate readability score using multiple metrics
pub fn readability_score(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "readability_score requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "readability_score requires a string or TextUnit".to_string(),
        }),
    };

    // Calculate Flesch Reading Ease score
    let words = text.split_whitespace().count() as f64;
    if words == 0.0 {
        return Ok(Value::Number(0.0));
    }

    let sentences = text.split(&['.', '!', '?'][..]).filter(|s| !s.trim().is_empty()).count() as f64;
    let syllables = count_syllables(text) as f64;

    let flesch_score = if sentences > 0.0 {
        206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    } else {
        0.0
    };

    Ok(Value::Number(flesch_score.max(0.0).min(100.0)))
}

/// Perform sentiment analysis on text
pub fn sentiment_analysis(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "sentiment_analysis requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "sentiment_analysis requires a string or TextUnit".to_string(),
        }),
    };

    // Simple lexicon-based sentiment analysis
    let positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "best", "awesome", "brilliant", "outstanding", "magnificent", "superb", "terrific"];
    let negative_words = ["bad", "terrible", "awful", "horrible", "hate", "dislike", "worst", "terrible", "dreadful", "appalling", "disgusting", "offensive", "unpleasant"];

    let words: Vec<&str> = text.to_lowercase().split_whitespace().collect();
    let mut positive_count = 0;
    let mut negative_count = 0;

    for word in &words {
        let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
        if positive_words.contains(&clean_word) {
            positive_count += 1;
        }
        if negative_words.contains(&clean_word) {
            negative_count += 1;
        }
    }

    let sentiment_score = if words.is_empty() {
        0.0
    } else {
        (positive_count as f64 - negative_count as f64) / words.len() as f64
    };

    // Create sentiment result map
    let mut result = HashMap::new();
    result.insert("score".to_string(), Value::Number(sentiment_score));
    result.insert("positive_count".to_string(), Value::Number(positive_count as f64));
    result.insert("negative_count".to_string(), Value::Number(negative_count as f64));
    result.insert("polarity".to_string(), Value::String(
        if sentiment_score > 0.1 { "positive" }
        else if sentiment_score < -0.1 { "negative" }
        else { "neutral" }.to_string()
    ));

    Ok(Value::Map(result))
}

/// Extract keywords from text using TF-IDF-like scoring
pub fn extract_keywords(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "extract_keywords requires 1-2 arguments (text, optional count)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "extract_keywords requires a string or TextUnit".to_string(),
        }),
    };

    let max_keywords = if args.len() > 1 {
        match &args[1] {
            Value::Number(n) => *n as usize,
            _ => 10,
        }
    } else {
        10
    };

    // Simple keyword extraction based on word frequency and length
    let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"];

    let words: Vec<&str> = text.to_lowercase()
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()))
        .filter(|w| w.len() > 3 && !stop_words.contains(w))
        .collect();

    let mut word_freq = HashMap::new();
    for word in &words {
        *word_freq.entry(*word).or_insert(0) += 1;
    }

    // Score words by frequency and length
    let mut scored_words: Vec<_> = word_freq.iter()
        .map(|(word, freq)| {
            let score = *freq as f64 * (word.len() as f64).ln();
            (*word, score)
        })
        .collect();

    scored_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let keywords: Vec<Value> = scored_words
        .into_iter()
        .take(max_keywords)
        .map(|(word, score)| {
            let mut keyword_info = HashMap::new();
            keyword_info.insert("word".to_string(), Value::String(word.to_string()));
            keyword_info.insert("score".to_string(), Value::Number(score));
            Value::Map(keyword_info)
        })
        .collect();

    Ok(Value::List(keywords))
}

/// Check if text contains a pattern
pub fn contains(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contains requires exactly 2 arguments (text, pattern)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contains first argument must be a string or TextUnit".to_string(),
        }),
    };

    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contains second argument must be a string".to_string(),
        }),
    };

    let contains_pattern = text.to_lowercase().contains(&pattern.to_lowercase());
    Ok(Value::Boolean(contains_pattern))
}

/// Extract patterns using regex-like matching
pub fn extract_patterns(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "extract_patterns requires exactly 2 arguments (text, pattern_type)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "extract_patterns first argument must be a string or TextUnit".to_string(),
        }),
    };

    let pattern_type = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "extract_patterns second argument must be a string".to_string(),
        }),
    };

    let patterns = match pattern_type.as_str() {
        "emails" => extract_email_patterns(text),
        "urls" => extract_url_patterns(text),
        "numbers" => extract_number_patterns(text),
        "dates" => extract_date_patterns(text),
        "citations" => extract_citation_patterns(text),
        _ => Vec::new(),
    };

    let pattern_values: Vec<Value> = patterns.into_iter()
        .map(|p| Value::String(p))
        .collect();

    Ok(Value::List(pattern_values))
}

/// Calculate n-gram probability in text
pub fn ngram_probability(args: Vec<Value>) -> Result<Value> {
    if args.len() != 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "ngram_probability requires exactly 3 arguments (text, ngram, n)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ngram_probability first argument must be a string or TextUnit".to_string(),
        }),
    };

    let ngram = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ngram_probability second argument must be a string".to_string(),
        }),
    };

    let n = match &args[2] {
        Value::Number(num) => *num as usize,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ngram_probability third argument must be a number".to_string(),
        }),
    };

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < n {
        return Ok(Value::Number(0.0));
    }

    let ngram_words: Vec<&str> = ngram.split_whitespace().collect();
    if ngram_words.len() != n {
        return Err(TurbulanceError::RuntimeError {
            message: format!("ngram length must match n ({})", n),
        });
    }

    // Count total n-grams and target n-gram occurrences
    let mut total_ngrams = 0;
    let mut target_count = 0;

    for i in 0..=words.len() - n {
        total_ngrams += 1;
        let current_ngram = &words[i..i + n];
        if current_ngram == ngram_words.as_slice() {
            target_count += 1;
        }
    }

    let probability = if total_ngrams > 0 {
        target_count as f64 / total_ngrams as f64
    } else {
        0.0
    };

    Ok(Value::Number(probability))
}

/// Calculate conditional probability P(B|A) in text
pub fn conditional_probability(args: Vec<Value>) -> Result<Value> {
    if args.len() != 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "conditional_probability requires exactly 3 arguments (text, event_a, event_b)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "conditional_probability first argument must be a string or TextUnit".to_string(),
        }),
    };

    let event_a = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "conditional_probability second argument must be a string".to_string(),
        }),
    };

    let event_b = match &args[2] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "conditional_probability third argument must be a string".to_string(),
        }),
    };

    let sentences: Vec<&str> = text.split(&['.', '!', '?'][..])
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut count_a = 0;
    let mut count_both = 0;

    for sentence in &sentences {
        let sentence_lower = sentence.to_lowercase();
        let has_a = sentence_lower.contains(&event_a.to_lowercase());
        let has_b = sentence_lower.contains(&event_b.to_lowercase());

        if has_a {
            count_a += 1;
            if has_b {
                count_both += 1;
            }
        }
    }

    let conditional_prob = if count_a > 0 {
        count_both as f64 / count_a as f64
    } else {
        0.0
    };

    Ok(Value::Number(conditional_prob))
}

/// Calculate positional distribution of words
pub fn positional_distribution(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "positional_distribution requires exactly 2 arguments (text, word)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "positional_distribution first argument must be a string or TextUnit".to_string(),
        }),
    };

    let target_word = match &args[1] {
        Value::String(s) => s.to_lowercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "positional_distribution second argument must be a string".to_string(),
        }),
    };

    let words: Vec<&str> = text.split_whitespace().collect();
    let total_words = words.len();

    if total_words == 0 {
        return Ok(Value::Map(HashMap::new()));
    }

    let mut positions = Vec::new();
    for (i, word) in words.iter().enumerate() {
        if word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()) == target_word {
            let relative_position = i as f64 / total_words as f64;
            positions.push(relative_position);
        }
    }

    let mut result = HashMap::new();
    result.insert("positions".to_string(), Value::List(
        positions.iter().map(|&pos| Value::Number(pos)).collect()
    ));
    result.insert("count".to_string(), Value::Number(positions.len() as f64));
    result.insert("mean_position".to_string(), Value::Number(
        if !positions.is_empty() {
            positions.iter().sum::<f64>() / positions.len() as f64
        } else {
            0.0
        }
    ));

    Ok(Value::Map(result))
}

/// Calculate entropy measure of text
pub fn entropy_measure(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "entropy_measure requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "entropy_measure requires a string or TextUnit".to_string(),
        }),
    };

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(Value::Number(0.0));
    }

    let mut word_counts = HashMap::new();
    for word in &words {
        let clean_word = word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic());
        *word_counts.entry(clean_word).or_insert(0) += 1;
    }

    let total_words = words.len() as f64;
    let mut entropy = 0.0;

    for count in word_counts.values() {
        let probability = *count as f64 / total_words;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }

    Ok(Value::Number(entropy))
}

/// Calculate sequence significance using statistical methods
pub fn sequence_significance(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "sequence_significance requires exactly 2 arguments (text, sequence)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "sequence_significance first argument must be a string or TextUnit".to_string(),
        }),
    };

    let sequence = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "sequence_significance second argument must be a string".to_string(),
        }),
    };

    let words: Vec<&str> = text.split_whitespace().collect();
    let sequence_words: Vec<&str> = sequence.split_whitespace().collect();
    let seq_len = sequence_words.len();

    if words.len() < seq_len || seq_len == 0 {
        return Ok(Value::Number(0.0));
    }

    // Count actual occurrences
    let mut actual_count = 0;
    for i in 0..=words.len() - seq_len {
        let window = &words[i..i + seq_len];
        if window.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>() == 
           sequence_words.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>() {
            actual_count += 1;
        }
    }

    // Calculate expected frequency based on individual word frequencies
    let mut word_probs = Vec::new();
    for seq_word in &sequence_words {
        let word_count = words.iter()
            .filter(|w| w.to_lowercase() == seq_word.to_lowercase())
            .count();
        word_probs.push(word_count as f64 / words.len() as f64);
    }

    let expected_prob: f64 = word_probs.iter().product();
    let possible_positions = words.len() - seq_len + 1;
    let expected_count = expected_prob * possible_positions as f64;

    // Calculate significance (simplified chi-square-like measure)
    let significance = if expected_count > 0.0 {
        (actual_count as f64 - expected_count).abs() / expected_count.sqrt()
    } else {
        0.0
    };

    let mut result = HashMap::new();
    result.insert("actual_count".to_string(), Value::Number(actual_count as f64));
    result.insert("expected_count".to_string(), Value::Number(expected_count));
    result.insert("significance".to_string(), Value::Number(significance));
    result.insert("over_expected".to_string(), Value::Boolean(actual_count as f64 > expected_count));

    Ok(Value::Map(result))
}

/// Calculate Markov transition probabilities
pub fn markov_transition(args: Vec<Value>) -> Result<Value> {
    if args.len() != 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "markov_transition requires exactly 3 arguments (text, from_word, to_word)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "markov_transition first argument must be a string or TextUnit".to_string(),
        }),
    };

    let from_word = match &args[1] {
        Value::String(s) => s.to_lowercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "markov_transition second argument must be a string".to_string(),
        }),
    };

    let to_word = match &args[2] {
        Value::String(s) => s.to_lowercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "markov_transition third argument must be a string".to_string(),
        }),
    };

    let words: Vec<String> = text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .collect();

    let mut from_count = 0;
    let mut transition_count = 0;

    for i in 0..words.len() - 1 {
        if words[i] == from_word {
            from_count += 1;
            if words[i + 1] == to_word {
                transition_count += 1;
            }
        }
    }

    let transition_probability = if from_count > 0 {
        transition_count as f64 / from_count as f64
    } else {
        0.0
    };

    Ok(Value::Number(transition_probability))
}

/// Perform Zipf's law analysis on text
pub fn zipf_analysis(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "zipf_analysis requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "zipf_analysis requires a string or TextUnit".to_string(),
        }),
    };

    let words: Vec<String> = text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    let mut word_freq = HashMap::new();
    for word in &words {
        *word_freq.entry(word.clone()).or_insert(0) += 1;
    }

    let mut freq_vec: Vec<_> = word_freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    let mut zipf_data = Vec::new();
    for (rank, (word, freq)) in freq_vec.iter().enumerate() {
        let expected_freq = if rank == 0 { *freq } else {
            freq_vec[0].1 as f64 / (rank + 1) as f64
        };
        
        let mut word_data = HashMap::new();
        word_data.insert("word".to_string(), Value::String((*word).clone()));
        word_data.insert("rank".to_string(), Value::Number((rank + 1) as f64));
        word_data.insert("frequency".to_string(), Value::Number(**freq as f64));
        word_data.insert("expected_frequency".to_string(), Value::Number(expected_freq));
        word_data.insert("zipf_deviation".to_string(), Value::Number(
            (**freq as f64 - expected_freq).abs() / expected_freq
        ));
        
        zipf_data.push(Value::Map(word_data));
        
        if zipf_data.len() >= 20 { // Limit to top 20 words
            break;
        }
    }

    Ok(Value::List(zipf_data))
}

/// Calculate positional entropy
pub fn positional_entropy(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "positional_entropy requires exactly 2 arguments (text, window_size)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "positional_entropy first argument must be a string or TextUnit".to_string(),
        }),
    };

    let window_size = match &args[1] {
        Value::Number(n) => *n as usize,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "positional_entropy second argument must be a number".to_string(),
        }),
    };

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < window_size {
        return Ok(Value::Number(0.0));
    }

    let mut position_entropies = Vec::new();
    
    for start in 0..=words.len() - window_size {
        let window = &words[start..start + window_size];
        let mut word_counts = HashMap::new();
        
        for word in window {
            let clean_word = word.to_lowercase().trim_matches(|c: char| !c.is_alphabetic());
            *word_counts.entry(clean_word).or_insert(0) += 1;
        }
        
        let total_words = window_size as f64;
        let mut entropy = 0.0;
        
        for count in word_counts.values() {
            let prob = *count as f64 / total_words;
            if prob > 0.0 {
                entropy -= prob * prob.log2();
            }
        }
        
        position_entropies.push(entropy);
    }

    let avg_entropy = if !position_entropies.is_empty() {
        position_entropies.iter().sum::<f64>() / position_entropies.len() as f64
    } else {
        0.0
    };

    Ok(Value::Number(avg_entropy))
}

/// Calculate contextual uniqueness of words
pub fn contextual_uniqueness(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contextual_uniqueness requires exactly 2 arguments (text, target_word)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contextual_uniqueness first argument must be a string or TextUnit".to_string(),
        }),
    };

    let target_word = match &args[1] {
        Value::String(s) => s.to_lowercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contextual_uniqueness second argument must be a string".to_string(),
        }),
    };

    let words: Vec<String> = text.split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .collect();

    // Find contexts where target word appears
    let mut contexts = Vec::new();
    let context_window = 3; // 3 words before and after

    for (i, word) in words.iter().enumerate() {
        if *word == target_word {
            let start = if i >= context_window { i - context_window } else { 0 };
            let end = std::cmp::min(i + context_window + 1, words.len());
            let context: Vec<String> = words[start..end].iter()
                .filter(|w| **w != target_word)
                .cloned()
                .collect();
            contexts.push(context);
        }
    }

    if contexts.is_empty() {
        return Ok(Value::Number(0.0));
    }

    // Calculate uniqueness based on context diversity
    let mut all_context_words = HashMap::new();
    let mut total_context_words = 0;

    for context in &contexts {
        for word in context {
            *all_context_words.entry(word.clone()).or_insert(0) += 1;
            total_context_words += 1;
        }
    }

    // Calculate Shannon diversity of contexts
    let mut diversity = 0.0;
    for count in all_context_words.values() {
        if total_context_words > 0 {
            let prob = *count as f64 / total_context_words as f64;
            if prob > 0.0 {
                diversity -= prob * prob.log2();
            }
        }
    }

    // Normalize by theoretical maximum diversity
    let max_diversity = if all_context_words.len() > 1 {
        (all_context_words.len() as f64).log2()
    } else {
        1.0
    };

    let uniqueness = if max_diversity > 0.0 {
        diversity / max_diversity
    } else {
        0.0
    };

    Ok(Value::Number(uniqueness))
}

// Helper functions

fn count_syllables(text: &str) -> usize {
    let vowels = ['a', 'e', 'i', 'o', 'u', 'y'];
    let mut syllable_count = 0;
    
    for word in text.split_whitespace() {
        let clean_word = word.to_lowercase().chars()
            .filter(|c| c.is_alphabetic())
            .collect::<String>();
        
        if clean_word.is_empty() {
            continue;
        }
        
        let mut word_syllables = 0;
        let mut prev_was_vowel = false;
        
        for ch in clean_word.chars() {
            let is_vowel = vowels.contains(&ch);
            if is_vowel && !prev_was_vowel {
                word_syllables += 1;
            }
            prev_was_vowel = is_vowel;
        }
        
        // Handle silent 'e'
        if clean_word.ends_with('e') && word_syllables > 1 {
            word_syllables -= 1;
        }
        
        // Every word has at least one syllable
        if word_syllables == 0 {
            word_syllables = 1;
        }
        
        syllable_count += word_syllables;
    }
    
    syllable_count
}

fn extract_email_patterns(text: &str) -> Vec<String> {
    let mut emails = Vec::new();
    for word in text.split_whitespace() {
        if word.contains('@') && word.contains('.') {
            emails.push(word.to_string());
        }
    }
    emails
}

fn extract_url_patterns(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    for word in text.split_whitespace() {
        if word.starts_with("http://") || word.starts_with("https://") || word.starts_with("www.") {
            urls.push(word.to_string());
        }
    }
    urls
}

fn extract_number_patterns(text: &str) -> Vec<String> {
    let mut numbers = Vec::new();
    for word in text.split_whitespace() {
        if word.chars().any(|c| c.is_numeric()) {
            numbers.push(word.to_string());
        }
    }
    numbers
}

fn extract_date_patterns(text: &str) -> Vec<String> {
    let mut dates = Vec::new();
    for word in text.split_whitespace() {
        // Simple date pattern detection
        if word.contains('/') || word.contains('-') {
            if word.chars().filter(|c| c.is_numeric()).count() >= 4 {
                dates.push(word.to_string());
            }
        }
    }
    dates
}

fn extract_citation_patterns(text: &str) -> Vec<String> {
    let mut citations = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    
    for i in 0..words.len() {
        let word = words[i];
        // Look for patterns like "Smith (2020)" or "[1]"
        if word.starts_with('(') && word.ends_with(')') && word.len() > 4 {
            if let Some(prev_word) = words.get(i.saturating_sub(1)) {
                citations.push(format!("{} {}", prev_word, word));
            }
        } else if word.starts_with('[') && word.ends_with(']') {
            citations.push(word.to_string());
        }
    }
    
    citations
} 