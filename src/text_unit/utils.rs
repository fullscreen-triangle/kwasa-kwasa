use std::collections::HashSet;

/// Calculate string similarity using Jaccard similarity of words
/// Returns a value between 0.0 (completely different) and 1.0 (identical)
pub fn string_similarity(s1: &str, s2: &str) -> f64 {
    let s1_lower = s1.to_lowercase();
    let s2_lower = s2.to_lowercase();
    
    let words1: HashSet<String> = s1_lower
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();
    
    let words2: HashSet<String> = s2_lower
        .split_whitespace()
        .map(|w| w.to_string())
        .collect();
    
    if words1.is_empty() && words2.is_empty() {
        return 1.0; // Both strings are empty
    }
    
    // Calculate proper Jaccard similarity using HashSet operations
    let intersection_size = words1.intersection(&words2).count();
    let union_size = words1.union(&words2).count();
    
    if union_size == 0 {
        return 0.0;
    }
    
    intersection_size as f64 / union_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_similarity() {
        // Identical strings
        assert_eq!(string_similarity("hello world", "hello world"), 1.0);
        
        // Completely different
        assert_eq!(string_similarity("hello world", "goodbye moon"), 0.0);
        
        // Partially similar
        let s1 = "Rust is a programming language";
        let s2 = "Rust is a systems programming language";
        let similarity = string_similarity(s1, s2);
        assert!(similarity > 0.6 && similarity < 1.0);
    }
} 