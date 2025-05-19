//! Pattern Analysis extension for Kwasa-Kwasa
//! 
//! This module provides types and operations for extracting meaning from
//! fundamental character/symbol patterns, regardless of semantic content.

use std::fmt::Debug;
use std::{collections::HashMap, marker::PhantomData};
use std::hash::Hash;

/// Re-exports from this module
pub mod prelude {
    pub use super::{
        PatternAnalyzer, OrthographicAnalyzer, FrequencyDistribution,
        Pattern, PatternSignificance, VisualDensityMap,
    };
}

/// Metadata for patterns
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PatternMetadata {
    /// Source of the pattern
    pub source: Option<String>,
    /// Additional key-value annotations
    pub annotations: HashMap<String, String>,
}

/// A pattern identified in a unit
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    /// The raw pattern content
    pub pattern: Vec<u8>,
    /// Pattern length
    pub label: String,
    /// Pattern occurrences
    pub positions: Vec<usize>,
    /// Pattern significance score
    pub significance: f64,
    /// Pattern metadata
    pub metadata: PatternMetadata,
}

impl Pattern {
    /// Create a new pattern
    pub fn new(content: impl Into<Vec<u8>>) -> Self {
        let content = content.into();
        let length = content.len();
        
        Self {
            pattern: content,
            label: String::new(),
            positions: Vec::new(),
            significance: 0.0,
            metadata: PatternMetadata::default(),
        }
    }
    
    /// Create a new pattern with occurrence information
    pub fn with_occurrences(content: impl Into<Vec<u8>>, occurrences: usize) -> Self {
        let content = content.into();
        let length = content.len();
        
        Self {
            pattern: content,
            label: String::new(),
            positions: Vec::new(),
            significance: 0.0,
            metadata: PatternMetadata::default(),
        }
    }
    
    /// Set the significance score for this pattern
    pub fn with_significance(mut self, significance: f64) -> Self {
        self.significance = significance;
        self
    }
    
    /// Get the raw content of this pattern
    pub fn content(&self) -> &[u8] {
        &self.pattern
    }
    
    /// Get the number of occurrences of this pattern
    pub fn occurrences(&self) -> usize {
        self.positions.len()
    }
    
    /// Get the significance score of this pattern
    pub fn significance(&self) -> f64 {
        self.significance
    }
}

/// Distribution of pattern frequencies
#[derive(Debug, Clone)]
pub struct FrequencyDistribution<T: Eq + Hash> {
    /// Raw frequency counts
    counts: HashMap<T, usize>,
    /// Total count of all patterns
    total: usize,
    /// Shannon entropy of the distribution
    entropy: f64,
}

impl<T: Eq + Hash + Clone> FrequencyDistribution<T> {
    /// Create a new empty frequency distribution
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            entropy: 0.0,
        }
    }
    
    /// Add a single occurrence of a pattern
    pub fn add(&mut self, pattern: T) {
        *self.counts.entry(pattern).or_insert(0) += 1;
        self.total += 1;
        self.recalculate_entropy();
    }
    
    /// Add multiple occurrences of a pattern
    pub fn add_multiple(&mut self, pattern: T, count: usize) {
        *self.counts.entry(pattern).or_insert(0) += count;
        self.total += count;
        self.recalculate_entropy();
    }
    
    /// Get the frequency of a pattern
    pub fn frequency(&self, pattern: &T) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        
        match self.counts.get(pattern) {
            Some(count) => *count as f64 / self.total as f64,
            None => 0.0,
        }
    }
    
    /// Get the Shannon entropy of this distribution
    pub fn entropy(&self) -> f64 {
        self.entropy
    }
    
    /// Recalculate entropy after changes
    fn recalculate_entropy(&mut self) {
        if self.total == 0 {
            self.entropy = 0.0;
            return;
        }
        
        let mut entropy = 0.0;
        for &count in self.counts.values() {
            let probability = count as f64 / self.total as f64;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        self.entropy = entropy;
    }
    
    /// Get the most frequent patterns, limited by count
    pub fn most_frequent(&self, limit: usize) -> Vec<(T, usize)> {
        let mut entries: Vec<_> = self.counts.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(limit);
        
        entries
    }
    
    /// Compare with an expected distribution to find anomalies
    pub fn compare_with_expected(&self, expected: &Self) -> Vec<(T, f64)> {
        let mut anomalies = Vec::new();
        
        for (pattern, &count) in &self.counts {
            let observed_freq = count as f64 / self.total as f64;
            let expected_freq = expected.frequency(pattern);
            
            if expected_freq > 0.0 {
                let ratio = observed_freq / expected_freq;
                if ratio < 0.5 || ratio > 2.0 {
                    anomalies.push((pattern.clone(), ratio));
                }
            } else if observed_freq > 0.01 {
                // Pattern present in observed but not in expected
                anomalies.push((pattern.clone(), f64::INFINITY));
            }
        }
        
        anomalies.sort_by(|a, b| {
            let a_dist = (a.1 - 1.0).abs();
            let b_dist = (b.1 - 1.0).abs();
            b_dist.partial_cmp(&a_dist).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        anomalies
    }
}

/// Statistical significance of a pattern
#[derive(Debug, Clone)]
pub struct PatternSignificance {
    /// The pattern being analyzed
    pattern: Pattern,
    /// Z-score relative to expected frequency
    z_score: f64,
    /// p-value (probability of occurring by chance)
    p_value: f64,
    /// Is this pattern statistically significant?
    is_significant: bool,
}

/// Visual density map for orthographic analysis
#[derive(Debug, Clone)]
pub struct VisualDensityMap {
    /// Width of the analyzed text
    width: usize,
    /// Height of the analyzed text
    height: usize,
    /// Density values (0.0 to 1.0)
    densities: Vec<f64>,
}

impl VisualDensityMap {
    /// Create a new density map with the given dimensions
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            densities: vec![0.0; width * height],
        }
    }
    
    /// Set the density at a specific position
    pub fn set_density(&mut self, x: usize, y: usize, density: f64) {
        if x < self.width && y < self.height {
            self.densities[y * self.width + x] = density;
        }
    }
    
    /// Get the density at a specific position
    pub fn density_at(&self, x: usize, y: usize) -> Option<f64> {
        if x < self.width && y < self.height {
            Some(self.densities[y * self.width + x])
        } else {
            None
        }
    }
    
    /// Get the average density of the entire map
    pub fn average_density(&self) -> f64 {
        if self.densities.is_empty() {
            return 0.0;
        }
        
        self.densities.iter().sum::<f64>() / self.densities.len() as f64
    }
}

/// Analyzer for pattern-based meaning extraction
#[derive(Debug)]
pub struct PatternAnalyzer<T> {
    _phantom: PhantomData<T>,
}

impl<T> PatternAnalyzer<T> {
    /// Create a new pattern analyzer
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

/// Analyzer for text units that works with u8 content
impl PatternAnalyzer<Vec<u8>> {
    /// Analyze n-gram frequency distribution
    pub fn analyze_ngrams(&self, content: &[u8], n: usize) -> FrequencyDistribution<Vec<u8>> {
        let mut distribution = FrequencyDistribution::new();
        
        if content.len() < n {
            return distribution;
        }
        
        for i in 0..=content.len() - n {
            let ngram = content[i..i+n].to_vec();
            distribution.add(ngram);
        }
        
        distribution
    }
    
    /// Calculate the Shannon entropy of the content
    pub fn shannon_entropy(&self, content: &[u8]) -> f64 {
        let distribution = self.analyze_ngrams(content, 1);
        distribution.entropy()
    }
    
    /// Detect statistically significant patterns
    pub fn detect_significant_patterns(&self, content: &[u8], min_len: usize, max_len: usize) -> Vec<Pattern> {
        let mut patterns = Vec::new();
        
        // Analyze n-grams of various lengths
        for n in min_len..=max_len {
            if n > content.len() {
                break;
            }
            
            let distribution = self.analyze_ngrams(content, n);
            
            for (ngram, count) in distribution.counts.iter() {
                let frequency = *count as f64 / (content.len() - n + 1) as f64;
                
                // Simple significance heuristic based on length and frequency
                let expected_freq = 1.0 / 4f64.powi(n as i32); // Assuming 4 nucleotides
                let significance = frequency / expected_freq;
                
                if significance > 2.0 {
                    patterns.push(Pattern::with_occurrences(ngram.clone(), *count)
                        .with_significance(significance));
                }
            }
        }
        
        // Sort by significance
        patterns.sort_by(|a, b| b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal));
        
        patterns
    }
    
    /// Analyze the pattern structure and return a compressed representation
    pub fn compress_by_patterns(&self, content: &[u8]) -> (Vec<Vec<u8>>, Vec<usize>) {
        let significant_patterns = self.detect_significant_patterns(content, 2, 10);
        
        if significant_patterns.is_empty() {
            return (vec![content.to_vec()], vec![0]);
        }
        
        let mut compressed_content = Vec::new();
        let mut pattern_indices = Vec::new();
        let patterns: Vec<_> = significant_patterns.iter().map(|p| p.pattern.clone()).collect();
        
        let mut i = 0;
        while i < content.len() {
            let mut matched = false;
            
            for (pattern_idx, pattern) in patterns.iter().enumerate() {
                if i + pattern.len() <= content.len() && content[i..i+pattern.len()] == *pattern {
                    pattern_indices.push(pattern_idx);
                    i += pattern.len();
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                if pattern_indices.is_empty() || pattern_indices.last() != Some(&(patterns.len() + compressed_content.len())) {
                    pattern_indices.push(patterns.len() + compressed_content.len());
                    compressed_content.push(vec![content[i]]);
                } else {
                    compressed_content.last_mut().unwrap().push(content[i]);
                }
                i += 1;
            }
        }
        
        (patterns.into_iter().chain(compressed_content.into_iter()).collect(), pattern_indices)
    }
}

/// Analyzer for orthographic features of text
pub struct OrthographicAnalyzer;

impl OrthographicAnalyzer {
    /// Create a new orthographic analyzer
    pub fn new() -> Self {
        Self
    }
    
    /// Generate a visual density map for the given text
    pub fn visual_density(&self, text: &[u8], width: usize) -> VisualDensityMap {
        // Determine density values based on character properties
        let mut density_values = HashMap::new();
        
        // Assign density values to different character types
        // These are arbitrary but aim to capture visual weight
        for &b in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" {
            let base_density = match b {
                b'i' | b'l' | b'j' | b'I' | b'.' | b',' | b'\'' | b'`' => 0.2,
                b'r' | b'c' | b's' | b't' | b'v' | b'x' | b'z' | b'y' => 0.5,
                b'o' | b'e' | b'a' | b'u' | b'n' => 0.6,
                b'm' | b'w' | b'O' | b'Q' | b'G' | b'B' => 0.8,
                b'W' | b'M' | b'%' | b'@' | b'#' => 0.9,
                _ => 0.4,
            };
            
            density_values.insert(b, base_density);
        }
        
        // Calculate height based on the width
        let height = (text.len() + width - 1) / width;
        let mut density_map = VisualDensityMap::new(width, height);
        
        for (i, &byte) in text.iter().enumerate() {
            let x = i % width;
            let y = i / width;
            let density = *density_values.get(&byte).unwrap_or(&0.1);
            density_map.set_density(x, y, density);
        }
        
        density_map
    }
    
    /// Extract visual rhythm data from text
    pub fn visual_rhythm(&self, text: &[u8]) -> Vec<f64> {
        let mut rhythm = Vec::new();
        let mut current_weight = 0.0;
        
        // Analyze the rhythm of density changes
        for &byte in text {
            let weight = match byte {
                // Short characters
                b'i' | b'l' | b'.' | b',' | b'\'' => 0.2,
                // Medium height characters
                b'a' | b'c' | b'e' | b'm' | b'n' | b'o' | b'r' | b's' | b'u' | b'v' | b'w' | b'x' | b'z' => 0.5,
                // Tall characters
                b'b' | b'd' | b'f' | b'h' | b'k' | b't' => 0.7,
                // Characters with descenders
                b'g' | b'j' | b'p' | b'q' | b'y' => 0.8,
                // Space
                b' ' => 0.1,
                // Other
                _ => 0.4,
            };
            
            // Create a moving average of visual weight
            current_weight = 0.8 * current_weight + 0.2 * weight;
            rhythm.push(current_weight);
        }
        
        rhythm
    }
    
    /// Find common visual patterns in the text
    pub fn find_visual_patterns(&self, text: &[u8], pattern_length: usize) -> Vec<(Vec<u8>, f64)> {
        if text.len() < pattern_length {
            return Vec::new();
        }
        
        // Extract visual patterns based on character shape classes
        let mut shape_map = HashMap::new();
        
        // Map characters to shape classes
        for (i, &byte) in text.iter().enumerate() {
            let shape_class = match byte {
                b'a' | b'c' | b'e' | b'o' | b's' => 0, // round shapes
                b'i' | b'l' | b'I' | b'j' | b'f' | b't' => 1, // vertical strokes
                b'm' | b'n' | b'h' | b'u' => 2, // arch shapes
                b'v' | b'w' | b'x' | b'y' | b'z' => 3, // angled shapes
                b'b' | b'd' | b'p' | b'q' | b'g' => 4, // circles with stems
                _ => 5, // other
            };
            
            if i + pattern_length <= text.len() {
                let mut shape_pattern = Vec::with_capacity(pattern_length);
                for j in 0..pattern_length {
                    let shape_class_j = match text[i + j] {
                        b'a' | b'c' | b'e' | b'o' | b's' => 0,
                        b'i' | b'l' | b'I' | b'j' | b'f' | b't' => 1,
                        b'm' | b'n' | b'h' | b'u' => 2,
                        b'v' | b'w' | b'x' | b'y' | b'z' => 3,
                        b'b' | b'd' | b'p' | b'q' | b'g' => 4,
                        _ => 5,
                    };
                    shape_pattern.push(shape_class_j);
                }
                
                *shape_map.entry(shape_pattern).or_insert(0) += 1;
            }
        }
        
        // Convert to vector and sort by frequency
        let mut patterns: Vec<_> = shape_map.into_iter()
            .filter(|(_, count)| *count > 1)
            .collect();
        
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Find actual text examples of these patterns
        let mut result = Vec::new();
        for (shape_pattern, count) in patterns {
            for i in 0..=text.len() - pattern_length {
                let mut matches = true;
                for j in 0..pattern_length {
                    let shape_class = match text[i + j] {
                        b'a' | b'c' | b'e' | b'o' | b's' => 0,
                        b'i' | b'l' | b'I' | b'j' | b'f' | b't' => 1,
                        b'm' | b'n' | b'h' | b'u' => 2,
                        b'v' | b'w' | b'x' | b'y' | b'z' => 3,
                        b'b' | b'd' | b'p' | b'q' | b'g' => 4,
                        _ => 5,
                    };
                    
                    if shape_class != shape_pattern[j] {
                        matches = false;
                        break;
                    }
                }
                
                if matches {
                    let text_pattern = text[i..i+pattern_length].to_vec();
                    let score = count as f64 / (text.len() - pattern_length + 1) as f64;
                    result.push((text_pattern, score));
                    break;
                }
            }
        }
        
        result
    }
} 