//! Sequence analysis module for genomic data
//!
//! This module provides sequence analysis operations including
//! composition analysis, motif finding, and sequence statistics.

use std::collections::HashMap;
use super::{NucleotideSequence, GenomicMetadata, UnitId, Strand, Position, Unit};

/// Sequence analyzer for genomic sequences
#[derive(Debug, Clone)]
pub struct SequenceAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
}

/// Configuration for sequence analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Minimum sequence length to analyze
    pub min_length: usize,
    /// K-mer size for analysis
    pub kmer_size: usize,
    /// Window size for sliding window analysis
    pub window_size: usize,
    /// Step size for windows
    pub step_size: usize,
}

/// Result of sequence analysis
#[derive(Debug, Clone)]
pub struct SequenceAnalysisResult {
    /// Sequence being analyzed
    pub sequence_id: String,
    /// Base composition
    pub base_composition: HashMap<char, usize>,
    /// GC content
    pub gc_content: f64,
    /// Length
    pub length: usize,
    /// K-mer frequencies
    pub kmer_frequencies: HashMap<String, usize>,
    /// Complexity score
    pub complexity: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_length: 1,
            kmer_size: 3,
            window_size: 100,
            step_size: 50,
        }
    }
}

impl SequenceAnalyzer {
    /// Create a new sequence analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AnalysisConfig::default())
    }

    /// Analyze a nucleotide sequence
    pub fn analyze(&self, sequence: &NucleotideSequence) -> SequenceAnalysisResult {
        let content_str = String::from_utf8_lossy(sequence.content());
        let length = content_str.len();
        
        // Base composition
        let mut base_composition = HashMap::new();
        for base in content_str.chars() {
            *base_composition.entry(base).or_insert(0) += 1;
        }
        
        // GC content
        let gc_count = base_composition.get(&'G').unwrap_or(&0) + 
                      base_composition.get(&'C').unwrap_or(&0);
        let gc_content = if length > 0 { gc_count as f64 / length as f64 } else { 0.0 };
        
        // K-mer frequencies
        let kmer_frequencies = self.calculate_kmer_frequencies(&content_str);
        
        // Complexity score (simplified Shannon entropy)
        let complexity = self.calculate_complexity(&base_composition, length);
        
        SequenceAnalysisResult {
            sequence_id: sequence.id().to_string(),
            base_composition,
            gc_content,
            length,
            kmer_frequencies,
            complexity,
        }
    }

    /// Calculate k-mer frequencies
    fn calculate_kmer_frequencies(&self, sequence: &str) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        
        if sequence.len() < self.config.kmer_size {
            return frequencies;
        }
        
        for i in 0..=(sequence.len() - self.config.kmer_size) {
            let kmer = &sequence[i..i + self.config.kmer_size];
            *frequencies.entry(kmer.to_uppercase()).or_insert(0) += 1;
        }
        
        frequencies
    }

    /// Calculate sequence complexity using Shannon entropy
    fn calculate_complexity(&self, composition: &HashMap<char, usize>, length: usize) -> f64 {
        if length == 0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &count in composition.values() {
            if count > 0 {
                let probability = count as f64 / length as f64;
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }

    /// Find repeats in sequence
    pub fn find_repeats(&self, sequence: &NucleotideSequence, min_repeat_length: usize) -> Vec<RepeatRegion> {
        let content_str = String::from_utf8_lossy(sequence.content());
        let mut repeats = Vec::new();
        
        // Simple tandem repeat detection
        for i in 0..content_str.len() {
            for repeat_len in min_repeat_length..=((content_str.len() - i) / 2) {
                let pattern = &content_str[i..i + repeat_len];
                let mut repeat_count = 1;
                let mut pos = i + repeat_len;
                
                while pos + repeat_len <= content_str.len() {
                    if &content_str[pos..pos + repeat_len] == pattern {
                        repeat_count += 1;
                        pos += repeat_len;
                    } else {
                        break;
                    }
                }
                
                if repeat_count >= 2 {
                    repeats.push(RepeatRegion {
                        start: i,
                        end: pos,
                        pattern: pattern.to_string(),
                        repeat_count,
                        repeat_type: RepeatType::Tandem,
                    });
                    break; // Found a repeat starting at this position
                }
            }
        }
        
        repeats
    }

    /// Predict open reading frames (ORFs)
    pub fn find_orfs(&self, sequence: &NucleotideSequence, min_length: usize) -> Vec<OpenReadingFrame> {
        let content_str = String::from_utf8_lossy(sequence.content());
        let mut orfs = Vec::new();
        
        // Start codons
        let start_codons = ["ATG"];
        // Stop codons
        let stop_codons = ["TAA", "TAG", "TGA"];
        
        // Search all reading frames
        for frame in 0..3 {
            let mut pos = frame;
            
            while pos + 3 <= content_str.len() {
                let codon = &content_str[pos..pos + 3];
                
                if start_codons.contains(&codon) {
                    // Found start codon, look for stop codon
                    let start_pos = pos;
                    pos += 3;
                    
                    while pos + 3 <= content_str.len() {
                        let stop_codon = &content_str[pos..pos + 3];
                        
                        if stop_codons.contains(&stop_codon) {
                            let end_pos = pos + 3;
                            let length = end_pos - start_pos;
                            
                            if length >= min_length {
                                orfs.push(OpenReadingFrame {
                                    start: start_pos,
                                    end: end_pos,
                                    frame: frame as i8,
                                    length,
                                    sequence: content_str[start_pos..end_pos].to_string(),
                                });
                            }
                            break;
                        }
                        pos += 3;
                    }
                } else {
                    pos += 3;
                }
            }
        }
        
        orfs
    }
}

/// Represents a repeat region in a sequence
#[derive(Debug, Clone)]
pub struct RepeatRegion {
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Repeat pattern
    pub pattern: String,
    /// Number of repeats
    pub repeat_count: usize,
    /// Type of repeat
    pub repeat_type: RepeatType,
}

/// Type of repeat
#[derive(Debug, Clone)]
pub enum RepeatType {
    /// Tandem repeat
    Tandem,
    /// Interspersed repeat
    Interspersed,
    /// Simple sequence repeat
    SimpleSequence,
}

/// Represents an open reading frame
#[derive(Debug, Clone)]
pub struct OpenReadingFrame {
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Reading frame (0, 1, or 2)
    pub frame: i8,
    /// Length in base pairs
    pub length: usize,
    /// ORF sequence
    pub sequence: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_analysis() {
        let analyzer = SequenceAnalyzer::default();
        let sequence = NucleotideSequence::new("ATGCGATCGATCG".as_bytes(), "test_seq");
        
        let result = analyzer.analyze(&sequence);
        
        assert_eq!(result.length, 13);
        assert!(result.gc_content > 0.0);
        assert!(!result.kmer_frequencies.is_empty());
    }

    #[test]
    fn test_orf_finding() {
        let analyzer = SequenceAnalyzer::default();
        let sequence = NucleotideSequence::new("ATGAAATAA".as_bytes(), "test_orf");
        
        let orfs = analyzer.find_orfs(&sequence, 6);
        
        assert_eq!(orfs.len(), 1);
        assert_eq!(orfs[0].start, 0);
        assert_eq!(orfs[0].end, 9);
    }
} 