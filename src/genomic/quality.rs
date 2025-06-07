//! Quality control module for genomic data
//!
//! This module provides quality assessment and filtering for genomic sequences.

use super::NucleotideSequence;
use std::collections::HashMap;

/// Quality control system
#[derive(Debug, Clone)]
pub struct QualityControl {
    /// Quality thresholds
    thresholds: QualityThresholds,
}

/// Quality thresholds
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Minimum sequence quality score
    pub min_sequence_quality: f64,
    /// Minimum sequence length
    pub min_length: usize,
    /// Maximum N content (fraction)
    pub max_n_content: f64,
    /// Minimum GC content
    pub min_gc_content: f64,
    /// Maximum GC content
    pub max_gc_content: f64,
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// Sequence ID
    pub sequence_id: String,
    /// Overall quality score
    pub overall_quality: f64,
    /// Individual quality metrics
    pub metrics: QualityMetrics,
    /// Pass/fail status
    pub passed: bool,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Sequence length
    pub length: usize,
    /// GC content
    pub gc_content: f64,
    /// N content (fraction of unknown bases)
    pub n_content: f64,
    /// Complexity score
    pub complexity: f64,
    /// Base composition
    pub base_composition: HashMap<char, usize>,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_sequence_quality: 20.0,
            min_length: 50,
            max_n_content: 0.1,
            min_gc_content: 0.2,
            max_gc_content: 0.8,
        }
    }
}

impl QualityControl {
    /// Create a new quality control system
    pub fn new(thresholds: QualityThresholds) -> Self {
        Self { thresholds }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(QualityThresholds::default())
    }

    /// Assess quality of a single sequence
    pub fn assess_sequence(&self, sequence: &NucleotideSequence) -> QualityAssessment {
        let metrics = self.calculate_metrics(sequence);
        let overall_quality = self.calculate_overall_quality(&metrics);
        let passed = overall_quality >= self.thresholds.min_sequence_quality;

        QualityAssessment {
            sequence_id: sequence.id().to_string(),
            overall_quality,
            metrics,
            passed,
        }
    }

    /// Calculate quality metrics for a sequence
    fn calculate_metrics(&self, sequence: &NucleotideSequence) -> QualityMetrics {
        let content = String::from_utf8_lossy(sequence.content());
        let length = content.len();
        
        // Base composition
        let mut base_composition = HashMap::new();
        for base in content.chars() {
            *base_composition.entry(base.to_ascii_uppercase()).or_insert(0) += 1;
        }
        
        // GC content
        let gc_count = base_composition.get(&'G').unwrap_or(&0) + 
                      base_composition.get(&'C').unwrap_or(&0);
        let gc_content = if length > 0 { gc_count as f64 / length as f64 } else { 0.0 };
        
        // N content
        let n_count = base_composition.get(&'N').unwrap_or(&0);
        let n_content = if length > 0 { *n_count as f64 / length as f64 } else { 0.0 };
        
        // Complexity (Shannon entropy)
        let complexity = self.calculate_complexity(&base_composition, length);
        
        QualityMetrics {
            length,
            gc_content,
            n_content,
            complexity,
            base_composition,
        }
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
        
        // Normalize to 0-1 range (maximum entropy for 4 bases is 2.0)
        entropy / 2.0
    }

    /// Calculate overall quality score
    fn calculate_overall_quality(&self, metrics: &QualityMetrics) -> f64 {
        let mut quality = 100.0;
        
        // Length check
        if metrics.length < self.thresholds.min_length {
            quality -= 30.0;
        }
        
        // N content check
        if metrics.n_content > self.thresholds.max_n_content {
            quality -= 25.0;
        }
        
        // GC content check
        if metrics.gc_content < self.thresholds.min_gc_content || 
           metrics.gc_content > self.thresholds.max_gc_content {
            quality -= 15.0;
        }
        
        // Reward high complexity
        quality += metrics.complexity * 10.0;
        
        quality.max(0.0).min(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_assessment() {
        let qc = QualityControl::default();
        let sequence = NucleotideSequence::new("ATCGATCGATCGATCG".as_bytes(), "test_seq");

        let assessment = qc.assess_sequence(&sequence);

        assert_eq!(assessment.sequence_id, "test_seq");
        assert!(assessment.overall_quality > 0.0);
        assert!(!assessment.metrics.base_composition.is_empty());
    }
} 