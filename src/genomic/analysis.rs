//! Genomic analysis module
//!
//! This module provides high-level analysis operations for genomic data.

use std::collections::HashMap;
use super::{NucleotideSequence, sequence::SequenceAnalyzer};

/// Main genomic analysis engine
#[derive(Debug, Clone)]
pub struct GenomicAnalyzer {
    /// Sequence analyzer
    sequence_analyzer: SequenceAnalyzer,
    /// Analysis configuration
    config: AnalysisConfig,
}

/// Configuration for genomic analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable gene prediction
    pub enable_gene_prediction: bool,
    /// Enable repeat detection
    pub enable_repeat_detection: bool,
    /// Minimum gene length
    pub min_gene_length: usize,
    /// Quality thresholds
    pub quality_threshold: f64,
}

/// Result of genomic analysis
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Analysis summary
    pub summary: AnalysisSummary,
    /// Detailed results
    pub details: AnalysisDetails,
}

/// Summary of analysis
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// Total sequences analyzed
    pub total_sequences: usize,
    /// Total base pairs
    pub total_base_pairs: usize,
    /// Average GC content
    pub average_gc_content: f64,
    /// Number of genes predicted
    pub predicted_genes: usize,
    /// Analysis quality score
    pub quality_score: f64,
}

/// Detailed analysis results
#[derive(Debug, Clone)]
pub struct AnalysisDetails {
    /// Per-sequence results
    pub sequence_results: Vec<SequenceResult>,
    /// Global statistics
    pub global_stats: GlobalStatistics,
}

/// Result for a single sequence
#[derive(Debug, Clone)]
pub struct SequenceResult {
    /// Sequence identifier
    pub sequence_id: String,
    /// Length
    pub length: usize,
    /// GC content
    pub gc_content: f64,
    /// Predicted features
    pub features: Vec<GenomicFeature>,
}

/// A genomic feature
#[derive(Debug, Clone)]
pub struct GenomicFeature {
    /// Feature type
    pub feature_type: FeatureType,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
    /// Feature attributes
    pub attributes: HashMap<String, String>,
}

/// Type of genomic feature
#[derive(Debug, Clone)]
pub enum FeatureType {
    /// Gene
    Gene,
    /// Exon
    Exon,
    /// Intron
    Intron,
    /// Repeat element
    Repeat,
    /// Regulatory element
    Regulatory,
}

/// Global statistics across all sequences
#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    /// Base composition
    pub base_composition: HashMap<char, usize>,
    /// K-mer distribution
    pub kmer_distribution: HashMap<String, usize>,
    /// Feature density
    pub feature_density: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_gene_prediction: true,
            enable_repeat_detection: true,
            min_gene_length: 300,
            quality_threshold: 0.7,
        }
    }
}

impl GenomicAnalyzer {
    /// Create a new genomic analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            sequence_analyzer: SequenceAnalyzer::default(),
            config,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AnalysisConfig::default())
    }

    /// Analyze a set of sequences
    pub fn analyze(&self, sequences: &[NucleotideSequence]) -> AnalysisResult {
        let mut sequence_results = Vec::new();
        let mut total_length = 0;
        let mut total_gc = 0.0;
        let mut total_genes = 0;

        // Analyze each sequence
        for sequence in sequences {
            let seq_analysis = self.sequence_analyzer.analyze(sequence);
            let features = self.predict_features(sequence);
            
            let genes_in_seq = features.iter()
                .filter(|f| matches!(f.feature_type, FeatureType::Gene))
                .count();
                
            total_genes += genes_in_seq;
            total_length += seq_analysis.length;
            total_gc += seq_analysis.gc_content * seq_analysis.length as f64;

            sequence_results.push(SequenceResult {
                sequence_id: seq_analysis.sequence_id,
                length: seq_analysis.length,
                gc_content: seq_analysis.gc_content,
                features,
            });
        }

        // Calculate summary statistics
        let average_gc_content = if total_length > 0 {
            total_gc / total_length as f64
        } else {
            0.0
        };

        let summary = AnalysisSummary {
            total_sequences: sequences.len(),
            total_base_pairs: total_length,
            average_gc_content,
            predicted_genes: total_genes,
            quality_score: self.calculate_quality_score(&sequence_results),
        };

        // Calculate global statistics
        let global_stats = self.calculate_global_stats(sequences);

        AnalysisResult {
            summary,
            details: AnalysisDetails {
                sequence_results,
                global_stats,
            },
        }
    }

    /// Predict genomic features in a sequence
    fn predict_features(&self, sequence: &NucleotideSequence) -> Vec<GenomicFeature> {
        let mut features = Vec::new();

        // Gene prediction
        if self.config.enable_gene_prediction {
            let orfs = self.sequence_analyzer.find_orfs(sequence, self.config.min_gene_length);
            for orf in orfs {
                features.push(GenomicFeature {
                    feature_type: FeatureType::Gene,
                    start: orf.start,
                    end: orf.end,
                    confidence: 0.8, // Simplified confidence score
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("frame".to_string(), orf.frame.to_string());
                        attrs.insert("length".to_string(), orf.length.to_string());
                        attrs
                    },
                });
            }
        }

        // Repeat detection
        if self.config.enable_repeat_detection {
            let repeats = self.sequence_analyzer.find_repeats(sequence, 10);
            for repeat in repeats {
                features.push(GenomicFeature {
                    feature_type: FeatureType::Repeat,
                    start: repeat.start,
                    end: repeat.end,
                    confidence: 0.7,
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("pattern".to_string(), repeat.pattern);
                        attrs.insert("count".to_string(), repeat.repeat_count.to_string());
                        attrs
                    },
                });
            }
        }

        features
    }

    /// Calculate overall quality score
    fn calculate_quality_score(&self, results: &[SequenceResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        let total_score: f64 = results.iter().map(|r| {
            // Simple quality metric based on feature density and GC content
            let feature_density = r.features.len() as f64 / r.length as f64 * 1000.0;
            let gc_quality = 1.0 - (r.gc_content - 0.5).abs() * 2.0; // Prefer ~50% GC
            (feature_density + gc_quality) / 2.0
        }).sum();

        total_score / results.len() as f64
    }

    /// Calculate global statistics
    fn calculate_global_stats(&self, sequences: &[NucleotideSequence]) -> GlobalStatistics {
        let mut base_composition = HashMap::new();
        let mut kmer_distribution = HashMap::new();
        let mut total_features = 0;
        let mut total_length = 0;

        for sequence in sequences {
            let analysis = self.sequence_analyzer.analyze(sequence);
            
            // Aggregate base composition
            for (base, count) in analysis.base_composition {
                *base_composition.entry(base).or_insert(0) += count;
            }
            
            // Aggregate k-mer distribution
            for (kmer, count) in analysis.kmer_frequencies {
                *kmer_distribution.entry(kmer).or_insert(0) += count;
            }
            
            // Count features
            let features = self.predict_features(sequence);
            total_features += features.len();
            total_length += analysis.length;
        }

        let feature_density = if total_length > 0 {
            total_features as f64 / total_length as f64 * 1000.0
        } else {
            0.0
        };

        GlobalStatistics {
            base_composition,
            kmer_distribution,
            feature_density,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genomic_analysis() {
        let analyzer = GenomicAnalyzer::default();
        let sequences = vec![
            NucleotideSequence::new("ATGAAATAA".as_bytes(), "seq1"),
            NucleotideSequence::new("GCTAGCTAGC".as_bytes(), "seq2"),
        ];

        let result = analyzer.analyze(&sequences);

        assert_eq!(result.summary.total_sequences, 2);
        assert!(result.summary.total_base_pairs > 0);
        assert!(!result.details.sequence_results.is_empty());
    }

    #[test]
    fn test_feature_prediction() {
        let analyzer = GenomicAnalyzer::default();
        let sequence = NucleotideSequence::new("ATGAAATAA".as_bytes(), "test_seq");

        let features = analyzer.predict_features(&sequence);

        // Should find at least one gene (ORF)
        assert!(!features.is_empty());
        assert!(features.iter().any(|f| matches!(f.feature_type, FeatureType::Gene)));
    }
} 