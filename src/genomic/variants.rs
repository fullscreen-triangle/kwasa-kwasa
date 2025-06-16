//! Variant analysis module for genomic data
//!
//! This module provides variant calling and analysis functionality.

use std::collections::HashMap;
use super::{NucleotideSequence, Unit};

/// Variant caller for genomic sequences
#[derive(Debug, Clone)]
pub struct VariantCaller {
    /// Configuration
    config: VariantConfig,
}

/// Configuration for variant calling
#[derive(Debug, Clone)]
pub struct VariantConfig {
    /// Minimum quality score
    pub min_quality: f64,
    /// Minimum coverage depth
    pub min_coverage: u32,
    /// Minimum allele frequency
    pub min_allele_frequency: f64,
}

/// A genomic variant
#[derive(Debug, Clone)]
pub struct Variant {
    /// Chromosome/contig
    pub chromosome: String,
    /// Position (0-based)
    pub position: usize,
    /// Reference allele
    pub reference: String,
    /// Alternative allele
    pub alternative: String,
    /// Variant type
    pub variant_type: VariantType,
    /// Quality score
    pub quality: f64,
    /// Coverage depth
    pub depth: u32,
    /// Allele frequency
    pub allele_frequency: f64,
}

/// Type of genomic variant
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VariantType {
    /// Single nucleotide polymorphism
    SNP,
    /// Insertion
    Insertion,
    /// Deletion
    Deletion,
    /// Complex structural variant
    Complex,
}

/// Result of variant calling
#[derive(Debug, Clone)]
pub struct VariantCallResult {
    /// List of variants found
    pub variants: Vec<Variant>,
    /// Statistics
    pub statistics: VariantStatistics,
}

/// Statistics about variant calling
#[derive(Debug, Clone)]
pub struct VariantStatistics {
    /// Total variants called
    pub total_variants: usize,
    /// Variants by type
    pub variants_by_type: HashMap<VariantType, usize>,
    /// Average quality
    pub average_quality: f64,
    /// Coverage statistics
    pub coverage_stats: CoverageStats,
}

/// Coverage statistics
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Mean coverage
    pub mean_coverage: f64,
    /// Median coverage
    pub median_coverage: f64,
    /// Coverage standard deviation
    pub coverage_std: f64,
}

impl Default for VariantConfig {
    fn default() -> Self {
        Self {
            min_quality: 20.0,
            min_coverage: 10,
            min_allele_frequency: 0.05,
        }
    }
}

impl VariantCaller {
    /// Create a new variant caller
    pub fn new(config: VariantConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(VariantConfig::default())
    }

    /// Call variants between reference and sample sequences
    pub fn call_variants(
        &self,
        reference: &NucleotideSequence,
        sample: &NucleotideSequence,
    ) -> VariantCallResult {
        let variants = self.find_variants(reference, sample);
        let statistics = self.calculate_statistics(&variants);

        VariantCallResult {
            variants,
            statistics,
        }
    }

    /// Find variants by comparing sequences
    fn find_variants(
        &self,
        reference: &NucleotideSequence,
        sample: &NucleotideSequence,
    ) -> Vec<Variant> {
        let mut variants = Vec::new();
        
        let ref_content = String::from_utf8_lossy(reference.content());
        let sample_content = String::from_utf8_lossy(sample.content());
        
        let ref_chars: Vec<char> = ref_content.chars().collect();
        let sample_chars: Vec<char> = sample_content.chars().collect();
        
        let min_len = ref_chars.len().min(sample_chars.len());
        
        // Simple variant detection by position comparison
        for i in 0..min_len {
            if ref_chars[i] != sample_chars[i] {
                let variant = Variant {
                    chromosome: reference.id().to_string(),
                    position: i,
                    reference: ref_chars[i].to_string(),
                    alternative: sample_chars[i].to_string(),
                    variant_type: VariantType::SNP,
                    quality: 30.0, // Simplified quality score
                    depth: 20, // Simplified depth
                    allele_frequency: 1.0, // Simplified frequency
                };
                
                if variant.quality >= self.config.min_quality 
                   && variant.depth >= self.config.min_coverage 
                   && variant.allele_frequency >= self.config.min_allele_frequency {
                    variants.push(variant);
                }
            }
        }
        
        // Handle length differences as indels
        if ref_chars.len() != sample_chars.len() {
            let (variant_type, position) = if ref_chars.len() > sample_chars.len() {
                (VariantType::Deletion, sample_chars.len())
            } else {
                (VariantType::Insertion, ref_chars.len())
            };
            
            let variant = Variant {
                chromosome: reference.id().to_string(),
                position,
                reference: if variant_type == VariantType::Deletion {
                    ref_chars[position..].iter().collect()
                } else {
                    "-".to_string()
                },
                alternative: if variant_type == VariantType::Insertion {
                    sample_chars[position..].iter().collect()
                } else {
                    "-".to_string()
                },
                variant_type,
                quality: 25.0,
                depth: 15,
                allele_frequency: 1.0,
            };
            
            if variant.quality >= self.config.min_quality 
               && variant.depth >= self.config.min_coverage 
               && variant.allele_frequency >= self.config.min_allele_frequency {
                variants.push(variant);
            }
        }
        
        variants
    }

    /// Calculate statistics for variant calls
    fn calculate_statistics(&self, variants: &[Variant]) -> VariantStatistics {
        let total_variants = variants.len();
        
        let mut variants_by_type = HashMap::new();
        let mut total_quality = 0.0;
        let mut total_depth = 0;
        
        for variant in variants {
            *variants_by_type.entry(variant.variant_type.clone()).or_insert(0) += 1;
            total_quality += variant.quality;
            total_depth += variant.depth;
        }
        
        let average_quality = if total_variants > 0 {
            total_quality / total_variants as f64
        } else {
            0.0
        };
        
        let mean_coverage = if total_variants > 0 {
            total_depth as f64 / total_variants as f64
        } else {
            0.0
        };
        
        VariantStatistics {
            total_variants,
            variants_by_type,
            average_quality,
            coverage_stats: CoverageStats {
                mean_coverage,
                median_coverage: mean_coverage, // Simplified
                coverage_std: 5.0, // Simplified
            },
        }
    }

    /// Filter variants based on quality criteria
    pub fn filter_variants(&self, variants: Vec<Variant>) -> Vec<Variant> {
        variants
            .into_iter()
            .filter(|v| {
                v.quality >= self.config.min_quality
                    && v.depth >= self.config.min_coverage
                    && v.allele_frequency >= self.config.min_allele_frequency
            })
            .collect()
    }

    /// Annotate variants with functional impact
    pub fn annotate_variants(&self, variants: &mut [Variant]) {
        // Simplified annotation - in a real implementation, 
        // this would use databases and prediction tools
        for _variant in variants.iter_mut() {
            // Add functional annotations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_calling() {
        let caller = VariantCaller::default();
        let reference = NucleotideSequence::new("ATCGATCG".as_bytes(), "ref");
        let sample = NucleotideSequence::new("ATGGATCG".as_bytes(), "sample");

        let result = caller.call_variants(&reference, &sample);

        // Should find at least one SNP (C->G at position 2)
        assert!(!result.variants.is_empty());
        assert!(result.variants.iter().any(|v| v.variant_type == VariantType::SNP));
    }

    #[test]
    fn test_variant_filtering() {
        let caller = VariantCaller::default();
        let variants = vec![
            Variant {
                chromosome: "chr1".to_string(),
                position: 100,
                reference: "A".to_string(),
                alternative: "T".to_string(),
                variant_type: VariantType::SNP,
                quality: 30.0,
                depth: 20,
                allele_frequency: 0.8,
            },
            Variant {
                chromosome: "chr1".to_string(),
                position: 200,
                reference: "G".to_string(),
                alternative: "C".to_string(),
                variant_type: VariantType::SNP,
                quality: 10.0, // Below threshold
                depth: 5, // Below threshold
                allele_frequency: 0.02, // Below threshold
            },
        ];

        let filtered = caller.filter_variants(variants);

        // Should only keep the high-quality variant
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].position, 100);
    }
} 