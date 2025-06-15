//! Annotation module for genomic features
//!
//! This module provides genomic annotation functionality.

use std::collections::HashMap;
use super::{NucleotideSequence, Unit};

/// Annotation database
#[derive(Debug, Clone)]
pub struct AnnotationDatabase {
    /// Gene annotations
    genes: HashMap<String, Vec<GeneAnnotation>>,
    /// Functional annotations
    functions: HashMap<String, FunctionalAnnotation>,
    /// Database configuration
    config: AnnotationConfig,
}

/// Configuration for annotation
#[derive(Debug, Clone)]
pub struct AnnotationConfig {
    /// Minimum gene length
    pub min_gene_length: usize,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Enable functional annotation
    pub enable_functional: bool,
}

/// Gene annotation
#[derive(Debug, Clone)]
pub struct GeneAnnotation {
    /// Gene ID
    pub gene_id: String,
    /// Gene symbol
    pub symbol: Option<String>,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Strand
    pub strand: char,
    /// Biotype
    pub biotype: GeneType,
    /// Confidence score
    pub confidence: f64,
    /// Exons
    pub exons: Vec<ExonAnnotation>,
}

/// Exon annotation
#[derive(Debug, Clone)]
pub struct ExonAnnotation {
    /// Exon ID
    pub exon_id: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Phase
    pub phase: Option<u8>,
}

/// Functional annotation
#[derive(Debug, Clone)]
pub struct FunctionalAnnotation {
    /// Gene ID
    pub gene_id: String,
    /// Product description
    pub product: Option<String>,
    /// GO terms
    pub go_terms: Vec<GOTerm>,
    /// Pathways
    pub pathways: Vec<String>,
    /// EC numbers
    pub ec_numbers: Vec<String>,
}

/// Gene Ontology term
#[derive(Debug, Clone)]
pub struct GOTerm {
    /// GO ID
    pub go_id: String,
    /// GO term name
    pub name: String,
    /// GO namespace
    pub namespace: GONamespace,
    /// Evidence code
    pub evidence: String,
}

/// GO namespace
#[derive(Debug, Clone)]
pub enum GONamespace {
    BiologicalProcess,
    MolecularFunction,
    CellularComponent,
}

/// Gene type
#[derive(Debug, Clone)]
pub enum GeneType {
    ProteinCoding,
    NonCoding,
    Pseudogene,
    rRNA,
    tRNA,
    snRNA,
    snoRNA,
    miRNA,
    lncRNA,
}

/// Annotation result
#[derive(Debug, Clone)]
pub struct AnnotationResult {
    /// Sequence ID
    pub sequence_id: String,
    /// Gene annotations
    pub genes: Vec<GeneAnnotation>,
    /// Functional annotations
    pub functions: HashMap<String, FunctionalAnnotation>,
    /// Statistics
    pub statistics: AnnotationStatistics,
}

/// Annotation statistics
#[derive(Debug, Clone)]
pub struct AnnotationStatistics {
    /// Total genes annotated
    pub total_genes: usize,
    /// Genes by type
    pub genes_by_type: HashMap<GeneType, usize>,
    /// Average gene length
    pub avg_gene_length: f64,
    /// Coding density
    pub coding_density: f64,
}

impl Default for AnnotationConfig {
    fn default() -> Self {
        Self {
            min_gene_length: 300,
            confidence_threshold: 0.5,
            enable_functional: true,
        }
    }
}

impl AnnotationDatabase {
    /// Create a new annotation database
    pub fn new(config: AnnotationConfig) -> Self {
        Self {
            genes: HashMap::new(),
            functions: HashMap::new(),
            config,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AnnotationConfig::default())
    }

    /// Annotate a sequence
    pub fn annotate(&self, sequence: &NucleotideSequence) -> AnnotationResult {
        let sequence_id = sequence.id().to_string();
        
        // Predict genes
        let genes = self.predict_genes(sequence);
        
        // Add functional annotations
        let mut functions = HashMap::new();
        if self.config.enable_functional {
            for gene in &genes {
                if let Some(functional) = self.get_functional_annotation(&gene.gene_id) {
                    functions.insert(gene.gene_id.clone(), functional);
                }
            }
        }
        
        // Calculate statistics
        let statistics = self.calculate_statistics(&genes, sequence);
        
        AnnotationResult {
            sequence_id,
            genes,
            functions,
            statistics,
        }
    }

    /// Predict genes in a sequence
    fn predict_genes(&self, sequence: &NucleotideSequence) -> Vec<GeneAnnotation> {
        let mut genes = Vec::new();
        let content = String::from_utf8_lossy(sequence.content());
        
        // Simple gene prediction based on ORFs
        let orfs = self.find_orfs(&content);
        
        for (i, orf) in orfs.iter().enumerate() {
            if orf.length >= self.config.min_gene_length {
                let gene = GeneAnnotation {
                    gene_id: format!("gene_{}", i),
                    symbol: Some(format!("GENE{}", i)),
                    start: orf.start,
                    end: orf.end,
                    strand: if orf.frame >= 0 { '+' } else { '-' },
                    biotype: GeneType::ProteinCoding,
                    confidence: 0.8, // Simplified confidence
                    exons: vec![ExonAnnotation {
                        exon_id: format!("exon_{}_{}", i, 1),
                        start: orf.start,
                        end: orf.end,
                        phase: Some(0),
                    }],
                };
                genes.push(gene);
            }
        }
        
        genes
    }

    /// Find open reading frames
    fn find_orfs(&self, sequence: &str) -> Vec<ORF> {
        let mut orfs = Vec::new();
        let start_codons = ["ATG"];
        let stop_codons = ["TAA", "TAG", "TGA"];
        
        // Search all reading frames
        for frame in 0..3 {
            let mut pos = frame;
            
            while pos + 3 <= sequence.len() {
                let codon = &sequence[pos..pos + 3];
                
                if start_codons.contains(&codon) {
                    let start_pos = pos;
                    pos += 3;
                    
                    while pos + 3 <= sequence.len() {
                        let stop_codon = &sequence[pos..pos + 3];
                        
                        if stop_codons.contains(&stop_codon) {
                            let end_pos = pos + 3;
                            let length = end_pos - start_pos;
                            
                            orfs.push(ORF {
                                start: start_pos,
                                end: end_pos,
                                frame: frame as i8,
                                length,
                            });
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

    /// Get functional annotation for a gene
    fn get_functional_annotation(&self, gene_id: &str) -> Option<FunctionalAnnotation> {
        // Simplified functional annotation
        Some(FunctionalAnnotation {
            gene_id: gene_id.to_string(),
            product: Some("Hypothetical protein".to_string()),
            go_terms: vec![
                GOTerm {
                    go_id: "GO:0003674".to_string(),
                    name: "molecular_function".to_string(),
                    namespace: GONamespace::MolecularFunction,
                    evidence: "IEA".to_string(),
                },
            ],
            pathways: vec!["Unknown pathway".to_string()],
            ec_numbers: Vec::new(),
        })
    }

    /// Calculate annotation statistics
    fn calculate_statistics(&self, genes: &[GeneAnnotation], sequence: &NucleotideSequence) -> AnnotationStatistics {
        let total_genes = genes.len();
        let sequence_length = sequence.content().len();
        
        let mut genes_by_type = HashMap::new();
        let mut total_gene_length = 0;
        
        for gene in genes {
            *genes_by_type.entry(gene.biotype.clone()).or_insert(0) += 1;
            total_gene_length += gene.end - gene.start;
        }
        
        let avg_gene_length = if total_genes > 0 {
            total_gene_length as f64 / total_genes as f64
        } else {
            0.0
        };
        
        let coding_density = if sequence_length > 0 {
            total_gene_length as f64 / sequence_length as f64
        } else {
            0.0
        };
        
        AnnotationStatistics {
            total_genes,
            genes_by_type,
            avg_gene_length,
            coding_density,
        }
    }

    /// Add gene annotation
    pub fn add_gene_annotation(&mut self, sequence_id: &str, gene: GeneAnnotation) {
        self.genes.entry(sequence_id.to_string()).or_insert_with(Vec::new).push(gene);
    }

    /// Add functional annotation
    pub fn add_functional_annotation(&mut self, gene_id: &str, annotation: FunctionalAnnotation) {
        self.functions.insert(gene_id.to_string(), annotation);
    }

    /// Get annotations for a sequence
    pub fn get_annotations(&self, sequence_id: &str) -> Option<&Vec<GeneAnnotation>> {
        self.genes.get(sequence_id)
    }
}

/// Simple ORF structure
#[derive(Debug, Clone)]
struct ORF {
    start: usize,
    end: usize,
    frame: i8,
    length: usize,
}

// Need to implement Hash and PartialEq for GeneType to use in HashMap
impl std::hash::Hash for GeneType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
    }
}

impl PartialEq for GeneType {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

impl Eq for GeneType {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gene_annotation() {
        let db = AnnotationDatabase::default();
        let sequence = NucleotideSequence::new("ATGAAATAAATGCCCTAA".as_bytes(), "test_seq");

        let result = db.annotate(&sequence);

        assert!(!result.genes.is_empty());
        assert_eq!(result.sequence_id, "test_seq");
        assert!(result.statistics.total_genes > 0);
    }

    #[test]
    fn test_orf_finding() {
        let db = AnnotationDatabase::default();
        let sequence = "ATGAAATAA";

        let orfs = db.find_orfs(sequence);

        assert!(!orfs.is_empty());
        assert_eq!(orfs[0].start, 0);
        assert_eq!(orfs[0].end, 9);
    }
} 