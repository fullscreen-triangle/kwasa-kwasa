//! Genomic extension for Kwasa-Kwasa
//! 
//! This module provides types and operations for working with genomic sequences
//! using the same powerful abstractions as text processing.

use std::fmt::Debug;
use std::{collections::HashMap, marker::PhantomData};

// Add the high-throughput module
pub mod high_throughput;

/// Re-exports from this module
pub mod prelude {
    pub use super::{
        NucleotideSequence, CodonUnit, GeneUnit, MotifUnit, ExonUnit, IntronUnit,
        GenomicBoundaryDetector, GenomicOperations,
        // Add exports for high-throughput components
        high_throughput::{
            HighThroughputGenomics, SequenceCompressor, CompressedSequence
        },
    };
}

/// Metadata for genomic units
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct GenomicMetadata {
    /// Source of the sequence (e.g., organism, database)
    pub source: Option<String>,
    /// Strand information (+/-)
    pub strand: Strand,
    /// Position in the genome
    pub position: Option<Position>,
    /// Additional key-value annotations
    pub annotations: HashMap<String, String>,
}

/// Strand direction
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum Strand {
    /// Forward strand (+)
    Forward,
    /// Reverse strand (-)
    Reverse,
    /// Unknown strand
    Unknown,
}

/// Position in a reference sequence
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Position {
    /// Starting position (0-based)
    pub start: usize,
    /// Ending position (0-based, exclusive)
    pub end: usize,
    /// Reference sequence identifier
    pub reference: String,
}

/// Unique identifier for genomic units
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct UnitId(String);

impl UnitId {
    /// Create a new unit ID
    pub fn new(id: impl Into<String>) -> Self {
        UnitId(id.into())
    }
}

impl std::fmt::Display for UnitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Universal trait for all units of analysis
pub trait Unit: Clone + Debug {
    /// The raw content of this unit
    fn content(&self) -> &[u8];
    
    /// Human-readable representation
    fn display(&self) -> String;
    
    /// Metadata associated with this unit
    fn metadata(&self) -> &dyn std::any::Any;
    
    /// Unique identifier for this unit
    fn id(&self) -> &UnitId;
}

/// Configuration for boundary detection
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Minimum unit size
    pub min_size: usize,
    /// Maximum unit size
    pub max_size: Option<usize>,
    /// Whether to include partial units at the ends
    pub include_partial: bool,
    /// Pattern to use for splitting
    pub pattern: Option<String>,
}

/// Generic trait for boundary detection in any domain
pub trait BoundaryDetector {
    type UnitType: Unit;
    
    /// Detect boundaries in the given content
    fn detect_boundaries(&self, content: &[u8]) -> Vec<Self::UnitType>;
    
    /// Configuration for the detection algorithm
    fn configuration(&self) -> &BoundaryConfig;
}

/// Generic operations on units
pub trait UnitOperations<T: Unit> {
    /// Split a unit into smaller units based on a pattern
    fn divide(&self, unit: &T, pattern: &str) -> Vec<T>;
    
    /// Combine two units with appropriate transitions
    fn multiply(&self, left: &T, right: &T) -> T;
    
    /// Concatenate units with intelligent joining
    fn add(&self, left: &T, right: &T) -> T;
    
    /// Remove elements from a unit
    fn subtract(&self, source: &T, to_remove: &T) -> T;
}

//------------------------------------------------------------------------------
// Nucleotide Sequence
//------------------------------------------------------------------------------

/// A sequence of nucleotides (DNA or RNA)
#[derive(Debug, Clone)]
pub struct NucleotideSequence {
    /// Raw sequence data
    content: Vec<u8>,
    /// Metadata for this sequence
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
}

impl NucleotideSequence {
    /// Create a new nucleotide sequence
    pub fn new(content: impl Into<Vec<u8>>, id: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            content,
            metadata: GenomicMetadata {
                source: None,
                strand: Strand::Unknown,
                position: None,
                annotations: HashMap::new(),
            },
            id: UnitId::new(id),
        }
    }
    
    /// Get the GC content of this sequence
    pub fn gc_content(&self) -> f64 {
        let gc_count = self.content.iter().filter(|&b| *b == b'G' || *b == b'C').count();
        gc_count as f64 / self.content.len() as f64
    }
    
    /// Get the reverse complement of this sequence
    pub fn reverse_complement(&self) -> Self {
        let mut result = Vec::with_capacity(self.content.len());
        
        for &nucleotide in self.content.iter().rev() {
            let complement = match nucleotide {
                b'A' => b'T',
                b'T' => b'A',
                b'G' => b'C',
                b'C' => b'G',
                b'U' => b'A',
                b'N' => b'N',
                _ => nucleotide,
            };
            
            result.push(complement);
        }
        
        let mut new_metadata = self.metadata.clone();
        new_metadata.strand = match self.metadata.strand {
            Strand::Forward => Strand::Reverse,
            Strand::Reverse => Strand::Forward,
            Strand::Unknown => Strand::Unknown,
        };
        
        Self {
            content: result,
            metadata: new_metadata,
            id: UnitId::new(format!("{}_revcomp", self.id.0)),
        }
    }
    
    /// Translate this DNA sequence to a protein sequence
    pub fn translate(&self) -> Vec<u8> {
        let mut result = Vec::new();
        
        // Fix the codon table HashMap collection with appropriate type conversion
        let codon_table: HashMap<Vec<u8>, u8> = [
            (b"TTT".to_vec(), b'F'), (b"TTC".to_vec(), b'F'), (b"TTA".to_vec(), b'L'), (b"TTG".to_vec(), b'L'),
            (b"CTT".to_vec(), b'L'), (b"CTC".to_vec(), b'L'), (b"CTA".to_vec(), b'L'), (b"CTG".to_vec(), b'L'),
            (b"ATT".to_vec(), b'I'), (b"ATC".to_vec(), b'I'), (b"ATA".to_vec(), b'I'), (b"ATG".to_vec(), b'M'),
            (b"GTT".to_vec(), b'V'), (b"GTC".to_vec(), b'V'), (b"GTA".to_vec(), b'V'), (b"GTG".to_vec(), b'V'),
            (b"TCT".to_vec(), b'S'), (b"TCC".to_vec(), b'S'), (b"TCA".to_vec(), b'S'), (b"TCG".to_vec(), b'S'),
            (b"CCT".to_vec(), b'P'), (b"CCC".to_vec(), b'P'), (b"CCA".to_vec(), b'P'), (b"CCG".to_vec(), b'P'),
            (b"ACT".to_vec(), b'T'), (b"ACC".to_vec(), b'T'), (b"ACA".to_vec(), b'T'), (b"ACG".to_vec(), b'T'),
            (b"GCT".to_vec(), b'A'), (b"GCC".to_vec(), b'A'), (b"GCA".to_vec(), b'A'), (b"GCG".to_vec(), b'A'),
            (b"TAT".to_vec(), b'Y'), (b"TAC".to_vec(), b'Y'), (b"TAA".to_vec(), b'*'), (b"TAG".to_vec(), b'*'),
            (b"CAT".to_vec(), b'H'), (b"CAC".to_vec(), b'H'), (b"CAA".to_vec(), b'Q'), (b"CAG".to_vec(), b'Q'),
            (b"AAT".to_vec(), b'N'), (b"AAC".to_vec(), b'N'), (b"AAA".to_vec(), b'K'), (b"AAG".to_vec(), b'K'),
            (b"GAT".to_vec(), b'D'), (b"GAC".to_vec(), b'D'), (b"GAA".to_vec(), b'E'), (b"GAG".to_vec(), b'E'),
            (b"TGT".to_vec(), b'C'), (b"TGC".to_vec(), b'C'), (b"TGA".to_vec(), b'*'), (b"TGG".to_vec(), b'W'),
            (b"CGT".to_vec(), b'R'), (b"CGC".to_vec(), b'R'), (b"CGA".to_vec(), b'R'), (b"CGG".to_vec(), b'R'),
            (b"AGT".to_vec(), b'S'), (b"AGC".to_vec(), b'S'), (b"AGA".to_vec(), b'R'), (b"AGG".to_vec(), b'R'),
            (b"GGT".to_vec(), b'G'), (b"GGC".to_vec(), b'G'), (b"GGA".to_vec(), b'G'), (b"GGG".to_vec(), b'G'),
        ].iter().cloned().collect();
        
        // Translate each codon
        for i in (0..self.content.len()).step_by(3) {
            if i + 3 <= self.content.len() {
                let codon = self.content[i..i+3].to_vec();
                if let Some(&amino_acid) = codon_table.get(&codon) {
                    // Stop at stop codon
                    if amino_acid == b'*' {
                        break;
                    }
                    result.push(amino_acid);
                } else {
                    // Unknown codon
                    result.push(b'X');
                }
            }
        }
        
        result
    }
    
    /// Set metadata for this sequence
    pub fn set_metadata(&mut self, metadata: GenomicMetadata) {
        self.metadata = metadata;
    }
    
    /// Get metadata for this sequence
    pub fn metadata(&self) -> &GenomicMetadata {
        &self.metadata
    }
}

impl Unit for NucleotideSequence {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        String::from_utf8_lossy(&self.content).to_string()
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Codon Unit
//------------------------------------------------------------------------------

/// A codon unit (triplet of nucleotides)
#[derive(Debug, Clone)]
pub struct CodonUnit {
    /// The codon sequence (three nucleotides)
    content: [u8; 3],
    /// Metadata for this codon
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
}

impl Unit for CodonUnit {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        String::from_utf8_lossy(&self.content).to_string()
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Gene Unit
//------------------------------------------------------------------------------

/// A gene unit (named sequence region)
#[derive(Debug, Clone)]
pub struct GeneUnit {
    /// The gene sequence
    content: Vec<u8>,
    /// Metadata for this gene
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
    /// Gene name
    name: String,
    /// Exons in this gene
    exons: Vec<ExonUnit>,
    /// Introns in this gene
    introns: Vec<IntronUnit>,
}

impl Unit for GeneUnit {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        format!("Gene: {} ({}bp)", self.name, self.content.len())
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Motif Unit
//------------------------------------------------------------------------------

/// A recurring pattern unit
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct MotifUnit {
    /// The motif sequence
    content: Vec<u8>,
    /// Metadata for this motif
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
    /// Motif name
    name: Option<String>,
    /// Position weight matrix
    pwm: Option<Vec<[f64; 4]>>,
}

impl MotifUnit {
    /// Create a new motif unit from a nucleotide sequence
    pub fn new(sequence: NucleotideSequence) -> Self {
        Self {
            content: sequence.content().to_vec(),
            metadata: sequence.metadata().clone(),
            id: sequence.id().clone(),
            name: None,
            pwm: None,
        }
    }
}

impl Unit for MotifUnit {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        if let Some(ref name) = self.name {
            format!("Motif: {} ({}bp)", name, self.content.len())
        } else {
            format!("Motif: ({}bp)", self.content.len())
        }
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Exon and Intron Units
//------------------------------------------------------------------------------

/// A coding region unit
#[derive(Debug, Clone)]
pub struct ExonUnit {
    /// The exon sequence
    content: Vec<u8>,
    /// Metadata for this exon
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
    /// Exon number
    number: usize,
}

impl Unit for ExonUnit {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        format!("Exon {} ({}bp)", self.number, self.content.len())
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

/// A non-coding region unit
#[derive(Debug, Clone)]
pub struct IntronUnit {
    /// The intron sequence
    content: Vec<u8>,
    /// Metadata for this intron
    metadata: GenomicMetadata,
    /// Unique identifier
    id: UnitId,
    /// Intron number
    number: usize,
}

impl Unit for IntronUnit {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        format!("Intron {} ({}bp)", self.number, self.content.len())
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Genomic Boundary Detector
//------------------------------------------------------------------------------

/// Detector for genomic sequence boundaries
#[derive(Debug)]
pub struct GenomicBoundaryDetector {
    /// Configuration for boundary detection
    config: BoundaryConfig,
    /// Type of boundaries to detect
    boundary_type: GenomicBoundaryType,
}

/// Types of genomic boundaries to detect
#[derive(Debug, Clone)]
pub enum GenomicBoundaryType {
    /// Nucleotide-level boundaries
    Nucleotide,
    /// Codon-level boundaries
    Codon,
    /// Gene-level boundaries
    Gene,
    /// Motif-level boundaries
    Motif(String),
    /// Exon/intron boundaries
    ExonIntron,
}

impl BoundaryDetector for GenomicBoundaryDetector {
    type UnitType = NucleotideSequence;
    
    fn detect_boundaries(&self, content: &[u8]) -> Vec<Self::UnitType> {
        // Implementation would depend on the boundary type
        match self.boundary_type {
            GenomicBoundaryType::Nucleotide => {
                // Each nucleotide is a separate unit
                content
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| {
                        NucleotideSequence::new(vec![n], format!("nuc_{}", i))
                    })
                    .collect()
            }
            GenomicBoundaryType::Codon => {
                // Every three nucleotides is a unit
                (0..content.len())
                    .step_by(3)
                    .filter_map(|i| {
                        if i + 3 <= content.len() {
                            Some(NucleotideSequence::new(
                                content[i..i+3].to_vec(),
                                format!("codon_{}", i/3)
                            ))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            // Other boundary types would be implemented similarly
            _ => vec![NucleotideSequence::new(content.to_vec(), "whole_sequence")]
        }
    }
    
    fn configuration(&self) -> &BoundaryConfig {
        &self.config
    }
}

//------------------------------------------------------------------------------
// Genomic Operations
//------------------------------------------------------------------------------

/// Operations for genomic sequences
pub struct GenomicOperations;

impl UnitOperations<NucleotideSequence> for GenomicOperations {
    fn divide(&self, unit: &NucleotideSequence, pattern: &str) -> Vec<NucleotideSequence> {
        // Split by the pattern
        let pattern_bytes = pattern.as_bytes();
        let mut result = Vec::new();
        let mut last_idx = 0;
        
        // Simple naive pattern matching
        for i in 0..=unit.content.len() - pattern_bytes.len() {
            if unit.content[i..i+pattern_bytes.len()] == *pattern_bytes {
                // Add the segment before the pattern
                if i > last_idx {
                    result.push(NucleotideSequence::new(
                        unit.content[last_idx..i].to_vec(),
                        format!("{}_part_{}", unit.id.0, result.len())
                    ));
                }
                
                last_idx = i + pattern_bytes.len();
            }
        }
        
        // Add the last segment
        if last_idx < unit.content.len() {
            result.push(NucleotideSequence::new(
                unit.content[last_idx..].to_vec(),
                format!("{}_part_{}", unit.id.0, result.len())
            ));
        }
        
        result
    }
    
    fn multiply(&self, left: &NucleotideSequence, right: &NucleotideSequence) -> NucleotideSequence {
        // In genomic context, this could be interpreted as recombination
        let mut result = left.content.clone();
        result.extend_from_slice(&right.content);
        
        NucleotideSequence::new(
            result,
            format!("{}_x_{}", left.id.0, right.id.0)
        )
    }
    
    fn add(&self, left: &NucleotideSequence, right: &NucleotideSequence) -> NucleotideSequence {
        // Simple concatenation
        let mut result = left.content.clone();
        result.extend_from_slice(&right.content);
        
        NucleotideSequence::new(
            result,
            format!("{}_{}", left.id.0, right.id.0)
        )
    }
    
    fn subtract(&self, source: &NucleotideSequence, to_remove: &NucleotideSequence) -> NucleotideSequence {
        // Remove all occurrences of to_remove from source
        let pattern = to_remove.content.as_slice();
        let mut result = Vec::new();
        let mut i = 0;
        
        while i < source.content.len() {
            if i + pattern.len() <= source.content.len() && source.content[i..i+pattern.len()] == *pattern {
                // Skip the pattern
                i += pattern.len();
            } else {
                // Add this nucleotide
                result.push(source.content[i]);
                i += 1;
            }
        }
        
        NucleotideSequence::new(
            result,
            format!("{}_minus_{}", source.id.0, to_remove.id.0)
        )
    }
} 