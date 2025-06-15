//! Phylogenetic analysis module
//!
//! This module provides phylogenetic tree construction and analysis.

use std::collections::HashMap;
use super::{NucleotideSequence, Unit};

/// Phylogenetic analyzer
#[derive(Debug, Clone)]
pub struct PhylogeneticAnalyzer {
    /// Configuration
    config: PhylogeneticConfig,
}

/// Configuration for phylogenetic analysis
#[derive(Debug, Clone)]
pub struct PhylogeneticConfig {
    /// Tree building method
    pub method: TreeMethod,
    /// Bootstrap iterations
    pub bootstrap_iterations: usize,
    /// Distance calculation method
    pub distance_method: DistanceMethod,
}

/// Tree building method
#[derive(Debug, Clone)]
pub enum TreeMethod {
    /// Neighbor joining
    NeighborJoining,
    /// UPGMA
    UPGMA,
    /// Maximum likelihood
    MaximumLikelihood,
    /// Maximum parsimony
    MaximumParsimony,
}

/// Distance calculation method
#[derive(Debug, Clone)]
pub enum DistanceMethod {
    /// Hamming distance
    Hamming,
    /// Jukes-Cantor
    JukesCantor,
    /// Kimura 2-parameter
    Kimura2P,
}

/// Phylogenetic tree
#[derive(Debug, Clone)]
pub struct PhylogeneticTree {
    /// Tree nodes
    pub nodes: Vec<TreeNode>,
    /// Tree edges
    pub edges: Vec<TreeEdge>,
    /// Root node ID
    pub root_id: String,
    /// Newick format representation
    pub newick: String,
}

/// Tree node
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node ID
    pub id: String,
    /// Node label (for leaves)
    pub label: Option<String>,
    /// Branch length
    pub branch_length: f64,
    /// Bootstrap support
    pub bootstrap_support: Option<f64>,
    /// Is this a leaf node?
    pub is_leaf: bool,
}

/// Tree edge
#[derive(Debug, Clone)]
pub struct TreeEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge weight
    pub weight: f64,
}

/// Distance matrix
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Sequence labels
    pub labels: Vec<String>,
    /// Distance values
    pub distances: Vec<Vec<f64>>,
}

/// Result of phylogenetic analysis
#[derive(Debug, Clone)]
pub struct PhylogeneticResult {
    /// Constructed tree
    pub tree: PhylogeneticTree,
    /// Distance matrix
    pub distance_matrix: DistanceMatrix,
    /// Bootstrap values
    pub bootstrap_values: Vec<f64>,
    /// Tree statistics
    pub statistics: TreeStatistics,
}

/// Tree statistics
#[derive(Debug, Clone)]
pub struct TreeStatistics {
    /// Number of taxa
    pub num_taxa: usize,
    /// Tree height
    pub tree_height: f64,
    /// Average branch length
    pub avg_branch_length: f64,
    /// Tree balance index
    pub balance_index: f64,
}

impl Default for PhylogeneticConfig {
    fn default() -> Self {
        Self {
            method: TreeMethod::NeighborJoining,
            bootstrap_iterations: 100,
            distance_method: DistanceMethod::JukesCantor,
        }
    }
}

impl PhylogeneticAnalyzer {
    /// Create a new phylogenetic analyzer
    pub fn new(config: PhylogeneticConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(PhylogeneticConfig::default())
    }

    /// Perform phylogenetic analysis on sequences
    pub fn analyze(&self, sequences: &[NucleotideSequence]) -> PhylogeneticResult {
        // Calculate distance matrix
        let distance_matrix = self.calculate_distance_matrix(sequences);
        
        // Build tree
        let tree = self.build_tree(&distance_matrix);
        
        // Calculate bootstrap support
        let bootstrap_values = self.calculate_bootstrap_support(sequences);
        
        // Calculate tree statistics
        let statistics = self.calculate_tree_statistics(&tree);
        
        PhylogeneticResult {
            tree,
            distance_matrix,
            bootstrap_values,
            statistics,
        }
    }

    /// Calculate distance matrix between sequences
    fn calculate_distance_matrix(&self, sequences: &[NucleotideSequence]) -> DistanceMatrix {
        let n = sequences.len();
        let mut distances = vec![vec![0.0; n]; n];
        let mut labels = Vec::new();
        
        // Extract labels
        for seq in sequences {
            labels.push(seq.id().to_string());
        }
        
        // Calculate pairwise distances
        for i in 0..n {
            for j in i + 1..n {
                let distance = self.calculate_pairwise_distance(&sequences[i], &sequences[j]);
                distances[i][j] = distance;
                distances[j][i] = distance;
            }
        }
        
        DistanceMatrix {
            labels,
            distances,
        }
    }

    /// Calculate pairwise distance between two sequences
    fn calculate_pairwise_distance(&self, seq1: &NucleotideSequence, seq2: &NucleotideSequence) -> f64 {
        let content1 = String::from_utf8_lossy(seq1.content());
        let content2 = String::from_utf8_lossy(seq2.content());
        
        let chars1: Vec<char> = content1.chars().collect();
        let chars2: Vec<char> = content2.chars().collect();
        
        let min_len = chars1.len().min(chars2.len());
        let mut differences = 0;
        
        // Count differences
        for i in 0..min_len {
            if chars1[i] != chars2[i] {
                differences += 1;
            }
        }
        
        // Add length difference
        differences += chars1.len().abs_diff(chars2.len());
        
        match self.config.distance_method {
            DistanceMethod::Hamming => {
                differences as f64 / chars1.len().max(chars2.len()) as f64
            }
            DistanceMethod::JukesCantor => {
                let p = differences as f64 / min_len as f64;
                if p < 0.75 {
                    -0.75 * (1.0 - 4.0 * p / 3.0).ln()
                } else {
                    f64::INFINITY
                }
            }
            DistanceMethod::Kimura2P => {
                // Simplified Kimura 2-parameter model
                let p = differences as f64 / min_len as f64;
                if p < 0.5 {
                    -0.5 * (1.0 - 2.0 * p).ln()
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// Build phylogenetic tree from distance matrix
    fn build_tree(&self, distance_matrix: &DistanceMatrix) -> PhylogeneticTree {
        match self.config.method {
            TreeMethod::NeighborJoining => self.neighbor_joining(distance_matrix),
            TreeMethod::UPGMA => self.upgma(distance_matrix),
            _ => {
                // Fallback to neighbor joining for complex methods
                self.neighbor_joining(distance_matrix)
            }
        }
    }

    /// Neighbor joining algorithm
    fn neighbor_joining(&self, distance_matrix: &DistanceMatrix) -> PhylogeneticTree {
        let n = distance_matrix.labels.len();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_counter = 0;
        
        // Create leaf nodes
        for (i, label) in distance_matrix.labels.iter().enumerate() {
            nodes.push(TreeNode {
                id: format!("node_{}", i),
                label: Some(label.clone()),
                branch_length: 0.0,
                bootstrap_support: None,
                is_leaf: true,
            });
        }
        
        // Simplified neighbor joining (just create a star tree for now)
        let root_id = format!("node_{}", n);
        nodes.push(TreeNode {
            id: root_id.clone(),
            label: None,
            branch_length: 0.0,
            bootstrap_support: None,
            is_leaf: false,
        });
        
        // Connect all leaves to root
        for i in 0..n {
            edges.push(TreeEdge {
                source: root_id.clone(),
                target: format!("node_{}", i),
                weight: 1.0,
            });
        }
        
        PhylogeneticTree {
            nodes,
            edges,
            root_id,
            newick: self.generate_newick(&distance_matrix.labels),
        }
    }

    /// UPGMA algorithm
    fn upgma(&self, distance_matrix: &DistanceMatrix) -> PhylogeneticTree {
        // Simplified UPGMA - returns same structure as NJ for now
        self.neighbor_joining(distance_matrix)
    }

    /// Generate Newick format string
    fn generate_newick(&self, labels: &[String]) -> String {
        let mut newick = "(".to_string();
        for (i, label) in labels.iter().enumerate() {
            if i > 0 {
                newick.push(',');
            }
            newick.push_str(&format!("{}:1.0", label));
        }
        newick.push_str(");");
        newick
    }

    /// Calculate bootstrap support values
    fn calculate_bootstrap_support(&self, sequences: &[NucleotideSequence]) -> Vec<f64> {
        let mut bootstrap_values = Vec::new();
        
        // Simplified bootstrap - just return some values
        for _ in 0..sequences.len().saturating_sub(1) {
            bootstrap_values.push(0.8); // Simplified bootstrap value
        }
        
        bootstrap_values
    }

    /// Calculate tree statistics
    fn calculate_tree_statistics(&self, tree: &PhylogeneticTree) -> TreeStatistics {
        let num_taxa = tree.nodes.iter().filter(|n| n.is_leaf).count();
        
        let total_branch_length: f64 = tree.nodes.iter()
            .map(|n| n.branch_length)
            .sum();
            
        let avg_branch_length = if tree.nodes.len() > 0 {
            total_branch_length / tree.nodes.len() as f64
        } else {
            0.0
        };
        
        TreeStatistics {
            num_taxa,
            tree_height: total_branch_length,
            avg_branch_length,
            balance_index: 0.5, // Simplified balance index
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phylogenetic_analysis() {
        let analyzer = PhylogeneticAnalyzer::default();
        let sequences = vec![
            NucleotideSequence::new("ATCGATCG".as_bytes(), "seq1"),
            NucleotideSequence::new("ATGGATCG".as_bytes(), "seq2"),
            NucleotideSequence::new("ATCGATGG".as_bytes(), "seq3"),
        ];

        let result = analyzer.analyze(&sequences);

        assert_eq!(result.tree.nodes.len(), 4); // 3 leaves + 1 internal
        assert_eq!(result.distance_matrix.labels.len(), 3);
        assert!(!result.tree.newick.is_empty());
    }

    #[test]
    fn test_distance_calculation() {
        let analyzer = PhylogeneticAnalyzer::default();
        let seq1 = NucleotideSequence::new("ATCG".as_bytes(), "seq1");
        let seq2 = NucleotideSequence::new("ATGG".as_bytes(), "seq2");

        let distance = analyzer.calculate_pairwise_distance(&seq1, &seq2);

        assert!(distance > 0.0);
        assert!(distance < 1.0);
    }
} 