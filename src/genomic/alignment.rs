//! Sequence alignment module
//!
//! This module provides pairwise and multiple sequence alignment functionality.

use super::{NucleotideSequence, Unit};

/// Sequence alignment engine
#[derive(Debug, Clone)]
pub struct AlignmentEngine {
    /// Alignment parameters
    parameters: AlignmentParameters,
}

/// Pairwise sequence aligner
#[derive(Debug, Clone)]
pub struct PairwiseAligner {
    /// Alignment parameters
    parameters: AlignmentParameters,
}

/// Multiple sequence aligner
#[derive(Debug, Clone)]
pub struct MultipleAligner {
    /// Alignment parameters
    parameters: AlignmentParameters,
}

/// Alignment parameters
#[derive(Debug, Clone)]
pub struct AlignmentParameters {
    /// Match score
    pub match_score: i32,
    /// Mismatch penalty
    pub mismatch_penalty: i32,
    /// Gap opening penalty
    pub gap_open_penalty: i32,
    /// Gap extension penalty
    pub gap_extend_penalty: i32,
    /// Alignment algorithm
    pub algorithm: AlignmentAlgorithm,
}

/// Alignment algorithm
#[derive(Debug, Clone)]
pub enum AlignmentAlgorithm {
    /// Needleman-Wunsch global alignment
    NeedlemanWunsch,
    /// Smith-Waterman local alignment
    SmithWaterman,
    /// BLAST-like heuristic alignment
    BLAST,
    /// Custom algorithm
    Custom(String),
}

/// Result of pairwise alignment
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// First aligned sequence
    pub sequence1: String,
    /// Second aligned sequence
    pub sequence2: String,
    /// Alignment score
    pub score: i32,
    /// Start position in first sequence
    pub start1: usize,
    /// End position in first sequence
    pub end1: usize,
    /// Start position in second sequence
    pub start2: usize,
    /// End position in second sequence
    pub end2: usize,
    /// Alignment statistics
    pub statistics: AlignmentStatistics,
}

/// Alignment statistics
#[derive(Debug, Clone)]
pub struct AlignmentStatistics {
    /// Alignment length
    pub length: usize,
    /// Number of matches
    pub matches: usize,
    /// Number of mismatches
    pub mismatches: usize,
    /// Number of gaps
    pub gaps: usize,
    /// Percent identity
    pub identity: f64,
    /// Percent similarity
    pub similarity: f64,
}

impl Default for AlignmentParameters {
    fn default() -> Self {
        Self {
            match_score: 2,
            mismatch_penalty: -1,
            gap_open_penalty: -2,
            gap_extend_penalty: -1,
            algorithm: AlignmentAlgorithm::NeedlemanWunsch,
        }
    }
}

impl AlignmentEngine {
    /// Create a new alignment engine
    pub fn new(parameters: AlignmentParameters) -> Self {
        Self { parameters }
    }

    /// Create with default parameters
    pub fn with_defaults() -> Self {
        Self { parameters: AlignmentParameters::default() }
    }

    /// Perform pairwise alignment
    pub fn align_pairwise(
        &self,
        seq1: &NucleotideSequence,
        seq2: &NucleotideSequence,
    ) -> AlignmentResult {
        let aligner = PairwiseAligner::new(self.parameters.clone());
        aligner.align(seq1, seq2)
    }
}

impl PairwiseAligner {
    /// Create a new pairwise aligner
    pub fn new(parameters: AlignmentParameters) -> Self {
        Self { parameters }
    }

    /// Align two sequences
    pub fn align(&self, seq1: &NucleotideSequence, seq2: &NucleotideSequence) -> AlignmentResult {
        self.needleman_wunsch(seq1, seq2)
    }

    /// Needleman-Wunsch global alignment
    fn needleman_wunsch(&self, seq1: &NucleotideSequence, seq2: &NucleotideSequence) -> AlignmentResult {
        let content1 = String::from_utf8_lossy(seq1.content());
        let content2 = String::from_utf8_lossy(seq2.content());
        
        let chars1: Vec<char> = content1.chars().collect();
        let chars2: Vec<char> = content2.chars().collect();
        
        let m = chars1.len();
        let n = chars2.len();
        
        // Initialize scoring matrix
        let mut dp = vec![vec![0i32; n + 1]; m + 1];
        
        // Initialize first row and column
        for i in 0..=m {
            dp[i][0] = i as i32 * self.parameters.gap_open_penalty;
        }
        for j in 0..=n {
            dp[0][j] = j as i32 * self.parameters.gap_open_penalty;
        }
        
        // Fill the matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_score = if chars1[i-1] == chars2[j-1] {
                    self.parameters.match_score
                } else {
                    self.parameters.mismatch_penalty
                };
                
                dp[i][j] = [
                    dp[i-1][j-1] + match_score,
                    dp[i-1][j] + self.parameters.gap_open_penalty,
                    dp[i][j-1] + self.parameters.gap_open_penalty,
                ].iter().max().copied().unwrap_or(0);
            }
        }
        
        // Traceback to get alignment
        let (aligned1, aligned2) = self.traceback(&dp, &chars1, &chars2);
        
        // Calculate statistics
        let statistics = self.calculate_statistics(&aligned1, &aligned2);
        
        AlignmentResult {
            sequence1: aligned1,
            sequence2: aligned2,
            score: dp[m][n],
            start1: 0,
            end1: m,
            start2: 0,
            end2: n,
            statistics,
        }
    }

    /// Traceback to reconstruct alignment
    fn traceback(&self, dp: &[Vec<i32>], chars1: &[char], chars2: &[char]) -> (String, String) {
        let mut aligned1 = String::new();
        let mut aligned2 = String::new();
        
        let mut i = chars1.len();
        let mut j = chars2.len();
        
        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let match_score = if chars1[i-1] == chars2[j-1] {
                    self.parameters.match_score
                } else {
                    self.parameters.mismatch_penalty
                };
                
                if dp[i][j] == dp[i-1][j-1] + match_score {
                    aligned1.insert(0, chars1[i-1]);
                    aligned2.insert(0, chars2[j-1]);
                    i -= 1;
                    j -= 1;
                } else if i > 0 && dp[i][j] == dp[i-1][j] + self.parameters.gap_open_penalty {
                    aligned1.insert(0, chars1[i-1]);
                    aligned2.insert(0, '-');
                    i -= 1;
                } else if j > 0 {
                    aligned1.insert(0, '-');
                    aligned2.insert(0, chars2[j-1]);
                    j -= 1;
                }
            } else if i > 0 {
                aligned1.insert(0, chars1[i-1]);
                aligned2.insert(0, '-');
                i -= 1;
            } else {
                aligned1.insert(0, '-');
                aligned2.insert(0, chars2[j-1]);
                j -= 1;
            }
        }
        
        (aligned1, aligned2)
    }

    /// Calculate alignment statistics
    fn calculate_statistics(&self, aligned1: &str, aligned2: &str) -> AlignmentStatistics {
        let chars1: Vec<char> = aligned1.chars().collect();
        let chars2: Vec<char> = aligned2.chars().collect();
        
        let length = chars1.len();
        let mut matches = 0;
        let mut mismatches = 0;
        let mut gaps = 0;
        
        for i in 0..length {
            if chars1[i] == '-' || chars2[i] == '-' {
                gaps += 1;
            } else if chars1[i] == chars2[i] {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }
        
        let identity = if length > 0 {
            matches as f64 / length as f64 * 100.0
        } else {
            0.0
        };
        
        let similarity = identity; // Simplified
        
        AlignmentStatistics {
            length,
            matches,
            mismatches,
            gaps,
            identity,
            similarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairwise_alignment() {
        let aligner = PairwiseAligner::new(AlignmentParameters::default());
        let seq1 = NucleotideSequence::new("ATCG".as_bytes(), "seq1");
        let seq2 = NucleotideSequence::new("ATGG".as_bytes(), "seq2");

        let result = aligner.align(&seq1, &seq2);

        assert_eq!(result.sequence1.len(), result.sequence2.len());
        assert!(result.statistics.identity <= 100.0);
    }
} 