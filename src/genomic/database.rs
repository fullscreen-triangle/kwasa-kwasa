//! Database module for genomic data storage
//!
//! This module provides storage and retrieval functionality for genomic data.

use std::collections::HashMap;
use super::{NucleotideSequence, Unit};

/// Genomic database
#[derive(Debug, Clone)]
pub struct GenomicDatabase {
    /// Stored sequences
    sequences: HashMap<String, NucleotideSequence>,
    /// Database metadata
    metadata: DatabaseMetadata,
    /// Configuration
    config: DatabaseConfig,
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Maximum sequences to store
    pub max_sequences: Option<usize>,
    /// Enable compression
    pub enable_compression: bool,
    /// Index sequences by content
    pub enable_indexing: bool,
}

/// Database metadata
#[derive(Debug, Clone)]
pub struct DatabaseMetadata {
    /// Creation timestamp
    pub created_at: u64,
    /// Last modified timestamp
    pub modified_at: u64,
    /// Total sequences stored
    pub sequence_count: usize,
    /// Total base pairs
    pub total_base_pairs: usize,
    /// Database version
    pub version: String,
}

/// Database query
#[derive(Debug, Clone)]
pub struct DatabaseQuery {
    /// Sequence ID pattern
    pub id_pattern: Option<String>,
    /// Minimum length
    pub min_length: Option<usize>,
    /// Maximum length
    pub max_length: Option<usize>,
    /// GC content range
    pub gc_range: Option<(f64, f64)>,
    /// Content pattern
    pub content_pattern: Option<String>,
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Matching sequences
    pub sequences: Vec<NucleotideSequence>,
    /// Query statistics
    pub statistics: QueryStatistics,
}

/// Query statistics
#[derive(Debug, Clone)]
pub struct QueryStatistics {
    /// Total matches found
    pub total_matches: usize,
    /// Query execution time (ms)
    pub execution_time_ms: u64,
    /// Sequences scanned
    pub sequences_scanned: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            max_sequences: Some(10000),
            enable_compression: false,
            enable_indexing: true,
        }
    }
}

impl GenomicDatabase {
    /// Create a new genomic database
    pub fn new(config: DatabaseConfig) -> Self {
        let metadata = DatabaseMetadata {
            created_at: chrono::Utc::now().timestamp() as u64,
            modified_at: chrono::Utc::now().timestamp() as u64,
            sequence_count: 0,
            total_base_pairs: 0,
            version: "1.0.0".to_string(),
        };

        Self {
            sequences: HashMap::new(),
            metadata,
            config,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(DatabaseConfig::default())
    }

    /// Add a sequence to the database
    pub fn add_sequence(&mut self, sequence: NucleotideSequence) -> Result<(), String> {
        // Check capacity limits
        if let Some(max_sequences) = self.config.max_sequences {
            if self.sequences.len() >= max_sequences {
                return Err("Database capacity exceeded".to_string());
            }
        }

        let sequence_id = sequence.id().to_string();
        let sequence_length = sequence.content().len();

        // Check if sequence already exists
        if self.sequences.contains_key(&sequence_id) {
            return Err(format!("Sequence with ID '{}' already exists", sequence_id));
        }

        // Add sequence
        self.sequences.insert(sequence_id, sequence);

        // Update metadata
        self.metadata.sequence_count += 1;
        self.metadata.total_base_pairs += sequence_length;
        self.metadata.modified_at = chrono::Utc::now().timestamp() as u64;

        Ok(())
    }

    /// Get a sequence by ID
    pub fn get_sequence(&self, id: &str) -> Option<&NucleotideSequence> {
        self.sequences.get(id)
    }

    /// Remove a sequence by ID
    pub fn remove_sequence(&mut self, id: &str) -> Option<NucleotideSequence> {
        if let Some(sequence) = self.sequences.remove(id) {
            self.metadata.sequence_count -= 1;
            self.metadata.total_base_pairs -= sequence.content().len();
            self.metadata.modified_at = chrono::Utc::now().timestamp() as u64;
            Some(sequence)
        } else {
            None
        }
    }

    /// Query sequences matching criteria
    pub fn query(&self, query: &DatabaseQuery) -> QueryResult {
        let start_time = std::time::Instant::now();
        let mut matching_sequences = Vec::new();
        let mut sequences_scanned = 0;

        for (id, sequence) in &self.sequences {
            sequences_scanned += 1;

            // Check ID pattern
            if let Some(ref pattern) = query.id_pattern {
                if !id.contains(pattern) {
                    continue;
                }
            }

            // Check length constraints
            let sequence_length = sequence.content().len();
            if let Some(min_length) = query.min_length {
                if sequence_length < min_length {
                    continue;
                }
            }
            if let Some(max_length) = query.max_length {
                if sequence_length > max_length {
                    continue;
                }
            }

            // Check GC content
            if let Some((min_gc, max_gc)) = query.gc_range {
                let gc_content = sequence.gc_content();
                if gc_content < min_gc || gc_content > max_gc {
                    continue;
                }
            }

            // Check content pattern
            if let Some(ref pattern) = query.content_pattern {
                let content = String::from_utf8_lossy(sequence.content());
                if !content.contains(pattern) {
                    continue;
                }
            }

            // Sequence matches all criteria
            matching_sequences.push(sequence.clone());
        }

        let execution_time = start_time.elapsed();

        QueryResult {
            sequences: matching_sequences.clone(),
            statistics: QueryStatistics {
                total_matches: matching_sequences.len(),
                execution_time_ms: execution_time.as_millis() as u64,
                sequences_scanned,
            },
        }
    }

    /// Get all sequence IDs
    pub fn list_sequence_ids(&self) -> Vec<String> {
        self.sequences.keys().cloned().collect()
    }

    /// Get database metadata
    pub fn metadata(&self) -> &DatabaseMetadata {
        &self.metadata
    }

    /// Get database statistics
    pub fn statistics(&self) -> DatabaseStatistics {
        let mut length_distribution = Vec::new();
        let mut gc_distribution = Vec::new();
        let mut total_length = 0;
        let mut total_gc = 0.0;

        for sequence in self.sequences.values() {
            let length = sequence.content().len();
            let gc_content = sequence.gc_content();
            
            length_distribution.push(length);
            gc_distribution.push(gc_content);
            total_length += length;
            total_gc += gc_content;
        }

        let avg_length = if self.metadata.sequence_count > 0 {
            total_length as f64 / self.metadata.sequence_count as f64
        } else {
            0.0
        };

        let avg_gc = if self.metadata.sequence_count > 0 {
            total_gc / self.metadata.sequence_count as f64
        } else {
            0.0
        };

        DatabaseStatistics {
            total_sequences: self.metadata.sequence_count,
            total_base_pairs: self.metadata.total_base_pairs,
            average_length: avg_length,
            average_gc_content: avg_gc,
            length_distribution,
            gc_distribution,
        }
    }

    /// Clear all sequences
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.metadata.sequence_count = 0;
        self.metadata.total_base_pairs = 0;
        self.metadata.modified_at = chrono::Utc::now().timestamp() as u64;
    }

    /// Get number of sequences
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    /// Total number of sequences
    pub total_sequences: usize,
    /// Total base pairs
    pub total_base_pairs: usize,
    /// Average sequence length
    pub average_length: f64,
    /// Average GC content
    pub average_gc_content: f64,
    /// Length distribution
    pub length_distribution: Vec<usize>,
    /// GC content distribution
    pub gc_distribution: Vec<f64>,
}

impl DatabaseQuery {
    /// Create a new empty query
    pub fn new() -> Self {
        Self {
            id_pattern: None,
            min_length: None,
            max_length: None,
            gc_range: None,
            content_pattern: None,
        }
    }

    /// Add ID pattern filter
    pub fn with_id_pattern(mut self, pattern: String) -> Self {
        self.id_pattern = Some(pattern);
        self
    }

    /// Add length range filter
    pub fn with_length_range(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.min_length = min;
        self.max_length = max;
        self
    }

    /// Add GC content range filter
    pub fn with_gc_range(mut self, min: f64, max: f64) -> Self {
        self.gc_range = Some((min, max));
        self
    }

    /// Add content pattern filter
    pub fn with_content_pattern(mut self, pattern: String) -> Self {
        self.content_pattern = Some(pattern);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_operations() {
        let mut db = GenomicDatabase::default();
        let sequence = NucleotideSequence::new("ATCGATCG".as_bytes(), "test_seq");

        // Add sequence
        assert!(db.add_sequence(sequence.clone()).is_ok());
        assert_eq!(db.len(), 1);

        // Retrieve sequence
        let retrieved = db.get_sequence("test_seq");
        assert!(retrieved.is_some());

        // Remove sequence
        let removed = db.remove_sequence("test_seq");
        assert!(removed.is_some());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_database_query() {
        let mut db = GenomicDatabase::default();
        
        let seq1 = NucleotideSequence::new("ATCGATCG".as_bytes(), "short_seq");
        let seq2 = NucleotideSequence::new("ATCGATCGATCGATCGATCG".as_bytes(), "long_seq");
        
        db.add_sequence(seq1).unwrap();
        db.add_sequence(seq2).unwrap();

        // Query by length
        let query = DatabaseQuery::new().with_length_range(Some(15), None);
        let result = db.query(&query);
        
        assert_eq!(result.sequences.len(), 1);
        assert_eq!(result.sequences[0].id().to_string(), "long_seq");
    }

    #[test]
    fn test_database_statistics() {
        let mut db = GenomicDatabase::default();
        
        let seq1 = NucleotideSequence::new("ATCG".as_bytes(), "seq1");
        let seq2 = NucleotideSequence::new("GCTA".as_bytes(), "seq2");
        
        db.add_sequence(seq1).unwrap();
        db.add_sequence(seq2).unwrap();

        let stats = db.statistics();
        
        assert_eq!(stats.total_sequences, 2);
        assert_eq!(stats.total_base_pairs, 8);
        assert_eq!(stats.average_length, 4.0);
    }
} 