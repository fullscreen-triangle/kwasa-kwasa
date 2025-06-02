use rusqlite::{Connection, Result, params};
use std::path::Path;
use std::fs;
use log::{info, error};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Knowledge entry representing a piece of information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// Unique ID for the entry
    pub id: i64,
    
    /// The content of the entry
    pub content: String,
    
    /// Source of the information
    pub source: String,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Timestamp when the entry was created
    pub created_at: i64,
    
    /// Timestamp when the entry was last accessed
    pub last_accessed: i64,
    
    /// Number of times the entry has been accessed
    pub access_count: i64,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry
    pub fn new(content: &str, source: &str, tags: Vec<String>, confidence: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        Self {
            id: 0, // Will be set when saved to database
            content: content.to_string(),
            source: source.to_string(),
            tags,
            confidence,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }
    
    /// Record an access to this entry
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
    }
}

/// Knowledge database for storing and retrieving information
pub struct KnowledgeDatabase {
    conn: Connection,
}

impl KnowledgeDatabase {
    /// Create a new knowledge database
    pub fn new(db_path: &Path) -> Result<Self> {
        // Ensure the directory exists
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent).map_err(|e| rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_IOERR),
                Some(format!("Failed to create directory: {}", e))
            ))?;
        }
        
        let conn = Connection::open(db_path)?;
        
        // Initialize the database schema
        Self::initialize_schema(&conn)?;
        
        Ok(Self { conn })
    }
    
    /// Initialize the database schema
    fn initialize_schema(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                access_count INTEGER NOT NULL
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY,
                knowledge_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge (id) ON DELETE CASCADE,
                UNIQUE (knowledge_id, tag)
            )",
            [],
        )?;
        
        // Create indices for faster searching
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags (tag)",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_content ON knowledge (content)",
            [],
        )?;
        
        Ok(())
    }
    
    /// Add a new entry to the knowledge database
    pub fn add_entry(&self, entry: &mut KnowledgeEntry) -> Result<i64> {
        // Begin a transaction
        let tx = self.conn.transaction()?;
        
        // Insert the knowledge entry
        tx.execute(
            "INSERT INTO knowledge (content, source, confidence, created_at, last_accessed, access_count)
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                entry.content,
                entry.source,
                entry.confidence,
                entry.created_at,
                entry.last_accessed,
                entry.access_count
            ],
        )?;
        
        // Get the ID of the inserted entry
        let id = tx.last_insert_rowid();
        entry.id = id;
        
        // Insert tags
        for tag in &entry.tags {
            tx.execute(
                "INSERT OR IGNORE INTO tags (knowledge_id, tag) VALUES (?, ?)",
                params![id, tag],
            )?;
        }
        
        // Commit the transaction
        tx.commit()?;
        
        info!("Added knowledge entry {}: {}", id, entry.content);
        
        Ok(id)
    }
    
    /// Get an entry by ID
    pub fn get_entry(&self, id: i64) -> Result<KnowledgeEntry> {
        // Get the knowledge entry
        let mut stmt = self.conn.prepare(
            "SELECT id, content, source, confidence, created_at, last_accessed, access_count
             FROM knowledge WHERE id = ?",
        )?;
        
        let mut entry = stmt.query_row(params![id], |row| {
            Ok(KnowledgeEntry {
                id: row.get(0)?,
                content: row.get(1)?,
                source: row.get(2)?,
                tags: Vec::new(), // Will be populated later
                confidence: row.get(3)?,
                created_at: row.get(4)?,
                last_accessed: row.get(5)?,
                access_count: row.get(6)?,
            })
        })?;
        
        // Get the tags
        let mut tag_stmt = self.conn.prepare(
            "SELECT tag FROM tags WHERE knowledge_id = ?",
        )?;
        
        let tags: Result<Vec<String>> = tag_stmt
            .query_map(params![id], |row| row.get(0))?
            .collect();
        
        entry.tags = tags?;
        
        // Update access statistics
        self.update_access_stats(id)?;
        
        Ok(entry)
    }
    
    /// Search for entries by content
    pub fn search(&self, query: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content FROM knowledge
             WHERE content LIKE ?
             ORDER BY confidence DESC, last_accessed DESC
             LIMIT 10",
        )?;
        
        let results: Result<Vec<String>> = stmt
            .query_map(params![format!("%{}%", query)], |row| {
                let id: i64 = row.get(0)?;
                let content: String = row.get(1)?;
                
                // Update access statistics
                self.update_access_stats(id).unwrap_or_else(|e| {
                    error!("Failed to update access stats: {}", e);
                });
                
                Ok(content)
            })?
            .collect();
        
        results
    }
    
    /// Search for entries by tag
    pub fn search_by_tag(&self, tag: &str) -> Result<Vec<KnowledgeEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT k.id, k.content, k.source, k.confidence, k.created_at, k.last_accessed, k.access_count
             FROM knowledge k
             JOIN tags t ON k.id = t.knowledge_id
             WHERE t.tag = ?
             ORDER BY k.confidence DESC, k.last_accessed DESC",
        )?;
        
        let entries: Result<Vec<KnowledgeEntry>> = stmt
            .query_map(params![tag], |row| {
                let id: i64 = row.get(0)?;
                
                // Update access statistics
                self.update_access_stats(id).unwrap_or_else(|e| {
                    error!("Failed to update access stats: {}", e);
                });
                
                Ok(KnowledgeEntry {
                    id,
                    content: row.get(1)?,
                    source: row.get(2)?,
                    tags: Vec::new(), // Will be populated later
                    confidence: row.get(3)?,
                    created_at: row.get(4)?,
                    last_accessed: row.get(5)?,
                    access_count: row.get(6)?,
                })
            })?
            .collect();
        
        let mut result = entries?;
        
        // Get tags for each entry
        for entry in &mut result {
            entry.tags = self.get_tags_for_entry(entry.id)?;
        }
        
        Ok(result)
    }
    
    /// Get all tags for an entry
    fn get_tags_for_entry(&self, id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag FROM tags WHERE knowledge_id = ?",
        )?;
        
        let tags: Result<Vec<String>> = stmt
            .query_map(params![id], |row| row.get(0))?
            .collect();
        
        tags
    }
    
    /// Update access statistics for an entry
    fn update_access_stats(&self, id: i64) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        self.conn.execute(
            "UPDATE knowledge SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            params![now, id],
        )?;
        
        Ok(())
    }
    
    /// Delete an entry from the database
    pub fn delete_entry(&self, id: i64) -> Result<()> {
        // Begin a transaction
        let tx = self.conn.transaction()?;
        
        // Delete tags first (due to foreign key constraints)
        tx.execute(
            "DELETE FROM tags WHERE knowledge_id = ?",
            params![id],
        )?;
        
        // Delete the knowledge entry
        tx.execute(
            "DELETE FROM knowledge WHERE id = ?",
            params![id],
        )?;
        
        // Commit the transaction
        tx.commit()?;
        
        info!("Deleted knowledge entry {}", id);
        
        Ok(())
    }
    
    /// Update an entry in the database
    pub fn update_entry(&self, entry: &KnowledgeEntry) -> Result<()> {
        // Begin a transaction
        let tx = self.conn.transaction()?;
        
        // Update the knowledge entry
        tx.execute(
            "UPDATE knowledge SET
                content = ?,
                source = ?,
                confidence = ?
             WHERE id = ?",
            params![
                entry.content,
                entry.source,
                entry.confidence,
                entry.id
            ],
        )?;
        
        // Delete existing tags
        tx.execute(
            "DELETE FROM tags WHERE knowledge_id = ?",
            params![entry.id],
        )?;
        
        // Insert new tags
        for tag in &entry.tags {
            tx.execute(
                "INSERT OR IGNORE INTO tags (knowledge_id, tag) VALUES (?, ?)",
                params![entry.id, tag],
            )?;
        }
        
        // Commit the transaction
        tx.commit()?;
        
        info!("Updated knowledge entry {}", entry.id);
        
        Ok(())
    }
    
    /// Get the total number of entries in the database
    pub fn count_entries(&self) -> Result<i64> {
        let mut stmt = self.conn.prepare("SELECT COUNT(*) FROM knowledge")?;
        let count: i64 = stmt.query_row([], |row| row.get(0))?;
        
        Ok(count)
    }
    
    /// Get all tags in the database, sorted by frequency
    pub fn get_all_tags(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT tag, COUNT(*) as count
             FROM tags
             GROUP BY tag
             ORDER BY count DESC",
        )?;
        
        let tags: Result<Vec<(String, i64)>> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect();
        
        tags
    }
}

/// For testing purposes
#[cfg(test)]
pub mod tests {
    use super::*;
    use tempfile::tempdir;
    
    /// Create a test database for tests
    pub fn create_test_db() -> KnowledgeDatabase {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_knowledge.db");
        
        let db = KnowledgeDatabase::new(&db_path).unwrap();
        
        // Add some test entries
        let mut entry1 = KnowledgeEntry::new(
            "Machine learning is a subset of artificial intelligence.",
            "Wikipedia",
            vec!["machine learning".to_string(), "ai".to_string()],
            0.9,
        );
        db.add_entry(&mut entry1).unwrap();
        
        let mut entry2 = KnowledgeEntry::new(
            "Neural networks are a class of machine learning models.",
            "Book: Deep Learning",
            vec!["neural networks".to_string(), "machine learning".to_string()],
            0.95,
        );
        db.add_entry(&mut entry2).unwrap();
        
        let mut entry3 = KnowledgeEntry::new(
            "Climate change is causing global temperatures to rise.",
            "Scientific Journal",
            vec!["climate change".to_string(), "global warming".to_string()],
            0.99,
        );
        db.add_entry(&mut entry3).unwrap();
        
        db
    }
    
    #[test]
    fn test_add_and_get_entry() {
        let db = create_test_db();
        
        let mut entry = KnowledgeEntry::new(
            "Test content",
            "Test source",
            vec!["test".to_string(), "example".to_string()],
            0.8,
        );
        
        let id = db.add_entry(&mut entry).unwrap();
        let retrieved = db.get_entry(id).unwrap();
        
        assert_eq!(retrieved.content, "Test content");
        assert_eq!(retrieved.source, "Test source");
        assert_eq!(retrieved.tags, vec!["test", "example"]);
        assert_eq!(retrieved.confidence, 0.8);
        assert_eq!(retrieved.access_count, 1); // Incremented by get_entry
    }
    
    #[test]
    fn test_search() {
        let db = create_test_db();
        
        let results = db.search("machine learning").unwrap();
        assert_eq!(results.len(), 2);
        
        // Results should contain both entries with "machine learning"
        assert!(results.iter().any(|r| r.contains("subset of artificial intelligence")));
        assert!(results.iter().any(|r| r.contains("class of machine learning models")));
        
        // Should not contain unrelated entries
        assert!(!results.iter().any(|r| r.contains("climate change")));
    }
    
    #[test]
    fn test_search_by_tag() {
        let db = create_test_db();
        
        let results = db.search_by_tag("machine learning").unwrap();
        assert_eq!(results.len(), 2);
        
        let climate_results = db.search_by_tag("global warming").unwrap();
        assert_eq!(climate_results.len(), 1);
        assert_eq!(climate_results[0].content, "Climate change is causing global temperatures to rise.");
    }
    
    #[test]
    fn test_update_entry() {
        let db = create_test_db();
        
        // Add a new entry
        let mut entry = KnowledgeEntry::new(
            "Original content",
            "Original source",
            vec!["original".to_string()],
            0.5,
        );
        
        let id = db.add_entry(&mut entry).unwrap();
        
        // Update the entry
        let mut updated = db.get_entry(id).unwrap();
        updated.content = "Updated content".to_string();
        updated.source = "Updated source".to_string();
        updated.tags = vec!["updated".to_string(), "modified".to_string()];
        updated.confidence = 0.7;
        
        db.update_entry(&updated).unwrap();
        
        // Retrieve and verify the updated entry
        let retrieved = db.get_entry(id).unwrap();
        assert_eq!(retrieved.content, "Updated content");
        assert_eq!(retrieved.source, "Updated source");
        assert_eq!(retrieved.tags, vec!["modified", "updated"]);
        assert_eq!(retrieved.confidence, 0.7);
    }
    
    #[test]
    fn test_delete_entry() {
        let db = create_test_db();
        
        // Add a new entry
        let mut entry = KnowledgeEntry::new(
            "To be deleted",
            "Test",
            vec!["delete".to_string()],
            0.5,
        );
        
        let id = db.add_entry(&mut entry).unwrap();
        
        // Verify it exists
        let result = db.get_entry(id);
        assert!(result.is_ok());
        
        // Delete it
        db.delete_entry(id).unwrap();
        
        // Verify it's gone
        let result = db.get_entry(id);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_get_all_tags() {
        let db = create_test_db();
        
        let tags = db.get_all_tags().unwrap();
        
        // Should have at least the tags we added
        assert!(tags.iter().any(|(tag, _)| tag == "machine learning"));
        assert!(tags.iter().any(|(tag, _)| tag == "ai"));
        assert!(tags.iter().any(|(tag, _)| tag == "neural networks"));
        assert!(tags.iter().any(|(tag, _)| tag == "climate change"));
        assert!(tags.iter().any(|(tag, _)| tag == "global warming"));
        
        // Check frequency
        let ml_tag = tags.iter().find(|(tag, _)| tag == "machine learning").unwrap();
        assert_eq!(ml_tag.1, 2); // Used in 2 entries
    }
}

// Knowledge database and integration module

// Export public sub-modules
pub mod database;
pub mod citation;
pub mod research;
pub mod verification;

// Re-export common types
pub use database::KnowledgeDatabase as DatabaseImpl;
pub use citation::Citation;
pub use research::ResearchQuery;
pub use verification::FactVerifier;

/// Represents the result of a knowledge query
#[derive(Debug, Clone)]
pub struct KnowledgeResult {
    /// The content of the result
    pub content: String,
    /// The source of the information
    pub source: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Optional citation information
    pub citation: Option<Citation>,
    /// Timestamp of when this information was last verified
    pub last_verified: chrono::DateTime<chrono::Utc>,
}

/// Represents a domain of knowledge
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KnowledgeDomain {
    /// General knowledge
    General,
    /// Science domain
    Science,
    /// Technology domain
    Technology,
    /// Medicine domain
    Medicine,
    /// Business domain
    Business,
    /// Arts domain
    Arts,
    /// Custom domain with a name
    Custom(String),
}

impl std::fmt::Display for KnowledgeDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KnowledgeDomain::General => write!(f, "General"),
            KnowledgeDomain::Science => write!(f, "Science"),
            KnowledgeDomain::Technology => write!(f, "Technology"),
            KnowledgeDomain::Medicine => write!(f, "Medicine"),
            KnowledgeDomain::Business => write!(f, "Business"),
            KnowledgeDomain::Arts => write!(f, "Arts"),
            KnowledgeDomain::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// A trait for knowledge providers
pub trait KnowledgeProvider {
    /// Query for knowledge on a specific topic
    fn query(&self, topic: &str, domain: KnowledgeDomain) -> Vec<KnowledgeResult>;
    
    /// Verify if a statement is factual
    fn verify_statement(&self, statement: &str) -> Option<FactVerification>;
    
    /// Get citation for a knowledge result
    fn get_citation(&self, result: &KnowledgeResult) -> Option<Citation>;
    
    /// Update the knowledge database with new information
    fn update_database(&mut self, topic: &str, content: &str, source: &str) -> Result<(), String>;
}

/// Represents a verification result for a factual statement
#[derive(Debug, Clone)]
pub struct FactVerification {
    /// Is the statement verified as factual?
    pub is_factual: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Option<String>,
    /// Source of verification
    pub source: String,
}

/// Provides a default implementation that returns no knowledge
pub struct NullKnowledgeProvider;

impl KnowledgeProvider for NullKnowledgeProvider {
    fn query(&self, _topic: &str, _domain: KnowledgeDomain) -> Vec<KnowledgeResult> {
        Vec::new()
    }
    
    fn verify_statement(&self, _statement: &str) -> Option<FactVerification> {
        None
    }
    
    fn get_citation(&self, _result: &KnowledgeResult) -> Option<Citation> {
        None
    }
    
    fn update_database(&mut self, _topic: &str, _content: &str, _source: &str) -> Result<(), String> {
        Err("NullKnowledgeProvider does not support updates".to_string())
    }
}

/// Create a new knowledge provider based on configuration
pub fn create_knowledge_provider() -> Box<dyn KnowledgeProvider> {
    // Try to create a database-backed knowledge provider
    match std::env::temp_dir().join("kwasa_knowledge.db") {
        db_path => {
            match DatabaseKnowledgeProvider::new(&db_path) {
                Ok(provider) => Box::new(provider),
                Err(_) => Box::new(NullKnowledgeProvider),
            }
        }
    }
}

/// A knowledge provider backed by the local SQLite database
pub struct DatabaseKnowledgeProvider {
    database: DatabaseImpl,
}

impl DatabaseKnowledgeProvider {
    /// Create a new database-backed knowledge provider
    pub fn new(db_path: &std::path::Path) -> Result<Self, String> {
        let database = DatabaseImpl::new(db_path)
            .map_err(|e| format!("Failed to create knowledge database: {}", e))?;
        
        Ok(DatabaseKnowledgeProvider { database })
    }
    
    /// Add a knowledge entry to the database
    pub fn add_knowledge(&mut self, content: &str, source: &str, tags: Vec<String>, confidence: f64) -> Result<i64, String> {
        let mut entry = KnowledgeEntry::new(content, source, tags, confidence);
        self.database.add_entry(&mut entry)
            .map_err(|e| format!("Failed to add knowledge entry: {}", e))
    }
    
    /// Search for knowledge entries
    pub fn search_knowledge(&self, query: &str) -> Result<Vec<KnowledgeEntry>, String> {
        // First try searching by content
        let content_results = self.database.search(query)
            .map_err(|e| format!("Search failed: {}", e))?;
        
        // Also search by tag
        let tag_results = self.database.search_by_tag(query)
            .map_err(|e| format!("Tag search failed: {}", e))?;
        
        // Combine results and deduplicate
        let mut all_results = tag_results;
        
        // For now, return the tag results as they are more specific
        // In a real implementation, we would merge and rank results
        Ok(all_results)
    }
}

impl KnowledgeProvider for DatabaseKnowledgeProvider {
    fn query(&self, topic: &str, domain: KnowledgeDomain) -> Vec<KnowledgeResult> {
        // Search for entries related to the topic
        let search_query = match domain {
            KnowledgeDomain::General => topic.to_string(),
            _ => format!("{} {}", domain, topic),
        };
        
        match self.database.search(&search_query) {
            Ok(_content_matches) => {
                // Also search by tag to get more structured results
                match self.database.search_by_tag(topic) {
                    Ok(entries) => {
                        entries.into_iter().map(|entry| {
                            KnowledgeResult {
                                content: entry.content,
                                source: entry.source,
                                confidence: entry.confidence,
                                citation: Some(Citation::new(
                                    &entry.source,
                                    &format!("Retrieved from Kwasa-Kwasa knowledge base, ID: {}", entry.id),
                                    chrono::Utc::now(),
                                    research::CitationType::Database,
                                )),
                                last_verified: chrono::DateTime::from_timestamp(entry.last_accessed, 0)
                                    .unwrap_or_else(|| chrono::Utc::now()),
                            }
                        }).collect()
                    },
                    Err(_) => Vec::new(),
                }
            },
            Err(_) => Vec::new(),
        }
    }
    
    fn verify_statement(&self, statement: &str) -> Option<FactVerification> {
        // Simple fact verification by checking if we have supporting knowledge
        let results = self.query(statement, KnowledgeDomain::General);
        
        if results.is_empty() {
            return Some(FactVerification {
                is_factual: false,
                confidence: 0.1,
                evidence: None,
                source: "No supporting evidence found in knowledge base".to_string(),
            });
        }
        
        // Calculate average confidence from matching results
        let avg_confidence = results.iter()
            .map(|r| r.confidence)
            .sum::<f64>() / results.len() as f64;
        
        let supporting_evidence = results.iter()
            .take(3)  // Take up to 3 results
            .map(|r| format!("• {} (source: {})", r.content, r.source))
            .collect::<Vec<_>>()
            .join("\n");
        
        Some(FactVerification {
            is_factual: avg_confidence > 0.6,
            confidence: avg_confidence,
            evidence: Some(supporting_evidence),
            source: "Kwasa-Kwasa Knowledge Database".to_string(),
        })
    }
    
    fn get_citation(&self, result: &KnowledgeResult) -> Option<Citation> {
        result.citation.clone()
    }
    
    fn update_database(&mut self, topic: &str, content: &str, source: &str) -> Result<(), String> {
        let tags = vec![topic.to_string()];
        let confidence = 0.8; // Default confidence for manually added entries
        
        self.add_knowledge(content, source, tags, confidence)
            .map(|_| ())
    }
} 