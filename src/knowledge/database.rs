use std::path::PathBuf;
use std::collections::HashMap;
use rusqlite::{Connection, params, Result as SqliteResult};
use chrono::{DateTime, Utc};

use crate::knowledge::{KnowledgeDomain, KnowledgeResult, Citation};

/// SQLite-based knowledge database for storing and retrieving information
pub struct KnowledgeDatabase {
    /// Database connection
    conn: Connection,
    /// Path to the database file
    db_path: PathBuf,
    /// Cache of recent queries
    cache: HashMap<String, Vec<KnowledgeResult>>,
}

impl KnowledgeDatabase {
    /// Create a new knowledge database with the specified database file
    pub fn new(db_path: PathBuf) -> SqliteResult<Self> {
        let conn = Connection::open(&db_path)?;
        
        // Initialize database schema if needed
        Self::initialize_schema(&conn)?;
        
        Ok(Self {
            conn,
            db_path,
            cache: HashMap::new(),
        })
    }
    
    /// Initialize the database schema if it doesn't exist
    fn initialize_schema(conn: &Connection) -> SqliteResult<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                last_verified TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY,
                knowledge_id INTEGER NOT NULL,
                citation_type TEXT NOT NULL,
                author TEXT,
                title TEXT,
                publication TEXT,
                url TEXT,
                date TEXT,
                page_numbers TEXT,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_entries (id)
            )",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_entries (topic)",
            [],
        )?;
        
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_entries (domain)",
            [],
        )?;
        
        Ok(())
    }
    
    /// Query the database for knowledge on a specific topic
    pub fn query(&self, topic: &str, domain: KnowledgeDomain) -> Vec<KnowledgeResult> {
        // Check cache first
        let cache_key = format!("{}:{}", topic, domain);
        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }
        
        // Query the database
        let domain_str = domain.to_string();
        let mut stmt = match self.conn.prepare(
            "SELECT id, content, source, confidence, last_verified 
             FROM knowledge_entries 
             WHERE topic LIKE ?1 AND domain = ?2 
             ORDER BY confidence DESC"
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        
        let query_results = match stmt.query_map(
            params![format!("%{}%", topic), domain_str],
            |row| {
                let id: i64 = row.get(0)?;
                let content: String = row.get(1)?;
                let source: String = row.get(2)?;
                let confidence: f64 = row.get(3)?;
                let last_verified: String = row.get(4)?;
                
                let last_verified_dt = DateTime::parse_from_rfc3339(&last_verified)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                
                // Get citation for this entry
                let citation = self.get_citation_for_entry(id).unwrap_or(None);
                
                Ok(KnowledgeResult {
                    content,
                    source,
                    confidence,
                    citation,
                    last_verified: last_verified_dt,
                })
            },
        ) {
            Ok(results) => results,
            Err(_) => return Vec::new(),
        };
        
        let mut results = Vec::new();
        for result in query_results {
            if let Ok(knowledge_result) = result {
                results.push(knowledge_result);
            }
        }
        
        // Update cache
        // ToDo: Implement cache expiry
        let cache_entry = results.clone();
        self.cache.insert(cache_key, cache_entry);
        
        results
    }
    
    /// Get the citation for a knowledge entry by ID
    fn get_citation_for_entry(&self, entry_id: i64) -> SqliteResult<Option<Citation>> {
        let mut stmt = self.conn.prepare(
            "SELECT citation_type, author, title, publication, url, date, page_numbers 
             FROM citations 
             WHERE knowledge_id = ?"
        )?;
        
        let citation_result = stmt.query_row(params![entry_id], |row| {
            let citation_type: String = row.get(0)?;
            let author: Option<String> = row.get(1)?;
            let title: Option<String> = row.get(2)?;
            let publication: Option<String> = row.get(3)?;
            let url: Option<String> = row.get(4)?;
            let date: Option<String> = row.get(5)?;
            let page_numbers: Option<String> = row.get(6)?;
            
            Ok(Citation {
                citation_type,
                author,
                title,
                publication,
                url,
                date,
                page_numbers,
            })
        });
        
        match citation_result {
            Ok(citation) => Ok(Some(citation)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e),
        }
    }
    
    /// Add a new knowledge entry to the database
    pub fn add_entry(
        &mut self,
        topic: &str,
        content: &str,
        domain: KnowledgeDomain,
        source: &str,
        confidence: f64,
        citation: Option<Citation>,
    ) -> SqliteResult<i64> {
        let now = Utc::now().to_rfc3339();
        let domain_str = domain.to_string();
        
        let tx = self.conn.transaction()?;
        
        let entry_id = tx.execute(
            "INSERT INTO knowledge_entries 
             (topic, content, domain, source, confidence, last_verified, created_at) 
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![topic, content, domain_str, source, confidence, now, now],
        )?;
        
        // Add citation if provided
        if let Some(citation) = citation {
            tx.execute(
                "INSERT INTO citations 
                 (knowledge_id, citation_type, author, title, publication, url, date, page_numbers) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    entry_id,
                    citation.citation_type,
                    citation.author,
                    citation.title,
                    citation.publication,
                    citation.url,
                    citation.date,
                    citation.page_numbers,
                ],
            )?;
        }
        
        tx.commit()?;
        
        // Clear cache for this topic/domain
        let cache_key = format!("{}:{}", topic, domain);
        self.cache.remove(&cache_key);
        
        Ok(entry_id)
    }
    
    /// Update an existing knowledge entry
    pub fn update_entry(
        &mut self,
        id: i64,
        content: &str,
        source: &str,
        confidence: f64,
    ) -> SqliteResult<()> {
        let now = Utc::now().to_rfc3339();
        
        let rows_affected = self.conn.execute(
            "UPDATE knowledge_entries 
             SET content = ?, source = ?, confidence = ?, last_verified = ? 
             WHERE id = ?",
            params![content, source, confidence, now, id],
        )?;
        
        if rows_affected == 0 {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        }
        
        // Clear all cache as this could affect multiple queries
        self.cache.clear();
        
        Ok(())
    }
    
    /// Delete a knowledge entry
    pub fn delete_entry(&mut self, id: i64) -> SqliteResult<()> {
        let tx = self.conn.transaction()?;
        
        // Delete citations first due to foreign key constraint
        tx.execute(
            "DELETE FROM citations WHERE knowledge_id = ?",
            params![id],
        )?;
        
        let rows_affected = tx.execute(
            "DELETE FROM knowledge_entries WHERE id = ?",
            params![id],
        )?;
        
        if rows_affected == 0 {
            return Err(rusqlite::Error::QueryReturnedNoRows);
        }
        
        tx.commit()?;
        
        // Clear all cache as this could affect multiple queries
        self.cache.clear();
        
        Ok(())
    }
    
    /// Get the path to the database file
    pub fn db_path(&self) -> &PathBuf {
        &self.db_path
    }
    
    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Get the total number of entries in the database
    pub fn count_entries(&self) -> SqliteResult<i64> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge_entries",
            [],
            |row| row.get(0),
        )?;
        
        Ok(count)
    }
    
    /// Perform a full-text search on the knowledge database
    pub fn full_text_search(&self, query: &str) -> Vec<KnowledgeResult> {
        // This is a simple implementation - a real one would use FTS5
        let mut stmt = match self.conn.prepare(
            "SELECT id, content, source, confidence, domain, last_verified 
             FROM knowledge_entries 
             WHERE content LIKE ? OR topic LIKE ? 
             ORDER BY confidence DESC"
        ) {
            Ok(stmt) => stmt,
            Err(_) => return Vec::new(),
        };
        
        let search_pattern = format!("%{}%", query);
        
        let query_results = match stmt.query_map(
            params![search_pattern.clone(), search_pattern],
            |row| {
                let id: i64 = row.get(0)?;
                let content: String = row.get(1)?;
                let source: String = row.get(2)?;
                let confidence: f64 = row.get(3)?;
                let domain: String = row.get(4)?;
                let last_verified: String = row.get(5)?;
                
                let last_verified_dt = DateTime::parse_from_rfc3339(&last_verified)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                
                // Get citation for this entry
                let citation = self.get_citation_for_entry(id).unwrap_or(None);
                
                Ok(KnowledgeResult {
                    content,
                    source,
                    confidence,
                    citation,
                    last_verified: last_verified_dt,
                })
            },
        ) {
            Ok(results) => results,
            Err(_) => return Vec::new(),
        };
        
        let mut results = Vec::new();
        for result in query_results {
            if let Ok(knowledge_result) = result {
                results.push(knowledge_result);
            }
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_database_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_knowledge.db");
        
        let db = KnowledgeDatabase::new(db_path).unwrap();
        assert_eq!(db.count_entries().unwrap(), 0);
    }
    
    #[test]
    fn test_add_entry() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_knowledge.db");
        
        let mut db = KnowledgeDatabase::new(db_path).unwrap();
        
        let id = db.add_entry(
            "Rust Programming",
            "Rust is a systems programming language focused on safety and performance.",
            KnowledgeDomain::Technology,
            "Official Rust Documentation",
            0.95,
            None,
        ).unwrap();
        
        assert!(id > 0);
        assert_eq!(db.count_entries().unwrap(), 1);
    }
    
    #[test]
    fn test_query() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_knowledge.db");
        
        let mut db = KnowledgeDatabase::new(db_path).unwrap();
        
        db.add_entry(
            "Rust Programming",
            "Rust is a systems programming language focused on safety and performance.",
            KnowledgeDomain::Technology,
            "Official Rust Documentation",
            0.95,
            None,
        ).unwrap();
        
        let results = db.query("Rust", KnowledgeDomain::Technology);
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("systems programming language"));
        
        // Query for nonexistent topic
        let results = db.query("Nonexistent", KnowledgeDomain::Technology);
        assert_eq!(results.len(), 0);
    }
} 