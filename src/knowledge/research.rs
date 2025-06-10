use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::knowledge::{KnowledgeDomain, KnowledgeResult, Citation};

/// Represents a research query for obtaining information
#[derive(Debug, Clone)]
pub struct ResearchQuery {
    /// The main topic or question
    pub topic: String,
    /// The knowledge domain
    pub domain: KnowledgeDomain,
    /// Minimum confidence required (0.0 to 1.0)
    pub min_confidence: f64,
    /// Maximum number of results
    pub max_results: usize,
    /// Whether to include citations
    pub include_citations: bool,
    /// Additional context or constraints
    pub context: HashMap<String, String>,
}

impl ResearchQuery {
    /// Create a new research query
    pub fn new(topic: &str, domain: KnowledgeDomain) -> Self {
        Self {
            topic: topic.to_string(),
            domain,
            min_confidence: 0.6, // Default minimum confidence
            max_results: 5,      // Default maximum results
            include_citations: true,
            context: HashMap::new(),
        }
    }
    
    /// Set the minimum confidence threshold
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }
    
    /// Set the maximum number of results
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }
    
    /// Set whether to include citations
    pub fn with_citations(mut self, include_citations: bool) -> Self {
        self.include_citations = include_citations;
        self
    }
    
    /// Add context to the query
    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }
}

/// Trait for research providers that can answer queries
pub trait ResearchProvider {
    /// Execute a research query and return results
    fn execute_query(&self, query: &ResearchQuery) -> Vec<KnowledgeResult>;
    
    /// Check if a provider can handle a specific domain
    fn supports_domain(&self, domain: &KnowledgeDomain) -> bool;
    
    /// Get the name of the provider
    fn name(&self) -> &str;
}

/// Research manager that coordinates multiple research providers
pub struct ResearchManager {
    /// List of available research providers
    providers: Vec<Box<dyn ResearchProvider>>,
}

impl ResearchManager {
    /// Create a new research manager
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }
    
    /// Add a research provider
    pub fn add_provider(&mut self, provider: Box<dyn ResearchProvider>) {
        self.providers.push(provider);
    }
    
    /// Execute a research query across all suitable providers
    pub fn execute_query(&self, query: &ResearchQuery) -> Vec<KnowledgeResult> {
        let mut results = Vec::new();
        
        // Find providers that support the requested domain
        let suitable_providers: Vec<&Box<dyn ResearchProvider>> = self.providers
            .iter()
            .filter(|p| p.supports_domain(&query.domain))
            .collect();
        
        // If no suitable providers, return empty results
        if suitable_providers.is_empty() {
            return results;
        }
        
        // Query each provider
        for provider in suitable_providers {
            let provider_results = provider.execute_query(query);
            for result in provider_results {
                // Only include results that meet the confidence threshold
                if result.confidence >= query.min_confidence {
                    results.push(result);
                }
            }
        }
        
        // Sort results by confidence (descending)
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to the requested number of results
        if results.len() > query.max_results {
            results.truncate(query.max_results);
        }
        
        results
    }
    
    /// List all available providers
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }
    
    /// Get providers that support a specific domain
    pub fn providers_for_domain(&self, domain: &KnowledgeDomain) -> Vec<&str> {
        self.providers
            .iter()
            .filter(|p| p.supports_domain(domain))
            .map(|p| p.name())
            .collect()
    }
}

/// Basic internal knowledge base provider
pub struct InternalKnowledgeProvider {
    /// Name of this provider
    name: String,
    /// Simulated knowledge base (topic -> content)
    knowledge_base: HashMap<String, (String, f64)>,
    /// Domains supported by this provider
    supported_domains: Vec<KnowledgeDomain>,
}

impl InternalKnowledgeProvider {
    /// Create a new internal knowledge provider
    pub fn new(name: &str) -> Self {
        let mut provider = Self {
            name: name.to_string(),
            knowledge_base: HashMap::new(),
            supported_domains: vec![KnowledgeDomain::General],
        };
        
        // Add some sample knowledge
        provider.add_knowledge(
            "Rust programming language",
            "Rust is a systems programming language focused on safety, speed, and concurrency.",
            KnowledgeDomain::Technology,
            0.95,
        );
        
        provider.add_knowledge(
            "Text processing",
            "Text processing involves operations like parsing, analyzing, and transforming text data.",
            KnowledgeDomain::Technology,
            0.9,
        );
        
        provider.add_knowledge(
            "Metacognition",
            "Metacognition refers to awareness and understanding of one's own thought processes.",
            KnowledgeDomain::Science,
            0.85,
        );
        
        provider
    }
    
    /// Add knowledge to this provider
    pub fn add_knowledge(&mut self, topic: &str, content: &str, domain: KnowledgeDomain, confidence: f64) {
        self.knowledge_base.insert(topic.to_lowercase(), (content.to_string(), confidence));
        
        // Ensure the domain is supported
        if !self.supported_domains.contains(&domain) {
            self.supported_domains.push(domain);
        }
    }
    
    /// Add support for a domain
    pub fn add_supported_domain(&mut self, domain: KnowledgeDomain) {
        if !self.supported_domains.contains(&domain) {
            self.supported_domains.push(domain);
        }
    }
}

impl ResearchProvider for InternalKnowledgeProvider {
    fn execute_query(&self, query: &ResearchQuery) -> Vec<KnowledgeResult> {
        let mut results = Vec::new();
        
        // Simple fuzzy matching on topics
        for (topic, (content, confidence)) in &self.knowledge_base {
            if topic.contains(&query.topic.to_lowercase()) {
                // Create a result with the internal knowledge
                let result = KnowledgeResult {
                    content: content.clone(),
                    source: format!("Internal Knowledge Base: {}", self.name),
                    confidence: *confidence,
                    citation: if query.include_citations {
                        Some(Citation::new(
                            "database",
                            Some("Kwasa-Kwasa Knowledge Base"),
                            Some(topic),
                            Some(&self.name),
                            None,
                            Some("2023"),
                            None,
                        ))
                    } else {
                        None
                    },
                    last_verified: chrono::Utc::now(),
                };
                
                results.push(result);
            }
        }
        
        results
    }
    
    fn supports_domain(&self, domain: &KnowledgeDomain) -> bool {
        self.supported_domains.contains(domain)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_research_query_creation() {
        let query = ResearchQuery::new("Rust programming", KnowledgeDomain::Technology)
            .with_min_confidence(0.7)
            .with_max_results(3)
            .with_citations(true)
            .with_context("purpose", "educational");
        
        assert_eq!(query.topic, "Rust programming");
        assert!(matches!(query.domain, KnowledgeDomain::Technology));
        assert_eq!(query.min_confidence, 0.7);
        assert_eq!(query.max_results, 3);
        assert_eq!(query.include_citations, true);
        assert_eq!(query.context.get("purpose"), Some(&"educational".to_string()));
    }
    
    #[test]
    fn test_internal_knowledge_provider() {
        let provider = InternalKnowledgeProvider::new("Test Provider");
        
        let query = ResearchQuery::new("Rust", KnowledgeDomain::Technology);
        let results = provider.execute_query(&query);
        
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
        assert!(results[0].confidence > 0.9);
    }
    
    #[test]
    fn test_research_manager() {
        let mut manager = ResearchManager::new();
        
        let provider = InternalKnowledgeProvider::new("Test Provider");
        manager.add_provider(Box::new(provider));
        
        let query = ResearchQuery::new("Rust", KnowledgeDomain::Technology);
        let results = manager.execute_query(&query);
        
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Rust"));
        
        let providers = manager.list_providers();
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0], "Test Provider");
    }
} 