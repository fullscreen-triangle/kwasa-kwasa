use crate::error::{Error, Result};
use crate::text_unit::TextUnit;
use crate::pattern::metacognitive::{MetaCognitive, MetaNode, MetaNodeType};
use std::collections::HashMap;

/// Text processor for advanced text analysis using metacognitive reasoning
pub struct TextProcessor {
    /// Metacognitive reasoning engine
    meta: MetaCognitive,
    /// Processing options
    options: ProcessingOptions,
}

/// Options for text processing
#[derive(Debug, Clone)]
pub struct ProcessingOptions {
    /// Minimum confidence threshold for patterns (0.0-1.0)
    pub min_confidence: f64,
    /// Maximum number of patterns to extract
    pub max_patterns: usize,
    /// Whether to use semantic analysis
    pub use_semantic: bool,
    /// Whether to use structural analysis
    pub use_structural: bool,
    /// Whether to generate reflections
    pub generate_reflections: bool,
}

impl Default for ProcessingOptions {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_patterns: 10,
            use_semantic: true,
            use_structural: true,
            generate_reflections: true,
        }
    }
}

/// Results from text processing
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Extracted patterns
    pub patterns: Vec<crate::pattern::Pattern>,
    /// Insights from metacognitive reflection
    pub insights: Vec<String>,
    /// Processed text units
    pub text_units: Vec<TextUnit>,
}

impl TextProcessor {
    /// Create a new text processor with default options
    pub fn new() -> Self {
        Self {
            meta: MetaCognitive::new(),
            options: ProcessingOptions::default(),
        }
    }
    
    /// Create a new text processor with custom options
    pub fn with_options(options: ProcessingOptions) -> Self {
        Self {
            meta: MetaCognitive::new(),
            options,
        }
    }
    
    /// Process text using metacognitive reasoning
    pub fn process(&mut self, text: &str) -> Result<ProcessingResult> {
        // Break text into meaningful units
        let units = self.extract_units(text)?;
        
        // Create nodes for each text unit
        for (i, unit) in units.iter().enumerate() {
            let node = MetaNode {
                id: format!("unit_{}", i),
                content: unit.content.clone(),
                confidence: 0.9,
                evidence: vec!["text extraction".to_string()],
                node_type: MetaNodeType::Concept,
                metadata: HashMap::new(),
            };
            
            self.meta.add_node(node)?;
        }
        
        // Apply metacognitive reasoning
        let focus_nodes: Vec<String> = (0..units.len())
            .map(|i| format!("unit_{}", i))
            .collect();
        
        let patterns = self.meta.reason(&focus_nodes)?;
        
        // Filter patterns by confidence
        let filtered_patterns: Vec<_> = patterns.into_iter()
            .filter(|p| p.confidence >= self.options.min_confidence)
            .take(self.options.max_patterns)
            .collect();
        
        // Generate reflections if enabled
        let insights = if self.options.generate_reflections {
            self.meta.reflect()?
        } else {
            Vec::new()
        };
        
        Ok(ProcessingResult {
            patterns: filtered_patterns,
            insights,
            text_units: units,
        })
    }
    
    /// Extract meaningful text units from input text
    fn extract_units(&self, text: &str) -> Result<Vec<TextUnit>> {
        // Simple paragraph-based extraction
        let paragraphs: Vec<&str> = text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        
        if paragraphs.is_empty() {
            // Fall back to line-based extraction for short texts
            return Ok(text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| TextUnit::new(line.trim().to_string()))
                .collect());
        }
        
        Ok(paragraphs.into_iter()
            .map(|p| TextUnit::new(p.trim().to_string()))
            .collect())
    }
    
    /// Apply semantic analysis to text units
    pub fn analyze_semantics(&self, text_units: &[TextUnit]) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        // This is a simple placeholder for semantic analysis
        // In a real implementation, this would use more sophisticated NLP techniques
        
        // Calculate average sentence length
        let total_sentences: usize = text_units.iter()
            .map(|unit| unit.content.split(['.', '!', '?']).count())
            .sum();
        
        let total_words: usize = text_units.iter()
            .map(|unit| unit.content.split_whitespace().count())
            .sum();
        
        if total_sentences > 0 {
            results.insert("avg_sentence_length".to_string(), 
                           total_words as f64 / total_sentences as f64);
        }
        
        // Calculate vocabulary diversity (unique words / total words)
        let mut unique_words = std::collections::HashSet::new();
        for unit in text_units {
            for word in unit.content.split_whitespace() {
                unique_words.insert(word.to_lowercase());
            }
        }
        
        if total_words > 0 {
            results.insert("vocabulary_diversity".to_string(), 
                           unique_words.len() as f64 / total_words as f64);
        }
        
        Ok(results)
    }
    
    /// Generate a summary from processing results
    pub fn generate_summary(&self, result: &ProcessingResult) -> String {
        let mut summary = String::new();
        
        // Add pattern information
        if !result.patterns.is_empty() {
            summary.push_str(&format!("Found {} significant patterns.\n", result.patterns.len()));
            
            // Add the top pattern description
            if let Some(top_pattern) = result.patterns.first() {
                summary.push_str(&format!("Primary pattern: {}\n", top_pattern.description));
            }
        } else {
            summary.push_str("No significant patterns found.\n");
        }
        
        // Add insights from metacognitive reflection
        if !result.insights.is_empty() {
            summary.push_str("\nKey insights:\n");
            for (i, insight) in result.insights.iter().take(3).enumerate() {
                summary.push_str(&format!("{}. {}\n", i+1, insight));
            }
        }
        
        summary
    }
    
    /// Extract patterns from text using sophisticated metacognitive reasoning
    pub fn extract_patterns(&mut self, text: &str) -> Result<Vec<crate::pattern::Pattern>> {
        // Process the text first
        let result = self.process(text)?;
        
        // Apply advanced pattern recognition
        let mut enhanced_patterns = Vec::new();
        
        // Group related patterns using clustering
        if !result.patterns.is_empty() {
            let mut clusters = self.cluster_patterns(&result.patterns);
            
            for cluster in clusters {
                if let Some(primary) = cluster.first() {
                    // Create an enhanced pattern with additional metadata
                    let mut enhanced = primary.clone();
                    
                    // Add metadata from cluster analysis
                    enhanced.add_annotation("cluster_size", &format!("{}", cluster.len()));
                    enhanced.add_annotation("derived_from", "metacognitive_analysis");
                    
                    // Calculate confidence boost based on cluster support
                    let confidence_boost = (cluster.len() as f64 * 0.05).min(0.3);
                    enhanced.confidence = (enhanced.confidence + confidence_boost).min(1.0);
                    
                    enhanced_patterns.push(enhanced);
                }
            }
        }
        
        // If no patterns found through clustering, return original patterns
        if enhanced_patterns.is_empty() {
            return Ok(result.patterns);
        }
        
        Ok(enhanced_patterns)
    }
    
    /// Cluster similar patterns together
    fn cluster_patterns(&self, patterns: &[crate::pattern::Pattern]) -> Vec<Vec<crate::pattern::Pattern>> {
        let mut clusters: Vec<Vec<crate::pattern::Pattern>> = Vec::new();
        
        // Simple clustering algorithm based on tag similarity
        for pattern in patterns {
            let mut added = false;
            
            // Try to add to existing cluster
            for cluster in &mut clusters {
                if let Some(first) = cluster.first() {
                    // Check if tags overlap significantly
                    let common_tags = pattern.tags.iter()
                        .filter(|tag| first.tags.contains(tag))
                        .count();
                    
                    let similarity = if first.tags.is_empty() || pattern.tags.is_empty() {
                        0.0
                    } else {
                        common_tags as f64 / first.tags.len().max(pattern.tags.len()) as f64
                    };
                    
                    if similarity > 0.5 {
                        cluster.push(pattern.clone());
                        added = true;
                        break;
                    }
                }
            }
            
            // Create new cluster if not added to existing one
            if !added {
                clusters.push(vec![pattern.clone()]);
            }
        }
        
        clusters
    }
    
    /// Apply metacognitive reasoning to discover complex semantic relationships
    pub fn discover_relationships(&mut self, text: &str) -> Result<Vec<(String, String, String)>> {
        // Process the text 
        self.process(text)?;
        
        // Extract relationship triplets (subject, predicate, object)
        let mut relationships = Vec::new();
        
        // Get nodes from the metacognitive engine
        let nodes = self.meta.get_nodes()?;
        let edges = self.meta.get_edges()?;
        
        // Convert edges to relationship triplets
        for edge in edges {
            // Find source and target nodes
            let source_node = nodes.iter().find(|n| n.id == edge.source);
            let target_node = nodes.iter().find(|n| n.id == edge.target);
            
            if let (Some(source), Some(target)) = (source_node, target_node) {
                // Create relationship triplet
                let subject = source.content.clone();
                let object = target.content.clone();
                let predicate = match edge.edge_type {
                    crate::pattern::metacognitive::MetaEdgeType::Causes => "causes".to_string(),
                    crate::pattern::metacognitive::MetaEdgeType::Contains => "contains".to_string(),
                    crate::pattern::metacognitive::MetaEdgeType::Implies => "implies".to_string(),
                    crate::pattern::metacognitive::MetaEdgeType::Supports => "supports".to_string(),
                    crate::pattern::metacognitive::MetaEdgeType::Contradicts => "contradicts".to_string(),
                    crate::pattern::metacognitive::MetaEdgeType::Relates => "relates to".to_string(),
                    _ => "is related to".to_string(),
                };
                
                relationships.push((subject, predicate, object));
            }
        }
        
        Ok(relationships)
    }
} 