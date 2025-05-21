use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::error::{Error, Result};
use crate::pattern::Pattern;

/// Represents a node in the metacognitive reasoning network
#[derive(Debug, Clone)]
pub struct MetaNode {
    /// Unique identifier for the node
    pub id: String,
    /// Content or representation of the node
    pub content: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    /// References to supporting evidence
    pub evidence: Vec<String>,
    /// Node type (concept, relationship, etc.)
    pub node_type: MetaNodeType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of metacognitive nodes
#[derive(Debug, Clone, PartialEq)]
pub enum MetaNodeType {
    /// Represents a fundamental concept
    Concept,
    /// Represents a relationship between concepts
    Relationship,
    /// Represents an inference drawn from other nodes
    Inference,
    /// Represents a hypothesis to be tested
    Hypothesis,
    /// Represents an observed pattern
    Pattern,
    /// Represents a feedback or reflection on the reasoning process
    Reflection,
}

/// Represents a connection between metacognitive nodes
#[derive(Debug, Clone)]
pub struct MetaEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Type of relationship
    pub edge_type: MetaEdgeType,
    /// Connection strength (0.0 - 1.0)
    pub strength: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of metacognitive edges
#[derive(Debug, Clone, PartialEq)]
pub enum MetaEdgeType {
    /// Causation relationship
    Causes,
    /// Part-whole relationship
    PartOf,
    /// Instance relationship
    InstanceOf,
    /// Inheritance relationship
    IsA,
    /// Opposition relationship
    Opposes,
    /// Similarity relationship
    SimilarTo,
    /// Sequential relationship
    Follows,
    /// Reflection relationship
    ReflectsOn,
}

/// The metacognitive reasoning engine
pub struct MetaCognitive {
    /// Nodes in the metacognitive network
    nodes: Arc<Mutex<HashMap<String, MetaNode>>>,
    /// Edges connecting nodes
    edges: Arc<Mutex<Vec<MetaEdge>>>,
    /// Patterns discovered through metacognition
    patterns: Arc<Mutex<Vec<Pattern>>>,
    /// Network activation threshold (0.0 - 1.0)
    activation_threshold: f64,
    /// Maximum nodes to consider in reasoning
    max_active_nodes: usize,
}

impl MetaCognitive {
    /// Creates a new metacognitive reasoning engine
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            edges: Arc::new(Mutex::new(Vec::new())),
            patterns: Arc::new(Mutex::new(Vec::new())),
            activation_threshold: 0.3,
            max_active_nodes: 100,
        }
    }

    /// Adds a node to the metacognitive network
    pub fn add_node(&self, node: MetaNode) -> Result<()> {
        let mut nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Adds an edge between nodes
    pub fn add_edge(&self, edge: MetaEdge) -> Result<()> {
        // Verify that source and target nodes exist
        let nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        if !nodes.contains_key(&edge.source) {
            return Err(Error::pattern(format!("Source node {} not found", edge.source)));
        }
        if !nodes.contains_key(&edge.target) {
            return Err(Error::pattern(format!("Target node {} not found", edge.target)));
        }
        drop(nodes);

        let mut edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
        edges.push(edge);
        Ok(())
    }

    /// Performs metacognitive reasoning to discover patterns
    pub fn reason(&self, focus_nodes: &[String]) -> Result<Vec<Pattern>> {
        // Start with focus nodes and spread activation
        let activated_nodes = self.spread_activation(focus_nodes)?;
        
        // Generate candidate patterns
        let candidates = self.generate_pattern_candidates(&activated_nodes)?;
        
        // Evaluate patterns
        let evaluated = self.evaluate_patterns(candidates)?;
        
        // Update internal patterns
        let mut patterns = self.patterns.lock().map_err(|_| Error::pattern("Failed to lock patterns"))?;
        for pattern in &evaluated {
            if !patterns.iter().any(|p| p.id == pattern.id) {
                patterns.push(pattern.clone());
            }
        }
        
        Ok(evaluated)
    }

    /// Spreads activation through the network from specified nodes
    fn spread_activation(&self, start_nodes: &[String]) -> Result<Vec<MetaNode>> {
        let nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
        
        // Track activation level of each node
        let mut activation: HashMap<String, f64> = HashMap::new();
        
        // Initialize activation with start nodes
        for node_id in start_nodes {
            if let Some(_) = nodes.get(node_id) {
                activation.insert(node_id.clone(), 1.0);
            }
        }
        
        // Spread activation for a fixed number of iterations
        for _ in 0..5 {
            let mut new_activation = activation.clone();
            
            // Propagate activation through edges
            for edge in edges.iter() {
                if let Some(source_activation) = activation.get(&edge.source) {
                    let propagated = source_activation * edge.strength;
                    
                    // Update target node's activation
                    let current = new_activation.entry(edge.target.clone()).or_insert(0.0);
                    *current = (*current + propagated).min(1.0);
                }
            }
            
            activation = new_activation;
        }
        
        // Extract activated nodes above threshold
        let mut activated_nodes: Vec<MetaNode> = activation
            .iter()
            .filter(|(_, &level)| level >= self.activation_threshold)
            .filter_map(|(id, _)| nodes.get(id).cloned())
            .collect();
        
        // Limit to max_active_nodes
        activated_nodes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        activated_nodes.truncate(self.max_active_nodes);
        
        Ok(activated_nodes)
    }

    /// Generates candidate patterns from activated nodes
    fn generate_pattern_candidates(&self, activated_nodes: &[MetaNode]) -> Result<Vec<Pattern>> {
        let mut candidates = Vec::new();
        
        // Group nodes by type
        let mut by_type: HashMap<MetaNodeType, Vec<&MetaNode>> = HashMap::new();
        for node in activated_nodes {
            by_type.entry(node.node_type.clone()).or_insert_with(Vec::new).push(node);
        }
        
        // Look for concept clusters
        if let Some(concepts) = by_type.get(&MetaNodeType::Concept) {
            if concepts.len() >= 3 {
                // Create pattern from related concepts
                let pattern = Pattern {
                    id: format!("pattern_concepts_{}", uuid::Uuid::new_v4()),
                    name: "Concept Cluster".to_string(),
                    description: format!("Cluster of {} related concepts", concepts.len()),
                    confidence: 0.7,
                    elements: concepts.iter().map(|n| n.content.clone()).collect(),
                    significance: 0.6,
                    source: Some("Metacognitive reasoning".to_string()),
                    tags: vec!["concept_cluster".to_string()],
                };
                candidates.push(pattern);
            }
        }
        
        // Look for causal chains
        let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
        let causal_edges: Vec<&MetaEdge> = edges.iter()
            .filter(|e| e.edge_type == MetaEdgeType::Causes)
            .filter(|e| 
                activated_nodes.iter().any(|n| n.id == e.source) && 
                activated_nodes.iter().any(|n| n.id == e.target)
            )
            .collect();
        
        if causal_edges.len() >= 2 {
            // Create pattern from causal chain
            let pattern = Pattern {
                id: format!("pattern_causal_{}", uuid::Uuid::new_v4()),
                name: "Causal Chain".to_string(),
                description: format!("Chain of {} causal relationships", causal_edges.len()),
                confidence: 0.75,
                elements: causal_edges.iter().flat_map(|e| vec![e.source.clone(), e.target.clone()]).collect(),
                significance: 0.8,
                source: Some("Metacognitive reasoning".to_string()),
                tags: vec!["causal_chain".to_string()],
            };
            candidates.push(pattern);
        }
        
        // Generate patterns from inferences
        if let Some(inferences) = by_type.get(&MetaNodeType::Inference) {
            for inference in inferences {
                let pattern = Pattern {
                    id: format!("pattern_inference_{}", uuid::Uuid::new_v4()),
                    name: "Inference Pattern".to_string(),
                    description: inference.content.clone(),
                    confidence: inference.confidence,
                    elements: inference.evidence.clone(),
                    significance: 0.65,
                    source: Some("Metacognitive reasoning".to_string()),
                    tags: vec!["inference".to_string()],
                };
                candidates.push(pattern);
            }
        }
        
        Ok(candidates)
    }

    /// Evaluates and ranks pattern candidates
    fn evaluate_patterns(&self, candidates: Vec<Pattern>) -> Result<Vec<Pattern>> {
        // For now, just return the candidates
        // In a more advanced implementation, this would evaluate and filter patterns
        Ok(candidates)
    }

    /// Reflects on the reasoning process and generates meta-level insights
    pub fn reflect(&self) -> Result<Vec<String>> {
        let nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
        let patterns = self.patterns.lock().map_err(|_| Error::pattern("Failed to lock patterns"))?;
        
        let mut insights = Vec::new();
        
        // Calculate network statistics
        let node_count = nodes.len();
        let edge_count = edges.len();
        let pattern_count = patterns.len();
        
        if node_count > 0 {
            let avg_confidence: f64 = nodes.values().map(|n| n.confidence).sum::<f64>() / node_count as f64;
            insights.push(format!("Average node confidence: {:.2}", avg_confidence));
        }
        
        if edge_count > 0 {
            let avg_strength: f64 = edges.iter().map(|e| e.strength).sum::<f64>() / edge_count as f64;
            insights.push(format!("Average edge strength: {:.2}", avg_strength));
        }
        
        // Identify knowledge gaps
        let concept_nodes: Vec<_> = nodes.values().filter(|n| n.node_type == MetaNodeType::Concept).collect();
        for node in &concept_nodes {
            let connected_edges: Vec<_> = edges.iter()
                .filter(|e| e.source == node.id || e.target == node.id)
                .collect();
            
            if connected_edges.is_empty() {
                insights.push(format!("Isolated concept detected: {}", node.content));
            }
        }
        
        // Analyze pattern distribution
        let pattern_types: HashMap<String, usize> = patterns.iter()
            .flat_map(|p| p.tags.clone())
            .fold(HashMap::new(), |mut map, tag| {
                *map.entry(tag).or_insert(0) += 1;
                map
            });
        
        if !pattern_types.is_empty() {
            let dominant_type = pattern_types.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(t, c)| format!("{} ({})", t, c))
                .unwrap_or_default();
            
            insights.push(format!("Dominant pattern type: {}", dominant_type));
        }
        
        // Suggest areas for exploration
        if pattern_count < 3 {
            insights.push("Consider generating more patterns through focused reasoning".to_string());
        }
        
        if edge_count < node_count {
            insights.push("Network connectivity is sparse; consider exploring relationships".to_string());
        }
        
        Ok(insights)
    }
}

impl Default for MetaCognitive {
    fn default() -> Self {
        Self::new()
    }
} 