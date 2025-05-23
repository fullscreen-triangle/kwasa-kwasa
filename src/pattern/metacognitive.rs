use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
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
                    id: format!("pattern_concepts_{}", Uuid::new_v4()),
                    name: "Concept Cluster".to_string(),
                    description: format!("Cluster of {} related concepts", concepts.len()),
                    confidence: 0.7,
                    elements: concepts.iter().map(|n| n.content.clone()).collect(),
                    significance: 0.6,
                    source: Some("Metacognitive reasoning".to_string()),
                    tags: vec!["concept_cluster".to_string()],
                    pattern_type: "concept_cluster".to_string(),
                    supporting_evidence: concepts.iter().flat_map(|n| n.evidence.clone()).collect(),
                    metadata: HashMap::new(),
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
                id: format!("pattern_causal_{}", Uuid::new_v4()),
                name: "Causal Chain".to_string(),
                description: format!("Chain of {} causal relationships", causal_edges.len()),
                confidence: 0.75,
                elements: causal_edges.iter().flat_map(|e| vec![e.source.clone(), e.target.clone()]).collect(),
                significance: 0.8,
                source: Some("Metacognitive reasoning".to_string()),
                tags: vec!["causal_chain".to_string()],
                pattern_type: "causal_chain".to_string(),
                supporting_evidence: causal_edges.iter().map(|e| format!("{} causes {}", e.source, e.target)).collect(),
                metadata: HashMap::new(),
            };
            candidates.push(pattern);
        }
        
        // Generate patterns from inferences
        if let Some(inferences) = by_type.get(&MetaNodeType::Inference) {
            for inference in inferences {
                let pattern = Pattern {
                    id: format!("pattern_inference_{}", Uuid::new_v4()),
                    name: "Inference Pattern".to_string(),
                    description: inference.content.clone(),
                    confidence: inference.confidence,
                    elements: inference.evidence.clone(),
                    significance: 0.65,
                    source: Some("Metacognitive reasoning".to_string()),
                    tags: vec!["inference".to_string()],
                    pattern_type: "inference".to_string(),
                    supporting_evidence: inference.evidence.clone(),
                    metadata: HashMap::new(),
                };
                candidates.push(pattern);
            }
        }
        
        Ok(candidates)
    }

    /// Evaluates candidate patterns and selects the most relevant ones
    fn evaluate_patterns(&self, candidates: Vec<Pattern>) -> Result<Vec<Pattern>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        
        // Create a score for each pattern based on:
        // 1. Confidence level
        // 2. Supporting evidence
        // 3. Consistency with existing patterns
        // 4. Novelty/interestingness
        
        let mut scored_patterns: Vec<(Pattern, f64)> = Vec::new();
        let existing_patterns = self.patterns.lock().map_err(|_| Error::pattern("Failed to lock patterns"))?;
        
        for pattern in candidates {
            let mut score = pattern.significance;
            
            // Add bonus for patterns with more evidence
            score += 0.1 * pattern.supporting_evidence.len() as f64;
            
            // Consistency check with existing patterns
            for existing in existing_patterns.iter() {
                // Reduce score if too similar to existing patterns (avoid redundancy)
                if pattern.is_similar_to(existing) {
                    score -= 0.2;
                }
                
                // Increase score if it complements existing patterns
                if pattern.complements(existing) {
                    score += 0.15;
                }
                
                // Decrease score if it contradicts existing high-confidence patterns
                if pattern.contradicts(existing) && existing.significance > 0.7 {
                    score -= 0.3;
                }
            }
            
            // Bonus for novel, interesting patterns
            if !existing_patterns.iter().any(|p| p.pattern_type == pattern.pattern_type) {
                score += 0.2; // Novelty bonus
            }
            
            // Ensure score is in valid range
            score = score.max(0.0).min(1.0);
            
            scored_patterns.push((pattern, score));
        }
        
        // Sort by score in descending order
        scored_patterns.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top patterns (limit to 10 for efficiency)
        let selected: Vec<Pattern> = scored_patterns.into_iter()
            .take(10)
            .map(|(pattern, _)| pattern)
            .collect();
            
        Ok(selected)
    }
    
    /// Generates insights based on current patterns and node activations
    pub fn generate_insights(&self) -> Result<Vec<String>> {
        let patterns = self.patterns.lock().map_err(|_| Error::pattern("Failed to lock patterns"))?;
        let nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        
        let mut insights = Vec::new();
        
        // Generate insights based on high-confidence patterns
        for pattern in patterns.iter().filter(|p| p.significance > 0.7) {
            insights.push(format!("High confidence pattern detected: {}", pattern.description));
            
            // Extract implications from pattern
            if let Some(implications) = &pattern.metadata.get("implications") {
                insights.push(format!("Implication: {}", implications));
            }
        }
        
        // Generate insights about connected concepts
        let concept_nodes: Vec<&MetaNode> = nodes.values()
            .filter(|n| n.node_type == MetaNodeType::Concept && n.confidence > 0.6)
            .collect();
            
        if concept_nodes.len() >= 2 {
            for i in 0..concept_nodes.len() {
                for j in (i+1)..concept_nodes.len() {
                    let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
                    
                    // Check if these concepts are directly connected
                    let connection = edges.iter().find(|e| 
                        (e.source == concept_nodes[i].id && e.target == concept_nodes[j].id) ||
                        (e.source == concept_nodes[j].id && e.target == concept_nodes[i].id)
                    );
                    
                    if let Some(edge) = connection {
                        insights.push(format!(
                            "Found strong relationship ({:?}) between '{}' and '{}'",
                            edge.edge_type,
                            concept_nodes[i].content,
                            concept_nodes[j].content
                        ));
                    }
                }
            }
        }
        
        // Generate insights about potential gaps or missing connections
        if !concept_nodes.is_empty() {
            let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
            
            // Identify isolated concepts (no incoming or outgoing edges)
            for node in &concept_nodes {
                let has_connections = edges.iter().any(|e| 
                    e.source == node.id || e.target == node.id
                );
                
                if !has_connections {
                    insights.push(format!(
                        "Isolated concept detected: '{}'. Consider exploring its relationships.",
                        node.content
                    ));
                }
            }
        }
        
        Ok(insights)
    }

    /// Performs self-reflection on the reasoning process
    pub fn reflect(&self) -> Result<Vec<String>> {
        let mut reflections = Vec::new();
        
        // Reflect on the pattern distribution
        let patterns = self.patterns.lock().map_err(|_| Error::pattern("Failed to lock patterns"))?;
        let pattern_types: HashMap<_, _> = patterns.iter()
            .map(|p| (&p.pattern_type, 1))
            .fold(HashMap::new(), |mut acc, (t, c)| {
                *acc.entry(t).or_insert(0) += c;
                acc
            });
            
        if !pattern_types.is_empty() {
            // Check for pattern type bias
            let total = patterns.len();
            let most_common = pattern_types.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(t, c)| (t, c));
                
            if let Some((dominant_type, count)) = most_common {
                let percentage = (*count as f64 / total as f64) * 100.0;
                if percentage > 70.0 {
                    reflections.push(format!(
                        "Potential reasoning bias: {:?} patterns represent {:.1}% of all patterns.",
                        dominant_type, percentage
                    ));
                }
            }
        }
        
        // Reflect on confidence distribution
        let confidence_values: Vec<f64> = patterns.iter().map(|p| p.significance).collect();
        if !confidence_values.is_empty() {
            // Calculate mean confidence
            let mean = confidence_values.iter().sum::<f64>() / confidence_values.len() as f64;
            
            if mean > 0.8 {
                reflections.push(
                    "High overall pattern confidence may indicate overconfidence or insufficient skepticism."
                    .to_string()
                );
            } else if mean < 0.3 {
                reflections.push(
                    "Low overall pattern confidence suggests uncertainty or lack of strong evidence."
                    .to_string()
                );
            }
        }
        
        // Reflect on network characteristics
        let nodes = self.nodes.lock().map_err(|_| Error::pattern("Failed to lock nodes"))?;
        let edges = self.edges.lock().map_err(|_| Error::pattern("Failed to lock edges"))?;
        
        // Check network connectivity
        if !nodes.is_empty() && !edges.is_empty() {
            let edge_to_node_ratio = edges.len() as f64 / nodes.len() as f64;
            
            if edge_to_node_ratio < 0.5 {
                reflections.push(
                    "Low connectivity in the metacognitive network. Consider exploring more relationships."
                    .to_string()
                );
            } else if edge_to_node_ratio > 5.0 {
                reflections.push(
                    "Very high connectivity may indicate over-generalization or spurious connections."
                    .to_string()
                );
            }
        }
        
        // Suggest potential improvements to reasoning process
        reflections.push(
            "Consider gathering more evidence for emerging patterns to increase confidence levels."
            .to_string()
        );
        
        if reflections.is_empty() {
            reflections.push("No significant issues found in the reasoning process.".to_string());
        }
        
        Ok(reflections)
    }
}

impl Default for MetaCognitive {
    fn default() -> Self {
        Self::new()
    }
} 