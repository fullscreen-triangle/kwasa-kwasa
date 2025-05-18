use std::collections::{HashMap, VecDeque, BTreeMap};
use super::proposition::Motion;

/// TextGraph represents relationships between text components
pub struct TextGraph {
    /// Nodes in the graph (text units)
    nodes: HashMap<String, Motion>,
    
    /// Edges between nodes (relationships)
    edges: HashMap<String, Vec<(String, f64)>>,
}

impl TextGraph {
    /// Create a new text graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, id: &str, motion: Motion) {
        self.nodes.insert(id.to_string(), motion);
        
        // Initialize empty edge list
        if !self.edges.contains_key(id) {
            self.edges.insert(id.to_string(), Vec::new());
        }
    }
    
    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) {
        if let Some(edges) = self.edges.get_mut(from) {
            edges.push((to.to_string(), weight));
        }
    }
    
    /// Get a node by id
    pub fn get_node(&self, id: &str) -> Option<&Motion> {
        self.nodes.get(id)
    }
    
    /// Get all edges from a node
    pub fn get_edges(&self, id: &str) -> Vec<(String, f64)> {
        self.edges.get(id)
            .map(|edges| edges.clone())
            .unwrap_or_default()
    }
    
    /// Find related nodes based on similarity
    pub fn find_related(&self, id: &str, threshold: f64) -> Vec<&Motion> {
        let mut related = Vec::new();
        
        if let Some(edges) = self.edges.get(id) {
            for (to_id, weight) in edges {
                if *weight >= threshold {
                    if let Some(node) = self.nodes.get(to_id) {
                        related.push(node);
                    }
                }
            }
        }
        
        related
    }
}

/// ConceptChain represents a sequence of ideas with cause-effect relationships
pub struct ConceptChain {
    /// The sequence of ideas (could be causes or effects)
    sequence: VecDeque<(String, Motion)>,
    
    /// The relationships between ideas (cause-effect)
    relationships: HashMap<String, String>,
}

impl ConceptChain {
    /// Create a new concept chain
    pub fn new() -> Self {
        Self {
            sequence: VecDeque::new(),
            relationships: HashMap::new(),
        }
    }
    
    /// Add a concept to the chain
    pub fn add_concept(&mut self, id: &str, motion: Motion) {
        self.sequence.push_back((id.to_string(), motion));
    }
    
    /// Add a cause-effect relationship
    pub fn add_relationship(&mut self, cause: &str, effect: &str) {
        self.relationships.insert(cause.to_string(), effect.to_string());
    }
    
    /// Get the sequence of ideas
    pub fn sequence(&self) -> Vec<&Motion> {
        self.sequence.iter().map(|(_, motion)| motion).collect()
    }
    
    /// Get the effect of a cause
    pub fn effect_of(&self, cause: &str) -> Option<&Motion> {
        if let Some(effect_id) = self.relationships.get(cause) {
            for (id, motion) in &self.sequence {
                if id == effect_id {
                    return Some(motion);
                }
            }
        }
        
        None
    }
    
    /// Get the cause of an effect
    pub fn cause_of(&self, effect: &str) -> Option<&Motion> {
        for (cause, effect_id) in &self.relationships {
            if effect_id == effect {
                for (id, motion) in &self.sequence {
                    if id == cause {
                        return Some(motion);
                    }
                }
            }
        }
        
        None
    }
}

/// IdeaHierarchy organizes ideas in a hierarchical structure
pub struct IdeaHierarchy {
    /// The hierarchy of ideas
    hierarchy: BTreeMap<String, Vec<String>>,
    
    /// The content of each idea
    content: HashMap<String, Motion>,
}

impl IdeaHierarchy {
    /// Create a new idea hierarchy
    pub fn new() -> Self {
        Self {
            hierarchy: BTreeMap::new(),
            content: HashMap::new(),
        }
    }
    
    /// Add a root-level idea
    pub fn add_root(&mut self, id: &str, motion: Motion) {
        self.hierarchy.insert(id.to_string(), Vec::new());
        self.content.insert(id.to_string(), motion);
    }
    
    /// Add a child idea to a parent
    pub fn add_child(&mut self, parent: &str, child: &str, motion: Motion) {
        // Add to hierarchy
        if let Some(children) = self.hierarchy.get_mut(parent) {
            children.push(child.to_string());
        } else {
            self.hierarchy.insert(parent.to_string(), vec![child.to_string()]);
        }
        
        // Initialize child's entry in hierarchy
        if !self.hierarchy.contains_key(child) {
            self.hierarchy.insert(child.to_string(), Vec::new());
        }
        
        // Store content
        self.content.insert(child.to_string(), motion);
    }
    
    /// Get children of an idea
    pub fn get_children(&self, id: &str) -> Vec<&Motion> {
        let mut children = Vec::new();
        
        if let Some(child_ids) = self.hierarchy.get(id) {
            for child_id in child_ids {
                if let Some(motion) = self.content.get(child_id) {
                    children.push(motion);
                }
            }
        }
        
        children
    }
    
    /// Get content of an idea
    pub fn get_content(&self, id: &str) -> Option<&Motion> {
        self.content.get(id)
    }
    
    /// Get all roots
    pub fn get_roots(&self) -> Vec<&Motion> {
        let mut roots = Vec::new();
        
        // Find all nodes that are not children of any other node
        let is_child: Vec<String> = self.hierarchy.values()
            .flat_map(|children| children.clone())
            .collect();
            
        for (id, _) in &self.hierarchy {
            if !is_child.contains(id) {
                if let Some(motion) = self.content.get(id) {
                    roots.push(motion);
                }
            }
        }
        
        roots
    }
}

/// ArgMap represents an argumentation map with claims and evidence
pub struct ArgMap {
    /// Claims made in the argument
    claims: HashMap<String, Motion>,
    
    /// Evidence supporting claims
    evidence: HashMap<String, Vec<(String, f64)>>,
    
    /// Objections to claims
    objections: HashMap<String, Vec<String>>,
}

impl ArgMap {
    /// Create a new argument map
    pub fn new() -> Self {
        Self {
            claims: HashMap::new(),
            evidence: HashMap::new(),
            objections: HashMap::new(),
        }
    }
    
    /// Add a claim to the map
    pub fn add_claim(&mut self, id: &str, motion: Motion) {
        self.claims.insert(id.to_string(), motion);
        self.evidence.insert(id.to_string(), Vec::new());
        self.objections.insert(id.to_string(), Vec::new());
    }
    
    /// Add evidence supporting a claim
    pub fn add_evidence(&mut self, claim: &str, evidence_id: &str, motion: Motion, strength: f64) {
        // Store the evidence
        self.claims.insert(evidence_id.to_string(), motion);
        
        // Link to the claim
        if let Some(evidence_list) = self.evidence.get_mut(claim) {
            evidence_list.push((evidence_id.to_string(), strength));
        }
    }
    
    /// Add an objection to a claim
    pub fn add_objection(&mut self, claim: &str, objection_id: &str, motion: Motion) {
        // Store the objection
        self.claims.insert(objection_id.to_string(), motion);
        
        // Link to the claim
        if let Some(objection_list) = self.objections.get_mut(claim) {
            objection_list.push(objection_id.to_string());
        }
    }
    
    /// Get a claim by id
    pub fn get_claim(&self, id: &str) -> Option<&Motion> {
        self.claims.get(id)
    }
    
    /// Get evidence for a claim
    pub fn get_evidence(&self, claim: &str) -> Vec<(&Motion, f64)> {
        let mut result = Vec::new();
        
        if let Some(evidence_list) = self.evidence.get(claim) {
            for (evidence_id, strength) in evidence_list {
                if let Some(motion) = self.claims.get(evidence_id) {
                    result.push((motion, *strength));
                }
            }
        }
        
        result
    }
    
    /// Get objections to a claim
    pub fn get_objections(&self, claim: &str) -> Vec<&Motion> {
        let mut result = Vec::new();
        
        if let Some(objection_list) = self.objections.get(claim) {
            for objection_id in objection_list {
                if let Some(motion) = self.claims.get(objection_id) {
                    result.push(motion);
                }
            }
        }
        
        result
    }
    
    /// Evaluate the strength of a claim based on evidence and objections
    pub fn evaluate_claim(&self, claim: &str) -> f64 {
        let mut strength = 0.0;
        
        // Add evidence strength
        if let Some(evidence_list) = self.evidence.get(claim) {
            for (_, evidence_strength) in evidence_list {
                strength += evidence_strength;
            }
        }
        
        // Subtract for objections (simple model: each objection reduces strength by 0.1)
        if let Some(objection_list) = self.objections.get(claim) {
            strength -= objection_list.len() as f64 * 0.1;
        }
        
        // Normalize to 0.0-1.0 range
        strength.max(0.0).min(1.0)
    }
}

/// EvidenceNetwork implements a Bayesian-based framework for representing 
/// conflicting evidence with quantified uncertainty
pub struct EvidenceNetwork {
    /// Evidence nodes in the network
    nodes: HashMap<String, EvidenceNode>,
    
    /// Adjacency list of relationships
    adjacency: HashMap<String, Vec<(String, EdgeType, f64)>>,
    
    /// Belief values for nodes
    beliefs: HashMap<String, f64>,
    
    /// Uncertainty metrics for evidence propagation
    uncertainty: UncertaintyQuantifier,
}

/// Types of evidence nodes
#[derive(Debug, Clone)]
pub enum EvidenceNode {
    Molecule { 
        structure: String, 
        formula: String,
        motion: Motion 
    },
    Spectra { 
        peaks: Vec<(f64, f64)>, 
        retention_time: f64,
        motion: Motion 
    },
    GenomicFeature { 
        sequence: Vec<u8>, 
        position: Option<String>,
        motion: Motion 
    },
    Evidence { 
        source: String, 
        timestamp: String,
        motion: Motion 
    },
}

/// Types of edges in the evidence network
#[derive(Debug, Clone)]
pub enum EdgeType {
    Supports { strength: f64 },
    Contradicts { strength: f64 },
    PartOf,
    Catalyzes { rate: f64 },
    Transforms,
    BindsTo { affinity: f64 },
}

/// Quantifier for uncertainty in evidence relationships
#[derive(Debug, Clone)]
pub struct UncertaintyQuantifier {
    /// Global uncertainty parameters
    pub global_params: HashMap<String, f64>,
    /// Node-specific uncertainty
    pub node_uncertainty: HashMap<String, f64>,
    /// Edge-specific uncertainty
    pub edge_uncertainty: HashMap<(String, String), f64>,
}

impl EvidenceNetwork {
    /// Create a new evidence network
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            beliefs: HashMap::new(),
            uncertainty: UncertaintyQuantifier {
                global_params: HashMap::new(),
                node_uncertainty: HashMap::new(),
                edge_uncertainty: HashMap::new(),
            },
        }
    }
    
    /// Add a node to the network
    pub fn add_node(&mut self, id: &str, node: EvidenceNode) {
        self.nodes.insert(id.to_string(), node);
        
        // Initialize empty adjacency list
        if !self.adjacency.contains_key(id) {
            self.adjacency.insert(id.to_string(), Vec::new());
        }
        
        // Set default belief value
        self.beliefs.insert(id.to_string(), 0.5); // Neutral belief by default
    }
    
    /// Add an edge between nodes
    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: EdgeType, uncertainty: f64) {
        if let Some(edges) = self.adjacency.get_mut(from) {
            edges.push((to.to_string(), edge_type, uncertainty));
        }
    }
    
    /// Set belief value for a node
    pub fn set_belief(&mut self, id: &str, belief: f64) {
        // Ensure belief is between 0 and 1
        let belief = belief.max(0.0).min(1.0);
        self.beliefs.insert(id.to_string(), belief);
    }
    
    /// Get belief value for a node
    pub fn get_belief(&self, id: &str) -> Option<f64> {
        self.beliefs.get(id).copied()
    }
    
    /// Get a node by id
    pub fn get_node(&self, id: &str) -> Option<&EvidenceNode> {
        self.nodes.get(id)
    }
    
    /// Get all edges from a node
    pub fn get_edges(&self, id: &str) -> Vec<(&str, &EdgeType, f64)> {
        self.adjacency.get(id)
            .map(|edges| {
                edges.iter()
                    .map(|(to, edge_type, uncertainty)| 
                        (to.as_str(), edge_type, *uncertainty))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Propagate belief changes through the network using Bayesian rules
    pub fn propagate_beliefs(&mut self) {
        // Store the original beliefs to avoid order-dependent updates
        let original_beliefs = self.beliefs.clone();
        
        // For each node
        for (id, _) in &self.nodes {
            // Get the current belief
            let current_belief = *original_beliefs.get(id).unwrap_or(&0.5);
            
            // Get all nodes that influence this node
            for (from, edges) in &self.adjacency {
                for (to, edge_type, uncertainty) in edges {
                    if to != id {
                        continue;
                    }
                    
                    let from_belief = *original_beliefs.get(from).unwrap_or(&0.5);
                    
                    // Update belief based on edge type
                    let new_belief = match edge_type {
                        EdgeType::Supports { strength } => {
                            // Supporting evidence increases belief (weighted by strength and uncertainty)
                            let impact = from_belief * strength * (1.0 - uncertainty);
                            // Combine using a Bayesian-inspired approach
                            (current_belief + impact) / (1.0 + impact)
                        },
                        EdgeType::Contradicts { strength } => {
                            // Contradicting evidence decreases belief (weighted by strength and uncertainty)
                            let impact = (1.0 - from_belief) * strength * (1.0 - uncertainty);
                            // Combine using a Bayesian-inspired approach
                            (current_belief - impact).max(0.0)
                        },
                        EdgeType::PartOf => {
                            // Part-of relationship transfers belief
                            // If the parent is believed, the part is more likely to be correct
                            (current_belief + from_belief) / 2.0
                        },
                        EdgeType::Catalyzes { rate } => {
                            // Catalysis slightly increases belief based on rate
                            current_belief + (0.1 * rate * from_belief * (1.0 - uncertainty))
                        },
                        EdgeType::Transforms => {
                            // Transformation has complex impact on belief
                            // If A transforms into B, and A is likely, B is more likely
                            (current_belief + 0.3 * from_belief) / 1.3
                        },
                        EdgeType::BindsTo { affinity } => {
                            // Binding increases correlation between beliefs
                            (current_belief + affinity * from_belief) / (1.0 + affinity)
                        },
                    };
                    
                    // Update the belief
                    self.beliefs.insert(id.to_string(), new_belief);
                }
            }
        }
    }
    
    /// Calculate uncertainty bounds for a given node
    pub fn calculate_uncertainty_bounds(&self, id: &str) -> Option<(f64, f64)> {
        let belief = *self.beliefs.get(id)?;
        
        // Get node-specific uncertainty
        let node_uncertainty = self.uncertainty.node_uncertainty.get(id).copied().unwrap_or(0.1);
        
        // Calculate bounds
        let lower_bound = (belief - node_uncertainty).max(0.0);
        let upper_bound = (belief + node_uncertainty).min(1.0);
        
        Some((lower_bound, upper_bound))
    }
    
    /// Identify conflicting evidence
    pub fn identify_conflicts(&self) -> Vec<(String, String, f64)> {
        let mut conflicts = Vec::new();
        
        // For each node
        for (id, _) in &self.nodes {
            // Get edges that contradict this node
            for (from, edges) in &self.adjacency {
                for (to, edge_type, uncertainty) in edges {
                    if to != id {
                        continue;
                    }
                    
                    if let EdgeType::Contradicts { strength } = edge_type {
                        conflicts.push((from.clone(), id.clone(), *strength));
                    }
                }
            }
        }
        
        conflicts
    }
    
    /// Find the critical nodes that most impact a given conclusion
    pub fn sensitivity_analysis(&self, target_id: &str) -> Vec<(String, f64)> {
        let mut sensitivities = Vec::new();
        
        // For each node, calculate how much it impacts the target belief
        for (id, _) in &self.nodes {
            if id == target_id {
                continue;
            }
            
            // Simple approximation of sensitivity
            let mut impact = 0.0;
            
            // Direct impact through edges
            for (to, edge_type, uncertainty) in self.get_edges(id) {
                if to == target_id {
                    match edge_type {
                        EdgeType::Supports { strength } => impact += strength * (1.0 - uncertainty),
                        EdgeType::Contradicts { strength } => impact += strength * (1.0 - uncertainty),
                        _ => impact += 0.1, // Small impact for other edge types
                    }
                }
            }
            
            // Indirect impact (simplified)
            for (mid_id, _) in &self.nodes {
                if mid_id == id || mid_id == target_id {
                    continue;
                }
                
                // Check for paths: id -> mid_id -> target_id
                let has_first_edge = self.adjacency.get(id)
                    .map(|edges| edges.iter().any(|(to, _, _)| to == mid_id))
                    .unwrap_or(false);
                    
                let has_second_edge = self.adjacency.get(mid_id)
                    .map(|edges| edges.iter().any(|(to, _, _)| to == target_id))
                    .unwrap_or(false);
                    
                if has_first_edge && has_second_edge {
                    impact += 0.05; // Small indirect impact
                }
            }
            
            if impact > 0.0 {
                sensitivities.push((id.clone(), impact));
            }
        }
        
        // Sort by impact (highest first)
        sensitivities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        sensitivities
    }
    
    /// Get all nodes in the network
    pub fn nodes(&self) -> &HashMap<String, EvidenceNode> {
        &self.nodes
    }
    
    /// Get all edges in the network
    pub fn edges(&self) -> HashMap<&str, Vec<(&str, &EdgeType, f64)>> {
        let mut result = HashMap::new();
        
        for (source, edges) in &self.adjacency {
            let mapped_edges: Vec<(&str, &EdgeType, f64)> = edges.iter()
                .map(|(target, edge_type, uncertainty)| 
                    (target.as_str(), edge_type, *uncertainty))
                .collect();
            
            result.insert(source.as_str(), mapped_edges);
        }
        
        result
    }
} 