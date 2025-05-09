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