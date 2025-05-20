use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::text_unit::boundary::{BoundaryType, TextUnit, detect_boundaries};
use crate::text_unit::operations::TextOperations;
use crate::text_unit::TextUnitRegistry;
use crate::text_unit::utils::string_similarity;

/// Enum representing the type of a node in the document hierarchy
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// The entire document
    Document,
    /// A major section (typically with a heading)
    Section,
    /// A subsection within a section
    Subsection,
    /// A paragraph
    Paragraph,
    /// A sentence
    Sentence,
    /// A phrase or clause
    Phrase,
    /// A word
    Word,
    /// A semantic block (represents a semantic unit regardless of structure)
    SemanticBlock,
    /// A custom node type with a name
    Custom(String),
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Document => write!(f, "Document"),
            NodeType::Section => write!(f, "Section"),
            NodeType::Subsection => write!(f, "Subsection"),
            NodeType::Paragraph => write!(f, "Paragraph"),
            NodeType::Sentence => write!(f, "Sentence"),
            NodeType::Phrase => write!(f, "Phrase"),
            NodeType::Word => write!(f, "Word"),
            NodeType::SemanticBlock => write!(f, "SemanticBlock"),
            NodeType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Convert BoundaryType to NodeType
impl From<BoundaryType> for NodeType {
    fn from(boundary_type: BoundaryType) -> Self {
        match boundary_type {
            BoundaryType::Document => NodeType::Document,
            BoundaryType::Sections => NodeType::Section,
            BoundaryType::Paragraphs => NodeType::Paragraph,
            BoundaryType::Sentences => NodeType::Sentence,
            BoundaryType::Words => NodeType::Word,
            BoundaryType::Characters => NodeType::Word, // Map to Word as closest equivalent
            BoundaryType::Semantic => NodeType::SemanticBlock,
            BoundaryType::Custom(name) => NodeType::Custom(name),
        }
    }
}

/// A node in the document hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    /// Node type
    node_type: NodeType,
    /// Content of this node as a text unit
    content: TextUnit,
    /// Metadata associated with this node
    metadata: HashMap<String, String>,
    /// Child nodes
    children: Vec<HierarchyNode>,
    /// Parent index, if any (None for root)
    parent_id: Option<usize>,
    /// Unique identifier for this node
    id: usize,
}

impl HierarchyNode {
    /// Create a new hierarchy node
    pub fn new(
        node_type: NodeType,
        content: TextUnit,
        id: usize,
        parent_id: Option<usize>,
    ) -> Self {
        Self {
            node_type,
            content,
            metadata: HashMap::new(),
            children: Vec::new(),
            parent_id,
            id,
        }
    }
    
    /// Get the node type
    pub fn node_type(&self) -> &NodeType {
        &self.node_type
    }
    
    /// Get the content as a text unit
    pub fn content(&self) -> &TextUnit {
        &self.content
    }
    
    /// Get mutable access to the content
    pub fn content_mut(&mut self) -> &mut TextUnit {
        &mut self.content
    }
    
    /// Get the content as a string reference
    pub fn text(&self) -> &str {
        &self.content.content
    }
    
    /// Get mutable access to the metadata
    pub fn metadata_mut(&mut self) -> &mut HashMap<String, String> {
        &mut self.metadata
    }
    
    /// Get reference to the metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// Get the children nodes
    pub fn children(&self) -> &[HierarchyNode] {
        &self.children
    }
    
    /// Get mutable access to the children
    pub fn children_mut(&mut self) -> &mut Vec<HierarchyNode> {
        &mut self.children
    }
    
    /// Get the parent ID, if any
    pub fn parent_id(&self) -> Option<usize> {
        self.parent_id
    }
    
    /// Get the ID of this node
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Set the parent ID
    pub fn set_parent_id(&mut self, parent_id: Option<usize>) {
        self.parent_id = parent_id;
    }
    
    /// Add a child node
    pub fn add_child(&mut self, child: HierarchyNode) {
        self.children.push(child);
    }
    
    /// Add a new metadata key-value pair
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Get a metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// Count the total number of nodes in the subtree (including this node)
    pub fn count_nodes(&self) -> usize {
        1 + self.children.iter().map(|child| child.count_nodes()).sum::<usize>()
    }
    
    /// Get the depth of this subtree (1 for a leaf node)
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|child| child.depth()).max().unwrap_or(0)
        }
    }
    
    /// Find nodes by type
    pub fn find_by_type(&self, node_type: &NodeType) -> Vec<&HierarchyNode> {
        let mut result = Vec::new();
        
        if &self.node_type == node_type {
            result.push(self);
        }
        
        for child in &self.children {
            result.extend(child.find_by_type(node_type));
        }
        
        result
    }
    
    /// Find nodes by content (substring match)
    pub fn find_by_content(&self, substring: &str) -> Vec<&HierarchyNode> {
        let mut result = Vec::new();
        
        if self.content.content.contains(substring) {
            result.push(self);
        }
        
        for child in &self.children {
            result.extend(child.find_by_content(substring));
        }
        
        result
    }
    
    /// Find a node by ID in this subtree
    pub fn find_by_id(&self, id: usize) -> Option<&HierarchyNode> {
        if self.id == id {
            return Some(self);
        }
        
        for child in &self.children {
            if let Some(node) = child.find_by_id(id) {
                return Some(node);
            }
        }
        
        None
    }
    
    /// Find the path to a node with the given ID
    pub fn path_to_node(&self, id: usize, current_path: &mut Vec<usize>) -> bool {
        current_path.push(self.id);
        
        if self.id == id {
            return true;
        }
        
        for child in &self.children {
            if child.path_to_node(id, current_path) {
                return true;
            }
        }
        
        current_path.pop();
        false
    }
    
    /// Apply an operation to this node and all its children
    pub fn apply_operation<F>(&mut self, operation: F)
    where
        F: Fn(&mut TextUnit) + Copy,
    {
        // Apply to this node
        operation(&mut self.content);
        
        // Apply to all children
        for child in &mut self.children {
            child.apply_operation(operation);
        }
    }
    
    /// Get nodes at a specific depth level (0 = this node, 1 = children, etc.)
    pub fn nodes_at_depth(&self, depth: usize) -> Vec<&HierarchyNode> {
        if depth == 0 {
            return vec![self];
        }
        
        let mut result = Vec::new();
        for child in &self.children {
            result.extend(child.nodes_at_depth(depth - 1));
        }
        result
    }
    
    /// Find sibling nodes (nodes with same parent)
    pub fn siblings<'a>(&self, hierarchy: &'a DocumentHierarchy) -> Vec<&'a HierarchyNode> {
        match self.parent_id {
            Some(parent_id) => {
                if let Some(parent) = hierarchy.find_node_by_id(parent_id) {
                    parent
                        .children()
                        .iter()
                        .filter(|node| node.id() != self.id)
                        .collect()
                } else {
                    Vec::new()
                }
            }
            None => Vec::new(), // Root node has no siblings
        }
    }
    
    /// Get the next sibling, if any
    pub fn next_sibling<'a>(&self, hierarchy: &'a DocumentHierarchy) -> Option<&'a HierarchyNode> {
        match self.parent_id {
            Some(parent_id) => {
                if let Some(parent) = hierarchy.find_node_by_id(parent_id) {
                    let children = parent.children();
                    let pos = children.iter().position(|node| node.id() == self.id)?;
                    if pos + 1 < children.len() {
                        Some(&children[pos + 1])
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None, // Root node has no siblings
        }
    }
    
    /// Get the previous sibling, if any
    pub fn prev_sibling<'a>(&self, hierarchy: &'a DocumentHierarchy) -> Option<&'a HierarchyNode> {
        match self.parent_id {
            Some(parent_id) => {
                if let Some(parent) = hierarchy.find_node_by_id(parent_id) {
                    let children = parent.children();
                    let pos = children.iter().position(|node| node.id() == self.id)?;
                    if pos > 0 {
                        Some(&children[pos - 1])
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            None => None, // Root node has no siblings
        }
    }
    
    /// Find ancestor nodes (parent, grandparent, etc.)
    pub fn ancestors<'a>(&self, hierarchy: &'a DocumentHierarchy) -> Vec<&'a HierarchyNode> {
        let mut result = Vec::new();
        let mut current_id = self.parent_id;
        
        while let Some(id) = current_id {
            if let Some(node) = hierarchy.find_node_by_id(id) {
                result.push(node);
                current_id = node.parent_id();
            } else {
                break;
            }
        }
        
        result
    }
    
    /// Find all descendants (children, grandchildren, etc.)
    pub fn descendants(&self) -> Vec<&HierarchyNode> {
        let mut result = Vec::new();
        
        for child in &self.children {
            result.push(child);
            result.extend(child.descendants());
        }
        
        result
    }
    
    /// Get the level (depth from root) of this node
    pub fn level(&self, hierarchy: &DocumentHierarchy) -> usize {
        self.ancestors(hierarchy).len()
    }
}

/// Document hierarchy representing the structure of a document
pub struct DocumentHierarchy {
    /// Root node of the hierarchy
    root: Option<HierarchyNode>,
    /// Next available ID for nodes
    next_id: usize,
}

impl DocumentHierarchy {
    /// Create a new empty document hierarchy
    pub fn new() -> Self {
        Self {
            root: None,
            next_id: 0,
        }
    }
    
    /// Create a document hierarchy from a text unit
    pub fn from_text_unit(text_unit: TextUnit) -> Self {
        let mut hierarchy = Self::new();
        
        // Create root node
        let root_id = hierarchy.next_id;
        hierarchy.next_id += 1;
        
        let node_type = match text_unit.unit_type {
            crate::text_unit::TextUnitType::Document => NodeType::Document,
            crate::text_unit::TextUnitType::Section => NodeType::Section,
            crate::text_unit::TextUnitType::Paragraph => NodeType::Paragraph,
            crate::text_unit::TextUnitType::Sentence => NodeType::Sentence,
            crate::text_unit::TextUnitType::Word => NodeType::Word,
            crate::text_unit::TextUnitType::Character => NodeType::Word,
            crate::text_unit::TextUnitType::Custom(_) => NodeType::Custom("Custom".to_string()),
        };
        
        let root = HierarchyNode::new(node_type, text_unit, root_id, None);
        hierarchy.root = Some(root);
        
        hierarchy.process_children(text_unit);
        
        hierarchy
    }
    
    /// Process children of a text unit and add them to the hierarchy
    fn process_children(&mut self, text_unit: TextUnit) {
        if let Some(root) = &mut self.root {
            // First process paragraphs
            if root.node_type() == &NodeType::Document {
                self.extract_and_add_children(&text_unit, root.id(), BoundaryType::Paragraphs);
            }
            
            // Then process sentences within paragraphs
            let paragraphs = self.find_nodes_by_type(&NodeType::Paragraph);
            for paragraph in &paragraphs {
                self.extract_and_add_children(&paragraph.content, paragraph.id(), BoundaryType::Sentences);
            }
            
            // Process sentences directly if no paragraphs
            if root.node_type() == &NodeType::Document && root.children().is_empty() {
                self.extract_and_add_children(&text_unit, root.id(), BoundaryType::Sentences);
            }
            
            // Process words within sentences
            let sentences = self.find_nodes_by_type(&NodeType::Sentence);
            for sentence in &sentences {
                self.extract_and_add_children(&sentence.content, sentence.id(), BoundaryType::Words);
            }
        }
    }
    
    /// Extract units of a certain type from parent and add them as children
    fn extract_and_add_children(
        &mut self,
        parent_unit: &TextUnit,
        parent_id: usize,
        child_boundary_type: BoundaryType,
    ) {
        // Create a temporary registry
        let mut registry = TextUnitRegistry::new();
        
        // Detect boundaries
        let child_ids = detect_boundaries(
            &parent_unit.content,
            child_boundary_type,
            &mut registry,
            None,
        );
        
        // Add children to parent
        if let Some(parent) = self.find_node_by_id_mut(parent_id) {
            for child_id in child_ids {
                if let Some(child_unit) = registry.get_unit(child_id) {
                    let node_id = self.next_id;
                    self.next_id += 1;
                    
                    let node_type = NodeType::from(child_boundary_type);
                    let node = HierarchyNode::new(
                        node_type,
                        child_unit.clone(),
                        node_id,
                        Some(parent_id),
                    );
                    
                    parent.add_child(node);
                }
            }
        }
    }
    
    /// Get reference to the root node
    pub fn root(&self) -> Option<&HierarchyNode> {
        self.root.as_ref()
    }
    
    /// Get mutable reference to the root node
    pub fn root_mut(&mut self) -> Option<&mut HierarchyNode> {
        self.root.as_mut()
    }
    
    /// Set the root node
    pub fn set_root(&mut self, root: HierarchyNode) {
        self.root = Some(root);
    }
    
    /// Find a node by ID
    pub fn find_node_by_id(&self, id: usize) -> Option<&HierarchyNode> {
        self.root.as_ref().and_then(|root| root.find_by_id(id))
    }
    
    /// Find a node by ID (mutable)
    pub fn find_node_by_id_mut(&mut self, id: usize) -> Option<&mut HierarchyNode> {
        // First check if the root matches
        if let Some(root) = self.root.as_mut() {
            if root.id() == id {
                return Some(root);
            }
            
            // Search within root using a non-recursive approach
            return Self::find_node_by_id_mut_helper(root, id);
        }
        
        None
    }
    
    /// Helper function to find a node by ID and get a mutable reference
    fn find_node_by_id_mut_helper(node: &mut HierarchyNode, id: usize) -> Option<&mut HierarchyNode> {
        // First, check if any direct child matches the ID
        let children_ids: Vec<(usize, usize)> = node.children_mut()
            .iter()
            .enumerate()
            .map(|(i, child)| (i, child.id()))
            .collect();
        
        // Check for direct matches
        for (i, child_id) in &children_ids {
            if *child_id == id {
                return Some(&mut node.children_mut()[*i]);
            }
        }
        
        // If no direct match, recurse one child at a time
        for (i, _) in children_ids {
            let child = &mut node.children_mut()[i];
            if let Some(found) = Self::find_node_by_id_mut_helper(child, id) {
                return Some(found);
            }
        }
        
        None
    }
    
    /// Find nodes by type
    pub fn find_nodes_by_type(&self, node_type: &NodeType) -> Vec<&HierarchyNode> {
        self.root.as_ref().map_or_else(Vec::new, |root| root.find_by_type(node_type))
    }
    
    /// Find nodes by content (substring match)
    pub fn find_nodes_by_content(&self, substring: &str) -> Vec<&HierarchyNode> {
        self.root.as_ref().map_or_else(Vec::new, |root| root.find_by_content(substring))
    }
    
    /// Find the path to a node by ID
    pub fn path_to_node(&self, id: usize) -> Vec<usize> {
        let mut path = Vec::new();
        if let Some(root) = &self.root {
            root.path_to_node(id, &mut path);
        }
        path
    }
    
    /// Get the text of the entire document
    pub fn text(&self) -> String {
        self.root.as_ref().map_or_else(String::new, |root| root.text().to_string())
    }
    
    /// Count the total number of nodes
    pub fn count_nodes(&self) -> usize {
        self.root.as_ref().map_or(0, |root| root.count_nodes())
    }
    
    /// Get the maximum depth (height) of the hierarchy
    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |root| root.depth())
    }
    
    /// Apply an operation to all nodes in the hierarchy
    pub fn apply_operation<F>(&mut self, operation: F)
    where
        F: Fn(&mut TextUnit) + Copy,
    {
        if let Some(root) = &mut self.root {
            root.apply_operation(operation);
        }
    }
    
    /// Get nodes at a specific level in the hierarchy (0 = root, 1 = root's children, etc.)
    pub fn nodes_at_level(&self, level: usize) -> Vec<&HierarchyNode> {
        self.root.as_ref().map_or_else(Vec::new, |root| root.nodes_at_depth(level))
    }
    
    /// Build a new hierarchy by organizing nodes according to semantic relationships
    pub fn build_semantic_hierarchy(&self) -> Self {
        // Start with a copy of the document node
        let mut new_hierarchy = DocumentHierarchy::new();
        
        if let Some(root) = &self.root {
            let new_root = HierarchyNode::new(
                NodeType::Document,
                root.content().clone(),
                0,
                None,
            );
            
            new_hierarchy.set_root(new_root);
            new_hierarchy.next_id = 1;
            
            // Group sentences by semantic similarity or topic
            let sentences = self.find_nodes_by_type(&NodeType::Sentence);
            
            // Simple algorithm: group consecutive sentences with similar content
            if !sentences.is_empty() {
                let mut current_block = Vec::new();
                let mut current_block_keywords = HashSet::new();
                
                for s in 0..sentences.len() {
                    let sentence = &sentences[s];
                    // Extract keywords from sentence
                    let keywords: HashSet<String> = sentence.text()
                        .split_whitespace()
                        .filter(|w| w.len() > 3) // Only words of significant length
                        .map(|w| w.to_lowercase())
                        .collect();
                    
                    // Calculate overlap with current block
                    let overlap = if current_block_keywords.is_empty() {
                        1.0 // First sentence always starts a new block
                    } else {
                        let common_count = keywords.intersection(&current_block_keywords).count();
                        if keywords.is_empty() {
                            0.0
                        } else {
                            common_count as f64 / keywords.len() as f64
                        }
                    };
                    
                    // If there's sufficient overlap, add to current block
                    if overlap > 0.2 {
                        current_block.push(sentence);
                        current_block_keywords.extend(keywords);
                    } else {
                        // Create a new semantic block from current sentences
                        if !current_block.is_empty() {
                            let sentences_slice: Vec<&HierarchyNode> = current_block.iter().cloned().collect();
                            new_hierarchy.add_semantic_block(&sentences_slice);
                        }
                        
                        // Start a new block with this sentence
                        current_block = vec![sentence];
                        current_block_keywords = keywords;
                    }
                }
                
                // Add the last block if it exists
                if !current_block.is_empty() {
                    let sentences_slice: Vec<&HierarchyNode> = current_block.iter().cloned().collect();
                    new_hierarchy.add_semantic_block(&sentences_slice);
                }
            }
        }
        
        new_hierarchy
    }
    
    /// Helper to add a semantic block to the hierarchy
    fn add_semantic_block(&mut self, sentences: &[&HierarchyNode]) {
        if let Some(root) = &mut self.root {
            // Combine the text of all sentences
            let combined_text = sentences.iter()
                .map(|s| s.text())
                .collect::<Vec<_>>()
                .join(" ");
            
            // Create a text unit for the semantic block
            let text_unit = TextUnit::new(
                combined_text,
                sentences.first().map_or(0, |s| s.content().start),
                sentences.last().map_or(0, |s| s.content().end),
                crate::text_unit::TextUnitType::Custom(0), // Using custom type for semantic block
                self.next_id,
            );
            
            // Create a new node
            let node_id = self.next_id;
            self.next_id += 1;
            
            let mut block_node = HierarchyNode::new(
                NodeType::SemanticBlock,
                text_unit,
                node_id,
                Some(root.id()),
            );
            
            // Add each sentence as a child of the semantic block
            for sentence in sentences {
                let child_id = self.next_id;
                self.next_id += 1;
                
                let child_node = HierarchyNode::new(
                    NodeType::Sentence,
                    sentence.content().clone(),
                    child_id,
                    Some(node_id),
                );
                
                block_node.add_child(child_node);
            }
            
            // Add to root
            root.add_child(block_node);
        }
    }
    
    /// Get all paths from root to leaf nodes
    pub fn all_paths(&self) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        
        if let Some(root) = &self.root {
            self.collect_paths(root, Vec::new(), &mut paths);
        }
        
        paths
    }
    
    /// Helper to collect all paths recursively
    fn collect_paths(&self, node: &HierarchyNode, current_path: Vec<usize>, paths: &mut Vec<Vec<usize>>) {
        let mut path = current_path;
        path.push(node.id());
        
        if node.children().is_empty() {
            // This is a leaf node
            paths.push(path);
        } else {
            // Continue recursion
            for child in node.children() {
                self.collect_paths(child, path.clone(), paths);
            }
        }
    }
    
    /// Compare two nodes for similarity (0.0-1.0)
    pub fn compare_nodes(&self, id1: usize, id2: usize) -> f64 {
        let node1 = self.find_node_by_id(id1);
        let node2 = self.find_node_by_id(id2);
        
        match (node1, node2) {
            (Some(n1), Some(n2)) => {
                // Use common string similarity function
                string_similarity(n1.text(), n2.text())
            },
            _ => 0.0,
        }
    }
    
    /// Traverse the hierarchy in breadth-first order
    pub fn breadth_first_traverse(&self) -> Vec<&HierarchyNode> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        
        if let Some(root) = &self.root {
            queue.push_back(root);
            
            while let Some(node) = queue.pop_front() {
                result.push(node);
                
                for child in node.children() {
                    queue.push_back(child);
                }
            }
        }
        
        result
    }
    
    /// Traverse the hierarchy in depth-first order
    pub fn depth_first_traverse(&self) -> Vec<&HierarchyNode> {
        let mut result = Vec::new();
        
        if let Some(root) = &self.root {
            self.depth_first_visit(root, &mut result);
        }
        
        result
    }
    
    /// Helper for depth-first traversal
    fn depth_first_visit<'a>(&'a self, node: &'a HierarchyNode, result: &mut Vec<&'a HierarchyNode>) {
        result.push(node);
        
        for child in node.children() {
            self.depth_first_visit(child, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_unit::boundary::BoundaryType;
    
    /// Helper to create a test text unit
    fn create_test_unit(content: &str, boundary_type: BoundaryType) -> TextUnit {
        TextUnit::new(
            content.to_string(),
            0,
            content.len(),
            match boundary_type {
                BoundaryType::Document => crate::text_unit::TextUnitType::Document,
                BoundaryType::Sections => crate::text_unit::TextUnitType::Section,
                BoundaryType::Paragraphs => crate::text_unit::TextUnitType::Paragraph,
                BoundaryType::Sentences => crate::text_unit::TextUnitType::Sentence,
                BoundaryType::Words => crate::text_unit::TextUnitType::Word,
                BoundaryType::Characters => crate::text_unit::TextUnitType::Character,
                _ => crate::text_unit::TextUnitType::Custom(0),
            },
            0,
        )
    }
    
    #[test]
    fn test_hierarchy_node_creation() {
        let unit = create_test_unit("Test content", BoundaryType::Document);
        let node = HierarchyNode::new(NodeType::Document, unit, 0, None);
        
        assert_eq!(node.node_type(), &NodeType::Document);
        assert_eq!(node.text(), "Test content");
        assert_eq!(node.id(), 0);
        assert_eq!(node.parent_id(), None);
        assert!(node.children().is_empty());
    }
    
    #[test]
    fn test_hierarchy_node_metadata() {
        let unit = create_test_unit("Test content", BoundaryType::Document);
        let mut node = HierarchyNode::new(NodeType::Document, unit, 0, None);
        
        node.add_metadata("key1".to_string(), "value1".to_string());
        node.add_metadata("key2".to_string(), "value2".to_string());
        
        assert_eq!(node.get_metadata("key1"), Some(&"value1".to_string()));
        assert_eq!(node.get_metadata("key2"), Some(&"value2".to_string()));
        assert_eq!(node.get_metadata("key3"), None);
    }
    
    #[test]
    fn test_document_hierarchy_creation() {
        let unit = create_test_unit("This is a test document. It has two sentences.", BoundaryType::Document);
        let hierarchy = DocumentHierarchy::from_text_unit(unit);
        
        assert!(hierarchy.root().is_some());
        assert_eq!(hierarchy.root().unwrap().node_type(), &NodeType::Document);
        assert!(hierarchy.count_nodes() > 1); // Should have created child nodes
    }
    
    #[test]
    fn test_hierarchy_navigation() {
        // Create a simple hierarchy manually
        let mut hierarchy = DocumentHierarchy::new();
        
        let doc_unit = create_test_unit("Document", BoundaryType::Document);
        let mut root = HierarchyNode::new(NodeType::Document, doc_unit, 0, None);
        
        let para1_unit = create_test_unit("Paragraph 1", BoundaryType::Paragraphs);
        let mut para1 = HierarchyNode::new(NodeType::Paragraph, para1_unit, 1, Some(0));
        
        let para2_unit = create_test_unit("Paragraph 2", BoundaryType::Paragraphs);
        let para2 = HierarchyNode::new(NodeType::Paragraph, para2_unit, 2, Some(0));
        
        let sent1_unit = create_test_unit("Sentence 1.", BoundaryType::Sentences);
        let sent1 = HierarchyNode::new(NodeType::Sentence, sent1_unit, 3, Some(1));
        
        let sent2_unit = create_test_unit("Sentence 2.", BoundaryType::Sentences);
        let sent2 = HierarchyNode::new(NodeType::Sentence, sent2_unit, 4, Some(1));
        
        para1.add_child(sent1);
        para1.add_child(sent2);
        
        root.add_child(para1);
        root.add_child(para2);
        
        hierarchy.set_root(root);
        
        // Test navigation
        assert_eq!(hierarchy.depth(), 3); // Doc -> Para -> Sent
        assert_eq!(hierarchy.count_nodes(), 5); // 1 doc + 2 para + 2 sent
        
        // Test finding nodes by type
        let paragraphs = hierarchy.find_nodes_by_type(&NodeType::Paragraphs);
        assert_eq!(paragraphs.len(), 2);
        
        let sentences = hierarchy.find_nodes_by_type(&NodeType::Sentences);
        assert_eq!(sentences.len(), 2);
        
        // Test path to node
        let path_to_sent2 = hierarchy.path_to_node(4);
        assert_eq!(path_to_sent2, vec![0, 1, 4]);
    }
    
    #[test]
    fn test_hierarchy_traversal() {
        // Create a simple hierarchy for testing
        let mut hierarchy = DocumentHierarchy::new();
        
        let doc_unit = create_test_unit("Document", BoundaryType::Document);
        let mut root = HierarchyNode::new(NodeType::Document, doc_unit, 0, None);
        
        let para1_unit = create_test_unit("Paragraph 1", BoundaryType::Paragraphs);
        let mut para1 = HierarchyNode::new(NodeType::Paragraph, para1_unit, 1, Some(0));
        
        let para2_unit = create_test_unit("Paragraph 2", BoundaryType::Paragraphs);
        let para2 = HierarchyNode::new(NodeType::Paragraph, para2_unit, 2, Some(0));
        
        let sent1_unit = create_test_unit("Sentence 1.", BoundaryType::Sentences);
        let sent1 = HierarchyNode::new(NodeType::Sentence, sent1_unit, 3, Some(1));
        
        let sent2_unit = create_test_unit("Sentence 2.", BoundaryType::Sentences);
        let sent2 = HierarchyNode::new(NodeType::Sentence, sent2_unit, 4, Some(1));
        
        para1.add_child(sent1);
        para1.add_child(sent2);
        
        root.add_child(para1);
        root.add_child(para2);
        
        hierarchy.set_root(root);
        
        // Test breadth-first traversal
        let bfs = hierarchy.breadth_first_traverse();
        assert_eq!(bfs.len(), 5);
        assert_eq!(bfs[0].id(), 0); // Root
        assert_eq!(bfs[1].id(), 1); // Para 1
        assert_eq!(bfs[2].id(), 2); // Para 2
        
        // Test depth-first traversal
        let dfs = hierarchy.depth_first_traverse();
        assert_eq!(dfs.len(), 5);
        assert_eq!(dfs[0].id(), 0); // Root
        assert_eq!(dfs[1].id(), 1); // Para 1
        assert_eq!(dfs[2].id(), 3); // Sent 1
        assert_eq!(dfs[3].id(), 4); // Sent 2
        assert_eq!(dfs[4].id(), 2); // Para 2
    }
    
    #[test]
    fn test_semantic_hierarchy_building() {
        let text = "This is about AI. Artificial intelligence is transforming how we work. \
                   Weather today is sunny. The forecast calls for clear skies all week.";
        let unit = create_test_unit(text, BoundaryType::Document);
        
        let hierarchy = DocumentHierarchy::from_text_unit(unit);
        let semantic_hierarchy = hierarchy.build_semantic_hierarchy();
        
        // Should have created semantic blocks
        assert!(semantic_hierarchy.find_nodes_by_type(&NodeType::SemanticBlock).len() > 0);
        
        // The semantic blocks should group related sentences
        assert!(semantic_hierarchy.count_nodes() < hierarchy.count_nodes());
    }
}
