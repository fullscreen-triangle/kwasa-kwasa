use std::collections::HashMap;
use std::fmt;

use crate::text_unit::boundary::{BoundaryType, TextUnit};

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
            BoundaryType::Section => NodeType::Section,
            BoundaryType::Paragraph => NodeType::Paragraph,
            BoundaryType::Sentence => NodeType::Sentence,
            BoundaryType::Word => NodeType::Word,
            BoundaryType::Character => NodeType::Word, // Map to Word as closest equivalent
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
    
    /// Find node by ID
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
    
    /// Get the path from the root to this node
    pub fn path_to_node(&self, id: usize, current_path: &mut Vec<usize>) -> bool {
        if self.id == id {
            return true;
        }
        
        for child in &self.children {
            current_path.push(child.id);
            if child.path_to_node(id, current_path) {
                return true;
            }
            current_path.pop();
        }
        
        false
    }
}

/// Document hierarchy representation for structured text processing
#[derive(Debug, Clone)]
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
    
    /// Create a hierarchy from a text unit
    pub fn from_text_unit(text_unit: TextUnit) -> Self {
        let mut hierarchy = Self::new();
        let root_id = hierarchy.next_id;
        hierarchy.next_id += 1;
        
        let root_node = HierarchyNode::new(
            NodeType::from(text_unit.boundary_type.clone()),
            text_unit.clone(),
            root_id,
            None,
        );
        
        hierarchy.root = Some(root_node);
        hierarchy.process_children(text_unit);
        
        hierarchy
    }
    
    /// Process children of a text unit and add them to the hierarchy
    fn process_children(&mut self, text_unit: TextUnit) {
        // If the root doesn't exist yet, we can't process children
        if self.root.is_none() {
            return;
        }
        
        // The current implementation is simplified
        // A more sophisticated implementation would analyze the text unit
        // and identify its hierarchical structure
        
        // For now, we'll create a simple hierarchy based on boundary types:
        // Document > Section > Paragraph > Sentence > Word
        
        // Process based on the boundary type of the current unit
        match text_unit.boundary_type {
            BoundaryType::Document => {
                // For a document, extract sections, then paragraphs, then sentences, etc.
                self.extract_and_add_children(
                    &text_unit,
                    self.root.as_ref().unwrap().id(),
                    BoundaryType::Section,
                );
            }
            BoundaryType::Section => {
                // For a section, extract paragraphs, then sentences, etc.
                self.extract_and_add_children(
                    &text_unit,
                    self.root.as_ref().unwrap().id(),
                    BoundaryType::Paragraph,
                );
            }
            BoundaryType::Paragraph => {
                // For a paragraph, extract sentences, then words, etc.
                self.extract_and_add_children(
                    &text_unit,
                    self.root.as_ref().unwrap().id(),
                    BoundaryType::Sentence,
                );
            }
            BoundaryType::Sentence => {
                // For a sentence, extract words
                self.extract_and_add_children(
                    &text_unit,
                    self.root.as_ref().unwrap().id(),
                    BoundaryType::Word,
                );
            }
            _ => {
                // Other boundary types don't have predefined children
                // We could extract custom boundaries, but we'll skip for now
            }
        }
    }
    
    /// Extract child units of a specific boundary type and add them to the hierarchy
    fn extract_and_add_children(
        &mut self,
        parent_unit: &TextUnit,
        parent_id: usize,
        child_boundary_type: BoundaryType,
    ) {
        // Find the parent node
        let parent_node = match self.find_node_by_id_mut(parent_id) {
            Some(node) => node,
            None => return, // Parent not found
        };
        
        // Extract child units
        let child_units = parent_unit.extract_units(&child_boundary_type);
        
        // Add each child unit as a node
        for child_unit in child_units {
            let child_id = self.next_id;
            self.next_id += 1;
            
            let child_node = HierarchyNode::new(
                NodeType::from(child_unit.boundary_type.clone()),
                child_unit.clone(),
                child_id,
                Some(parent_id),
            );
            
            parent_node.add_child(child_node);
            
            // Recursively process children of this child
            // This is a simplified approach; more sophistication needed for real docs
            match child_unit.boundary_type {
                BoundaryType::Section => {
                    self.extract_and_add_children(&child_unit, child_id, BoundaryType::Paragraph);
                }
                BoundaryType::Paragraph => {
                    self.extract_and_add_children(&child_unit, child_id, BoundaryType::Sentence);
                }
                BoundaryType::Sentence => {
                    self.extract_and_add_children(&child_unit, child_id, BoundaryType::Word);
                }
                _ => {} // No further extraction for other types
            }
        }
    }
    
    /// Get the root node
    pub fn root(&self) -> Option<&HierarchyNode> {
        self.root.as_ref()
    }
    
    /// Get mutable access to the root node
    pub fn root_mut(&mut self) -> Option<&mut HierarchyNode> {
        self.root.as_mut()
    }
    
    /// Set the root node
    pub fn set_root(&mut self, root: HierarchyNode) {
        self.root = Some(root);
    }
    
    /// Find a node by ID
    pub fn find_node_by_id(&self, id: usize) -> Option<&HierarchyNode> {
        if let Some(root) = &self.root {
            root.find_by_id(id)
        } else {
            None
        }
    }
    
    /// Find a node by ID (mutable)
    pub fn find_node_by_id_mut(&mut self, id: usize) -> Option<&mut HierarchyNode> {
        // This is more complex because we need mutable access
        // We'll do a recursive search
        Self::find_node_by_id_mut_recursive(&mut self.root, id)
    }
    
    /// Helper for recursively finding a node by ID
    fn find_node_by_id_mut_recursive(
        node: &mut Option<HierarchyNode>,
        id: usize,
    ) -> Option<&mut HierarchyNode> {
        match node {
            Some(n) if n.id() == id => node.as_mut(),
            Some(n) => {
                for child in n.children_mut().iter_mut() {
                    if child.id() == id {
                        return Some(child);
                    }
                    
                    if let Some(found) = 
                        Self::find_node_by_id_mut_recursive(&mut Some(child.clone()), id) {
                        return Some(found);
                    }
                }
                None
            }
            None => None,
        }
    }
    
    /// Find all nodes of a specific type
    pub fn find_nodes_by_type(&self, node_type: &NodeType) -> Vec<&HierarchyNode> {
        if let Some(root) = &self.root {
            root.find_by_type(node_type)
        } else {
            Vec::new()
        }
    }
    
    /// Find all nodes containing a specific substring
    pub fn find_nodes_by_content(&self, substring: &str) -> Vec<&HierarchyNode> {
        if let Some(root) = &self.root {
            root.find_by_content(substring)
        } else {
            Vec::new()
        }
    }
    
    /// Get the path from root to a specific node
    pub fn path_to_node(&self, id: usize) -> Vec<usize> {
        let mut path = Vec::new();
        
        if let Some(root) = &self.root {
            path.push(root.id());
            root.path_to_node(id, &mut path);
        }
        
        path
    }
    
    /// Get the document text
    pub fn text(&self) -> String {
        if let Some(root) = &self.root {
            root.text().to_string()
        } else {
            String::new()
        }
    }
    
    /// Count the total number of nodes in the hierarchy
    pub fn count_nodes(&self) -> usize {
        if let Some(root) = &self.root {
            root.count_nodes()
        } else {
            0
        }
    }
    
    /// Get the maximum depth of the hierarchy
    pub fn depth(&self) -> usize {
        if let Some(root) = &self.root {
            root.depth()
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_unit::boundary::TextUnit;
    
    /// Create a test text unit with the given content and boundary type
    fn create_test_unit(content: &str, boundary_type: BoundaryType) -> TextUnit {
        TextUnit {
            content: content.to_string(),
            boundary_type,
            start_pos: 0,
            end_pos: content.len(),
            metadata: HashMap::new(),
        }
    }
    
    #[test]
    fn test_hierarchy_node_creation() {
        let text_unit = create_test_unit("Test content", BoundaryType::Paragraph);
        let node = HierarchyNode::new(NodeType::Paragraph, text_unit, 1, None);
        
        assert_eq!(node.node_type(), &NodeType::Paragraph);
        assert_eq!(node.text(), "Test content");
        assert_eq!(node.id(), 1);
        assert_eq!(node.parent_id(), None);
        assert!(node.children().is_empty());
    }
    
    #[test]
    fn test_hierarchy_node_metadata() {
        let text_unit = create_test_unit("Test content", BoundaryType::Paragraph);
        let mut node = HierarchyNode::new(NodeType::Paragraph, text_unit, 1, None);
        
        node.add_metadata("author".to_string(), "Test Author".to_string());
        node.add_metadata("category".to_string(), "Test".to_string());
        
        assert_eq!(node.get_metadata("author"), Some(&"Test Author".to_string()));
        assert_eq!(node.get_metadata("category"), Some(&"Test".to_string()));
        assert_eq!(node.get_metadata("nonexistent"), None);
    }
    
    #[test]
    fn test_document_hierarchy_creation() {
        let document_unit = create_test_unit("This is a test document. It has multiple sentences.", BoundaryType::Document);
        let hierarchy = DocumentHierarchy::from_text_unit(document_unit);
        
        assert!(hierarchy.root().is_some());
        if let Some(root) = hierarchy.root() {
            assert_eq!(root.node_type(), &NodeType::Document);
            assert_eq!(
                root.text(),
                "This is a test document. It has multiple sentences."
            );
        }
    }
    
    // More tests would be added for the full implementation
}
