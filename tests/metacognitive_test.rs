use kwasa_kwasa::pattern::metacognitive::{
    MetaCognitive, MetaNode, MetaEdge, MetaNodeType, MetaEdgeType
};
use std::collections::HashMap;

#[test]
fn test_metacognitive_node_creation() {
    // Create a metacognitive engine
    let meta = MetaCognitive::new();
    
    // Create a node
    let node = MetaNode {
        id: "concept1".to_string(),
        content: "The sky is blue".to_string(),
        confidence: 0.9,
        evidence: vec!["observation".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    // Add the node
    let result = meta.add_node(node);
    assert!(result.is_ok(), "Should be able to add a node");
}

#[test]
fn test_metacognitive_edge_creation() {
    // Create a metacognitive engine
    let meta = MetaCognitive::new();
    
    // Create nodes
    let node1 = MetaNode {
        id: "concept1".to_string(),
        content: "Rain".to_string(),
        confidence: 0.9,
        evidence: vec!["observation".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    let node2 = MetaNode {
        id: "concept2".to_string(),
        content: "Wet ground".to_string(),
        confidence: 0.9,
        evidence: vec!["observation".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    // Add the nodes
    let _ = meta.add_node(node1);
    let _ = meta.add_node(node2);
    
    // Create an edge
    let edge = MetaEdge {
        source: "concept1".to_string(),
        target: "concept2".to_string(),
        edge_type: MetaEdgeType::Causes,
        strength: 0.8,
        metadata: HashMap::new(),
    };
    
    // Add the edge
    let result = meta.add_edge(edge);
    assert!(result.is_ok(), "Should be able to add an edge between existing nodes");
}

#[test]
fn test_metacognitive_invalid_edge() {
    // Create a metacognitive engine
    let meta = MetaCognitive::new();
    
    // Create a node
    let node = MetaNode {
        id: "concept1".to_string(),
        content: "Rain".to_string(),
        confidence: 0.9,
        evidence: vec!["observation".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    // Add the node
    let _ = meta.add_node(node);
    
    // Create an invalid edge
    let edge = MetaEdge {
        source: "concept1".to_string(),
        target: "nonexistent".to_string(),
        edge_type: MetaEdgeType::Causes,
        strength: 0.8,
        metadata: HashMap::new(),
    };
    
    // Try to add the edge
    let result = meta.add_edge(edge);
    assert!(result.is_err(), "Should not be able to add an edge to a nonexistent node");
}

#[test]
fn test_metacognitive_reasoning() {
    // Create a metacognitive engine
    let meta = MetaCognitive::new();
    
    // Create nodes
    let nodes = vec![
        MetaNode {
            id: "cloud".to_string(),
            content: "Dark clouds".to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: MetaNodeType::Concept,
            metadata: HashMap::new(),
        },
        MetaNode {
            id: "rain".to_string(),
            content: "Rain".to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: MetaNodeType::Concept,
            metadata: HashMap::new(),
        },
        MetaNode {
            id: "ground".to_string(),
            content: "Wet ground".to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: MetaNodeType::Concept,
            metadata: HashMap::new(),
        },
        MetaNode {
            id: "thunder".to_string(),
            content: "Thunder sound".to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: MetaNodeType::Concept,
            metadata: HashMap::new(),
        },
        MetaNode {
            id: "lightning".to_string(),
            content: "Lightning flash".to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: MetaNodeType::Concept,
            metadata: HashMap::new(),
        },
    ];
    
    // Add all nodes
    for node in nodes {
        let _ = meta.add_node(node);
    }
    
    // Create edges
    let edges = vec![
        MetaEdge {
            source: "cloud".to_string(),
            target: "rain".to_string(),
            edge_type: MetaEdgeType::Causes,
            strength: 0.8,
            metadata: HashMap::new(),
        },
        MetaEdge {
            source: "rain".to_string(),
            target: "ground".to_string(),
            edge_type: MetaEdgeType::Causes,
            strength: 0.9,
            metadata: HashMap::new(),
        },
        MetaEdge {
            source: "lightning".to_string(),
            target: "thunder".to_string(),
            edge_type: MetaEdgeType::Causes,
            strength: 0.7,
            metadata: HashMap::new(),
        },
        MetaEdge {
            source: "cloud".to_string(),
            target: "lightning".to_string(),
            edge_type: MetaEdgeType::Causes,
            strength: 0.6,
            metadata: HashMap::new(),
        },
    ];
    
    // Add all edges
    for edge in edges {
        let _ = meta.add_edge(edge);
    }
    
    // Perform reasoning starting from cloud
    let patterns = meta.reason(&["cloud".to_string()]);
    assert!(patterns.is_ok(), "Reasoning should complete without errors");
    
    let found_patterns = patterns.unwrap();
    assert!(!found_patterns.is_empty(), "Should find at least one pattern");
    
    // At least one pattern should be a causal chain
    let has_causal_chain = found_patterns.iter().any(|p| 
        p.tags.contains(&"causal_chain".to_string())
    );
    assert!(has_causal_chain, "Should find a causal chain pattern");
}

#[test]
fn test_metacognitive_reflection() {
    // Create a metacognitive engine with some test data
    let meta = MetaCognitive::new();
    
    // Create some nodes and edges (simplified test setup)
    let node1 = MetaNode {
        id: "concept1".to_string(),
        content: "Test concept 1".to_string(),
        confidence: 0.9,
        evidence: vec!["test".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    let node2 = MetaNode {
        id: "concept2".to_string(),
        content: "Test concept 2".to_string(),
        confidence: 0.8,
        evidence: vec!["test".to_string()],
        node_type: MetaNodeType::Concept,
        metadata: HashMap::new(),
    };
    
    // Add the nodes
    let _ = meta.add_node(node1);
    let _ = meta.add_node(node2);
    
    // Create an edge
    let edge = MetaEdge {
        source: "concept1".to_string(),
        target: "concept2".to_string(),
        edge_type: MetaEdgeType::Causes,
        strength: 0.7,
        metadata: HashMap::new(),
    };
    
    // Add the edge
    let _ = meta.add_edge(edge);
    
    // Get reflections
    let reflections = meta.reflect();
    assert!(reflections.is_ok(), "Reflection should complete without errors");
    
    let insights = reflections.unwrap();
    assert!(!insights.is_empty(), "Should generate at least one insight");
    
    // Check that some insights contain expected text about confidence
    let has_confidence_insight = insights.iter().any(|i| i.contains("confidence"));
    assert!(has_confidence_insight, "Should have an insight about confidence");
} 