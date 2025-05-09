use kwasa_kwasa::turbulance;
use kwasa_kwasa::turbulance::proposition::{Proposition, Motion};
use kwasa_kwasa::turbulance::datastructures::{TextGraph, ArgMap};
use kwasa_kwasa::text_unit::TextUnitType;

#[test]
fn test_proposition_creation() {
    // Create a new proposition
    let mut prop = Proposition::new("TestProposition");
    
    // Add some motions
    prop.add_motion_from_content("This is a test motion.", TextUnitType::Paragraph);
    prop.add_motion_from_content("This is another test motion.", TextUnitType::Paragraph);
    
    // Check motion count
    assert_eq!(prop.motions().len(), 2);
    
    // Check proposition name
    assert_eq!(prop.name(), "TestProposition");
}

#[test]
fn test_motion_analysis() {
    // Create a motion with content containing a capitalization error
    let mut motion = Motion::new("this sentence should start with a capital letter.", TextUnitType::Sentence);
    
    // Use the capitalization check method
    let cap_result = motion.capitalization();
    
    // Should have found an issue
    assert!(!cap_result.issues.is_empty());
    assert_eq!(cap_result.issues[0], "this sentence should start with a capital letter");
    
    // Check a custom pattern
    let check_result = motion.check_this_exactly("capital letter");
    assert!(check_result.found);
    assert_eq!(check_result.count, 1);
}

#[test]
fn test_text_graph() {
    // Create a new text graph
    let mut graph = TextGraph::new();
    
    // Create some motions
    let motion1 = Motion::new("First idea", TextUnitType::Paragraph);
    let motion2 = Motion::new("Second idea", TextUnitType::Paragraph);
    let motion3 = Motion::new("Third idea", TextUnitType::Paragraph);
    
    // Add nodes to the graph
    graph.add_node("n1", motion1);
    graph.add_node("n2", motion2);
    graph.add_node("n3", motion3);
    
    // Add connections
    graph.add_edge("n1", "n2", 0.8);
    graph.add_edge("n2", "n3", 0.7);
    
    // Find related nodes
    let related = graph.find_related("n1", 0.7);
    
    // Should have found one related node
    assert_eq!(related.len(), 1);
    assert_eq!(related[0].content(), "Second idea");
}

#[test]
fn test_arg_map() {
    // Create a new argument map
    let mut arg_map = ArgMap::new();
    
    // Add a main claim
    let main_claim = Motion::new("Climate change is a serious threat", TextUnitType::Paragraph);
    arg_map.add_claim("main", main_claim);
    
    // Add evidence
    let evidence1 = Motion::new("Global temperatures are rising", TextUnitType::Paragraph);
    let evidence2 = Motion::new("Extreme weather events are increasing", TextUnitType::Paragraph);
    
    arg_map.add_evidence("main", "e1", evidence1, 0.9);
    arg_map.add_evidence("main", "e2", evidence2, 0.8);
    
    // Add objection
    let objection1 = Motion::new("Natural climate cycles also exist", TextUnitType::Paragraph);
    arg_map.add_objection("main", "o1", objection1);
    
    // Evaluate the claim strength
    let strength = arg_map.evaluate_claim("main");
    
    // With 2 pieces of strong evidence and 1 objection, strength should be high
    assert!(strength > 0.7);
    
    // Check evidence count
    let evidence = arg_map.get_evidence("main");
    assert_eq!(evidence.len(), 2);
    
    // Check objection count
    let objections = arg_map.get_objections("main");
    assert_eq!(objections.len(), 1);
} 