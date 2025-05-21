use kwasa_kwasa::{error::{Error, ErrorReporter}, pattern::metacognitive::MetaCognitive};
use std::collections::HashMap;

#[test]
fn test_error_reporter() {
    // Create a simple source for testing
    let source = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n";
    
    // Create error reporter with the source
    let mut reporter = ErrorReporter::default().with_source(source.to_string());
    
    // Add a parse error
    reporter.add_error(Error::parse("Invalid syntax", 3, 5));
    
    // Add a semantic error
    reporter.add_error(Error::semantic("Undefined variable"));
    
    // Generate the report
    let report = reporter.report();
    
    // Verify report contains expected information
    assert!(report.contains("error(s) found"), "Report should mention errors found");
    assert!(report.contains("Invalid syntax"), "Report should contain error message");
    assert!(report.contains("Line 3"), "Report should contain line number");
    assert!(report.contains("Undefined variable"), "Report should contain semantic error");
}

#[test]
fn test_error_propagation() {
    // Test that errors are properly propagated
    fn inner_function() -> kwasa_kwasa::error::Result<()> {
        Err(Error::runtime("Inner error"))
    }
    
    fn middle_function() -> kwasa_kwasa::error::Result<()> {
        inner_function()?;
        Ok(())
    }
    
    fn outer_function() -> kwasa_kwasa::error::Result<()> {
        middle_function()?;
        Ok(())
    }
    
    // Verify error propagation
    let result = outer_function();
    assert!(result.is_err());
    
    // Convert to string and check message
    let err_string = format!("{}", result.unwrap_err());
    assert!(err_string.contains("Inner error"), "Error should propagate with original message");
}

#[test]
fn test_metacognitive_integration() {
    // Test integration with error handling
    let meta = MetaCognitive::new();
    
    // Attempt to reason with non-existent nodes should produce error
    let result = meta.reason(&["nonexistent".to_string()]);
    
    // Result should be ok (empty patterns, not an error)
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    
    // Reflect should succeed even with empty network
    let reflections = meta.reflect();
    assert!(reflections.is_ok());
    
    // Try to create an invalid edge and check error
    let edge_result = meta.add_edge(kwasa_kwasa::pattern::metacognitive::MetaEdge {
        source: "source".to_string(),
        target: "target".to_string(),
        edge_type: kwasa_kwasa::pattern::metacognitive::MetaEdgeType::Causes,
        strength: 0.5,
        metadata: HashMap::new(),
    });
    
    assert!(edge_result.is_err());
    let err = edge_result.unwrap_err();
    match err {
        Error::Pattern(msg) => {
            assert!(msg.contains("not found"), "Error should mention node not found");
        },
        _ => panic!("Expected Pattern error type"),
    }
} 