use kwasa_kwasa::{error::{Error, ErrorReporter}, pattern::metacognitive::MetaCognitive};
use std::collections::HashMap;
use kwasa_kwasa::turbulance::context::Context;

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

#[test]
fn test_enhanced_error_reporting() {
    // Create test source with deliberate errors
    let source = "Line 1: This is fine\nLine 2: This has a syntax error ]\nLine 3: This is fine\nLine 4: Another error here $$$\nLine 5: This is fine\n";
    
    // Create error reporter with source context
    let mut reporter = ErrorReporter::default().with_source(source.to_string());
    
    // Add parse errors with specific locations
    reporter.add_error(Error::Parse {
        message: "Unexpected closing bracket".to_string(),
        line: 2,
        column: 35,
    });
    
    reporter.add_error(Error::Parse {
        message: "Invalid token sequence".to_string(),
        line: 4,
        column: 19,
    });
    
    // Generate the report
    let report = reporter.report();
    
    // Check for line context in the error report
    assert!(report.contains("Line 2:"), "Report should show the line with the error");
    assert!(report.contains("syntax error"), "Report should show context around error");
    assert!(report.contains("^"), "Report should include position markers");
    
    // Verify surrounding context
    assert!(report.contains("Line 1:"), "Report should show line before error");
    assert!(report.contains("Line 3:"), "Report should show line after error");
    
    // Verify recovery feasibility reporting
    assert!(!reporter.recover(), "Should indicate recovery not possible for parse errors");
}

#[test]
fn test_context_error_integration() {
    // Create a context for execution
    let mut context = Context::new();
    
    // Start execution tracking
    context.begin_execution();
    
    // Simulate function calls with tracking
    assert!(context.enter_function("main").is_ok());
    assert!(context.enter_function("process_data").is_ok());
    
    // Add an error during execution
    context.add_error(Error::Runtime("Failed to process data".to_string()));
    
    // Continue execution (error doesn't halt progress)
    assert!(context.enter_function("cleanup").is_ok());
    context.exit_function(); // Exit cleanup
    context.exit_function(); // Exit process_data
    context.exit_function(); // Exit main
    
    // End execution
    context.end_execution();
    
    // Verify error was recorded
    assert!(context.error_reporter().has_errors());
    assert_eq!(context.error_reporter().error_count(), 1);
    
    // Verify call stack tracking
    let performance_report = context.get_performance_report();
    assert!(performance_report.contains("Execution time:"), "Performance report should include timing");
}

#[test]
fn test_error_recovery_workflow() {
    // Create a context
    let mut context = Context::new();
    
    // Begin execution
    context.begin_execution();
    
    // Record metrics
    context.record_metric("memory_usage_mb", 128.5);
    context.record_metric("query_time_ms", 45.2);
    
    // Simulate some non-fatal errors
    context.add_error(Error::Evidence("Missing optional data".to_string()));
    
    // Enter recovery mode
    context.enter_recovery_mode();
    assert!(context.is_in_recovery_mode());
    
    // In recovery mode, we can continue with degraded functionality
    context.set_variable("recovery_mode", kwasa_kwasa::turbulance::context::Value::Boolean(true));
    
    // Exit recovery mode after handling the issue
    context.exit_recovery_mode();
    assert!(!context.is_in_recovery_mode());
    
    // End execution
    context.end_execution();
    
    // Verify performance metrics were recorded
    let report = context.get_performance_report();
    assert!(report.contains("memory_usage_mb: 128.5"));
    assert!(report.contains("query_time_ms: 45.2"));
} 