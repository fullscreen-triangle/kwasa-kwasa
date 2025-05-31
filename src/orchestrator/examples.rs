use std::sync::Arc;
use tokio::sync::mpsc::Receiver;

use super::metacognitive::{
    DefaultContextLayer, DefaultReasoningLayer, DefaultIntuitionLayer,
    MetacognitiveOrchestrator,
};
use super::stream::{StreamProcessor, FunctionProcessor};
use super::types::StreamData;

/// Create a default orchestrator with the three standard layers
pub fn create_default_orchestrator() -> MetacognitiveOrchestrator {
    MetacognitiveOrchestrator::new(
        DefaultContextLayer::new(),
        DefaultReasoningLayer::new(),
        DefaultIntuitionLayer::new(),
    )
}

/// Create a custom orchestrator with function-based layers
pub fn create_custom_orchestrator() -> MetacognitiveOrchestrator {
    // Context layer: Extract key concepts
    let context_layer = FunctionProcessor::new("CustomContext", |mut data| {
        // Extract key concepts from the text
        let concepts = extract_concepts(&data.content);
        
        // Store in metadata
        data = data.with_metadata("concepts", &concepts.join(","))
            .with_confidence(0.6);
        data
    });
    
    // Reasoning layer: Apply logical rules
    let reasoning_layer = FunctionProcessor::new("CustomReasoning", |mut data| {
        // Get concepts from previous layer
        let concepts = data.metadata.get("concepts")
            .map(|s| s.split(',').map(String::from).collect::<Vec<_>>())
            .unwrap_or_default();
        
        // Apply logical rules based on concepts
        let inferences = apply_logical_rules(&concepts);
        
        // Store in metadata
        data.with_metadata("inferences", &inferences.join(","))
            .with_confidence(0.8)
    });
    
    // Intuition layer: Generate creative insights
    let intuition_layer = FunctionProcessor::new("CustomIntuition", |mut data| {
        // Get inferences from previous layer
        let inferences = data.metadata.get("inferences")
            .map(|s| s.split(',').map(String::from).collect::<Vec<_>>())
            .unwrap_or_default();
        
        // Generate creative insights
        let insights = generate_insights(&inferences);
        
        // Update content with enriched text
        let mut content = data.content.clone();
        content.push_str("\n\nInsights:\n");
        for insight in insights {
            content.push_str(&format!("- {}\n", insight));
        }
        
        data.content = content;
        data.with_confidence(1.0)
    });
    
    MetacognitiveOrchestrator::new(
        context_layer,
        reasoning_layer,
        intuition_layer,
    )
}

/// Example: Process text using the orchestrator
pub async fn process_text_example(text: &str) -> Vec<StreamData> {
    // Create orchestrator
    let orchestrator = create_default_orchestrator();
    
    // Add some knowledge
    orchestrator.add_knowledge("domain", "creative writing");
    orchestrator.add_knowledge("style", "academic");
    
    // Process input
    let input = vec![StreamData::new(text.to_string())];
    let results = orchestrator.process(input).await;
    
    results
}

/// Example: Stream data through the orchestrator
pub async fn stream_text_example(text: &str) {
    // Create orchestrator
    let orchestrator = create_custom_orchestrator();
    
    // Create input stream
    let (tx, rx) = orchestrator.create_input_stream().await;
    
    // Process streaming with callback
    let processing = orchestrator.process_stream(rx, |data| async move {
        println!("Received result: {}", data.content);
        println!("Confidence: {}", data.confidence);
        println!("Metadata: {:?}", data.metadata);
        println!("---");
    });
    
    // Send input data
    let _ = tx.send(StreamData::new(text.to_string())).await;
    
    // Process more text segments as they become available
    let _ = tx.send(StreamData::new("More input text...".to_string())).await;
    let _ = tx.send(StreamData::new("Final text segment.".to_string())).await;
    
    // Wait for processing to complete
    // Note: In a real application, you would properly manage channel closure
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
}

// Example helper functions
fn extract_concepts(text: &str) -> Vec<String> {
    // Simplified concept extraction - just split on spaces and take words > 5 chars
    text.split_whitespace()
        .filter(|word| word.len() > 5)
        .map(|word| word.to_string())
        .collect()
}

fn apply_logical_rules(concepts: &[String]) -> Vec<String> {
    // Simplified logical rules
    concepts.iter()
        .map(|concept| format!("logical-{}", concept))
        .collect()
}

fn generate_insights(inferences: &[String]) -> Vec<String> {
    // Simplified insight generation
    inferences.iter()
        .map(|inference| format!("Creative insight based on {}", inference))
        .collect()
} 