// Intelligence Modules Examples
// Demonstrates how to use the five intelligence modules

use crate::orchestrator::{
    StreamData, StreamPipeline, StreamProcessor,
    MzekezkeBayesianEngine, DiggidenAdversarialSystem, HatataDecisionSystem,
    SpectacularHandler, NicotineContextValidator, IntegratedMetacognitiveOrchestrator
};
use log::info;

/// Example showing basic usage of individual intelligence modules
pub async fn basic_intelligence_example() {
    info!("=== Basic Intelligence Modules Example ===");

    // Create the five intelligence modules
    let mzekezeke = MzekezkeBayesianEngine::new();
    let diggiden = DiggidenAdversarialSystem::new();
    let hatata = HatataDecisionSystem::new();
    let spectacular = SpectacularHandler::new();
    let nicotine = NicotineContextValidator::new();

    // Create sample text data
    let sample_text = "This is a revolutionary breakthrough in machine learning that demonstrates unprecedented paradigm shifts across multiple domains of science and technology.";
    let data = StreamData::new(sample_text.to_string()).with_confidence(0.6);

    info!("Processing sample text: {}", sample_text);

    // Process through each module individually
    let mzekezeke_result = mzekezeke.process(tokio::sync::mpsc::channel(1).1).await;
    info!("Mzekezeke (Bayesian Learning): Ready for belief optimization");

    let diggiden_result = diggiden.process(tokio::sync::mpsc::channel(1).1).await;
    info!("Diggiden (Adversarial System): Ready for vulnerability testing");

    let hatata_result = hatata.process(tokio::sync::mpsc::channel(1).1).await;
    info!("Hatata (Decision System): Ready for utility optimization");

    let spectacular_result = spectacular.process(tokio::sync::mpsc::channel(1).1).await;
    info!("Spectacular (Extraordinary Handler): Ready for significance assessment");

    let nicotine_result = nicotine.process(tokio::sync::mpsc::channel(1).1).await;
    info!("Nicotine (Context Validator): Ready for context validation");

    info!("All intelligence modules initialized successfully!");
}

/// Example showing pipeline usage of intelligence modules
pub async fn pipeline_intelligence_example() {
    info!("=== Intelligence Pipeline Example ===");

    // Create a processing pipeline with all five modules
    let mut pipeline = StreamPipeline::new("IntelligencePipeline");
    
    pipeline
        .add_processor(MzekezkeBayesianEngine::new())      // Bayesian learning
        .add_processor(DiggidenAdversarialSystem::new())   // Adversarial testing
        .add_processor(HatataDecisionSystem::new())        // Decision optimization
        .add_processor(SpectacularHandler::new())          // Extraordinary content
        .add_processor(NicotineContextValidator::new());   // Context validation

    // Sample texts with different characteristics
    let texts = vec![
        "Regular text about daily activities and normal observations.",
        "Groundbreaking research reveals revolutionary approaches to quantum computing that could transform technology forever.",
        "The study contradicts previous findings while demonstrating clear evidence of paradigm shifts in medical science.",
        "Simple statement with basic information and standard content structure.",
    ];

    for (i, text) in texts.iter().enumerate() {
        info!("--- Processing Text {} ---", i + 1);
        info!("Input: {}", text);
        
        let input_data = StreamData::new(text.to_string()).with_confidence(0.5);
        let results = pipeline.execute(vec![input_data]).await;
        
        if let Some(result) = results.first() {
            info!("Output: {}", result.content);
            info!("Final Confidence: {:.3}", result.confidence);
            
            // Display metadata from intelligence modules
            for (key, value) in &result.metadata {
                if key.starts_with("mzekezeke_") || 
                   key.starts_with("diggiden_") || 
                   key.starts_with("hatata_") || 
                   key.starts_with("spectacular_") || 
                   key.starts_with("nicotine_") {
                    info!("  {}: {}", key, value);
                }
            }
        }
        info!("");
    }
}

/// Example showing the integrated orchestrator
pub async fn integrated_orchestrator_example() {
    info!("=== Integrated Metacognitive Orchestrator Example ===");

    // Create the integrated orchestrator (combines all five modules)
    let orchestrator = IntegratedMetacognitiveOrchestrator::new();

    // Sample texts for intelligent processing
    let test_cases = vec![
        ("Regular Content", "This is normal text content for standard processing."),
        ("Scientific Discovery", "Revolutionary breakthrough in quantum mechanics demonstrates unprecedented energy efficiency gains."),
        ("Contradictory Information", "The new study contradicts established theories while providing clear evidence of effectiveness."),
        ("Cross-Domain Innovation", "This technology bridges neuroscience, computer science, and psychology to create transformative solutions."),
        ("Complex Analysis", "The evidence strongly supports the hypothesis through rigorous methodology and comprehensive data analysis."),
    ];

    for (label, text) in test_cases {
        info!("--- {} ---", label);
        info!("Input: {}", text);
        
        // Process text through integrated intelligence
        let result = orchestrator.process_text_intelligently(text.to_string()).await;
        
        info!("Processed Result: {}", result);
        info!("");
    }
    
    info!("Integrated orchestrator demonstration complete!");
}

/// Example showing custom configuration of modules
pub async fn custom_configuration_example() {
    info!("=== Custom Configuration Example ===");

    // Create modules with custom settings
    let mzekezeke = MzekezkeBayesianEngine::new();
    
    let diggiden = DiggidenAdversarialSystem::new()
        .with_attack_frequency(std::time::Duration::from_secs(10)); // More frequent attacks
    
    let hatata = HatataDecisionSystem::new();
    
    let spectacular = SpectacularHandler::new()
        .with_significance_threshold(0.7)  // Lower threshold for extraordinariness
        .with_atp_investment(1000.0);      // Higher ATP investment
    
    let nicotine = NicotineContextValidator::new()
        .with_validation_frequency(std::time::Duration::from_secs(30)); // More frequent validation

    info!("Custom intelligence modules configured:");
    info!("- Diggiden: 10-second attack frequency");
    info!("- Spectacular: 0.7 significance threshold, 1000 ATP investment");
    info!("- Nicotine: 30-second validation frequency");

    // Create custom pipeline
    let mut custom_pipeline = StreamPipeline::new("CustomIntelligence");
    custom_pipeline
        .add_processor(mzekezeke)
        .add_processor(diggiden)
        .add_processor(hatata)
        .add_processor(spectacular)
        .add_processor(nicotine);

    let test_text = "This innovative approach demonstrates significant advances in artificial intelligence through novel methodologies.";
    let input_data = StreamData::new(test_text.to_string()).with_confidence(0.6);
    
    info!("Processing with custom configuration...");
    let results = custom_pipeline.execute(vec![input_data]).await;
    
    if let Some(result) = results.first() {
        info!("Custom processing completed successfully!");
        info!("Final confidence: {:.3}", result.confidence);
        info!("Metadata entries: {}", result.metadata.len());
    }
}

/// Run all intelligence examples
pub async fn run_all_intelligence_examples() {
    info!("=== Running All Intelligence Module Examples ===");
    
    basic_intelligence_example().await;
    pipeline_intelligence_example().await;
    integrated_orchestrator_example().await;
    custom_configuration_example().await;
    
    info!("=== All Intelligence Examples Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_intelligence() {
        basic_intelligence_example().await;
    }
    
    #[tokio::test] 
    async fn test_integrated_orchestrator() {
        let orchestrator = IntegratedMetacognitiveOrchestrator::new();
        let result = orchestrator.process_text_intelligently("Test text".to_string()).await;
        assert!(!result.is_empty());
    }
} 