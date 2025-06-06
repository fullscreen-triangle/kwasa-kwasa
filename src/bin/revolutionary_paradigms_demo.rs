/// Revolutionary Paradigms Demo for Kwasa-Kwasa
/// 
/// This demo showcases all four revolutionary paradigms:
/// 1. Points and Resolutions: Probabilistic Language Processing
/// 2. Positional Semantics: Position as Primary Meaning
/// 3. Perturbation Validation: Testing Probabilistic Robustness
/// 4. Hybrid Processing with Probabilistic Loops

use std::collections::HashMap;
use kwasa_kwasa::turbulance::{
    // Points and Resolutions
    point, ResolutionStrategy, ResolutionManager, ResolutionContext,
    
    // Positional Semantics
    PositionalAnalyzer, PositionalSentence, SemanticRole,
    
    // Perturbation Validation
    PerturbationValidator, ValidationConfig, ValidationDepth, validate_resolution_quality,
    
    // Debate Platforms
    DebatePlatform, DebatePlatformManager, Affirmation, Contention, ChallengeAspect,
    EvidenceType, EvidenceSource, SourceType,
    
    // Hybrid Processing
    HybridProcessor, ProbabilisticFloor, HybridConfig, ProcessingMode,
    
    // Turbulance Syntax
    TurbulanceProcessor, TurbulanceFunction, TurbulanceOperation, TurbulanceType,
    
    // Integration
    KwasaKwasaPipeline, PipelineConfig, demonstrate_complete_framework,
    
    // Core types
    interpreter::Value, Result, TurbulanceError,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🌟 Kwasa-Kwasa Revolutionary Paradigms Demonstration 🌟");
    println!("=".repeat(70));
    
    // Demo 1: Points and Resolutions
    println!("\n📍 Demo 1: Points and Resolutions - Probabilistic Language Processing");
    println!("-".repeat(70));
    demo_points_and_resolutions().await?;
    
    // Demo 2: Positional Semantics
    println!("\n📍 Demo 2: Positional Semantics - Position as Primary Meaning");
    println!("-".repeat(70));
    demo_positional_semantics().await?;
    
    // Demo 3: Perturbation Validation
    println!("\n📍 Demo 3: Perturbation Validation - Testing Probabilistic Robustness");
    println!("-".repeat(70));
    demo_perturbation_validation().await?;
    
    // Demo 4: Hybrid Processing
    println!("\n📍 Demo 4: Hybrid Processing with Probabilistic Loops");
    println!("-".repeat(70));
    demo_hybrid_processing().await?;
    
    // Demo 5: All Paradigms Together
    println!("\n📍 Demo 5: Revolutionary Synthesis - All Paradigms Working Together");
    println!("-".repeat(70));
    demo_revolutionary_synthesis().await?;
    
    println!("\n🎉 Revolutionary Paradigms Demo Complete! 🎉");
    println!("The future of text processing is probabilistic, position-aware, validated, and adaptive!");
    
    Ok(())
}

/// Demo 1: Points and Resolutions - Probabilistic Language Processing
async fn demo_points_and_resolutions() -> Result<()> {
    println!("Core Insight: 'No point is 100% certain'");
    println!("Instead of deterministic functions, we use probabilistic debate platforms.");
    
    // Create a point with inherent uncertainty
    let research_claim = point!(
        "AI systems demonstrate emergent reasoning capabilities at scale",
        certainty: 0.72,
        evidence_strength: 0.65,
        contextual_relevance: 0.88
    );
    
    println!("📝 Created Point: '{}'", research_claim.content);
    println!("   Certainty: {:.1%}", research_claim.certainty);
    
    // Create a debate platform for this point
    let mut debate_manager = DebatePlatformManager::new();
    let platform_id = debate_manager.create_platform(
        research_claim.clone(),
        ResolutionStrategy::Bayesian,
        None
    );
    
    let platform = debate_manager.get_platform_mut(&platform_id).unwrap();
    
    // Add affirmations (supporting evidence)
    let aff1_id = platform.add_affirmation(
        "Large language models show reasoning patterns in mathematical proofs".to_string(),
        "OpenAI Research Paper".to_string(),
        0.78,
        0.82
    ).await?;
    
    let aff2_id = platform.add_affirmation(
        "Chain-of-thought prompting enables step-by-step logical reasoning".to_string(),
        "Google Research".to_string(),
        0.85,
        0.90
    ).await?;
    
    // Add contentions (challenges)
    let con1_id = platform.add_contention(
        "Apparent reasoning may be sophisticated pattern matching, not true reasoning".to_string(),
        "NYU AI Critique".to_string(),
        0.71,
        0.75,
        ChallengeAspect::LogicalReasoning
    ).await?;
    
    // Update the score based on debate
    platform.update_score().await?;
    
    let resolution = platform.get_resolution();
    println!("🏛️  Debate Platform Resolution:");
    println!("   Final Score: {:.1%}", resolution.confidence);
    println!("   Resolution: {}", resolution.resolution_text);
    println!("   Affirmations: {}", platform.affirmations.len());
    println!("   Contentions: {}", platform.contentions.len());
    
    Ok(())
}

/// Demo 2: Positional Semantics - Position as Primary Meaning
async fn demo_positional_semantics() -> Result<()> {
    println!("Core Insight: 'The location of a word is the whole point behind its probable meaning'");
    println!("Word position becomes a first-class semantic feature.");
    
    let mut analyzer = PositionalAnalyzer::new();
    
    // Analyze sentences with different positional structures
    let sentences = vec![
        "The AI quickly learned the complex task",
        "Quickly, the AI learned the complex task", 
        "The AI learned quickly the complex task",
        "The complex task was learned quickly by the AI"
    ];
    
    for (i, sentence) in sentences.iter().enumerate() {
        println!("\n📝 Sentence {}: '{}'", i + 1, sentence);
        
        let analysis = analyzer.analyze(sentence)?;
        
        println!("   Order Dependency: {:.1%}", analysis.order_dependency_score);
        println!("   Semantic Signature: {}", analysis.semantic_signature);
        
        // Show positional weights for key words
        for word in &analysis.words {
            if word.is_content_word && word.positional_weight > 0.7 {
                println!("   '{}' (pos {}): weight={:.2}, role={:?}", 
                    word.text, word.position, word.positional_weight, word.semantic_role);
            }
        }
        
        // Convert to TextPoint for further processing
        let text_point = analysis.to_text_point();
        println!("   Point Certainty: {:.1%}", text_point.certainty);
    }
    
    // Compare positional similarity
    let analysis1 = analyzer.analyze(&sentences[0])?;
    let analysis2 = analyzer.analyze(&sentences[1])?;
    let similarity = analysis1.positional_similarity(&analysis2);
    
    println!("\n🔍 Positional Similarity between sentences 1 & 2: {:.1%}", similarity);
    println!("   (Lower similarity indicates position significantly affects meaning)");
    
    Ok(())
}

/// Demo 3: Perturbation Validation - Testing Probabilistic Robustness
async fn demo_perturbation_validation() -> Result<()> {
    println!("Core Insight: 'Since everything is probabilistic, there still should be a way");
    println!("to disentangle these seemingly fleeting quantities'");
    
    // Create a point to validate
    let point = point!(
        "Machine learning algorithms can detect early signs of disease in medical imaging",
        certainty: 0.84,
        evidence_strength: 0.79,
        contextual_relevance: 0.91
    );
    
    println!("📝 Testing Point: '{}'", point.content);
    println!("   Initial Certainty: {:.1%}", point.certainty);
    
    // Create an initial resolution
    let mut resolution_manager = ResolutionManager::new();
    let resolution = resolution_manager.create_resolution(
        point.clone(),
        ResolutionStrategy::Conservative,
        ResolutionContext::default()
    ).await?;
    
    // Run perturbation validation
    let config = ValidationConfig {
        validation_depth: ValidationDepth::Thorough,
        min_stability_score: 0.7,
        enable_word_removal: true,
        enable_positional_rearrangement: true,
        enable_synonym_substitution: true,
        enable_negation_tests: true,
        ..Default::default()
    };
    
    println!("\n🧪 Running Perturbation Validation Tests...");
    let validation_result = validate_resolution_quality(&point, &resolution, Some(config)).await?;
    
    println!("✅ Validation Results:");
    println!("   Stability Score: {:.1%}", validation_result.stability_score);
    println!("   Reliability: {:?}", validation_result.quality_assessment.reliability_category);
    println!("   Tests Performed: {}", validation_result.perturbation_results.len());
    println!("   Tests Passed: {}", 
        validation_result.perturbation_results.iter().filter(|r| r.passed).count());
    
    // Show specific test results
    for result in &validation_result.perturbation_results[..3] { // Show first 3
        println!("   📋 {:?}: {:.1%} stability, {} confidence change", 
            result.test_type, result.stability_score, 
            if result.confidence_change > 0.0 { "+" } else { "" });
    }
    
    // Show recommendations
    if !validation_result.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for (i, rec) in validation_result.recommendations.iter().take(2).enumerate() {
            println!("   {}. {}", i + 1, rec);
        }
    }
    
    Ok(())
}

/// Demo 4: Hybrid Processing with Probabilistic Loops
async fn demo_hybrid_processing() -> Result<()> {
    println!("Core Insight: 'The whole probabilistic system can be tucked inside probabilistic processes'");
    println!("Dynamic switching between deterministic and probabilistic processing modes.");
    
    let config = HybridConfig {
        probabilistic_threshold: 0.75,
        settlement_threshold: 0.85,
        max_roll_iterations: 10,
        enable_adaptive_loops: true,
        density_resolution: 100,
        stream_buffer_size: 1024,
    };
    
    let mut processor = HybridProcessor::new(config);
    
    // Create a probabilistic floor
    let mut floor = ProbabilisticFloor::new(0.3);
    
    let points = vec![
        ("hypothesis1", "Deep learning can predict protein folding", 0.72),
        ("hypothesis2", "Quantum computers will break current encryption", 0.65),
        ("hypothesis3", "AGI will emerge through scale alone", 0.48),
        ("hypothesis4", "Renewable energy will replace fossil fuels by 2050", 0.83),
    ];
    
    for (key, content, weight) in points {
        let point = point!(content, certainty: weight * 0.9, evidence_strength: weight);
        floor.add_point(key.to_string(), point, weight);
    }
    
    println!("🎯 Created Probabilistic Floor with {} points", floor.points.len());
    
    // Demo 1: Cycle operation
    println!("\n🔄 Cycle Operation - Basic iteration through floor:");
    let cycle_results = processor.cycle(&floor, |point, weight| {
        println!("   Processing: '{}' (weight: {:.2})", 
            point.content.chars().take(50).collect::<String>(), weight);
        Ok(kwasa_kwasa::turbulance::ResolutionResult {
            confidence: point.certainty * weight,
            resolution_text: format!("Analyzed with weight {:.2}", weight),
            strategy: ResolutionStrategy::MaximumLikelihood,
            metadata: HashMap::new(),
        })
    }).await?;
    
    println!("   Completed {} cycles", cycle_results.len());
    
    // Demo 2: Drift operation  
    println!("\n🌊 Drift Operation - Probabilistic corpus exploration:");
    let corpus = "Machine learning is transforming science. AI can predict molecular behavior. \
                  Deep learning models show remarkable capabilities. Quantum computing may revolutionize AI.";
    
    let drift_results = processor.drift(corpus).await?;
    println!("   Processed {} drift operations through corpus", drift_results.len());
    
    // Demo 3: Roll until settled
    println!("\n🎲 Roll Until Settled - Iterative resolution:");
    let uncertain_point = point!(
        "Consciousness can emerge from computational processes",
        certainty: 0.45,
        evidence_strength: 0.38
    );
    
    let roll_result = processor.roll_until_settled(&uncertain_point).await?;
    println!("   Settled after {} iterations", roll_result.iterations);
    println!("   Final confidence: {:.1%}", roll_result.confidence);
    println!("   Settlement achieved: {}", roll_result.settled);
    
    // Demo 4: Hybrid function with conditional probabilistic processing
    println!("\n⚡ Hybrid Function - Conditional probabilistic processing:");
    let paragraph = "AI ethics is crucial for development. Some concerns are overblown. \
                     Regulation should be thoughtful. Technology can solve its own problems.";
    
    let hybrid_results = processor.hybrid_function(paragraph, 0.7, |sentence| {
        let uncertainty = sentence.split_whitespace().count() as f64 * 0.1;
        Ok(uncertainty < 0.6) // Return true if we should continue processing
    }).await?;
    
    println!("   Processed {} sentences with hybrid approach", hybrid_results.len());
    for result in &hybrid_results {
        println!("   '{}': {:.1%} confidence, {} mode", 
            result.input.chars().take(30).collect::<String>(), 
            result.confidence, result.mode);
    }
    
    // Show stats
    let stats = processor.get_stats();
    println!("\n📊 Processing Statistics:");
    println!("   Cycles: {}, Drifts: {}, Flows: {}, Rolls: {}", 
        stats.cycles_performed, stats.drifts_executed, 
        stats.flows_processed, stats.rolls_until_settled);
    
    Ok(())
}

/// Demo 5: Revolutionary Synthesis - All Paradigms Working Together
async fn demo_revolutionary_synthesis() -> Result<()> {
    println!("🌟 Revolutionary Synthesis: All Four Paradigms Working Together");
    println!("This demonstrates the complete framework integration:");
    println!("1. Points with inherent uncertainty");
    println!("2. Positional semantics enriching meaning");
    println!("3. Perturbation validation ensuring robustness");
    println!("4. Hybrid processing adapting to uncertainty");
    
    // Use the complete integration pipeline
    let config = PipelineConfig::default();
    let mut pipeline = KwasaKwasaPipeline::new(config);
    
    // Process a complex research paper abstract
    let research_text = "
Recent advances in large language models have demonstrated remarkable capabilities \
in reasoning and problem-solving. These systems exhibit emergent behaviors that \
were not explicitly programmed, suggesting potential pathways toward artificial \
general intelligence. However, fundamental questions remain about the nature of \
understanding in these models. Critics argue that sophisticated pattern matching \
should not be confused with genuine comprehension. The debate continues regarding \
whether scale alone can bridge the gap to true artificial intelligence.
";
    
    println!("\n📄 Processing Research Text ({} characters):", research_text.len());
    println!("'{}'", research_text.trim());
    
    let result = pipeline.process_text(research_text).await?;
    
    println!("\n🔍 Revolutionary Processing Results:");
    println!("   Points Extracted: {}", result.points.len());
    println!("   Debates Created: {}", result.debates.len());
    println!("   Overall Quality: {:.1%}", result.quality_assessment.overall_score);
    
    // Show detailed analysis for first few points
    for (i, validated_point) in result.points.iter().take(3).enumerate() {
        println!("\n   📍 Point {}: '{}'", i + 1, 
            validated_point.point.content.chars().take(60).collect::<String>());
        
        // Positional analysis
        println!("      🎯 Position Analysis:");
        println!("         Order Dependency: {:.1%}", 
            validated_point.positional_analysis.order_dependency_score);
        println!("         Analysis Confidence: {:.1%}", 
            validated_point.positional_analysis.analysis_confidence);
        
        // Validation results
        println!("      🧪 Validation Results:");
        println!("         Stability Score: {:.1%}", 
            validated_point.validation.stability_score);
        println!("         Reliability: {:?}", 
            validated_point.validation.quality_assessment.reliability_category);
        
        // Resolution
        println!("      ⚖️  Final Resolution:");
        println!("         Confidence: {:.1%}", validated_point.resolution.confidence);
        
        // Debate platform (if created)
        if let Some(debate_id) = validated_point.debate_id {
            if let Some(debate_info) = result.debates.iter().find(|d| d.id == debate_id) {
                println!("      🏛️  Debate Platform:");
                println!("         Current Score: {:.1%}", debate_info.current_score);
                println!("         Contributions: {} affirmations, {} contentions", 
                    debate_info.contributions.0, debate_info.contributions.1);
            }
        }
    }
    
    // Show debates created
    if !result.debates.is_empty() {
        println!("\n🏛️  Debates Created for Uncertain Points:");
        for debate in &result.debates {
            println!("   • '{}': {:.1%} confidence, {} state", 
                debate.point_content.chars().take(50).collect::<String>(),
                debate.current_score, debate.resolution_state);
        }
    }
    
    // Quality assessment
    println!("\n📊 Quality Assessment:");
    println!("   Overall Score: {:.1%}", result.quality_assessment.overall_score);
    println!("   Positional Coherence: {:.1%}", result.quality_assessment.positional_coherence);
    println!("   Validation Stability: {:.1%}", result.quality_assessment.validation_stability);
    println!("   Resolution Confidence: {:.1%}", result.quality_assessment.resolution_confidence);
    println!("   📋 Recommendation: {}", result.quality_assessment.recommendation);
    
    // Processing metadata
    println!("\n⏱️  Processing Metadata:");
    println!("   Duration: {} ms", result.metadata.duration_ms);
    println!("   Sentences Analyzed: {}", result.metadata.sentences_analyzed);
    println!("   Validation Tests: {}", result.metadata.validation_tests);
    println!("   Debates Initiated: {}", result.metadata.debates_initiated);
    
    if !result.metadata.warnings.is_empty() {
        println!("   ⚠️  Warnings: {}", result.metadata.warnings.len());
    }
    
    // Pipeline statistics
    let stats = pipeline.get_stats();
    println!("\n📈 Pipeline Statistics:");
    println!("   Texts Processed: {}", stats.texts_processed);
    println!("   Points Extracted: {}", stats.points_extracted);
    println!("   Debates Created: {}", stats.debates_created);
    println!("   Validations Performed: {}", stats.validations_performed);
    println!("   Resolution Success Rate: {:.1%}", stats.resolution_success_rate);
    
    Ok(())
} 