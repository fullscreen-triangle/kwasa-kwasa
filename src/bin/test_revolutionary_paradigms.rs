/// Comprehensive Test Suite for Revolutionary Paradigms
/// 
/// This test suite validates that all four revolutionary paradigms are
/// implemented correctly and working together as intended.

use std::collections::HashMap;
use kwasa_kwasa::turbulance::{
    // Core types and traits
    Result, TurbulanceError, Value,
    
    // Points and Resolutions
    point, ResolutionStrategy, ResolutionManager, ResolutionContext,
    
    // Positional Semantics
    PositionalAnalyzer, SemanticRole,
    
    // Perturbation Validation
    ValidationConfig, ValidationDepth, validate_resolution_quality,
    
    // Debate Platforms
    DebatePlatformManager, ChallengeAspect,
    
    // Hybrid Processing
    HybridProcessor, ProbabilisticFloor, HybridConfig,
    
    // Integration
    KwasaKwasaPipeline, PipelineConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Comprehensive Test Suite for Revolutionary Paradigms");
    println!("=".repeat(70));
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Test 1: Points and Resolutions
    println!("\n🔬 Test Suite 1: Points and Resolutions");
    println!("-".repeat(50));
    match test_points_and_resolutions().await {
        Ok(_) => {
            println!("✅ Points and Resolutions: PASSED");
            passed += 1;
        },
        Err(e) => {
            println!("❌ Points and Resolutions: FAILED - {}", e);
            failed += 1;
        }
    }
    
    // Test 2: Positional Semantics
    println!("\n🔬 Test Suite 2: Positional Semantics");
    println!("-".repeat(50));
    match test_positional_semantics().await {
        Ok(_) => {
            println!("✅ Positional Semantics: PASSED");
            passed += 1;
        },
        Err(e) => {
            println!("❌ Positional Semantics: FAILED - {}", e);
            failed += 1;
        }
    }
    
    // Test 3: Perturbation Validation
    println!("\n🔬 Test Suite 3: Perturbation Validation");
    println!("-".repeat(50));
    match test_perturbation_validation().await {
        Ok(_) => {
            println!("✅ Perturbation Validation: PASSED");
            passed += 1;
        },
        Err(e) => {
            println!("❌ Perturbation Validation: FAILED - {}", e);
            failed += 1;
        }
    }
    
    // Test 4: Hybrid Processing
    println!("\n🔬 Test Suite 4: Hybrid Processing");
    println!("-".repeat(50));
    match test_hybrid_processing().await {
        Ok(_) => {
            println!("✅ Hybrid Processing: PASSED");
            passed += 1;
        },
        Err(e) => {
            println!("❌ Hybrid Processing: FAILED - {}", e);
            failed += 1;
        }
    }
    
    // Test 5: Integration
    println!("\n🔬 Test Suite 5: Revolutionary Integration");
    println!("-".repeat(50));
    match test_revolutionary_integration().await {
        Ok(_) => {
            println!("✅ Revolutionary Integration: PASSED");
            passed += 1;
        },
        Err(e) => {
            println!("❌ Revolutionary Integration: FAILED - {}", e);
            failed += 1;
        }
    }
    
    // Final results
    println!("\n🏁 Test Results Summary");
    println!("=".repeat(70));
    println!("✅ Passed: {}", passed);
    println!("❌ Failed: {}", failed);
    println!("📊 Success Rate: {:.1%}", passed as f64 / (passed + failed) as f64);
    
    if failed == 0 {
        println!("🎉 ALL TESTS PASSED! Revolutionary paradigms are fully implemented!");
    } else {
        println!("⚠️  Some tests failed. Please check implementation.");
    }
    
    Ok(())
}

/// Test Points and Resolutions paradigm
async fn test_points_and_resolutions() -> Result<()> {
    println!("Testing: 'No point is 100% certain'");
    
    // Test 1: Point creation with uncertainty
    let point = point!(
        "Artificial intelligence will exceed human performance in most cognitive tasks",
        certainty: 0.78,
        evidence_strength: 0.65,
        contextual_relevance: 0.82
    );
    
    // Verify point properties
    assert!(point.certainty < 1.0, "Points should have uncertainty");
    assert!(point.certainty > 0.0, "Points should have positive certainty");
    assert!(!point.content.is_empty(), "Points should have content");
    println!("   ✓ Point creation with uncertainty");
    
    // Test 2: Debate platform creation
    let mut debate_manager = DebatePlatformManager::new();
    let platform_id = debate_manager.create_platform(
        point.clone(),
        ResolutionStrategy::Bayesian,
        None
    );
    
    let platform = debate_manager.get_platform(&platform_id)
        .ok_or_else(|| TurbulanceError::RuntimeError { 
            message: "Failed to create debate platform".to_string() 
        })?;
    
    assert_eq!(platform.point.content, point.content, "Platform should contain the point");
    println!("   ✓ Debate platform creation");
    
    // Test 3: Affirmations and contentions
    let platform = debate_manager.get_platform_mut(&platform_id).unwrap();
    
    let aff_id = platform.add_affirmation(
        "Recent AI benchmarks show consistent improvement".to_string(),
        "Research Study".to_string(),
        0.85,
        0.78
    ).await?;
    
    let con_id = platform.add_contention(
        "AI lacks true understanding and consciousness".to_string(),
        "Philosophy Paper".to_string(),
        0.72,
        0.81,
        ChallengeAspect::LogicalReasoning
    ).await?;
    
    assert_eq!(platform.affirmations.len(), 1, "Should have one affirmation");
    assert_eq!(platform.contentions.len(), 1, "Should have one contention");
    println!("   ✓ Affirmations and contentions");
    
    // Test 4: Score update through debate
    let initial_score = platform.current_score;
    platform.update_score().await?;
    let final_score = platform.current_score;
    
    assert!(final_score >= 0.0 && final_score <= 1.0, "Score should be in valid range");
    println!("   ✓ Probabilistic score calculation (initial: {:.2}, final: {:.2})", 
        initial_score, final_score);
    
    // Test 5: Resolution generation
    let resolution = platform.get_resolution();
    assert!(resolution.confidence > 0.0, "Resolution should have confidence");
    assert!(!resolution.resolution_text.is_empty(), "Resolution should have text");
    println!("   ✓ Resolution generation");
    
    Ok(())
}

/// Test Positional Semantics paradigm
async fn test_positional_semantics() -> Result<()> {
    println!("Testing: 'The location of a word is the whole point behind its probable meaning'");
    
    let mut analyzer = PositionalAnalyzer::new();
    
    // Test 1: Basic positional analysis
    let sentence = "The quick brown fox jumps over the lazy dog";
    let analysis = analyzer.analyze(sentence)?;
    
    assert_eq!(analysis.original_text, sentence, "Should preserve original text");
    assert!(!analysis.words.is_empty(), "Should have analyzed words");
    assert!(analysis.order_dependency_score >= 0.0 && analysis.order_dependency_score <= 1.0, 
        "Order dependency should be in valid range");
    println!("   ✓ Basic positional analysis");
    
    // Test 2: Positional weights
    let content_words: Vec<_> = analysis.words.iter()
        .filter(|w| w.is_content_word)
        .collect();
    
    assert!(!content_words.is_empty(), "Should identify content words");
    
    for word in &content_words {
        assert!(word.positional_weight >= 0.0 && word.positional_weight <= 1.0,
            "Positional weight should be in valid range");
        assert!(word.position > 0, "Position should be 1-indexed");
    }
    println!("   ✓ Positional weight assignment");
    
    // Test 3: Semantic role assignment
    let roles: Vec<_> = analysis.words.iter()
        .map(|w| &w.semantic_role)
        .collect();
    
    assert!(roles.contains(&&SemanticRole::Subject) || roles.contains(&&SemanticRole::Predicate),
        "Should identify major semantic roles");
    println!("   ✓ Semantic role assignment");
    
    // Test 4: Position-dependent similarity
    let sentence1 = "The AI quickly learned the task";
    let sentence2 = "Quickly, the AI learned the task";
    
    let analysis1 = analyzer.analyze(sentence1)?;
    let analysis2 = analyzer.analyze(sentence2)?;
    
    let similarity = analysis1.positional_similarity(&analysis2);
    assert!(similarity >= 0.0 && similarity <= 1.0, "Similarity should be in valid range");
    assert!(similarity < 1.0, "Position changes should reduce similarity");
    println!("   ✓ Position-dependent similarity (similarity: {:.2})", similarity);
    
    // Test 5: TextPoint conversion
    let text_point = analysis1.to_text_point();
    assert!(!text_point.content.is_empty(), "Should create valid text point");
    assert!(text_point.certainty > 0.0, "Should have positive certainty");
    println!("   ✓ TextPoint conversion");
    
    Ok(())
}

/// Test Perturbation Validation paradigm
async fn test_perturbation_validation() -> Result<()> {
    println!("Testing: 'Since everything is probabilistic, there still should be a way");
    println!("         to disentangle these seemingly fleeting quantities'");
    
    // Test 1: Create point and resolution for validation
    let point = point!(
        "Climate change is primarily caused by human activities",
        certainty: 0.92,
        evidence_strength: 0.88,
        contextual_relevance: 0.95
    );
    
    let mut resolution_manager = ResolutionManager::new();
    let resolution = resolution_manager.create_resolution(
        point.clone(),
        ResolutionStrategy::Conservative,
        ResolutionContext::default()
    ).await?;
    
    assert!(resolution.confidence > 0.0, "Resolution should have confidence");
    println!("   ✓ Point and resolution creation");
    
    // Test 2: Validation configuration
    let config = ValidationConfig {
        validation_depth: ValidationDepth::Quick, // Use quick for testing
        min_stability_score: 0.6,
        enable_word_removal: true,
        enable_positional_rearrangement: true,
        enable_synonym_substitution: false, // Disable for simpler testing
        enable_negation_tests: true,
        max_perturbations_per_type: 3,
        ..Default::default()
    };
    
    // Test 3: Run perturbation validation
    let validation_result = validate_resolution_quality(&point, &resolution, Some(config)).await?;
    
    assert!(validation_result.stability_score >= 0.0 && validation_result.stability_score <= 1.0,
        "Stability score should be in valid range");
    assert!(!validation_result.perturbation_results.is_empty(), "Should have perturbation results");
    println!("   ✓ Perturbation validation execution");
    
    // Test 4: Validation results structure
    for result in &validation_result.perturbation_results {
        assert!(result.stability_score >= 0.0 && result.stability_score <= 1.0,
            "Individual stability scores should be valid");
        assert!(!result.description.is_empty(), "Should have test description");
    }
    println!("   ✓ Validation results structure");
    
    // Test 5: Quality assessment
    let quality = &validation_result.quality_assessment;
    assert!(quality.confidence_in_resolution >= 0.0 && quality.confidence_in_resolution <= 1.0,
        "Quality confidence should be valid");
    assert!(!quality.quality_metrics.is_empty(), "Should have quality metrics");
    println!("   ✓ Quality assessment generation");
    
    // Test 6: Recommendations
    assert!(!validation_result.recommendations.is_empty() || 
            validation_result.stability_score > 0.9, 
            "Should provide recommendations unless very stable");
    println!("   ✓ Recommendation generation");
    
    Ok(())
}

/// Test Hybrid Processing paradigm
async fn test_hybrid_processing() -> Result<()> {
    println!("Testing: 'The whole probabilistic system can be tucked inside probabilistic processes'");
    
    let config = HybridConfig {
        probabilistic_threshold: 0.75,
        settlement_threshold: 0.85,
        max_roll_iterations: 5, // Keep low for testing
        enable_adaptive_loops: true,
        density_resolution: 10,
        stream_buffer_size: 256,
    };
    
    let mut processor = HybridProcessor::new(config);
    
    // Test 1: Probabilistic floor creation
    let mut floor = ProbabilisticFloor::new(0.3);
    
    let test_point = point!(
        "Quantum computing will solve optimization problems",
        certainty: 0.67,
        evidence_strength: 0.54
    );
    
    floor.add_point("test_hypothesis".to_string(), test_point, 0.75);
    
    assert_eq!(floor.points.len(), 1, "Floor should contain added point");
    assert!(floor.total_mass > 0.0, "Floor should have positive total mass");
    println!("   ✓ Probabilistic floor creation");
    
    // Test 2: Cycle operation
    let cycle_results = processor.cycle(&floor, |point, weight| {
        assert!(weight > 0.0, "Weight should be positive");
        assert!(!point.content.is_empty(), "Point should have content");
        
        Ok(kwasa_kwasa::turbulance::ResolutionResult {
            confidence: point.certainty * weight,
            resolution_text: format!("Processed with weight {:.2}", weight),
            strategy: ResolutionStrategy::MaximumLikelihood,
            metadata: HashMap::new(),
        })
    }).await?;
    
    assert!(!cycle_results.is_empty(), "Cycle should produce results");
    assert_eq!(cycle_results.len(), floor.points.len(), "Should process all points");
    println!("   ✓ Cycle operation");
    
    // Test 3: Drift operation
    let corpus = "Artificial intelligence is advancing rapidly. Machine learning shows promise. \
                  Deep learning achieves remarkable results. Neural networks continue to improve.";
    
    let drift_results = processor.drift(corpus).await?;
    assert!(!drift_results.is_empty(), "Drift should produce results");
    println!("   ✓ Drift operation");
    
    // Test 4: Roll until settled
    let uncertain_point = point!(
        "Time travel is theoretically possible",
        certainty: 0.35,
        evidence_strength: 0.28
    );
    
    let roll_result = processor.roll_until_settled(&uncertain_point).await?;
    assert!(roll_result.iterations > 0, "Should perform at least one iteration");
    assert!(roll_result.confidence >= 0.0 && roll_result.confidence <= 1.0,
        "Final confidence should be valid");
    println!("   ✓ Roll until settled");
    
    // Test 5: Hybrid function
    let paragraph = "AI research is progressing. Some challenges remain. Progress is being made.";
    
    let hybrid_results = processor.hybrid_function(paragraph, 0.7, |sentence| {
        Ok(sentence.len() > 10) // Simple condition for testing
    }).await?;
    
    assert!(!hybrid_results.is_empty(), "Hybrid function should produce results");
    println!("   ✓ Hybrid function with conditional processing");
    
    // Test 6: Processing statistics
    let stats = processor.get_stats();
    assert!(stats.cycles_performed > 0, "Should track cycle operations");
    println!("   ✓ Processing statistics tracking");
    
    Ok(())
}

/// Test Revolutionary Integration (all paradigms working together)
async fn test_revolutionary_integration() -> Result<()> {
    println!("Testing: All four paradigms working together in integrated pipeline");
    
    // Test 1: Pipeline creation
    let config = PipelineConfig {
        enable_comprehensive_validation: false, // Disable for faster testing
        auto_create_debates: true,
        debate_uncertainty_threshold: 0.6,
        max_processing_time: 30, // 30 seconds max
        ..Default::default()
    };
    
    let mut pipeline = KwasaKwasaPipeline::new(config);
    println!("   ✓ Pipeline creation");
    
    // Test 2: Text processing integration
    let test_text = "
    Artificial intelligence has made remarkable progress in recent years. Large language models 
    demonstrate impressive capabilities in reasoning and text generation. However, questions 
    remain about whether these systems truly understand language or merely perform sophisticated 
    pattern matching. The debate continues in the scientific community.
    ";
    
    let result = pipeline.process_text(test_text).await?;
    
    // Test 3: Validate integration results
    assert!(!result.points.is_empty(), "Should extract points from text");
    assert!(result.quality_assessment.overall_score >= 0.0 && 
            result.quality_assessment.overall_score <= 1.0,
        "Quality score should be valid");
    println!("   ✓ Text processing with {} points extracted", result.points.len());
    
    // Test 4: Validate paradigm integration
    for (i, validated_point) in result.points.iter().enumerate() {
        // Points and Resolutions
        assert!(!validated_point.point.content.is_empty(), "Point should have content");
        assert!(validated_point.point.certainty > 0.0, "Point should have uncertainty");
        
        // Positional Semantics
        assert!(!validated_point.positional_analysis.words.is_empty(), 
            "Should have positional analysis");
        assert!(validated_point.positional_analysis.order_dependency_score >= 0.0,
            "Should have order dependency score");
        
        // Perturbation Validation
        assert!(validated_point.validation.stability_score >= 0.0 && 
                validated_point.validation.stability_score <= 1.0,
            "Should have valid stability score");
        
        // Resolution
        assert!(validated_point.resolution.confidence > 0.0,
            "Should have resolution confidence");
        
        if i == 0 { // Only print details for first point
            println!("   ✓ Point {}: All paradigms integrated", i + 1);
            println!("      - Content: '{}'", 
                validated_point.point.content.chars().take(50).collect::<String>());
            println!("      - Positional analysis: {:.1%} order dependency", 
                validated_point.positional_analysis.order_dependency_score);
            println!("      - Validation stability: {:.1%}", 
                validated_point.validation.stability_score);
            println!("      - Final confidence: {:.1%}", 
                validated_point.resolution.confidence);
        }
    }
    
    // Test 5: Quality assessment integration
    let quality = &result.quality_assessment;
    assert!(quality.positional_coherence >= 0.0 && quality.positional_coherence <= 1.0,
        "Positional coherence should be valid");
    assert!(quality.validation_stability >= 0.0 && quality.validation_stability <= 1.0,
        "Validation stability should be valid");
    assert!(quality.resolution_confidence >= 0.0 && quality.resolution_confidence <= 1.0,
        "Resolution confidence should be valid");
    println!("   ✓ Integrated quality assessment");
    
    // Test 6: Pipeline statistics
    let stats = pipeline.get_stats();
    assert!(stats.texts_processed > 0, "Should track processed texts");
    assert!(stats.points_extracted > 0, "Should track extracted points");
    println!("   ✓ Pipeline statistics tracking");
    
    println!("   🎉 Revolutionary Integration: All paradigms working together!");
    
    Ok(())
}

/// Helper function to assert conditions with custom error messages
fn assert(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(TurbulanceError::RuntimeError { 
            message: message.to_string() 
        });
    }
    Ok(())
} 