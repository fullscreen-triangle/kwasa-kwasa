/// Comprehensive Hybrid Processing Demonstration
/// 
/// This binary demonstrates the revolutionary hybrid processing capabilities
/// that seamlessly blend deterministic and probabilistic operations within
/// the same control flow constructs.

use kwasa_kwasa::turbulance::{
    demonstrate_hybrid_processing,
    demonstrate_turbulance_syntax,
    HybridProcessor, HybridConfig, ProbabilisticFloor,
    TurbulanceProcessor, TurbulanceFunction, TurbulanceOperation,
    TurbulanceCondition, ConfidenceOperator, TurbulanceType,
    ProcessingMode,
    TextPoint, Value,
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎵 Kwasa-Kwasa Hybrid Processing - Revolutionary Loop Constructs 🎵");
    println!("==================================================================\n");
    
    // Part 1: Basic hybrid processing demonstration
    println!("PART 1: Core Hybrid Processing Operations");
    println!("=========================================");
    demonstrate_hybrid_processing().await?;
    
    println!("\n" + "=".repeat(80).as_str() + "\n");
    
    // Part 2: Turbulance syntax demonstration
    println!("PART 2: Turbulance Language Syntax");
    println!("==================================");
    demonstrate_turbulance_syntax().await?;
    
    println!("\n" + "=".repeat(80).as_str() + "\n");
    
    // Part 3: Advanced hybrid processing scenarios
    println!("PART 3: Advanced Hybrid Processing Scenarios");
    println!("===========================================");
    demonstrate_advanced_scenarios().await?;
    
    println!("\n" + "=".repeat(80).as_str() + "\n");
    
    // Part 4: Real-world application example
    println!("PART 4: Real-World Application Example");
    println!("=====================================");
    demonstrate_real_world_application().await?;
    
    println!("\n🎵 Complete hybrid processing demonstration finished! 🎵");
    println!("This showcases a completely new paradigm in text processing:");
    println!("  ✨ Probabilistic loops that adapt based on uncertainty");
    println!("  🔄 Seamless switching between deterministic and probabilistic modes");
    println!("  🎲 Iterative resolution until settlement");
    println!("  🌊 Streaming processing with adaptive control flow");
    println!("  🏛️  Debate platforms for resolving uncertain points");
    println!("  📊 Positional semantics awareness");
    println!("  🧪 Perturbation validation for robustness");
    
    Ok(())
}

/// Demonstrate advanced hybrid processing scenarios
async fn demonstrate_advanced_scenarios() -> Result<()> {
    println!("🔬 Advanced Scenarios\n");
    
    // Scenario 1: Scientific paper analysis with uncertainty handling
    println!("Scenario 1: 📄 Scientific Paper Analysis");
    let mut processor = HybridProcessor::new(HybridConfig {
        probabilistic_threshold: 0.6,
        settlement_threshold: 0.8,
        max_roll_iterations: 15,
        ..HybridConfig::default()
    });
    
    let scientific_text = "The experimental results demonstrate a significant improvement in performance. However, the statistical significance (p=0.047) is borderline. Additional validation studies are recommended. The methodology appears sound but sample size limitations may affect generalizability.";
    
    println!("   Analyzing: '{}'", scientific_text);
    let results = processor.drift(scientific_text).await?;
    
    for result in &results {
        println!("   📊 {}: confidence {:.2}, mode: {}", 
                 result.input, result.confidence, result.mode);
    }
    println!();
    
    // Scenario 2: Multi-threshold processing
    println!("Scenario 2: 🎚️  Multi-Threshold Processing");
    let uncertain_claim = TextPoint::new("This controversial finding requires further investigation".to_string(), 0.2);
    
    println!("   Rolling uncertain claim until settled...");
    let roll_result = processor.roll_until_settled(&uncertain_claim).await?;
    
    println!("   📈 Original: {:.2} → Final: {:.2} (iterations: {})", 
             uncertain_claim.confidence, roll_result.confidence, roll_result.iterations);
    println!("   ✅ Settled: {}", roll_result.settled);
    println!();
    
    // Scenario 3: Streaming with context awareness
    println!("Scenario 3: 📡 Context-Aware Streaming");
    let context_lines = vec![
        "Artificial intelligence has advanced rapidly.".to_string(),
        "Machine learning models show promising results.".to_string(),
        "However, concerns about bias remain significant.".to_string(),
        "Ethical considerations must guide development.".to_string(),
        "The future potential is both exciting and uncertain.".to_string(),
    ];
    
    let flow_results = processor.flow(&context_lines).await?;
    
    for (i, result) in flow_results.iter().enumerate() {
        println!("   Line {}: '{}' (confidence: {:.2}, settled: {})", 
                 i+1, result.input, result.confidence, result.settled);
    }
    println!();
    
    Ok(())
}

/// Demonstrate real-world application
async fn demonstrate_real_world_application() -> Result<()> {
    println!("🌍 Real-World Application: Research Paper Evaluation\n");
    
    let mut turbulance_processor = TurbulanceProcessor::new();
    
    // Set up research paper content
    let research_paper = "Recent advances in quantum computing have demonstrated quantum advantage in specific domains. The IBM quantum processor achieved a 100x speedup over classical simulation. However, error rates remain high at 0.1% per gate operation. Noise mitigation techniques show promise but require further development. Commercial applications may be feasible within the next decade, though significant challenges remain.";
    
    turbulance_processor.variables.insert("paper".to_string(), Value::String(research_paper.to_string()));
    
    // Define a comprehensive evaluation function
    let evaluation_function = TurbulanceFunction {
        name: "evaluate_research_paper".to_string(),
        parameters: vec!["paper".to_string()],
        operations: vec![
            // First, do a basic analysis pass
            TurbulanceOperation::Drift {
                text_var: "claim".to_string(),
                corpus_var: "paper".to_string(),
                body: Box::new(TurbulanceOperation::Block(vec![
                    TurbulanceOperation::Resolution {
                        operation: "analyze".to_string(),
                        target: "claim".to_string(),
                    },
                ])),
            },
            
            // Then do detailed sentence-by-sentence evaluation
            TurbulanceOperation::Considering {
                item_var: "sentence".to_string(),
                collection_var: "paper".to_string(),
                condition: Box::new(TurbulanceCondition::ProbabilisticConfidence {
                    resolution_var: "resolution_confidence".to_string(),
                    threshold: 0.7,
                    operator: ConfidenceOperator::LessThan,
                }),
                body: Box::new(TurbulanceOperation::Block(vec![
                    TurbulanceOperation::Assignment {
                        var_name: "current_point".to_string(),
                        value: Value::String("Evaluating research claim".to_string()),
                    },
                    TurbulanceOperation::RollUntilSettled {
                        body: Box::new(TurbulanceOperation::Resolution {
                            operation: "guess".to_string(),
                            target: "current_point".to_string(),
                        }),
                    },
                ])),
            },
            
            TurbulanceOperation::Return(Value::String("comprehensive evaluation completed".to_string())),
        ],
        return_type: TurbulanceType::Value,
    };
    
    turbulance_processor.register_function(evaluation_function);
    
    println!("📋 Evaluating research paper with hybrid processing...");
    println!("   Paper: '{}'", research_paper);
    println!();
    
    let args = vec![Value::String(research_paper.to_string())];
    let result = turbulance_processor.execute_function("evaluate_research_paper", args).await?;
    
    println!("✅ Evaluation completed: {}", turbulance_processor.value_to_string(&result));
    println!();
    
    // Show the power of probabilistic floor processing
    println!("📊 Probabilistic Floor Analysis:");
    let mut floor = ProbabilisticFloor::new(0.6);
    
    floor.add_point("quantum_advantage".to_string(), 
                    TextPoint::new("Quantum advantage demonstrated".to_string(), 0.8), 1.0);
    floor.add_point("error_rates".to_string(), 
                    TextPoint::new("Error rates remain high".to_string(), 0.9), 1.2);
    floor.add_point("commercial_viability".to_string(), 
                    TextPoint::new("Commercial applications feasible".to_string(), 0.4), 0.6);
    floor.add_point("future_potential".to_string(), 
                    TextPoint::new("Significant challenges remain".to_string(), 0.7), 0.9);
    
    // Demonstrate probabilistic sampling
    println!("   Probabilistic floor contains {} points", floor.points.len());
    println!("   Total probability mass: {:.2}", floor.total_mass);
    
    for (key, point, weight) in floor.probabilistic_iter() {
        println!("   • {} (weight: {:.2}): '{}' (confidence: {:.2})", 
                 key, weight, point.content, point.confidence);
    }
    
    // Sample a few points
    println!("\n   🎲 Random sampling from floor:");
    for i in 1..=3 {
        if let Some((key, point)) = floor.sample_point() {
            println!("   Sample {}: {} → '{}'", i, key, point.content);
        }
    }
    
    println!("\n🎯 This demonstrates how hybrid processing enables:");
    println!("   • Adaptive analysis based on content uncertainty");
    println!("   • Iterative refinement for uncertain claims");
    println!("   • Probabilistic sampling for diverse perspectives");
    println!("   • Seamless integration of multiple processing modes");
    
    Ok(())
} 