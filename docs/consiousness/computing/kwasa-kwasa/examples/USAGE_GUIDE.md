# Kwasa-Kwasa Usage Guide

This guide provides comprehensive documentation for using the Kwasa-Kwasa text processing framework with the Turbulance language.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kwasa-kwasa
cd kwasa-kwasa

# Build the project
cargo build --release

# Run examples
cargo run -- run examples/basic_example.trb
```

### Basic Usage

```bash
# Run a Turbulance script
kwasa-kwasa run script.turb

# Start interactive REPL
kwasa-kwasa repl

# Process a document
kwasa-kwasa process document.txt --format json

# Validate syntax
kwasa-kwasa validate script.turb
```

## Turbulance Language Reference

### Basic Syntax

```turbulance
// Function declaration
funxn analyze_text(text, domain="general"):
    item score = readability_score(text)
    item keywords = extract_keywords(text, 10)
    
    within text:
        given score < 50:
            simplify_sentences()
        given contains("technical"):
            ensure_explanation_follows()
    
    return score

// Project declaration
project research_analysis:
    source academic_papers from "papers/"
    goal "Extract key insights from research papers"
    
    funxn process_paper(paper):
        item summary = extract_summary(paper)
        item citations = extract_citations(paper)
        return {summary: summary, citations: citations}
```

### Text Unit Operations

```turbulance
// Text division and manipulation
item paragraphs = text / "paragraph"
item sentences = text / "sentence"
item words = text / "word"

// Text combination
item combined = text1 + text2
item repeated = text * 3

// Text filtering
item filtered = text - stopwords
```

### Control Flow

```turbulance
// Conditional processing
within document:
    given readability_score(text) < 60:
        simplify_sentences()
        replace_jargon()
    else:
        enhance_vocabulary()

// Ensuring conditions
ensure word_count(text) > 500
ensure contains(text, "conclusion")
```

## Standard Library Functions

### Text Analysis

```turbulance
// Basic analysis
readability_score(text)              // Returns 0-100 score
sentiment_analysis(text)             // Returns sentiment object
extract_keywords(text, count=10)     // Extracts top keywords
contains(text, pattern)              // Pattern matching
extract_patterns(text, regex)       // Regex extraction

// Statistical Analysis
ngram_probability(text, sequence, n=3)
conditional_probability(text, sequence, condition)
positional_distribution(text, pattern)
entropy_measure(text, window_size=50)
sequence_significance(text, sequence)
markov_transition(text, order=1)
zipf_analysis(text)
positional_entropy(text, unit="paragraph")
contextual_uniqueness(text, sequence)
```

### Text Transformation

```turbulance
// Content modification
simplify_sentences(text, level="moderate")
replace_jargon(text, domain="general")
formalize(text)
expand_abbreviations(text)
normalize_style(text)
```

### Research Integration

```turbulance
// Knowledge integration
research_context(topic, depth="medium")
fact_check(statement)
ensure_explanation_follows(term)
cite_sources(text)
verify_claims(text)
```

### Cross-Domain Analysis

```turbulance
// Scientific analysis
motif_enrichment(genomic_sequence, motif)
spectral_correlation(spectrum1, spectrum2)
evidence_likelihood(evidence_network, hypothesis)
uncertainty_propagation(evidence_network, node_id)
bayesian_update(prior_belief, new_evidence)
confidence_interval(measurement, confidence_level)
cross_domain_correlation(genomic_data, spectral_data)
false_discovery_rate(matches, null_model)
permutation_significance(observed, randomized)

// Positional importance
positional_importance(text, unit="paragraph")
section_weight_map(document)
structural_prominence(text, structure_type="heading")
```

## Advanced Features

### Goal-Oriented Processing

```turbulance
// Set processing goals
project academic_writing:
    goal "Write a comprehensive research paper"
    relevance_threshold 0.8
    
    funxn enhance_for_goal(text):
        item alignment = evaluate_alignment(text)
        given alignment < 0.6:
            suggest_improvements()
        return text
```

### Knowledge Integration

```turbulance
// Using the knowledge database
funxn research_enhanced_writing(topic):
    item context = research_context(topic)
    item facts = fact_check(topic)
    
    ensure len(context) > 0
    return build_content(context, facts)
```

### Multi-Domain Processing

```turbulance
// Genomic analysis
funxn analyze_dna_sequence(sequence):
    item motifs = extract_motifs(sequence)
    item enrichment = motif_enrichment(sequence, "ATCG")
    return {motifs: motifs, enrichment: enrichment}

// Chemical analysis  
funxn analyze_compound(formula):
    item properties = calculate_properties(formula)
    item reactions = predict_reactions(formula)
    return {properties: properties, reactions: reactions}

// Spectral analysis
funxn compare_spectra(spec1, spec2):
    item correlation = spectral_correlation(spec1, spec2)
    item peaks = identify_peaks(spec1)
    return {correlation: correlation, peaks: peaks}
```

## CLI Commands

### Project Management

```bash
# Initialize new project
kwasa-kwasa init my_project --template research

# Analyze project complexity
kwasa-kwasa analyze

# Show project information
kwasa-kwasa info

# Format code files
kwasa-kwasa format src/ --check

# Generate documentation
kwasa-kwasa docs --format html

# Run tests
kwasa-kwasa test --filter "text_analysis"
```

### Configuration

```bash
# Show current configuration
kwasa-kwasa config show

# Set configuration values
kwasa-kwasa config set editor.theme dark
kwasa-kwasa config set analysis.default_domain science

# Reset to defaults
kwasa-kwasa config reset
```

### Benchmarking

```bash
# Run performance benchmarks
kwasa-kwasa bench

# Run specific benchmark
kwasa-kwasa bench --filter "text_operations"
```

## WebAssembly Integration

### Basic Setup

```javascript
import init, { KwasaWasm, KwasaConfig } from './pkg/kwasa_kwasa.js';

async function setupKwasa() {
    await init();
    
    const config = new KwasaConfig();
    config.set_goal("Improve text readability");
    config.set_relevance_threshold(0.7);
    
    const kwasa = new KwasaWasm(config);
    return kwasa;
}
```

### Usage in Web Applications

```javascript
const kwasa = await setupKwasa();

// Execute Turbulance code
const result = kwasa.execute_code(`
    funxn improve_text(text):
        item score = readability_score(text)
        given score < 60:
            return simplify_sentences(text)
        return text
    
    improve_text("This is a complex sentence that needs simplification.")
`);

console.log(result);

// Process text with orchestrator
const processed = kwasa.process_text("Your text here");

// Check goal alignment
const alignment = kwasa.evaluate_alignment("Your text here");
```

## Project Templates

### Research Template

```turbulance
project research_paper:
    source papers from "./data/papers/"
    source citations from "./data/references.bib"
    
    goal "Synthesize research findings into coherent analysis"
    relevance_threshold 0.8
    
    funxn analyze_literature():
        item papers = load_papers()
        item summaries = []
        
        for paper in papers:
            item summary = extract_summary(paper)
            item keywords = extract_keywords(summary, 10)
            summaries.append({paper: paper, summary: summary, keywords: keywords})
        
        return summaries
    
    funxn synthesize_findings(summaries):
        item themes = identify_themes(summaries)
        item connections = find_connections(themes)
        return build_synthesis(themes, connections)
```

### Analysis Template

```turbulance
project data_analysis:
    source datasets from "./data/"
    
    goal "Extract insights from multi-domain datasets"
    
    funxn cross_domain_analysis(genomic_data, spectral_data):
        item correlation = cross_domain_correlation(genomic_data, spectral_data)
        item significance = permutation_significance(correlation, random_baseline)
        
        ensure significance < 0.05
        
        return {
            correlation: correlation,
            significance: significance,
            confidence: calculate_confidence(correlation)
        }
```

## Best Practices

### Code Organization

```turbulance
// Use descriptive function names
funxn analyze_research_paper_readability(paper):
    // Function implementation

// Group related functions in projects
project academic_writing:
    // Related functions here

// Use meaningful variable names
item readability_threshold = 65
item technical_term_density = 0.15
```

### Error Handling

```turbulance
funxn safe_analysis(text):
    ensure len(text) > 0
    ensure typeof(text) == "string"
    
    item result = analyze_text(text)
    
    given result == null:
        return default_analysis()
    
    return result
```

### Performance Optimization

```turbulance
// Use appropriate text units for operations
item sentences = text / "sentence"  // More efficient than word-level for sentence analysis

// Cache expensive computations
item cached_score = memoize(readability_score, text)

// Use streaming for large documents
funxn process_large_document(document):
    item chunks = document / "paragraph"
    item results = []
    
    for chunk in chunks:
        item result = process_chunk(chunk)
        results.append(result)
    
    return merge_results(results)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required dependencies are installed
2. **Performance Issues**: Use appropriate text unit granularity
3. **Memory Usage**: Process large documents in chunks
4. **WebAssembly Issues**: Check browser compatibility and WASM support

### Debug Mode

```bash
# Enable debug output
kwasa-kwasa run script.turb --debug --verbose 3

# Check syntax issues
kwasa-kwasa validate script.turb
```

### Configuration Issues

```bash
# Check current configuration
kwasa-kwasa config show

# Reset corrupted configuration
kwasa-kwasa config reset
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_example.trb` - Basic text processing
- `genomic_analysis.turb` - DNA sequence analysis
- `chemistry_analysis.turb` - Chemical compound analysis
- `spectrometry_analysis.turb` - Spectral data processing
- `cross_domain_analysis.turb` - Multi-domain correlation analysis
- `pattern_analysis.turb` - Advanced pattern recognition
- `proposition_example.turb` - Logic and reasoning
- `wasm_demo.html` - Web integration example

## Contributing

To contribute to the Kwasa-Kwasa framework:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See `CONTRIBUTING.md` for detailed guidelines.

## License

Kwasa-Kwasa is released under the MIT License. See `LICENSE` for details.

## Revolutionary Features Guide

This guide covers the revolutionary paradigms introduced in Kwasa-Kwasa that fundamentally change how text processing and semantic analysis are approached.

## Points and Resolutions Paradigm

### Understanding Points

Points are semantic units with inherent uncertainty, replacing traditional deterministic variables:

```turbulance
// Creating a point with uncertainty
point research_claim = {
    content: "AI improves medical diagnosis accuracy by 23%",
    certainty: 0.78,
    evidence_strength: 0.65,
    contextual_relevance: 0.89,
    semantic_clarity: 0.93
}

// Points automatically track uncertainty propagation
point derived_conclusion = analyze_implications(research_claim)
// derived_conclusion inherits and compounds uncertainty
print(f"Derived certainty: {derived_conclusion.certainty}")  // ~0.73
```

### Creating Resolutions

Resolutions are debate platforms that process affirmations and contentions:

```turbulance
// Define a resolution for evaluating claims
resolution evaluate_medical_claim(point: MedicalPoint) -> MedicalOutcome {
    // Gather supporting evidence (affirmations)
    affirmations = [
        Affirmation {
            content: "Clinical trial with 1,000 patients shows 23% improvement",
            evidence_type: EvidenceType::ClinicalTrial,
            strength: 0.91,
            relevance: 0.88
        },
        Affirmation {
            content: "Peer-reviewed meta-analysis confirms findings",
            evidence_type: EvidenceType::MetaAnalysis,
            strength: 0.95,
            relevance: 0.92
        }
    ]
    
    // Gather challenges (contentions)
    contentions = [
        Contention {
            content: "Study population may not represent general patients",
            evidence_type: EvidenceType::PopulationBias,
            strength: 0.73,
            impact: 0.68
        },
        Contention {
            content: "Short follow-up period limits long-term validity",
            evidence_type: EvidenceType::TemporalLimitation,
            strength: 0.81,
            impact: 0.72
        }
    ]
    
    // Choose resolution strategy based on requirements
    strategy = match confidence_requirement {
        "clinical_decision" => ResolutionStrategy::Conservative,
        "research_hypothesis" => ResolutionStrategy::Exploratory,
        "regulatory_approval" => ResolutionStrategy::Bayesian
    }
    
    return resolve_medical_debate(affirmations, contentions, strategy)
}

// Execute the resolution
item outcome = evaluate_medical_claim(research_claim)
print(f"Resolution confidence: {outcome.confidence}")
print(f"Clinical recommendation: {outcome.recommendation}")
```

### Probabilistic Discourse Networks

Create networks of interconnected points for complex reasoning:

```turbulance
// Build a discourse network
item network = DiscourseNetwork::new()

// Add multiple related points
network.add_point("main_hypothesis", research_claim)
network.add_point("supporting_study", supporting_evidence)
network.add_point("methodology", study_methodology)

// Define relationships between points
network.add_causal_link("supporting_study", "main_hypothesis", strength: 0.84)
network.add_supporting_link("methodology", "supporting_study", strength: 0.91)

// Perform network-wide resolution
item network_outcome = network.global_resolution()
print(f"Network coherence: {network_outcome.coherence_score}")
print(f"Overall confidence: {network_outcome.confidence_interval}")
```

## Positional Semantics

### Understanding Position-Dependent Meaning

Positional semantics recognizes that word location fundamentally affects meaning:

```turbulance
// Analyze positional semantics
item analyzer = PositionalAnalyzer::new()
item sentence = "Significantly, the new treatment reduces mortality."

item analysis = analyzer.analyze(sentence)

// Each word has position-dependent properties
for word in analysis.words {
    print(f"Word: '{word.lexeme}'")
    print(f"  Semantic role: {word.semantic_role}")
    print(f"  Position weight: {word.position_weight}")
    print(f"  Order dependency: {word.order_dependency}")
    print(f"  Structural prominence: {word.structural_prominence}")
}

// Output example:
// Word: 'Significantly'
//   Semantic role: Epistemic
//   Position weight: 0.89 (sentence-initial intensifier)
//   Order dependency: 0.93
//   Structural prominence: 0.76
//
// Word: 'treatment'
//   Semantic role: Subject
//   Position weight: 0.85
//   Order dependency: 0.78
//   Structural prominence: 0.45
```

### Position-Aware Text Operations

All text operations can account for positional semantics:

```turbulance
// Position-weighted similarity comparison
funxn compare_medical_statements(statement1, statement2) -> PositionalSimilarity {
    item analysis1 = PositionalAnalyzer::analyze(statement1)
    item analysis2 = PositionalAnalyzer::analyze(statement2)
    
    item similarity_score = 0.0
    item total_weight = 0.0
    
    // Weight similarity by positional importance
    for (word1, word2) in align_word_sequences(analysis1, analysis2) {
        item semantic_similarity = calculate_semantic_similarity(word1, word2)
        item position_weight = min(word1.position_weight, word2.position_weight)
        item role_compatibility = assess_role_compatibility(word1.semantic_role, word2.semantic_role)
        
        similarity_score += semantic_similarity * position_weight * role_compatibility
        total_weight += position_weight
    }
    
    return PositionalSimilarity {
        weighted_similarity: similarity_score / total_weight,
        role_alignment: calculate_role_alignment(analysis1, analysis2),
        structural_consistency: assess_structural_consistency(analysis1, analysis2)
    }
}

// Usage
item stmt1 = "The treatment significantly improves patient outcomes."
item stmt2 = "Patient outcomes improve significantly with treatment."
item similarity = compare_medical_statements(stmt1, stmt2)

print(f"Positional similarity: {similarity.weighted_similarity}")  // ~0.73
print(f"Role alignment: {similarity.role_alignment}")              // ~0.68
print(f"Structural consistency: {similarity.structural_consistency}") // ~0.81
```

### Position-Dependent Embeddings

Generate embeddings that capture position-dependent meaning:

```turbulance
// Create position-aware embeddings
item embedding_model = PositionalEmbeddingModel::new()

// Same word, different positions = different embeddings
item critical_beginning = embedding_model.embed("critical", Position::SentenceInitial)
item critical_middle = embedding_model.embed("critical", Position::MidSentence)
item critical_end = embedding_model.embed("critical", Position::SentenceFinal)

// Compare positional embedding differences
item beg_mid_sim = cosine_similarity(critical_beginning, critical_middle)  // ~0.74
item mid_end_sim = cosine_similarity(critical_middle, critical_end)        // ~0.82
item beg_end_sim = cosine_similarity(critical_beginning, critical_end)     // ~0.69

print(f"Position affects meaning: beginning-end similarity = {beg_end_sim}")
```

## Perturbation Validation

### Systematic Stability Testing

Test the robustness of resolutions through systematic perturbations:

```turbulance
// Create perturbation validator
item validator = PerturbationValidator::new()

// Original text and resolution
item original = "Machine learning algorithms demonstrate significant improvements in diagnostic accuracy."
item original_resolution = evaluate_claim(extract_point(original))

// Apply comprehensive perturbation testing
item validation = validator.validate_comprehensive(original, original_resolution)

// Analyze results
print(f"Overall stability: {validation.overall_stability}")
print(f"Reliability category: {validation.reliability_category}")

for perturbation in validation.perturbation_results {
    print(f"Perturbation type: {perturbation.type}")
    print(f"  Modified text: {perturbation.modified_text}")
    print(f"  Stability score: {perturbation.stability_score}")
    print(f"  Semantic preservation: {perturbation.semantic_preservation}")
}
```

### Specific Perturbation Types

#### Word Removal Testing
```turbulance
// Test which words are essential for meaning
item removal_analysis = validator.word_removal_analysis(original)

for word_test in removal_analysis.results {
    print(f"Removing '{word_test.word}': semantic impact = {word_test.impact}")
    
    if word_test.impact > 0.8 {
        print(f"  → Critical word for meaning preservation")
    } else if word_test.impact < 0.3 {
        print(f"  → Redundant word, safe to remove")
    }
}
```

#### Positional Rearrangement Testing
```turbulance
// Test sensitivity to word order changes
item rearrangement = validator.positional_rearrangement_analysis(original)

print(f"Most stable arrangement: {rearrangement.most_stable.text}")
print(f"  Stability score: {rearrangement.most_stable.score}")

print(f"Least stable arrangement: {rearrangement.least_stable.text}")
print(f"  Stability score: {rearrangement.least_stable.score}")

print(f"Order sensitivity: {rearrangement.order_sensitivity}")
```

#### Synonym Substitution Testing
```turbulance
// Test robustness to word choice variations
item substitution = validator.synonym_substitution_analysis(original)

for sub_test in substitution.results {
    print(f"'{sub_test.original}' → '{sub_test.replacement}'")
    print(f"  Semantic distance: {sub_test.semantic_distance}")
    print(f"  Resolution stability: {sub_test.resolution_stability}")
    print(f"  Acceptable substitution: {sub_test.is_acceptable}")
}
```

## Hybrid Processing with Probabilistic Loops

### Probabilistic Floor Management

Create and manage collections of points with uncertainty:

```turbulance
// Initialize a probabilistic floor
item floor = ProbabilisticFloor::new()

// Add points with varying certainty and weights
floor.add_point("primary_claim", certainty: 0.84, weight: 0.91)
floor.add_point("supporting_evidence", certainty: 0.72, weight: 0.78)
floor.add_point("methodology", certainty: 0.95, weight: 0.65)
floor.add_point("limitations", certainty: 0.88, weight: 0.44)

// Sample points based on combined certainty and weight
item selected_points = floor.sample_weighted(count: 3)
print(f"Selected points for processing: {selected_points}")

// Get probability distribution across points
item distribution = floor.get_probability_distribution()
for (point_id, probability) in distribution {
    print(f"Point '{point_id}': selection probability = {probability}")
}
```

### Hybrid Loop Types

#### Cycle Loop - Confidence-Based Iteration
```turbulance
// Basic cycle with confidence tracking
cycle item over floor:
    item analysis_result = resolution.analyze(item)
    
    // Switch processing mode based on confidence
    if analysis_result.confidence > 0.75:
        // High confidence: use fast deterministic processing
        continue_deterministic()
        resolution.fast_process(item)
    else:
        // Low confidence: switch to probabilistic mode
        switch_to_probabilistic_mode()
        resolution.deep_probabilistic_analysis(item)
    
    // Track confidence evolution
    track_confidence_evolution(item, analysis_result.confidence)
```

#### Drift Loop - Probabilistic Exploration
```turbulance
// Probabilistic drift through content
drift text_chunk in large_corpus:
    item complexity = assess_chunk_complexity(text_chunk)
    item uncertainty = assess_chunk_uncertainty(text_chunk)
    
    // Probability of deep analysis depends on complexity and uncertainty
    item analysis_probability = calculate_drift_probability(complexity, uncertainty)
    
    if random() < analysis_probability:
        // Deep probabilistic analysis
        resolution.comprehensive_analysis(text_chunk)
    else:
        // Quick scanning, move to next cluster
        resolution.quick_scan(text_chunk)
        jump_to_next_semantic_cluster()
```

#### Flow Loop - Streaming Processing
```turbulance
// Adaptive streaming processing
flow line on floor:
    item line_properties = analyze_line_properties(line)
    item processing_mode = determine_optimal_mode(line_properties)
    
    match processing_mode {
        ProcessingMode::FastDeterministic => {
            resolution.fast_parse(line)
        },
        ProcessingMode::ProbabilisticAnalysis => {
            resolution.probabilistic_deep_parse(line)
        },
        ProcessingMode::HybridAdaptive => {
            item quick_result = resolution.fast_parse(line)
            if quick_result.confidence < 0.7 {
                // Refine with probabilistic methods
                resolution.probabilistic_refinement(quick_result)
            }
        }
    }
```

#### Roll Until Settled Loop - Iterative Convergence
```turbulance
// Iterative resolution until convergence
roll until settled:
    item current_state = resolution.assess_current_state()
    
    // Check multiple convergence criteria
    if current_state.is_settled(tolerance: 0.05) {
        break settled(current_state)
    }
    
    // Update evidence base for next iteration
    resolution.gather_additional_evidence()
    resolution.update_affirmations(current_state.new_support)
    resolution.update_contentions(current_state.new_challenges)
    
    // Adaptive convergence criteria
    if iteration_count > 5:
        increase_tolerance(0.02)  // Relax criteria if taking too long
    
    // Track convergence progress
    track_convergence_progress(current_state)
```

### Complete Hybrid Function Example

```turbulance
// Full hybrid function demonstrating all features
funxn comprehensive_document_analysis(document, confidence_threshold=0.75) -> HybridAnalysisResult {
    // Initialize probabilistic floor from document
    item floor = ProbabilisticFloor::from_document(document)
    item processor = HybridProcessor::new(confidence_threshold)
    
    // Multi-stage analysis with adaptive processing
    item analysis_results = []
    
    // Stage 1: Stream through document sections
    flow section on floor:
        item section_complexity = calculate_complexity(section)
        
        // Adaptive mode selection
        processor.set_mode_based_on_complexity(section_complexity)
        
        considering sentence in section:
            // Extract points with positional awareness
            item points = extract_points_with_position(sentence)
            
            // Conditional probabilistic processing
            if sentence.contains_high_uncertainty_points():
                switch_to_probabilistic_mode()
                
                // Process each point through resolution
                for point in points:
                    item resolution = create_adaptive_resolution(point)
                    
                    // Probabilistic convergence loop
                    roll until settled:
                        item assessment = resolution.assess(point)
                        
                        if assessment.confidence > confidence_threshold:
                            // Validate with perturbation testing
                            item validation = perturbation_validate(point, assessment)
                            
                            if validation.reliability >= ReliabilityCategory::Reliable:
                                accept_point_resolution(assessment)
                                break settled(assessment)
                            else:
                                // Expand evidence search if validation fails
                                resolution.expand_evidence_search()
                        else:
                            // Gather more evidence and continue
                            resolution.enhance_evidence_base()
                            resolution.update_debate_parameters()
            else:
                // Use deterministic mode for simple content
                switch_to_deterministic_mode()
                simple_deterministic_processing(sentence)
        
        analysis_results.push(section_analysis)
    
    // Stage 2: Network-wide coherence analysis
    item discourse_network = build_discourse_network(analysis_results)
    item network_coherence = discourse_network.assess_global_coherence()
    
    // Stage 3: Final confidence assessment
    cycle result over analysis_results:
        item result_confidence = result.calculate_confidence()
        
        if result_confidence > 0.8:
            continue_deterministic()
        else:
            switch_to_probabilistic_mode()
            refine_analysis_result(result)
    
    // Generate comprehensive report
    return HybridAnalysisResult {
        processed_document: document,
        point_resolutions: collect_all_resolutions(),
        positional_analysis: get_positional_analysis_summary(),
        perturbation_validation: get_validation_summary(),
        confidence_distribution: calculate_confidence_distribution(),
        processing_mode_history: get_mode_transition_history(),
        network_coherence: network_coherence,
        overall_reliability: calculate_overall_reliability(),
        recommendations: generate_actionable_recommendations()
    }
}

// Execute comprehensive analysis
item document = load_document("complex_research_paper.txt")
item result = comprehensive_document_analysis(document, confidence_threshold: 0.8)

// Display results
print(f"Document Analysis Complete")
print(f"==========================")
print(f"Overall Reliability: {result.overall_reliability}")
print(f"Network Coherence: {result.network_coherence}")
print(f"Validated Claims: {result.perturbation_validation.validated_count}")
print(f"Processing Modes Used: {result.processing_mode_history}")
print(f"Recommendations: {result.recommendations}")
```

## Best Practices

### When to Use Points vs Traditional Variables
- Use **Points** when dealing with uncertain, subjective, or context-dependent information
- Use traditional variables for deterministic, objective data

### Choosing Resolution Strategies
- **Bayesian**: When you have prior knowledge and want rigorous statistical inference
- **Maximum Likelihood**: When you want the most probable interpretation
- **Conservative**: When the cost of false positives is high (medical, legal)
- **Exploratory**: When discovering new patterns or hypotheses

### Optimizing Positional Analysis
- Use higher position weights for content words (nouns, verbs, adjectives)
- Consider discourse markers and transition words for structural analysis
- Account for domain-specific positional conventions

### Effective Perturbation Testing
- Start with word removal to identify critical terms
- Use positional rearrangement for order-sensitive content
- Apply synonym substitution for robustness testing
- Combine multiple perturbation types for comprehensive validation

### Hybrid Processing Guidelines
- Use probabilistic loops for uncertain, complex content
- Switch to deterministic mode for routine, high-confidence processing
- Monitor convergence carefully in "roll until settled" loops
- Track processing mode transitions for optimization

This completes the comprehensive usage guide for Kwasa-Kwasa's revolutionary features. 