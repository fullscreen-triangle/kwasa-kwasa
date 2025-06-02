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
    var score = readability_score(text)
    var keywords = extract_keywords(text, 10)
    
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
        var summary = extract_summary(paper)
        var citations = extract_citations(paper)
        return {summary: summary, citations: citations}
```

### Text Unit Operations

```turbulance
// Text division and manipulation
var paragraphs = text / "paragraph"
var sentences = text / "sentence"
var words = text / "word"

// Text combination
var combined = text1 + text2
var repeated = text * 3

// Text filtering
var filtered = text - stopwords
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
        var alignment = evaluate_alignment(text)
        given alignment < 0.6:
            suggest_improvements()
        return text
```

### Knowledge Integration

```turbulance
// Using the knowledge database
funxn research_enhanced_writing(topic):
    var context = research_context(topic)
    var facts = fact_check(topic)
    
    ensure len(context) > 0
    return build_content(context, facts)
```

### Multi-Domain Processing

```turbulance
// Genomic analysis
funxn analyze_dna_sequence(sequence):
    var motifs = extract_motifs(sequence)
    var enrichment = motif_enrichment(sequence, "ATCG")
    return {motifs: motifs, enrichment: enrichment}

// Chemical analysis  
funxn analyze_compound(formula):
    var properties = calculate_properties(formula)
    var reactions = predict_reactions(formula)
    return {properties: properties, reactions: reactions}

// Spectral analysis
funxn compare_spectra(spec1, spec2):
    var correlation = spectral_correlation(spec1, spec2)
    var peaks = identify_peaks(spec1)
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
        var score = readability_score(text)
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
        var papers = load_papers()
        var summaries = []
        
        for paper in papers:
            var summary = extract_summary(paper)
            var keywords = extract_keywords(summary, 10)
            summaries.append({paper: paper, summary: summary, keywords: keywords})
        
        return summaries
    
    funxn synthesize_findings(summaries):
        var themes = identify_themes(summaries)
        var connections = find_connections(themes)
        return build_synthesis(themes, connections)
```

### Analysis Template

```turbulance
project data_analysis:
    source datasets from "./data/"
    
    goal "Extract insights from multi-domain datasets"
    
    funxn cross_domain_analysis(genomic_data, spectral_data):
        var correlation = cross_domain_correlation(genomic_data, spectral_data)
        var significance = permutation_significance(correlation, random_baseline)
        
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
var readability_threshold = 65
var technical_term_density = 0.15
```

### Error Handling

```turbulance
funxn safe_analysis(text):
    ensure len(text) > 0
    ensure typeof(text) == "string"
    
    var result = analyze_text(text)
    
    given result == null:
        return default_analysis()
    
    return result
```

### Performance Optimization

```turbulance
// Use appropriate text units for operations
var sentences = text / "sentence"  // More efficient than word-level for sentence analysis

// Cache expensive computations
var cached_score = memoize(readability_score, text)

// Use streaming for large documents
funxn process_large_document(document):
    var chunks = document / "paragraph"
    var results = []
    
    for chunk in chunks:
        var result = process_chunk(chunk)
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