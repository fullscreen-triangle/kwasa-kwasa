# Turbulance - Universal Scientific Experiment DSL

[![Crates.io](https://img.shields.io/crates/v/turbulance.svg)](https://crates.io/crates/turbulance)
[![Documentation](https://docs.rs/turbulance/badge.svg)](https://docs.rs/turbulance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turbulance** is a domain-specific language (DSL) designed for formalizing scientific methods and experiments. It enables scientists to express experimental procedures, hypotheses, and data transformations in a programmatic yet natural way while preserving semantic meaning.

## Philosophy

Just as [kwasa-kwasa music](https://en.wikipedia.org/wiki/Kwassa_kwassa) transcended language barriers to communicate meaning that required no translation, Turbulance aims to create a universal language for scientific expression that preserves the essential nature of scientific thought through computational transformation.

## Key Features

- **🔬 Scientific Focus**: Built-in constructs for hypotheses, experiments, and data analysis
- **📊 Semantic Operations**: Text and data transformations that preserve meaning
- **🎯 Proposition System**: Formalize scientific claims and test hypotheses
- **🔄 Reproducible**: Designed for reproducible scientific workflows
- **⚡ Lightweight**: Minimal dependencies, fast execution
- **🌐 Cross-platform**: Desktop, web (WASM), and embedded systems
- **🔧 Extensible**: Plugin system for domain-specific extensions

## Quick Start

### Installation

```bash
# Install as a CLI tool
cargo install turbulance

# Or add to your Rust project
cargo add turbulance
```

### Basic Usage

Create a simple experiment:

```turbulance
// hypothesis.turb
proposition DataQualityHypothesis:
    motion Hypothesis("Higher data quality leads to better model performance")
    
    within experiment:
        given data_quality(dataset) > 0.8:
            item model = train_model(dataset)
            ensure model.accuracy > 0.9
        
        alternatively:
            research "data quality improvement techniques"

funxn validate_hypothesis(dataset):
    item cleaned_data = dataset / noise / outliers
    item quality_score = assess_quality(cleaned_data)
    
    given quality_score > 0.8:
        return "Hypothesis supported"
    alternatively:
        return "Need better data preprocessing"

// Execute
item result = validate_hypothesis(load_data("experiment.csv"))
result
```

Run it:

```bash
turbulance run hypothesis.turb
```

### Using as a Library

```rust
use turbulance::{Engine, Script};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new();
    let script = Script::from_source(r#"
        funxn analyze_correlation(x, y):
            item correlation = pearson(x, y)
            given correlation > 0.7:
                return "Strong positive correlation"
            given correlation < -0.7:
                return "Strong negative correlation"
            alternatively:
                return "Weak correlation"
        
        analyze_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    "#)?;
    
    let result = engine.execute(&script)?;
    println!("Result: {}", result);
    Ok(())
}
```

## Language Features

### Scientific Constructs

#### Propositions and Hypotheses

```turbulance
proposition MemoryRetentionHypothesis:
    motion Hypothesis("Spaced repetition improves long-term retention")
    
    sources:
        local("data/memory_test_results.csv")
        web_search(engines = ["pubmed"])
    
    within experiment:
        given sample_size > 30:
            item control_group = select_controls()
            item treatment_group = apply_spaced_repetition()
            item results = measure_retention(control_group, treatment_group)
            ensure statistical_significance(results) < 0.05
```

#### Semantic Data Operations

```turbulance
// Clean and transform data semantically
item raw_data = load_dataset("experiment.csv")
item cleaned = raw_data / missing_values / outliers / noise
item normalized = cleaned * standardization
item features = extract_features(normalized)

// Semantic addition combines meaningful information
item comprehensive_analysis = statistical_summary(features) + 
                             correlation_analysis(features) + 
                             regression_model(features)
```

#### Experimental Workflows

```turbulance
project ClinicalTrialAnalysis(
    title: "Drug Efficacy Study",
    version: "1.0",
    reproducible: true
):
    funxn setup_trial():
        item participants = recruit_participants(n=100)
        item (control, treatment) = randomize(participants)
        ensure groups_balanced(control, treatment)
        return (control, treatment)
    
    funxn run_trial(control, treatment):
        item baseline = measure_baseline(control + treatment)
        item intervention = apply_treatment(treatment)
        item follow_up = measure_outcomes(control + treatment, weeks=12)
        
        return trial_data {
            baseline: baseline,
            intervention: intervention,
            outcomes: follow_up
        }
    
    funxn analyze_efficacy(trial_data):
        item effect_size = cohen_d(trial_data.control, trial_data.treatment)
        item p_value = t_test(trial_data.control, trial_data.treatment)
        
        given p_value < 0.05 && effect_size > 0.5:
            return "Treatment shows significant efficacy"
        alternatively:
            return "No significant treatment effect"
```

## CLI Usage

### Available Commands

```bash
# Run a Turbulance script
turbulance run experiment.turb

# Validate syntax
turbulance validate analysis.turb

# Interactive mode
turbulance repl

# Create templates
turbulance new my-experiment --type experiment
turbulance new hypothesis --type hypothesis
turbulance new data-analysis --type analysis

# Format code
turbulance format *.turb

# Get help
turbulance --help
```

### Template Types

- **hypothesis**: Creates a template for hypothesis testing
- **experiment**: Creates a full experimental workflow template
- **analysis**: Creates a data analysis workflow template

## Scientific Use Cases

### 1. Data Analysis Workflows

```turbulance
funxn comprehensive_analysis(dataset):
    // Data preprocessing
    item cleaned = dataset / missing_values / outliers
    item features = engineer_features(cleaned)
    
    // Exploratory analysis
    item summary = describe(features)
    item correlations = correlation_matrix(features)
    
    // Statistical modeling
    item model = fit_regression(features)
    item validation = cross_validate(model)
    
    // Interpretation
    given validation.r_squared > 0.8:
        return "Model explains data well"
    alternatively:
        research "alternative modeling approaches"
```

### 2. Hypothesis Testing

```turbulance
proposition DrugEfficacyHypothesis:
    motion Hypothesis("New drug reduces symptoms by >50%")
    
    within clinical_trial:
        given sample_size >= 50:
            item control_outcomes = measure(control_group)
            item treatment_outcomes = measure(treatment_group)
            item improvement = (control_outcomes - treatment_outcomes) / control_outcomes
            
            ensure improvement > 0.5 && p_value(improvement) < 0.05
        
        alternatively:
            motion AdjustDosage("Increase sample size or modify dosage")
```

### 3. Literature Integration

```turbulance
sources:
    web_search(engines = ["pubmed", "arxiv"], query = "machine learning cancer diagnosis")
    local("references/systematic_review.csv")

funxn synthesize_evidence():
    item papers = load_literature()
    item filtered = papers / low_quality / irrelevant
    item effect_sizes = extract_effect_sizes(filtered)
    item meta_analysis = meta_analyze(effect_sizes)
    
    return research_synthesis {
        papers_included: count(filtered),
        pooled_effect: meta_analysis.effect,
        confidence_interval: meta_analysis.ci,
        heterogeneity: meta_analysis.i2
    }
```

## Advanced Features

### Cross-Modal Operations

```turbulance
// Combine textual and numerical data
item clinical_notes = load_text("patient_notes.txt")
item lab_results = load_data("lab_values.csv")
item imaging_data = load_images("scans/")

item comprehensive_assessment = clinical_notes + lab_results + imaging_data
item diagnosis = predict_diagnosis(comprehensive_assessment)
```

### Probabilistic Reasoning

```turbulance
point diagnostic_confidence = {
    content: "Patient has pneumonia based on symptoms and imaging",
    certainty: 0.85,
    evidence_strength: 0.72
}

resolution diagnose_condition(point: DiagnosticPoint) -> DiagnosticOutcome {
    given point.certainty > 0.8 && point.evidence_strength > 0.7:
        return "High confidence diagnosis"
    alternatively:
        return "Additional tests recommended"
}
```

## Extension System

Turbulance supports domain-specific extensions:

```rust
// In your Rust code
use turbulance::{Engine, extension::Extension};

struct BioinformaticsExtension;

impl Extension for BioinformaticsExtension {
    fn name(&self) -> &str { "bioinformatics" }
    
    fn register_functions(&self, engine: &mut Engine) {
        engine.register_function("sequence_align", sequence_alignment);
        engine.register_function("blast_search", blast_search);
        engine.register_function("phylogenetic_tree", build_tree);
    }
}

let mut engine = Engine::new();
engine.add_extension(BioinformaticsExtension);
```

## Performance

Turbulance is designed for performance:

- **Fast parsing**: Uses efficient lexing and parsing techniques
- **Minimal overhead**: Lightweight runtime with minimal memory footprint
- **Streaming support**: Process large datasets without loading everything into memory
- **Parallel execution**: Built-in support for parallel processing of independent operations

## Contributing

We welcome contributions from the scientific community! Areas where we need help:

1. **Domain-specific extensions** (bioinformatics, chemistry, psychology, etc.)
2. **Statistical functions** and scientific computing primitives
3. **Documentation** and examples from your field
4. **Performance optimizations**
5. **Integration with popular scientific tools**

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Comparison with Other Tools

| Feature | Turbulance | R | Python | MATLAB |
|---------|------------|---|--------|--------|
| Scientific DSL | ✅ | ❌ | ❌ | ❌ |
| Semantic Operations | ✅ | ❌ | ❌ | ❌ |
| Hypothesis Formalization | ✅ | ❌ | ❌ | ❌ |
| Cross-platform | ✅ | ✅ | ✅ | ❌ |
| WebAssembly | ✅ | ❌ | ❌ | ❌ |
| Reproducible by Design | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Learning Curve | Low | High | Medium | Medium |

## Examples Repository

Check out the [examples repository](https://github.com/fullscreen-triangle/turbulance-examples) for:

- Complete experimental workflows
- Domain-specific analyses
- Integration patterns
- Best practices
- Tutorial notebooks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the universality of kwasa-kwasa music
- Built on the foundation of modern language design principles
- Designed with input from working scientists across multiple disciplines

---

*"There is no reason for your scientific ideas to be misunderstood"* - The Turbulance Philosophy 