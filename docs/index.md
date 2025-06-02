# Kwasa-Kwasa Documentation

Welcome to the complete documentation for Kwasa-Kwasa, a revolutionary text processing framework that treats text as a first-class computational medium.

## Getting Started

### Quick Start Guides
- **[Complete System Guide](complete_system_guide.md)** - Comprehensive explanation of the entire Kwasa-Kwasa system from text units to metacognitive orchestration
- **[Practical Getting Started](examples/getting_started_practical.md)** - Hands-on guide with installation, first programs, and running Turbulance code
- **[Basic Usage](examples/basic-usage.md)** - Fundamental operations and concepts

### Installation
- **System Requirements**: Rust 1.70+, Python 3.8+, Git
- **Quick Install**: `cargo install --path .` after cloning the repository
- **Verification**: `kwasa-kwasa --version`

## Language Reference

### Complete Language Documentation
- **[Turbulance Language Overview](language/turbulance-language.md)** - Complete language reference with syntax, features, and examples
- **[Special Language Features](language/special_features.md)** - Advanced constructs like Propositions, Motions, Evidence structures, Metacognitive structures
- **[Goal System](language/goal.md)** - Comprehensive goal definition, tracking, and achievement system

### Technical Specifications
- **[Operations Specification](spec/operations.md)** - Complete specification of all text operations, mathematical functions, and specialized operations
- **[Turbulance Syntax Specification](spec/turbulance-syntax.md)** - Formal grammar, syntax rules, and language structure specification

## Core Concepts

### Text Units - The Foundation
Text units are the fundamental building blocks of Kwasa-Kwasa:

```turbulance
// Text units exist in a natural hierarchy
var document = "Sample text here."
var sentences = document / sentence  // Division creates smaller units
var words = document / word          // Further division
var paragraph = sentence1 * sentence2 // Multiplication combines units
```

**Hierarchy**: Document → Section → Paragraph → Sentence → Phrase → Word → Character

### Mathematical Operations on Text
- **Division (/)**: Split text into smaller units
- **Multiplication (*)**: Combine units intelligently
- **Addition (+)**: Concatenate while preserving type
- **Subtraction (-)**: Remove content maintaining structure

### The Turbulance Language

**Core Syntax Features:**
- `funxn` - Function declarations
- `given` - Conditional logic (instead of "if")
- `within` - Contextual processing scopes
- `considering all` - Iteration over collections
- `allow`/`cause` - Variable declarations
- `proposition`/`motion` - Hypothesis-driven processing

### Syntax Examples

#### Basic Function
```turbulance
funxn analyze_text(content):
    var readability = readability_score(content)
    given readability < 70:
        improve_readability(content)
    return content
```

#### Propositions and Motions
```turbulance
proposition TextQuality:
    motion Clarity("Text should be clear and unambiguous")
    motion Conciseness("Text should be concise without losing meaning")
    
    within document:
        given readability_score() > 70:
            support Clarity
        given word_count() / idea_count() < 20:
            support Conciseness
```

#### Text Processing Pipeline
```turbulance
var processed = raw_text |>
    normalize_whitespace() |>
    correct_spelling() |>
    improve_readability() |>
    add_section_headers()
```

## Advanced Features

### Specialized Data Structures
- **[Domain Extensions](examples/domain-extensions.md)** - Extensions for genomics, chemistry, mass spectrometry, physics, and materials science

**Core Structures:**
- **TextGraph**: Network relationships between concepts
- **ConceptChain**: Cause-and-effect relationships  
- **ArgMap**: Argument mapping with claims, evidence, objections
- **EvidenceNetwork**: Bayesian networks for scientific evidence
- **IdeaHierarchy**: Hierarchical organization of ideas

### Metacognitive Orchestration
- **[Metacognitive Orchestration](metacognitive-orchestration.md)** - Intelligent control system with three-layer architecture

**Three-Layer Processing:**
1. **Context Layer**: Domain understanding and knowledge base
2. **Reasoning Layer**: Logical processing and rule application  
3. **Intuition Layer**: Pattern recognition and heuristic analysis

### Goal-Oriented Writing
The Goal System enables intelligent, objective-driven text processing:

**Goal Features:**
- **Progress Tracking**: Automatic evaluation of writing objectives
- **Adaptive Guidance**: Dynamic suggestions based on goal alignment
- **Hierarchical Goals**: Complex goals with sub-objectives and dependencies
- **Real-Time Feedback**: Continuous assessment and recommendations

```turbulance
var tutorial_goal = Goal.new("Create beginner-friendly tutorial") {
    success_threshold: 0.85,
    keywords: ["tutorial", "beginner", "step-by-step"],
    domain: "education",
    metrics: {
        readability_score: 65,
        explanation_coverage: 0.9
    }
}
```

## Practical Examples

### Comprehensive Examples
- **[Cross-Domain Analysis](examples/cross_domain_analysis.md)** - Drug discovery combining genomics, chemistry, and physics
- **[Evidence Integration](examples/evidence_integration.md)** - Sophisticated evidence handling with uncertainty quantification

### Domain-Specific Applications

#### Academic Writing Assistant
```turbulance
proposition AcademicWriting:
    motion Formality("Writing maintains academic tone")
    motion Citations("All claims are properly cited")
    
funxn enhance_paper(paper):
    considering all paragraph in paper:
        given needs_citation(paragraph):
            suggest_citations(paragraph)
        given readability_score(paragraph) < 60:
            improve_clarity(paragraph)
```

#### Genomic Analysis
```turbulance
import genomic

var dna = genomic.NucleotideSequence.new("ATGCGATCG", "gene_123")
var codons = dna / codon
var gc_rich = dna.filter("gc_content() > 0.6")

proposition GeneExpression:
    motion HighExpression("Gene shows high expression")
    within dna:
        given contains("TATAAA"):  // TATA box
            support HighExpression
```

#### Chemical Structure Analysis
```turbulance
import chemistry

var caffeine = chemistry.Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
var fragments = caffeine / functional_group

proposition DrugLikeness:
    motion LipiniskiCompliant("Follows Lipinski's Rule of Five")
    within caffeine:
        given molecular_weight() < 500:
            support LipiniskiCompliant
```

## Development and Extensions

### Pattern Analysis
- **[Pattern-Based Analysis](DOMAIN_EXPANSION.md)** - Advanced pattern recognition and analysis techniques

### Scientific Data Extensions
- **[Scientific Data Extensions](SCIENTIFIC_DATA_EXTENSIONS.md)** - Working with scientific datasets and specialized data types

### Performance and Optimization
- **Parallel Processing**: Automatic parallelization of text operations
- **Streaming**: Memory-efficient processing of large documents
- **Caching**: Intelligent caching of expensive operations

## API Reference

### Built-in Functions

**Text Analysis:**
- `readability_score(text)` - Flesch-Kincaid readability score
- `sentiment_analysis(text)` - Polarity and subjectivity analysis
- `extract_keywords(text, count)` - Extract significant keywords
- `word_count(text)`, `sentence_count(text)` - Basic statistics

**Text Transformation:**
- `simplify_sentences(text, level)` - Reduce sentence complexity
- `improve_readability(text)` - Enhance overall readability
- `normalize_whitespace(text)` - Clean whitespace formatting
- `correct_spelling(text)` - Fix spelling errors

**Research Integration:**
- `research_context(topic, depth)` - Retrieve contextual information
- `fact_check(statement)` - Verify factual claims
- `ensure_explanation_follows(term)` - Ensure terms are explained

**Pattern Recognition:**
- `find_patterns(text, pattern_type)` - Identify recurring patterns
- `detect_boundaries(text, boundary_type)` - Find text unit boundaries
- `extract_structure(text)` - Analyze document structure

**Goal Management:**
- `Goal.new(description, threshold)` - Create new writing goals
- `track_progress(goal, text)` - Monitor goal achievement
- `evaluate_alignment(text, goal)` - Assess text-goal alignment
- `generate_suggestions(goal, gaps)` - Get improvement recommendations

### Domain Extension APIs

**Genomic Extension:**
```turbulance
import genomic
var sequence = genomic.NucleotideSequence.new("ATGC...", "gene_id")
var codons = sequence / codon
var gc_content = sequence.gc_content()
```

**Chemistry Extension:**
```turbulance
import chemistry
var molecule = chemistry.Molecule.from_smiles("CCO")
var molecular_weight = molecule.molecular_weight()
var fragments = molecule / functional_group
```

**Mass Spectrometry Extension:**
```turbulance
import mass_spec
var spectrum = mass_spec.Spectrum.from_file("data.mzML")
var peaks = spectrum / peak
var base_peak = spectrum.base_peak()
```

## Mathematical Operations Reference

### Text Unit Arithmetic
```turbulance
// Division - Split text into smaller components
var sentences = paragraph / sentence
var words = sentence / word
var characters = word / character

// Multiplication - Intelligent combination
var paragraph = sentence1 * sentence2 * sentence3
var document = section1 * section2

// Addition - Semantic concatenation  
var extended = text + " additional content"
var clarified = term + " (explanation)"

// Subtraction - Content removal
var cleaned = text - "unwanted phrase"
var formal = text - informal_expressions
```

### Statistical Functions
```turbulance
// Probability analysis
ngram_probability(text, "ing", 3)
conditional_probability(text, "word", "previous_word")
entropy_measure(text, window_size=50)

// Pattern analysis
positional_distribution(text, "keyword")
sequence_significance(text, "pattern")
markov_transition(text, order=2)
```

### Cross-Domain Analysis
```turbulance
// Genomic analysis
motif_enrichment(dna_sequence, "TATAAA")
sequence_alignment(seq1, seq2)

// Chemical analysis
spectral_correlation(spectrum1, spectrum2)
chemical_similarity(molecule1, molecule2)

// Evidence networks
evidence_likelihood(network, hypothesis)
bayesian_update(prior, evidence)
```

## Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-org/kwasa-kwasa.git
cd kwasa-kwasa

# Build project
cargo build --release

# Run tests
cargo test

# Install development tools
cargo install --path .
```

### Architecture Overview
- **Core Engine**: Rust-based text processing engine
- **Language Parser**: Turbulance language interpreter
- **Metacognitive Layer**: Goal management and orchestration
- **Domain Extensions**: Pluggable scientific computing modules
- **API Layer**: Interface for external integrations

## License and Credits

Kwasa-Kwasa is open source software. See LICENSE file for details.

**Philosophy**: *"There is no reason for your soul to be misunderstood"* - The driving principle behind making meaning computationally accessible.

---

## Quick Navigation

**Essential Reading:**
1. [Complete System Guide](complete_system_guide.md) - Start here for full understanding
2. [Practical Getting Started](examples/getting_started_practical.md) - Hands-on learning
3. [Turbulance Language](language/turbulance-language.md) - Language reference

**Advanced Topics:**
- [Domain Extensions](examples/domain-extensions.md) - Specialized applications
- [Metacognitive Orchestration](metacognitive-orchestration.md) - Intelligent control
- [Cross-Domain Analysis](examples/cross_domain_analysis.md) - Complex examples

**For Developers:**
- [Implementation Status](implementation_status.md) - Development progress
- [Scientific Extensions](SCIENTIFIC_DATA_EXTENSIONS.md) - Scientific computing
- [Pattern Analysis](DOMAIN_EXPANSION.md) - Advanced techniques 