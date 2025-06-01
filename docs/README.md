# Kwasa-Kwasa: Metacognitive Text Processing Framework

<p align="center"><em>There is no reason for your soul to be misunderstood</em></p>

Kwasa-Kwasa is a revolutionary text processing framework that treats text as a first-class computational medium. It provides a complete environment where text becomes mathematically manipulable while preserving semantic meaning.

## Quick Start

```bash
# Install Kwasa-Kwasa
cargo install --path .

# Run your first Turbulance program
kwasa-kwasa run hello.turb
```

## Core Innovation: Text as Computational Medium

Kwasa-Kwasa transforms text processing by introducing **mathematical operations on text** with semantic awareness:

```turbulance
// Mathematical text operations
var sentences = paragraph / sentence    // Division: split into smaller units
var combined = sentence1 * sentence2    // Multiplication: intelligent combination
var extended = text + " more content"   // Addition: semantic concatenation
var filtered = text - "unwanted phrase" // Subtraction: content removal
```

## Turbulance: A Language for Text

Turbulance is a domain-specific language designed exclusively for text operations. It provides a rich, expressive syntax for text manipulation with semantic awareness.

### Key Language Features

- **Boundaries and Text Units**: Define and operate on specific text structures
- **Contextual Transformations**: Apply transformations based on semantic context
- **Knowledge Integration**: Connect with external research sources
- **State Management**: Maintain context across transformations
- **Semantic Operations**: Operate on text while preserving meaning

### Syntax Example

```turbulance
funxn enhance_paragraph(paragraph, domain="general"):
    within paragraph:
        given contains("technical_term"):
            research_context(domain)
            ensure_explanation_follows()
        given readability_score < 65:
            simplify_sentences()
            replace_jargon()
        return processed
```

### Language Structure

Turbulance's lexical structure includes:
- **Keywords**: `funxn`, `within`, `given`, `project`, `ensure`, `return`, etc.
- **Operators**: `/` (division), `*` (multiplication), `+` (addition), `-` (subtraction)
- **Control Structures**: Block expressions, conditional execution, iterations
- **Function System**: Declarations, parameters, closures, return values
- **Special Constructs**: `motion`, `proposition`, `cause`, `considering`, `allow`

### Standard Library

The Turbulance standard library provides built-in functions for text manipulation:

```turbulance
// Text analysis
readability_score(text)              // Returns Flesch-Kincaid score (0-100)
sentiment_analysis(text)             // Returns polarity and subjectivity
extract_keywords(text, count=10)     // Extracts significant keywords

// Text transformation
simplify_sentences(text, level="moderate")  // Simplifies complex sentences
replace_jargon(text, domain="general")      // Replaces specialized terms
formalize(text)                             // Increases formality

// Research assistance
research_context(topic, depth="medium")     // Retrieves contextual information
fact_check(statement)                       // Verifies factual claims
ensure_explanation_follows(term)            // Ensures term is explained

// Utilities
print(value)                                // Outputs to console
len(collection)                             // Returns collection length
typeof(value)                               // Returns type information
```

### Statistical Analysis Functions
- `ngram_probability(text, sequence, n=3)`: Returns probability of a letter sequence given surrounding context
- `conditional_probability(text, sequence, condition)`: Calculates probability of sequence given conditional context
- `positional_distribution(text, pattern)`: Maps occurrences of pattern across different positions in text
- `entropy_measure(text, window_size=50)`: Calculates information entropy within sliding windows
- `sequence_significance(text, sequence)`: Tests statistical significance of a sequence compared to baseline
- `markov_transition(text, order=1)`: Generates transition probability matrix for text elements
- `zipf_analysis(text)`: Analyzes token frequency distribution against Zipf's law
- `positional_entropy(text, unit="paragraph")`: Measures information distribution across structural units
- `contextual_uniqueness(text, sequence)`: Evaluates how distinctive a sequence is in different contexts

### Cross-Domain Statistical Analysis
- `motif_enrichment(genomic_sequence, motif)`: Calculates statistical enrichment of genomic motifs
- `spectral_correlation(spectrum1, spectrum2)`: Computes correlation between mass spectral patterns
- `evidence_likelihood(evidence_network, hypothesis)`: Calculates probability of hypothesis given evidence
- `uncertainty_propagation(evidence_network, node_id)`: Models how uncertainty propagates through evidence
- `bayesian_update(prior_belief, new_evidence)`: Updates belief based on new evidence using Bayes' theorem
- `confidence_interval(measurement, confidence_level)`: Calculates confidence intervals for measurements
- `cross_domain_correlation(genomic_data, spectral_data)`: Finds correlations between multi-domain datasets
- `false_discovery_rate(matches, null_model)`: Estimates false discovery rate in pattern matching results
- `permutation_significance(observed, randomized)`: Calculates significance through permutation testing

### Positional Importance Analysis
- `positional_importance(text, unit="paragraph")`: Calculates importance score based on position within document
- `section_weight_map(document)`: Creates heatmap of importance weights across document sections
- `structural_prominence(text, structure_type="heading")`: Measures text importance based on structural context
- `proximity_weight(text, anchor_points)`: Weights text importance by proximity to key document anchors
- `transition_importance(text)`: Assigns higher importance to text at section transitions or logical boundaries
- `opening_closing_emphasis(text)`: Weights text at the beginning and end of units more heavily
- `local_global_context(text)`: Compares importance of text in its local context versus the entire document
- `hierarchical_importance(document)`: Cascades importance scores through document hierarchy levels
- `citation_proximity(text)`: Weights text based on proximity to citations or evidence
- `rhetorical_position_score(text)`: Assigns scores based on position within rhetorical structures

## Metacognitive Orchestration

The framework doesn't just process text; it understands your goals and guides the writing process through the Metacognitive Orchestrator.

### Orchestrator Features

- **Goal Representation**: Define and track writing objectives
- **Context Awareness**: Maintain knowledge of document state and domain
- **Intelligent Intervention**: Provide suggestions based on goals and context
- **Progress Evaluation**: Assess alignment with intended outcomes

### Goal-Oriented Writing

```turbulance
// Setting up a writing goal
var goal = new Goal("Write a technical tutorial for beginners", 0.4)
goal.add_keywords(["tutorial", "beginner", "step-by-step", "explanation"])

// Track progress towards the goal
goal.update_progress(0.3)  // 30% complete
goal.is_complete()         // Returns false

// Evaluating alignment with goals
var alignment = orchestrator.evaluate_alignment(text)
if alignment < 0.3:
    suggest_improvements()
```

### Advanced Processing Architecture

The Metacognitive Orchestrator implements a streaming-based concurrent processing model with three nested layers:

1. **Context Layer**: Establishes the relevant frame for processing
2. **Reasoning Layer**: Handles logical processing and analytical computation
3. **Intuition Layer**: Focuses on pattern recognition and heuristic reasoning

This architecture enables:
- Processing to begin before complete input is available
- Continuous refinement of results as more information becomes available
- Enhanced ability to handle complex, open-ended tasks

## Propositions and Motions

The most unique feature of Turbulance is the Proposition and Motion system for hypothesis-driven text processing.

### Propositions and Motions

Propositions are containers for related ideas that can be tested and validated:

```turbulance
proposition TextQuality:
    motion Clarity("Text should be clear and unambiguous")
    motion Conciseness("Text should be concise without losing meaning") 
    motion Accuracy("Text should be factually correct")
    
    // Test the motions against actual content
    within document:
        given readability_score() > 70:
            support Clarity
        given word_count() / idea_count() < 20:
            support Conciseness
        given fact_check_score() > 0.9:
            support Accuracy
```

### Motions

Motions are the fundamental building blocks within propositions:

```turbulance
// Working with motions directly
motion claim = Motion("Text should be programmatically manipulable", "claim")
motion evidence = Motion("Word processors lack semantic awareness", "evidence")

// Apply motion-specific analysis
spelling_issues = claim.spelling()
capitalization_issues = evidence.capitalization()

// Check for cognitive biases
if claim.check_sunken_cost_fallacy().has_bias:
    print("Warning: Potential sunken cost fallacy detected")
```

## Specialized Data Structures

The framework introduces data structures specifically for metacognitive text processing and scientific data analysis:

1. **TextGraph**: Represents relationships between text components as a weighted directed graph
2. **ConceptChain**: Represents sequences of ideas with cause-effect relationships
3. **IdeaHierarchy**: Organizes ideas in a hierarchical tree structure
4. **ArgMap**: Creates argumentation maps with claims, evidence, and objections
5. **EvidenceNetwork**: Implements a Bayesian framework for managing conflicting scientific evidence

```turbulance
// Create an evidence network for scientific analysis
var network = new EvidenceNetwork()

// Add nodes representing different types of evidence
network.add_node("genomic_1", EvidenceNode.GenomicFeature {
    sequence: dna_sequence,
    position: "chr7:55249071-55249143",
    motion: Motion("EGFR exon sequence with activating mutation")
})

network.add_node("spectrum_1", EvidenceNode.Spectra {
    peaks: mass_spec_peaks,
    retention_time: 15.7,
    motion: Motion("Mass spectrum showing drug metabolite")
})

// Add edges representing relationships between evidence
network.add_edge("genomic_1", "spectrum_1", EdgeType.Supports { strength: 0.85 }, 0.2)

// Propagate beliefs through the network
network.propagate_beliefs()

// Analyze sensitivity to understand which evidence is most critical
sensitivity = network.sensitivity_analysis("genomic_1")  // Returns impact scores
```

## Extended Language Syntax

Turbulance includes unique language constructs for text processing:

### 1. Considering Statements
Process collections contextually:

```turbulance
considering these paragraphs where contains("important"):
    highlight(paragraph)

considering all motions in this:
    check_spelling(motion)
    check_capitalization(motion)
```

### 2. Cause Declarations
Model relationships between concepts:

```turbulance
cause BiasedReasoning = {
    primary: "emotional investment",
    effects: ["selective evidence consideration", "overconfidence in judgment"],
    confidence: 0.75
}
```

### 3. Allow Statements
Control permissions for text transformations:

```turbulance
allow fact_checking on Abstract
allow sentiment_analysis on MainPoint
allow coherence_check on Conclusion
```

### 4. Within Contexts
Define processing scopes:

```turbulance
within document:
    // Document-level processing
    ensure_consistent_terminology()
    
    considering all section in document:
        // Section-level processing
        ensure_logical_flow()
```

## Text Unit Hierarchy

Text units exist in a natural hierarchy that can be mathematically manipulated:

```
Document
├── Section
│   ├── Paragraph
│   │   ├── Sentence
│   │   │   ├── Phrase
│   │   │   │   ├── Word
│   │   │   │   │   └── Character
```

### Mathematical Operations on Text Units

```turbulance
// Division: Split into smaller units
var sentences = paragraph / sentence
var words = sentence / word

// Multiplication: Combine with intelligent connectors  
var paragraph = sentence1 * sentence2 * sentence3

// Addition: Concatenate while preserving type
var extended = original_text + " additional content"

// Subtraction: Remove content while maintaining structure
var cleaned = text - "unwanted phrase"
```

## Domain Extensions

### Genomic Extension

Process DNA sequences as text units:

```turbulance
import genomic

// Create a DNA sequence
var dna = genomic.NucleotideSequence.new("ATGCTAGCTAGCTAGCTA", "gene_123")

// Apply text operations to genetic data
var codons = dna / codon  // Split into codons
var gc_rich_regions = dna.filter("gc_content() > 0.6")

// Use propositions for genetic analysis
proposition GeneRegulation:
    motion Activation("Gene X activates Gene Y")
    motion Inhibition("Gene Z inhibits Gene X")
    
    within dna:
        given contains("TATAAA"):
            print("Found TATA box promoter")
            ensure_follows("Gene body")
```

### Chemistry Extension

Manipulate molecular structures:

```turbulance
import chemistry

// Create a molecule from SMILES
var caffeine = chemistry.Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

// Apply text operations to molecules
var fragments = caffeine / functional_group
var aromatic_parts = caffeine.filter("is_aromatic() == true")

// Chemical analysis with propositions
proposition DrugLikeness:
    motion LipiniskiCompliant("Molecule follows Lipinski's Rule of Five")
    
    within caffeine:
        given molecular_weight() < 500:
            support LipiniskiCompliant
        given hydrogen_donors() <= 5:
            support LipiniskiCompliant
```

### Mass Spectrometry Extension

Analyze spectral data:

```turbulance
import mass_spec

// Load and process spectrum
var spectrum = mass_spec.Spectrum.from_file("sample.mzML")
var peaks = spectrum / peak
var high_intensity = peaks.filter("intensity > 1000")

// Spectral analysis propositions
proposition CompoundIdentification:
    motion MolecularIon("Peak represents molecular ion")
    motion FragmentIon("Peak represents fragment ion")
    
    within spectrum:
        given peak_at_mz(molecular_weight):
            support MolecularIon
```

## Real-World Applications

### Academic Writing Assistant

```turbulance
proposition AcademicWriting:
    motion Formality("Writing maintains academic tone")
    motion Citations("All claims are properly cited") 
    motion Clarity("Ideas are clearly expressed")
    
funxn enhance_academic_paper(paper):
    considering all section in paper:
        given not has_clear_topic_sentence(section):
            add_topic_sentence(section)
        
        considering all paragraph in section:
            given readability_score(paragraph) < 60:
                improve_clarity(paragraph)
            given needs_citation(paragraph):
                suggest_citations(paragraph)
```

### Technical Documentation Generator

```turbulance
funxn generate_api_docs(code_base):
    var docs = Document.new("API Documentation")
    
    considering all module in code_base:
        considering all function in module:
            var func_doc = document_function(function)
            given complexity_score(func_doc) > 0.7:
                add_examples(func_doc)
                simplify_explanations(func_doc)
```

### Creative Writing Enhancement

```turbulance
proposition CreativeWriting:
    motion Engagement("Writing engages the reader")
    motion Flow("Ideas flow naturally between sentences")
    motion Vivid("Descriptions are vivid and compelling")

funxn enhance_creative_writing(story):
    considering all paragraph in story:
        given descriptive_density(paragraph) < 0.3:
            add_sensory_details(paragraph)
        given transition_quality(paragraph) < 0.6:
            improve_transitions(paragraph)
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/kwasa-kwasa.git
cd kwasa-kwasa

# Build the project
cargo build --release

# Install globally
cargo install --path .

# Verify installation
kwasa-kwasa --version
```

### Your First Program

Create `hello.turb`:

```turbulance
funxn main():
    var text = "Hello, Kwasa-Kwasa! This framework transforms how we work with text."
    
    // Split into sentences using division
    var sentences = text / sentence
    print("Number of sentences: {}", len(sentences))
    
    // Check readability
    given readability_score(text) > 70:
        print("Text is highly readable")
    
    // Use propositions for analysis
    proposition TextQuality:
        motion Clarity("Text should be clear")
        within text:
            given readability_score() > 60:
                support Clarity
                print("✓ Text is clear enough")

main()
```

Run it:

```bash
kwasa-kwasa run hello.turb
```

## Documentation

- **[Complete System Guide](complete_system_guide.md)** - Comprehensive explanation of the entire system
- **[Practical Getting Started](examples/getting_started_practical.md)** - Hands-on tutorial with examples
- **[Language Reference](language/turbulance-language.md)** - Complete Turbulance language documentation
- **[Domain Extensions](examples/domain-extensions.md)** - Genomics, chemistry, and scientific computing extensions
- **[Metacognitive Orchestration](metacognitive-orchestration.md)** - Advanced orchestration features

## Philosophy

Kwasa-Kwasa represents a fundamental shift in text processing philosophy. Traditional approaches treat text as strings to be manipulated. Kwasa-Kwasa treats text as a rich, semantic medium that can be mathematically manipulated while preserving meaning.

**Vision**: Create a computational environment where text becomes as manipulable as data structures in programming languages, while maintaining semantic richness and meaning.

---

*"There is no reason for your soul to be misunderstood"* - The core philosophy driving Kwasa-Kwasa's approach to making meaning computationally accessible. 