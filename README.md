<h1 align="center">Kwasa Kwasa</h1>
<p align="center"><em>There is no reason for your soul to be misunderstood</em></p>

<p align="center">
  <img src="horizontal_film.gif" alt="Logo">
</p>

## A Metacognitive Text Processing Framework with Turbulance Syntax

Kwasa-Kwasa is a specialized framework designed for writers who need programmatic control over complex text operations with semantic awareness. It combines a powerful text processing language ("Turbulance") with an intelligent orchestration system to create a comprehensive solution for serious writing projects.

---

## Table of Contents

- [Vision](#vision)
- [Core Concepts](#core-concepts)
  - [Turbulance Language](#turbulance-a-language-for-text)
  - [Metacognitive Orchestration](#metacognitive-orchestration)
  - [Proposition and Motion System](#proposition-and-motion-system)
  - [Text Unit System](#text-unit-system)
- [System Architecture](#system-architecture)
- [Installation and Usage](#using-kwasa-kwasa)
- [Real-World Use Cases](#real-world-use-cases)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

---

## Vision

Kwasa-Kwasa addresses fundamental limitations in how we interact with text. While code has evolved sophisticated tooling for manipulation, refactoring, and analysis, text remains constrained by simplistic word processors or overly complicated publishing workflows.

This project rejects the notion that text should be treated merely as strings or formatting challenges. Instead, it recognizes text as semantically rich units that can be programmatically manipulated while preserving their meaning and context.

> "The way we interact with text hasn't fundamentally changed in decades. Kwasa-Kwasa is not just another text editor or document processor; it's a new paradigm for how writers can leverage computation to enhance their craft."

## Core Concepts

### Turbulance: A Language for Text

Turbulance is a domain-specific language designed exclusively for text operations. It provides a rich, expressive syntax for text manipulation with semantic awareness.

#### Key Language Features

- **Boundaries and Text Units**: Define and operate on specific text structures
- **Contextual Transformations**: Apply transformations based on semantic context
- **Knowledge Integration**: Connect with external research sources
- **State Management**: Maintain context across transformations
- **Semantic Operations**: Operate on text while preserving meaning

#### Syntax Example

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

#### Language Structure

Turbulance's lexical structure includes:
- **Keywords**: `funxn`, `within`, `given`, `project`, `ensure`, `return`, etc.
- **Operators**: `/` (division), `*` (multiplication), `+` (addition), `-` (subtraction)
- **Control Structures**: Block expressions, conditional execution, iterations
- **Function System**: Declarations, parameters, closures, return values
- **Special Constructs**: `motion`, `proposition`, `cause`, `considering`, `allow`

#### Standard Library

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

### Metacognitive Orchestration

The framework doesn't just process text; it understands your goals and guides the writing process through the Metacognitive Orchestrator.

#### Orchestrator Features

- **Goal Representation**: Define and track writing objectives
- **Context Awareness**: Maintain knowledge of document state and domain
- **Intelligent Intervention**: Provide suggestions based on goals and context
- **Progress Evaluation**: Assess alignment with intended outcomes

#### Goal-Oriented Writing

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

#### Advanced Processing Architecture

The Metacognitive Orchestrator implements a streaming-based concurrent processing model with three nested layers:

1. **Context Layer**: Establishes the relevant frame for processing
2. **Reasoning Layer**: Handles logical processing and analytical computation
3. **Intuition Layer**: Focuses on pattern recognition and heuristic reasoning

This architecture enables:
- Processing to begin before complete input is available
- Continuous refinement of results as more information becomes available
- Enhanced ability to handle complex, open-ended tasks

### Proposition and Motion System

Kwasa-Kwasa introduces a paradigm shift from traditional object-oriented programming by replacing classes with **Propositions** that contain **Motions**—pieces of ideas with semantic meaning.

#### Propositions

A Proposition serves as a container for related semantic units:

```turbulance
// Define a proposition with motions
proposition TextAnalysis:
    // Define motions within the proposition
    motion Introduction("The text analysis begins with understanding the context.")
    motion MainPoint("Proper analysis requires both syntactic and semantic understanding.")
    motion Conclusion("By analyzing text with these methods, we gain deeper insights.")
    
    // Add metadata to the proposition
    with_metadata("domain", "linguistics")
    with_metadata("confidence", "0.95")
    
    // Process all motions in this proposition
    considering all motions in this:
        check_spelling(motion)
        check_capitalization(motion)
        
    // Allow specific operations on specific motions
    allow fact_checking on Introduction
    allow coherence_check on Conclusion
```

#### Motions

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

#### Specialized Data Structures

The framework introduces data structures specifically for metacognitive text processing:

1. **TextGraph**: Represents relationships between text components as a weighted directed graph
2. **ConceptChain**: Represents sequences of ideas with cause-effect relationships
3. **IdeaHierarchy**: Organizes ideas in a hierarchical tree structure
4. **ArgMap**: Creates argumentation maps with claims, evidence, and objections

```turbulance
// Create an argument map
var argmap = new ArgMap()
argmap.add_claim("main", Motion("Text analysis should be taught in schools", "claim"))

// Add supporting evidence
argmap.add_evidence(
    "main", 
    "evidence1", 
    Motion("Improves critical thinking skills", "evidence"),
    0.8  // Strong evidence
)

// Evaluate claim strength based on evidence and objections
strength = argmap.evaluate_claim("main")  // Returns a value between 0 and 1
```

#### Extended Language Syntax

Turbulance includes unique language constructs for text processing:

1. **Considering Statements**: Process collections contextually
   ```turbulance
   considering these paragraphs where contains("important"):
       highlight(paragraph)
   ```

2. **Cause Declarations**: Model relationships between concepts
   ```turbulance
   cause BiasedReasoning = {
       primary: "emotional investment",
       effects: ["selective evidence consideration", "overconfidence in judgment"]
   }
   ```

3. **Allow Statements**: Control permissions for text transformations
   ```turbulance
   allow fact_checking on Abstract
   ```

### Text Unit System

Kwasa-Kwasa's text unit system provides a way to work with text at varying levels of granularity.

#### Unit Hierarchy

The system recognizes multiple levels of text units:
- **Document**: The entire text
- **Section**: Major divisions with headings
- **Paragraph**: Standard paragraph breaks
- **Sentence**: Complete sentences with terminal punctuation
- **Clause**: Grammatical clauses within sentences
- **Phrase**: Meaningful word groups
- **Word**: Individual words
- **Character**: Individual characters

#### Boundary Detection

Text units are identified through:
- Structural markers (headings, paragraph breaks)
- Syntactic analysis (sentence boundaries)
- Semantic coherence (topic-based segmentation)
- User-defined markers (custom delimiters)

#### Working with Text Units

```turbulance
// Process specific unit types
within text as paragraphs:
    // Operate on each paragraph
    print("Found paragraph: " + paragraph)
    
    within paragraph as sentences:
        // Operate on each sentence within this paragraph
        ensure sentences.length < 50  // Ensure sentences aren't too long
```

#### Mathematical Operations on Text

Turbulance enables mathematical-like operations on text:

1. **Division (/)**: Segments text into units
   ```turbulance
   var paragraphs = document / "paragraph"
   var topics = document / "topic"
   ```

2. **Multiplication (*)**: Combines with transitions
   ```turbulance
   var section = historical_context * current_regulations
   ```

3. **Addition (+)**: Combines with connectives
   ```turbulance
   var comprehensive_view = pilot_testimony + expert_analysis
   ```

4. **Subtraction (-)**: Removes elements
   ```turbulance
   var accessible_explanation = technical_explanation - jargon
   ```

#### Transformation Pipelines

Pipelines allow chaining operations for text transformations:

```turbulance
// Basic pipeline with the |> operator
section("Introduction") |>
    analyze_sentiment() |>
    extract_key_themes() |>
    enhance_clarity(level="moderate") |>
    ensure_consistency(with="conclusion")
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      KWASA-KWASA FRAMEWORK                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐      ┌──────────────────────────┐     │
│  │                 │      │                          │     │
│  │   Turbulance    │◄────►│  Metacognitive           │     │
│  │   Language      │      │  Orchestrator            │     │
│  │   Engine        │      │                          │     │
│  │                 │      │                          │     │
│  └────────┬────────┘      └──────────┬───────────────┘     │
│           │                          │                      │
│           ▼                          ▼                      │
│  ┌─────────────────┐      ┌──────────────────────────┐     │
│  │                 │      │                          │     │
│  │   Text Unit     │      │  Knowledge               │     │
│  │   Processor     │◄────►│  Integration             │     │
│  │                 │      │  Engine                  │     │
│  │                 │      │                          │     │
│  └─────────────────┘      └──────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Turbulance Language Engine**
   - Parser and interpreter for Turbulance syntax
   - Compiler to optimize text operations
   - Runtime environment for executing text transformations
   - Standard library of common text functions

2. **Text Unit Processor**
   - Boundary detection for natural text units
   - Hierarchical representation of document structure
   - Transformation pipeline for applying operations
   - State management between operations

3. **Metacognitive Orchestrator**
   - Document goal representation and tracking
   - Contextual understanding of content domains
   - Intervention decision system
   - Progress evaluation against goals

4. **Knowledge Integration Engine**
   - Research interface for contextual information retrieval
   - Knowledge database for storing domain information
   - Citation and reference management
   - Fact verification system

## Using Kwasa-Kwasa

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kwasa-kwasa.git
cd kwasa-kwasa

# Build the project
cargo build --release

# Install globally
cargo install --path .
```

### CLI Commands

Kwasa-Kwasa provides a comprehensive command-line interface:

#### Running Scripts

```bash
kwasa-kwasa run <script_path>
```

This executes a standalone Turbulance script, parsing and interpreting it.

#### Validating Scripts

```bash
kwasa-kwasa validate <script_path>
```

Checks a script for syntax errors without executing it.

#### Processing Documents

```bash
kwasa-kwasa process <document_path> [--interactive]
```

Processes a document with embedded Turbulance commands, applying transformations directly to the content.

#### Interactive REPL

```bash
kwasa-kwasa repl
```

Starts an interactive Turbulance shell for experimentation.

### WebAssembly Integration

Kwasa-Kwasa can run directly in the browser through WebAssembly:

```javascript
// Execute Turbulance code
const result = kwasa.execute_code(`
  var greeting = "Hello, world!"
  print(greeting)
`);

// Process text with the metacognitive orchestrator
const processed = kwasa.process_text(text);

// Divide text into semantic units
const sentences = kwasa.divide_text(text, "sentence");
```

## Technology Stack

Kwasa-Kwasa is built with modern, high-performance technologies:

- **Rust** - For memory safety, performance, and concurrency
  - Logos for lexical analysis
  - Chumsky for parsing
  - Memory safety through ownership and borrowing system

- **SQLite** - Embedded database for knowledge storage
  - Zero-configuration, serverless database
  - Efficient for document metadata and knowledge indexing

- **WebAssembly** - For browser integrations
  - Compiled Rust code deployable to browser environments
  - Near-native performance in web applications

- **gRPC** - For efficient service communication
  - High-performance remote procedure calls
  - Multi-language support for system extensions

## Real-World Use Cases

### Academic Writing

```turbulance
project thesis(title="Quantum Computing Applications in Cryptography"):
    require sections = ["introduction", "literature_review", "methodology", 
                       "results", "discussion", "conclusion"]
    
    for each section:
        ensure readability_score > 40
        ensure citation_density appropriate_for(section)
        
    within section("methodology"):
        ensure contains(experiment_details)
        ensure formula_accuracy()
```

### Technical Documentation

```turbulance
project api_docs(api="GraphQL Schema"):
    for each endpoint:
        ensure has(description, parameters, return_values, examples)
        ensure code_samples are_valid()
        
    within all_sections:
        detect jargon()
        if audience != "expert":
            provide_explanation()
```

### Research-Intensive Writing

```turbulance
project research_article(topic="Aerodynamics"):
    setup knowledge_base from ["aviation", "aerodynamics", "physics"]
    
    within section("introduction"):
        research context(topic)
        ensure historical_context(topic)
    
    before completion:
        verify technical_accuracy()
        ensure all_claims are_supported()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
