<h1 align="center">Kwasa Kwasa</h1>
<p align="center"><em> There is no reason for your soul to be misunderstood</em></p>


<p align="center">
  <img src="horizontal_film.gif" alt="Logo">
</p>

## A Metacognitive Text Processing Framework with Turbulance Syntax

Kwasa-Kwasa is a specialized framework designed for writers who need programmatic control over complex text operations with semantic awareness. It combines a powerful text processing language ("Turbulance") with an intelligent orchestration system to create a comprehensive solution for serious writing projects.

---

## Vision

Kwasa-Kwasa addresses fundamental limitations in how we interact with text. While code has evolved sophisticated tooling for manipulation, refactoring, and analysis, text remains constrained by simplistic word processors or overly complicated publishing workflows.

This project rejects the notion that text should be treated merely as strings or formatting challenges. Instead, it recognizes text as semantically rich units that can be programmatically manipulated while preserving their meaning and context.

## Core Concepts

### Turbulance: A Language for Text

Turbulance is a domain-specific language designed exclusively for text operations. It allows writers to:

- Define boundaries around "text units" that can be operated on programmatically
- Create functions that transform text in contextually-aware ways
- Build pipelines for processing, analyzing, and enhancing written content
- Interface with external knowledge sources and NLP services
- Maintain state between transformations while preserving semantic integrity

Syntax example:

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

### Metacognitive Orchestration

The framework doesn't just process text; it understands your goals and guides the writing process:

- **Context-Aware Processing**: Operations understand the semantic context and subject domain
- **Goal Orientation**: The system guides writing toward intended outcomes
- **Intelligent Intervention**: Suggestions happen only when relevant to goals
- **Knowledge Integration**: Connects writing with research sources contextually
- **State Preservation**: Maintains awareness of the document's evolving state

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

Kwasa-Kwasa provides a comprehensive command-line interface powered by the Clap framework:

#### Running Scripts

Execute a standalone Turbulance script:

```bash
kwasa-kwasa run <script_path>
```

This command:
- Tokenizes the script using the Logos-based lexer
- Parses it with the recursive descent parser
- Constructs the AST (Abstract Syntax Tree)
- Executes the script using the interpreter
- Outputs the results to stdout

Example:
```bash
kwasa-kwasa run examples/text_analysis.turb
```

#### Validating Scripts

Check a Turbulance script for syntax errors without executing it:

```bash
kwasa-kwasa validate <script_path>
```

This performs lexical and syntax validation, reporting:
- Lexical errors (invalid tokens)
- Syntax errors (malformed expressions)
- Semantic errors (type mismatches, undefined variables)

Example:
```bash
kwasa-kwasa validate my_script.turb
```

#### Processing Documents

Process a document with embedded Turbulance commands:

```bash
kwasa-kwasa process <document_path> [--interactive]
```

This command:
- Extracts Turbulance code blocks from the document
- Executes them in sequence, maintaining state between blocks
- Applies transformations directly to the document content
- Generates output with the processed text

The `--interactive` flag enables prompts for user input during processing.

Example:
```bash
kwasa-kwasa process article.md --interactive
```

#### Interactive REPL

Start an interactive Turbulance shell for experimentation:

```bash
kwasa-kwasa repl
```

The REPL (Read-Eval-Print Loop) provides:
- Immediate feedback for Turbulance expressions
- Command history navigation
- Auto-completion of keywords and variables
- Contextual help for standard library functions

## Turbulance Language

The Turbulance language is the core of Kwasa-Kwasa. It provides a rich, expressive syntax for text manipulation with semantic awareness.

### Lexical Structure

Turbulance has a well-defined token structure:
- **Keywords**: `funxn`, `within`, `given`, `project`, `ensure`, `return`, etc.
- **Operators**: `/` (division), `*` (multiplication), `+` (addition), `-` (subtraction), etc.
- **Literals**: String literals (`"text"`), numbers, booleans (`true`/`false`)
- **Identifiers**: Variable and function names with Unicode support
- **Comments**: Single-line (`// comment`) and multi-line (`/* comment */`)

### Syntax Highlights

```turbulance
// Variable declaration
var greeting = "Hello, world!"

// Function declaration with parameters and default values
funxn greet(name, formal=false):
    given formal:
        return "Greetings, " + name + "."
    return "Hello, " + name + "!"

// Function call
greet("Writer")

// Control flow with expressions
var mood = "happy"
given mood == "happy":
    print("Glad to hear that!")
else:
    print("Hope things improve!")

// Block expressions
var result = {
    var x = compute_value()
    x * 2
}
```

### Parser Design

The Turbulance parser is implemented as a recursive descent parser with robust error handling:

- **Top-down parsing**: Builds the AST directly from the token stream
- **Operator precedence**: Correctly handles complex expressions with appropriate precedence rules
- **Error recovery**: Provides detailed error messages with location information
- **Lookahead**: Uses limited lookahead for efficient parsing decisions

```turbulance
// This expression correctly follows operator precedence
var result = 2 + 3 * 4       // Evaluates to 14, not 20
var complex = (x + y) * (z / w) - factor  // Grouping with parentheses
```

### Abstract Syntax Tree (AST)

The AST representation provides a structured view of Turbulance programs:

- **Node hierarchy**: Specialized node types for different language constructs
- **Visitor pattern**: Allows for operations like interpretation, static analysis, and optimization
- **Source mapping**: Preserves source location information for error reporting
- **Semantic analysis**: Type checking and variable resolution

AST node categories include:

- **Declarations**: Functions, projects, variables
- **Statements**: Assignment, return, control flow
- **Expressions**: Binary, unary, literals, function calls
- **Text operations**: Division, multiplication, addition, subtraction
- **Special constructs**: Within blocks, given conditions, ensure assertions

### Interpreter Implementation

The Turbulance interpreter executes AST nodes with the following features:

- **Evaluation strategy**: Eager evaluation with short-circuiting for logical operators
- **Scoping**: Lexical scoping with closures
- **Memory model**: Value semantics with reference counting for efficient text handling
- **Error handling**: Runtime errors with stack traces and location information
- **Standard library**: Built-in functions accessible to all Turbulance code

```turbulance
// Example demonstrating interpreter features
funxn process_text(text):
    // Lexical scope with closure
    var transformations = []
    
    funxn add_transformation(name, func):
        transformations.push({
            "name": name,
            "apply": func
        })
    
    // Add some transformations
    add_transformation("simplify", x => simplify_text(x))
    add_transformation("enhance", x => enhance_readability(x))
    
    // Apply all transformations
    var result = text
    for each t in transformations:
        result = t.apply(result)
        
    return result
```

### Standard Library

The Turbulance standard library provides a rich set of built-in functions for text analysis and manipulation:

#### Text Analysis Functions

```turbulance
// Readability assessment
readability_score(text)              // Returns Flesch-Kincaid score (0-100)
grade_level(text)                    // Estimated grade level needed to understand
complexity_analysis(text)            // Returns detailed complexity metrics

// Content analysis
extract_keywords(text, count=10)     // Extracts most significant keywords
extract_entities(text, types=["person", "organization", "location"])
extract_topics(text, depth=2)        // Hierarchical topic extraction
sentiment_analysis(text)             // Returns polarity and subjectivity scores
contains(text, pattern)              // Searches for pattern (string or regex)
```

#### Text Transformation Functions

```turbulance
// Text simplification
simplify_sentences(text, level="moderate")  // Simplifies complex sentences
replace_jargon(text, domain="general")     // Replaces specialized terms
formalize(text)                            // Increases formality
casualize(text)                            // Makes text more conversational

// Text enhancement
enhance_clarity(text)                      // Improves clarity
expand_acronyms(text)                      // Expands acronyms with definitions
add_transitions(text)                      // Adds appropriate transition phrases
restructure(text, pattern="problem-solution") // Restructures following pattern
```

#### Knowledge Integration Functions

```turbulance
// Research assistance
research_context(topic, depth="medium")    // Retrieves contextual information
find_citations(claim, max_results=5)       // Finds supporting citations
fact_check(statement)                      // Verifies factual claims
ensure_explanation_follows(term)           // Ensures term is explained
```

#### Utility Functions

```turbulance
// General utilities
print(value)                               // Outputs to console
len(collection)                            // Returns collection length
typeof(value)                              // Returns type information
join(collection, separator=" ")            // Joins collection elements
format(template, ...args)                  // String formatting
```

## Text Unit System

Kwasa-Kwasa's text unit system provides a powerful way to work with text at varying levels of granularity:

### Text Unit Types

The system recognizes multiple levels of text units:

- **Document**: The entire text
- **Section**: Major divisions with headings
- **Paragraph**: Standard paragraph breaks
- **Sentence**: Complete sentences with terminal punctuation
- **Clause**: Grammatical clauses within sentences
- **Phrase**: Meaningful word groups
- **Word**: Individual words
- **Character**: Individual characters

### Boundary Detection

Text units are identified through a combination of approaches:

- **Structural markers**: Headings, paragraph breaks, list items
- **Syntactic analysis**: Sentence and clause boundaries
- **Semantic coherence**: Topic-based segmentation
- **User-defined markers**: Custom delimiters specified in text

### Working with Boundaries

The `within` block allows operations on specific text units:

```turbulance
// Process specific unit types
within paragraph:
    given contains("technical term"):
        highlight()
        add_definition()
    given readability_score() < 70:
        simplify()

// Nested units
within section("Methods"):
    within paragraph:
        ensure_consistent_terminology()
    within sentence.first():
        ensure_topic_sentence()

// Custom boundary definitions
define boundary "concept" as segment(by="topic", min_length=100)
within concept:
    ensure_coherence()
```

### Unit Selection and Filtering

Text units can be selected and filtered using various criteria:

```turbulance
// Selection by position
paragraph.first()
paragraph.last()
paragraph[3]  // zero-based indexing
paragraph.slice(2, 5)

// Selection by content
paragraph.containing("keyword")
paragraph.matching(/pattern/)

// Selection by property
paragraph.where(length > 100)
paragraph.where(readability_score() < 60)

// Combining selections
paragraph.where(contains("technical") && readability_score() < 70)
```

### Mathematical Operations on Text Units

Turbulance enables powerful mathematical-like operations on text:

#### Division (/)

The division operator segments text into units:

```turbulance
// Divide text into paragraphs
var paragraphs = document / "paragraph"

// Divide by semantic units
var topics = document / "topic"
var concepts = document / "concept"

// Custom division criteria
var segments = document / {min_length: 200, coherence: "high"}

// Chained division
var sentences = (document / "paragraph")[0] / "sentence"
```

#### Multiplication (*)

Multiplication combines text units with appropriate transitions:

```turbulance
// Combine with context-appropriate transitions
var section = historical_context * current_regulations

// Multiply with specific joining strategy
var narrative = anecdote * explanation * conclusion * "causal"

// Controlled multiplication with options
var detailed_view = summary * details * {
    transition_style: "elaborative",
    ensure_coherence: true
}
```

#### Addition (+)

Addition combines information with connectives:

```turbulance
// Simple combination
var comprehensive_view = pilot_testimony + expert_analysis

// Addition with specific connective style
var contrasting_views = view_a + view_b + "contrastive"

// Complex addition with options
var balanced_perspective = for_arguments + against_arguments + {
    balance: "equal",
    structure: "point-counterpoint"
}
```

#### Subtraction (-)

Subtraction removes elements while preserving coherence:

```turbulance
// Remove jargon
var accessible_explanation = technical_explanation - jargon

// Remove specific content
var summary = full_text - examples - digressions

// Subtraction with options
var concise_version = original - {
    target: "redundancies",
    preserve: "key_points",
    max_reduction: 0.3  // max 30% reduction
}
```

### Transformation Pipelines

Pipelines allow chaining operations for sophisticated text transformations:

```turbulance
// Basic pipeline with the |> operator
section("Introduction") |>
    analyze_sentiment() |>
    extract_key_themes() |>
    enhance_clarity(level="moderate") |>
    ensure_consistency(with="conclusion")

// Pipeline with conditional branches
document |>
    divide("section") |>
    for_each() |> {
        given type == "introduction":
            ensure_hook() |> ensure_context()
        given type == "methods":
            ensure_clarity() |> ensure_completeness()
        given type == "results":
            visualize_data() |> highlight_key_findings()
        given type == "discussion":
            connect_to_literature() |> suggest_implications()
    } |>
    reassemble()

// Pipeline with error handling
document |>
    try {
        process_complex_transformations() |>
        validate_results()
    } catch (e) {
        log_error(e) |>
        fallback_to_simple_processing()
    } |>
    finalize()
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

### Components

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

## Technology Stack

Kwasa-Kwasa is built with modern, high-performance technologies:

- **Rust** - For memory safety, performance, and concurrency
  - Logos for lexical analysis
  - Chumsky for parsing
  - Memory safety through ownership and borrowing system
  - Immutable by default with controlled mutability

- **SQLite** - Embedded database for knowledge storage
  - Zero-configuration, serverless database
  - Cross-platform compatibility
  - Efficient for document metadata and knowledge indexing

- **WebAssembly** - For potential editor integrations
  - Compiled Rust code deployable to browser environments
  - Near-native performance in web applications
  - Direct integration with text editors and IDEs

- **gRPC** - For efficient service communication
  - High-performance remote procedure calls
  - Strongly typed interface definitions with Protocol Buffers
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
    
    within all_sections:
        given contains(terminology, "quantum"):
            ensure defined_or_cited()
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
project research_article(topic="Pugachev Cobra Aerodynamics"):
    setup knowledge_base from ["aviation", "aerodynamics", "fighter_aircraft"]
    
    within section("introduction"):
        research context(topic)
        ensure historical_context(topic)
    
    during writing:
        given mention("Pugachev"):
            provide relevant_background()
            suggest relevant_citations()
        
    before completion:
        verify technical_accuracy()
        ensure all_claims are_supported()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> "The way we interact with text hasn't fundamentally changed in decades. Kwasa-Kwasa is not just another text editor or document processor; it's a new paradigm for how writers can leverage computation to enhance their craft."

---

## Metacognitive Orchestrator

The Kwasa-Kwasa framework includes a powerful Metacognitive Orchestrator that manages the writing process:

### Goal Representation

The Orchestrator maintains a clear representation of writing goals:

```turbulance
// Setting up a writing goal
var goal = new Goal("Write a technical tutorial for beginners", 0.4)
goal.add_keywords(["tutorial", "beginner", "step-by-step", "explanation"])

// Track progress towards the goal
goal.update_progress(0.3)  // 30% complete
goal.is_complete()         // Returns false
```

### Context Awareness

The Orchestrator maintains awareness of the current writing context:

```turbulance
// The context tracks important elements of the writing
var context = orchestrator.context()

// Access contextual information
context.get_keywords()             // Active keywords
context.get_recent_keywords()      // Recent keywords by frequency
context.get_research_terms()       // Research topics
context.most_common_transition("algorithm")  // Common topic transitions
```

### Intervention System

Interventions automatically improve text based on goals and context:

```turbulance
// Register interventions
orchestrator.register_intervention(new ReadabilityIntervention())
orchestrator.register_intervention(new CoherenceIntervention())
orchestrator.register_intervention(new ResearchIntervention())

// Process text with interventions
var improved = orchestrator.process_text(original_text)
```

### Progress Evaluation

The Orchestrator evaluates writing progress against goals:

```turbulance
// Evaluating alignment with goals
var alignment = orchestrator.evaluate_alignment(text)
if alignment < 0.3:
    suggest_improvements()
    
// Getting progress metrics
var progress = goal.progress()
var estimated_completion = orchestrator.estimate_completion()
```

## WebAssembly Integration

Kwasa-Kwasa can run directly in the browser through WebAssembly integration:

### Setting up the WASM Module

```html
<script type="module">
  // Import the Kwasa-Kwasa WebAssembly module
  import init, { init_kwasa_wasm, KwasaConfig } from './pkg/kwasa_kwasa.js';

  async function start() {
    // Initialize the WASM module
    await init();
    
    // Configure with your preferences
    const config = new KwasaConfig();
    config.set_goal("Technical writing");
    config.set_debug(true);
    
    // Create the Kwasa-Kwasa instance
    const kwasa = init_kwasa_wasm();
    
    // Now you can use all the Kwasa-Kwasa functionality
    // directly in your web application
  }
  
  start();
</script>
```

### Available API

The WebAssembly bindings expose the core Kwasa-Kwasa functionality:

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

// Set writing goals
kwasa.set_goal("Create technical documentation", 0.3);

// Evaluate goal alignment
const alignment = kwasa.evaluate_alignment(text);

// Perform research
const results = kwasa.research("programming languages");
```

### Example Web Application

The framework includes a demo web application showing how to use Kwasa-Kwasa in a browser environment. To run the demo:

```bash
# Build the WebAssembly module
wasm-pack build --target web

# Serve the demo application
cd examples
python -m http.server
```

Then navigate to http://localhost:8000/wasm_demo.html to see the demo in action.
