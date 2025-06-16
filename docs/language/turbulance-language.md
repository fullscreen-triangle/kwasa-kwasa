# Turbulance Language Overview

Turbulance is a domain-specific programming language designed for pattern analysis, evidence-based reasoning, and scientific computing. It combines imperative programming constructs with declarative pattern matching and evidence evaluation capabilities.

## Language Philosophy

Turbulance is built around the concept that **patterns are fundamental to understanding**. The language provides first-class support for:

- **Pattern Recognition**: Identifying recurring structures in data
- **Evidence-Based Reasoning**: Building conclusions from observable patterns
- **Hypothetical Thinking**: Testing propositions against evidence
- **Cross-Domain Analysis**: Applying patterns across different scientific domains

## Core Language Features

### 1. Variables and Data Types

Turbulance uses dynamic typing with strong type inference:

```turbulance
// Basic variable declaration
item temperature = 23.5
item molecule_name = "caffeine"
item is_valid = true

// Typed declarations for clarity
item data: TimeSeries = load_series("temperature.csv")
item patterns: PatternSet = {}
```

### 2. Functions

Functions in Turbulance use the `funxn` keyword (a play on "function"):

```turbulance
// Basic function
funxn calculate_average(numbers):
    item sum = 0
    item count = 0
    
    for each number in numbers:
        sum = sum + number
        count = count + 1
    
    return sum / count

// Function with type hints
funxn analyze_sequence(sequence: DNASequence) -> AnalysisResult:
    // Function body
```

### 3. Control Flow

#### Conditional Statements

Turbulance uses `given` instead of `if` to emphasize conditional reasoning:

```turbulance
given temperature > 30:
    print("Hot weather")
given temperature < 10:
    print("Cold weather")
given otherwise:
    print("Moderate weather")
```

#### Loops

Multiple loop constructs for different scenarios:

```turbulance
// For-each loop
for each item in collection:
    process(item)

// While loop
while condition_holds:
    update_condition()

// Within loop (pattern-based iteration)
within dataset as records:
    given record.matches(pattern):
        collect(record)
```

### 4. Pattern Matching

Pattern matching is a core feature:

```turbulance
// Simple pattern matching
within text_data:
    given matches("scientific.*paper"):
        classify_as("academic")
    given matches("news.*article"):
        classify_as("journalism")

// Complex pattern with variables
within genetic_sequence:
    given matches("ATG(?<codon>...)TAG"):
        item protein_start = codon
        analyze_protein(protein_start)
```

### 5. Data Structures

#### Lists and Arrays
```turbulance
item numbers = [1, 2, 3, 4, 5]
item mixed_data = ["text", 42, true, 3.14]

// Dynamic arrays
item temperatures = []
temperatures.append(23.5)
temperatures.extend([24.1, 22.8, 25.3])
```

#### Dictionaries/Maps
```turbulance
item person = {
    "name": "Dr. Smith",
    "age": 45,
    "department": "Physics"
}

// Accessing values
item name = person["name"]
item age = person.age  // Dot notation also works
```

#### Sets
```turbulance
item unique_values = {1, 2, 3, 4, 5}
item gene_names = {"BRCA1", "TP53", "EGFR"}

// Set operations
item intersection = set1 & set2
item union = set1 | set2
```

## Domain-Specific Extensions

### Scientific Computing

```turbulance
// Mathematical operations
item matrix = [[1, 2], [3, 4]]
item eigenvalues = calculate_eigenvalues(matrix)

// Statistical functions
item mean_value = mean(dataset)
item correlation = pearson_correlation(x_values, y_values)
```

### Text Analysis

```turbulance
// Natural language processing
item sentiment = analyze_sentiment(text)
item topics = extract_topics(documents, num_topics=5)
item entities = recognize_entities(text)
```

### Bioinformatics

```turbulance
// Genomic analysis
item gc_content = calculate_gc_content(dna_sequence)
item alignment = align_sequences(seq1, seq2, algorithm="needleman-wunsch")
item variants = call_variants(reference, sample)
```

## Advanced Features

### 1. Propositions and Motions

Core constructs for hypothesis testing:

```turbulance
proposition DrugEfficacy:
    motion ReducesSymptoms("Drug reduces symptom severity")
    motion MinimalSideEffects("Drug has minimal side effects")
    
    within clinical_data:
        given symptom_reduction > 0.5:
            support ReducesSymptoms
        given side_effect_rate < 0.1:
            support MinimalSideEffects
```

### 2. Evidence Collection

Specialized structures for gathering and validating evidence:

```turbulance
evidence ExperimentalData:
    sources:
        - patient_records: MedicalDatabase
        - lab_results: LabDatabase
    
    validation:
        - cross_reference_patients()
        - verify_lab_protocols()
        - check_data_integrity()
```

### 3. Metacognitive Operations

Self-reflective analysis capabilities:

```turbulance
metacognitive AnalysisReview:
    track:
        - reasoning_steps
        - confidence_levels
        - potential_biases
    
    evaluate:
        - logical_consistency()
        - evidence_quality()
        - conclusion_strength()
```

## Error Handling

Turbulance provides comprehensive error handling:

```turbulance
try:
    item result = risky_operation(data)
catch PatternNotFound as e:
    print("Pattern matching failed: {}".format(e.message))
    use_fallback_method()
catch DataCorruption as e:
    print("Data integrity issue: {}".format(e.message))
    request_data_reload()
finally:
    cleanup_resources()
```

## Modules and Imports

Organize code using modules:

```turbulance
// Import entire module
import statistics

// Import specific functions
from genomics import analyze_sequence, calculate_gc_content

// Import with alias
import chemistry as chem

// Conditional imports based on availability
try:
    import experimental_features
    item use_experimental = true
catch ImportError:
    item use_experimental = false
```

## Concurrency and Parallelism

Built-in support for parallel processing:

```turbulance
// Parallel processing
parallel analyze_samples(samples):
    workers: 4
    load_balancing: dynamic

// Async operations
async funxn fetch_data(url):
    item response = await http_get(url)
    return parse_response(response)

// Concurrent evidence gathering
concurrent gather_evidence():
    item text_evidence = analyze_text_async(documents)
    item numeric_evidence = analyze_numbers_async(datasets)
    item pattern_evidence = find_patterns_async(sequences)
    
    await all([text_evidence, numeric_evidence, pattern_evidence])
```

## Type System

Turbulance features a gradual type system:

```turbulance
// Dynamic typing (default)
item data = load_file("data.csv")

// Optional type annotations
item temperature: Float = 23.5
item genes: List[String] = ["BRCA1", "TP53"]

// Generic types
funxn process_items<T>(items: List[T]) -> List[T]:
    // Process items of any type T

// Union types
item identifier: String | Integer = "sample_001"

// Optional types
item optional_value: String? = might_return_none()
```

## Memory Management

Automatic memory management with manual control when needed:

```turbulance
// Automatic cleanup
item large_dataset = load_massive_file("huge_data.csv")
// Automatically freed when out of scope

// Manual memory management for large operations
with managed_memory(limit="2GB"):
    item results = process_big_data(dataset)
    
// Streaming for memory efficiency
stream process_large_file(filename):
    chunk_size: 1000
    yield_frequency: 100
```

## Development Tools

### Debugging

```turbulance
// Debug assertions
assert temperature > 0, "Temperature cannot be negative"
assert len(sequence) > 0, "Sequence cannot be empty"

// Debug printing with context
debug print("Processing sample: {}", sample_id)
debug print("Intermediate result: {}", calculation_step)
```

### Testing

```turbulance
// Unit tests
test "calculate_gc_content works correctly":
    item sequence = "ATCGATCG"
    item expected = 0.5
    item actual = calculate_gc_content(sequence)
    assert_equals(expected, actual)

// Property-based testing
property "sequence length is preserved":
    for any sequence in generate_dna_sequences():
        item processed = process_sequence(sequence)
        assert len(processed) == len(sequence)
```

## Language Syntax Summary

### Keywords
- `var` - variable declaration
- `funxn` - function definition
- `given` - conditional statement
- `within` - pattern-based iteration
- `proposition` - hypothesis definition
- `motion` - sub-hypothesis
- `evidence` - evidence collection
- `metacognitive` - self-reflective analysis
- `try`/`catch`/`finally` - error handling
- `import`/`from` - module imports
- `parallel`/`async`/`await` - concurrency

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `**`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`
- Pattern: `matches`, `contains`, `in`
- Assignment: `=`, `+=`, `-=`, `*=`, `/=`

### Comments
```turbulance
// Single line comment

/*
Multi-line
comment
*/

/// Documentation comment
funxn documented_function():
    /// This function is documented
```

## Next Steps

- Explore [Special Language Features](language/special_features.md) for advanced constructs
- See [Examples](examples/index.md) for practical applications
- Check [Domain Extensions](domain-extensions.md) for specialized functionality 