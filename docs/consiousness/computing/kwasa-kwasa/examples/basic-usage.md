# Basic Usage Guide

This guide covers the fundamental concepts and common usage patterns for the Kwasa-Kwasa framework and the Turbulance language.

## Getting Started

### Prerequisites

Before using Kwasa-Kwasa, ensure you have:
- Rust 1.70 or later
- Python 3.8+ (for Python bindings)
- Git (for development)

### Installation

```bash
# Install from crates.io
cargo install kwasa-kwasa

# Or build from source
git clone https://github.com/yourusername/kwasa-kwasa.git
cd kwasa-kwasa
cargo build --release
```

### Your First Turbulance Program

Create a file called `hello.turb`:

```turbulance
// hello.turb - Your first Turbulance program

// Import standard modules
import io
import text

funxn main():
    // Basic text processing
    item message = "Hello, Kwasa-Kwasa World!"
    item words = message.split(" ")
    
    print("Original message: {}", message)
    print("Word count: {}", len(words))
    
    // Pattern analysis
    within message:
        given contains("Kwasa"):
            print("Found framework name!")
            
    // Simple proposition
    proposition GreetingAnalysis:
        motion IsPolite("Message contains polite greeting")
        
        within message:
            given contains("Hello"):
                support IsPolite
                print("Polite greeting detected")
```

Run it with:
```bash
kwasa run hello.turb
```

## Core Concepts

### 1. Variables and Assignment

```turbulance
// Dynamic typing
item temperature = 23.5
item name = "Sample_001"
item is_valid = true

// Type annotations (optional but recommended)
item data: List[Float] = [1.2, 3.4, 5.6]
item metadata: Dict[String, Any] = {"source": "sensor", "quality": "high"}

// Constants
const PI = 3.14159
const MAX_ITERATIONS = 1000
```

### 2. Functions

```turbulance
// Basic function
funxn calculate_mean(values: List[Float]) -> Float:
    item sum = 0.0
    for each value in values:
        sum += value
    return sum / len(values)

// Function with default parameters
funxn analyze_data(data: List[Float], method: String = "standard", threshold: Float = 0.05):
    given method == "standard":
        return standard_analysis(data, threshold)
    given method == "robust":
        return robust_analysis(data, threshold)
    given otherwise:
        throw ValueError("Unknown method: {}".format(method))

// Higher-order functions
funxn apply_to_all(items: List[Any], func: Function) -> List[Any]:
    item results = []
    for each item in items:
        results.append(func(item))
    return results
```

### 3. Control Flow

#### Conditional Logic

```turbulance
// Basic conditionals using 'given'
given temperature > 30:
    print("Hot weather")
given temperature > 20:
    print("Warm weather")
given temperature > 10:
    print("Cool weather")
given otherwise:
    print("Cold weather")

// Pattern-based conditionals
within data:
    given matches("error.*pattern"):
        handle_error()
    given matches("warning.*pattern"):
        handle_warning()
    given contains("success"):
        handle_success()
```

#### Loops

```turbulance
// For-each loop
item samples = ["sample1", "sample2", "sample3"]
for each sample in samples:
    item result = process_sample(sample)
    print("Processed: {}", sample)

// While loop
item counter = 0
while counter < 10:
    process_iteration(counter)
    counter += 1

// Pattern-based iteration
within dataset as records:
    given record.quality > 0.8:
        high_quality_records.append(record)
    given record.quality > 0.5:
        medium_quality_records.append(record)
    given otherwise:
        low_quality_records.append(record)
```

### 4. Data Structures

#### Working with Lists

```turbulance
// Creating lists
item numbers = [1, 2, 3, 4, 5]
item mixed = ["text", 42, 3.14, true]
item empty_list = []

// List operations
numbers.append(6)                    // Add element
numbers.extend([7, 8, 9])           // Add multiple elements
item first = numbers.first()          // Get first element
item last = numbers.last()            // Get last element
item slice = numbers[1:4]             // Slice: [2, 3, 4]

// List comprehensions
item squares = [x*x for x in range(10)]
item evens = [x for x in numbers if x % 2 == 0]
```

#### Working with Dictionaries

```turbulance
// Creating dictionaries
item person = {
    "name": "Dr. Smith",
    "age": 45,
    "department": "Physics",
    "publications": 23
}

// Accessing values
item name = person["name"]           // Bracket notation
item age = person.age                // Dot notation
item papers = person.get("publications", 0)  // With default

// Updating values
person["age"] = 46
person.email = "smith@university.edu"

// Iterating
for each key, value in person:
    print("{}: {}", key, value)
```

#### Working with Sets

```turbulance
// Creating sets
item unique_genes = {"BRCA1", "TP53", "EGFR"}
item numbers_set = {1, 2, 3, 4, 5}

// Set operations
item set_a = {1, 2, 3}
item set_b = {3, 4, 5}

item union = set_a | set_b           // {1, 2, 3, 4, 5}
item intersection = set_a & set_b    // {3}
item difference = set_a - set_b      // {1, 2}
```

## Pattern Matching

Pattern matching is a core feature of Turbulance:

### Basic Pattern Matching

```turbulance
item text = "The research paper discusses BRCA1 gene mutations"

within text:
    given matches("research.*paper"):
        classify_as("academic")
    given matches("news.*article"):
        classify_as("journalism")
    given contains("gene"):
        add_tag("genetics")
```

### Advanced Pattern Matching

```turbulance
// Named capture groups
within genetic_sequence:
    given matches("ATG(?<start_codon>...)(?<gene_body>.*)TAA"):
        item start = start_codon
        item body = gene_body
        analyze_gene(start, body)

// Multiple patterns
within data_stream:
    given matches(["pattern1", "pattern2", "pattern3"]):
        handle_multiple_matches()
```

## Propositions and Evidence

### Basic Proposition

```turbulance
proposition DataQuality:
    motion CompleteData("Dataset has no missing values")
    motion ConsistentFormat("All entries follow expected format")
    motion ReasonableValues("Values are within expected ranges")
    
    within dataset:
        given missing_values_count == 0:
            support CompleteData
        given format_validation_passed:
            support ConsistentFormat
        given all_values_in_range(min_val, max_val):
            support ReasonableValues
```

### Evidence Collection

```turbulance
evidence ExperimentalEvidence:
    sources:
        - lab_data: LabDatabase("experiments_2024")
        - sensor_data: SensorStream("temperature_humidity")
        - literature: PubmedSearch("climate change effects")
    
    collection:
        frequency: daily
        validation: cross_reference
        quality_threshold: 0.9
    
    processing:
        - remove_outliers(method="iqr")
        - normalize_units()
        - timestamp_validation()
```

## Error Handling

### Basic Error Handling

```turbulance
// Try-catch blocks
try:
    item data = load_file("experiment.csv")
    item results = analyze_data(data)
catch FileNotFound as e:
    print("File not found: {}", e.filename)
    item data = generate_sample_data()
catch DataCorruption as e:
    print("Data corruption detected: {}", e.message)
    item data = restore_from_backup()
finally:
    cleanup_temporary_files()
```

### Custom Exceptions

```turbulance
// Define custom exceptions
exception InsufficientDataError:
    message: "Not enough data points for reliable analysis"
    min_required: Integer
    actual_count: Integer

// Throw custom exceptions
funxn analyze_sample(data: List[Float]):
    given len(data) < 10:
        throw InsufficientDataError(
            min_required=10,
            actual_count=len(data)
        )
    
    // Proceed with analysis
    return perform_analysis(data)
```

## Working with Modules

### Importing Modules

```turbulance
// Import entire module
import statistics
import genomics
import chemistry

// Import specific functions
from math import sqrt, log, exp
from text_analysis import extract_keywords, sentiment_analysis

// Import with aliases
import numpy as np
import matplotlib.pyplot as plt

// Conditional imports
try:
    import advanced_features
    item has_advanced = true
catch ImportError:
    item has_advanced = false
    print("Advanced features not available")
```

### Creating Your Own Modules

Create a file `my_analysis.turb`:

```turbulance
// my_analysis.turb - Custom analysis module

/// Calculate the coefficient of variation
funxn coefficient_of_variation(data: List[Float]) -> Float:
    item mean_val = mean(data)
    item std_val = standard_deviation(data)
    return std_val / mean_val

/// Detect outliers using IQR method
funxn detect_outliers_iqr(data: List[Float]) -> List[Float]:
    item q1 = quantile(data, 0.25)
    item q3 = quantile(data, 0.75)
    item iqr = q3 - q1
    item lower_bound = q1 - 1.5 * iqr
    item upper_bound = q3 + 1.5 * iqr
    
    item outliers = []
    for each value in data:
        given value < lower_bound or value > upper_bound:
            outliers.append(value)
    
    return outliers

// Export functions for use in other modules
export coefficient_of_variation, detect_outliers_iqr
```

Use it in another file:

```turbulance
import my_analysis

item data = [1, 2, 3, 4, 5, 100]  // 100 is an outlier
item cv = my_analysis.coefficient_of_variation(data)
item outliers = my_analysis.detect_outliers_iqr(data)

print("Coefficient of variation: {}", cv)
print("Outliers: {}", outliers)
```

## Common Patterns and Idioms

### 1. Data Validation Pipeline

```turbulance
funxn validate_experimental_data(data: List[Dict]):
    item validated_data = []
    
    for each record in data:
        // Check required fields
        given not record.has_key("sample_id"):
            continue  // Skip invalid records
            
        // Validate ranges
        given record.temperature < -273 or record.temperature > 1000:
            print("Invalid temperature for sample: {}", record.sample_id)
            continue
            
        // Add validation metadata
        record.validated = true
        record.validation_timestamp = current_time()
        validated_data.append(record)
    
    return validated_data
```

### 2. Result Aggregation

```turbulance
funxn aggregate_results(experiments: List[Dict]) -> Dict:
    item aggregated = {
        "total_experiments": len(experiments),
        "success_rate": 0.0,
        "average_duration": 0.0,
        "error_types": {}
    }
    
    item successful = 0
    item total_duration = 0.0
    
    for each experiment in experiments:
        given experiment.status == "success":
            successful += 1
            total_duration += experiment.duration
        given experiment.status == "error":
            item error_type = experiment.error_type
            given error_type not in aggregated.error_types:
                aggregated.error_types[error_type] = 0
            aggregated.error_types[error_type] += 1
    
    aggregated.success_rate = successful / len(experiments)
    aggregated.average_duration = total_duration / successful if successful > 0 else 0.0
    
    return aggregated
```

### 3. Configuration Management

```turbulance
// config.turb - Configuration management
item default_config = {
    "analysis": {
        "method": "standard",
        "threshold": 0.05,
        "iterations": 1000
    },
    "output": {
        "format": "json",
        "precision": 3,
        "include_metadata": true
    },
    "logging": {
        "level": "info",
        "file": "analysis.log"
    }
}

funxn load_config(config_file: String = "config.json") -> Dict:
    try:
        item user_config = load_json(config_file)
        return merge_configs(default_config, user_config)
    catch FileNotFound:
        print("Config file not found, using defaults")
        return default_config

funxn merge_configs(default: Dict, user: Dict) -> Dict:
    item merged = default.copy()
    for each key, value in user:
        given isinstance(value, Dict) and key in merged:
            merged[key] = merge_configs(merged[key], value)
        given otherwise:
            merged[key] = value
    return merged
```

## Performance Tips

### 1. Use Appropriate Data Structures

```turbulance
// Use sets for membership testing
item valid_ids = {"ID001", "ID002", "ID003"}  // Set lookup: O(1)
given sample_id in valid_ids:  // Fast lookup
    process_sample(sample_id)

// Use dictionaries for key-value lookups
item gene_functions = {
    "BRCA1": "DNA repair",
    "TP53": "tumor suppressor",
    "EGFR": "growth factor receptor"
}
```

### 2. Lazy Evaluation

```turbulance
// Use generators for large datasets
funxn read_large_file(filename: String):
    with open(filename) as file:
        for each line in file:
            yield parse_line(line)

// Process in chunks
for each chunk in chunked(read_large_file("huge_dataset.csv"), 1000):
    results = process_chunk(chunk)
    save_intermediate_results(results)
```

### 3. Parallel Processing

```turbulance
// Parallel data processing
parallel process_samples(samples: List[String]):
    workers: 4
    for each sample in samples:
        item result = analyze_sample(sample)
        yield result

// Collect results
item all_results = []
for each result in process_samples(sample_list):
    all_results.append(result)
```

## Debugging and Testing

### Debug Output

```turbulance
// Debug assertions
debug assert temperature > 0, "Temperature must be positive"
debug assert len(sequence) > 0, "Sequence cannot be empty"

// Debug printing
debug print("Processing sample: {}", sample_id)
debug print("Intermediate calculation: {}", intermediate_result)

// Conditional debugging
given DEBUG_MODE:
    print("Debug info: {}", debug_data)
```

### Unit Testing

```turbulance
// test_analysis.turb - Unit tests

test "mean calculation works correctly":
    item data = [1, 2, 3, 4, 5]
    item expected = 3.0
    item actual = calculate_mean(data)
    assert_equals(expected, actual, tolerance=0.001)

test "outlier detection finds correct outliers":
    item data = [1, 2, 3, 4, 5, 100]
    item outliers = detect_outliers_iqr(data)
    assert_equals([100], outliers)

test "empty data handling":
    item empty_data = []
    try:
        calculate_mean(empty_data)
        assert_fail("Should have thrown an exception")
    catch EmptyDataError:
        // Expected behavior
        pass
```

## Next Steps

Now that you understand the basics, explore:

1. [Special Language Features](language/special_features.md) - Advanced constructs like propositions and metacognitive analysis
2. [Domain Extensions](examples/domain-extensions.md) - Specialized modules for genomics, chemistry, etc.
3. [Examples](examples/index.md) - Complete examples showing real-world usage
4. [API Reference](spec/index.md) - Detailed API documentation

## Common Gotchas

1. **Pattern Matching Scope**: Remember that `within` creates a new scope
2. **Type Inference**: While optional, type annotations help catch errors early
3. **Error Handling**: Always handle potential errors, especially when working with external data
4. **Memory Management**: Use streaming for large datasets to avoid memory issues
5. **Parallel Processing**: Be careful with shared state in parallel operations 