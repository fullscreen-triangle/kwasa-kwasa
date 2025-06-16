# Getting Started with Kwasa-Kwasa: A Practical Guide

## Prerequisites and Installation

### System Requirements

- **Rust**: Latest stable version (1.70+)
- **Python**: 3.8+ (for research integration and some extensions)
- **Git**: For cloning the repository
- **Operating System**: Linux, macOS, or Windows

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/kwasa-kwasa.git
cd kwasa-kwasa
```

2. **Build the Framework**
```bash
# Build in release mode for better performance
cargo build --release

# Run the test suite to verify installation
cargo test
```

3. **Install the CLI Tool**
```bash
# Install the kwasa-kwasa command globally
cargo install --path .

# Verify installation
kwasa-kwasa --version
```

4. **Optional: Python Integration**
```bash
# Install Python dependencies for research features
pip install -r python-requirements.txt
```

## Your First Turbulance Program

Create a file called `hello.turb`:

```turbulance
// hello.turb - Your first Turbulance program

funxn main():
    // Create some text to work with
    item text = "Hello, world! This is my first Turbulance program. It's quite exciting to work with text in this way."
    
    print("Original text: {}", text)
    
    // Split text into sentences using the division operator
    item sentences = text / sentence
    print("Number of sentences: {}", len(sentences))
    
    // Print each sentence
    considering all sentence in sentences:
        print("Sentence: {}", sentence)
    
    // Check readability
    item readability = readability_score(text)
    print("Readability score: {}", readability)
    
    // Use conditional processing
    given readability < 70:
        print("Text could be simpler")
    given otherwise:
        print("Text is reasonably readable")

// Run the main function
main()
```

Run your first program:

```bash
kwasa-kwasa run hello.turb
```

Expected output:
```
Original text: Hello, world! This is my first Turbulance program. It's quite exciting to work with text in this way.
Number of sentences: 3
Sentence: Hello, world!
Sentence: This is my first Turbulance program.
Sentence: It's quite exciting to work with text in this way.
Readability score: 82.3
Text is reasonably readable
```

## Understanding Text Units

Let's explore how text units work with a more detailed example:

```turbulance
// text_units.turb - Understanding text unit operations

funxn explore_text_units():
    item document = "Machine learning is a subset of artificial intelligence. It focuses on algorithms that can learn from data. These algorithms improve automatically through experience."
    
    print("=== Working with Text Units ===")
    
    // Split into different unit types
    item paragraphs = document / paragraph
    item sentences = document / sentence
    item words = document / word
    
    print("Paragraphs: {}", len(paragraphs))
    print("Sentences: {}", len(sentences))
    print("Words: {}", len(words))
    
    // Mathematical operations on text
    print("\n=== Text Mathematics ===")
    
    // Combine sentences with multiplication
    item combined = sentences[0] * sentences[1]
    print("Combined sentences: {}", combined)
    
    // Add content
    item extended = sentences[0] + " (especially in computer science)"
    print("Extended sentence: {}", extended)
    
    // Remove content with subtraction
    item shortened = document - "artificial intelligence"
    print("Shortened text: {}", shortened)

explore_text_units()
```

## Working with Propositions and Motions

Create a file called `propositions.turb`:

```turbulance
// propositions.turb - Understanding propositions and motions

funxn analyze_with_propositions():
    item academic_text = "The research methodology employed in this study utilized a mixed-methods approach. Quantitative data was collected through surveys, while qualitative insights were gathered via interviews. The results indicate a significant correlation between variables."
    
    // Define a proposition for academic writing quality
    proposition AcademicQuality:
        motion Formality("Text maintains appropriate academic tone")
        motion Clarity("Ideas are expressed clearly")
        motion Evidence("Claims are supported by evidence")
        
        // Test our text against these motions
        within academic_text:
            // Check for formal language indicators
            given contains("methodology") or contains("quantitative") or contains("qualitative"):
                support Formality
                print("✓ Formal academic language detected")
            
            // Check readability for clarity
            given readability_score() > 50:
                support Clarity
                print("✓ Text is reasonably clear")
            
            // Check for evidence-based language
            given contains("results") or contains("data") or contains("study"):
                support Evidence
                print("✓ Evidence-based language found")
    
    // Test individual motions
    item formality_check = academic_text.check_formality()
    print("Formality score: {}", formality_check.score)

analyze_with_propositions()
```

## Text Processing Pipeline Example

Create `pipeline.turb` to see how to chain operations:

```turbulance
// pipeline.turb - Building text processing pipelines

funxn create_processing_pipeline():
    item raw_text = "this is some text that needs improvement. it has poor capitalization and might be hard to read for some people. we should make it better."
    
    print("Original: {}", raw_text)
    
    // Method 1: Step by step processing
    print("\n=== Step by Step Processing ===")
    item step1 = capitalize_sentences(raw_text)
    print("After capitalization: {}", step1)
    
    item step2 = improve_readability(step1)
    print("After readability improvement: {}", step2)
    
    item step3 = add_transitions(step2)
    print("After adding transitions: {}", step3)
    
    // Method 2: Pipeline processing (more elegant)
    print("\n=== Pipeline Processing ===")
    item processed = raw_text |>
        capitalize_sentences() |>
        improve_readability() |>
        add_transitions() |>
        check_grammar()
    
    print("Final result: {}", processed)
    
    // Analyze the improvement
    item original_score = readability_score(raw_text)
    item improved_score = readability_score(processed)
    
    print("\nReadability improvement: {} -> {}", original_score, improved_score)

create_processing_pipeline()
```

## Using the Text Analysis Functions

Create `analysis.turb`:

```turbulance
// analysis.turb - Using built-in analysis functions

funxn comprehensive_analysis():
    item text = "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once. It's commonly used for testing purposes."
    
    print("=== Comprehensive Text Analysis ===")
    print("Text: {}", text)
    print()
    
    // Basic statistics
    print("Character count: {}", len(text))
    print("Word count: {}", word_count(text))
    print("Sentence count: {}", sentence_count(text))
    print()
    
    // Readability analysis
    print("Readability score: {}", readability_score(text))
    print("Grade level: {}", grade_level(text))
    print()
    
    // Sentiment analysis
    item sentiment = sentiment_analysis(text)
    print("Sentiment polarity: {}", sentiment.polarity)
    print("Sentiment subjectivity: {}", sentiment.subjectivity)
    print()
    
    // Keyword extraction
    item keywords = extract_keywords(text, 5)
    print("Top keywords:")
    considering all keyword in keywords:
        print("  - {}", keyword)
    print()
    
    // Pattern analysis
    item patterns = find_patterns(text, "common_english")
    print("Detected patterns: {}", len(patterns))
    
    // Check for specific content
    given contains_technical_terms(text):
        print("Contains technical terms")
    given otherwise:
        print("No technical terms detected")

comprehensive_analysis()
```

## Working with Files

Create `file_processing.turb`:

```turbulance
// file_processing.turb - Working with external files

funxn process_file():
    // Read a text file
    item content = read_file("sample.txt")
    
    given content.is_error():
        print("Error reading file: {}", content.error())
        return
    
    item text = content.unwrap()
    print("Loaded {} characters from file", len(text))
    
    // Process the content
    item processed = text |>
        normalize_whitespace() |>
        correct_spelling() |>
        improve_readability()
    
    // Save the processed version
    item result = write_file("processed_sample.txt", processed)
    
    given result.is_ok():
        print("Processed file saved successfully")
    given otherwise:
        print("Error saving file: {}", result.error())

// Alternative: streaming processing for large files
funxn process_large_file():
    item file_stream = open_file_stream("large_document.txt")
    item output_stream = create_file_stream("processed_large_document.txt")
    
    // Process in chunks to manage memory
    file_stream.process_chunks(chunk_size=1000) { chunk ->
        item processed_chunk = chunk |>
            normalize_whitespace() |>
            correct_spelling()
        
        output_stream.write(processed_chunk)
    }
    
    print("Large file processing complete")

process_file()
```

## Interactive Mode

Kwasa-Kwasa provides an interactive REPL (Read-Eval-Print Loop) for experimentation:

```bash
# Start interactive mode
kwasa-kwasa repl
```

In the REPL, you can try:

```turbulance
>>> item text = "Hello, interactive mode!"
>>> print(text)
Hello, interactive mode!

>>> item words = text / word
>>> print(len(words))
3

>>> item readability = readability_score(text)
>>> print("Readability: {}", readability)
Readability: 91.5

>>> // Define a quick proposition
>>> proposition Test:
...     motion Simple("Text should be simple")
...     within text:
...         given readability_score() > 80:
...             support Simple
...
✓ Simple motion supported

>>> exit
```

## Configuration and Settings

Create a configuration file `kwasa.toml`:

```toml
[general]
# Default readability target
readability_target = 70

# Enable debug mode
debug = false

# Default text unit types for operations
default_division_unit = "sentence"

[research]
# Enable research integration
enabled = true

# Research depth (shallow, medium, deep)
default_depth = "medium"

# External sources
sources = ["wikipedia", "academic_papers"]

[extensions]
# Enable domain extensions
genomic = true
chemistry = true
mass_spec = false

[processing]
# Number of parallel processing threads
threads = 4

# Memory limit for large files (MB)
memory_limit = 1024

# Enable caching
cache_enabled = true
cache_size = 100
```

Load configuration in your programs:

```turbulance
// Load configuration
item config = load_config("kwasa.toml")

// Use configuration settings
item target_readability = config.general.readability_target
given readability_score(text) < target_readability:
    improve_readability(text)
```

## Debugging and Error Handling

```turbulance
// debugging.turb - Error handling and debugging

funxn debug_example():
    // Enable debug mode
    debug_mode(true)
    
    item text = "Sample text for debugging"
    
    // Log processing steps
    log_debug("Starting analysis of text: {}", text)
    
    try:
        item result = analyze_text(text)
        log_debug("Analysis successful: {}", result)
        print("Result: {}", result)
        
    catch AnalysisError as e:
        log_error("Analysis failed: {}", e.message)
        print("Error during analysis: {}", e.message)
        
    catch FileError as e:
        log_error("File operation failed: {}", e.message)
        
    finally:
        log_debug("Analysis complete")
        debug_mode(false)

// Error handling with text operations
funxn safe_text_operations():
    item text = "Test text"
    
    // Safe division that handles errors
    item sentences = try_divide(text, sentence)
    
    given sentences.is_ok():
        print("Successfully divided into {} sentences", len(sentences.unwrap()))
    given otherwise:
        print("Division failed: {}", sentences.error())
        // Fallback behavior
        item fallback = [text]  // Treat entire text as one unit
        print("Using fallback: single unit")

debug_example()
safe_text_operations()
```

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for more complex use cases
2. **Read Documentation**: Review `docs/complete_system_guide.md` for comprehensive understanding
3. **Try Domain Extensions**: Experiment with genomic or chemistry extensions
4. **Build Your Own Tools**: Create custom text processing tools for your specific needs
5. **Join the Community**: Contribute to the project or ask questions

## Common Issues and Solutions

### Installation Problems

**Issue**: Cargo build fails with dependency errors
```bash
# Solution: Update Rust and clear cache
rustup update
cargo clean
cargo build --release
```

**Issue**: Python integration not working
```bash
# Solution: Check Python version and reinstall dependencies
python3 --version  # Should be 3.8+
pip install -r python-requirements.txt
```

### Runtime Errors

**Issue**: "Text unit type not recognized"
```turbulance
// Problem: Using undefined text unit type
item result = text / unknown_type

// Solution: Use defined types
item result = text / sentence  // or word, paragraph, etc.
```

**Issue**: "Function not found"
```turbulance
// Problem: Function name doesn't exist
item score = unknown_function(text)

// Solution: Check available functions
item score = readability_score(text)
```

### Performance Issues

**Issue**: Slow processing on large files
```turbulance
// Problem: Loading entire file into memory
item content = read_file("huge_file.txt")

// Solution: Use streaming
item stream = open_file_stream("huge_file.txt")
stream.process_chunks(1000) { chunk -> process(chunk) }
```

You're now ready to start building with Kwasa-Kwasa! The framework provides a powerful foundation for text processing that goes far beyond traditional string manipulation. 