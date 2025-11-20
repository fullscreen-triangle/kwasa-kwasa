# Basic Turbulance Example

This example demonstrates the core features of the Turbulance language, including project setup, text processing, and basic operations.

## Source Code

```turbulance
// Basic Turbulance example script
// This demonstrates the core language features

// Project declaration with metadata
project "Simple Demo" {
    author: "Kwasa-Kwasa Team",
    version: "0.1.0",
    description: "A basic demo of Turbulance language features"
}

// Source declarations
sources {
    "data/sample.txt" as "text",
    "data/reference.bib" as "bibliography"
}

// Function declaration
funxn greet(name) {
    return "Hello, " + name + "!";
}

// Text processing function
funxn process_paragraph(text) {
    within text {
        // Apply text operations
        simplify();
        
        // Conditional processing
        given contains("complex") {
            replace_jargon();
        }
        
        ensure readability_score() > 70;
    }
}

// Main execution block
{
    // Variable assignment
    let welcome_message = greet("Turbulance User");
    print(welcome_message);
    
    // String and number operations
    let count = 5;
    let stars = "*" * count;  // String repetition
    print(stars);
    
    // Text unit creation and processing
    let paragraph = "The complex algorithm utilizes advanced computational techniques...";
    
    // Process text with the function
    let processed = process_paragraph(paragraph);
    print("Processed text:");
    print(processed);
}
```

## Code Explanation

### 1. Project Setup

The example starts with a project declaration that includes metadata:

```turbulance
project "Simple Demo" {
    author: "Kwasa-Kwasa Team",
    version: "0.1.0",
    description: "A basic demo of Turbulance language features"
}
```

This metadata helps organize and document your Turbulance scripts.

### 2. Source Declarations

```turbulance
sources {
    "data/sample.txt" as "text",
    "data/reference.bib" as "bibliography"
}
```

The `sources` block declares external files that the script will use, giving them logical names for reference.

### 3. Basic Function Declaration

```turbulance
funxn greet(name) {
    return "Hello, " + name + "!";
}
```

Functions in Turbulance are declared using the `funxn` keyword. This example shows basic string concatenation.

### 4. Text Processing

```turbulance
funxn process_paragraph(text) {
    within text {
        simplify();
        given contains("complex") {
            replace_jargon();
        }
        ensure readability_score() > 70;
    }
}
```

This function demonstrates several key Turbulance features:
- `within` blocks for text unit operations
- `given` for conditional processing
- `ensure` for validation
- Built-in functions like `simplify()` and `readability_score()`

### 5. Main Execution

The main block shows:
- Variable assignment
- String operations
- Text processing
- Output using `print()`

## Running the Example

1. Save the code in a file with `.trb` extension (e.g., `basic_example.trb`)
2. Create the required data files (`data/sample.txt` and `data/reference.bib`)
3. Run using the Kwasa-Kwasa interpreter:
   ```bash
   kwasa run basic_example.trb
   ```

## Expected Output

```
Hello, Turbulance User!
*****
Processed text:
The algorithm uses computational methods to analyze text data and find meaning through language models and neural networks.
```

## Key Concepts Demonstrated

1. **Project Organization**: Metadata and source declarations
2. **Functions**: Basic and text-processing functions
3. **Text Operations**: 
   - String concatenation (`+`)
   - String repetition (`*`)
   - Text simplification
   - Jargon replacement
4. **Control Structures**:
   - `within` blocks
   - `given` conditions
   - `ensure` statements
5. **Variables and Types**:
   - String variables
   - Number variables
   - Text units 