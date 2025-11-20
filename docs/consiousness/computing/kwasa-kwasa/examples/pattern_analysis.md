# Pattern Analysis Example

This example demonstrates Kwasa-Kwasa's powerful pattern analysis capabilities, showing how to extract meaning from text through various pattern recognition techniques, including visual patterns, letter frequencies, and orthographic analysis.

## Source Code

```turbulance
// Example of pattern-based meaning extraction using Turbulance

// Some example text with potentially interesting patterns
item text = "The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump!"

// Function to analyze letter frequency
funxn letter_frequency(text):
    item frequencies = {}
    item total = 0
    
    within text as characters:
        given character.is_alpha():
            item char_lower = character.lower()
            
            given char_lower in frequencies:
                frequencies[char_lower] = frequencies[char_lower] + 1
            given otherwise:
                frequencies[char_lower] = 1
                
            total = total + 1
    
    // Convert to percentages
    for each letter in frequencies:
        frequencies[letter] = frequencies[letter] / total
    
    return frequencies

// Calculate Shannon entropy of text
funxn calculate_entropy(text):
    item frequencies = letter_frequency(text)
    item entropy = 0.0
    
    for each letter, freq in frequencies:
        given freq > 0:
            entropy = entropy - freq * log2(freq)
    
    return entropy

// Detect recurring visual patterns
funxn detect_visual_patterns(text, pattern_length=3):
    // Map letters to shape classes
    item shape_classes = {
        'a': 0, 'c': 0, 'e': 0, 'o': 0, 's': 0,  // round shapes
        'i': 1, 'l': 1, 'j': 1, 'f': 1, 't': 1,  // vertical strokes
        'm': 2, 'n': 2, 'h': 2, 'u': 2,          // arch shapes
        'v': 3, 'w': 3, 'x': 3, 'y': 3, 'z': 3,  // angled shapes
        'b': 4, 'd': 4, 'p': 4, 'q': 4, 'g': 4   // circles with stems
    }
    // ... rest of the function
```

## Code Explanation

### 1. Letter Frequency Analysis

```turbulance
funxn letter_frequency(text):
    item frequencies = {}
    item total = 0
    
    within text as characters:
        given character.is_alpha():
            // ... character counting logic
```

This function:
- Iterates through text characters
- Counts frequency of each letter
- Converts counts to percentages
- Useful for:
  - Statistical analysis
  - Language identification
  - Cryptographic analysis

### 2. Information Theory Analysis

```turbulance
funxn calculate_entropy(text):
    item frequencies = letter_frequency(text)
    item entropy = 0.0
    // ... entropy calculation
```

Calculates Shannon entropy of text:
- Measures information content
- Higher entropy = more randomness/information
- Lower entropy = more predictable patterns

### 3. Visual Pattern Recognition

```turbulance
funxn detect_visual_patterns(text, pattern_length=3):
    item shape_classes = {
        'a': 0, 'c': 0, 'e': 0, 'o': 0, 's': 0,  // round shapes
        'i': 1, 'l': 1, 'j': 1, 'f': 1, 't': 1,  // vertical strokes
        // ... more shape classes
    }
```

Features:
- Groups letters by visual shape
- Detects recurring shape patterns
- Useful for:
  - Typography analysis
  - Visual rhythm detection
  - Pattern-based meaning extraction

### 4. Orthographic Analysis

```turbulance
funxn visual_rhythm(text):
    // ... rhythm analysis code

funxn orthographic_density(text, width=40):
    // ... density mapping code
```

These functions analyze:
- Visual rhythm of text
- Character density patterns
- Typographic weight distribution
- Spatial relationships between characters

### 5. Statistical Pattern Analysis

```turbulance
funxn unusual_combinations(text, ngram_size=2):
    // ... n-gram analysis code
```

Identifies:
- Unusual letter combinations
- Deviations from expected frequencies
- Statistical anomalies in text

## Running the Example

1. Save the code in a file with `.turb` extension
2. Run using the Kwasa-Kwasa interpreter:
   ```bash
   kwasa run pattern_analysis.turb
   ```

## Expected Output

```
Analyzing text: The quick brown fox jumps over the lazy dog...

Letter frequencies:
  a: 0.0476
  b: 0.0238
  c: 0.0238
  ...

Text entropy: 4.2876 bits

Recurring visual patterns (by shape class):
  Pattern (0, 1, 2): ['ace', 'ane']
  Pattern (1, 0, 3): ['lex', 'laz']
  ...

Visual rhythm analysis (first 10 points):
  [0.4, 0.5, 0.7, 0.5, 0.2, 0.4, 0.5, 0.7, 0.4, 0.5]

Orthographic density map:
  ▁▂▃▅▂▁▃▄▂▁
  ▂▄▆▃▁▂▄▃▁▂
  ▁▃▂▁▄▅▂▁▃▄
```

## Key Concepts Demonstrated

1. **Statistical Analysis**:
   - Letter frequency counting
   - Shannon entropy calculation
   - N-gram analysis
   - Statistical anomaly detection

2. **Visual Analysis**:
   - Shape-based pattern recognition
   - Visual rhythm detection
   - Orthographic density mapping
   - Typographic weight analysis

3. **Pattern Recognition**:
   - Recurring shape patterns
   - Letter combination analysis
   - Visual pattern matching
   - Multi-level pattern detection

4. **Information Theory**:
   - Entropy calculation
   - Information density analysis
   - Pattern frequency analysis
   - Statistical significance testing

## Applications

1. **Text Analysis**:
   - Style analysis
   - Author identification
   - Language detection
   - Text complexity measurement

2. **Visual Design**:
   - Typography analysis
   - Layout optimization
   - Visual rhythm enhancement
   - Design pattern detection

3. **Pattern Discovery**:
   - Hidden pattern detection
   - Anomaly identification
   - Structure analysis
   - Pattern-based meaning extraction 