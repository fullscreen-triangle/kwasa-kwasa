# Kwasa-Kwasa Domain Expansions

This document explains how to use the new domain expansions in Kwasa-Kwasa, which extend the framework beyond text processing to handle genomic data and pattern-based meaning extraction.

## Overview

The core philosophy of Kwasa-Kwasa is to work with arbitrarily defined units and their relationships, regardless of their semantic meaning to humans. The domain expansions take this concept further by applying the same powerful abstractions to:

1. **Genomic Sequence Analysis** - Treating DNA/RNA sequences as units that can be manipulated with the same operators as text
2. **Pattern-Based Meaning Extraction** - Finding meaning in fundamental patterns of symbols, independent of their conventional semantic meaning

## Getting Started with Genomic Analysis

### Setting Up

First, add the genomic module to your Turbulance imports:

```turbulance
import genomic
```

### Basic Genomic Operations

#### Creating a Sequence

```turbulance
// Create a new DNA sequence
var dna = genomic.NucleotideSequence.new("ATGCTAGCTAGCTAGCTA", "gene_123")

// Access properties
print("GC content: {:.2f}%".format(dna.gc_content() * 100))
print("Length: {} bp".format(len(dna.content())))
```

#### Manipulating Sequences with Mathematical Operators

```turbulance
// Split into fragments using division operator
var motifs = dna / "GCT"

// Combine sequences with multiplication (recombination)
var exon1 = genomic.NucleotideSequence.new("ATGCCC", "exon1")
var exon2 = genomic.NucleotideSequence.new("GGGTGA", "exon2")
var joined = exon1 * exon2

// Concatenate sequences with addition
var concatenated = exon1 + exon2

// Remove pattern with subtraction
var filtered = dna - "GCT"
```

#### Using Within Blocks for Genomic Processing

```turbulance
within dna:
    // Process the sequence
    given contains("ATG"):
        print("Found start codon at position {}".format(index_of("ATG")))
        
    given gc_content() > 0.5:
        print("High GC content region detected")
```

#### Processing Sequences with Different Unit Types

```turbulance
// Convert sequence to codons
within dna as codons:
    for each codon:
        given codon == "ATG":
            print("Start codon found")
        given codon in ["TAA", "TAG", "TGA"]:
            print("Stop codon found")
```

### Working with Genomic Propositions

```turbulance
proposition GeneRegulation:
    motion Activation("Gene X activates Gene Y")
    motion Inhibition("Gene Z inhibits Gene X")
    
    within dna:
        given contains("TATAAA"):
            print("Found TATA box promoter")
            ensure_follows("Gene body")
```

### Pipeline Processing for Genomic Data

```turbulance
// Create a genomic analysis pipeline
var result = dna |>
    find_open_reading_frames() |>
    filter_by_length(min_length=100) |>
    translate_to_protein() |>
    predict_secondary_structure()
```

## Getting Started with Pattern Analysis

### Setting Up

First, add the pattern module to your Turbulance imports:

```turbulance
import pattern
```

### Basic Pattern Operations

#### Creating Pattern Analyzers

```turbulance
// Create analyzers
var analyzer = pattern.PatternAnalyzer.new()
var ortho_analyzer = pattern.OrthographicAnalyzer.new()
```

#### Analyzing Character Distributions

```turbulance
// Analyze n-gram frequencies
var text = "The quick brown fox jumps over the lazy dog"
var trigrams = analyzer.analyze_ngrams(text, 3)

// Calculate entropy
var entropy = analyzer.shannon_entropy(text)
print("Text entropy: {:.2f} bits".format(entropy))
```

#### Finding Significant Patterns

```turbulance
// Detect statistically significant patterns
var patterns = analyzer.detect_significant_patterns(text, 2, 5)

for each p in patterns:
    print("Pattern '{}' occurs {} times (significance: {:.2f})".format(
        p.content(), p.occurrences(), p.significance()
    ))
```

#### Analyzing Visual Patterns

```turbulance
// Generate visual density map
var density_map = ortho_analyzer.visual_density(text, 40)
print("Average density: {:.2f}".format(density_map.average_density()))

// Extract visual rhythm
var rhythm = ortho_analyzer.visual_rhythm(text)
```

### Mathematical Operators for Patterns

```turbulance
// Division: Split text by pattern type
var visual_units = text / "visual_class"

// Multiplication: Combine based on pattern similarity
var combined = text1 * text2

// Addition: Concatenate with pattern-aware joining
var joined = text1 + text2

// Subtraction: Remove common patterns
var uncommon = text - common_patterns
```

### Working with Pattern Propositions

```turbulance
proposition TextPatternAnalysis:
    motion FrequencyDistribution("Letter distribution shows specific patterns")
    motion VisualDensity("Text has visual density anomalies")
    
    within text:
        given entropy > 4.5:
            print("High information density detected")
        given contains_unusual_patterns():
            print("Text contains statistically unusual patterns")
```

## Advanced Usage

### Combining Genomic and Pattern Analysis

One powerful aspect of Kwasa-Kwasa is the ability to apply pattern analysis techniques to genomic data:

```turbulance
// Analyze patterns in a genomic sequence
var dna = genomic.NucleotideSequence.new("ATGCTAGCTAGCTAGCTA", "gene_123")
var pattern_analyzer = pattern.PatternAnalyzer.new()

// Find repeating patterns in DNA
var significant_patterns = pattern_analyzer.detect_significant_patterns(dna.content(), 3, 7)

for each p in significant_patterns:
    print("DNA pattern '{}' occurs {} times".format(p.content(), p.occurrences()))
```

### Creating Custom Unit Types

You can define your own unit types for specialized analysis:

```turbulance
struct CustomUnit:
    content: bytes
    metadata: any
    
    funxn new(content, name):
        return CustomUnit {
            content: content,
            metadata: { "name": name }
        }
    
    // Implement Unit trait
    funxn content(self):
        return self.content
    
    funxn display(self):
        return string(self.content)
    
    funxn metadata(self):
        return self.metadata
```

## Example Projects

Check out the examples directory for complete projects that demonstrate these capabilities:

- `examples/genomic_analysis.turb` - Demonstrates genomic sequence analysis
- `examples/pattern_analysis.turb` - Shows pattern-based meaning extraction
- `examples/combined_analysis.turb` - Combines both approaches

## Next Steps

To learn more about these domain expansions, refer to the following documents:

- `domain_expansion_plan.md` - Detailed implementation plan
- API documentation under `docs/api/genomic/` and `docs/api/pattern/`
- The source code in `src/genomic/` and `src/pattern/` 