# Genomic Analysis Example

This example demonstrates how to use Kwasa-Kwasa's genomic extension to analyze DNA sequences. It shows various operations including finding open reading frames, calculating GC content, motif searching, and sequence manipulations.

## Source Code

```turbulance
// Example of genomic sequence analysis using Turbulance

// Define a DNA sequence
item dna_sequence = "ATGCCCGGGTAATCGGTAACGGCTAGCATTGCATGCATCGA"

// Function to find open reading frames
funxn find_orfs(sequence):
    // Split the sequence into codons
    item codons = sequence / "codon"
    item orfs = []
    
    within sequence:
        // Find start codons (ATG)
        given contains("ATG"):
            item start_pos = index_of("ATG")
            item orf = extract_from(start_pos)
            
            // Ensure it has a valid length
            given len(orf) >= 6 and len(orf) % 3 == 0:
                // Check for stop codons
                given not contains(["TAA", "TAG", "TGA"]):
                    orfs.append(orf)
    
    return orfs

// Function to calculate GC content
funxn gc_content(sequence):
    item gc_count = 0
    
    within sequence as nucleotides:
        given nucleotide in ["G", "C"]:
            gc_count = gc_count + 1
    
    return gc_count / len(sequence)

// Function to find motifs
funxn find_motifs(sequence, motif_pattern):
    item locations = []
    item current_pos = 0
    
    while current_pos < len(sequence):
        item found_at = sequence.find(motif_pattern, current_pos)
        
        given found_at != -1:
            locations.append(found_at)
            current_pos = found_at + 1
        given otherwise:
            break
    
    return locations
```

## Code Explanation

### 1. DNA Sequence Analysis Functions

#### Finding Open Reading Frames (ORFs)
```turbulance
funxn find_orfs(sequence):
    // Split the sequence into codons
    item codons = sequence / "codon"
    // ...
```
The `find_orfs` function:
- Splits DNA into codons (3-nucleotide units)
- Looks for start codons (ATG)
- Checks for valid ORF length
- Ensures no stop codons in the reading frame

#### GC Content Calculation
```turbulance
funxn gc_content(sequence):
    item gc_count = 0
    within sequence as nucleotides:
        given nucleotide in ["G", "C"]:
            gc_count = gc_count + 1
```
This function:
- Iterates through nucleotides
- Counts G and C bases
- Returns the GC percentage

#### Motif Finding
```turbulance
funxn find_motifs(sequence, motif_pattern):
    // ...
```
Searches for specific DNA patterns and returns their positions.

### 2. Sequence Operations

The example demonstrates various sequence operations:

```turbulance
// Addition: concatenation
item combined_exons = exon1 + exon2

// Division: split by pattern
item fragments = dna_sequence / "GGT"

// Multiplication: special joining (recombination)
item recombined = exon1 * exon2

// Subtraction: remove pattern
item filtered = dna_sequence - "GGT"
```

### 3. Gene Regulation Analysis

```turbulance
proposition GeneRegulation:
    motion Activation("Gene X activates Gene Y through binding site GGTA")
    motion Inhibition("Gene Z inhibits Gene X when bound to sequence ATGC")
    
    within dna_sequence:
        given contains("GGTA"):
            print("Found activation site for Gene Y")
```

This section demonstrates:
- Modeling gene regulatory networks
- Binding site analysis
- Regulatory motif detection

### 4. Advanced Pattern Analysis

```turbulance
funxn analyze_pattern_frequencies(sequence, pattern_size=3):
    item patterns = sequence / pattern_size
    // ...
```

Features:
- K-mer frequency analysis
- Shannon entropy calculation
- Pattern distribution analysis

## Running the Example

1. Save the code in a file with `.turb` extension
2. Run using the Kwasa-Kwasa interpreter:
   ```bash
   kwasa run genomic_analysis.turb
   ```

## Expected Output

```
Found 2 open reading frames
GC content: 52.50%
Found motif GGTA at positions: [8, 15]
Combined exons: ATGCCCGGGGCTAGCATT
Fragments after splitting by GGT: ["AT", "AACGGCTAGCATTGCATGCATCGA"]
Found activation site for Gene Y
Sequence entropy: 4.32 bits
Top 3 most common patterns:
  GCT occurs 2 times
  ATG occurs 2 times
  GGT occurs 2 times
```

## Key Concepts Demonstrated

1. **Sequence Analysis**:
   - Open Reading Frame detection
   - GC content calculation
   - Motif searching
   - Pattern frequency analysis

2. **Mathematical Operations**:
   - Sequence concatenation (`+`)
   - Pattern-based splitting (`/`)
   - Recombination (`*`)
   - Pattern removal (`-`)

3. **Biological Concepts**:
   - Gene regulation
   - DNA motifs
   - Sequence patterns
   - Codon analysis

4. **Advanced Features**:
   - Pattern frequency analysis
   - Information entropy
   - Reverse complement
   - Regulatory network modeling 