# Scientific Data Extensions for Kwasa-Kwasa

This document provides comprehensive documentation for using Kwasa-Kwasa with scientific data types beyond text. These extensions apply the same powerful abstraction principles to specialized domains, allowing seamless processing of diverse scientific data formats.

## Table of Contents
- [Core Philosophy](#core-philosophy)
- [Extension Architecture](#extension-architecture)
- [Mass Spectrometry Extension](#mass-spectrometry-extension)
- [Cheminformatics Extension](#cheminformatics-extension)
- [Genomic Extension (Previously Documented)](#genomic-extension)
- [Cross-Domain Applications](#cross-domain-applications)
- [Extension Development Guide](#extension-development-guide)

## Core Philosophy

Kwasa-Kwasa's foundation is built on the principle that arbitrary boundaries and mathematical operations can extract meaning from any sequence of symbols. The scientific data extensions demonstrate that this philosophy extends naturally beyond text to:

1. **Molecular Structures** - Where boundaries can be functional groups, rings, or bonds
2. **Spectral Data** - Where boundaries can be peaks, intensity ranges, or m/z windows
3. **Genomic Sequences** - Where boundaries can be genes, codons, or motifs

All these domains benefit from the same core operations:
- **Division (/)** - Splitting data into meaningful units
- **Multiplication (*)** - Combining data in domain-specific ways
- **Addition (+)** - Concatenating data preserving domain context
- **Subtraction (-)** - Removing specific elements while maintaining structure

## Extension Architecture

Each domain extension implements the following core traits:

```rust
pub trait Unit: Clone + Debug {
    fn content(&self) -> &[u8];
    fn display(&self) -> String;
    fn metadata(&self) -> &dyn std::any::Any;
    fn id(&self) -> &UnitId;
}

pub trait BoundaryDetector {
    type UnitType: Unit;
    fn detect_boundaries(&self, content: &[u8]) -> Vec<Self::UnitType>;
    fn configuration(&self) -> &BoundaryConfig;
}

pub trait UnitOperations<T: Unit> {
    fn divide(&self, unit: &T, pattern: &str) -> Vec<T>;
    fn multiply(&self, left: &T, right: &T) -> T;
    fn add(&self, left: &T, right: &T) -> T;
    fn subtract(&self, source: &T, to_remove: &T) -> T;
}
```

This consistent architecture ensures that all extensions benefit from the same core capabilities and can be used with the standard Turbulance syntax.

## Mass Spectrometry Extension

The Mass Spectrometry extension allows processing of spectral data commonly produced by analytical chemistry instruments.

### Key Types

- `MassSpectrum` - Represents a mass spectrum with peaks and metadata
- `Peak` - Individual m/z and intensity data points
- `SpectrumBoundaryDetector` - Identifies boundaries in spectral data
- `SpectrumOperations` - Implements mathematical operations for spectra

### Getting Started

```turbulance
import spectrometry

// Create a spectrum from raw data
item spectrum = spectrometry.MassSpectrum.from_numeric_data(
    [100.1, 120.2, 130.3, 145.6, 180.9, 212.4, 258.3],  // m/z values
    [1050, 320, 5200, 750, 3500, 12300, 8400],          // intensities
    "sample_spectrum"
)

// Basic properties and operations
print("Number of peaks: {}", len(spectrum.peaks()))
print("Base peak m/z: {:.4}", spectrum.base_peak().mz)
print("Total intensity: {:.1}", sum(spectrum.peaks().map(p => p.intensity)))
```

### Mathematical Operations

```turbulance
// Division: split by pattern
item spectrum_ops = spectrometry.SpectrumOperations.new()

// Split by m/z ranges (e.g., 100 m/z windows)
item range_parts = spectrum_ops.divide(spectrum, "mz_range")

// Split by intensity levels (low, medium, high)
item intensity_parts = spectrum_ops.divide(spectrum, "intensity")

// Split into individual peaks
item peak_parts = spectrum_ops.divide(spectrum, "peak")

// Addition: combining spectra
item combined = spectrum_ops.add(spectrum1, spectrum2)

// Multiplication: spectral convolution
item convolved = spectrum_ops.multiply(spectrum1, spectrum2)

// Subtraction: background removal
item background = create_background_spectrum()
item background_subtracted = spectrum_ops.subtract(spectrum, background)
```

### Integration with Propositions and Motions

```turbulance
proposition MassSpectralAnalysis:
    motion IsotopicPattern("Isotopic distribution follows predicted pattern")
    motion FragmentIdentification("Fragment ions match predicted structures")
    
    within spectrum:
        given contains_peak_at(212.4, tolerance=0.1):
            print("Found molecular ion")
        given intensity_at(258.3) > 5000:
            print("High abundance of fragment ion at m/z 258.3")
```

### Practical Applications

1. **Data preprocessing**:
   ```turbulance
   // Normalize spectrum
   item normalized = spectrum.normalize()
   
   // Filter by minimum intensity
   item filtered = spectrum.filter_by_intensity(1000)
   
   // Extract a specific m/z range
   item range_of_interest = spectrum.extract_range(200, 300)
   ```

2. **Fragment searching**:
   ```turbulance
   funxn find_fragments(spectrum, fragment_masses, tolerance=0.1):
       item matches = []
       for each mass in fragment_masses:
           item peaks = spectrum.peaks_in_range(mass - tolerance, mass + tolerance)
           if len(peaks) > 0:
               matches.append((mass, peaks[0]))
       return matches
   ```

3. **Spectral comparison**:
   ```turbulance
   funxn spectral_similarity(spec1, spec2, tolerance=0.1):
       item common_peaks = 0
       item total_peaks = len(spec1.peaks())
       
       for each peak1 in spec1.peaks():
           item matching = spec2.peaks_in_range(peak1.mz - tolerance, peak1.mz + tolerance)
           if len(matching) > 0:
               common_peaks += 1
       
       return common_peaks / total_peaks
   ```

## Cheminformatics Extension

The Cheminformatics extension enables processing of molecular structures, primarily represented in SMILES notation.

### Key Types

- `Molecule` - Represents a chemical structure with atoms, bonds, and metadata
- `Atom` - Individual atoms with properties
- `Bond` - Connections between atoms with bond types
- `MoleculeBoundaryDetector` - Identifies boundaries in molecular structures
- `MoleculeOperations` - Implements mathematical operations for molecules

### Getting Started

```turbulance
import chemistry

// Create molecules from SMILES strings
item ethanol = chemistry.Molecule.from_smiles("CCO", "ethanol")
item aspirin = chemistry.Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin")

// Basic properties
print("Ethanol SMILES: {}", ethanol.smiles())
print("Aspirin formula: {}", aspirin.formula())
print("Aspirin molecular weight: {:.2f}", aspirin.molecular_weight())
```

### Mathematical Operations

```turbulance
// Division: split by pattern
item molecule_ops = chemistry.MoleculeOperations.new()

// Split by functional groups
item functional_groups = molecule_ops.divide(aspirin, "functional_group")

// Split by rings
item rings = molecule_ops.divide(aspirin, "ring")

// Addition: combining molecules (mixing)
item mixture = molecule_ops.add(ethanol, aspirin)

// Multiplication: simulating reaction
item reaction_product = molecule_ops.multiply(ethanol, aspirin)

// Subtraction: removing a functional group
item hydroxyl = chemistry.Molecule.from_smiles("O", "hydroxyl")
item dehydroxylated = molecule_ops.subtract(ethanol, hydroxyl)
```

### Integration with Propositions and Motions

```turbulance
proposition MolecularAnalysis:
    motion DrugLikeness("Lipinski's Rule of Five predicts drug-like properties")
    motion ReactivityPrediction("Functional groups determine reaction potential")
    
    within aspirin:
        given contains_substructure("C(=O)O"):
            print("Contains carboxylic acid group - potential for esterification")
        given molecular_weight() < 500:
            print("Molecular weight within drug-like range")
```

### Practical Applications

1. **Structure analysis**:
   ```turbulance
   funxn analyze_functional_groups(molecule):
       item smiles = molecule.smiles()
       item groups = {}
       
       if smiles.contains("OH"): groups["hydroxyl"] = smiles.count("OH")
       if smiles.contains("C(=O)O"): groups["carboxyl"] = smiles.count("C(=O)O")
       if smiles.contains("C=O"): groups["carbonyl"] = smiles.count("C=O")
       
       return groups
   ```

2. **Molecular similarity**:
   ```turbulance
   funxn tanimoto_similarity(mol1, mol2):
       item fp1 = generate_fingerprint(mol1)
       item fp2 = generate_fingerprint(mol2)
       
       item intersection = bit_intersection(fp1, fp2)
       item union = bit_union(fp1, fp2)
       
       return len(intersection) / len(union)
   ```

3. **Drug-likeness prediction**:
   ```turbulance
   funxn check_lipinski_rule_of_five(molecule):
       item violations = 0
       
       if molecular_weight(molecule) > 500: violations += 1
       if count_h_donors(molecule) > 5: violations += 1
       if count_h_acceptors(molecule) > 10: violations += 1
       if calculate_logp(molecule) > 5: violations += 1
       
       return violations <= 1  // Allow one violation
   ```

## Genomic Extension

The Genomic extension is documented in more detail in `DOMAIN_EXPANSION.md`, but here's a brief overview for completeness:

```turbulance
import genomic

// Create a DNA sequence
item dna = genomic.NucleotideSequence.new("ATGCTAGCTAGCTAGCTA", "gene_123")

// Calculate GC content
print("GC content: {:.2f}%", dna.gc_content() * 100)

// Get reverse complement
item rev_comp = dna.reverse_complement()

// Translate to protein
item protein = dna.translate()
```

## Cross-Domain Applications

One of the most powerful features of Kwasa-Kwasa's domain extensions is the ability to work across multiple domains in the same script.

### Combined Analysis Workflows

```turbulance
import genomic
import spectrometry
import chemistry

// Analyze a protein-metabolite interaction
funxn analyze_interaction(protein_sequence, metabolite_smiles, binding_spectrum):
    // Genomic analysis of protein
    item protein = genomic.NucleotideSequence.new(protein_sequence, "protein")
    item domains = protein / "domain"
    
    // Chemical analysis of metabolite
    item metabolite = chemistry.Molecule.from_smiles(metabolite_smiles, "metabolite")
    item functional_groups = chemistry.MoleculeOperations.new().divide(metabolite, "functional_group")
    
    // Spectral analysis of binding
    item binding_data = spectrometry.MassSpectrum.from_file(binding_spectrum)
    item binding_peaks = binding_data.peaks_in_range(500, 1500)
    
    // Cross-domain correlation
    item correlations = correlate_binding_with_structure(binding_peaks, functional_groups, domains)
    
    return correlations
```

### Data Transformation Between Domains

```turbulance
// Convert spectral data to sequence representation
funxn spectrum_to_sequence(spectrum):
    item normalized = spectrum.normalize()
    item sequence = ""
    
    for each peak in normalized.peaks():
        // Convert intensity to amino acid letter (0-100% maps to A-Y)
        item intensity_scaled = min(int(peak.intensity / 4), 24)
        item amino = char(65 + intensity_scaled)  // ASCII: A=65, B=66, etc.
        sequence = sequence + amino
    
    return genomic.NucleotideSequence.new(sequence, "spectrum_derived")

// Now we can apply genomic analysis techniques to spectral data!
item spec_seq = spectrum_to_sequence(mass_spectrum)
item patterns = spec_seq / "pattern"
```

## Extension Development Guide

Want to create your own domain extension? Follow these steps:

1. **Define your domain unit types**:
   ```rust
   pub struct MyDomainUnit {
       content: Vec<u8>,
       metadata: MyDomainMetadata,
       id: UnitId,
       // Domain-specific fields
   }
   
   impl Unit for MyDomainUnit {
       // Implement required methods
   }
   ```

2. **Implement boundary detector**:
   ```rust
   pub struct MyDomainBoundaryDetector {
       config: BoundaryConfig,
       boundary_type: MyDomainBoundaryType,
   }
   
   impl BoundaryDetector for MyDomainBoundaryDetector {
       type UnitType = MyDomainUnit;
       // Implement required methods
   }
   ```

3. **Implement operations**:
   ```rust
   pub struct MyDomainOperations;
   
   impl UnitOperations<MyDomainUnit> for MyDomainOperations {
       // Implement division, multiplication, addition, subtraction
   }
   ```

4. **Create example scripts**:
   ```turbulance
   import my_domain
   
   // Demonstrate usage of your extension
   item my_unit = my_domain.MyDomainUnit.new(...)
   ```

For detailed implementation guidance, see the existing extensions in `src/spectrometry/mod.rs`, `src/chemistry/mod.rs`, and `src/genomic/mod.rs`. 