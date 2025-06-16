# Chemical Structure Analysis Example

This example demonstrates how to use Kwasa-Kwasa's chemistry extension to analyze molecular structures, perform chemical operations, and compare molecular similarities. It shows how the framework's pattern-based approach can be applied to chemical structures.

## Source Code

```turbulance
// Example of chemical molecule analysis using Turbulance

// Import the chemistry module
import chemistry

// Create molecules from SMILES strings
item ethanol = chemistry.Molecule.from_smiles("CCO", "ethanol").with_name("Ethanol")
item aspirin = chemistry.Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin").with_name("Aspirin")
item caffeine = chemistry.Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine").with_name("Caffeine")

// Function to analyze a molecule
funxn analyze_molecule(molecule):
    print("Analyzing molecule: {}".format(molecule.display()))
    print("SMILES: {}".format(molecule.smiles()))
    print("Formula: {}".format(molecule.formula()))
    print("Molecular weight: {:.2f}".format(molecule.molecular_weight()))
    
    // Display atom count
    print("Number of atoms: {}".format(len(molecule.atoms())))
    
    // Display bond count
    print("Number of bonds: {}".format(len(molecule.bonds())))
    
    // Calculate functional groups
    item functional_groups = identify_functional_groups(molecule)
    print("Functional groups:")
    for each group, count in functional_groups:
        print("  - {}: {}".format(group, count))
```

## Code Explanation

### 1. Molecule Creation and Basic Analysis

```turbulance
item ethanol = chemistry.Molecule.from_smiles("CCO", "ethanol").with_name("Ethanol")
```

The example shows how to:
- Create molecules from SMILES notation
- Add metadata (names, IDs)
- Access basic molecular properties
- Calculate structural features

### 2. Functional Group Analysis

```turbulance
funxn identify_functional_groups(molecule):
    item groups = {}
    item smiles = molecule.smiles()
    
    // Check for common functional groups
    if smiles.contains("OH"):
        groups["hydroxyl"] = smiles.count("OH")
    // ... more group checks
```

Features:
- Identifies common functional groups
- Counts group occurrences
- Pattern-based structure analysis
- SMILES pattern matching

### 3. Mathematical Operations on Molecules

```turbulance
// Division: split by functional groups
item ethanol_parts = molecule_ops.divide(ethanol, "functional_group")

// Multiplication: chemical reaction
item reaction_product = molecule_ops.multiply(ethanol, aspirin)

// Addition: combine molecules
item mixture = molecule_ops.add(caffeine, aspirin)

// Subtraction: remove a functional group
item dehydroxylated = molecule_ops.subtract(ethanol, chemistry.Molecule.from_smiles("O", "hydroxyl"))
```

Demonstrates:
- Splitting molecules by groups
- Simulating chemical reactions
- Combining molecular structures
- Removing structural elements

### 4. Chemical Analysis Propositions

```turbulance
proposition ChemicalAnalysis:
    motion DrugLikeness("Molecules with specific properties can act as drugs")
    motion SolubilityPrediction("Polar functional groups increase water solubility")
    
    within caffeine:
        given contains_substructure("CN1C"):
            print("Contains N-methylated nitrogen - common in psychoactive compounds")
```

Shows how to:
- Define chemical properties
- Check structural features
- Make chemical predictions
- Analyze drug-like properties

### 5. Molecular Complexity and Similarity

```turbulance
funxn molecular_complexity(molecule):
    item complexity = 0
    // Calculate complexity based on structure
    
funxn find_similar(molecule, candidates, threshold=0.7):
    // Compare molecular similarities
```

Features:
- Complexity scoring
- Structural similarity comparison
- Functional group matching
- Threshold-based filtering

## Running the Example

1. Save the code in a file with `.turb` extension
2. Run using the Kwasa-Kwasa interpreter:
   ```bash
   kwasa run chemistry_analysis.turb
   ```

## Expected Output

```
Analyzing molecule: Ethanol
SMILES: CCO
Formula: C2H6O
Molecular weight: 46.07
Number of atoms: 9
Number of bonds: 8
Functional groups:
  - hydroxyl: 1

Analyzing molecule: Aspirin
SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
Formula: C9H8O4
Molecular weight: 180.16
Functional groups:
  - carboxyl: 1
  - carbonyl: 2
  - aromatic: 1

Dividing molecules by functional groups:
Ethanol divided into 2 functional groups
Aspirin divided into 4 functional groups

Molecular complexity comparison:
Ethanol: 4.5
Aspirin: 28.5
Caffeine: 35.0

Molecules similar to aspirin:
Acetaminophen: 0.85 similarity
Ibuprofen: 0.78 similarity
```

## Key Concepts Demonstrated

1. **Molecular Representation**:
   - SMILES notation
   - Structural formulas
   - Molecular properties
   - Chemical metadata

2. **Chemical Analysis**:
   - Functional group identification
   - Structure decomposition
   - Complexity calculation
   - Similarity comparison

3. **Chemical Operations**:
   - Molecular division
   - Chemical reactions
   - Structure combination
   - Group removal

4. **Pattern Recognition**:
   - Substructure matching
   - Functional group patterns
   - Structural motifs
   - Chemical fingerprints

## Applications

1. **Drug Discovery**:
   - Structure-activity relationships
   - Drug-like property analysis
   - Molecular similarity search
   - Lead compound identification

2. **Chemical Informatics**:
   - Structure database searching
   - Molecular property prediction
   - Reaction pathway analysis
   - Structure visualization

3. **Material Science**:
   - Polymer analysis
   - Structure-property relationships
   - Material compatibility
   - Composite design 