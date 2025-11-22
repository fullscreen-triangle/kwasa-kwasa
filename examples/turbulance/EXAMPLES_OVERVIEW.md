# Turbulance Examples - Progressive Complexity

## Overview

This directory contains **6 Turbulance examples** progressing from simple to advanced:

### Basic Level (Conceptual Introduction)
1. `simple_point.trb` - Points (uncertain measurements)
2. `simple_resolution.trb` - Resolutions (debate platforms)
3. `simple_bmd.trb` - BMDs (categorical completion)

### Advanced Level (Detailed Demonstrations)
4. `01_point_demo_executable.trb` - Full uncertainty propagation
5. `02_resolution_demo_executable.trb` - Full Bayesian integration
6. `03_bmd_demo_executable.trb` - Full frame selection mechanism

---

## Quick Start

### Run Basic Examples

```bash
cd python

# Basic demonstrations
python turbulance.py ../examples/turbulance/simple_point.trb
python turbulance.py ../examples/turbulance/simple_resolution.trb
python turbulance.py ../examples/turbulance/simple_bmd.trb
```

### Run Advanced Examples

```bash
# Advanced demonstrations
python turbulance.py ../examples/turbulance/01_point_demo_executable.trb
python turbulance.py ../examples/turbulance/02_resolution_demo_executable.trb
python turbulance.py ../examples/turbulance/03_bmd_demo_executable.trb
```

### Run All with Output Saving

```bash
test_all_examples.bat  # Windows
```

---

## Progression Path

### Level 1: Basic Concepts

**Audience:** First-time readers, stakeholders  
**Goal:** Understand what each paradigm does  
**Time:** 2-3 minutes each  

#### `simple_point.trb`
- Shows 3 measurements with certainty
- Demonstrates explicit uncertainty
- ~30 lines

#### `simple_resolution.trb`
- Shows affirmations vs contentions
- Demonstrates debate outcome
- ~45 lines

#### `simple_bmd.trb`
- Shows 3 predetermined frames
- Demonstrates frame selection
- ~40 lines

### Level 2: Advanced Mechanisms

**Audience:** Researchers, implementers  
**Goal:** Understand HOW each paradigm works  
**Time:** 5-10 minutes each  

#### `01_point_demo_executable.trb`
- Full uncertainty propagation formulas
- Statistical significance calculations
- Clinical decision thresholds
- ~120 lines

#### `02_resolution_demo_executable.trb`
- Complete Bayesian integration
- Step-by-step posterior calculation
- Evidence weighting formulas
- Recommendation generation
- ~150 lines

#### `03_bmd_demo_executable.trb`
- Complete frame database (5 frames)
- Distance calculation for all frames
- Alternative scenarios (different H+ states)
- Variance minimization explanation
- ~170 lines

---

## Example Comparison

### What Gets More Complex?

| Aspect | Basic | Advanced |
|--------|-------|----------|
| **Length** | 30-45 lines | 120-170 lines |
| **Depth** | Shows result | Shows calculation |
| **Examples** | 1 scenario | Multiple scenarios |
| **Formulas** | Mentioned | Step-by-step |
| **Theory** | Implicit | Explicit |

### Basic Example (simple_point.trb)

```
Measuring H+ field coherence...
  Value: 0.67
  Certainty: 0.89
  
=> All measurements are Points
```

**What it shows:** Points have certainty

### Advanced Example (01_point_demo_executable.trb)

```
Measuring H+ field coherence...
  Value: 0.67
  Certainty: 0.89
  Evidence strength: 0.92
  Variance: 0.03
  
AGGREGATE CONSCIOUSNESS STATE:
  Overall certainty: 0.86
    -> Calculated: (0.89 * 0.76 * 0.94)^(1/3)
  
Delta H+ coherence: +0.15
  Combined certainty: 0.90
    -> sqrt(0.89^2 + 0.91^2) / sqrt(2)
  Statistical significance: 0.95
    -> z-score: |0.15| / sqrt(0.03)
```

**What it shows:** HOW certainty propagates with formulas

---

## Recommended Learning Path

### For Stakeholders / Executives

```
1. simple_point.trb          (understand uncertainty)
2. simple_resolution.trb      (understand debate)
3. simple_bmd.trb            (understand selection)
```

**Time:** 10 minutes  
**Outcome:** Conceptual understanding

### For Researchers / Scientists

```
1. simple_point.trb
   ↓
2. 01_point_demo_executable.trb      (see the math)
   ↓
3. simple_resolution.trb
   ↓
4. 02_resolution_demo_executable.trb (see Bayesian integration)
   ↓
5. simple_bmd.trb
   ↓
6. 03_bmd_demo_executable.trb       (see frame selection)
```

**Time:** 45 minutes  
**Outcome:** Implementation-ready understanding

### For Implementers / Developers

```
All 6 examples + reading the compiler source code
```

**Time:** 2 hours  
**Outcome:** Can extend the framework

---

## Output Examples

### Basic Output (simple_point.trb)

```
============================================================
Example 1: Oscillatory State as Point
============================================================

KEY INSIGHT: Biological measurements are uncertain!
SOLUTION: Represent as Points with explicit certainty

Measuring H+ field coherence...
  Value: 0.67 (measured)
  Certainty: 0.89 (89% confident)
...

=> This is how we 'compile consciousness' - as probabilistic data!
```

**Focus:** What Points do

### Advanced Output (01_point_demo_executable.trb)

```
============================================================
TURBULANCE PARADIGM: Oscillatory States as Points
============================================================

--- BASELINE: Patient 001 (Depressed State) ---

[Full measurements with variance, evidence strength]

AGGREGATE CONSCIOUSNESS STATE:
  Overall certainty: 0.86
    -> Calculated: (0.89 * 0.76 * 0.94)^(1/3)
  Weighted evidence: 0.90
    -> H+ (30%), O2 (20%), theta-gamma (50%)

--- TREATMENT: Sertraline + Omega-3 for 14 days ---

[Full post-treatment measurements]

COMPARING STATES (Uncertainty Propagation):
  [Step-by-step calculations with formulas]

CLINICAL ASSESSMENT:
  => CLINICALLY SIGNIFICANT IMPROVEMENT
  Criteria: Delta > 0.3 AND Certainty > 0.8
  Result: 0.44 > 0.3 AND 0.95 > 0.8 => TRUE
```

**Focus:** HOW Points work (with math)

---

## Validation Outputs

All examples can save outputs for validation:

```bash
python turbulance.py simple_point.trb --save-output
```

Creates:
- `validation_outputs/simple_point_latest.txt`
- `validation_outputs/simple_point_TIMESTAMP.txt`

These outputs serve as:
- ✅ Proof of execution
- ✅ Documentation examples
- ✅ Regression test baselines
- ✅ Stakeholder demonstrations

---

## Theory-to-Code Mapping

### Points → Biological Measurements

| Theory | Code Example |
|--------|--------------|
| H⁺ field coherence (uncertain) | `simple_point.trb` |
| Uncertainty propagation | `01_point_demo_executable.trb` |
| Statistical significance | `01_point_demo_executable.trb` |

**Papers:**
- `FORMAL_SPECIFICATION_PROBABILISTIC_POINTS.md`
- `KWASA_KWASA_BIOLOGICAL_OSCILLATORY_COMPUTING.md`

### Resolutions → Therapeutic Claims

| Theory | Code Example |
|--------|--------------|
| Affirmations vs contentions | `simple_resolution.trb` |
| Bayesian integration | `02_resolution_demo_executable.trb` |
| Evidence weighting | `02_resolution_demo_executable.trb` |

**Papers:**
- `RESOLUTION_VALIDATION_THROUGH_PERTURBATION.md`
- `kuramoto-oscillator-phase-computing.tex`

### BMDs → Thought Selection

| Theory | Code Example |
|--------|--------------|
| Frame database | `simple_bmd.trb` |
| H+ field navigation | `03_bmd_demo_executable.trb` |
| Variance minimization | `03_bmd_demo_executable.trb` |
| Categorical completion | `03_bmd_demo_executable.trb` |

**Papers:**
- `KWASA_KWASA_BIOLOGICAL_OSCILLATORY_COMPUTING.md`
- `categorical-completion-consciousness.tex`

---

## Next Steps

### After Running Examples

1. **Read the outputs** in `validation_outputs/`
2. **Compare theory to implementation**
3. **Try modifying values** (e.g., change H+ coupling)
4. **Create your own examples**

### Writing Your Own

Start from a template:

```turbulance
// My Consciousness Experiment

funxn main():
    print("============================================================")
    print("My Experiment Title")
    print("============================================================")
    print("")
    
    // Your measurements here
    
    print("=> Key insight!")
```

Run it:

```bash
python turbulance.py my_experiment.trb --save-output
```

### Extending the Compiler

Current compiler supports:
- ✅ `funxn main()`
- ✅ `print()` statements
- ✅ Comments (`//`)

To add:
- Complex expressions
- Multiple functions
- Control flow (if/else, loops)
- Type system
- Import system

See: `python/turbulance.py`

---

## File Reference

```
examples/turbulance/
├── README.md                           ← General overview
├── EXAMPLES_OVERVIEW.md                ← This file
│
├── simple_point.trb                    ← Basic: Points
├── simple_resolution.trb               ← Basic: Resolutions
├── simple_bmd.trb                      ← Basic: BMDs
│
├── 01_point_demo_executable.trb        ← Advanced: Points
├── 02_resolution_demo_executable.trb   ← Advanced: Resolutions
├── 03_bmd_demo_executable.trb          ← Advanced: BMDs
│
├── 01_oscillatory_state_as_point.trb   ← Full syntax (future)
└── 02_drug_efficacy_resolution.trb     ← Full syntax (future)
```

**Note:** Files with `_demo_executable` run with current compiler.  
Files without it are full Turbulance syntax (for future compiler versions).

---

## Summary

**6 Examples, 2 Levels, 3 Paradigms**

| Paradigm | Basic | Advanced |
|----------|-------|----------|
| Points | `simple_point.trb` | `01_point_demo_executable.trb` |
| Resolutions | `simple_resolution.trb` | `02_resolution_demo_executable.trb` |
| BMDs | `simple_bmd.trb` | `03_bmd_demo_executable.trb` |

**Run them:**

```bash
cd python
python turbulance.py ../examples/turbulance/simple_point.trb
```

**Save outputs:**

```bash
python turbulance.py ../examples/turbulance/simple_point.trb --save-output
```

**Run all:**

```bash
test_all_examples.bat
```

🧠⚡ **Consciousness compiled, progressively demonstrated!**

