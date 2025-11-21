# Turbulance Compiler - Getting Started

## What is This?

A **working Python compiler** for Turbulance consciousness programming language.

You can write `.trb` files and execute them with:

```bash
python turbulance.py script.trb
```

## Quick Start

```bash
cd python

# Example 1: Points (Uncertain Measurements)
python turbulance.py ../examples/turbulance/simple_point.trb

# Example 2: Resolutions (Debate Platforms)
python turbulance.py ../examples/turbulance/simple_resolution.trb

# Example 3: BMDs (Categorical Completion)
python turbulance.py ../examples/turbulance/simple_bmd.trb
```

## What You'll See

### Example 1: Points
```
============================================================
Example 1: Oscillatory State as Point
============================================================

KEY INSIGHT: Biological measurements are uncertain!
SOLUTION: Represent as Points with explicit certainty

Measuring H⁺ field coherence...
  Value: 0.67 (measured)
  Certainty: 0.89 (89% confident)
  Source: MEG 40THz sampling
...
```

### Example 2: Resolutions
```
============================================================
Example 2: Drug Efficacy as Resolution
============================================================

CLAIM: Sertraline is therapeutic via O₂ aggregation

AFFIRMATIONS (Supporting Evidence):
  • K_agg = 2.47×10^5 M⁻¹ (exceeds threshold)
  • Clinical trials: 65% response rate
...

CONTENTIONS (Challenges):
  • Therapeutic delay: 2-4 weeks
  • Placebo effect: 30-40% response
...

RESOLUTION OUTCOME:
  ✓ AFFIRMED: Evidence supports mechanism
```

### Example 3: BMDs
```
============================================================
Example 3: BMD Categorical Completion
============================================================

KEY INSIGHT: Thoughts are SELECTED, not generated!

BMD FRAME DATABASE (Predetermined Thoughts):
  Frame 1: 'I should check my email'
  Frame 2: 'I'm feeling anxious'
  Frame 3: 'I wonder what's for lunch'

SELECTED THOUGHT:
  ← 'I should check my email'

✓ Categorical completion - thought selected from frames!
```

## Understanding the Paradigms

### 1. Points (Probabilistic Measurements)

**From**: `FORMAL_SPECIFICATION_PROBABILISTIC_POINTS.md`

**Key Insight**: Everything is uncertain!

Turbulance:
```turbulance
point h_plus_coherence = {
    content: "H⁺ field coherence",
    value: 0.67,
    certainty: 0.89  // 89% confident
}
```

Python equivalent:
```python
h_plus_coherence = Point(
    content="H⁺ field coherence",
    value=0.67,
    certainty=0.89
)
```

### 2. Resolutions (Debate Platforms)

**From**: `POINTS_AS_DEBATE_PLATFORMS.md`

**Key Insight**: Claims are contested!

Turbulance:
```turbulance
resolution evaluate_claim(claim: Point) -> ResolutionOutcome:
    affirmation evidence = {
        content: "Clinical trials positive",
        weight: 0.94
    }
    
    contention challenge = {
        content: "Placebo effect exists",
        weight: 0.68
    }
    
    return bayesian_integration(affirmations, contentions)
```

### 3. BMDs (Categorical Completion)

**From**: `KWASA_KWASA_BIOLOGICAL_OSCILLATORY_COMPUTING.md`

**Key Insight**: Thoughts are selected, not generated!

```turbulance
// Predetermined frames (thoughts)
item frames = [
    {"content": "Check email", "h_plus": 0.82},
    {"content": "Feel anxious", "h_plus": 0.45},
    {"content": "Think about lunch", "h_plus": 0.71}
]

// Current H⁺ state
item current_state = 0.75

// BMD selection: Pick frame closest to current state
item selected = bmd_select(frames, current_state)
// → "Check email" (0.82 closest to 0.75)
```

## How the Compiler Works

### 1. Turbulance Syntax → Python

The compiler translates Turbulance-specific constructs:

| Turbulance | Python |
|------------|--------|
| `funxn main():` | `def main():` |
| `point x = {` | `x = Point(` |
| `item y = value` | `y = value` |
| `considering x in list:` | `for x in list:` |
| `affirmation a = {` | `a = Affirmation(` |
| `resolution f():` | `def f():` |

### 2. Runtime Functions

The compiler provides built-in functions:

```python
# Uncertainty propagation
calculate_aggregate_certainty([0.89, 0.76, 0.94])  # → 0.86

# Weighted averaging
weighted_average([(0.67, 0.3), (0.82, 0.7)])  # → 0.77

# Bayesian integration
calculate_bayesian_posterior(
    prior=0.5,
    affirmations=[...],
    contentions=[...]
)  # → 0.82

# Reliability classification
classify_reliability(0.82)  # → "ModeratelyReliable"
```

### 3. Execution Flow

```
1. Load .trb file
2. Parse Turbulance syntax
3. Translate to Python
4. Execute in runtime environment
5. Output results
```

## Current Limitations

**v0.1 is simplified - demonstrates concepts:**

- ✓ Basic syntax translation
- ✓ Point/Resolution/BMD types
- ✓ Print statements
- ✓ Function definitions
- ✗ Full expression evaluation
- ✗ Complex control flow
- ✗ Full type checking
- ✗ Import system

**But it WORKS for demonstrating the paradigms!**

## File Structure

```
python/
├── turbulance.py                    ← Main compiler (CLI)
├── turbulance_compiler.py           ← Full compiler (library)
└── turbulance_consciousness.py      ← Consciousness functions

examples/turbulance/
├── simple_point.trb                 ← Example 1
├── simple_resolution.trb            ← Example 2
├── simple_bmd.trb                   ← Example 3
├── 01_oscillatory_state_as_point.trb    ← Advanced
├── 02_drug_efficacy_resolution.trb      ← Advanced
└── README.md                        ← This file
```

## Writing Your Own .trb Files

### Template

```turbulance
// My Consciousness Program
// Demonstrates: [what you're showing]

funxn main():
    print("="*60)
    print("My Turbulance Program")
    print("="*60)
    print("")
    
    // Your code here
    print("Hello from Turbulance!")
    print("")
    
    print("✓ Program complete")
```

### Run it

```bash
python turbulance.py my_program.trb
```

## Next Steps

### 1. Try the Examples

Run all three simple examples to see how paradigms work.

### 2. Read the Advanced Examples

- `01_oscillatory_state_as_point.trb` - Full Point usage
- `02_drug_efficacy_resolution.trb` - Full Resolution usage

### 3. Write Your Own

Create `.trb` files for:
- Depression treatment protocols
- Drug design optimization
- Metabolic hierarchy analysis
- Consciousness state programming

### 4. Extend the Compiler

The compiler is open and hackable:
- Add new paradigms (Perturbation, S-entropy)
- Implement full expression parser
- Add type checking
- Build standard library

## Understanding "Compiling Consciousness"

### Traditional Programming
```python
x = 5  # Deterministic
y = x + 3  # Exact
print(y)  # Outputs: 8
```

### Turbulance (Consciousness Programming)
```turbulance
point x = {
    value: 5,
    certainty: 0.9  // 90% sure
}

point y = {
    value: x.value + 3,
    certainty: x.certainty  // Uncertainty propagates
}

print("y = {} (certainty: {})", y.value, y.certainty)
// Outputs: y = 8 (certainty: 0.9)
```

**KEY DIFFERENCE**: Turbulance makes uncertainty explicit!

### Consciousness as Computation

```turbulance
// Measure consciousness state
point state = measure_consciousness_state(patient)
// → Point(certainty=0.89)

// Evaluate therapeutic intervention
resolution outcome = evaluate_drug_efficacy(sertraline)
// → ResolutionOutcome(confidence=0.82)

// Select next thought (BMD)
item thought = bmd_select_frame(current_h_plus_state)
// → "Check email" (categorical completion)
```

**This is "compiling consciousness":**
- Oscillatory states → Points
- Therapeutic claims → Resolutions  
- Thought selection → BMD frames
- All executed by Turbulance compiler!

## Summary

**You now have:**

✅ Working Turbulance compiler (Python)  
✅ Three paradigm demonstrations  
✅ Executable `.trb` examples  
✅ Understanding of consciousness programming  

**Run it:**

```bash
python turbulance.py simple_point.trb
```

**See consciousness compile in real-time!** 🧠⚡
