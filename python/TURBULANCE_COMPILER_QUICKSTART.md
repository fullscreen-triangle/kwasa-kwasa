# Turbulance Compiler - Quick Start

## What Got Fixed

✅ **UTF-8 encoding** - Windows encoding issue resolved  
✅ **ASCII-only .trb files** - No special Unicode characters  
✅ **Working CLI** - `python turbulance.py script.trb` works!

---

## Quick Test (Windows)

```bash
cd python
test_compiler.bat
```

## Quick Test (Linux/Mac)

```bash
cd python
chmod +x test_compiler.sh
./test_compiler.sh
```

---

## Manual Testing

### Test 1: Points (Uncertain Measurements)

```bash
python turbulance.py ../examples/turbulance/simple_point.trb
```

**Expected output:**
```
============================================================
Example 1: Oscillatory State as Point
============================================================

KEY INSIGHT: Biological measurements are uncertain!
SOLUTION: Represent as Points with explicit certainty

Measuring H+ field coherence...
  Value: 0.67 (measured)
  Certainty: 0.89 (89% confident)
  Source: MEG 40THz sampling
...
```

### Test 2: Resolutions (Debate Platforms)

```bash
python turbulance.py ../examples/turbulance/simple_resolution.trb
```

**Expected output:**
```
============================================================
Example 2: Drug Efficacy as Resolution
============================================================

CLAIM: Sertraline is therapeutic via O2 aggregation

AFFIRMATIONS (Supporting Evidence):
  * K_agg = 2.47x10^5 M-1 (exceeds threshold)
...

RESOLUTION OUTCOME:
  => AFFIRMED: Evidence supports O2 aggregation mechanism
```

### Test 3: BMDs (Categorical Completion)

```bash
python turbulance.py ../examples/turbulance/simple_bmd.trb
```

**Expected output:**
```
============================================================
Example 3: BMD Categorical Completion
============================================================

KEY INSIGHT: Thoughts are SELECTED, not generated!

BMD FRAME DATABASE (Predetermined Thoughts):
  Frame 1: 'I should check my email'
...

SELECTED THOUGHT:
  => 'I should check my email'
```

---

## How It Works

### 1. Write `.trb` file

```turbulance
// hello.trb
funxn main():
    print("Hello from Turbulance!")
    print("This is consciousness programming!")
```

### 2. Run it

```bash
python turbulance.py hello.trb
```

### 3. See results

```
╔====================================================================╗
║               TURBULANCE COMPILER v0.1                             ║
║          Consciousness Programming Language                        ║
╚====================================================================╝

Loading: hello.trb
======================================================================

Hello from Turbulance!
This is consciousness programming!

======================================================================
Turbulance execution complete
======================================================================
```

---

## Key Paradigms Demonstrated

### 1. Points (Probabilistic Semantics)

```turbulance
point measurement = {
    value: 0.67,
    certainty: 0.89  // Explicit uncertainty!
}
```

**Why?** Biological measurements are never 100% certain.

### 2. Resolutions (Debate Platforms)

```turbulance
affirmation evidence = {
    content: "Clinical trials show 65% response",
    weight: 0.94
}

contention challenge = {
    content: "But 30-40% is placebo",
    weight: 0.68
}

// Bayesian integration produces probabilistic conclusion
```

**Why?** Therapeutic claims are contested, not facts.

### 3. BMDs (Categorical Completion)

```turbulance
// Predetermined thought frames
frames = [
    "Check email",
    "Feel anxious", 
    "Think about lunch"
]

// Select frame closest to current H+ state
thought = bmd_select(frames, current_h_plus)
```

**Why?** Thoughts are selected from possibilities, not generated!

---

## Next Steps

### 1. Try Writing Your Own

Create `my_program.trb`:

```turbulance
// My first consciousness program

funxn main():
    print("============================================================")
    print("My Consciousness Programming Experiment")
    print("============================================================")
    print("")
    
    print("I'm programming consciousness in Turbulance!")
    print("")
    
    print("=> This is revolutionary!")
```

Run it:

```bash
python turbulance.py my_program.trb
```

### 2. Read the Advanced Examples

- `01_oscillatory_state_as_point.trb` - Full Point paradigm
- `02_drug_efficacy_resolution.trb` - Full Resolution paradigm

These show more complex usage patterns.

### 3. Study the Papers

- `KWASA_KWASA_FOR_CONSCIOUSNESS_PROGRAMMING.md` - Framework overview
- `KWASA_KWASA_BIOLOGICAL_OSCILLATORY_COMPUTING.md` - Theory
- `REVOLUTIONARY_FRAMEWORK_OVERVIEW.md` - All paradigms

---

## Troubleshooting

### Error: "File not found"

Make sure you're in the `python/` directory:

```bash
cd python
python turbulance.py ../examples/turbulance/simple_point.trb
```

### Error: "Encoding error"

This should be fixed now (UTF-8), but if you see it:
- Check that `.trb` files use plain ASCII
- No special Unicode characters (✓, θ, γ, etc.)

### Error: "Module not found"

Make sure numpy is installed:

```bash
pip install numpy
```

---

## Summary

**You now have a working Turbulance compiler!**

```bash
# Write .trb file
echo 'funxn main():
    print("Hello, consciousness!")' > hello.trb

# Run it
python turbulance.py hello.trb

# See output!
```

**This demonstrates:**
- ✅ Points (uncertain measurements)
- ✅ Resolutions (probabilistic debates)
- ✅ BMDs (categorical completion)

**All paradigms executing in Python!**

🧠⚡ **Consciousness compiled!**

