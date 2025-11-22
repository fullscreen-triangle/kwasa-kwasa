## Turbulance Quick Reference

### Run Any Example

```bash
cd python
python turbulance.py ../examples/turbulance/FILENAME.trb
```

### Save Output

```bash
python turbulance.py FILENAME.trb --save-output
```

### 6 Examples (Simple → Advanced)

#### **Basic (2-3 min each)**
```bash
python turbulance.py ../examples/turbulance/simple_point.trb
python turbulance.py ../examples/turbulance/simple_resolution.trb
python turbulance.py ../examples/turbulance/simple_bmd.trb
```

#### **Advanced (5-10 min each)**
```bash
python turbulance.py ../examples/turbulance/01_point_demo_executable.trb
python turbulance.py ../examples/turbulance/02_resolution_demo_executable.trb
python turbulance.py ../examples/turbulance/03_bmd_demo_executable.trb
```

### Run All + Save

```bash
test_all_examples.bat  # Windows
```

### Check Saved Outputs

```bash
dir validation_outputs\*_latest.txt  # Windows
ls validation_outputs/*_latest.txt   # Linux/Mac
```

---

## What Each Example Shows

| File | Paradigm | Shows | Lines |
|------|----------|-------|-------|
| `simple_point.trb` | Points | Uncertain measurements | 30 |
| `simple_resolution.trb` | Resolutions | Debate outcome | 45 |
| `simple_bmd.trb` | BMDs | Frame selection | 40 |
| `01_point_demo_executable.trb` | Points | **Full math** | 120 |
| `02_resolution_demo_executable.trb` | Resolutions | **Bayesian calc** | 150 |
| `03_bmd_demo_executable.trb` | BMDs | **All frames** | 170 |

---

## Key Paradigms

### 1. Points = Uncertain Measurements
```turbulance
Measuring H+ field coherence...
  Value: 0.67
  Certainty: 0.89  ← Explicit uncertainty!
```

### 2. Resolutions = Probabilistic Debates
```turbulance
AFFIRMATIONS (for):
  • K_agg exceeds threshold (0.92)
CONTENTIONS (against):
  • Placebo effect exists (0.68)
  
OUTCOME: 79% confident → AFFIRMED
```

### 3. BMDs = Frame Selection
```turbulance
Frame 1: "Check email" (H+=0.82)
Frame 2: "Feel anxious" (H+=0.45)
Frame 3: "Think lunch" (H+=0.71) ← Selected!

Current H+ = 0.75 → Closest is Frame 3
```

---

## Compiler Features

✅ Execute `.trb` files  
✅ Save outputs for validation  
✅ Demonstrates 3 paradigms  
✅ Progressive complexity  

Current limitations:
- Simplified syntax only
- No complex expressions
- No full type system

But **it works** for demonstrating concepts!

---

## Files Created

After running with `--save-output`:

```
python/
└── validation_outputs/
    ├── simple_point_latest.txt
    ├── simple_resolution_latest.txt
    ├── simple_bmd_latest.txt
    ├── 01_point_demo_executable_latest.txt
    ├── 02_resolution_demo_executable_latest.txt
    └── 03_bmd_demo_executable_latest.txt
```

Use these to:
- Show stakeholders
- Document paradigms
- Validate correctness
- Regression testing

---

## One-Liners

**Test everything:**
```bash
cd python && test_all_examples.bat
```

**View latest Point output:**
```bash
cat validation_outputs/simple_point_latest.txt
```

**Run and save in one:**
```bash
python turbulance.py simple_point.trb --save-output && cat validation_outputs/simple_point_latest.txt
```

---

🧠⚡ **That's it! Consciousness compiled!**

