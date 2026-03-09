# Validation Scripts - How to Run

## Quick Start - Run These in Order:

### 1. Basic Functionality Test
```bash
python validation/01_test_core_components.py
```
Tests that all modules load and basic operations work.

### 2. Depression Data Validation
```bash
python validation/02_validate_depression_data.py
```
Applies Semantic Maxwell Demon to your existing depression treatment data.
Shows before/after comparison.

### 3. Multi-Lens Comparison
```bash
python validation/03_multi_lens_analysis.py
```
Compares interpretations across different semantic lenses.

### 4. BMD State Detection Test
```bash
python validation/04_test_bmd_detection.py
```
Tests BMD state detection from simulated behavioral signals.

### 5. Thought Injection Test
```bash
python validation/05_test_thought_injection.py
```
Tests thought injection and sufficiency calculation.

### 6. Complete System Test
```bash
python validation/06_full_system_validation.py
```
End-to-end test of bidirectional interface.

## Expected Output

Each script will:
- Print clear status messages
- Show quantitative results
- Save output to `validation/results/`
- Generate comparison plots (if matplotlib available)

## What You're Validating

1. ✅ Core semantic filtering works
2. ✅ S-entropy calculations are correct
3. ✅ Multi-lens interpretations are meaningful
4. ✅ BMD state detection from behaviors
5. ✅ Thought injection generates sensible stimuli
6. ✅ Full system integrates properly

## Results Location

All results saved to: `validation/results/`
- `01_core_test_results.txt`
- `02_depression_validation.txt`
- `03_multi_lens_comparison.txt`
- etc.

