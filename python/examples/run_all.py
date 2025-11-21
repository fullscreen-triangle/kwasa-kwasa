#!/usr/bin/env python3
"""
Run all grounded examples in sequence
"""

import subprocess
import sys
import os

def run_example(script_name):
    """Run a single example script"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(__file__) or '.',
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {script_name} completed successfully\n")
            return True
        else:
            print(f"\n✗ {script_name} failed with return code {result.returncode}\n")
            return False
    except Exception as e:
        print(f"\n✗ {script_name} failed with exception: {e}\n")
        return False


def main():
    print("="*80)
    print("GROUNDED CONSCIOUSNESS PROGRAMMING EXAMPLES")
    print("Executable implementations of theoretical frameworks")
    print("="*80)
    print()
    print("These examples map directly to:")
    print("  - hybrid-meta-language-pharmacodynamics.tex")
    print("  - metabolic-hierarchy-computing.tex")
    print("  - kuramoto-oscillator-phase-computing.tex")
    print()
    
    examples = [
        "01_cheminformatics_basics.py",
        "02_kuramoto_phase_synchronization.py",
        "03_drug_oxygen_aggregation.py",
        "04_metabolic_hierarchy.py",
    ]
    
    results = []
    for example in examples:
        success = run_example(example)
        results.append((example, success))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    for example, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8} {example}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print()
    print(f"Total: {passed}/{total} examples passed")
    
    if passed == total:
        print()
        print("="*80)
        print("✓✓✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        print()
        print("Key achievements:")
        print("  1. ✓ Calculated molecular properties (K_agg, frequencies)")
        print("  2. ✓ Simulated Kuramoto phase synchronization")
        print("  3. ✓ Analyzed drug-O₂ aggregation and EM coupling")
        print("  4. ✓ Computed metabolic hierarchy flux cascades")
        print()
        print("All theoretical frameworks are EXECUTABLE!")
        print("All predictions are TESTABLE!")
        print()
        print("Next step: Express these in Turbulance syntax")
        return 0
    else:
        print()
        print(f"✗ {total - passed} examples failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

