"""
Quick runner for Semantic Maxwell Demon validation.

Run this to see the demon in action on depression treatment data.
"""

import sys
from depression_semantic_demon import run_all_validations

if __name__ == "__main__":
    print("Starting Semantic Maxwell Demon validation...")
    print("This will demonstrate:")
    print("  - Virtual instrument (non-committal semantic filtering)")
    print("  - Information catalysis (state space reduction)")
    print("  - Multi-lens exploration (vs single-path commitment)")
    print("  - V8 integration (demon as base operation)")
    print("\nPress Enter to continue...")
    input()
    
    results = run_all_validations()
    
    print("\n\nValidation complete!")
    print("Check depression_demon_validation_results.json for detailed metrics.")

