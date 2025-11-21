#!/usr/bin/env python3
"""
Example 1: Basic Cheminformatics
Ground zero for consciousness programming - calculate molecular properties
"""

from turbulance_consciousness import ConsciousnessProgramming
import numpy as np


def calculate_molecular_properties(smiles: str, name: str):
    """Calculate basic molecular properties from SMILES"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"SMILES: {smiles}")
    print(f"{'='*60}")
    
    # Parse SMILES (simplified - in production would use RDKit)
    atom_count = smiles.count('C') + smiles.count('N') + smiles.count('O')
    aromatic_rings = smiles.count('c')  # lowercase = aromatic
    
    # Estimate molecular weight (simplified)
    mw = atom_count * 12  # rough estimate
    
    # Estimate LogP (lipophilicity)
    # More carbons = more lipophilic
    # More N/O = more hydrophilic
    hydrophobic = smiles.count('C') + smiles.count('c')
    hydrophilic = smiles.count('N') + smiles.count('O')
    logP = 0.5 * hydrophobic - 0.3 * hydrophilic
    
    # Estimate O2 aggregation constant (from paper: K_agg > 10^4 M^-1 is therapeutic)
    # Aromatic rings and heteroatoms increase aggregation
    K_agg = 10**2 * (1 + aromatic_rings) * (1 + hydrophilic * 0.5)
    
    # Estimate vibrational frequency (THz range from paper)
    # Lighter molecules vibrate faster
    freq = 1e13 / (mw / 100)**0.5
    
    print(f"\nMolecular Properties:")
    print(f"  Atom count: {atom_count}")
    print(f"  Molecular weight: {mw:.1f} g/mol")
    print(f"  LogP (lipophilicity): {logP:.2f}")
    print(f"  Aromatic rings: {aromatic_rings}")
    print(f"  ")
    print(f"Consciousness Programming Properties:")
    print(f"  O₂ aggregation constant (K_agg): {K_agg:.2e} M⁻¹")
    print(f"  Therapeutic threshold: 10^4 M⁻¹")
    print(f"  Status: {'✓ EXCEEDS' if K_agg > 1e4 else '✗ BELOW'} threshold")
    print(f"  Vibrational frequency: {freq:.2e} Hz ({freq/1e12:.1f} THz)")
    print(f"  O₂ frequency match: {'✓ YES' if 0.5e13 < freq < 5e13 else '✗ NO'}")
    
    return {
        'smiles': smiles,
        'name': name,
        'mw': mw,
        'logP': logP,
        'K_agg': K_agg,
        'frequency': freq,
        'therapeutic': K_agg > 1e4
    }


def main():
    print("="*60)
    print("EXAMPLE 1: BASIC CHEMINFORMATICS")
    print("Grounding consciousness programming in molecular properties")
    print("="*60)
    print()
    print("From papers: Drug-O₂ aggregation (K_agg > 10^4 M⁻¹) enables")
    print("phase-lock programming. Vibrational frequencies ~10^13 Hz")
    print("match O₂ molecular vibrations for resonance coupling.")
    print()
    
    # Test molecules from the papers
    molecules = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N", "Tryptophan (serotonin precursor)"),
        ("c1ccc2c(c1)c(cn2)CCN", "Tryptamine (simpler)"),
        ("CC(C)NCC(COc1ccccc1)O", "Propranolol (beta-blocker)"),
    ]
    
    results = []
    for smiles, name in molecules:
        result = calculate_molecular_properties(smiles, name)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Therapeutic Potential")
    print(f"{'='*60}")
    print()
    print(f"{'Molecule':<30} {'K_agg (M⁻¹)':<15} {'Therapeutic?':<15}")
    print("-"*60)
    for r in results:
        status = "✓ YES" if r['therapeutic'] else "✗ NO"
        print(f"{r['name']:<30} {r['K_agg']:<15.2e} {status:<15}")
    
    print()
    print("KEY INSIGHT: O₂ aggregation constant determines if a molecule")
    print("can modulate phase-lock coupling → consciousness programming.")
    print()
    print("Next: Example 2 - Kuramoto oscillator dynamics")


if __name__ == '__main__':
    main()

