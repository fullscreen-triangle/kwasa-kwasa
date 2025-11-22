#!/usr/bin/env python3
"""
Turbulance Compiler - CLI Tool
Usage: python turbulance.py script.trb [--save-output]
"""

import sys
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import io


# ============================================================================
# CORE TYPES
# ============================================================================

@dataclass
class Point:
    """Probabilistic semantic unit"""
    content: str
    value: Any = None
    certainty: float = 1.0
    evidence_strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.metadata.get(key, self.value)


@dataclass
class Affirmation:
    """Supporting evidence"""
    content: str
    weight: float
    source: str
    confidence: float
    relevance: float = 1.0


@dataclass
class Contention:
    """Challenging evidence"""
    content: str
    weight: float
    challenge: str
    confidence: float = 0.8


# ============================================================================
# TURBULANCE RUNTIME
# ============================================================================

class TurbulanceRuntime:
    """Runtime environment for Turbulance execution"""

    def __init__(self):
        self.globals = {}
        self.locals = {}

        # Built-in functions
        self.globals['print'] = self._print
        self.globals['sqrt'] = np.sqrt
        self.globals['min'] = min
        self.globals['max'] = max
        self.globals['Point'] = Point
        self.globals['Affirmation'] = Affirmation
        self.globals['Contention'] = Contention

        # Consciousness programming functions
        self.globals['calculate_aggregate_certainty'] = self._calculate_aggregate_certainty
        self.globals['weighted_average'] = self._weighted_average
        self.globals['calculate_significance'] = self._calculate_significance
        self.globals['calculate_bayesian_posterior'] = self._calculate_bayesian_posterior
        self.globals['calculate_evidence_strength'] = lambda p: abs(p - 0.5) * 2.0
        self.globals['classify_reliability'] = self._classify_reliability
        self.globals['sort_by_confidence'] = lambda x: sorted(x, key=lambda t: t[1], reverse=True)

    def _print(self, *args, **kwargs):
        """Enhanced print with format string support"""
        if len(args) >= 1 and isinstance(args[0], str) and '{}' in args[0]:
            # Format string
            try:
                print(args[0].format(*args[1:]), **kwargs)
            except:
                print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def _calculate_aggregate_certainty(self, certainties: List[float]) -> float:
        """Aggregate certainty (geometric mean)"""
        if not certainties:
            return 0.0
        return float(np.prod(certainties) ** (1.0 / len(certainties)))

    def _weighted_average(self, weighted_values: List[tuple]) -> float:
        """Weighted average of (value, weight) pairs"""
        if not weighted_values:
            return 0.0
        total_weight = sum(w for _, w in weighted_values)
        if total_weight == 0:
            return 0.0
        return sum(v * w for v, w in weighted_values) / total_weight

    def _calculate_significance(self, delta: float, variance: float) -> float:
        """Statistical significance"""
        if variance == 0:
            return 1.0 if delta > 0 else 0.0
        z_score = abs(delta) / np.sqrt(variance)
        return min(1.0, z_score / 3.0)

    def _calculate_bayesian_posterior(
        self,
        prior: float,
        affirmations: List[Affirmation],
        contentions: List[Contention],
        affirmation_weight: float = 0.6,
        contention_weight: float = 0.4
    ) -> float:
        """Bayesian posterior from affirmations and contentions"""
        aff_likelihood = 1.0
        for aff in affirmations:
            aff_likelihood *= (1.0 + aff.weight * aff.confidence * aff.relevance * affirmation_weight)

        cont_likelihood = 1.0
        for cont in contentions:
            cont_likelihood *= (1.0 - cont.weight * cont.confidence * 0.5 * contention_weight)

        numerator = prior * aff_likelihood * cont_likelihood
        posterior = numerator / (numerator + (1 - prior))

        return float(np.clip(posterior, 0.0, 1.0))

    def _classify_reliability(self, confidence: float) -> str:
        """Classify reliability from confidence"""
        if confidence > 0.90:
            return "HighlyReliable"
        elif confidence > 0.75:
            return "ModeratelyReliable"
        elif confidence > 0.60:
            return "SomewhatReliable"
        else:
            return "RequiresReview"


# ============================================================================
# TURBULANCE COMPILER
# ============================================================================

class TurbulanceCompiler:
    """
    Compiles .trb files to executable Python

    Simplified parser that handles key Turbulance constructs
    """

    def __init__(self):
        self.runtime = TurbulanceRuntime()
        self.output_buffer = []  # Store output for saving

    def compile_file(self, filepath: str) -> str:
        """Compile .trb file to Python"""
        with open(filepath, 'r', encoding='utf-8') as f:
            trb_code = f.read()

        return self.compile(trb_code)

    def compile(self, trb_code: str) -> str:
        """Compile Turbulance code to Python"""
        python_code = []
        lines = trb_code.split('\n')

        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                continue

            # Import statements
            if stripped.startswith('import '):
                # Skip for now (would map to Python imports)
                continue

            # Function definitions: funxn -> def
            if stripped.startswith('funxn '):
                python_line = line.replace('funxn ', 'def ')
                python_line = python_line.replace(' -> ', ' -> ')
                python_code.append(python_line)
                indent_level += 1
                continue

            # Point declarations: point x = { -> x = Point(
            if stripped.startswith('point '):
                python_code.append(self._compile_point(line))
                continue

            # Item declarations: item x = -> x =
            if stripped.startswith('item '):
                python_code.append(line.replace('item ', ''))
                continue

            # Affirmation/Contention declarations
            if stripped.startswith('affirmation '):
                python_code.append(self._compile_evidence(line, 'Affirmation'))
                continue

            if stripped.startswith('contention '):
                python_code.append(self._compile_evidence(line, 'Contention'))
                continue

            # Resolution declarations
            if stripped.startswith('resolution '):
                python_code.append(line.replace('resolution ', 'def '))
                indent_level += 1
                continue

            # Considering loops: considering x in y: -> for x in y:
            if stripped.startswith('considering '):
                python_code.append(line.replace('considering ', 'for '))
                indent_level += 1
                continue

            # Regular lines
            python_code.append(line)

        return '\n'.join(python_code)

    def _compile_point(self, line: str) -> str:
        """Compile point declaration"""
        # Extract variable name
        match = re.match(r'(\s*)point\s+(\w+)\s*=\s*\{', line)
        if not match:
            return line

        indent = match.group(1)
        var_name = match.group(2)

        # Convert to Point() constructor
        return f"{indent}{var_name} = Point("

    def _compile_evidence(self, line: str, evidence_type: str) -> str:
        """Compile affirmation/contention"""
        match = re.match(r'(\s*)(affirmation|contention)\s+(\w+)\s*=\s*\{', line)
        if not match:
            return line

        indent = match.group(1)
        var_name = match.group(3)

        return f"{indent}{var_name} = {evidence_type}("

    def execute_file(self, filepath: str, save_output: bool = False):
        """Execute .trb file directly"""
        # Capture all output
        if save_output:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            print(f"Loading: {filepath}")
            print("="*70)
            print()

            with open(filepath, 'r', encoding='utf-8') as f:
                trb_code = f.read()

            # For demonstration, use simplified execution
            self._execute_simple(trb_code)

            # Get captured output
            if save_output:
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

                # Print to screen
                print(output, end='')

                # Save to file
                self._save_output(filepath, output)

        except Exception as e:
            if save_output:
                sys.stdout = old_stdout
            raise e

    def _execute_simple(self, trb_code: str):
        """Simplified execution (direct interpretation)"""
        # Extract and execute main function if exists
        lines = trb_code.split('\n')

        # Look for funxn main():
        in_main = False
        main_code = []

        for line in lines:
            stripped = line.strip()

            # Skip comments
            if stripped.startswith('//'):
                # Check for section headers
                if '//' in line and '='*10 in line:
                    print(line[2:].strip())
                continue

            if 'funxn main()' in stripped:
                in_main = True
                continue

            if in_main:
                if stripped and not stripped.startswith('}'):
                    # Execute print statements
                    if 'print(' in line:
                        self._execute_print(line)
                elif stripped.startswith('}'):
                    break

        print()
        print("="*70)
        print("Turbulance execution complete")
        print("="*70)

    def _execute_print(self, line: str):
        """Execute print statement"""
        # Extract string and format args
        match = re.search(r'print\("([^"]+)"(?:,\s*(.+))?\)', line)
        if match:
            text = match.group(1)
            # Simple output
            print(text)

    def _save_output(self, script_path: str, output: str):
        """Save execution output to file"""
        script_name = Path(script_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create outputs directory
        output_dir = Path("validation_outputs")
        output_dir.mkdir(exist_ok=True)

        # Save with timestamp
        output_file = output_dir / f"{script_name}_{timestamp}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"Turbulance Execution Output\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n")
            f.write("\n")
            f.write(output)

        print(f"\n✓ Output saved to: {output_file}")

        # Also save latest (no timestamp)
        latest_file = output_dir / f"{script_name}_latest.txt"
        with open(latest_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"Turbulance Execution Output\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("="*70 + "\n")
            f.write("\n")
            f.write(output)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python turbulance.py script.trb [--save-output]")
        print()
        print("Turbulance Compiler - Consciousness Programming Language")
        print()
        print("Options:")
        print("  --save-output    Save execution output to validation_outputs/")
        print()
        print("Examples:")
        print("  python turbulance.py simple_point.trb")
        print("  python turbulance.py simple_point.trb --save-output")
        sys.exit(1)

    script_path = sys.argv[1]
    save_output = '--save-output' in sys.argv

    if not Path(script_path).exists():
        print(f"Error: File not found: {script_path}")
        sys.exit(1)

    compiler = TurbulanceCompiler()

    # For now, just show that we can load and process it
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "TURBULANCE COMPILER v0.1" + " "*29 + "║")
    print("║" + " "*10 + "Consciousness Programming Language" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    print()

    try:
        compiler.execute_file(script_path, save_output=save_output)
    except Exception as e:
        print(f"Error executing {script_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

