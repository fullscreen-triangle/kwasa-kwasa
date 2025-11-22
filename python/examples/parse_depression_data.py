# parse_depression_treatment_data.py
"""
Parse actual depression treatment files to extract data for visualization
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any

class DepressionTreatmentParser:
    """Parser for Turbulance depression treatment files"""

    def __init__(self, base_path: str = './'):
        self.base_path = Path(base_path)
        self.data = {}

    def parse_trb_file(self, filename: str = 'depression_treatment.trb') -> Dict:
        """Parse .trb protocol specification"""
        filepath = self.base_path / filename

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract imports
        imports = re.findall(r'import\s+([\w.]+)', content)

        # Extract semantic hypothesis
        hypothesis_match = re.search(r'SEMANTIC HYPOTHESIS.*?Framework for understanding:(.*?)(?=//|\n\n)',
                                     content, re.DOTALL)
        hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else None

        # Extract consciousness parameters
        params = {
            'h_plus_frequency': self._extract_frequency(content, 'H+'),
            'o2_states': self._extract_number(content, 'O2.*states'),
            'theta_band': self._extract_range(content, 'theta'),
            'gamma_band': self._extract_range(content, 'gamma')
        }

        self.data['trb'] = {
            'imports': imports,
            'hypothesis': hypothesis,
            'parameters': params
        }

        return self.data['trb']

    def parse_fs_file(self, filename: str = 'depression_treatment.fs.fs') -> Dict:
        """Parse .fs flux state monitoring"""
        filepath = self.base_path / filename

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract data inputs
        data_inputs = re.findall(r'├── (\w+)\s*\((.*?)\)', content)

        # Extract V8 module states
        v8_states = {}
        v8_modules = ['mzekezeke', 'zengeza', 'diggiden', 'spectacular',
                      'hatata', 'champagne', 'nicotine', 'pungwe']

        for module in v8_modules:
            pattern = f'{module}.*?status:\s*(\w+)'
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                v8_states[module] = match.group(1)

        # Extract consciousness metrics
        metrics = {
            'psi_0': self._extract_float(content, r'Ψ₀.*?(\d+\.\d+)'),
            'theta_0': self._extract_float(content, r'Θ₀.*?(\d+\.\d+)'),
            'plv': self._extract_float(content, r'PLV.*?(\d+\.\d+)'),
            'consciousness_quality': self._extract_float(content, r'consciousness.*quality.*?(\d+\.\d+)')
        }

        self.data['fs'] = {
            'data_inputs': dict(data_inputs),
            'v8_states': v8_states,
            'metrics': metrics
        }

        return self.data['fs']

    def parse_ghd_file(self, filename: str = 'depression_treatment.ghd') -> Dict:
        """Parse .ghd dependency specification"""
        filepath = self.base_path / filename

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract external databases
        databases = re.findall(r'-\s+(\w+):\s*"([^"]+)"', content)

        # Extract computational resources
        resources = {
            'cpu_cores': self._extract_number(content, r'cpu_cores:\s*(\d+)'),
            'gpu_devices': self._extract_number(content, r'gpu.*?(\d+)'),
            'memory_gb': self._extract_number(content, r'memory.*?(\d+)')
        }

        self.data['ghd'] = {
            'databases': dict(databases),
            'resources': resources
        }

        return self.data['ghd']

    def parse_hre_file(self, filename: str = 'depression_treatment.hre') -> Dict:
        """Parse .hre decision log"""
        filepath = self.base_path / filename

        with open(filepath, 'r') as f:
            content = f.read()

        # Extract scientific hypothesis
        hypothesis_match = re.search(r'scientific_hypothesis:\s*"([^"]+)"', content)
        hypothesis = hypothesis_match.group(1) if hypothesis_match else None

        # Extract decision timeline
        decisions = []
        decision_blocks = re.findall(r'(\w+_phase):.*?timestamp:\s*"([^"]+)".*?decision:\s*"([^"]+)".*?reasoning:\s*"([^"]+)"',
                                     content, re.DOTALL)

        for phase, timestamp, decision, reasoning in decision_blocks:
            decisions.append({
                'phase': phase,
                'timestamp': timestamp,
                'decision': decision,
                'reasoning': reasoning
            })

        # Extract metacognitive insights
        insights = re.findall(r'insight:\s*"([^"]+)"', content)

        self.data['hre'] = {
            'hypothesis': hypothesis,
            'decisions': decisions,
            'insights': insights
        }

        return self.data['hre']

    def parse_all(self) -> Dict:
        """Parse all four files"""
        self.parse_trb_file()
        self.parse_fs_file()
        self.parse_ghd_file()
        self.parse_hre_file()
        return self.data

    def export_json(self, output_file: str = 'depression_treatment_data.json'):
        """Export parsed data to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Exported data to {output_file}")

    # Helper methods
    def _extract_frequency(self, text: str, pattern: str) -> float:
        match = re.search(f'{pattern}.*?(\d+\.?\d*)\s*(THz|GHz|MHz)', text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit == 'thz':
                return value * 1e12
            elif unit == 'ghz':
                return value * 1e9
            elif unit == 'mhz':
                return value * 1e6
        return None

    def _extract_number(self, text: str, pattern: str) -> int:
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_float(self, text: str, pattern: str) -> float:
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def _extract_range(self, text: str, pattern: str) -> tuple:
        match = re.search(f'{pattern}.*?(\d+)-(\d+)\s*Hz', text, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None

# Usage
if __name__ == '__main__':
    parser = DepressionTreatmentParser()
    data = parser.parse_all()
    parser.export_json()

    print("\n=== PARSED DATA ===")
    print(json.dumps(data, indent=2))
