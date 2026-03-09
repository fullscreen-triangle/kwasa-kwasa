# parse_turbulance_files.py
"""
Parse Turbulance depression treatment files and extract visualization data
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any

class TurbulanceFileParser:
    """Parse all four Turbulance file types"""
    
    def __init__(self, public_dir: str = './public'):
        self.public_dir = Path(public_dir)
        self.data = {}
    
    def parse_trb(self, filename: str = 'depression_treatment.trb') -> Dict:
        """Parse .trb protocol file"""
        filepath = self.public_dir / filename
        print(f"Parsing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract imports
        imports = re.findall(r'import\s+([\w.]+)', content)
        
        # Extract comments/documentation
        doc_lines = re.findall(r'^//\s*(.+)$', content, re.MULTILINE)
        
        # Extract consciousness parameters
        h_plus_freq = re.search(r'(\d+)\s*THz.*?H\+', content, re.IGNORECASE)
        theta_band = re.search(r'theta.*?(\d+)-(\d+)\s*Hz', content, re.IGNORECASE)
        gamma_band = re.search(r'gamma.*?(\d+)-(\d+)\s*Hz', content, re.IGNORECASE)
        
        self.data['trb'] = {
            'filename': filename,
            'imports': imports,
            'documentation': doc_lines[:5],  # First 5 doc lines
            'consciousness_params': {
                'h_plus_frequency_thz': float(h_plus_freq.group(1)) if h_plus_freq else None,
                'theta_band_hz': (int(theta_band.group(1)), int(theta_band.group(2))) if theta_band else None,
                'gamma_band_hz': (int(gamma_band.group(1)), int(gamma_band.group(2))) if gamma_band else None
            }
        }
        
        print(f"  ✓ Found {len(imports)} imports")
        print(f"  ✓ Extracted consciousness parameters")
        
        return self.data['trb']
    
    def parse_fs(self, filename: str = 'depression_treatment.fs.fs') -> Dict:
        """Parse .fs flux state file"""
        filepath = self.public_dir / filename
        print(f"\nParsing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract data inputs
        data_inputs = {}
        input_matches = re.findall(r'│\s+├──\s+(\w+)\s*\(([^)]+)\)', content)
        for name, description in input_matches:
            data_inputs[name] = description
        
        # Extract V8 module states
        v8_modules = {}
        module_names = ['mzekezeke', 'zengeza', 'diggiden', 'spectacular', 
                       'hatata', 'champagne', 'nicotine', 'pungwe']
        
        for module in module_names:
            # Look for module status
            status_match = re.search(f'{module}.*?status:\s*(\w+)', content, re.IGNORECASE | re.DOTALL)
            if status_match:
                v8_modules[module] = {'status': status_match.group(1)}
                
                # Look for metrics
                metrics_section = re.search(f'{module}(.*?)(?={"|".join(module_names)}|consciousness_metrics)', 
                                          content, re.IGNORECASE | re.DOTALL)
                if metrics_section:
                    metrics_text = metrics_section.group(1)
                    # Extract numeric metrics
                    numbers = re.findall(r'(\w+):\s*(\d+\.?\d*)', metrics_text)
                    v8_modules[module]['metrics'] = dict(numbers)
        
        # Extract consciousness metrics
        consciousness_metrics = {}
        metric_patterns = [
            (r'Ψ₀.*?(\d+\.\d+)', 'psi_0'),
            (r'Θ₀.*?(\d+\.\d+)', 'theta_0'),
            (r'PLV.*?(\d+\.\d+)', 'plv'),
            (r'consciousness.*quality.*?(\d+\.\d+)', 'consciousness_quality'),
            (r'hierarchical.*depth.*?(\d+\.\d+)', 'hierarchical_depth')
        ]
        
        for pattern, key in metric_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                consciousness_metrics[key] = float(match.group(1))
        
        self.data['fs'] = {
            'filename': filename,
            'data_inputs': data_inputs,
            'v8_modules': v8_modules,
            'consciousness_metrics': consciousness_metrics
        }
        
        print(f"  ✓ Found {len(data_inputs)} data inputs")
        print(f"  ✓ Found {len(v8_modules)} V8 modules")
        print(f"  ✓ Extracted {len(consciousness_metrics)} consciousness metrics")
        
        return self.data['fs']
    
    def parse_ghd(self, filename: str = 'depression_treatment.ghd') -> Dict:
        """Parse .ghd dependency file"""
        filepath = self.public_dir / filename
        print(f"\nParsing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract external databases
        databases = {}
        db_matches = re.findall(r'-\s+(\w+):\s*"([^"]+)"', content)
        for name, url in db_matches:
            databases[name] = url
        
        # Extract computational resources
        resources = {}
        resource_patterns = [
            (r'cpu.*?cores?:\s*(\d+)', 'cpu_cores'),
            (r'gpu.*?devices?:\s*(\d+)', 'gpu_devices'),
            (r'memory.*?(\d+)', 'memory_gb'),
            (r'storage.*?(\d+)', 'storage_gb')
        ]
        
        for pattern, key in resource_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                resources[key] = int(match.group(1))
        
        self.data['ghd'] = {
            'filename': filename,
            'databases': databases,
            'resources': resources
        }
        
        print(f"  ✓ Found {len(databases)} external databases")
        print(f"  ✓ Extracted {len(resources)} resource specifications")
        
        return self.data['ghd']
    
    def parse_hre(self, filename: str = 'depression_treatment.hre') -> Dict:
        """Parse .hre decision log file"""
        filepath = self.public_dir / filename
        print(f"\nParsing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract orchestrator session
        session_match = re.search(r'orchestrator_session:\s*"([^"]+)"', content)
        session = session_match.group(1) if session_match else None
        
        # Extract scientific hypothesis
        hypothesis_match = re.search(r'scientific_hypothesis:\s*"([^"]+)"', content)
        hypothesis = hypothesis_match.group(1) if hypothesis_match else None
        
        # Extract decision phases
        decisions = []
        phase_pattern = r'(\w+_phase):.*?timestamp:\s*"([^"]+)".*?decision:\s*"([^"]+)".*?reasoning:\s*"([^"]+)"'
        phase_matches = re.findall(phase_pattern, content, re.DOTALL)
        
        for phase, timestamp, decision, reasoning in phase_matches:
            decisions.append({
                'phase': phase,
                'timestamp': timestamp,
                'decision': decision,
                'reasoning': reasoning[:200]  # Truncate long reasoning
            })
        
        # Extract metacognitive insights
        insights = re.findall(r'insight:\s*"([^"]+)"', content)
        
        # Extract key learnings
        learnings = re.findall(r'learning:\s*"([^"]+)"', content)
        
        self.data['hre'] = {
            'filename': filename,
            'session': session,
            'hypothesis': hypothesis,
            'decisions': decisions,
            'insights': insights,
            'learnings': learnings
        }
        
        print(f"  ✓ Found session: {session}")
        print(f"  ✓ Extracted {len(decisions)} decision phases")
        print(f"  ✓ Found {len(insights)} metacognitive insights")
        
        return self.data['hre']
    
    def parse_all(self) -> Dict:
        """Parse all four files"""
        print("=" * 70)
        print("PARSING TURBULANCE DEPRESSION TREATMENT FILES")
        print("=" * 70)
        
        self.parse_trb()
        self.parse_fs()
        self.parse_ghd()
        self.parse_hre()
        
        print("\n" + "=" * 70)
        print("PARSING COMPLETE")
        print("=" * 70)
        
        return self.data
    
    def export_json(self, output_file: str = 'depression_treatment_parsed.json'):
        """Export parsed data to JSON"""
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Exported parsed data to: {output_path.absolute()}")
        return output_path
    
    def print_summary(self):
        """Print a summary of parsed data"""
        print("\n" + "=" * 70)
        print("SUMMARY OF PARSED DATA")
        print("=" * 70)
        
        if 'trb' in self.data:
            print(f"\n.trb Protocol:")
            print(f"  - {len(self.data['trb']['imports'])} imports")
            print(f"  - H+ frequency: {self.data['trb']['consciousness_params']['h_plus_frequency_thz']} THz")
        
        if 'fs' in self.data:
            print(f"\n.fs Flux State:")
            print(f"  - {len(self.data['fs']['data_inputs'])} data inputs")
            print(f"  - {len(self.data['fs']['v8_modules'])} V8 modules active")
            if self.data['fs']['consciousness_metrics']:
                print(f"  - Consciousness metrics:")
                for key, value in self.data['fs']['consciousness_metrics'].items():
                    print(f"    • {key}: {value}")
        
        if 'ghd' in self.data:
            print(f"\n.ghd Dependencies:")
            print(f"  - {len(self.data['ghd']['databases'])} external databases")
            print(f"  - Computational resources:")
            for key, value in self.data['ghd']['resources'].items():
                print(f"    • {key}: {value}")
        
        if 'hre' in self.data:
            print(f"\n.hre Decision Log:")
            print(f"  - Session: {self.data['hre']['session']}")
            print(f"  - {len(self.data['hre']['decisions'])} decision phases")
            print(f"  - {len(self.data['hre']['insights'])} insights")
            print(f"  - Hypothesis: {self.data['hre']['hypothesis'][:100]}...")

# Run the parser
if __name__ == '__main__':
    parser = TurbulanceFileParser(public_dir='./public')
    data = parser.parse_all()
    parser.export_json()
    parser.print_summary()
