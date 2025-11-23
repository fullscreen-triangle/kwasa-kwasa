"""Semantic richness detection via compression"""

import numpy as np
from typing import Any
import zlib

class RichnessDetector:
    """Detect semantic richness through compression resistance"""
    
    @staticmethod
    def measure_richness(data: str) -> float:
        """Measure semantic richness (compression ratio)"""
        if not data:
            return 0.0
        
        # Compress data
        compressed = zlib.compress(data.encode())
        
        # Compression ratio: original / compressed
        # High ratio = compressible = low richness
        # Low ratio = incompressible = high richness
        original_size = len(data.encode())
        compressed_size = len(compressed)
        
        richness = 1.0 - (compressed_size / original_size)
        return richness
    
    @staticmethod
    def find_rich_regions(text: str, window_size: int = 100) -> list:
        """Find semantically rich regions in text"""
        rich_regions = []
        
        for i in range(0, len(text) - window_size, window_size // 2):
            window = text[i:i+window_size]
            richness = RichnessDetector.measure_richness(window)
            
            if richness > 0.5:  # Threshold for "rich"
                rich_regions.append((i, i+window_size, richness))
        
        return rich_regions

