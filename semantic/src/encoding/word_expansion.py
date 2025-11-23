"""Word expansion for semantic richness"""

from typing import List

class WordExpander:
    """Expand words into semantic variations"""

    def expand(self, word: str, depth: int = 3) -> List[str]:
        """Expand word into semantic neighborhood"""
        expansions = [word]

        # Add simple variations (in production: use word embeddings)
        expansions.append(f"{word}_concept")
        expansions.append(f"{word}_meaning")
        expansions.append(f"related_to_{word}")

        return expansions[:depth]

    def semantic_distance(self, word1: str, word2: str) -> float:
        """Compute semantic distance between words"""
        # Simple implementation (in production: use embeddings)
        if word1 == word2:
            return 0.0

        # Levenshtein-like distance
        return len(set(word1) ^ set(word2)) / max(len(word1), len(word2))

