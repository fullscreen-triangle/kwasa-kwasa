"""Cardinal direction encoding for semantic navigation"""

from enum import Enum
import numpy as np

class Direction(Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UP = "up"
    DOWN = "down"

class CardinalEncoder:
    """Encode semantic concepts as cardinal directions"""

    def encode(self, concept: str) -> Direction:
        """Map concept to cardinal direction"""
        hash_val = hash(concept) % 6
        return list(Direction)[hash_val]

    def to_vector(self, direction: Direction) -> np.ndarray:
        """Convert direction to unit vector"""
        vectors = {
            Direction.NORTH: np.array([0, 1, 0]),
            Direction.SOUTH: np.array([0, -1, 0]),
            Direction.EAST: np.array([1, 0, 0]),
            Direction.WEST: np.array([-1, 0, 0]),
            Direction.UP: np.array([0, 0, 1]),
            Direction.DOWN: np.array([0, 0, -1])
        }
        return vectors[direction]

