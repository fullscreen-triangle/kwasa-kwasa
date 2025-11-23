"""Encoding modules - Multi-dimensional semantic encoding"""

from .cardinal_directions import CardinalEncoder, Direction
from .word_expansion import WordExpander
from .positional_context import PositionalEncoder

__all__ = ["CardinalEncoder", "Direction", "WordExpander", "PositionalEncoder"]

