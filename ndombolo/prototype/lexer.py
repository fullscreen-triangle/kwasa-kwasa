"""
Turbulance lexer for the ndombolo deterministic core.

Mirrors the keyword and operator surface of the reference JavaScript
implementation (long-grass/src/lib/turbulance/index.js) so that a script
accepted there is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

KEYWORDS = {
    "funxn", "item", "proposition", "hypothesis", "motion", "support",
    "contradict", "inconclusive", "within", "given", "considering", "for",
    "each", "in", "while", "return", "ensure", "allow", "research", "cause",
    "point", "resolution", "resolve", "cycle", "drift", "flow", "roll",
    "until", "settled", "over", "on", "goal", "metacognitive", "import",
    "from", "as", "otherwise", "with_confidence", "and", "or", "not",
}

# `considering` quantifiers; `item` is the default when none is written.
QUANTS = {"all", "these"}

# Keywords recognised by the grammar but with no deterministic semantics in
# this core. Parsed to a Noop so a script using them still runs.
LOOSE = {"allow", "research", "cause", "goal", "metacognitive", "resolution"}

# Longest first: the scanner takes the first match.
OPERATORS = [
    "==", "!=", "<=", ">=", "=>", "|>", "&&", "||",
    "=", "<", ">", "+", "-", "*", "/", "%",
    "(", ")", "[", "]", "{", "}", ",", ":", ".", "|",
]


class LexError(Exception):
    def __init__(self, message: str, line: int):
        super().__init__(f"line {line}: {message}")
        self.message = message
        self.line = line


@dataclass(frozen=True)
class Token:
    kind: str            # kw | ident | num | str | op | newline | indent | dedent | eof
    value: str
    line: int
    col: int

    def is_kw(self, *names: str) -> bool:
        return self.kind == "kw" and self.value in names

    def is_op(self, *names: str) -> bool:
        return self.kind == "op" and self.value in names


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(source: str, first_line: int = 1) -> List[Token]:
    """Scan `source` into tokens, emitting explicit indent/dedent.

    Layout is significant: a block is opened by `:` at end of line and
    delimited by indentation, as in the reference implementation.
    """
    toks: List[Token] = []
    indents: List[int] = [0]
    bracket_depth = 0
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for lineno, raw in enumerate(lines, start=first_line):
        stripped = raw.strip()
        # Blank lines and full-line comments carry no layout information.
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue

        line_toks = _scan_line(raw, lineno)

        # Inside brackets, layout carries no meaning: a multi-line list or map
        # is one logical line, so indent/dedent are suppressed until it closes.
        if bracket_depth > 0:
            toks.extend(line_toks)
            bracket_depth += _bracket_delta(line_toks)
            continue

        depth = len(raw) - len(raw.lstrip(" \t"))
        depth = _expand_tabs(raw[:depth])

        if depth > indents[-1]:
            indents.append(depth)
            toks.append(Token("indent", "", lineno, depth))
        else:
            while depth < indents[-1]:
                indents.pop()
                toks.append(Token("dedent", "", lineno, depth))
            if depth != indents[-1]:
                raise LexError("inconsistent indentation", lineno)

        toks.extend(line_toks)
        bracket_depth += _bracket_delta(line_toks)
        if bracket_depth == 0:
            toks.append(Token("newline", "", lineno, len(raw)))

    while len(indents) > 1:
        indents.pop()
        toks.append(Token("dedent", "", first_line + len(lines) - 1, 0))
    toks.append(Token("eof", "", first_line + len(lines) - 1, 0))
    return toks


def _bracket_delta(toks: List[Token]) -> int:
    """Net change in open brackets across one line's tokens."""
    return sum(t.is_op("(", "[", "{") - t.is_op(")", "]", "}") for t in toks)


def _expand_tabs(prefix: str, width: int = 4) -> int:
    col = 0
    for ch in prefix:
        col = col + width - (col % width) if ch == "\t" else col + 1
    return col


def _scan_line(raw: str, lineno: int) -> List[Token]:
    out: List[Token] = []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]

        if ch in " \t":
            i += 1
            continue

        # Trailing comment ends the line.
        if raw.startswith("//", i) or ch == "#":
            break

        if ch == '"':
            j, buf = i + 1, []
            while j < n and raw[j] != '"':
                if raw[j] == "\\" and j + 1 < n:
                    buf.append(_unescape(raw[j + 1]))
                    j += 2
                    continue
                buf.append(raw[j])
                j += 1
            if j >= n:
                raise LexError("unterminated string", lineno)
            out.append(Token("str", "".join(buf), lineno, i))
            i = j + 1
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and raw[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (raw[j].isdigit() or (raw[j] == "." and not seen_dot)):
                seen_dot = seen_dot or raw[j] == "."
                j += 1
            out.append(Token("num", raw[i:j], lineno, i))
            i = j
            continue

        if _is_ident_start(ch):
            j = i
            while j < n and _is_ident_part(raw[j]):
                j += 1
            word = raw[i:j]
            out.append(Token("kw" if word in KEYWORDS else "ident", word, lineno, i))
            i = j
            continue

        for op in OPERATORS:
            if raw.startswith(op, i):
                out.append(Token("op", op, lineno, i))
                i += len(op)
                break
        else:
            raise LexError(f"unexpected character {ch!r}", lineno)

    return out


def _unescape(ch: str) -> str:
    return {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(ch, ch)


def find_token(toks: List[Token], line: int) -> Optional[Token]:
    """First token on `line`, for error reporting."""
    for t in toks:
        if t.line == line and t.kind not in ("indent", "dedent", "newline"):
            return t
    return None
