"""
Dump the oracle's token stream, so the Rust harness needs no Python at test time.

`prototype/` is a reference implementation, not a performance claim (paper, §8),
and it is frozen. It stays off the execution path; its only remaining job is to
be the oracle the Rust port is checked against. This script freezes that oracle
into JSON.

    python figures/dump_tokens.py

Writes `crates/ndombolo-core/tests/tokens.json`, which the differential test
embeds with `include_str!`. Re-run after any change to the Python lexer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from prototype.lexer import tokenize          # noqa: E402
from prototype.run import split_cells         # noqa: E402

sys.path.insert(0, str(HERE))
from corpus import make_script, sweep         # noqa: E402

OUT = ROOT / "crates" / "ndombolo-core" / "tests" / "tokens.json"


# ---------------------------------------------------------------------------
# Edge cases the generated corpus never produces.
#
# `corpus.py` emits clean machine-written Turbulance: no tabs, no string
# escapes, no leading-dot numbers, no trailing comments, and only 16 of the 26
# operators. Agreement on that corpus alone is broad but shallow -- it would
# leave the lexer's fiddly paths resting on the port author's expectations
# rather than on the oracle's behaviour. These scripts are hand-written to
# reach them. They are not required to be meaningful Turbulance; they are
# required to tokenize.
# ---------------------------------------------------------------------------

EDGE: List[Tuple[str, str]] = [
    # The eight operators the corpus never emits: == != => |> && || * / % |
    ("ops-compare", 'item a = 1\nitem b = a == 1\nitem c = a != 1\n'
                    'item d = a <= 1\nitem e = a >= 1\n'),
    ("ops-arith", 'item i = 6 * 2 / 3 % 4 + 1 - 1\n'),
    ("ops-logic", 'item p = a && b\nitem q = a || b\nitem r = a | b\n'),
    ("ops-arrow", 'item f = x => y\nitem g = x |> y\n'),

    # Tab expansion at width 4, including the mixed-whitespace prefix that
    # makes the tab stop (rather than a count of characters) decide the depth.
    ("tabs", 'funxn f(x):\n\titem a = 1\n'),
    ("tabs-nested", 'funxn f(x):\n\titem a = 1\n\t\titem b = 2\n'),
    ("tabs-mixed", 'funxn f(x):\n  \titem a = 1\n'),

    # Every branch of `unescape`, plus the fallthrough for an unknown escape.
    ("escapes", 'item s = "a\\nb"\nitem t = "a\\tb"\n'
                'item u = "a\\\\b"\nitem v = "a\\"b"\nitem w = "a\\qb"\n'),

    # A leading dot is a number only when a digit follows.
    ("leading-dot", 'item a = .5\nitem b = 0.5\nitem c = [.25, .75]\n'
                    'item d = a.b\n'),

    # Both comment markers, trailing and whole-line. `#` matters because a
    # markdown heading pasted into a cell would otherwise vanish silently.
    ("comments", 'item a = 1  // trailing\nitem b = 2  # hash trailing\n'
                 '// whole line\n# hash line\nitem c = 3\n'),

    # Layout corners: bracket suppression, deep dedent, blank lines, and a
    # block left open at end of input.
    ("brackets", 'item xs = [\n  1,\n  2,\n]\nitem y = 1\n'),
    ("nested-dedent", 'funxn f(x):\n    given x > 1:\n'
                      '        given x > 2:\n            item a = 1\n'
                      'item b = 2\n'),
    ("blank-lines", 'item a = 1\n\n\n    \n\nitem b = 2\n'),
    ("trailing-indent", 'funxn f(x):\n    item a = 1\n'),
    ("empty", ''),

    # `col` is a byte offset in the Python and the record carries it, so a
    # non-ASCII line is where a char-indexed port would silently disagree.
    ("unicode-str", 'item s = "naïve µg"\nitem t = 1\n'),
]


def dump(name: str, source: str) -> Dict[str, Any]:
    """Tokenize cell by cell, exactly as the runtime does.

    `split_cells` keeps line numbers script-absolute, so the recorded `line`
    is a place in the document. Tokenizing the whole source at once would
    silently lose that and make the comparison weaker than the real path.
    """
    cells = []
    for first_line, text in split_cells(source):
        toks = tokenize(text, first_line)
        cells.append({
            "first_line": first_line,
            "text": text,
            "tokens": [[t.kind, t.value, t.line, t.col] for t in toks],
        })
    return {"name": name, "cells": cells}


def main() -> int:
    out: List[Dict[str, Any]] = []

    # Built exactly as measure.py builds it, so the tokens dumped here are the
    # tokens the measured records came from rather than a parallel corpus.
    for spec in sweep():
        src = make_script(spec["n_items"], spec["n_funxns"],
                          spec["n_motions"], spec["depth"], spec["seed"])
        out.append(dump(spec["name"], src))

    for name, source in EDGE:
        out.append(dump(f"edge-{name}", source))

    for ex in ("signal", "assay"):
        p = ROOT / "prototype" / "examples" / f"{ex}.tb"
        out.append(dump(ex, p.read_text(encoding="utf-8")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scripts": out}, indent=1), encoding="utf-8")

    cells = sum(len(s["cells"]) for s in out)
    toks = sum(len(c["tokens"]) for s in out for c in s["cells"])
    print(f"{len(out)} scripts, {cells} cells, {toks} tokens -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
