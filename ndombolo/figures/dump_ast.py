"""
Dump the oracle's parse trees, so the Rust harness needs no Python at test time.

The companion to `dump_tokens.py`, over the same corpus and the same cell-wise
path. `prototype/` is a reference implementation, not a performance claim
(paper, §8), and it is frozen; its only remaining job is to be the oracle the
Rust port is checked against.

    python figures/dump_ast.py

Writes `crates/ndombolo-core/tests/ast.json`, which the differential test
embeds with `include_str!`. Re-run after any change to the Python parser.

The AST is plain dicts (`parser.Node = Dict[str, Any]`), so it serialises
without a codec and the comparison can be made on the JSON directly. That is
what pins the Rust `Node` enum's serialisation: it has to produce these dicts,
key for key, including which keys are *absent* -- `graph.py` walks statements
by probing field names rather than by matching on kind, so a key that appears
where the Python omits it changes the graph rather than merely the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# `prototype/` itself goes on the path, not its parent: the modules import
# each other flat (`from lexer import ...`), so the package form only works
# for the leaves. This is how measure.py does it.
sys.path.insert(0, str(ROOT / "prototype"))
sys.path.insert(0, str(HERE))

from corpus import make_script, sweep             # noqa: E402
from parser import ParseError, parse              # noqa: E402
from run import split_cells                       # noqa: E402

OUT = ROOT / "crates" / "ndombolo-core" / "tests" / "ast.json"


# ---------------------------------------------------------------------------
# Edge cases the generated corpus never produces.
#
# `corpus.py` emits clean machine-written Turbulance, and it exercises a small
# part of the grammar: it never writes a pipe, a map, an index, a unary
# operator, an `otherwise` branch, a loose keyword, an import, or a verdict
# with a confidence. Agreement on that corpus alone would leave most of the
# parser resting on the port author's expectations rather than on the oracle's
# behaviour. These scripts are hand-written to reach the rest.
#
# They are required to parse, not to mean anything. A few are here precisely
# because they are shaped oddly -- what is under test is that the Rust agrees
# with the Python about the tree, including where the Python's tree is
# surprising.
# ---------------------------------------------------------------------------

EDGE: List[Tuple[str, str]] = [
    # Precedence. The cascade is six levels and the corpus reaches two of
    # them, so the shape of a mixed expression is otherwise unchecked.
    ("prec-arith", 'item a = 1 + 2 * 3 - 4 / 5 % 6\n'),
    ("prec-compare", 'item a = 1 + 2 == 3 * 4\nitem b = 1 < 2 == 3 > 4\n'),
    ("prec-logic", 'item a = 1 < 2 and 3 > 4 or 5 == 6\n'),
    # `&&`/`||` are the same rungs as `and`/`or`, and `_canon` folds the
    # spelling, so these two lines must produce identical trees but for line.
    ("prec-canon", 'item a = x && y || z\nitem b = x and y or z\n'),
    ("prec-right", 'item a = 1 - 2 - 3\nitem b = 1 - (2 - 3)\n'),

    # A parenthesised expression returns `inner` unwrapped: there is no Paren
    # node and the line is not rewritten to the paren's. Easy to get wrong in
    # a port that adds a grouping node for symmetry.
    ("paren-unwrapped", 'item a = (1)\nitem b = ((x))\nitem c = (x + y) * z\n'),

    # Unary, including the stacking `parse_unary` recurses for.
    ("unary", 'item a = -1\nitem b = not x\nitem c = - -1\n'
              'item d = not not x\nitem e = -x + 1\n'),

    # Postfix chains, which the corpus writes only as a bare call.
    ("postfix", 'item a = f(x)\nitem b = f(x)(y)\nitem c = a.b.c\n'
                'item d = a[0][1]\nitem e = f(x).b[2](y)\nitem f = g()\n'),

    # Pipes sit above the whole binary cascade, so a pipe of sums groups the
    # sums first. Chained pipes are left-associative.
    ("pipe", 'item a = x |> f\nitem b = x |> f |> g\n'
             'item c = 1 + 2 |> f\n'),

    # Collections, including the trailing comma and the multi-line layout that
    # only parses because brackets suppress newlines in the lexer.
    ("collections", 'item a = []\nitem b = [1, 2, 3]\nitem c = [1, 2,]\n'
                    'item d = [[1], [2]]\nitem e = [\n  1,\n  2,\n]\n'),
    # Map keys may be ident, str, or keyword -- the keyword case is the one a
    # port is likely to reject, since `item` is a `kw` token here.
    ("maps", 'item a = {}\nitem b = {x: 1, y: 2}\nitem c = {"k": 1}\n'
             'item d = {item: 1, given: 2}\nitem e = {x: 1,}\n'
             'item f = {\n  x: 1,\n  y: 2,\n}\n'),

    # Literals. Every number becomes a float, so `1` parses to 1.0 and the
    # serialisation has to keep the `.0` for the comparison to hold.
    ("literals", 'item a = 1\nitem b = 1.5\nitem c = .5\nitem d = 0\n'
                 'item e = true\nitem f = false\nitem g = none\n'
                 'item h = "text"\n'),

    # `given` with and without `otherwise`, and the lookahead that has to be
    # rolled back when the newlines are not followed by one.
    ("given-otherwise", 'given x > 1:\n    item a = 1\notherwise:\n'
                        '    item b = 2\n'),
    ("given-bare", 'given x > 1:\n    item a = 1\n\nitem b = 2\n'),
    ("given-inline", 'given x > 1: item a = 1\n'),
    ("given-nested", 'given x:\n    given y:\n        item a = 1\n'
                     '    otherwise:\n        item b = 2\n'),

    # Every loop form, including `considering`'s three-way quantifier rule:
    # an explicit quantifier, the `item` keyword, or nothing at all.
    ("loops", 'for each x in xs:\n    item a = 1\n'
              'while x < 10:\n    item b = 2\n'
              'considering all x in xs:\n    item c = 3\n'
              'considering these x in xs:\n    item d = 4\n'
              'considering item x in xs:\n    item e = 5\n'
              'considering x in xs:\n    item f = 6\n'),

    # Propositions partition their block: motions are lifted into `motions`
    # and everything else stays in `body`. The only statement whose block does
    # not appear verbatim.
    # Both verdict kinds appear with and without a confidence. Absent `conf`
    # is present-and-null, not a missing key, and a port that skips the key
    # when it is empty produces a tree `graph.py` walks differently. One bare
    # verdict would leave the other kind's null unchecked -- which is exactly
    # what a sabotage run found -- so there are two of each.
    ("proposition", 'proposition P:\n    motion M("text")\n'
                    '    motion N()\n    motion O\n    item a = 1\n'
                    '    support M with_confidence(0.8)\n'
                    '    support N\n'
                    '    contradict N with_confidence(0.2)\n'
                    '    contradict M\n    inconclusive O\n'),
    ("proposition-empty", 'proposition P:\n    item a = 1\n'),
    ("hypothesis", 'hypothesis H:\n    item a = 1\n'),

    # Statement forms the corpus never writes.
    ("misc-stmts", 'point p = 0.5\nresolve p\nensure x > 1\n'
                   'funxn f():\n    return\n'
                   'funxn g(a, b):\n    return a + b\n'),
    # Assignment is recognised only for a bare name (one token of lookahead),
    # so `a.b = 3` is a parse error rather than a field assignment: the grammar
    # has no such form. Kept out of the corpus for that reason.
    ("assign", 'item a = 1\na = 2\nb = a + 1\n'),

    # `import` consumes to end of line and produces a Noop with `text` but no
    # `body`; a loose keyword produces a Noop with both. Two different key
    # sets under one kind, which a single Rust variant has to reproduce.
    # Only an `import`-leading line parses. `from` is a keyword with no
    # handler and is not loose, so `parse_statement` falls through to
    # `parse_expr_statement` and `parse_primary` rejects it -- a hard error,
    # not a Noop. The Rust dispatch has to reproduce that rather than treat
    # every unhandled keyword as loose.
    ("noop-import", 'import x from y\nimport a as b\nitem a = 1\n'),
    ("noop-loose", 'allow x\nresearch y\ncause z\ngoal g:\n    item a = 1\n'
                   'metacognitive m:\n    item b = 2\nresolution r\n'),
    # A loose keyword's scan tracks bracket depth, so a `:` inside brackets
    # does not open its block.
    ("noop-brackets", 'allow f(a, b)\ngoal g(x: 1):\n    item a = 1\n'),

    # Bare expression statements, which are `ExprStmt` rather than `Assign`
    # unless a bare name is followed by `=`.
    ("expr-stmts", 'f(x)\nx\nx + 1\nx.y\n'),

    ("empty", ''),
]


def dump(name: str, source: str) -> Dict[str, Any]:
    """Parse cell by cell, exactly as the runtime does.

    `split_cells` keeps line numbers script-absolute, so every `line` in the
    tree is a place in the document. Parsing the whole source at once would
    lose that and make the comparison weaker than the real path.
    """
    cells = []
    for first_line, text in split_cells(source):
        cells.append({
            "first_line": first_line,
            "text": text,
            "ast": parse(text, first_line),
        })
    return {"name": name, "cells": cells}


def main() -> int:
    out: List[Dict[str, Any]] = []

    # Built exactly as measure.py builds it, so the trees dumped here are the
    # trees the measured records came from rather than a parallel corpus.
    for spec in sweep():
        src = make_script(spec["n_items"], spec["n_funxns"],
                          spec["n_motions"], spec["depth"], spec["seed"])
        out.append(dump(spec["name"], src))

    for name, source in EDGE:
        try:
            out.append(dump(f"edge-{name}", source))
        except ParseError as e:
            # An edge script that does not parse is a bug in this file, not a
            # finding about the parser: fail loudly rather than dumping a
            # corpus with a hole in it.
            print(f"edge-{name} does not parse: {e}", file=sys.stderr)
            return 1

    for ex in ("signal", "assay"):
        p = ROOT / "prototype" / "examples" / f"{ex}.tb"
        out.append(dump(ex, p.read_text(encoding="utf-8")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scripts": out}, indent=1), encoding="utf-8")

    cells = sum(len(s["cells"]) for s in out)
    nodes = sum(_count(c["ast"]) for s in out for c in s["cells"])
    print(f"{len(out)} scripts, {cells} cells, {nodes} nodes -> {OUT}")
    return 0


def _count(x: Any) -> int:
    """Every dict carrying a `kind`, at any depth."""
    if isinstance(x, list):
        return sum(_count(i) for i in x)
    if isinstance(x, dict):
        return ("kind" in x) + sum(_count(v) for v in x.values())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
