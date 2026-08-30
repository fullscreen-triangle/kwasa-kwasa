"""
Dump the oracle's script graphs, so the Rust harness needs no Python.

The fourth of the dumps, after `dump_tokens.py`, `dump_ast.py` and
`dump_eval.py`, over the same corpus. `prototype/` is a reference
implementation, not a performance claim (paper, Sec. 8), and it is frozen; its
only remaining job is to be the oracle the Rust port is checked against.

    python figures/dump_graph.py

Writes `crates/ndombolo-core/tests/graph.json`, which the differential test
embeds with `include_str!`. Re-run after any change to `prototype/graph.py`.

# What is dumped, and why per cell

The graph is *accretive*: `add_cell` extends it and no method removes an item
or decreases a weight, which is what makes prefix containment hold (paper,
Prop. 2.7). Dumping only the final graph would test the sum and leave the
accretion untested -- a port that rebuilt the whole graph from scratch on every
cell would agree on every script here and still be the wrong runtime.

So each cell records the graph *as it stands after that cell*, and the harness
checks the sequence. That also puts `stage` under test, since an item declared
in cell 2 must carry stage 2 forever after.

# The parse stage is not under test here

A cell that fails to parse is skipped, not recorded as an error: `dump_eval.py`
already pins every parse failure's message and line, and repeating that here
would be a second copy of the same assertion drifting independently. The graph
of an unparseable cell is not a thing the Python defines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# `prototype/` itself goes on the path, not its parent: the modules import
# each other flat (`from lexer import ...`), so the package form only works
# for the leaves. This is how measure.py does it.
sys.path.insert(0, str(ROOT / "prototype"))
sys.path.insert(0, str(HERE))

from corpus import make_script, sweep       # noqa: E402
from graph import GraphBuilder              # noqa: E402
from lexer import LexError                  # noqa: E402
from parser import ParseError, parse        # noqa: E402
from run import split_cells                 # noqa: E402

OUT = ROOT / "crates" / "ndombolo-core" / "tests" / "graph.json"


# ---------------------------------------------------------------------------
# Edge cases for the graph specifically.
#
# The evaluator's edge scripts pin evaluator hazards, and most of them build a
# trivial graph -- a handful of items and no interesting contacts. The graph's
# own hazards are elsewhere: in which fields are probed, which endpoints are
# dropped, and which declaration wins a redeclaration. Each entry below exists
# because a plausible wrong port passes without it.
#
# The rule this list is held to: a comment claiming a hazard is pinned is
# unearned until a mutation of that exact behaviour turns the harness red. Two
# entries here were added *after* sabotage found them missing, and say so.
# ---------------------------------------------------------------------------

EDGE: List[tuple[str, str]] = [
    # `free_names` returns a set, so a name repeated inside one expression is
    # one contact. A port using a Vec doubles the weight.
    ("dedup", 'item a = 1\nitem b = a + a\nitem c = a * a + a\n'),

    # ...but across statements the weights do accumulate, so a port that
    # deduplicated globally would collapse these to 1.
    ("accumulate", 'item a = 1\nitem b = a\nitem c = a\nitem d = a + b\n'),

    # `setdefault`: a redeclaration keeps the FIRST line and the first stage.
    # A port using plain insertion keeps the last and reports line 3.
    ("redeclare", 'item a = 1\nitem b = 2\nitem a = 3\n'),

    # Both guards in `contact`. `undeclared` is never declared, so `a ->
    # undeclared` is dropped; `item s = s` is a self-contact and is dropped.
    # `len` is a builtin, not an item, so calling it contributes nothing.
    ("dropped-edges", 'item a = undeclared + 1\nitem s = s\n'
                      'item n = len([1, 2])\n'),

    # `Assign` is not in DECL_KINDS, so at top level its target falls back to
    # `opener`, which is None -- the contact vanishes entirely rather than
    # landing on the assigned name. A port that treated assignment as a
    # declaration would both add an item and add an edge.
    ("assign-not-decl", 'item a = 1\nitem b = 2\nb = a\n'),

    # A funxn opens a scope: its body contacts the funxn, and its params are
    # not items, so a mention of a param contributes nothing.
    ("funxn-scope", 'item g = 10\nfunxn f(v):\n    item local = v + g\n'
                    '    return local\nitem r = f(1)\n'),

    # `given` probes `then` and `otherwise` -- two separate block fields. A
    # port that only walked `body` would miss both branches entirely.
    ("given-branches", 'item a = 1\nitem b = 2\nitem c = 3\n'
                       'given a > 0:\n    item t = b\notherwise:\n'
                       '    item e = c\n'),

    # `within` puts its subject in `target`, an EXPRESSION field, so the
    # subject contacts the enclosing opener rather than opening one itself --
    # `within` is not a declaration.
    ("within-target", 'item subj = [1, 2]\nfunxn f():\n'
                      '    within subj:\n        item inner = 1\n'
                      '    return 0\n'),

    # A proposition contacts its motions (the `motions` clause) and its body
    # contacts it (the scope). Both directions, from different sites.
    ("prop-motions", 'proposition P:\n    motion M("first")\n'
                     '    motion N("second")\n    support M with_weight 0.9\n'
                     '    contradict N with_weight 0.4\n'),

    # A verdict's `conf` is an expression field, so a confidence that mentions
    # an item contacts the enclosing proposition too. `inconclusive` carries no
    # `conf` key at all, which the probe must tolerate rather than trip on.
    ("verdict-conf", 'item w = 0.7\nproposition P:\n    motion M("m")\n'
                     '    motion N("n")\n    support M with_weight w\n'
                     '    inconclusive N\n'),

    # Loops: `considering` and `for each` put the sequence in `iter` and open a
    # scope through `body`. The loop variable is not an item.
    ("loops", 'item xs = [1, 2, 3]\nfunxn f():\n'
              '    for each x in xs:\n        item seen = x\n'
              '    return 0\n'),

    # A `while` puts its test in `cond`, the same field name `given` uses.
    ("while-cond", 'item limit = 3\nfunxn f():\n    item i = 0\n'
                   '    while i < limit:\n        i = i + 1\n'
                   '    return i\n'),

    # An import is a Noop with NO `body` key; a loose keyword is a Noop WITH
    # one. The probe must distinguish absent from empty. The indented block
    # is the part sabotage found missing: `allow a` has a body too, but the
    # body is one expression statement that declares nothing and contacts
    # nothing, so a port that never walked a Noop's body agreed with the
    # oracle anyway. The second script declares INSIDE the block, and its
    # contacts fall through to the outer opener -- a Noop declares nothing,
    # so it is not itself a target.
    ("noop-shapes", 'import x from y\nitem a = 1\nallow a\n'),
    ("noop-body", 'item a = 1\nitem b = 2\nallow:\n    item c = a + b\n'),

    # Nesting: a funxn inside a funxn, a proposition inside a funxn. The inner
    # opener has to replace the outer one, not merely shadow the target.
    ("nested-scopes", 'item g = 1\nfunxn outer():\n    funxn inner():\n'
                      '        item deep = g\n        return deep\n'
                      '    item mid = g\n    return inner()\n'),

    # Points and hypotheses are declaration kinds too, with their own labels.
    ("decl-kinds", 'item i = 1\npoint p = 0.6\nfunxn f():\n    return 1\n'
                   'hypothesis H:\n    item inside = i\n'
                   'proposition P:\n    motion M("m")\n'),

    # Forward reference within one cell: `b` mentions `a`, declared later. The
    # two-pass build is what makes this an edge rather than a dropped one.
    ("forward-ref", 'funxn f():\n    return later\nitem later = 1\n'),

    # Maps and lists: only a map's VALUES are names, never its keys, and a
    # nested structure has to be walked all the way down.
    ("collections", 'item a = 1\nitem b = 2\n'
                    'item m = {a: a, b: b}\nitem l = [a, [b, [a]]]\n'),

    # Indexing and field access, the two `target`-carrying expressions, plus a
    # pipe. All reach their operands through the fallthrough probe, not through
    # a named branch.
    ("expr-shapes", 'item xs = [1, 2]\nitem i = 0\nitem p = 0.5\n'
                    'funxn double(v):\n    return v * 2\n'
                    'item hit = xs[i]\nitem conf = p.confidence\n'
                    'item piped = i |> double\n'),
]

# Multi-cell scripts, which are where `stage` and accretion are actually
# tested. A single-cell script pins neither.
MULTI: List[tuple[str, str]] = [
    # Stage numbering, and an item in cell 2 referring to one from cell 1.
    ("stages", 'item a = 1\n// ---\nitem b = a + 1\n// ---\nitem c = a + b\n'),

    # A redeclaration in a LATER cell must keep the earlier stage and line.
    # This is the accretion case: a port rebuilding per cell gets stage 2.
    ("restage", 'item a = 1\n// ---\nitem a = 2\nitem b = a\n'),

    # A cell that adds only contacts, no items: the item count must hold while
    # the contact weights climb.
    ("weights-only", 'item a = 1\nitem b = a\n// ---\nb = a\nitem c = a\n'
                     '// ---\nitem d = a + b + c\n'),
]


def graph_of(source: str) -> Dict[str, Any]:
    """Build the graph cell by cell, recording it after each.

    Mirrors `run.py`'s loop: one builder across all cells, `stage` being the
    cell index. A cell that fails to parse ends the script, as it does there.
    """
    builder = GraphBuilder()
    cells: List[Dict[str, Any]] = []

    for k, (first_line, text) in enumerate(split_cells(source)):
        try:
            body = parse(text, first_line)
        except (LexError, ParseError):
            # Not recorded as an error: `dump_eval.py` owns parse failures.
            break
        builder.add_cell(body, k)
        cells.append({
            "index": k,
            "first_line": first_line,
            "text": text,
            "graph": builder.graph.to_json(),
        })

    return {"cells": cells, "final": builder.graph.to_json()}


def dump(name: str, source: str) -> Dict[str, Any]:
    out = graph_of(source)
    out["name"] = name
    return out


def main() -> int:
    out: List[Dict[str, Any]] = []

    # The same generated sweep the other dumps use, so the graphs here are the
    # graphs of the scripts already under test elsewhere.
    for spec in sweep():
        src = make_script(spec["n_items"], spec["n_funxns"],
                          spec["n_motions"], spec["depth"], spec["seed"])
        out.append(dump(spec["name"], src))

    for name, source in EDGE + MULTI:
        try:
            out.append(dump(f"edge-{name}", source))
        except Exception as e:                       # noqa: BLE001
            # Dumping a corpus with a hole in it would be worse than stopping.
            print(f"edge-{name} raised {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1

    for ex in ("signal", "assay"):
        p = ROOT / "prototype" / "examples" / f"{ex}.tb"
        out.append(dump(ex, p.read_text(encoding="utf-8")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scripts": out}, indent=1), encoding="utf-8")

    cells = sum(len(s["cells"]) for s in out)
    items = sum(s["final"]["item_count"] for s in out)
    contacts = sum(s["final"]["contact_count"] for s in out)
    print(f"{OUT.relative_to(ROOT)}: {len(out)} scripts, {cells} cells, "
          f"{items} items, {contacts} contacts")

    empty = [s["name"] for s in out if not s["cells"]]
    if empty:
        print(f"  WARNING: {len(empty)} script(s) produced no cells: "
              f"{', '.join(empty[:5])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
