"""
Dump the oracle's evaluation results, so the Rust harness needs no Python.

The third of the dumps, after `dump_tokens.py` and `dump_ast.py`, over the same
corpus and the same cell-wise path. `prototype/` is a reference implementation,
not a performance claim (paper, Sec. 8), and it is frozen; its only remaining
job is to be the oracle the Rust port is checked against.

    python figures/dump_eval.py

Writes `crates/ndombolo-core/tests/eval.json`, which the differential test
embeds with `include_str!`. Re-run after any change to the Python evaluator.

What is dumped, per cell, is the *trace* -- every rule application with its
site, its reads, and its write -- plus the store delta and the printed output.
The trace is the thing that matters: the paper's Thm. 5.2 makes every atomic
question a projection of it, so a port that agreed on final values while
disagreeing on the trace would have ported the arithmetic and lost the point.

Cells run against a single `Compiler`, exactly as `run.py` does, so the store,
the trace counter and the proposition table all accrete across cells. Running
each cell fresh would test a runtime nobody uses.

A cell that raises is dumped with its error and ends the script, again as
`run.py` does. Those entries are not failures of this dump: several edge
scripts exist precisely to pin an error's message and line, which a port is as
able to get wrong as it is a value.
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

from corpus import make_script, sweep                       # noqa: E402
from evaluator import Compiler, RuntimeErr, render, tag     # noqa: E402
from lexer import LexError                                  # noqa: E402
from parser import ParseError, parse                        # noqa: E402
from run import split_cells                                 # noqa: E402

OUT = ROOT / "crates" / "ndombolo-core" / "tests" / "eval.json"


# ---------------------------------------------------------------------------
# Edge cases the generated corpus never reaches.
#
# `corpus.py` emits clean machine-written Turbulance that computes with numbers
# and declares propositions. It never divides, never indexes, never calls a
# builtin other than through its own funxns, never writes a `within`, and never
# fails. Agreement on it alone would leave most of the evaluator resting on the
# port author's expectations rather than on the oracle's behaviour.
#
# Several of these are here for one line of the Python each, and the comment
# says which -- because on the Rust side each is a place where the obvious
# translation is the wrong one.
# ---------------------------------------------------------------------------

EDGE: List[Tuple[str, str]] = [
    # `render` collapses an integral float to an int, so 1.0 serialises as `1`
    # -- the exact opposite of the AST, where `Num.value` stays 1.0. Same
    # numbers, opposite JSON, depending on which side of the runtime they are
    # on. A port that picks one rule for both is wrong twice.
    ("num-integral", 'item a = 1\nitem b = 2.0\nitem c = 1.5\n'
                     'item d = 3 / 2\nitem e = 4 / 2\nitem f = 0 - 0\n'),

    # Python's bool is a subclass of int, so `render(True) == render(1.0)` and
    # `1 == true` is true. A Rust Value enum with distinct Bool and Num says
    # false unless equality is written to cross the tags deliberately.
    ("bool-is-num", 'item a = 1 == true\nitem b = 0 == false\n'
                    'item c = true + 1\nitem d = true and 1\n'
                    'item e = "1" == 1\nitem f = none == false\n'),

    # `_text` is %g: six significant digits, trailing zeros stripped, and an
    # integral float printed through `int()` so 1e21 comes out as full digits
    # rather than in exponent form. This is what `print` writes and what the
    # `note` of an expr/return/resolve event carries, so it is observable.
    ("text-formatting", 'print(1)\nprint(1.5)\nprint(1 / 3)\n'
                        'print(0.1 + 0.2)\nprint(100000)\nprint(0.000001)\n'
                        'print(true)\nprint(false)\nprint(none)\n'
                        'print("s")\nprint([1, 2])\nprint({a: 1})\n'
                        'print(1, 2)\nprint()\n'),

    # `%` is math.fmod, which takes the sign of the dividend. Rust's `%` on
    # f64 agrees; Python's own `%` operator does not, so a port that reads the
    # source as `a % b` and writes `a.rem_euclid(b)` diverges on negatives.
    ("arith", 'item a = 7 % 3\nitem b = 0 - 7 % 3\nitem c = 7 % (0 - 3)\n'
              'item d = 2 * 3 + 1\nitem e = 0 - 5\nitem f = - -5\n'),

    # A map literal keeps insertion order, in the value, in `_text`, and in the
    # JSON. A BTreeMap would sort it and the divergence would show up only in
    # printed output and rendered values -- never in a length or a count.
    ("map-order", 'item m = {z: 1, a: 2, m: 3}\nprint(m)\n'
                  'item n = {b: 1, a: 2}\nitem o = n.a\nitem p = m["z"]\n'),

    # Indexing: lists by number (truncated to int), maps by text. A negative or
    # out-of-range list index is an error, but a missing map key is `none`.
    ("indexing", 'item l = [10, 20, 30]\nitem a = l[0]\nitem b = l[2]\n'
                 'item c = l[1.7]\nitem m = {k: 1}\nitem d = m["k"]\n'
                 'item e = m["nope"]\n'),

    # Truthiness by tag: zero, empty string, empty list, empty map, none and
    # false are falsy; a point is truthy when its confidence exceeds zero.
    ("truthy", 'given 0: print("no")\notherwise: print("zero falsy")\n'
               'given "": print("no")\notherwise: print("empty str falsy")\n'
               'given []: print("no")\notherwise: print("empty list falsy")\n'
               'given {}: print("no")\notherwise: print("empty map falsy")\n'
               'given none: print("no")\notherwise: print("none falsy")\n'
               'given 0.5: print("nonzero truthy")\n'
               'given [0]: print("nonempty list truthy")\n'),

    # `and`/`or` short-circuit, so the right operand is not evaluated when it
    # cannot matter -- and, crucially, emits no events when skipped. They also
    # return a bool rather than the operand, unlike most languages'.
    ("short-circuit", 'item a = false and undefined_name\n'
                      'item b = true or undefined_name\n'
                      'item c = true and 5\nitem d = false or 5\n'),

    # A point's confidence falls back certainty -> confidence -> 1.0, is
    # clamped, and is reachable under either field name whatever the map said.
    ("points", 'point p = {certainty: 0.7, x: 1}\npoint q = {confidence: 0.3}\n'
               'point r = {x: 1}\npoint s = {certainty: 5}\n'
               'point t = {certainty: 0 - 1}\n'
               'item a = p.confidence\nitem b = p.certainty\nitem c = p.x\n'
               'item d = p.missing\nprint(p)\n'),

    # Noisy-or over verdicts, which is not a sum and not a max: two 0.5
    # supports give 0.75, and a contradict multiplies the complement in, so M
    # scores 0.375 rather than any of 1.5, 0.5 or 1.0.
    #
    # An inconclusive carries no confidence *and cannot*: `parse_inconclusive`
    # is a separate function that stops after the name, so the evaluator's
    # `if stance == "inconclusive": conf = 0.0` can only ever be zeroing the
    # 1.0 default. Reachable in the AST, unreachable from source. The rejection
    # is pinned below as `err-inconclusive-conf`.
    ("verdicts", 'proposition P:\n    motion M("m")\n    motion N\n'
                 '    support M with_confidence(0.5)\n'
                 '    support M with_confidence(0.5)\n'
                 '    contradict M with_confidence(0.5)\n'
                 '    support N\n'
                 '    inconclusive N\n'),

    # `resolve-motion`'s reads are built as a dict keyed by motion name, so
    # repeated verdicts on one motion collapse to a single entry with the last
    # value -- not one entry per verdict. Building a Vec there is the obvious
    # port and is wrong.
    ("verdict-reads", 'proposition Q:\n    motion M\n'
                      '    support M with_confidence(0.25)\n'
                      '    support M with_confidence(0.75)\n'),

    # Closures capture the defining environment, so a funxn sees names declared
    # before it; a parameter shadows an outer name of the same spelling for the
    # duration of the call and no longer.
    ("closures", 'item a = 1\nfunxn f():\n    return a\n'
                 'item x = 1\nfunxn g(x):\n    return x + 1\n'
                 'item b = f()\nitem c = g(10)\nitem d = x\n'
                 'funxn h(p, q):\n    return p * q\nitem e = h(2, 3)\n'
                 'funxn noret():\n    item z = 1\nitem k = noret()\n'),

    # A `within` opens a scope whichever way it goes: on a map it binds the
    # keys, on anything else it binds nothing, and either way what the body
    # declares is gone at the end. So the store delta sees nothing from it.
    ("within", 'item m = {p: 1, q: 2}\nwithin m:\n    item r = p + q\n'
               '    print(r)\nitem s = 5\nwithin s:\n    item t = 1\n'
               '    print(t)\n'),

    # Every loop form, including `considering item` taking the head only and a
    # non-list iterable being wrapped in a singleton rather than rejected.
    ("loops", 'item xs = [1, 2, 3]\nfor each x in xs:\n    print(x)\n'
              'considering all y in xs:\n    print(y)\n'
              'considering item z in xs:\n    print(z)\n'
              'considering w in 5:\n    print(w)\n'
              'item n = 0\nwhile n < 3:\n    n = n + 1\nprint(n)\n'
              'while false:\n    print("never")\n'),

    # The builtins, including the empty-argument cases that return 0.0 rather
    # than raising, and `round`, which is banker's rounding in Python: round(2.5)
    # is 2, not 3. Rust's f64::round is half-away-from-zero and diverges.
    ("builtins", 'print(len([1, 2, 3]))\nprint(len("abc"))\n'
                 'print(len({a: 1}))\nprint(sum(1, 2, 3))\n'
                 'print(min(3, 1, 2))\nprint(max(3, 1, 2))\n'
                 'print(abs(0 - 4))\nprint(round(2.5))\nprint(round(3.5))\n'
                 'print(round(2.4))\nprint(sum())\nprint(min())\n'),

    # Pipes are application, so `x |> f` traces exactly as `f(x)` would.
    ("pipe", 'funxn double(v):\n    return v * 2\n'
             'item a = 3 |> double\nitem b = 3 |> double |> double\n'
             'item c = double(3)\n'),

    # `ensure` that holds continues; a failing one is an error with its own
    # message, and it emits its event before raising.
    ("ensure-ok", 'item a = 1\nensure a == 1\nprint("past")\n'),

    # Statement forms that only exist for the trace: resolve and a bare
    # expression both emit a note carrying the rendered value.
    ("resolve-expr", 'point p = {certainty: 0.5}\nresolve p\n'
                     'item a = 2\na\na + 1\n"text"\n'),

    # A loose keyword and an import are Noops at runtime too: one event each,
    # carrying the text, and a loose keyword's body is *not* executed.
    ("noops", 'import x from y\nallow a\nresearch b\n'
              'goal g:\n    item never = 1\nitem after = 2\n'),

    # -- errors. Each ends its script, and what is pinned is the message and
    # the line, which a port can get wrong as readily as a value.
    ("err-undefined", 'item a = 1\nitem b = undefined_name\nitem c = 3\n'),
    ("err-div-zero", 'item a = 1 / 0\n'),
    ("err-mod-zero", 'item a = 1 % 0\n'),
    ("err-not-num", 'item a = "s" - 1\n'),
    ("err-index-range", 'item l = [1]\nitem a = l[5]\n'),
    ("err-not-indexable", 'item a = 5\nitem b = a[0]\n'),
    ("err-no-field", 'item a = 5\nitem b = a.f\n'),
    ("err-not-callable", 'item a = 5\nitem b = a()\n'),
    ("err-arity", 'funxn f(a, b):\n    return a\nitem x = f(1)\n'),
    ("err-ensure", 'item a = 1\nensure a == 2\n'),
    ("err-verdict-outside", 'motion M\nsupport M\n'),
    ("err-point-not-map", 'point p = 5\n'),
    ("err-len", 'print(len(5))\n'),

    # `inconclusive` takes no confidence -- a parse error, not a runtime one,
    # so this is the only edge script that fails before the evaluator runs.
    ("err-inconclusive-conf", 'proposition P:\n    motion M\n'
                              '    inconclusive M with_confidence(0.5)\n'),

    # An assignment to a name that was never declared *defines* it: `_st_Assign`
    # falls back to `define` when `env.has` is false. So `Env.assign`'s
    # "assignment to undefined name" has exactly one caller, guarded by that
    # test, and cannot fire from any Turbulance source at all. A port that reads
    # `Env` on its own and implements the error faithfully would be adding a
    # failure the oracle does not have.
    ("assign-undeclared", 'undeclared = 1\nundeclared = 2\n'),

    # A cell that fails ends the script, so a later cell must not run. The
    # separator is what makes this two cells rather than one.
    ("err-stops-later-cells", 'item a = 1\n// ---\nitem b = undefined_name\n'
                              '// ---\nitem c = 3\n'),

    # A multi-cell script whose store, trace counter and propositions all carry
    # forward -- the property that makes this a session rather than three runs.
    ("multi-cell", 'item a = 1\n// ---\nitem b = a + 1\n// ---\n'
                   'funxn f():\n    return a + b\n// ---\nitem c = f()\n'),

    ("empty", ''),
]


def dump(name: str, source: str) -> Dict[str, Any]:
    """Run the cells against one compiler, exactly as `run.py` does.

    Not a call to `run_script`: that would also build the graph, which is a
    later stage of the port, and would bury the trace inside a larger object
    whose other fields are not yet under test. The loop here is the same loop,
    stopped at what the evaluator produces.
    """
    compiler = Compiler()
    cells: List[Dict[str, Any]] = []
    # Builtins are the initial environment, not something a cell wrote, so
    # they are excluded from the first cell's delta by seeding `prev_store`.
    prev_store = {n: render(v) for n, v in compiler.env.flatten().items()}
    prev_trace = 0
    prev_output = 0

    for k, (first_line, text) in enumerate(split_cells(source)):
        cell: Dict[str, Any] = {"index": k, "first_line": first_line,
                                "text": text}
        try:
            body = parse(text, first_line)
        except (LexError, ParseError) as e:
            cell.update(ok=False, error={"phase": "parse",
                                         "message": str(e), "line": e.line})
            cells.append(cell)
            break

        try:
            compiler.exec_block(body)
            cell["ok"] = True
        except RuntimeErr as e:
            cell.update(ok=False, error={"phase": "run",
                                         "message": str(e), "line": e.line})

        store = compiler.env.flatten()
        delta: Dict[str, Any] = {}
        for nm, value in store.items():
            rendered = render(value)
            if nm not in prev_store:
                delta[nm] = {"change": "added", "value": rendered,
                             "tag": tag(value)}
            elif prev_store[nm] != rendered:
                delta[nm] = {"change": "updated", "from": prev_store[nm],
                             "value": rendered, "tag": tag(value)}
        prev_store = {n: render(v) for n, v in store.items()}

        cell["store_delta"] = delta
        cell["trace"] = compiler.trace[prev_trace:]
        prev_trace = len(compiler.trace)
        cell["output"] = compiler.output[prev_output:]
        prev_output = len(compiler.output)

        cells.append(cell)
        if not cell["ok"]:
            break

    propositions = {
        name_: {
            "motions": [{"name": m.name, "text": m.text,
                         "score": state.score(m.name)}
                        for m in state.motions],
            "verdicts": [{"motion": v.motion, "stance": v.stance,
                          "confidence": v.confidence, "line": v.line}
                         for v in state.verdicts],
        }
        for name_, state in compiler.propositions.items()
    }

    return {
        "name": name,
        "cells": cells,
        "final": {
            "store": prev_store,
            "output": compiler.output,
            "propositions": propositions,
            "points": {n: render(p) for n, p in compiler.points.items()},
            "trace_length": len(compiler.trace),
        },
    }


def main() -> int:
    out: List[Dict[str, Any]] = []

    # Built exactly as measure.py builds it, so the traces dumped here are the
    # traces the measured records came from rather than a parallel corpus.
    for spec in sweep():
        src = make_script(spec["n_items"], spec["n_funxns"],
                          spec["n_motions"], spec["depth"], spec["seed"])
        out.append(dump(spec["name"], src))

    for name, source in EDGE:
        try:
            out.append(dump(f"edge-{name}", source))
        except Exception as e:                       # noqa: BLE001
            # A RuntimeErr is caught inside `dump` and recorded; anything
            # reaching here is a bug in this file or an escaping ReturnSignal,
            # and dumping a corpus with a hole in it would be worse than
            # stopping.
            print(f"edge-{name} raised {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1

    for ex in ("signal", "assay"):
        p = ROOT / "prototype" / "examples" / f"{ex}.tb"
        out.append(dump(ex, p.read_text(encoding="utf-8")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scripts": out}, indent=1), encoding="utf-8")

    cells = sum(len(s["cells"]) for s in out)
    events = sum(s["final"]["trace_length"] for s in out)
    failed = sum(1 for s in out for c in s["cells"] if not c.get("ok"))
    print(f"{len(out)} scripts, {cells} cells, {events} events, "
          f"{failed} failing cells -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
