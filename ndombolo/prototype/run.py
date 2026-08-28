"""
The cell-wise runner (paper, Sec. 9).

Splits a script into cells, compiles each with the deterministic compiler, and
emits one JSON object per stage carrying the stage index, the store delta, the
trace, the stage graph, the deposit summary, and the record counter.

Two things this prototype deliberately does NOT do (paper, Rem. 9.3):

  * it never invokes a model -- the deposit map is a counter, standing in for
    the record without pretending to be one;
  * it computes `seek` (forward reachability) and never `nec` (backward
    ablation), and says so in the JSON it writes. Per Thm. 5.4 the trace cannot
    express `nec`, so a prototype that reported one would be reporting a
    number it did not compute.

Usage:
    python run.py examples/signal.tb
    python run.py examples/signal.tb --out out/signal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import Compiler, RuntimeErr, render, tag          # noqa: E402
from graph import GraphBuilder                                   # noqa: E402
from lexer import LexError                                       # noqa: E402
from parser import ParseError, parse                             # noqa: E402

CELL_SEP = "// ---"          # a line beginning with this opens the next cell


# -- cell splitting ------------------------------------------------------


def split_cells(source: str) -> List[Tuple[int, str]]:
    """Split on separator lines, keeping each cell's first line number.

    Line numbers stay script-absolute so that a trace site names a place in
    the script, not an offset into whichever cell happened to produce it.
    """
    cells: List[Tuple[int, str]] = []
    current: List[str] = []
    start = 1
    text = source.replace("\r\n", "\n").split("\n")
    for lineno, line in enumerate(text, 1):
        if line.strip().startswith(CELL_SEP):
            cells.append((start, "\n".join(current)))
            current = []
            start = lineno + 1
            continue
        current.append(line)
    cells.append((start, "\n".join(current)))
    kept = [(ln, c) for ln, c in cells if c.strip()]
    return kept or [(1, source)]


# -- reachability --------------------------------------------------------


def seek(graph, seeds: List[str]) -> List[str]:
    """Forward reachability from `seeds` over contacts (paper, Thm. 5.4).

    This is `seek`, not `nec`. It answers "what did this touch", never "what
    was load-bearing" -- the trace does not carry the information the latter
    needs.
    """
    adj: Dict[str, List[str]] = {}
    for (a, b) in graph.contacts:
        adj.setdefault(a, []).append(b)
    seen: Set[str] = set()
    stack = [s for s in seeds if s in graph.items]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return sorted(seen)


# -- the run -------------------------------------------------------------


def run_script(source: str, path: str = "<script>") -> Dict[str, Any]:
    cells = split_cells(source)
    compiler = Compiler()
    builder = GraphBuilder()

    stages: List[Dict[str, Any]] = []
    record = 0                    # rec: the deposit counter, strictly increasing
    # Builtins are the initial environment, not something a cell wrote.
    prev_store: Dict[str, Any] = {n: render(v)
                                  for n, v in compiler.env.flatten().items()}
    prev_trace = 0
    prev_output = 0

    for k, (first_line, text) in enumerate(cells):
        stage: Dict[str, Any] = {"stage": k, "line": first_line,
                                 "source": text.strip()}

        try:
            body = parse(text, first_line)
        except (LexError, ParseError) as e:
            stage.update(ok=False, error={"phase": "parse", "message": str(e),
                                          "line": e.line})
            stages.append(stage)
            break

        builder.add_cell(body, k)

        try:
            compiler.exec_block(body)
            stage["ok"] = True
        except RuntimeErr as e:
            stage.update(ok=False, error={"phase": "run", "message": str(e),
                                          "line": e.line})

        # -- store delta: what this stage added or changed
        store = compiler.env.flatten()
        delta: Dict[str, Any] = {}
        for name, value in store.items():
            rendered = render(value)
            if name not in prev_store:
                delta[name] = {"change": "added", "value": rendered,
                               "tag": tag(value)}
            elif prev_store[name] != rendered:
                delta[name] = {"change": "updated", "from": prev_store[name],
                               "value": rendered, "tag": tag(value)}
        prev_store = {n: render(v) for n, v in store.items()}

        stage["store_delta"] = delta
        stage["trace"] = compiler.trace[prev_trace:]
        prev_trace = len(compiler.trace)

        stage["output"] = compiler.output[prev_output:]
        prev_output = len(compiler.output)

        # -- the stage graph, containing every earlier stage graph
        stage["graph"] = builder.graph.to_json()

        # -- the deposit: one committing act per stage, rec strictly increases
        record += 1
        stage["deposit"] = {
            "committed": True,
            "events": len(stage["trace"]),
            "items_touched": sorted(delta.keys()),
            "record_before": record - 1,
        }
        stage["record"] = record

        stages.append(stage)
        if not stage["ok"]:
            break

    propositions = {
        name: {
            "motions": [
                {"name": m.name, "text": m.text, "score": state.score(m.name)}
                for m in state.motions
            ],
            "verdicts": [
                {"motion": v.motion, "stance": v.stance,
                 "confidence": v.confidence, "line": v.line}
                for v in state.verdicts
            ],
        }
        for name, state in compiler.propositions.items()
    }

    graph_json = builder.graph.to_json()
    seeds = [i["name"] for i in graph_json["items"] if i["stage"] == 0]

    return {
        "source": path,
        "cells": len(cells),
        "ok": all(s.get("ok") for s in stages),
        "stages": stages,
        "final": {
            "store": prev_store,
            "output": compiler.output,
            "propositions": propositions,
            "points": {n: render(p) for n, p in compiler.points.items()},
            "graph": graph_json,
            "record": record,
            "trace_length": len(compiler.trace),
        },
        "reachability": {
            "operator": "seek",
            "note": ("forward reachability only; `nec` is not computed, "
                     "because the trace cannot express it (paper, Thm. 5.4)"),
            "seeds": seeds,
            "reached": seek(builder.graph, seeds),
        },
    }


# -- entry point ---------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run Turbulance cells, emit JSON.")
    ap.add_argument("script", help="path to a .tb script")
    ap.add_argument("--out", help="write JSON here (default: out/<name>.json)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the human-readable summary")
    args = ap.parse_args(argv)

    src_path = Path(args.script)
    if not src_path.exists():
        print(f"no such script: {src_path}", file=sys.stderr)
        return 2

    result = run_script(src_path.read_text(encoding="utf-8"), str(src_path))

    out_path = Path(args.out) if args.out else \
        Path(__file__).resolve().parent / "out" / f"{src_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not args.quiet:
        _summary(result, out_path)
    return 0 if result["ok"] else 1


def _summary(result: Dict[str, Any], out_path: Path) -> None:
    g = result["final"]["graph"]
    print(f"{result['source']}: {result['cells']} cell(s), "
          f"{'ok' if result['ok'] else 'FAILED'}")
    for st in result["stages"]:
        mark = "ok " if st.get("ok") else "ERR"
        n = len(st.get("trace", []))
        touched = ", ".join(st.get("deposit", {}).get("items_touched", []))
        print(f"  [{mark}] stage {st['stage']}: {n:>4} events  rec={st.get('record')}"
              + (f"  ->  {touched}" if touched else ""))
        if not st.get("ok"):
            print(f"        {st['error']['phase']}: {st['error']['message']}")
    for line in result["final"]["output"]:
        print(f"  | {line}")
    print(f"  graph: {g['item_count']} items, {g['contact_count']} contacts")
    print(f"  trace: {result['final']['trace_length']} events, "
          f"record={result['final']['record']}")
    print(f"  json:  {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
