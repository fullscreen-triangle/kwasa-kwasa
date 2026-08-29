"""
Run the corpus and measure it. Every number the panels plot is produced here.

Two of the measurements go beyond what the prototype computes, deliberately:

  * `nec` by dominator tree. The prototype implements `seek` only and says so
    (paper, Rem. 9.3), because the *trace* cannot express `nec` (Thm. 5.4).
    That is a statement about traces, not about graphs -- given the graph, the
    dominator characterisation of Prop. 6.4 is computable, and computing it
    here is what lets us plot the gap between the two operators rather than
    merely assert it.
  * a static-only pass, parsing without executing, to check Prop. 2.4 on every
    script in the corpus rather than on the two examples.

Results land in measurements.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototype"))
sys.path.insert(0, str(HERE))

from corpus import make_script, sweep          # noqa: E402
from graph import GraphBuilder                 # noqa: E402
from parser import parse                       # noqa: E402
from run import run_script, seek, split_cells  # noqa: E402


# -- nec, by dominators (paper, Prop. 6.4) -------------------------------


def _adj(contacts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    a: Dict[str, List[str]] = {}
    for c in contacts:
        a.setdefault(c["from"], []).append(c["to"])
    return a


def _reach(adj: Dict[str, List[str]], seeds: List[str]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(seeds)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return seen


def nec(items: List[str], contacts: List[Dict[str, Any]],
        seeds: List[str], target: str) -> Set[str]:
    """Items whose removal disconnects `target` from the seeds.

    This is the dominator set of `target`, computed by the direct definition
    (remove one item, re-reach) rather than Lengauer-Tarjan. The paper's
    near-linear claim is about the algorithm; correctness of the *set* is what
    matters for the gap we plot, and the direct method is the one that cannot
    be wrong about it.
    """
    adj = _adj(contacts)
    if target not in _reach(adj, seeds):
        return set()
    out: Set[str] = set()
    for u in items:
        if u == target or u in seeds:
            continue
        pruned = [c for c in contacts if c["from"] != u and c["to"] != u]
        if target not in _reach(_adj(pruned), seeds):
            out.add(u)
    return out


# -- one script ----------------------------------------------------------


def measure(name: str, family: str, source: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = run_script(source, name)
    run_ms = (time.perf_counter() - t0) * 1000.0

    g = res["final"]["graph"]
    items = [i["name"] for i in g["items"]]
    seeds = [i["name"] for i in g["items"] if i["stage"] == 0]

    # -- static graph, no execution at all (Prop. 2.4)
    t1 = time.perf_counter()
    static = GraphBuilder()
    for k, (first_line, text) in enumerate(split_cells(source)):
        static.add_cell(parse(text, first_line), k)
    static_ms = (time.perf_counter() - t1) * 1000.0
    sg = static.graph.to_json()
    static_matches = (
        [(i["name"], i["kind"]) for i in sg["items"]]
        == [(i["name"], i["kind"]) for i in g["items"]]
        and sg["contacts"] == g["contacts"]
    )

    # -- seek vs nec, over every motion in the script
    ops: List[Dict[str, Any]] = []
    t2 = time.perf_counter()
    for it in g["items"]:
        if it["kind"] != "motion":
            continue
        tgt = it["name"]
        reached = _reach(_adj(g["contacts"]), [tgt])
        loadbearing = nec(items, g["contacts"], seeds, tgt)
        ops.append({"target": tgt, "seek": len(reached),
                    "nec": len(loadbearing)})
    nec_ms = (time.perf_counter() - t2) * 1000.0

    # -- per-stage series (Prop. 2.7, Def. 4.1)
    stages = [{
        "stage": s["stage"],
        "events": len(s["trace"]),
        "items": s["graph"]["item_count"],
        "contacts": s["graph"]["contact_count"],
        "record": s["record"],
        "touched": len(s["deposit"]["items_touched"]),
    } for s in res["stages"]]

    rules: Dict[str, int] = {}
    for s in res["stages"]:
        for e in s["trace"]:
            rules[e["rule"]] = rules.get(e["rule"], 0) + 1

    # -- reads per event: the width of the contact each event realises
    reads = [len(e["reads"]) for s in res["stages"] for e in s["trace"]]

    return {
        "name": name, "family": family, "ok": res["ok"],
        "cells": res["cells"],
        "items": g["item_count"], "contacts": g["contact_count"],
        "events": res["final"]["trace_length"],
        "record": res["final"]["record"],
        "run_ms": run_ms, "static_ms": static_ms, "nec_ms": nec_ms,
        "static_matches_run": static_matches,
        "stages": stages, "rules": rules, "ops": ops,
        "reads_per_event": reads,
        "density": (g["contact_count"] / max(g["item_count"] ** 2, 1)),
    }


def main() -> int:
    rows: List[Dict[str, Any]] = []

    for spec in sweep():
        src = make_script(spec["n_items"], spec["n_funxns"],
                          spec["n_motions"], spec["depth"], spec["seed"])
        rows.append(measure(spec["name"], spec["family"], src))

    # The two hand-written examples, measured the same way.
    for ex in ("signal", "assay"):
        p = HERE.parent / "prototype" / "examples" / f"{ex}.tb"
        rows.append(measure(ex, "example", p.read_text(encoding="utf-8")))

    bad = [r["name"] for r in rows if not r["ok"]]
    mismatch = [r["name"] for r in rows if not r["static_matches_run"]]

    out = {
        "scripts": rows,
        "checks": {
            "all_ran": not bad, "failed": bad,
            "static_equals_run": not mismatch, "mismatched": mismatch,
            "n_scripts": len(rows),
            "total_events": sum(r["events"] for r in rows),
        },
    }
    (HERE / "measurements.json").write_text(json.dumps(out, indent=2),
                                            encoding="utf-8")

    print(f"{len(rows)} scripts, {out['checks']['total_events']} events")
    print(f"  all ran clean:        {not bad}" + (f"  {bad}" if bad else ""))
    print(f"  static == run graph:  {not mismatch}"
          + (f"  {mismatch}" if mismatch else ""))
    return 0 if (not bad and not mismatch) else 1


if __name__ == "__main__":
    raise SystemExit(main())
