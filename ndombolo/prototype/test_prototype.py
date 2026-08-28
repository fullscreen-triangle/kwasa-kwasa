"""
Checks of the properties the manuscript asserts of the prototype.

Each test names the result it exercises. They are properties of the runtime,
not of the examples: an example is only the witness.

    python test_prototype.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import GraphBuilder
from parser import parse
from run import run_script, split_cells

HERE = Path(__file__).resolve().parent
EXAMPLES = sorted((HERE / "examples").glob("*.tb"))


def _blank_separators(src: str) -> str:
    """Whole-script text, line-identical to the cellwise source."""
    return chr(10).join("" if l.strip().startswith("// ---") else l
                        for l in src.split(chr(10)))


def test_determinism():
    """Premise 1: the same source yields the same run, byte for byte."""
    for path in EXAMPLES:
        src = path.read_text(encoding="utf-8")
        a, b = run_script(src), run_script(src)
        assert a == b, f"{path.name}: two runs of one source disagreed"


def test_cellwise_equals_whole_script():
    """Thm. 9.4: cell-wise execution agrees with whole-script execution.

    The stage labels on graph items are exempt: a one-cell run has no stages
    to label, which is a difference in what was recorded, not in what ran.
    """
    for path in EXAMPLES:
        src = path.read_text(encoding="utf-8")
        whole = run_script(_blank_separators(src))
        cellwise = run_script(src)

        for field in ("store", "output", "propositions", "points"):
            assert whole["final"][field] == cellwise["final"][field],                 f"{path.name}: {field} differs"

        ev = lambda r: [(e["rule"], e["site"]["line"], e.get("writes"), e["reads"])
                        for st in r["stages"] for e in st["trace"]]
        assert ev(whole) == ev(cellwise), f"{path.name}: traces differ"

        bare = lambda g: (
            {(i["name"], i["kind"], i["line"]) for i in g["items"]},
            {(c["from"], c["to"], c["weight"]) for c in g["contacts"]})
        assert bare(whole["final"]["graph"]) == bare(cellwise["final"]["graph"]),             f"{path.name}: graphs differ"


def test_prefix_containment():
    """Prop. 2.7: each stage graph contains every earlier stage graph."""
    for path in EXAMPLES:
        r = run_script(path.read_text(encoding="utf-8"))
        prev_items, prev_contacts = set(), {}
        for st in r["stages"]:
            g = st["graph"]
            items = {i["name"] for i in g["items"]}
            contacts = {(c["from"], c["to"]): c["weight"] for c in g["contacts"]}
            assert prev_items <= items, f"{path.name}: stage {st['stage']} lost items"
            for key, w in prev_contacts.items():
                assert contacts.get(key, -1) >= w,                     f"{path.name}: stage {st['stage']} weakened {key}"
            prev_items, prev_contacts = items, contacts


def test_record_strictly_increases():
    """Def. 4.1: rec advances on every committing act and never repeats."""
    for path in EXAMPLES:
        r = run_script(path.read_text(encoding="utf-8"))
        recs = [st["record"] for st in r["stages"] if "record" in st]
        assert recs == sorted(set(recs)) and recs == list(range(1, len(recs) + 1)),             f"{path.name}: record did not strictly increase: {recs}"


def test_graph_needs_no_execution():
    """Prop. 2.4: the graph is recoverable from the source alone."""
    for path in EXAMPLES:
        src = path.read_text(encoding="utf-8")
        builder = GraphBuilder()
        for k, (first_line, text) in enumerate(split_cells(src)):
            builder.add_cell(parse(text, first_line), k)
        static = builder.graph.to_json()
        ran = run_script(src)["final"]["graph"]
        assert static == ran, f"{path.name}: static graph differs from run graph"


def test_sites_are_script_absolute():
    """A trace site names a line of the script, not an offset into a cell."""
    for path in EXAMPLES:
        src = path.read_text(encoding="utf-8")
        n = len(src.split(chr(10)))
        r = run_script(src)
        for st in r["stages"]:
            for e in st["trace"]:
                line = e["site"]["line"]
                assert 1 <= line <= n,                     f"{path.name}: site {line} outside 1..{n}"
            if st["stage"] > 0 and st["trace"]:
                assert max(e["site"]["line"] for e in st["trace"]) >= st["line"],                     f"{path.name}: stage {st['stage']} sites precede its first line"


def test_reports_seek_not_nec():
    """Rem. 9.3: the prototype computes forward reachability and says so.

    Thm. 5.4 shows the trace cannot express `nec`, so reporting one would be
    reporting a number the prototype did not compute.
    """
    for path in EXAMPLES:
        r = run_script(path.read_text(encoding="utf-8"))
        assert r["reachability"]["operator"] == "seek"
        assert "nec" not in r["final"]


def test_examples_run_clean():
    for path in EXAMPLES:
        r = run_script(path.read_text(encoding="utf-8"), str(path))
        assert r["ok"], f"{path.name}: {[s.get('error') for s in r['stages']]}"


def main() -> int:
    assert EXAMPLES, "no examples found"
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed "
          f"over {len(EXAMPLES)} example(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
