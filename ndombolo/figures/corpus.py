"""
Generate a corpus of Turbulance scripts of controlled size.

The two hand-written examples are enough to demonstrate the runtime but not to
measure it: a claim about how cost scales needs a sweep, not two points. These
scripts are synthetic in shape but not in behaviour -- they are run by the same
compiler, and every number plotted downstream comes from that run.

Each script is parameterised by (n_items, n_funxns, n_motions, depth), which
between them move the item count, the trace length, and the graph density
independently.
"""

from __future__ import annotations

from typing import List


def _readings(n: int, seed: int) -> str:
    # A deterministic spread in [0.05, 0.95]; no RNG, so the corpus is fixed.
    vals = [0.05 + ((i * 37 + seed * 11) % 91) / 100.0 for i in range(n)]
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def make_script(n_items: int, n_funxns: int, n_motions: int,
                depth: int, seed: int = 0) -> str:
    """Build one script. Cells are separated by `// ---`."""
    cells: List[str] = []

    # -- cell 0: the seeds
    head = [f"item threshold = 0.{4 + seed % 5}"]
    for i in range(n_items):
        head.append(f"item series{i} = {_readings(3 + (i % 4), seed + i)}")
    cells.append("\n".join(head))

    # -- cell 1..: one funxn per cell, each with `depth` nested guards
    for f in range(n_funxns):
        lines = [f"funxn count{f}(xs, t):", "    item hits = 0",
                 "    considering all x in xs:"]
        pad = "        "
        for d in range(depth):
            lines.append(f"{pad}given x > t - 0.{d}:")
            pad += "    "
        lines.append(f"{pad}hits = hits + 1")
        lines.append("    return hits")
        cells.append("\n".join(lines))

    # -- final cell: a proposition whose motions are contested
    prop = ["proposition Claim:"]
    for m in range(n_motions):
        prop.append(f'    motion Motion{m}("motion {m} of the claim")')
    prop.append("")
    for m in range(n_motions):
        fn = f"count{m % max(n_funxns, 1)}" if n_funxns else None
        src = f"series{m % max(n_items, 1)}" if n_items else "[]"
        if fn:
            prop.append(f"    item tally{m} = {fn}({src}, threshold)")
            prop.append(f"    given tally{m} >= 1:")
            prop.append(f"        support Motion{m} with_confidence(0.{5 + m % 4})")
            prop.append("    otherwise:")
            prop.append(f"        inconclusive Motion{m}")
        if m % 3 == 2:
            prop.append(f"    contradict Motion{m} with_confidence(0.{3 + m % 5})")
    cells.append("\n".join(prop))

    return "\n\n// ---\n\n".join(cells) + "\n"


def sweep() -> List[dict]:
    """The corpus: a size ladder plus shape variations at fixed size."""
    out: List[dict] = []

    # Size ladder -- everything grows together.
    for k in range(1, 13):
        out.append({
            "name": f"ladder{k:02d}", "family": "ladder",
            "n_items": k, "n_funxns": max(1, k // 2),
            "n_motions": max(1, k // 2), "depth": 1 + (k % 3), "seed": k,
        })

    # Depth variation at fixed width -- moves trace length, not item count.
    for d in range(1, 7):
        out.append({
            "name": f"depth{d}", "family": "depth",
            "n_items": 4, "n_funxns": 3, "n_motions": 3, "depth": d, "seed": d,
        })

    # Width variation at fixed depth -- moves item count, not nesting.
    for w in range(1, 9):
        out.append({
            "name": f"width{w}", "family": "width",
            "n_items": w, "n_funxns": w, "n_motions": 2, "depth": 2, "seed": w,
        })

    return out


if __name__ == "__main__":
    print(make_script(2, 1, 2, 2, seed=1))
