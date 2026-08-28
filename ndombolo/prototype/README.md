# ndombolo prototype

A Python reference implementation of the deterministic core described in
[`../docs/semantic-runtime-graph/semantic-runtime-graph.tex`](../docs/semantic-runtime-graph/semantic-runtime-graph.tex).
It runs cells of Turbulance, compiles each with a deterministic compiler, and
stores the results as JSON.

No dependencies beyond the standard library.

```
python run.py examples/signal.tb        # writes out/signal.json
python run.py examples/assay.tb --quiet
python test_prototype.py                # the paper's claims, as checks
```

## Modules

| File | Paper |
|---|---|
| `lexer.py` | layout-significant scanner; brackets suspend layout |
| `parser.py` | statement dispatch + expression cascade → AST |
| `graph.py` | the static script graph — Prop. 2.4, Prop. 2.7 |
| `evaluator.py` | the deterministic compiler 𝒦 — Def. 3.2, Def. 2.6 (trace) |
| `run.py` | cell-wise runner and JSON emitter — §9.2 |
| `test_prototype.py` | eight properties the manuscript asserts |

## Cells

A line beginning `// ---` opens the next cell. Line numbers stay
script-absolute across the split, so a trace site names a place in the script
rather than an offset into whichever cell produced it. A script with no
separator is a single cell.

## Emitted JSON

One object per stage, per §9.2:

```jsonc
{
  "stage": 2, "line": 17, "source": "...", "ok": true,
  "store_delta": { "Strong": { "change": "added", "value": ..., "tag": "motion" } },
  "trace": [ { "seq": 3, "rule": "declare-motion", "site": {"line": 19},
               "reads": [], "writes": {"item": "Strong", "value": ..., "tag": "motion"} } ],
  "graph":  { "items": [...], "contacts": [{"from": ..., "to": ..., "weight": 1}] },
  "deposit": { "committed": true, "events": 16, "items_touched": ["Strong"] },
  "record": 3
}
```

Every trace event carries the five fields of Def. 2.6 — rule, site, items read,
item written, value written — plus `seq` for application order. That is what
makes an atomic question a projection of the JSON (Thm. 5.2): *which rule wrote
`hits` on its second iteration, reading what?* is a filter, not an inference.

## Two deliberate omissions

Per Rem. 9.3, both are absences the paper requires, not gaps:

- **No model is invoked.** The prototype emits the record a model would read.
  `deposit` is a counter standing in for `rec` — the interface is the deposit
  map alone, so the missing half is separable rather than entangled.
- **`nec` is not implemented.** The prototype computes `seek` (forward
  reachability) and labels it as such in the emitted `reachability` block.
  Thm. 5.4 shows the trace cannot express `nec`; reporting reachability *as*
  necessity would be reporting a number that was not computed.

## What the tests check

`test_prototype.py` runs each property over every example:

- **determinism** — one source, two runs, byte-identical results (Premise 1)
- **cell-wise ≡ whole-script** — store, output, propositions, points, trace and
  graph all agree (Thm. 9.4). Per-item `stage` labels are exempt: a one-cell
  run has no stages to label, which is a difference in what was *recorded*, not
  in what ran.
- **prefix containment** — no stage loses an item or weakens a contact (Prop. 2.7)
- **record strictly increases** — 1, 2, 3, … with no repeat (Def. 4.1)
- **graph needs no execution** — the graph built by parsing alone equals the
  graph after running (Prop. 2.4). This is the whole content of "the script
  *is* the graph".
- **sites are script-absolute**, **`seek` not `nec`**, **examples run clean**

## The deterministic fragment

Implemented: `item`, `funxn`, `given`/`otherwise`, `considering` (`all`,
`these`, default single), `for each`, `while`, `within`, `return`, `ensure`,
`point`, `proposition`/`motion`/`support`/`contradict`/`inconclusive`,
`resolve`, the `|>` pipe, and builtins `print len sum min max abs round`.

Motion scores aggregate by noisy-or over supports, discounted by noisy-or over
contradictions: `(1 - Π(1-sᵢ)) · Π(1-cⱼ)`.

Keywords the grammar recognises but the deterministic core gives no meaning —
`allow`, `research`, `cause`, `goal`, `metacognitive`, `resolution` — parse to
a no-op so a script using them still runs. They are the points at which a
non-deterministic layer would attach.
