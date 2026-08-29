# Panel captions

Every value plotted is measured. The corpus is 28 Turbulance scripts — a size
ladder, a depth family, a width family, and the two hand-written examples — run
through the prototype's own compiler, producing 1780 trace events. Graphs span
6–33 items and 6–42 contacts; traces span 19–137 events. Nothing here is drawn
by hand, and no chart is a diagram, a table, or a text box.

The corpus is generated without an RNG, so it is fixed: `python measure.py &&
python panels.py` reproduces these figures exactly.

---

## Panel 1 — the script is the graph

**Static and executed graphs coincide on every script in the corpus.**

**(a)** Graph size after parsing alone against graph size after running, for all
28 scripts. Items (open circles) lie on the identity line without exception:
parsing recovers the same item set, with the same kinds, that execution
produces. Contacts (filled) are plotted on the same axes for scale. This is
Prop. 2.4 checked on the corpus rather than on the two examples — the whole
content of the claim that the script *is* the graph, since a graph needing
execution to exist could not be read off a script before running it.

**(b)** Items in the stage graph against stage, one line per script, coloured by
family. Every line is non-decreasing, and the underlying check is stronger than
the picture: no stage loses an item or weakens a contact in any of the 28
scripts (Prop. 2.7). The three families separate by construction — the depth
family holds item count fixed while lengthening the trace, so its lines run
flat where the ladder and width lines climb.

**(c)** The accretion surface over the size ladder: stage on one axis, script on
the other, items in the stage graph as height. The surface is monotone in
stage along every script, which is prefix containment seen as a shape rather
than as twelve separate curves. It rises to 33 items at the top of the ladder.

**(d)** Contacts against items, by family, with a least-squares line
(1.23·items − 1.25, r = 0.95). Contacts grow linearly in items rather than
quadratically: the graph a script induces stays sparse as the script grows,
which is what makes the reachability work in Panels 3 and 4 cheap.

---

## Panel 2 — the trace

**What the record contains, and how much of it there is.**

**(a)** Trace events against items, by family. The families separate cleanly:
the depth family reaches ~100 events at 16 items, where the width family is
still near 50 at 23 items. Trace length is driven by nesting, not by item
count — the two are independent axes of cost, which is why the corpus varies
them separately.

**(b)** Rule composition across the size ladder: script on one axis, the seven
most frequent of 15 rule kinds on the other, event count as height. The mix is
stable in shape as scripts grow: from the smallest ladder script to the largest,
7 items to 33, `given` moves from 27% to 24% of events and `declare-item` from
18% to 19%. The trace lengthens by repeating the same rules rather than by
recruiting new ones. Each bar counts events carrying that rule label in the
emitted JSON.

**(c)** Distribution of items read per event, over all 1780 events. 95.8% read
two items or fewer, and the mean is 0.89. Each event carries the five fields of
Def. 2.6 — rule, site, items read, item written, value written — so this is the
width of the contact each event realises. A narrow distribution is what makes an
atomic question a projection of the JSON (Thm. 5.2): *which rule wrote `hits` on
its second iteration, reading what?* is a filter over these events, not an
inference from them.

**(d)** Record against stage, one line per script, with the reference rec =
stage + 1 dashed; dots mark where each script stops. Every script traces the
reference exactly — the record strictly increases, 1, 2, 3, …, with no repeat,
across up to 10 stages (Def. 4.1). One committing act per stage, and the
counter never stalls.

---

## Panel 3 — seek is not nec

**The trace's operator over-reports by a factor of four.**

Across 74 motion targets: `seek` returns 4.22 items on average, where 1.04 are
load-bearing. `nec` here is computed from the *graph* by direct ablation —
remove one item, re-reach — which is the dominator characterisation of Prop.
6.4. The prototype itself computes only `seek` and labels its output as such
(Rem. 9.3), because Thm. 5.4 shows the *trace* cannot express `nec`. That is a
statement about traces, not about graphs; computing `nec` from the graph is
what lets these charts plot the gap rather than assert it.

**(a)** `nec` against `seek`, one point per target, with the diagonal marking
where the two operators would agree. Every point lies on or below it, and
almost all lie far below: `seek` ranges to 7 while `nec` never exceeds 2. The
points are jittered slightly so that coincident targets remain countable.

**(b)** The over-report `seek − nec` per target. Mean 3.18 (dashed), maximum 6.
The distribution has almost no mass at zero — `nec < seek` in 71 of 74 targets,
and the 3 exceptions are exactly the targets with two load-bearing items.

**(c)** Both operators' means against graph size, with the gap filled. `seek`
climbs from 2 to 7 as graphs grow from 6 to 33 items; `nec` rises to 2 and then
stays pinned at 1. The gap widens with graph size, so this is not a small-graph
artefact: reporting reachability as necessity gets worse the more there is to
report.

**(d)** The over-report as a surface over graph size and seek breadth,
smoothed by inverse-distance weighting (a smoothing of the scatter, not a model
fitted to it), with floor contours. The surface is monotone in both directions
and has no interior structure — the gap is a consequence of breadth, not of any
particular script shape. The obstruction in Thm. 5.4 is expressibility, not
complexity: no amount of trace detail closes this gap, because the trace does
not carry the information `nec` needs.

---

## Panel 4 — cost

**What the runtime costs, measured on the same corpus.**

**(a)** Run time over the (items, events) plane, with the 28 measured scripts
overlaid as points. The surface is an inverse-distance smoothing of those
points. Cost rises with events far more steeply than with items — the plane
tilts along one axis — which is the trace, not the graph, doing the work.

**(b)** Compile-and-run against parse-only, against trace length. Parsing is
0.70–4.84 ms and executing 1.11–12.47 ms, a mean ratio of 1.75. The static pass
that builds the graph is the cheaper half of the run, which is what makes
Panel 1(a) practical: the graph can be had for well under the price of the
answer.

**(c)** Ablation cost against graph size, with a quadratic reference. The direct
`nec` computation is quadratic by construction — one re-reach per item — and
tops out at 4.14 ms on the largest graph. The paper's near-linear claim (Prop.
6.4) is about Lengauer–Tarjan; this measures the direct method, which is the one
that cannot be wrong about the *set*, and it is already cheap at corpus scale.

**(d)** Events per item by family, with means marked. The depth family costs
4.80 events per item against the width family's 3.10, and the two hand-written
examples sit at 3.63 — inside the synthetic range rather than outside it, which
is the check that the corpus is representative of the scripts it generalises.
