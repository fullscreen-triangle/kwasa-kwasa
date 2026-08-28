# Ndombolo — the orchestration module of kwasa-kwasa

**Status:** design plan. No code written yet.

Ndombolo is a *mode* of the framework, not a new language. When the user is in
ndombolo, **only Turbulance scripts are written.** No `.ghd`/`.trb`/`.hre`/`.fs`
quartet to hand-author, no per-module DSLs to learn, no prompts. One language,
executed cell-wise, that speaks for the whole federation.

Where kwasa-kwasa is the conductor, ndombolo is the conducting — the module that
turns a person's intent into dispatches across the orchestra, and judges when
enough has been done.

---

## 0. The three names

| Name | What it is | Algebra |
|---|---|---|
| **Orchestra** | the code — chunks, modules, the players | — |
| **Ensemble** | all the code in all the cells; the durable node catalogue | convergence (idempotent join on τ) |
| **Conductor** | this framework; ndombolo is its baton | — |

Inside an ensemble live **goals** (mutable, nested, splitting and merging)
against a single **context** (invariant).

---

## 1. The load-bearing asymmetry

> **A goal can split into more goals that become one goal again.
> A node does not split into more nodes.**

This is the design's spine, and every structural decision below follows from it.

Nodes *converge*: two agents raising the same τ meet at a node that was already
there; merging chunk bags is idempotent and never forks anything. Goals *fan out
and fuse*: a goal elaborates into subgoals which are separately worked and later
joined. Those are two different algebras. Neither may be expressed in terms of
the other.

Therefore:

- **A goal is not a node, and not a tree of nodes.** A goal *ranges over* nodes;
  a node *participates in* goals. Many-to-many.
- **Goal fusion is cheap** precisely because nodes do not split: when two
  subgoals merge, any node both touched appears once, because it always *was*
  one node. There is no reconciliation problem — no conflicting copies to
  resolve, only a union of references into a catalogue that never forked.
- **Node merging is not a user-facing operation.** It is what convergence does
  automatically on τ-match. The language exposes no verb for it.
- **`consolidate` merges goals, never nodes.** Its real content is therefore not
  the node sets (those union for free) but the *constraints* and *sufficiency
  thresholds* the subgoals accumulated.

### What splits and merges, and how

| Component of a goal | Under split | Under merge |
|---|---|---|
| Node references | partition / overlap | union — free |
| Constraints | inherited from parent | **conjunction** — may be unsatisfiable |
| Context | does not participate | does not participate |
| Sufficiency threshold | **must be derived** (§6) | **must be recomputed** (§6) |

An unsatisfiable conjunction on merge is **not an error**. It is the split
reporting that the parent goal was incoherent — the most informative thing
`consolidate` can return, and the first place the necessity gate does real work.

Context not participating is what earns it its name: it is what makes two
subgoals *the same ensemble* rather than two unrelated activities running
concurrently.

---

## 2. Why nothing here contradicts the runtime

Ndombolo sits *above* the CKG runtime and must not weaken its guarantees.

- **Thm. 2 (no exit code) / Cor. 4 (run to completion)** — ndombolo introduces
  no predicate of correctness at the runtime layer. The typed-intent verbs (§4)
  select *which module realises an act*; they never judge its output. A throwing
  chunk still emits an error value; the graph still never halts.
- **Thm. 5 (trajectory emergence)** — goals are mutable and run-constituted;
  they are the trajectory side of the split and are *not* storable as a plan.
  The ensemble (nodes) is durable and freezable. Reproducibility still attaches
  to the node set, never to a goal history.
- **Prop. 7 (prefix containment)** — a cell is a length-k address prefix.
  Cell-wise execution is not a new affordance; it is the already-proved
  resolution control exposed at source-text granularity.
- **Prop. 8 / provenance** — the fingerprint continues to exclude values.
  Context is the linguistic form of exactly that: the thing whose invariance you
  check while goals and results vary freely.
- **No-Local-Necessity** — modules judge correctness; only the conductor judges
  necessity. Ndombolo *is* that gate, and §6 is where it stops being a socket.

---

## 3. Three kinds of keyword, not one flat list

The proposed vocabulary contains three categorically different things, and the
runtime treats them differently. Naming the split keeps the language honest —
otherwise users cannot predict which keywords cost a dispatch.

**Declarations** — shape a node *before* dispatch. Free; no act consumed.
`item` · `constant` · `fact` · `rule` · `constraint` · `assert`

**Acts** — become chunks; run; emit value-deltas. Cost acts.
`query_all` · `consolidate` · the typed-intent verbs (§4)

**Scoping** — determine address depth and what is in play. Free.
`given` (for) · `within`

`item`, `given`, `within` already exist as live Turbulance prefixes in
`route-input.js`. `constant`, `fact`, `rule`, `query_all`, `consolidate`,
`constraint`, `assert` are new.

The declaration/act/scope axis is **orthogonal** to the typed-intent axis. Typed
intent says *what kind of act*; this says *whether it is an act at all*.

---

## 4. Typed intent — a closed verb set

`analyze` · `generate` · `refactor` · `critique` · `translate` · `prove`

Closed so the speech act is never guessed. The verb selects the realising module
and the expected shape of the emitted delta; it does **not** rank or validate
what comes back.

Note the deliberate parallel: the module vocabulary is closed at four verbs
(identify / read / transform / emit) with no fifth verb of correctness. The
intent vocabulary is closed at six with no seventh verb of *approval*. Ndombolo
asks for kinds of work; it does not certify them.

---

## 5. The question structure

A question is a **partition**: a certain surround, a located gap, and a
criterion for when the gap is filled. Uncertainty is the condition for asking —
with nothing uncertain there is nothing to say. The four parts:

1. **Constraints** — the certain surround. First-class, never buried in prose.
2. **Typed intent + scope** — locate the gap. Which files/symbols/sections are
   in play, and which are `target` (mutable) versus `reference` / `example`
   (read-only).
3. **Output contract** — the sufficiency criterion. `diff` · `code` ·
   `json-schema` · `prose` · `structured-report`. If unspecified it is guessed,
   and guessing is where ambiguity re-enters.
4. **Emphasis** — a weighting on the gap.

**Context roles and scope are the same distinction, generalised** — do not build
both. `role="target"` is the mutation target; `role="reference"` and
`role="example"` are read-only context. One mechanism, two syntaxes at most.

### Decision: constraints bind the CHECKING, not the asking

Two independent arguments converge here, so this is settled rather than open:

1. Constraints held back and tested against the response are checkable
   **blindly** — a response can be rejected on constraint survival without ever
   reading it for meaning. That is the whole point of the floor: judgement
   without extraction.
2. `consolidate` requires it. Constraints merge by conjunction, which is
   well-defined and decidably unsatisfiable. Two *prompts* cannot be conjoined
   at all. If constraints bound the asking, goal fusion would have no semantics.

Constraints may additionally be shown to the model as a courtesy. They are
*enforced* on the way back.

---

## 6. Sufficiency is closure, not a threshold

This section was the plan's one open problem. The directional-pair paper closes
most of it, and the resolution is that the framing was wrong: **sufficiency is
not a number to be split and recombined.** It is a structural property of the
reachable class set, and it is decidable.

### 6.1 The stopping rule

A determination from a seed is **closed** when, for every probe available but
not yet invoked, extending by that probe cannot add a new equivalence class to
the reachable set. Not "the answer looks good" — "the space of answers has
stopped producing new destinations."

Closure is *strictly stronger* than any confidence threshold, and provably so.
A single internally-dense cluster reached through one probe reports terminal
alignment at the floor and therefore satisfies **any** threshold trivially,
while a second uninvoked probe reaches an entirely different class. Threshold
stopping halts at whichever source it happened to reach first and calls it
sufficient. That is the failure mode, stated exactly.

So `sufficiencyThreshold` on a `LiveTask` is the wrong shape of field. What
ndombolo's gate computes is not a scalar comparison but the closure test.

### 6.2 Termination has exactly two outcomes

Over a finite probe registry every determination terminates in one of:

- **Convergent closure** — the reachable class set stabilises to a single
  class; report its representative.
- **Contested closure** — it stabilises with more than one class; report
  **decline**, together with the distinct classes found.

Nothing runs forever, because the registry is finite and the class partition
can only grow finitely often.

**Decline is a first-class output, not a failure to retry.** By the no-verdict
theorem the system could not adjudicate between the classes even if instructed
to. And a contested closure has located precisely where independent probing
diverges — which is ordinarily worth more than either class alone, and is
exactly what a threshold procedure discards.

This lands directly on §1's table. An unsatisfiable constraint conjunction on
merge was already identified there as informative rather than erroneous. It is
the same phenomenon: contested closure at the goal layer.

### 6.3 What split and merge actually do

With closure as the criterion, the two missing functions change character:

- **`split_threshold`** — mostly dissolves. Subgoals do not inherit a fraction
  of a parent's number; each closes against its own reachable class set. What a
  split *does* distribute is the **probe budget**, and that is a solved
  allocation problem, not an open one (§6.4).
- **`merge_sufficiency`** — becomes a class-set union followed by a
  re-partition under endpoint-indistinguishability. The parent is convergently
  closed iff the union collapses to one class. If subgoals closed convergently
  but on *mutually distinguishable* representatives, the parent is
  **contested** — which is the correct and informative answer, not a
  contradiction to resolve.

The residual open piece is small and concrete: ndombolo must supply the
endpoint-indistinguishability test at the goal layer — when do two subgoal
results count as the same class? That is a real decision, but it is one
predicate, not two functions.

### 6.4 Budget division, when it is needed

Closure says *when to stop*; it does not say *which probe next* under cost. Two
regimes, and the paper is explicit about which applies:

- **Graded probes** (effort is continuous, returns diminish) — optimal division
  is **water-filling**: equalise marginal gain across engaged probes at a single
  shadow price, and drop any probe whose first unit of effort returns less than
  that price. The price falls as budget rises and rises as competing probes
  multiply.
- **All-or-nothing probes** (a threshold below which the probe returns nothing)
  — diminishing returns fails, water-filling does not apply, and the correct
  model is a **0/1 knapsack**. Admit in decreasing order of
  `(1/cost) · log(Ω / (Ω − floor))`.

The concavity premise is a claim about the environment being probed, not about
ndombolo. It should be recorded per probe kind rather than assumed globally.
A `purpose ask` is closer to all-or-nothing; an LLM act with a token budget is
closer to graded.

**Budget exhaustion is never the stopping condition.** A determination that
does not settle to within the floor should descend further or decline — never
fabricate. Budget selects the *next* probe; closure decides *whether to stop*.

---

## 7. AI enters as the process side of one table

This is the section that answers "how does AI fit in," and the answer is that
the question is malformed. There is no component to bolt on.

### 7.1 One table at two settings

A causal knowledge graph and a generative model are **the same table evaluated
at two settings** of the process parameter. Restricted to a single item with
the rest of the process held at rest, the table *is* the graph — the resting
cut around that item. Opened to a nontrivial process, the same table is the
model. The catalogue is the model at rest; the model is the catalogue in
motion.

So the pair is `(Graph, Model)`, and what joins them is not an interface but a
**deposit map**: every propagation on the process side commits its residue back
into the graph as new contacts. There is no protocol, no adapter, no boundary
to cross. Direction is strict — **the catalogue accretes; the process side
consumes and deposits.** Residue never flows the other way.

This is the formal grounding of the naming the framework already had:

| Framework name | Pair side |
|---|---|
| **Ensemble** (durable node catalogue) | Graph — the resting side |
| **The LLM / the acts** | Model — the process side |
| **Ndombolo's gate** | the deposit map plus the closure test |

### 7.2 The consequence that matters most

**The model alone has no invariant and no record.** It is a family of
evaluations, not a structure. The pair's invariant is the graph's invariant;
the pair's record is the graph's record. *The catalogue is the process side's
identity.*

Which settles a question the framework would otherwise have had to argue about
at length: the LLM is not a participant with standing in the ensemble. It has
no τ, occupies no node, and cannot be a party to convergence. It is how
propagation happens. What persists of it is exactly what it deposited.

Note also what this forbids. A **stateless** predictor — perfectly repeatable
and leaving no trace — occupies a formally forbidden position: it would require
both a zero floor and a non-advancing record, each independently impossible. An
LLM call in ndombolo is therefore *never* modelled as a pure function. It
advances the record. That is not an implementation detail to be optimised away;
it is what makes the pair admissible at all.

### 7.3 Six invariants ndombolo must satisfy

Each is checkable, and none follows from the others.

| Invariant | Commitment | Where it bites in this plan |
|---|---|---|
| **Conserved invariant** | something is conserved under relabelling | τ-identity; the provenance fingerprint that excludes values (§2) |
| **Never-resetting record** | the record only advances; "undo" is a *compensating commit that increments* | §9.2 — session reset must be a new commit, never a rewind |
| **Determination by search** | no cache of prior determinations; determine, do not look up | forbids memoising `query_all`; forbids a "we asked this already" shortcut |
| **Deposit on every propagation** | every act commits residue | every LLM act writes back, including ones whose output is discarded |
| **Accretive update** | no teardown-then-rebuild; a resting cut exists at every intermediate stage | §9.2's accretion recommendation, now a hard requirement |
| **No verdict** | no success/failure vocabulary | already the runtime's no-exit-code theorem (§2) and the closed verb sets (§4) |

Two of these have immediate teeth. **Determination by search** means the
obvious optimisation — cache the answer to a repeated question — is not an
optimisation but a violation. And **repeated queries need not agree, and that
is correct**: the first determination committed cuts that altered the graph, so
the second is being asked of a different structure. Divergence between two runs
of the same cell is *not* a bug report. Reproducibility relocates from "same
answer" to protocol plus provenance.

That is the sharpest confirmation of the accretion decision, arriving
independently: it was a recommendation, and it is now a requirement with a
theorem behind it.

---

## 8. Search — and the operator `query_all` is missing

Two separable capabilities. One is much harder than it looks, and one is absent
entirely.

### 8.1 Finding and pruning are different operations

`query_all` as previously planned — three retrieval primitives, chosen by
question kind — implements **`seek` alone**: forward reachability, *what does
this target touch.* That is precisely what RAG does, and it is half of correct
probing.

The other half is **`nec`**: backward ablation, *what does dropping this
change.* The two operators are provably distinct and provably disagree. And the
missing one **cannot be obtained by making the retriever better** — it is not a
quality problem.

Correct probing is the composition, in this order:

```
nec ∘ seek
```

Reach forward to the candidate set; then ablate to find which members are
load-bearing. Implementation is tractable: a non-seed item is necessary exactly
when it **dominates** some reachable item, and dominator trees are near-linear
(Lengauer–Tarjan).

Three warnings, each of which has already caught a real implementation:

- **Orientation.** The resolving measure must *grow* with resolving power. A
  min cut has the wrong sign — use the reachable-slice measure.
- **Do not count the item's own disappearance** when measuring its
  contribution. Doing so collapses `nec` onto `seek` and silently reproduces the
  bug the whole distinction exists to avoid. A validation suite caught exactly
  this.
- **Seeds are a genuine boundary case.** A seed that reaches only itself
  dominates nothing, yet is necessary. Seeds need separate handling; the
  dominance rule is stated for non-seed items.

For ndombolo this is a new verb, not a flag on an existing one. `query_all`
finds; something like `prune` or `necessary_of` ablates. A goal that has only
sought has not finished probing.

### 8.2 Why a store cannot close the gap

There is a hard limit here worth stating so nobody tries to engineer past it.
**Retrieval cannot express admissibility.** Two graphs with identical edge sets
— identical to *every* retrieval query — yield opposite verdicts, because the
floor is a minimum over every item, including items lying on no path at all.
Adding numeric attributes to edges does not help: *the obstruction is
expressibility, not hardness.* The same separation extends to corpus-trained
predictors.

Stated plainly: **accountability is a global property expressed in local terms,
and a store records only the local terms.**

This is the principled reason `purpose` and `spraypaint` have the blind spots
they have, and the reason the fix is a second operator rather than a better
index.

### 8.3 Finding files — the blind spot

`purpose` does not index filenames or paths. `spraypaint` is the same:
filenames and paths contribute nothing to ranking. **Neither tool answers
"where is the file called X."**

| Question | Primitive |
|---|---|
| where is `X` defined | `purpose ask` — one stem, never a question |
| which passage argues this | `spraypaint ask` |
| what file is named like this | **Glob** — must be added; neither tool does it |
| which modules is this goal about / what breaks if I touch this | `purpose ckg ask` / `ckg why` |
| **what here is load-bearing** | **`nec` — does not exist yet (§8.1)** |

Replacing Windows Explorer is the Glob row. The last row is the one the theory
says is missing, and it is the more consequential absence.

### 8.4 Safety rails on orchestrated search — mandatory

`spraypaint` is dangerous under automation in two specific ways:

- **Monotone committed count, no decrement path.** Every committed ask
  permanently inflates it. Under orchestration the count is no longer under a
  human's eye. → **`--dry-run` is the default in any ndombolo-issued ask;**
  committing is an explicit opt-in.
- **`.gitignore` is not consulted, and it re-reads snippets from disk at query
  time.** A dry run over credential-shaped terms was observed to allocate a
  `.claude/settings.local.json` passage; a committed ask would have printed a
  live credential. → **credential-shaped terms are refused at the ndombolo
  layer,** before the subprocess. Never `spraypaint serve` on a non-loopback
  address.

Note the tension with §7.3's *determination by search* invariant, and that it
is only apparent: the invariant forbids caching *determinations*, not
rate-limiting *probes*. Dry-run-by-default changes which probes are invoked,
never whether a result is recomputed.

Also: **a miss proves nothing.** Neither tool indexes call sites, imports,
config values, or string literals. `query_all` must never report absence from a
`purpose`/`spraypaint` miss; it falls back to Grep/Glob before concluding
anything.

### 8.5 Writing files

docx / pdf / latex output is the **output contract made concrete** — the shape
you asked for, rendered. This is the cleanest place for the contract idea to
prove itself, because format conformance is mechanically checkable without
reading the content for meaning. Same principle as §5: blind judgement.

---

## 9. Implementation order

### 9.1 Session-persistent interpreter — the prerequisite

Nothing else can be built first. `run()` in `turbulance/index.js` constructs a
fresh `Interpreter` per call and discards it, so today it is
whole-script-or-nothing.

The good news: `Interpreter` state is small and entirely in its constructor —
`output`, `propositions`, `points`, `debateStack`, `global` (an `Env`),
`thresholds`, `files`, `onStatus`. Splitting `run()` into `createSession()` +
`evalInSession(session, cellSource)` is a modest, contained refactor. `run()`
stays as a thin wrapper over both, so existing callers and the eight tutorials
do not change.

Two things to get right:

- Cell splitting must respect block DSLs. `splitStatements()` already handles
  bracket depth and string state; explicit user cell delimiters largely moot the
  problem.
- Accumulating `output` / `propositions` / `points` across cells needs a
  per-cell watermark so a cell reports only what it produced.

### 9.2 Re-execution semantics — now settled by theorem

**Jupyter overwrites. The pair accretes.** This was a recommendation; §7.3 makes
it a requirement. Re-running a graph-touching cell adds chunks to the same node
and reports that it converged onto an existing node. Divergence from the
previous run is expected and correct, not a defect.

Session reset must be implemented as a **compensating commit that advances the
record**, never as a rewind. There is no rewind; the record does not decrease.

Pure-computation cells (no dispatch, no deposit) may be re-run freely; they are
just rebinding in `Env`.

### 9.3 Then, in order

1. `context` as an ensemble-level invariant, `goal` as a mutable nested
   container.
2. Declarations: `constant`, `fact`, `rule`, `constraint`, `assert` — free, no
   acts.
3. The deposit map (§7.1) and the six invariant predicates (§7.3), as checks
   that run in CI from the first commit. They are cheap now and expensive to
   retrofit.
4. `query_all` over the retrieval primitives, Glob first (§8.3), with the §8.4
   rails in place from the first commit, not retrofitted.
5. `nec` — the ablation operator (§8.1). Dominator-based; the three warnings as
   test cases *before* the implementation, since two of them fail silently.
6. Typed intent verbs + scope/role + output contract.
7. `consolidate` — constraint conjunction, unsatisfiability as a real result.
8. The closure test and decline as a first-class outcome (§6). No longer last
   because it is undetermined — it is not — but it depends on 5, since closure
   is a statement about reachable classes.

---

## 10. Open

- **The endpoint-indistinguishability predicate at the goal layer** (§6.3).
  When do two subgoal results count as the same class? One predicate, and the
  only genuinely undetermined thing left.
- **Per-probe concavity** (§6.4). Which probe kinds are graded and which are
  all-or-nothing must be recorded, not assumed. Guessing wrong picks the wrong
  allocation rule.
- **Is an asked question a node?** Questions converge and answers accrete, which
  argues yes — but then question identity τ becomes load-bearing, and two
  differently-worded questions with the same intent must either merge or not.
  Deferred; nothing in §9.1–§9.3 depends on it.

---

## 11. Settled

- Goal ≠ node; many-to-many; goal fusion is free (§1).
- `consolidate` merges goals only; node merging has no verb (§1).
- Constraints bind the checking, not the asking (§5).
- Context roles and scope are one mechanism (§5).
- **Sufficiency is closure, not a threshold** (§6.1); threshold-stopping is a
  named failure mode with a proof.
- **Decline is a first-class output** (§6.2), carrying the divergent classes.
- **Budget never stops a determination** (§6.4); closure does.
- **AI is the process side of one table, joined by a deposit map, not an
  interface** (§7.1). The catalogue is its identity; it holds no standing in the
  ensemble.
- **An LLM call is never a pure function** (§7.2) — stateless prediction is a
  forbidden position.
- **Six invariants, checked in CI** (§7.3); no caching of determinations;
  repeated queries need not agree.
- **Accretion, not overwrite** (§9.2) — now required, not recommended.
- **`query_all` alone is `seek`; correct probing is `nec ∘ seek`** (§8.1).
- `--dry-run` default; credential-shaped terms refused at the ndombolo layer
  (§8.4).
