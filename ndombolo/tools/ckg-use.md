# Using `purpose ckg` in your repository

You are an AI working in a repository. This document tells you how to build a
**causal knowledge graph** of that repository's modules, tune it to the questions
you actually need to answer, and query it.

The short version: `purpose ask` finds where a symbol is defined. `purpose ckg`
answers a different kind of question — *which modules is this goal actually about,
which of them are load-bearing, and what would I break by touching this one.* If
you only need a definition site, use `ask` and stop reading; it is cheaper. See
[When not to use this](#when-not-to-use-this).

---

## 1. The thing to understand before you run anything

The graph is built from a **term map** τ: a function from each module to the set
of distinctions it draws. Two modules are in contact when they share a
distinction; the weight of that contact is what it would cost to tell them apart.

**τ is yours to choose.** That is the whole point of this tool. The underlying
calculus is proved correct for *any* τ — no theorem inspects where an edge came
from — so nothing you write in the lens file can make a determination unsound.
What your choice decides is whether the graph can tell your modules apart *at
all*.

A badly-chosen τ does not lie to you. It produces a graph in which everything is
in contact with everything, every query returns the whole repository, and you
learn nothing. That failure is visible, and §4 is about seeing it.

**There is deliberately no score to maximise.** If you look for one you will not
find it, and this is not an oversight — see §6.

---

## 2. Bootstrap

```bash
purpose index                      # once; builds .purpose/index.json
purpose ckg lens --init            # writes a commented .purpose/lens.toml
purpose ckg lens                   # diagnostics — what that lens did to the structure
#   ... edit .purpose/lens.toml ...
purpose ckg lens                   # look again; iterate
purpose ckg build                  # induce and store the graph
```

`purpose ckg lens` **never writes the graph.** It builds τ in memory and reports.
Run it as often as you like while tuning; nothing is committed until you run
`build`.

Both `.purpose/lens.toml` and `.purpose/ckg.json` live under `.purpose/`.
**Check `lens.toml` in; gitignore the rest.** The lens is a decision about how
this repository should be read, and it belongs in version control where it can be
diffed. The index and graph are derived caches.

If `purpose ckg --help` reports `unrecognized subcommand`, the binary on your PATH
predates this feature. Rebuild:

```bash
cargo install --path crates/purpose-cli --force   # from the purpose implementation repo
```

---

## 3. The lens file

`purpose ckg lens --init` writes this with explanatory comments. The schema in
full:

```toml
[lens]
granularity = "file"        # file | dir — one module per source file, or per directory
floor = 1.0                 # β; every contact carries at least this weight

[include]
paths = []                  # empty = all files. `*` stays in one path segment, `**` crosses
kinds = ["fn", "struct", "trait", "heading"]   # omit entirely for all kinds

[terms]
min_len = 3                 # shortest prose token that counts as a distinction
split_camel_case = false    # also emit the pieces of parseRequest, not only the whole
prose_kinds = ["heading", "section"]   # names here are phrases to split, not identifiers
stopwords = ["the", "and", ...]        # 45 by default

[terms.alias]               # rename a term wherever it appears, so two spellings merge
"ckg" = "contact-knowledge-graph"

[terms.weight]              # what a distinction is worth; first match wins, default 1.0
"trait:*" = 3.0
"heading:*" = 0.5
"fn:parse" = 2.0

[edges]
weight = "sum"              # sum | count | jaccard
```

### Things that will catch you out

**Stopwords apply to prose only — never to symbol names.** A `fn over` really does
define `over`, and no stopword list will remove it. This is deliberate: a function
someone named `over` draws a genuine distinction in the codebase. The consequence
is practical and important — if `new` shows at 33% spread in your diagnostics
because ten files each define `fn new`, **the stopword list cannot touch it**.
Reach for `include.kinds` or `[terms.weight]` instead.

**`[terms.weight]` patterns match the *kind that contributed the term*, not a
textual prefix.** Prose terms are emitted bare, so `"heading:*"` means *any term
contributed by a heading*, not *terms literally starting with `heading:`*. When
one term arrives from several kinds in the same module, the **greater** weight
wins — merging by sum would make weight depend on how many times the indexer's
line scan happened to fire.

**A shared term is worth `min` of what each module gives it.** If `u` draws `t` as
a `trait` (3.0) and `v` only in a heading (0.5), what they genuinely hold in
common is worth 0.5. The weaker party bounds the contact; otherwise one module's
emphasis could inflate a contact the other barely makes.

**`edges = "jaccard"` with `floor = 1.0` flattens the graph.** Jaccard weights land
in (0,1], the floor clamps every one of them to β, and every contact becomes
identical. If you use `jaccard`, set `floor = 0.01` or similar. The tool warns
about this on stderr; the warning is worth heeding.

**Unknown keys are refused, with a suggestion.** A silently-ignored typo would
mean you think you tuned something and did not.

### Precedence

**CLI flag > lens file > built-in default.** `purpose ckg build --floor 2` beats
the lens; the tool prints a note on stderr when a flag overrides a lens key, so an
override is never silent.

---

## 4. Reading the diagnostics

`purpose ckg lens` prints eight sections. `--raw` gives the same thing as JSON,
which is what you want when tuning in a loop.

| Section | What you are looking for |
|---|---|
| **LENS** | Every setting as *resolved*, plus source and digest. Confirm the file you edited is the file in play. |
| **COVERAGE** | Entries indexed / admitted by paths / admitted by kinds / yielded no terms; modules in τ. **A lens quietly dropping most of the index is the first thing to notice.** |
| **COMPONENTS** | Sizes, largest first, medium excluded. One giant component ⇒ τ is not discriminating. All singletons ⇒ it is not connecting. |
| **DENSITY** | `n`, `e`, density, mean/median weight, β, β*, Ω. |
| **DEGREE HUBS** | Top modules by degree with σ. Where a degenerate term is hiding. |
| **TERM SPREAD** | Top terms by how many modules carry them. **The actionable section** — the top of this list is your stopword candidate list, subject to the prose-only caveat above. |
| **GOAL SATURATION** | For each `--goal`, how much of the repo it seeds. Reported per goal, never averaged. **A goal seeding >50% of the repo is one the graph cannot discriminate.** |
| **THE CAVEAT** | Always last. Read it. §6. |

Pass goals you actually care about:

```bash
purpose ckg lens --goal "resolver compile" --goal "contact floor graph"
```

### Which levers actually work

Measured on this repository (451 symbols, 30 modules), in increasing order of
effect. This ordering is empirical and it **contradicts** the intuitive guess:

1. **Stopwords — weakest.** Density 0.232 → 0.189. They cannot touch symbol names,
   which is where high-spread terms usually live.
2. **`[terms.weight]` — moves σ, not the structure.** Weighting by kind spread
   separation costs from a flat 26.00 to 11.00–25.00 and halved Ω. But edges are
   contact-only by design, so weight changes what a contact *costs*, never whether
   it *exists*. Components do not move.
3. **`include.kinds` — the real lever.** Dropping `fn` took the giant component
   from 25 modules to 11 (14 components; density 0.232 → 0.050; top term spread
   37% → 8%).

**Start at `include.kinds`, not the stopword list.**

And note that lever 3 is a genuine trade, not a free win: on this repo, dropping
`fn` also cost 5 modules leaving τ entirely and 302 of 451 entries. Structure
against coverage. Decide deliberately, and record the decision — a commented-out
setting with a note on what it costs is better than a silent omission.

---

## 5. Querying

```bash
purpose ckg ask "resolver compile"        # determination for a goal
purpose ckg why crates/foo/src/bar.rs     # one module's position
purpose ckg why crates/foo/src/bar.rs --goal "resolver"
purpose ckg floor                         # β* and its witness
```

### `ckg ask` — the determination

Reports a verdict, then two classified sets:

- **ACCOUNTABLE** — `σ ≤ β* + εΩ`. The goal's alignment sits within the system's
  own floor.
- **CONTESTED** — it does not; the determination is declined and the classes are
  reported anyway so you can see why.
- **DECLINED** — nothing in the repository draws a distinction meeting this goal.
  Usually means your goal terms do not match the vocabulary τ extracted; check
  TERM SPREAD.

- **NECESSARY** — load-bearing. Dropping one changes what the goal resolves.
  Marked `seed` (met the goal directly) or `dominates N` (every route from the
  goal passes through it).
- **REACHABLE BUT REDUNDANT** — reachable, but droppable without changing what the
  goal resolves.

Necessity is **relative to what the goal reached**, not to the repository. A
module can be structurally important and still redundant for your goal; that is
the correct answer, not a bug.

### `ckg why` — one module

σ, β*, Ω; and with `--goal`, whether the module is reachable and necessary for it,
and what it dominates. Then the **resting cut** — the actual set of edges whose
removal separates this module, which is the concrete answer to *what would I break
by touching this*. Then its contacts.

### ε

**The default is ε = 0**, and this is deliberate. Measured over six goals on this
repo, the threshold behaves as a cliff, not a gradient: ε = 0 gives 1/6
accountable, ε = 0.01 through 0.03 sits flat at 2/6, then ε = 0.05 gives 6/6. Any
default near the cliff makes the verdict meaningless. ε = 0 is the only value with
a defensible reading — *σ is within the system's own floor, with no allowance* —
and `--eps` is there when you want to relax it consciously.

If everything comes back CONTESTED, the fix is usually a sharper lens, not a
larger ε.

---

## 6. The rule about β\*

**β\* is a monotonicity signal, not a score.** Refining the term map cannot lower
it. So a floor that is not rising under attempted refinement tells you the map is
not being refined.

It does **not** follow that a higher floor is better. A lens under which every
module draws identical distinctions induces among the *highest* floors while
discriminating *worst*. β* rising and components collapsing to one is the
degenerate case, not success.

This is why there is no aggregate score anywhere in the diagnostics, and why there
is a test asserting no key in the raw output matches `score|quality|rating|grade|
overall`. A scalar to maximise would be optimised, and optimising this one
produces the degenerate lens.

**Never report a rising β\* as an improvement.** Judge a lens by component sizes,
term spread, and goal saturation. On this repository β* stayed at 1.00 through
every tuning pass that materially improved the graph.

---

## 7. Staleness

`ckg.json` stores the lens that induced it, in full — not just a hash, so a stale
graph can tell you *which setting moved*.

`ckg ask` and `ckg why` **reconstruct τ from the stored lens, not from
`lens.toml` on disk.** If you edit the lens without rebuilding, they warn on
stderr and continue against the stored graph. That result is still sound — it is a
correct determination about the graph you built — it is just about a τ you have
moved on from. Run `purpose ckg build` to catch up.

---

## 8. When not to use this

`ckg ask` is **not a cheaper `ask`** — it answers a different question, and its
cost does not reliably undercut it. Measured on this repository:

| goal | `ask` | `ckg ask` |
|---|---|---|
| `resolver compile` | 1330 B | 1913 B |
| `contact floor graph` | 2652 B | 1625 B |

Which one is cheaper depends on the goal, so do not choose between them on cost.
Choose on the question:

| You want | Use |
|---|---|
| Where is `Resolver` defined? | `purpose ask "Resolver"` |
| Which modules is this task actually about? | `purpose ckg ask "<goal>"` |
| What breaks if I change this file? | `purpose ckg why <module>` |
| Which turns of this conversation still matter? | `purpose ledger` |

Two more limits worth knowing before you trust a negative result:

- **The index carries definitions and headings only.** Call sites, imports, config
  values, and string literals are absent — from `ask` and from τ alike, so they
  cannot create contacts either. A module whose only relationship to another is a
  call will not be in contact with it.
- **Filenames and paths are not indexed as terms.** Use `include.paths` to scope,
  and Glob to find files.

Never conclude something does not exist from a `purpose` miss. Fall back to
Grep/Glob.

---

## 9. A worked loop

```bash
purpose index
purpose ckg lens --init
purpose ckg lens --goal "the goal you actually have"
```

Read COMPONENTS. If one component holds most modules, the graph cannot yet
discriminate. Read TERM SPREAD for the terms putting everything in contact, then:

- Term is prose (a heading word)? → add to `stopwords`.
- Term is a symbol name (`new`, `build`, `run`)? → stopwords will not help.
  Narrow `include.kinds`, or down-weight its kind (`"fn:*" = 0.25`).
- Two spellings of one idea? → `[terms.alias]`.

Re-run `purpose ckg lens`. Repeat until components have split and your goal's
saturation is well under 50%. Then:

```bash
purpose ckg build
purpose ckg ask "the goal you actually have"
```

Commit `.purpose/lens.toml`. The next AI in this repository inherits your reading
of it, and can diff it if they disagree.
