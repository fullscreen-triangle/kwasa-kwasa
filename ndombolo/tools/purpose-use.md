# Purpose — Integration Instructions

Purpose is **two tools** that share a name and a principle, but not a codebase, a
language, or an entry point. Use the right one for what you are doing:

| | **Tool B — AI-probe CLI** | **Tool A — DSL-model generator** |
|---|---|---|
| **What it is** | An installed command-line binary | JS code that lives *inside* Buhera OS |
| **Language** | Rust (`purpose.exe` on your PATH) | JavaScript (Next.js, in `long-grass`) |
| **What it does** | Lets an AI query a repo's structure instead of reading its files | Turns natural-language instructions into valid DSL code and runs it |
| **Where it runs** | Any repo/project on your machine | The Buhera OS module runtime |
| **Use it when** | An AI needs to understand a codebase cheaply | An uninformed user needs to drive a module without writing its DSL |

The shared principle is the **empty-dictionary principle**: Purpose ships no domain
facts. Tool B retrieves the slice deterministically; Tool A lets a real compiler judge
generated code. Neither memorizes the thing it operates on.

---

# Part 1 — Tool B: Use Purpose independently in any repo

**Goal:** an AI agent working in one of your repos answers "where is X / how is Y
structured" by calling `purpose` instead of reading dozens of files. A `purpose ask`
costs ~200 tokens; reading the files to answer the same question costs thousands.

## 1.1 Prerequisites

`purpose` is already installed on your PATH (`~/.cargo/bin/purpose`). Confirm:

```bash
purpose --version        # -> purpose 0.1.0
```

If it is ever missing (fresh machine), reinstall from the workspace:

```bash
cd mechanistic-synthesis/implementation
cargo install --path crates/purpose-cli --force
```

## 1.2 The two-step workflow: index once, ask many times

From **any** repo, build the index once, then ask questions against it.

```bash
cd /path/to/any/repo         # e.g. bioinformatics/borgia
purpose index                # scans the repo -> .purpose/index.json
purpose ask "where is the spin echo partition count"
```

`purpose ask` returns a ranked context slice — `file:line [kind] name` plus a one-line
snippet — for example:

```
20 matching symbol(s):

  honjo-masamune/docs/loschmidt/loschmidt-echo-progression.tex:741  [section] The Nuclear Spin Echo: A Complete Account
      \section{The Nuclear Spin Echo: A Complete Account}
  dmitri/sources/partition-graph-propagation.tex:345  [section] The Bijection and the Partition Count
      \subsection{The Bijection and the Partition Count}
  ...
```

That slice is the answer. The AI reads it (a few hundred tokens) instead of grepping and
opening files across the tree.

## 1.3 Command reference (installed binary)

| Command | Purpose |
|---|---|
| `purpose index [--root <path>]` | Line-scan the repo into `.purpose/index.json`. Defaults to the project root detected from the current directory (walks up to `.git`/`.purpose`). |
| `purpose ask "<question>" [--root <path>] [--dry-run]` | Return the context slice for a question. `--dry-run` prints the compiled vaHera fragment instead of executing. |
| `purpose query "<utterance>"` | The protein/UniProt domain (worked example domain). |
| `purpose operations` | List the registered operations. |

**What gets indexed:** function/struct/enum/trait/type/class/def definitions across ~20
source extensions, plus Markdown and LaTeX headings. Generated/minified files,
`node_modules`, `target`, `.git`, build output, etc. are skipped. Prose files contribute
headings only; code files contribute definitions only.

## 1.4 Wiring an AI agent to call Purpose

The point is that an agent probes Purpose *instead of* reading files. Give the agent this
instruction (in its system prompt, `CLAUDE.md`, or tool description):

> To understand this codebase, run `purpose ask "<your question>"` from the repo root
> and read the returned slice. Run `purpose index` first if `.purpose/index.json` is
> missing or stale. Prefer this over reading files directly — it is far cheaper in tokens.

For a **Claude Code / MCP-style** agent, expose it as an allowed shell command:

```jsonc
// .claude/settings.json (in the target repo)
{
  "permissions": { "allow": ["Bash(purpose index)", "Bash(purpose ask:*)"] }
}
```

## 1.5 Keeping the index fresh

The index is a snapshot. Re-run `purpose index` after significant code changes. Cheap
options:

- **Manual:** re-index when an agent reports stale/missing symbols.
- **Git hook:** add `purpose index` to a `post-merge` / `post-checkout` hook.
- **Ignore the artifact:** add `.purpose/` to the repo's `.gitignore` — it is a local
  cache, not source.

## 1.6 Multi-repo (forthcoming — Tool B Layer 2)

Today `ask` is scoped to the current repo (or `--root`). Cross-repo querying — a registry
of named repos so you can `purpose ask "..." --repo borgia` from anywhere, and
`--all-repos` to merge indices — is the planned next layer for Tool B. Until then, `cd`
into the target repo (or pass `--root`) and index each repo once.

---

# Part 2 — Tool A: Insert Purpose into Buhera OS

**Goal:** an *uninformed* user types natural-language instructions; Purpose compiles them
into a valid script for a specific module's DSL and runs it — reaching the **same
executor** an *informed* user's hand-written DSL script would. Purpose is the equalizer
between the two user types.

Tool A lives in the Buhera OS web app (`long-grass`). It is already wired in; this section
documents how it fits together and how to extend it to a new DSL.

## 2.1 The execution model

Both user paths converge at one funnel — `dispatch(moduleId, source)` in
`src/lib/modules/registry.js`:

```
informed user   → hand-writes DSL script ─────────────────────────────┐
                                                                        ▼
uninformed user → NL instructions → [Purpose: generate → validate →   → dispatch(moduleId, source) → ActResult
                                     repair] → DSL script                 (the ONE executor both reach)
```

The generator never invents DSL syntax on faith: it drafts code, then the module's **own
compiler** validates it (reject → feed the error back → regenerate). The compiler is the
ground truth.

## 2.2 The pieces (all in `long-grass`)

| File | Role |
|---|---|
| `knowledge-packs/<dsl>/` | Grammar + worked examples that ground generation (examples, not facts). |
| `src/lib/purpose/dsl/validators.js` | Per-DSL validator adapters + the `DSL_REGISTRY` (`dslId → {validate, moduleId, packId}`). |
| `src/lib/purpose/federation-orchestrator.js` | FKAC: N parallel LLM drafts + integration + aggregate-floor confidence. |
| `src/lib/purpose/dsl-generator.js` | The `generateDsl(...)` loop: ground → generate → validate → repair. |
| `src/pages/api/dsl-generate.js` | HTTP route; optionally dispatches the validated code. |
| `src/lib/modules/dsl-writer-module.js` | Makes the generator itself a dispatchable module. |

## 2.3 Prerequisites: a schema-capable LLM provider

The generator forces the model to emit `{ code }` via JSON-schema-constrained output.
Only **schema-capable** providers work: Ollama, Gemini, or OpenAI. HuggingFace is
automatically filtered out when a schema is required.

Set at least one in `long-grass/.env.local`:

```bash
# Any ONE of these. Ollama is local, free, and schema-capable — the surest for offline dev.
OLLAMA_URL=http://localhost:11434
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

If none is set (or all are out of quota), generation returns
`{ ok:false, errors:[{ stage:"provider" | "draft" }] }` with the per-provider reason —
retrieval/validation still work, only drafting is blocked.

## 2.4 Using it: HTTP route

```bash
curl -X POST http://localhost:3000/api/dsl-generate \
  -H "Content-Type: application/json" \
  -d '{
        "dslId": "vahera",
        "instructions": "store a memory named x with value 1, then find the nearest memory to x",
        "execute": true
      }'
```

Response:

```jsonc
{
  "ok": true,
  "dslId": "vahera",
  "code": "memory store \"x\" = \"1\"\nmemory find nearest \"x\"",
  "federation": { "draft_models": [...], "aggregate_floor": 12.3, "confidence": 0.88 },
  "repairs": 0,
  "attempts": 1,
  "act": { "ok": true, "output_delta": { "kind": "vahera_results", ... }, "completed": true }
}
```

- `execute: false` → generate + validate only (returns `code`, no `act`).
- `execute: true` → also `dispatch(moduleId, code)`; `act` is the module's ActResult.
- `maxRepairs` (default 3, cap 6) → repair attempts against the real compiler.

## 2.5 Using it: from a Turbulance script or the terminal

The generator is also a module, so an informed script can call it:

```
dispatch("dsl-writer", "store a memory \"x\"=\"1\" then find nearest to x")
dispatch("dsl-writer", { dslId: "vahera", instructions: "...", execute: true })
dispatch("dsl-writer", { dslId: "vahera", instructions: "...", execute: false })  // generate + validate only
```

It is registered in `src/components/BuheraTerminal.js` alongside the other modules.

## 2.6 Adding a new DSL to Purpose

Each Buhera module carries its own DSL; the modules arrive in `long-grass` via `npm link`
(or a vendored package), so their real compilers are importable. To make Purpose generate
a new DSL, you add three small things — **no change to the generator or orchestrator**:

1. **A knowledge pack** — `knowledge-packs/<dsl>/manifest.json` + a `reference.md` with the
   grammar and a few worked scripts. Match the manifest shape used by
   `src/lib/purpose/knowledge-packs.js` (`id`, `name`, `trigger_keywords`, `files`, …).

2. **A validator adapter** in `src/lib/purpose/dsl/validators.js` that calls the DSL's
   real compiler and normalizes to `{ ok, errors: [{ message, line? }] }`:
   - returns `{valid, errors}` → map `valid` → `ok` (e.g. SBS `validateSBS`);
   - returns `{ok, errors}` → pass through (e.g. SCOPE `compile`);
   - returns a single `{error}` → lift to `errors:[error]` (e.g. Turbulance `run`);
   - **throws** → wrap in try/catch (e.g. vaHera `parseVahera`).

3. **A `DSL_REGISTRY` entry**: `{ <dslId>: { label, validate, moduleId, packId } }` where
   `moduleId` is the registry id that executes the code and `packId` is the pack from (1).

That is the whole extension surface. `generateDsl` and the federation are DSL-agnostic.

### DSL fit notes (from the current module set)

| DSL | Module | Validate | Execute-from-text? | Status |
|---|---|---|---|---|
| **vaHera** | `vahera` | `parseVahera` (throws) | yes | ✅ implemented |
| **SBS** | `sbs` | `validateSBS` → `{valid,errors}` | yes (`runSBS(source)`) | ready to add |
| **Turbulance** | (turbulance) | `run(source)` (validate+execute) | yes | ready to add |
| **SCOPE** | `scope` | `compile(source)` → `{ok,errors}` | validate-only (execution needs an image payload) | add as generate+validate |
| **lavoisier experiment** | `lavoisier` | — | not a language (`runExperiment(config)`) | out of scope for DSL gen |

## 2.7 Verifying an install

Run the unit suite (no LLM needed — validates against the real compilers):

```bash
cd long-grass
npm test        # node:test; 17 tests over validator, federation, repair loop
```

For a live end-to-end check (needs a schema-capable provider with quota):

```bash
npm run dev
# then POST to /api/dsl-generate as in 2.4 and confirm `code` parses and `act.ok` is true
```

---

# Appendix — Which tool answers which question

- "Where is X defined in this repo?" / "How is this framework structured?" → **Tool B**
  (`purpose ask`), in any repo.
- "I want to run a module but I don't know its DSL." → **Tool A** (`/api/dsl-generate` or
  `dispatch("dsl-writer", ...)`), inside Buhera OS.
- "I know the DSL and wrote the script myself." → you don't need Purpose; `dispatch()`
  directly. Tool A exists precisely to bring the uninformed user to that same door.
