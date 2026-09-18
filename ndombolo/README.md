# ndombolo

Run the cells of a `.ndo` document.

A `.ndo` file is prose with fenced code blocks. A block tagged `turbulance` is a
**cell**; running it writes an `output` block directly beneath it. The document
is the report. The `.record` file beside it is the history of every run that
produced the report.

That split is the whole idea. The document shows you the last run. The record
knows about all of them, including the ones whose output you have since cleared.

```
notes.ndo          the report -- prose, cells, and the output of the last run
notes.ndo.record   the history -- one deposit per cell run, append-only
```

---

## Requirements

**Rust 1.75 or newer.** That is the only thing you need to build and run
ndombolo.

If you do not have it, install from [rustup.rs](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # macOS / Linux
```

On Windows, download and run `rustup-init.exe` from the same site.

Check it:

```bash
cargo --version    # cargo 1.75.0 or newer
```

**No npm, no Node, no Python, no database, no Docker.** The whole dependency
list is `serde` and `serde_json`. The HTTP server, the editor page, and the
charting are written against the standard library or vendored into the binary.
There is nothing to `npm install` and nothing is fetched at page load.

**Ollama is optional.** It is needed only for the editor's "ask" panel, which
answers questions about a run by searching its trace. Everything else — running
cells, the editor, the charts — works without it. See
[Asking about a run](#asking-about-a-run-optional) below.

---

## Build

From the `ndombolo/` directory:

```bash
cargo build
```

That produces `target/debug/ndombolo` (`target\debug\ndombolo.exe` on Windows).
For a faster binary, `cargo build --release` puts one in `target/release/`.

Confirm it works:

```bash
cargo test --workspace
```

You should see **71 tests passing and no warnings**. If you get compile errors,
your Rust is likely older than 1.75 — `rustup update` fixes it.

### Putting it on your PATH

Optional, but the rest of this README assumes you can type `ndombolo`:

```bash
cargo install --path crates/ndombolo
```

Otherwise substitute `./target/debug/ndombolo` wherever you see `ndombolo`.

---

## Your first document

```bash
ndombolo new hello.ndo
ndombolo run hello.ndo
```

`new` writes a short starter document that explains itself. `run` executes its
cells and writes the results back into the file. Open `hello.ndo` in any editor
and you will see `output` blocks that were not there before:

````markdown
```turbulance
item greeting = "hello"
print(greeting)
```

```output
hello

greeting = hello
```
````

An output block has two parts: anything the cell printed, then a blank line,
then the bindings the cell left in the store. The second part is why a later
cell can see what an earlier one did.

Run it a second time and the output block is *replaced*, not appended — so a
cell run ten times leaves one block. Meanwhile:

```bash
ndombolo record hello.ndo
```

...shows the record has advanced ten times. The document forgets; the record
does not.

---

## The editor

```bash
ndombolo edit hello.ndo
```

Open <http://127.0.0.1:7749> in a browser. The document renders as prose with
editable cells.

- **Ctrl-Enter** (Cmd-Enter on a Mac) runs the cell your cursor is in.
- **Escape** closes the menu or the ask panel.
- Editing prose and saving writes straight back to the file on disk.

To serve a whole folder instead of one file:

```bash
ndombolo edit tutorials/
```

Now `/` is an index listing every `.ndo` in the directory, each with its record
count. Click one to open it; the index shows each document's own heading rather
than its filename, so you can tell what you are opening.

### The editor is a second face on the runtime, not a second runtime

Anything the editor does to a document, `ndombolo run` does identically — same
session, same replay discipline, same deposits. The editor just adds a browser
and the ability to ask a local model about the trace.

### It binds to loopback only, and that is not configurable

The server reads and writes files on disk and has no authentication. Reachable
from another machine, it would be a file-read primitive for anyone on your
network. So it binds `127.0.0.1` and nothing else, deliberately.

The `--port` flag changes the port, not the address:

```bash
ndombolo edit tutorials/ --port 8080
```

---

## The tutorials

Eleven documents that teach the system by being run. Start the editor on the
folder:

```bash
ndombolo edit tutorials/
```

They form two arcs:

**1–7 — running an experiment.** Intent, entities, protocol, provenance,
observation, acceptance, and then the whole thing assembled. This arc is about
keeping intent, plan, execution, observation and inference distinct in a
document you can hand to someone.

**8–11 — querying what is already known.** Federated querying across
heterogeneous sources, and the distinction that decides how a data layer gets
built: a schema declares *shape*, which is not the same as *admissibility*.
These four are the ones with figures in them.

Each document links to the next, so you can read straight through.

---

## Commands

```
ndombolo run    <file.ndo> [--cell N]   run cells and write outputs back
ndombolo clear  <file.ndo> [--cell N]   remove output blocks
ndombolo cells  <file.ndo>              list cells without running them
ndombolo graph  <file.ndo>              the script graph, as JSON
ndombolo trace  <file.ndo>              every trace event, as JSON
ndombolo record <file.ndo>              the deposits behind this document
ndombolo new    <file.ndo>              write a starter document
ndombolo edit   <file.ndo|dir>          open a document, or a directory of them
```

**Options**

```
--cell N       act on cell N only (0-based)
--port N       the port the editor listens on, on loopback (default 7749)
--model NAME   the ollama model the editor asks (default llama3.2)
--host URL     where ollama listens (default http://127.0.0.1:11434)
```

### Two behaviours worth knowing before they surprise you

**`--cell N` selects what is written, not what is run.** Cells 0 through N all
execute, because a cell's meaning depends on the store the cells before it
left — but only cell N's output block is replaced. So the record advances by
more than one:

```bash
$ ndombolo run demo.ndo --cell 1
demo.ndo: ran 2 cells, record 4     # was 2; both cells ran, both deposited
```

This is not a quirk to work around. There is no cache of a previous run's store
to resume from, and adding one would change what a run *means*.

**`clear` removes output blocks but the record still stands.**

```bash
$ ndombolo clear demo.ndo
demo.ndo: cleared 2 output blocks, record still 4
```

Clearing tidies the report. It does not rewrite history.

---

## Asking about a run (optional)

The editor can answer questions about what a run did — "why is `n` 2?", "which
cell last wrote `greeting`?" — by searching the trace.

This needs [ollama](https://ollama.com) running locally:

```bash
ollama serve                 # in one terminal
ollama pull llama3.2         # once
```

Then open the editor and use the ask panel. To point at a different model or
host:

```bash
ndombolo edit notes.ndo --model llama3.1 --host http://127.0.0.1:11434
```

**If ollama is not running, only the ask panel fails** — with an error in the
panel. The document still renders, cells still run, charts still draw.

Two design rules govern this, and both are load-bearing:

- **The model searches; it is not handed the trace.** A real run emits thousands
  of events. Pasting them into a prompt would make the answer a summary of text
  the model was given, rather than a result of searching the events.
- **The model can write prose and nothing else.** It never writes a cell and
  never writes an output block. What the runtime determined is not something the
  model may report — only something it may describe.

---

## Charts

Tutorials 9, 10 and 11 draw figures. A chart is declared in a ` ```chart ` fence
in the **prose**, not emitted from a cell:

````markdown
```chart
{"chart": "ladder", "title": "three rungs against a target of 0.95",
 "powers": "rungs", "target": 0.95}
```
````

The fence names **bindings**, never data. `"powers": "rungs"` means *draw the
value the cell bound to `rungs`*. The chart reads that value from the live
session, so a figure cannot state a number the run did not produce.

Two consequences follow from that, and they are the reason it is built this way:

- Charts live in prose because output blocks are runtime-written and replaced on
  every run. A chart in an output block would not survive `ndombolo run`.
- Charts read the session's values, not the text of an output block. An output
  block is a *rendering* — floats already rounded, lists flattened onto one line
  — so a chart built from it would be charting a rendering, and could drift from
  the run without anything being broken.

Four kinds are available: `ladder`, `cuts`, `graph`, `stages`. d3 v7 is vendored
into the binary; nothing is fetched from a CDN.

---

## The language

Cells are written in **Turbulance**. The deterministic core implemented here is
small and worth knowing exactly:

```turbulance
item x = 3                          # bind a name
funxn double(n):                    # define a function
    return n * 2

item xs = [1, 2, 3]                 # lists
item m = {name: "a", weight: 0.8}   # maps, and m.name to read

for each v in xs:                   # iterate
    print(v)

given x > 2:                        # conditional
    print("big")
```

**Builtins, and there are exactly seven:** `print`, `len`, `sum`, `min`, `max`,
`abs`, `round`.

They do not work the way the Python names suggest. Only `len` takes a
collection. The other five take **numbers as separate arguments** and raise on a
list:

```turbulance
print(len([1, 2, 3]))      # 3    -- lists, strings and maps
print(min(4, 1))           # 1    -- NOT min([4, 1]), which is an error
print(sum(1, 2, 3))        # 6    -- NOT sum([1, 2, 3])
print(round(2.5))          # 2    -- banker's rounding, and no digits argument
```

To total a list, write the loop:

```turbulance
item total = 0
for each v in xs:
    total = total + v
```

### Three traps that bite in practice

**`from` and `to` are keywords.** You cannot write `m.from`. Use quoted keys and
index access:

```turbulance
item edge = {"from": "a", "to": "b", weight: 0.5}
print(edge["from"])
```

**There is no integer division.** `/` is always float division. Code that
assumes truncation — bit-mask tricks, index arithmetic — will silently
misbehave.

**There is no `elif`.** Ordered dispatch is written as early `return` inside
`given`, and the order is then normative:

```turbulance
funxn classify(term, alphabet, cap):
    given not has(alphabet, term):
        return "unexpressed"
    given not has(cap, term):
        return "unsupported"
    return "ok"
```

Some keywords parse but have no deterministic semantics in this core and are
no-ops: `allow`, `research`, `cause`, `goal`, `metacognitive`, `resolution`.
`cycle`, `drift`, `flow` and `roll until settled` are keywords with no parse
rule yet.

---

## Layout

```
ndombolo/
├── crates/
│   ├── ndombolo-core/     the language: lexer, parser, evaluator, session
│   │   └── tests/         differential tests against a frozen Python oracle
│   └── ndombolo/          the CLI, HTTP server, editor, charts
│       └── src/
│           ├── editor.rs  routes and the document/record boundary
│           ├── http.rs    the server; loopback bind lives here
│           ├── page.html  the editor page
│           ├── charts.js  d3 renderings
│           └── vendor/    d3 v7.9.0, compiled in
├── tutorials/             eleven .ndo documents
└── docs/
```

The core is tested differentially against a frozen Python oracle — the Rust
lexer and parser must agree with it token for token. That is what
`cargo test --workspace` is mostly checking.

---

## Troubleshooting

**`cargo build` fails on Windows with `os error 5` / `Zugriff verweigert`.**
A running `ndombolo.exe` is holding the binary. Stop the server first:

```bash
taskkill /F /IM ndombolo.exe
```

**Port 7749 already in use.** Another instance is running, or something else
holds the port. Use `--port 8081`, or stop the other instance.

**The editor page loads but cells will not run.** Check the terminal where you
started `ndombolo edit` — a parse error in the document is reported there.

**The ask panel errors.** Ollama is not running or the model is not pulled. See
[Asking about a run](#asking-about-a-run-optional). Nothing else is affected.

**A tutorial stops partway through.** Tutorials 2 and 3 stop at cells 5 and 7 by
design — they demonstrate errors. That is the lesson, not a bug.

---

## License

MIT.
