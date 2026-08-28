# spraypaint

Full-text search for any repo, ranked by **BM25 within scenes** and allocated
across scenes by the **water-filling** rule of the *Split-Attention Synchronised
Agents* calculus. A sibling to `purpose`: where `purpose` locates symbols,
`spraypaint` retrieves ranked passages of file **content**.

It is a token-cheap retrieval primitive for AI agents — `spraypaint ask "..."`
returns a ranked context slice instead of the agent reading whole files.

## Install

A single self-contained executable — no Rust toolchain, no Node, no runtime
dependencies. The web UI is compiled into it.

**macOS / Linux:**

```sh
curl -fsSL https://raw.githubusercontent.com/fullscreen-triangle/graffiti/main/spraypaint/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/fullscreen-triangle/graffiti/main/spraypaint/install.ps1 | iex
```

Both scripts verify the download against the published `SHA256SUMS` and abort on
mismatch. They install to `~/.local/bin` and `%LOCALAPPDATA%\spraypaint\bin`
respectively — user scope only, no administrator rights. Set
`SPRAYPAINT_INSTALL_DIR` to override.

Or download an archive directly from
[Releases](https://github.com/fullscreen-triangle/graffiti/releases) and put the
binary on your `PATH`.

The binaries are unsigned, so the first run triggers a warning: SmartScreen on
Windows ("More info" → "Run anyway"), Gatekeeper on macOS. On macOS you can
clear the quarantine attribute instead:

```sh
xattr -d com.apple.quarantine ~/.local/bin/spraypaint
```

**From source**, if you have a Rust toolchain:

```bash
cargo install --path spraypaint --force   # from the graffiti repo root
spraypaint --version
```

That build embeds whatever is in `spraypaint/ui/dist/`, which is empty in a
fresh clone — `serve` still runs and serves the JSON API, but `GET /` returns a
plain-text notice rather than the UI. To get the interface too, build it first:

```bash
cd spraypaint-web/spraypaint-web && npm ci && npm run build
cp -r out/* ../../spraypaint/ui/dist/
```

Pass `--no-default-features` for a CLI-only binary with no HTTP server.

Releases are cut by pushing a `spraypaint-v<version>` tag, which runs
`.github/workflows/spraypaint-release.yml`: it exports the UI **once** and
embeds that same build into all five targets, so a bug reproduced on one
platform is not an artifact of a differently-resolved asset bundle on another.
The workflow fails rather than publishing if the tag disagrees with
`Cargo.toml` or if the UI is missing from `ui/dist/` at compile time.

## Use: index once, ask many times

```bash
cd /path/to/any/repo
spraypaint index                       # scans the repo -> .spraypaint/index.json
spraypaint ask "how is attention divided across scenes"
spraypaint ask "water filling" --json  # machine output for agents
spraypaint ask "kuramoto" -k 20        # widen the result budget
spraypaint ask "identity" --scenes crates,docs   # restrict to named scenes
```

`ask` returns passages grouped by scene, so a dense area cannot crowd out the
rest. `--flat` prints that same set in one globally descending order instead of
grouped under scene headings.

`--flat` is a **presentation** flag, not a second retrieval mode: which passages
come back is always decided by water-filling, so a flat listing and a grouped
one contain exactly the same passages. There is no "global top-k" mode to
compare against — the allocator *is* the retrieval rule.

## Commands

| Command | Purpose |
|---|---|
| `spraypaint index [--root DIR] [--dry-run] [--window N] [--overlap N]` | Build `.spraypaint/index.json`. |
| `spraypaint ask "<query>" [--root DIR] [-k N] [--json] [--flat] [--scenes a,b] [--dry-run]` | Search. `--dry-run` prints diagnostics (price, allocation) and does **not** commit an act. |
| `spraypaint identity [--json]` | The conserved identity fingerprint + χ (Inv 1). |
| `spraypaint count [--json]` | The monotone committed count (Inv 2). |
| `spraypaint scenes [--json]` | Detected/overridden scenes. |
| `spraypaint verify [--json] [--allow-degenerate]` | Re-check all four invariants. See the exit contract below. |
| `spraypaint serve [--port N] [--host H] [--open] [--allow-index]` | Serve the JSON API and the embedded web UI on loopback. |
| `spraypaint serve --pair <ORIGIN>` | Additionally authorise a hosted copy of the UI at ORIGIN, via a printed token. |

### `verify` exit codes

| Exit | Meaning |
|---|---|
| `0` | Every check passed and every check applied. |
| `1` | At least one check **failed** — a real breach. |
| `2` | Nothing failed, but at least one check was **not applicable**. |

Exit 2 is new in 0.2.0 and is the one thing in this release that can turn a
green pipeline red. It fires on *degenerate* repositories — an empty index, a
single document, a single scene, or a corpus whose documents share no vocabulary
at all. In those regimes a check does not have enough structure to discriminate
against, so reporting PASS would overclaim: the check did not pass, it did not
run. Nothing has regressed in such a repository; it was simply never verified in
the first place, and now says so.

Pass `--allow-degenerate` to map `2 → 0` if that is the intended reading for
your corpus. The JSON output keeps its top-level `pass` boolean, so parsers
written against 0.1 keep working; `overall`, `degeneracies[]`, and per-check
`status`/`detail` are additive.

## Scenes

By default each **top-level directory** is a scene (loose root files form
`(root)`). Override with `.spraypaint/scenes.toml`:

```toml
[scenes]
core = ["crates/core", "crates/s-entropy"]
docs = ["docs", "README.md"]
```

The result budget `k` is divided across scenes by a single price `p*`
(water-filling, Algorithm 1 of the paper): each scene contributes passages while
their relevance clears `p*`; scenes below it drop. A huge dense scene cannot take
all `k` slots unless its passages genuinely out-score the others.

## The four invariants

`spraypaint` is a faithful runtime for the blueprint of
`docs/sources/split-attention-agents.tex`:

1. **Conserved identity** — the index *is* the self-graph; a relabelling-
   invariant fingerprint (+ χ, the min-cut) is recomputed and checked on every
   load.
2. **Never-resetting count** — one committed act per non-dry-run `ask`; never
   decremented, survives re-index and restart.
3. **Search-not-fetch** — every query is a fresh BM25 + water-filling walk; the
   index stores **no answers** (snippets are re-read from disk at query time).
4. **Exclusive phases** — `index` (construction) holds an exclusive lock; `ask`
   (commitment) a shared one; they never overlap.

Run `spraypaint verify` for a one-command conformance certificate — read its
exit code, not just its output, and read the degeneracy list before treating a
pass as evidence.

## The web interface

`spraypaint serve` binds `127.0.0.1:7373` and serves both the JSON API and a web
UI compiled into the executable. No Node, no network, no install step beyond the
binary itself:

```
spraypaint serve --open
```

The UI is a view over the same `actions::*` calls the CLI makes, so it cannot
show a number the CLI would not. Two properties are worth knowing because they
are load-bearing rather than cosmetic:

- **Interactive changes preview; only an explicit Run commits.** Dragging the
  budget slider calls `/api/dry-run`, which returns allocation and price without
  incrementing the committed count. The count is monotone with no decrement
  path, so a UI that committed on every drag would inflate it irreversibly.
- **The server is loopback-only and checks the `Host` header**, rejecting
  anything else with 421. That is what stops a web page you happen to be
  visiting from querying your local index via DNS rebinding — binding to
  127.0.0.1 alone does *not* prevent it.

Serving on a non-loopback address is possible but requires a second explicit
flag, because the server reads arbitrary file content, has no authentication,
and lets anyone who can reach it irreversibly inflate your count.

### Pairing a hosted copy of the UI

`serve --open` is the path with no caveats, and if it works for you, use it. But
the UI is also deployed as a static page, and `--pair` lets that page drive the
binary on *your* machine — nothing is uploaded, and the search still runs
locally:

```
spraypaint serve --pair https://acrylic-spray-paint-inky.vercel.app
```

This prints a token. Paste it into the page's pairing form together with the
server URL, and the page becomes a front-end for your local binary.

Pairing deliberately gives up the same-origin assumption the rest of the server
rests on, so a **bearer token** replaces it as the control:

- exactly one origin is authorised — the one named on the command line, matched
  in full, so a lookalike host is still rejected with 403;
- every `/api/*` request must carry `Authorization: Bearer <token>`, compared in
  constant time; a leaked token from the wrong origin still gets nothing;
- the token is 160 bits from the OS CSPRNG, exists only in that process, and
  never touches disk. Stopping the server revokes it; restarting mints a new one.

Two browser limits are worth knowing before you try, because both surface as the
same unhelpful "failed to fetch":

- **Chrome 142+ asks permission** the first time an HTTPS page reaches your local
  network. Choose Allow; dismissing it fails the request.
- **Safari cannot do this at all.** WebKit forbids HTTPS pages from reaching
  `http://127.0.0.1` and offers no prompt. There is no server-side fix — use
  `serve --open` instead.

## Notes

- `.spraypaint/` is a local cache — add it to `.gitignore`.
- Ranking uses global IDF (comparable across scenes); `--scene-idf` is planned as
  an escape hatch for a per-scene reading.
- Hand-rolled BM25 over an inverted index (no external search service);
  deterministic given the same index and query.
```