<h1 align="center">Kwasa Kwasa</h1>
<p align="center"><em>There is no reason for your soul to be misunderstood</em></p>

<p align="center">
  <img src="horizontal_film.gif" alt="Logo">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-%23000000.svg?e&logo=rust&logoColor=white)](#)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=fff)](#)

## Table of Contents

- [Introduction](#introduction)
- [Historical Context](#historical-context)
- [Theoretical Foundation](#theoretical-foundation)
- [The Turbulance Language](#the-turbulance-language)
- [System Architecture](#system-architecture)
- [Validation](#validation)
- [Implementation](#implementation)
- [Installation and Usage](#installation-and-usage)
- [Repository Layout](#repository-layout)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Kwasa-Kwasa is a semantic computing framework built around a single language,
*Turbulance*, and a small formal theory of how meaning is individuated and
transferred. The theory treats a unit of meaning as a part told apart from a
larger whole, and the cost of telling it apart as a quantity that can be
computed exactly. From that one quantity the framework derives a notion of
identity, a criterion for when two utterances mean the same thing, and a
procedure for translating between them.

The framework is organised in two layers. A formal layer states the theory as a
set of theorems about finite weighted graphs and supplies a validation suite
that checks each theorem numerically. An implementation layer realises the
theory as a programming language whose programs describe how textual,
and where available visual and auditory, inputs are individuated, compared, and
transformed. The two layers are kept separate on purpose: the theory is intended
to be checkable independently of any particular implementation, and the
implementation is intended to be replaceable without disturbing the theory.

## Historical Context

### The Philosophy Behind Kwasa-Kwasa

Kwasa-Kwasa takes its name from the musical style that emerged in the Democratic
Republic of Congo in the 1980s. During a period when many African nations had
recently gained independence, kwasa-kwasa represented a form of expression that
transcended language barriers. Despite lyrics often being in Lingala, the music
achieved widespread popularity across Africa because it communicated something
that required no translation.

### Understanding Without Translation

In the early 1970s across Africa, leaders faced the rising restlessness of Black
youth born after independence. This generation knew nothing of the hardships of
war or rural living—they had been born in bustling city hospitals, educated by
the continent's finest experts, had disposable income, and free weekends. Music
had always been a medium for dancing, but European customs of seated listening
were fundamentally misaligned with how music was experienced on the continent.

The breakthrough came when a musician named Kanda Bongo Man broke the rules of
soukous (modern "Congolese Rhumba") by making a consequential structural change:
he encouraged his guitarist, known as Diblo "Machine Gun" Dibala, to play solo
guitar riffs after every verse.

Just as DJ Kool Herc recognized the potential of extended breaks in "Amen
Brother," a mechanic from Kinshasa named Jenoaro saw similar possibilities in
these guitar breaks. The dance was intensely physical—deliberately so. In
regions where political independence was still a distant dream, kwasa-kwasa
became a covert meeting ground for insurgent groups. Instead of clandestine
gatherings, people could congregate at venues playing this popular music.

The lyrics? No one fully understood them, nor did they need to—the souls of the
performers were understood without their words being comprehended. Artists like
Awilo Longomba, Papa Wemba, Pepe Kale, and Alan Nkuku weren't merely
performing—they were expressing their souls in a way that needed no translation.

This framework aims at a similar preservation: it treats the transfer of meaning
as the goal, and asks under what conditions that transfer can be certified
without first reducing meaning to a fixed value.

## Theoretical Foundation

The framework rests on a self-contained development carried out entirely over
finite weighted graphs. The full treatment, including proofs and references, is
given in the accompanying paper, *Semantic Uncertainty Propagation*
([`transmission-lever/docs/semantic-uncertainty-propagation/`](transmission-lever/docs/semantic-uncertainty-propagation/)).
The account below summarises its premises and main results.

### Objects

Every object is a finite weighted graph. Items of interest are vertices; a
distinguished vertex, the *medium*, stands for "everything else" and is adjacent
to every item. The *separation cost* of an item is the minimum weight of a cut
that places the item on one side and the medium on the other—the item-to-medium
minimum cut, computable exactly by maximum flow. This cut is the item's
*resting cut*.

### The floor

The central quantity is a strictly positive lower bound, the *floor* `β > 0`, on
the cost of any separation. It is not assumed but derived: identifying an item
means comparing it against everything it is not, and if the whole cannot be
completed, that comparison never closes, so an irreducible residue remains. The
floor is the trace this incompleteness leaves on each part. A direct consequence
is that there is no zero-cost ("sharp") cut: no item is individuated for free.

### Identity as a region

The identity of an item is taken to be its resting cut, not its label. The
resting cut is invariant under relabelling, whereas the label is not; and the
minimising side of the cut is in general not a single vertex. Identity is
therefore a region of contrasts of weight at least `β`, never a point. We call
this the *least sufficient unknowability* of an item: the cheapest commitment of
*what the item is not* that suffices to tell it apart.

### Content, meaning, and names

The *content* of a unit is its label and its dictionary correspondents; its
*meaning* is its resting cut together with what that cut propagates. The two come
apart, and meaning cannot be read off as a value: doing so would require
resolving the cut to zero residue, which the floor forbids. A *name* is the
content-free handle on a unit's resting cut—a reference to the cut that adds no
contrast of its own. Names are not reached by any sequence of negations and are
not compositional; they are resolved by how they propagate in context, not by
decomposition.

### Detecting identity of meaning

Because meaning cannot be extracted, the framework does not attempt to know it;
it detects when two sides *share* it. Two columns—a source unit and a candidate
rendering—are relaxed against each other by propagating their differences as
demands. The process either reaches *quiescence*, where no difference remains to
propagate, or it does not halt. Quiescence is taken to constitute identity of
meaning, and it is certified without any party reading the meaning, at a positive
residual of form. Where no fixed point exists, the honest output is to decline
rather than to emit a divergent rendering.

### Receivers and conversation

A meaning is relative to the receiver that individuates it: distinct decoders
register distinct cells, each correct in its own graph, and there is no
privileged value of which they are approximations. A *conversation* is modelled
as a sequence of such individuations in which the residue of each utterance
prompts the next. A response is required precisely because no point-meaning can
be extracted, and the next utterance is governed by the standing demand rather
than by a comprehended value.

### The four-column translator

Translation is extended from two columns to four: the two central translation
columns, plus two external columns holding a plausible *response* generated for
each side. Quiescence of the four-column system tests not only whether the two
units have the same content but whether they invite the same response—an audit
of the route rather than the endpoints. This distinguishes a faithful rendering
from a false friend, whose endpoints align while its responses diverge.

## The Turbulance Language

Turbulance is the framework's programming language. A program describes how units
are individuated, compared, and transformed; the constructs correspond to the
elements of the theory above.

The language is implemented in two settings. A deterministic interpreter runs in
the browser with no backend
([`web/src/lib/turbulance/`](web/src/lib/turbulance/)) and powers a set of
worked tutorials and a sandbox. A formal engine, written in Rust, implements the
same language together with the parts of the theory that require it. The
validated browser examples serve as acceptance tests for the Rust engine.

### Points and resolutions

A *point* is a unit carrying explicit residual uncertainty; a *resolution*
propagates that uncertainty rather than collapsing it to a single value.

```turbulance
point claim = {
    content: "The patient shows improvement in respiratory function.",
    certainty: 0.73
}

resolution assess(p: Point) -> Outcome {
    item residue = propagate_uncertainty(p)
    return integrate(residue)
}
```

### Propositions and motions

A *proposition* groups related claims, called *motions*, that are evaluated
against a unit.

```turbulance
proposition ClinicalReading:
    motion Improvement("Respiratory function is improving")

    within report:
        given mentions(report, "respiratory") and trend(report) == Positive:
            support Improvement
```

### Units and operations

Text is divided into units that can be compared and recombined. Operations are
defined so that the relevant invariants—resting cuts and the demands between
them—are what the program manipulates, in keeping with the theory.

```turbulance
item paragraph = "Machine learning improves diagnosis. However, limitations exist."

item claims        = paragraph / claim
item qualifications = paragraph / qualification

item combined = claims + qualifications
```

## System Architecture

The framework is organised as follows.

```
+----------------------------------------------------------+
|                    Kwasa-Kwasa                           |
+----------------------------------------------------------+
|  Formal layer                                            |
|    - Semantic Uncertainty Propagation (theory + proofs)  |
|    - Validation suite (exact minimum cut)                |
+----------------------------------------------------------+
|  Language layer                                          |
|    - Turbulance interpreter (deterministic, browser)     |
|    - Turbulance engine (Rust)                            |
+----------------------------------------------------------+
|  Processing modules                                      |
|    - Text units and operations                           |
|    - Optional: chemistry, biology, spectrometry, media   |
+----------------------------------------------------------+
|  Reasoning delegation                                    |
|    - External probabilistic engine (Autobahn), optional  |
+----------------------------------------------------------+
```

The text layer is the reference implementation. The optional modules apply the
same unit-and-operation model to other domains and are compiled only when
selected. Probabilistic reasoning, where it is needed, is delegated to an
external engine rather than implemented in the core.

## Validation

The formal layer ships with a validation suite that checks each theorem of the
theory numerically on finite weighted graphs. Minimum cuts are computed exactly
by a self-contained maximum-flow routine, so the suite has no external
dependencies. The suite is organised as one category of checks per result, run
under a fixed seed for reproducibility, and reports a per-category pass count and
a summary written to JSON.

The current suite comprises twenty categories covering, among others: the floor
holding on every item of randomly generated graphs; complementation as an
involution; invariance of the minimum cut under relabelling; the region-valued
character of identity; the alignment lower bound; strict monotonicity of the
committed record; the residue-to-successor relation; the equivalence of
quiescence and an identical resting cut; the positive form-residual; the
relaxation dichotomy; receiver-relativity; the four-column route audit;
non-compositionality of names; and the master equivalence. The suite and its
output are in
[`transmission-lever/docs/semantic-uncertainty-propagation/validation/`](transmission-lever/docs/semantic-uncertainty-propagation/validation/).

```bash
cd transmission-lever/docs/semantic-uncertainty-propagation/validation
python validate.py        # writes results.json
```

## Implementation

### Core modules

- `turbulance/` — language implementation (parser, evaluator, semantic operations)
- `text_unit/` — text unit processing
- `semantic_bmds/` — information-catalyst operations across modalities
- `knowledge/` — knowledge representation and retrieval
- `cli/` — command-line interface and REPL

### Optional modules

Compiled only when their feature is enabled:

- `kwasa-cheminformatics` — chemistry processing
- `kwasa-systems-biology` — biology analysis
- `kwasa-spectrometry` — spectrometry processing
- `kwasa-multimedia` — image and audio handling

### Status

The deterministic Turbulance interpreter and its tutorials run in the browser and
are the most complete part of the implementation. The Rust engine is under
development; portions of the wider system are scaffolding. Where behaviour is not
yet implemented, the source indicates so rather than silently approximating.

## Installation and Usage

### Prerequisites

- Rust 1.70 or later (for the engine)
- A recent Python with the standard library (for the validation suite)
- Node.js (for the browser playground and tutorials)

### Build

```bash
git clone https://github.com/fullscreen-triangle/kwasa-kwasa.git
cd kwasa-kwasa

# Core build
cargo build --release

# With optional modules
cargo build --release --features="full"
```

### Run

```bash
# Execute a Turbulance script
./target/release/kwasa-kwasa run script.turb

# Interactive REPL
./target/release/kwasa-kwasa repl

# Validate a script
./target/release/kwasa-kwasa validate script.turb
```

### Programming interface

```rust
use kwasa_kwasa::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let framework = KwasaFramework::with_defaults().await?;

    let result = framework.process_text(
        "The patient shows improvement in respiratory function.",
        None,
    ).await?;

    println!("{:?}", result);
    Ok(())
}
```

### Browser playground

The `web/` directory contains a Next.js application with a Turbulance sandbox and
the worked tutorials. It runs without a backend.

```bash
cd web
npm install
npm run dev
```

## Repository Layout

- `transmission-lever/docs/` — the formal papers, figures, and validation suites
- `web/` — the browser playground, tutorials, and the deterministic interpreter
- `src/` — the Rust engine and processing modules
- `examples/` — sample Turbulance programs

## Contributing

Contributions are welcome in the following areas:

1. Language development: extending Turbulance syntax and semantics
2. Processing engines: text, image, and audio units and operations
3. Integration: external reasoning delegation and optional modules
4. Documentation: examples, tutorials, and use cases
5. Validation: extending the theorem-checking suite

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for
details.

## Technology

- **Rust** — core engine
- **WebAssembly** — browser deployment
- **JavaScript / Next.js** — playground and tutorials
- **Python** — theorem validation suite
- **SQLite** — knowledge persistence
- **Logos / Chumsky** — parsing infrastructure
