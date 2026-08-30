//! The Rust evaluator against the frozen Python one, cell by cell.
//!
//! The third of the differential harnesses, after the lexer's and the parser's.
//! `tests/eval.json` is written by `figures/dump_eval.py` from `prototype/`,
//! which is a reference implementation and not a performance claim (paper, §8);
//! its only remaining job is to be the oracle this checks against.
//!
//! # What is actually under test
//!
//! Not the final values. The paper's Thm. 5.2 makes every atomic question a
//! projection of the **trace**, so a port that agreed on computed values while
//! disagreeing on which rules fired, in what order, reading what, would have
//! ported the arithmetic and lost the property the runtime exists for. The
//! per-cell trace slice is therefore compared event by event and field by
//! field, ahead of anything else.
//!
//! # Why the comparison is written out rather than derived
//!
//! `serde_json::Value` has `PartialEq`, and comparing two whole documents with
//! it would be three lines. It would also report a failure as two multi-kilobyte
//! blobs. [`first_diff`] walks instead, reporting a dotted path and the two
//! values at it, and it **checks key sets before values** -- so a missing field
//! is a named failure rather than a silent `null == absent`.

use ndombolo_core::session::{Session, StoreChange};
use serde::Deserialize;
use serde_json::{json, Map, Value as Json};

const CORPUS: &str = include_str!("eval.json");

#[derive(Deserialize)]
struct Corpus {
    scripts: Vec<Script>,
}

#[derive(Deserialize)]
struct Script {
    name: String,
    cells: Vec<Cell>,
    #[serde(rename = "final")]
    final_: Json,
}

/// A cell as the oracle recorded it. Everything but the identifying fields is
/// left as raw JSON: the harness compares against the oracle's own shape rather
/// than round-tripping it through a Rust type, which could only launder a
/// disagreement into agreement.
#[derive(Deserialize)]
struct Cell {
    index: usize,
    first_line: usize,
    text: String,
    ok: bool,
    #[serde(default)]
    error: Option<Json>,
    #[serde(default)]
    store_delta: Option<Json>,
    #[serde(default)]
    trace: Option<Json>,
    #[serde(default)]
    output: Option<Json>,
}

/// The first place two documents differ, as a dotted path.
///
/// Key sets are compared before values, so a field the port forgot to emit is
/// reported as a missing key rather than surviving as an absent-equals-null.
fn first_diff(rust: &Json, python: &Json, path: &str) -> Option<String> {
    match (rust, python) {
        (Json::Object(a), Json::Object(b)) => {
            let mut ak: Vec<&String> = a.keys().collect();
            let mut bk: Vec<&String> = b.keys().collect();
            ak.sort();
            bk.sort();
            if ak != bk {
                let missing: Vec<&&String> = bk.iter().filter(|k| !a.contains_key(**k)).collect();
                let extra: Vec<&&String> = ak.iter().filter(|k| !b.contains_key(**k)).collect();
                return Some(format!(
                    "{path}: key sets differ\n  missing in rust: {missing:?}\n  extra in rust:   {extra:?}"
                ));
            }
            // Iterate the oracle's order so the first reported difference is the
            // earliest one a reader of the JSON would reach.
            for (k, bv) in b {
                if let Some(d) = first_diff(&a[k], bv, &format!("{path}.{k}")) {
                    return Some(d);
                }
            }
            None
        }
        (Json::Array(a), Json::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "{path}: length differs\n  rust:   {}\n  python: {}",
                    a.len(),
                    b.len()
                ));
            }
            for (i, (av, bv)) in a.iter().zip(b).enumerate() {
                if let Some(d) = first_diff(av, bv, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            None
        }
        _ if rust == python => None,
        _ => Some(format!(
            "{path}: differs\n  rust:   {rust}\n  python: {python}"
        )),
    }
}

/// A store delta in the oracle's shape: a map from name to a change record.
fn delta_json(delta: &[(String, StoreChange)]) -> Json {
    let mut out = Map::new();
    for (name, change) in delta {
        let entry = match change {
            StoreChange::Added { value, tag } => {
                json!({"change": "added", "value": value, "tag": tag})
            }
            StoreChange::Updated { from, value, tag } => {
                json!({"change": "updated", "from": from, "value": value, "tag": tag})
            }
        };
        out.insert(name.clone(), entry);
    }
    Json::Object(out)
}

fn store_json(pairs: &[(String, Json)]) -> Json {
    let mut out = Map::new();
    for (k, v) in pairs {
        out.insert(k.clone(), v.clone());
    }
    Json::Object(out)
}

/// The `final` block: store, output, propositions, points, trace length.
fn final_json(s: &Session) -> Json {
    let mut props = Map::new();
    for (name, state) in &s.compiler.propositions {
        let motions: Vec<Json> = state
            .motions
            .iter()
            .map(|m| json!({"name": m.name, "text": m.text, "score": state.score(&m.name)}))
            .collect();
        let verdicts: Vec<Json> = state
            .verdicts
            .iter()
            .map(|v| {
                json!({
                    "motion": v.motion,
                    "stance": v.stance,
                    "confidence": v.confidence,
                    "line": v.line,
                })
            })
            .collect();
        props.insert(
            name.clone(),
            json!({"motions": motions, "verdicts": verdicts}),
        );
    }

    let mut points = Map::new();
    for (name, p) in &s.compiler.points {
        points.insert(
            name.clone(),
            ndombolo_core::value::render(&ndombolo_core::Value::Point(p.clone())),
        );
    }

    json!({
        "store": store_json(&s.store()),
        "output": s.compiler.output,
        "propositions": Json::Object(props),
        "points": Json::Object(points),
        "trace_length": s.compiler.trace.len(),
    })
}

#[test]
fn rust_evaluator_matches_the_python_oracle() {
    let corpus: Corpus = serde_json::from_str(CORPUS).expect("eval.json should parse");

    let mut checked_cells = 0usize;
    let mut checked_events = 0usize;
    let mut failing_cells = 0usize;

    for script in &corpus.scripts {
        let mut session = Session::new();

        // The oracle stores each cell's text, not the whole script, and the
        // separator lines it split on are gone. Running the recorded cells
        // directly is both simpler and more faithful than reconstituting a
        // source and re-splitting it -- and it checks `first_line`
        // independently, since the oracle's value is passed in rather than
        // derived here. `split_cells` has its own tests in the crate.
        for cell in &script.cells {
            let got = session.run_cell(cell.index, cell.first_line, &cell.text);
            checked_cells += 1;
            if !cell.ok {
                failing_cells += 1;
            }

            let ctx = |what: &str, d: String| -> String {
                format!(
                    "{} cell {} (first_line {}) -- {}\n{}\n--- cell source ---\n{}",
                    script.name, cell.index, cell.first_line, what, d, cell.text
                )
            };

            if got.ok != cell.ok {
                panic!(
                    "{}",
                    ctx(
                        "ok differs",
                        format!(
                            "  rust:   ok={} err={:?}\n  python: ok={} err={:?}",
                            got.ok, got.error, cell.ok, cell.error
                        )
                    )
                );
            }

            // The error, when there is one. Its message carries the `line N: `
            // prefix, so a port that got the line right and the text wrong --
            // or the reverse -- fails here rather than passing on the flag.
            match (&got.error, &cell.error) {
                (Some(g), Some(p)) => {
                    let mine = json!({
                        "phase": g.phase,
                        "message": g.message,
                        "line": g.line,
                    });
                    if let Some(d) = first_diff(&mine, p, "error") {
                        panic!("{}", ctx("error differs", d));
                    }
                }
                (None, None) => {}
                _ => unreachable!("ok flags already agreed"),
            }

            // The trace first: it is the thing under test.
            if let Some(want) = &cell.trace {
                let mine = Json::Array(got.trace.clone());
                checked_events += got.trace.len();
                if let Some(d) = first_diff(&mine, want, "trace") {
                    panic!("{}", ctx("trace differs", d));
                }
            } else if !got.trace.is_empty() {
                panic!(
                    "{}",
                    ctx(
                        "trace present in rust, absent in oracle",
                        format!("  rust: {} events", got.trace.len())
                    )
                );
            }

            if let Some(want) = &cell.store_delta {
                let mine = delta_json(&got.store_delta);
                if let Some(d) = first_diff(&mine, want, "store_delta") {
                    panic!("{}", ctx("store delta differs", d));
                }
            } else if !got.store_delta.is_empty() {
                panic!("{}", ctx("store delta present in rust, absent in oracle", String::new()));
            }

            if let Some(want) = &cell.output {
                let mine = json!(got.output);
                if let Some(d) = first_diff(&mine, want, "output") {
                    panic!("{}", ctx("output differs", d));
                }
            } else if !got.output.is_empty() {
                panic!("{}", ctx("output present in rust, absent in oracle", String::new()));
            }

            if !cell.ok {
                break;
            }
        }

        let mine = final_json(&session);
        if let Some(d) = first_diff(&mine, &script.final_, "final") {
            panic!("{} -- final state differs\n{}", script.name, d);
        }
    }

    // A corpus that shrank -- a truncated regeneration, a dump that failed
    // partway -- would make this test pass by checking almost nothing. The
    // counts are the ones `dump_eval.py` reports.
    assert_eq!(corpus.scripts.len(), 66, "corpus lost scripts");
    assert!(checked_cells > 150, "only {checked_cells} cells checked");
    assert!(checked_events > 1500, "only {checked_events} events checked");
    assert!(failing_cells > 10, "only {failing_cells} failing cells checked");
}

/// The edge scripts are where the port's hazards live: each pins one line of
/// the Python where the obvious Rust is the wrong Rust. If a regenerated corpus
/// dropped them the test above would still pass on the generated sweep, which
/// exercises almost none of them.
#[test]
fn edge_scripts_are_in_the_corpus() {
    let corpus: Corpus = serde_json::from_str(CORPUS).expect("eval.json should parse");
    let names: Vec<&str> = corpus.scripts.iter().map(|s| s.name.as_str()).collect();

    for wanted in [
        "edge-num-integral",      // render collapses integral floats
        "edge-bool-is-num",       // 1 == true
        "edge-text-formatting",   // %g, and int() for integral floats
        "edge-arith",             // % is fmod, not rem_euclid
        "edge-map-order",         // insertion order, not sorted
        "edge-indexing",          // missing map key is none; list index is not
        "edge-truthy",            // truthiness by tag
        "edge-short-circuit",     // and/or return a bool
        "edge-points",            // confidence fallback and clamp
        "edge-verdicts",          // noisy-or
        "edge-verdict-reads",     // reads is a dict keyed by motion: last wins
        "edge-closures",          // capture the defining environment
        "edge-within",            // opens a scope whichever way it goes
        "edge-loops",             // per-iteration bind events, while-exit
        "edge-builtins",          // banker's rounding
        "edge-pipe",              // right operand evaluated first
        "edge-err-pipe-order",    // ... observably so: the error names `f`, not `x`
        "edge-resolve-expr",      // notes are str(render(v))
        "edge-noops",             // a loose keyword's body does not run
        "edge-assign-undeclared", // assignment defines
        "edge-err-inconclusive-conf", // the one parse-stage failure
        "edge-err-stops-later-cells",
        "edge-multi-cell",
        "signal",
        "assay",
    ] {
        assert!(
            names.contains(&wanted),
            "corpus is missing {wanted}; regenerate with figures/dump_eval.py"
        );
    }
}
