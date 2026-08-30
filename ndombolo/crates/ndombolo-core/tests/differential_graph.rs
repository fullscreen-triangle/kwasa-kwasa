//! The Rust graph builder against the frozen Python one, cell by cell.
//!
//! The fourth of the differential harnesses. `tests/graph.json` is written by
//! `figures/dump_graph.py` from `prototype/`, which is a reference
//! implementation and not a performance claim (paper, §8); its only remaining
//! job is to be the oracle this checks against.
//!
//! # Why the comparison is per cell rather than final
//!
//! The graph is accretive: `add_cell` extends it and nothing removes an item or
//! decreases a weight, which is what makes prefix containment hold (paper,
//! Prop. 2.7). Comparing only the final graph would test the sum and leave the
//! accretion untested -- a port that rebuilt the whole graph from scratch on
//! every cell would agree on every final graph in this corpus and still be the
//! wrong runtime. So each cell's graph is compared as it stands after that cell,
//! which is also what puts `stage` under test: an item declared in cell 1 must
//! carry stage 1 forever after, not the stage of the cell that last mentioned
//! it.
//!
//! # What the comparison is not
//!
//! It is not a count check. Two graphs with the same item and contact counts can
//! disagree about which names are in contact, and the counts are compared last,
//! after the lists that determine them, so a difference is reported at the pair
//! it happened at rather than as a number that came out wrong.

use ndombolo_core::graph::GraphBuilder;
use ndombolo_core::parser::parse;
use serde::Deserialize;
use serde_json::Value as Json;

const CORPUS: &str = include_str!("graph.json");

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

/// A cell as the oracle recorded it. The graph is left as raw JSON: the harness
/// compares against the oracle's own shape rather than round-tripping it through
/// a Rust type, which could only launder a disagreement into agreement.
#[derive(Deserialize)]
struct Cell {
    index: usize,
    first_line: usize,
    text: String,
    graph: Json,
}

/// The first place two documents differ, as a dotted path.
///
/// The same walker the evaluator's harness uses, and for the same reason: key
/// sets are compared before values, so a field the port forgot to emit is
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
                    "{path}: length differs\n  rust:   {} entries\n  python: {} entries\n  rust:   {rust}\n  python: {python}",
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

#[test]
fn rust_graph_matches_the_python_oracle() {
    let corpus: Corpus = serde_json::from_str(CORPUS).expect("graph.json should parse");

    let mut checked_cells = 0usize;
    let mut checked_items = 0usize;
    let mut checked_contacts = 0usize;

    for script in &corpus.scripts {
        // One builder across the whole script, as `run.py` keeps one: the stage
        // is the cell index, and the graph carries over.
        let mut builder = GraphBuilder::new();

        for cell in &script.cells {
            // The oracle only recorded cells that parsed, so a parse failure
            // here is a disagreement with the parser stage, not with this one --
            // and it is reported as such rather than being swallowed.
            let body = match parse(&cell.text, cell.first_line) {
                Ok(body) => body,
                Err(e) => panic!(
                    "{} cell {} (first_line {}) -- the oracle parsed this cell and the Rust \
                     parser did not: {e}\n--- cell source ---\n{}",
                    script.name, cell.index, cell.first_line, cell.text
                ),
            };

            builder.add_cell(&body, cell.index);
            checked_cells += 1;

            let mine = builder.graph.to_json();
            if let Some(d) = first_diff(&mine, &cell.graph, "graph") {
                panic!(
                    "{} cell {} (first_line {}) -- graph differs after this cell\n{}\n\
                     --- cell source ---\n{}",
                    script.name, cell.index, cell.first_line, d, cell.text
                );
            }
        }

        checked_items += builder.graph.item_count();
        checked_contacts += builder.graph.contact_count();

        let mine = builder.graph.to_json();
        if let Some(d) = first_diff(&mine, &script.final_, "final") {
            panic!("{} -- final graph differs\n{}", script.name, d);
        }
    }

    // A corpus that shrank -- a truncated regeneration, a dump that failed
    // partway -- would make this test pass by checking almost nothing. The
    // counts are the ones `dump_graph.py` reports.
    assert_eq!(corpus.scripts.len(), 50, "corpus lost scripts");
    assert!(checked_cells > 150, "only {checked_cells} cells checked");
    assert!(checked_items > 500, "only {checked_items} items checked");
    assert!(
        checked_contacts > 500,
        "only {checked_contacts} contacts checked"
    );
}

/// Accretion is only under test where a script has more than one cell, and the
/// generated sweep is single-cell throughout. If a regenerated corpus lost the
/// multi-cell scripts, the test above would still pass while checking the graph
/// of a one-shot build -- which is exactly the wrong port it exists to catch.
#[test]
fn the_corpus_actually_exercises_accretion() {
    let corpus: Corpus = serde_json::from_str(CORPUS).expect("graph.json should parse");

    let multi: Vec<&str> = corpus
        .scripts
        .iter()
        .filter(|s| s.cells.len() > 1)
        .map(|s| s.name.as_str())
        .collect();

    for wanted in ["edge-stages", "edge-restage", "edge-weights-only"] {
        assert!(
            multi.contains(&wanted),
            "corpus is missing multi-cell script {wanted}; regenerate with \
             figures/dump_graph.py. Note the separator is `// ---`, not `---`: \
             with the wrong one the script parses as a single failing cell and \
             silently contributes nothing."
        );
    }

    // And the accretion has to be visible, not just present: a later cell must
    // add to an earlier cell's graph rather than starting from nothing.
    let restage = corpus
        .scripts
        .iter()
        .find(|s| s.name == "edge-restage")
        .expect("checked above");
    let first = &restage.cells[0].graph["items"];
    let last = &restage.cells[restage.cells.len() - 1].graph["items"];
    assert!(
        last.as_array().map(|a| a.len()) > first.as_array().map(|a| a.len()),
        "edge-restage does not grow across cells, so it pins no accretion"
    );
}

/// The edge scripts are where the port's hazards live: each pins one line of the
/// Python where the obvious Rust is the wrong Rust. The generated sweep builds
/// large but structurally uniform graphs and exercises almost none of them.
#[test]
fn edge_scripts_are_in_the_corpus() {
    let corpus: Corpus = serde_json::from_str(CORPUS).expect("graph.json should parse");
    let names: Vec<&str> = corpus.scripts.iter().map(|s| s.name.as_str()).collect();

    for wanted in [
        "edge-dedup",           // free_names is a set: a + a is one contact
        "edge-accumulate",      // ...but weights accumulate across statements
        "edge-redeclare",       // setdefault keeps the first line and stage
        "edge-dropped-edges",   // undeclared endpoint, and self-contact
        "edge-assign-not-decl", // assignment is not a declaration
        "edge-funxn-scope",     // a funxn opens a scope; params are not items
        "edge-given-branches",  // then and otherwise are two block fields
        "edge-within-target",   // the subject is an expression field
        "edge-prop-motions",    // proposition -> motions, and body -> proposition
        "edge-verdict-conf",    // conf is probed; inconclusive has no conf key
        "edge-loops",           // iter is an expression field, body a block
        "edge-while-cond",      // cond, the field given also uses
        "edge-noop-shapes",     // import has no body key; a loose keyword has one
        "edge-noop-body",       // ...and that body is walked: sabotage found this one
        "edge-nested-scopes",   // the inner opener replaces the outer
        "edge-decl-kinds",      // point, funxn, hypothesis, proposition
        "edge-forward-ref",     // declare-then-link, not one pass
        "edge-collections",     // map values are names, keys are not
        "edge-expr-shapes",     // index, field access, pipe: the fallthrough probe
        "signal",
        "assay",
    ] {
        assert!(
            names.contains(&wanted),
            "corpus is missing {wanted}; regenerate with figures/dump_graph.py"
        );
    }
}
