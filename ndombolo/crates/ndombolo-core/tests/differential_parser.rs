//! Differential test: the Rust parser against the frozen Python oracle.
//!
//! `ast.json` is produced by `figures/dump_ast.py` over the same corpus as the
//! lexer harness, plus a hand-written edge set reaching the grammar the
//! generated scripts never write — pipes, maps, indexes, unary operators,
//! `otherwise`, loose keywords, imports, verdict confidences.
//!
//! The comparison is on the **serialised tree**, not on a structural walk, and
//! that is deliberate. The Python AST is plain dicts, so what a port has to
//! reproduce is the dict — key for key, including which keys are *absent*.
//! `graph.py` walks statements by probing field names rather than by matching
//! on kind, so a key present where the Python omits it changes the graph rather
//! than merely the JSON. Comparing `serde_json::Value` is the only comparison
//! that sees that; a walk over the Rust enum would be blind to exactly the
//! distinction it needs to check.
//!
//! Regenerate after any parser change:
//!     python figures/dump_ast.py

use ndombolo_core::parser::parse;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Corpus {
    scripts: Vec<Script>,
}

#[derive(Deserialize)]
struct Script {
    name: String,
    cells: Vec<Cell>,
}

#[derive(Deserialize)]
struct Cell {
    first_line: usize,
    text: String,
    /// The oracle's tree, left as raw JSON: it is the comparison target, so
    /// giving it a Rust type would beg the question.
    ast: Value,
}

/// The first path where two trees differ, as a dotted trail.
///
/// A whole-tree diff of a 200-node cell is unreadable, and the failures that
/// matter here are one wrong key deep inside an otherwise-identical tree, so
/// the report has to name the site rather than dump both sides.
fn first_diff(rust: &Value, python: &Value, path: &str) -> Option<String> {
    match (rust, python) {
        (Value::Object(r), Value::Object(p) ) => {
            // Key sets first: `Noop.body` and `Support.conf` are the two places
            // where presence, not value, is the whole question.
            let mut keys: Vec<&String> = r.keys().chain(p.keys()).collect();
            keys.sort_unstable();
            keys.dedup();
            for k in keys {
                match (r.get(k), p.get(k)) {
                    (Some(rv), Some(pv)) => {
                        if let Some(d) = first_diff(rv, pv, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    (Some(rv), None) => {
                        return Some(format!(
                            "{path}.{k}: rust has this key ({rv}), python omits it"
                        ))
                    }
                    (None, Some(pv)) => {
                        return Some(format!(
                            "{path}.{k}: python has this key ({pv}), rust omits it"
                        ))
                    }
                    (None, None) => unreachable!(),
                }
            }
            None
        }
        (Value::Array(r), Value::Array(p)) => {
            if r.len() != p.len() {
                return Some(format!(
                    "{path}: length {} vs {}",
                    r.len(),
                    p.len()
                ));
            }
            for (i, (rv, pv)) in r.iter().zip(p).enumerate() {
                if let Some(d) = first_diff(rv, pv, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            None
        }
        _ if rust == python => None,
        _ => Some(format!("{path}: rust {rust} vs python {python}")),
    }
}

fn count_nodes(v: &Value) -> usize {
    match v {
        Value::Array(items) => items.iter().map(count_nodes).sum(),
        Value::Object(m) => {
            usize::from(m.contains_key("kind")) + m.values().map(count_nodes).sum::<usize>()
        }
        _ => 0,
    }
}

#[test]
fn rust_parser_matches_python_oracle() {
    let raw = include_str!("ast.json");
    let corpus: Corpus = serde_json::from_str(raw).expect("parse ast.json");

    let mut cells = 0usize;
    let mut nodes = 0usize;

    for script in &corpus.scripts {
        for (k, cell) in script.cells.iter().enumerate() {
            let tree = parse(&cell.text, cell.first_line)
                .unwrap_or_else(|e| panic!("{} cell {k}: rust failed: {e}", script.name));
            let got = serde_json::to_value(&tree).expect("serialise rust tree");

            if let Some(d) = first_diff(&got, &cell.ast, "") {
                panic!(
                    "{} cell {k} (line {}): {d}\n--- source ---\n{}",
                    script.name, cell.first_line, cell.text
                );
            }

            cells += 1;
            nodes += count_nodes(&cell.ast);
        }
    }

    assert!(cells > 100, "corpus looks truncated: {cells} cells");
    eprintln!("{} scripts, {cells} cells, {nodes} nodes agree", corpus.scripts.len());
}

/// Floats have to survive the round trip as floats.
///
/// Every Turbulance number is `float(t.value)` in the Python, so `1` is `1.0`.
/// If the Rust serialised it as an integer the trees would differ on every
/// literal — worth its own test, since a `serde_json` change could silently
/// start emitting `1` for a whole `f64`.
#[test]
fn integral_literals_stay_floats() {
    let tree = parse("item a = 1", 1).unwrap();
    let v = serde_json::to_value(&tree).unwrap();
    assert!(v[0]["expr"]["value"].is_f64(), "got {}", v[0]["expr"]["value"]);
    assert_eq!(serde_json::to_string(&v[0]["expr"]["value"]).unwrap(), "1.0");
}

/// The oracle's own edge scripts must be present.
///
/// The generated corpus exercises a small part of the grammar; if the edge set
/// went missing from `ast.json` the main test would still pass while checking
/// much less. Naming a few pins that.
#[test]
fn edge_scripts_are_in_the_corpus() {
    let corpus: Corpus = serde_json::from_str(include_str!("ast.json")).unwrap();
    let names: Vec<&str> = corpus.scripts.iter().map(|s| s.name.as_str()).collect();
    for want in [
        "edge-paren-unwrapped",
        "edge-noop-import",
        "edge-noop-loose",
        "edge-maps",
        "edge-pipe",
        "edge-proposition",
    ] {
        assert!(names.contains(&want), "{want} missing from ast.json");
    }
}
