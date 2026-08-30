//! Differential test: the Rust lexer against the frozen Python oracle.
//!
//! `tokens.json` is produced by `figures/dump_tokens.py` from the same 28-script
//! corpus the paper's measurements use — a size ladder, a depth family, a width
//! family, and the two hand-written examples. It is generated without an RNG, so
//! the corpus is fixed and this comparison is reproducible.
//!
//! The comparison is token for token on all four fields, not on counts: a port
//! that agreed on how many tokens it produced while disagreeing on where they
//! sit would pass a weaker test and fail in the record, where `line` and `col`
//! are what make a trace site name a place in the document.
//!
//! Regenerate after any lexer change:
//!     python figures/dump_tokens.py

use ndombolo_core::lexer::{tokenize, Kind, Token};
use serde::Deserialize;

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
    /// `[kind, value, line, col]`, as the Python spells them.
    tokens: Vec<(String, String, usize, usize)>,
}

fn show(t: &Token) -> String {
    format!("{} {:?} @{}:{}", t.kind.as_str(), t.value, t.line, t.col)
}

#[test]
fn rust_lexer_matches_python_oracle() {
    let raw = include_str!("tokens.json");
    let corpus: Corpus = serde_json::from_str(raw).expect("parse tokens.json");

    let mut cells = 0usize;
    let mut tokens = 0usize;

    for script in &corpus.scripts {
        for (k, cell) in script.cells.iter().enumerate() {
            let got = tokenize(&cell.text, cell.first_line)
                .unwrap_or_else(|e| panic!("{} cell {k}: rust failed: {e}", script.name));

            assert_eq!(
                got.len(),
                cell.tokens.len(),
                "{} cell {k}: token count differs\n  rust:   {:?}\n  python: {:?}",
                script.name,
                got.iter().map(show).collect::<Vec<_>>(),
                cell.tokens,
            );

            for (i, (want, have)) in cell.tokens.iter().zip(&got).enumerate() {
                let (kind, value, line, col) = want;
                assert_eq!(
                    (have.kind.as_str(), have.value.as_str(), have.line, have.col),
                    (kind.as_str(), value.as_str(), *line, *col),
                    "{} cell {k} token {i}: mismatch",
                    script.name,
                );
            }

            cells += 1;
            tokens += got.len();
        }
    }

    // Guard against a silently empty or truncated fixture: a harness that
    // compares nothing would pass just as quietly as one that compares
    // everything. Counted in two parts, because the generated corpus and the
    // hand-written edge cases can each vanish without changing the other --
    // and a bare total would not notice.
    let edge = corpus.scripts.iter().filter(|s| s.name.starts_with("edge-")).count();
    assert_eq!(corpus.scripts.len() - edge, 28, "generated corpus size changed");
    assert_eq!(edge, 16, "edge-case corpus size changed");
    assert!(tokens > 10_000, "only {tokens} tokens compared");
    eprintln!("{} scripts, {cells} cells, {tokens} tokens agree", corpus.scripts.len());
}

#[test]
fn oracle_covers_every_operator_and_subtlety() {
    // The generated corpus is clean machine-written Turbulance: measured, it
    // reaches 16 of the 26 operators and none of the lexer's fiddly paths. The
    // agreement it establishes is therefore broad but shallow, and the parts
    // most likely to drift -- tab stops, `unescape`, the leading-dot rule --
    // were the parts resting on the port author's expectations. The `edge-`
    // scripts exist to move them onto the oracle; this test fails if they are
    // dropped, so that the coverage cannot quietly regress.
    let raw = include_str!("tokens.json");
    let corpus: Corpus = serde_json::from_str(raw).expect("parse tokens.json");

    let cells = || corpus.scripts.iter().flat_map(|s| &s.cells);
    let toks = || cells().flat_map(|c| &c.tokens);

    for op in [
        "==", "!=", "<=", ">=", "=>", "|>", "&&", "||",
        "=", "<", ">", "+", "-", "*", "/", "%",
        "(", ")", "[", "]", "{", "}", ",", ":", ".", "|",
    ] {
        assert!(
            toks().any(|(k, v, ..)| k == "op" && v == op),
            "corpus exercises no `{op}` operator; agreement on it is untested",
        );
    }

    assert!(
        cells().any(|c| c.text.contains('\t')),
        "no cell contains a tab; tab-stop expansion is untested",
    );
    // The oracle already applied `unescape`, so an escape shows up as the
    // decoded character in a `str` value, never as a backslash pair.
    for (what, ch) in [("\\n", '\n'), ("\\t", '\t'), ("\\\\", '\\'), ("\\\"", '"')] {
        assert!(
            toks().any(|(k, v, ..)| k == "str" && v.contains(ch)),
            "no string decodes `{what}`; that branch of unescape is untested",
        );
    }
    assert!(
        toks().any(|(k, v, ..)| k == "num" && v.starts_with('.')),
        "no leading-dot number; the `.5` vs `a.b` rule is untested",
    );
    assert!(
        toks().any(|(k, v, ..)| k == "str" && !v.is_ascii()),
        "no non-ASCII string; `col` as a byte offset is untested",
    );
}

#[test]
fn oracle_covers_every_token_kind() {
    // A differential test is only as strong as the corpus's variety. If a kind
    // never appears, agreement on it is untested rather than established.
    let raw = include_str!("tokens.json");
    let corpus: Corpus = serde_json::from_str(raw).expect("parse tokens.json");

    let mut seen: Vec<&str> = corpus
        .scripts
        .iter()
        .flat_map(|s| &s.cells)
        .flat_map(|c| &c.tokens)
        .map(|(kind, ..)| kind.as_str())
        .collect();
    seen.sort_unstable();
    seen.dedup();

    for kind in [
        Kind::Kw, Kind::Ident, Kind::Num, Kind::Str, Kind::Op,
        Kind::Newline, Kind::Indent, Kind::Dedent, Kind::Eof,
    ] {
        assert!(
            seen.contains(&kind.as_str()),
            "corpus exercises no `{}` token; agreement on it is untested",
            kind.as_str(),
        );
    }
}
