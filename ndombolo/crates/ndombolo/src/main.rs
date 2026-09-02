//! `ndombolo` -- run the cells of a `.ndo` document.
//!
//! A `.ndo` file is prose with fenced blocks. A block tagged `turbulance` is a
//! cell; running it writes an `output` block below it. The document is the
//! report; the record beside it is the history.
//!
//! ```text
//! ndombolo run   notes.ndo [--cell N]   run cells, write outputs back
//! ndombolo clear notes.ndo [--cell N]   remove output blocks (the record still moves)
//! ndombolo cells notes.ndo              list the cells without running them
//! ndombolo graph notes.ndo              the script graph, as JSON
//! ndombolo trace notes.ndo              every trace event, as JSON
//! ndombolo record notes.ndo             the deposits behind the document
//! ndombolo new   notes.ndo              write a starter document
//! ndombolo edit  notes.ndo              open the document in the editor
//! ```
//!
//! # The editor is a second face on this runtime, not a second runtime
//!
//! `edit` serves the same document over loopback and runs cells through the
//! same [`Session`] with the same replay discipline. Anything the editor can do
//! to a document, `run` does identically; the difference is that the editor can
//! also ask a local model about the trace, and the model's answer enters the
//! file only as prose the user places.
//!
//! # One session per invocation
//!
//! `run` builds a fresh [`Session`] and replays *every* cell in order up to the
//! one asked for, because a cell's meaning depends on the store the cells before
//! it left. Determination by search, not lookup (paper, B3): there is no cache
//! of a previous run's store to resume from, and adding one would be a violation
//! rather than an optimisation.
//!
//! # `--cell` selects what is *written*, not what is run
//!
//! With `--cell 3`, cells 0..=3 all run -- they have to, to build the store --
//! but only cell 3's output block is spliced. The earlier runs still deposit
//! (paper, B4: every propagation deposits, including those whose output is
//! discarded), which is why the record advances by more than one.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ndombolo_core::doc::{format_result, Document};
use ndombolo_core::graph::GraphBuilder;
use ndombolo_core::parser::parse;
use ndombolo_core::session::Session;

mod editor;
mod http;
mod ollama;
mod record;
use record::Record;

const USAGE: &str = "\
ndombolo -- run the cells of a .ndo document

    ndombolo run    <file.ndo> [--cell N]   run cells and write outputs back
    ndombolo clear  <file.ndo> [--cell N]   remove output blocks
    ndombolo cells  <file.ndo>              list cells without running them
    ndombolo graph  <file.ndo>              the script graph, as JSON
    ndombolo trace  <file.ndo>              every trace event, as JSON
    ndombolo record <file.ndo>              the deposits behind this document
    ndombolo new    <file.ndo>              write a starter document
    ndombolo edit   <file.ndo>              open the document in the editor

Options
    --cell N     act on cell N only (0-based). Earlier cells still run: a
                 cell's meaning depends on the store the ones before it left.
    --port N     the port the editor listens on, on loopback (default 7749)
    --model NAME the ollama model the editor asks (default llama3.2)
    --host URL   where ollama listens (default http://127.0.0.1:11434)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ndombolo: {e}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(args: &[String]) -> Result<(), String> {
    let mut positional: Vec<&str> = Vec::new();
    let mut cell: Option<usize> = None;
    let mut port: u16 = 7749;
    let mut model = ollama::DEFAULT_MODEL.to_string();
    let mut host = ollama::DEFAULT_HOST.to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" | "help" => {
                print!("{USAGE}");
                return Ok(());
            }
            "--cell" => {
                let n = args
                    .get(i + 1)
                    .ok_or_else(|| "--cell needs a number".to_string())?;
                cell = Some(
                    n.parse()
                        .map_err(|_| format!("--cell wants a number, got {n:?}"))?,
                );
                i += 2;
            }
            "--port" => {
                let n = value(args, i, "--port")?;
                port = n
                    .parse()
                    .map_err(|_| format!("--port wants a number, got {n:?}"))?;
                i += 2;
            }
            "--model" => {
                model = value(args, i, "--model")?.to_string();
                i += 2;
            }
            "--host" => {
                host = value(args, i, "--host")?.to_string();
                i += 2;
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other}\n\n{USAGE}"));
            }
            other => {
                positional.push(other);
                i += 1;
            }
        }
    }

    let (cmd, file) = match positional.as_slice() {
        [] => {
            print!("{USAGE}");
            return Ok(());
        }
        [cmd] => return Err(format!("{cmd} needs a file\n\n{USAGE}")),
        [cmd, file, ..] => (*cmd, PathBuf::from(file)),
    };

    match cmd {
        "run" => cmd_run(&file, cell),
        "clear" => cmd_clear(&file, cell),
        "cells" => cmd_cells(&file),
        "graph" => cmd_graph(&file),
        "trace" => cmd_trace(&file),
        "record" => cmd_record(&file),
        "new" => cmd_new(&file),
        "edit" => cmd_edit(&file, port, &model, &host),
        other => Err(format!("unknown command {other:?}\n\n{USAGE}")),
    }
}

/// The argument after `flag`, or an error naming it.
fn value<'a>(args: &'a [String], i: usize, flag: &str) -> Result<&'a str, String> {
    args.get(i + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} needs a value"))
}

// -- reading and writing ---------------------------------------------------

fn load(path: &Path) -> Result<Document, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(Document::parse(&text))
}

/// Write via a temporary file in the same directory, then rename.
///
/// The document is the user's report and may hold prose that exists nowhere
/// else. A partial write from a crash mid-`write_all` would destroy it; a
/// rename either happens or does not.
fn save(path: &Path, doc: &Document) -> Result<(), String> {
    let tmp = path.with_extension("ndo.tmp");
    std::fs::write(&tmp, doc.render()).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot replace {}: {e}", path.display())
    })
}

// -- run -------------------------------------------------------------------

fn cmd_run(path: &Path, only: Option<usize>) -> Result<(), String> {
    let mut doc = load(path)?;
    let cells = doc.cell_indices();

    if cells.is_empty() {
        println!("{}: no turbulance cells", path.display());
        return Ok(());
    }
    if let Some(n) = only {
        if n >= cells.len() {
            return Err(format!(
                "--cell {n}: the document has {} cell{}",
                cells.len(),
                if cells.len() == 1 { "" } else { "s" }
            ));
        }
    }

    let mut rec = Record::open(path).map_err(|e| format!("cannot open the record: {e}"))?;
    let mut session = Session::new();
    let last = only.unwrap_or(cells.len() - 1);

    // Splices are collected and applied afterwards: `splice` inserts blocks,
    // which shifts every later block index, and `cells` was taken before.
    let mut splices: Vec<(usize, String)> = Vec::new();
    let mut stopped_at: Option<usize> = None;

    for (n, &block) in cells.iter().enumerate().take(last + 1) {
        let text = doc.cell_text(block).expect("cell_indices yields cells");
        let first_line = doc.line_of(block) + 1; // the fence line itself is not source
        let result = session.run_cell(n, first_line, &text);

        // Every run deposits, including the ones whose output is not written
        // back (paper, B4).
        let touched: Vec<String> = result.store_delta.iter().map(|(k, _)| k.clone()).collect();
        rec.deposit(n, result.trace.len(), touched, result.ok)
            .map_err(|e| format!("cannot deposit: {e}"))?;

        if only.is_none() || only == Some(n) {
            splices.push((
                block,
                format_result(&result.output, &result.store_delta, result.error.as_ref()),
            ));
        }

        if !result.ok {
            stopped_at = Some(n);
            break;
        }
    }

    // Later blocks first, so an earlier splice cannot invalidate a later index.
    splices.sort_by_key(|(b, _)| std::cmp::Reverse(*b));
    for (block, text) in &splices {
        doc.splice(*block, text);
    }
    save(path, &doc)?;

    let ran = stopped_at.map(|n| n + 1).unwrap_or(last + 1);
    println!(
        "{}: ran {ran} cell{}, record {}",
        path.display(),
        if ran == 1 { "" } else { "s" },
        rec.count()
    );
    if let Some(n) = stopped_at {
        // Not a verdict on the document (paper, B6) -- a statement of where the
        // run stopped, so the reader knows the later cells did not run.
        println!("stopped at cell {n}; cells after it did not run");
    }
    Ok(())
}

// -- clear -----------------------------------------------------------------

fn cmd_clear(path: &Path, only: Option<usize>) -> Result<(), String> {
    let mut doc = load(path)?;
    let cells = doc.cell_indices();

    let targets: Vec<usize> = match only {
        Some(n) => vec![*cells
            .get(n)
            .ok_or_else(|| format!("--cell {n}: the document has {} cells", cells.len()))?],
        None => cells,
    };

    // Descending, for the same index-shift reason as `run`.
    let mut cleared = 0;
    for block in targets.into_iter().rev() {
        if doc.clear_output(block) {
            cleared += 1;
        }
    }
    save(path, &doc)?;

    // Clearing removes a view; it does not remove the deposits that produced
    // it. The record is printed here so that is visible rather than assumed.
    let rec = Record::open(path).map_err(|e| format!("cannot open the record: {e}"))?;
    println!(
        "{}: cleared {cleared} output block{}, record still {}",
        path.display(),
        if cleared == 1 { "" } else { "s" },
        rec.count()
    );
    Ok(())
}

// -- reading the document --------------------------------------------------

fn cmd_cells(path: &Path) -> Result<(), String> {
    let doc = load(path)?;
    for (n, &block) in doc.cell_indices().iter().enumerate() {
        let text = doc.cell_text(block).unwrap_or_default();
        let lines = text.lines().count();
        let first = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
        println!(
            "cell {n}  line {}  {lines} line{}  {}",
            doc.line_of(block),
            if lines == 1 { "" } else { "s" },
            first.trim()
        );
    }
    Ok(())
}

/// The script graph after every cell, accretive (paper, Prop. 2.7).
///
/// Cells that do not parse are skipped rather than fatal: the graph is a static
/// reading of the script, and a half-written cell should not stop you seeing
/// what the rest declares.
fn cmd_graph(path: &Path) -> Result<(), String> {
    let doc = load(path)?;
    let mut builder = GraphBuilder::new();
    let mut skipped: Vec<usize> = Vec::new();

    for (n, &block) in doc.cell_indices().iter().enumerate() {
        let text = doc.cell_text(block).unwrap_or_default();
        match parse(&text, doc.line_of(block) + 1) {
            Ok(body) => builder.add_cell(&body, n),
            Err(_) => skipped.push(n),
        }
    }

    let mut out = builder.graph.to_json();
    if !skipped.is_empty() {
        out["cells_that_did_not_parse"] = serde_json::json!(skipped);
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    Ok(())
}

/// Every event the run emitted, in order.
///
/// This is the atomic-interrogation surface: each event names the rule that
/// fired and the item it touched, so a question about the output can be
/// answered by search over the events rather than by re-reasoning about the
/// script.
fn cmd_trace(path: &Path) -> Result<(), String> {
    let doc = load(path)?;
    let mut session = Session::new();
    let mut events: Vec<serde_json::Value> = Vec::new();

    for (n, &block) in doc.cell_indices().iter().enumerate() {
        let text = doc.cell_text(block).unwrap_or_default();
        let result = session.run_cell(n, doc.line_of(block) + 1, &text);
        for ev in result.trace {
            events.push(serde_json::json!({ "cell": n, "event": ev }));
        }
        if !result.ok {
            break;
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "events": events.len(),
            "trace": events,
        }))
        .unwrap()
    );
    Ok(())
}

fn cmd_record(path: &Path) -> Result<(), String> {
    let rec = Record::open(path).map_err(|e| format!("cannot open the record: {e}"))?;
    let entries = rec.entries().map_err(|e| format!("cannot read: {e}"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "record": rec.count(),
            "file": rec.path().display().to_string(),
            "deposits": entries,
        }))
        .unwrap()
    );
    Ok(())
}

// -- new -------------------------------------------------------------------

const STARTER: &str = "\
# A new document

Write prose here as you would in any note. It is already rendered -- there is
no source view and no preview, just the one document.

A block tagged `turbulance` is a cell. Run it and its output appears below.

```turbulance
item greeting = \"hello\"
print(greeting)
```

Cells share a store, so a later cell sees what an earlier one left:

```turbulance
item names = [\"ndombolo\", \"turbulance\"]
item n = len(names)
print(greeting, n)
```

Run them with:

    ndombolo run this-file.ndo

The output block is a view of the last run. Running a cell again replaces it,
so a cell run ten times leaves one block -- while the record beside this file
advances ten times. `ndombolo record this-file.ndo` shows it.
";

fn cmd_new(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "{} already exists; not overwriting it",
            path.display()
        ));
    }
    std::fs::write(path, STARTER).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    println!("{}", path.display());
    Ok(())
}

// -- edit ------------------------------------------------------------------

/// Serve the document on loopback and open it.
///
/// The file must already exist: `edit` is a second face on a document, and
/// creating one silently here would mean a typo in a filename produced a new
/// empty report rather than an error. `new` is the command that creates.
fn cmd_edit(path: &Path, port: u16, model: &str, host: &str) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "{} does not exist; `ndombolo new {}` writes one",
            path.display(),
            path.display()
        ));
    }
    // Read once before binding, so a document that cannot be parsed fails here
    // rather than as a 500 in the browser.
    load(path)?;

    let mut editor = editor::Editor::new(path, model, host);
    println!("{}", path.display());
    http::serve(port, |req| editor.handle(req))
        .map_err(|e| format!("cannot serve on 127.0.0.1:{port}: {e}"))
}
