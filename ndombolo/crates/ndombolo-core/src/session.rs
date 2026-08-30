//! Cells, and running a sequence of them against one compiler.
//!
//! A `.ndo` document is prose with fenced Turbulance in it. Splitting that
//! document into cells and running them is the same operation `prototype/run.py`
//! performs on a script split by `// ---`, so it lives in the crate rather than
//! in the editor or the test: one definition, checked against the oracle, used
//! by both.
//!
//! # A session, not N runs
//!
//! [`Session`] holds a single [`Compiler`], so the store, the trace counter, the
//! output counter and the proposition table all accrete across cells.
//! Re-creating the compiler per cell is the obvious reading of "run this cell",
//! and it would test a runtime nobody uses -- it would also reset the record,
//! which the blueprint's B2 forbids.

use serde_json::Value as Json;

use crate::eval::{env_flatten, Compiler, Flow};
use crate::parser::parse;
use crate::value::render;

/// A line whose leading whitespace is followed by this opens the next cell.
pub const CELL_SEP: &str = "// ---";

/// Split on separator lines, keeping each cell's first line number.
///
/// Line numbers stay **script-absolute**, so a trace site names a place in the
/// document rather than an offset into whichever cell happened to produce it.
/// That is what lets an event be pointed at from the editor at all.
pub fn split_cells(source: &str) -> Vec<(usize, String)> {
    let normalised = source.replace("\r\n", "\n");
    let mut cells: Vec<(usize, String)> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut start = 1usize;

    for (i, line) in normalised.split('\n').enumerate() {
        let lineno = i + 1;
        if line.trim_start().starts_with(CELL_SEP) {
            cells.push((start, current.join("\n")));
            current.clear();
            start = lineno + 1;
            continue;
        }
        current.push(line);
    }
    cells.push((start, current.join("\n")));

    let kept: Vec<(usize, String)> = cells
        .into_iter()
        .filter(|(_, c)| !c.trim().is_empty())
        .collect();
    // An all-blank source is one empty cell, not zero: the caller still gets
    // something to report against.
    if kept.is_empty() {
        vec![(1, source.to_string())]
    } else {
        kept
    }
}

/// Where a cell failed, and what it said.
#[derive(Debug, Clone, PartialEq)]
pub struct CellError {
    /// `parse` or `run`.
    pub phase: &'static str,
    /// The prefixed form, `line N: ...`, as the oracle records it.
    pub message: String,
    pub line: usize,
}

/// How a name in the store moved during a cell.
#[derive(Debug, Clone, PartialEq)]
pub enum StoreChange {
    Added {
        value: Json,
        tag: &'static str,
    },
    Updated {
        from: Json,
        value: Json,
        tag: &'static str,
    },
}

/// One cell's result: what it wrote, what it traced, what it printed.
#[derive(Debug, Clone)]
pub struct CellResult {
    pub index: usize,
    pub first_line: usize,
    pub ok: bool,
    pub error: Option<CellError>,
    /// Names added or updated by this cell, in store order.
    pub store_delta: Vec<(String, StoreChange)>,
    /// The events this cell appended, not the whole trace.
    pub trace: Vec<Json>,
    pub output: Vec<String>,
}

/// A live document: one compiler, cells run against it in order.
pub struct Session {
    pub compiler: Compiler,
    prev_store: Vec<(String, Json)>,
    prev_trace: usize,
    prev_output: usize,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        let compiler = Compiler::new();
        // Builtins are the initial environment, not something a cell wrote, so
        // seeding the previous store excludes them from the first delta.
        let prev_store = env_flatten(&compiler.env)
            .iter()
            .map(|(k, v)| (k.clone(), render(v)))
            .collect();
        Session {
            compiler,
            prev_store,
            prev_trace: 0,
            prev_output: 0,
        }
    }

    /// Run one cell. A failure is returned, not raised: the caller decides
    /// whether to stop, and the record keeps whatever events the cell emitted
    /// before it failed.
    pub fn run_cell(&mut self, index: usize, first_line: usize, text: &str) -> CellResult {
        let body = match parse(text, first_line) {
            Ok(body) => body,
            Err(e) => {
                // A parse failure moves neither store nor trace, so the cell is
                // reported bare -- and, in the oracle, without even the
                // `store_delta`/`trace`/`output` keys.
                return CellResult {
                    index,
                    first_line,
                    ok: false,
                    error: Some(CellError {
                        phase: "parse",
                        message: e.to_string(),
                        line: e.line,
                    }),
                    store_delta: Vec::new(),
                    trace: Vec::new(),
                    output: Vec::new(),
                };
            }
        };

        let error = match self.compiler.exec_block(&body) {
            Ok(()) => None,
            Err(Flow::Err(e)) => Some(CellError {
                phase: "run",
                message: e.to_string(),
                line: e.line,
            }),
            // A top-level `return` escapes the Python as an uncaught
            // ReturnSignal rather than a RuntimeErr, so `dump_eval.py` does not
            // catch it and the corpus contains no such script. Recording it as
            // a distinct phase keeps it from being silently folded into a
            // runtime error the oracle would never produce.
            Err(Flow::Return(_)) => Some(CellError {
                phase: "return",
                message: "return outside a funxn".into(),
                line: first_line,
            }),
        };

        let store = env_flatten(&self.compiler.env);
        let mut delta = Vec::new();
        let mut next_store: Vec<(String, Json)> = Vec::with_capacity(store.len());
        for (name, value) in store.iter() {
            let rendered = render(value);
            match self.prev_store.iter().find(|(k, _)| k == name) {
                None => delta.push((
                    name.clone(),
                    StoreChange::Added {
                        value: rendered.clone(),
                        tag: value.tag(),
                    },
                )),
                Some((_, before)) if *before != rendered => delta.push((
                    name.clone(),
                    StoreChange::Updated {
                        from: before.clone(),
                        value: rendered.clone(),
                        tag: value.tag(),
                    },
                )),
                Some(_) => {}
            }
            next_store.push((name.clone(), rendered));
        }
        self.prev_store = next_store;

        let trace = self.compiler.trace[self.prev_trace..].to_vec();
        self.prev_trace = self.compiler.trace.len();
        let output = self.compiler.output[self.prev_output..].to_vec();
        self.prev_output = self.compiler.output.len();

        CellResult {
            index,
            first_line,
            ok: error.is_none(),
            error,
            store_delta: delta,
            trace,
            output,
        }
    }

    /// Run a whole source, stopping at the first failing cell.
    ///
    /// Stopping is what `run.py` does, and it is not merely convenient: a cell
    /// that failed left the store in a state its successors were not written
    /// against, so continuing would record a run nobody asked for.
    pub fn run_source(&mut self, source: &str) -> Vec<CellResult> {
        let mut out = Vec::new();
        for (k, (first_line, text)) in split_cells(source).into_iter().enumerate() {
            let r = self.run_cell(k, first_line, &text);
            let failed = !r.ok;
            out.push(r);
            if failed {
                break;
            }
        }
        out
    }

    /// The store as the oracle's `final.store` records it.
    pub fn store(&self) -> Vec<(String, Json)> {
        self.prev_store.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separator_splits_and_keeps_absolute_lines() {
        let cells = split_cells("item a = 1\n// ---\nitem b = 2\n");
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].0, 1);
        // The cell after the separator starts at line 3, not line 1.
        assert_eq!(cells[1].0, 3);
    }

    #[test]
    fn blank_source_is_one_cell() {
        assert_eq!(split_cells("").len(), 1);
        assert_eq!(split_cells("\n\n").len(), 1);
    }

    #[test]
    fn store_accretes_across_cells() {
        let mut s = Session::new();
        let cells = s.run_source("item a = 1\n// ---\nitem b = a + 1\n");
        assert!(cells.iter().all(|c| c.ok), "{:?}", cells);
        // The second cell sees the first cell's binding, and reports only its
        // own addition.
        assert_eq!(cells[1].store_delta.len(), 1);
        assert_eq!(cells[1].store_delta[0].0, "b");
    }

    #[test]
    fn builtins_are_not_in_the_first_delta() {
        let mut s = Session::new();
        let cells = s.run_source("item a = 1\n");
        assert_eq!(cells[0].store_delta.len(), 1);
    }

    #[test]
    fn a_failing_cell_stops_the_rest() {
        let mut s = Session::new();
        let cells = s.run_source("item a = 1\n// ---\nitem b = nope\n// ---\nitem c = 3\n");
        assert_eq!(cells.len(), 2, "the third cell must not run");
        assert!(!cells[1].ok);
    }
}
