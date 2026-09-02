//! The editor: routes over one document, and the page that draws it.
//!
//! # The document is edited block by block, never as a string
//!
//! The tempting design sends the whole document to the browser as markdown,
//! lets a WYSIWYG editor rewrite it, and posts the result back. That destroys
//! the file. [`Document`] preserves the backtick run each fence used, its
//! indent, the newline convention of the file, and whether the last line ended
//! in one; a round trip through HTML and back regenerates all four from
//! whatever the serialiser prefers. So the wire format here is *blocks by
//! index*: the browser says "block 4 now reads this", and prose is the only
//! kind it may say that about. Everything else in the file is untouched by
//! construction rather than by care.
//!
//! # What each route is allowed to write
//!
//! | route | writes |
//! |---|---|
//! | `POST /api/prose` | one prose block, from the user or the model |
//! | `POST /api/run`   | output blocks, and one deposit per cell run |
//! | `POST /api/cell`  | one cell's source, from the user only |
//! | `POST /api/ask`   | nothing -- it reads the record and answers |
//!
//! `/api/ask` returning prose that the page then posts to `/api/prose` is the
//! no-backflow wall as a route table: the model's words reach the document
//! only by the same door a user's words do, and there is no door at all onto a
//! cell or an output block.

use std::path::{Path, PathBuf};

use ndombolo_core::doc::{format_result, Block, Document};
use ndombolo_core::session::Session;
use serde_json::{json, Value as Json};

use crate::http::{Request, Response};
use crate::ollama::Model;
use crate::record::Record;

/// Settings the page can change: the document's own context menu.
pub struct Settings {
    pub host: String,
    pub model: String,
}

pub struct Editor {
    path: PathBuf,
    settings: Settings,
}

impl Editor {
    pub fn new(path: &Path, model: &str, host: &str) -> Editor {
        Editor {
            path: path.to_path_buf(),
            settings: Settings {
                host: host.to_string(),
                model: model.to_string(),
            },
        }
    }

    pub fn handle(&mut self, req: &Request) -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") => Response::html(&page(&self.path)),
            ("GET", "/api/doc") => self.get_doc(),
            ("GET", "/api/record") => self.get_record(),
            ("GET", "/api/settings") => Response::json(&json!({
                "host": self.settings.host,
                "model": self.settings.model,
                "file": self.path.display().to_string(),
            })),
            ("POST", "/api/settings") => self.set_settings(req),
            ("POST", "/api/prose") => self.set_prose(req),
            ("POST", "/api/cell") => self.set_cell(req),
            ("POST", "/api/run") => self.run(req),
            ("POST", "/api/ask") => self.ask(req),
            _ => Response::error(404, "no such route"),
        }
    }

    // -- reading -----------------------------------------------------------

    fn load(&self) -> Result<Document, String> {
        std::fs::read_to_string(&self.path)
            .map(|t| Document::parse(&t))
            .map_err(|e| format!("cannot read {}: {e}", self.path.display()))
    }

    /// Write via a temporary file and rename, as the CLI does: the document
    /// holds prose that exists nowhere else.
    fn save(&self, doc: &Document) -> Result<(), String> {
        let tmp = self.path.with_extension("ndo.tmp");
        std::fs::write(&tmp, doc.render()).map_err(|e| format!("cannot write: {e}"))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("cannot replace {}: {e}", self.path.display())
        })
    }

    /// The document as blocks, each tagged with what it is.
    ///
    /// `editable` and `model_writable` are sent per block rather than inferred
    /// in the page: the rule about who may write what belongs to the runtime,
    /// and a copy of it in JavaScript would be a second place for it to drift.
    fn get_doc(&self) -> Response {
        let doc = match self.load() {
            Ok(d) => d,
            Err(e) => return Response::error(500, &e),
        };
        let mut cell_number = 0usize;
        let blocks: Vec<Json> = doc
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| match b {
                Block::Prose { text } => json!({
                    "index": i, "kind": "prose", "text": text,
                    "editable": true, "model_writable": true,
                }),
                Block::Fence { lang, body, .. } => {
                    let kind = if b.is_cell() {
                        let n = cell_number;
                        cell_number += 1;
                        return json!({
                            "index": i, "kind": "cell", "cell": n, "text": body,
                            "line": doc.line_of(i) + 1,
                            "editable": true, "model_writable": false,
                        });
                    } else if b.is_output() {
                        "output"
                    } else {
                        "fence"
                    };
                    json!({
                        "index": i, "kind": kind, "lang": lang, "text": body,
                        "editable": kind == "fence", "model_writable": false,
                    })
                }
            })
            .collect();

        let record = Record::open(&self.path).map(|r| r.count()).unwrap_or(0);
        Response::json(&json!({ "blocks": blocks, "record": record }))
    }

    fn get_record(&self) -> Response {
        match Record::open(&self.path).and_then(|r| {
            let count = r.count();
            r.entries().map(|e| (count, e))
        }) {
            Ok((count, entries)) => Response::json(&json!({
                "record": count, "deposits": entries,
            })),
            Err(e) => Response::error(500, &format!("cannot read the record: {e}")),
        }
    }

    // -- writing -----------------------------------------------------------

    fn set_settings(&mut self, req: &Request) -> Response {
        let body = match parse_body(req) {
            Ok(b) => b,
            Err(e) => return Response::error(400, &e),
        };
        if let Some(m) = body.get("model").and_then(Json::as_str) {
            self.settings.model = m.to_string();
        }
        if let Some(h) = body.get("host").and_then(Json::as_str) {
            self.settings.host = h.to_string();
        }
        Response::json(&json!({
            "host": self.settings.host, "model": self.settings.model,
        }))
    }

    /// Replace one prose block.
    ///
    /// Refuses any other kind. The page already hides the control, but the
    /// check is here because the page is not what enforces it: a request that
    /// named a cell index would otherwise let the model author a cell.
    fn set_prose(&mut self, req: &Request) -> Response {
        self.set_block(req, true)
    }

    /// Replace one cell's source. Not model-writable; see [`set_prose`].
    fn set_cell(&mut self, req: &Request) -> Response {
        self.set_block(req, false)
    }

    fn set_block(&mut self, req: &Request, prose: bool) -> Response {
        let body = match parse_body(req) {
            Ok(b) => b,
            Err(e) => return Response::error(400, &e),
        };
        let index = match body.get("index").and_then(Json::as_u64) {
            Some(i) => i as usize,
            None => return Response::error(400, "no block index"),
        };
        let text = body.get("text").and_then(Json::as_str).unwrap_or("");

        let mut doc = match self.load() {
            Ok(d) => d,
            Err(e) => return Response::error(500, &e),
        };
        // Asked before the mutable borrow: the check reads the document, and
        // it cannot read it while `get_mut` holds it.
        let is_cell = block_is_cell(index, &doc);
        let followed_by_fence = matches!(doc.blocks.get(index + 1), Some(Block::Fence { .. }));
        let block = match doc.blocks.get_mut(index) {
            Some(b) => b,
            None => return Response::error(400, "no such block"),
        };

        if prose {
            match block {
                Block::Prose { text: t } => *t = separate(text, followed_by_fence),
                _ => {
                    return Response::error(
                        400,
                        "that block is not prose; only prose may be written this way",
                    )
                }
            }
        } else {
            match block {
                Block::Fence { body: b, .. } if is_cell => *b = text.to_string(),
                _ => return Response::error(400, "that block is not a cell"),
            }
        }

        match self.save(&doc) {
            Ok(()) => Response::json(&json!({ "written": index })),
            Err(e) => Response::error(500, &e),
        }
    }

    /// Run the document's cells, exactly as `ndombolo run` does.
    ///
    /// The same replay-from-the-top discipline: a fresh session, every cell up
    /// to the requested one, every run depositing. The editor is a second face
    /// on one runtime, not a second runtime.
    fn run(&mut self, req: &Request) -> Response {
        let body = parse_body(req).unwrap_or(json!({}));
        let only = body.get("cell").and_then(Json::as_u64).map(|n| n as usize);

        let mut doc = match self.load() {
            Ok(d) => d,
            Err(e) => return Response::error(500, &e),
        };
        let cells = doc.cell_indices();
        if cells.is_empty() {
            return Response::json(&json!({ "ran": 0, "record": 0, "cells": [] }));
        }
        if let Some(n) = only {
            if n >= cells.len() {
                return Response::error(400, &format!("the document has {} cells", cells.len()));
            }
        }

        let mut rec = match Record::open(&self.path) {
            Ok(r) => r,
            Err(e) => return Response::error(500, &format!("cannot open the record: {e}")),
        };
        let mut session = Session::new();
        let last = only.unwrap_or(cells.len() - 1);

        let mut splices: Vec<(usize, String)> = Vec::new();
        let mut reports: Vec<Json> = Vec::new();
        let mut stopped_at: Option<usize> = None;

        for (n, &block) in cells.iter().enumerate().take(last + 1) {
            let text = doc.cell_text(block).unwrap_or_default();
            let result = session.run_cell(n, doc.line_of(block) + 1, &text);

            let touched: Vec<String> =
                result.store_delta.iter().map(|(k, _)| k.clone()).collect();
            if let Err(e) = rec.deposit(n, result.trace.len(), touched, result.ok) {
                return Response::error(500, &format!("cannot deposit: {e}"));
            }

            reports.push(json!({
                "cell": n,
                "completed": result.ok,
                "events": result.trace.len(),
                "error": result.error.as_ref().map(|e| e.message.clone()),
            }));

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

        // Later blocks first: a splice shifts every index after it.
        splices.sort_by_key(|(b, _)| std::cmp::Reverse(*b));
        for (block, text) in &splices {
            doc.splice(*block, text);
        }
        if let Err(e) = self.save(&doc) {
            return Response::error(500, &e);
        }

        Response::json(&json!({
            "ran": stopped_at.map(|n| n + 1).unwrap_or(last + 1),
            "record": rec.count(),
            "stopped_at": stopped_at,
            "cells": reports,
        }))
    }

    /// Ask the model a question about the run.
    ///
    /// Writes nothing. The answer comes back as prose for the page to show;
    /// putting it in the document is a separate, explicit `POST /api/prose`,
    /// so nothing the model says enters the file without the user placing it.
    fn ask(&mut self, req: &Request) -> Response {
        let body = match parse_body(req) {
            Ok(b) => b,
            Err(e) => return Response::error(400, &e),
        };
        let question = body.get("question").and_then(Json::as_str).unwrap_or("");
        if question.trim().is_empty() {
            return Response::error(400, "no question");
        }

        let doc = match self.load() {
            Ok(d) => d,
            Err(e) => return Response::error(500, &e),
        };

        // Replay to collect the events. Determination by search, not lookup
        // (B3): there is no stored trace from the last run to consult, and
        // keeping one so that repeated questions could skip this would be the
        // cache the blueprint forbids.
        let mut session = Session::new();
        let mut events: Vec<Json> = Vec::new();
        for (n, &block) in doc.cell_indices().iter().enumerate() {
            let text = doc.cell_text(block).unwrap_or_default();
            let result = session.run_cell(n, doc.line_of(block) + 1, &text);
            for ev in result.trace {
                events.push(json!({ "cell": n, "event": ev }));
            }
            if !result.ok {
                break;
            }
        }

        // Reading the record is a probe, not a determination -- rate-limiting
        // probes is allowed, caching determinations is not -- so this replay
        // deposits nothing.
        let model = Model::new(&self.settings.host, &self.settings.model, events);
        match model.ask(question) {
            Ok(a) => Response::json(&json!({
                "text": a.text, "searches": a.searches,
            })),
            Err(e) => Response::error(500, &e),
        }
    }
}

/// Prose as it must sit in the file, given what follows it.
///
/// `render` writes exactly one newline between blocks, so the blank line that
/// separates a paragraph from the fence below it lives inside the prose block
/// itself. The browser strips trailing whitespace from a contenteditable, so
/// prose arriving here has lost it, and writing that back would glue the
/// paragraph onto the fence. Restoring it here rather than in `render` keeps
/// `render` from inventing blank lines the document never had.
fn separate(text: &str, followed_by_fence: bool) -> String {
    let trimmed = text.trim_end_matches(['\n', ' ', '\t']);
    if followed_by_fence && !trimmed.is_empty() {
        format!("{trimmed}\n")
    } else {
        trimmed.to_string()
    }
}

/// Whether block `index` is a cell, checked against a fresh view of the doc.
fn block_is_cell(index: usize, doc: &Document) -> bool {
    doc.blocks.get(index).map(Block::is_cell).unwrap_or(false)
}

fn parse_body(req: &Request) -> Result<Json, String> {
    if req.body.is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_slice(&req.body).map_err(|e| format!("body is not json: {e}"))
}

// -- the page ---------------------------------------------------------------

/// The editor, as one self-contained page.
///
/// No CDN and no bundle: the highlighter is a tokenizer written into the page,
/// over the same keyword set the lexer uses. An editor for a local file should
/// not stop colouring its cells because the network is down, and shipping a
/// syntax highlighter is not worth an npm step in a project that has none.
fn page(path: &Path) -> String {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "document".into());
    PAGE.replace("{{NAME}}", &escape(&name))
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

const PAGE: &str = include_str!("page.html");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_page_names_the_document() {
        let html = page(Path::new("/tmp/notes.ndo"));
        assert!(html.contains("notes.ndo"));
        assert!(!html.contains("{{NAME}}"));
    }

    #[test]
    fn prose_above_a_fence_keeps_its_blank_line() {
        // The browser sends "text" with the trailing newline stripped; written
        // back as-is it would abut the fence below.
        assert_eq!(separate("# Title\n\nA line.", true), "# Title\n\nA line.\n");
        // Already separated: not doubled.
        assert_eq!(separate("A line.\n", true), "A line.\n");
    }

    #[test]
    fn prose_not_above_a_fence_gains_nothing() {
        assert_eq!(separate("A line.", false), "A line.");
        // An empty block stays empty rather than becoming a stray newline.
        assert_eq!(separate("", true), "");
        assert_eq!(separate("\n\n", true), "");
    }

    #[test]
    fn a_name_with_markup_in_it_is_escaped() {
        let html = page(Path::new("/tmp/<script>.ndo"));
        assert!(!html.contains("<script>.ndo"));
        assert!(html.contains("&lt;script&gt;.ndo"));
    }
}
