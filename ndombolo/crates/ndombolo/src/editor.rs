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

use crate::http::{percent_encode, Request, Response};
use crate::ollama::Model;
use crate::record::Record;

/// Settings the page can change: the document's own context menu.
pub struct Settings {
    pub host: String,
    pub model: String,
}

pub struct Editor {
    /// The directory served. Every document handled is a file directly in it.
    root: PathBuf,
    /// `Some(name)` when one file was named rather than a directory: the index
    /// then lists just that document.
    single: Option<String>,
    settings: Settings,
}

impl Editor {
    /// Serve one document.
    pub fn file(path: &Path, model: &str, host: &str) -> Editor {
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let name = path.file_name().map(|s| s.to_string_lossy().into_owned());
        Editor {
            root,
            single: name,
            settings: Settings {
                host: host.to_string(),
                model: model.to_string(),
            },
        }
    }

    /// Serve every `.ndo` in a directory.
    pub fn dir(root: &Path, model: &str, host: &str) -> Editor {
        Editor {
            root: root.to_path_buf(),
            single: None,
            settings: Settings {
                host: host.to_string(),
                model: model.to_string(),
            },
        }
    }

    /// The documents this editor serves, in filename order.
    fn documents(&self) -> Vec<String> {
        if let Some(name) = &self.single {
            return vec![name.clone()];
        }
        let mut names: Vec<String> = std::fs::read_dir(&self.root)
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.ends_with(".ndo"))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    /// Resolve the `doc` parameter to a path inside [`Editor::root`].
    ///
    /// # The guard is the security boundary, not the loopback bind
    ///
    /// This is the only place a client-supplied string becomes a path. Loopback
    /// limits *who* may ask; it does not make an arbitrary path safe to read,
    /// and this process reads and writes whatever it is given. So the name must
    /// be a single ordinary component ending in `.ndo` -- no separators, no
    /// `..`, no root, no drive prefix. `../../.ssh/id_rsa` is rejected here and
    /// nowhere else.
    fn doc_path(&self, req: &Request) -> Result<PathBuf, String> {
        let name = match req.param("doc") {
            Some(n) if !n.is_empty() => n,
            // No `doc` given: the single-file case keeps working unqualified.
            _ => match &self.single {
                Some(n) => n.clone(),
                None => return Err("no document named".into()),
            },
        };
        if !is_safe_name(&name) {
            return Err(format!("{name:?} is not a document in this directory"));
        }
        Ok(self.root.join(name))
    }

    pub fn handle(&mut self, req: &Request) -> Response {
        // `/` with no document is the index; with one it is the reader.
        if (req.method.as_str(), req.path.as_str()) == ("GET", "/")
            && req.param("doc").is_none()
            && self.single.is_none()
        {
            return Response::html(&self.index());
        }

        let path = match self.doc_path(req) {
            Ok(p) => p,
            Err(e) => return Response::error(400, &e),
        };

        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/") => Response::html(&page(&path)),
            ("GET", "/api/doc") => self.get_doc(&path),
            ("GET", "/api/record") => self.get_record(&path),
            ("GET", "/api/settings") => Response::json(&json!({
                "host": self.settings.host,
                "model": self.settings.model,
                "file": path.display().to_string(),
            })),
            ("POST", "/api/settings") => self.set_settings(req),
            ("POST", "/api/prose") => self.set_prose(req, &path),
            ("POST", "/api/cell") => self.set_cell(req, &path),
            ("POST", "/api/run") => self.run(req, &path),
            ("POST", "/api/ask") => self.ask(req, &path),
            _ => Response::error(404, "no such route"),
        }
    }

    // -- reading -----------------------------------------------------------

    fn load(&self, path: &Path) -> Result<Document, String> {
        std::fs::read_to_string(path)
            .map(|t| Document::parse(&t))
            .map_err(|e| format!("cannot read {}: {e}", path.display()))
    }

    /// Write via a temporary file and rename, as the CLI does: the document
    /// holds prose that exists nowhere else.
    fn save(&self, path: &Path, doc: &Document) -> Result<(), String> {
        let tmp = path.with_extension("ndo.tmp");
        std::fs::write(&tmp, doc.render()).map_err(|e| format!("cannot write: {e}"))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("cannot replace {}: {e}", path.display())
        })
    }

    /// The document as blocks, each tagged with what it is.
    ///
    /// `editable` and `model_writable` are sent per block rather than inferred
    /// in the page: the rule about who may write what belongs to the runtime,
    /// and a copy of it in JavaScript would be a second place for it to drift.
    fn get_doc(&self, path: &Path) -> Response {
        let doc = match self.load(path) {
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

        let record = Record::open(path).map(|r| r.count()).unwrap_or(0);
        Response::json(&json!({ "blocks": blocks, "record": record }))
    }

    fn get_record(&self, path: &Path) -> Response {
        match Record::open(path).and_then(|r| {
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
    fn set_prose(&mut self, req: &Request, path: &Path) -> Response {
        self.set_block(req, path, true)
    }

    /// Replace one cell's source. Not model-writable; see [`set_prose`].
    fn set_cell(&mut self, req: &Request, path: &Path) -> Response {
        self.set_block(req, path, false)
    }

    fn set_block(&mut self, req: &Request, path: &Path, prose: bool) -> Response {
        let body = match parse_body(req) {
            Ok(b) => b,
            Err(e) => return Response::error(400, &e),
        };
        let index = match body.get("index").and_then(Json::as_u64) {
            Some(i) => i as usize,
            None => return Response::error(400, "no block index"),
        };
        let text = body.get("text").and_then(Json::as_str).unwrap_or("");

        let mut doc = match self.load(path) {
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

        match self.save(path, &doc) {
            Ok(()) => Response::json(&json!({ "written": index })),
            Err(e) => Response::error(500, &e),
        }
    }

    /// Run the document's cells, exactly as `ndombolo run` does.
    ///
    /// The same replay-from-the-top discipline: a fresh session, every cell up
    /// to the requested one, every run depositing. The editor is a second face
    /// on one runtime, not a second runtime.
    fn run(&mut self, req: &Request, path: &Path) -> Response {
        let body = parse_body(req).unwrap_or(json!({}));
        let only = body.get("cell").and_then(Json::as_u64).map(|n| n as usize);

        let mut doc = match self.load(path) {
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

        let mut rec = match Record::open(path) {
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
        if let Err(e) = self.save(path, &doc) {
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
    fn ask(&mut self, req: &Request, path: &Path) -> Response {
        let body = match parse_body(req) {
            Ok(b) => b,
            Err(e) => return Response::error(400, &e),
        };
        let question = body.get("question").and_then(Json::as_str).unwrap_or("");
        if question.trim().is_empty() {
            return Response::error(400, "no question");
        }

        let doc = match self.load(path) {
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

impl Editor {
    /// The index: every document served, with the count of its record.
    ///
    /// Navigation only. Nothing here writes, and a document is opened by the
    /// same reader whichever way you arrive at it.
    fn index(&self) -> String {
        let mut rows = String::new();
        for name in self.documents() {
            let count = Record::open(&self.root.join(&name))
                .map(|r| r.count())
                .unwrap_or(0);
            rows.push_str(&format!(
                "<li><a href=\"/?doc={}\">{}</a><span>record {}</span></li>
",
                percent_encode(&name),
                escape(&name),
                count
            ));
        }
        if rows.is_empty() {
            rows.push_str("<li><em>no .ndo documents in this directory</em></li>");
        }
        INDEX
            .replace("{{ROOT}}", &escape(&self.root.display().to_string()))
            .replace("{{ROWS}}", &rows)
    }
}

/// Whether `name` is a plain document name in the served directory.
///
/// One ordinary component, ending in `.ndo`, and not the record or temp file
/// beside one. Anything with a separator, a `..`, a root or a drive prefix is
/// refused: see [`Editor::doc_path`] for why this is the boundary.
fn is_safe_name(name: &str) -> bool {
    use std::path::Component;
    if !name.ends_with(".ndo") || name.contains(' ') {
        return false;
    }
    let path = Path::new(name);
    let mut parts = path.components();
    let one = matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none();
    one && path.file_name().map(|f| f == name).unwrap_or(false)
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

/// The index, inline rather than a second file: it is a list of links.
///
/// The palette is the one `page.html` uses, repeated rather than shared because
/// a stylesheet route would be a third thing to serve for eleven declarations.
const INDEX: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ndombolo</title>
<style>
  :root {
    --bg: #16171a; --fg: #dcdcd6; --muted: #82827c; --rule: #2c2e33;
    --cell-bg: #1d1f23; --accent: #8fb0e0;
  }
  html, body { margin: 0; background: var(--bg); color: var(--fg); }
  body {
    font: 16px/1.6 ui-serif, Georgia, "Times New Roman", serif;
    max-width: 44rem; margin: 0 auto; padding: 4rem 1.5rem;
  }
  h1 { font-size: 1.3rem; font-weight: 600; margin: 0 0 0.2rem; }
  p.root {
    margin: 0 0 2rem; color: var(--muted); font-size: 0.85rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  ul { list-style: none; margin: 0; padding: 0; }
  li {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem; padding: 0.7rem 0.2rem; border-bottom: 1px solid var(--rule);
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  li span { color: var(--muted); font-size: 0.8rem; white-space: nowrap; }
  footer { margin-top: 2.5rem; color: var(--muted); font-size: 0.8rem; }
</style>
</head>
<body>
  <h1>ndombolo</h1>
  <p class="root">{{ROOT}}</p>
  <ul>
{{ROWS}}  </ul>
  <footer>Open one to read it. Cells run in place with ctrl-enter.</footer>
</body>
</html>
"#;

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
    fn a_document_name_cannot_escape_the_directory() {
        // The whole security boundary of serving a directory is this function.
        assert!(is_safe_name("05-the-evidential-layer.ndo"));
        assert!(!is_safe_name("../secrets.ndo"));
        assert!(!is_safe_name("../../.ssh/id_rsa"));
        assert!(!is_safe_name("/etc/passwd"));
        assert!(!is_safe_name("sub/dir.ndo"));
        assert!(!is_safe_name(r"sub\dir.ndo"));
        assert!(!is_safe_name("notes.txt"));
        assert!(!is_safe_name("notes.ndo.record"));
        assert!(!is_safe_name(""));
        assert!(!is_safe_name(".."));
    }

    #[test]
    fn an_index_lists_documents_in_filename_order() {
        let dir = std::env::temp_dir().join("ndombolo-index-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["02-second.ndo", "01-first.ndo", "notes.txt"] {
            std::fs::write(dir.join(name), "# x
").unwrap();
        }
        let ed = Editor::dir(&dir, "m", "h");
        assert_eq!(ed.documents(), vec!["01-first.ndo", "02-second.ndo"]);

        let html = ed.index();
        let first = html.find("01-first.ndo").unwrap();
        let second = html.find("02-second.ndo").unwrap();
        assert!(first < second, "filename order is the reading order");
        // A non-document in the directory is not listed.
        assert!(!html.contains("notes.txt"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_single_file_editor_serves_only_that_file() {
        let ed = Editor::file(Path::new("/tmp/notes.ndo"), "m", "h");
        assert_eq!(ed.documents(), vec!["notes.ndo"]);
    }

    #[test]
    fn a_name_with_markup_in_it_is_escaped() {
        let html = page(Path::new("/tmp/<script>.ndo"));
        assert!(!html.contains("<script>.ndo"));
        assert!(html.contains("&lt;script&gt;.ndo"));
    }
}
