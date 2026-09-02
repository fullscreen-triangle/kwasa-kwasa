//! The `.ndo` document: prose, cells, and the outputs written beneath them.
//!
//! A document is a report. It holds prose the user wrote, prose the user wrote
//! with the model, fenced `turbulance` cells, and each cell's output as plain
//! text. All four are the same artefact -- there is no source view and no
//! rendered view, because there is only one file.
//!
//! # The block grammar
//!
//! Everything is prose except a fence. A fence opens on a line whose first
//! non-space content is three or more backticks, and closes on the next line
//! whose backtick run is at least as long. The info string -- `turbulance`,
//! `output`, or anything else -- decides what the block *is*:
//!
//! * `turbulance` is a cell: it runs.
//! * `output` is a cell's result: it is written by the runtime and is never
//!   read back as input.
//! * anything else, including a bare fence, is prose that happens to look like
//!   code. It is carried through untouched.
//!
//! Fence length is tracked rather than fixed at three so a cell may contain a
//! fence -- a document about `.ndo` documents is the obvious case, and this
//! file's own doc comment is the second.
//!
//! # Why output is a block and not a comment
//!
//! Output has to be *replaceable*. A cell run twice must leave one output
//! block, not two, or the document grows a transcript every time the user
//! presses run and stops being a report. Delimiting it lets [`splice`] find the
//! previous one and replace it exactly.
//!
//! That replacement is **not** a rewind. The visible block is a view of the
//! last run; the record is the history of every run, and it only ever
//! increases (paper, B2, and B4: a propagation deposits *"including those whose
//! output is discarded"*). Replacing the view while the record advances is the
//! two-sided thing the blueprint asks for, and conflating them -- either by
//! appending forever, or by resetting the record to match the document -- is
//! the failure mode this split exists to prevent.
//!
//! # What the model may write
//!
//! Prose. Never a `turbulance` block and never an `output` block, because
//! residue never flows from the model into a run (paper, Def. 4.x, the runtime
//! pair). [`Block::is_model_writable`] is that rule in one place, so the
//! constraint is checkable rather than remembered.

use std::fmt::Write as _;

/// The info string marking a block the runtime runs.
pub const CELL_LANG: &str = "turbulance";

/// The info string marking a block the runtime writes.
pub const OUTPUT_LANG: &str = "output";

/// One block of a document, in source order.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Anything that is not a fence: markdown, headings, blank lines.
    Prose { text: String },
    /// A fenced block. `lang` is the info string, empty for a bare fence.
    Fence {
        /// The backtick run that opened it, kept so the block round-trips.
        ticks: usize,
        /// Leading spaces on the opening line, kept for the same reason.
        indent: String,
        lang: String,
        /// The info string's trailing part, if any: ```` ```turbulance foo ````.
        info_rest: String,
        body: String,
        /// False when the document ended before a closing fence. Kept so an
        /// unterminated fence round-trips as it was written rather than
        /// acquiring a delimiter the user did not type.
        closed: bool,
    },
}

impl Block {
    /// Whether this is a `turbulance` cell.
    pub fn is_cell(&self) -> bool {
        matches!(self, Block::Fence { lang, .. } if lang == CELL_LANG)
    }

    /// Whether this is a runtime-written output block.
    pub fn is_output(&self) -> bool {
        matches!(self, Block::Fence { lang, .. } if lang == OUTPUT_LANG)
    }

    /// Whether the observing model may author this block.
    ///
    /// Prose only. A model that could write a cell would be feeding residue
    /// back into a run, and one that could write an output would be reporting
    /// a determination it did not compute.
    pub fn is_model_writable(&self) -> bool {
        matches!(self, Block::Prose { .. })
    }
}

/// A parsed document: blocks in source order, plus the newline it used.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub blocks: Vec<Block>,
    /// `\r\n` if the file used it anywhere, else `\n`. Preserved so writing a
    /// document back does not rewrite every line on a Windows checkout.
    pub newline: &'static str,
    /// Whether the source ended with a newline.
    pub trailing_newline: bool,
}

/// Where a fence opens: the tick run and the info string, if this line is one.
fn fence_open(line: &str) -> Option<(String, usize, String)> {
    let indent: String = line.chars().take_while(|c| *c == ' ').collect();
    let rest = &line[indent.len()..];
    let ticks = rest.chars().take_while(|c| *c == '`').count();
    if ticks < 3 {
        return None;
    }
    let info = rest[ticks..].trim().to_string();
    // A closing fence has no info string, so an info string containing a
    // backtick would be ambiguous. CommonMark forbids it; so do we.
    if info.contains('`') {
        return None;
    }
    Some((indent, ticks, info))
}

/// Whether this line closes a fence opened with `ticks` backticks.
fn fence_close(line: &str, ticks: usize) -> bool {
    let rest = line.trim_start_matches(' ');
    let run = rest.chars().take_while(|c| *c == '`').count();
    run >= ticks && rest[run..].trim().is_empty()
}

impl Document {
    /// Parse a `.ndo` document. Never fails: anything unrecognised is prose.
    pub fn parse(src: &str) -> Document {
        let newline = if src.contains("\r\n") { "\r\n" } else { "\n" };
        let trailing_newline = src.ends_with('\n');

        // Splitting on '\n' and trimming '\r' handles a mixed-ending file
        // without a second code path, and `newline` above decides what is
        // written back.
        let lines: Vec<&str> = src
            .split('\n')
            .map(|l| l.strip_suffix('\r').unwrap_or(l))
            .collect();
        // A trailing newline yields a final empty element that is not a line.
        let lines = if trailing_newline && !lines.is_empty() {
            &lines[..lines.len() - 1]
        } else {
            &lines[..]
        };

        let mut blocks: Vec<Block> = Vec::new();
        let mut prose: Vec<&str> = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            let Some((indent, ticks, info)) = fence_open(lines[i]) else {
                prose.push(lines[i]);
                i += 1;
                continue;
            };

            if !prose.is_empty() {
                blocks.push(Block::Prose {
                    text: prose.join("\n"),
                });
                prose.clear();
            }

            let mut parts = info.splitn(2, char::is_whitespace);
            let lang = parts.next().unwrap_or("").to_string();
            let info_rest = parts.next().unwrap_or("").trim().to_string();

            let mut body: Vec<&str> = Vec::new();
            let mut j = i + 1;
            let mut closed = false;
            while j < lines.len() {
                if fence_close(lines[j], ticks) {
                    closed = true;
                    break;
                }
                body.push(lines[j]);
                j += 1;
            }

            blocks.push(Block::Fence {
                ticks,
                indent,
                lang,
                info_rest,
                body: body.join("\n"),
                closed,
            });
            i = if closed { j + 1 } else { j };
        }

        if !prose.is_empty() {
            blocks.push(Block::Prose {
                text: prose.join("\n"),
            });
        }

        Document {
            blocks,
            newline,
            trailing_newline,
        }
    }

    /// Render back to text.
    ///
    /// `parse` then `render` is the identity on every document this can parse,
    /// which is what makes running a cell a local edit: the prose the user
    /// wrote comes back byte for byte, including the fences it did not
    /// understand.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut first = true;
        for b in &self.blocks {
            if !first {
                out.push('\n');
            }
            first = false;
            match b {
                Block::Prose { text } => out.push_str(text),
                Block::Fence {
                    ticks,
                    indent,
                    lang,
                    info_rest,
                    body,
                    closed,
                } => {
                    let bars = "`".repeat(*ticks);
                    out.push_str(indent);
                    out.push_str(&bars);
                    out.push_str(lang);
                    if !info_rest.is_empty() {
                        out.push(' ');
                        out.push_str(info_rest);
                    }
                    if !body.is_empty() {
                        out.push('\n');
                        out.push_str(body);
                    }
                    if *closed {
                        out.push('\n');
                        out.push_str(indent);
                        out.push_str(&bars);
                    }
                }
            }
        }
        if self.trailing_newline {
            out.push('\n');
        }
        if self.newline == "\r\n" {
            out.replace('\n', "\r\n")
        } else {
            out
        }
    }

    /// The indices of every `turbulance` block, in document order.
    ///
    /// This is the cell numbering the whole system uses: a cell's index is its
    /// position among cells, not among blocks, so inserting prose above a cell
    /// does not renumber it.
    pub fn cell_indices(&self) -> Vec<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_cell())
            .map(|(i, _)| i)
            .collect()
    }

    /// The source of every cell, in document order.
    pub fn cells(&self) -> Vec<&str> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Fence { lang, body, .. } if lang == CELL_LANG => Some(body.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The source of the cell at `block_index`, if that block is a cell.
    ///
    /// Addressed by block index rather than cell number because the caller
    /// already holds one from [`cell_indices`](Document::cell_indices), and a
    /// second numbering would be a second chance to be off by one.
    pub fn cell_text(&self, block_index: usize) -> Option<String> {
        match self.blocks.get(block_index) {
            Some(Block::Fence { lang, body, .. }) if lang == CELL_LANG => Some(body.clone()),
            _ => None,
        }
    }

    /// The 1-based line the block at `block_index` starts on.
    ///
    /// Used to give a trace site a place in the *document*, so an error names
    /// a line the user can navigate to rather than an offset into a cell.
    pub fn line_of(&self, block_index: usize) -> usize {
        let mut line = 1;
        for b in &self.blocks[..block_index] {
            line += match b {
                Block::Prose { text } => text.split('\n').count(),
                Block::Fence { body, closed, .. } => {
                    let body_lines = if body.is_empty() {
                        0
                    } else {
                        body.split('\n').count()
                    };
                    1 + body_lines + usize::from(*closed)
                }
            };
        }
        line
    }

    /// Replace the output block belonging to the cell at `block_index`, or
    /// insert one if there is none.
    ///
    /// The output block belonging to a cell is the `output` fence immediately
    /// after it, ignoring blank prose. Anything else between them -- a
    /// paragraph the user wrote, a heading -- means the cell has no output
    /// block yet and one is inserted directly beneath the cell, above that
    /// prose.
    ///
    /// Replacement, not appending: a cell run ten times leaves one block. The
    /// record still advances ten times, which is where the ten runs are
    /// remembered.
    pub fn splice(&mut self, block_index: usize, text: &str) {
        debug_assert!(self.blocks[block_index].is_cell());

        let out = Block::Fence {
            ticks: 3,
            indent: String::new(),
            lang: OUTPUT_LANG.to_string(),
            info_rest: String::new(),
            body: text.trim_end().to_string(),
            closed: true,
        };

        // Skip a blank prose separator, but only a blank one.
        let mut at = block_index + 1;
        if let Some(Block::Prose { text }) = self.blocks.get(at) {
            if text.trim().is_empty() {
                at += 1;
            }
        }

        match self.blocks.get(at) {
            Some(b) if b.is_output() => self.blocks[at] = out,
            _ => {
                // Insert with a blank line above, so the document reads as
                // markdown rather than as two fences jammed together.
                self.blocks.insert(
                    block_index + 1,
                    Block::Prose {
                        text: String::new(),
                    },
                );
                self.blocks.insert(block_index + 2, out);
            }
        }
    }

    /// Remove the output block belonging to the cell at `block_index`.
    ///
    /// This is a document edit, not a rewind: the record does not move, and
    /// the run that produced the removed text remains deposited. It exists so
    /// a user can tidy a report before sharing it.
    ///
    /// Returns whether a block was removed, so a caller can report what it did
    /// rather than what it attempted.
    pub fn clear_output(&mut self, block_index: usize) -> bool {
        let mut at = block_index + 1;
        if let Some(Block::Prose { text }) = self.blocks.get(at) {
            if text.trim().is_empty() {
                at += 1;
            }
        }
        if !self.blocks.get(at).is_some_and(|b| b.is_output()) {
            return false;
        }
        self.blocks.remove(at);
        // Take the blank separator with it, if that is all it was.
        if at > block_index + 1 {
            if let Some(Block::Prose { text }) = self.blocks.get(block_index + 1) {
                if text.trim().is_empty() {
                    self.blocks.remove(block_index + 1);
                }
            }
        }
        true
    }
}

/// A rendered value as plain text for the output block.
///
/// Not [`crate::value::text`]: a store delta carries values that have already
/// been through [`crate::value::render`], which is lossy -- a point arrives as
/// a tagged object, a closure as another. Rendering the JSON is therefore a
/// distinct job rather than a duplicate of `text`, and the tagged forms are
/// unwrapped back to the shapes `text` would have produced so the two agree on
/// everything a document displays.
fn text_of(v: &serde_json::Value) -> String {
    use serde_json::Value as J;
    match v {
        J::Null => "none".into(),
        J::Bool(true) => "true".into(),
        J::Bool(false) => "false".into(),
        J::Number(n) => n.to_string(),
        J::String(s) => s.clone(),
        J::Array(a) => {
            let parts: Vec<String> = a.iter().map(text_of).collect();
            format!("[{}]", parts.join(", "))
        }
        J::Object(m) => {
            // The tagged forms `render` builds by hand, unwrapped.
            for (tag, label) in [("point", "point"), ("motion", "motion"), ("closure", "closure")]
            {
                if let Some(J::String(name)) = m.get(tag) {
                    if tag == "point" {
                        let conf = m.get("confidence").map(text_of).unwrap_or_default();
                        return format!("point({name}, conf={conf})");
                    }
                    return format!("{label}({name})");
                }
            }
            if let Some(J::String(name)) = m.get("builtin") {
                return format!("builtin({name})");
            }
            let parts: Vec<String> = m.iter().map(|(k, v)| format!("{k}: {}", text_of(v))).collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Format a cell's result as the plain text that goes in its output block.
///
/// Printed lines first, then changed names, then the error if there was one.
/// No success vocabulary: a cell that ran without error and printed nothing
/// yields an empty block, not "ok" or a checkmark (paper, B6).
pub fn format_result(
    output: &[String],
    delta: &[(String, crate::session::StoreChange)],
    error: Option<&crate::session::CellError>,
) -> String {
    use crate::session::StoreChange;

    let mut s = String::new();
    for line in output {
        s.push_str(line);
        s.push('\n');
    }

    if !delta.is_empty() {
        if !s.is_empty() {
            s.push('\n');
        }
        for (name, change) in delta {
            match change {
                StoreChange::Added { value, .. } => {
                    let _ = writeln!(s, "{name} = {}", text_of(value));
                }
                StoreChange::Updated { from, value, .. } => {
                    let _ = writeln!(
                        s,
                        "{name} = {}  (was {})",
                        text_of(value),
                        text_of(from)
                    );
                }
            }
        }
    }

    if let Some(e) = error {
        if !s.is_empty() {
            s.push('\n');
        }
        let _ = writeln!(s, "{}: {}", e.phase, e.message);
    }

    s
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole design rests on: running a cell must not disturb
    /// a byte the user wrote. If parse/render is not the identity, every run
    /// silently rewrites the report.
    fn round_trips(src: &str) {
        let d = Document::parse(src);
        assert_eq!(d.render(), src, "round trip failed for {src:?}");
    }

    #[test]
    fn round_trip_holds() {
        for src in [
            "",
            "just prose",
            "just prose\n",
            "# heading\n\ntext\n",
            "```turbulance\nitem a = 1\n```\n",
            "before\n\n```turbulance\nitem a = 1\n```\n\nafter\n",
            "```\nbare fence\n```\n",
            "```python\nprint(1)\n```\n",
            "```turbulance\n```\n",
            "text\n```turbulance\nunterminated\n",
            "  ```turbulance\n  indented\n  ```\n",
            "````\n```\nnested\n```\n````\n",
            "```turbulance extra info\nitem a = 1\n```\n",
            "\n\n\n",
            "a\n\n```output\nold\n```\n\nb\n",
        ] {
            round_trips(src);
        }
    }

    #[test]
    fn crlf_survives() {
        let src = "text\r\n\r\n```turbulance\r\nitem a = 1\r\n```\r\n";
        assert_eq!(Document::parse(src).render(), src);
    }

    #[test]
    fn only_turbulance_is_a_cell() {
        let d = Document::parse(
            "```turbulance\nitem a = 1\n```\n\
             ```python\nnot a cell\n```\n\
             ```\nnot a cell either\n```\n\
             ```output\nnor this\n```\n",
        );
        assert_eq!(d.cells(), vec!["item a = 1"]);
    }

    #[test]
    fn splice_replaces_rather_than_appends() {
        let mut d = Document::parse("```turbulance\nitem a = 1\n```\n");
        let i = d.cell_indices()[0];
        d.splice(i, "a = 1");
        let once = d.render();
        assert!(once.contains("```output\na = 1\n```"), "{once}");

        // The second run must leave one block, not two. This is the whole
        // reason output is delimited.
        let mut d = Document::parse(&once);
        let i = d.cell_indices()[0];
        d.splice(i, "a = 2");
        let twice = d.render();
        assert_eq!(twice.matches("```output").count(), 1, "{twice}");
        // Scoped to the output block, not to the whole document: the cell's
        // own source is `item a = 1`, so a document-wide search for the old
        // text finds the *source* and reports a stale block that is not there.
        let block = twice
            .split_once("```output
")
            .and_then(|(_, rest)| rest.split_once("```"))
            .map(|(body, _)| body.to_string())
            .expect("checked above that one exists");
        assert_eq!(block.trim(), "a = 2", "{twice}");
    }

    #[test]
    fn splice_does_not_claim_prose_as_output() {
        // An `output` block separated from the cell by real prose belongs to
        // nothing; the cell gets its own, above the prose.
        let mut d = Document::parse(
            "```turbulance\nitem a = 1\n```\n\nsome commentary\n\n```output\nunrelated\n```\n",
        );
        let i = d.cell_indices()[0];
        d.splice(i, "a = 1");
        let s = d.render();
        assert_eq!(s.matches("```output").count(), 2, "{s}");
        assert!(s.contains("unrelated"), "commentary's block was eaten:\n{s}");
        assert!(
            s.find("a = 1").unwrap() < s.find("some commentary").unwrap(),
            "output landed below the prose:\n{s}"
        );
    }

    #[test]
    fn clear_output_removes_block_and_separator() {
        let mut d = Document::parse("```turbulance\nitem a = 1\n```\n");
        let i = d.cell_indices()[0];
        d.splice(i, "a = 1");
        let mut d = Document::parse(&d.render());
        let i = d.cell_indices()[0];
        d.clear_output(i);
        assert_eq!(d.render(), "```turbulance\nitem a = 1\n```\n");
    }

    #[test]
    fn line_of_points_at_the_document() {
        let d = Document::parse("# title\n\nintro\n\n```turbulance\nitem a = 1\n```\n");
        let i = d.cell_indices()[0];
        // Lines 1-4 are prose; the fence opens on 5.
        assert_eq!(d.line_of(i), 5);
    }

    #[test]
    fn the_model_may_write_prose_only() {
        let d = Document::parse("prose\n```turbulance\nitem a = 1\n```\n```output\nx\n```\n");
        let writable: Vec<bool> = d.blocks.iter().map(|b| b.is_model_writable()).collect();
        assert_eq!(writable, vec![true, false, false]);
    }
}
