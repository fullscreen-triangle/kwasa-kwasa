//! The observing model: a local ollama chat with tools over the run's record.
//!
//! # Why tools rather than a stuffed prompt
//!
//! The obvious design pastes the trace into the prompt and asks the question.
//! That fails on both ends. A real run emits thousands of events, so the trace
//! does not fit; and pasting it would make the model's answer a summary of text
//! it was handed, when the claim the whole design rests on is that a question
//! about the output is answerable *by search over the events*. Giving the model
//! [`query_trace`] makes that literal: it locates the events that bear on the
//! question and reads those.
//!
//! # The no-backflow wall
//!
//! The model reads the record and deposits prose. It never writes a cell and
//! never writes an output block -- the runtime's determination is not something
//! it may report, only something it may describe. That rule lives in
//! `Block::is_model_writable`; this module simply never produces anything else,
//! and [`Answer`] has no field that could carry a cell.
//!
//! # Not a cache
//!
//! Each question runs the tool loop afresh. Nothing here remembers a previous
//! answer to skip a search, because a lookup that let a repeated question skip
//! the search would be exactly the cache the blueprint forbids (B3). Repeated
//! probes are cheap; caching determinations is not permitted.

use serde_json::{json, Value as Json};
use std::time::Duration;

/// Where ollama listens by default.
pub const DEFAULT_HOST: &str = "http://127.0.0.1:11434";

/// The model the editor asks unless told otherwise.
///
/// Small on purpose: it runs on a laptop beside the document, and the job is
/// reading a trace and describing it, not reasoning from nothing.
pub const DEFAULT_MODEL: &str = "llama3.2";

/// A model call can be slow on a cold load; a short timeout would turn a
/// working setup into a mysterious failure.
const TIMEOUT: Duration = Duration::from_secs(180);

/// How many tool round trips before we stop and answer with what we have.
///
/// A bound is necessary: a model that keeps calling tools without concluding
/// would hang the editor. Six is enough for "find the events for this item,
/// then look at the rule that wrote it" with room to spare.
const MAX_ROUNDS: usize = 6;

/// What the model said, and what it looked at to say it.
pub struct Answer {
    /// Prose. This is the only thing that may enter the document.
    pub text: String,
    /// The tool calls made, in order, for display: the answer's provenance is
    /// visible rather than asserted.
    pub searches: Vec<Json>,
}

/// One conversation with the model, over one document's events.
pub struct Model {
    host: String,
    name: String,
    /// Every trace event of the current run, as `ndombolo trace` reports them.
    /// Held rather than re-derived so a tool call is a search, not a re-run.
    events: Vec<Json>,
}

impl Model {
    pub fn new(host: &str, name: &str, events: Vec<Json>) -> Model {
        Model {
            host: host.to_string(),
            name: name.to_string(),
            events,
        }
    }

    /// Ask a question, letting the model search the events to answer it.
    pub fn ask(&self, question: &str) -> Result<Answer, String> {
        let mut messages = vec![
            json!({ "role": "system", "content": SYSTEM }),
            json!({ "role": "user", "content": question }),
        ];
        let mut searches: Vec<Json> = Vec::new();

        for _ in 0..MAX_ROUNDS {
            let reply = self.chat(&messages)?;
            let message = reply
                .get("message")
                .cloned()
                .ok_or_else(|| format!("ollama returned no message: {reply}"))?;

            let calls = message
                .get("tool_calls")
                .and_then(Json::as_array)
                .cloned()
                .unwrap_or_default();

            if calls.is_empty() {
                let text = message
                    .get("content")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                return Ok(Answer { text, searches });
            }

            messages.push(message);
            for call in calls {
                let function = call.get("function").cloned().unwrap_or(json!({}));
                let name = function.get("name").and_then(Json::as_str).unwrap_or("");
                let args = function.get("arguments").cloned().unwrap_or(json!({}));

                let result = self.call_tool(name, &args);
                searches.push(json!({ "tool": name, "arguments": args,
                                      "events_found": result.get("count").cloned() }));

                // The result goes back as a tool message; ollama expects the
                // content as a string, not as nested JSON.
                messages.push(json!({
                    "role": "tool",
                    "content": serde_json::to_string(&result).unwrap_or_default(),
                }));
            }
        }

        // Out of rounds. Say so rather than presenting a partial search as an
        // answer -- the blueprint forbids success vocabulary (B6), and it would
        // be equally wrong to imply a conclusion the model did not reach.
        Ok(Answer {
            text: format!(
                "I searched the record {MAX_ROUNDS} times without settling on an \
                 answer. The searches are listed below; a narrower question \
                 (naming a rule or an item) usually resolves it."
            ),
            searches,
        })
    }

    /// Dispatch one tool call against the events.
    fn call_tool(&self, name: &str, args: &Json) -> Json {
        match name {
            "query_trace" => {
                let rule = args.get("rule").and_then(Json::as_str);
                let item = args.get("item").and_then(Json::as_str);
                let limit = args
                    .get("limit")
                    .and_then(Json::as_u64)
                    .unwrap_or(20)
                    .min(50) as usize;
                self.query_trace(rule, item, limit)
            }
            "list_rules" => self.list_rules(),
            other => json!({ "error": format!("no tool named {other}") }),
        }
    }

    /// Events matching a rule, an item, or both.
    ///
    /// An item matches if the event reads it or writes it -- the distinction
    /// the caller usually wants is *which* of those, and that is visible in the
    /// event returned, so filtering on it here would only hide the answer.
    fn query_trace(&self, rule: Option<&str>, item: Option<&str>, limit: usize) -> Json {
        let matched: Vec<&Json> = self
            .events
            .iter()
            .filter(|ev| {
                let e = ev.get("event").unwrap_or(ev);
                if let Some(r) = rule {
                    if e.get("rule").and_then(Json::as_str) != Some(r) {
                        return false;
                    }
                }
                if let Some(i) = item {
                    if !touches(e, i) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // The count is of everything that matched; the events are the first
        // `limit` of them. Reporting a truncated count would let the model
        // conclude "only three" from a window.
        json!({
            "count": matched.len(),
            "returned": matched.len().min(limit),
            "events": matched.into_iter().take(limit).collect::<Vec<_>>(),
        })
    }

    /// Every rule that fired, with how often.
    ///
    /// The entry point for a model that does not yet know the vocabulary: it
    /// cannot filter by a rule name it has never seen.
    fn list_rules(&self) -> Json {
        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for ev in &self.events {
            let e = ev.get("event").unwrap_or(ev);
            if let Some(r) = e.get("rule").and_then(Json::as_str) {
                *counts.entry(r.to_string()).or_insert(0) += 1;
            }
        }
        json!({ "events": self.events.len(), "rules": counts })
    }

    fn chat(&self, messages: &[Json]) -> Result<Json, String> {
        let body = json!({
            "model": self.name,
            "messages": messages,
            "stream": false,
            "tools": tools(),
            // Deterministic-ish: the model is describing a record, and a
            // different retelling on each ask would make its answers harder to
            // check against the events it names.
            "options": { "temperature": 0.2 },
        });
        let url = format!("{}/api/chat", self.host);
        let reply = crate::http::post_json(&url, &body, TIMEOUT)?;
        if let Some(err) = reply.get("error").and_then(Json::as_str) {
            return Err(format!("ollama: {err}"));
        }
        Ok(reply)
    }
}

/// Whether an event reads or writes `item`.
fn touches(event: &Json, item: &str) -> bool {
    let reads = event
        .get("reads")
        .and_then(Json::as_array)
        .map(|a| {
            a.iter().any(|r| {
                r.as_str() == Some(item) || r.get("item").and_then(Json::as_str) == Some(item)
            })
        })
        .unwrap_or(false);
    let writes = event
        .get("writes")
        .map(|w| {
            w.as_str() == Some(item) || w.get("item").and_then(Json::as_str) == Some(item)
        })
        .unwrap_or(false);
    reads || writes
}

fn tools() -> Json {
    json!([
        {
            "type": "function",
            "function": {
                "name": "query_trace",
                "description": "Search the events this run emitted. Filter by rule name, \
                                by item name, or both. An item matches if the event reads \
                                or writes it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rule": { "type": "string",
                                  "description": "Rule name, e.g. declare-item" },
                        "item": { "type": "string",
                                  "description": "Item name, e.g. greeting" },
                        "limit": { "type": "integer",
                                   "description": "Max events to return (default 20)" }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_rules",
                "description": "List every rule that fired in this run, with counts. \
                                Use this first if you do not know the rule names.",
                "parameters": { "type": "object", "properties": {} }
            }
        }
    ])
}

/// The model's brief.
///
/// The last paragraph is the wall in words the model can act on: it may not
/// claim a determination, because it did not compute one. That is a restriction
/// on what it should say; `Block::is_model_writable` is the restriction on what
/// it can write, and the two are deliberately not the same mechanism.
const SYSTEM: &str = "\
You are reading the record of a Turbulance run inside an ndombolo document.

The script was compiled deterministically. You did not run it and you cannot \
run it. What you have is the trace: an ordered list of events, each naming the \
rule that fired, the line it fired at, the items it read, and the item it \
wrote.

Answer questions by searching that trace with the tools. Call list_rules first \
if you do not know the vocabulary. Cite the events you used -- by rule and \
line -- so the reader can check you against the record.

Write prose. Do not write Turbulance code, and do not state what a cell would \
output: the compiler determines that, not you. If the events do not settle the \
question, say which search you ran and what it returned, rather than filling \
the gap.";

#[cfg(test)]
mod tests {
    use super::*;

    fn events() -> Vec<Json> {
        vec![
            json!({ "cell": 0, "event": { "seq": 0, "rule": "declare-item",
                    "site": { "line": 4 }, "reads": [],
                    "writes": { "item": "a", "value": 1 } } }),
            json!({ "cell": 0, "event": { "seq": 1, "rule": "call-builtin",
                    "site": { "line": 5 }, "reads": ["a"], "writes": null } }),
            json!({ "cell": 1, "event": { "seq": 2, "rule": "declare-item",
                    "site": { "line": 9 }, "reads": ["a"],
                    "writes": { "item": "b", "value": 2 } } }),
        ]
    }

    #[test]
    fn an_item_matches_whether_read_or_written() {
        let m = Model::new(DEFAULT_HOST, "test", events());
        // `a` is written once and read twice.
        let r = m.query_trace(None, Some("a"), 20);
        assert_eq!(r["count"], json!(3));
        // `b` only appears as a write.
        assert_eq!(m.query_trace(None, Some("b"), 20)["count"], json!(1));
    }

    #[test]
    fn rule_and_item_filters_compose() {
        let m = Model::new(DEFAULT_HOST, "test", events());
        assert_eq!(m.query_trace(Some("declare-item"), None, 20)["count"], json!(2));
        assert_eq!(
            m.query_trace(Some("declare-item"), Some("a"), 20)["count"],
            json!(2)
        );
        assert_eq!(
            m.query_trace(Some("call-builtin"), Some("b"), 20)["count"],
            json!(0)
        );
    }

    #[test]
    fn the_count_is_of_matches_not_of_the_window() {
        // A limit must not let the model conclude "only one".
        let m = Model::new(DEFAULT_HOST, "test", events());
        let r = m.query_trace(None, Some("a"), 1);
        assert_eq!(r["count"], json!(3));
        assert_eq!(r["returned"], json!(1));
        assert_eq!(r["events"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn rules_are_listed_with_counts() {
        let m = Model::new(DEFAULT_HOST, "test", events());
        let r = m.list_rules();
        assert_eq!(r["rules"]["declare-item"], json!(2));
        assert_eq!(r["rules"]["call-builtin"], json!(1));
        assert_eq!(r["events"], json!(3));
    }
}
