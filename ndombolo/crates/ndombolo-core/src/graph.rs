//! The static script graph (paper, Prop. 2.4).
//!
//! A port of `prototype/graph.py`. One traversal of the parse tree with a scope
//! stack, emitting every item and every contact exactly once. No execution is
//! involved: the graph exists before any run, which is the sense in which the
//! script *is* the causal knowledge graph rather than describing one.
//!
//! # The hazard that shapes this file
//!
//! `graph.py` walks statements **structurally**. It probes four block field
//! names and five expression field names on every node regardless of kind, and
//! matches on kind at only three sites. A Rust port that dispatched on the enum
//! -- the obvious translation -- would silently visit different children,
//! because absent keys matter: a `Noop` from `import` has no `body` at all, and
//! `_child_blocks` keeps only **non-empty** lists.
//!
//! That probe lives in [`Node::child_blocks`] and [`Node::exprs_of`], written
//! during the parser stage for this purpose. This module transcribes the
//! traversal and never re-derives which fields count.
//!
//! # Other hazards, each pinned by a corpus case
//!
//! * `free_names` returns a **set**, so `a + a` is one contact, not two.
//! * A redeclaration keeps the **first** site (`setdefault`), not the last.
//! * `contact` silently drops an edge whose endpoint is not a declared item, so
//!   a reference to a builtin or an undeclared name contributes nothing.
//! * A self-contact is dropped: `item a = a` yields no edge.
//! * `Assign` is not a declaration, so at top level its target falls back to
//!   `opener`, which is `None` -- the contact vanishes rather than landing on
//!   the assigned name.
//! * A verdict contacts the motion it names, and the same statement's `conf`
//!   expression is picked up by the field probe under the same target.
//! * `to_json` sorts items by `(stage, line, name)` and contacts by the
//!   `(from, to)` pair, so the output is stable while the builder's own maps
//!   need not be.

use crate::ast::{free_names, Node};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// An item: a named binding site, with where and when it was declared.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Item {
    pub name: String,
    pub kind: &'static str,
    pub line: usize,
    pub stage: usize,
}

/// A weighted contact between two declared items.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Contact {
    pub from: String,
    pub to: String,
    pub weight: usize,
}

/// Items, weighted contacts, and the stage each item was declared at.
#[derive(Debug, Default, Clone)]
pub struct ScriptGraph {
    /// Insertion order is irrelevant -- `to_json` sorts -- but a map keyed by
    /// name is what `setdefault` needs.
    items: BTreeMap<String, Item>,
    contacts: BTreeMap<(String, String), usize>,
}

impl ScriptGraph {
    /// Record a declaration.
    ///
    /// A redeclaration keeps the **first** site: items are not undeclared, and
    /// the earliest declaration is the one later stages contain. This is
    /// `setdefault` in the Python, and the direction matters -- keeping the
    /// last site would break prefix containment (paper, Prop. 2.7).
    pub fn declare(&mut self, name: &str, kind: &'static str, line: usize, stage: usize) {
        self.items.entry(name.to_string()).or_insert_with(|| Item {
            name: name.to_string(),
            kind,
            line,
            stage,
        });
    }

    /// Record a contact, if both endpoints are declared items and differ.
    ///
    /// The two guards are why a graph never names anything it did not declare:
    /// a mention of a builtin, of an undeclared name, or of the item itself
    /// contributes no edge at all.
    pub fn contact(&mut self, src: &str, dst: &str) {
        if src == dst || !self.items.contains_key(src) || !self.items.contains_key(dst) {
            return;
        }
        *self
            .contacts
            .entry((src.to_string(), dst.to_string()))
            .or_insert(0) += 1;
    }

    pub fn items(&self) -> impl Iterator<Item = &Item> {
        self.items.values()
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }

    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// The oracle's shape: items sorted by `(stage, line, name)`, contacts by
    /// the pair.
    pub fn to_json(&self) -> serde_json::Value {
        let mut items: Vec<&Item> = self.items.values().collect();
        items.sort_by(|a, b| (a.stage, a.line, &a.name).cmp(&(b.stage, b.line, &b.name)));
        let contacts: Vec<Contact> = self
            .contacts
            .iter()
            .map(|((a, b), w)| Contact {
                from: a.clone(),
                to: b.clone(),
                weight: *w,
            })
            .collect();
        serde_json::json!({
            "items": items,
            "contacts": contacts,
            "item_count": self.items.len(),
            "contact_count": self.contacts.len(),
        })
    }
}

/// Accretive builder: [`add_cell`](GraphBuilder::add_cell) extends the graph,
/// never rewrites it.
///
/// Prefix containment (paper, Prop. 2.7) holds because no method removes an
/// item or decreases a weight. That is the same no-backflow discipline the
/// runtime record keeps, and it is why the editor's "undo" increments the
/// record rather than resetting it (blueprint, B2).
#[derive(Debug, Default)]
pub struct GraphBuilder {
    pub graph: ScriptGraph,
    stage: usize,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Extend the graph with one cell's statements.
    ///
    /// Two passes: declare, then link. A statement may mention an item declared
    /// later in the same cell, so the declaration pass has to complete first --
    /// otherwise the "is it a declared item" guard in [`ScriptGraph::contact`]
    /// would drop the forward reference.
    pub fn add_cell(&mut self, body: &[Node], stage: usize) {
        self.stage = stage;
        self.declare(body);
        self.link(body, None);
    }

    // -- pass 1: declarations ---------------------------------------------

    fn declare(&mut self, body: &[Node]) {
        for st in body {
            if let (Some(kind), Some(name)) = (st.decl_kind(), st.name()) {
                self.graph.declare(name, kind, st.line(), self.stage);
            }
            for child in st.child_blocks() {
                self.declare(child);
            }
        }
    }

    // -- pass 2: contacts --------------------------------------------------

    fn link(&mut self, body: &[Node], opener: Option<&str>) {
        for st in body {
            self.link_stmt(st, opener);
        }
    }

    fn link_stmt(&mut self, st: &Node, opener: Option<&str>) {
        // A declaration is the target of contacts from what its term mentions;
        // anything else contacts the scope it sits in.
        let target: Option<&str> = if st.decl_kind().is_some() {
            st.name()
        } else {
            opener
        };

        for expr in st.exprs_of() {
            let mut names: BTreeSet<String> = BTreeSet::new();
            free_names(expr, &mut names);
            for name in &names {
                // The Python falls back to `opener` when `target` is None. It
                // reads like a dead arm -- `target` already *is* `opener` for a
                // non-declaration -- but it is reachable for a declaration node
                // carrying no `name` key, where `st.get("name")` is None while
                // `opener` is set. Every declaration the parser builds has a
                // name, so no corpus case distinguishes the two; the fallback
                // is transcribed anyway rather than argued away, because the
                // argument depends on the parser and this file does not.
                if let Some(t) = target.or(opener) {
                    self.graph.contact(name, t);
                }
            }
        }

        // A verdict contacts the motion it is about.
        if let Node::Support { motion, .. }
        | Node::Contradict { motion, .. }
        | Node::Inconclusive { motion, .. } = st
        {
            // Same `target`-then-`opener` fallback as above, for the same
            // reason: a verdict is never a declaration, so `target` is `opener`
            // here and the `or` is inert -- but it is what the Python writes.
            if let Some(t) = target.or(opener) {
                self.graph.contact(motion, t);
            }
        }

        // A declaration with a body opens a scope: everything inside is in
        // contact with it (paper, Def. 2.3, second clause).
        let inner_opener: Option<&str> = if st.decl_kind().is_some() {
            st.name()
        } else {
            opener
        };

        if let Node::Proposition { name, motions, .. } = st {
            for m in motions {
                if let Some(mn) = m.name() {
                    self.graph.contact(name, mn);
                }
            }
        }

        for child in st.child_blocks() {
            self.link(child, inner_opener);
        }
    }
}
