//! The Turbulance AST.
//!
//! A port of the node shapes in `prototype/parser.py`, where the tree is plain
//! dicts (`Node = Dict[str, Any]`) so it serialises without a codec. Rust gets
//! an enum instead, but the enum's *serialisation* has to reproduce those dicts
//! key for key -- including which keys are absent -- because that is what the
//! differential harness compares.
//!
//! Two consequences shape everything here, and neither is cosmetic:
//!
//! * **Absent is not null.** `Support` carries `conf: null` when no confidence
//!   was written, while a `Noop` from `import` has no `body` key at all where
//!   one from a loose keyword has `body: []`. Both distinctions are invisible
//!   in Python and both are load-bearing, so `conf` serialises as a null and
//!   `body` is skipped when `None`.
//!
//! * **`graph.py` walks statements structurally**, probing four block field
//!   names and five expression field names on every node regardless of kind,
//!   with a further five-key fallthrough for unrecognised expressions. Only
//!   three sites there match on kind at all. So [`Node::child_blocks`] and
//!   [`Node::exprs_of`] reproduce that field probe rather than reinterpreting
//!   it per variant: the graph stage is then a transcription, and cannot drift
//!   by disagreeing about which fields count.

use serde::Serialize;

/// One `{key: value}` entry of a map literal.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Entry {
    pub key: String,
    pub value: Node,
}

/// A node of the parse tree.
///
/// `kind` is the tag and the remaining fields are flattened beside it, so each
/// variant serialises to exactly the dict the Python builds.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Node {
    // -- statements ------------------------------------------------------
    Funxn {
        name: String,
        params: Vec<String>,
        body: Vec<Node>,
        line: usize,
    },
    Item {
        name: String,
        expr: Box<Node>,
        line: usize,
    },
    Point {
        name: String,
        expr: Box<Node>,
        line: usize,
    },
    Assign {
        name: String,
        expr: Box<Node>,
        line: usize,
    },
    /// The one statement whose block does not appear verbatim: motions are
    /// lifted out of the body into their own list.
    Proposition {
        name: String,
        motions: Vec<Node>,
        body: Vec<Node>,
        line: usize,
    },
    Hypothesis {
        name: String,
        body: Vec<Node>,
        line: usize,
    },
    Motion {
        name: String,
        text: String,
        line: usize,
    },
    Support {
        motion: String,
        conf: Option<Box<Node>>,
        line: usize,
    },
    Contradict {
        motion: String,
        conf: Option<Box<Node>>,
        line: usize,
    },
    /// Carries no `conf` key at all -- the Python builds it without one.
    Inconclusive {
        motion: String,
        line: usize,
    },
    Given {
        cond: Box<Node>,
        then: Vec<Node>,
        otherwise: Vec<Node>,
        line: usize,
    },
    Within {
        target: Box<Node>,
        body: Vec<Node>,
        line: usize,
    },
    Considering {
        quant: String,
        var: String,
        iter: Box<Node>,
        body: Vec<Node>,
        line: usize,
    },
    ForEach {
        var: String,
        iter: Box<Node>,
        body: Vec<Node>,
        line: usize,
    },
    While {
        cond: Box<Node>,
        body: Vec<Node>,
        line: usize,
    },
    Return {
        expr: Option<Box<Node>>,
        line: usize,
    },
    Ensure {
        expr: Box<Node>,
        line: usize,
    },
    Resolve {
        expr: Box<Node>,
        line: usize,
    },
    ExprStmt {
        expr: Box<Node>,
        line: usize,
    },
    /// Both `import` and the loose keywords land here, with *different* key
    /// sets: an import has no `body`, a loose keyword has one (empty when it
    /// opened no block). Hence the skip rather than a defaulted `Vec`.
    Noop {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        body: Option<Vec<Node>>,
        line: usize,
    },

    // -- expressions -----------------------------------------------------
    Pipe {
        left: Box<Node>,
        right: Box<Node>,
        line: usize,
    },
    Binary {
        op: String,
        left: Box<Node>,
        right: Box<Node>,
        line: usize,
    },
    Unary {
        op: String,
        operand: Box<Node>,
        line: usize,
    },
    Call {
        callee: Box<Node>,
        args: Vec<Node>,
        line: usize,
    },
    Field {
        target: Box<Node>,
        field: String,
        line: usize,
    },
    Index {
        target: Box<Node>,
        index: Box<Node>,
        line: usize,
    },
    /// Every numeric literal is `float(t.value)` in the Python, so `1` is
    /// `1.0` and the serialiser must keep the `.0`.
    Num {
        value: f64,
        line: usize,
    },
    Str {
        value: String,
        line: usize,
    },
    Bool {
        value: bool,
        line: usize,
    },
    #[serde(rename = "None")]
    NoneLit {
        line: usize,
    },
    Var {
        name: String,
        line: usize,
    },
    List {
        items: Vec<Node>,
        line: usize,
    },
    Map {
        entries: Vec<Entry>,
        line: usize,
    },
}

impl Node {
    /// The `line` every node carries.
    pub fn line(&self) -> usize {
        use Node::*;
        match self {
            Funxn { line, .. }
            | Item { line, .. }
            | Point { line, .. }
            | Assign { line, .. }
            | Proposition { line, .. }
            | Hypothesis { line, .. }
            | Motion { line, .. }
            | Support { line, .. }
            | Contradict { line, .. }
            | Inconclusive { line, .. }
            | Given { line, .. }
            | Within { line, .. }
            | Considering { line, .. }
            | ForEach { line, .. }
            | While { line, .. }
            | Return { line, .. }
            | Ensure { line, .. }
            | Resolve { line, .. }
            | ExprStmt { line, .. }
            | Noop { line, .. }
            | Pipe { line, .. }
            | Binary { line, .. }
            | Unary { line, .. }
            | Call { line, .. }
            | Field { line, .. }
            | Index { line, .. }
            | Num { line, .. }
            | Str { line, .. }
            | Bool { line, .. }
            | NoneLit { line, .. }
            | Var { line, .. }
            | List { line, .. }
            | Map { line, .. } => *line,
        }
    }

    /// The name this node declares, if it is a declaration form.
    ///
    /// `graph.py` reads `st.get("name")` and relies on the key being absent for
    /// everything else, so this returns `None` for non-declarations even though
    /// several other variants do carry a `name` field.
    pub fn decl_kind(&self) -> Option<&'static str> {
        match self {
            Node::Item { .. } => Some("item"),
            Node::Funxn { .. } => Some("funxn"),
            Node::Proposition { .. } => Some("proposition"),
            Node::Motion { .. } => Some("motion"),
            Node::Point { .. } => Some("point"),
            Node::Hypothesis { .. } => Some("hypothesis"),
            _ => None,
        }
    }

    /// The `name` key, for the nodes that have one.
    pub fn name(&self) -> Option<&str> {
        match self {
            Node::Funxn { name, .. }
            | Node::Item { name, .. }
            | Node::Point { name, .. }
            | Node::Assign { name, .. }
            | Node::Proposition { name, .. }
            | Node::Hypothesis { name, .. }
            | Node::Motion { name, .. }
            | Node::Var { name, .. } => Some(name),
            _ => None,
        }
    }

    /// Child statement blocks, reproducing `graph.py`'s `_child_blocks`.
    ///
    /// The Python probes the field names `body`, `then`, `otherwise`, `motions`
    /// in that order and keeps each that holds a **non-empty** list of nodes.
    /// The order matters (it fixes traversal order) and so does the emptiness
    /// test, so both are reproduced rather than tidied.
    pub fn child_blocks(&self) -> Vec<&[Node]> {
        use Node::*;
        let fields: [&[Node]; 4] = match self {
            // body, then, otherwise, motions
            Funxn { body, .. }
            | Hypothesis { body, .. }
            | Within { body, .. }
            | Considering { body, .. }
            | ForEach { body, .. }
            | While { body, .. } => [body, &[], &[], &[]],
            Noop {
                body: Some(body), ..
            } => [body, &[], &[], &[]],
            Given {
                then, otherwise, ..
            } => [&[], then, otherwise, &[]],
            // A proposition's `body` and `motions` are both probed, in that
            // order: `body` is the first field name and `motions` the last.
            Proposition { body, motions, .. } => [body, &[], &[], motions],
            _ => [&[], &[], &[], &[]],
        };
        fields.into_iter().filter(|b| !b.is_empty()).collect()
    }

    /// Sub-expressions, reproducing `graph.py`'s `_exprs_of`.
    ///
    /// The Python probes `expr`, `cond`, `iter`, `target`, `conf` in that order
    /// and keeps each that holds a node. `Within`'s `target` is an expression
    /// and is picked up here; so are `Field`'s and `Index`'s, under the same
    /// name, which is why this is a field probe and not a per-kind match. A
    /// `conf` of `None` is skipped, matching the Python's `isinstance` test
    /// against a null.
    pub fn exprs_of(&self) -> Vec<&Node> {
        use Node::*;
        let fields: [Option<&Node>; 5] = match self {
            // expr, cond, iter, target, conf
            Item { expr, .. }
            | Point { expr, .. }
            | Assign { expr, .. }
            | Ensure { expr, .. }
            | Resolve { expr, .. }
            | ExprStmt { expr, .. } => [Some(expr), None, None, None, None],
            Return { expr, .. } => [expr.as_deref(), None, None, None, None],
            Given { cond, .. } | While { cond, .. } => [None, Some(cond), None, None, None],
            Considering { iter, .. } | ForEach { iter, .. } => [None, None, Some(iter), None, None],
            Within { target, .. } | Field { target, .. } | Index { target, .. } => {
                [None, None, None, Some(target), None]
            }
            Support { conf, .. } | Contradict { conf, .. } => {
                [None, None, None, None, conf.as_deref()]
            }
            _ => [None, None, None, None, None],
        };
        fields.into_iter().flatten().collect()
    }
}

/// Names occurring free in an expression, as `graph.py`'s `free_names`.
///
/// Deliberately a walk with a fallthrough rather than an exhaustive match, so
/// that it keeps agreeing with the Python's `_walk_expr`: that function
/// recognises four kinds and then probes `left`, `right`, `operand`, `target`,
/// `index` on anything else. A node with none of those keys contributes
/// nothing -- including `Str`, `Num`, and, notably, a `Call`'s `args` reached
/// only through the explicit `Call` branch.
pub fn free_names(expr: &Node, out: &mut std::collections::BTreeSet<String>) {
    match expr {
        Node::Var { name, .. } => {
            out.insert(name.clone());
        }
        Node::Call { callee, args, .. } => {
            free_names(callee, out);
            for a in args {
                free_names(a, out);
            }
        }
        Node::Map { entries, .. } => {
            // Only values: a key is a bare word, not a reference.
            for e in entries {
                free_names(&e.value, out);
            }
        }
        Node::List { items, .. } => {
            for i in items {
                free_names(i, out);
            }
        }
        // The fallthrough: `left`, `right`, `operand`, `target`, `index`.
        Node::Pipe { left, right, .. } | Node::Binary { left, right, .. } => {
            free_names(left, out);
            free_names(right, out);
        }
        Node::Unary { operand, .. } => free_names(operand, out),
        Node::Field { target, .. } => free_names(target, out),
        Node::Index { target, index, .. } => {
            free_names(target, out);
            free_names(index, out);
        }
        _ => {}
    }
}
