//! The deterministic compiler K, ported from `prototype/evaluator.py`.
//!
//! Big-step evaluation of the deterministic Turbulance fragment. Every rule
//! application emits one trace event recording the rule, the site, the items
//! read with their values, and the item written with its value -- which is what
//! makes every atomic question a projection of the trace (paper, Thm. 5.2).
//!
//! No rule consults a clock, an address, a random source, or a model. That is
//! checkable by reading this file: the only inputs to [`Compiler::exec_stmt`]
//! and [`Compiler::eval_expr`] are the node and the environment.
//!
//! # What is actually under test
//!
//! The trace, not the final store. A port that agreed on computed values while
//! disagreeing on which rules fired, in what order, reading what, would have
//! ported the arithmetic and lost the property the runtime exists for. So the
//! event stream is reproduced event for event, including the events that carry
//! no value and exist only to mark that a rule ran.
//!
//! # Where the faithful port is not the obvious Rust
//!
//! Each of these is pinned by a script in `figures/dump_eval.py`, and each is a
//! place where writing what Rust makes easy would diverge:
//!
//! * `==` compares **rendered** values, so `1 == true`.
//! * `%` is `math.fmod`, which is Rust's `%` -- *not* `rem_euclid`.
//! * `round` is banker's rounding: `round(2.5)` is 2. `f64::round` gives 3.
//! * `and`/`or` return a **bool**, not the operand: `true and 5` is `true`.
//! * An assignment to an undeclared name **defines** it. There is no
//!   "assignment to undefined name" error reachable from any source.
//! * `resolve-motion`'s reads are a **map keyed by motion**, so repeated
//!   verdicts on one motion collapse to one entry holding the last value.
//! * A `Pipe` evaluates its **right operand first**, then its left.
//! * An `expr`/`resolve`/`return` note is `str(render(v))` -- a Python repr of
//!   JSON, not the language's own text form.
//! * `_reads_for` **drops unbound names**, so reads are not the free-name set.

use std::cell::RefCell;
use std::rc::Rc;

use serde::Serialize;
use serde_json::{json, Value as Json};

use crate::ast::{Entry, Node};
use crate::value::{
    free_names_sorted, note_of, render, text, Builtin, BuiltinKind, Closure, Motion, OrderedMap,
    Point, Value,
};

/// Guards non-termination; exceeding it is an error.
pub const MAX_STEPS: u64 = 200_000;
pub const MAX_LOOP: u64 = 100_000;

/// A runtime error, carrying the line it was raised at.
///
/// `message` is the bare text and `to_string` is the `line N: ...` form, which
/// is what the oracle dumps -- the Python's `RuntimeErr.__init__` passes the
/// prefixed string to `Exception` while keeping the bare one as an attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeErr {
    pub message: String,
    pub line: usize,
}

impl RuntimeErr {
    fn new(message: impl Into<String>, line: usize) -> Self {
        Self {
            message: message.into(),
            line,
        }
    }
}

impl std::fmt::Display for RuntimeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for RuntimeErr {}

/// A `return`, which unwinds like an exception rather than propagating a value.
///
/// Kept distinct from [`RuntimeErr`] because a top-level `return` escapes the
/// Python as an uncaught `ReturnSignal` -- a different failure mode from an
/// error, and one the harness must not silently convert into one.
#[derive(Debug)]
pub enum Flow {
    Err(RuntimeErr),
    Return(Value),
}

impl From<RuntimeErr> for Flow {
    fn from(e: RuntimeErr) -> Self {
        Flow::Err(e)
    }
}

type Res<T> = Result<T, Flow>;

// -- environment ---------------------------------------------------------

/// A shared, mutable scope.
///
/// `Rc<RefCell<_>>` because a closure captures its *defining* environment and
/// outlives the call that created it, and because `flatten` must see writes
/// made through any other handle to the same scope.
pub type EnvRef = Rc<RefCell<Env>>;

#[derive(Debug, Default)]
pub struct Env {
    vars: OrderedMap,
    parent: Option<EnvRef>,
}

impl PartialEq for Env {
    /// Compared by binding, not by identity, so that a `Closure` deriving
    /// `PartialEq` does not recurse into a cycle through its captured scope.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Env {
    pub fn root() -> EnvRef {
        Rc::new(RefCell::new(Env::default()))
    }

    pub fn child(parent: &EnvRef) -> EnvRef {
        Rc::new(RefCell::new(Env {
            vars: OrderedMap::new(),
            parent: Some(Rc::clone(parent)),
        }))
    }
}

fn env_get(env: &EnvRef, name: &str, line: usize) -> Result<Value, RuntimeErr> {
    let mut cur = Rc::clone(env);
    loop {
        if let Some(v) = cur.borrow().vars.get(name) {
            return Ok(v.clone());
        }
        let next = cur.borrow().parent.clone();
        match next {
            Some(p) => cur = p,
            // The `!r` of a Python string is single-quoted, and the message is
            // compared verbatim.
            None => return Err(RuntimeErr::new(format!("undefined name '{}'", name), line)),
        }
    }
}

fn env_has(env: &EnvRef, name: &str) -> bool {
    let mut cur = Rc::clone(env);
    loop {
        if cur.borrow().vars.get(name).is_some() {
            return true;
        }
        let next = cur.borrow().parent.clone();
        match next {
            Some(p) => cur = p,
            None => return false,
        }
    }
}

fn env_define(env: &EnvRef, name: &str, value: Value) {
    env.borrow_mut().vars.insert(name.to_string(), value);
}

/// Assign to the innermost scope that already binds the name.
///
/// The caller must have checked [`env_has`] first, exactly as `_st_Assign`
/// does; this returns quietly if it finds nothing, because the Python's error
/// branch on the same path is unreachable and reproducing it would add a
/// failure the oracle does not have.
fn env_assign(env: &EnvRef, name: &str, value: Value) {
    let mut cur = Rc::clone(env);
    loop {
        let found = cur.borrow().vars.get(name).is_some();
        if found {
            cur.borrow_mut().vars.insert(name.to_string(), value);
            return;
        }
        let next = cur.borrow().parent.clone();
        match next {
            Some(p) => cur = p,
            None => return,
        }
    }
}

/// The visible store, innermost binding winning.
///
/// Built outermost-first so that an inner rebinding overwrites in place and
/// keeps the *outer* scope's insertion position -- which is what Python's
/// `dict.update` over a reversed chain does, and what fixes the key order of
/// every store snapshot the harness compares.
pub fn env_flatten(env: &EnvRef) -> OrderedMap {
    let mut chain: Vec<EnvRef> = Vec::new();
    let mut cur = Some(Rc::clone(env));
    while let Some(e) = cur {
        cur = e.borrow().parent.clone();
        chain.push(e);
    }
    let mut out = OrderedMap::new();
    for e in chain.into_iter().rev() {
        for (k, v) in e.borrow().vars.iter() {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

// -- propositions --------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Verdict {
    pub motion: String,
    /// `support` | `contradict` | `inconclusive`.
    pub stance: String,
    pub confidence: f64,
    pub line: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PropositionState {
    pub name: String,
    pub motions: Vec<Motion>,
    pub verdicts: Vec<Verdict>,
}

impl PropositionState {
    /// Noisy-or aggregation over verdicts for one motion.
    ///
    /// Not a sum and not a max: two supports of 0.5 give 0.75, and a
    /// contradict multiplies the complement in rather than subtracting.
    pub fn score(&self, motion: &str) -> f64 {
        let pos: Vec<f64> = self
            .verdicts
            .iter()
            .filter(|v| v.motion == motion && v.stance == "support")
            .map(|v| v.confidence)
            .collect();
        let neg: Vec<f64> = self
            .verdicts
            .iter()
            .filter(|v| v.motion == motion && v.stance == "contradict")
            .map(|v| v.confidence)
            .collect();
        noisy_or(&pos) * (1.0 - noisy_or(&neg))
    }
}

fn noisy_or(cs: &[f64]) -> f64 {
    let mut acc = 1.0;
    for c in cs {
        acc *= 1.0 - clamp(*c);
    }
    1.0 - acc
}

fn clamp(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

// -- the compiler --------------------------------------------------------

/// K, carried across cells: the store and the trace both accrete.
pub struct Compiler {
    pub env: EnvRef,
    pub trace: Vec<Json>,
    pub output: Vec<String>,
    /// Insertion-ordered, because the dump serialises it and a sorted map
    /// would reorder propositions declared out of alphabetical order.
    pub propositions: Vec<(String, PropositionState)>,
    pub points: Vec<(String, Rc<Point>)>,
    pub steps: u64,
    prop_stack: Vec<usize>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        let env = Env::root();
        let mut c = Compiler {
            env,
            trace: Vec::new(),
            output: Vec::new(),
            propositions: Vec::new(),
            points: Vec::new(),
            steps: 0,
            prop_stack: Vec::new(),
        };
        c.install_builtins();
        c
    }

    fn install_builtins(&mut self) {
        use BuiltinKind::*;
        for (name, which) in [
            ("print", Print),
            ("len", Len),
            ("sum", Sum),
            ("min", Min),
            ("max", Max),
            ("abs", Abs),
            ("round", Round),
        ] {
            env_define(&self.env, name, Value::Builtin(Builtin { which }));
        }
    }

    // -- tracing ---------------------------------------------------------

    /// Emit one event.
    ///
    /// `writes` is omitted entirely when there is nothing written, and `note`
    /// likewise -- absent keys, not nulls, because that is what the Python's
    /// conditional `event[...] = ...` produces and what the harness compares.
    fn emit(
        &mut self,
        rule: &str,
        line: usize,
        reads: &[(String, Value)],
        writes: Option<(&str, &Value)>,
        note: Option<String>,
    ) {
        let reads_json: Vec<Json> = reads
            .iter()
            .map(|(k, v)| json!({"item": k, "value": render(v)}))
            .collect();
        let mut event = json!({
            "seq": self.trace.len(),
            "rule": rule,
            "site": {"line": line},
            "reads": reads_json,
        });
        let obj = event.as_object_mut().expect("object");
        if let Some((item, value)) = writes {
            obj.insert(
                "writes".into(),
                json!({"item": item, "value": render(value), "tag": value.tag()}),
            );
        }
        if let Some(n) = note {
            obj.insert("note".into(), Json::String(n));
        }
        self.trace.push(event);
    }

    /// The bound names an expression mentions, with their current values.
    ///
    /// Sorted, and **filtered to names that are bound**: a free name with no
    /// binding contributes nothing. So `reads` is not the free-name set, and a
    /// port that emitted the whole set would diverge on any expression naming
    /// something not yet declared -- including the short-circuit cases where
    /// the unbound name is never evaluated at all.
    fn reads_for(&self, expr: Option<&Node>) -> Vec<(String, Value)> {
        let Some(expr) = expr else {
            return Vec::new();
        };
        let line = expr.line();
        free_names_sorted(expr)
            .into_iter()
            .filter_map(|n| {
                if env_has(&self.env, &n) {
                    env_get(&self.env, &n, line).ok().map(|v| (n, v))
                } else {
                    None
                }
            })
            .collect()
    }

    fn tick(&mut self, line: usize) -> Result<(), RuntimeErr> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(RuntimeErr::new("step limit exceeded", line));
        }
        Ok(())
    }

    // -- statements -------------------------------------------------------

    pub fn exec_block(&mut self, body: &[Node]) -> Res<()> {
        for st in body {
            self.exec_stmt(st)?;
        }
        Ok(())
    }

    pub fn exec_stmt(&mut self, st: &Node) -> Res<()> {
        self.tick(st.line())?;
        match st {
            Node::Noop { text, line, .. } => {
                self.emit("noop", *line, &[], None, Some(text.clone()));
            }

            Node::Funxn {
                name,
                params,
                body,
                line,
            } => {
                let clo = Value::Closure(Rc::new(Closure {
                    name: name.clone(),
                    params: params.clone(),
                    body: body.clone(),
                    // The *defining* environment, captured now.
                    env: Rc::clone(&self.env),
                }));
                env_define(&self.env, name, clo.clone());
                self.emit("declare-funxn", *line, &[], Some((name, &clo)), None);
            }

            Node::Item { name, expr, line } => {
                let reads = self.reads_for(Some(expr));
                let value = self.eval_expr(expr)?;
                env_define(&self.env, name, value.clone());
                self.emit("declare-item", *line, &reads, Some((name, &value)), None);
            }

            Node::Assign { name, expr, line } => {
                let reads = self.reads_for(Some(expr));
                let value = self.eval_expr(expr)?;
                // Defines when the name is unbound. This is why there is no
                // reachable "assignment to undefined name".
                if env_has(&self.env, name) {
                    env_assign(&self.env, name, value.clone());
                } else {
                    env_define(&self.env, name, value.clone());
                }
                self.emit("assign", *line, &reads, Some((name, &value)), None);
            }

            Node::Point { name, expr, line } => {
                let reads = self.reads_for(Some(expr));
                let raw = self.eval_expr(expr)?;
                let Value::Map(content) = raw else {
                    return Err(RuntimeErr::new("point requires a map literal", *line).into());
                };
                // certainty, then confidence, then 1.0 -- and clamped rather
                // than rejected, so `certainty: 5` is 1.0 and not an error.
                let conf = content
                    .get("certainty")
                    .or_else(|| content.get("confidence"))
                    .cloned();
                let conf = match conf {
                    Some(v) => clamp(num_of(&v, *line)?),
                    None => 1.0,
                };
                let pt = Rc::new(Point {
                    name: name.clone(),
                    content,
                    confidence: conf,
                });
                let value = Value::Point(Rc::clone(&pt));
                env_define(&self.env, name, value.clone());
                upsert(&mut self.points, name, pt);
                self.emit("declare-point", *line, &reads, Some((name, &value)), None);
            }

            Node::Hypothesis { name, body, line } => {
                // Writes the name *as a string value*, not the block.
                let as_value = Value::Str(name.clone());
                self.emit(
                    "declare-hypothesis",
                    *line,
                    &[],
                    Some((name, &as_value)),
                    None,
                );
                self.exec_block(body)?;
            }

            Node::Proposition {
                name,
                motions,
                body,
                line,
            } => self.exec_proposition(name, motions, body, *line)?,

            Node::Motion { name, text, line } => {
                let motion = Value::Motion(Rc::new(Motion {
                    name: name.clone(),
                    text: text.clone(),
                }));
                env_define(&self.env, name, motion.clone());
                self.emit("declare-motion", *line, &[], Some((name, &motion)), None);
            }

            Node::Support { motion, conf, line } => {
                self.verdict("support", motion, conf.as_deref(), *line)?
            }
            Node::Contradict { motion, conf, line } => {
                self.verdict("contradict", motion, conf.as_deref(), *line)?
            }
            Node::Inconclusive { motion, line } => {
                self.verdict("inconclusive", motion, None, *line)?
            }

            Node::Given {
                cond,
                then,
                otherwise,
                line,
            } => {
                let reads = self.reads_for(Some(cond));
                let c = self.eval_expr(cond)?.truthy();
                let note = if c { "then" } else { "otherwise" };
                self.emit("given", *line, &reads, None, Some(note.into()));
                self.exec_block(if c { then } else { otherwise })?;
            }

            Node::Within { target, body, line } => {
                let reads = self.reads_for(Some(target));
                let t = self.eval_expr(target)?;
                self.emit("within", *line, &reads, None, None);
                let inner = Env::child(&self.env);
                // A map binds its keys; anything else binds nothing, and the
                // scope is opened either way.
                if let Value::Map(m) = &t {
                    for (k, v) in m.iter() {
                        env_define(&inner, k, v.clone());
                    }
                }
                let saved = std::mem::replace(&mut self.env, inner);
                let r = self.exec_block(body);
                self.env = saved;
                r?;
            }

            Node::Considering {
                quant,
                var,
                iter,
                body,
                line,
            } => self.iterate("considering", Some(quant), var, iter, body, *line)?,

            Node::ForEach {
                var,
                iter,
                body,
                line,
            } => self.iterate("for-each", None, var, iter, body, *line)?,

            Node::While { cond, body, line } => {
                let mut rounds: u64 = 0;
                // The condition is evaluated before the event is emitted, and
                // the exit test happens once more than the body runs.
                while self.eval_expr(cond)?.truthy() {
                    rounds += 1;
                    if rounds > MAX_LOOP {
                        return Err(RuntimeErr::new("loop limit exceeded", *line).into());
                    }
                    let reads = self.reads_for(Some(cond));
                    self.emit("while", *line, &reads, None, Some(format!("round {}", rounds)));
                    self.exec_block(body)?;
                }
                self.emit(
                    "while-exit",
                    *line,
                    &[],
                    None,
                    Some(format!("{} rounds", rounds)),
                );
            }

            Node::Return { expr, line } => {
                let reads = self.reads_for(expr.as_deref());
                let value = match expr {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::None,
                };
                self.emit("return", *line, &reads, None, Some(note_of(&value)));
                return Err(Flow::Return(value));
            }

            Node::Ensure { expr, line } => {
                let reads = self.reads_for(Some(expr));
                let ok = self.eval_expr(expr)?.truthy();
                let note = if ok { "held" } else { "failed" };
                // The event is emitted before the raise, so a failing ensure
                // still leaves its mark on the trace.
                self.emit("ensure", *line, &reads, None, Some(note.into()));
                if !ok {
                    return Err(RuntimeErr::new("ensure failed", *line).into());
                }
            }

            Node::Resolve { expr, line } => {
                let reads = self.reads_for(Some(expr));
                let value = self.eval_expr(expr)?;
                self.emit("resolve", *line, &reads, None, Some(note_of(&value)));
            }

            Node::ExprStmt { expr, line } => {
                let reads = self.reads_for(Some(expr));
                let value = self.eval_expr(expr)?;
                self.emit("expr", *line, &reads, None, Some(note_of(&value)));
            }

            other => {
                return Err(RuntimeErr::new(
                    format!("no rule for {}", stmt_kind(other)),
                    other.line(),
                )
                .into())
            }
        }
        Ok(())
    }

    fn exec_proposition(
        &mut self,
        name: &str,
        motions: &[Node],
        body: &[Node],
        line: usize,
    ) -> Res<()> {
        let mut state = PropositionState {
            name: name.to_string(),
            ..Default::default()
        };
        // Motions are declared and traced before the proposition itself, each
        // at its own line.
        for m in motions {
            let (mname, mtext, mline) = match m {
                Node::Motion { name, text, line } => (name, text, *line),
                _ => continue,
            };
            let motion = Motion {
                name: mname.clone(),
                text: mtext.clone(),
            };
            state.motions.push(motion.clone());
            let value = Value::Motion(Rc::new(motion));
            env_define(&self.env, mname, value.clone());
            self.emit("declare-motion", mline, &[], Some((mname, &value)), None);
        }

        let names: Vec<Value> = state
            .motions
            .iter()
            .map(|m| Value::Str(m.name.clone()))
            .collect();
        let idx = upsert_prop(&mut self.propositions, name, state);
        let listed = Value::List(names);
        self.emit("declare-proposition", line, &[], Some((name, &listed)), None);

        self.prop_stack.push(idx);
        let r = self.exec_block(body);
        self.prop_stack.pop();
        r?;

        // One `resolve-motion` per motion, all at the proposition's line.
        //
        // The reads are a dict comprehension keyed by motion name, so two
        // verdicts on one motion collapse to a single entry carrying the last
        // confidence. A `Vec` here is the obvious port and is wrong.
        let motion_names: Vec<String> = self.propositions[idx]
            .1
            .motions
            .iter()
            .map(|m| m.name.clone())
            .collect();
        for mname in motion_names {
            let mut reads: Vec<(String, Value)> = Vec::new();
            for v in &self.propositions[idx].1.verdicts {
                if v.motion == mname {
                    let entry = (v.motion.clone(), Value::Num(v.confidence));
                    match reads.iter_mut().find(|(k, _)| *k == v.motion) {
                        Some(slot) => slot.1 = entry.1,
                        None => reads.push(entry),
                    }
                }
            }
            let score = Value::Num(self.propositions[idx].1.score(&mname));
            let item = format!("{}.{}", name, mname);
            self.emit("resolve-motion", line, &reads, Some((&item, &score)), None);
        }
        Ok(())
    }

    fn verdict(
        &mut self,
        stance: &str,
        motion: &str,
        conf: Option<&Node>,
        line: usize,
    ) -> Res<()> {
        let mut confidence = 1.0;
        let mut reads: Vec<(String, Value)> = Vec::new();
        if let Some(c) = conf {
            reads = self.reads_for(Some(c));
            let v = self.eval_expr(c)?;
            confidence = clamp(num_of(&v, line)?);
        }
        if stance == "inconclusive" {
            // Only ever zeroing the 1.0 default: `parse_inconclusive` takes no
            // confidence, so this branch cannot see a written one.
            confidence = 0.0;
        }
        // The check comes *after* the confidence is evaluated, so a verdict
        // outside a proposition still evaluates (and can fail inside) its
        // confidence expression first.
        let Some(&idx) = self.prop_stack.last() else {
            return Err(RuntimeErr::new(format!("{} outside a proposition", stance), line).into());
        };
        let pname = self.propositions[idx].1.name.clone();
        self.propositions[idx].1.verdicts.push(Verdict {
            motion: motion.to_string(),
            stance: stance.to_string(),
            confidence,
            line,
        });
        let item = format!("{}.{}", pname, motion);
        let value = Value::Num(confidence);
        self.emit(stance, line, &reads, Some((&item, &value)), None);
        Ok(())
    }

    fn iterate(
        &mut self,
        rule: &str,
        quant: Option<&str>,
        var: &str,
        iter: &Node,
        body: &[Node],
        line: usize,
    ) -> Res<()> {
        let reads = self.reads_for(Some(iter));
        let seq = self.eval_expr(iter)?;
        // A non-list is wrapped in a singleton rather than rejected.
        let mut items = match seq {
            Value::List(l) => l,
            other => vec![other],
        };
        if quant == Some("item") {
            items.truncate(1);
        }
        self.emit(
            rule,
            line,
            &reads,
            None,
            Some(format!("{} iterations", items.len())),
        );
        for value in items {
            let inner = Env::child(&self.env);
            env_define(&inner, var, value.clone());
            let saved = std::mem::replace(&mut self.env, inner);
            // The `bind` event is emitted from inside the loop scope.
            self.emit("bind", line, &[], Some((var, &value)), None);
            let r = self.exec_block(body);
            self.env = saved;
            r?;
        }
        Ok(())
    }

    // -- expressions ------------------------------------------------------

    pub fn eval_expr(&mut self, e: &Node) -> Res<Value> {
        self.tick(e.line())?;
        match e {
            Node::Num { value, .. } => Ok(Value::Num(*value)),
            Node::Str { value, .. } => Ok(Value::Str(value.clone())),
            Node::Bool { value, .. } => Ok(Value::Bool(*value)),
            Node::NoneLit { .. } => Ok(Value::None),
            Node::Var { name, line } => Ok(env_get(&self.env, name, *line)?),

            Node::List { items, .. } => {
                let mut out = Vec::with_capacity(items.len());
                for i in items {
                    out.push(self.eval_expr(i)?);
                }
                Ok(Value::List(out))
            }

            Node::Map { entries, .. } => {
                let mut m = OrderedMap::new();
                for Entry { key, value } in entries {
                    let v = self.eval_expr(value)?;
                    m.insert(key.clone(), v);
                }
                Ok(Value::Map(m))
            }

            Node::Unary { op, operand, line } => {
                let v = self.eval_expr(operand)?;
                if op == "not" {
                    Ok(Value::Bool(!v.truthy()))
                } else {
                    Ok(Value::Num(-num_of(&v, *line)?))
                }
            }

            Node::Binary {
                op,
                left,
                right,
                line,
            } => self.binary(op, left, right, *line),

            Node::Index {
                target,
                index,
                line,
            } => {
                let t = self.eval_expr(target)?;
                let i = self.eval_expr(index)?;
                match t {
                    Value::List(l) => {
                        // Truncation toward zero, as Python's `int()`.
                        let n = num_of(&i, *line)? as i64;
                        if n < 0 || n as usize >= l.len() {
                            return Err(
                                RuntimeErr::new(format!("index {} out of range", n), *line).into()
                            );
                        }
                        Ok(l[n as usize].clone())
                    }
                    // A missing key is `none`, not an error -- unlike a list.
                    Value::Map(m) => Ok(m.get(&text(&i)).cloned().unwrap_or(Value::None)),
                    _ => Err(RuntimeErr::new("value is not indexable", *line).into()),
                }
            }

            Node::Field {
                target,
                field,
                line,
            } => {
                let t = self.eval_expr(target)?;
                match &t {
                    Value::Point(p) => {
                        if field == "confidence" || field == "certainty" {
                            // The *clamped* confidence, not whatever the map
                            // held under that key.
                            Ok(Value::Num(p.confidence))
                        } else {
                            Ok(p.content.get(field).cloned().unwrap_or(Value::None))
                        }
                    }
                    Value::Motion(m) => Ok(match field.as_str() {
                        "name" => Value::Str(m.name.clone()),
                        "text" => Value::Str(m.text.clone()),
                        _ => Value::None,
                    }),
                    Value::Map(m) => Ok(m.get(field).cloned().unwrap_or(Value::None)),
                    other => Err(RuntimeErr::new(
                        format!("no field '{}' on {}", field, other.tag()),
                        *line,
                    )
                    .into()),
                }
            }

            Node::Call { callee, args, line } => {
                let c = self.eval_expr(callee)?;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval_expr(a)?);
                }
                self.apply(&c, vals, *line)
            }

            // `x |> f` is `f(x)` -- and the right operand is evaluated first.
            Node::Pipe { left, right, line } => {
                let f = self.eval_expr(right)?;
                let x = self.eval_expr(left)?;
                self.apply(&f, vec![x], *line)
            }

            other => Err(RuntimeErr::new(
                format!("no rule for expression {}", stmt_kind(other)),
                other.line(),
            )
            .into()),
        }
    }

    fn binary(&mut self, op: &str, left: &Node, right: &Node, line: usize) -> Res<Value> {
        // Short-circuit, and a **bool** result: `true and 5` is `true`, not 5.
        // The right operand is not evaluated when it cannot matter, so it emits
        // no events and an undefined name there raises nothing.
        if op == "and" {
            let a = self.eval_expr(left)?.truthy();
            if !a {
                return Ok(Value::Bool(false));
            }
            return Ok(Value::Bool(self.eval_expr(right)?.truthy()));
        }
        if op == "or" {
            let a = self.eval_expr(left)?.truthy();
            if a {
                return Ok(Value::Bool(true));
            }
            return Ok(Value::Bool(self.eval_expr(right)?.truthy()));
        }

        let a = self.eval_expr(left)?;
        let b = self.eval_expr(right)?;

        match op {
            // On rendered values, which is what makes `1 == true` true.
            "==" => Ok(Value::Bool(a.eq_rendered(&b))),
            "!=" => Ok(Value::Bool(!a.eq_rendered(&b))),
            "+" => {
                // Concatenation wins over addition when *either* side is text,
                // and the number is stringified through `_text`.
                if matches!(a, Value::Str(_)) || matches!(b, Value::Str(_)) {
                    return Ok(Value::Str(format!("{}{}", text(&a), text(&b))));
                }
                if let (Value::List(x), Value::List(y)) = (&a, &b) {
                    let mut out = x.clone();
                    out.extend(y.iter().cloned());
                    return Ok(Value::List(out));
                }
                Ok(Value::Num(num_of(&a, line)? + num_of(&b, line)?))
            }
            "-" => Ok(Value::Num(num_of(&a, line)? - num_of(&b, line)?)),
            "*" => Ok(Value::Num(num_of(&a, line)? * num_of(&b, line)?)),
            "/" => {
                let d = num_of(&b, line)?;
                if d == 0.0 {
                    return Err(RuntimeErr::new("division by zero", line).into());
                }
                Ok(Value::Num(num_of(&a, line)? / d))
            }
            "%" => {
                let d = num_of(&b, line)?;
                if d == 0.0 {
                    return Err(RuntimeErr::new("modulo by zero", line).into());
                }
                // `math.fmod`, which takes the sign of the dividend. Rust's `%`
                // on f64 is exactly that; `rem_euclid` is not.
                Ok(Value::Num(num_of(&a, line)? % d))
            }
            "<" | "<=" | ">" | ">=" => {
                let x = num_of(&a, line)?;
                let y = num_of(&b, line)?;
                Ok(Value::Bool(match op {
                    "<" => x < y,
                    "<=" => x <= y,
                    ">" => x > y,
                    _ => x >= y,
                }))
            }
            _ => Err(RuntimeErr::new(format!("unknown operator '{}'", op), line).into()),
        }
    }

    fn apply(&mut self, callee: &Value, args: Vec<Value>, line: usize) -> Res<Value> {
        match callee {
            Value::Builtin(b) => self.call_builtin(b.which, args, line),
            Value::Closure(c) => {
                if args.len() != c.params.len() {
                    return Err(RuntimeErr::new(
                        format!(
                            "{} expects {} argument(s), got {}",
                            c.name,
                            c.params.len(),
                            args.len()
                        ),
                        line,
                    )
                    .into());
                }
                // The child of the *defining* environment, not the caller's.
                let inner = Env::child(&c.env);
                for (p, a) in c.params.iter().zip(args) {
                    env_define(&inner, p, a);
                }
                let saved = std::mem::replace(&mut self.env, inner);
                let r = self.exec_block(&c.body);
                self.env = saved;
                match r {
                    // A function that falls off its end returns none.
                    Ok(()) => Ok(Value::None),
                    Err(Flow::Return(v)) => Ok(v),
                    Err(e) => Err(e),
                }
            }
            other => Err(
                RuntimeErr::new(format!("{} is not callable", other.tag()), line).into(),
            ),
        }
    }

    fn call_builtin(&mut self, which: BuiltinKind, args: Vec<Value>, line: usize) -> Res<Value> {
        use BuiltinKind::*;
        match which {
            Print => {
                let joined = args
                    .iter()
                    .map(text)
                    .collect::<Vec<_>>()
                    .join(" ");
                self.output.push(joined.clone());
                self.emit("print", line, &[], None, Some(joined));
                Ok(Value::None)
            }
            Len => {
                let v = args.first().cloned().unwrap_or(Value::None);
                match &v {
                    Value::List(l) => Ok(Value::Num(l.len() as f64)),
                    // Characters, not bytes: Python's `len` of a `str`.
                    Value::Str(s) => Ok(Value::Num(s.chars().count() as f64)),
                    Value::Map(m) => Ok(Value::Num(m.len() as f64)),
                    other => {
                        Err(RuntimeErr::new(format!("len of {}", other.tag()), line).into())
                    }
                }
            }
            // The rest coerce every argument first, so a non-number raises even
            // when it would not have been looked at.
            Sum | Min | Max | Abs | Round => {
                let mut xs = Vec::with_capacity(args.len());
                for a in &args {
                    xs.push(num_of(a, line)?);
                }
                Ok(Value::Num(match which {
                    Sum => xs.iter().sum(),
                    // An empty call is 0.0, not an error.
                    Min => xs.iter().copied().fold(f64::INFINITY, f64::min),
                    Max => xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    Abs => xs.first().map(|x| x.abs()).unwrap_or(0.0),
                    _ => xs.first().map(|x| banker_round(*x)).unwrap_or(0.0),
                }))
                .map(|v| match (which, args.is_empty()) {
                    (Min | Max, true) => Value::Num(0.0),
                    _ => v,
                })
            }
        }
    }
}

/// Python's `round`: half to even, not half away from zero.
///
/// `round(2.5)` is 2 and `round(3.5)` is 4. `f64::round` gives 3 for the first,
/// so this is the one arithmetic builtin that cannot use the obvious call.
fn banker_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

/// A value as a number, for the arithmetic and comparison rules.
///
/// A bool is 1.0 / 0.0 and a point is its confidence; everything else raises.
fn num_of(v: &Value, line: usize) -> Result<f64, RuntimeErr> {
    match v {
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Num(n) => Ok(*n),
        Value::Point(p) => Ok(p.confidence),
        other => Err(RuntimeErr::new(
            format!("expected a number, found {}", other.tag()),
            line,
        )),
    }
}

/// The `kind` string a node serialises under, for the two "no rule" messages.
fn stmt_kind(n: &Node) -> String {
    serde_json::to_value(n)
        .ok()
        .and_then(|v| v.get("kind").and_then(|k| k.as_str().map(String::from)))
        .unwrap_or_default()
}

fn upsert(list: &mut Vec<(String, Rc<Point>)>, name: &str, value: Rc<Point>) {
    match list.iter_mut().find(|(k, _)| k == name) {
        Some(slot) => slot.1 = value,
        None => list.push((name.to_string(), value)),
    }
}

fn upsert_prop(list: &mut Vec<(String, PropositionState)>, name: &str, v: PropositionState) -> usize {
    match list.iter().position(|(k, _)| k == name) {
        Some(i) => {
            list[i].1 = v;
            i
        }
        None => {
            list.push((name.to_string(), v));
            list.len() - 1
        }
    }
}

/// The store as the dump renders it: name to rendered value, in order.
pub fn render_store(env: &EnvRef) -> Vec<(String, Json)> {
    env_flatten(env)
        .iter()
        .map(|(k, v)| (k.clone(), render(v)))
        .collect()
}
