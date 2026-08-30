//! Runtime values, and the two renderings the trace is made of.
//!
//! A port of the value types in `prototype/evaluator.py`. Three of the
//! decisions here look wrong in isolation and are not; each reproduces a
//! behaviour of the Python that a Turbulance script can observe.
//!
//! * **Maps are insertion-ordered.** `{z: 1, a: 2}` renders, prints and
//!   serialises as `z` then `a`. A `BTreeMap` would sort it, and nothing that
//!   counts or measures would notice -- only the text.
//!
//! * **`Bool` and `Num` compare equal across the tags.** Python's `bool` is a
//!   subclass of `int`, so `render(True) == render(1.0)` and `1 == true` is
//!   true in Turbulance. [`Value::eq_rendered`] crosses the tags deliberately;
//!   `PartialEq` on the enum does not, and is not what `==` uses.
//!
//! * **[`render`] collapses an integral float to an integer**, so a runtime
//!   `1.0` serialises as `1`. This is the *opposite* of the AST, where
//!   `Num.value` keeps its `.0`. One rule for both sides would be wrong twice.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::rc::Rc;

use serde_json::{json, Map as JsonMap, Value as Json};

use crate::ast::Node;

/// An insertion-ordered map, as Python's `dict` is.
///
/// Small and linear on purpose: a Turbulance map is a literal a person wrote,
/// so it holds a handful of keys and the order is the part that matters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderedMap {
    entries: Vec<(String, Value)>,
}

impl OrderedMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert, replacing in place so a repeated key keeps its first position --
    /// which is what Python's `dict` does.
    pub fn insert(&mut self, key: String, value: Value) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.entries.push((key, value)),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

/// A user-defined function, closed over the environment it was defined in.
#[derive(Debug, Clone, PartialEq)]
pub struct Closure {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Node>,
    /// The *defining* environment, not the calling one.
    pub env: crate::eval::EnvRef,
}

/// A point: a map, plus the confidence read out of it at declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub name: String,
    pub content: OrderedMap,
    pub confidence: f64,
}

/// A motion: a named claim a proposition can carry verdicts about.
#[derive(Debug, Clone, PartialEq)]
pub struct Motion {
    pub name: String,
    pub text: String,
}

/// A built-in function.
///
/// The `name` is what [`render`] emits, and it is *not* the name the builtin is
/// bound to. In the Python the five numeric builtins are all the same closure
/// `f`, produced by `_num_of`, so `__name__` is `"f"` for `sum`, `min`, `max`,
/// `abs` and `round` alike -- they are indistinguishable once rendered. That
/// indistinguishability is observable, so it is reproduced rather than fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Builtin {
    pub which: BuiltinKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinKind {
    Print,
    Len,
    Sum,
    Min,
    Max,
    Abs,
    Round,
}

impl BuiltinKind {
    /// The rendered name: `_print`, `_len`, or `f` for everything `_num_of`
    /// wrapped.
    pub fn render_name(self) -> &'static str {
        match self {
            BuiltinKind::Print => "_print",
            BuiltinKind::Len => "_len",
            _ => "f",
        }
    }
}

/// A runtime value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    /// Every number is a float, as in the Python: there is no integer type.
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Map(OrderedMap),
    Point(Rc<Point>),
    Motion(Rc<Motion>),
    Closure(Rc<Closure>),
    Builtin(Builtin),
}

impl Value {
    /// The tag written into a trace event's `writes`.
    ///
    /// `bool` is checked before `num` because in the Python a `bool` *is* an
    /// `int`, so the order of the `isinstance` tests is what separates them.
    pub fn tag(&self) -> &'static str {
        match self {
            Value::None => "none",
            Value::Bool(_) => "bool",
            Value::Num(_) => "num",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Point(_) => "point",
            Value::Motion(_) => "motion",
            Value::Closure(_) => "closure",
            Value::Builtin(_) => "builtin",
            Value::Map(_) => "map",
        }
    }

    /// Truthiness, by tag rather than by value.
    ///
    /// An empty string, list or map is false; a point is true when its
    /// confidence exceeds zero; a closure or builtin is always true.
    pub fn truthy(&self) -> bool {
        match self {
            Value::None => false,
            Value::Bool(b) => *b,
            Value::Num(n) => *n != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
            Value::Point(p) => p.confidence > 0.0,
            _ => true,
        }
    }

    /// Equality as `==` and `!=` use it: on the *rendered* values, compared
    /// the way Python compares them.
    ///
    /// Two separate things had to be got right here, and getting one of them
    /// right looks like getting both.
    ///
    /// The values compared are the **rendered** ones, so `1.0 == 1` and a map
    /// equals a map of equal renderings. That much `render(a) == render(b)`
    /// gives.
    ///
    /// But `render` leaves a bool a bool -- it falls through to `return v` --
    /// and Python's `bool` subclasses `int`, so `True == 1`. `serde_json`'s
    /// `PartialEq` has no such coercion and calls them different. Hence
    /// [`json_eq`], which is `==` on JSON *as Python would perform it*.
    pub fn eq_rendered(&self, other: &Value) -> bool {
        json_eq(&render(self), &render(other))
    }
}

/// `==` on two rendered values, as Python performs it.
///
/// Identical to `PartialEq` except across the boolean/number boundary, where
/// Python's `bool` is an `int`: `True == 1` and `False == 0`. The coercion has
/// to recurse, since a list or map of booleans is compared elementwise by the
/// same rule -- `[true] == [1]` is true in Turbulance for exactly this reason.
///
/// A string is never numeric here: `"1" == 1` is false, as it is in Python.
pub fn json_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Bool(x), Json::Number(_)) => num_of_json(b) == Some(if *x { 1.0 } else { 0.0 }),
        (Json::Number(_), Json::Bool(y)) => num_of_json(a) == Some(if *y { 1.0 } else { 0.0 }),
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Json::Object(x), Json::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).map(|w| json_eq(v, w)).unwrap_or(false))
        }
        _ => a == b,
    }
}

fn num_of_json(v: &Json) -> Option<f64> {
    v.as_f64()
}

/// A JSON-safe rendering of a runtime value.
///
/// An integral float becomes an integer here, which is why `1 == true`: both
/// render to `1`. Note the consequence for equality of a whole map or list --
/// `[1]` and `[true]` render alike and so compare equal.
pub fn render(v: &Value) -> Json {
    match v {
        Value::None => Json::Null,
        Value::Bool(b) => Json::Bool(*b),
        Value::Num(n) => render_num(*n),
        Value::Str(s) => Json::String(s.clone()),
        Value::List(l) => Json::Array(l.iter().map(render).collect()),
        Value::Map(m) => {
            let mut out = JsonMap::new();
            for (k, val) in m.iter() {
                out.insert(k.clone(), render(val));
            }
            Json::Object(out)
        }
        Value::Point(p) => {
            let mut content = JsonMap::new();
            for (k, val) in p.content.iter() {
                content.insert(k.clone(), render(val));
            }
            json!({
                "point": p.name,
                "content": Json::Object(content),
                // Raw, not `render_num`: `render` builds this dict by hand
                // and never recurses on the confidence, so a point of 1.0
                // serialises as `1.0` where every other number would be `1`.
                "confidence": p.confidence,
            })
        }
        Value::Motion(m) => json!({"motion": m.name, "text": m.text}),
        Value::Closure(c) => json!({"closure": c.name, "params": c.params}),
        Value::Builtin(b) => json!({"builtin": b.which.render_name()}),
    }
}

/// A number as `render` writes it: an integer when it is one.
///
/// The Python's `float.is_integer()` is false for the infinities and for NaN,
/// which then reach `json.dumps` as bare `Infinity` / `NaN` -- invalid JSON
/// that a reader would reject. `serde_json` has no such token, so those become
/// null here. The divergence is unreachable from any script the corpus admits
/// (division by zero raises rather than producing an infinity), and the
/// alternative is emitting a document that is not JSON.
fn render_num(n: f64) -> Json {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e17 {
        return json!(n as i64);
    }
    if !n.is_finite() {
        return Json::Null;
    }
    json!(n)
}

/// The text form: what `print` writes and what a `note` carries.
///
/// Floats go through Python's `%g` -- six significant digits, trailing zeros
/// stripped, exponent form outside that range -- *except* that an integral
/// float is printed through `int()` first, so `1e21` comes out in full digits
/// rather than as `1e+21`.
pub fn text(v: &Value) -> String {
    match v {
        Value::None => "none".into(),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Num(n) => text_num(*n),
        Value::Str(s) => s.clone(),
        Value::Point(p) => format!("point({}, conf={})", p.name, format_g(p.confidence)),
        Value::Motion(m) => format!("motion({})", m.name),
        Value::Closure(c) => format!("closure({})", c.name),
        Value::Builtin(b) => format!("builtin({})", b.which.render_name()),
        Value::List(l) => {
            let mut out = String::from("[");
            for (i, x) in l.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&text(x));
            }
            out.push(']');
            out
        }
        Value::Map(m) => {
            let mut out = String::from("{");
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{}: {}", k, text(val));
            }
            out.push('}');
            out
        }
    }
}

fn text_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 {
        // `str(int(v))`: full digits, however large.
        return format_integral(n);
    }
    format_g(n)
}

/// `str(int(v))` for an integral float, without going through `i64`, which
/// would overflow on the large ones Python prints in full.
fn format_integral(n: f64) -> String {
    if n.abs() < 1e17 {
        return format!("{}", n as i64);
    }
    // Beyond i64's exact range, `{:.0}` gives the same digits `int()` does:
    // both print the float's exact integer value.
    format!("{:.0}", n)
}

/// Python's `%g` with the default precision of six significant digits.
///
/// Rust has no `%g`, so this is the rule spelled out: exponent form when the
/// exponent is below -4 or at least the precision, fixed form otherwise, with
/// trailing zeros and a trailing point stripped from either.
pub fn format_g(n: f64) -> String {
    if n == 0.0 {
        return "0".into();
    }
    if n.is_nan() {
        return "nan".into();
    }
    if n.is_infinite() {
        return if n < 0.0 { "-inf".into() } else { "inf".into() };
    }

    const P: i32 = 6;
    let exp = n.abs().log10().floor() as i32;
    // Round to P significant digits first: the exponent decides the format, and
    // rounding can carry it (9.9999995e2 formats as 1000, not 1e+03).
    let exp = {
        let scaled = format!("{:.*e}", (P - 1) as usize, n);
        scaled
            .split('e')
            .nth(1)
            .and_then(|e| e.parse::<i32>().ok())
            .unwrap_or(exp)
    };

    if exp < -4 || exp >= P {
        let s = format!("{:.*e}", (P - 1) as usize, n);
        let (mant, e) = s.split_once('e').unwrap_or((s.as_str(), "0"));
        let mant = strip_zeros(mant);
        let ev: i32 = e.parse().unwrap_or(0);
        format!("{}e{}{:02}", mant, if ev < 0 { '-' } else { '+' }, ev.abs())
    } else {
        let decimals = (P - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{:.*}", decimals, n))
    }
}

fn strip_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.into();
    }
    s.trim_end_matches('0').trim_end_matches('.').into()
}

/// Names occurring free in an expression, sorted.
///
/// A thin wrapper over [`crate::ast::free_names`], returning the sorted set the
/// Python's `sorted(free_names(expr))` produces -- the order is what fixes the
/// order of a trace event's `reads`.
pub fn free_names_sorted(expr: &Node) -> Vec<String> {
    let mut set = BTreeSet::new();
    crate::ast::free_names(expr, &mut set);
    set.into_iter().collect()
}

/// Python's `str()` of a rendered value: what an `expr`, `resolve` or `return`
/// event carries as its note.
///
/// Not [`text`]. The Python writes `str(render(value))`, so the note is the
/// *repr* of a JSON document rather than the language's own text form: `none`
/// prints as `None`, a boolean as `True`/`False`, and a map with single-quoted
/// keys. Using `_text` here would be the natural reading and would disagree on
/// every one of those.
pub fn repr_json(v: &Json) -> String {
    match v {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Number(n) => n.to_string(),
        Json::String(s) => {
            let esc = s.replace('\\', "\\\\")
                        .replace('\'', "\'");
            format!("'{}'", esc)
        }
        Json::Array(a) => {
            let inner: Vec<String> = a.iter().map(repr_json).collect();
            format!("[{}]", inner.join(", "))
        }
        Json::Object(o) => {
            let inner: Vec<String> = o
                .iter()
                .map(|(k, val)| format!("'{}': {}", k, repr_json(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// The note of a top-level `expr`/`resolve`/`return`, where a bare string is
/// *not* quoted -- `str("text")` is `text`, since `str` of a `str` is itself.
pub fn note_of(v: &Value) -> String {
    let rendered = render(v);
    match &rendered {
        Json::String(s) => s.clone(),
        other => repr_json(other),
    }
}
