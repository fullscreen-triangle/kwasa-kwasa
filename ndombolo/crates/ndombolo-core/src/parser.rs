//! Turbulance parser.
//!
//! A port of `prototype/parser.py`: statements dispatched on a leading keyword,
//! expressions through a six-level precedence cascade with a Pratt-style binary
//! loop. The Python builds plain dicts; here the same trees are [`Node`] values
//! whose serialisation reproduces those dicts key for key.
//!
//! Two places where the Python's behaviour is easy to improve on by accident,
//! and where doing so would be a port bug rather than a fix:
//!
//! * **The dispatch is `getattr(self, f"parse_{t.value}")`.** A keyword with no
//!   method of that name is not an error and is not loose -- it falls through to
//!   `parse_expr_statement`, where `parse_primary` rejects it. So `from`, `as`,
//!   `each`, `in`, `otherwise`, `with_confidence`, `until`, `settled`, `over`,
//!   `on`, `cycle`, `drift`, `flow`, `roll`, `and`, `or`, `not` are hard errors
//!   in statement position. [`is_statement_keyword`] pins that set, and a test
//!   walks every keyword to check the Rust agrees.
//!
//! * **Operator nodes carry the operator's line**, not the left operand's, and
//!   a collection carries its bracket's. Since `line` is what a trace site
//!   names, a node built with the wrong one moves a reported site.

use crate::ast::{Entry, Node};
use crate::lexer::{tokenize, Kind, LexError, Token, LOOSE, QUANTS};

/// Lowest binding first. `|>` is handled above the cascade, in [`Parser::pipe`].
const BINARY_LEVELS: [&[&str]; 6] = [
    &["or", "||"],
    &["and", "&&"],
    &["==", "!="],
    &["<", "<=", ">", ">="],
    &["+", "-"],
    &["*", "/", "%"],
];

/// Keywords that name a statement handler.
///
/// The Python has no such list -- it looks for a method called `parse_<kw>` and
/// falls through when there is none. Writing the set out is the only way to
/// port that faithfully, so it is stated once here and checked against the
/// lexer's keyword set by a test rather than trusted.
fn is_statement_keyword(kw: &str) -> bool {
    matches!(
        kw,
        "funxn"
            | "item"
            | "proposition"
            | "hypothesis"
            | "motion"
            | "support"
            | "contradict"
            | "inconclusive"
            | "within"
            | "given"
            | "considering"
            | "for"
            | "while"
            | "return"
            | "ensure"
            | "point"
            | "resolve"
            | "import"
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        ParseError {
            message: e.message,
            line: e.line,
        }
    }
}

type PResult<T> = Result<T, ParseError>;

pub struct Parser {
    toks: Vec<Token>,
    i: usize,
}

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Parser { toks, i: 0 }
    }

    // -- token plumbing ---------------------------------------------------

    fn peek_at(&self, ahead: usize) -> &Token {
        let j = (self.i + ahead).min(self.toks.len() - 1);
        &self.toks[j]
    }

    fn peek(&self) -> &Token {
        self.peek_at(0)
    }

    /// Does **not** advance past `eof`. That is what stops the parser spinning
    /// on a truncated script: every loop that tests `at(Eof)` stays true.
    fn next(&mut self) -> Token {
        let t = self.toks[self.i].clone();
        if t.kind != Kind::Eof {
            self.i += 1;
        }
        t
    }

    fn at(&self, kind: Kind) -> bool {
        self.peek().kind == kind
    }

    fn at_val(&self, kind: Kind, value: &str) -> bool {
        let t = self.peek();
        t.kind == kind && t.value == value
    }

    fn at_op(&self, op: &str) -> bool {
        self.at_val(Kind::Op, op)
    }

    fn accept(&mut self, kind: Kind) -> bool {
        if self.at(kind) {
            self.next();
            true
        } else {
            false
        }
    }

    fn accept_op(&mut self, op: &str) -> bool {
        if self.at_op(op) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: Kind) -> PResult<Token> {
        if self.at(kind) {
            return Ok(self.next());
        }
        let t = self.peek();
        Err(self.unexpected(kind.as_str(), t))
    }

    fn expect_val(&mut self, kind: Kind, value: &str) -> PResult<Token> {
        if self.at_val(kind, value) {
            return Ok(self.next());
        }
        let t = self.peek();
        Err(self.unexpected(value, t))
    }

    fn expect_op(&mut self, op: &str) -> PResult<Token> {
        self.expect_val(Kind::Op, op)
    }

    /// The Python's `expected {want!r}, found {t.value or t.kind!r}`: an empty
    /// value falls back to the kind name, which is how a missing token reads as
    /// `'newline'` rather than as an empty string.
    fn unexpected(&self, want: &str, found: &Token) -> ParseError {
        let found_desc = if found.value.is_empty() {
            found.kind.as_str().to_string()
        } else {
            found.value.clone()
        };
        ParseError {
            message: format!("expected '{}', found '{}'", want, found_desc),
            line: found.line,
        }
    }

    fn skip_newlines(&mut self) {
        while self.at(Kind::Newline) {
            self.next();
        }
    }

    // -- program ----------------------------------------------------------

    pub fn parse_program(&mut self) -> PResult<Vec<Node>> {
        let mut body = Vec::new();
        self.skip_newlines();
        while !self.at(Kind::Eof) {
            body.push(self.statement()?);
            self.skip_newlines();
        }
        Ok(body)
    }

    /// A `:`-introduced indented block, or a single inline statement.
    fn block(&mut self) -> PResult<Vec<Node>> {
        self.expect_op(":")?;
        if self.accept(Kind::Newline) {
            self.skip_newlines();
            self.expect(Kind::Indent)?;
            let mut body = Vec::new();
            self.skip_newlines();
            while !self.at(Kind::Dedent) && !self.at(Kind::Eof) {
                body.push(self.statement()?);
                self.skip_newlines();
            }
            self.accept(Kind::Dedent);
            return Ok(body);
        }
        Ok(vec![self.statement()?])
    }

    // -- statements -------------------------------------------------------

    fn statement(&mut self) -> PResult<Node> {
        let t = self.peek();
        if t.kind == Kind::Kw {
            let kw = t.value.clone();
            if LOOSE.contains(&kw.as_str()) {
                return self.loose();
            }
            if is_statement_keyword(&kw) {
                return match kw.as_str() {
                    "funxn" => self.funxn(),
                    "item" => self.item(),
                    "proposition" => self.proposition(),
                    "hypothesis" => self.hypothesis(),
                    "motion" => self.motion(),
                    "support" => self.verdict("Support"),
                    "contradict" => self.verdict("Contradict"),
                    "inconclusive" => self.inconclusive(),
                    "within" => self.within(),
                    "given" => self.given(),
                    "considering" => self.considering(),
                    "for" => self.for_each(),
                    "while" => self.while_(),
                    "return" => self.return_(),
                    "ensure" => self.ensure(),
                    "point" => self.point(),
                    "resolve" => self.resolve(),
                    "import" => self.import(),
                    _ => unreachable!("is_statement_keyword and dispatch disagree"),
                };
            }
        }
        self.expr_statement()
    }

    fn funxn(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        self.expect_op("(")?;
        let mut params = Vec::new();
        while !self.at_op(")") {
            params.push(self.expect(Kind::Ident)?.value);
            if !self.accept_op(",") {
                break;
            }
        }
        self.expect_op(")")?;
        let body = self.block()?;
        Ok(Node::Funxn {
            name,
            params,
            body,
            line,
        })
    }

    fn item(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        self.expect_op("=")?;
        let expr = Box::new(self.expr()?);
        Ok(Node::Item { name, expr, line })
    }

    /// Motions are lifted out of the block into their own list: the one
    /// statement whose body does not appear verbatim.
    fn proposition(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        let body = self.block()?;
        let (motions, rest): (Vec<Node>, Vec<Node>) = body
            .into_iter()
            .partition(|s| matches!(s, Node::Motion { .. }));
        Ok(Node::Proposition {
            name,
            motions,
            body: rest,
            line,
        })
    }

    fn hypothesis(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        let body = self.block()?;
        Ok(Node::Hypothesis { name, body, line })
    }

    fn motion(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        let mut text = String::new();
        if self.accept_op("(") {
            if self.at(Kind::Str) {
                text = self.next().value;
            }
            self.expect_op(")")?;
        }
        Ok(Node::Motion { name, text, line })
    }

    fn verdict(&mut self, kind: &str) -> PResult<Node> {
        let line = self.next().line;
        let motion = self.expect(Kind::Ident)?.value;
        let mut conf = None;
        if self.at_val(Kind::Kw, "with_confidence") {
            self.next();
            self.expect_op("(")?;
            conf = Some(Box::new(self.expr()?));
            self.expect_op(")")?;
        }
        Ok(match kind {
            "Support" => Node::Support { motion, conf, line },
            _ => Node::Contradict { motion, conf, line },
        })
    }

    fn inconclusive(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let motion = self.expect(Kind::Ident)?.value;
        Ok(Node::Inconclusive { motion, line })
    }

    fn given(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let cond = Box::new(self.expr()?);
        let then = self.block()?;
        let mut otherwise = Vec::new();
        // The newlines before `otherwise` have to be consumed to see it, and
        // put back when it is not there -- a blank line after a `given` block
        // otherwise gets eaten and the following statement mis-parsed.
        let save = self.i;
        self.skip_newlines();
        if self.at_val(Kind::Kw, "otherwise") {
            self.next();
            otherwise = self.block()?;
        } else {
            self.i = save;
        }
        Ok(Node::Given {
            cond,
            then,
            otherwise,
            line,
        })
    }

    fn within(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let target = Box::new(self.expr()?);
        let body = self.block()?;
        Ok(Node::Within { target, body, line })
    }

    fn considering(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let mut quant = "item".to_string();
        // Three ways in: an explicit quantifier (as either an ident or a kw --
        // neither `all` nor `these` is a keyword, but the Python tests both),
        // the literal `item` keyword, or nothing.
        let t = self.peek();
        if (t.kind == Kind::Ident || t.kind == Kind::Kw) && QUANTS.contains(&t.value.as_str()) {
            quant = self.next().value;
        } else if self.at_val(Kind::Kw, "item") {
            self.next();
        }
        let var = self.expect(Kind::Ident)?.value;
        self.expect_val(Kind::Kw, "in")?;
        let iter = Box::new(self.expr()?);
        let body = self.block()?;
        Ok(Node::Considering {
            quant,
            var,
            iter,
            body,
            line,
        })
    }

    fn for_each(&mut self) -> PResult<Node> {
        let line = self.next().line;
        self.expect_val(Kind::Kw, "each")?;
        let var = self.expect(Kind::Ident)?.value;
        self.expect_val(Kind::Kw, "in")?;
        let iter = Box::new(self.expr()?);
        let body = self.block()?;
        Ok(Node::ForEach {
            var,
            iter,
            body,
            line,
        })
    }

    fn while_(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let cond = Box::new(self.expr()?);
        let body = self.block()?;
        Ok(Node::While { cond, body, line })
    }

    /// A bare `return` is only bare at a newline. At `eof` the Python parses an
    /// expression and fails, so a script ending in `return` is an error.
    fn return_(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let expr = if self.at(Kind::Newline) {
            None
        } else {
            Some(Box::new(self.expr()?))
        };
        Ok(Node::Return { expr, line })
    }

    fn ensure(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let expr = Box::new(self.expr()?);
        Ok(Node::Ensure { expr, line })
    }

    fn point(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let name = self.expect(Kind::Ident)?.value;
        self.expect_op("=")?;
        let expr = Box::new(self.expr()?);
        Ok(Node::Point { name, expr, line })
    }

    fn resolve(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let expr = Box::new(self.expr()?);
        Ok(Node::Resolve { expr, line })
    }

    /// Consumes to end of line. The resulting `Noop` has **no** `body` key and
    /// its `text` excludes the `import` itself -- both unlike [`Parser::loose`].
    fn import(&mut self) -> PResult<Node> {
        let line = self.next().line;
        let mut parts = Vec::new();
        while !self.at(Kind::Newline) && !self.at(Kind::Eof) {
            parts.push(self.next().value);
        }
        Ok(Node::Noop {
            text: parts.join(" "),
            body: None,
            line,
        })
    }

    /// A keyword recognised by the grammar with no deterministic meaning.
    ///
    /// Its `text` *includes* the keyword and it always has a `body` (empty when
    /// it opened no block). The scan tracks bracket depth so a `:` inside
    /// brackets does not open one.
    fn loose(&mut self) -> PResult<Node> {
        let t = self.next();
        let line = t.line;
        let mut parts = vec![t.value];
        let mut depth: i32 = 0;
        while !self.at(Kind::Eof) {
            if depth == 0 && (self.at(Kind::Newline) || self.at_op(":")) {
                break;
            }
            let tok = self.next();
            depth += tok.is_op(&["(", "[", "{"]) as i32 - tok.is_op(&[")", "]", "}"]) as i32;
            parts.push(tok.value);
        }
        let body = if self.at_op(":") {
            self.block()?
        } else {
            Vec::new()
        };
        Ok(Node::Noop {
            text: parts.join(" "),
            body: Some(body),
            line,
        })
    }

    fn expr_statement(&mut self) -> PResult<Node> {
        let line = self.peek().line;
        // Assignment is recognised only for a bare name, on one token of
        // lookahead. There is no field-assignment form: `a.b = 3` parses `a.b`
        // and then chokes on the `=`.
        if self.peek().kind == Kind::Ident && self.peek_at(1).is_op(&["="]) {
            let name = self.next().value;
            self.next();
            let expr = Box::new(self.expr()?);
            return Ok(Node::Assign { name, expr, line });
        }
        let expr = Box::new(self.expr()?);
        Ok(Node::ExprStmt { expr, line })
    }

    // -- expressions ------------------------------------------------------

    fn expr(&mut self) -> PResult<Node> {
        self.pipe()
    }

    /// Above the whole binary cascade, so `1 + 2 |> f` pipes the sum.
    fn pipe(&mut self) -> PResult<Node> {
        let mut left = self.binary(0)?;
        while self.at_op("|>") {
            let line = self.next().line;
            let right = self.binary(0)?;
            left = Node::Pipe {
                left: Box::new(left),
                right: Box::new(right),
                line,
            };
        }
        Ok(left)
    }

    fn binary(&mut self, level: usize) -> PResult<Node> {
        if level >= BINARY_LEVELS.len() {
            return self.unary();
        }
        let ops = BINARY_LEVELS[level];
        let mut left = self.binary(level + 1)?;
        loop {
            let t = self.peek();
            // `kw` is matched against the level lists too: that is the only
            // way `and`/`or`, which lex as keywords, reach the cascade.
            let matched =
                (t.kind == Kind::Op || t.kind == Kind::Kw) && ops.contains(&t.value.as_str());
            if !matched {
                return Ok(left);
            }
            let t = self.next();
            let right = self.binary(level + 1)?;
            left = Node::Binary {
                op: canon(&t.value),
                left: Box::new(left),
                right: Box::new(right),
                line: t.line,
            };
        }
    }

    fn unary(&mut self) -> PResult<Node> {
        let t = self.peek();
        if t.is_op(&["-"]) || (t.kind == Kind::Kw && t.value == "not") {
            let t = self.next();
            let operand = Box::new(self.unary()?);
            return Ok(Node::Unary {
                op: canon(&t.value),
                operand,
                line: t.line,
            });
        }
        self.postfix()
    }

    fn postfix(&mut self) -> PResult<Node> {
        let mut node = self.primary()?;
        loop {
            if self.at_op("(") {
                let line = self.next().line;
                let mut args = Vec::new();
                while !self.at_op(")") {
                    args.push(self.expr()?);
                    if !self.accept_op(",") {
                        break;
                    }
                }
                self.expect_op(")")?;
                node = Node::Call {
                    callee: Box::new(node),
                    args,
                    line,
                };
            } else if self.at_op(".") {
                let line = self.next().line;
                let field = self.expect(Kind::Ident)?.value;
                node = Node::Field {
                    target: Box::new(node),
                    field,
                    line,
                };
            } else if self.at_op("[") {
                let line = self.next().line;
                let index = Box::new(self.expr()?);
                self.expect_op("]")?;
                node = Node::Index {
                    target: Box::new(node),
                    index,
                    line,
                };
            } else {
                return Ok(node);
            }
        }
    }

    fn primary(&mut self) -> PResult<Node> {
        // Consumed before dispatch, so an unrecognised token is already past
        // and the error names it rather than looping on it.
        let t = self.next();

        match t.kind {
            Kind::Num => {
                // `float(t.value)` in the Python: every literal is a float, so
                // `1` is `1.0` and the serialisation keeps the `.0`.
                let value: f64 = t.value.parse().map_err(|_| ParseError {
                    message: format!("bad number '{}'", t.value),
                    line: t.line,
                })?;
                return Ok(Node::Num {
                    value,
                    line: t.line,
                });
            }
            Kind::Str => {
                return Ok(Node::Str {
                    value: t.value,
                    line: t.line,
                })
            }
            Kind::Ident => {
                // `true`/`false`/`none` are idents in the lexer, promoted to
                // literals here rather than being keywords.
                return Ok(match t.value.as_str() {
                    "true" => Node::Bool {
                        value: true,
                        line: t.line,
                    },
                    "false" => Node::Bool {
                        value: false,
                        line: t.line,
                    },
                    "none" => Node::NoneLit { line: t.line },
                    _ => Node::Var {
                        name: t.value,
                        line: t.line,
                    },
                });
            }
            _ => {}
        }

        if t.is_op(&["("]) {
            // Returned unwrapped: there is no grouping node, and the inner
            // node keeps its own line rather than the paren's.
            let inner = self.expr()?;
            self.expect_op(")")?;
            return Ok(inner);
        }

        if t.is_op(&["["]) {
            let mut items = Vec::new();
            self.skip_newlines();
            while !self.at_op("]") {
                items.push(self.expr()?);
                self.skip_newlines();
                if !self.accept_op(",") {
                    break;
                }
                self.skip_newlines();
            }
            self.expect_op("]")?;
            // The bracket's line, not the first item's.
            return Ok(Node::List {
                items,
                line: t.line,
            });
        }

        if t.is_op(&["{"]) {
            let mut entries = Vec::new();
            self.skip_newlines();
            while !self.at_op("}") {
                let key_tok = self.next();
                // A key may be a keyword: `{item: 1}` is a map, not a syntax
                // error, because the check is on kind rather than on reserved
                // words.
                if !matches!(key_tok.kind, Kind::Ident | Kind::Str | Kind::Kw) {
                    return Err(ParseError {
                        message: "expected map key".to_string(),
                        line: key_tok.line,
                    });
                }
                self.expect_op(":")?;
                entries.push(Entry {
                    key: key_tok.value,
                    value: self.expr()?,
                });
                self.skip_newlines();
                if !self.accept_op(",") {
                    break;
                }
                self.skip_newlines();
            }
            self.expect_op("}")?;
            return Ok(Node::Map {
                entries,
                line: t.line,
            });
        }

        let desc = if t.value.is_empty() {
            t.kind.as_str().to_string()
        } else {
            t.value.clone()
        };
        Err(ParseError {
            message: format!("unexpected '{}'", desc),
            line: t.line,
        })
    }
}

/// `&&`/`||` fold to their word spellings, so the two ways of writing a
/// conjunction produce identical trees.
fn canon(op: &str) -> String {
    match op {
        "&&" => "and".to_string(),
        "||" => "or".to_string(),
        other => other.to_string(),
    }
}

/// Parse `source`; `first_line` makes sites script-absolute across cells.
pub fn parse(source: &str, first_line: usize) -> PResult<Vec<Node>> {
    let toks = tokenize(source, first_line)?;
    Parser::new(toks).parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::KEYWORDS;

    fn p(src: &str) -> Vec<Node> {
        parse(src, 1).expect("should parse")
    }

    fn json(src: &str) -> String {
        serde_json::to_string(&p(src)).unwrap()
    }

    /// The dispatch set has to be exactly the keywords the Python has a
    /// `parse_<kw>` method for. Any keyword that is neither loose nor a
    /// statement keyword must fail in statement position -- silently treating
    /// one as loose is the failure mode this guards.
    #[test]
    fn keyword_dispatch_is_total_and_matches_oracle() {
        let mut fallthrough = Vec::new();
        for kw in KEYWORDS {
            let loose = LOOSE.contains(kw);
            let stmt = is_statement_keyword(kw);
            assert!(!(loose && stmt), "{kw} is both loose and a statement");
            if !loose && !stmt {
                fallthrough.push(*kw);
            }
        }
        fallthrough.sort_unstable();
        assert_eq!(
            fallthrough,
            [
                "and",
                "as",
                "cycle",
                "drift",
                "each",
                "flow",
                "from",
                "in",
                "not",
                "on",
                "or",
                "otherwise",
                "over",
                "roll",
                "settled",
                "until",
                "with_confidence",
            ],
            "the set of keywords with no handler has changed"
        );
    }

    /// Each of those, standing alone, must fail rather than parse.
    ///
    /// `and`/`or`/`not` are the near-miss: they are reached by the expression
    /// cascade rather than rejected at dispatch, so they fail a token later --
    /// but they do fail, which is what a port treating them as loose would not.
    #[test]
    fn every_unhandled_keyword_fails_in_statement_position() {
        for kw in KEYWORDS {
            if LOOSE.contains(kw) || is_statement_keyword(kw) {
                continue;
            }
            let src = format!(
                "{kw}
"
            );
            assert!(
                parse(&src, 1).is_err(),
                "'{kw}' parsed as a statement; the Python has no handler for it"
            );
        }
    }

    /// `from` leads no statement, so it reaches `parse_primary` and is
    /// rejected there. Confirmed against the oracle while building the corpus.
    #[test]
    fn unhandled_keyword_is_an_error_not_a_noop() {
        let e = parse("from a import b\n", 1).unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.message.contains("from"), "{}", e.message);
    }

    #[test]
    fn parens_leave_no_node() {
        assert_eq!(json("item a = (1)"), json("item a = 1"));
    }

    #[test]
    fn numbers_are_floats() {
        assert!(json("item a = 1").contains("\"value\":1.0"));
    }

    /// The two `Noop` shapes, which is why `body` is an `Option`.
    #[test]
    fn noop_key_sets_differ() {
        let import = json("import x from y");
        assert!(import.contains("\"text\":\"x from y\""), "{import}");
        assert!(!import.contains("body"), "import Noop must have no body");

        let loose = json("allow x");
        assert!(loose.contains("\"text\":\"allow x\""), "{loose}");
        assert!(loose.contains("\"body\":[]"), "loose Noop must have a body");
    }

    /// Absent confidence is a null, not a missing key.
    #[test]
    fn verdict_conf_is_present_and_null() {
        let s = json("proposition P:\n    motion M\n    support M");
        assert!(s.contains("\"conf\":null"), "{s}");
    }

    #[test]
    fn canon_folds_operator_spellings() {
        let a = json("item a = x && y || z");
        let b = json("item a = x and y or z");
        assert_eq!(a, b);
        assert!(a.contains("\"op\":\"and\""));
    }

    /// A pipe binds looser than every binary operator.
    #[test]
    fn pipe_is_above_the_cascade() {
        let s = json("item a = 1 + 2 |> f");
        let v = serde_json::from_str::<serde_json::Value>(&s).unwrap();
        assert_eq!(v[0]["expr"]["kind"], "Pipe");
        assert_eq!(v[0]["expr"]["left"]["kind"], "Binary");
    }

    /// An operator node takes the operator's line, not the left operand's.
    #[test]
    fn binary_takes_the_operator_line() {
        let s = json("item a = [\n  1,\n  2,\n]");
        let v = serde_json::from_str::<serde_json::Value>(&s).unwrap();
        // The bracket's line, not the first item's.
        assert_eq!(v[0]["expr"]["line"], 1);
    }

    #[test]
    fn proposition_lifts_motions() {
        let s = json("proposition P:\n    motion M\n    item a = 1");
        let v = serde_json::from_str::<serde_json::Value>(&s).unwrap();
        assert_eq!(v[0]["motions"].as_array().unwrap().len(), 1);
        assert_eq!(v[0]["body"].as_array().unwrap().len(), 1);
        assert_eq!(v[0]["body"][0]["kind"], "Item");
    }

    /// The rolled-back lookahead: a blank line after the block must not eat
    /// the following statement.
    #[test]
    fn given_without_otherwise_keeps_the_next_statement() {
        let v: serde_json::Value =
            serde_json::from_str(&json("given x:\n    item a = 1\n\nitem b = 2\n")).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2);
        assert_eq!(v[0]["otherwise"].as_array().unwrap().len(), 0);
        assert_eq!(v[1]["kind"], "Item");
    }

    #[test]
    fn considering_defaults_to_item() {
        let v: serde_json::Value =
            serde_json::from_str(&json("considering x in xs:\n    item a = 1")).unwrap();
        assert_eq!(v[0]["quant"], "item");
        let v: serde_json::Value =
            serde_json::from_str(&json("considering all x in xs:\n    item a = 1")).unwrap();
        assert_eq!(v[0]["quant"], "all");
    }

    /// Keywords are valid map keys.
    #[test]
    fn map_key_may_be_a_keyword() {
        let v: serde_json::Value =
            serde_json::from_str(&json("item a = {item: 1, given: 2}")).unwrap();
        assert_eq!(v[0]["expr"]["entries"][0]["key"], "item");
    }

    /// There is no field-assignment form.
    #[test]
    fn field_assignment_is_an_error() {
        assert!(parse("a.b = 3\n", 1).is_err());
    }

    /// `first_line` makes every site script-absolute.
    #[test]
    fn first_line_offsets_every_node() {
        let v: serde_json::Value = serde_json::from_str(
            &serde_json::to_string(&parse("item a = 1", 42).unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(v[0]["line"], 42);
        assert_eq!(v[0]["expr"]["line"], 42);
    }

    /// `next()` does not advance past `eof`, so a truncated script errors
    /// rather than spinning.
    #[test]
    fn truncated_input_terminates() {
        assert!(parse("item a =", 1).is_err());
        assert!(parse("given x:", 1).is_err());
        assert!(parse("item a = [1,", 1).is_err());
    }
}
