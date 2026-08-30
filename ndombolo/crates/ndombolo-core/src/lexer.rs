//! Turbulance lexer.
//!
//! A port of `prototype/lexer.py`, which is the reference the Rust must agree
//! with token for token. Where a natural Rust idiom would diverge from the
//! Python, the Python wins and the reason is noted inline: the differential
//! harness compares records, and a token-level difference surfaces there as an
//! unexplained parse failure rather than as anything legible.

use std::fmt;

/// Keywords. Anything here lexes as `Kw`, everything else as `Ident`.
pub const KEYWORDS: &[&str] = &[
    "funxn", "item", "proposition", "hypothesis", "motion", "support",
    "contradict", "inconclusive", "within", "given", "considering", "for",
    "each", "in", "while", "return", "ensure", "allow", "research", "cause",
    "point", "resolution", "resolve", "cycle", "drift", "flow", "roll",
    "until", "settled", "over", "on", "goal", "metacognitive", "import",
    "from", "as", "otherwise", "with_confidence", "and", "or", "not",
];

/// `considering` quantifiers; `item` is the default when none is written.
pub const QUANTS: &[&str] = &["all", "these"];

/// Recognised by the grammar but without deterministic semantics in this core.
/// Parsed to a no-op so a script using them still runs.
pub const LOOSE: &[&str] = &[
    "allow", "research", "cause", "goal", "metacognitive", "resolution",
];

/// Longest first: the scanner takes the first match, so `==` must precede `=`.
const OPERATORS: &[&str] = &[
    "==", "!=", "<=", ">=", "=>", "|>", "&&", "||",
    "=", "<", ">", "+", "-", "*", "/", "%",
    "(", ")", "[", "]", "{", "}", ",", ":", ".", "|",
];

pub fn is_keyword(word: &str) -> bool {
    KEYWORDS.contains(&word)
}

pub fn is_quant(word: &str) -> bool {
    QUANTS.contains(&word)
}

pub fn is_loose(word: &str) -> bool {
    LOOSE.contains(&word)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub message: String,
    pub line: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for LexError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Kw,
    Ident,
    Num,
    Str,
    Op,
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl Kind {
    /// The spelling used by the Python, for diagnostics that cross the boundary.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Kw => "kw",
            Kind::Ident => "ident",
            Kind::Num => "num",
            Kind::Str => "str",
            Kind::Op => "op",
            Kind::Newline => "newline",
            Kind::Indent => "indent",
            Kind::Dedent => "dedent",
            Kind::Eof => "eof",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: Kind,
    pub value: String,
    pub line: usize,
    pub col: usize,
}

impl Token {
    fn new(kind: Kind, value: impl Into<String>, line: usize, col: usize) -> Self {
        Token { kind, value: value.into(), line, col }
    }

    pub fn is_kw(&self, names: &[&str]) -> bool {
        self.kind == Kind::Kw && names.contains(&self.value.as_str())
    }

    pub fn is_op(&self, names: &[&str]) -> bool {
        self.kind == Kind::Op && names.contains(&self.value.as_str())
    }
}

fn is_ident_start(ch: char) -> bool {
    ch.is_alphabetic() || ch == '_'
}

fn is_ident_part(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// Net change in open brackets across one line's tokens.
fn bracket_delta(toks: &[Token]) -> i32 {
    toks.iter()
        .map(|t| {
            if t.is_op(&["(", "[", "{"]) {
                1
            } else if t.is_op(&[")", "]", "}"]) {
                -1
            } else {
                0
            }
        })
        .sum()
}

/// Tab stops every `width` columns, matching the Python's `_expand_tabs`.
fn expand_tabs(prefix: &str, width: usize) -> usize {
    let mut col = 0usize;
    for ch in prefix.chars() {
        col = if ch == '\t' { col + width - (col % width) } else { col + 1 };
    }
    col
}

fn unescape(ch: char) -> char {
    match ch {
        'n' => '\n',
        't' => '\t',
        '\\' => '\\',
        '"' => '"',
        other => other,
    }
}

/// Scan `source` into tokens, emitting explicit indent/dedent.
///
/// Layout is significant: a block is opened by `:` at end of line and delimited
/// by indentation. `first_line` keeps line numbers script-absolute, so a trace
/// site names a place in the document rather than an offset into a cell.
pub fn tokenize(source: &str, first_line: usize) -> Result<Vec<Token>, LexError> {
    let mut toks: Vec<Token> = Vec::new();
    let mut indents: Vec<usize> = vec![0];
    let mut bracket_depth: i32 = 0;

    let normalised = source.replace("\r\n", "\n").replace('\r', "\n");
    let lines: Vec<&str> = normalised.split('\n').collect();

    for (offset, raw) in lines.iter().enumerate() {
        let lineno = first_line + offset;
        let stripped = raw.trim();

        // Blank lines and full-line comments carry no layout information.
        if stripped.is_empty() || stripped.starts_with("//") || stripped.starts_with('#') {
            continue;
        }

        let line_toks = scan_line(raw, lineno)?;

        // Inside brackets layout carries no meaning: a multi-line list or map is
        // one logical line, so indent/dedent stay suppressed until it closes.
        if bracket_depth > 0 {
            bracket_depth += bracket_delta(&line_toks);
            toks.extend(line_toks);
            continue;
        }

        let ws = raw.len() - raw.trim_start_matches([' ', '\t']).len();
        let depth = expand_tabs(&raw[..ws], 4);

        if depth > *indents.last().expect("indents is never empty") {
            indents.push(depth);
            toks.push(Token::new(Kind::Indent, "", lineno, depth));
        } else {
            while depth < *indents.last().expect("indents is never empty") {
                indents.pop();
                toks.push(Token::new(Kind::Dedent, "", lineno, depth));
            }
            if depth != *indents.last().expect("indents is never empty") {
                return Err(LexError {
                    message: "inconsistent indentation".to_string(),
                    line: lineno,
                });
            }
        }

        bracket_depth += bracket_delta(&line_toks);
        toks.extend(line_toks);
        if bracket_depth == 0 {
            // `len(raw)` in the Python counts characters, so a line holding
            // non-ASCII text would place this token past the end of the line
            // if measured in bytes.
            toks.push(Token::new(Kind::Newline, "", lineno, raw.chars().count()));
        }
    }

    // Trailing dedents and EOF carry the last line's number, as in the Python:
    // `first_line + len(lines) - 1`.
    let last_line = first_line + lines.len().saturating_sub(1);
    while indents.len() > 1 {
        indents.pop();
        toks.push(Token::new(Kind::Dedent, "", last_line, 0));
    }
    toks.push(Token::new(Kind::Eof, "", last_line, 0));
    Ok(toks)
}

fn scan_line(raw: &str, lineno: usize) -> Result<Vec<Token>, LexError> {
    let mut out: Vec<Token> = Vec::new();
    // Character indices throughout. A Python string is a sequence of code
    // points, so every `col` the oracle emits counts characters, not bytes.
    // The distinction is not cosmetic: `col` is carried in the emitted record,
    // and a trace site is supposed to name a place in the document, so on any
    // line holding non-ASCII text a byte offset would point at the wrong
    // column. Caught by the differential harness on `edge-unicode-str`.
    let b: Vec<char> = raw.chars().collect();
    let n = b.len();
    let mut i = 0usize;

    while i < n {
        let ch = b[i];

        if ch == ' ' || ch == '\t' {
            i += 1;
            continue;
        }

        // A trailing comment ends the line.
        if (ch == '/' && i + 1 < n && b[i + 1] == '/') || ch == '#' {
            break;
        }

        if ch == '"' {
            let mut j = i + 1;
            let mut buf = String::new();
            let mut closed = false;
            while j < n {
                let cj = b[j];
                if cj == '"' {
                    closed = true;
                    break;
                }
                if cj == '\\' && j + 1 < n {
                    buf.push(unescape(b[j + 1]));
                    j += 2;
                    continue;
                }
                buf.push(cj);
                j += 1;
            }
            if !closed {
                return Err(LexError {
                    message: "unterminated string".to_string(),
                    line: lineno,
                });
            }
            out.push(Token::new(Kind::Str, buf, lineno, i));
            i = j + 1;
            continue;
        }

        // A leading dot starts a number only when a digit follows, so `.5` is a
        // number while `a.b` keeps its `.` as an operator.
        let starts_number =
            ch.is_ascii_digit() || (ch == '.' && i + 1 < n && b[i + 1].is_ascii_digit());
        if starts_number {
            let mut j = i;
            let mut seen_dot = false;
            while j < n {
                let cj = b[j];
                if cj.is_ascii_digit() {
                    j += 1;
                } else if cj == '.' && !seen_dot {
                    seen_dot = true;
                    j += 1;
                } else {
                    break;
                }
            }
            let word: String = b[i..j].iter().collect();
            out.push(Token::new(Kind::Num, word, lineno, i));
            i = j;
            continue;
        }

        if is_ident_start(ch) {
            let mut j = i;
            while j < n && is_ident_part(b[j]) {
                j += 1;
            }
            let word: String = b[i..j].iter().collect();
            let kind = if is_keyword(&word) { Kind::Kw } else { Kind::Ident };
            out.push(Token::new(kind, word, lineno, i));
            i = j;
            continue;
        }

        // Compared character-wise for the same reason the index is: `op.len()`
        // would be a byte count and would desynchronise `i` from `col`.
        let matched = OPERATORS.iter().find(|op| {
            let o: Vec<char> = op.chars().collect();
            i + o.len() <= n && b[i..i + o.len()] == o[..]
        });
        match matched {
            Some(op) => {
                out.push(Token::new(Kind::Op, *op, lineno, i));
                i += op.chars().count();
            }
            None => {
                // Match the Python's `repr`-style quoting of the offending char.
                return Err(LexError {
                    message: format!("unexpected character '{}'", ch),
                    line: lineno,
                });
            }
        }
    }

    Ok(out)
}

/// First token on `line`, for error reporting.
pub fn find_token(toks: &[Token], line: usize) -> Option<&Token> {
    toks.iter().find(|t| {
        t.line == line
            && !matches!(t.kind, Kind::Indent | Kind::Dedent | Kind::Newline)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<(Kind, String)> {
        tokenize(src, 1)
            .expect("tokenize")
            .into_iter()
            .map(|t| (t.kind, t.value))
            .collect()
    }

    #[test]
    fn keywords_and_idents_separate() {
        let t = kinds("item threshold = 0.4");
        assert_eq!(t[0], (Kind::Kw, "item".to_string()));
        assert_eq!(t[1], (Kind::Ident, "threshold".to_string()));
        assert_eq!(t[2], (Kind::Op, "=".to_string()));
        assert_eq!(t[3], (Kind::Num, "0.4".to_string()));
    }

    #[test]
    fn longest_operator_wins() {
        let t = kinds("a >= b");
        assert!(t.iter().any(|(k, v)| *k == Kind::Op && v == ">="));
        assert!(!t.iter().any(|(_, v)| v == ">"));
    }

    #[test]
    fn layout_emits_indent_and_dedent() {
        let t = kinds("given x:\n    y = 1\nz = 2");
        let seq: Vec<Kind> = t.iter().map(|(k, _)| *k).collect();
        assert!(seq.contains(&Kind::Indent));
        assert!(seq.contains(&Kind::Dedent));
    }

    #[test]
    fn brackets_suppress_layout() {
        // A multi-line list carries no layout at all. The closing line is
        // itself scanned while the depth is still positive, so it takes the
        // suppressed branch and no newline is emitted for the whole
        // construct -- verified against the Python, which agrees.
        let t = kinds("item xs = [\n  1,\n  2,\n]");
        let seq: Vec<Kind> = t.iter().map(|(k, _)| *k).collect();
        assert!(!seq.contains(&Kind::Indent), "layout leaked inside brackets");
        assert_eq!(seq.iter().filter(|k| **k == Kind::Newline).count(), 0);
    }

    #[test]
    fn comments_and_blanks_carry_no_layout() {
        let t = kinds("item a = 1\n\n  // indented comment\nitem b = 2");
        let seq: Vec<Kind> = t.iter().map(|(k, _)| *k).collect();
        assert!(!seq.contains(&Kind::Indent));
    }

    #[test]
    fn leading_dot_number_but_dot_operator_otherwise() {
        assert!(kinds(".5").iter().any(|(k, v)| *k == Kind::Num && v == ".5"));
        assert!(kinds("a.b").iter().any(|(k, v)| *k == Kind::Op && v == "."));
    }

    #[test]
    fn string_escapes_and_unterminated() {
        let t = kinds(r#"item s = "a\nb""#);
        assert!(t.iter().any(|(k, v)| *k == Kind::Str && v == "a\nb"));
        let e = tokenize(r#"item s = "oops"#, 1).expect_err("should fail");
        assert_eq!(e.message, "unterminated string");
    }

    #[test]
    fn tabs_expand_to_width_four() {
        assert_eq!(expand_tabs("\t", 4), 4);
        assert_eq!(expand_tabs("  \t", 4), 4);
        assert_eq!(expand_tabs("    ", 4), 4);
    }

    #[test]
    fn first_line_offsets_are_script_absolute() {
        let t = tokenize("item a = 1", 47).expect("tokenize");
        assert_eq!(t[0].line, 47);
    }

    #[test]
    fn cols_count_characters_not_bytes() {
        // A first port used byte indices, which agree with the oracle on every
        // ASCII line and so passed the whole generated corpus. `col` is carried
        // in the record and is meant to name a place in the document, so being
        // wrong here means pointing at the wrong column in any file with
        // non-ASCII prose -- which a .ndo report routinely has.
        let t = tokenize("item s = \"µ\" + 1", 1).expect("tokenize");
        let plus = t.iter().find(|t| t.value == "+").expect("the + operator");
        assert_eq!(plus.col, 13, "byte indexing would say 14");
        assert_eq!(t.last().expect("eof").kind, Kind::Eof);
    }

    #[test]
    fn inconsistent_indentation_is_an_error() {
        let e = tokenize("given x:\n      a = 1\n   b = 2", 1).expect_err("should fail");
        assert_eq!(e.message, "inconsistent indentation");
    }
}
