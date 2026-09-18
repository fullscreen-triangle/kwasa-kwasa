//! Lexical analysis for the Turbulance language

use logos::{Logos, Span};
use std::fmt;
use crate::error::{TurbulanceError, Result};

/// Token types in the Turbulance language
#[derive(Logos, Debug, Clone, Hash, Eq, PartialEq)]
pub enum TokenKind {
    /// The `funxn` token.
    #[token("funxn")]
    FunctionDecl,

    /// The `project` token.
    #[token("project")]
    ProjectDecl,

    /// The `proposition` token.
    #[token("proposition")]
    Proposition,

    /// The `motion` token.
    #[token("motion")]
    Motion,

    /// The `hypothesis` token.
    #[token("hypothesis")]
    Hypothesis,

    /// The `experiment` token.
    #[token("experiment")]
    Experiment,

    /// The `analysis` token.
    #[token("analysis")]
    Analysis,

    /// The `item` token.
    #[token("item")]
    Item,

    /// The `var` token.
    #[token("var")]
    Var,

    /// The `point` token.
    #[token("point")]
    Point,

    /// The `resolution` token.
    #[token("resolution")]
    Resolution,

    /// The `given` token.
    #[token("given")]
    Given,

    /// The `within` token.
    #[token("within")]
    Within,

    /// The `considering` token.
    #[token("considering")]
    Considering,

    /// The `ensure` token.
    #[token("ensure")]
    Ensure,

    /// The `alternatively` token.
    #[token("alternatively")]
    Alternatively,

    /// The `if` token.
    #[token("if")]
    If,

    /// The `else` token.
    #[token("else")]
    Else,

    /// The `for` token.
    #[token("for")]
    For,

    /// The `each` token.
    #[token("each")]
    Each,

    /// The `in` token.
    #[token("in")]
    In,

    /// The `all` token.
    #[token("all")]
    All,

    /// The `these` token.
    #[token("these")]
    These,

    /// The `return` token.
    #[token("return")]
    Return,

    /// The `research` token.
    #[token("research")]
    Research,

    /// The `apply` token.
    #[token("apply")]
    Apply,

    /// The `to_all` token.
    #[token("to_all")]
    ToAll,

    /// The `sources` token.
    #[token("sources")]
    SourcesDecl,

    /// The `cause` token.
    #[token("cause")]
    Cause,

    /// The `allow` token.
    #[token("allow")]
    Allow,

    /// The `true` token.
    #[token("true")]
    True,

    /// The `false` token.
    #[token("false")]
    False,

    /// The `null` token.
    #[token("null")]
    Null,

    /// The `+` token.
    #[token("+")]
    Plus,

    /// The `-` token.
    #[token("-")]
    Minus,

    /// The `*` token.
    #[token("*")]
    Multiply,

    /// The `/` token.
    #[token("/")]
    Divide,

    /// The `%` token.
    #[token("%")]
    Modulo,

    /// The `**` token.
    #[token("**")]
    Power,

    /// The `&&` token.
    #[token("&&")]
    And,

    /// The `||` token.
    #[token("||")]
    Or,

    /// The `!` token.
    #[token("!")]
    Not,

    /// The `==` token.
    #[token("==")]
    Equal,

    /// The `!=` token.
    #[token("!=")]
    NotEqual,

    /// The `<` token.
    #[token("<")]
    LessThan,

    /// The `>` token.
    #[token(">")]
    GreaterThan,

    /// The `<=` token.
    #[token("<=")]
    LessThanEqual,

    /// The `>=` token.
    #[token(">=")]
    GreaterThanEqual,

    /// The `=` token.
    #[token("=")]
    Assign,

    /// The `=>` token.
    #[token("=>")]
    Arrow,

    /// The `|` token.
    #[token("|")]
    Pipe,
    
    /// The `|>` token.
    #[token("|>")]
    PipeForward,

    /// The `(` token.
    #[token("(")]
    LeftParen,

    /// The `)` token.
    #[token(")")]
    RightParen,

    /// The `{` token.
    #[token("{")]
    LeftBrace,

    /// The `}` token.
    #[token("}")]
    RightBrace,

    /// The `[` token.
    #[token("[")]
    LeftBracket,

    /// The `]` token.
    #[token("]")]
    RightBracket,

    /// The `,` token.
    #[token(",")]
    Comma,

    /// The `:` token.
    #[token(":")]
    Colon,

    /// The `;` token.
    #[token(";")]
    Semicolon,

    /// The `.` token.
    #[token(".")]
    Dot,

    /// The `..` token.
    #[token("..")]
    Range,

    /// The `...` token.
    #[token("...")]
    Spread,

    /// A name: a letter or underscore, then letters, digits or underscores.
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,

    /// A double-quoted string, with backslash escapes.
    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    /// A decimal number, optionally with a fraction and an exponent.
    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")]
    NumberLiteral,

    /// A `//` line comment or a `/* */` block comment. Skipped.
    #[regex(r"//[^\n]*", logos::skip)]
    #[regex(r"/\*([^*]|\*[^/])*\*/", logos::skip)]
    Comment,

    /// A run of spaces, tabs or newlines. Skipped.
    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,

    /// End of input.
    Eof,

    /// A character sequence that matches no other rule.
    Error,
}

/// Token with its type, location, and lexeme
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The type of token
    pub kind: TokenKind,
    /// The location span in the source
    pub span: Span,
    /// The actual text content
    pub lexeme: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
}

impl Token {
    /// Create a new token
    pub fn new(kind: TokenKind, span: Span, lexeme: String, line: usize, column: usize) -> Self {
        Self {
            kind,
            span,
            lexeme,
            line,
            column,
        }
    }

    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::FunctionDecl
                | TokenKind::ProjectDecl
                | TokenKind::Proposition
                | TokenKind::Motion
                | TokenKind::Hypothesis
                | TokenKind::Experiment
                | TokenKind::Analysis
                | TokenKind::Item
                | TokenKind::Var
                | TokenKind::Point
                | TokenKind::Resolution
                | TokenKind::Given
                | TokenKind::Within
                | TokenKind::Considering
                | TokenKind::Ensure
                | TokenKind::Alternatively
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::For
                | TokenKind::Each
                | TokenKind::In
                | TokenKind::All
                | TokenKind::These
                | TokenKind::Return
                | TokenKind::Research
                | TokenKind::Apply
                | TokenKind::ToAll
                | TokenKind::SourcesDecl
                | TokenKind::Cause
                | TokenKind::Allow
                | TokenKind::True
                | TokenKind::False
                | TokenKind::Null
        )
    }

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Multiply
                | TokenKind::Divide
                | TokenKind::Modulo
                | TokenKind::Power
                | TokenKind::And
                | TokenKind::Or
                | TokenKind::Not
                | TokenKind::Equal
                | TokenKind::NotEqual
                | TokenKind::LessThan
                | TokenKind::GreaterThan
                | TokenKind::LessThanEqual
                | TokenKind::GreaterThanEqual
                | TokenKind::Assign
                | TokenKind::Arrow
                | TokenKind::Pipe
                | TokenKind::PipeForward
        )
    }

    /// Check if this token is a delimiter
    pub fn is_delimiter(&self) -> bool {
        matches!(
            self.kind,
            TokenKind::LeftParen
                | TokenKind::RightParen
                | TokenKind::LeftBrace
                | TokenKind::RightBrace
                | TokenKind::LeftBracket
                | TokenKind::RightBracket
                | TokenKind::Comma
                | TokenKind::Colon
                | TokenKind::Semicolon
                | TokenKind::Dot
                | TokenKind::Range
                | TokenKind::Spread
        )
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::FunctionDecl => write!(f, "funxn"),
            TokenKind::ProjectDecl => write!(f, "project"),
            TokenKind::Proposition => write!(f, "proposition"),
            TokenKind::Motion => write!(f, "motion"),
            TokenKind::Hypothesis => write!(f, "hypothesis"),
            TokenKind::Experiment => write!(f, "experiment"),
            TokenKind::Analysis => write!(f, "analysis"),
            TokenKind::Item => write!(f, "item"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::Point => write!(f, "point"),
            TokenKind::Resolution => write!(f, "resolution"),
            TokenKind::Given => write!(f, "given"),
            TokenKind::Within => write!(f, "within"),
            TokenKind::Considering => write!(f, "considering"),
            TokenKind::Ensure => write!(f, "ensure"),
            TokenKind::Alternatively => write!(f, "alternatively"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Each => write!(f, "each"),
            TokenKind::In => write!(f, "in"),
            TokenKind::All => write!(f, "all"),
            TokenKind::These => write!(f, "these"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Research => write!(f, "research"),
            TokenKind::Apply => write!(f, "apply"),
            TokenKind::ToAll => write!(f, "to_all"),
            TokenKind::SourcesDecl => write!(f, "sources"),
            TokenKind::Cause => write!(f, "cause"),
            TokenKind::Allow => write!(f, "allow"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Null => write!(f, "null"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Multiply => write!(f, "*"),
            TokenKind::Divide => write!(f, "/"),
            TokenKind::Modulo => write!(f, "%"),
            TokenKind::Power => write!(f, "**"),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::Equal => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::LessThan => write!(f, "<"),
            TokenKind::GreaterThan => write!(f, ">"),
            TokenKind::LessThanEqual => write!(f, "<="),
            TokenKind::GreaterThanEqual => write!(f, ">="),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Arrow => write!(f, "=>"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::PipeForward => write!(f, "|>"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Range => write!(f, ".."),
            TokenKind::Spread => write!(f, "..."),
            TokenKind::Identifier => write!(f, "identifier"),
            TokenKind::StringLiteral => write!(f, "string"),
            TokenKind::NumberLiteral => write!(f, "number"),
            TokenKind::Comment => write!(f, "comment"),
            TokenKind::Whitespace => write!(f, "whitespace"),
            TokenKind::Eof => write!(f, "end of file"),
            TokenKind::Error => write!(f, "error"),
        }
    }
}

/// Lexer for tokenizing Turbulance source code
pub struct Lexer<'a> {
    source: &'a str,
    lexer: logos::Lexer<'a, TokenKind>,
    line: usize,
    column: usize,
    last_newline: usize,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source code
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            lexer: TokenKind::lexer(source),
            line: 1,
            column: 1,
            last_newline: 0,
        }
    }

    /// Update line and column tracking
    fn update_position(&mut self, span: &Span) {
        let text = &self.source[self.last_newline..span.end];
        let newlines: Vec<_> = text.match_indices('\n').collect();
        
        if !newlines.is_empty() {
            self.line += newlines.len();
            self.last_newline = span.start + newlines.last().unwrap().0 + 1;
            self.column = span.end - self.last_newline + 1;
        } else {
            self.column = span.end - self.last_newline + 1;
        }
    }

    /// Get the lexeme for a span
    fn get_lexeme(&self, span: &Span) -> String {
        self.source[span.clone()].to_string()
    }

    /// Tokenize the entire source into a vector of tokens
    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        
        while let Some(token_result) = self.lexer.next() {
            let span = self.lexer.span();
            let lexeme = self.get_lexeme(&span);
            
            self.update_position(&span);
            
            match token_result {
                Ok(token_kind) => {
                    tokens.push(Token::new(token_kind, span, lexeme, self.line, self.column));
                }
                Err(_) => {
                    return Err(TurbulanceError::lexical(
                        span.start,
                        format!("Unexpected character: '{}'", lexeme)
                    ));
                }
            }
        }
        
        // Add EOF token
        tokens.push(Token::new(
            TokenKind::Eof,
            self.source.len()..self.source.len(),
            String::new(),
            self.line,
            self.column,
        ));
        
        Ok(tokens)
    }
}

/// Convenience function to tokenize source code
pub fn tokenize(source: &str) -> Result<Vec<Token>> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_keywords() {
        let source = "funxn proposition motion given";
        let tokens = tokenize(source).unwrap();
        
        assert_eq!(tokens.len(), 5); // 4 keywords + EOF
        assert_eq!(tokens[0].kind, TokenKind::FunctionDecl);
        assert_eq!(tokens[1].kind, TokenKind::Proposition);
        assert_eq!(tokens[2].kind, TokenKind::Motion);
        assert_eq!(tokens[3].kind, TokenKind::Given);
        assert_eq!(tokens[4].kind, TokenKind::Eof);
    }

    #[test]
    fn test_tokenize_operators() {
        let source = "+ - * / == != <= >= && ||";
        let tokens = tokenize(source).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::Plus);
        assert_eq!(tokens[1].kind, TokenKind::Minus);
        assert_eq!(tokens[2].kind, TokenKind::Multiply);
        assert_eq!(tokens[3].kind, TokenKind::Divide);
        assert_eq!(tokens[4].kind, TokenKind::Equal);
        assert_eq!(tokens[5].kind, TokenKind::NotEqual);
        assert_eq!(tokens[6].kind, TokenKind::LessThanEqual);
        assert_eq!(tokens[7].kind, TokenKind::GreaterThanEqual);
        assert_eq!(tokens[8].kind, TokenKind::And);
        assert_eq!(tokens[9].kind, TokenKind::Or);
    }

    #[test]
    fn test_tokenize_literals() {
        let source = r#"42 3.14 "hello world" true false null"#;
        let tokens = tokenize(source).unwrap();
        
        assert_eq!(tokens[0].kind, TokenKind::NumberLiteral);
        assert_eq!(tokens[0].lexeme, "42");
        assert_eq!(tokens[1].kind, TokenKind::NumberLiteral);
        assert_eq!(tokens[1].lexeme, "3.14");
        assert_eq!(tokens[2].kind, TokenKind::StringLiteral);
        assert_eq!(tokens[2].lexeme, r#""hello world""#);
        assert_eq!(tokens[3].kind, TokenKind::True);
        assert_eq!(tokens[4].kind, TokenKind::False);
        assert_eq!(tokens[5].kind, TokenKind::Null);
    }

    #[test]
    fn test_tokenize_scientific_code() {
        let source = r#"
        proposition DataQualityHypothesis:
            motion Hypothesis("Quality improves accuracy")
            
            given data_quality > 0.8:
                item model = train_model(data)
                ensure model.accuracy > 0.9
        "#;
        
        let tokens = tokenize(source).unwrap();
        
        // Should contain proposition, identifier, colon, etc.
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Proposition));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Motion));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Hypothesis));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Given));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Item));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::Ensure));
    }

    #[test]
    fn test_position_tracking() {
        let source = "line1\nline2\nline3";
        let tokens = tokenize(source).unwrap();
        
        // Check that line numbers are tracked correctly
        assert_eq!(tokens[0].line, 1); // line1
        assert_eq!(tokens[1].line, 2); // line2  
        assert_eq!(tokens[2].line, 3); // line3
    }

    #[test]
    fn test_invalid_token() {
        let source = "valid_identifier @#$%";
        let result = tokenize(source);
        
        assert!(result.is_err());
        if let Err(TurbulanceError::LexicalError { position, message }) = result {
            assert!(message.contains("Unexpected character"));
        } else {
            panic!("Expected lexical error");
        }
    }
} 