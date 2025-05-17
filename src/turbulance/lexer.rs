use logos::{Logos, Span};
use std::fmt;

/// TokenKind represents all possible token types in the Turbulance language
#[derive(Logos, Debug, Clone, Hash, Eq, PartialEq)]
pub enum TokenKind {
    // Keywords
    #[token("funxn")]
    FunctionDecl,

    #[token("project")]
    ProjectDecl,

    #[token("sources")]
    SourcesDecl,

    #[token("within")]
    Within,

    #[token("given")]
    Given,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("for")]
    For,

    #[token("each")]
    Each,

    #[token("considering")]
    Considering,

    #[token("all")]
    All,

    #[token("these")]
    These,

    #[token("item")]
    Item,

    #[token("in")]
    In,

    #[token("return")]
    Return,

    #[token("ensure")]
    Ensure,
    
    #[token("research")]
    Research,
    
    #[token("apply")]
    Apply,
    
    #[token("to_all")]
    ToAll,

    #[token("allow")]
    Allow,

    #[token("cause")]
    Cause,

    #[token("motion")]
    Motion,

    // Operators
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Multiply,

    #[token("/")]
    Divide,

    #[token("|")]
    Pipe,
    
    #[token("|>")]
    PipeForward,

    #[token("=>")]
    Arrow,

    #[token("=")]
    Assign,

    #[token("==")]
    Equal,

    #[token("!=")]
    NotEqual,

    #[token("<")]
    LessThan,

    #[token(">")]
    GreaterThan,

    #[token("<=")]
    LessThanEqual,

    #[token(">=")]
    GreaterThanEqual,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    // Delimiters
    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token(".")]
    Dot,

    // Complex tokens
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,

    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    #[regex(r"[0-9]+(\.[0-9]+)?")]
    NumberLiteral,

    // Comments and whitespace
    #[regex(r"//[^\n]*", logos::skip)]
    Comment,

    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,

    // Error token (without the error attribute as Logos 0.13+ doesn't require it)
    Error,
}

/// Token represents a token with its type and span (location in source)
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::FunctionDecl => write!(f, "funxn"),
            TokenKind::ProjectDecl => write!(f, "project"),
            TokenKind::SourcesDecl => write!(f, "sources"),
            TokenKind::Within => write!(f, "within"),
            TokenKind::Given => write!(f, "given"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Each => write!(f, "each"),
            TokenKind::Considering => write!(f, "considering"),
            TokenKind::All => write!(f, "all"),
            TokenKind::These => write!(f, "these"),
            TokenKind::Item => write!(f, "item"),
            TokenKind::In => write!(f, "in"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Ensure => write!(f, "ensure"),
            TokenKind::Research => write!(f, "research"),
            TokenKind::Apply => write!(f, "apply"),
            TokenKind::ToAll => write!(f, "to_all"),
            TokenKind::Allow => write!(f, "allow"),
            TokenKind::Cause => write!(f, "cause"),
            TokenKind::Motion => write!(f, "motion"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Multiply => write!(f, "*"),
            TokenKind::Divide => write!(f, "/"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::PipeForward => write!(f, "|>"),
            TokenKind::Arrow => write!(f, "=>"),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Equal => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::LessThan => write!(f, "<"),
            TokenKind::GreaterThan => write!(f, ">"),
            TokenKind::LessThanEqual => write!(f, "<="),
            TokenKind::GreaterThanEqual => write!(f, ">="),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Identifier => write!(f, "identifier"),
            TokenKind::StringLiteral => write!(f, "string"),
            TokenKind::NumberLiteral => write!(f, "number"),
            TokenKind::Comment => write!(f, "comment"),
            TokenKind::Whitespace => write!(f, "whitespace"),
            TokenKind::Error => write!(f, "error"),
        }
    }
}

/// The Lexer struct for tokenizing Turbulance source code
pub struct Lexer<'a> {
    lex: logos::Lexer<'a, TokenKind>,
    source: &'a str,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source code
    pub fn new(source: &'a str) -> Self {
        Self {
            lex: TokenKind::lexer(source),
            source,
        }
    }

    /// Get the current token's lexeme (actual text)
    fn get_lexeme(&self, span: Span) -> String {
        self.source[span.start..span.end].to_string()
    }

    /// Tokenize the entire source code into a vector of tokens
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        while let Some(token_kind) = self.lex.next() {
            let span = self.lex.span();
            let lexeme = self.get_lexeme(span.clone());
            
            tokens.push(Token {
                kind: token_kind,
                span,
                lexeme,
            });
        }
        
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_keywords() {
        let source = "funxn project sources within given if else for each considering all these item in return ensure research apply to_all allow cause motion";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::FunctionDecl,
            TokenKind::ProjectDecl,
            TokenKind::SourcesDecl,
            TokenKind::Within,
            TokenKind::Given,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::For,
            TokenKind::Each,
            TokenKind::Considering,
            TokenKind::All,
            TokenKind::These,
            TokenKind::Item,
            TokenKind::In,
            TokenKind::Return,
            TokenKind::Ensure,
            TokenKind::Research,
            TokenKind::Apply,
            TokenKind::ToAll,
            TokenKind::Allow,
            TokenKind::Cause,
            TokenKind::Motion,
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_operators() {
        let source = "+ - * / | |> => = == != < > <= >= && || !";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Multiply,
            TokenKind::Divide,
            TokenKind::Pipe,
            TokenKind::PipeForward,
            TokenKind::Arrow,
            TokenKind::Assign,
            TokenKind::Equal,
            TokenKind::NotEqual,
            TokenKind::LessThan,
            TokenKind::GreaterThan,
            TokenKind::LessThanEqual,
            TokenKind::GreaterThanEqual,
            TokenKind::And,
            TokenKind::Or,
            TokenKind::Not,
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_identifiers_and_literals() {
        let source = r#"identifier 123 123.456 "string literal" another_id"#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::Identifier,  // identifier
            TokenKind::NumberLiteral,  // 123
            TokenKind::NumberLiteral,  // 123.456
            TokenKind::StringLiteral,  // "string literal"
            TokenKind::Identifier,  // another_id
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_complex_code() {
        let source = r#"
            funxn enhance_paragraph(paragraph, domain="general"):
                within paragraph:
                    given contains("technical_term"):
                        research_context(domain)
                        ensure_explanation_follows()
                    given readability_score < 65:
                        simplify_sentences()
                        replace_jargon()
                    return processed
        "#;
        
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        // This is just a basic smoke test to ensure we don't panic on valid code
        assert!(tokens.len() > 20);  // Should have plenty of tokens
        
        // Check if specific important tokens are present
        let has_function = tokens.iter().any(|t| t.kind == TokenKind::FunctionDecl);
        let has_within = tokens.iter().any(|t| t.kind == TokenKind::Within);
        let has_given = tokens.iter().any(|t| t.kind == TokenKind::Given);
        let has_return = tokens.iter().any(|t| t.kind == TokenKind::Return);
        
        assert!(has_function && has_within && has_given && has_return);
    }
}
