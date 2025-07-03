//! Parser for the Turbulance language

use crate::ast::{BinaryOp, Motion, Node, Parameter, Position, Source, Span, TextOp, UnaryOp};
use crate::error::{Result, TurbulanceError};
use crate::lexer::{Token, TokenKind};
use std::collections::HashMap;

/// Recursive descent parser for Turbulance
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    /// Create a new parser
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, current: 0 }
    }

    /// Parse tokens into an AST
    pub fn parse(&mut self) -> Result<Node> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            if self.check(&TokenKind::Eof) {
                break;
            }
            statements.push(self.declaration()?);
        }

        let span = if let Some(first) = self.tokens.first() {
            let last = self.tokens.last().unwrap();
            Span::new(
                Position::new(first.line, first.column, first.span.start),
                Position::new(last.line, last.column, last.span.end),
            )
        } else {
            Span::new(Position::new(1, 1, 0), Position::new(1, 1, 0))
        };

        Ok(Node::program(statements, span))
    }

    /// Parse a declaration
    fn declaration(&mut self) -> Result<Node> {
        if self.match_token(&TokenKind::FunctionDecl) {
            return self.function_declaration();
        }

        if self.match_token(&TokenKind::ProjectDecl) {
            return self.project_declaration();
        }

        if self.match_token(&TokenKind::Proposition) {
            return self.proposition_declaration();
        }

        if self.match_token(&TokenKind::SourcesDecl) {
            return self.sources_declaration();
        }

        if self.match_token(&TokenKind::Point) {
            return self.point_declaration();
        }

        if self.match_token(&TokenKind::Resolution) {
            return self.resolution_declaration();
        }

        self.statement()
    }

    /// Parse a function declaration
    fn function_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let name = self.consume_identifier("Expected function name")?;

        self.consume(&TokenKind::LeftParen, "Expected '(' after function name")?;
        let parameters = self.parse_parameters()?;
        self.consume(&TokenKind::RightParen, "Expected ')' after parameters")?;

        self.consume(&TokenKind::Colon, "Expected ':' after function signature")?;
        let body = self.block_or_expression()?;

        let span = self.span_from_to(&start_pos, body.span());

        Ok(Node::FunctionDecl {
            name,
            parameters,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a project declaration
    fn project_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let name = self.consume_identifier("Expected project name")?;

        let attributes = if self.match_token(&TokenKind::LeftParen) {
            let attrs = self.parse_attributes()?;
            self.consume(
                &TokenKind::RightParen,
                "Expected ')' after project attributes",
            )?;
            attrs
        } else {
            HashMap::new()
        };

        self.consume(&TokenKind::Colon, "Expected ':' after project declaration")?;
        let body = self.block_or_expression()?;

        let span = self.span_from_to(&start_pos, body.span());

        Ok(Node::ProjectDecl {
            name,
            attributes,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a proposition declaration
    fn proposition_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let name = self.consume_identifier("Expected proposition name")?;
        self.consume(&TokenKind::Colon, "Expected ':' after proposition name")?;

        let mut motions = Vec::new();
        let mut statements = Vec::new();

        // Parse motions and statements
        while !self.check(&TokenKind::Eof) && !self.is_block_end() {
            if self.match_token(&TokenKind::Motion) {
                motions.push(self.parse_motion()?);
            } else {
                statements.push(self.statement()?);
            }
        }

        let body_span = if let Some(last_stmt) = statements.last() {
            self.span_from_token_to_node(&start_pos, last_stmt)
        } else {
            self.span_from_to(&start_pos, &start_pos)
        };

        let body = Node::block(statements, body_span);
        let span = self.span_from_to(&start_pos, &self.previous());

        Ok(Node::Proposition {
            name,
            motions,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a sources declaration
    fn sources_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        self.consume(&TokenKind::Colon, "Expected ':' after 'sources'")?;

        let mut sources = Vec::new();

        while !self.is_at_end() && !self.is_block_end() {
            sources.push(self.parse_source()?);
        }

        let span = self.span_from_to(&start_pos, &self.previous());

        Ok(Node::SourcesDecl { sources, span })
    }

    /// Parse a point declaration
    fn point_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let name = self.consume_identifier("Expected point name")?;
        self.consume(&TokenKind::Assign, "Expected '=' after point name")?;
        self.consume(
            &TokenKind::LeftBrace,
            "Expected '{' to start point properties",
        )?;

        let mut properties = HashMap::new();

        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let key = self.consume_identifier("Expected property name")?;
            self.consume(&TokenKind::Colon, "Expected ':' after property name")?;
            let value = self.expression()?;
            properties.insert(key, value);

            if !self.check(&TokenKind::RightBrace) {
                self.consume(&TokenKind::Comma, "Expected ',' between properties")?;
            }
        }

        self.consume(
            &TokenKind::RightBrace,
            "Expected '}' after point properties",
        )?;

        let span = self.span_from_to(&start_pos, &self.previous());

        Ok(Node::Point {
            name,
            properties,
            span,
        })
    }

    /// Parse a resolution declaration
    fn resolution_declaration(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let name = self.consume_identifier("Expected resolution name")?;

        self.consume(&TokenKind::LeftParen, "Expected '(' after resolution name")?;
        let parameters = self.parse_parameters()?;
        self.consume(&TokenKind::RightParen, "Expected ')' after parameters")?;

        let return_type = if self.match_token(&TokenKind::Arrow) {
            Some(self.consume_identifier("Expected return type")?)
        } else {
            None
        };

        self.consume(&TokenKind::Colon, "Expected ':' after resolution signature")?;
        let body = self.block_or_expression()?;

        let span = self.span_from_to(&start_pos, body.span());

        Ok(Node::Resolution {
            name,
            parameters,
            body: Box::new(body),
            return_type,
            span,
        })
    }

    /// Parse a statement
    fn statement(&mut self) -> Result<Node> {
        if self.match_token(&TokenKind::Return) {
            return self.return_statement();
        }

        if self.match_token(&TokenKind::Given) {
            return self.given_statement();
        }

        if self.match_token(&TokenKind::Within) {
            return self.within_statement();
        }

        if self.match_token(&TokenKind::Considering) {
            return self.considering_statement();
        }

        if self.match_token(&TokenKind::Ensure) {
            return self.ensure_statement();
        }

        if self.match_token(&TokenKind::Research) {
            return self.research_statement();
        }

        if self.match_token(&TokenKind::Item) || self.match_token(&TokenKind::Var) {
            return self.assignment_statement();
        }

        self.expression_statement()
    }

    /// Parse a return statement
    fn return_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let value = if self.is_statement_end() {
            None
        } else {
            Some(Box::new(self.expression()?))
        };

        let span = if let Some(ref val) = value {
            self.span_from_token_to_node(&start_pos, &val)
        } else {
            self.span_from_to(&start_pos, &start_pos)
        };

        Ok(Node::Return { value, span })
    }

    /// Parse a given statement
    fn given_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let condition = self.expression()?;
        self.consume(&TokenKind::Colon, "Expected ':' after given condition")?;
        let then_branch = self.block_or_expression()?;

        let else_branch = if self.match_token(&TokenKind::Alternatively) {
            self.consume(&TokenKind::Colon, "Expected ':' after 'alternatively'")?;
            Some(Box::new(self.block_or_expression()?))
        } else {
            None
        };

        let end_span = else_branch
            .as_ref()
            .map(|e| e.span())
            .unwrap_or(then_branch.span());
        let span = self.span_from_spans(&self.token_span(&start_pos), end_span);

        Ok(Node::Given {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
            span,
        })
    }

    /// Parse a within statement
    fn within_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let target = self.expression()?;
        self.consume(&TokenKind::Colon, "Expected ':' after within target")?;
        let body = self.block_or_expression()?;

        let span = self.span_from_to(&start_pos, body.span());

        Ok(Node::Within {
            target: Box::new(target),
            body: Box::new(body),
            span,
        })
    }

    /// Parse a considering statement
    fn considering_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let items = self.expression()?;
        self.consume(&TokenKind::Colon, "Expected ':' after considering items")?;
        let body = self.block_or_expression()?;

        let span = self.span_from_to(&start_pos, body.span());

        Ok(Node::Considering {
            items: Box::new(items),
            body: Box::new(body),
            span,
        })
    }

    /// Parse an ensure statement
    fn ensure_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let condition = self.expression()?;
        let span = self.span_from_to(&start_pos, condition.span());

        Ok(Node::Ensure {
            condition: Box::new(condition),
            span,
        })
    }

    /// Parse a research statement
    fn research_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let query = self.expression()?;
        let span = self.span_from_to(&start_pos, query.span());

        Ok(Node::Research {
            query: Box::new(query),
            span,
        })
    }

    /// Parse an assignment statement
    fn assignment_statement(&mut self) -> Result<Node> {
        let start_pos = self.previous().clone();

        let target = self.primary()?; // This should be an identifier
        self.consume(&TokenKind::Assign, "Expected '=' in assignment")?;
        let value = self.expression()?;

        let span = self.span_from_to(&start_pos, value.span());

        Ok(Node::Assignment {
            target: Box::new(target),
            value: Box::new(value),
            span,
        })
    }

    /// Parse an expression statement
    fn expression_statement(&mut self) -> Result<Node> {
        let expr = self.expression()?;
        let span = expr.span().clone();

        Ok(Node::ExpressionStatement {
            expression: Box::new(expr),
            span,
        })
    }

    /// Parse an expression
    fn expression(&mut self) -> Result<Node> {
        self.pipe()
    }

    /// Parse pipe operations
    fn pipe(&mut self) -> Result<Node> {
        let mut expr = self.logical_or()?;

        while self.match_token(&TokenKind::Pipe) || self.match_token(&TokenKind::PipeForward) {
            let operator = match self.previous().kind {
                TokenKind::Pipe => BinaryOp::Pipe,
                TokenKind::PipeForward => BinaryOp::PipeForward,
                _ => unreachable!(),
            };
            let right = self.logical_or()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, operator, right, span);
        }

        Ok(expr)
    }

    /// Parse logical OR
    fn logical_or(&mut self) -> Result<Node> {
        let mut expr = self.logical_and()?;

        while self.match_token(&TokenKind::Or) {
            let right = self.logical_and()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, BinaryOp::Or, right, span);
        }

        Ok(expr)
    }

    /// Parse logical AND
    fn logical_and(&mut self) -> Result<Node> {
        let mut expr = self.equality()?;

        while self.match_token(&TokenKind::And) {
            let right = self.equality()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, BinaryOp::And, right, span);
        }

        Ok(expr)
    }

    /// Parse equality operations
    fn equality(&mut self) -> Result<Node> {
        let mut expr = self.comparison()?;

        while self.match_token(&TokenKind::Equal) || self.match_token(&TokenKind::NotEqual) {
            let operator = match self.previous().kind {
                TokenKind::Equal => BinaryOp::Equal,
                TokenKind::NotEqual => BinaryOp::NotEqual,
                _ => unreachable!(),
            };
            let right = self.comparison()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, operator, right, span);
        }

        Ok(expr)
    }

    /// Parse comparison operations
    fn comparison(&mut self) -> Result<Node> {
        let mut expr = self.term()?;

        while self.match_token(&TokenKind::GreaterThan)
            || self.match_token(&TokenKind::GreaterThanEqual)
            || self.match_token(&TokenKind::LessThan)
            || self.match_token(&TokenKind::LessThanEqual)
        {
            let operator = match self.previous().kind {
                TokenKind::GreaterThan => BinaryOp::GreaterThan,
                TokenKind::GreaterThanEqual => BinaryOp::GreaterThanEqual,
                TokenKind::LessThan => BinaryOp::LessThan,
                TokenKind::LessThanEqual => BinaryOp::LessThanEqual,
                _ => unreachable!(),
            };
            let right = self.term()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, operator, right, span);
        }

        Ok(expr)
    }

    /// Parse addition and subtraction
    fn term(&mut self) -> Result<Node> {
        let mut expr = self.factor()?;

        while self.match_token(&TokenKind::Plus) || self.match_token(&TokenKind::Minus) {
            let operator = match self.previous().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Subtract,
                _ => unreachable!(),
            };
            let right = self.factor()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, operator, right, span);
        }

        Ok(expr)
    }

    /// Parse multiplication and division
    fn factor(&mut self) -> Result<Node> {
        let mut expr = self.unary()?;

        while self.match_token(&TokenKind::Multiply) || self.match_token(&TokenKind::Divide) {
            let operator = match self.previous().kind {
                TokenKind::Multiply => BinaryOp::Multiply,
                TokenKind::Divide => BinaryOp::Divide,
                _ => unreachable!(),
            };
            let right = self.unary()?;
            let span = self.span_from_to(expr.span(), right.span());
            expr = Node::binary_op(expr, operator, right, span);
        }

        Ok(expr)
    }

    /// Parse unary operations
    fn unary(&mut self) -> Result<Node> {
        if self.match_token(&TokenKind::Not) || self.match_token(&TokenKind::Minus) {
            let start_pos = self.previous().clone();
            let operator = match self.previous().kind {
                TokenKind::Not => UnaryOp::Not,
                TokenKind::Minus => UnaryOp::Negate,
                _ => unreachable!(),
            };
            let operand = self.unary()?;
            let span = self.span_from_to(&start_pos, operand.span());

            Ok(Node::UnaryOp {
                operator,
                operand: Box::new(operand),
                span,
            })
        } else {
            self.call()
        }
    }

    /// Parse function calls and member access
    fn call(&mut self) -> Result<Node> {
        let mut expr = self.primary()?;

        loop {
            if self.match_token(&TokenKind::LeftParen) {
                let mut arguments = Vec::new();

                if !self.check(&TokenKind::RightParen) {
                    loop {
                        arguments.push(self.expression()?);
                        if !self.match_token(&TokenKind::Comma) {
                            break;
                        }
                    }
                }

                let end_pos =
                    self.consume(&TokenKind::RightParen, "Expected ')' after arguments")?;
                let span = self.span_from_to(expr.span(), end_pos);
                expr = Node::call(expr, arguments, span);
            } else if self.match_token(&TokenKind::Dot) {
                let property = self.consume_identifier("Expected property name after '.'")?;
                let span = self.span_from_to(expr.span(), &self.previous());
                expr = Node::Member {
                    object: Box::new(expr),
                    property,
                    span,
                };
            } else {
                break;
            }
        }

        Ok(expr)
    }

    /// Parse primary expressions
    fn primary(&mut self) -> Result<Node> {
        if self.match_token(&TokenKind::True) {
            let token = self.previous();
            return Ok(Node::Boolean {
                value: true,
                span: self.token_span(token),
            });
        }

        if self.match_token(&TokenKind::False) {
            let token = self.previous();
            return Ok(Node::Boolean {
                value: false,
                span: self.token_span(token),
            });
        }

        if self.match_token(&TokenKind::Null) {
            let token = self.previous();
            return Ok(Node::Null {
                span: self.token_span(token),
            });
        }

        if self.match_token(&TokenKind::NumberLiteral) {
            let token = self.previous();
            let value = token
                .lexeme
                .parse::<f64>()
                .map_err(|_| TurbulanceError::syntax(token.line, token.column, "Invalid number"))?;
            return Ok(Node::number(value, self.token_span(token)));
        }

        if self.match_token(&TokenKind::StringLiteral) {
            let token = self.previous();
            let value = token.lexeme[1..token.lexeme.len() - 1].to_string(); // Remove quotes
            return Ok(Node::string(value, self.token_span(token)));
        }

        if self.match_token(&TokenKind::Identifier) {
            let token = self.previous();
            return Ok(Node::identifier(
                token.lexeme.clone(),
                self.token_span(token),
            ));
        }

        if self.match_token(&TokenKind::LeftParen) {
            let expr = self.expression()?;
            self.consume(&TokenKind::RightParen, "Expected ')' after expression")?;
            return Ok(expr);
        }

        if self.match_token(&TokenKind::LeftBracket) {
            return self.array_literal();
        }

        if self.match_token(&TokenKind::LeftBrace) {
            return self.object_literal();
        }

        Err(TurbulanceError::syntax(
            self.peek().line,
            self.peek().column,
            "Expected expression",
        ))
    }

    /// Parse array literal
    fn array_literal(&mut self) -> Result<Node> {
        let start_pos = self.tokens[self.current - 1].clone();
        let mut elements = Vec::new();

        if !self.check(&TokenKind::RightBracket) {
            loop {
                elements.push(self.expression()?);
                if !self.match_token(&TokenKind::Comma) {
                    break;
                }
            }
        }

        let end_pos = self.consume(
            &TokenKind::RightBracket,
            "Expected ']' after array elements",
        )?;
        let span = self.span_from_to(&start_pos, end_pos);

        Ok(Node::Array { elements, span })
    }

    /// Parse object literal
    fn object_literal(&mut self) -> Result<Node> {
        let start_pos = self.tokens[self.current - 1].clone();
        let mut fields = HashMap::new();

        if !self.check(&TokenKind::RightBrace) {
            loop {
                let key = if self.match_token(&TokenKind::Identifier) {
                    self.previous().lexeme.clone()
                } else if self.match_token(&TokenKind::StringLiteral) {
                    let token = self.previous();
                    token.lexeme[1..token.lexeme.len() - 1].to_string() // Remove quotes
                } else {
                    return Err(TurbulanceError::syntax(
                        self.peek().line,
                        self.peek().column,
                        "Expected property name",
                    ));
                };

                self.consume(&TokenKind::Colon, "Expected ':' after property name")?;
                let value = self.expression()?;
                fields.insert(key, value);

                if !self.match_token(&TokenKind::Comma) {
                    break;
                }
            }
        }

        let end_pos = self.consume(&TokenKind::RightBrace, "Expected '}' after object fields")?;
        let span = self.span_from_to(&start_pos, end_pos);

        Ok(Node::Object { fields, span })
    }

    /// Parse function parameters
    fn parse_parameters(&mut self) -> Result<Vec<Parameter>> {
        let mut parameters = Vec::new();

        if !self.check(&TokenKind::RightParen) {
            loop {
                let name = self.consume_identifier("Expected parameter name")?;
                let param = Parameter::new(name);
                parameters.push(param);

                if !self.match_token(&TokenKind::Comma) {
                    break;
                }
            }
        }

        Ok(parameters)
    }

    /// Parse attributes (key-value pairs)
    fn parse_attributes(&mut self) -> Result<HashMap<String, Node>> {
        let mut attributes = HashMap::new();

        if !self.check(&TokenKind::RightParen) {
            loop {
                let key = self.consume_identifier("Expected attribute name")?;
                self.consume(&TokenKind::Colon, "Expected ':' after attribute name")?;
                let value = self.expression()?;
                attributes.insert(key, value);

                if !self.match_token(&TokenKind::Comma) {
                    break;
                }
            }
        }

        Ok(attributes)
    }

    /// Parse a motion
    fn parse_motion(&mut self) -> Result<Motion> {
        let motion_type = self.consume_identifier("Expected motion type")?;

        self.consume(&TokenKind::LeftParen, "Expected '(' after motion type")?;

        // For now, just parse the content as a string
        let content = if self.match_token(&TokenKind::StringLiteral) {
            let token = self.previous();
            token.lexeme[1..token.lexeme.len() - 1].to_string() // Remove quotes
        } else {
            return Err(TurbulanceError::syntax(
                self.peek().line,
                self.peek().column,
                "Expected motion content string",
            ));
        };

        self.consume(&TokenKind::RightParen, "Expected ')' after motion content")?;

        Ok(Motion::new(motion_type, content))
    }

    /// Parse a source
    fn parse_source(&mut self) -> Result<Source> {
        let source_type = self.consume_identifier("Expected source type")?;

        self.consume(&TokenKind::LeftParen, "Expected '(' after source type")?;

        let path = if self.match_token(&TokenKind::StringLiteral) {
            let token = self.previous();
            token.lexeme[1..token.lexeme.len() - 1].to_string() // Remove quotes
        } else {
            return Err(TurbulanceError::syntax(
                self.peek().line,
                self.peek().column,
                "Expected source path string",
            ));
        };

        self.consume(&TokenKind::RightParen, "Expected ')' after source path")?;

        Ok(Source::with_type(path, source_type))
    }

    /// Parse a block or single expression
    fn block_or_expression(&mut self) -> Result<Node> {
        if self.check(&TokenKind::LeftBrace) {
            self.advance(); // consume '{'
            let mut statements = Vec::new();

            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                statements.push(self.declaration()?);
            }

            let end_pos = self.consume(&TokenKind::RightBrace, "Expected '}' after block")?;
            let span = if let Some(first) = statements.first() {
                self.span_from_to(first.span(), end_pos)
            } else {
                self.span_from_to(&self.previous(), end_pos)
            };

            Ok(Node::block(statements, span))
        } else {
            self.statement()
        }
    }

    // Helper methods

    fn match_token(&mut self, token_type: &TokenKind) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, token_type: &TokenKind) -> bool {
        if self.is_at_end() {
            false
        } else {
            &self.peek().kind == token_type
        }
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || self.peek().kind == TokenKind::Eof
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn consume(&mut self, token_type: &TokenKind, message: &str) -> Result<&Token> {
        if self.check(token_type) {
            Ok(self.advance())
        } else {
            let token = self.peek();
            Err(TurbulanceError::syntax(token.line, token.column, message))
        }
    }

    fn consume_identifier(&mut self, message: &str) -> Result<String> {
        if self.match_token(&TokenKind::Identifier) {
            Ok(self.previous().lexeme.clone())
        } else {
            let token = self.peek();
            Err(TurbulanceError::syntax(token.line, token.column, message))
        }
    }

    fn is_statement_end(&self) -> bool {
        self.is_at_end() || self.is_block_end()
    }

    fn is_block_end(&self) -> bool {
        matches!(self.peek().kind, TokenKind::RightBrace | TokenKind::Eof)
    }

    fn token_span(&self, token: &Token) -> Span {
        Span::new(
            Position::new(token.line, token.column, token.span.start),
            Position::new(
                token.line,
                token.column + token.lexeme.len(),
                token.span.end,
            ),
        )
    }

    fn span_from_to(&self, start: &Token, end: &Token) -> Span {
        Span::new(
            Position::new(start.line, start.column, start.span.start),
            Position::new(end.line, end.column + end.lexeme.len(), end.span.end),
        )
    }

    fn span_from_spans(&self, start: &Span, end: &Span) -> Span {
        Span::new(start.start, end.end)
    }

    fn span_from_node_to_token(&self, start: &Node, end: &Token) -> Span {
        Span::new(
            start.span().start,
            Position::new(end.line, end.column + end.lexeme.len(), end.span.end),
        )
    }

    fn span_from_token_to_node(&self, start: &Token, end: &Node) -> Span {
        Span::new(
            Position::new(start.line, start.column, start.span.start),
            end.span().end,
        )
    }
}

/// Convenience function to parse tokens into an AST
pub fn parse(tokens: Vec<Token>) -> Result<Node> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    fn parse_source(source: &str) -> Result<Node> {
        let tokens = tokenize(source)?;
        parse(tokens)
    }

    #[test]
    fn test_parse_number() {
        let ast = parse_source("42").unwrap();
        if let Node::Program { statements, .. } = ast {
            if let Node::ExpressionStatement { expression, .. } = &statements[0] {
                if let Node::Number { value, .. } = expression.as_ref() {
                    assert_eq!(*value, 42.0);
                } else {
                    panic!("Expected number");
                }
            } else {
                panic!("Expected expression statement");
            }
        } else {
            panic!("Expected program");
        }
    }

    #[test]
    fn test_parse_binary_op() {
        let ast = parse_source("1 + 2").unwrap();
        if let Node::Program { statements, .. } = ast {
            if let Node::ExpressionStatement { expression, .. } = &statements[0] {
                if let Node::BinaryOp { operator, .. } = expression.as_ref() {
                    assert_eq!(*operator, BinaryOp::Add);
                } else {
                    panic!("Expected binary operation");
                }
            }
        }
    }

    #[test]
    fn test_parse_function() {
        let ast = parse_source("funxn test(): return 42").unwrap();
        if let Node::Program { statements, .. } = ast {
            if let Node::FunctionDecl { name, .. } = &statements[0] {
                assert_eq!(name, "test");
            } else {
                panic!("Expected function declaration");
            }
        }
    }

    #[test]
    fn test_parse_proposition() {
        let ast = parse_source(
            r#"
            proposition TestHypothesis:
                motion Hypothesis("Test hypothesis")
                given true:
                    return "success"
        "#,
        )
        .unwrap();

        if let Node::Program { statements, .. } = ast {
            if let Node::Proposition { name, motions, .. } = &statements[0] {
                assert_eq!(name, "TestHypothesis");
                assert_eq!(motions.len(), 1);
                assert_eq!(motions[0].motion_type, "Hypothesis");
            } else {
                panic!("Expected proposition declaration");
            }
        }
    }
}
