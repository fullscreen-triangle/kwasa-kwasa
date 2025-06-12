use crate::turbulance::ast::{self, Node, BinaryOp, UnaryOp, Parameter, Span, Position, Source};
use crate::turbulance::lexer::{Token, TokenKind};
use crate::turbulance::TurbulanceError;
use std::collections::HashMap;

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            current: 0,
        }
    }

    /// Parse the tokens into an AST
    pub fn parse(&mut self) -> Result<Node, TurbulanceError> {
        let mut statements = Vec::new();
        
        while !self.is_at_end() {
            statements.push(self.declaration()?);
        }
        
        // Create a span for the entire program
        let start_pos = if let Some(first_token) = self.tokens.first() {
            Position::new(0, 0, first_token.span.start)
        } else {
            Position::new(0, 0, 0)
        };
        
        let end_pos = if let Some(last_token) = self.tokens.last() {
            Position::new(0, 0, last_token.span.end)
        } else {
            Position::new(0, 0, 0)
        };
        
        let span = Span::new(start_pos, end_pos);
        
        Ok(ast::program(statements, span))
    }
    
    /// Parse a declaration
    fn declaration(&mut self) -> Result<Node, TurbulanceError> {
        if self.match_token(&[TokenKind::FunctionDecl]) {
            return self.function_declaration();
        }
        
        if self.match_token(&[TokenKind::ProjectDecl]) {
            return self.project_declaration();
        }
        
        if self.match_token(&[TokenKind::SourcesDecl]) {
            return self.sources_declaration();
        }
        
        if self.match_token(&[TokenKind::Cause]) {
            return self.cause_declaration();
        }
        
        if self.match_token(&[TokenKind::Motion]) {
            return self.motion_declaration();
        }
        
        self.statement()
    }
    
    /// Parse a function declaration
    fn function_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse function name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected function name"));
        };
        
        // Parse parameters
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let parameters = self.parameters()?;
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        
        // Parse function body
        self.consume(TokenKind::Colon, "Expected ':' after function parameters")?;
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::FunctionDecl {
            name,
            parameters,
            body: Box::new(body),
            span,
        })
    }
    
    /// Parse a project declaration
    fn project_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse project name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected project name"));
        };
        
        // Parse attributes
        self.consume(TokenKind::LeftParen, "Expected '(' after project name")?;
        let attributes = self.attributes()?;
        self.consume(TokenKind::RightParen, "Expected ')' after attributes")?;
        
        // Parse project body
        self.consume(TokenKind::Colon, "Expected ':' after project attributes")?;
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::ProjectDecl {
            name,
            attributes,
            body: Box::new(body),
            span,
        })
    }
    
    /// Parse a sources declaration
    fn sources_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse sources
        self.consume(TokenKind::Colon, "Expected ':' after 'sources'")?;
        let mut sources = Vec::new();
        
        // Parse source entries
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            sources.push(self.source()?);
        }
        
        let token = self.peek();
        let end_span = token.span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::SourcesDecl {
            sources,
            span,
        })
    }
    
    /// Parse a source entry
    fn source(&mut self) -> Result<Source, TurbulanceError> {
        if self.match_token(&[TokenKind::Identifier]) {
            let source_type = self.previous().lexeme.clone();
            
            self.consume(TokenKind::LeftParen, "Expected '(' after source type")?;
            
            match source_type.as_str() {
                "local" => {
                    let path = self.string_literal()?;
                    self.consume(TokenKind::RightParen, "Expected ')' after local path")?;
                    Ok(Source {
                        path,
                        source_type: Some("local".to_string()),
                    })
                },
                "web_search" => {
                    self.consume(TokenKind::Identifier, "Expected 'engines' parameter")?;
                    self.consume(TokenKind::Assign, "Expected '=' after 'engines'")?;
                    let engines = self.string_array()?;
                    self.consume(TokenKind::RightParen, "Expected ')' after engines")?;
                    Ok(Source {
                        path: engines.join(","),
                        source_type: Some("web_search".to_string()),
                    })
                },
                "knowledge_base" => {
                    let name = self.string_literal()?;
                    self.consume(TokenKind::RightParen, "Expected ')' after knowledge base name")?;
                    Ok(Source {
                        path: name,
                        source_type: Some("knowledge_base".to_string()),
                    })
                },
                "domain_experts" => {
                    let experts = self.string_array()?;
                    self.consume(TokenKind::RightParen, "Expected ')' after domain experts")?;
                    Ok(Source {
                        path: experts.join(","),
                        source_type: Some("domain_experts".to_string()),
                    })
                },
                _ => Err(self.error(&format!("Unknown source type: {}", source_type)))
            }
        } else {
            Err(self.error("Expected source type"))
        }
    }
    
    /// Parse a statement
    fn statement(&mut self) -> Result<Node, TurbulanceError> {
        if self.match_token(&[TokenKind::Return]) {
            return self.return_statement();
        }
        
        if self.match_token(&[TokenKind::Within]) {
            return self.within_statement();
        }
        
        if self.match_token(&[TokenKind::Given]) {
            return self.given_statement();
        }
        
        if self.match_token(&[TokenKind::Ensure]) {
            return self.ensure_statement();
        }
        
        if self.match_token(&[TokenKind::Research]) {
            return self.research_statement();
        }
        
        if self.match_token(&[TokenKind::For]) {
            return self.for_statement();
        }
        
        if self.match_token(&[TokenKind::Considering]) {
            return self.considering_statement();
        }
        
        if self.match_token(&[TokenKind::Allow]) {
            return self.allow_statement();
        }
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            return self.block();
        }
        
        self.expression_statement()
    }
    
    /// Parse a block of statements
    fn block(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        let mut statements = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            statements.push(self.declaration()?);
        }
        
        let end_token = self.consume(TokenKind::RightBrace, "Expected '}' after block")?;
        let end_span = end_token.span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Block {
            statements,
            span,
        })
    }
    
    /// Parse a return statement
    fn return_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let value = if !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            Some(Box::new(self.expression()?))
        } else {
            None
        };
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::ReturnStmt {
            value,
            span,
        })
    }
    
    /// Parse a within statement
    fn within_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let target = Box::new(self.expression()?);
        
        self.consume(TokenKind::Colon, "Expected ':' after target expression")?;
        
        let body = Box::new(self.block()?);
        
        let end_span = if let Node::Block { span, .. } = &*body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::WithinBlock {
            target,
            body,
            span,
        })
    }
    
    /// Parse a given statement
    fn given_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let condition = Box::new(self.expression()?);
        
        self.consume(TokenKind::Colon, "Expected ':' after condition")?;
        
        let body = Box::new(self.block()?);
        
        let end_span = if let Node::Block { span, .. } = &*body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::GivenBlock {
            condition,
            body,
            span,
        })
    }
    
    /// Parse an ensure statement
    fn ensure_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let condition = Box::new(self.expression()?);
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::EnsureStmt {
            condition,
            span,
        })
    }
    
    /// Parse a research statement
    fn research_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let query = Box::new(self.expression()?);
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::ResearchStmt {
            query,
            span,
        })
    }
    
    /// Parse an expression statement
    fn expression_statement(&mut self) -> Result<Node, TurbulanceError> {
        let expr = self.expression()?;
        
        Ok(expr)
    }
    
    /// Parse an expression
    fn expression(&mut self) -> Result<Node, TurbulanceError> {
        self.assignment()
    }
    
    /// Parse an assignment expression
    fn assignment(&mut self) -> Result<Node, TurbulanceError> {
        let expr = self.pipe()?;
        
        if self.match_token(&[TokenKind::Assign]) {
            let equals_token = self.previous().clone(); // Clone the token to avoid the borrow conflict
            let value = Box::new(self.assignment()?);
            
            match expr {
                Node::Identifier(name, span) => {
                    return Ok(Node::Assignment {
                        target: Box::new(Node::Identifier(name, span.clone())),
                        value,
                        span: Span::new(
                            Position::new(0, 0, span.start.offset),
                            Position::new(0, 0, self.previous().span.end),
                        ),
                    });
                },
                _ => return Err(self.error_at_token(&equals_token, "Invalid assignment target")),
            }
        }
        
        Ok(expr)
    }
    
    /// Parse a pipe expression
    fn pipe(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.or()?;
        
        while self.match_token(&[TokenKind::Pipe, TokenKind::PipeForward]) {
            let operator = match self.previous().kind {
                TokenKind::Pipe => BinaryOp::Pipe,
                TokenKind::PipeForward => BinaryOp::PipeForward,
                _ => unreachable!(),
            };
            
            let right = Box::new(self.or()?);
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse an or expression
    fn or(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.and()?;
        
        while self.match_token(&[TokenKind::Or]) {
            let operator = BinaryOp::Or;
            let right = Box::new(self.and()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse an and expression
    fn and(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.equality()?;
        
        while self.match_token(&[TokenKind::And]) {
            let operator = BinaryOp::And;
            let right = Box::new(self.equality()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse an equality expression
    fn equality(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.comparison()?;
        
        while self.match_token(&[TokenKind::Equal, TokenKind::NotEqual]) {
            let operator = match self.previous().kind {
                TokenKind::Equal => BinaryOp::Equal,
                TokenKind::NotEqual => BinaryOp::NotEqual,
                _ => unreachable!(),
            };
            
            let right = Box::new(self.comparison()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse a comparison expression
    fn comparison(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.term()?;
        
        while self.match_token(&[TokenKind::LessThan, TokenKind::GreaterThan, TokenKind::LessThanEqual, TokenKind::GreaterThanEqual]) {
            let operator = match self.previous().kind {
                TokenKind::LessThan => BinaryOp::LessThan,
                TokenKind::GreaterThan => BinaryOp::GreaterThan,
                TokenKind::LessThanEqual => BinaryOp::LessThanEqual,
                TokenKind::GreaterThanEqual => BinaryOp::GreaterThanEqual,
                _ => unreachable!(),
            };
            
            let right = Box::new(self.term()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse a term expression
    fn term(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.factor()?;
        
        while self.match_token(&[TokenKind::Plus, TokenKind::Minus]) {
            let operator = match self.previous().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Subtract,
                _ => unreachable!(),
            };
            
            let right = Box::new(self.factor()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse a factor expression
    fn factor(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.unary()?;
        
        while self.match_token(&[TokenKind::Multiply, TokenKind::Divide]) {
            let operator = match self.previous().kind {
                TokenKind::Multiply => BinaryOp::Multiply,
                TokenKind::Divide => BinaryOp::Divide,
                _ => unreachable!(),
            };
            
            let right = Box::new(self.unary()?);
            
            let span = Span::new(
                if let Some(start) = expr.span() {
                    start.start
                } else {
                    Position::new(0, 0, 0)
                },
                if let Some(end) = right.span() {
                    end.end
                } else {
                    Position::new(0, 0, 0)
                },
            );
            
            expr = Node::BinaryExpr {
                left: Box::new(expr),
                operator,
                right,
                span,
            };
        }
        
        Ok(expr)
    }
    
    /// Parse a unary expression
    fn unary(&mut self) -> Result<Node, TurbulanceError> {
        if self.match_token(&[TokenKind::Not, TokenKind::Minus]) {
            let operator = match self.previous().kind {
                TokenKind::Not => UnaryOp::Not,
                TokenKind::Minus => UnaryOp::Negate,
                _ => unreachable!(),
            };
            
            let start_span = self.previous().span.clone();
            let operand = Box::new(self.unary()?);
            
            let end_span = if let Some(span) = operand.span() {
                span.end
            } else {
                Position::new(0, 0, 0)
            };
            
            let span = Span::new(
                Position::new(0, 0, start_span.start),
                end_span,
            );
            
            return Ok(Node::UnaryExpr {
                operator,
                operand,
                span,
            });
        }
        
        self.call()
    }
    
    /// Parse a function call
    fn call(&mut self) -> Result<Node, TurbulanceError> {
        let mut expr = self.primary()?;
        
        loop {
            if self.match_token(&[TokenKind::LeftParen]) {
                expr = self.finish_call(expr)?;
            } else if self.match_token(&[TokenKind::Dot]) {
                let name = self.consume(TokenKind::Identifier, "Expected property name after '.'")?;
                
                let name_span = name.span.clone();
                
                let span = Span::new(
                    if let Some(start) = expr.span() {
                        start.start
                    } else {
                        Position::new(0, 0, 0)
                    },
                    Position::new(0, 0, name_span.end),
                );
                
                // For now, we'll treat property access as a special kind of function call
                expr = Node::FunctionCall {
                    function: Box::new(Node::Identifier("property_access".to_string(), span.clone())),
                    arguments: vec![expr, Node::StringLiteral(name.lexeme, name_span)],
                    span,
                };
            } else {
                break;
            }
        }
        
        Ok(expr)
    }
    
    /// Finish parsing a function call
    fn finish_call(&mut self, callee: Node) -> Result<Node, TurbulanceError> {
        let start_span = if let Some(span) = callee.span() {
            span.start
        } else {
            Position::new(0, 0, 0)
        };
        
        let mut arguments = Vec::new();
        
        if !self.check(&TokenKind::RightParen) {
            loop {
                arguments.push(self.expression()?);
                
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        let end_token = self.consume(TokenKind::RightParen, "Expected ')' after arguments")?;
        let end_span = end_token.span.clone();
        
        let span = Span::new(
            start_span,
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::FunctionCall {
            function: Box::new(callee),
            arguments,
            span,
        })
    }
    
    /// Parse a primary expression
    fn primary(&mut self) -> Result<Node, TurbulanceError> {
        if self.match_token(&[TokenKind::StringLiteral]) {
            let lexeme = self.previous().lexeme.clone();
            let span = self.previous().span.clone();
            
            // Remove the quotes from the string literal
            let content = lexeme[1..lexeme.len()-1].to_string();
            
            return Ok(Node::StringLiteral(content, Span::new(
                Position::new(0, 0, span.start),
                Position::new(0, 0, span.end),
            )));
        }
        
        if self.match_token(&[TokenKind::NumberLiteral]) {
            let lexeme = self.previous().lexeme.clone();
            let span = self.previous().span.clone();
            
            let number = lexeme.parse::<f64>().map_err(|_| {
                self.error(&format!("Invalid number: {}", lexeme))
            })?;
            
            return Ok(Node::NumberLiteral(number, Span::new(
                Position::new(0, 0, span.start),
                Position::new(0, 0, span.end),
            )));
        }
        
        if self.match_token(&[TokenKind::Identifier]) {
            let name = self.previous().lexeme.clone();
            let span = self.previous().span.clone();
            
            return Ok(Node::Identifier(name, Span::new(
                Position::new(0, 0, span.start),
                Position::new(0, 0, span.end),
            )));
        }
        
        if self.match_token(&[TokenKind::LeftParen]) {
            let expr = self.expression()?;
            self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
            return Ok(expr);
        }
        
        Err(self.error("Expected expression"))
    }
    
    /// Parse function parameters
    fn parameters(&mut self) -> Result<Vec<Parameter>, TurbulanceError> {
        let mut parameters = Vec::new();
        
        if !self.check(&TokenKind::RightParen) {
            loop {
                let name = self.consume(TokenKind::Identifier, "Expected parameter name")?.lexeme.clone();
                let name_span = self.previous().span.clone();
                
                let default_value = if self.match_token(&[TokenKind::Assign]) {
                    Some(self.expression()?)
                } else {
                    None
                };
                
                let end_span = if let Some(ref expr) = default_value {
                    if let Some(span) = expr.span() {
                        span.end
                    } else {
                        Position::new(0, 0, name_span.end)
                    }
                } else {
                    Position::new(0, 0, name_span.end)
                };
                
                parameters.push(Parameter {
                    name,
                    default_value,
                    span: Span::new(
                        Position::new(0, 0, name_span.start),
                        end_span,
                    ),
                });
                
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        Ok(parameters)
    }
    
    /// Parse project attributes
    fn attributes(&mut self) -> Result<HashMap<String, Node>, TurbulanceError> {
        let mut attributes = HashMap::new();
        
        if !self.check(&TokenKind::RightParen) {
            loop {
                let name = self.consume(TokenKind::Identifier, "Expected attribute name")?.lexeme.clone();
                self.consume(TokenKind::Assign, "Expected '=' after attribute name")?;
                let value = self.expression()?;
                
                attributes.insert(name, value);
                
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        Ok(attributes)
    }
    
    /// Parse a string literal
    fn string_literal(&mut self) -> Result<String, TurbulanceError> {
        let token = self.consume(TokenKind::StringLiteral, "Expected string literal")?;
        let lexeme = token.lexeme.clone();
        
        // Remove the quotes from the string literal
        Ok(lexeme[1..lexeme.len()-1].to_string())
    }
    
    /// Parse an array of string literals
    fn string_array(&mut self) -> Result<Vec<String>, TurbulanceError> {
        self.consume(TokenKind::LeftBracket, "Expected '['")?;
        
        let mut strings = Vec::new();
        
        if !self.check(&TokenKind::RightBracket) {
            loop {
                strings.push(self.string_literal()?);
                
                if !self.match_token(&[TokenKind::Comma]) {
                    break;
                }
            }
        }
        
        self.consume(TokenKind::RightBracket, "Expected ']'")?;
        
        Ok(strings)
    }
    
    /// Check if the current token matches any of the given types
    fn match_token(&mut self, types: &[TokenKind]) -> bool {
        for token_type in types {
            if self.check(token_type) {
                self.advance();
                return true;
            }
        }
        
        false
    }
    
    /// Check if the current token is of the given type
    fn check(&self, token_type: &TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        
        &self.peek().kind == token_type
    }
    
    /// Advance to the next token and return the previous one
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        
        self.previous()
    }
    
    /// Check if we've reached the end of the token stream
    fn is_at_end(&self) -> bool {
        self.current >= self.tokens.len() || matches!(self.peek().kind, TokenKind::Error)
    }
    
    /// Get the current token
    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }
    
    /// Get the previous token
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    
    /// Consume a token of the expected type or return an error
    fn consume(&mut self, token_type: TokenKind, message: &str) -> Result<&Token, TurbulanceError> {
        if self.check(&token_type) {
            return Ok(self.advance());
        }
        
        Err(self.error_at_token(self.peek(), message))
    }
    
    /// Create an error at the current position
    fn error(&self, message: &str) -> TurbulanceError {
        self.error_at_token(self.peek(), message)
    }
    
    /// Create an error at the given token
    fn error_at_token(&self, token: &Token, message: &str) -> TurbulanceError {
        TurbulanceError::SyntaxError {
            position: token.span.start,
            message: message.to_string(),
        }
    }

    fn cause_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = self.consume(TokenKind::Identifier, "Expected cause name after 'cause'")?
            .lexeme.clone();
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after cause name")?;
        
        let content = self.block()?;
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::CauseDecl {
            name,
            value: Box::new(content),
            span,
        })
    }
}

impl Node {
    /// Get the span of a node, if available
    fn span(&self) -> Option<Span> {
        match self {
            Node::NumberLiteral(_, span) => Some(*span),
            Node::StringLiteral(_, span) => Some(*span),
            Node::BoolLiteral(_, span) => Some(*span),
            Node::Identifier(_, span) => Some(*span),
            Node::BinaryExpr { span, .. } => Some(*span),
            Node::UnaryExpr { span, .. } => Some(*span),
            Node::FunctionCall { span, .. } => Some(*span),
            Node::IfExpr { span, .. } => Some(*span),
            Node::Block { span, .. } => Some(*span),
            Node::Assignment { span, .. } => Some(*span),
            Node::ReturnStmt { span, .. } => Some(*span),
            Node::WithinBlock { span, .. } => Some(*span),
            Node::GivenBlock { span, .. } => Some(*span),
            Node::EnsureStmt { span, .. } => Some(*span),
            Node::ResearchStmt { span, .. } => Some(*span),
            Node::FunctionDecl { span, .. } => Some(*span),
            Node::ProjectDecl { span, .. } => Some(*span),
            Node::SourcesDecl { span, .. } => Some(*span),
            Node::ForEach { span, .. } => Some(*span),
            Node::ConsideringAll { span, .. } => Some(*span),
            Node::ConsideringThese { span, .. } => Some(*span),
            Node::ConsideringItem { span, .. } => Some(*span),
            Node::Motion { span, .. } => Some(*span),
            Node::CauseDecl { span, .. } => Some(*span),
            Node::AllowStmt { span, .. } => Some(*span),
            Node::Error(_, span) => Some(*span),
            // Add a catch-all for any future variants
            _ => None,
        }
    }
}

impl Parser {
    /// Parse a motion declaration
    fn motion_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = self.consume(TokenKind::Identifier, "Expected motion name after 'motion'")?
            .lexeme.clone();
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after motion name")?;
        
        let content = self.block()?;
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Motion {
            name,
            content: Box::new(content),
            span,
        })
    }

    /// Parse a for statement
    fn for_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::Each, "Expected 'each' after 'for'")?;
        
        let variable = self.consume(TokenKind::Identifier, "Expected variable name after 'each'")?
            .lexeme.clone();
        
        self.consume(TokenKind::In, "Expected 'in' after variable name")?;
        
        let iterable = self.expression()?;
        
        self.consume(TokenKind::Colon, "Expected ':' after iterable expression")?;
        
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::ForEach {
            iterable: Box::new(iterable),
            variable,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a considering statement (replaces for each)
    fn considering_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Determine which type of considering statement we have
        if self.match_token(&[TokenKind::All]) {
            return self.considering_all_statement(start_span);
        } else if self.match_token(&[TokenKind::These]) {
            return self.considering_these_statement(start_span);
        } else if self.match_token(&[TokenKind::Item]) {
            return self.considering_item_statement(start_span);
        } else {
            return Err(self.error("Expected 'all', 'these', or 'item' after 'considering'"));
        }
    }

    /// Parse a "considering all X" statement
    fn considering_all_statement(&mut self, start_span: crate::turbulance::lexer::Span) -> Result<Node, TurbulanceError> {
        let variable = self.consume(TokenKind::Identifier, "Expected variable name after 'all'")?
            .lexeme.clone();
        
        self.consume(TokenKind::In, "Expected 'in' after variable name")?;
        
        let iterable = self.expression()?;
        
        self.consume(TokenKind::Colon, "Expected ':' after iterable expression")?;
        
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::ConsideringAll {
            iterable: Box::new(iterable),
            variable,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a "considering these X" statement
    fn considering_these_statement(&mut self, start_span: crate::turbulance::lexer::Span) -> Result<Node, TurbulanceError> {
        let variable = self.consume(TokenKind::Identifier, "Expected variable name after 'these'")?
            .lexeme.clone();
        
        self.consume(TokenKind::In, "Expected 'in' after variable name")?;
        
        let iterable = self.expression()?;
        
        self.consume(TokenKind::Colon, "Expected ':' after iterable expression")?;
        
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::ConsideringThese {
            iterable: Box::new(iterable),
            variable,
            body: Box::new(body),
            span,
        })
    }

    /// Parse a "considering item X" statement
    fn considering_item_statement(&mut self, start_span: crate::turbulance::lexer::Span) -> Result<Node, TurbulanceError> {
        let variable = self.consume(TokenKind::Identifier, "Expected variable name after 'item'")?
            .lexeme.clone();
        
        self.consume(TokenKind::Colon, "Expected ':' after variable name")?;
        
        let item = self.expression()?;
        
        self.consume(TokenKind::Colon, "Expected ':' after item expression")?;
        
        let body = self.block()?;
        
        let end_span = if let Node::Block { span, .. } = &body {
            span.end
        } else {
            return Err(self.error("Expected block"));
        };
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.offset),
        );
        
        Ok(Node::ConsideringItem {
            item: Box::new(item),
            variable,
            body: Box::new(body),
            span,
        })
    }

    /// Parse an "allow" statement (replaces "let")
    fn allow_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let value = self.expression()?;
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::AllowStmt {
            value: Box::new(value),
            span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::lexer::Lexer;
    
    fn parse(source: &str) -> Result<Node, TurbulanceError> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse()
    }
    
    #[test]
    fn test_parse_simple_expression() {
        let result = parse("40 + 2");
        assert!(result.is_ok());
        
        match result.unwrap() {
            Node::Block { statements, .. } => {
                assert_eq!(statements.len(), 1);
                
                match &statements[0] {
                    Node::BinaryExpr { operator, .. } => {
                        assert_eq!(*operator, BinaryOp::Add);
                    },
                    _ => panic!("Expected BinaryExpr"),
                }
            },
            _ => panic!("Expected Block"),
        }
    }
    
    #[test]
    fn test_parse_function_declaration() {
        let result = parse("funxn test(a, b = 42): return a + b");
        assert!(result.is_ok());
        
        match result.unwrap() {
            Node::Block { statements, .. } => {
                assert_eq!(statements.len(), 1);
                
                match &statements[0] {
                    Node::FunctionDecl { name, parameters, .. } => {
                        assert_eq!(name, "test");
                        assert_eq!(parameters.len(), 2);
                        assert_eq!(parameters[0].name, "a");
                        assert_eq!(parameters[1].name, "b");
                        
                        // Check that the second parameter has a default value
                        assert!(parameters[1].default_value.is_some());
                    },
                    _ => panic!("Expected FunctionDecl"),
                }
            },
            _ => panic!("Expected Block"),
        }
    }
    
    #[test]
    fn test_parse_within_block() {
        let result = parse("
            within paragraph:
                given contains(\"technical_term\"):
                    ensure has_explanation()
        ");
        assert!(result.is_ok());
        
        match result.unwrap() {
            Node::Block { statements, .. } => {
                assert_eq!(statements.len(), 1);
                
                match &statements[0] {
                    Node::WithinBlock { .. } => {
                        // Successfully parsed
                    },
                    _ => panic!("Expected WithinBlock"),
                }
            },
            _ => panic!("Expected Block"),
        }
    }
}
