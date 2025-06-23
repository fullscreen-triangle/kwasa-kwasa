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
        
        // New scientific reasoning constructs
        if self.match_token(&[TokenKind::Proposition]) {
            return self.proposition_declaration();
        }
        
        if self.match_token(&[TokenKind::Evidence]) {
            return self.evidence_declaration();
        }
        
        if self.match_token(&[TokenKind::Pattern]) {
            return self.pattern_declaration();
        }
        
        if self.match_token(&[TokenKind::Meta]) {
            return self.meta_analysis_declaration();
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
        
        // Scientific reasoning statements
        if self.match_token(&[TokenKind::Support]) {
            return self.support_statement();
        }
        
        if self.match_token(&[TokenKind::Contradict]) {
            return self.contradict_statement();
        }
        
        if self.match_token(&[TokenKind::Inconclusive]) {
            return self.inconclusive_statement();
        }
        
        if self.match_token(&[TokenKind::DeriveHypotheses]) {
            return self.derive_hypotheses_statement();
        }
        
        // Advanced orchestration statements
        if self.match_token(&[TokenKind::Flow]) {
            return self.flow_statement();
        }
        
        if self.match_token(&[TokenKind::Catalyze]) {
            return self.catalyze_statement();
        }
        
        if self.match_token(&[TokenKind::CrossScale]) {
            return self.cross_scale_coordinate();
        }
        
        if self.match_token(&[TokenKind::Drift]) {
            return self.drift_statement();
        }
        
        if self.match_token(&[TokenKind::Cycle]) {
            return self.cycle_statement();
        }
        
        if self.match_token(&[TokenKind::Roll]) {
            return self.roll_statement();
        }
        
        if self.match_token(&[TokenKind::Resolve]) {
            return self.resolve_statement();
        }
        
        if self.match_token(&[TokenKind::Point]) {
            return self.point_declaration();
        }
        
        // Autobahn reference statements
        if self.match_token(&[TokenKind::Funxn]) {
            return self.funxn_declaration();
        }
        
        if self.match_token(&[TokenKind::Goal]) {
            return self.goal_declaration();
        }
        
        if self.match_token(&[TokenKind::Metacognitive]) {
            return self.metacognitive_block();
        }
        
        if self.match_token(&[TokenKind::Try]) {
            return self.try_statement();
        }
        
        if self.match_token(&[TokenKind::Parallel]) {
            return self.parallel_block();
        }
        
        if self.match_token(&[TokenKind::QuantumState]) {
            return self.quantum_state_declaration();
        }
        
        if self.match_token(&[TokenKind::OptimizeUntil]) {
            return self.optimize_until_statement();
        }
        
        if self.match_token(&[TokenKind::For]) {
            return self.for_statement();
        }
        
        if self.match_token(&[TokenKind::While]) {
            return self.while_statement();
        }
        
        if self.match_token(&[TokenKind::Import]) {
            return self.import_statement();
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
                    arguments: vec![expr, Node::StringLiteral(name.lexeme.clone(), Span::new(
                        Position::new(0, 0, name_span.start),
                        Position::new(0, 0, name_span.end),
                    ))],
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
    fn considering_all_statement(&mut self, start_span: logos::Span) -> Result<Node, TurbulanceError> {
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
    fn considering_these_statement(&mut self, start_span: logos::Span) -> Result<Node, TurbulanceError> {
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
    fn considering_item_statement(&mut self, start_span: logos::Span) -> Result<Node, TurbulanceError> {
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
        
        let token = self.peek();
        let end_span = token.span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::AllowStmt {
            value: Box::new(value),
            span,
        })
    }
    
    /// Parse a proposition declaration
    fn proposition_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse proposition name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected proposition name"));
        };
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after proposition name")?;
        
        // Parse optional description (first string literal)
        let mut description = None;
        let mut requirements = None;
        let mut body = None;
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            if self.match_token(&[TokenKind::StringLiteral]) {
                if description.is_none() {
                    description = Some(self.previous().lexeme.trim_matches('"').to_string());
                }
            } else if self.match_token(&[TokenKind::Requirements]) {
                self.consume(TokenKind::LeftBrace, "Expected '{' after requirements")?;
                requirements = Some(Box::new(self.structured_data()?));
                self.consume(TokenKind::RightBrace, "Expected '}' after requirements")?;
            } else {
                // Parse other statements as body
                if body.is_none() {
                    let mut statements = Vec::new();
                    while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                        statements.push(self.statement()?);
                    }
                    let end_pos = self.peek().span.end;
                    let body_span = Span::new(
                        Position::new(0, 0, start_span.start),
                        Position::new(0, 0, end_pos),
                    );
                    body = Some(Box::new(Node::Block { statements, span: body_span }));
                    break;
                }
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after proposition body")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::PropositionDecl {
            name,
            description,
            requirements,
            body,
            span,
        })
    }
    
    /// Parse an evidence declaration
    fn evidence_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse evidence name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected evidence name"));
        };
        
        self.consume(TokenKind::Assign, "Expected '=' after evidence name")?;
        
        // Parse collection method (function call)
        let collection_method = self.expression()?;
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after collection method")?;
        
        // Parse data structure
        let data_structure = self.structured_data()?;
        
        self.consume(TokenKind::RightBrace, "Expected '}' after evidence data")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::EvidenceDecl {
            name,
            collection_method: Box::new(collection_method),
            data_structure: Box::new(data_structure),
            span,
        })
    }
    
    /// Parse a pattern declaration
    fn pattern_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse pattern name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected pattern name"));
        };
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after pattern name")?;
        
        // Parse signature
        self.consume(TokenKind::Signature, "Expected 'signature' in pattern")?;
        self.consume(TokenKind::Colon, "Expected ':' after signature")?;
        self.consume(TokenKind::LeftBrace, "Expected '{' after signature:")?;
        let signature = self.structured_data()?;
        self.consume(TokenKind::RightBrace, "Expected '}' after signature")?;
        
        // Parse optional within clause
        let mut within_clause = None;
        if self.match_token(&[TokenKind::Within]) {
            within_clause = Some(Box::new(self.expression()?));
        }
        
        // Parse match clauses
        let mut match_clauses = Vec::new();
        while self.match_token(&[TokenKind::Match]) {
            let condition = self.expression()?;
            self.consume(TokenKind::LeftBrace, "Expected '{' after match condition")?;
            let action = self.structured_data()?;
            self.consume(TokenKind::RightBrace, "Expected '}' after match action")?;
            
            let match_span = Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, self.previous().span.end),
            );
            
            match_clauses.push(ast::MatchClause {
                condition: Box::new(condition),
                action: Box::new(action),
                span: match_span,
            });
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after pattern body")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::PatternDecl {
            name,
            signature: Box::new(signature),
            within_clause,
            match_clauses,
            span,
        })
    }
    
    /// Parse a meta analysis declaration
    fn meta_analysis_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse meta analysis name
        let name = if self.match_token(&[TokenKind::Identifier]) {
            self.previous().lexeme.clone()
        } else {
            return Err(self.error("Expected meta analysis name"));
        };
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after meta analysis name")?;
        
        // Parse studies
        self.consume(TokenKind::Identifier, "Expected 'studies' field")?;
        if self.previous().lexeme != "studies" {
            return Err(self.error("Expected 'studies' field in meta analysis"));
        }
        self.consume(TokenKind::Colon, "Expected ':' after 'studies'")?;
        let studies = self.expression()?;
        
        self.consume(TokenKind::Semicolon, "Expected ';' after studies")?;
        
        // Parse analysis
        let analysis = self.structured_data()?;
        
        self.consume(TokenKind::RightBrace, "Expected '}' after meta analysis body")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::MetaAnalysis {
            name,
            studies: Box::new(studies),
            analysis: Box::new(analysis),
            span,
        })
    }
    
    /// Parse structured data (objects with key-value pairs)
    fn structured_data(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.peek().span.clone();
        let mut fields = HashMap::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            // Parse field name
            let field_name = if self.match_token(&[TokenKind::Identifier]) {
                self.previous().lexeme.clone()
            } else {
                return Err(self.error("Expected field name in structured data"));
            };
            
            self.consume(TokenKind::Colon, "Expected ':' after field name")?;
            
            // Parse field value
            let field_value = if self.match_token(&[TokenKind::LeftBrace]) {
                // Nested structured data
                let nested = self.structured_data()?;
                self.consume(TokenKind::RightBrace, "Expected '}' after nested data")?;
                nested
            } else if self.match_token(&[TokenKind::LeftBracket]) {
                // Array literal
                let mut elements = Vec::new();
                while !self.check(&TokenKind::RightBracket) && !self.is_at_end() {
                    elements.push(self.expression()?);
                    if !self.match_token(&[TokenKind::Comma]) && !self.check(&TokenKind::RightBracket) {
                        return Err(self.error("Expected ',' or ']' in array"));
                    }
                }
                self.consume(TokenKind::RightBracket, "Expected ']' after array elements")?;
                
                let array_span = Span::new(
                    Position::new(0, 0, start_span.start),
                    Position::new(0, 0, self.previous().span.end),
                );
                
                Node::ArrayLiteral {
                    elements,
                    span: array_span,
                }
            } else {
                self.expression()?
            };
            
            fields.insert(field_name, field_value);
            
            // Optional semicolon separator
            self.match_token(&[TokenKind::Semicolon]);
        }
        
        let end_span = self.peek().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::StructuredData { fields, span })
    }
    
    /// Parse a support statement
    fn support_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse hypothesis
        let hypothesis = self.expression()?;
        
        self.consume(TokenKind::With, "Expected 'with' after hypothesis in support statement")?;
        
        // Parse evidence
        self.consume(TokenKind::LeftBrace, "Expected '{' after 'with'")?;
        let evidence = self.structured_data()?;
        self.consume(TokenKind::RightBrace, "Expected '}' after evidence")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::SupportStmt {
            hypothesis: Box::new(hypothesis),
            evidence: Box::new(evidence),
            span,
        })
    }
    
    /// Parse a contradict statement
    fn contradict_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse hypothesis
        let hypothesis = self.expression()?;
        
        self.consume(TokenKind::With, "Expected 'with' after hypothesis in contradict statement")?;
        
        // Parse evidence
        self.consume(TokenKind::LeftBrace, "Expected '{' after 'with'")?;
        let evidence = self.structured_data()?;
        self.consume(TokenKind::RightBrace, "Expected '}' after evidence")?;
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::ContradictStmt {
            hypothesis: Box::new(hypothesis),
            evidence: Box::new(evidence),
            span,
        })
    }
    
    /// Parse an inconclusive statement
    fn inconclusive_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        // Parse message (string literal)
        let message = if self.match_token(&[TokenKind::StringLiteral]) {
            self.previous().lexeme.trim_matches('"').to_string()
        } else {
            return Err(self.error("Expected message string in inconclusive statement"));
        };
        
        // Parse optional recommendations
        let mut recommendations = None;
        if self.match_token(&[TokenKind::With]) {
            self.consume(TokenKind::LeftBrace, "Expected '{' after 'with'")?;
            recommendations = Some(Box::new(self.structured_data()?));
            self.consume(TokenKind::RightBrace, "Expected '}' after recommendations")?;
        }
        
        let end_span = self.previous().span.clone();
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::InconclusiveStmt {
            message,
            recommendations,
            span,
        })
    }
    
    /// Parse a derive hypotheses statement
    fn derive_hypotheses_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'derive_hypotheses'
        
        let mut hypotheses = Vec::new();
        
        while !self.check(&TokenKind::Semicolon) && !self.is_at_end() {
            if let TokenKind::String(hypothesis) = &self.current_token().kind {
                hypotheses.push(hypothesis.clone());
                self.advance();
            } else {
                return Err(self.error("Expected hypothesis string"));
            }
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::Semicolon, "Expected ';' after derive_hypotheses statement")?;
        let end = self.previous().span.end;
        
        Ok(Node::DeriveHypotheses {
            hypotheses,
            span: Span { start, end },
        })
    }

    // Advanced orchestration parsing methods
    fn flow_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let variable = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected variable name after 'flow'"));
        };
        
        self.consume(TokenKind::On, "Expected 'on' after flow variable")?;
        let collection = Box::new(self.expression()?);
        
        self.consume(TokenKind::LeftBrace, "Expected '{' after flow collection")?;
        let body = Box::new(self.block()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Flow(FlowStatement {
            variable,
            collection,
            body,
            span,
        }))
    }

    fn catalyze_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let target = Box::new(self.expression()?);
        self.consume(TokenKind::With, "Expected 'with' after catalyze target")?;
        
        let scale = match &self.peek().kind {
            TokenKind::Quantum => ScaleType::Quantum,
            TokenKind::Molecular => ScaleType::Molecular,
            TokenKind::Environmental => ScaleType::Environmental,
            TokenKind::Hardware => ScaleType::Hardware,
            TokenKind::Cognitive => ScaleType::Cognitive,
            _ => return Err(self.error("Expected valid scale type after 'catalyze'")),
        };
        
        let end_span = self.advance().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Catalyze(CatalyzeStatement {
            target,
            scale,
            span,
        }))
    }

    fn cross_scale_coordinate(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'cross_scale'
        self.consume(TokenKind::Coordinate, "Expected 'coordinate' after 'cross_scale'")?;
        
        let mut pairs = Vec::new();
        
        loop {
            let pair_start = self.current_token().span.start;
            
            let scale1 = match &self.current_token().kind {
                TokenKind::Quantum => ScaleType::Quantum,
                TokenKind::Molecular => ScaleType::Molecular,
                TokenKind::Environmental => ScaleType::Environmental,
                TokenKind::Hardware => ScaleType::Hardware,
                TokenKind::Cognitive => ScaleType::Cognitive,
                _ => return Err(self.error("Expected valid scale type in coordinate pair")),
            };
            self.advance();
            
            self.consume(TokenKind::With, "Expected 'with' between scale types")?;
            
            let scale2 = match &self.current_token().kind {
                TokenKind::Quantum => ScaleType::Quantum,
                TokenKind::Molecular => ScaleType::Molecular,
                TokenKind::Environmental => ScaleType::Environmental,
                TokenKind::Hardware => ScaleType::Hardware,
                TokenKind::Cognitive => ScaleType::Cognitive,
                _ => return Err(self.error("Expected valid scale type in coordinate pair")),
            };
            let pair_end = self.advance().span.end;
            
            pairs.push(CoordinationPair {
                scale1,
                scale2,
                span: Span { start: pair_start, end: pair_end },
            });
            
            if !self.check(&TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        
        let end = self.previous().span.end;
        
        Ok(Node::CrossScaleCoordinate(CrossScaleCoordinate {
            pairs,
            span: Span { start, end },
        }))
    }

    fn drift_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'drift'
        
        let parameters = Box::new(self.expression()?);
        self.consume(TokenKind::Until, "Expected 'until' after drift parameters")?;
        let condition = Box::new(self.expression()?);
        
        let body = Box::new(self.block()?);
        let end = self.previous().span.end;
        
        Ok(Node::Drift(DriftStatement {
            parameters,
            condition,
            body,
            span: Span { start, end },
        }))
    }

    fn cycle_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'cycle'
        
        let variable = self.consume_identifier("Expected variable name after 'cycle'")?;
        self.consume(TokenKind::On, "Expected 'on' after cycle variable")?;
        let collection = Box::new(self.expression()?);
        
        let body = Box::new(self.block()?);
        let end = self.previous().span.end;
        
        Ok(Node::Cycle(CycleStatement {
            variable,
            collection,
            body,
            span: Span { start, end },
        }))
    }

    fn roll_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'roll'
        
        let variable = self.consume_identifier("Expected variable name after 'roll'")?;
        self.consume(TokenKind::Until, "Expected 'until' after roll variable")?;
        let condition = Box::new(self.expression()?);
        
        let body = Box::new(self.block()?);
        let end = self.previous().span.end;
        
        Ok(Node::Roll(RollStatement {
            variable,
            condition,
            body,
            span: Span { start, end },
        }))
    }

    fn resolve_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.advance(); // consume 'resolve'
        
        let function_call = Box::new(self.expression()?);
        
        let context = if self.check(&TokenKind::Given) {
            self.advance(); // consume 'given'
            Some(Box::new(self.expression()?))
        } else {
            None
        };
        
        let end = self.previous().span.end;
        
        Ok(Node::Resolve(ResolveStatement {
            function_call,
            context,
            span: Span { start, end },
        }))
    }

    fn point_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected point name after 'point'"));
        };
        
        self.consume(TokenKind::Assign, "Expected '=' after point name")?;
        
        let properties = Box::new(self.expression()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Point(PointDeclaration {
            name,
            properties,
            span,
        }))
    }

    // Helper method for structured data with parameters
    fn parse_function_with_parameters(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        let name = self.consume_identifier("Expected function name")?;
        
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if let TokenKind::Identifier(param_name) = &self.current_token().kind {
                let param_name = param_name.clone();
                self.advance();
                
                if self.check(&TokenKind::Colon) {
                    self.advance(); // consume ':'
                    let param_value = self.expression()?;
                    parameters.push((param_name, Box::new(param_value)));
                } else {
                    // Positional parameter
                    parameters.push((param_name, Box::new(Node::Identifier { 
                        name: param_name.clone(), 
                        span: self.previous().span 
                    })));
                }
            } else {
                let param_value = self.expression()?;
                parameters.push((String::new(), Box::new(param_value)));
            }
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        let end = self.previous().span.end;
        
        Ok(Node::FunctionCall {
            name,
            args: parameters.into_iter().map(|(_, v)| *v).collect(),
            span: Span { start, end },
        })
    }

    // Helper method for range specifications
    fn parse_range(&mut self) -> Result<Node, TurbulanceError> {
        let start = self.current_token().span.start;
        self.consume(TokenKind::LeftBracket, "Expected '[' for range")?;
        
        let range_start = Box::new(self.expression()?);
        self.consume(TokenKind::Comma, "Expected ',' in range")?;
        let range_end = Box::new(self.expression()?);
        
        self.consume(TokenKind::RightBracket, "Expected ']' after range")?;
        let end = self.previous().span.end;
        
        Ok(Node::ArrayLiteral(ArrayLiteral {
            elements: vec![*range_start, *range_end],
            span: Span { start, end },
        }))
    }

    // Autobahn reference parsing methods
    fn funxn_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected function name after 'funxn'"));
        };
        
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            let param_start = self.current_token().span.start;
            
            let param_name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
                self.advance();
                name
            } else {
                return Err(self.error("Expected parameter name"));
            };
            
            let param_type = if self.check(&TokenKind::Colon) {
                self.advance(); // consume ':'
                if let Some(TokenKind::Identifier(type_name)) = self.peek().kind.clone() {
                    self.advance();
                    Some(type_name)
                } else {
                    None
                }
            } else {
                None
            };
            
            let default_value = if self.check(&TokenKind::Assign) {
                self.advance(); // consume '='
                Some(Box::new(self.expression()?))
            } else {
                None
            };
            
            let param_end = self.previous().span.end;
            
            parameters.push(Parameter {
                name: param_name,
                param_type,
                default_value,
                span: Span { start: param_start, end: param_end },
            });
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;
        self.consume(TokenKind::Colon, "Expected ':' after function signature")?;
        
        let body = Box::new(self.block()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Funxn(FunxnDeclaration {
            name,
            parameters,
            body,
            span,
        }))
    }

    fn goal_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected goal name after 'goal'"));
        };
        
        self.consume(TokenKind::Colon, "Expected ':' after goal name")?;
        
        let mut description = None;
        let mut success_threshold = None;
        let mut metrics = Vec::new();
        let mut subgoals = Vec::new();
        let mut constraints = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            if self.check(&TokenKind::Description) {
                self.advance();
                self.consume(TokenKind::Colon, "Expected ':' after 'description'")?;
                if let Some(TokenKind::String(desc)) = self.peek().kind.clone() {
                    self.advance();
                    description = Some(desc);
                }
            } else if self.check(&TokenKind::SuccessThreshold) {
                self.advance();
                self.consume(TokenKind::Colon, "Expected ':' after 'success_threshold'")?;
                success_threshold = Some(Box::new(self.expression()?));
            } else if self.check(&TokenKind::Metrics) {
                self.advance();
                self.consume(TokenKind::Colon, "Expected ':' after 'metrics'")?;
                // Parse metrics object
                metrics = self.parse_key_value_pairs()?;
            } else if self.check(&TokenKind::Subgoals) {
                self.advance();
                self.consume(TokenKind::Colon, "Expected ':' after 'subgoals'")?;
                // Parse subgoals
                subgoals = self.parse_subgoals()?;
            } else if self.check(&TokenKind::Constraints) {
                self.advance();
                self.consume(TokenKind::Colon, "Expected ':' after 'constraints'")?;
                // Parse constraints array
                constraints = self.parse_constraint_array()?;
            } else {
                break;
            }
        }
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Goal(GoalDeclaration {
            name,
            description,
            success_threshold,
            metrics,
            subgoals,
            constraints,
            span,
        }))
    }

    fn metacognitive_block(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected metacognitive block name"));
        };
        
        self.consume(TokenKind::Colon, "Expected ':' after metacognitive block name")?;
        
        let mut operations = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let operation = match &self.peek().kind {
                TokenKind::TrackReasoning => {
                    self.advance();
                    self.consume(TokenKind::LeftParen, "Expected '(' after track_reasoning")?;
                    if let Some(TokenKind::String(topic)) = self.peek().kind.clone() {
                        self.advance();
                        self.consume(TokenKind::RightParen, "Expected ')' after topic")?;
                        MetacognitiveOperation::TrackReasoning(topic)
                    } else {
                        return Err(self.error("Expected topic string for track_reasoning"));
                    }
                },
                TokenKind::EvaluateConfidence => {
                    self.advance();
                    MetacognitiveOperation::EvaluateConfidence
                },
                TokenKind::DetectBias => {
                    self.advance();
                    self.consume(TokenKind::LeftParen, "Expected '(' after detect_bias")?;
                    if let Some(TokenKind::String(bias_type)) = self.peek().kind.clone() {
                        self.advance();
                        self.consume(TokenKind::RightParen, "Expected ')' after bias type")?;
                        MetacognitiveOperation::DetectBias(bias_type)
                    } else {
                        return Err(self.error("Expected bias type string for detect_bias"));
                    }
                },
                TokenKind::AdaptBehavior => {
                    self.advance();
                    self.consume(TokenKind::LeftParen, "Expected '(' after adapt_behavior")?;
                    if let Some(TokenKind::String(behavior)) = self.peek().kind.clone() {
                        self.advance();
                        self.consume(TokenKind::RightParen, "Expected ')' after behavior")?;
                        MetacognitiveOperation::AdaptBehavior(behavior)
                    } else {
                        return Err(self.error("Expected behavior string for adapt_behavior"));
                    }
                },
                TokenKind::AnalyzeDecisionHistory => {
                    self.advance();
                    MetacognitiveOperation::AnalyzeDecisionHistory
                },
                TokenKind::UpdateDecisionStrategies => {
                    self.advance();
                    MetacognitiveOperation::UpdateDecisionStrategies
                },
                TokenKind::IncreaseEvidenceRequirements => {
                    self.advance();
                    MetacognitiveOperation::IncreaseEvidenceRequirements
                },
                TokenKind::ReduceComputationalOverhead => {
                    self.advance();
                    MetacognitiveOperation::ReduceComputationalOverhead
                },
                _ => break,
            };
            
            operations.push(operation);
        }
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Metacognitive(MetacognitiveBlock {
            name,
            operations,
            span,
        }))
    }

    fn try_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::Colon, "Expected ':' after 'try'")?;
        let try_block = Box::new(self.block()?);
        
        let mut catch_blocks = Vec::new();
        
        while self.check(&TokenKind::Catch) {
            let catch_start = self.advance().span.clone();
            
            let (exception_type, exception_name) = if let Some(TokenKind::Identifier(exc_type)) = self.peek().kind.clone() {
                self.advance();
                let exc_type = exc_type;
                
                let exc_name = if self.check(&TokenKind::As) {
                    self.advance(); // consume 'as'
                    if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
                        self.advance();
                        Some(name)
                    } else {
                        None
                    }
                } else {
                    None
                };
                
                (Some(exc_type), exc_name)
            } else {
                (None, None)
            };
            
            self.consume(TokenKind::Colon, "Expected ':' after catch clause")?;
            let catch_body = Box::new(self.block()?);
            let catch_end = self.previous().span.clone();
            
            catch_blocks.push(CatchBlock {
                exception_type,
                exception_name,
                body: catch_body,
                span: Span::new(
                    Position::new(0, 0, catch_start.start),
                    Position::new(0, 0, catch_end.end),
                ),
            });
        }
        
        let finally_block = if self.check(&TokenKind::Finally) {
            self.advance();
            self.consume(TokenKind::Colon, "Expected ':' after 'finally'")?;
            Some(Box::new(self.block()?))
        } else {
            None
        };
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Try(TryStatement {
            try_block,
            catch_blocks,
            finally_block,
            span,
        }))
    }

    fn parallel_block(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::ParallelExecute, "Expected 'parallel_execute' after 'parallel'")?;
        self.consume(TokenKind::Colon, "Expected ':' after 'parallel_execute'")?;
        
        let mut tasks = Vec::new();
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let task_start = self.current_token().span.start;
            
            let task_name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
                self.advance();
                name
            } else {
                return Err(self.error("Expected task name"));
            };
            
            self.consume(TokenKind::Colon, "Expected ':' after task name")?;
            let task_body = Box::new(self.expression()?);
            let task_end = self.previous().span.end;
            
            tasks.push(ParallelTask {
                name: task_name,
                body: task_body,
                span: Span { start: task_start, end: task_end },
            });
        }
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Parallel(ParallelBlock {
            tasks,
            span,
        }))
    }

    fn quantum_state_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected quantum state name"));
        };
        
        self.consume(TokenKind::Colon, "Expected ':' after quantum state name")?;
        
        let properties = self.parse_key_value_pairs()?;
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::QuantumState(QuantumStateDeclaration {
            name,
            properties,
            span,
        }))
    }

    fn optimize_until_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let condition = Box::new(self.expression()?);
        self.consume(TokenKind::Colon, "Expected ':' after optimize_until condition")?;
        
        let body = Box::new(self.block()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::OptimizeUntil(OptimizeUntilStatement {
            condition,
            body,
            span,
        }))
    }

    fn for_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::Each, "Expected 'each' after 'for'")?;
        
        let variable = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected variable name after 'each'"));
        };
        
        self.consume(TokenKind::In, "Expected 'in' after variable")?;
        let collection = Box::new(self.expression()?);
        self.consume(TokenKind::Colon, "Expected ':' after collection")?;
        
        let body = Box::new(self.block()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::For(ForStatement {
            variable,
            collection,
            body,
            span,
        }))
    }

    fn while_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let condition = Box::new(self.expression()?);
        self.consume(TokenKind::Colon, "Expected ':' after while condition")?;
        
        let body = Box::new(self.block()?);
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::While(WhileStatement {
            condition,
            body,
            span,
        }))
    }

    fn import_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        let module = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else if let Some(TokenKind::String(name)) = self.peek().kind.clone() {
            self.advance();
            name
        } else {
            return Err(self.error("Expected module name after 'import'"));
        };
        
        let (items, alias) = if self.check(&TokenKind::From) {
            // Handle "from module import items" syntax
            let from_items = if self.check(&TokenKind::LeftBrace) {
                self.advance(); // consume '{'
                let mut import_items = Vec::new();
                
                while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                    if let Some(TokenKind::Identifier(item)) = self.peek().kind.clone() {
                        self.advance();
                        import_items.push(item);
                    }
                    
                    if self.check(&TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                
                self.consume(TokenKind::RightBrace, "Expected '}' after import items")?;
                Some(import_items)
            } else {
                None
            };
            
            let import_alias = if self.check(&TokenKind::As) {
                self.advance(); // consume 'as'
                if let Some(TokenKind::Identifier(alias_name)) = self.peek().kind.clone() {
                    self.advance();
                    Some(alias_name)
                } else {
                    None
                }
            } else {
                None
            };
            
            (from_items, import_alias)
        } else {
            (None, None)
        };
        
        let end_span = self.previous().span.clone();
        
        let span = Span::new(
            Position::new(0, 0, start_span.start),
            Position::new(0, 0, end_span.end),
        );
        
        Ok(Node::Import(ImportStatement {
            module,
            items,
            alias,
            span,
        }))
    }

    // Helper methods for parsing complex structures
    fn parse_key_value_pairs(&mut self) -> Result<Vec<(String, Box<Node>)>, TurbulanceError> {
        let mut pairs = Vec::new();
        
        self.consume(TokenKind::LeftBrace, "Expected '{' for key-value pairs")?;
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let key = if let Some(TokenKind::Identifier(key)) = self.peek().kind.clone() {
                self.advance();
                key
            } else if let Some(TokenKind::String(key)) = self.peek().kind.clone() {
                self.advance();
                key
            } else {
                return Err(self.error("Expected key name"));
            };
            
            self.consume(TokenKind::Colon, "Expected ':' after key")?;
            let value = Box::new(self.expression()?);
            
            pairs.push((key, value));
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after key-value pairs")?;
        Ok(pairs)
    }

    fn parse_subgoals(&mut self) -> Result<Vec<SubGoal>, TurbulanceError> {
        let mut subgoals = Vec::new();
        
        self.consume(TokenKind::LeftBrace, "Expected '{' for subgoals")?;
        
        while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
            let subgoal_start = self.current_token().span.start;
            
            let name = if let Some(TokenKind::Identifier(name)) = self.peek().kind.clone() {
                self.advance();
                name
            } else {
                return Err(self.error("Expected subgoal name"));
            };
            
            self.consume(TokenKind::Colon, "Expected ':' after subgoal name")?;
            self.consume(TokenKind::LeftBrace, "Expected '{' for subgoal properties")?;
            
            let mut weight = None;
            let mut threshold = None;
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Weight) {
                    self.advance();
                    self.consume(TokenKind::Colon, "Expected ':' after 'weight'")?;
                    weight = Some(Box::new(self.expression()?));
                } else if self.check(&TokenKind::Threshold) {
                    self.advance();
                    self.consume(TokenKind::Colon, "Expected ':' after 'threshold'")?;
                    threshold = Some(Box::new(self.expression()?));
                } else {
                    break;
                }
                
                if self.check(&TokenKind::Comma) {
                    self.advance();
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after subgoal properties")?;
            let subgoal_end = self.previous().span.end;
            
            subgoals.push(SubGoal {
                name,
                weight,
                threshold,
                span: Span { start: subgoal_start, end: subgoal_end },
            });
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after subgoals")?;
        Ok(subgoals)
    }

    fn parse_constraint_array(&mut self) -> Result<Vec<Box<Node>>, TurbulanceError> {
        let mut constraints = Vec::new();
        
        self.consume(TokenKind::LeftBracket, "Expected '[' for constraints")?;
        
        while !self.check(&TokenKind::RightBracket) && !self.is_at_end() {
            constraints.push(Box::new(self.expression()?));
            
            if self.check(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(TokenKind::RightBracket, "Expected ']' after constraints")?;
        Ok(constraints)
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
