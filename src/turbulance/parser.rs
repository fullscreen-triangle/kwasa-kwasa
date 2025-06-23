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
        
        // Bene Gesserit masterclass constructs
        if self.match_token(&[TokenKind::SuccessFramework]) {
            return self.success_framework_declaration();
        }
        
        if self.match_token(&[TokenKind::BiologicalComputer]) {
            return self.biological_computer_declaration();
        }
        
        if self.match_token(&[TokenKind::PatternAnalysis]) {
            return self.pattern_analysis_block();
        }
        
        if self.match_token(&[TokenKind::SpatiotemporalAnalysis]) {
            return self.spatiotemporal_analysis_block();
        }
        
        if self.match_token(&[TokenKind::DataProcessing]) {
            return self.data_processing_block();
        }
        
        if self.match_token(&[TokenKind::UncertaintyPropagation]) {
            return self.uncertainty_propagation_block();
        }
        
        if self.match_token(&[TokenKind::CausalAnalysis]) {
            return self.causal_analysis_block();
        }
        
        if self.match_token(&[TokenKind::BiasAnalysis]) {
            return self.bias_analysis_block();
        }
        
        if self.match_token(&[TokenKind::QuantumClassicalInterface]) {
            return self.quantum_classical_interface_block();
        }
        
        // Imhotep Framework: Revolutionary Self-Aware Neural Networks
        if self.match_token(&[TokenKind::NeuralConsciousness]) {
            return self.neural_consciousness_declaration();
        }
        
        if self.match_token(&[TokenKind::CreateBmdNeuron]) {
            return self.create_bmd_neuron_statement();
        }
        
        if self.match_token(&[TokenKind::ConnectPattern]) {
            return self.connect_pattern_statement();
        }
        
        if self.match_token(&[TokenKind::ConfigureSelfAwareness]) {
            return self.configure_self_awareness_statement();
        }
        
        if self.match_token(&[TokenKind::ActivateSelfAwareness]) {
            return self.activate_self_awareness_statement();
        }
        
        if self.match_token(&[TokenKind::ProcessWithMetacognitiveMonitoring]) {
            return self.process_with_metacognitive_monitoring_statement();
        }
        
        if self.match_token(&[TokenKind::AssessReasoningQuality]) {
            return self.assess_reasoning_quality_statement();
        }
        
        if self.match_token(&[TokenKind::BeginMetacognitiveReasoning]) {
            return self.begin_metacognitive_reasoning_statement();
        }
        
        if self.match_token(&[TokenKind::AnalyzeWithMetacognitiveOversight]) {
            return self.analyze_with_metacognitive_oversight_statement();
        }
        
        if self.match_token(&[TokenKind::InterpretWithSelfAwareness]) {
            return self.interpret_with_self_awareness_statement();
        }
        
        if self.match_token(&[TokenKind::AnalyzePathwaysWithMetacognition]) {
            return self.analyze_pathways_with_metacognition_statement();
        }
        
        if self.match_token(&[TokenKind::DemonstrateSelfAwarenessVsConsciousness]) {
            return self.demonstrate_self_awareness_vs_consciousness_statement();
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

    fn parse_boolean(&mut self) -> Result<bool, ParseError> {
        if let Some(token) = self.advance() {
            match token.kind {
                TokenKind::True => Ok(true),
                TokenKind::False => Ok(false),
                TokenKind::Identifier(name) if name == "true" => Ok(true),
                TokenKind::Identifier(name) if name == "false" => Ok(false),
                _ => Err(ParseError::UnexpectedToken(format!("Expected boolean value, found {:?}", token.kind))),
            }
        } else {
            Err(ParseError::UnexpectedToken("Expected boolean value".to_string()))
        }
    }

    // Helper parsing methods for Bene Gesserit constructs
    fn parse_quantum_targets(&mut self) -> Result<Vec<QuantumTarget>, ParseError> {
        let mut targets = Vec::new();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                let start_pos = self.current_position();
                
                let name = if let Some(token) = self.advance() {
                    match token.kind {
                        TokenKind::Identifier(name) => name,
                        _ => return Err(ParseError::UnexpectedToken("Expected quantum target name".to_string())),
                    }
                } else {
                    return Err(ParseError::UnexpectedToken("Expected quantum target name".to_string()));
                };
                
                self.consume(TokenKind::Colon, "Expected ':' after quantum target name")?;
                
                let quantum_state = if let Some(token) = self.advance() {
                    match token.kind {
                        TokenKind::String(s) => s,
                        TokenKind::Identifier(name) => name,
                        _ => return Err(ParseError::UnexpectedToken("Expected quantum state".to_string())),
                    }
                } else {
                    return Err(ParseError::UnexpectedToken("Expected quantum state".to_string()));
                };
                
                let end_pos = self.current_position();
                
                targets.push(QuantumTarget {
                    name,
                    quantum_state,
                    span: Span::new(start_pos, end_pos),
                });
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
                break;
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after quantum targets")?;
        }
        
        Ok(targets)
    }

    fn parse_oscillatory_dynamics(&mut self) -> Result<Vec<OscillatoryDynamic>, ParseError> {
        let mut dynamics = Vec::new();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                let start_pos = self.current_position();
                
                let name = if let Some(token) = self.advance() {
                    match token.kind {
                        TokenKind::Identifier(name) => name,
                        _ => return Err(ParseError::UnexpectedToken("Expected oscillatory dynamic name".to_string())),
                    }
                } else {
                    return Err(ParseError::UnexpectedToken("Expected oscillatory dynamic name".to_string()));
                };
                
                self.consume(TokenKind::Colon, "Expected ':' after oscillatory dynamic name")?;
                
                let frequency = self.expression()?;
                
                let end_pos = self.current_position();
                
                dynamics.push(OscillatoryDynamic {
                    name,
                    frequency,
                    span: Span::new(start_pos, end_pos),
                });
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
                break;
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after oscillatory dynamics")?;
        }
        
        Ok(dynamics)
    }

    fn parse_molecular_patterns(&mut self) -> Result<MolecularPatternAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut binding_pose_clustering = None;
        let mut pharmacophore_identification = None;
        let mut admet_pattern_detection = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::BindingPoseClustering]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'binding_pose_clustering'")?;
                    binding_pose_clustering = Some(self.parse_clustering_parameters()?);
                } else if self.match_token(&[TokenKind::PharmacophoreIdentification]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'pharmacophore_identification'")?;
                    pharmacophore_identification = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::AdmetPatternDetection]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'admet_pattern_detection'")?;
                    admet_pattern_detection = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after molecular patterns")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(MolecularPatternAnalysis {
            binding_pose_clustering,
            pharmacophore_identification,
            admet_pattern_detection,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_clustering_parameters(&mut self) -> Result<ClusteringParameters, ParseError> {
        let start_pos = self.current_position();
        
        let method = if let Some(token) = self.advance() {
            match token.kind {
                TokenKind::Identifier(name) => name,
                TokenKind::String(s) => s,
                _ => return Err(ParseError::UnexpectedToken("Expected clustering method".to_string())),
            }
        } else {
            return Err(ParseError::UnexpectedToken("Expected clustering method".to_string()));
        };
        
        let mut eps = None;
        let mut min_samples = None;
        
        if self.match_token(&[TokenKind::LeftParen]) {
            while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                if self.match_token(&[TokenKind::Eps]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'eps'")?;
                    eps = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::MinSamples]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'min_samples'")?;
                    min_samples = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightParen, "Expected ')' after clustering parameters")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(ClusteringParameters {
            method,
            eps,
            min_samples,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_clinical_patterns(&mut self) -> Result<ClinicalPatternAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut responder_phenotyping = None;
        let mut disease_progression_trajectories = None;
        let mut adverse_event_clustering = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::ResponderPhenotyping]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'responder_phenotyping'")?;
                    responder_phenotyping = Some(self.parse_gaussian_mixture_parameters()?);
                } else if self.match_token(&[TokenKind::DiseaseProgressionTrajectories]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'disease_progression_trajectories'")?;
                    disease_progression_trajectories = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::AdverseEventClustering]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'adverse_event_clustering'")?;
                    adverse_event_clustering = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after clinical patterns")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(ClinicalPatternAnalysis {
            responder_phenotyping,
            disease_progression_trajectories,
            adverse_event_clustering,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_gaussian_mixture_parameters(&mut self) -> Result<GaussianMixtureParameters, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftParen]) {
            self.consume(TokenKind::NComponents, "Expected 'n_components' parameter")?;
            self.consume(TokenKind::Colon, "Expected ':' after 'n_components'")?;
            let n_components = self.expression()?;
            self.consume(TokenKind::RightParen, "Expected ')' after gaussian mixture parameters")?;
            
            let end_pos = self.current_position();
            
            Ok(GaussianMixtureParameters {
                n_components,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '(' after gaussian mixture model".to_string()))
        }
    }

    fn parse_omics_integration(&mut self) -> Result<OmicsIntegrationAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut multi_block_pls = None;
        let mut network_medicine_analysis = None;
        let mut pathway_enrichment = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::MultiBlockPls]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'multi_block_pls'")?;
                    multi_block_pls = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::NetworkMedicineAnalysis]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'network_medicine_analysis'")?;
                    network_medicine_analysis = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::PathwayEnrichment]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'pathway_enrichment'")?;
                    pathway_enrichment = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after omics integration")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(OmicsIntegrationAnalysis {
            multi_block_pls,
            network_medicine_analysis,
            pathway_enrichment,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_spatial_modeling(&mut self) -> Result<SpatialModelingAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut local_adaptation = None;
        let mut environmental_gradients = None;
        let mut population_structure = None;
        let mut migration_patterns = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::LocalAdaptation]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'local_adaptation'")?;
                    local_adaptation = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::EnvironmentalGradients]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'environmental_gradients'")?;
                    environmental_gradients = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::PopulationStructure]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'population_structure'")?;
                    population_structure = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::MigrationPatterns]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'migration_patterns'")?;
                    migration_patterns = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after spatial modeling")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(SpatialModelingAnalysis {
            local_adaptation,
            environmental_gradients,
            population_structure,
            migration_patterns,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_temporal_modeling(&mut self) -> Result<TemporalModelingAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut evolutionary_trajectories = None;
        let mut selection_dynamics = None;
        let mut demographic_inference = None;
        let mut cultural_evolution = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::EvolutionaryTrajectories]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'evolutionary_trajectories'")?;
                    evolutionary_trajectories = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::SelectionDynamics]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'selection_dynamics'")?;
                    selection_dynamics = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::DemographicInference]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'demographic_inference'")?;
                    demographic_inference = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::CulturalEvolution]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'cultural_evolution'")?;
                    cultural_evolution = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after temporal modeling")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(TemporalModelingAnalysis {
            evolutionary_trajectories,
            selection_dynamics,
            demographic_inference,
            cultural_evolution,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_association_analysis(&mut self) -> Result<AssociationAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        let mut environmental_gwas = None;
        let mut polygenic_adaptation = None;
        let mut balancing_selection = None;
        let mut introgression_analysis = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::EnvironmentalGwas]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'environmental_gwas'")?;
                    environmental_gwas = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::PolygenicAdaptation]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'polygenic_adaptation'")?;
                    polygenic_adaptation = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::BalancingSelection]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'balancing_selection'")?;
                    balancing_selection = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::IntrogressionAnalysis]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'introgression_analysis'")?;
                    introgression_analysis = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after association analysis")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(AssociationAnalysis {
            environmental_gwas,
            polygenic_adaptation,
            balancing_selection,
            introgression_analysis,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_quality_control(&mut self) -> Result<QualityControlParameters, ParseError> {
        let start_pos = self.current_position();
        
        let mut missing_data_threshold = None;
        let mut outlier_detection = None;
        let mut batch_effect_correction = None;
        let mut technical_replicate_correlation = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::MissingDataThreshold]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'missing_data_threshold'")?;
                    missing_data_threshold = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::OutlierDetection]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'outlier_detection'")?;
                    outlier_detection = Some(self.parse_outlier_detection_parameters()?);
                } else if self.match_token(&[TokenKind::BatchEffectCorrection]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'batch_effect_correction'")?;
                    batch_effect_correction = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::TechnicalReplicateCorrelation]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'technical_replicate_correlation'")?;
                    technical_replicate_correlation = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after quality control")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(QualityControlParameters {
            missing_data_threshold,
            outlier_detection,
            batch_effect_correction,
            technical_replicate_correlation,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_outlier_detection_parameters(&mut self) -> Result<OutlierDetectionParameters, ParseError> {
        let start_pos = self.current_position();
        
        let method = if let Some(token) = self.advance() {
            match token.kind {
                TokenKind::Identifier(name) => name,
                TokenKind::String(s) => s,
                _ => return Err(ParseError::UnexpectedToken("Expected outlier detection method".to_string())),
            }
        } else {
            return Err(ParseError::UnexpectedToken("Expected outlier detection method".to_string()));
        };
        
        let mut contamination = None;
        
        if self.match_token(&[TokenKind::LeftParen]) {
            while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
                if self.match_token(&[TokenKind::Contamination]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'contamination'")?;
                    contamination = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightParen, "Expected ')' after outlier detection parameters")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(OutlierDetectionParameters {
            method,
            contamination,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_harmonization(&mut self) -> Result<HarmonizationParameters, ParseError> {
        let start_pos = self.current_position();
        
        let mut unit_standardization = None;
        let mut temporal_alignment = None;
        let mut population_stratification = None;
        let mut covariate_adjustment = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::UnitStandardization]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'unit_standardization'")?;
                    unit_standardization = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::TemporalAlignment]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'temporal_alignment'")?;
                    temporal_alignment = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::PopulationStratification]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'population_stratification'")?;
                    population_stratification = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::CovariateAdjustment]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'covariate_adjustment'")?;
                    covariate_adjustment = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after harmonization")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(HarmonizationParameters {
            unit_standardization,
            temporal_alignment,
            population_stratification,
            covariate_adjustment,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_feature_engineering(&mut self) -> Result<FeatureEngineeringParameters, ParseError> {
        let start_pos = self.current_position();
        
        let mut molecular_descriptors = None;
        let mut clinical_composite_scores = None;
        let mut time_series_features = None;
        let mut network_features = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::MolecularDescriptors]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'molecular_descriptors'")?;
                    molecular_descriptors = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::ClinicalCompositeScores]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'clinical_composite_scores'")?;
                    clinical_composite_scores = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::TimeSeriesFeatures]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'time_series_features'")?;
                    time_series_features = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::NetworkFeatures]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'network_features'")?;
                    network_features = Some(self.expression()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after feature engineering")?;
        }
        
        let end_pos = self.current_position();
        
        Ok(FeatureEngineeringParameters {
            molecular_descriptors,
            clinical_composite_scores,
            time_series_features,
            network_features,
            span: Span::new(start_pos, end_pos),
        })
    }

    fn parse_uncertainty_component(&mut self) -> Result<UncertaintyComponent, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut source = String::new();
            let mut quantification = String::new();
            let mut propagation = String::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "source" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'source'")?;
                                    source = self.parse_string_value()?;
                                }
                                "quantification" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'quantification'")?;
                                    quantification = self.parse_string_value()?;
                                }
                                "propagation" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'propagation'")?;
                                    propagation = self.parse_string_value()?;
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after uncertainty component")?;
            
            let end_pos = self.current_position();
            
            Ok(UncertaintyComponent {
                source,
                quantification,
                propagation,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for uncertainty component".to_string()))
        }
    }

    fn parse_causal_method(&mut self) -> Result<CausalMethod, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut method = String::new();
            let mut adjustment = None;
            let mut validation = None;
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "method" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'method'")?;
                                    method = self.parse_string_value()?;
                                }
                                "adjustment" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'adjustment'")?;
                                    adjustment = Some(self.parse_string_value()?);
                                }
                                "validation" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'validation'")?;
                                    validation = Some(self.parse_string_value()?);
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after causal method")?;
            
            let end_pos = self.current_position();
            
            Ok(CausalMethod {
                method,
                adjustment,
                validation,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for causal method".to_string()))
        }
    }

    fn parse_bias_component(&mut self) -> Result<BiasComponent, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut detection = String::new();
            let mut severity_assessment = String::new();
            let mut mitigation = String::new();
            let mut monitoring = String::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "detection" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'detection'")?;
                                    detection = self.parse_string_value()?;
                                }
                                "severity_assessment" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'severity_assessment'")?;
                                    severity_assessment = self.parse_string_value()?;
                                }
                                "mitigation" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'mitigation'")?;
                                    mitigation = self.parse_string_value()?;
                                }
                                "monitoring" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'monitoring'")?;
                                    monitoring = self.parse_string_value()?;
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after bias component")?;
            
            let end_pos = self.current_position();
            
            Ok(BiasComponent {
                detection,
                severity_assessment,
                mitigation,
                monitoring,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for bias component".to_string()))
        }
    }

    fn parse_coherence_analysis(&mut self) -> Result<CoherenceAnalysis, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut coherence_time_measurement = String::new();
            let mut decoherence_pathway_analysis = String::new();
            let mut environmental_coupling_analysis = String::new();
            let mut coherence_protection_mechanisms = String::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "coherence_time_measurement" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'coherence_time_measurement'")?;
                                    coherence_time_measurement = self.parse_string_value()?;
                                }
                                "decoherence_pathway_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'decoherence_pathway_analysis'")?;
                                    decoherence_pathway_analysis = self.parse_string_value()?;
                                }
                                "environmental_coupling_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'environmental_coupling_analysis'")?;
                                    environmental_coupling_analysis = self.parse_string_value()?;
                                }
                                "coherence_protection_mechanisms" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'coherence_protection_mechanisms'")?;
                                    coherence_protection_mechanisms = self.parse_string_value()?;
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after coherence analysis")?;
            
            let end_pos = self.current_position();
            
            Ok(CoherenceAnalysis {
                coherence_time_measurement,
                decoherence_pathway_analysis,
                environmental_coupling_analysis,
                coherence_protection_mechanisms,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for coherence analysis".to_string()))
        }
    }

    fn parse_neural_quantum_correlation(&mut self) -> Result<NeuralQuantumCorrelation, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut phase_locking_analysis = String::new();
            let mut quantum_neural_synchronization = String::new();
            let mut information_theoretic_analysis = String::new();
            let mut causal_analysis = String::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "phase_locking_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'phase_locking_analysis'")?;
                                    phase_locking_analysis = self.parse_string_value()?;
                                }
                                "quantum_neural_synchronization" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'quantum_neural_synchronization'")?;
                                    quantum_neural_synchronization = self.parse_string_value()?;
                                }
                                "information_theoretic_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'information_theoretic_analysis'")?;
                                    information_theoretic_analysis = self.parse_string_value()?;
                                }
                                "causal_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'causal_analysis'")?;
                                    causal_analysis = self.parse_string_value()?;
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after neural quantum correlation")?;
            
            let end_pos = self.current_position();
            
            Ok(NeuralQuantumCorrelation {
                phase_locking_analysis,
                quantum_neural_synchronization,
                information_theoretic_analysis,
                causal_analysis,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for neural quantum correlation".to_string()))
        }
    }

    fn parse_consciousness_classification(&mut self) -> Result<ConsciousnessClassification, ParseError> {
        let start_pos = self.current_position();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            let mut machine_learning_classification = String::new();
            let mut bayesian_state_estimation = String::new();
            let mut hidden_markov_modeling = String::new();
            let mut neural_network_analysis = String::new();
            
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.check(&TokenKind::Identifier) {
                    if let Some(token) = self.peek() {
                        if let TokenKind::Identifier(name) = &token.kind {
                            match name.as_str() {
                                "machine_learning_classification" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'machine_learning_classification'")?;
                                    machine_learning_classification = self.parse_string_value()?;
                                }
                                "bayesian_state_estimation" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'bayesian_state_estimation'")?;
                                    bayesian_state_estimation = self.parse_string_value()?;
                                }
                                "hidden_markov_modeling" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'hidden_markov_modeling'")?;
                                    hidden_markov_modeling = self.parse_string_value()?;
                                }
                                "neural_network_analysis" => {
                                    self.advance();
                                    self.consume(TokenKind::Colon, "Expected ':' after 'neural_network_analysis'")?;
                                    neural_network_analysis = self.parse_string_value()?;
                                }
                                _ => break,
                            }
                        }
                    }
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after consciousness classification")?;
            
            let end_pos = self.current_position();
            
            Ok(ConsciousnessClassification {
                machine_learning_classification,
                bayesian_state_estimation,
                hidden_markov_modeling,
                neural_network_analysis,
                span: Span::new(start_pos, end_pos),
            })
        } else {
            Err(ParseError::UnexpectedToken("Expected '{' for consciousness classification".to_string()))
        }
    }

    fn parse_string_value(&mut self) -> Result<String, ParseError> {
        if let Some(token) = self.advance() {
            match token.kind {
                TokenKind::String(s) => Ok(s),
                TokenKind::Identifier(name) => Ok(name),
                _ => Err(ParseError::UnexpectedToken(format!("Expected string value, found {:?}", token.kind))),
            }
        } else {
            Err(ParseError::UnexpectedToken("Expected string value".to_string()))
        }
    }

    // Imhotep Framework: Revolutionary Self-Aware Neural Networks Parser Methods
    
    fn neural_consciousness_declaration(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'neural_consciousness'")?;
        
        let mut session_name = String::new();
        let mut consciousness_level = None;
        let mut self_awareness = false;
        let mut metacognitive_monitoring = false;
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if self.match_token(&[TokenKind::SessionName]) {
                self.consume(TokenKind::Colon, "Expected ':' after 'session_name'")?;
                session_name = self.string_literal()?;
            } else if self.match_token(&[TokenKind::ConsciousnessLevel]) {
                self.consume(TokenKind::Colon, "Expected ':' after 'consciousness_level'")?;
                consciousness_level = Some(self.expression()?);
            } else if self.match_token(&[TokenKind::SelfAwareness]) {
                self.consume(TokenKind::Colon, "Expected ':' after 'self_awareness'")?;
                self_awareness = self.parse_boolean()?;
            } else if self.match_token(&[TokenKind::MetacognitiveMonitoring]) {
                self.consume(TokenKind::Colon, "Expected ':' after 'metacognitive_monitoring'")?;
                metacognitive_monitoring = self.parse_boolean()?;
            } else {
                break;
            }
            
            if self.match_token(&[TokenKind::Comma]) {
                continue;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after neural consciousness parameters")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::NeuralConsciousnessDecl(
            NeuralConsciousnessSession {
                session_name,
                consciousness_level: Box::new(consciousness_level.unwrap_or(Node::NumberLiteral(0.9, start_span.clone()))),
                self_awareness,
                metacognitive_monitoring,
                bmd_neurons: Vec::new(),
                neural_connections: Vec::new(),
                self_awareness_config: None,
                span: Span::new(
                    Position::new(0, 0, start_span.start),
                    Position::new(0, 0, end_span.end),
                ),
            }
        )))
    }
    
    fn create_bmd_neuron_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'create_bmd_neuron'")?;
        
        let name = self.string_literal()?;
        self.consume(TokenKind::Comma, "Expected ',' after neuron name")?;
        
        let mut activation = String::new();
        let mut parameters = Vec::new();
        let mut subsystem = String::new();
        let mut question = String::new();
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::Activation]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'activation'")?;
                    activation = self.string_literal()?;
                } else if self.match_token(&[TokenKind::Subsystem]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'subsystem'")?;
                    subsystem = self.string_literal()?;
                } else if self.match_token(&[TokenKind::Question]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'question'")?;
                    question = self.string_literal()?;
                } else if self.check(&TokenKind::Identifier) {
                    // Parse other neuron parameters
                    let param_name = self.string_literal()?;
                    self.consume(TokenKind::Colon, "Expected ':' after parameter name")?;
                    let param_value = self.expression()?;
                    parameters.push(NeuronParameter {
                        name: param_name,
                        value: Box::new(param_value),
                        span: self.previous().span.clone(),
                    });
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after neuron parameters")?;
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after create_bmd_neuron")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::CreateBmdNeuron {
            name,
            activation,
            parameters,
            subsystem,
            question,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn connect_pattern_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'connect_pattern'")?;
        
        let mut connections = Vec::new();
        
        if self.match_token(&[TokenKind::LeftBracket]) {
            while !self.check(&TokenKind::RightBracket) && !self.is_at_end() {
                if self.match_token(&[TokenKind::LeftParen]) {
                    let from_neuron = self.string_literal()?;
                    self.consume(TokenKind::Comma, "Expected ',' after from neuron")?;
                    let to_neuron = self.string_literal()?;
                    self.consume(TokenKind::Comma, "Expected ',' after to neuron")?;
                    let connection_type = self.string_literal()?;
                    self.consume(TokenKind::RightParen, "Expected ')' after connection")?;
                    
                    connections.push(NeuralConnection {
                        from_neuron,
                        to_neuron,
                        connection_type,
                        span: self.previous().span.clone(),
                    });
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBracket, "Expected ']' after connections")?;
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after connect_pattern")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::ConnectPattern {
            connections,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn configure_self_awareness_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'configure_self_awareness'")?;
        
        let mut metacognitive_depth = None;
        let mut self_reflection_threshold = None;
        let mut thought_quality_standards = None;
        let mut knowledge_audit_frequency = None;
        let mut reasoning_chain_logging = None;
        let mut decision_trail_persistence = None;
        
        if self.match_token(&[TokenKind::LeftBrace]) {
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if self.match_token(&[TokenKind::MetacognitiveDepth]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'metacognitive_depth'")?;
                    metacognitive_depth = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::SelfReflectionThreshold]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'self_reflection_threshold'")?;
                    self_reflection_threshold = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::ThoughtQualityStandards]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'thought_quality_standards'")?;
                    thought_quality_standards = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::KnowledgeAuditFrequency]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'knowledge_audit_frequency'")?;
                    knowledge_audit_frequency = Some(self.expression()?);
                } else if self.match_token(&[TokenKind::ReasoningChainLogging]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'reasoning_chain_logging'")?;
                    reasoning_chain_logging = Some(self.parse_boolean()?);
                } else if self.match_token(&[TokenKind::DecisionTrailPersistence]) {
                    self.consume(TokenKind::Colon, "Expected ':' after 'decision_trail_persistence'")?;
                    decision_trail_persistence = Some(self.parse_boolean()?);
                } else {
                    break;
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after self-awareness config")?;
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after configure_self_awareness")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::ConfigureSelfAwareness(
            SelfAwarenessConfiguration {
                metacognitive_depth: metacognitive_depth.map(Box::new),
                self_reflection_threshold: self_reflection_threshold.map(Box::new),
                thought_quality_standards: thought_quality_standards.map(Box::new),
                knowledge_audit_frequency: knowledge_audit_frequency.map(Box::new),
                reasoning_chain_logging,
                decision_trail_persistence,
                span: Span::new(
                    Position::new(0, 0, start_span.start),
                    Position::new(0, 0, end_span.end),
                ),
            }
        )))
    }
    
    fn activate_self_awareness_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'activate_self_awareness'")?;
        let session = self.string_literal()?;
        self.consume(TokenKind::RightParen, "Expected ')' after session name")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::ActivateSelfAwareness {
            session,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn process_with_metacognitive_monitoring_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'process_with_metacognitive_monitoring'")?;
        
        let mut data = None;
        let mut processing_steps = Vec::new();
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if self.check(&TokenKind::Identifier) {
                if let Some(token) = self.peek() {
                    if let TokenKind::Identifier(name) = &token.kind {
                        match name.as_str() {
                            "data" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'data'")?;
                                data = Some(self.expression()?);
                            }
                            "processing_steps" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'processing_steps'")?;
                                processing_steps = self.string_array()?;
                            }
                            _ => break,
                        }
                    }
                }
            } else {
                break;
            }
            
            if self.match_token(&[TokenKind::Comma]) {
                continue;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after process_with_metacognitive_monitoring")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::ProcessWithMetacognitiveMonitoring {
            data: Box::new(data.unwrap_or(Node::StringLiteral("null".to_string(), start_span.clone()))),
            processing_steps,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn assess_reasoning_quality_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'assess_reasoning_quality'")?;
        let session = self.string_literal()?;
        self.consume(TokenKind::RightParen, "Expected ')' after session name")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::AssessReasoningQuality {
            session,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn begin_metacognitive_reasoning_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'begin_metacognitive_reasoning'")?;
        let session = self.string_literal()?;
        self.consume(TokenKind::Comma, "Expected ',' after session name")?;
        let analysis_name = self.string_literal()?;
        self.consume(TokenKind::RightParen, "Expected ')' after analysis name")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::BeginMetacognitiveReasoning {
            session,
            analysis_name,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn analyze_with_metacognitive_oversight_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'analyze_with_metacognitive_oversight'")?;
        
        let mut data = None;
        let mut analysis_type = String::new();
        let mut metacognitive_monitoring = false;
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if self.check(&TokenKind::Identifier) {
                if let Some(token) = self.peek() {
                    if let TokenKind::Identifier(name) = &token.kind {
                        match name.as_str() {
                            "data" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'data'")?;
                                data = Some(self.expression()?);
                            }
                            "analysis_type" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'analysis_type'")?;
                                analysis_type = self.string_literal()?;
                            }
                            "metacognitive_monitoring" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'metacognitive_monitoring'")?;
                                metacognitive_monitoring = self.parse_boolean()?;
                            }
                            _ => break,
                        }
                    }
                }
            } else {
                break;
            }
            
            if self.match_token(&[TokenKind::Comma]) {
                continue;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after analyze_with_metacognitive_oversight")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::AnalyzeWithMetacognitiveOversight {
            data: Box::new(data.unwrap_or(Node::StringLiteral("null".to_string(), start_span.clone()))),
            analysis_type,
            metacognitive_monitoring,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn interpret_with_self_awareness_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'interpret_with_self_awareness'")?;
        
        let mut results = None;
        let mut interpretation_context = String::new();
        let mut uncertainty_tracking = false;
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if self.check(&TokenKind::Identifier) {
                if let Some(token) = self.peek() {
                    if let TokenKind::Identifier(name) = &token.kind {
                        match name.as_str() {
                            "results" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'results'")?;
                                results = Some(self.expression()?);
                            }
                            "interpretation_context" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'interpretation_context'")?;
                                interpretation_context = self.string_literal()?;
                            }
                            "uncertainty_tracking" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'uncertainty_tracking'")?;
                                uncertainty_tracking = self.parse_boolean()?;
                            }
                            _ => break,
                        }
                    }
                }
            } else {
                break;
            }
            
            if self.match_token(&[TokenKind::Comma]) {
                continue;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after interpret_with_self_awareness")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::InterpretWithSelfAwareness {
            results: Box::new(results.unwrap_or(Node::StringLiteral("null".to_string(), start_span.clone()))),
            interpretation_context,
            uncertainty_tracking,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn analyze_pathways_with_metacognition_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'analyze_pathways_with_metacognition'")?;
        
        let mut metabolites = None;
        let mut self_reflection = false;
        let mut knowledge_gap_detection = false;
        
        while !self.check(&TokenKind::RightParen) && !self.is_at_end() {
            if self.check(&TokenKind::Identifier) {
                if let Some(token) = self.peek() {
                    if let TokenKind::Identifier(name) = &token.kind {
                        match name.as_str() {
                            "metabolites" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'metabolites'")?;
                                metabolites = Some(self.expression()?);
                            }
                            "self_reflection" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'self_reflection'")?;
                                self_reflection = self.parse_boolean()?;
                            }
                            "knowledge_gap_detection" => {
                                self.advance();
                                self.consume(TokenKind::Colon, "Expected ':' after 'knowledge_gap_detection'")?;
                                knowledge_gap_detection = self.parse_boolean()?;
                            }
                            _ => break,
                        }
                    }
                }
            } else {
                break;
            }
            
            if self.match_token(&[TokenKind::Comma]) {
                continue;
            }
        }
        
        self.consume(TokenKind::RightParen, "Expected ')' after analyze_pathways_with_metacognition")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::AnalyzePathwaysWithMetacognition {
            metabolites: Box::new(metabolites.unwrap_or(Node::StringLiteral("null".to_string(), start_span.clone()))),
            self_reflection,
            knowledge_gap_detection,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
    }
    
    fn demonstrate_self_awareness_vs_consciousness_statement(&mut self) -> Result<Node, TurbulanceError> {
        let start_span = self.previous().span.clone();
        
        self.consume(TokenKind::LeftParen, "Expected '(' after 'demonstrate_self_awareness_vs_consciousness'")?;
        let traditional_session = self.string_literal()?;
        self.consume(TokenKind::Comma, "Expected ',' after traditional session")?;
        let self_aware_session = self.string_literal()?;
        self.consume(TokenKind::RightParen, "Expected ')' after self-aware session")?;
        
        let end_span = self.previous().span.clone();
        
        Ok(Node::SelfAware(SelfAwareStatement::DemonstrateSelfAwarenessVsConsciousness {
            traditional_session,
            self_aware_session,
            span: Span::new(
                Position::new(0, 0, start_span.start),
                Position::new(0, 0, end_span.end),
            ),
        }))
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

    fn import_statement(&mut self) -> Result<Box<Node>, ParseError> {
        let start_pos = self.current_position();
        
        // 'import' already consumed
        let mut items = Vec::new();
        let mut alias = None;
        
        // Parse import items
        if self.match_token(&[TokenKind::LeftBrace]) {
            // Selective import: import { item1, item2, item3 }
            while !self.check(&TokenKind::RightBrace) && !self.is_at_end() {
                if let Some(token) = self.advance() {
                    if let TokenKind::Identifier(name) = token.kind {
                        items.push(name);
                    }
                }
                
                if self.match_token(&[TokenKind::Comma]) {
                    continue;
                }
                break;
            }
            
            self.consume(TokenKind::RightBrace, "Expected '}' after import items")?;
        } else {
            // Single import
            if let Some(token) = self.advance() {
                if let TokenKind::Identifier(name) = token.kind {
                    items.push(name);
                }
            }
        }
        
        // Check for 'as' alias
        if self.match_token(&[TokenKind::As]) {
            if let Some(token) = self.advance() {
                if let TokenKind::Identifier(name) = token.kind {
                    alias = Some(name);
                }
            }
        }
        
        // Parse 'from' clause
        self.consume(TokenKind::From, "Expected 'from' after import items")?;
        
        let module = if let Some(token) = self.advance() {
            match token.kind {
                TokenKind::String(s) => s,
                TokenKind::Identifier(name) => name,
                _ => return Err(ParseError::UnexpectedToken(format!("Expected module name, found {:?}", token.kind))),
            }
        } else {
            return Err(ParseError::UnexpectedToken("Expected module name".to_string()));
        };
        
        let end_pos = self.current_position();
        
        Ok(Box::new(Node {
            kind: NodeKind::Import(ImportStatement {
                items,
                module,
                alias,
                span: Span::new(start_pos, end_pos),
            }),
            span: Span::new(start_pos, end_pos),
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
