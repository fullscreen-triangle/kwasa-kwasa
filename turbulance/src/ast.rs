//! Abstract Syntax Tree definitions for Turbulance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Position in source code
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Byte offset in source
    pub offset: usize,
}

impl Position {
    /// Create a new position
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }
}

/// Span representing a range in source code
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Start position
    pub start: Position,
    /// End position
    pub end: Position,
}

impl Span {
    /// Create a new span
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }
}

/// AST node representing any language construct
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    // Literals
    Number {
        value: f64,
        span: Span,
    },
    String {
        value: String,
        span: Span,
    },
    Boolean {
        value: bool,
        span: Span,
    },
    Null {
        span: Span,
    },
    Identifier {
        name: String,
        span: Span,
    },

    // Collections
    Array {
        elements: Vec<Node>,
        span: Span,
    },
    Object {
        fields: HashMap<String, Node>,
        span: Span,
    },

    // Binary operations
    BinaryOp {
        left: Box<Node>,
        operator: BinaryOp,
        right: Box<Node>,
        span: Span,
    },

    // Unary operations
    UnaryOp {
        operator: UnaryOp,
        operand: Box<Node>,
        span: Span,
    },

    // Function call
    Call {
        callee: Box<Node>,
        arguments: Vec<Node>,
        span: Span,
    },

    // Member access
    Member {
        object: Box<Node>,
        property: String,
        span: Span,
    },

    // Variable assignment
    Assignment {
        target: Box<Node>,
        value: Box<Node>,
        span: Span,
    },

    // Function declaration
    FunctionDecl {
        name: String,
        parameters: Vec<Parameter>,
        body: Box<Node>,
        span: Span,
    },

    // Project declaration
    ProjectDecl {
        name: String,
        attributes: HashMap<String, Node>,
        body: Box<Node>,
        span: Span,
    },

    // Proposition
    Proposition {
        name: String,
        motions: Vec<Motion>,
        body: Box<Node>,
        span: Span,
    },

    // Sources declaration
    SourcesDecl {
        sources: Vec<Source>,
        span: Span,
    },

    // Control flow
    If {
        condition: Box<Node>,
        then_branch: Box<Node>,
        else_branch: Option<Box<Node>>,
        span: Span,
    },

    Given {
        condition: Box<Node>,
        then_branch: Box<Node>,
        else_branch: Option<Box<Node>>,
        span: Span,
    },

    Within {
        target: Box<Node>,
        body: Box<Node>,
        span: Span,
    },

    Considering {
        items: Box<Node>,
        body: Box<Node>,
        span: Span,
    },

    Ensure {
        condition: Box<Node>,
        span: Span,
    },

    // Statements
    Return {
        value: Option<Box<Node>>,
        span: Span,
    },

    Research {
        query: Box<Node>,
        span: Span,
    },

    Block {
        statements: Vec<Node>,
        span: Span,
    },

    // Expression statement
    ExpressionStatement {
        expression: Box<Node>,
        span: Span,
    },

    // Program root
    Program {
        statements: Vec<Node>,
        span: Span,
    },

    // Text operations (semantic)
    TextOperation {
        operation: TextOp,
        target: Box<Node>,
        arguments: Vec<Node>,
        span: Span,
    },

    // Points and resolutions
    Point {
        name: String,
        properties: HashMap<String, Node>,
        span: Span,
    },

    Resolution {
        name: String,
        parameters: Vec<Parameter>,
        body: Box<Node>,
        return_type: Option<String>,
        span: Span,
    },
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,

    // Comparison
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,

    // Logical
    And,
    Or,

    // Semantic operations
    Pipe,
    PipeForward,
    Arrow,

    // Scientific operations
    SemanticAdd,      // Meaningful combination
    SemanticSubtract, // Removal of elements
    SemanticMultiply, // Amplification/repetition
    SemanticDivide,   // Extraction/filtering
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Negate,
    Not,
    Plus,
}

/// Text operations for semantic processing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextOp {
    Simplify,
    Expand,
    Formalize,
    Informalize,
    Translate,
    Summarize,
    Extract,
    Rewrite,
    Understand,
    Clarify,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Optional type annotation
    pub type_annotation: Option<String>,
    /// Optional default value
    pub default_value: Option<Node>,
}

impl Parameter {
    /// Create a new parameter
    pub fn new(name: String) -> Self {
        Self {
            name,
            type_annotation: None,
            default_value: None,
        }
    }

    /// Create a parameter with type annotation
    pub fn with_type(name: String, type_annotation: String) -> Self {
        Self {
            name,
            type_annotation: Some(type_annotation),
            default_value: None,
        }
    }

    /// Create a parameter with default value
    pub fn with_default(name: String, default_value: Node) -> Self {
        Self {
            name,
            type_annotation: None,
            default_value: Some(default_value),
        }
    }
}

/// Motion in a proposition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Motion {
    /// Motion type (e.g., "Hypothesis", "Procedure")
    pub motion_type: String,
    /// Motion content/description
    pub content: String,
    /// Optional parameters
    pub parameters: Option<HashMap<String, Node>>,
}

impl Motion {
    /// Create a new motion
    pub fn new(motion_type: String, content: String) -> Self {
        Self {
            motion_type,
            content,
            parameters: None,
        }
    }

    /// Create a motion with parameters
    pub fn with_parameters(
        motion_type: String,
        content: String,
        parameters: HashMap<String, Node>,
    ) -> Self {
        Self {
            motion_type,
            content,
            parameters: Some(parameters),
        }
    }
}

/// Source declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    /// Source path or identifier
    pub path: String,
    /// Source type (local, web_search, database, etc.)
    pub source_type: Option<String>,
    /// Additional parameters
    pub parameters: Option<HashMap<String, Node>>,
}

impl Source {
    /// Create a new source
    pub fn new(path: String) -> Self {
        Self {
            path,
            source_type: None,
            parameters: None,
        }
    }

    /// Create a typed source
    pub fn with_type(path: String, source_type: String) -> Self {
        Self {
            path,
            source_type: Some(source_type),
            parameters: None,
        }
    }
}

/// Text unit for semantic operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextUnit {
    /// The text content
    pub content: String,
    /// Semantic metadata
    pub metadata: HashMap<String, String>,
    /// Processing confidence (0.0 to 1.0)
    pub confidence: f64,
}

impl TextUnit {
    /// Create a new text unit
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
            confidence: 1.0,
        }
    }

    /// Create a text unit with metadata
    pub fn with_metadata(content: String, metadata: HashMap<String, String>) -> Self {
        Self {
            content,
            metadata,
            confidence: 1.0,
        }
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

impl fmt::Display for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl Node {
    /// Get the span of this node
    pub fn span(&self) -> &Span {
        match self {
            Node::Number { span, .. }
            | Node::String { span, .. }
            | Node::Boolean { span, .. }
            | Node::Null { span }
            | Node::Identifier { span, .. }
            | Node::Array { span, .. }
            | Node::Object { span, .. }
            | Node::BinaryOp { span, .. }
            | Node::UnaryOp { span, .. }
            | Node::Call { span, .. }
            | Node::Member { span, .. }
            | Node::Assignment { span, .. }
            | Node::FunctionDecl { span, .. }
            | Node::ProjectDecl { span, .. }
            | Node::Proposition { span, .. }
            | Node::SourcesDecl { span, .. }
            | Node::If { span, .. }
            | Node::Given { span, .. }
            | Node::Within { span, .. }
            | Node::Considering { span, .. }
            | Node::Ensure { span, .. }
            | Node::Return { span, .. }
            | Node::Research { span, .. }
            | Node::Block { span, .. }
            | Node::ExpressionStatement { span, .. }
            | Node::Program { span, .. }
            | Node::TextOperation { span, .. }
            | Node::Point { span, .. }
            | Node::Resolution { span, .. } => span,
        }
    }

    /// Check if this node is valid (basic validation)
    pub fn is_valid(&self) -> bool {
        match self {
            Node::Program { statements, .. } => statements.iter().all(|stmt| stmt.is_valid()),
            Node::Block { statements, .. } => statements.iter().all(|stmt| stmt.is_valid()),
            Node::FunctionDecl { body, .. } => body.is_valid(),
            Node::BinaryOp { left, right, .. } => left.is_valid() && right.is_valid(),
            Node::UnaryOp { operand, .. } => operand.is_valid(),
            Node::Call {
                callee, arguments, ..
            } => callee.is_valid() && arguments.iter().all(|arg| arg.is_valid()),
            _ => true, // Most nodes are valid by construction
        }
    }

    /// Check if this node represents a statement
    pub fn is_statement(&self) -> bool {
        matches!(
            self,
            Node::FunctionDecl { .. }
                | Node::ProjectDecl { .. }
                | Node::Proposition { .. }
                | Node::SourcesDecl { .. }
                | Node::Assignment { .. }
                | Node::Return { .. }
                | Node::Research { .. }
                | Node::Ensure { .. }
                | Node::ExpressionStatement { .. }
                | Node::Given { .. }
                | Node::Within { .. }
                | Node::Considering { .. }
        )
    }

    /// Check if this node represents an expression
    pub fn is_expression(&self) -> bool {
        matches!(
            self,
            Node::Number { .. }
                | Node::String { .. }
                | Node::Boolean { .. }
                | Node::Null { .. }
                | Node::Identifier { .. }
                | Node::Array { .. }
                | Node::Object { .. }
                | Node::BinaryOp { .. }
                | Node::UnaryOp { .. }
                | Node::Call { .. }
                | Node::Member { .. }
                | Node::TextOperation { .. }
        )
    }
}

// Convenience constructors
impl Node {
    /// Create a program node
    pub fn program(statements: Vec<Node>, span: Span) -> Self {
        Node::Program { statements, span }
    }

    /// Create a number node
    pub fn number(value: f64, span: Span) -> Self {
        Node::Number { value, span }
    }

    /// Create a string node
    pub fn string(value: String, span: Span) -> Self {
        Node::String { value, span }
    }

    /// Create an identifier node
    pub fn identifier(name: String, span: Span) -> Self {
        Node::Identifier { name, span }
    }

    /// Create a binary operation node
    pub fn binary_op(left: Node, operator: BinaryOp, right: Node, span: Span) -> Self {
        Node::BinaryOp {
            left: Box::new(left),
            operator,
            right: Box::new(right),
            span,
        }
    }

    /// Create a function call node
    pub fn call(callee: Node, arguments: Vec<Node>, span: Span) -> Self {
        Node::Call {
            callee: Box::new(callee),
            arguments,
            span,
        }
    }

    /// Create a block node
    pub fn block(statements: Vec<Node>, span: Span) -> Self {
        Node::Block { statements, span }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new(10, 5, 100);
        assert_eq!(pos.line, 10);
        assert_eq!(pos.column, 5);
        assert_eq!(pos.offset, 100);
    }

    #[test]
    fn test_span_creation() {
        let start = Position::new(1, 1, 0);
        let end = Position::new(1, 10, 9);
        let span = Span::new(start, end);
        assert_eq!(span.start.line, 1);
        assert_eq!(span.end.column, 10);
    }

    #[test]
    fn test_node_validation() {
        let span = Span::new(Position::new(1, 1, 0), Position::new(1, 2, 1));
        let node = Node::number(42.0, span);
        assert!(node.is_valid());
        assert!(node.is_expression());
        assert!(!node.is_statement());
    }

    #[test]
    fn test_text_unit_creation() {
        let unit = TextUnit::new("test content".to_string()).with_confidence(0.85);
        assert_eq!(unit.content, "test content");
        assert_eq!(unit.confidence, 0.85);
    }

    #[test]
    fn test_parameter_creation() {
        let param = Parameter::new("x".to_string());
        assert_eq!(param.name, "x");
        assert!(param.type_annotation.is_none());

        let typed_param = Parameter::with_type("y".to_string(), "Number".to_string());
        assert_eq!(typed_param.type_annotation, Some("Number".to_string()));
    }
}
