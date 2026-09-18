//! Abstract Syntax Tree definitions for Turbulance

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Position in source code
///
/// `Copy`: three `usize` fields. Spans are threaded through every parser
/// helper by reference, and without `Copy` each read of `span.start` is a
/// move out of a shared reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// A numeric literal. All Turbulance numbers are `f64`.
    Number {
        /// The parsed value.
        value: f64,
        /// The source range this node covers.
        span: Span,
    },
    /// A string literal, already unescaped.
    String {
        /// The literal's contents.
        value: String,
        /// The source range this node covers.
        span: Span,
    },
    /// A `true` or `false` literal.
    Boolean {
        /// Which of the two.
        value: bool,
        /// The source range this node covers.
        span: Span,
    },
    /// The null literal.
    Null {
        /// The source range this node covers.
        span: Span,
    },
    /// A bare name, resolved against the environment at evaluation.
    Identifier {
        /// The name as written.
        name: String,
        /// The source range this node covers.
        span: Span,
    },

    /// A list literal, `[a, b, c]`.
    Array {
        /// The elements, in source order.
        elements: Vec<Node>,
        /// The source range this node covers.
        span: Span,
    },
    /// A map literal, `{k: v}`.
    Object {
        /// The fields, keyed by name.
        fields: HashMap<String, Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// An infix operation.
    BinaryOp {
        /// The left operand.
        left: Box<Node>,
        /// Which operation.
        operator: BinaryOp,
        /// The right operand.
        right: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A prefix operation.
    UnaryOp {
        /// Which operation.
        operator: UnaryOp,
        /// The single operand.
        operand: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A function call, `callee(arguments)`.
    Call {
        /// The expression being called.
        callee: Box<Node>,
        /// The arguments, in source order.
        arguments: Vec<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// Property access, `object.property`.
    Member {
        /// The expression on the left of the dot.
        object: Box<Node>,
        /// The property name.
        property: String,
        /// The source range this node covers.
        span: Span,
    },

    /// Assignment to an existing binding.
    Assignment {
        /// The place being assigned to.
        target: Box<Node>,
        /// The expression producing the new value.
        value: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `funxn` declaration.
    FunctionDecl {
        /// The function's name.
        name: String,
        /// Its parameters, in order.
        parameters: Vec<Parameter>,
        /// The body, a block or a single expression.
        body: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `project` declaration.
    ProjectDecl {
        /// The project's name.
        name: String,
        /// Attributes given in the parentheses after the name.
        attributes: HashMap<String, Node>,
        /// The project body.
        body: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `proposition`: a named claim together with the motions that argue it.
    Proposition {
        /// The proposition's name.
        name: String,
        /// Its motions.
        motions: Vec<Motion>,
        /// Statements in the proposition's body.
        body: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `sources` block, declaring where evidence comes from.
    SourcesDecl {
        /// The declared sources.
        sources: Vec<Source>,
        /// The source range this node covers.
        span: Span,
    },

    /// An `if` conditional.
    If {
        /// The test.
        condition: Box<Node>,
        /// Taken when the test holds.
        then_branch: Box<Node>,
        /// Taken otherwise, when one was written.
        else_branch: Option<Box<Node>>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `given` conditional. Turbulance's own spelling of a conditional; there is no `elif`, so ordered dispatch is written as early `return` inside successive `given`s.
    Given {
        /// The test.
        condition: Box<Node>,
        /// Taken when the test holds.
        then_branch: Box<Node>,
        /// Taken otherwise, when one was written.
        else_branch: Option<Box<Node>>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `within` statement, scoping a body to a target.
    Within {
        /// What the body is scoped to.
        target: Box<Node>,
        /// The scoped statements.
        body: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `considering` statement, iterating a collection.
    Considering {
        /// The collection iterated over.
        items: Box<Node>,
        /// The body run per item.
        body: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// An `ensure` assertion. Evaluation fails if the condition does not hold.
    Ensure {
        /// The condition that must hold.
        condition: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `return`, with or without a value.
    Return {
        /// The returned expression, absent for a bare `return`.
        value: Option<Box<Node>>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `research` statement.
    Research {
        /// The query expression.
        query: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A sequence of statements sharing a scope.
    Block {
        /// The statements, in order.
        statements: Vec<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// An expression evaluated for its effect, its value discarded.
    ExpressionStatement {
        /// The expression.
        expression: Box<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// The root of a parsed source file.
    Program {
        /// The top-level statements, in order.
        statements: Vec<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A semantic operation applied to text.
    TextOperation {
        /// Which operation.
        operation: TextOp,
        /// The text operated on.
        target: Box<Node>,
        /// Any further arguments.
        arguments: Vec<Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `point` declaration: a named bundle of properties.
    Point {
        /// The point's name.
        name: String,
        /// Its properties, keyed by name.
        properties: HashMap<String, Node>,
        /// The source range this node covers.
        span: Span,
    },

    /// A `resolution` declaration.
    Resolution {
        /// The resolution's name.
        name: String,
        /// Its parameters, in order.
        parameters: Vec<Parameter>,
        /// The body.
        body: Box<Node>,
        /// The declared return type, when one was written.
        return_type: Option<String>,
        /// The source range this node covers.
        span: Span,
    },
}

/// Binary operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    /// Arithmetic `+`.
    Add,
    /// Arithmetic `-`.
    Subtract,
    /// Arithmetic `*`.
    Multiply,
    /// Arithmetic `/`. Always float division; there is no integer division.
    Divide,
    /// Arithmetic `%`.
    Modulo,
    /// Exponentiation.
    Power,

    /// Equality, `==`.
    Equal,
    /// Inequality, `!=`.
    NotEqual,
    /// Ordering, `<`.
    LessThan,
    /// Ordering, `>`.
    GreaterThan,
    /// Ordering, `<=`.
    LessThanEqual,
    /// Ordering, `>=`.
    GreaterThanEqual,

    /// Logical conjunction.
    And,
    /// Logical disjunction.
    Or,

    /// `|`: pipe the left value into the right.
    Pipe,
    /// `|>`: pipe forward.
    PipeForward,
    /// `=>`: arrow, used where a mapping rather than a value is meant.
    Arrow,

    /// Semantic `+`: meaningful combination rather than numeric addition.
    SemanticAdd,
    /// Semantic `-`: removal of elements.
    SemanticSubtract,
    /// Semantic `*`: amplification or repetition.
    SemanticMultiply,
    /// Semantic `/`: extraction or filtering.
    SemanticDivide,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Arithmetic negation, prefix `-`.
    Negate,
    /// Logical negation, `not`.
    Not,
    /// Prefix `+`, which leaves its operand unchanged.
    Plus,
}

/// Text operations for semantic processing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextOp {
    /// Reduce to plainer language.
    Simplify,
    /// Draw out what was stated compactly.
    Expand,
    /// Raise the register toward formal prose.
    Formalize,
    /// Lower the register toward informal prose.
    Informalize,
    /// Render in another language.
    Translate,
    /// Shorten while keeping the substance.
    Summarize,
    /// Pull out a named part.
    Extract,
    /// Restate differently without changing what is said.
    Rewrite,
    /// Produce a reading of the text.
    Understand,
    /// Resolve what the text leaves ambiguous.
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
