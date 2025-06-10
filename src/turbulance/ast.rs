use std::collections::HashMap;
use std::fmt;

/// Represents a position in the source code
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

impl Position {
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// Represents a span in the source code
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: Position,
    pub end: Position,
}

impl Span {
    pub fn new(start: Position, end: Position) -> Self {
        Self { start, end }
    }
}

/// Represents a value in the Turbulance language
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Number(f64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
    Function(FunctionDef),
    TextUnit(TextUnit),
    Cause(String, Box<Value>),
    Motion(String, Box<Value>),
    None,
}

/// Represents a text unit, which is a block of text with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct TextUnit {
    pub content: String,
    pub metadata: HashMap<String, Value>,
}

impl TextUnit {
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(content: String, metadata: HashMap<String, Value>) -> Self {
        Self { content, metadata }
    }
}

/// Represents a node in the AST (Abstract Syntax Tree)
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    // Literal values
    StringLiteral(String, Span),
    NumberLiteral(f64, Span),
    BoolLiteral(bool, Span),
    
    // Variables and expressions
    Identifier(String, Span),
    BinaryExpr {
        left: Box<Node>,
        operator: BinaryOp,
        right: Box<Node>,
        span: Span,
    },
    UnaryExpr {
        operator: UnaryOp,
        operand: Box<Node>,
        span: Span,
    },
    FunctionCall {
        function: Box<Node>,
        arguments: Vec<Node>,
        span: Span,
    },
    
    // Control flow
    IfExpr {
        condition: Box<Node>,
        then_branch: Box<Node>,
        else_branch: Option<Box<Node>>,
        span: Span,
    },
    
    // Iteration expressions (including new considering expressions)
    ForEach {
        iterable: Box<Node>,
        variable: String,
        body: Box<Node>,
        span: Span,
    },
    ConsideringAll {
        iterable: Box<Node>,
        variable: String,
        body: Box<Node>,
        span: Span,
    },
    ConsideringThese {
        iterable: Box<Node>,
        variable: String,
        body: Box<Node>,
        span: Span,
    },
    ConsideringItem {
        item: Box<Node>,
        variable: String,
        body: Box<Node>,
        span: Span,
    },
    
    // Declarations
    FunctionDecl {
        name: String,
        parameters: Vec<Parameter>,
        body: Box<Node>,
        span: Span,
    },
    ProjectDecl {
        name: String,
        attributes: HashMap<String, Node>,
        body: Box<Node>,
        span: Span,
    },
    SourcesDecl {
        sources: Vec<Source>,
        span: Span,
    },
    Motion {
        name: String,
        content: Box<Node>,
        span: Span,
    },
    
    // Statements
    Block {
        statements: Vec<Node>,
        span: Span,
    },
    Assignment {
        target: Box<Node>,
        value: Box<Node>,
        span: Span,
    },
    CauseDecl {
        name: String,
        value: Box<Node>,
        span: Span,
    },
    ReturnStmt {
        value: Option<Box<Node>>,
        span: Span,
    },
    AllowStmt {
        value: Box<Node>,
        span: Span,
    },
    
    // Text operations
    WithinBlock {
        target: Box<Node>,
        body: Box<Node>,
        span: Span,
    },
    GivenBlock {
        condition: Box<Node>,
        body: Box<Node>,
        span: Span,
    },
    EnsureStmt {
        condition: Box<Node>,
        span: Span,
    },
    ResearchStmt {
        query: Box<Node>,
        span: Span,
    },
    
    // Special operations
    TextOperation {
        operation: TextOp,
        target: Box<Node>,
        arguments: Vec<Node>,
        span: Span,
    },
    
    // Error node
    Error(String, Span),
    
    // Member access (obj.property)
    MemberAccess {
        object: Box<Node>,
        property: Box<Node>,
        span: Span,
    },
    
    // Index access (obj[index])
    IndexAccess {
        object: Box<Node>,
        index: Box<Node>,
        span: Span,
    },
}

/// Represents a function definition
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Box<Node>,
}

/// Represents a parameter in a function declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub default_value: Option<Node>,
    pub span: Span,
}

/// Represents a source declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub path: String,
    pub source_type: Option<String>,
}

/// Represents binary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,        // +
    Subtract,   // -
    Multiply,   // *
    Divide,     // /
    Equal,      // ==
    NotEqual,   // !=
    LessThan,   // <
    GreaterThan, // >
    LessThanEqual, // <=
    GreaterThanEqual, // >=
    And,        // &&
    Or,         // ||
    Pipe,       // |
    PipeForward, // |>
    Arrow,      // =>
}

/// Represents unary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Negate,     // -
    Not,        // !
}

/// Represents text operations
#[derive(Debug, Clone, PartialEq)]
pub enum TextOp {
    Simplify,
    Expand,
    Formalize,
    Informalize,
    Rewrite,
    Translate,
    Extract,
    Summarize,
    Divide,
    Multiply,
    Add,
    Subtract,
    Filter,
    Transform,
    Analyze,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Number(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Map(map) => {
                write!(f, "{{")?;
                for (i, (key, value)) in map.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", key, value)?;
                }
                write!(f, "}}")
            }
            Value::Function(func) => write!(f, "funxn {}", func.name),
            Value::TextUnit(unit) => write!(f, "TextUnit(\"{}\")", 
                                          if unit.content.len() > 20 {
                                              format!("{}...", &unit.content[..20])
                                          } else {
                                              unit.content.clone()
                                          }),
            Value::Cause(cause, value) => write!(f, "Cause({}, {})", cause, value),
            Value::Motion(motion, value) => write!(f, "Motion({}, {})", motion, value),
            Value::None => write!(f, "None"),
        }
    }
}

/// Creates a program AST from a list of top-level nodes
pub fn program(statements: Vec<Node>, span: Span) -> Node {
    Node::Block { statements, span }
}

/// Helper function to create a cause declaration
pub fn cause_decl(name: String, value: Node, span: Span) -> Node {
    Node::CauseDecl {
        name,
        value: Box::new(value),
        span,
    }
}

/// Helper function to create a motion declaration
pub fn motion(name: String, content: Node, span: Span) -> Node {
    Node::Motion {
        name,
        content: Box::new(content),
        span,
    }
}

/// Helper function to create a considering_all statement
pub fn considering_all(iterable: Node, variable: String, body: Node, span: Span) -> Node {
    Node::ConsideringAll {
        iterable: Box::new(iterable),
        variable,
        body: Box::new(body),
        span,
    }
}

/// Helper function to create a considering_these statement
pub fn considering_these(iterable: Node, variable: String, body: Node, span: Span) -> Node {
    Node::ConsideringThese {
        iterable: Box::new(iterable),
        variable,
        body: Box::new(body),
        span,
    }
}

/// Helper function to create a considering_item statement
pub fn considering_item(item: Node, variable: String, body: Node, span: Span) -> Node {
    Node::ConsideringItem {
        item: Box::new(item),
        variable,
        body: Box::new(body),
        span,
    }
}

/// Helper function to create an allow statement
pub fn allow_stmt(value: Node, span: Span) -> Node {
    Node::AllowStmt {
        value: Box::new(value),
        span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        let pos = Position::new(10, 5, 100);
        assert_eq!(format!("{}", pos), "line 10, column 5");
    }

    #[test]
    fn test_value_display() {
        let string_val = Value::String("hello".to_string());
        assert_eq!(format!("{}", string_val), r#""hello""#);

        let num_val = Value::Number(42.5);
        assert_eq!(format!("{}", num_val), "42.5");

        let bool_val = Value::Bool(true);
        assert_eq!(format!("{}", bool_val), "true");

        let list_val = Value::List(vec![
            Value::Number(1.0),
            Value::Number(2.0),
            Value::Number(3.0),
        ]);
        assert_eq!(format!("{}", list_val), "[1, 2, 3]");

        let mut map = HashMap::new();
        map.insert("key1".to_string(), Value::String("value1".to_string()));
        map.insert("key2".to_string(), Value::Number(42.0));
        let map_val = Value::Map(map);
        // Since HashMap doesn't guarantee order, we'll check for key-value pairs
        let s = format!("{}", map_val);
        assert!(s.contains(r#""key1": "value1""#));
        assert!(s.contains(r#""key2": 42"#));
    }

    #[test]
    fn test_text_unit() {
        let simple_unit = TextUnit::new("Sample text".to_string());
        assert_eq!(simple_unit.content, "Sample text");
        assert_eq!(simple_unit.metadata.len(), 0);

        let mut metadata = HashMap::new();
        metadata.insert("language".to_string(), Value::String("en".to_string()));
        metadata.insert("sentiment".to_string(), Value::Number(0.75));
        
        let complex_unit = TextUnit::with_metadata("Complex text".to_string(), metadata);
        assert_eq!(complex_unit.content, "Complex text");
        assert_eq!(complex_unit.metadata.len(), 2);
        
        match complex_unit.metadata.get("language") {
            Some(Value::String(lang)) => assert_eq!(lang, "en"),
            _ => panic!("Expected language metadata"),
        }
        
        match complex_unit.metadata.get("sentiment") {
            Some(Value::Number(score)) => assert_eq!(*score, 0.75),
            _ => panic!("Expected sentiment metadata"),
        }
    }

    #[test]
    fn test_program_creation() {
        let start_pos = Position::new(1, 1, 0);
        let end_pos = Position::new(10, 1, 100);
        let span = Span::new(start_pos, end_pos);
        
        let nodes = vec![
            Node::StringLiteral("test".to_string(), span),
            Node::NumberLiteral(42.0, span),
        ];
        
        let program = program(nodes.clone(), span);
        
        match program {
            Node::Block { statements, span: prog_span } => {
                assert_eq!(statements.len(), 2);
                assert_eq!(statements, nodes);
                assert_eq!(prog_span, span);
            },
            _ => panic!("Expected Block node"),
        }
    }
}
