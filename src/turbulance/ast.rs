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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    
    // Scientific reasoning constructs
    PropositionDecl {
        name: String,
        description: Option<String>,
        requirements: Option<Box<Node>>,
        body: Option<Box<Node>>,
        span: Span,
    },
    EvidenceDecl {
        name: String,
        collection_method: Box<Node>,
        data_structure: Box<Node>,
        span: Span,
    },
    PatternDecl {
        name: String,
        signature: Box<Node>,
        within_clause: Option<Box<Node>>,
        match_clauses: Vec<MatchClause>,
        span: Span,
    },
    MatchClause {
        condition: Box<Node>,
        action: Box<Node>,
        span: Span,
    },
    SupportStmt {
        hypothesis: Box<Node>,
        evidence: Box<Node>,
        span: Span,
    },
    ContradictStmt {
        hypothesis: Box<Node>,
        evidence: Box<Node>,
        span: Span,
    },
    InconclusiveStmt {
        message: String,
        recommendations: Option<Box<Node>>,
        span: Span,
    },
    MetaAnalysis(MetaAnalysis),
    DeriveHypotheses(DeriveHypotheses),
    
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
        else_branch: Option<Box<Node>>,
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
    
    // Complex data structures for scientific data
    StructuredData {
        fields: HashMap<String, Node>,
        span: Span,
    },
    
    // Array/List literals
    ArrayLiteral {
        elements: Vec<Node>,
        span: Span,
    },
    
    // Advanced orchestration statements
    Flow(FlowStatement),
    Catalyze(CatalyzeStatement), 
    CrossScaleCoordinate(CrossScaleCoordinate),
    Drift(DriftStatement),
    Cycle(CycleStatement),
    Roll(RollStatement),
    Resolve(ResolveStatement),
    Point(PointDeclaration),
    
    // Autobahn reference constructs
    Funxn(FunxnDeclaration),
    Goal(GoalDeclaration),
    Metacognitive(MetacognitiveBlock),
    Try(TryStatement),
    Parallel(ParallelBlock),
    QuantumState(QuantumStateDeclaration),
    BiologicalOperation(BiologicalOperation),
    OptimizeUntil(OptimizeUntilStatement),
    Within(WithinStatement),
    For(ForStatement),
    While(WhileStatement),
    Import(ImportStatement),
    
    // Bene Gesserit masterclass constructs
    SuccessFramework(SuccessFrameworkDeclaration),
    BiologicalComputer(BiologicalComputerDeclaration),
    PatternAnalysis(PatternAnalysisBlock),
    SpatiotemporalAnalysis(SpatiotemporalAnalysisBlock),
    DataProcessing(DataProcessingBlock),
    EvidenceIntegration(EvidenceIntegrationBlock),
    UncertaintyPropagation(UncertaintyPropagationBlock),
    CausalAnalysis(CausalAnalysisBlock),
    BiasAnalysis(BiasAnalysisBlock),
    QuantumClassicalInterface(QuantumClassicalInterface),
    
    // Imhotep Framework: Revolutionary Self-Aware Neural Networks
    SelfAware(SelfAwareStatement),
    
    // Space Computer Biomechanical Analysis Framework
    Config(ConfigDeclaration),
    Datasources(DatasourcesDeclaration),
    Segment(SegmentDeclaration),
    EvidenceIntegratorDecl(EvidenceIntegratorDeclaration),
    RealTimeOrchestrator(RealTimeOrchestratorDeclaration),
    VerificationSystem(VerificationSystemDeclaration),
    Interface(InterfaceDeclaration),
    OrchestratorSystem(OrchestratorSystemDeclaration),
    BiomechanicalAnalysis(BiomechanicalAnalysisBlock),
}

/// Represents a function definition
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub parameters: Vec<Parameter>,
    #[serde(skip, default = "default_node")]
    pub body: Box<Node>,
}

fn default_node() -> Box<Node> {
    Box::new(Node::Identifier("".to_string(), default_span()))
}

/// Represents a parameter in a function declaration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Parameter {
    pub name: String,
    #[serde(skip, default)]
    pub default_value: Option<Node>,
    #[serde(skip, default = "default_span")]
    pub span: Span,
}

fn default_span() -> Span {
    Span::new(
        Position::new(0, 0, 0),
        Position::new(0, 0, 0)
    )
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

#[derive(Debug, Clone, PartialEq)]
pub struct MatchClause {
    pub condition: Box<Node>,
    pub action: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeriveHypotheses {
    pub hypotheses: Vec<String>,
    pub span: Span,
}

// Advanced orchestration constructs
#[derive(Debug, Clone, PartialEq)]
pub struct FlowStatement {
    pub variable: String,
    pub collection: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatalyzeStatement {
    pub target: Box<Node>,
    pub scale: ScaleType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScaleType {
    Quantum,
    Molecular,
    Environmental,
    Hardware,
    Cognitive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossScaleCoordinate {
    pub pairs: Vec<CoordinationPair>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinationPair {
    pub scale1: ScaleType,
    pub scale2: ScaleType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftStatement {
    pub parameters: Box<Node>,
    pub condition: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CycleStatement {
    pub variable: String,
    pub collection: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RollStatement {
    pub variable: String,
    pub condition: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolveStatement {
    pub function_call: Box<Node>,
    pub context: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointDeclaration {
    pub name: String,
    pub properties: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InformationCatalysis {
    pub input_filter: Box<Node>,
    pub output_filter: Box<Node>,
    pub context: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternRecognizer {
    pub pattern: Box<Node>,
    pub sensitivity: Option<Box<Node>>,
    pub specificity: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionChanneler {
    pub amplification: Box<Node>,
    pub focus: Option<Box<Node>>,
    pub scope: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentalCapture {
    pub region: Option<Box<Node>>,
    pub focus: Option<Box<Node>>,
    pub context: Option<Box<Node>>,
    pub parameters: Vec<(String, Box<Node>)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeSpecification {
    pub start: Box<Node>,
    pub end: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterMap {
    pub parameters: Vec<(String, Box<Node>)>,
    pub span: Span,
}

// Autobahn reference constructs
#[derive(Debug, Clone, PartialEq)]
pub struct FunxnDeclaration {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Option<String>,
    pub default_value: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GoalDeclaration {
    pub name: String,
    pub description: Option<String>,
    pub success_threshold: Option<Box<Node>>,
    pub metrics: Vec<(String, Box<Node>)>,
    pub subgoals: Vec<SubGoal>,
    pub constraints: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubGoal {
    pub name: String,
    pub weight: Option<Box<Node>>,
    pub threshold: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetacognitiveBlock {
    pub name: String,
    pub operations: Vec<MetacognitiveOperation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetacognitiveOperation {
    TrackReasoning(String),
    EvaluateConfidence,
    DetectBias(String),
    AdaptBehavior(String),
    AnalyzeDecisionHistory,
    UpdateDecisionStrategies,
    IncreaseEvidenceRequirements,
    ReduceComputationalOverhead,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryStatement {
    pub try_block: Box<Node>,
    pub catch_blocks: Vec<CatchBlock>,
    pub finally_block: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CatchBlock {
    pub exception_type: Option<String>,
    pub exception_name: Option<String>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelBlock {
    pub tasks: Vec<ParallelTask>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelTask {
    pub name: String,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumStateDeclaration {
    pub name: String,
    pub properties: Vec<(String, Box<Node>)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiologicalOperation {
    pub operation_type: BiologicalOperationType,
    pub target: Option<Box<Node>>,
    pub parameters: Vec<(String, Box<Node>)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BiologicalOperationType {
    ProcessMolecule,
    HarvestEnergy,
    ExtractInformation,
    UpdateMembraneState,
    ConfigureMembrane,
    CalculateEntropyChange,
    GibbsFreeEnergy,
    Shannon,
    MutualInfo,
    InfoGain,
    CalculateMw,
    CalculateKa,
    AnalyzeFlux,
    CalculateKcatKm,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizeUntilStatement {
    pub condition: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WithinStatement {
    pub target: Box<Node>,
    pub alias: Option<String>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForStatement {
    pub variable: String,
    pub collection: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhileStatement {
    pub condition: Box<Node>,
    pub body: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportStatement {
    pub module: String,
    pub items: Option<Vec<String>>,
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Temporal,
    Spatial,
    Oscillatory,
    Emergent,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternExpression {
    pub pattern_type: PatternType,
    pub name: String,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceCollection {
    pub collection_type: EvidenceCollectionType,
    pub sources: Vec<String>,
    pub validation_rules: Vec<Box<Node>>,
    pub processing_pipeline: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceCollectionType {
    Basic,
    Batch,
    Comprehensive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumOperation {
    pub operation_type: QuantumOperationType,
    pub targets: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumOperationType {
    ApplyHadamard,
    ApplyCnot,
    Measure,
    MeasureEntanglement,
}

// Bene Gesserit masterclass constructs
#[derive(Debug, Clone, PartialEq)]
pub struct SuccessFrameworkDeclaration {
    pub primary_threshold: Option<Box<Node>>,
    pub secondary_threshold: Option<Box<Node>>,
    pub safety_threshold: Option<Box<Node>>,
    pub evidence_quality_modulation: bool,
    pub uncertainty_penalty: Option<Box<Node>>,
    pub fda_guidance_compliance: bool,
    pub ema_scientific_advice_integration: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiologicalComputerDeclaration {
    pub name: String,
    pub atp_budget: Option<Box<Node>>,
    pub time_horizon: Option<Box<Node>>,
    pub quantum_targets: Vec<QuantumTarget>,
    pub oscillatory_dynamics: Vec<OscillatoryDynamic>,
    pub operations: Vec<BiologicalComputerOperation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumTarget {
    pub name: String,
    pub quantum_state: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OscillatoryDynamic {
    pub name: String,
    pub frequency: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BiologicalComputerOperation {
    QuantumMolecularDocking {
        protein: String,
        ligand: Box<Node>,
        conformational_sampling: String,
        scoring_function: String,
    },
    QuantumMembraneSimulation {
        membrane_model: String,
        compound: Box<Node>,
        transport_mechanisms: Vec<String>,
        quantum_tunneling: bool,
    },
    BiologicalMaxwellsDemon {
        input_patterns: Box<Node>,
        recognition_threshold: Box<Node>,
        catalysis_efficiency: Box<Node>,
    },
    OptimizeATPEfficiency,
    TrackOscillationEndpoints,
    MeasureQuantumFidelity,
    CalculateInformationCatalysisEfficiency,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternAnalysisBlock {
    pub name: String,
    pub molecular_patterns: Option<MolecularPatternAnalysis>,
    pub clinical_patterns: Option<ClinicalPatternAnalysis>,
    pub omics_integration: Option<OmicsIntegrationAnalysis>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MolecularPatternAnalysis {
    pub binding_pose_clustering: Option<ClusteringParameters>,
    pub pharmacophore_identification: Option<Box<Node>>,
    pub admet_pattern_detection: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusteringParameters {
    pub method: String,
    pub eps: Option<Box<Node>>,
    pub min_samples: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClinicalPatternAnalysis {
    pub responder_phenotyping: Option<GaussianMixtureParameters>,
    pub disease_progression_trajectories: Option<Box<Node>>,
    pub adverse_event_clustering: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianMixtureParameters {
    pub n_components: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OmicsIntegrationAnalysis {
    pub multi_block_pls: Option<Box<Node>>,
    pub network_medicine_analysis: Option<Box<Node>>,
    pub pathway_enrichment: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatiotemporalAnalysisBlock {
    pub name: String,
    pub spatial_modeling: Option<SpatialModelingAnalysis>,
    pub temporal_modeling: Option<TemporalModelingAnalysis>,
    pub association_analysis: Option<AssociationAnalysis>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialModelingAnalysis {
    pub local_adaptation: Option<Box<Node>>,
    pub environmental_gradients: Option<Box<Node>>,
    pub population_structure: Option<Box<Node>>,
    pub migration_patterns: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalModelingAnalysis {
    pub evolutionary_trajectories: Option<Box<Node>>,
    pub selection_dynamics: Option<Box<Node>>,
    pub demographic_inference: Option<Box<Node>>,
    pub cultural_evolution: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociationAnalysis {
    pub environmental_gwas: Option<Box<Node>>,
    pub polygenic_adaptation: Option<Box<Node>>,
    pub balancing_selection: Option<Box<Node>>,
    pub introgression_analysis: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataProcessingBlock {
    pub quality_control: Option<QualityControlParameters>,
    pub harmonization: Option<HarmonizationParameters>,
    pub feature_engineering: Option<FeatureEngineeringParameters>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualityControlParameters {
    pub missing_data_threshold: Option<Box<Node>>,
    pub outlier_detection: Option<OutlierDetectionParameters>,
    pub batch_effect_correction: Option<Box<Node>>,
    pub technical_replicate_correlation: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutlierDetectionParameters {
    pub method: String,
    pub contamination: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HarmonizationParameters {
    pub unit_standardization: Option<Box<Node>>,
    pub temporal_alignment: Option<Box<Node>>,
    pub population_stratification: Option<Box<Node>>,
    pub covariate_adjustment: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureEngineeringParameters {
    pub molecular_descriptors: Option<Box<Node>>,
    pub clinical_composite_scores: Option<Box<Node>>,
    pub time_series_features: Option<Box<Node>>,
    pub network_features: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceIntegrationBlock {
    pub molecular_sources: Vec<DataSource>,
    pub clinical_sources: Vec<DataSource>,
    pub real_world_sources: Vec<DataSource>,
    pub omics_sources: Vec<DataSource>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataSource {
    pub name: String,
    pub database_type: String,
    pub database_name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UncertaintyPropagationBlock {
    pub aleatory_uncertainty: Option<UncertaintyComponent>,
    pub epistemic_uncertainty: Option<UncertaintyComponent>,
    pub model_uncertainty: Option<UncertaintyComponent>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UncertaintyComponent {
    pub source: String,
    pub quantification: String,
    pub propagation: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalAnalysisBlock {
    pub confounding_control: Option<CausalMethod>,
    pub reverse_causation: Option<CausalMethod>,
    pub mediation_analysis: Option<CausalMethod>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMethod {
    pub method: String,
    pub adjustment: Option<String>,
    pub validation: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiasAnalysisBlock {
    pub selection_bias: Option<BiasComponent>,
    pub confirmation_bias: Option<BiasComponent>,
    pub publication_bias: Option<BiasComponent>,
    pub measurement_bias: Option<BiasComponent>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiasComponent {
    pub detection: String,
    pub severity_assessment: String,
    pub mitigation: String,
    pub monitoring: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantumClassicalInterface {
    pub coherence_analysis: Option<CoherenceAnalysis>,
    pub neural_quantum_correlation: Option<NeuralQuantumCorrelation>,
    pub consciousness_classification: Option<ConsciousnessClassification>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceAnalysis {
    pub coherence_time_measurement: String,
    pub decoherence_pathway_analysis: String,
    pub environmental_coupling_analysis: String,
    pub coherence_protection_mechanisms: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralQuantumCorrelation {
    pub phase_locking_analysis: String,
    pub quantum_neural_synchronization: String,
    pub information_theoretic_analysis: String,
    pub causal_analysis: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsciousnessClassification {
    pub machine_learning_classification: String,
    pub bayesian_state_estimation: String,
    pub hidden_markov_modeling: String,
    pub neural_network_analysis: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedHypothesisTesting {
    pub confidence_interval: Option<ConfidenceIntervalAnalysis>,
    pub ensemble_prediction: Option<EnsemblePredictionAnalysis>,
    pub effect_size_analysis: Option<EffectSizeAnalysis>,
    pub longitudinal_modeling: Option<LongitudinalModelingAnalysis>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceIntervalAnalysis {
    pub method: String,
    pub n_bootstrap: Option<Box<Node>>,
    pub lower_bound: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnsemblePredictionAnalysis {
    pub methods: Vec<String>,
    pub ensemble_agreement: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectSizeAnalysis {
    pub method: String,
    pub treatment_group: Box<Node>,
    pub placebo_group: Box<Node>,
    pub clinical_significance: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LongitudinalModelingAnalysis {
    pub fixed_effects: Vec<String>,
    pub random_effects: Vec<String>,
    pub treatment_effect: Option<Box<Node>>,
    pub span: Span,
}

// Imhotep Framework: Revolutionary Self-Aware Neural Networks
#[derive(Debug, Clone, PartialEq)]
pub struct NeuralConsciousnessSession {
    pub session_name: String,
    pub consciousness_level: Box<Node>,
    pub self_awareness: bool,
    pub metacognitive_monitoring: bool,
    pub bmd_neurons: Vec<BmdNeuron>,
    pub neural_connections: Vec<NeuralConnection>,
    pub self_awareness_config: Option<SelfAwarenessConfiguration>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BmdNeuron {
    pub name: String,
    pub activation: String,
    pub parameters: Vec<NeuronParameter>,
    pub subsystem: String,
    pub question: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuronParameter {
    pub name: String,
    pub value: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralConnection {
    pub from_neuron: String,
    pub to_neuron: String,
    pub connection_type: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfAwarenessConfiguration {
    pub metacognitive_depth: Option<Box<Node>>,
    pub self_reflection_threshold: Option<Box<Node>>,
    pub thought_quality_standards: Option<Box<Node>>,
    pub knowledge_audit_frequency: Option<Box<Node>>,
    pub reasoning_chain_logging: Option<bool>,
    pub decision_trail_persistence: Option<bool>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetacognitiveState {
    pub current_thought_focus: String,
    pub self_awareness_level: Box<Node>,
    pub reasoning_chain: Vec<String>,
    pub thought_quality_assessment: Box<Node>,
    pub uncertainties_identified: Vec<UncertaintyItem>,
    pub knowledge_gaps_identified: Vec<String>,
    pub decision_history: Vec<DecisionRecord>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UncertaintyItem {
    pub description: String,
    pub confidence: Box<Node>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub decision: String,
    pub reasoning: String,
    pub confidence: Box<Node>,
    pub external_knowledge_used: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfAwareProcessing {
    pub processing_type: String,
    pub processing_steps: Vec<String>,
    pub metacognitive_monitoring: bool,
    pub uncertainty_tracking: bool,
    pub reasoning_quality_assessment: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningQualityAssessment {
    pub overall_quality: Box<Node>,
    pub statistical_quality: Option<Box<Node>>,
    pub biological_quality: Option<Box<Node>>,
    pub uncertainties: Vec<UncertaintyItem>,
    pub knowledge_gaps: Vec<KnowledgeGap>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeGap {
    pub domain: String,
    pub impact_level: String,
    pub impact_on_conclusions: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConsciousnessComparison {
    pub traditional_result: Box<Node>,
    pub self_aware_result: Box<Node>,
    pub metacognitive_insights: MetacognitiveState,
    pub span: Span,
}

// Imhotep Framework: Revolutionary Self-Aware Neural Networks
#[derive(Debug, Clone, PartialEq)]
pub enum SelfAwareStatement {
    // Neural Consciousness Creation
    NeuralConsciousnessDecl(NeuralConsciousnessSession),
    
    // BMD Neuron Creation
    CreateBmdNeuron {
        name: String,
        activation: String,
        parameters: Vec<NeuronParameter>,
        subsystem: String,
        question: String,
        span: Span,
    },
    
    // Neural Connection Pattern
    ConnectPattern {
        connections: Vec<NeuralConnection>,
        span: Span,
    },
    
    // Self-Awareness Configuration
    ConfigureSelfAwareness(SelfAwarenessConfiguration),
    
    // Self-Aware Processing Operations
    ActivateSelfAwareness {
        session: String,
        span: Span,
    },
    
    ProcessWithMetacognitiveMonitoring {
        data: Box<Node>,
        processing_steps: Vec<String>,
        span: Span,
    },
    
    AssessReasoningQuality {
        session: String,
        span: Span,
    },
    
    EnhanceMetacognitiveMonitoring {
        session: String,
        span: Span,
    },
    
    // Scientific Self-Aware Reasoning
    BeginMetacognitiveReasoning {
        session: String,
        analysis_name: String,
        span: Span,
    },
    
    AnalyzeWithMetacognitiveOversight {
        data: Box<Node>,
        analysis_type: String,
        metacognitive_monitoring: bool,
        span: Span,
    },
    
    InterpretWithSelfAwareness {
        results: Box<Node>,
        interpretation_context: String,
        uncertainty_tracking: bool,
        span: Span,
    },
    
    AnalyzePathwaysWithMetacognition {
        metabolites: Box<Node>,
        self_reflection: bool,
        knowledge_gap_detection: bool,
        span: Span,
    },
    
    // Consciousness vs Self-Awareness Comparison
    DemonstrateSelfAwarenessVsConsciousness {
        traditional_session: String,
        self_aware_session: String,
        span: Span,
    },
    
    // Metacognitive State Access
    GetMetacognitiveState {
        session: String,
        span: Span,
    },
    
    GetCurrentReasoningState {
        session: String,
        span: Span,
    },
    
    IdentifyKnowledgeGaps {
        session: String,
        span: Span,
    },
}

// Space Computer Biomechanical Analysis Framework Structures

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigDeclaration {
    pub platform_version: Option<String>,
    pub uncertainty_model: Option<String>,
    pub confidence_threshold: Option<Box<Node>>,
    pub verification_required: bool,
    pub real_time_analysis: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DatasourcesDeclaration {
    pub video_analysis: Option<VideoAnalysisConfig>,
    pub ground_reaction_forces: Option<ForceAnalysisConfig>,
    pub expert_annotations: Option<ExpertAnnotationsConfig>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoAnalysisConfig {
    pub pose_models: Vec<String>,
    pub fps: Option<Box<Node>>,
    pub resolution: Option<String>,
    pub pose_confidence: Option<Box<Node>>,
    pub occlusion_handling: bool,
    pub multi_camera_fusion: bool,
    pub landmarks: Vec<String>,
    pub coordinate_accuracy: Option<Box<Node>>,
    pub temporal_consistency: bool,
    pub missing_data_interpolation: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForceAnalysisConfig {
    pub sampling_rate: Option<Box<Node>>,
    pub force_accuracy: Option<Box<Node>>,
    pub moment_accuracy: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertAnnotationsConfig {
    pub inter_rater_reliability: Option<Box<Node>>,
    pub expert_confidence: Option<Box<Node>>,
    pub bias_correction: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentDeclaration {
    pub name: String,
    pub phases: Vec<PhaseDeclaration>,
    pub metrics: Vec<BiomechanicalMetric>,
    pub analysis_functions: Vec<AnalysisFunction>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhaseDeclaration {
    pub name: String,
    pub phase_type: PhaseType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PhaseType {
    ExtractPhase,
    StartPhase,
    DrivePhase,
    MaxVelocityPhase,
    ImpactPhase,
    PunchInitiation,
    WindUp,
    Contact,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiomechanicalMetric {
    pub name: String,
    pub metric_type: BiomechanicalMetricType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BiomechanicalMetricType {
    BlockAngle,
    ShinAngle,
    FirstStepLength,
    LegLength,
    GroundContactAngle,
    StrideFrequency,
    VerticalOscillation,
    StrideLength,
    GroundContactTime,
    FlightTime,
    HipRotation,
    ShoulderSeparation,
    WeightTransfer,
    WristAlignment,
    ElbowExtension,
    FollowThrough,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisFunction {
    pub name: String,
    pub function_type: AnalysisFunctionType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisFunctionType {
    OptimalRange,
    DecreasesLinearly,
    IncreasesOptimally,
    AtOptimalFrequencyRatio,
    HipRotationLeadsSequence,
    MaintainsStraight,
    ExtensionComplete,
    WithinOptimalRange,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceIntegratorDeclaration {
    pub fusion_methods: Vec<FusionMethod>,
    pub validation_pipeline: ValidationPipeline,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusionMethod {
    pub method_type: FusionMethodType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FusionMethodType {
    BayesianInference,
    UncertaintyPropagation,
    MultiFidelityFusion,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationPipeline {
    pub cross_validation: bool,
    pub bootstrap_validation: bool,
    pub external_validation: bool,
    pub prior_construction: Option<Box<Node>>,
    pub likelihood_modeling: Option<Box<Node>>,
    pub posterior_sampling: Option<Box<Node>>,
    pub markov_chain_monte_carlo: bool,
    pub convergence_diagnostics: bool,
    pub gelman_rubin_statistic: bool,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealTimeOrchestratorDeclaration {
    pub stream_processing: StreamProcessingConfig,
    pub continuous_evaluation: bool,
    pub temporal_weighting: bool,
    pub recency_bias_correction: bool,
    pub predictive_modeling: Option<PredictiveModelingConfig>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamProcessingConfig {
    pub video_feed: Option<StreamConfig>,
    pub sensor_data: Option<StreamConfig>,
    pub environmental: Option<StreamConfig>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StreamConfig {
    pub latency: Option<Box<Node>>,
    pub frequency: Option<Box<Node>>,
    pub update_rate: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveModelingConfig {
    pub prediction_horizon: Option<Box<Node>>,
    pub intervention_window: Option<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerificationSystemDeclaration {
    pub verification_methods: Vec<VerificationMethod>,
    pub verification_levels: Vec<VerificationLevel>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerificationMethod {
    pub method_type: VerificationMethodType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerificationMethodType {
    VisualSimilarityVerification,
    BiomechanicalConsistencyCheck,
    CrossReferenceValidation,
    UncertaintyQuantificationValidation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerificationLevel {
    pub level: VerificationLevelType,
    pub validation_time: Option<Box<Node>>,
    pub use_case: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerificationLevelType {
    Level1Basic,
    Level2Standard,
    Level3Comprehensive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceDeclaration {
    pub components: Vec<InterfaceComponent>,
    pub interactions: Vec<InterfaceInteraction>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceComponent {
    pub component_type: InterfaceComponentType,
    pub properties: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterfaceComponentType {
    ProbabilisticVisualization,
    GoalProgressDashboard,
    EvidenceExplorer,
    VerificationStatusPanel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceInteraction {
    pub interaction_type: InterfaceInteractionType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterfaceInteractionType {
    HypothesisModification,
    EvidenceExploration,
    RecommendationCustomization,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorSystemDeclaration {
    pub name: String,
    pub phases: Vec<OrchestratorPhase>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorPhase {
    pub phase_type: OrchestratorPhaseType,
    pub operations: Vec<OrchestratorOperation>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrchestratorPhaseType {
    Initialize,
    Execute,
    Monitor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrchestratorOperation {
    pub operation_type: OrchestratorOperationType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrchestratorOperationType {
    DataQualityAssessment,
    PropositionEvaluation,
    GoalOptimization,
    QualityAssurance,
    RecommendationSynthesis,
    ParallelEvaluate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BiomechanicalAnalysisBlock {
    pub name: String,
    pub elite_athletes: Option<Box<Node>>,
    pub sport_filtering: Option<Box<Node>>,
    pub pattern_extraction: Option<Box<Node>>,
    pub statistical_methods: Vec<StatisticalMethod>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatisticalMethod {
    pub method_type: StatisticalMethodType,
    pub parameters: Vec<Box<Node>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatisticalMethodType {
    WeightedHarmonicMean,
    BayesianUpdate,
    MonteCarloSimulation,
    GeometricMean,
    WeightedAverage,
    HarmonicMean,
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
