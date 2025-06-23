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

    #[token("proposition")]
    Proposition,

    // New scientific reasoning keywords
    #[token("evidence")]
    Evidence,

    #[token("pattern")]
    Pattern,

    #[token("support")]
    Support,

    #[token("contradict")]
    Contradict,

    #[token("inconclusive")]
    Inconclusive,

    #[token("requirements")]
    Requirements,

    #[token("signature")]
    Signature,

    #[token("match")]
    Match,

    #[token("meta")]
    Meta,

    #[token("derive_hypotheses")]
    DeriveHypotheses,

    #[token("alternatively")]
    Alternatively,

    #[token("with")]
    With,

    #[token("classify_as")]
    ClassifyAs,

    #[token("confidence")]
    Confidence,

    #[token("emergent_behaviors")]
    EmergentBehaviors,

    #[token("mechanisms")]
    Mechanisms,

    #[token("clinical_relevance")]
    ClinicalRelevance,

    #[token("refined_hypotheses")]
    RefinedHypotheses,

    #[token("recommendations")]
    Recommendations,

    // Advanced orchestration keywords  
    #[token("flow")]
    Flow,

    #[token("on")]
    On,

    #[token("catalyze")]
    Catalyze,

    #[token("cross_scale")]
    CrossScale,

    #[token("coordinate")]
    Coordinate,

    #[token("drift")]
    Drift,

    #[token("until")]
    Until,

    #[token("cycle")]
    Cycle,

    #[token("roll")]
    Roll,

    #[token("resolve")]
    Resolve,

    #[token("execute_information_catalysis")]
    ExecuteInformationCatalysis,

    #[token("create_pattern_recognizer")]
    CreatePatternRecognizer,

    #[token("create_action_channeler")]
    CreateActionChanneler,

    #[token("capture_screen_pixels")]
    CaptureScreenPixels,

    #[token("point")]
    Point,

    #[token("content")]
    Content,

    #[token("certainty")]
    Certainty,

    #[token("evidence_strength")]
    EvidenceStrength,

    #[token("contextual_relevance")]
    ContextualRelevance,

    #[token("urgency_factor")]
    UrgencyFactor,

    // Autobahn reference keywords
    #[token("funxn")]
    Funxn,

    #[token("metacognitive")]
    Metacognitive,

    #[token("goal")]
    Goal,

    #[token("optimize_until")]
    OptimizeUntil,

    #[token("try")]
    Try,

    #[token("catch")]
    Catch,

    #[token("finally")]
    Finally,

    #[token("parallel")]
    Parallel,

    #[token("async")]
    Async,

    #[token("await")]
    Await,

    #[token("import")]
    Import,

    #[token("from")]
    From,

    #[token("otherwise")]
    Otherwise,

    #[token("within")]
    Within,

    #[token("each")]
    Each,

    #[token("all")]
    All,

    #[token("these")]
    These,

    #[token("for")]
    For,

    #[token("while")]
    While,

    #[token("break")]
    Break,

    #[token("continue")]
    Continue,

    #[token("description")]
    Description,

    #[token("success_threshold")]
    SuccessThreshold,

    #[token("metrics")]
    Metrics,

    #[token("subgoals")]
    Subgoals,

    #[token("weight")]
    Weight,

    #[token("threshold")]
    Threshold,

    #[token("constraints")]
    Constraints,

    #[token("requires_evidence")]
    RequiresEvidence,

    #[token("support")]
    Support,

    #[token("with_weight")]
    WithWeight,

    #[token("collect")]
    Collect,

    #[token("collect_batch")]
    CollectBatch,

    #[token("validation_rules")]
    ValidationRules,

    #[token("processing_pipeline")]
    ProcessingPipeline,

    #[token("track_reasoning")]
    TrackReasoning,

    #[token("evaluate_confidence")]
    EvaluateConfidence,

    #[token("detect_bias")]
    DetectBias,

    #[token("adapt_behavior")]
    AdaptBehavior,

    #[token("analyze_decision_history")]
    AnalyzeDecisionHistory,

    #[token("update_decision_strategies")]
    UpdateDecisionStrategies,

    #[token("increase_evidence_requirements")]
    IncreaseEvidenceRequirements,

    #[token("reduce_computational_overhead")]
    ReduceComputationalOverhead,

    // Biological operations
    #[token("process_molecule")]
    ProcessMolecule,

    #[token("harvest_energy")]
    HarvestEnergy,

    #[token("extract_information")]
    ExtractInformation,

    #[token("update_membrane_state")]
    UpdateMembraneState,

    #[token("configure_membrane")]
    ConfigureMembrane,

    // Scientific functions
    #[token("calculate_entropy_change")]
    CalculateEntropyChange,

    #[token("gibbs_free_energy")]
    GibbsFreeEnergy,

    #[token("shannon")]
    Shannon,

    #[token("mutual_info")]
    MutualInfo,

    #[token("info_gain")]
    InfoGain,

    #[token("calculate_mw")]
    CalculateMw,

    #[token("calculate_ka")]
    CalculateKa,

    #[token("analyze_flux")]
    AnalyzeFlux,

    #[token("calculate_kcat_km")]
    CalculateKcatKm,

    // Quantum operations
    #[token("quantum_state")]
    QuantumState,

    #[token("amplitude")]
    Amplitude,

    #[token("phase")]
    Phase,

    #[token("coherence_time")]
    CoherenceTime,

    #[token("apply_hadamard")]
    ApplyHadamard,

    #[token("apply_cnot")]
    ApplyCnot,

    #[token("measure")]
    Measure,

    #[token("measure_entanglement")]
    MeasureEntanglement,

    #[token("parallel_execute")]
    ParallelExecute,

    #[token("await_all_tasks")]
    AwaitAllTasks,

    // Pattern types
    #[token("temporal")]
    Temporal,

    #[token("spatial")]
    Spatial,

    #[token("oscillatory")]
    Oscillatory,

    #[token("emergent")]
    Emergent,

    // Additional scientific keywords
    #[token("matches")]
    Matches,

    #[token("contains")]
    Contains,

    #[token("temperature")]
    Temperature,

    #[token("ph_level")]
    PhLevel,

    #[token("concentration")]
    Concentration,

    #[token("catalyst")]
    Catalyst,

    #[token("monitor_efficiency")]
    MonitorEfficiency,

    #[token("target_yield")]
    TargetYield,

    #[token("adaptive_optimization")]
    AdaptiveOptimization,

    #[token("processing_method")]
    ProcessingMethod,

    #[token("noise_filtering")]
    NoiseFiltering,

    #[token("confidence_threshold")]
    ConfidenceThreshold,

    #[token("permeability")]
    Permeability,

    #[token("selectivity")]
    Selectivity,

    #[token("transport_rate")]
    TransportRate,

    #[token("energy_requirement")]
    EnergyRequirement,

    // Scale identifiers
    #[token("quantum")]
    Quantum,

    #[token("molecular")]
    Molecular,

    #[token("environmental")]
    Environmental,

    #[token("hardware")]
    Hardware,

    #[token("cognitive")]
    Cognitive,

    // Context and loading keywords
    #[token("load_sequence")]
    LoadSequence,

    #[token("load_molecules")]
    LoadMolecules,

    #[token("context")]
    Context,

    #[token("region")]
    Region,

    #[token("focus")]
    Focus,

    #[token("wavelength_range")]
    WavelengthRange,

    #[token("wavelength_scan")]
    WavelengthScan,

    #[token("sensitivity")]
    Sensitivity,

    #[token("specificity")]
    Specificity,

    #[token("amplification")]
    Amplification,

    #[token("duration")]
    Duration,

    #[token("size")]
    Size,

    #[token("diversity")]
    Diversity,

    #[token("var")]
    Var,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("sources")]
    SourcesDecl,

    #[token("given")]
    Given,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("considering")]
    Considering,

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

    #[token(";")]
    Semicolon,

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
            TokenKind::Proposition => write!(f, "proposition"),
            TokenKind::Evidence => write!(f, "evidence"),
            TokenKind::Pattern => write!(f, "pattern"),
            TokenKind::Support => write!(f, "support"),
            TokenKind::Contradict => write!(f, "contradict"),
            TokenKind::Inconclusive => write!(f, "inconclusive"),
            TokenKind::Requirements => write!(f, "requirements"),
            TokenKind::Signature => write!(f, "signature"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::Meta => write!(f, "meta"),
            TokenKind::DeriveHypotheses => write!(f, "derive_hypotheses"),
            TokenKind::Alternatively => write!(f, "alternatively"),
            TokenKind::With => write!(f, "with"),
            TokenKind::ClassifyAs => write!(f, "classify_as"),
            TokenKind::Confidence => write!(f, "confidence"),
            TokenKind::EmergentBehaviors => write!(f, "emergent_behaviors"),
            TokenKind::Mechanisms => write!(f, "mechanisms"),
            TokenKind::ClinicalRelevance => write!(f, "clinical_relevance"),
            TokenKind::RefinedHypotheses => write!(f, "refined_hypotheses"),
            TokenKind::Recommendations => write!(f, "recommendations"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
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
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Identifier => write!(f, "identifier"),
            TokenKind::StringLiteral => write!(f, "string"),
            TokenKind::NumberLiteral => write!(f, "number"),
            TokenKind::Comment => write!(f, "comment"),
            TokenKind::Whitespace => write!(f, "whitespace"),
            TokenKind::Error => write!(f, "error"),
            TokenKind::Flow => write!(f, "flow"),
            TokenKind::On => write!(f, "on"),
            TokenKind::Catalyze => write!(f, "catalyze"),
            TokenKind::CrossScale => write!(f, "cross_scale"),
            TokenKind::Coordinate => write!(f, "coordinate"),
            TokenKind::Drift => write!(f, "drift"),
            TokenKind::Until => write!(f, "until"),
            TokenKind::Cycle => write!(f, "cycle"),
            TokenKind::Roll => write!(f, "roll"),
            TokenKind::Resolve => write!(f, "resolve"),
            TokenKind::ExecuteInformationCatalysis => write!(f, "execute_information_catalysis"),
            TokenKind::CreatePatternRecognizer => write!(f, "create_pattern_recognizer"),
            TokenKind::CreateActionChanneler => write!(f, "create_action_channeler"),
            TokenKind::CaptureScreenPixels => write!(f, "capture_screen_pixels"),
            TokenKind::Point => write!(f, "point"),
            TokenKind::Content => write!(f, "content"),
            TokenKind::Certainty => write!(f, "certainty"),
            TokenKind::EvidenceStrength => write!(f, "evidence_strength"),
            TokenKind::ContextualRelevance => write!(f, "contextual_relevance"),
            TokenKind::UrgencyFactor => write!(f, "urgency_factor"),
            TokenKind::Quantum => write!(f, "quantum"),
            TokenKind::Molecular => write!(f, "molecular"),
            TokenKind::Environmental => write!(f, "environmental"),
            TokenKind::Hardware => write!(f, "hardware"),
            TokenKind::Cognitive => write!(f, "cognitive"),
            TokenKind::LoadSequence => write!(f, "load_sequence"),
            TokenKind::LoadMolecules => write!(f, "load_molecules"),
            TokenKind::Context => write!(f, "context"),
            TokenKind::Region => write!(f, "region"),
            TokenKind::Focus => write!(f, "focus"),
            TokenKind::WavelengthRange => write!(f, "wavelength_range"),
            TokenKind::WavelengthScan => write!(f, "wavelength_scan"),
            TokenKind::Sensitivity => write!(f, "sensitivity"),
            TokenKind::Specificity => write!(f, "specificity"),
            TokenKind::Amplification => write!(f, "amplification"),
            TokenKind::Duration => write!(f, "duration"),
            TokenKind::Size => write!(f, "size"),
            TokenKind::Diversity => write!(f, "diversity"),
            TokenKind::Funxn => write!(f, "funxn"),
            TokenKind::Metacognitive => write!(f, "metacognitive"),
            TokenKind::Goal => write!(f, "goal"),
            TokenKind::OptimizeUntil => write!(f, "optimize_until"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Catch => write!(f, "catch"),
            TokenKind::Finally => write!(f, "finally"),
            TokenKind::Parallel => write!(f, "parallel"),
            TokenKind::Async => write!(f, "async"),
            TokenKind::Await => write!(f, "await"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::From => write!(f, "from"),
            TokenKind::Otherwise => write!(f, "otherwise"),
            TokenKind::Description => write!(f, "description"),
            TokenKind::SuccessThreshold => write!(f, "success_threshold"),
            TokenKind::Metrics => write!(f, "metrics"),
            TokenKind::Subgoals => write!(f, "subgoals"),
            TokenKind::Weight => write!(f, "weight"),
            TokenKind::Threshold => write!(f, "threshold"),
            TokenKind::Constraints => write!(f, "constraints"),
            TokenKind::RequiresEvidence => write!(f, "requires_evidence"),
            TokenKind::WithWeight => write!(f, "with_weight"),
            TokenKind::Collect => write!(f, "collect"),
            TokenKind::CollectBatch => write!(f, "collect_batch"),
            TokenKind::ValidationRules => write!(f, "validation_rules"),
            TokenKind::ProcessingPipeline => write!(f, "processing_pipeline"),
            TokenKind::TrackReasoning => write!(f, "track_reasoning"),
            TokenKind::EvaluateConfidence => write!(f, "evaluate_confidence"),
            TokenKind::DetectBias => write!(f, "detect_bias"),
            TokenKind::AdaptBehavior => write!(f, "adapt_behavior"),
            TokenKind::AnalyzeDecisionHistory => write!(f, "analyze_decision_history"),
            TokenKind::UpdateDecisionStrategies => write!(f, "update_decision_strategies"),
            TokenKind::IncreaseEvidenceRequirements => write!(f, "increase_evidence_requirements"),
            TokenKind::ReduceComputationalOverhead => write!(f, "reduce_computational_overhead"),
            TokenKind::ProcessMolecule => write!(f, "process_molecule"),
            TokenKind::HarvestEnergy => write!(f, "harvest_energy"),
            TokenKind::ExtractInformation => write!(f, "extract_information"),
            TokenKind::UpdateMembraneState => write!(f, "update_membrane_state"),
            TokenKind::ConfigureMembrane => write!(f, "configure_membrane"),
            TokenKind::CalculateEntropyChange => write!(f, "calculate_entropy_change"),
            TokenKind::GibbsFreeEnergy => write!(f, "gibbs_free_energy"),
            TokenKind::Shannon => write!(f, "shannon"),
            TokenKind::MutualInfo => write!(f, "mutual_info"),
            TokenKind::InfoGain => write!(f, "info_gain"),
            TokenKind::CalculateMw => write!(f, "calculate_mw"),
            TokenKind::CalculateKa => write!(f, "calculate_ka"),
            TokenKind::AnalyzeFlux => write!(f, "analyze_flux"),
            TokenKind::CalculateKcatKm => write!(f, "calculate_kcat_km"),
            TokenKind::QuantumState => write!(f, "quantum_state"),
            TokenKind::Amplitude => write!(f, "amplitude"),
            TokenKind::Phase => write!(f, "phase"),
            TokenKind::CoherenceTime => write!(f, "coherence_time"),
            TokenKind::ApplyHadamard => write!(f, "apply_hadamard"),
            TokenKind::ApplyCnot => write!(f, "apply_cnot"),
            TokenKind::Measure => write!(f, "measure"),
            TokenKind::MeasureEntanglement => write!(f, "measure_entanglement"),
            TokenKind::ParallelExecute => write!(f, "parallel_execute"),
            TokenKind::AwaitAllTasks => write!(f, "await_all_tasks"),
            TokenKind::Temporal => write!(f, "temporal"),
            TokenKind::Spatial => write!(f, "spatial"),
            TokenKind::Oscillatory => write!(f, "oscillatory"),
            TokenKind::Emergent => write!(f, "emergent"),
            TokenKind::Matches => write!(f, "matches"),
            TokenKind::Contains => write!(f, "contains"),
            TokenKind::Temperature => write!(f, "temperature"),
            TokenKind::PhLevel => write!(f, "ph_level"),
            TokenKind::Concentration => write!(f, "concentration"),
            TokenKind::Catalyst => write!(f, "catalyst"),
            TokenKind::MonitorEfficiency => write!(f, "monitor_efficiency"),
            TokenKind::TargetYield => write!(f, "target_yield"),
            TokenKind::AdaptiveOptimization => write!(f, "adaptive_optimization"),
            TokenKind::ProcessingMethod => write!(f, "processing_method"),
            TokenKind::NoiseFiltering => write!(f, "noise_filtering"),
            TokenKind::ConfidenceThreshold => write!(f, "confidence_threshold"),
            TokenKind::Permeability => write!(f, "permeability"),
            TokenKind::Selectivity => write!(f, "selectivity"),
            TokenKind::TransportRate => write!(f, "transport_rate"),
            TokenKind::EnergyRequirement => write!(f, "energy_requirement"),
            _ => write!(f, "{:?}", self),
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
        
        while let Some(result) = self.lex.next() {
            let span = self.lex.span();
            let lexeme = self.get_lexeme(span.clone());
            
            let token_kind = match result {
                Ok(kind) => kind,
                Err(_) => TokenKind::Error,
            };
            
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
