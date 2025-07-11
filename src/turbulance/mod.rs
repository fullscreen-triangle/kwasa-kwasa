use std::collections::HashMap;

pub mod ast;
pub mod audio_syntax;
pub mod context;
pub mod datastructures;
pub mod debate_platform;
pub mod domain_extensions;
pub mod four_file_system;
pub mod hybrid_processing;
pub mod image_syntax;
pub mod integration;
pub mod interpreter;
pub mod lexer;
pub mod parser;
pub mod perturbation_validation;
pub mod polyglot_syntax;
pub mod polyglot;
pub mod positional_semantics;
pub mod probabilistic;
pub mod proposition;
pub mod quantum_lexer;
pub mod semantic_engine;
pub mod stdlib;
pub mod streaming;
pub mod turbulance_syntax;
pub mod v8_intelligence;
pub mod bmd_pattern_learning;

pub use ast::*;
pub use audio_syntax::*;
pub use context::*;
pub use datastructures::*;
pub use debate_platform::*;
pub use domain_extensions::*;
pub use four_file_system::*;
pub use hybrid_processing::*;
pub use image_syntax::*;
pub use integration::*;
pub use interpreter::*;
pub use lexer::*;
pub use parser::*;
pub use perturbation_validation::*;
pub use polyglot_syntax::*;
pub use polyglot::*;
pub use positional_semantics::*;
pub use probabilistic::*;
pub use proposition::*;
pub use quantum_lexer::*;
pub use semantic_engine::*;
pub use stdlib::*;
pub use streaming::*;
pub use turbulance_syntax::*;
pub use v8_intelligence::*;
pub use bmd_pattern_learning::*;

pub use context::Context;
pub use debate_platform::{Affirmation, Contention, DebatePlatform, DebatePlatformManager};
pub use lexer::TokenKind;
pub use perturbation_validation::{PerturbationValidator, ValidationConfig, ValidationResult};
pub use positional_semantics::{PositionalAnalyzer, PositionalSentence};
pub use probabilistic::{ResolutionResult, TextPoint};
pub use streaming::{StreamConfig, StreamState, TextStream};

// Include generated code
// mod generated {
//     pub(crate) mod prelude {
//         pub use super::super::interpreter::{NativeFunction, Value};
//         // Re-export common types
//         pub use super::super::{Result, TokenKind, TurbulanceError};
//     }
//     pub(crate) use prelude::*;

//     include!(concat!(env!("OUT_DIR"), "/generated/parser_tables.rs"));
//     include!(concat!(env!("OUT_DIR"), "/generated/stdlib_bindings.rs"));
//     include!(concat!(env!("OUT_DIR"), "/generated/token_definitions.rs"));
// }

// Re-export generated types
// pub use generated::*;

/// Error types for the Turbulance language
#[derive(Debug, Clone, thiserror::Error)]
pub enum TurbulanceError {
    #[error("Lexical error at position {position}: {message}")]
    LexicalError { position: usize, message: String },

    #[error("Syntax error at position {position}: {message}")]
    SyntaxError { position: usize, message: String },

    #[error("Semantic error: {message}")]
    SemanticError { message: String },

    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    #[error("IO error: {message}")]
    IoError { message: String },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Processing timeout exceeded")]
    ProcessingTimeout,
}

/// Result type for Turbulance operations
pub type Result<T> = std::result::Result<T, TurbulanceError>;

/// Parse and run a Turbulance script
pub fn run(source: &str) -> Result<()> {
    // Initialize language components using generated code
    let keyword_table = keywords_table();
    let operator_precedence = operator_precedence();
    let stdlib = stdlib_functions();

    // Tokenize the source code
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();

    // Parse tokens into AST
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse()?;

    // Execute program with standard library
    let mut interpreter = interpreter::Interpreter::new();
    interpreter.register_stdlib_functions(stdlib);

    // Register domain extensions
    domain_extensions::register_domain_extensions(&mut interpreter)
        .map_err(|e| TurbulanceError::RuntimeError { message: e.to_string() })?;

    let _ = interpreter.execute(&ast)?;

    Ok(())
}

/// Version of the Turbulance language
pub const VERSION: &str = "0.1.0";

/// Check if the given source code is syntactically valid
pub fn validate(source: &str) -> Result<bool> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();

    let mut parser = parser::Parser::new(tokens);
    match parser.parse() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

// Serialize AST to JSON
pub fn serialize_ast(program: &ast::Program) -> String {
    let serializable = SerializableAst::from(program);
    serde_json::to_string_pretty(&serializable).unwrap_or_else(|_| "{}".to_string())
}

/// Run Turbulance code with a specific context
pub fn run_with_context(source: &str, context: &mut Context) -> Result<String> {
    let tokens = lexer::lex(source)?;
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse()?;
    let result = interpreter::interpret(ast, context)?;
    Ok(result)
}

impl TurbulanceError {
    /// Determine if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::LexicalError { .. } | Self::SyntaxError { .. } => false,
            _ => true,
        }
    }
}

impl From<std::io::Error> for TurbulanceError {
    fn from(error: std::io::Error) -> Self {
        Self::IoError { message: error.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_run_placeholder() {
        let result = run("funxn test(): return 42");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_placeholder() {
        let result = validate("funxn test(): return 42");
        assert!(result.is_ok());
        assert!(result.unwrap());
    }
}

/// Scientific Argument Validation System
/// Validates the logical coherence and methodological soundness of scientific reasoning
pub struct ScientificArgumentValidator {
    pub propositions: HashMap<String, PropositionValidation>,
    pub evidence_chains: HashMap<String, EvidenceValidation>,
    pub reasoning_graph: ReasoningGraph,
    pub methodological_checks: MethodologicalChecks,
}

#[derive(Debug, Clone)]
pub struct PropositionValidation {
    pub name: String,
    pub hypothesis: String,
    pub testable: bool,
    pub falsifiable: bool,
    pub specific: bool,
    pub measurable: bool,
    pub evidence_requirements: Vec<EvidenceRequirement>,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone)]
pub struct EvidenceRequirement {
    pub evidence_type: EvidenceType,
    pub minimum_quality: f64,
    pub sample_size_required: Option<usize>,
    pub statistical_power: Option<f64>,
    pub controls_required: bool,
}

#[derive(Debug, Clone)]
pub enum EvidenceType {
    Experimental,
    Observational,
    Computational,
    MetaAnalysis,
    SystematicReview,
    CaseStudy,
    Expert Opinion,
}

#[derive(Debug, Clone)]
pub struct EvidenceValidation {
    pub name: String,
    pub evidence_type: EvidenceType,
    pub quality_score: f64,
    pub sample_size: Option<usize>,
    pub statistical_power: Option<f64>,
    pub controls_present: bool,
    pub bias_assessment: BiasAssessment,
    pub supports_propositions: Vec<String>,
    pub contradicts_propositions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BiasAssessment {
    pub selection_bias: BiasLevel,
    pub confirmation_bias: BiasLevel,
    pub publication_bias: BiasLevel,
    pub measurement_bias: BiasLevel,
    pub overall_bias_risk: BiasLevel,
}

#[derive(Debug, Clone)]
pub enum BiasLevel {
    Low,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ReasoningGraph {
    pub nodes: HashMap<String, ReasoningNode>,
    pub edges: Vec<ReasoningEdge>,
    pub circular_reasoning_detected: Vec<String>,
    pub logical_fallacies: Vec<LogicalFallacy>,
}

#[derive(Debug, Clone)]
pub struct ReasoningNode {
    pub id: String,
    pub node_type: ReasoningNodeType,
    pub content: String,
    pub confidence: f64,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ReasoningNodeType {
    Proposition,
    Evidence,
    Inference,
    Conclusion,
    Assumption,
}

#[derive(Debug, Clone)]
pub struct ReasoningEdge {
    pub from: String,
    pub to: String,
    pub relationship: ReasoningRelationship,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub enum ReasoningRelationship {
    Supports,
    Contradicts,
    Implies,
    RequiredFor,
    EnhancedBy,
    WeakenedBy,
}

#[derive(Debug, Clone)]
pub struct LogicalFallacy {
    pub fallacy_type: FallacyType,
    pub location: String,
    pub description: String,
    pub severity: FallacySeverity,
}

#[derive(Debug, Clone)]
pub enum FallacyType {
    CircularReasoning,
    FalseCorrelation,
    CherryPicking,
    StrawMan,
    AdHominem,
    AppealToAuthority,
    FalseDichotomy,
    SlipperySlope,
    Overgeneralization,
    UnderpoweredStudy,
    MultipleTesting,
}

#[derive(Debug, Clone)]
pub enum FallacySeverity {
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct MethodologicalChecks {
    pub statistical_validity: StatisticalValidation,
    pub experimental_design: ExperimentalDesignValidation,
    pub reproducibility: ReproducibilityChecks,
    pub ethical_considerations: EthicalValidation,
}

#[derive(Debug, Clone)]
pub struct StatisticalValidation {
    pub power_analysis_present: bool,
    pub appropriate_tests_used: bool,
    pub multiple_testing_corrected: bool,
    pub effect_sizes_reported: bool,
    pub confidence_intervals_provided: bool,
    pub assumptions_checked: bool,
}

#[derive(Debug, Clone)]
pub struct ExperimentalDesignValidation {
    pub randomization_used: bool,
    pub blinding_appropriate: bool,
    pub controls_adequate: bool,
    pub sample_size_justified: bool,
    pub confounders_addressed: bool,
}

#[derive(Debug, Clone)]
pub struct ReproducibilityChecks {
    pub data_available: bool,
    pub code_available: bool,
    pub methods_detailed: bool,
    pub materials_specified: bool,
    pub protocols_standardized: bool,
}

#[derive(Debug, Clone)]
pub struct EthicalValidation {
    pub consent_obtained: bool,
    pub irb_approval: bool,
    pub animal_welfare: Option<bool>,
    pub data_privacy: bool,
    pub conflicts_disclosed: bool,
}

#[derive(Debug, Clone)]
pub enum ValidationStatus {
    Valid,
    Warning(Vec<String>),
    Invalid(Vec<String>),
    RequiresReview,
}

impl ScientificArgumentValidator {
    pub fn new() -> Self {
        Self {
            propositions: HashMap::new(),
            evidence_chains: HashMap::new(),
            reasoning_graph: ReasoningGraph {
                nodes: HashMap::new(),
                edges: Vec::new(),
                circular_reasoning_detected: Vec::new(),
                logical_fallacies: Vec::new(),
            },
            methodological_checks: MethodologicalChecks {
                statistical_validity: StatisticalValidation {
                    power_analysis_present: false,
                    appropriate_tests_used: false,
                    multiple_testing_corrected: false,
                    effect_sizes_reported: false,
                    confidence_intervals_provided: false,
                    assumptions_checked: false,
                },
                experimental_design: ExperimentalDesignValidation {
                    randomization_used: false,
                    blinding_appropriate: false,
                    controls_adequate: false,
                    sample_size_justified: false,
                    confounders_addressed: false,
                },
                reproducibility: ReproducibilityChecks {
                    data_available: false,
                    code_available: false,
                    methods_detailed: false,
                    materials_specified: false,
                    protocols_standardized: false,
                },
                ethical_considerations: EthicalValidation {
                    consent_obtained: false,
                    irb_approval: false,
                    animal_welfare: None,
                    data_privacy: false,
                    conflicts_disclosed: false,
                },
            },
        }
    }

    /// Validate a complete scientific argument
    pub fn validate_argument(&mut self, ast: &ast::Node) -> Result<ArgumentValidationReport, TurbulanceError> {
        // Extract propositions, evidence, and reasoning from AST
        self.extract_scientific_elements(ast)?;

        // Validate individual propositions
        self.validate_propositions()?;

        // Validate evidence quality and relevance
        self.validate_evidence()?;

        // Check reasoning coherence
        self.validate_reasoning_coherence()?;

        // Check for logical fallacies
        self.detect_logical_fallacies()?;

        // Validate methodology
        self.validate_methodology()?;

        // Generate comprehensive report
        Ok(self.generate_validation_report())
    }

    fn extract_scientific_elements(&mut self, ast: &ast::Node) -> Result<(), TurbulanceError> {
        match ast {
            ast::Node::PropositionDecl { name, description, requirements, body, .. } => {
                let proposition = PropositionValidation {
                    name: name.clone(),
                    hypothesis: description.clone().unwrap_or_else(|| "No description provided".to_string()),
                    testable: self.assess_testability(&description),
                    falsifiable: self.assess_falsifiability(&description),
                    specific: self.assess_specificity(&description),
                    measurable: self.assess_measurability(&description),
                    evidence_requirements: self.extract_evidence_requirements(requirements),
                    validation_status: ValidationStatus::RequiresReview,
                };

                self.propositions.insert(name.clone(), proposition);

                // Add to reasoning graph
                self.reasoning_graph.nodes.insert(name.clone(), ReasoningNode {
                    id: name.clone(),
                    node_type: ReasoningNodeType::Proposition,
                    content: description.clone().unwrap_or_default(),
                    confidence: 0.5, // Default confidence
                    dependencies: Vec::new(),
                });
            },

            ast::Node::EvidenceDecl { name, collection_method, data_structure, .. } => {
                let evidence = EvidenceValidation {
                    name: name.clone(),
                    evidence_type: self.determine_evidence_type(collection_method),
                    quality_score: self.assess_evidence_quality(collection_method, data_structure),
                    sample_size: self.extract_sample_size(data_structure),
                    statistical_power: self.calculate_statistical_power(data_structure),
                    controls_present: self.check_controls_present(collection_method),
                    bias_assessment: self.assess_bias_risk(collection_method),
                    supports_propositions: Vec::new(),
                    contradicts_propositions: Vec::new(),
                };

                self.evidence_chains.insert(name.clone(), evidence);

                // Add to reasoning graph
                self.reasoning_graph.nodes.insert(name.clone(), ReasoningNode {
                    id: name.clone(),
                    node_type: ReasoningNodeType::Evidence,
                    content: format!("Evidence: {}", name),
                    confidence: 0.7, // Evidence typically has higher confidence
                    dependencies: Vec::new(),
                });
            },

            ast::Node::SupportStmt { hypothesis, evidence, .. } => {
                // Create reasoning edge
                if let (ast::Node::Identifier(hyp_name, _), ast::Node::Identifier(ev_name, _)) = (hypothesis.as_ref(), evidence.as_ref()) {
                    self.reasoning_graph.edges.push(ReasoningEdge {
                        from: ev_name.clone(),
                        to: hyp_name.clone(),
                        relationship: ReasoningRelationship::Supports,
                        strength: 0.8,
                    });

                    // Update evidence validation
                    if let Some(evidence_val) = self.evidence_chains.get_mut(ev_name) {
                        evidence_val.supports_propositions.push(hyp_name.clone());
                    }
                }
            },

            ast::Node::ContradictStmt { hypothesis, evidence, .. } => {
                // Create reasoning edge
                if let (ast::Node::Identifier(hyp_name, _), ast::Node::Identifier(ev_name, _)) = (hypothesis.as_ref(), evidence.as_ref()) {
                    self.reasoning_graph.edges.push(ReasoningEdge {
                        from: ev_name.clone(),
                        to: hyp_name.clone(),
                        relationship: ReasoningRelationship::Contradicts,
                        strength: 0.8,
                    });

                    // Update evidence validation
                    if let Some(evidence_val) = self.evidence_chains.get_mut(ev_name) {
                        evidence_val.contradicts_propositions.push(hyp_name.clone());
                    }
                }
            },

            ast::Node::Block { statements, .. } => {
                for statement in statements {
                    self.extract_scientific_elements(statement)?;
                }
            },

            _ => {
                // Handle other node types as needed
            }
        }

        Ok(())
    }

    fn validate_propositions(&mut self) -> Result<(), TurbulanceError> {
        for (name, proposition) in &mut self.propositions {
            let mut warnings = Vec::new();
            let mut errors = Vec::new();

            // Check if proposition is testable
            if !proposition.testable {
                warnings.push("Proposition may not be testable with current methods".to_string());
            }

            // Check if proposition is falsifiable
            if !proposition.falsifiable {
                errors.push("Proposition is not falsifiable - violates scientific method".to_string());
            }

            // Check if proposition is specific enough
            if !proposition.specific {
                warnings.push("Proposition lacks specificity - consider narrowing scope".to_string());
            }

            // Check if outcomes are measurable
            if !proposition.measurable {
                errors.push("Proposition outcomes are not measurable".to_string());
            }

            // Update validation status
            proposition.validation_status = if !errors.is_empty() {
                ValidationStatus::Invalid(errors)
            } else if !warnings.is_empty() {
                ValidationStatus::Warning(warnings)
            } else {
                ValidationStatus::Valid
            };
        }

        Ok(())
    }

    fn validate_evidence(&mut self) -> Result<(), TurbulanceError> {
        for (name, evidence) in &mut self.evidence_chains {
            // Check evidence quality threshold
            if evidence.quality_score < 0.6 {
                self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                    fallacy_type: FallacyType::UnderpoweredStudy,
                    location: name.clone(),
                    description: "Evidence quality below acceptable threshold".to_string(),
                    severity: FallacySeverity::Warning,
                });
            }

            // Check for adequate sample size
            if let Some(sample_size) = evidence.sample_size {
                if sample_size < 30 {
                    self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                        fallacy_type: FallacyType::UnderpoweredStudy,
                        location: name.clone(),
                        description: "Sample size may be too small for reliable conclusions".to_string(),
                        severity: FallacySeverity::Warning,
                    });
                }
            }

            // Check bias risk
            if matches!(evidence.bias_assessment.overall_bias_risk, BiasLevel::High | BiasLevel::Critical) {
                self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                    fallacy_type: FallacyType::CherryPicking,
                    location: name.clone(),
                    description: "High risk of bias detected in evidence".to_string(),
                    severity: FallacySeverity::Error,
                });
            }
        }

        Ok(())
    }

    fn validate_reasoning_coherence(&mut self) -> Result<(), TurbulanceError> {
        // Check for circular reasoning
        self.detect_circular_reasoning()?;

        // Check for contradictory evidence
        self.detect_contradictory_evidence()?;

        // Check for unsupported claims
        self.detect_unsupported_claims()?;

        Ok(())
    }

    fn detect_circular_reasoning(&mut self) -> Result<(), TurbulanceError> {
        // Implement cycle detection in reasoning graph
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for node_id in self.reasoning_graph.nodes.keys() {
            if !visited.contains(node_id) {
                if self.has_cycle(node_id, &mut visited, &mut rec_stack) {
                    self.reasoning_graph.circular_reasoning_detected.push(node_id.clone());
                    self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                        fallacy_type: FallacyType::CircularReasoning,
                        location: node_id.clone(),
                        description: "Circular reasoning detected in argument chain".to_string(),
                        severity: FallacySeverity::Error,
                    });
                }
            }
        }

        Ok(())
    }

    fn has_cycle(&self, node_id: &str, visited: &mut std::collections::HashSet<String>, rec_stack: &mut std::collections::HashSet<String>) -> bool {
        visited.insert(node_id.to_string());
        rec_stack.insert(node_id.to_string());

        // Check all adjacent nodes
        for edge in &self.reasoning_graph.edges {
            if edge.from == node_id {
                let neighbor = &edge.to;
                if !visited.contains(neighbor) && self.has_cycle(neighbor, visited, rec_stack) {
                    return true;
                } else if rec_stack.contains(neighbor) {
                    return true;
                }
            }
        }

        rec_stack.remove(node_id);
        false
    }

    fn detect_contradictory_evidence(&mut self) -> Result<(), TurbulanceError> {
        // Check for evidence that both supports and contradicts the same proposition
        for (prop_name, _) in &self.propositions {
            let supporting_evidence: Vec<_> = self.evidence_chains.iter()
                .filter(|(_, ev)| ev.supports_propositions.contains(prop_name))
                .collect();

            let contradicting_evidence: Vec<_> = self.evidence_chains.iter()
                .filter(|(_, ev)| ev.contradicts_propositions.contains(prop_name))
                .collect();

            if !supporting_evidence.is_empty() && !contradicting_evidence.is_empty() {
                self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                    fallacy_type: FallacyType::FalseCorrelation,
                    location: prop_name.clone(),
                    description: "Conflicting evidence detected - requires resolution".to_string(),
                    severity: FallacySeverity::Warning,
                });
            }
        }

        Ok(())
    }

    fn detect_unsupported_claims(&mut self) -> Result<(), TurbulanceError> {
        // Check for propositions without supporting evidence
        for (prop_name, _) in &self.propositions {
            let has_support = self.evidence_chains.iter()
                .any(|(_, ev)| ev.supports_propositions.contains(prop_name));

            if !has_support {
                self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                    fallacy_type: FallacyType::AppealToAuthority,
                    location: prop_name.clone(),
                    description: "Proposition lacks empirical support".to_string(),
                    severity: FallacySeverity::Error,
                });
            }
        }

        Ok(())
    }

    fn detect_logical_fallacies(&mut self) -> Result<(), TurbulanceError> {
        // Additional fallacy detection logic would go here
        // This is already partially implemented in other methods
        Ok(())
    }

    fn validate_methodology(&mut self) -> Result<(), TurbulanceError> {
        // Check statistical methodology
        if !self.methodological_checks.statistical_validity.power_analysis_present {
            self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                fallacy_type: FallacyType::UnderpoweredStudy,
                location: "methodology".to_string(),
                description: "Power analysis not provided".to_string(),
                severity: FallacySeverity::Warning,
            });
        }

        // Check for multiple testing without correction
        if !self.methodological_checks.statistical_validity.multiple_testing_corrected {
            self.reasoning_graph.logical_fallacies.push(LogicalFallacy {
                fallacy_type: FallacyType::MultipleTesting,
                location: "methodology".to_string(),
                description: "Multiple testing correction not applied".to_string(),
                severity: FallacySeverity::Error,
            });
        }

        Ok(())
    }

    fn generate_validation_report(&self) -> ArgumentValidationReport {
        ArgumentValidationReport {
            overall_validity: self.calculate_overall_validity(),
            proposition_validations: self.propositions.clone(),
            evidence_validations: self.evidence_chains.clone(),
            logical_fallacies: self.reasoning_graph.logical_fallacies.clone(),
            methodological_issues: self.identify_methodological_issues(),
            recommendations: self.generate_recommendations(),
        }
    }

    fn calculate_overall_validity(&self) -> OverallValidity {
        let critical_errors = self.reasoning_graph.logical_fallacies.iter()
            .filter(|f| matches!(f.severity, FallacySeverity::Critical))
            .count();

        let errors = self.reasoning_graph.logical_fallacies.iter()
            .filter(|f| matches!(f.severity, FallacySeverity::Error))
            .count();

        let warnings = self.reasoning_graph.logical_fallacies.iter()
            .filter(|f| matches!(f.severity, FallacySeverity::Warning))
            .count();

        if critical_errors > 0 {
            OverallValidity::Invalid
        } else if errors > 0 {
            OverallValidity::RequiresRevision
        } else if warnings > 0 {
            OverallValidity::AcceptableWithConcerns
        } else {
            OverallValidity::Valid
        }
    }

    fn identify_methodological_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if !self.methodological_checks.statistical_validity.power_analysis_present {
            issues.push("Power analysis not provided".to_string());
        }

        if !self.methodological_checks.experimental_design.randomization_used {
            issues.push("Randomization not used in experimental design".to_string());
        }

        if !self.methodological_checks.reproducibility.data_available {
            issues.push("Data not made available for reproducibility".to_string());
        }

        issues
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Generate specific recommendations based on detected issues
        for fallacy in &self.reasoning_graph.logical_fallacies {
            match fallacy.fallacy_type {
                FallacyType::UnderpoweredStudy => {
                    recommendations.push("Consider increasing sample size or conducting power analysis".to_string());
                },
                FallacyType::CircularReasoning => {
                    recommendations.push("Restructure argument to avoid circular reasoning".to_string());
                },
                FallacyType::CherryPicking => {
                    recommendations.push("Include all relevant evidence, not just supporting data".to_string());
                },
                _ => {}
            }
        }

        recommendations
    }

    // Helper methods for assessment
    fn assess_testability(&self, description: &Option<String>) -> bool {
        if let Some(desc) = description {
            // Simple heuristic - look for measurable terms
            desc.contains("measure") || desc.contains("test") || desc.contains("compare") || desc.contains("correlate")
        } else {
            false
        }
    }

    fn assess_falsifiability(&self, description: &Option<String>) -> bool {
        if let Some(desc) = description {
            // Look for specific, testable predictions
            !desc.contains("always") && !desc.contains("never") && !desc.contains("all")
        } else {
            false
        }
    }

    fn assess_specificity(&self, description: &Option<String>) -> bool {
        if let Some(desc) = description {
            // Check for specific terms, numbers, conditions
            desc.len() > 50 && (desc.contains("when") || desc.contains("if") || desc.contains("under"))
        } else {
            false
        }
    }

    fn assess_measurability(&self, description: &Option<String>) -> bool {
        if let Some(desc) = description {
            // Look for quantifiable outcomes
            desc.contains("increase") || desc.contains("decrease") || desc.contains("rate") ||
            desc.contains("level") || desc.contains("amount") || desc.contains("percentage")
        } else {
            false
        }
    }

    fn extract_evidence_requirements(&self, requirements: &Option<Box<ast::Node>>) -> Vec<EvidenceRequirement> {
        // Extract evidence requirements from AST node
        // This would be implemented based on the specific AST structure
        vec![
            EvidenceRequirement {
                evidence_type: EvidenceType::Experimental,
                minimum_quality: 0.7,
                sample_size_required: Some(100),
                statistical_power: Some(0.8),
                controls_required: true,
            }
        ]
    }

    fn determine_evidence_type(&self, collection_method: &ast::Node) -> EvidenceType {
        // Determine evidence type based on collection method
        // This would analyze the AST node to classify the evidence type
        EvidenceType::Experimental
    }

    fn assess_evidence_quality(&self, collection_method: &ast::Node, data_structure: &ast::Node) -> f64 {
        // Assess evidence quality based on methodology
        0.75 // Placeholder
    }

    fn extract_sample_size(&self, data_structure: &ast::Node) -> Option<usize> {
        // Extract sample size from data structure
        Some(100) // Placeholder
    }

    fn calculate_statistical_power(&self, data_structure: &ast::Node) -> Option<f64> {
        // Calculate statistical power if possible
        Some(0.8) // Placeholder
    }

    fn check_controls_present(&self, collection_method: &ast::Node) -> bool {
        // Check if appropriate controls are present
        true // Placeholder
    }

    fn assess_bias_risk(&self, collection_method: &ast::Node) -> BiasAssessment {
        // Assess various types of bias
        BiasAssessment {
            selection_bias: BiasLevel::Low,
            confirmation_bias: BiasLevel::Moderate,
            publication_bias: BiasLevel::Low,
            measurement_bias: BiasLevel::Low,
            overall_bias_risk: BiasLevel::Low,
        }
    }
}

#[derive(Debug)]
pub struct ArgumentValidationReport {
    pub overall_validity: OverallValidity,
    pub proposition_validations: HashMap<String, PropositionValidation>,
    pub evidence_validations: HashMap<String, EvidenceValidation>,
    pub logical_fallacies: Vec<LogicalFallacy>,
    pub methodological_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug)]
pub enum OverallValidity {
    Valid,
    AcceptableWithConcerns,
    RequiresRevision,
    Invalid,
}

impl ArgumentValidationReport {
    pub fn print_report(&self) {
        println!("🔬 Scientific Argument Validation Report");
        println!("==========================================");

        match self.overall_validity {
            OverallValidity::Valid => println!("✅ Overall Validity: VALID"),
            OverallValidity::AcceptableWithConcerns => println!("⚠️  Overall Validity: ACCEPTABLE WITH CONCERNS"),
            OverallValidity::RequiresRevision => println!("❌ Overall Validity: REQUIRES REVISION"),
            OverallValidity::Invalid => println!("🚫 Overall Validity: INVALID"),
        }

        println!("\n📋 Proposition Analysis:");
        for (name, prop) in &self.proposition_validations {
            println!("  {} - {:?}", name, prop.validation_status);
        }

        println!("\n🔍 Evidence Analysis:");
        for (name, evidence) in &self.evidence_validations {
            println!("  {} - Quality: {:.2}", name, evidence.quality_score);
        }

        if !self.logical_fallacies.is_empty() {
            println!("\n⚠️  Logical Issues Detected:");
            for fallacy in &self.logical_fallacies {
                println!("  {:?}: {} ({})", fallacy.severity, fallacy.description, fallacy.location);
            }
        }

        if !self.methodological_issues.is_empty() {
            println!("\n🔬 Methodological Issues:");
            for issue in &self.methodological_issues {
                println!("  • {}", issue);
            }
        }

        if !self.recommendations.is_empty() {
            println!("\n💡 Recommendations:");
            for rec in &self.recommendations {
                println!("  • {}", rec);
            }
        }
    }
}

// Re-export key quantum computing interface types
pub use quantum_lexer::QuantumTokenKind;
pub use semantic_engine::{
    SemanticEngine, CognitiveFrame, UserProfile, ContaminationSequence,
    ContaminationMetrics, SemanticCatalyst, CatalyticResult
};
pub use v8_intelligence::{
    V8IntelligenceNetwork, IntelligenceModule, ProcessingInput, ProcessingOutput,
    MzekezekeBayesian, ZengezaSignal, DiggidenAdversarial, SpectacularParadigm,
    NetworkStatus
};
pub use four_file_system::{
    FourFileSystem, ProcessedFile, FileType, ProcessedContent,
    TrbScript, FsVisualization, GhdNetwork, HreLog,
    SystemState, ExecutionResult
};
