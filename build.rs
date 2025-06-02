// Build script for kwasa-kwasa
// This script handles code generation and build configuration for the framework

use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn generate_prelude(out_dir: &Path) -> io::Result<()> {
    let prelude_path = out_dir.join("prelude.rs");
    let mut file = File::create(&prelude_path)?;
    
    writeln!(file, "// Auto-generated prelude for Turbulance generated code")?;
    writeln!(file, "// DO NOT EDIT MANUALLY")?;
    writeln!(file, "")?;
    writeln!(file, "pub use std::collections::HashMap;")?;
    writeln!(file, "pub use crate::turbulance::{{TokenKind, Result, TurbulanceError}};")?;
    writeln!(file, "pub use crate::turbulance::interpreter::{{Value, NativeFunction}};")?;
    
    Ok(())
}

fn main() -> io::Result<()> {
    // Track dependencies for rebuild
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=src/turbulance/");
    println!("cargo:rerun-if-changed=grammar/");
    println!("cargo:rerun-if-changed=build.rs");
    
    // Get output directory for generated files
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Create necessary directories
    let generated_dir = out_dir.join("generated");
    fs::create_dir_all(&generated_dir)?;
    
    // Generate prelude first
    generate_prelude(&generated_dir)?;
    
    // Generate parser tables from grammar definitions
    generate_parser_tables(&generated_dir)?;
    
    // Generate language bindings for stdlib functions
    generate_stdlib_bindings(&generated_dir)?;
    
    // Generate token definitions from lexer specification
    generate_token_definitions(&generated_dir)?;
    
    // Generate serialization code for AST nodes
    generate_ast_serialization(&generated_dir)?;
    
    // Set feature flags for build configuration
    configure_build_features();
    
    // Include path to generated files
    println!("cargo:rustc-env=TURBULANCE_GENERATED_DIR={}", generated_dir.display());
    
    println!("cargo:warning=Build script executed successfully");
    Ok(())
}

fn generate_parser_tables(out_dir: &Path) -> io::Result<()> {
    // Create the parser tables file
    let parser_tables_path = out_dir.join("parser_tables.rs");
    let mut file = File::create(&parser_tables_path)?;
    
    // Generate parser lookup tables for keywords, operators and productions
    writeln!(file, "// Auto-generated parser tables for Turbulance language")?;
    writeln!(file, "// DO NOT EDIT MANUALLY")?;
    writeln!(file, "")?;
    writeln!(file, "use super::prelude::*;")?;
    writeln!(file, "")?;
    
    // Keywords table
    writeln!(file, "pub fn keywords_table() -> HashMap<&'static str, crate::turbulance::TokenKind> {{")?;
    writeln!(file, "    let mut keywords = HashMap::new();")?;
    writeln!(file, "    keywords.insert(\"funxn\", crate::turbulance::TokenKind::FunctionDecl);")?;
    writeln!(file, "    keywords.insert(\"within\", crate::turbulance::TokenKind::Within);")?;
    writeln!(file, "    keywords.insert(\"given\", crate::turbulance::TokenKind::Given);")?;
    writeln!(file, "    keywords.insert(\"project\", crate::turbulance::TokenKind::ProjectDecl);")?;
    writeln!(file, "    keywords.insert(\"ensure\", crate::turbulance::TokenKind::Ensure);")?;
    writeln!(file, "    keywords.insert(\"return\", crate::turbulance::TokenKind::Return);")?;
    writeln!(file, "    keywords.insert(\"proposition\", crate::turbulance::TokenKind::Proposition);")?;
    writeln!(file, "    keywords.insert(\"motion\", crate::turbulance::TokenKind::Motion);")?;
    writeln!(file, "    keywords.insert(\"considering\", crate::turbulance::TokenKind::Considering);")?;
    writeln!(file, "    keywords.insert(\"allow\", crate::turbulance::TokenKind::Allow);")?;
    writeln!(file, "    keywords.insert(\"var\", crate::turbulance::TokenKind::Var);")?;
    writeln!(file, "    keywords.insert(\"if\", crate::turbulance::TokenKind::If);")?;
    writeln!(file, "    keywords.insert(\"else\", crate::turbulance::TokenKind::Else);")?;
    writeln!(file, "    keywords.insert(\"true\", crate::turbulance::TokenKind::True);")?;
    writeln!(file, "    keywords.insert(\"false\", crate::turbulance::TokenKind::False);")?;
    writeln!(file, "    keywords")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    
    // Operator precedence table
    writeln!(file, "pub fn operator_precedence() -> HashMap<crate::turbulance::TokenKind, u8> {{")?;
    writeln!(file, "    let mut precedence = HashMap::new();")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::Plus, 10);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::Minus, 10);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::Multiply, 20);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::Divide, 20);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::Equal, 5);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::NotEqual, 5);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::LessThan, 5);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::GreaterThan, 5);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::LessThanEqual, 5);")?;
    writeln!(file, "    precedence.insert(crate::turbulance::TokenKind::GreaterThanEqual, 5);")?;
    writeln!(file, "    precedence")?;
    writeln!(file, "}}")?;
    
    Ok(())
}

fn generate_stdlib_bindings(out_dir: &Path) -> io::Result<()> {
    // Create stdlib bindings file
    let stdlib_path = out_dir.join("stdlib_bindings.rs");
    let mut file = File::create(&stdlib_path)?;
    
    // Generate bindings between Rust implementations and Turbulance functions
    writeln!(file, "// Auto-generated standard library bindings for Turbulance")?;
    writeln!(file, "// DO NOT EDIT MANUALLY")?;
    writeln!(file, "")?;
    writeln!(file, "use super::prelude::*;")?;
    writeln!(file, "")?;
    writeln!(file, "// Define standard function type for consistent casting")?;
    writeln!(file, "type StdlibFnType = fn(Vec<Value>) -> Result<Value>;")?;
    writeln!(file, "")?;
    
    // Generate function registry
    writeln!(file, "pub fn stdlib_functions() -> HashMap<&'static str, NativeFunction> {{")?;
    writeln!(file, "    let mut functions = HashMap::new();")?;
    writeln!(file, "")?;
    
    // Text analysis functions
    writeln!(file, "    // Text analysis functions")?;
    writeln!(file, "    functions.insert(\"readability_score\", Box::new(crate::turbulance::stdlib::text_analysis::readability_score as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"sentiment_analysis\", Box::new(crate::turbulance::stdlib::text_analysis::sentiment_analysis as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"extract_keywords\", Box::new(crate::turbulance::stdlib::text_analysis::extract_keywords as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"contains\", Box::new(crate::turbulance::stdlib::text_analysis::contains as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"extract_patterns\", Box::new(crate::turbulance::stdlib::text_analysis::extract_patterns as StdlibFnType));")?;
    
    // Statistical analysis functions
    writeln!(file, "")?;
    writeln!(file, "    // Statistical analysis functions")?;
    writeln!(file, "    functions.insert(\"ngram_probability\", Box::new(crate::turbulance::stdlib::text_analysis::ngram_probability as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"conditional_probability\", Box::new(crate::turbulance::stdlib::text_analysis::conditional_probability as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"positional_distribution\", Box::new(crate::turbulance::stdlib::text_analysis::positional_distribution as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"entropy_measure\", Box::new(crate::turbulance::stdlib::text_analysis::entropy_measure as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"sequence_significance\", Box::new(crate::turbulance::stdlib::text_analysis::sequence_significance as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"markov_transition\", Box::new(crate::turbulance::stdlib::text_analysis::markov_transition as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"zipf_analysis\", Box::new(crate::turbulance::stdlib::text_analysis::zipf_analysis as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"positional_entropy\", Box::new(crate::turbulance::stdlib::text_analysis::positional_entropy as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"contextual_uniqueness\", Box::new(crate::turbulance::stdlib::text_analysis::contextual_uniqueness as StdlibFnType));")?;
    
    // Cross-domain statistical analysis functions
    writeln!(file, "")?;
    writeln!(file, "    // Cross-domain statistical analysis functions")?;
    writeln!(file, "    functions.insert(\"motif_enrichment\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::motif_enrichment as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"spectral_correlation\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::spectral_correlation as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"evidence_likelihood\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::evidence_likelihood as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"uncertainty_propagation\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::uncertainty_propagation as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"bayesian_update\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::bayesian_update as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"confidence_interval\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::confidence_interval as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"cross_domain_correlation\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::cross_domain_correlation as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"false_discovery_rate\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::false_discovery_rate as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"permutation_significance\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::permutation_significance as StdlibFnType));")?;
    
    // Positional importance analysis functions
    writeln!(file, "")?;
    writeln!(file, "    // Positional importance analysis functions")?;
    writeln!(file, "    functions.insert(\"positional_importance\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::positional_importance as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"section_weight_map\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::section_weight_map as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"structural_prominence\", Box::new(crate::turbulance::stdlib::cross_domain_analysis::structural_prominence as StdlibFnType));")?;
    
    // Text transformation functions
    writeln!(file, "")?;
    writeln!(file, "    // Text transformation functions")?;
    writeln!(file, "    functions.insert(\"simplify_sentences\", Box::new(crate::turbulance::stdlib::text_transform::simplify_sentences as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"replace_jargon\", Box::new(crate::turbulance::stdlib::text_transform::replace_jargon as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"formalize\", Box::new(crate::turbulance::stdlib::text_transform::formalize as StdlibFnType));")?;
    
    // Research assistance functions
    writeln!(file, "")?;
    writeln!(file, "    // Research assistance functions")?;
    writeln!(file, "    functions.insert(\"research_context\", Box::new(crate::turbulance::stdlib::research::research_context as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"fact_check\", Box::new(crate::turbulance::stdlib::research::fact_check as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"ensure_explanation_follows\", Box::new(crate::turbulance::stdlib::research::ensure_explanation_follows as StdlibFnType));")?;
    
    // Utility functions
    writeln!(file, "")?;
    writeln!(file, "    // Utility functions")?;
    writeln!(file, "    functions.insert(\"print\", Box::new(crate::turbulance::stdlib::utils::print as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"len\", Box::new(crate::turbulance::stdlib::utils::len as StdlibFnType));")?;
    writeln!(file, "    functions.insert(\"typeof\", Box::new(crate::turbulance::stdlib::utils::typeof_fn as StdlibFnType));")?;
    
    writeln!(file, "")?;
    writeln!(file, "    functions")?;
    writeln!(file, "}}")?;
    
    Ok(())
}

fn generate_token_definitions(out_dir: &Path) -> io::Result<()> {
    // Create token definitions file
    let tokens_path = out_dir.join("token_definitions.rs");
    let mut file = File::create(&tokens_path)?;
    
    // Generate token definitions for the lexer
    writeln!(file, "// Auto-generated token definitions for Turbulance language")?;
    writeln!(file, "// DO NOT EDIT MANUALLY")?;
    writeln!(file, "")?;
    writeln!(file, "use logos::Logos;")?;
    writeln!(file, "")?;
    writeln!(file, "#[derive(Logos, Debug, Clone, PartialEq, Eq, Hash)]")?;
    writeln!(file, "#[logos(skip r\"[ \\t\\n\\f]+\")]")?;
    writeln!(file, "#[logos(skip r\"//.*\")]")?;
    writeln!(file, "pub enum TokenKindGenerated {{")?;
    writeln!(file, "    #[token(\"funxn\")]")?;
    writeln!(file, "    Function,")?;
    writeln!(file, "    #[token(\"within\")]")?;
    writeln!(file, "    Within,")?;
    writeln!(file, "    #[token(\"given\")]")?;
    writeln!(file, "    Given,")?;
    writeln!(file, "    #[token(\"project\")]")?;
    writeln!(file, "    Project,")?;
    writeln!(file, "    #[token(\"ensure\")]")?;
    writeln!(file, "    Ensure,")?;
    writeln!(file, "    #[token(\"return\")]")?;
    writeln!(file, "    Return,")?;
    writeln!(file, "    #[token(\"proposition\")]")?;
    writeln!(file, "    Proposition,")?;
    writeln!(file, "    #[token(\"motion\")]")?;
    writeln!(file, "    Motion,")?;
    writeln!(file, "    #[token(\"considering\")]")?;
    writeln!(file, "    Considering,")?;
    writeln!(file, "    #[token(\"allow\")]")?;
    writeln!(file, "    Allow,")?;
    writeln!(file, "    #[token(\"var\")]")?;
    writeln!(file, "    Var,")?;
    writeln!(file, "    #[token(\"if\")]")?;
    writeln!(file, "    If,")?;
    writeln!(file, "    #[token(\"else\")]")?;
    writeln!(file, "    Else,")?;
    writeln!(file, "    #[token(\"true\")]")?;
    writeln!(file, "    True,")?;
    writeln!(file, "    #[token(\"false\")]")?;
    writeln!(file, "    False,")?;
    writeln!(file, "    #[token(\"(\")]")?;
    writeln!(file, "    LeftParen,")?;
    writeln!(file, "    #[token(\")\")]")?;
    writeln!(file, "    RightParen,")?;
    writeln!(file, "    #[token(\"{{\")]")?;
    writeln!(file, "    LeftBrace,")?;
    writeln!(file, "    #[token(\"}}\")]")?;
    writeln!(file, "    RightBrace,")?;
    writeln!(file, "    #[token(\"[\")]")?;
    writeln!(file, "    LeftBracket,")?;
    writeln!(file, "    #[token(\"]\")]")?;
    writeln!(file, "    RightBracket,")?;
    writeln!(file, "    #[token(\",\")]")?;
    writeln!(file, "    Comma,")?;
    writeln!(file, "    #[token(\".\")]")?;
    writeln!(file, "    Dot,")?;
    writeln!(file, "    #[token(\":\")]")?;
    writeln!(file, "    Colon,")?;
    writeln!(file, "    #[token(\"+\")]")?;
    writeln!(file, "    Plus,")?;
    writeln!(file, "    #[token(\"-\")]")?;
    writeln!(file, "    Minus,")?;
    writeln!(file, "    #[token(\"*\")]")?;
    writeln!(file, "    Star,")?;
    writeln!(file, "    #[token(\"/\")]")?;
    writeln!(file, "    Slash,")?;
    writeln!(file, "    #[token(\"=\")]")?;
    writeln!(file, "    Assign,")?;
    writeln!(file, "    #[token(\"==\")]")?;
    writeln!(file, "    Equal,")?;
    writeln!(file, "    #[token(\"!=\")]")?;
    writeln!(file, "    NotEqual,")?;
    writeln!(file, "    #[token(\"<\")]")?;
    writeln!(file, "    LessThan,")?;
    writeln!(file, "    #[token(\">\")]")?;
    writeln!(file, "    GreaterThan,")?;
    writeln!(file, "    #[token(\"<=\")]")?;
    writeln!(file, "    LessThanEqual,")?;
    writeln!(file, "    #[token(\">=\")]")?;
    writeln!(file, "    GreaterThanEqual,")?;
    writeln!(file, "    #[regex(r#\"[a-zA-Z_][a-zA-Z0-9_]*\"#)]")?;
    writeln!(file, "    Identifier,")?;
    writeln!(file, "    #[regex(r#\"\\d+\"#)]")?;
    writeln!(file, "    Number,")?;
    writeln!(file, "    #[regex(r#\"\\d+\\.\\d+\"#)]")?;
    writeln!(file, "    Float,")?;
    writeln!(file, "    #[regex(r#\"\"[^\"]*\"\"#)]")?;
    writeln!(file, "    String,")?;
    writeln!(file, "    #[error]")?;
    writeln!(file, "    Error,")?;
    writeln!(file, "    Eof,")?;
    writeln!(file, "}}")?;
    
    Ok(())
}

fn generate_ast_serialization(out_dir: &Path) -> io::Result<()> {
    // Create AST serialization file
    let ast_path = out_dir.join("ast_serialization.rs");
    let mut file = File::create(&ast_path)?;
    
    // Generate serialization code for AST nodes
    writeln!(file, "// Auto-generated AST serialization code for Turbulance language")?;
    writeln!(file, "// DO NOT EDIT MANUALLY")?;
    writeln!(file, "")?;
    writeln!(file, "use serde::{{Serialize, Deserialize}};")?;
    writeln!(file, "use crate::turbulance::ast::*;")?;
    writeln!(file, "")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub struct SerializableAst {{")?;
    writeln!(file, "    pub declarations: Vec<SerializableDeclaration>,")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "impl From<&Program> for SerializableAst {{")?;
    writeln!(file, "    fn from(program: &Program) -> Self {{")?;
    writeln!(file, "        SerializableAst {{")?;
    writeln!(file, "            declarations: program.declarations.iter().map(|d| d.into()).collect(),")?;
    writeln!(file, "        }}")?;
    writeln!(file, "    }}")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub enum SerializableDeclaration {{")?;
    writeln!(file, "    Function {{name: String, params: Vec<String>, body: SerializableStatement}},")?;
    writeln!(file, "    Project {{name: String, body: SerializableStatement}},")?;
    writeln!(file, "    Proposition {{name: String, motions: Vec<SerializableMotion>}},")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "impl From<&crate::turbulance::ast::Declaration> for SerializableDeclaration {{")?;
    writeln!(file, "    fn from(decl: &crate::turbulance::ast::Declaration) -> Self {{")?;
    writeln!(file, "        match decl {{")?;
    writeln!(file, "            crate::turbulance::ast::Declaration::Function {{name, params, body}} => {{")?;
    writeln!(file, "                SerializableDeclaration::Function {{")?;
    writeln!(file, "                    name: name.clone(),")?;
    writeln!(file, "                    params: params.clone(),")?;
    writeln!(file, "                    body: body.into(),")?;
    writeln!(file, "                }}")?;
    writeln!(file, "            }},")?;
    writeln!(file, "            crate::turbulance::ast::Declaration::Project {{name, body}} => {{")?;
    writeln!(file, "                SerializableDeclaration::Project {{")?;
    writeln!(file, "                    name: name.clone(),")?;
    writeln!(file, "                    body: body.into(),")?;
    writeln!(file, "                }}")?;
    writeln!(file, "            }},")?;
    writeln!(file, "            // Add other declaration types based on your AST structure")?;
    writeln!(file, "            _ => unimplemented!(\"Other declaration types\"),")?;
    writeln!(file, "        }}")?;
    writeln!(file, "    }}")?;
    writeln!(file, "}}")?;
    
    // Add more serialization implementations as needed
    writeln!(file, "")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub enum SerializableStatement {{")?;
    writeln!(file, "    Block {{ statements: Vec<SerializableStatement> }},")?;
    writeln!(file, "    Within {{ target: SerializableExpression, body: Box<SerializableStatement> }},")?;
    writeln!(file, "    Given {{ condition: SerializableExpression, body: Box<SerializableStatement> }},")?;
    writeln!(file, "    Ensure {{ condition: SerializableExpression }},")?;
    writeln!(file, "    Return {{ value: Option<SerializableExpression> }},")?;
    writeln!(file, "    Expression {{ expr: SerializableExpression }},")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "// Placeholder for motion serialization")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub struct SerializableMotion {{")?;
    writeln!(file, "    pub name: String,")?;
    writeln!(file, "    pub content: String,")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub enum SerializableExpression {{")?;
    writeln!(file, "    Literal {{ value: SerializableValue }},")?;
    writeln!(file, "    Variable {{ name: String }},")?;
    writeln!(file, "    Binary {{ left: Box<SerializableExpression>, operator: String, right: Box<SerializableExpression> }},")?;
    writeln!(file, "    Call {{ callee: String, arguments: Vec<SerializableExpression> }},")?;
    writeln!(file, "}}")?;
    writeln!(file, "")?;
    writeln!(file, "#[derive(Serialize, Deserialize, Debug, Clone)]")?;
    writeln!(file, "pub enum SerializableValue {{")?;
    writeln!(file, "    Number(f64),")?;
    writeln!(file, "    String(String),")?;
    writeln!(file, "    Boolean(bool),")?;
    writeln!(file, "    Null,")?;
    writeln!(file, "}}")?;
    
    Ok(())
}

fn configure_build_features() {
    // Enable wasm feature when building for wasm target
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("wasm32") {
        println!("cargo:rustc-cfg=feature=\"wasm\"");
    }
    
    // Set optimization level for release builds
    if env::var("PROFILE").unwrap_or_default() == "release" {
        println!("cargo:rustc-cfg=feature=\"optimized\"");
    }
    
    // Enable debugging features in debug mode
    if env::var("PROFILE").unwrap_or_default() == "debug" {
        println!("cargo:rustc-cfg=feature=\"debug_mode\"");
    }
}
