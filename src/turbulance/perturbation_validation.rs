/// Perturbation Validation System for Kwasa-Kwasa
/// 
/// This module implements systematic linguistic perturbation as a validation
/// mechanism for probabilistic resolutions, allowing us to test the robustness
/// and reliability of uncertain interpretations.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::probabilistic::{TextPoint, ResolutionResult, ResolutionManager, ResolutionContext};
use crate::turbulance::positional_semantics::{PositionalSentence, PositionalAnalyzer};

/// Validator that tests resolution quality through systematic perturbation
pub struct PerturbationValidator {
    /// The point being validated
    point: TextPoint,
    
    /// Initial resolution result
    initial_resolution: ResolutionResult,
    
    /// Perturbation tests to run
    perturbation_tests: Vec<PerturbationTest>,
    
    /// Threshold for stability scoring
    stability_threshold: f64,
    
    /// Resolution manager for creating new resolutions
    resolution_manager: ResolutionManager,
    
    /// Positional analyzer for linguistic analysis
    positional_analyzer: PositionalAnalyzer,
    
    /// Cache for perturbation results
    perturbation_cache: HashMap<String, ValidationResult>,
}

/// Configuration for perturbation validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum stability score for acceptance
    pub min_stability_score: f64,
    
    /// Enable word removal tests
    pub enable_word_removal: bool,
    
    /// Enable positional rearrangement tests
    pub enable_positional_rearrangement: bool,
    
    /// Enable synonym substitution tests
    pub enable_synonym_substitution: bool,
    
    /// Enable negation consistency tests
    pub enable_negation_tests: bool,
    
    /// Maximum number of perturbations per test type
    pub max_perturbations_per_type: usize,
    
    /// Context for resolution testing
    pub resolution_context: ResolutionContext,
    
    /// Depth of validation (quick, thorough, exhaustive)
    pub validation_depth: ValidationDepth,
}

/// Depth of validation testing
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ValidationDepth {
    /// Quick validation with basic tests
    Quick,
    
    /// Thorough validation with comprehensive tests
    Thorough,
    
    /// Exhaustive validation with all possible perturbations
    Exhaustive,
    
    /// Custom validation with specified test types
    Custom(Vec<PerturbationType>),
}

/// Types of perturbation tests
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PerturbationType {
    /// Remove individual words
    WordRemoval,
    
    /// Rearrange word positions
    PositionalRearrangement,
    
    /// Substitute with synonyms
    SynonymSubstitution,
    
    /// Test logical consistency with negation
    NegationConsistency,
    
    /// Add noise words
    NoiseAddition,
    
    /// Test grammatical variants
    GrammaticalVariation,
    
    /// Test punctuation sensitivity
    PunctuationVariation,
    
    /// Test case sensitivity
    CaseVariation,
}

/// Individual perturbation test
#[derive(Clone, Debug)]
pub struct PerturbationTest {
    /// Type of perturbation
    pub test_type: PerturbationType,
    
    /// Original text point
    pub original_point: TextPoint,
    
    /// Perturbed version
    pub perturbed_point: TextPoint,
    
    /// Description of the perturbation
    pub description: String,
    
    /// Expected impact level
    pub expected_impact: ImpactLevel,
}

/// Expected impact of a perturbation
#[derive(Clone, Debug, PartialEq)]
pub enum ImpactLevel {
    /// Minimal impact expected (function words, punctuation)
    Minimal,
    
    /// Moderate impact expected (word order changes)
    Moderate,
    
    /// Major impact expected (content word removal)
    Major,
    
    /// Logical impact expected (negation, contradiction)
    Logical,
}

/// Result of perturbation validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall stability score (0.0-1.0)
    pub stability_score: f64,
    
    /// Results of individual perturbation tests
    pub perturbation_results: Vec<PerturbationResult>,
    
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
    
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
    
    /// Vulnerable aspects identified
    pub vulnerable_aspects: Vec<String>,
    
    /// Robust aspects identified
    pub robust_aspects: Vec<String>,
    
    /// Processing metadata
    pub metadata: ValidationMetadata,
}

/// Result of a single perturbation test
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerturbationResult {
    /// Test type performed
    pub test_type: PerturbationType,
    
    /// Description of perturbation
    pub description: String,
    
    /// Original resolution confidence
    pub original_confidence: f64,
    
    /// Perturbed resolution confidence
    pub perturbed_confidence: f64,
    
    /// Confidence change (delta)
    pub confidence_change: f64,
    
    /// Stability score for this test (0.0-1.0)
    pub stability_score: f64,
    
    /// Whether this test passed stability threshold
    pub passed: bool,
    
    /// Impact assessment
    pub impact_assessment: ImpactAssessment,
}

/// Assessment of perturbation impact
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Actual impact level observed
    pub actual_impact: ImpactLevel,
    
    /// Expected impact level
    pub expected_impact: ImpactLevel,
    
    /// Whether impact matched expectations
    pub impact_as_expected: bool,
    
    /// Explanation of impact
    pub explanation: String,
    
    /// Linguistic features affected
    pub affected_features: Vec<String>,
}

/// Overall quality assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Confidence in the resolution
    pub confidence_in_resolution: f64,
    
    /// Resolution reliability category
    pub reliability_category: ReliabilityCategory,
    
    /// Specific quality metrics
    pub quality_metrics: HashMap<String, f64>,
    
    /// Risk factors identified
    pub risk_factors: Vec<String>,
    
    /// Strength factors identified
    pub strength_factors: Vec<String>,
}

/// Categories of resolution reliability
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ReliabilityCategory {
    /// Highly reliable resolution
    HighlyReliable,
    
    /// Moderately reliable resolution
    ModeratelyReliable,
    
    /// Questionable reliability
    Questionable,
    
    /// Unreliable resolution
    Unreliable,
    
    /// Requires human review
    RequiresReview,
}

/// Metadata about validation processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Total validation time (ms)
    pub validation_time_ms: u64,
    
    /// Number of tests performed
    pub tests_performed: usize,
    
    /// Tests that passed
    pub tests_passed: usize,
    
    /// Average stability score
    pub average_stability: f64,
    
    /// Validation depth used
    pub validation_depth: ValidationDepth,
    
    /// Warnings generated
    pub warnings: Vec<String>,
}

impl PerturbationValidator {
    /// Create a new perturbation validator
    pub fn new(
        point: TextPoint,
        initial_resolution: ResolutionResult,
        config: ValidationConfig,
    ) -> Self {
        Self {
            point,
            initial_resolution,
            perturbation_tests: Vec::new(),
            stability_threshold: config.min_stability_score,
            resolution_manager: ResolutionManager::new(),
            positional_analyzer: PositionalAnalyzer::new(),
            perturbation_cache: HashMap::new(),
        }
    }
    
    /// Run complete validation process
    pub async fn run_validation(&mut self, config: &ValidationConfig) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        
        // Generate perturbation tests based on configuration
        self.generate_perturbation_tests(config)?;
        
        // Execute all perturbation tests
        let mut perturbation_results = Vec::new();
        for test in &self.perturbation_tests {
            let result = self.execute_perturbation_test(test, config).await?;
            perturbation_results.push(result);
        }
        
        // Calculate overall stability score
        let stability_score = self.calculate_overall_stability(&perturbation_results);
        
        // Assess quality
        let quality_assessment = self.assess_quality(&perturbation_results, stability_score);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&perturbation_results, &quality_assessment);
        
        // Identify vulnerable and robust aspects
        let (vulnerable_aspects, robust_aspects) = self.identify_aspects(&perturbation_results);
        
        // Create metadata
        let metadata = ValidationMetadata {
            validation_time_ms: start_time.elapsed().as_millis() as u64,
            tests_performed: perturbation_results.len(),
            tests_passed: perturbation_results.iter().filter(|r| r.passed).count(),
            average_stability: perturbation_results.iter().map(|r| r.stability_score).sum::<f64>() / perturbation_results.len() as f64,
            validation_depth: config.validation_depth.clone(),
            warnings: Vec::new(),
        };
        
        Ok(ValidationResult {
            stability_score,
            perturbation_results,
            quality_assessment,
            recommendations,
            vulnerable_aspects,
            robust_aspects,
            metadata,
        })
    }
    
    /// Generate perturbation tests based on configuration
    fn generate_perturbation_tests(&mut self, config: &ValidationConfig) -> Result<()> {
        self.perturbation_tests.clear();
        
        let test_types = match &config.validation_depth {
            ValidationDepth::Quick => vec![PerturbationType::WordRemoval, PerturbationType::NegationConsistency],
            ValidationDepth::Thorough => vec![
                PerturbationType::WordRemoval,
                PerturbationType::PositionalRearrangement,
                PerturbationType::SynonymSubstitution,
                PerturbationType::NegationConsistency,
            ],
            ValidationDepth::Exhaustive => vec![
                PerturbationType::WordRemoval,
                PerturbationType::PositionalRearrangement,
                PerturbationType::SynonymSubstitution,
                PerturbationType::NegationConsistency,
                PerturbationType::NoiseAddition,
                PerturbationType::GrammaticalVariation,
                PerturbationType::PunctuationVariation,
                PerturbationType::CaseVariation,
            ],
            ValidationDepth::Custom(types) => types.clone(),
        };
        
        for test_type in test_types {
            let tests = self.generate_tests_for_type(&test_type, config.max_perturbations_per_type)?;
            self.perturbation_tests.extend(tests);
        }
        
        Ok(())
    }
    
    /// Generate tests for a specific perturbation type
    fn generate_tests_for_type(&mut self, test_type: &PerturbationType, max_tests: usize) -> Result<Vec<PerturbationTest>> {
        match test_type {
            PerturbationType::WordRemoval => self.generate_word_removal_tests(max_tests),
            PerturbationType::PositionalRearrangement => self.generate_rearrangement_tests(max_tests),
            PerturbationType::SynonymSubstitution => self.generate_synonym_tests(max_tests),
            PerturbationType::NegationConsistency => self.generate_negation_tests(max_tests),
            PerturbationType::NoiseAddition => self.generate_noise_tests(max_tests),
            PerturbationType::GrammaticalVariation => self.generate_grammar_tests(max_tests),
            PerturbationType::PunctuationVariation => self.generate_punctuation_tests(max_tests),
            PerturbationType::CaseVariation => self.generate_case_tests(max_tests),
        }
    }
    
    /// Generate word removal tests
    fn generate_word_removal_tests(&mut self, max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        let words: Vec<&str> = self.point.content.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate().take(max_tests) {
            let mut perturbed_words = words.clone();
            perturbed_words.remove(i);
            let perturbed_content = perturbed_words.join(" ");
            
            if !perturbed_content.trim().is_empty() {
                let perturbed_point = TextPoint::new(perturbed_content, self.point.confidence * 0.9);
                
                let expected_impact = if self.is_content_word(word) {
                    ImpactLevel::Major
                } else {
                    ImpactLevel::Minimal
                };
                
                tests.push(PerturbationTest {
                    test_type: PerturbationType::WordRemoval,
                    original_point: self.point.clone(),
                    perturbed_point,
                    description: format!("Remove word: '{}'", word),
                    expected_impact,
                });
            }
        }
        
        Ok(tests)
    }
    
    /// Generate positional rearrangement tests
    fn generate_rearrangement_tests(&mut self, max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        let words: Vec<&str> = self.point.content.split_whitespace().collect();
        
        if words.len() < 2 {
            return Ok(tests);
        }
        
        // Generate a few common rearrangements
        let rearrangements = [
            // Reverse order
            {
                let mut reversed = words.clone();
                reversed.reverse();
                reversed
            },
            // Move first word to end
            {
                let mut moved = words.clone();
                if !moved.is_empty() {
                    let first = moved.remove(0);
                    moved.push(first);
                }
                moved
            },
            // Move last word to beginning
            {
                let mut moved = words.clone();
                if !moved.is_empty() {
                    let last = moved.pop().unwrap();
                    moved.insert(0, last);
                }
                moved
            },
        ];
        
        for (i, rearranged) in rearrangements.iter().enumerate().take(max_tests) {
            let perturbed_content = rearranged.join(" ");
            let perturbed_point = TextPoint::new(perturbed_content, self.point.confidence * 0.8);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::PositionalRearrangement,
                original_point: self.point.clone(),
                perturbed_point,
                description: format!("Rearrangement #{}", i + 1),
                expected_impact: ImpactLevel::Moderate,
            });
        }
        
        Ok(tests)
    }
    
    /// Generate synonym substitution tests
    fn generate_synonym_tests(&mut self, max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        let words: Vec<&str> = self.point.content.split_whitespace().collect();
        
        // Simple synonym mappings (in a real implementation, this would use a thesaurus)
        let synonyms = HashMap::from([
            ("good", vec!["excellent", "great", "fine"]),
            ("bad", vec!["poor", "terrible", "awful"]),
            ("big", vec!["large", "huge", "massive"]),
            ("small", vec!["tiny", "little", "minor"]),
            ("fast", vec!["quick", "rapid", "swift"]),
            ("slow", vec!["gradual", "leisurely", "sluggish"]),
            ("happy", vec!["joyful", "pleased", "content"]),
            ("sad", vec!["unhappy", "depressed", "sorrowful"]),
        ]);
        
        for (i, word) in words.iter().enumerate().take(max_tests) {
            if let Some(word_synonyms) = synonyms.get(&word.to_lowercase()) {
                for synonym in word_synonyms.iter().take(1) { // Just test one synonym per word
                    let mut perturbed_words = words.clone();
                    perturbed_words[i] = synonym;
                    let perturbed_content = perturbed_words.join(" ");
                    let perturbed_point = TextPoint::new(perturbed_content, self.point.confidence * 0.95);
                    
                    tests.push(PerturbationTest {
                        test_type: PerturbationType::SynonymSubstitution,
                        original_point: self.point.clone(),
                        perturbed_point,
                        description: format!("Replace '{}' with '{}'", word, synonym),
                        expected_impact: ImpactLevel::Minimal,
                    });
                }
            }
        }
        
        Ok(tests)
    }
    
    /// Generate negation consistency tests
    fn generate_negation_tests(&mut self, _max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        
        // Add "not" to negate the statement
        let negated_content = if self.point.content.contains("is ") {
            self.point.content.replace("is ", "is not ")
        } else if self.point.content.contains("are ") {
            self.point.content.replace("are ", "are not ")
        } else if self.point.content.contains("will ") {
            self.point.content.replace("will ", "will not ")
        } else {
            format!("It is not true that {}", self.point.content.to_lowercase())
        };
        
        let perturbed_point = TextPoint::new(negated_content, 1.0 - self.point.confidence);
        
        tests.push(PerturbationTest {
            test_type: PerturbationType::NegationConsistency,
            original_point: self.point.clone(),
            perturbed_point,
            description: "Add negation".to_string(),
            expected_impact: ImpactLevel::Logical,
        });
        
        Ok(tests)
    }
    
    /// Generate noise addition tests
    fn generate_noise_tests(&mut self, max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        let noise_words = ["um", "uh", "like", "you know", "actually"];
        
        for (i, noise) in noise_words.iter().enumerate().take(max_tests) {
            let perturbed_content = format!("{} {}", noise, self.point.content);
            let perturbed_point = TextPoint::new(perturbed_content, self.point.confidence * 0.9);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::NoiseAddition,
                original_point: self.point.clone(),
                perturbed_point,
                description: format!("Add noise word: '{}'", noise),
                expected_impact: ImpactLevel::Minimal,
            });
        }
        
        Ok(tests)
    }
    
    /// Generate grammatical variation tests
    fn generate_grammar_tests(&mut self, _max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        
        // Convert to question form if possible
        if !self.point.content.ends_with('?') {
            let question_content = if self.point.content.starts_with("The ") {
                format!("Is {}?", self.point.content.strip_prefix("The ").unwrap_or(&self.point.content))
            } else {
                format!("Is it true that {}?", self.point.content.to_lowercase())
            };
            
            let perturbed_point = TextPoint::new(question_content, self.point.confidence);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::GrammaticalVariation,
                original_point: self.point.clone(),
                perturbed_point,
                description: "Convert to question form".to_string(),
                expected_impact: ImpactLevel::Minimal,
            });
        }
        
        Ok(tests)
    }
    
    /// Generate punctuation variation tests
    fn generate_punctuation_tests(&mut self, _max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        
        // Remove all punctuation
        let no_punct_content: String = self.point.content.chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect();
            
        if no_punct_content != self.point.content {
            let perturbed_point = TextPoint::new(no_punct_content, self.point.confidence * 0.95);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::PunctuationVariation,
                original_point: self.point.clone(),
                perturbed_point,
                description: "Remove all punctuation".to_string(),
                expected_impact: ImpactLevel::Minimal,
            });
        }
        
        Ok(tests)
    }
    
    /// Generate case variation tests
    fn generate_case_tests(&mut self, _max_tests: usize) -> Result<Vec<PerturbationTest>> {
        let mut tests = Vec::new();
        
        // Convert to all lowercase
        let lowercase_content = self.point.content.to_lowercase();
        if lowercase_content != self.point.content {
            let perturbed_point = TextPoint::new(lowercase_content, self.point.confidence * 0.98);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::CaseVariation,
                original_point: self.point.clone(),
                perturbed_point,
                description: "Convert to lowercase".to_string(),
                expected_impact: ImpactLevel::Minimal,
            });
        }
        
        // Convert to all uppercase
        let uppercase_content = self.point.content.to_uppercase();
        if uppercase_content != self.point.content {
            let perturbed_point = TextPoint::new(uppercase_content, self.point.confidence * 0.98);
            
            tests.push(PerturbationTest {
                test_type: PerturbationType::CaseVariation,
                original_point: self.point.clone(),
                perturbed_point,
                description: "Convert to uppercase".to_string(),
                expected_impact: ImpactLevel::Minimal,
            });
        }
        
        Ok(tests)
    }
    
    /// Execute a single perturbation test
    async fn execute_perturbation_test(&mut self, test: &PerturbationTest, config: &ValidationConfig) -> Result<PerturbationResult> {
        // Get original resolution confidence
        let original_confidence = match &self.initial_resolution {
            ResolutionResult::Certain(value) => 1.0,
            ResolutionResult::Uncertain { aggregated_confidence, .. } => *aggregated_confidence,
            ResolutionResult::Contextual { .. } => 0.7, // Default for contextual
            ResolutionResult::Fuzzy { central_tendency, .. } => *central_tendency,
        };
        
        // Create resolution for perturbed point
        let perturbed_resolution = self.resolution_manager.resolve(
            "probabilistic_len", // Use a standard function for testing
            &test.perturbed_point,
            Some(&config.resolution_context),
        )?;
        
        // Get perturbed resolution confidence
        let perturbed_confidence = match perturbed_resolution {
            ResolutionResult::Certain(_) => 1.0,
            ResolutionResult::Uncertain { aggregated_confidence, .. } => aggregated_confidence,
            ResolutionResult::Contextual { .. } => 0.7,
            ResolutionResult::Fuzzy { central_tendency, .. } => central_tendency,
        };
        
        // Calculate confidence change
        let confidence_change = (original_confidence - perturbed_confidence).abs();
        
        // Calculate stability score (higher is more stable)
        let stability_score = 1.0 - (confidence_change / original_confidence.max(0.1));
        
        // Assess impact
        let impact_assessment = self.assess_impact(test, confidence_change);
        
        // Check if test passed
        let passed = stability_score >= self.stability_threshold;
        
        Ok(PerturbationResult {
            test_type: test.test_type.clone(),
            description: test.description.clone(),
            original_confidence,
            perturbed_confidence,
            confidence_change,
            stability_score,
            passed,
            impact_assessment,
        })
    }
    
    /// Assess the impact of a perturbation
    fn assess_impact(&self, test: &PerturbationTest, confidence_change: f64) -> ImpactAssessment {
        let actual_impact = if confidence_change < 0.1 {
            ImpactLevel::Minimal
        } else if confidence_change < 0.3 {
            ImpactLevel::Moderate
        } else if confidence_change < 0.6 {
            ImpactLevel::Major
        } else {
            ImpactLevel::Logical
        };
        
        let impact_as_expected = actual_impact == test.expected_impact;
        
        let explanation = if impact_as_expected {
            "Impact matched expectations".to_string()
        } else {
            format!("Expected {:?} impact, but observed {:?}", test.expected_impact, actual_impact)
        };
        
        ImpactAssessment {
            actual_impact,
            expected_impact: test.expected_impact.clone(),
            impact_as_expected,
            explanation,
            affected_features: vec![], // Could be populated with detailed analysis
        }
    }
    
    /// Calculate overall stability score
    fn calculate_overall_stability(&self, results: &[PerturbationResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        
        let total_stability: f64 = results.iter().map(|r| r.stability_score).sum();
        total_stability / results.len() as f64
    }
    
    /// Assess overall quality
    fn assess_quality(&self, results: &[PerturbationResult], stability_score: f64) -> QualityAssessment {
        let reliability_category = if stability_score >= 0.9 {
            ReliabilityCategory::HighlyReliable
        } else if stability_score >= 0.7 {
            ReliabilityCategory::ModeratelyReliable
        } else if stability_score >= 0.5 {
            ReliabilityCategory::Questionable
        } else if stability_score >= 0.3 {
            ReliabilityCategory::Unreliable
        } else {
            ReliabilityCategory::RequiresReview
        };
        
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let pass_rate = passed_tests as f64 / results.len() as f64;
        
        let mut quality_metrics = HashMap::new();
        quality_metrics.insert("stability_score".to_string(), stability_score);
        quality_metrics.insert("pass_rate".to_string(), pass_rate);
        quality_metrics.insert("average_confidence_change".to_string(),
            results.iter().map(|r| r.confidence_change).sum::<f64>() / results.len() as f64);
        
        QualityAssessment {
            confidence_in_resolution: stability_score,
            reliability_category,
            quality_metrics,
            risk_factors: vec![],
            strength_factors: vec![],
        }
    }
    
    /// Generate recommendations
    fn generate_recommendations(&self, results: &[PerturbationResult], quality: &QualityAssessment) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match quality.reliability_category {
            ReliabilityCategory::HighlyReliable => {
                recommendations.push("Resolution is highly stable and reliable".to_string());
            },
            ReliabilityCategory::ModeratelyReliable => {
                recommendations.push("Resolution is generally reliable but monitor for edge cases".to_string());
            },
            ReliabilityCategory::Questionable => {
                recommendations.push("Resolution reliability is questionable - gather additional evidence".to_string());
            },
            ReliabilityCategory::Unreliable => {
                recommendations.push("Resolution is unreliable - requires significant improvement".to_string());
            },
            ReliabilityCategory::RequiresReview => {
                recommendations.push("Resolution requires human review before use".to_string());
            },
        }
        
        // Add specific recommendations based on failed tests
        let failed_tests: Vec<&PerturbationResult> = results.iter().filter(|r| !r.passed).collect();
        if !failed_tests.is_empty() {
            recommendations.push(format!("Address {} failed perturbation tests", failed_tests.len()));
        }
        
        recommendations
    }
    
    /// Identify vulnerable and robust aspects
    fn identify_aspects(&self, results: &[PerturbationResult]) -> (Vec<String>, Vec<String>) {
        let mut vulnerable = Vec::new();
        let mut robust = Vec::new();
        
        for result in results {
            if result.stability_score < 0.5 {
                vulnerable.push(format!("{:?}: {}", result.test_type, result.description));
            } else if result.stability_score > 0.8 {
                robust.push(format!("{:?}: {}", result.test_type, result.description));
            }
        }
        
        (vulnerable, robust)
    }
    
    /// Check if a word is likely a content word
    fn is_content_word(&self, word: &str) -> bool {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "has", "have", "had", "will", "would", "could", "should", "may",
            "might", "can", "do", "does", "did", "to", "of", "in", "on", "at",
            "by", "for", "with", "from", "up", "about", "into", "through", "during",
        ];
        
        !function_words.contains(&word.to_lowercase().as_str())
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_stability_score: 0.7,
            enable_word_removal: true,
            enable_positional_rearrangement: true,
            enable_synonym_substitution: true,
            enable_negation_tests: true,
            max_perturbations_per_type: 5,
            resolution_context: ResolutionContext {
                domain: Some("general".to_string()),
                culture: None,
                time_period: None,
                purpose: None,
                parameters: HashMap::new(),
                resolution_strategy: crate::turbulance::probabilistic::ResolutionStrategy::MaximumLikelihood,
            },
            validation_depth: ValidationDepth::Thorough,
        }
    }
}

/// Validate resolution quality through perturbation testing
pub async fn validate_resolution_quality(
    point: &TextPoint,
    resolution: &ResolutionResult,
    config: Option<ValidationConfig>,
) -> Result<ValidationResult> {
    let config = config.unwrap_or_default();
    let mut validator = PerturbationValidator::new(point.clone(), resolution.clone(), config.clone());
    validator.run_validation(&config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbulance::probabilistic::ResolutionStrategy;
    
    #[tokio::test]
    async fn test_validator_creation() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.8);
        let resolution = ResolutionResult::Certain(Value::String("test".to_string()));
        let config = ValidationConfig::default();
        
        let validator = PerturbationValidator::new(point, resolution, config);
        assert_eq!(validator.stability_threshold, 0.7);
    }
    
    #[tokio::test]
    async fn test_word_removal_tests() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.8);
        let resolution = ResolutionResult::Certain(Value::String("test".to_string()));
        let config = ValidationConfig::default();
        
        let mut validator = PerturbationValidator::new(point, resolution, config);
        let tests = validator.generate_word_removal_tests(10).unwrap();
        
        assert!(!tests.is_empty());
        assert!(tests.len() <= 4); // One test per word
    }
    
    #[tokio::test]
    async fn test_perturbation_validation() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.8);
        let resolution = ResolutionResult::Certain(Value::String("test".to_string()));
        let config = ValidationConfig::default();
        
        let mut validator = PerturbationValidator::new(point, resolution, config.clone());
        let result = validator.run_validation(&config).await.unwrap();
        
        assert!(result.stability_score >= 0.0);
        assert!(result.stability_score <= 1.0);
        assert!(!result.perturbation_results.is_empty());
    }
    
    #[tokio::test]
    async fn test_negation_consistency() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.8);
        let resolution = ResolutionResult::Certain(Value::String("test".to_string()));
        let config = ValidationConfig::default();
        
        let mut validator = PerturbationValidator::new(point, resolution, config);
        let tests = validator.generate_negation_tests(1).unwrap();
        
        assert_eq!(tests.len(), 1);
        assert!(tests[0].perturbed_point.content.contains("not"));
        assert_eq!(tests[0].expected_impact, ImpactLevel::Logical);
    }
    
    #[tokio::test]
    async fn test_validation_quality_function() {
        let point = TextPoint::new("The solution is optimal".to_string(), 0.8);
        let resolution = ResolutionResult::Certain(Value::String("test".to_string()));
        
        let result = validate_resolution_quality(&point, &resolution, None).await.unwrap();
        
        assert!(result.stability_score >= 0.0);
        assert!(result.stability_score <= 1.0);
        assert!(!result.recommendations.is_empty());
    }
} 