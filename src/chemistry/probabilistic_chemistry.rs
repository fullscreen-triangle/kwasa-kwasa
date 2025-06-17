//! Probabilistic Chemistry Module
//!
//! Implements fuzzy/deterministic hybrid logic for chemical analysis,
//! integrating with the Turbulance probabilistic framework.

use std::collections::HashMap;
use crate::chemistry::{Molecule, Atom, Bond, BondType};
use crate::turbulance::probabilistic::{TextPoint, ResolutionResult, ResolutionContext, ResolutionFunction};
use crate::turbulance::interpreter::Value;
use crate::error::TurbulanceError;
use serde::{Serialize, Deserialize};
use rand::prelude::*;
use rand_distr::{Normal, Beta, Gamma};

/// Probabilistic chemical engine that combines fuzzy and deterministic logic
#[derive(Debug, Clone)]
pub struct ProbabilisticChemicalEngine {
    /// Configuration for probabilistic analysis
    pub config: ProbabilisticConfig,
    /// Fuzzy rule base for chemical reasoning
    pub fuzzy_rules: Vec<FuzzyChemicalRule>,
    /// Deterministic chemical models
    pub deterministic_models: Vec<DeterministicModel>,
    /// Uncertainty quantification engine
    pub uncertainty_engine: UncertaintyEngine,
}

/// Configuration for probabilistic chemical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticConfig {
    /// Enable fuzzy logic
    pub enable_fuzzy: bool,
    /// Enable deterministic calculations
    pub enable_deterministic: bool,
    /// Uncertainty propagation method
    pub uncertainty_method: UncertaintyMethod,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Monte Carlo sample size
    pub monte_carlo_samples: usize,
    /// Bayesian prior strength
    pub prior_strength: f64,
}

/// Methods for uncertainty propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyMethod {
    /// Monte Carlo simulation
    MonteCarlo,
    /// Fuzzy arithmetic
    FuzzyArithmetic,
    /// Interval arithmetic
    IntervalArithmetic,
    /// Probabilistic programming
    ProbabilisticProgramming,
    /// Hybrid approach
    Hybrid,
}

/// Fuzzy chemical rule with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyChemicalRule {
    /// Rule identifier
    pub id: String,
    /// Antecedent conditions (IF part)
    pub antecedents: Vec<FuzzyCondition>,
    /// Consequent actions (THEN part)
    pub consequents: Vec<FuzzyConsequent>,
    /// Rule confidence/weight
    pub confidence: f64,
    /// Uncertainty in the rule itself
    pub rule_uncertainty: f64,
    /// Application domain
    pub domain: ChemicalDomain,
}

/// Fuzzy condition with probability distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyCondition {
    /// Property being evaluated
    pub property: String,
    /// Linguistic term (e.g., "high", "medium", "low")
    pub linguistic_term: String,
    /// Membership function parameters
    pub membership_params: MembershipParams,
    /// Confidence in this condition
    pub confidence: f64,
}

/// Parameters for membership functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MembershipParams {
    /// Triangular: (left, center, right)
    Triangular(f64, f64, f64),
    /// Trapezoidal: (left, left_top, right_top, right)
    Trapezoidal(f64, f64, f64, f64),
    /// Gaussian: (mean, std_dev)
    Gaussian(f64, f64),
    /// Sigmoid: (slope, center)
    Sigmoid(f64, f64),
}

/// Fuzzy consequent with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyConsequent {
    /// Predicted property
    pub property: String,
    /// Linguistic value
    pub linguistic_value: String,
    /// Confidence in prediction
    pub confidence: f64,
    /// Uncertainty bounds
    pub uncertainty_bounds: (f64, f64),
}

/// Chemical domains for specialized rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChemicalDomain {
    /// Drug-like properties
    DrugLikeness,
    /// Toxicity prediction
    Toxicity,
    /// Reactivity analysis
    Reactivity,
    /// Physical properties
    PhysicalProperties,
    /// Environmental fate
    Environmental,
    /// Catalysis
    Catalysis,
}

/// Deterministic chemical model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicModel {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Parameter uncertainties
    pub parameter_uncertainties: HashMap<String, f64>,
    /// Model accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
}

/// Types of deterministic models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear regression model
    LinearRegression,
    /// Non-linear regression
    NonLinearRegression,
    /// Quantum mechanical model
    QuantumMechanical,
    /// Force field model
    ForceField,
    /// Machine learning model
    MachineLearning(String),
    /// Empirical correlation
    Empirical,
}

/// Model accuracy metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Root mean square error
    pub rmse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Cross-validation score
    pub cv_score: f64,
}

/// Uncertainty quantification engine
#[derive(Debug, Clone)]
pub struct UncertaintyEngine {
    /// Random number generator
    rng: StdRng,
    /// Uncertainty propagation method
    method: UncertaintyMethod,
    /// Sample size for stochastic methods
    sample_size: usize,
}

/// Chemical property with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertainChemicalProperty {
    /// Property name
    pub name: String,
    /// Central value
    pub value: f64,
    /// Uncertainty (standard deviation)
    pub uncertainty: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Probability distribution type
    pub distribution: DistributionType,
    /// Sources of uncertainty
    pub uncertainty_sources: Vec<UncertaintySource>,
}

/// Types of probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Normal distribution
    Normal { mean: f64, std_dev: f64 },
    /// Log-normal distribution
    LogNormal { mu: f64, sigma: f64 },
    /// Beta distribution
    Beta { alpha: f64, beta: f64 },
    /// Gamma distribution
    Gamma { shape: f64, rate: f64 },
    /// Uniform distribution
    Uniform { min: f64, max: f64 },
    /// Custom empirical distribution
    Empirical { samples: Vec<f64> },
}

/// Sources of uncertainty in chemical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    /// Source type
    pub source_type: UncertaintySourceType,
    /// Contribution to total uncertainty (%)
    pub contribution: f64,
    /// Description
    pub description: String,
}

/// Types of uncertainty sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintySourceType {
    /// Experimental measurement uncertainty
    Experimental,
    /// Model uncertainty
    Model,
    /// Parameter uncertainty
    Parameter,
    /// Structural uncertainty
    Structural,
    /// Environmental variability
    Environmental,
    /// Aleatory uncertainty (natural variability)
    Aleatory,
    /// Epistemic uncertainty (knowledge gaps)
    Epistemic,
}

/// Results of probabilistic chemical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalUncertaintyAnalysis {
    /// Molecule identifier
    pub molecule_id: String,
    /// Uncertain properties
    pub uncertain_properties: Vec<UncertainChemicalProperty>,
    /// Fuzzy analysis results
    pub fuzzy_results: FuzzyAnalysisResults,
    /// Deterministic model results
    pub deterministic_results: Vec<DeterministicResult>,
    /// Hybrid consensus predictions
    pub consensus_predictions: Vec<ConsensusPrediction>,
    /// Sensitivity analysis
    pub sensitivity_analysis: SensitivityAnalysis,
    /// Reliability assessment
    pub reliability_assessment: ReliabilityAssessment,
}

/// Results from fuzzy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyAnalysisResults {
    /// Fuzzy property predictions
    pub property_predictions: Vec<FuzzyPropertyPrediction>,
    /// Rule activation strengths
    pub rule_activations: HashMap<String, f64>,
    /// Linguistic interpretations
    pub linguistic_interpretations: Vec<LinguisticInterpretation>,
}

/// Fuzzy property prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyPropertyPrediction {
    /// Property name
    pub property: String,
    /// Fuzzy value
    pub fuzzy_value: FuzzyNumber,
    /// Defuzzified crisp value
    pub crisp_value: f64,
    /// Confidence in prediction
    pub confidence: f64,
}

/// Fuzzy number representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyNumber {
    /// Membership function points (value, membership)
    pub membership_points: Vec<(f64, f64)>,
    /// Alpha cuts for interval representation
    pub alpha_cuts: Vec<AlphaCut>,
    /// Support (non-zero membership region)
    pub support: (f64, f64),
    /// Core (membership = 1 region)
    pub core: (f64, f64),
}

/// Alpha cut of fuzzy number
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaCut {
    /// Alpha level (0 to 1)
    pub alpha: f64,
    /// Lower bound at this alpha level
    pub lower: f64,
    /// Upper bound at this alpha level
    pub upper: f64,
}

/// Linguistic interpretation of results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticInterpretation {
    /// Property being described
    pub property: String,
    /// Natural language description
    pub description: String,
    /// Confidence in interpretation
    pub confidence: f64,
    /// Hedges and modifiers
    pub modifiers: Vec<String>,
}

/// Result from deterministic model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicResult {
    /// Model name
    pub model_name: String,
    /// Predicted value
    pub predicted_value: f64,
    /// Model uncertainty
    pub model_uncertainty: f64,
    /// Prediction interval
    pub prediction_interval: (f64, f64),
    /// Model confidence
    pub model_confidence: f64,
}

/// Consensus prediction combining multiple approaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusPrediction {
    /// Property name
    pub property: String,
    /// Consensus value
    pub consensus_value: f64,
    /// Consensus uncertainty
    pub consensus_uncertainty: f64,
    /// Weight of each contributing method
    pub method_weights: HashMap<String, f64>,
    /// Disagreement between methods
    pub method_disagreement: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// Parameter sensitivities
    pub parameter_sensitivities: Vec<ParameterSensitivity>,
    /// Most influential parameters
    pub influential_parameters: Vec<String>,
    /// Interaction effects
    pub interaction_effects: Vec<InteractionEffect>,
}

/// Sensitivity of output to parameter changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSensitivity {
    /// Parameter name
    pub parameter: String,
    /// Sensitivity coefficient
    pub sensitivity: f64,
    /// Normalized sensitivity
    pub normalized_sensitivity: f64,
    /// Rank by importance
    pub rank: usize,
}

/// Interaction effect between parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEffect {
    /// First parameter
    pub parameter1: String,
    /// Second parameter
    pub parameter2: String,
    /// Interaction strength
    pub interaction_strength: f64,
    /// Synergistic or antagonistic
    pub effect_type: InteractionType,
}

/// Types of parameter interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Parameters enhance each other's effects
    Synergistic,
    /// Parameters reduce each other's effects
    Antagonistic,
    /// No significant interaction
    Independent,
}

/// Reliability assessment of predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityAssessment {
    /// Overall reliability score (0-1)
    pub overall_reliability: f64,
    /// Data quality score
    pub data_quality: f64,
    /// Model applicability score
    pub model_applicability: f64,
    /// Prediction uncertainty score
    pub uncertainty_score: f64,
    /// Validation score
    pub validation_score: f64,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            enable_fuzzy: true,
            enable_deterministic: true,
            uncertainty_method: UncertaintyMethod::Hybrid,
            confidence_threshold: 0.8,
            monte_carlo_samples: 10000,
            prior_strength: 0.1,
        }
    }
}

impl ProbabilisticChemicalEngine {
    /// Create new probabilistic chemical engine
    pub fn new(config: ProbabilisticConfig) -> Self {
        Self {
            config,
            fuzzy_rules: Vec::new(),
            deterministic_models: Vec::new(),
            uncertainty_engine: UncertaintyEngine::new(UncertaintyMethod::Hybrid, 10000),
        }
    }

    /// Analyze molecule with probabilistic methods
    pub fn analyze_with_uncertainty(&self, molecule: &Molecule) -> Result<ChemicalUncertaintyAnalysis, TurbulanceError> {
        let molecule_id = molecule.id().to_string();

        // Generate uncertain properties
        let uncertain_properties = self.calculate_uncertain_properties(molecule)?;

        // Fuzzy analysis
        let fuzzy_results = if self.config.enable_fuzzy {
            self.perform_fuzzy_analysis(molecule)?
        } else {
            FuzzyAnalysisResults::default()
        };

        // Deterministic analysis
        let deterministic_results = if self.config.enable_deterministic {
            self.perform_deterministic_analysis(molecule)?
        } else {
            Vec::new()
        };

        // Consensus predictions
        let consensus_predictions = self.build_consensus(&fuzzy_results, &deterministic_results)?;

        // Sensitivity analysis
        let sensitivity_analysis = self.perform_sensitivity_analysis(molecule)?;

        // Reliability assessment
        let reliability_assessment = self.assess_reliability(molecule, &uncertain_properties)?;

        Ok(ChemicalUncertaintyAnalysis {
            molecule_id,
            uncertain_properties,
            fuzzy_results,
            deterministic_results,
            consensus_predictions,
            sensitivity_analysis,
            reliability_assessment,
        })
    }

    /// Calculate properties with uncertainty quantification
    fn calculate_uncertain_properties(&self, molecule: &Molecule) -> Result<Vec<UncertainChemicalProperty>, TurbulanceError> {
        let mut properties = Vec::new();

        // Molecular weight with uncertainty
        let mw = molecule.molecular_weight();
        let mw_uncertainty = mw * 0.001; // 0.1% relative uncertainty
        properties.push(UncertainChemicalProperty {
            name: "molecular_weight".to_string(),
            value: mw,
            uncertainty: mw_uncertainty,
            confidence_interval: (mw - 2.0 * mw_uncertainty, mw + 2.0 * mw_uncertainty),
            distribution: DistributionType::Normal { mean: mw, std_dev: mw_uncertainty },
            uncertainty_sources: vec![
                UncertaintySource {
                    source_type: UncertaintySourceType::Experimental,
                    contribution: 80.0,
                    description: "Atomic mass uncertainties".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Model,
                    contribution: 20.0,
                    description: "Calculation method approximations".to_string(),
                },
            ],
        });

        // Add more uncertain properties based on molecular structure
        // LogP prediction with uncertainty
        let logp = self.predict_logp_with_uncertainty(molecule)?;
        properties.push(logp);

        // Solubility prediction with uncertainty
        let solubility = self.predict_solubility_with_uncertainty(molecule)?;
        properties.push(solubility);

        Ok(properties)
    }

    /// Predict LogP with uncertainty
    fn predict_logp_with_uncertainty(&self, molecule: &Molecule) -> Result<UncertainChemicalProperty, TurbulanceError> {
        // Simplified LogP prediction with uncertainty
        let atom_count = molecule.atoms().len() as f64;
        let base_logp = (atom_count * 0.3 - 2.0).max(-5.0).min(8.0);
        let uncertainty = 0.5 + atom_count * 0.05; // Uncertainty increases with size

        Ok(UncertainChemicalProperty {
            name: "logP".to_string(),
            value: base_logp,
            uncertainty,
            confidence_interval: (base_logp - 1.96 * uncertainty, base_logp + 1.96 * uncertainty),
            distribution: DistributionType::Normal { mean: base_logp, std_dev: uncertainty },
            uncertainty_sources: vec![
                UncertaintySource {
                    source_type: UncertaintySourceType::Model,
                    contribution: 60.0,
                    description: "Prediction model uncertainty".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Parameter,
                    contribution: 25.0,
                    description: "Fragment contribution uncertainties".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Structural,
                    contribution: 15.0,
                    description: "Conformational flexibility".to_string(),
                },
            ],
        })
    }

    /// Predict solubility with uncertainty
    fn predict_solubility_with_uncertainty(&self, molecule: &Molecule) -> Result<UncertainChemicalProperty, TurbulanceError> {
        // Simplified solubility prediction
        let heavy_atoms = molecule.atoms().iter().filter(|a| a.symbol != "H").count() as f64;
        let base_solubility = (-0.5 * heavy_atoms + 5.0).max(-8.0).min(2.0);
        let uncertainty = 0.8 + heavy_atoms * 0.1;

        Ok(UncertainChemicalProperty {
            name: "log_solubility".to_string(),
            value: base_solubility,
            uncertainty,
            confidence_interval: (base_solubility - 1.96 * uncertainty, base_solubility + 1.96 * uncertainty),
            distribution: DistributionType::Normal { mean: base_solubility, std_dev: uncertainty },
            uncertainty_sources: vec![
                UncertaintySource {
                    source_type: UncertaintySourceType::Model,
                    contribution: 50.0,
                    description: "Solubility model limitations".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Environmental,
                    contribution: 30.0,
                    description: "Temperature and pH effects".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Epistemic,
                    contribution: 20.0,
                    description: "Unknown interaction effects".to_string(),
                },
            ],
        })
    }

    /// Perform fuzzy analysis
    fn perform_fuzzy_analysis(&self, molecule: &Molecule) -> Result<FuzzyAnalysisResults, TurbulanceError> {
        let mut property_predictions = Vec::new();
        let mut rule_activations = HashMap::new();
        let mut linguistic_interpretations = Vec::new();

        // Example fuzzy analysis for drug-likeness
        let drug_likeness = self.evaluate_drug_likeness_fuzzy(molecule)?;
        property_predictions.push(drug_likeness);

        // Example linguistic interpretation
        linguistic_interpretations.push(LinguisticInterpretation {
            property: "overall_assessment".to_string(),
            description: "The molecule shows moderate drug-like characteristics with some uncertainty in bioavailability".to_string(),
            confidence: 0.75,
            modifiers: vec!["moderate".to_string(), "some".to_string()],
        });

        Ok(FuzzyAnalysisResults {
            property_predictions,
            rule_activations,
            linguistic_interpretations,
        })
    }

    /// Evaluate drug-likeness using fuzzy logic
    fn evaluate_drug_likeness_fuzzy(&self, molecule: &Molecule) -> Result<FuzzyPropertyPrediction, TurbulanceError> {
        // Simplified fuzzy drug-likeness evaluation
        let mw = molecule.molecular_weight();
        let atom_count = molecule.atoms().len() as f64;

        // Create fuzzy membership for molecular weight
        let mw_membership = if mw < 300.0 {
            (mw / 300.0).min(1.0)
        } else if mw <= 500.0 {
            1.0
        } else if mw <= 800.0 {
            (800.0 - mw) / 300.0
        } else {
            0.0
        };

        // Create fuzzy number for drug-likeness
        let fuzzy_value = FuzzyNumber {
            membership_points: vec![
                (0.0, 0.0),
                (0.3, 0.2),
                (0.7, 1.0),
                (0.9, 0.5),
                (1.0, 0.0),
            ],
            alpha_cuts: vec![
                AlphaCut { alpha: 0.0, lower: 0.0, upper: 1.0 },
                AlphaCut { alpha: 0.5, lower: 0.5, upper: 0.85 },
                AlphaCut { alpha: 1.0, lower: 0.7, upper: 0.7 },
            ],
            support: (0.0, 1.0),
            core: (0.7, 0.7),
        };

        Ok(FuzzyPropertyPrediction {
            property: "drug_likeness".to_string(),
            fuzzy_value,
            crisp_value: 0.7,
            confidence: 0.8,
        })
    }

    /// Perform deterministic analysis
    fn perform_deterministic_analysis(&self, molecule: &Molecule) -> Result<Vec<DeterministicResult>, TurbulanceError> {
        let mut results = Vec::new();

        // Example deterministic calculation
        results.push(DeterministicResult {
            model_name: "Ghose-Crippen LogP".to_string(),
            predicted_value: 2.3,
            model_uncertainty: 0.4,
            prediction_interval: (1.5, 3.1),
            model_confidence: 0.85,
        });

        Ok(results)
    }

    /// Build consensus predictions
    fn build_consensus(&self, fuzzy: &FuzzyAnalysisResults, deterministic: &[DeterministicResult]) -> Result<Vec<ConsensusPrediction>, TurbulanceError> {
        let mut consensus = Vec::new();

        // Example consensus building
        consensus.push(ConsensusPrediction {
            property: "drug_likeness".to_string(),
            consensus_value: 0.73,
            consensus_uncertainty: 0.12,
            method_weights: [
                ("fuzzy_analysis".to_string(), 0.6),
                ("deterministic_model".to_string(), 0.4),
            ].into(),
            method_disagreement: 0.05,
        });

        Ok(consensus)
    }

    /// Perform sensitivity analysis
    fn perform_sensitivity_analysis(&self, molecule: &Molecule) -> Result<SensitivityAnalysis, TurbulanceError> {
        // Simplified sensitivity analysis
        Ok(SensitivityAnalysis {
            parameter_sensitivities: vec![
                ParameterSensitivity {
                    parameter: "molecular_weight".to_string(),
                    sensitivity: 0.8,
                    normalized_sensitivity: 0.9,
                    rank: 1,
                },
                ParameterSensitivity {
                    parameter: "lipophilicity".to_string(),
                    sensitivity: 0.6,
                    normalized_sensitivity: 0.7,
                    rank: 2,
                },
            ],
            influential_parameters: vec!["molecular_weight".to_string(), "lipophilicity".to_string()],
            interaction_effects: vec![],
        })
    }

    /// Assess reliability of predictions
    fn assess_reliability(&self, molecule: &Molecule, properties: &[UncertainChemicalProperty]) -> Result<ReliabilityAssessment, TurbulanceError> {
        Ok(ReliabilityAssessment {
            overall_reliability: 0.78,
            data_quality: 0.85,
            model_applicability: 0.75,
            uncertainty_score: 0.80,
            validation_score: 0.72,
        })
    }
}

impl UncertaintyEngine {
    /// Create new uncertainty engine
    pub fn new(method: UncertaintyMethod, sample_size: usize) -> Self {
        Self {
            rng: StdRng::seed_from_u64(42), // Fixed seed for reproducibility
            method,
            sample_size,
        }
    }

    /// Propagate uncertainty through calculations
    pub fn propagate_uncertainty(&mut self, inputs: &[UncertainChemicalProperty], calculation: fn(&[f64]) -> f64) -> Result<UncertainChemicalProperty, TurbulanceError> {
        match self.method {
            UncertaintyMethod::MonteCarlo => self.monte_carlo_propagation(inputs, calculation),
            UncertaintyMethod::FuzzyArithmetic => self.fuzzy_propagation(inputs, calculation),
            UncertaintyMethod::IntervalArithmetic => self.interval_propagation(inputs, calculation),
            UncertaintyMethod::Hybrid => self.hybrid_propagation(inputs, calculation),
            _ => self.monte_carlo_propagation(inputs, calculation), // Default
        }
    }

    /// Monte Carlo uncertainty propagation
    fn monte_carlo_propagation(&mut self, inputs: &[UncertainChemicalProperty], calculation: fn(&[f64]) -> f64) -> Result<UncertainChemicalProperty, TurbulanceError> {
        let mut samples = Vec::new();

        for _ in 0..self.sample_size {
            let mut input_sample = Vec::new();
            
            for input in inputs {
                let sample_value = match &input.distribution {
                    DistributionType::Normal { mean, std_dev } => {
                        let normal = Normal::new(*mean, *std_dev).unwrap();
                        normal.sample(&mut self.rng)
                    },
                    DistributionType::Uniform { min, max } => {
                        self.rng.gen_range(*min..*max)
                    },
                    _ => input.value, // Fallback to mean value
                };
                input_sample.push(sample_value);
            }

            let result = calculation(&input_sample);
            samples.push(result);
        }

        // Calculate statistics
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
        let std_dev = variance.sqrt();

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ci_lower = samples[(0.025 * samples.len() as f64) as usize];
        let ci_upper = samples[(0.975 * samples.len() as f64) as usize];

        Ok(UncertainChemicalProperty {
            name: "calculated_property".to_string(),
            value: mean,
            uncertainty: std_dev,
            confidence_interval: (ci_lower, ci_upper),
            distribution: DistributionType::Empirical { samples },
            uncertainty_sources: vec![
                UncertaintySource {
                    source_type: UncertaintySourceType::Parameter,
                    contribution: 100.0,
                    description: "Propagated from input uncertainties".to_string(),
                },
            ],
        })
    }

    /// Fuzzy arithmetic uncertainty propagation
    fn fuzzy_propagation(&mut self, inputs: &[UncertainChemicalProperty], _calculation: fn(&[f64]) -> f64) -> Result<UncertainChemicalProperty, TurbulanceError> {
        // Simplified fuzzy arithmetic implementation
        let mean_value = inputs.iter().map(|p| p.value).sum::<f64>() / inputs.len() as f64;
        let combined_uncertainty = inputs.iter().map(|p| p.uncertainty.powi(2)).sum::<f64>().sqrt();

        Ok(UncertainChemicalProperty {
            name: "fuzzy_calculated".to_string(),
            value: mean_value,
            uncertainty: combined_uncertainty,
            confidence_interval: (mean_value - combined_uncertainty, mean_value + combined_uncertainty),
            distribution: DistributionType::Normal { mean: mean_value, std_dev: combined_uncertainty },
            uncertainty_sources: vec![],
        })
    }

    /// Interval arithmetic uncertainty propagation
    fn interval_propagation(&mut self, inputs: &[UncertainChemicalProperty], _calculation: fn(&[f64]) -> f64) -> Result<UncertainChemicalProperty, TurbulanceError> {
        // Simplified interval arithmetic
        let lower_bounds: Vec<f64> = inputs.iter().map(|p| p.confidence_interval.0).collect();
        let upper_bounds: Vec<f64> = inputs.iter().map(|p| p.confidence_interval.1).collect();
        
        let min_result = lower_bounds.iter().sum::<f64>();
        let max_result = upper_bounds.iter().sum::<f64>();
        let mean_result = (min_result + max_result) / 2.0;
        let uncertainty = (max_result - min_result) / 4.0; // Approximate std dev

        Ok(UncertainChemicalProperty {
            name: "interval_calculated".to_string(),
            value: mean_result,
            uncertainty,
            confidence_interval: (min_result, max_result),
            distribution: DistributionType::Uniform { min: min_result, max: max_result },
            uncertainty_sources: vec![],
        })
    }

    /// Hybrid uncertainty propagation
    fn hybrid_propagation(&mut self, inputs: &[UncertainChemicalProperty], calculation: fn(&[f64]) -> f64) -> Result<UncertainChemicalProperty, TurbulanceError> {
        // Combine Monte Carlo and fuzzy approaches
        let mc_result = self.monte_carlo_propagation(inputs, calculation)?;
        let fuzzy_result = self.fuzzy_propagation(inputs, calculation)?;

        // Weighted combination
        let combined_value = 0.7 * mc_result.value + 0.3 * fuzzy_result.value;
        let combined_uncertainty = (0.7 * mc_result.uncertainty.powi(2) + 0.3 * fuzzy_result.uncertainty.powi(2)).sqrt();

        Ok(UncertainChemicalProperty {
            name: "hybrid_calculated".to_string(),
            value: combined_value,
            uncertainty: combined_uncertainty,
            confidence_interval: (
                combined_value - 1.96 * combined_uncertainty,
                combined_value + 1.96 * combined_uncertainty
            ),
            distribution: DistributionType::Normal { mean: combined_value, std_dev: combined_uncertainty },
            uncertainty_sources: vec![
                UncertaintySource {
                    source_type: UncertaintySourceType::Model,
                    contribution: 70.0,
                    description: "Monte Carlo contribution".to_string(),
                },
                UncertaintySource {
                    source_type: UncertaintySourceType::Parameter,
                    contribution: 30.0,
                    description: "Fuzzy arithmetic contribution".to_string(),
                },
            ],
        })
    }
}

// Default implementations

impl Default for FuzzyAnalysisResults {
    fn default() -> Self {
        Self {
            property_predictions: Vec::new(),
            rule_activations: HashMap::new(),
            linguistic_interpretations: Vec::new(),
        }
    }
}

/// Probabilistic resolution function for chemical properties
pub struct ChemicalProbabilisticResolution {
    engine: ProbabilisticChemicalEngine,
}

impl ChemicalProbabilisticResolution {
    pub fn new() -> Self {
        Self {
            engine: ProbabilisticChemicalEngine::new(ProbabilisticConfig::default()),
        }
    }
}

impl ResolutionFunction for ChemicalProbabilisticResolution {
    fn name(&self) -> &str {
        "chemical_probabilistic_resolution"
    }

    fn resolve(&self, point: &TextPoint, _context: &ResolutionContext) -> Result<ResolutionResult, TurbulanceError> {
        // Parse chemical information from text point
        let content = &point.content;
        
        if content.contains("SMILES") || content.contains("molecule") {
            // Simulate probabilistic chemical analysis
            let uncertainty = 0.15 + (1.0 - point.confidence) * 0.2;
            
            Ok(ResolutionResult::Uncertain {
                possibilities: vec![
                    (Value::Float(0.75), 0.4),
                    (Value::Float(0.65), 0.35),
                    (Value::Float(0.85), 0.25),
                ],
                confidence_interval: (0.60, 0.90),
                aggregated_confidence: point.confidence * (1.0 - uncertainty),
            })
        } else {
            Ok(ResolutionResult::Certain(Value::Float(0.5)))
        }
    }

    fn uncertainty_factor(&self) -> f64 {
        0.20 // Chemical predictions have inherent uncertainty
    }

    fn can_handle(&self, point: &TextPoint) -> bool {
        point.content.contains("chemical") ||
        point.content.contains("molecule") ||
        point.content.contains("property") ||
        point.content.contains("SMILES") ||
        point.content.contains("drug")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probabilistic_chemical_engine() {
        let config = ProbabilisticConfig::default();
        let engine = ProbabilisticChemicalEngine::new(config);
        assert!(engine.config.enable_fuzzy);
        assert!(engine.config.enable_deterministic);
    }

    #[test]
    fn test_uncertainty_engine() {
        let mut engine = UncertaintyEngine::new(UncertaintyMethod::MonteCarlo, 1000);
        assert_eq!(engine.sample_size, 1000);
    }

    #[test]
    fn test_chemical_probabilistic_resolution() {
        let resolver = ChemicalProbabilisticResolution::new();
        let point = TextPoint::new("SMILES: CCO".to_string(), 0.9);
        assert!(resolver.can_handle(&point));
    }
} 