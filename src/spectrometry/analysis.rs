//! Spectrometry analysis module
//!
//! This module provides high-level analysis functionality across all spectrometry techniques.

use std::collections::HashMap;
use super::{mass_spec::MassSpecAnalysisResult, nmr::NMRAnalysisResult, ir::IRAnalysisResult, uv_vis::UVVisAnalysisResult};

/// Comprehensive spectrometry analyzer
#[derive(Debug, Clone)]
pub struct SpectrometryAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
}

/// Configuration for spectrometry analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Enable correlation analysis
    pub enable_correlation: bool,
    /// Confidence threshold for identifications
    pub confidence_threshold: f64,
    /// Maximum results to return
    pub max_results: usize,
}

/// Combined analysis result
#[derive(Debug, Clone)]
pub struct CombinedAnalysisResult {
    /// Sample ID
    pub sample_id: String,
    /// Individual technique results
    pub technique_results: HashMap<String, TechniqueResult>,
    /// Cross-technique correlations
    pub correlations: Vec<Correlation>,
    /// Compound identifications
    pub compound_identifications: Vec<CompoundIdentification>,
}

/// Result from individual technique
#[derive(Debug, Clone)]
pub enum TechniqueResult {
    /// Mass spectrometry result
    MassSpec(MassSpecAnalysisResult),
    /// NMR result
    NMR(NMRAnalysisResult),
    /// IR result
    IR(IRAnalysisResult),
    /// UV-Vis result
    UVVis(UVVisAnalysisResult),
}

/// Correlation between techniques
#[derive(Debug, Clone)]
pub struct Correlation {
    /// Technique pair
    pub techniques: (String, String),
    /// Correlation type
    pub correlation_type: CorrelationType,
    /// Confidence score
    pub confidence: f64,
    /// Description
    pub description: String,
}

/// Types of correlations
#[derive(Debug, Clone)]
pub enum CorrelationType {
    /// Molecular weight confirmation
    MolecularWeight,
    /// Functional group confirmation
    FunctionalGroup,
    /// Structural confirmation
    Structural,
    /// Elemental composition
    ElementalComposition,
}

/// Compound identification
#[derive(Debug, Clone)]
pub struct CompoundIdentification {
    /// Compound name
    pub name: String,
    /// CAS number
    pub cas_number: Option<String>,
    /// Molecular formula
    pub molecular_formula: String,
    /// Overall confidence
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<Evidence>,
}

/// Evidence for compound identification
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Source technique
    pub technique: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Confidence
    pub confidence: f64,
    /// Description
    pub description: String,
}

/// Types of evidence
#[derive(Debug, Clone)]
pub enum EvidenceType {
    /// Molecular ion peak
    MolecularIon,
    /// Characteristic fragment
    Fragment,
    /// Chemical shift
    ChemicalShift,
    /// Vibrational frequency
    Vibrational,
    /// Electronic transition
    Electronic,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_correlation: true,
            confidence_threshold: 0.7,
            max_results: 10,
        }
    }
}

impl SpectrometryAnalyzer {
    /// Create new analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(AnalysisConfig::default())
    }

    /// Perform combined analysis
    pub fn analyze_combined(&self, results: HashMap<String, TechniqueResult>) -> CombinedAnalysisResult {
        let correlations = if self.config.enable_correlation {
            self.find_correlations(&results)
        } else {
            Vec::new()
        };

        let compound_identifications = self.identify_compounds(&results, &correlations);

        CombinedAnalysisResult {
            sample_id: "sample".to_string(),
            technique_results: results,
            correlations,
            compound_identifications,
        }
    }

    /// Find correlations between techniques
    fn find_correlations(&self, results: &HashMap<String, TechniqueResult>) -> Vec<Correlation> {
        let mut correlations = Vec::new();

        // Check for molecular weight correlations between MS and other techniques
        if let (Some(ms_result), Some(nmr_result)) = (
            self.get_mass_spec_result(results),
            self.get_nmr_result(results)
        ) {
            if let Some(mol_ion) = &ms_result.molecular_ion {
                correlations.push(Correlation {
                    techniques: ("MS".to_string(), "NMR".to_string()),
                    correlation_type: CorrelationType::MolecularWeight,
                    confidence: 0.8,
                    description: format!("Molecular weight: {:.2}", mol_ion.mz),
                });
            }
        }

        // Check for functional group correlations between IR and NMR
        if let (Some(ir_result), Some(nmr_result)) = (
            self.get_ir_result(results),
            self.get_nmr_result(results)
        ) {
            for fg in &ir_result.functional_groups {
                if fg.group_name.contains("O-H") && !nmr_result.chemical_shifts.is_empty() {
                    correlations.push(Correlation {
                        techniques: ("IR".to_string(), "NMR".to_string()),
                        correlation_type: CorrelationType::FunctionalGroup,
                        confidence: fg.confidence,
                        description: "O-H group confirmed by both techniques".to_string(),
                    });
                }
            }
        }

        correlations
    }

    /// Identify compounds
    fn identify_compounds(&self, results: &HashMap<String, TechniqueResult>, correlations: &[Correlation]) -> Vec<CompoundIdentification> {
        let mut identifications = Vec::new();

        // Simple compound identification based on available data
        if let Some(ms_result) = self.get_mass_spec_result(results) {
            if let Some(mol_ion) = &ms_result.molecular_ion {
                let evidence = vec![Evidence {
                    technique: "MS".to_string(),
                    evidence_type: EvidenceType::MolecularIon,
                    confidence: 0.8,
                    description: format!("Molecular ion at m/z {:.2}", mol_ion.mz),
                }];

                identifications.push(CompoundIdentification {
                    name: "Unknown compound".to_string(),
                    cas_number: None,
                    molecular_formula: format!("M+{:.0}", mol_ion.mz),
                    confidence: 0.6,
                    evidence,
                });
            }
        }

        identifications
    }

    /// Get mass spec result
    fn get_mass_spec_result(&self, results: &HashMap<String, TechniqueResult>) -> Option<&MassSpecAnalysisResult> {
        results.values().find_map(|r| match r {
            TechniqueResult::MassSpec(ms) => Some(ms),
            _ => None,
        })
    }

    /// Get NMR result
    fn get_nmr_result(&self, results: &HashMap<String, TechniqueResult>) -> Option<&NMRAnalysisResult> {
        results.values().find_map(|r| match r {
            TechniqueResult::NMR(nmr) => Some(nmr),
            _ => None,
        })
    }

    /// Get IR result
    fn get_ir_result(&self, results: &HashMap<String, TechniqueResult>) -> Option<&IRAnalysisResult> {
        results.values().find_map(|r| match r {
            TechniqueResult::IR(ir) => Some(ir),
            _ => None,
        })
    }

    /// Get UV-Vis result
    fn get_uv_vis_result(&self, results: &HashMap<String, TechniqueResult>) -> Option<&UVVisAnalysisResult> {
        results.values().find_map(|r| match r {
            TechniqueResult::UVVis(uv) => Some(uv),
            _ => None,
        })
    }
} 