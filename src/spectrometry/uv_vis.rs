//! UV-Vis spectroscopy analysis module
//!
//! This module provides functionality for ultraviolet-visible absorption spectroscopy analysis.

use std::collections::HashMap;

/// UV-Vis analyzer
#[derive(Debug, Clone)]
pub struct UVVisAnalyzer {
    /// Analysis configuration
    config: UVVisConfig,
}

/// Configuration for UV-Vis analysis
#[derive(Debug, Clone)]
pub struct UVVisConfig {
    /// Minimum absorbance threshold
    pub min_absorbance: f64,
    /// Peak detection sensitivity
    pub peak_sensitivity: f64,
    /// Baseline correction enabled
    pub baseline_correction: bool,
}

/// UV-Vis analysis result
#[derive(Debug, Clone)]
pub struct UVVisAnalysisResult {
    /// Spectrum ID
    pub spectrum_id: String,
    /// Electronic transitions
    pub transitions: Vec<ElectronicTransition>,
    /// Chromophore identifications
    pub chromophores: Vec<ChromophoreID>,
    /// Extinction coefficients
    pub extinction_coefficients: Vec<ExtinctionCoefficient>,
    /// Band gap analysis
    pub band_gap: Option<BandGap>,
}

/// Electronic transition information
#[derive(Debug, Clone)]
pub struct ElectronicTransition {
    /// Wavelength (nm)
    pub wavelength: f64,
    /// Absorbance
    pub absorbance: f64,
    /// Molar absorptivity (if known)
    pub molar_absorptivity: Option<f64>,
    /// Transition type
    pub transition_type: TransitionType,
    /// Assignment
    pub assignment: Option<String>,
}

/// Types of electronic transitions
#[derive(Debug, Clone)]
pub enum TransitionType {
    /// π → π* transition
    PiToPiStar,
    /// n → π* transition
    NToPiStar,
    /// n → σ* transition
    NToSigmaStar,
    /// σ → σ* transition
    SigmaToSigmaStar,
    /// Charge transfer
    ChargeTransfer,
    /// d-d transition
    DDTransition,
    /// Unknown
    Unknown,
}

/// Chromophore identification
#[derive(Debug, Clone)]
pub struct ChromophoreID {
    /// Chromophore name
    pub chromophore_name: String,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Expected wavelength range (nm)
    pub wavelength_range: (f64, f64),
    /// Observed transitions
    pub observed_transitions: Vec<f64>,
}

/// Extinction coefficient data
#[derive(Debug, Clone)]
pub struct ExtinctionCoefficient {
    /// Wavelength (nm)
    pub wavelength: f64,
    /// Extinction coefficient (M⁻¹cm⁻¹)
    pub coefficient: f64,
    /// Uncertainty
    pub uncertainty: f64,
}

/// Band gap information
#[derive(Debug, Clone)]
pub struct BandGap {
    /// Band gap energy (eV)
    pub energy: f64,
    /// Band gap type
    pub gap_type: BandGapType,
    /// Tauc plot analysis
    pub tauc_analysis: Option<TaucAnalysis>,
}

/// Band gap types
#[derive(Debug, Clone)]
pub enum BandGapType {
    /// Direct band gap
    Direct,
    /// Indirect band gap
    Indirect,
    /// Unknown
    Unknown,
}

/// Tauc plot analysis
#[derive(Debug, Clone)]
pub struct TaucAnalysis {
    /// Linear fit slope
    pub slope: f64,
    /// Linear fit intercept
    pub intercept: f64,
    /// R-squared value
    pub r_squared: f64,
    /// Extracted band gap (eV)
    pub extracted_band_gap: f64,
}

impl Default for UVVisConfig {
    fn default() -> Self {
        Self {
            min_absorbance: 0.01,
            peak_sensitivity: 0.05,
            baseline_correction: true,
        }
    }
}

impl UVVisAnalyzer {
    /// Create a new UV-Vis analyzer
    pub fn new(config: UVVisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(UVVisConfig::default())
    }

    /// Analyze UV-Vis spectrum
    pub fn analyze(&self, spectrum_data: &UVVisSpectrumData) -> UVVisAnalysisResult {
        // Apply baseline correction if enabled
        let corrected_data = if self.config.baseline_correction {
            self.apply_baseline_correction(spectrum_data)
        } else {
            spectrum_data.clone()
        };
        
        // Find electronic transitions
        let transitions = self.find_electronic_transitions(&corrected_data);
        
        // Identify chromophores
        let chromophores = self.identify_chromophores(&transitions);
        
        // Calculate extinction coefficients
        let extinction_coefficients = self.calculate_extinction_coefficients(&transitions);
        
        // Analyze band gap
        let band_gap = self.analyze_band_gap(&corrected_data);
        
        UVVisAnalysisResult {
            spectrum_id: spectrum_data.id.clone(),
            transitions,
            chromophores,
            extinction_coefficients,
            band_gap,
        }
    }

    /// Apply baseline correction
    fn apply_baseline_correction(&self, data: &UVVisSpectrumData) -> UVVisSpectrumData {
        let mut corrected = data.clone();
        
        if !data.absorbance_data.is_empty() {
            // Simple linear baseline correction
            let min_absorbance = data.absorbance_data.iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            
            for absorbance in &mut corrected.absorbance_data {
                *absorbance -= min_absorbance;
            }
        }
        
        corrected
    }

    /// Find electronic transitions
    fn find_electronic_transitions(&self, data: &UVVisSpectrumData) -> Vec<ElectronicTransition> {
        let mut transitions = Vec::new();
        let mut i = 1;

        while i < data.absorbance_data.len() - 1 {
            let prev_abs = data.absorbance_data[i - 1];
            let curr_abs = data.absorbance_data[i];
            let next_abs = data.absorbance_data[i + 1];

            // Peak detection
            if curr_abs > prev_abs && curr_abs > next_abs && 
               curr_abs > self.config.min_absorbance {
                
                let wavelength = data.wavelength_data[i];
                let transition_type = self.classify_transition_type(wavelength);
                
                transitions.push(ElectronicTransition {
                    wavelength,
                    absorbance: curr_abs,
                    molar_absorptivity: None,
                    transition_type,
                    assignment: None,
                });
            }
            i += 1;
        }

        // Sort by intensity (absorbance)
        transitions.sort_by(|a, b| b.absorbance.partial_cmp(&a.absorbance).unwrap());
        transitions
    }

    /// Classify transition type based on wavelength
    fn classify_transition_type(&self, wavelength: f64) -> TransitionType {
        match wavelength {
            200.0..=280.0 => TransitionType::PiToPiStar,    // Aromatic π → π*
            280.0..=320.0 => TransitionType::NToPiStar,     // n → π*
            320.0..=400.0 => TransitionType::ChargeTransfer, // Extended conjugation
            400.0..=700.0 => TransitionType::DDTransition,   // Transition metals
            _ => TransitionType::Unknown,
        }
    }

    /// Identify chromophores
    fn identify_chromophores(&self, transitions: &[ElectronicTransition]) -> Vec<ChromophoreID> {
        let mut chromophores = Vec::new();
        
        // Check for common chromophores
        if let Some(benzene) = self.check_benzene_chromophore(transitions) {
            chromophores.push(benzene);
        }
        
        if let Some(carbonyl) = self.check_carbonyl_chromophore(transitions) {
            chromophores.push(carbonyl);
        }
        
        if let Some(conjugated) = self.check_conjugated_system(transitions) {
            chromophores.push(conjugated);
        }
        
        chromophores
    }

    /// Check for benzene chromophore
    fn check_benzene_chromophore(&self, transitions: &[ElectronicTransition]) -> Option<ChromophoreID> {
        let benzene_transitions: Vec<_> = transitions.iter()
            .filter(|t| t.wavelength >= 250.0 && t.wavelength <= 270.0)
            .collect();
        
        if !benzene_transitions.is_empty() {
            let confidence = (benzene_transitions.len() as f64 * 0.3).min(1.0);
            let observed_transitions: Vec<f64> = benzene_transitions.iter()
                .map(|t| t.wavelength).collect();
            
            Some(ChromophoreID {
                chromophore_name: "Benzene".to_string(),
                confidence,
                wavelength_range: (250.0, 270.0),
                observed_transitions,
            })
        } else {
            None
        }
    }

    /// Check for carbonyl chromophore
    fn check_carbonyl_chromophore(&self, transitions: &[ElectronicTransition]) -> Option<ChromophoreID> {
        let carbonyl_transitions: Vec<_> = transitions.iter()
            .filter(|t| t.wavelength >= 280.0 && t.wavelength <= 320.0)
            .collect();
        
        if !carbonyl_transitions.is_empty() {
            let confidence = (carbonyl_transitions.len() as f64 * 0.4).min(1.0);
            let observed_transitions: Vec<f64> = carbonyl_transitions.iter()
                .map(|t| t.wavelength).collect();
            
            Some(ChromophoreID {
                chromophore_name: "Carbonyl (n→π*)".to_string(),
                confidence,
                wavelength_range: (280.0, 320.0),
                observed_transitions,
            })
        } else {
            None
        }
    }

    /// Check for conjugated system
    fn check_conjugated_system(&self, transitions: &[ElectronicTransition]) -> Option<ChromophoreID> {
        let conjugated_transitions: Vec<_> = transitions.iter()
            .filter(|t| t.wavelength >= 320.0 && t.wavelength <= 500.0)
            .collect();
        
        if !conjugated_transitions.is_empty() {
            let confidence = (conjugated_transitions.len() as f64 * 0.2).min(1.0);
            let observed_transitions: Vec<f64> = conjugated_transitions.iter()
                .map(|t| t.wavelength).collect();
            
            Some(ChromophoreID {
                chromophore_name: "Conjugated System".to_string(),
                confidence,
                wavelength_range: (320.0, 500.0),
                observed_transitions,
            })
        } else {
            None
        }
    }

    /// Calculate extinction coefficients
    fn calculate_extinction_coefficients(&self, transitions: &[ElectronicTransition]) -> Vec<ExtinctionCoefficient> {
        transitions.iter().map(|t| {
            // Simplified calculation - would need concentration and path length
            let coefficient = t.absorbance * 1000.0; // Placeholder calculation
            
            ExtinctionCoefficient {
                wavelength: t.wavelength,
                coefficient,
                uncertainty: coefficient * 0.1, // 10% uncertainty
            }
        }).collect()
    }

    /// Analyze band gap
    fn analyze_band_gap(&self, data: &UVVisSpectrumData) -> Option<BandGap> {
        // Simple band gap estimation from absorption edge
        let absorption_edge = self.find_absorption_edge(data);
        
        if let Some(edge_wavelength) = absorption_edge {
            // Convert wavelength to energy: E (eV) = 1240 / λ (nm)
            let energy = 1240.0 / edge_wavelength;
            
            Some(BandGap {
                energy,
                gap_type: BandGapType::Direct, // Simplified assumption
                tauc_analysis: None, // Would need more complex analysis
            })
        } else {
            None
        }
    }

    /// Find absorption edge
    fn find_absorption_edge(&self, data: &UVVisSpectrumData) -> Option<f64> {
        // Find wavelength where absorbance starts to increase significantly
        let threshold = 0.1;
        
        for (i, &absorbance) in data.absorbance_data.iter().enumerate() {
            if absorbance > threshold {
                return Some(data.wavelength_data[i]);
            }
        }
        
        None
    }
}

/// UV-Vis spectrum data
#[derive(Debug, Clone)]
pub struct UVVisSpectrumData {
    /// Spectrum identifier
    pub id: String,
    /// Wavelength data (nm)
    pub wavelength_data: Vec<f64>,
    /// Absorbance data
    pub absorbance_data: Vec<f64>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl UVVisSpectrumData {
    /// Create new UV-Vis spectrum data
    pub fn new(id: String) -> Self {
        Self {
            id,
            wavelength_data: Vec::new(),
            absorbance_data: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add data point
    pub fn add_data_point(&mut self, wavelength: f64, absorbance: f64) {
        self.wavelength_data.push(wavelength);
        self.absorbance_data.push(absorbance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uv_vis_analysis() {
        let analyzer = UVVisAnalyzer::default();
        let mut spectrum = UVVisSpectrumData::new("test_uv_vis".to_string());
        
        // Add some test data points
        spectrum.add_data_point(260.0, 0.8); // Benzene-like absorption
        spectrum.add_data_point(300.0, 0.3); // n→π* transition
        spectrum.add_data_point(400.0, 0.1); // Weak absorption
        
        let result = analyzer.analyze(&spectrum);
        
        assert_eq!(result.spectrum_id, "test_uv_vis");
        assert!(!result.transitions.is_empty());
    }
} 