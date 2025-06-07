//! NMR spectroscopy analysis module
//!
//! This module provides functionality for Nuclear Magnetic Resonance (NMR) data analysis.

use std::collections::HashMap;

/// NMR analyzer
#[derive(Debug, Clone)]
pub struct NMRAnalyzer {
    /// Analysis configuration
    config: NMRConfig,
}

/// Configuration for NMR analysis
#[derive(Debug, Clone)]
pub struct NMRConfig {
    /// Chemical shift tolerance (ppm)
    pub chemical_shift_tolerance: f64,
    /// Minimum peak intensity
    pub min_peak_intensity: f64,
    /// Integration threshold
    pub integration_threshold: f64,
}

/// NMR analysis result
#[derive(Debug, Clone)]
pub struct NMRAnalysisResult {
    /// Spectrum ID
    pub spectrum_id: String,
    /// NMR nucleus type
    pub nucleus: NMRNucleus,
    /// Chemical shift peaks
    pub chemical_shifts: Vec<ChemicalShiftPeak>,
    /// Integration data
    pub integrations: Vec<Integration>,
    /// Multipicity analysis
    pub multiplicities: Vec<Multiplicity>,
}

/// NMR nucleus types
#[derive(Debug, Clone)]
pub enum NMRNucleus {
    /// Proton (1H)
    Proton,
    /// Carbon-13 (13C)
    Carbon13,
    /// Nitrogen-15 (15N)
    Nitrogen15,
    /// Phosphorus-31 (31P)
    Phosphorus31,
    /// Fluorine-19 (19F)
    Fluorine19,
    /// Custom nucleus
    Custom(String),
}

/// Chemical shift peak
#[derive(Debug, Clone)]
pub struct ChemicalShiftPeak {
    /// Chemical shift (ppm)
    pub chemical_shift: f64,
    /// Peak intensity
    pub intensity: f64,
    /// Peak width
    pub width: f64,
    /// Assignment
    pub assignment: Option<String>,
}

/// Integration data
#[derive(Debug, Clone)]
pub struct Integration {
    /// Integration range (ppm)
    pub range: (f64, f64),
    /// Integrated area
    pub area: f64,
    /// Relative integration
    pub relative_integration: f64,
    /// Number of protons (for 1H NMR)
    pub proton_count: Option<usize>,
}

/// Multiplicity information
#[derive(Debug, Clone)]
pub struct Multiplicity {
    /// Chemical shift center (ppm)
    pub center: f64,
    /// Multiplicity type
    pub multiplicity_type: MultiplicityType,
    /// Coupling constants (Hz)
    pub coupling_constants: Vec<f64>,
}

/// Types of multiplicity
#[derive(Debug, Clone)]
pub enum MultiplicityType {
    /// Singlet
    Singlet,
    /// Doublet
    Doublet,
    /// Triplet
    Triplet,
    /// Quartet
    Quartet,
    /// Quintet
    Quintet,
    /// Multiplet
    Multiplet,
    /// Complex pattern
    Complex,
}

impl Default for NMRConfig {
    fn default() -> Self {
        Self {
            chemical_shift_tolerance: 0.1,
            min_peak_intensity: 100.0,
            integration_threshold: 0.01,
        }
    }
}

impl NMRAnalyzer {
    /// Create a new NMR analyzer
    pub fn new(config: NMRConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(NMRConfig::default())
    }

    /// Analyze NMR spectrum data
    pub fn analyze(&self, spectrum_data: &NMRSpectrumData) -> NMRAnalysisResult {
        let chemical_shifts = self.find_chemical_shifts(spectrum_data);
        let integrations = self.calculate_integrations(spectrum_data, &chemical_shifts);
        let multiplicities = self.analyze_multiplicities(&chemical_shifts);

        NMRAnalysisResult {
            spectrum_id: spectrum_data.id.clone(),
            nucleus: spectrum_data.nucleus.clone(),
            chemical_shifts,
            integrations,
            multiplicities,
        }
    }

    /// Find chemical shift peaks
    fn find_chemical_shifts(&self, data: &NMRSpectrumData) -> Vec<ChemicalShiftPeak> {
        let mut peaks = Vec::new();
        let mut i = 1;

        while i < data.intensity_data.len() - 1 {
            let prev_intensity = data.intensity_data[i - 1];
            let curr_intensity = data.intensity_data[i];
            let next_intensity = data.intensity_data[i + 1];

            // Simple peak detection
            if curr_intensity > prev_intensity && curr_intensity > next_intensity && 
               curr_intensity > self.config.min_peak_intensity {
                
                let chemical_shift = data.chemical_shift_data[i];
                let width = self.calculate_peak_width(data, i);
                
                peaks.push(ChemicalShiftPeak {
                    chemical_shift,
                    intensity: curr_intensity,
                    width,
                    assignment: None,
                });
            }
            i += 1;
        }

        peaks
    }

    /// Calculate peak width
    fn calculate_peak_width(&self, data: &NMRSpectrumData, peak_index: usize) -> f64 {
        let peak_intensity = data.intensity_data[peak_index];
        let half_height = peak_intensity / 2.0;
        
        // Find width at half height
        let mut left_index = peak_index;
        let mut right_index = peak_index;
        
        // Go left
        while left_index > 0 && data.intensity_data[left_index] > half_height {
            left_index -= 1;
        }
        
        // Go right
        while right_index < data.intensity_data.len() - 1 && data.intensity_data[right_index] > half_height {
            right_index += 1;
        }
        
        if right_index > left_index {
            data.chemical_shift_data[right_index] - data.chemical_shift_data[left_index]
        } else {
            0.1 // Default width
        }
    }

    /// Calculate integrations
    fn calculate_integrations(&self, data: &NMRSpectrumData, peaks: &[ChemicalShiftPeak]) -> Vec<Integration> {
        let mut integrations = Vec::new();
        
        for peak in peaks {
            let start_shift = peak.chemical_shift - peak.width / 2.0;
            let end_shift = peak.chemical_shift + peak.width / 2.0;
            
            let area = self.integrate_region(data, start_shift, end_shift);
            
            integrations.push(Integration {
                range: (start_shift, end_shift),
                area,
                relative_integration: area, // Will normalize later
                proton_count: None,
            });
        }
        
        // Normalize integrations
        if !integrations.is_empty() {
            let total_area: f64 = integrations.iter().map(|i| i.area).sum();
            for integration in &mut integrations {
                integration.relative_integration = integration.area / total_area;
            }
        }
        
        integrations
    }

    /// Integrate a region
    fn integrate_region(&self, data: &NMRSpectrumData, start_shift: f64, end_shift: f64) -> f64 {
        let mut area = 0.0;
        
        for i in 0..data.chemical_shift_data.len() - 1 {
            let shift = data.chemical_shift_data[i];
            if shift >= start_shift && shift <= end_shift {
                let width = (data.chemical_shift_data[i + 1] - data.chemical_shift_data[i]).abs();
                area += data.intensity_data[i] * width;
            }
        }
        
        area
    }

    /// Analyze multiplicities
    fn analyze_multiplicities(&self, peaks: &[ChemicalShiftPeak]) -> Vec<Multiplicity> {
        peaks.iter().map(|peak| {
            let multiplicity_type = self.determine_multiplicity_type(peak);
            let coupling_constants = self.calculate_coupling_constants(peak);
            
            Multiplicity {
                center: peak.chemical_shift,
                multiplicity_type,
                coupling_constants,
            }
        }).collect()
    }

    /// Determine multiplicity type
    fn determine_multiplicity_type(&self, peak: &ChemicalShiftPeak) -> MultiplicityType {
        // Simplified multiplicity detection based on peak width
        if peak.width < 0.01 {
            MultiplicityType::Singlet
        } else if peak.width < 0.05 {
            MultiplicityType::Doublet
        } else if peak.width < 0.1 {
            MultiplicityType::Triplet
        } else {
            MultiplicityType::Multiplet
        }
    }

    /// Calculate coupling constants
    fn calculate_coupling_constants(&self, peak: &ChemicalShiftPeak) -> Vec<f64> {
        // Simplified: return estimated coupling constant based on width
        if peak.width > 0.01 {
            vec![peak.width * 100.0] // Convert to Hz (rough approximation)
        } else {
            Vec::new()
        }
    }
}

/// NMR spectrum data
#[derive(Debug, Clone)]
pub struct NMRSpectrumData {
    /// Spectrum identifier
    pub id: String,
    /// NMR nucleus
    pub nucleus: NMRNucleus,
    /// Chemical shift data (ppm)
    pub chemical_shift_data: Vec<f64>,
    /// Intensity data
    pub intensity_data: Vec<f64>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl NMRSpectrumData {
    /// Create new NMR spectrum data
    pub fn new(id: String, nucleus: NMRNucleus) -> Self {
        Self {
            id,
            nucleus,
            chemical_shift_data: Vec::new(),
            intensity_data: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add data point
    pub fn add_data_point(&mut self, chemical_shift: f64, intensity: f64) {
        self.chemical_shift_data.push(chemical_shift);
        self.intensity_data.push(intensity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmr_analysis() {
        let analyzer = NMRAnalyzer::default();
        let mut spectrum = NMRSpectrumData::new("test_nmr".to_string(), NMRNucleus::Proton);
        
        // Add some test data points
        spectrum.add_data_point(7.2, 1000.0);
        spectrum.add_data_point(3.8, 800.0);
        spectrum.add_data_point(1.2, 1200.0);
        
        let result = analyzer.analyze(&spectrum);
        
        assert_eq!(result.spectrum_id, "test_nmr");
        assert!(matches!(result.nucleus, NMRNucleus::Proton));
    }
} 