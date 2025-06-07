//! Infrared spectroscopy analysis module
//!
//! This module provides functionality for IR spectrum analysis and interpretation.

use std::collections::HashMap;

/// IR analyzer
#[derive(Debug, Clone)]
pub struct IRAnalyzer {
    /// Analysis configuration
    config: IRConfig,
}

/// Configuration for IR analysis
#[derive(Debug, Clone)]
pub struct IRConfig {
    /// Minimum peak intensity
    pub min_peak_intensity: f64,
    /// Peak detection threshold
    pub peak_threshold: f64,
    /// Baseline correction method
    pub baseline_method: BaselineMethod,
}

/// Baseline correction methods
#[derive(Debug, Clone)]
pub enum BaselineMethod {
    /// No correction
    None,
    /// Linear baseline
    Linear,
    /// Polynomial baseline
    Polynomial(usize),
    /// Rubber band
    RubberBand,
}

/// IR analysis result
#[derive(Debug, Clone)]
pub struct IRAnalysisResult {
    /// Spectrum ID
    pub spectrum_id: String,
    /// Vibrational bands
    pub vibrational_bands: Vec<VibrationalBand>,
    /// Functional group identifications
    pub functional_groups: Vec<FunctionalGroupID>,
    /// Fingerprint region analysis
    pub fingerprint_analysis: FingerprintAnalysis,
}

/// Vibrational band information
#[derive(Debug, Clone)]
pub struct VibrationalBand {
    /// Wavenumber (cm⁻¹)
    pub wavenumber: f64,
    /// Intensity (absorbance or transmittance)
    pub intensity: f64,
    /// Band width (cm⁻¹)
    pub width: f64,
    /// Vibration type
    pub vibration_type: VibrationType,
    /// Assignment
    pub assignment: Option<String>,
}

/// Types of vibrational modes
#[derive(Debug, Clone)]
pub enum VibrationType {
    /// Stretching vibration
    Stretching,
    /// Bending vibration
    Bending,
    /// Rocking vibration
    Rocking,
    /// Wagging vibration
    Wagging,
    /// Twisting vibration
    Twisting,
    /// Out-of-plane vibration
    OutOfPlane,
    /// Unknown
    Unknown,
}

/// Functional group identification
#[derive(Debug, Clone)]
pub struct FunctionalGroupID {
    /// Group name
    pub group_name: String,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Supporting bands (wavenumbers)
    pub supporting_bands: Vec<f64>,
    /// Expected bands for this group
    pub expected_bands: Vec<f64>,
}

/// Fingerprint region analysis
#[derive(Debug, Clone)]
pub struct FingerprintAnalysis {
    /// Fingerprint region (typically 500-1500 cm⁻¹)
    pub region: (f64, f64),
    /// Characteristic peaks in fingerprint region
    pub characteristic_peaks: Vec<VibrationalBand>,
    /// Pattern similarity to reference spectra
    pub similarity_scores: HashMap<String, f64>,
}

impl Default for IRConfig {
    fn default() -> Self {
        Self {
            min_peak_intensity: 0.01,
            peak_threshold: 0.05,
            baseline_method: BaselineMethod::Linear,
        }
    }
}

impl IRAnalyzer {
    /// Create a new IR analyzer
    pub fn new(config: IRConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(IRConfig::default())
    }

    /// Analyze IR spectrum
    pub fn analyze(&self, spectrum_data: &IRSpectrumData) -> IRAnalysisResult {
        // Apply baseline correction
        let corrected_data = self.apply_baseline_correction(spectrum_data);
        
        // Find vibrational bands
        let vibrational_bands = self.find_vibrational_bands(&corrected_data);
        
        // Identify functional groups
        let functional_groups = self.identify_functional_groups(&vibrational_bands);
        
        // Analyze fingerprint region
        let fingerprint_analysis = self.analyze_fingerprint_region(&vibrational_bands);
        
        IRAnalysisResult {
            spectrum_id: spectrum_data.id.clone(),
            vibrational_bands,
            functional_groups,
            fingerprint_analysis,
        }
    }

    /// Apply baseline correction
    fn apply_baseline_correction(&self, data: &IRSpectrumData) -> IRSpectrumData {
        match self.config.baseline_method {
            BaselineMethod::None => data.clone(),
            BaselineMethod::Linear => self.linear_baseline_correction(data),
            BaselineMethod::Polynomial(order) => self.polynomial_baseline_correction(data, order),
            BaselineMethod::RubberBand => self.rubber_band_correction(data),
        }
    }

    /// Linear baseline correction
    fn linear_baseline_correction(&self, data: &IRSpectrumData) -> IRSpectrumData {
        let mut corrected = data.clone();
        
        if !data.intensity_data.is_empty() {
            let first_intensity = data.intensity_data[0];
            let last_intensity = data.intensity_data[data.intensity_data.len() - 1];
            let slope = (last_intensity - first_intensity) / (data.intensity_data.len() as f64 - 1.0);
            
            for (i, intensity) in corrected.intensity_data.iter_mut().enumerate() {
                let baseline = first_intensity + slope * i as f64;
                *intensity -= baseline;
            }
        }
        
        corrected
    }

    /// Polynomial baseline correction
    fn polynomial_baseline_correction(&self, data: &IRSpectrumData, _order: usize) -> IRSpectrumData {
        // Simplified - just use linear for now
        self.linear_baseline_correction(data)
    }

    /// Rubber band baseline correction
    fn rubber_band_correction(&self, data: &IRSpectrumData) -> IRSpectrumData {
        // Simplified - use linear for now
        self.linear_baseline_correction(data)
    }

    /// Find vibrational bands
    fn find_vibrational_bands(&self, data: &IRSpectrumData) -> Vec<VibrationalBand> {
        let mut bands = Vec::new();
        let mut i = 1;

        while i < data.intensity_data.len() - 1 {
            let prev_intensity = data.intensity_data[i - 1];
            let curr_intensity = data.intensity_data[i];
            let next_intensity = data.intensity_data[i + 1];

            // Peak detection (looking for absorption peaks - negative values after baseline correction)
            if curr_intensity < prev_intensity && curr_intensity < next_intensity && 
               curr_intensity.abs() > self.config.min_peak_intensity {
                
                let wavenumber = data.wavenumber_data[i];
                let width = self.calculate_band_width(data, i);
                let vibration_type = self.classify_vibration_type(wavenumber);
                
                bands.push(VibrationalBand {
                    wavenumber,
                    intensity: curr_intensity.abs(),
                    width,
                    vibration_type,
                    assignment: None,
                });
            }
            i += 1;
        }

        bands.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
        bands
    }

    /// Calculate band width
    fn calculate_band_width(&self, data: &IRSpectrumData, peak_index: usize) -> f64 {
        let peak_intensity = data.intensity_data[peak_index].abs();
        let half_height = peak_intensity / 2.0;
        
        let mut left_index = peak_index;
        let mut right_index = peak_index;
        
        // Find width at half height
        while left_index > 0 && data.intensity_data[left_index].abs() > half_height {
            left_index -= 1;
        }
        
        while right_index < data.intensity_data.len() - 1 && data.intensity_data[right_index].abs() > half_height {
            right_index += 1;
        }
        
        if right_index > left_index {
            (data.wavenumber_data[right_index] - data.wavenumber_data[left_index]).abs()
        } else {
            10.0 // Default width
        }
    }

    /// Classify vibration type based on wavenumber
    fn classify_vibration_type(&self, wavenumber: f64) -> VibrationType {
        match wavenumber {
            3200.0..=3600.0 => VibrationType::Stretching, // O-H, N-H stretch
            2800.0..=3200.0 => VibrationType::Stretching, // C-H stretch
            1600.0..=1800.0 => VibrationType::Stretching, // C=O stretch
            1400.0..=1600.0 => VibrationType::Bending,    // CH2, CH3 bending
            1000.0..=1400.0 => VibrationType::Stretching, // C-O stretch
            600.0..=1000.0 => VibrationType::Bending,     // Out-of-plane bending
            _ => VibrationType::Unknown,
        }
    }

    /// Identify functional groups
    fn identify_functional_groups(&self, bands: &[VibrationalBand]) -> Vec<FunctionalGroupID> {
        let mut functional_groups = Vec::new();
        
        // Check for common functional groups
        if let Some(oh_group) = self.check_oh_group(bands) {
            functional_groups.push(oh_group);
        }
        
        if let Some(co_group) = self.check_carbonyl_group(bands) {
            functional_groups.push(co_group);
        }
        
        if let Some(ch_group) = self.check_ch_group(bands) {
            functional_groups.push(ch_group);
        }
        
        functional_groups
    }

    /// Check for O-H group
    fn check_oh_group(&self, bands: &[VibrationalBand]) -> Option<FunctionalGroupID> {
        let oh_bands: Vec<_> = bands.iter()
            .filter(|b| b.wavenumber >= 3200.0 && b.wavenumber <= 3600.0)
            .collect();
        
        if !oh_bands.is_empty() {
            let confidence = (oh_bands.len() as f64 * 0.3).min(1.0);
            let supporting_bands: Vec<f64> = oh_bands.iter().map(|b| b.wavenumber).collect();
            
            Some(FunctionalGroupID {
                group_name: "Hydroxyl (O-H)".to_string(),
                confidence,
                supporting_bands,
                expected_bands: vec![3200.0, 3400.0, 3600.0],
            })
        } else {
            None
        }
    }

    /// Check for C=O group
    fn check_carbonyl_group(&self, bands: &[VibrationalBand]) -> Option<FunctionalGroupID> {
        let co_bands: Vec<_> = bands.iter()
            .filter(|b| b.wavenumber >= 1650.0 && b.wavenumber <= 1750.0)
            .collect();
        
        if !co_bands.is_empty() {
            let confidence = (co_bands.len() as f64 * 0.4).min(1.0);
            let supporting_bands: Vec<f64> = co_bands.iter().map(|b| b.wavenumber).collect();
            
            Some(FunctionalGroupID {
                group_name: "Carbonyl (C=O)".to_string(),
                confidence,
                supporting_bands,
                expected_bands: vec![1650.0, 1700.0, 1750.0],
            })
        } else {
            None
        }
    }

    /// Check for C-H group
    fn check_ch_group(&self, bands: &[VibrationalBand]) -> Option<FunctionalGroupID> {
        let ch_bands: Vec<_> = bands.iter()
            .filter(|b| b.wavenumber >= 2800.0 && b.wavenumber <= 3200.0)
            .collect();
        
        if !ch_bands.is_empty() {
            let confidence = (ch_bands.len() as f64 * 0.2).min(1.0);
            let supporting_bands: Vec<f64> = ch_bands.iter().map(|b| b.wavenumber).collect();
            
            Some(FunctionalGroupID {
                group_name: "Alkyl C-H".to_string(),
                confidence,
                supporting_bands,
                expected_bands: vec![2850.0, 2950.0, 3000.0],
            })
        } else {
            None
        }
    }

    /// Analyze fingerprint region
    fn analyze_fingerprint_region(&self, bands: &[VibrationalBand]) -> FingerprintAnalysis {
        let fingerprint_region = (500.0, 1500.0);
        
        let characteristic_peaks: Vec<_> = bands.iter()
            .filter(|b| b.wavenumber >= fingerprint_region.0 && b.wavenumber <= fingerprint_region.1)
            .cloned()
            .collect();
        
        // Simplified similarity scoring
        let mut similarity_scores = HashMap::new();
        similarity_scores.insert("alkane".to_string(), 0.5);
        similarity_scores.insert("alcohol".to_string(), 0.3);
        
        FingerprintAnalysis {
            region: fingerprint_region,
            characteristic_peaks,
            similarity_scores,
        }
    }
}

/// IR spectrum data
#[derive(Debug, Clone)]
pub struct IRSpectrumData {
    /// Spectrum identifier
    pub id: String,
    /// Wavenumber data (cm⁻¹)
    pub wavenumber_data: Vec<f64>,
    /// Intensity data (absorbance or transmittance)
    pub intensity_data: Vec<f64>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl IRSpectrumData {
    /// Create new IR spectrum data
    pub fn new(id: String) -> Self {
        Self {
            id,
            wavenumber_data: Vec::new(),
            intensity_data: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add data point
    pub fn add_data_point(&mut self, wavenumber: f64, intensity: f64) {
        self.wavenumber_data.push(wavenumber);
        self.intensity_data.push(intensity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_analysis() {
        let analyzer = IRAnalyzer::default();
        let mut spectrum = IRSpectrumData::new("test_ir".to_string());
        
        // Add some test data points (typical IR peaks)
        spectrum.add_data_point(3400.0, -0.5); // O-H stretch
        spectrum.add_data_point(1700.0, -0.8); // C=O stretch
        spectrum.add_data_point(2950.0, -0.3); // C-H stretch
        
        let result = analyzer.analyze(&spectrum);
        
        assert_eq!(result.spectrum_id, "test_ir");
        assert!(!result.vibrational_bands.is_empty());
    }
} 