//! Mass spectrometry analysis module
//!
//! This module provides specific functionality for mass spectrometry data analysis.

use std::collections::HashMap;
use super::{MassSpectrum, Peak, Unit};

/// Mass spectrometry analyzer
#[derive(Debug, Clone)]
pub struct MassSpecAnalyzer {
    /// Analysis configuration
    config: MassSpecConfig,
}

/// Configuration for mass spectrometry analysis
#[derive(Debug, Clone)]
pub struct MassSpecConfig {
    /// Mass accuracy tolerance (ppm)
    pub mass_accuracy_ppm: f64,
    /// Minimum peak intensity
    pub min_peak_intensity: f64,
    /// Signal-to-noise threshold
    pub snr_threshold: f64,
    /// Enable isotope pattern analysis
    pub enable_isotope_analysis: bool,
}

/// Mass spectrometry analysis result
#[derive(Debug, Clone)]
pub struct MassSpecAnalysisResult {
    /// Spectrum ID
    pub spectrum_id: String,
    /// Detected peaks
    pub peaks: Vec<AnalyzedPeak>,
    /// Molecular ion information
    pub molecular_ion: Option<MolecularIon>,
    /// Fragment analysis
    pub fragments: Vec<Fragment>,
    /// Isotope patterns
    pub isotope_patterns: Vec<IsotopePattern>,
}

/// Analyzed peak with additional information
#[derive(Debug, Clone)]
pub struct AnalyzedPeak {
    /// Base peak information
    pub peak: Peak,
    /// Possible assignments
    pub assignments: Vec<PeakAssignment>,
    /// Confidence score
    pub confidence: f64,
}

/// Peak assignment
#[derive(Debug, Clone)]
pub struct PeakAssignment {
    /// Assignment type
    pub assignment_type: AssignmentType,
    /// Description
    pub description: String,
    /// Confidence
    pub confidence: f64,
}

/// Type of peak assignment
#[derive(Debug, Clone)]
pub enum AssignmentType {
    /// Molecular ion
    MolecularIon,
    /// Fragment ion
    Fragment,
    /// Isotope peak
    Isotope,
    /// Adduct
    Adduct,
    /// Background
    Background,
}

/// Molecular ion information
#[derive(Debug, Clone)]
pub struct MolecularIon {
    /// m/z value
    pub mz: f64,
    /// Intensity
    pub intensity: f64,
    /// Charge state
    pub charge: i32,
    /// Mass accuracy (ppm)
    pub mass_accuracy: f64,
}

/// Fragment ion information
#[derive(Debug, Clone)]
pub struct Fragment {
    /// m/z value
    pub mz: f64,
    /// Intensity
    pub intensity: f64,
    /// Fragment type
    pub fragment_type: FragmentType,
    /// Neutral loss from parent
    pub neutral_loss: Option<f64>,
}

/// Type of fragment
#[derive(Debug, Clone)]
pub enum FragmentType {
    /// Simple fragmentation
    Simple,
    /// Rearrangement
    Rearrangement,
    /// Loss
    Loss,
    /// Unknown
    Unknown,
}

/// Isotope pattern information
#[derive(Debug, Clone)]
pub struct IsotopePattern {
    /// Monoisotopic peak
    pub monoisotopic_peak: Peak,
    /// Isotope peaks
    pub isotope_peaks: Vec<IsotopePeak>,
    /// Pattern quality
    pub quality: f64,
}

/// Individual isotope peak
#[derive(Debug, Clone)]
pub struct IsotopePeak {
    /// m/z value
    pub mz: f64,
    /// Relative intensity
    pub relative_intensity: f64,
    /// Isotope number
    pub isotope_number: usize,
}

impl Default for MassSpecConfig {
    fn default() -> Self {
        Self {
            mass_accuracy_ppm: 5.0,
            min_peak_intensity: 1000.0,
            snr_threshold: 3.0,
            enable_isotope_analysis: true,
        }
    }
}

impl MassSpecAnalyzer {
    /// Create a new mass spec analyzer
    pub fn new(config: MassSpecConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MassSpecConfig::default())
    }

    /// Analyze a mass spectrum
    pub fn analyze(&self, spectrum: &MassSpectrum) -> MassSpecAnalysisResult {
        let spectrum_id = spectrum.id().to_string();
        
        // Analyze peaks
        let analyzed_peaks = self.analyze_peaks(spectrum);
        
        // Find molecular ion
        let molecular_ion = self.find_molecular_ion(&analyzed_peaks);
        
        // Analyze fragments
        let fragments = self.analyze_fragments(&analyzed_peaks, &molecular_ion);
        
        // Find isotope patterns
        let isotope_patterns = if self.config.enable_isotope_analysis {
            self.find_isotope_patterns(&analyzed_peaks)
        } else {
            Vec::new()
        };
        
        MassSpecAnalysisResult {
            spectrum_id,
            peaks: analyzed_peaks,
            molecular_ion,
            fragments,
            isotope_patterns,
        }
    }

    /// Analyze individual peaks
    fn analyze_peaks(&self, spectrum: &MassSpectrum) -> Vec<AnalyzedPeak> {
        spectrum.peaks()
            .iter()
            .filter(|peak| peak.intensity >= self.config.min_peak_intensity)
            .filter(|peak| {
                if let Some(snr) = peak.snr {
                    snr >= self.config.snr_threshold
                } else {
                    true // Assume good quality if SNR not available
                }
            })
            .map(|peak| {
                let assignments = self.assign_peak(peak);
                let confidence = self.calculate_peak_confidence(peak, &assignments);
                
                AnalyzedPeak {
                    peak: peak.clone(),
                    assignments,
                    confidence,
                }
            })
            .collect()
    }

    /// Assign a peak to possible identities
    fn assign_peak(&self, peak: &Peak) -> Vec<PeakAssignment> {
        let mut assignments = Vec::new();
        
        // Simple heuristic assignments
        if peak.mz > 50.0 && peak.intensity > 10000.0 {
            assignments.push(PeakAssignment {
                assignment_type: AssignmentType::MolecularIon,
                description: "Potential molecular ion".to_string(),
                confidence: 0.7,
            });
        }
        
        if peak.mz < 200.0 {
            assignments.push(PeakAssignment {
                assignment_type: AssignmentType::Fragment,
                description: "Fragment ion".to_string(),
                confidence: 0.8,
            });
        }
        
        assignments
    }

    /// Calculate confidence for peak assignments
    fn calculate_peak_confidence(&self, peak: &Peak, assignments: &[PeakAssignment]) -> f64 {
        if assignments.is_empty() {
            return 0.5;
        }
        
        let max_confidence = assignments.iter()
            .map(|a| a.confidence)
            .fold(0.0f64, f64::max);
        
        // Factor in intensity
        let intensity_factor = (peak.intensity / 10000.0).min(1.0);
        
        max_confidence * intensity_factor
    }

    /// Find molecular ion
    fn find_molecular_ion(&self, peaks: &[AnalyzedPeak]) -> Option<MolecularIon> {
        // Simple heuristic: highest m/z peak with reasonable intensity
        peaks.iter()
            .filter(|p| p.peak.intensity > self.config.min_peak_intensity * 5.0)
            .max_by(|a, b| a.peak.mz.partial_cmp(&b.peak.mz).unwrap())
            .map(|peak| MolecularIon {
                mz: peak.peak.mz,
                intensity: peak.peak.intensity,
                charge: 1, // Assume singly charged
                mass_accuracy: 2.0, // Simplified
            })
    }

    /// Analyze fragments
    fn analyze_fragments(&self, peaks: &[AnalyzedPeak], molecular_ion: &Option<MolecularIon>) -> Vec<Fragment> {
        let mut fragments = Vec::new();
        
        if let Some(mol_ion) = molecular_ion {
            for peak in peaks {
                if peak.peak.mz < mol_ion.mz - 1.0 { // Not the molecular ion
                    let neutral_loss = mol_ion.mz - peak.peak.mz;
                    
                    fragments.push(Fragment {
                        mz: peak.peak.mz,
                        intensity: peak.peak.intensity,
                        fragment_type: FragmentType::Simple,
                        neutral_loss: Some(neutral_loss),
                    });
                }
            }
        }
        
        fragments
    }

    /// Find isotope patterns
    fn find_isotope_patterns(&self, peaks: &[AnalyzedPeak]) -> Vec<IsotopePattern> {
        let mut patterns = Vec::new();
        
        // Simple isotope pattern detection
        for (i, peak) in peaks.iter().enumerate() {
            let mut isotope_peaks = Vec::new();
            
            // Look for peaks 1 Da higher (simplified)
            for other_peak in peaks.iter().skip(i + 1) {
                let mass_diff = other_peak.peak.mz - peak.peak.mz;
                if (mass_diff - 1.0).abs() < 0.1 {
                    isotope_peaks.push(IsotopePeak {
                        mz: other_peak.peak.mz,
                        relative_intensity: other_peak.peak.intensity / peak.peak.intensity,
                        isotope_number: 1,
                    });
                }
            }
            
            if !isotope_peaks.is_empty() {
                patterns.push(IsotopePattern {
                    monoisotopic_peak: peak.peak.clone(),
                    isotope_peaks,
                    quality: 0.8, // Simplified quality score
                });
            }
        }
        
        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_spec_analysis() {
        let analyzer = MassSpecAnalyzer::default();
        let mut spectrum = MassSpectrum::new(b"test spectrum", "test");
        
        // Add some test peaks
        spectrum.add_peak(Peak::new(100.0, 5000.0));
        spectrum.add_peak(Peak::new(200.0, 15000.0));
        spectrum.add_peak(Peak::new(300.0, 8000.0));
        
        let result = analyzer.analyze(&spectrum);
        
        assert!(!result.peaks.is_empty());
        assert_eq!(result.spectrum_id, "test");
    }
} 