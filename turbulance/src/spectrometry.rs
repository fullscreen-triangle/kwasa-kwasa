//! Spectrometry analysis module
//! 
//! This module provides spectrometry analysis capabilities:
//! - Mass spectrometry (MS, MS/MS, SWATH-MS)
//! - NMR spectroscopy
//! - IR spectroscopy  
//! - UV-Vis spectroscopy

use crate::interpreter::Value;
use crate::error::Result;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Mass spectrum representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MassSpectrum {
    /// Mass-to-charge ratios
    pub mz: Vec<f64>,
    /// Intensity values
    pub intensity: Vec<f64>,
    /// Spectrum metadata
    pub metadata: HashMap<String, String>,
}

/// NMR spectrum representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NMRSpectrum {
    /// Chemical shifts (ppm)
    pub chemical_shifts: Vec<f64>,
    /// Signal intensities
    pub intensities: Vec<f64>,
    /// Nucleus type (1H, 13C, etc.)
    pub nucleus: String,
    /// Frequency (MHz)
    pub frequency: f64,
}

/// Peak in a spectrum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peak {
    /// Peak position (m/z, ppm, wavenumber, etc.)
    pub position: f64,
    /// Peak intensity
    pub intensity: f64,
    /// Peak width
    pub width: f64,
    /// Signal-to-noise ratio
    pub snr: f64,
}

impl MassSpectrum {
    /// Create new mass spectrum
    pub fn new(mz: Vec<f64>, intensity: Vec<f64>) -> Self {
        Self {
            mz,
            intensity,
            metadata: HashMap::new(),
        }
    }

    /// Find peaks in the spectrum
    pub fn find_peaks(&self, threshold: f64) -> Vec<Peak> {
        let mut peaks = Vec::new();
        
        for i in 1..self.intensity.len() - 1 {
            if self.intensity[i] > threshold &&
               self.intensity[i] > self.intensity[i-1] &&
               self.intensity[i] > self.intensity[i+1] {
                peaks.push(Peak {
                    position: self.mz[i],
                    intensity: self.intensity[i],
                    width: 0.1, // Mock width
                    snr: self.intensity[i] / 1000.0, // Mock SNR
                });
            }
        }
        
        peaks
    }

    /// Calculate base peak
    pub fn base_peak(&self) -> Option<Peak> {
        if let Some((max_idx, &max_intensity)) = self.intensity.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            Some(Peak {
                position: self.mz[max_idx],
                intensity: max_intensity,
                width: 0.1,
                snr: max_intensity / 1000.0,
            })
        } else {
            None
        }
    }

    /// Calculate molecular ion peak
    pub fn molecular_ion(&self) -> Option<Peak> {
        // Simple heuristic: find highest m/z peak above threshold
        let threshold = self.intensity.iter().fold(0.0, |max, &x| max.max(x)) * 0.1;
        
        for i in (0..self.intensity.len()).rev() {
            if self.intensity[i] > threshold {
                return Some(Peak {
                    position: self.mz[i],
                    intensity: self.intensity[i],
                    width: 0.1,
                    snr: self.intensity[i] / 1000.0,
                });
            }
        }
        None
    }
}

/// Analyze mass spectrum for compound identification
pub fn identify_compound(spectrum: &MassSpectrum, database: &str) -> Result<Vec<HashMap<String, Value>>> {
    // Mock compound identification
    let mut matches = Vec::new();
    
    let compound = {
        let mut c = HashMap::new();
        c.insert("name".to_string(), Value::String("Caffeine".to_string()));
        c.insert("formula".to_string(), Value::String("C8H10N4O2".to_string()));
        c.insert("molecular_weight".to_string(), Value::Number(194.19));
        c.insert("score".to_string(), Value::Number(0.85));
        c.insert("database".to_string(), Value::String(database.to_string()));
        c
    };
    
    matches.push(compound);
    Ok(matches)
}

/// Perform protein identification from MS/MS data
pub fn protein_identification(spectra: &[MassSpectrum]) -> Result<Vec<HashMap<String, Value>>> {
    // Mock protein identification
    let mut proteins = Vec::new();
    
    let protein = {
        let mut p = HashMap::new();
        p.insert("accession".to_string(), Value::String("P12345".to_string()));
        p.insert("name".to_string(), Value::String("Example protein".to_string()));
        p.insert("species".to_string(), Value::String("Homo sapiens".to_string()));
        p.insert("coverage".to_string(), Value::Number(0.65));
        p.insert("peptides".to_string(), Value::Number(12.0));
        p.insert("score".to_string(), Value::Number(95.0));
        p
    };
    
    proteins.push(protein);
    Ok(proteins)
}

/// Quantify compounds using peak areas
pub fn quantify_compounds(spectrum: &MassSpectrum, targets: &[f64]) -> Result<HashMap<String, f64>> {
    let mut quantities = HashMap::new();
    
    for (i, &target_mz) in targets.iter().enumerate() {
        // Find closest peak
        let mut best_match = 0;
        let mut min_diff = f64::INFINITY;
        
        for (j, &mz) in spectrum.mz.iter().enumerate() {
            let diff = (mz - target_mz).abs();
            if diff < min_diff {
                min_diff = diff;
                best_match = j;
            }
        }
        
        if min_diff < 0.1 { // Within 0.1 Da
            quantities.insert(format!("compound_{}", i + 1), spectrum.intensity[best_match]);
        }
    }
    
    Ok(quantities)
}

/// Analyze NMR spectrum
pub fn analyze_nmr(spectrum: &NMRSpectrum) -> Result<HashMap<String, Value>> {
    let mut analysis = HashMap::new();
    
    // Mock NMR analysis
    analysis.insert("num_signals".to_string(), Value::Number(spectrum.chemical_shifts.len() as f64));
    analysis.insert("chemical_shift_range".to_string(), Value::Array(vec![
        Value::Number(spectrum.chemical_shifts.iter().fold(f64::INFINITY, |min, &x| min.min(x))),
        Value::Number(spectrum.chemical_shifts.iter().fold(f64::NEG_INFINITY, |max, &x| max.max(x))),
    ]));
    analysis.insert("total_intensity".to_string(), Value::Number(spectrum.intensities.iter().sum::<f64>()));
    
    Ok(analysis)
}

/// Predict molecular structure from spectra
pub fn structure_prediction(mass_spectrum: &MassSpectrum, nmr_spectrum: Option<&NMRSpectrum>) -> Result<HashMap<String, Value>> {
    let mut prediction = HashMap::new();
    
    // Mock structure prediction
    if let Some(mol_ion) = mass_spectrum.molecular_ion() {
        prediction.insert("molecular_weight".to_string(), Value::Number(mol_ion.position));
    }
    
    prediction.insert("predicted_formula".to_string(), Value::String("C8H10N4O2".to_string()));
    prediction.insert("confidence".to_string(), Value::Number(0.75));
    prediction.insert("structure_suggestions".to_string(), Value::Array(vec![
        Value::String("Caffeine".to_string()),
        Value::String("Theophylline".to_string()),
    ]));
    
    if nmr_spectrum.is_some() {
        prediction.insert("nmr_analysis".to_string(), Value::String("Consistent with predicted structure".to_string()));
    }
    
    Ok(prediction)
}

/// Perform metabolomics analysis
pub fn metabolomics_analysis(spectra: &[MassSpectrum]) -> Result<HashMap<String, Value>> {
    let mut analysis = HashMap::new();
    
    // Mock metabolomics analysis
    analysis.insert("total_features".to_string(), Value::Number(spectra.len() as f64));
    analysis.insert("identified_metabolites".to_string(), Value::Number(156.0));
    analysis.insert("pathway_coverage".to_string(), Value::Number(0.68));
    analysis.insert("quality_score".to_string(), Value::Number(0.82));
    
    // Mock pathway analysis
    let pathways = vec![
        Value::String("Glycolysis".to_string()),
        Value::String("TCA cycle".to_string()),
        Value::String("Fatty acid metabolism".to_string()),
    ];
    analysis.insert("enriched_pathways".to_string(), Value::Array(pathways));
    
    Ok(analysis)
}

/// Calibrate mass spectrometer
pub fn calibrate_mass_spec(reference_masses: &[f64], observed_masses: &[f64]) -> Result<HashMap<String, f64>> {
    let mut calibration = HashMap::new();
    
    // Simple linear calibration
    if reference_masses.len() != observed_masses.len() || reference_masses.is_empty() {
        return Err(crate::error::TurbulanceError::argument_error("Reference and observed masses must have same length"));
    }
    
    // Calculate mean differences (mock calibration)
    let mean_error: f64 = reference_masses.iter()
        .zip(observed_masses.iter())
        .map(|(r, o)| (o - r) / r * 1e6) // ppm error
        .sum::<f64>() / reference_masses.len() as f64;
    
    calibration.insert("mass_accuracy_ppm".to_string(), mean_error.abs());
    calibration.insert("calibration_slope".to_string(), 1.000001);
    calibration.insert("calibration_intercept".to_string(), 0.0001);
    calibration.insert("r_squared".to_string(), 0.9998);
    
    Ok(calibration)
}

/// Process LC-MS data
pub fn process_lcms_data(retention_times: &[f64], spectra: &[MassSpectrum]) -> Result<HashMap<String, Value>> {
    let mut analysis = HashMap::new();
    
    // Mock LC-MS processing
    analysis.insert("total_peaks".to_string(), Value::Number(1250.0));
    analysis.insert("rt_range_min".to_string(), Value::Number(retention_times.iter().fold(f64::INFINITY, |min, &x| min.min(x))));
    analysis.insert("rt_range_max".to_string(), Value::Number(retention_times.iter().fold(f64::NEG_INFINITY, |max, &x| max.max(x))));
    analysis.insert("num_timepoints".to_string(), Value::Number(spectra.len() as f64));
    
    // Mock compound detection
    let compounds = vec![
        Value::String("Caffeine (RT: 3.2 min)".to_string()),
        Value::String("Glucose (RT: 1.8 min)".to_string()),
        Value::String("Aspirin (RT: 5.1 min)".to_string()),
    ];
    analysis.insert("detected_compounds".to_string(), Value::Array(compounds));
    
    Ok(analysis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_spectrum_creation() {
        let mz = vec![100.0, 200.0, 300.0];
        let intensity = vec![1000.0, 5000.0, 2000.0];
        let spectrum = MassSpectrum::new(mz.clone(), intensity.clone());
        
        assert_eq!(spectrum.mz, mz);
        assert_eq!(spectrum.intensity, intensity);
    }

    #[test]
    fn test_base_peak() {
        let spectrum = MassSpectrum::new(
            vec![100.0, 200.0, 300.0],
            vec![1000.0, 5000.0, 2000.0]
        );
        
        let base_peak = spectrum.base_peak().unwrap();
        assert_eq!(base_peak.position, 200.0);
        assert_eq!(base_peak.intensity, 5000.0);
    }

    #[test]
    fn test_peak_finding() {
        let spectrum = MassSpectrum::new(
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
            vec![100.0, 5000.0, 100.0, 3000.0, 100.0]
        );
        
        let peaks = spectrum.find_peaks(1000.0);
        assert_eq!(peaks.len(), 2); // Should find peaks at 200.0 and 400.0
    }
} 