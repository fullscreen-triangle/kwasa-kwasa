use std::sync::Arc;
use rayon::prelude::*;
use crate::spectrometry::{MassSpectrum, Peak, UnitId, SpectrumMetadata, MzRange};
use std::collections::{HashMap, BTreeMap, HashSet};

/// High-throughput mass spectrometry operations for parallel processing
pub struct HighThroughputSpectrometry;

impl HighThroughputSpectrometry {
    /// Create a new instance
    pub fn new() -> Self {
        Self
    }
    
    /// Process multiple spectra in parallel
    /// 
    /// Applies a transformation function to each spectrum in parallel
    pub fn process_spectra_parallel<F>(&self, 
                                      spectra: &[MassSpectrum], 
                                      processor: F) -> Vec<MassSpectrum>
    where
        F: Fn(&MassSpectrum) -> MassSpectrum + Send + Sync,
    {
        spectra.par_iter()
            .map(|spectrum| processor(spectrum))
            .collect()
    }
    
    /// Find peaks in multiple spectra in parallel
    /// 
    /// This function identifies peaks above a threshold in multiple spectra simultaneously
    pub fn find_peaks_parallel(&self, 
                              spectra: &[MassSpectrum], 
                              min_intensity: f64,
                              min_snr: Option<f64>) -> Vec<Vec<Peak>> {
        spectra.par_iter()
            .map(|spectrum| self.find_peaks_in_spectrum(spectrum, min_intensity, min_snr))
            .collect()
    }
    
    /// Find peaks in a single spectrum (used as a subroutine)
    fn find_peaks_in_spectrum(&self, 
                             spectrum: &MassSpectrum, 
                             min_intensity: f64,
                             min_snr: Option<f64>) -> Vec<Peak> {
        let peaks = spectrum.peaks();
        
        // Filter by intensity
        let filtered_peaks: Vec<Peak> = peaks.iter()
            .filter(|peak| {
                let intensity_ok = peak.intensity >= min_intensity;
                
                // Filter by SNR if specified
                if let Some(min_snr_value) = min_snr {
                    if let Some(snr) = peak.snr {
                        intensity_ok && snr >= min_snr_value
                    } else {
                        intensity_ok
                    }
                } else {
                    intensity_ok
                }
            })
            .cloned()
            .collect();
        
        filtered_peaks
    }
    
    /// Extract chromatograms for multiple m/z values in parallel from LC-MS data
    /// 
    /// For each m/z value in the input vector, extract an extracted ion chromatogram (XIC)
    /// from a series of spectra (representing an LC-MS run)
    pub fn extract_chromatograms_parallel(&self, 
                                         spectra: &[MassSpectrum], 
                                         mz_values: &[f64], 
                                         tolerance: f64) -> HashMap<f64, Vec<(f64, f64)>> {
        // Process each m/z value in parallel
        let chromatograms: Vec<(f64, Vec<(f64, f64)>)> = mz_values.par_iter()
            .map(|&mz| {
                let chromatogram = self.extract_single_chromatogram(spectra, mz, tolerance);
                (mz, chromatogram)
            })
            .collect();
        
        // Convert to hashmap
        let mut result = HashMap::new();
        for (mz, chromatogram) in chromatograms {
            result.insert(mz, chromatogram);
        }
        
        result
    }
    
    /// Extract a single chromatogram (used as a subroutine)
    fn extract_single_chromatogram(&self, 
                                  spectra: &[MassSpectrum], 
                                  mz: f64, 
                                  tolerance: f64) -> Vec<(f64, f64)> {
        let mut chromatogram = Vec::with_capacity(spectra.len());
        
        // For each spectrum (representing a time point)
        for (i, spectrum) in spectra.iter().enumerate() {
            // Get the retention time (use index if not available)
            let retention_time = spectrum.metadata().annotations
                .get("retention_time")
                .and_then(|rt_str| rt_str.parse::<f64>().ok())
                .unwrap_or(i as f64);
            
            // Find peaks within the m/z range
            let intensity = spectrum.peaks_in_range(mz - tolerance, mz + tolerance)
                .iter()
                .map(|peak| peak.intensity)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            
            chromatogram.push((retention_time, intensity));
        }
        
        // Sort by retention time
        chromatogram.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        chromatogram
    }
    
    /// Parallel deconvolution of spectra (simplified implementation)
    /// 
    /// Deconvolutes isotope patterns and charge states to identify monoisotopic masses
    pub fn deconvolute_spectra_parallel(&self, 
                                       spectra: &[MassSpectrum]) -> Vec<Vec<DeconvolutedPeak>> {
        spectra.par_iter()
            .map(|spectrum| self.deconvolute_spectrum(spectrum))
            .collect()
    }
    
    /// Deconvolute a single spectrum (used as a subroutine)
    fn deconvolute_spectrum(&self, spectrum: &MassSpectrum) -> Vec<DeconvolutedPeak> {
        let peaks = spectrum.peaks();
        let mut deconvoluted = Vec::new();
        
        // This is a simplified implementation - a real one would use sophisticated algorithms
        // for isotope pattern detection and charge state determination
        
        // Simple approach: look for peak patterns separated by 1/z (for different charge states)
        // Group peaks that could be part of the same isotope pattern
        let mut isotope_groups: HashMap<(usize, usize), Vec<&Peak>> = HashMap::new();
        
        // Try different charge states
        for charge in 1..=4 {
            let mass_diff = 1.0 / charge as f64; // m/z difference for this charge
            
            // Start with each peak as a potential monoisotopic peak
            for (i, peak) in peaks.iter().enumerate() {
                let mut group = vec![peak];
                let base_mz = peak.mz;
                
                // Look for peaks that could be isotopes
                for j in 1..5 { // Look for up to 4 isotope peaks
                    let expected_mz = base_mz + j as f64 * mass_diff;
                    let tolerance = 0.01; // m/z tolerance
                    
                    // Find a matching peak
                    if let Some(isotope_peak) = peaks.iter().find(|p| 
                        (p.mz - expected_mz).abs() < tolerance && p.intensity < peak.intensity
                    ) {
                        group.push(isotope_peak);
                    } else {
                        break;
                    }
                }
                
                // Only save groups with at least 2 peaks (monoisotopic + at least one isotope)
                if group.len() >= 2 {
                    isotope_groups.insert((i, charge), group);
                }
            }
        }
        
        // Convert isotope groups to deconvoluted peaks
        for ((i, charge), group) in isotope_groups {
            if let Some(monoisotopic) = group.first() {
                let monoisotopic_mass = monoisotopic.mz * charge as f64 - charge as f64 * 1.007276; // Proton mass
                
                let peak = DeconvolutedPeak {
                    mz: monoisotopic.mz,
                    intensity: monoisotopic.intensity,
                    monoisotopic_mass,
                    charge,
                    isotope_pattern: group.iter().map(|p| (p.mz, p.intensity)).collect(),
                };
                
                deconvoluted.push(peak);
            }
        }
        
        // Sort by intensity (descending)
        deconvoluted.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap_or(std::cmp::Ordering::Equal));
        
        deconvoluted
    }
    
    /// Parallel feature detection across multiple LC-MS runs
    /// 
    /// Detects consistent features (m/z, RT pairs) across multiple LC-MS runs
    pub fn detect_features_parallel(&self, 
                                   runs: &[Vec<MassSpectrum>], 
                                   mz_tolerance: f64,
                                   rt_tolerance: f64) -> Vec<Feature> {
        // Extract features from each run in parallel
        let all_run_features: Vec<Vec<RawFeature>> = runs.par_iter()
            .map(|run| self.extract_features_from_run(run))
            .collect();
        
        // Merge features across runs
        self.merge_features(all_run_features, mz_tolerance, rt_tolerance)
    }
    
    /// Extract features from a single LC-MS run (used as a subroutine)
    fn extract_features_from_run(&self, run: &[MassSpectrum]) -> Vec<RawFeature> {
        let mut features = Vec::new();
        
        // This is a simplified implementation - a real one would use techniques like
        // 2D peak picking, wavelet transformations, etc.
        
        // For simplicity, assume each spectrum has retention time metadata
        let mut mz_to_chromatogram: HashMap<(usize, usize), Vec<(f64, f64)>> = HashMap::new();
        
        // Group peaks by similar m/z values
        for (scan_idx, spectrum) in run.iter().enumerate() {
            // Get retention time
            let rt = spectrum.metadata().annotations
                .get("retention_time")
                .and_then(|rt_str| rt_str.parse::<f64>().ok())
                .unwrap_or(scan_idx as f64);
            
            // Process peaks
            for peak in spectrum.peaks() {
                // Discretize m/z for binning (using 2 decimal places)
                let mz_bin = (peak.mz * 100.0).round() as usize;
                
                // Add intensity to chromatogram
                mz_to_chromatogram
                    .entry((mz_bin, scan_idx / 10)) // Group by m/z bin and scan region
                    .or_insert_with(Vec::new)
                    .push((rt, peak.intensity));
            }
        }
        
        // Process each chromatogram to find peaks
        for ((mz_bin, _), chromatogram) in mz_to_chromatogram {
            if chromatogram.len() < 3 {
                continue; // Skip noise
            }
            
            // Sort by retention time
            let mut chrom = chromatogram.clone();
            chrom.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            
            // Simple peak detection in the chromatogram
            let mut i = 1;
            while i < chrom.len() - 1 {
                let (rt_prev, intensity_prev) = chrom[i-1];
                let (rt, intensity) = chrom[i];
                let (rt_next, intensity_next) = chrom[i+1];
                
                // Local maximum
                if intensity > intensity_prev && intensity > intensity_next {
                    // Calculate feature properties
                    let mz = mz_bin as f64 / 100.0;
                    
                    // Simple area calculation (trapezoidal)
                    let area = 0.5 * ((rt - rt_prev) * (intensity + intensity_prev) + 
                                    (rt_next - rt) * (intensity + intensity_next));
                    
                    features.push(RawFeature {
                        mz,
                        rt,
                        intensity,
                        area,
                    });
                }
                
                i += 1;
            }
        }
        
        features
    }
    
    /// Merge features from multiple runs (used as a subroutine)
    fn merge_features(&self, 
                     run_features: Vec<Vec<RawFeature>>, 
                     mz_tolerance: f64,
                     rt_tolerance: f64) -> Vec<Feature> {
        let mut feature_clusters: Vec<Vec<&RawFeature>> = Vec::new();
        
        // For each run's features
        for run_idx in 0..run_features.len() {
            let features = &run_features[run_idx];
            
            for feature in features {
                let mut found_cluster = false;
                
                // Try to add to an existing cluster
                for cluster in &mut feature_clusters {
                    if let Some(ref_feature) = cluster.first() {
                        if (feature.mz - ref_feature.mz).abs() <= mz_tolerance &&
                           (feature.rt - ref_feature.rt).abs() <= rt_tolerance {
                            cluster.push(feature);
                            found_cluster = true;
                            break;
                        }
                    }
                }
                
                // Create a new cluster if needed
                if !found_cluster {
                    feature_clusters.push(vec![feature]);
                }
            }
        }
        
        // Convert clusters to merged features
        let merged_features: Vec<Feature> = feature_clusters.iter()
            .filter(|cluster| cluster.len() >= 2) // At least 2 runs must have the feature
            .map(|cluster| {
                // Calculate average properties
                let mz_sum: f64 = cluster.iter().map(|f| f.mz).sum();
                let rt_sum: f64 = cluster.iter().map(|f| f.rt).sum();
                let intensity_sum: f64 = cluster.iter().map(|f| f.intensity).sum();
                let area_sum: f64 = cluster.iter().map(|f| f.area).sum();
                let count = cluster.len() as f64;
                
                Feature {
                    mz: mz_sum / count,
                    rt: rt_sum / count,
                    intensity: intensity_sum / count,
                    area: area_sum / count,
                    occurrence: cluster.len(),
                }
            })
            .collect();
        
        merged_features
    }
    
    /// Parallel spectrum alignment for comparing samples
    /// 
    /// Aligns multiple spectra to a reference for comparison
    pub fn align_spectra_parallel(&self, 
                                 spectra: &[MassSpectrum],
                                 reference: &MassSpectrum,
                                 mz_tolerance: f64) -> Vec<AlignedSpectrum> {
        spectra.par_iter()
            .map(|spectrum| self.align_spectrum(spectrum, reference, mz_tolerance))
            .collect()
    }
    
    /// Align a single spectrum to a reference (used as a subroutine)
    fn align_spectrum(&self, 
                     spectrum: &MassSpectrum,
                     reference: &MassSpectrum,
                     mz_tolerance: f64) -> AlignedSpectrum {
        let mut aligned_peaks = Vec::new();
        let ref_peaks = reference.peaks();
        
        // For each reference peak, find a matching peak in the spectrum
        for ref_peak in ref_peaks {
            let matched_peak = spectrum.peaks_in_range(ref_peak.mz - mz_tolerance, ref_peak.mz + mz_tolerance)
                .iter()
                .max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap_or(std::cmp::Ordering::Equal))
                .cloned();
            
            aligned_peaks.push(AlignedPeak {
                ref_mz: ref_peak.mz,
                ref_intensity: ref_peak.intensity,
                aligned_mz: matched_peak.as_ref().map(|p| p.mz),
                aligned_intensity: matched_peak.as_ref().map(|p| p.intensity),
                mz_shift: matched_peak.as_ref().map(|p| p.mz - ref_peak.mz),
            });
        }
        
        AlignedSpectrum {
            id: UnitId::new(format!("aligned_{}", spectrum.id())),
            peaks: aligned_peaks,
            similarity_score: calculate_similarity(&aligned_peaks),
        }
    }
}

/// Calculate a similarity score between aligned spectra
fn calculate_similarity(aligned_peaks: &[AlignedPeak]) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_ref = 0.0;
    let mut norm_aligned = 0.0;
    
    for peak in aligned_peaks {
        let ref_intensity = peak.ref_intensity;
        norm_ref += ref_intensity * ref_intensity;
        
        if let Some(aligned_intensity) = peak.aligned_intensity {
            dot_product += ref_intensity * aligned_intensity;
            norm_aligned += aligned_intensity * aligned_intensity;
        }
    }
    
    if norm_ref > 0.0 && norm_aligned > 0.0 {
        dot_product / (norm_ref.sqrt() * norm_aligned.sqrt())
    } else {
        0.0
    }
}

/// A deconvoluted peak with charge state and monoisotopic mass
#[derive(Debug, Clone)]
pub struct DeconvolutedPeak {
    /// Observed m/z
    pub mz: f64,
    /// Observed intensity
    pub intensity: f64,
    /// Calculated monoisotopic mass
    pub monoisotopic_mass: f64,
    /// Determined charge state
    pub charge: usize,
    /// Observed isotope pattern
    pub isotope_pattern: Vec<(f64, f64)>,
}

/// A raw feature detected in a single LC-MS run
#[derive(Debug, Clone)]
struct RawFeature {
    /// m/z value
    pub mz: f64,
    /// Retention time
    pub rt: f64,
    /// Intensity
    pub intensity: f64,
    /// Integrated area
    pub area: f64,
}

/// A feature detected across multiple LC-MS runs
#[derive(Debug, Clone)]
pub struct Feature {
    /// Average m/z value
    pub mz: f64,
    /// Average retention time
    pub rt: f64,
    /// Average intensity
    pub intensity: f64,
    /// Average integrated area
    pub area: f64,
    /// Number of runs where this feature was detected
    pub occurrence: usize,
}

/// A peak aligned between a reference and sample spectrum
#[derive(Debug, Clone)]
pub struct AlignedPeak {
    /// m/z in reference spectrum
    pub ref_mz: f64,
    /// Intensity in reference spectrum
    pub ref_intensity: f64,
    /// m/z in aligned spectrum (if found)
    pub aligned_mz: Option<f64>,
    /// Intensity in aligned spectrum (if found)
    pub aligned_intensity: Option<f64>,
    /// m/z shift between reference and aligned peak (if found)
    pub mz_shift: Option<f64>,
}

/// A spectrum aligned to a reference
#[derive(Debug, Clone)]
pub struct AlignedSpectrum {
    /// Unique identifier
    pub id: UnitId,
    /// Aligned peaks
    pub peaks: Vec<AlignedPeak>,
    /// Overall similarity score to reference
    pub similarity_score: f64,
} 