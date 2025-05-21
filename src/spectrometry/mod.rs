//! Mass Spectrometry extension for Kwasa-Kwasa
//! 
//! This module provides types and operations for working with mass spectrometry data
//! using the same powerful abstractions as text processing.

use std::fmt::Debug;
use std::{collections::HashMap, marker::PhantomData};

// Add the high-throughput module
pub mod high_throughput;

/// Re-exports from this module
pub mod prelude {
    pub use super::{
        MassSpectrum, Peak, SpectrumMetadata, SpectrumBoundaryDetector, SpectrumOperations,
        // Add exports for high-throughput components
        high_throughput::{
            HighThroughputSpectrometry, DeconvolutedPeak, Feature, AlignedSpectrum, AlignedPeak
        },
    };
}

/// Metadata for mass spectrum
#[derive(Debug, Clone)]
pub struct SpectrumMetadata {
    /// Source of the spectrum (e.g., instrument, experiment)
    pub source: Option<String>,
    /// Ionization method
    pub ionization_method: Option<String>,
    /// Resolution
    pub resolution: Option<f64>,
    /// Additional key-value annotations
    pub annotations: HashMap<String, String>,
}

/// Position in m/z space
#[derive(Debug, Clone, PartialEq)]
pub struct MzRange {
    /// Starting m/z
    pub start: f64,
    /// Ending m/z
    pub end: f64,
}

/// Unique identifier for spectrometry units
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct UnitId(String);

impl UnitId {
    /// Create a new unit ID
    pub fn new(id: impl Into<String>) -> Self {
        UnitId(id.into())
    }
}

impl std::fmt::Display for UnitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Universal trait for all units of analysis
pub trait Unit: Clone + Debug {
    /// The raw content of this unit
    fn content(&self) -> &[u8];
    
    /// Human-readable representation
    fn display(&self) -> String;
    
    /// Metadata associated with this unit
    fn metadata(&self) -> &dyn std::any::Any;
    
    /// Unique identifier for this unit
    fn id(&self) -> &UnitId;
}

/// Configuration for boundary detection
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    /// Minimum peak intensity
    pub min_intensity: f64,
    /// Minimum signal-to-noise ratio
    pub min_snr: Option<f64>,
    /// Whether to include partial units at the ends
    pub include_partial: bool,
    /// Pattern to use for splitting
    pub pattern: Option<String>,
}

/// Generic trait for boundary detection in any domain
pub trait BoundaryDetector {
    type UnitType: Unit;
    
    /// Detect boundaries in the given content
    fn detect_boundaries(&self, content: &[u8]) -> Vec<Self::UnitType>;
    
    /// Configuration for the detection algorithm
    fn configuration(&self) -> &BoundaryConfig;
}

/// Generic operations on units
pub trait UnitOperations<T: Unit> {
    /// Split a unit into smaller units based on a pattern
    fn divide(&self, unit: &T, pattern: &str) -> Vec<T>;
    
    /// Combine two units with appropriate transitions
    fn multiply(&self, left: &T, right: &T) -> T;
    
    /// Concatenate units with intelligent joining
    fn add(&self, left: &T, right: &T) -> T;
    
    /// Remove elements from a unit
    fn subtract(&self, source: &T, to_remove: &T) -> T;
}

//------------------------------------------------------------------------------
// Mass Spectrum
//------------------------------------------------------------------------------

/// A single peak in a mass spectrum
#[derive(Debug, Clone)]
pub struct Peak {
    /// m/z value
    pub mz: f64,
    /// Intensity value
    pub intensity: f64,
    /// Signal-to-noise ratio
    pub snr: Option<f64>,
    /// Peak annotations
    pub annotations: HashMap<String, String>,
}

impl Peak {
    /// Create a new peak
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self {
            mz,
            intensity,
            snr: None,
            annotations: HashMap::new(),
        }
    }
    
    /// Create a new peak with SNR
    pub fn with_snr(mz: f64, intensity: f64, snr: f64) -> Self {
        Self {
            mz,
            intensity,
            snr: Some(snr),
            annotations: HashMap::new(),
        }
    }
    
    /// Add an annotation to this peak
    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }
}

/// A mass spectrum
#[derive(Debug, Clone)]
pub struct MassSpectrum {
    /// The raw spectral data as bytes
    content: Vec<u8>,
    /// Peaks in this spectrum
    peaks: Vec<Peak>,
    /// Metadata for this spectrum
    metadata: SpectrumMetadata,
    /// Unique identifier
    id: UnitId,
}

impl MassSpectrum {
    /// Create a new mass spectrum from raw bytes
    pub fn new(content: impl Into<Vec<u8>>, id: impl Into<String>) -> Self {
        let content = content.into();
        Self {
            content,
            peaks: Vec::new(),
            metadata: SpectrumMetadata {
                source: None,
                ionization_method: None,
                resolution: None,
                annotations: HashMap::new(),
            },
            id: UnitId::new(id),
        }
    }
    
    /// Create a mass spectrum from m/z and intensity vectors
    pub fn from_numeric_data(
        mz_values: Vec<f64>, 
        intensities: Vec<f64>, 
        id: impl Into<String>
    ) -> Self {
        let mut peaks = Vec::with_capacity(mz_values.len());
        let mut content = Vec::new();
        
        // Create peaks from paired vectors
        for (i, (mz, intensity)) in mz_values.iter().zip(intensities.iter()).enumerate() {
            let peak = Peak::new(*mz, *intensity);
            peaks.push(peak);
            
            // Create a simple textual representation for the content field
            let peak_text = format!("{:.4},{:.1}\n", mz, intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        Self {
            content,
            peaks,
            metadata: SpectrumMetadata {
                source: None,
                ionization_method: None,
                resolution: None,
                annotations: HashMap::new(),
            },
            id: UnitId::new(id),
        }
    }
    
    /// Get the peaks in this spectrum
    pub fn peaks(&self) -> &[Peak] {
        &self.peaks
    }
    
    /// Find the base peak (most intense)
    pub fn base_peak(&self) -> Option<&Peak> {
        self.peaks.iter().max_by(|a, b| a.intensity.partial_cmp(&b.intensity).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Find peaks in a specific m/z range
    pub fn peaks_in_range(&self, min_mz: f64, max_mz: f64) -> Vec<&Peak> {
        self.peaks.iter()
            .filter(|p| p.mz >= min_mz && p.mz <= max_mz)
            .collect()
    }
    
    /// Filter peaks by minimum intensity
    pub fn filter_by_intensity(&self, min_intensity: f64) -> Self {
        let filtered_peaks: Vec<_> = self.peaks.iter()
            .filter(|p| p.intensity >= min_intensity)
            .cloned()
            .collect();
        
        // Create new content from filtered peaks
        let mut content = Vec::new();
        for peak in &filtered_peaks {
            let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        Self {
            content,
            peaks: filtered_peaks,
            metadata: self.metadata.clone(),
            id: UnitId::new(format!("{}_filtered", self.id.0)),
        }
    }
    
    /// Normalize peak intensities relative to base peak
    pub fn normalize(&self) -> Self {
        if let Some(base_peak) = self.base_peak() {
            let base_intensity = base_peak.intensity;
            
            let normalized_peaks: Vec<_> = self.peaks.iter()
                .map(|p| Peak {
                    mz: p.mz,
                    intensity: (p.intensity / base_intensity) * 100.0,
                    snr: p.snr,
                    annotations: p.annotations.clone(),
                })
                .collect();
            
            // Create new content from normalized peaks
            let mut content = Vec::new();
            for peak in &normalized_peaks {
                let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
                content.extend_from_slice(peak_text.as_bytes());
            }
            
            Self {
                content,
                peaks: normalized_peaks,
                metadata: self.metadata.clone(),
                id: UnitId::new(format!("{}_normalized", self.id.0)),
            }
        } else {
            self.clone()
        }
    }
    
    /// Extract a subset of the spectrum within a m/z range
    pub fn extract_range(&self, min_mz: f64, max_mz: f64) -> Self {
        let filtered_peaks: Vec<_> = self.peaks.iter()
            .filter(|p| p.mz >= min_mz && p.mz <= max_mz)
            .cloned()
            .collect();
        
        // Create new content from filtered peaks
        let mut content = Vec::new();
        for peak in &filtered_peaks {
            let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        Self {
            content,
            peaks: filtered_peaks,
            metadata: self.metadata.clone(),
            id: UnitId::new(format!("{}_{}_to_{}", self.id.0, min_mz, max_mz)),
        }
    }
    
    /// Set metadata for this spectrum
    pub fn set_metadata(&mut self, metadata: SpectrumMetadata) {
        self.metadata = metadata;
    }
    
    /// Get metadata for this spectrum
    pub fn metadata(&self) -> &SpectrumMetadata {
        &self.metadata
    }
    
    /// Add a peak to the spectrum
    pub fn add_peak(&mut self, peak: Peak) {
        self.peaks.push(peak);
    }
}

impl Unit for MassSpectrum {
    fn content(&self) -> &[u8] {
        &self.content
    }
    
    fn display(&self) -> String {
        if let Some(base_peak) = self.base_peak() {
            format!("Spectrum: {} peaks, base peak m/z: {:.4}", 
                self.peaks.len(), base_peak.mz)
        } else {
            format!("Spectrum: {} peaks", self.peaks.len())
        }
    }
    
    fn metadata(&self) -> &dyn std::any::Any {
        &self.metadata
    }
    
    fn id(&self) -> &UnitId {
        &self.id
    }
}

//------------------------------------------------------------------------------
// Spectrum Boundary Detector
//------------------------------------------------------------------------------

/// Detector for mass spectrum boundaries
#[derive(Debug)]
pub struct SpectrumBoundaryDetector {
    /// Configuration for boundary detection
    config: BoundaryConfig,
    /// Type of boundaries to detect
    boundary_type: SpectrumBoundaryType,
}

/// Types of spectrum boundaries to detect
#[derive(Debug, Clone)]
pub enum SpectrumBoundaryType {
    /// Individual peak boundaries
    Peak,
    /// m/z ranges
    MzRange(f64), // width of each range
    /// Intensity clusters
    IntensityCluster(f64), // clustering threshold
    /// Feature boundaries
    Feature,
}

impl SpectrumBoundaryDetector {
    /// Create a new spectrum boundary detector
    pub fn new(boundary_type: SpectrumBoundaryType, config: BoundaryConfig) -> Self {
        Self {
            config,
            boundary_type,
        }
    }
}

impl BoundaryDetector for SpectrumBoundaryDetector {
    type UnitType = MassSpectrum;
    
    fn detect_boundaries(&self, content: &[u8]) -> Vec<Self::UnitType> {
        // Parse content to get the spectrum data
        // This is a simplified implementation - in reality, this would parse
        // the actual file format (mzXML, mzML, etc.)
        
        // For demonstration, we'll just create a dummy spectrum
        let dummy_spectrum = MassSpectrum::new(content.to_vec(), "parsed_spectrum");
        
        // Implementation would depend on the boundary type
        match self.boundary_type {
            SpectrumBoundaryType::Peak => {
                // Return each peak as a separate unit
                vec![dummy_spectrum]
            },
            SpectrumBoundaryType::MzRange(width) => {
                // Divide the spectrum into m/z ranges
                vec![dummy_spectrum]
            },
            SpectrumBoundaryType::IntensityCluster(threshold) => {
                // Cluster peaks by intensity
                vec![dummy_spectrum]
            },
            SpectrumBoundaryType::Feature => {
                // Detect features (isotope patterns, etc.)
                vec![dummy_spectrum]
            },
        }
    }
    
    fn configuration(&self) -> &BoundaryConfig {
        &self.config
    }
}

//------------------------------------------------------------------------------
// Spectrum Operations
//------------------------------------------------------------------------------

/// Operations for mass spectra
pub struct SpectrumOperations;

impl UnitOperations<MassSpectrum> for SpectrumOperations {
    fn divide(&self, unit: &MassSpectrum, pattern: &str) -> Vec<MassSpectrum> {
        // Different division strategies based on pattern
        match pattern {
            "peak" => {
                // Each peak becomes a separate spectrum
                unit.peaks().iter().enumerate().map(|(i, peak)| {
                    let content = format!("{:.4},{:.1}\n", peak.mz, peak.intensity).into_bytes();
                    let mut spectrum = MassSpectrum::new(content, format!("{}_peak_{}", unit.id().0, i));
                    spectrum.peaks = vec![peak.clone()];
                    spectrum
                }).collect()
            },
            "mz_range" => {
                // Divide into m/z ranges
                // Example: divide by 100 m/z ranges
                if let (Some(first), Some(last)) = (unit.peaks.first(), unit.peaks.last()) {
                    let min_mz = first.mz.min(last.mz);
                    let max_mz = first.mz.max(last.mz);
                    let range_size = 100.0;
                    
                    let mut ranges = Vec::new();
                    let mut current_mz = min_mz;
                    
                    while current_mz < max_mz {
                        let range_end = current_mz + range_size;
                        ranges.push(unit.extract_range(current_mz, range_end));
                        current_mz = range_end;
                    }
                    
                    ranges
                } else {
                    vec![unit.clone()]
                }
            },
            "intensity" => {
                // Divide by intensity percentiles
                // Example: divide into low, medium, high intensity
                if let Some(base_peak) = unit.base_peak() {
                    let max_intensity = base_peak.intensity;
                    let low = unit.filter_by_intensity(0.0).filter_by_intensity(max_intensity * 0.33);
                    let medium = unit.filter_by_intensity(max_intensity * 0.33).filter_by_intensity(max_intensity * 0.66);
                    let high = unit.filter_by_intensity(max_intensity * 0.66);
                    
                    vec![
                        MassSpectrum {
                            content: low.content,
                            peaks: low.peaks,
                            metadata: low.metadata,
                            id: UnitId::new(format!("{}_low", unit.id().0)),
                        },
                        MassSpectrum {
                            content: medium.content,
                            peaks: medium.peaks,
                            metadata: medium.metadata,
                            id: UnitId::new(format!("{}_medium", unit.id().0)),
                        },
                        MassSpectrum {
                            content: high.content,
                            peaks: high.peaks,
                            metadata: high.metadata,
                            id: UnitId::new(format!("{}_high", unit.id().0)),
                        },
                    ]
                } else {
                    vec![unit.clone()]
                }
            },
            _ => vec![unit.clone()],
        }
    }
    
    fn multiply(&self, left: &MassSpectrum, right: &MassSpectrum) -> MassSpectrum {
        // In spectrum context, multiplication could be interpreted as spectral convolution
        // or peak correlation analysis
        
        // This is a simplified implementation - in reality, this would
        // perform proper spectral convolution
        
        // For demonstration, we'll just join the peaks
        let mut combined_peaks = left.peaks.clone();
        combined_peaks.extend(right.peaks.clone());
        
        // Sort by m/z
        combined_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create new content
        let mut content = Vec::new();
        for peak in &combined_peaks {
            let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        MassSpectrum {
            content,
            peaks: combined_peaks,
            metadata: SpectrumMetadata {
                source: Some(format!("Convolution of {} and {}", left.id().0, right.id().0)),
                ionization_method: left.metadata.ionization_method.clone(),
                resolution: left.metadata.resolution,
                annotations: left.metadata.annotations.clone(),
            },
            id: UnitId::new(format!("{}_x_{}", left.id().0, right.id().0)),
        }
    }
    
    fn add(&self, left: &MassSpectrum, right: &MassSpectrum) -> MassSpectrum {
        // Simple addition of spectra (summing intensities at matching m/z values)
        
        // Create a map of m/z values to combined intensities
        let mut mz_map: HashMap<u64, f64> = HashMap::new();
        
        // Add left peaks
        for peak in &left.peaks {
            // Use discretized m/z as key (multiply by 10000 and round to integer)
            let key = (peak.mz * 10000.0).round() as u64;
            *mz_map.entry(key).or_insert(0.0) += peak.intensity;
        }
        
        // Add right peaks
        for peak in &right.peaks {
            let key = (peak.mz * 10000.0).round() as u64;
            *mz_map.entry(key).or_insert(0.0) += peak.intensity;
        }
        
        // Convert back to peaks
        let mut combined_peaks = Vec::with_capacity(mz_map.len());
        for (key, intensity) in mz_map {
            let mz = key as f64 / 10000.0;
            combined_peaks.push(Peak::new(mz, intensity));
        }
        
        // Sort by m/z
        combined_peaks.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap_or(std::cmp::Ordering::Equal));
        
        // Create new content
        let mut content = Vec::new();
        for peak in &combined_peaks {
            let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        MassSpectrum {
            content,
            peaks: combined_peaks,
            metadata: SpectrumMetadata {
                source: Some(format!("Sum of {} and {}", left.id().0, right.id().0)),
                ionization_method: left.metadata.ionization_method.clone(),
                resolution: left.metadata.resolution,
                annotations: left.metadata.annotations.clone(),
            },
            id: UnitId::new(format!("{}_{}", left.id().0, right.id().0)),
        }
    }
    
    fn subtract(&self, source: &MassSpectrum, to_remove: &MassSpectrum) -> MassSpectrum {
        // Subtract spectrum (remove intensities at matching m/z values)
        
        // Create a map of m/z values to intensities for the spectrum to remove
        let mut remove_map: HashMap<u64, f64> = HashMap::new();
        
        // Add peaks to remove
        for peak in &to_remove.peaks {
            let key = (peak.mz * 10000.0).round() as u64;
            *remove_map.entry(key).or_insert(0.0) += peak.intensity;
        }
        
        // Create new peaks by subtracting
        let mut result_peaks = Vec::with_capacity(source.peaks.len());
        for peak in &source.peaks {
            let key = (peak.mz * 10000.0).round() as u64;
            if let Some(remove_intensity) = remove_map.get(&key) {
                let new_intensity = (peak.intensity - remove_intensity).max(0.0);
                if new_intensity > 0.0 {
                    result_peaks.push(Peak::new(peak.mz, new_intensity));
                }
            } else {
                result_peaks.push(peak.clone());
            }
        }
        
        // Create new content
        let mut content = Vec::new();
        for peak in &result_peaks {
            let peak_text = format!("{:.4},{:.1}\n", peak.mz, peak.intensity);
            content.extend_from_slice(peak_text.as_bytes());
        }
        
        MassSpectrum {
            content,
            peaks: result_peaks,
            metadata: SpectrumMetadata {
                source: Some(format!("{} minus {}", source.id().0, to_remove.id().0)),
                ionization_method: source.metadata.ionization_method.clone(),
                resolution: source.metadata.resolution,
                annotations: source.metadata.annotations.clone(),
            },
            id: UnitId::new(format!("{}_minus_{}", source.id().0, to_remove.id().0)),
        }
    }
} 