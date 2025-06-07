//! Spectral database module
//!
//! This module provides functionality for storing and querying spectral data.

use std::collections::HashMap;

/// Spectral database
#[derive(Debug, Clone)]
pub struct SpectralDatabase {
    /// Stored spectra
    spectra: HashMap<String, SpectrumEntry>,
    /// Database configuration
    config: DatabaseConfig,
    /// Search index
    search_index: SearchIndex,
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// NIST database enabled
    pub nist_enabled: bool,
    /// Local database path
    pub local_database: Option<String>,
    /// Online search enabled
    pub online_search: bool,
    /// Cache settings
    pub cache_settings: CacheSettings,
}

/// Cache settings
#[derive(Debug, Clone)]
pub struct CacheSettings {
    /// Enable caching
    pub enable_cache: bool,
    /// Cache size (MB)
    pub cache_size_mb: usize,
    /// Cache TTL (seconds)
    pub cache_ttl: u64,
}

/// Spectrum entry in database
#[derive(Debug, Clone)]
pub struct SpectrumEntry {
    /// Spectrum ID
    pub id: String,
    /// Compound name
    pub compound_name: String,
    /// Molecular formula
    pub molecular_formula: Option<String>,
    /// CAS number
    pub cas_number: Option<String>,
    /// Spectrum type
    pub spectrum_type: SpectrumType,
    /// Spectral data
    pub data: SpectralData,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Quality score
    pub quality_score: f64,
}

/// Types of spectra
#[derive(Debug, Clone)]
pub enum SpectrumType {
    /// Mass spectrum
    MassSpectrum,
    /// NMR spectrum
    NMR(String), // nucleus type
    /// IR spectrum
    IR,
    /// UV-Vis spectrum
    UVVis,
    /// Raman spectrum
    Raman,
}

/// Spectral data
#[derive(Debug, Clone)]
pub struct SpectralData {
    /// X-axis data
    pub x_data: Vec<f64>,
    /// Y-axis data
    pub y_data: Vec<f64>,
    /// Peak list
    pub peaks: Vec<DatabasePeak>,
}

/// Peak in database
#[derive(Debug, Clone)]
pub struct DatabasePeak {
    /// Position (m/z, ppm, wavenumber, wavelength)
    pub position: f64,
    /// Intensity
    pub intensity: f64,
    /// Assignment
    pub assignment: Option<String>,
}

/// Search index for fast searching
#[derive(Debug, Clone)]
pub struct SearchIndex {
    /// Molecular weight index
    molecular_weight_index: HashMap<u32, Vec<String>>,
    /// Formula index
    formula_index: HashMap<String, Vec<String>>,
    /// Peak index
    peak_index: HashMap<u32, Vec<String>>,
}

/// Search query
#[derive(Debug, Clone)]
pub struct SearchQuery {
    /// Spectrum type filter
    pub spectrum_type: Option<SpectrumType>,
    /// Molecular weight range
    pub molecular_weight_range: Option<(f64, f64)>,
    /// Peak list for matching
    pub peaks: Vec<f64>,
    /// Peak tolerance
    pub peak_tolerance: f64,
    /// Minimum similarity score
    pub min_similarity: f64,
    /// Maximum results
    pub max_results: usize,
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Matching spectra
    pub matches: Vec<SpectrumMatch>,
    /// Total number of matches
    pub total_matches: usize,
    /// Search time (ms)
    pub search_time: u64,
}

/// Spectrum match
#[derive(Debug, Clone)]
pub struct SpectrumMatch {
    /// Spectrum entry
    pub spectrum: SpectrumEntry,
    /// Similarity score
    pub similarity_score: f64,
    /// Matching peaks
    pub matching_peaks: Vec<PeakMatch>,
}

/// Peak match
#[derive(Debug, Clone)]
pub struct PeakMatch {
    /// Query peak
    pub query_peak: f64,
    /// Database peak
    pub database_peak: f64,
    /// Mass difference
    pub mass_difference: f64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            nist_enabled: false,
            local_database: None,
            online_search: false,
            cache_settings: CacheSettings {
                enable_cache: true,
                cache_size_mb: 100,
                cache_ttl: 3600,
            },
        }
    }
}

impl SpectralDatabase {
    /// Create new spectral database
    pub fn new(config: DatabaseConfig) -> Self {
        Self {
            spectra: HashMap::new(),
            config,
            search_index: SearchIndex::new(),
        }
    }

    /// Add spectrum to database
    pub fn add_spectrum(&mut self, spectrum: SpectrumEntry) {
        let id = spectrum.id.clone();
        self.search_index.index_spectrum(&spectrum);
        self.spectra.insert(id, spectrum);
    }

    /// Get spectrum by ID
    pub fn get_spectrum(&self, id: &str) -> Option<&SpectrumEntry> {
        self.spectra.get(id)
    }

    /// Search database
    pub fn search(&self, query: &SearchQuery) -> SearchResult {
        let start_time = std::time::Instant::now();
        let mut matches = Vec::new();

        for spectrum in self.spectra.values() {
            if self.matches_query(spectrum, query) {
                let similarity = self.calculate_similarity(spectrum, query);
                if similarity >= query.min_similarity {
                    let matching_peaks = self.find_matching_peaks(spectrum, query);
                    matches.push(SpectrumMatch {
                        spectrum: spectrum.clone(),
                        similarity_score: similarity,
                        matching_peaks,
                    });
                }
            }
        }

        // Sort by similarity score
        matches.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        
        // Limit results
        if matches.len() > query.max_results {
            matches.truncate(query.max_results);
        }

        let total_matches = matches.len();
        let search_time = start_time.elapsed().as_millis() as u64;

        SearchResult {
            matches,
            total_matches,
            search_time,
        }
    }

    /// Check if spectrum matches query
    fn matches_query(&self, spectrum: &SpectrumEntry, query: &SearchQuery) -> bool {
        // Check spectrum type
        if let Some(ref query_type) = query.spectrum_type {
            if !self.spectrum_types_match(&spectrum.spectrum_type, query_type) {
                return false;
            }
        }

        // Check molecular weight range
        if let Some((min_mw, max_mw)) = query.molecular_weight_range {
            let spectrum_mw = self.estimate_molecular_weight(spectrum);
            if spectrum_mw < min_mw || spectrum_mw > max_mw {
                return false;
            }
        }

        true
    }

    /// Check if spectrum types match
    fn spectrum_types_match(&self, spectrum_type: &SpectrumType, query_type: &SpectrumType) -> bool {
        match (spectrum_type, query_type) {
            (SpectrumType::MassSpectrum, SpectrumType::MassSpectrum) => true,
            (SpectrumType::NMR(n1), SpectrumType::NMR(n2)) => n1 == n2,
            (SpectrumType::IR, SpectrumType::IR) => true,
            (SpectrumType::UVVis, SpectrumType::UVVis) => true,
            (SpectrumType::Raman, SpectrumType::Raman) => true,
            _ => false,
        }
    }

    /// Estimate molecular weight from spectrum
    fn estimate_molecular_weight(&self, spectrum: &SpectrumEntry) -> f64 {
        match spectrum.spectrum_type {
            SpectrumType::MassSpectrum => {
                // Find highest m/z peak
                spectrum.data.peaks.iter()
                    .map(|p| p.position)
                    .fold(0.0f64, f64::max)
            }
            _ => 0.0, // Can't estimate from other spectrum types
        }
    }

    /// Calculate similarity between spectrum and query
    fn calculate_similarity(&self, spectrum: &SpectrumEntry, query: &SearchQuery) -> f64 {
        if query.peaks.is_empty() {
            return 0.5; // Default similarity
        }

        let matching_peaks = self.find_matching_peaks(spectrum, query);
        let match_ratio = matching_peaks.len() as f64 / query.peaks.len() as f64;
        
        // Simple similarity calculation
        match_ratio
    }

    /// Find matching peaks between spectrum and query
    fn find_matching_peaks(&self, spectrum: &SpectrumEntry, query: &SearchQuery) -> Vec<PeakMatch> {
        let mut matches = Vec::new();

        for &query_peak in &query.peaks {
            for db_peak in &spectrum.data.peaks {
                let mass_diff = (query_peak - db_peak.position).abs();
                if mass_diff <= query.peak_tolerance {
                    matches.push(PeakMatch {
                        query_peak,
                        database_peak: db_peak.position,
                        mass_difference: mass_diff,
                    });
                    break; // Take first match
                }
            }
        }

        matches
    }
}

impl SearchIndex {
    /// Create new search index
    pub fn new() -> Self {
        Self {
            molecular_weight_index: HashMap::new(),
            formula_index: HashMap::new(),
            peak_index: HashMap::new(),
        }
    }

    /// Index a spectrum
    pub fn index_spectrum(&mut self, spectrum: &SpectrumEntry) {
        let id = spectrum.id.clone();

        // Index by molecular weight
        let mw = self.estimate_molecular_weight(spectrum) as u32;
        self.molecular_weight_index.entry(mw).or_insert_with(Vec::new).push(id.clone());

        // Index by formula
        if let Some(ref formula) = spectrum.molecular_formula {
            self.formula_index.entry(formula.clone()).or_insert_with(Vec::new).push(id.clone());
        }

        // Index by peaks
        for peak in &spectrum.data.peaks {
            let peak_key = (peak.position as u32);
            self.peak_index.entry(peak_key).or_insert_with(Vec::new).push(id.clone());
        }
    }

    /// Estimate molecular weight from spectrum
    fn estimate_molecular_weight(&self, spectrum: &SpectrumEntry) -> f64 {
        match spectrum.spectrum_type {
            SpectrumType::MassSpectrum => {
                spectrum.data.peaks.iter()
                    .map(|p| p.position)
                    .fold(0.0f64, f64::max)
            }
            _ => 0.0,
        }
    }
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            spectrum_type: None,
            molecular_weight_range: None,
            peaks: Vec::new(),
            peak_tolerance: 0.1,
            min_similarity: 0.5,
            max_results: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_database() {
        let mut db = SpectralDatabase::new(DatabaseConfig::default());
        
        let spectrum = SpectrumEntry {
            id: "test_spectrum".to_string(),
            compound_name: "Test Compound".to_string(),
            molecular_formula: Some("C6H6".to_string()),
            cas_number: None,
            spectrum_type: SpectrumType::MassSpectrum,
            data: SpectralData {
                x_data: vec![78.0, 77.0, 79.0],
                y_data: vec![100.0, 50.0, 20.0],
                peaks: vec![
                    DatabasePeak { position: 78.0, intensity: 100.0, assignment: Some("M+".to_string()) },
                ],
            },
            metadata: HashMap::new(),
            quality_score: 0.9,
        };

        db.add_spectrum(spectrum);

        let query = SearchQuery {
            peaks: vec![78.0],
            ..Default::default()
        };

        let results = db.search(&query);
        assert_eq!(results.matches.len(), 1);
    }
} 