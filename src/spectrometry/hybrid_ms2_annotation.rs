//! Hybrid MS2 Fragment Annotation Module
//!
//! This module implements MS2 fragment annotation using the hybrid programming paradigm,
//! probabilistic reasoning, and uncertainty handling for complex spectral interpretation.

use std::collections::HashMap;
use crate::turbulance::{
    probabilistic::{TextPoint, ResolutionResult, ResolutionStrategy, ResolutionContext, ResolutionFunction},
    hybrid_processing::{HybridProcessor, ProbabilisticFloor, HybridConfig, HybridResult, ProcessingMode},
    streaming::{TextStream, StreamConfig},
    interpreter::Value,
    Result, TurbulanceError,
};
use super::{MassSpectrum, Peak, Fragment, FragmentType};

/// Hybrid MS2 fragment annotator using probabilistic methods
#[derive(Debug)]
pub struct HybridMS2Annotator {
    /// Hybrid processor for adaptive computation
    hybrid_processor: HybridProcessor,
    
    /// Spectral library for fragment matching
    spectral_library: SpectralLibrary,
    
    /// Fragmentation rules engine
    fragmentation_engine: FragmentationEngine,
    
    /// Configuration
    config: MS2AnnotationConfig,
}

/// Configuration for MS2 annotation
#[derive(Debug, Clone)]
pub struct MS2AnnotationConfig {
    /// Mass tolerance for fragment matching (Da)
    pub mass_tolerance: f64,
    
    /// Minimum intensity threshold
    pub min_intensity_threshold: f64,
    
    /// Confidence threshold for probabilistic mode switching
    pub probabilistic_threshold: f64,
    
    /// Maximum iterations for iterative annotation
    pub max_annotation_iterations: u32,
    
    /// Enable neural network predictions
    pub enable_neural_predictions: bool,
    
    /// Enable retention time predictions
    pub enable_rt_predictions: bool,
}

/// MS2 annotation result with uncertainty quantification
#[derive(Debug, Clone)]
pub struct MS2AnnotationResult {
    /// Spectrum ID
    pub spectrum_id: String,
    
    /// Annotated fragments with confidence
    pub annotated_fragments: Vec<AnnotatedFragment>,
    
    /// Peptide sequence predictions
    pub peptide_predictions: Vec<PeptidePrediction>,
    
    /// Overall annotation confidence
    pub overall_confidence: f64,
    
    /// Processing metadata
    pub processing_metadata: AnnotationMetadata,
}

/// Fragment annotation with probabilistic confidence
#[derive(Debug, Clone)]
pub struct AnnotatedFragment {
    /// Original peak
    pub peak: Peak,
    
    /// Fragment assignments with probabilities
    pub assignments: Vec<FragmentAssignment>,
    
    /// Annotation confidence
    pub confidence: f64,
    
    /// Fragmentation pathway
    pub pathway: Option<String>,
    
    /// Supporting evidence
    pub evidence: Vec<AnnotationEvidence>,
}

/// Probabilistic fragment assignment
#[derive(Debug, Clone)]
pub struct FragmentAssignment {
    /// Fragment ion type (b, y, a, x, c, z, etc.)
    pub ion_type: String,
    
    /// Position in peptide sequence
    pub position: Option<usize>,
    
    /// Theoretical m/z
    pub theoretical_mz: f64,
    
    /// Mass error (ppm)
    pub mass_error_ppm: f64,
    
    /// Assignment probability
    pub probability: f64,
    
    /// Neutral losses
    pub neutral_losses: Vec<NeutralLoss>,
    
    /// Charge state
    pub charge_state: i32,
}

/// Peptide sequence prediction with uncertainty
#[derive(Debug, Clone)]
pub struct PeptidePrediction {
    /// Predicted sequence
    pub sequence: String,
    
    /// Prediction confidence
    pub confidence: f64,
    
    /// Supporting fragments
    pub supporting_fragments: Vec<String>,
    
    /// Alternative sequences
    pub alternatives: Vec<(String, f64)>,
    
    /// Modification predictions
    pub modifications: Vec<ModificationPrediction>,
}

/// Neutral loss information
#[derive(Debug, Clone)]
pub struct NeutralLoss {
    /// Loss mass
    pub mass: f64,
    
    /// Loss type (H2O, NH3, etc.)
    pub loss_type: String,
    
    /// Probability of this loss
    pub probability: f64,
}

/// Modification prediction
#[derive(Debug, Clone)]
pub struct ModificationPrediction {
    /// Modification type
    pub modification_type: String,
    
    /// Position in sequence
    pub position: usize,
    
    /// Mass delta
    pub mass_delta: f64,
    
    /// Confidence
    pub confidence: f64,
}

/// Evidence supporting fragment annotation
#[derive(Debug, Clone)]
pub enum AnnotationEvidence {
    /// Library match evidence
    LibraryMatch {
        library_id: String,
        score: f64,
        similarity: f64,
    },
    
    /// Theoretical prediction evidence
    TheoreticalPrediction {
        model: String,
        score: f64,
        parameters: HashMap<String, f64>,
    },
    
    /// Fragmentation rule evidence
    FragmentationRule {
        rule_id: String,
        applicability: f64,
        confidence: f64,
    },
    
    /// Neural network prediction
    NeuralPrediction {
        model_id: String,
        confidence: f64,
        features: HashMap<String, f64>,
    },
}

/// Annotation processing metadata
#[derive(Debug, Clone)]
pub struct AnnotationMetadata {
    /// Processing time (ms)
    pub processing_time_ms: u64,
    
    /// Number of iterations performed
    pub iterations: u32,
    
    /// Processing mode used
    pub processing_mode: String,
    
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    
    /// Uncertainty metrics
    pub uncertainty_metrics: HashMap<String, f64>,
}

/// Spectral library for fragment matching
#[derive(Debug)]
pub struct SpectralLibrary {
    /// Library entries
    entries: HashMap<String, LibraryEntry>,
    
    /// Index for fast searching
    mass_index: Vec<(f64, String)>,
}

/// Library entry
#[derive(Debug, Clone)]
pub struct LibraryEntry {
    /// Peptide sequence
    pub sequence: String,
    
    /// Theoretical fragments
    pub fragments: Vec<TheoreticalFragment>,
    
    /// Retention time
    pub retention_time: Option<f64>,
    
    /// Charge state
    pub charge_state: i32,
    
    /// Confidence score
    pub confidence: f64,
}

/// Theoretical fragment
#[derive(Debug, Clone)]
pub struct TheoreticalFragment {
    /// Ion type
    pub ion_type: String,
    
    /// Position
    pub position: usize,
    
    /// m/z value
    pub mz: f64,
    
    /// Relative intensity
    pub relative_intensity: f64,
    
    /// Charge state
    pub charge_state: i32,
}

/// Fragmentation rules engine
#[derive(Debug)]
pub struct FragmentationEngine {
    /// Fragmentation rules
    rules: Vec<FragmentationRule>,
    
    /// Amino acid properties
    aa_properties: HashMap<char, AminoAcidProperties>,
}

/// Fragmentation rule
#[derive(Debug, Clone)]
pub struct FragmentationRule {
    /// Rule ID
    pub id: String,
    
    /// Conditions for rule applicability
    pub conditions: Vec<RuleCondition>,
    
    /// Expected fragments
    pub expected_fragments: Vec<ExpectedFragment>,
    
    /// Rule confidence
    pub confidence: f64,
}

/// Rule condition
#[derive(Debug, Clone)]
pub enum RuleCondition {
    /// Amino acid at position
    AminoAcidAt { position: usize, aa: char },
    
    /// Sequence motif
    SequenceMotif { motif: String, position: Option<usize> },
    
    /// Charge state
    ChargeState { min_charge: i32, max_charge: i32 },
    
    /// Collision energy
    CollisionEnergy { min_energy: f64, max_energy: f64 },
}

/// Expected fragment from rule
#[derive(Debug, Clone)]
pub struct ExpectedFragment {
    /// Ion type
    pub ion_type: String,
    
    /// Position relative to condition
    pub relative_position: i32,
    
    /// Expected relative intensity
    pub relative_intensity: f64,
    
    /// Confidence
    pub confidence: f64,
}

/// Amino acid properties
#[derive(Debug, Clone)]
pub struct AminoAcidProperties {
    /// Single letter code
    pub code: char,
    
    /// Mass
    pub mass: f64,
    
    /// Hydrophobicity
    pub hydrophobicity: f64,
    
    /// Basicity
    pub basicity: f64,
    
    /// Fragmentation propensity
    pub fragmentation_propensity: f64,
}

/// Resolution function for fragment annotation uncertainty
pub struct FragmentAnnotationResolver;

impl ResolutionFunction for FragmentAnnotationResolver {
    fn name(&self) -> &str {
        "fragment_annotation"
    }
    
    fn resolve(&self, point: &TextPoint, context: &ResolutionContext) -> Result<ResolutionResult> {
        // Parse fragment annotation from point content
        let fragment_info: HashMap<String, Value> = serde_json::from_str(&point.content)
            .map_err(|e| TurbulanceError::ParseError { 
                message: format!("Failed to parse fragment info: {}", e) 
            })?;
        
        // Extract key information
        let mz = fragment_info.get("mz")
            .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
            .unwrap_or(0.0);
            
        let intensity = fragment_info.get("intensity")
            .and_then(|v| if let Value::Number(n) = v { Some(*n) } else { None })
            .unwrap_or(0.0);
        
        // Apply resolution strategy based on confidence
        if point.confidence > 0.8 {
            // High confidence - return certain result
            Ok(ResolutionResult::Certain(Value::String(
                format!("High confidence annotation at m/z {:.4}", mz)
            )))
        } else if point.confidence > 0.5 {
            // Medium confidence - return uncertain result with possibilities
            let possibilities = vec![
                (Value::String(format!("b-ion at m/z {:.4}", mz)), 0.4),
                (Value::String(format!("y-ion at m/z {:.4}", mz)), 0.35),
                (Value::String(format!("a-ion at m/z {:.4}", mz)), 0.15),
                (Value::String(format!("neutral loss at m/z {:.4}", mz)), 0.1),
            ];
            
            Ok(ResolutionResult::Uncertain {
                possibilities,
                confidence_interval: (point.confidence * 0.8, point.confidence * 1.2),
                aggregated_confidence: point.confidence,
            })
        } else {
            // Low confidence - contextual result
            let mut context_variants = HashMap::new();
            
            if let Some(domain) = &context.domain {
                match domain.as_str() {
                    "proteomics" => {
                        context_variants.insert(
                            "proteomics".to_string(),
                            (Value::String(format!("Potential peptide fragment at m/z {:.4}", mz)), 0.6)
                        );
                    },
                    "metabolomics" => {
                        context_variants.insert(
                            "metabolomics".to_string(),
                            (Value::String(format!("Potential metabolite fragment at m/z {:.4}", mz)), 0.4)
                        );
                    },
                    _ => {}
                }
            }
            
            Ok(ResolutionResult::Contextual {
                base_result: Value::String(format!("Unknown fragment at m/z {:.4}", mz)),
                context_variants,
                resolution_strategy: context.resolution_strategy.clone(),
            })
        }
    }
    
    fn uncertainty_factor(&self) -> f64 {
        0.3 // Fragment annotation inherently has significant uncertainty
    }
    
    fn can_handle(&self, point: &TextPoint) -> bool {
        point.content.contains("mz") && point.content.contains("intensity")
    }
}

impl Default for MS2AnnotationConfig {
    fn default() -> Self {
        Self {
            mass_tolerance: 0.02,
            min_intensity_threshold: 1000.0,
            probabilistic_threshold: 0.7,
            max_annotation_iterations: 5,
            enable_neural_predictions: true,
            enable_rt_predictions: true,
        }
    }
}

impl HybridMS2Annotator {
    /// Create a new hybrid MS2 annotator
    pub fn new(config: MS2AnnotationConfig) -> Self {
        let hybrid_config = HybridConfig {
            probabilistic_threshold: config.probabilistic_threshold,
            settlement_threshold: 0.85,
            max_roll_iterations: config.max_annotation_iterations as u64,
            enable_adaptive_loops: true,
            density_resolution: 100,
            stream_buffer_size: 1024,
        };
        
        Self {
            hybrid_processor: HybridProcessor::new(hybrid_config),
            spectral_library: SpectralLibrary::new(),
            fragmentation_engine: FragmentationEngine::new(),
            config,
        }
    }
    
    /// Annotate MS2 spectrum using hybrid processing
    pub async fn annotate_spectrum(&mut self, spectrum: &MassSpectrum) -> Result<MS2AnnotationResult> {
        let start_time = std::time::Instant::now();
        
        // Create probabilistic floor from spectrum peaks
        let mut fragment_floor = self.create_fragment_floor(spectrum)?;
        
        // Perform hybrid annotation using different processing modes
        let mut annotated_fragments = Vec::new();
        
        // Phase 1: Cycle through high-confidence peaks (deterministic)
        let high_confidence_results = self.hybrid_processor.cycle(&fragment_floor, |point, weight| {
            self.annotate_fragment_deterministic(point, weight)
        }).await?;
        
        // Phase 2: Drift through uncertain regions (probabilistic)
        let uncertain_peaks = self.extract_uncertain_peaks(spectrum);
        let drift_results = self.hybrid_processor.drift(&uncertain_peaks).await?;
        
        // Phase 3: Roll until settled for difficult annotations
        let mut final_annotations = Vec::new();
        
        for peak in spectrum.peaks() {
            if self.is_difficult_peak(peak) {
                let peak_point = self.create_peak_point(peak)?;
                let roll_result = self.hybrid_processor.roll_until_settled(&peak_point).await?;
                final_annotations.push(roll_result);
            }
        }
        
        // Phase 4: Hybrid function for comprehensive analysis
        let comprehensive_results = self.hybrid_processor.hybrid_function(
            &uncertain_peaks,
            self.config.probabilistic_threshold,
            |point| {
                // Custom resolution logic for each uncertain point
                Box::pin(async move {
                    // Implement iterative refinement here
                    Ok(format!("Refined annotation for {}", point.content))
                })
            }
        ).await?;
        
        // Compile all results
        let peptide_predictions = self.predict_peptide_sequences(&high_confidence_results, &drift_results).await?;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(MS2AnnotationResult {
            spectrum_id: spectrum.id().to_string(),
            annotated_fragments,
            peptide_predictions,
            overall_confidence: self.calculate_overall_confidence(&high_confidence_results, &drift_results),
            processing_metadata: AnnotationMetadata {
                processing_time_ms: processing_time,
                iterations: self.config.max_annotation_iterations,
                processing_mode: "hybrid".to_string(),
                quality_metrics: HashMap::new(),
                uncertainty_metrics: HashMap::new(),
            },
        })
    }
    
    /// Create probabilistic floor from spectrum peaks
    fn create_fragment_floor(&self, spectrum: &MassSpectrum) -> Result<ProbabilisticFloor> {
        let mut floor = ProbabilisticFloor::new(self.config.probabilistic_threshold);
        
        for (i, peak) in spectrum.peaks().iter().enumerate() {
            if peak.intensity >= self.config.min_intensity_threshold {
                // Calculate confidence based on intensity, mass accuracy, etc.
                let confidence = self.calculate_peak_confidence(peak);
                
                // Create TextPoint for this peak
                let peak_info = serde_json::json!({
                    "mz": peak.mz,
                    "intensity": peak.intensity,
                    "snr": peak.snr.unwrap_or(1.0),
                    "peak_index": i
                });
                
                let point = TextPoint::new(peak_info.to_string(), confidence);
                
                // Weight based on relative intensity
                let weight = peak.intensity / spectrum.base_peak_intensity().unwrap_or(peak.intensity);
                
                floor.add_point(format!("peak_{}", i), point, weight);
            }
        }
        
        Ok(floor)
    }
    
    /// Deterministic fragment annotation for high-confidence peaks
    fn annotate_fragment_deterministic(&self, point: &TextPoint, weight: f64) -> Result<crate::turbulance::probabilistic::ResolutionResult> {
        // Parse peak information
        let peak_info: serde_json::Value = serde_json::from_str(&point.content)?;
        let mz = peak_info["mz"].as_f64().unwrap_or(0.0);
        let intensity = peak_info["intensity"].as_f64().unwrap_or(0.0);
        
        // Simple deterministic assignment based on common fragments
        let assignment = if mz < 200.0 {
            "Low mass fragment (likely immonium ion)"
        } else if mz > 1000.0 {
            "High mass fragment (likely y-ion)"
        } else {
            "Medium mass fragment (likely b-ion or y-ion)"
        };
        
        Ok(crate::turbulance::probabilistic::ResolutionResult::Certain(
            Value::String(format!("{}: m/z {:.4}, intensity {:.0}", assignment, mz, intensity))
        ))
    }
    
    /// Extract uncertain peaks as text for drift processing
    fn extract_uncertain_peaks(&self, spectrum: &MassSpectrum) -> String {
        let uncertain_peaks: Vec<String> = spectrum.peaks()
            .iter()
            .filter(|peak| self.calculate_peak_confidence(peak) < self.config.probabilistic_threshold)
            .map(|peak| format!("m/z {:.4} intensity {:.0}", peak.mz, peak.intensity))
            .collect();
        
        uncertain_peaks.join("\n")
    }
    
    /// Check if a peak is difficult to annotate
    fn is_difficult_peak(&self, peak: &Peak) -> bool {
        // Consider peaks difficult if they have low S/N, unusual m/z, or low confidence
        let low_snr = peak.snr.map_or(true, |snr| snr < 5.0);
        let unusual_mz = peak.mz < 50.0 || peak.mz > 2000.0;
        let low_confidence = self.calculate_peak_confidence(peak) < 0.5;
        
        low_snr || unusual_mz || low_confidence
    }
    
    /// Create a TextPoint from a peak
    fn create_peak_point(&self, peak: &Peak) -> Result<TextPoint> {
        let peak_info = serde_json::json!({
            "mz": peak.mz,
            "intensity": peak.intensity,
            "snr": peak.snr.unwrap_or(1.0)
        });
        
        let confidence = self.calculate_peak_confidence(peak);
        Ok(TextPoint::new(peak_info.to_string(), confidence))
    }
    
    /// Calculate confidence for a peak
    fn calculate_peak_confidence(&self, peak: &Peak) -> f64 {
        let intensity_factor = (peak.intensity / 10000.0).min(1.0);
        let snr_factor = peak.snr.map_or(0.5, |snr| (snr / 10.0).min(1.0));
        
        (intensity_factor * 0.6 + snr_factor * 0.4).max(0.1).min(1.0)
    }
    
    /// Predict peptide sequences from annotation results
    async fn predict_peptide_sequences(
        &self,
        high_confidence: &[HybridResult],
        drift_results: &[HybridResult],
    ) -> Result<Vec<PeptidePrediction>> {
        // Implement peptide sequence prediction logic
        // This would integrate with the spectral library and fragmentation rules
        
        let mut predictions = Vec::new();
        
        // Simple example prediction
        if !high_confidence.is_empty() || !drift_results.is_empty() {
            predictions.push(PeptidePrediction {
                sequence: "EXAMPLE".to_string(),
                confidence: 0.75,
                supporting_fragments: vec!["b2".to_string(), "y3".to_string()],
                alternatives: vec![
                    ("EXAMPL".to_string(), 0.65),
                    ("EXAMPLI".to_string(), 0.55),
                ],
                modifications: Vec::new(),
            });
        }
        
        Ok(predictions)
    }
    
    /// Calculate overall annotation confidence
    fn calculate_overall_confidence(&self, high_confidence: &[HybridResult], drift_results: &[HybridResult]) -> f64 {
        let total_results = high_confidence.len() + drift_results.len();
        if total_results == 0 {
            return 0.0;
        }
        
        let high_conf_score: f64 = high_confidence.iter().map(|r| r.confidence).sum();
        let drift_score: f64 = drift_results.iter().map(|r| r.confidence).sum();
        
        (high_conf_score + drift_score) / total_results as f64
    }
}

impl SpectralLibrary {
    /// Create a new spectral library
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            mass_index: Vec::new(),
        }
    }
    
    /// Add an entry to the library
    pub fn add_entry(&mut self, id: String, entry: LibraryEntry) {
        // Add to mass index for fast searching
        for fragment in &entry.fragments {
            self.mass_index.push((fragment.mz, id.clone()));
        }
        
        self.entries.insert(id, entry);
        
        // Keep mass index sorted
        self.mass_index.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
    
    /// Search for similar fragments
    pub fn search_fragments(&self, mz: f64, tolerance: f64) -> Vec<&LibraryEntry> {
        let min_mz = mz - tolerance;
        let max_mz = mz + tolerance;
        
        self.mass_index
            .iter()
            .filter(|(entry_mz, _)| *entry_mz >= min_mz && *entry_mz <= max_mz)
            .filter_map(|(_, id)| self.entries.get(id))
            .collect()
    }
}

impl FragmentationEngine {
    /// Create a new fragmentation engine
    pub fn new() -> Self {
        Self {
            rules: Self::create_default_rules(),
            aa_properties: Self::create_aa_properties(),
        }
    }
    
    /// Create default fragmentation rules
    fn create_default_rules() -> Vec<FragmentationRule> {
        vec![
            FragmentationRule {
                id: "proline_effect".to_string(),
                conditions: vec![
                    RuleCondition::AminoAcidAt { position: 0, aa: 'P' }
                ],
                expected_fragments: vec![
                    ExpectedFragment {
                        ion_type: "b".to_string(),
                        relative_position: -1,
                        relative_intensity: 0.8,
                        confidence: 0.9,
                    }
                ],
                confidence: 0.85,
            },
            // Add more rules...
        ]
    }
    
    /// Create amino acid properties
    fn create_aa_properties() -> HashMap<char, AminoAcidProperties> {
        let mut properties = HashMap::new();
        
        properties.insert('A', AminoAcidProperties {
            code: 'A',
            mass: 71.037114,
            hydrophobicity: 1.8,
            basicity: 9.87,
            fragmentation_propensity: 0.3,
        });
        
        // Add all amino acids...
        
        properties
    }
    
    /// Predict fragments for a peptide sequence
    pub fn predict_fragments(&self, sequence: &str, charge_state: i32) -> Vec<TheoreticalFragment> {
        let mut fragments = Vec::new();
        
        // Generate b-ions
        for i in 1..sequence.len() {
            if let Some(mz) = self.calculate_b_ion_mz(&sequence[..i], charge_state) {
                fragments.push(TheoreticalFragment {
                    ion_type: "b".to_string(),
                    position: i,
                    mz,
                    relative_intensity: self.predict_relative_intensity(&sequence[..i], "b"),
                    charge_state,
                });
            }
        }
        
        // Generate y-ions
        for i in 1..sequence.len() {
            if let Some(mz) = self.calculate_y_ion_mz(&sequence[i..], charge_state) {
                fragments.push(TheoreticalFragment {
                    ion_type: "y".to_string(),
                    position: sequence.len() - i,
                    mz,
                    relative_intensity: self.predict_relative_intensity(&sequence[i..], "y"),
                    charge_state,
                });
            }
        }
        
        fragments
    }
    
    /// Calculate b-ion m/z
    fn calculate_b_ion_mz(&self, sequence: &str, charge_state: i32) -> Option<f64> {
        let mass = self.calculate_peptide_mass(sequence);
        if mass > 0.0 {
            Some((mass + charge_state as f64 * 1.007276) / charge_state as f64)
        } else {
            None
        }
    }
    
    /// Calculate y-ion m/z
    fn calculate_y_ion_mz(&self, sequence: &str, charge_state: i32) -> Option<f64> {
        let mass = self.calculate_peptide_mass(sequence) + 18.010565; // Add H2O
        if mass > 0.0 {
            Some((mass + charge_state as f64 * 1.007276) / charge_state as f64)
        } else {
            None
        }
    }
    
    /// Calculate peptide mass
    fn calculate_peptide_mass(&self, sequence: &str) -> f64 {
        sequence.chars()
            .filter_map(|aa| self.aa_properties.get(&aa))
            .map(|props| props.mass)
            .sum()
    }
    
    /// Predict relative intensity of fragment
    fn predict_relative_intensity(&self, sequence: &str, ion_type: &str) -> f64 {
        // Simple intensity prediction based on sequence properties
        let length_factor = 1.0 - (sequence.len() as f64 / 20.0).min(0.5);
        let aa_factor = sequence.chars()
            .filter_map(|aa| self.aa_properties.get(&aa))
            .map(|props| props.fragmentation_propensity)
            .sum::<f64>() / sequence.len() as f64;
        
        let ion_factor = match ion_type {
            "b" => 0.7,
            "y" => 0.8,
            "a" => 0.3,
            _ => 0.5,
        };
        
        (length_factor * aa_factor * ion_factor).max(0.1).min(1.0)
    }
}

/// Utility function to create default MS2 annotator
pub fn create_default_ms2_annotator() -> HybridMS2Annotator {
    HybridMS2Annotator::new(MS2AnnotationConfig::default())
}

/// Turbulance syntax for MS2 annotation
/// 
/// ```turbulance
/// funxn annotate_ms2_spectrum(spectrum) -> MS2AnnotationResult {
///     item fragment_floor = ProbabilisticFloor::from_spectrum(spectrum)
///     
///     // Cycle through high-confidence peaks
///     cycle peak over fragment_floor:
///         given peak.confidence > 0.8:
///             resolution.annotate_deterministic(peak)
///         else:
///             continue_to_probabilistic_mode()
///     
///     // Drift through uncertain regions
///     drift uncertain_peak in spectrum.uncertain_peaks():
///         resolution.probabilistic_annotation(uncertain_peak)
///         
///         // Switch mode based on uncertainty
///         if uncertain_peak.uncertainty > 0.7:
///             roll until settled:
///                 item refined_annotation = resolution.iterative_refinement(uncertain_peak)
///                 if refined_annotation.confidence > 0.85:
///                     break settled(refined_annotation)
///                 else:
///                     resolution.gather_more_evidence()
///     
///     // Flow processing for peptide sequence prediction
///     flow fragment in annotated_fragments:
///         resolution.sequence_prediction(fragment)
///         
///     return comprehensive_ms2_result
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hybrid_ms2_annotation() {
        let mut annotator = create_default_ms2_annotator();
        
        // Create a test spectrum
        let mut peaks = Vec::new();
        peaks.push(Peak::new(100.0, 1000.0, Some(5.0)));
        peaks.push(Peak::new(200.0, 5000.0, Some(10.0)));
        peaks.push(Peak::new(300.0, 2000.0, Some(3.0)));
        
        let spectrum = MassSpectrum::new("test_spectrum".to_string(), peaks);
        
        let result = annotator.annotate_spectrum(&spectrum).await;
        assert!(result.is_ok());
        
        let annotation_result = result.unwrap();
        assert_eq!(annotation_result.spectrum_id, "test_spectrum");
        assert!(annotation_result.overall_confidence > 0.0);
    }
    
    #[test]
    fn test_fragmentation_engine() {
        let engine = FragmentationEngine::new();
        let fragments = engine.predict_fragments("PEPTIDE", 2);
        
        assert!(!fragments.is_empty());
        assert!(fragments.iter().any(|f| f.ion_type == "b"));
        assert!(fragments.iter().any(|f| f.ion_type == "y"));
    }
    
    #[test]
    fn test_fragment_annotation_resolver() {
        let resolver = FragmentAnnotationResolver;
        let point = TextPoint::new(
            r#"{"mz": 150.0, "intensity": 1000.0}"#.to_string(),
            0.9
        );
        let context = ResolutionContext::default();
        
        let result = resolver.resolve(&point, &context);
        assert!(result.is_ok());
        
        match result.unwrap() {
            ResolutionResult::Certain(_) => {}, // Expected for high confidence
            _ => panic!("Expected certain result for high confidence point"),
        }
    }
}