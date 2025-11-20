/// Temporal Scale Operations
/// 
/// Multi-scale hierarchical processing from cellular (T1) to organismal (T5)
/// Each scale operates at different frequencies and timescales

use super::types::*;

/// Navigate between temporal scales
pub fn navigate_temporal_scales(
    current_scale: TemporalScale,
    target_scale: TemporalScale,
) -> Vec<TemporalScale> {
    use TemporalScale::*;
    
    let scales = [T1Cellular, T2Population, T3Tissue, T4Functional, T5Organismal];
    
    let current_idx = scales.iter().position(|&s| s == current_scale).unwrap();
    let target_idx = scales.iter().position(|&s| s == target_scale).unwrap();
    
    if current_idx <= target_idx {
        scales[current_idx..=target_idx].to_vec()
    } else {
        scales[target_idx..=current_idx].iter().rev().copied().collect()
    }
}

/// Calculate phase coherence at specific temporal scale
pub fn measure_scale_coherence(
    scale: TemporalScale,
    measurements: &[f64],
) -> f64 {
    if measurements.is_empty() {
        return 0.5;
    }
    
    // Filter measurements to scale's frequency range
    let (min_freq, max_freq) = scale.frequency_range_hz();
    
    // Calculate coherence within frequency band
    let mean = measurements.iter().sum::<f64>() / measurements.len() as f64;
    let variance = measurements.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / measurements.len() as f64;
    
    // Coherence inversely related to variance
    1.0 / (1.0 + variance)
}

/// Synchronize across multiple temporal scales
pub fn synchronize_multi_scale(
    scales: &[TemporalScale],
    target_coherence: f64,
) -> Result<Vec<ScaleSyncSpec>, String> {
    let mut specs = Vec::new();
    
    for scale in scales {
        let (min_freq, max_freq) = scale.frequency_range_hz();
        let center_freq = ((min_freq * max_freq).sqrt()) * 1e12; // Geometric mean, convert to Hz
        
        specs.push(ScaleSyncSpec {
            scale: *scale,
            target_frequency: center_freq,
            target_coherence,
            coupling_requirement: target_coherence * 0.8,
        });
    }
    
    Ok(specs)
}

/// Calculate cross-scale coupling
pub fn calculate_cross_scale_coupling(
    scale1: TemporalScale,
    scale2: TemporalScale,
    measurements: &[(f64, f64)],
) -> f64 {
    if measurements.is_empty() {
        return 0.5;
    }
    
    // Calculate correlation between scales
    let n = measurements.len() as f64;
    let sum_x: f64 = measurements.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = measurements.iter().map(|(_, y)| y).sum();
    let sum_xy: f64 = measurements.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f64 = measurements.iter().map(|(x, _)| x * x).sum();
    let sum_y2: f64 = measurements.iter().map(|(_, y)| y * y).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator > 0.0 {
        (numerator / denominator).abs()
    } else {
        0.5
    }
}

/// Map temporal scale to intervention timing
pub fn scale_to_intervention_timing(scale: TemporalScale) -> InterventionTiming {
    use TemporalScale::*;
    
    match scale {
        T1Cellular => InterventionTiming {
            administration_interval_hours: 6.0,
            effect_onset_hours: 0.5,
            effect_duration_hours: 12.0,
            monitoring_interval_hours: 1.0,
        },
        T2Population => InterventionTiming {
            administration_interval_hours: 24.0,
            effect_onset_hours: 2.0,
            effect_duration_hours: 48.0,
            monitoring_interval_hours: 6.0,
        },
        T3Tissue => InterventionTiming {
            administration_interval_hours: 168.0, // Weekly
            effect_onset_hours: 24.0,
            effect_duration_hours: 336.0, // 2 weeks
            monitoring_interval_hours: 24.0,
        },
        T4Functional => InterventionTiming {
            administration_interval_hours: 720.0, // Monthly
            effect_onset_hours: 168.0,
            effect_duration_hours: 2160.0, // 3 months
            monitoring_interval_hours: 168.0,
        },
        T5Organismal => InterventionTiming {
            administration_interval_hours: 8760.0, // Yearly
            effect_onset_hours: 720.0,
            effect_duration_hours: 17520.0, // 2 years
            monitoring_interval_hours: 720.0,
        },
    }
}

#[derive(Debug, Clone)]
pub struct ScaleSyncSpec {
    pub scale: TemporalScale,
    pub target_frequency: f64,
    pub target_coherence: f64,
    pub coupling_requirement: f64,
}

#[derive(Debug, Clone)]
pub struct InterventionTiming {
    pub administration_interval_hours: f64,
    pub effect_onset_hours: f64,
    pub effect_duration_hours: f64,
    pub monitoring_interval_hours: f64,
}

/// Calculate optimal temporal scale for intervention
pub fn optimal_scale_for_intervention(
    target_effect_duration_hours: f64,
) -> TemporalScale {
    use TemporalScale::*;
    
    if target_effect_duration_hours < 24.0 {
        T1Cellular
    } else if target_effect_duration_hours < 168.0 {
        T2Population
    } else if target_effect_duration_hours < 720.0 {
        T3Tissue
    } else if target_effect_duration_hours < 4320.0 {
        T4Functional
    } else {
        T5Organismal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scale_navigation() {
        let path = navigate_temporal_scales(
            TemporalScale::T1Cellular,
            TemporalScale::T3Tissue,
        );
        
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], TemporalScale::T1Cellular);
        assert_eq!(path[2], TemporalScale::T3Tissue);
    }
    
    #[test]
    fn test_intervention_timing() {
        let timing = scale_to_intervention_timing(TemporalScale::T1Cellular);
        assert!(timing.administration_interval_hours < 24.0);
        assert!(timing.effect_onset_hours < 2.0);
    }
    
    #[test]
    fn test_optimal_scale_selection() {
        let scale = optimal_scale_for_intervention(48.0); // 2 days
        assert_eq!(scale, TemporalScale::T2Population);
    }
}

