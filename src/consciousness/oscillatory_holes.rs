/// Oscillatory Hole Operations
/// 
/// Oscillatory holes are the fundamental computation units
/// Thoughts are formed via hole-filling (categorical completion)

use super::types::*;

/// Detect oscillatory holes from variance analysis
pub fn detect_oscillatory_holes(
    field: &HydrogenFieldState,
    variance_threshold: f64,
) -> Vec<OscillatoryHole> {
    let mut holes = Vec::new();
    
    // Holes are regions of low variance in the oscillatory field
    if field.variance < variance_threshold {
        // Detected a stable hole
        holes.push(OscillatoryHole {
            formation_time: 0.0,
            stability_duration: 0.1, // 100 ms typical
            variance: field.variance,
            pcet_events: Vec::new(),
            location: Vec3::new(0.0, 0.0, 0.0),
            hole_type: HoleType::PositiveCharge,
        });
    }
    
    holes
}

/// Measure hole stability duration
pub fn measure_hole_stability(hole: &OscillatoryHole) -> f64 {
    hole.stability_duration
}

/// Track proton-coupled electron transfer events
pub fn track_pcet_events(
    hole: &OscillatoryHole,
    time_window: f64,
) -> Vec<ElectronTransfer> {
    // Return PCET events within time window
    hole.pcet_events.iter()
        .filter(|e| e.time >= time_window)
        .cloned()
        .collect()
}

/// Calculate hole formation rate (thoughts per second)
pub fn calculate_hole_formation_rate(holes: &[OscillatoryHole]) -> f64 {
    if holes.is_empty() {
        return 0.0;
    }
    
    // Average formation rate
    let total_duration: f64 = holes.iter()
        .map(|h| h.stability_duration)
        .sum();
    
    if total_duration > 0.0 {
        holes.len() as f64 / total_duration
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hole_detection() {
        let field = HydrogenFieldState {
            frequency: 40e12,
            coherence: 0.9,
            variance: 0.3, // Low variance
            spatial_extent: Vec3::default(),
            field_map: std::collections::HashMap::new(),
        };
        
        let holes = detect_oscillatory_holes(&field, 0.5);
        assert!(!holes.is_empty());
    }
}

