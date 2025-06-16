//! Peak picking and analysis module
//!
//! This module provides functionality for peak detection and analysis across different spectroscopy techniques.

use std::collections::HashMap;

/// Peak picking engine
#[derive(Debug, Clone)]
pub struct PeakPicking {
    /// Configuration
    config: PeakPickingConfig,
}

/// Configuration for peak picking
#[derive(Debug, Clone)]
pub struct PeakPickingConfig {
    /// Signal-to-noise threshold
    pub snr_threshold: f64,
    /// Minimum peak height
    pub min_peak_height: f64,
    /// Peak width tolerance
    pub peak_width_tolerance: f64,
    /// Baseline method
    pub baseline_method: BaselineMethod,
    /// Smoothing parameters
    pub smoothing: SmoothingConfig,
}

/// Baseline correction methods
#[derive(Debug, Clone)]
pub enum BaselineMethod {
    /// Linear baseline
    Linear,
    /// Polynomial baseline
    Polynomial(usize),
    /// Asymmetric least squares
    AsLS,
    /// Rolling ball
    RollingBall(f64),
    /// None
    None,
}

/// Smoothing configuration
#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    /// Enable smoothing
    pub enabled: bool,
    /// Smoothing method
    pub method: SmoothingMethod,
    /// Window size
    pub window_size: usize,
}

/// Smoothing methods
#[derive(Debug, Clone)]
pub enum SmoothingMethod {
    /// Moving average
    MovingAverage,
    /// Savitzky-Golay
    SavitzkyGolay(usize),
    /// Gaussian
    Gaussian(f64),
    /// Median filter
    Median,
}

/// Generic peak structure
#[derive(Debug, Clone)]
pub struct Peak {
    /// X-axis value (wavelength, m/z, wavenumber, etc.)
    pub x_value: f64,
    /// Y-axis value (intensity, absorbance, etc.)
    pub y_value: f64,
    /// Peak width
    pub width: f64,
    /// Signal-to-noise ratio
    pub snr: Option<f64>,
    /// Peak area
    pub area: f64,
    /// Peak properties
    pub properties: HashMap<String, String>,
}

/// Peak picking result
#[derive(Debug, Clone)]
pub struct PeakPickingResult {
    /// Detected peaks
    pub peaks: Vec<Peak>,
    /// Baseline
    pub baseline: Vec<f64>,
    /// Smoothed data
    pub smoothed_data: Option<Vec<f64>>,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for peak picking
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Total number of peaks
    pub total_peaks: usize,
    /// Average signal-to-noise ratio
    pub average_snr: f64,
    /// Baseline quality score
    pub baseline_quality: f64,
    /// Peak resolution score
    pub peak_resolution: f64,
}

impl Default for PeakPickingConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 3.0,
            min_peak_height: 100.0,
            peak_width_tolerance: 0.1,
            baseline_method: BaselineMethod::Linear,
            smoothing: SmoothingConfig {
                enabled: true,
                method: SmoothingMethod::MovingAverage,
                window_size: 5,
            },
        }
    }
}

impl PeakPicking {
    /// Create new peak picking engine
    pub fn new(config: PeakPickingConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(PeakPickingConfig::default())
    }

    /// Pick peaks from spectral data
    pub fn pick_peaks(&self, x_data: &[f64], y_data: &[f64]) -> PeakPickingResult {
        assert_eq!(x_data.len(), y_data.len(), "X and Y data must have same length");

        // Apply smoothing if enabled
        let smoothed_data = if self.config.smoothing.enabled {
            Some(self.apply_smoothing(y_data))
        } else {
            None
        };

        let data_to_use: &[f64] = if let Some(ref smoothed) = smoothed_data {
            smoothed.as_slice()
        } else {
            y_data
        };

        // Calculate baseline
        let baseline = self.calculate_baseline(data_to_use);

        // Subtract baseline
        let corrected_data: Vec<f64> = data_to_use.iter()
            .zip(baseline.iter())
            .map(|(y, b)| y - b)
            .collect();

        // Find peaks
        let peaks = self.find_peaks(x_data, &corrected_data);

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&peaks, &corrected_data);

        PeakPickingResult {
            peaks,
            baseline,
            smoothed_data,
            quality_metrics,
        }
    }

    /// Apply smoothing to data
    fn apply_smoothing(&self, data: &[f64]) -> Vec<f64> {
        match self.config.smoothing.method {
            SmoothingMethod::MovingAverage => self.moving_average(data),
            SmoothingMethod::Median => self.median_filter(data),
            SmoothingMethod::Gaussian(_) => self.gaussian_smooth(data),
            SmoothingMethod::SavitzkyGolay(_) => self.savgol_smooth(data),
        }
    }

    /// Moving average smoothing
    fn moving_average(&self, data: &[f64]) -> Vec<f64> {
        let window = self.config.smoothing.window_size;
        let mut smoothed = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = if i >= window / 2 { i - window / 2 } else { 0 };
            let end = (i + window / 2 + 1).min(data.len());
            
            let sum: f64 = data[start..end].iter().sum();
            let count = end - start;
            smoothed.push(sum / count as f64);
        }

        smoothed
    }

    /// Median filter
    fn median_filter(&self, data: &[f64]) -> Vec<f64> {
        let window = self.config.smoothing.window_size;
        let mut smoothed = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let start = if i >= window / 2 { i - window / 2 } else { 0 };
            let end = (i + window / 2 + 1).min(data.len());
            
            let mut window_data: Vec<f64> = data[start..end].to_vec();
            window_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let median = if window_data.len() % 2 == 0 {
                (window_data[window_data.len() / 2 - 1] + window_data[window_data.len() / 2]) / 2.0
            } else {
                window_data[window_data.len() / 2]
            };
            
            smoothed.push(median);
        }

        smoothed
    }

    /// Gaussian smoothing (simplified)
    fn gaussian_smooth(&self, data: &[f64]) -> Vec<f64> {
        // Simplified Gaussian - use moving average for now
        self.moving_average(data)
    }

    /// Savitzky-Golay smoothing (simplified)
    fn savgol_smooth(&self, data: &[f64]) -> Vec<f64> {
        // Simplified - use moving average for now
        self.moving_average(data)
    }

    /// Calculate baseline
    fn calculate_baseline(&self, data: &[f64]) -> Vec<f64> {
        match self.config.baseline_method {
            BaselineMethod::None => vec![0.0; data.len()],
            BaselineMethod::Linear => self.linear_baseline(data),
            BaselineMethod::Polynomial(order) => self.polynomial_baseline(data, order),
            BaselineMethod::AsLS => self.als_baseline(data),
            BaselineMethod::RollingBall(radius) => self.rolling_ball_baseline(data, radius),
        }
    }

    /// Linear baseline
    fn linear_baseline(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }

        let first = data[0];
        let last = data[data.len() - 1];
        let slope = (last - first) / (data.len() - 1) as f64;

        (0..data.len())
            .map(|i| first + slope * i as f64)
            .collect()
    }

    /// Polynomial baseline (simplified)
    fn polynomial_baseline(&self, data: &[f64], _order: usize) -> Vec<f64> {
        // Use linear for simplicity
        self.linear_baseline(data)
    }

    /// Asymmetric least squares baseline (simplified)
    fn als_baseline(&self, data: &[f64]) -> Vec<f64> {
        // Use linear for simplicity
        self.linear_baseline(data)
    }

    /// Rolling ball baseline (simplified)
    fn rolling_ball_baseline(&self, data: &[f64], _radius: f64) -> Vec<f64> {
        // Use linear for simplicity
        self.linear_baseline(data)
    }

    /// Find peaks in data
    fn find_peaks(&self, x_data: &[f64], y_data: &[f64]) -> Vec<Peak> {
        let mut peaks = Vec::new();
        let mut i = 1;

        while i < y_data.len() - 1 {
            let prev_y = y_data[i - 1];
            let curr_y = y_data[i];
            let next_y = y_data[i + 1];

            // Check if this is a peak
            if curr_y > prev_y && curr_y > next_y && curr_y > self.config.min_peak_height {
                let width = self.calculate_peak_width(y_data, i);
                let area = self.calculate_peak_area(y_data, i, width);
                let snr = self.calculate_snr(y_data, i);

                if snr >= self.config.snr_threshold {
                    peaks.push(Peak {
                        x_value: x_data[i],
                        y_value: curr_y,
                        width,
                        snr: Some(snr),
                        area,
                        properties: HashMap::new(),
                    });
                }
            }
            i += 1;
        }

        // Sort by intensity
        peaks.sort_by(|a, b| b.y_value.partial_cmp(&a.y_value).unwrap());
        peaks
    }

    /// Calculate peak width
    fn calculate_peak_width(&self, data: &[f64], peak_index: usize) -> f64 {
        let peak_height = data[peak_index];
        let half_height = peak_height / 2.0;

        let mut left = peak_index;
        let mut right = peak_index;

        // Find left half-height
        while left > 0 && data[left] > half_height {
            left -= 1;
        }

        // Find right half-height
        while right < data.len() - 1 && data[right] > half_height {
            right += 1;
        }

        (right - left) as f64
    }

    /// Calculate peak area
    fn calculate_peak_area(&self, data: &[f64], peak_index: usize, width: f64) -> f64 {
        let start = ((peak_index as f64 - width / 2.0).max(0.0) as usize).min(data.len() - 1);
        let end = ((peak_index as f64 + width / 2.0) as usize).min(data.len() - 1);

        data[start..=end].iter().sum()
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, data: &[f64], peak_index: usize) -> f64 {
        let signal = data[peak_index];
        
        // Estimate noise from local region
        let window_size = 20;
        let start = if peak_index >= window_size { peak_index - window_size } else { 0 };
        let end = (peak_index + window_size).min(data.len());
        
        let noise_region = &data[start..end];
        let noise_std = self.calculate_std(noise_region);
        
        if noise_std > 0.0 {
            signal / noise_std
        } else {
            f64::INFINITY
        }
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        
        variance.sqrt()
    }

    /// Calculate quality metrics
    fn calculate_quality_metrics(&self, peaks: &[Peak], data: &[f64]) -> QualityMetrics {
        let total_peaks = peaks.len();
        
        let average_snr = if !peaks.is_empty() {
            peaks.iter()
                .filter_map(|p| p.snr)
                .sum::<f64>() / peaks.len() as f64
        } else {
            0.0
        };

        let baseline_quality = 0.8; // Simplified metric
        let peak_resolution = if peaks.len() > 1 { 0.7 } else { 1.0 }; // Simplified metric

        QualityMetrics {
            total_peaks,
            average_snr,
            baseline_quality,
            peak_resolution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peak_picking() {
        let peak_picker = PeakPicking::default();
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![100.0, 500.0, 1000.0, 400.0, 150.0];

        let result = peak_picker.pick_peaks(&x_data, &y_data);
        
        assert!(!result.peaks.is_empty());
        assert_eq!(result.baseline.len(), y_data.len());
    }
} 