//! Utility functions and helper modules

use crate::error::Result;

/// Mathematical utility functions
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Standard deviation calculation
pub fn std_dev(values: &[f64]) -> f64 {
    let m = mean(values);
    let variance = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(crate::error::TurbulanceError::argument_error("Arrays must have same length"));
    }
    
    let x_mean = mean(x);
    let y_mean = mean(y);
    
    let numerator: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum();
    
    let x_var: f64 = x.iter().map(|xi| (xi - x_mean).powi(2)).sum();
    let y_var: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    
    let denominator = (x_var * y_var).sqrt();
    
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
} 