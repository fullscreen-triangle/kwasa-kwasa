//! Calibration module for spectrometry
//!
//! This module provides functionality for calibrating different types of spectrometers.

use std::collections::HashMap;

/// Calibration manager
#[derive(Debug, Clone)]
pub struct CalibrationManager {
    /// Mass calibrations
    mass_calibrations: HashMap<String, CalibrationCurve>,
    /// Chemical shift calibrations
    chemical_shift_calibrations: HashMap<String, CalibrationCurve>,
    /// Wavelength calibrations
    wavelength_calibrations: HashMap<String, CalibrationCurve>,
    /// Standard references
    standard_references: Vec<StandardReference>,
}

/// Calibration curve
#[derive(Debug, Clone)]
pub struct CalibrationCurve {
    /// Calibration points
    pub points: Vec<CalibrationPoint>,
    /// Polynomial coefficients
    pub coefficients: Vec<f64>,
    /// R-squared value
    pub r_squared: f64,
    /// Calibration method
    pub method: CalibrationMethod,
}

/// Calibration point
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    /// Measured value
    pub measured: f64,
    /// Reference value
    pub reference: f64,
    /// Weight
    pub weight: f64,
}

/// Calibration methods
#[derive(Debug, Clone)]
pub enum CalibrationMethod {
    /// Linear calibration
    Linear,
    /// Polynomial calibration
    Polynomial(usize),
    /// Cubic spline
    CubicSpline,
    /// Piecewise linear
    PiecewiseLinear,
}

/// Standard reference
#[derive(Debug, Clone)]
pub struct StandardReference {
    /// Reference name
    pub name: String,
    /// Reference values
    pub values: HashMap<String, f64>,
    /// Uncertainty
    pub uncertainty: f64,
}

/// Calibration settings
#[derive(Debug, Clone)]
pub struct CalibrationSettings {
    /// Auto-calibration enabled
    pub auto_calibration: bool,
    /// Standards database
    pub standards_database: Option<String>,
    /// Mass accuracy tolerance (ppm)
    pub mass_accuracy_ppm: f64,
    /// Retention time tolerance (minutes)
    pub rt_tolerance_min: f64,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            auto_calibration: true,
            standards_database: None,
            mass_accuracy_ppm: 5.0,
            rt_tolerance_min: 0.1,
        }
    }
}

impl CalibrationManager {
    /// Create new calibration manager
    pub fn new() -> Self {
        Self {
            mass_calibrations: HashMap::new(),
            chemical_shift_calibrations: HashMap::new(),
            wavelength_calibrations: HashMap::new(),
            standard_references: Vec::new(),
        }
    }

    /// Add mass calibration
    pub fn add_mass_calibration(&mut self, instrument: String, curve: CalibrationCurve) {
        self.mass_calibrations.insert(instrument, curve);
    }

    /// Add chemical shift calibration
    pub fn add_chemical_shift_calibration(&mut self, solvent: String, curve: CalibrationCurve) {
        self.chemical_shift_calibrations.insert(solvent, curve);
    }

    /// Add wavelength calibration
    pub fn add_wavelength_calibration(&mut self, instrument: String, curve: CalibrationCurve) {
        self.wavelength_calibrations.insert(instrument, curve);
    }

    /// Calibrate mass value
    pub fn calibrate_mass(&self, instrument: &str, measured_mass: f64) -> Option<f64> {
        self.mass_calibrations.get(instrument)
            .map(|curve| curve.apply(measured_mass))
    }

    /// Calibrate chemical shift
    pub fn calibrate_chemical_shift(&self, solvent: &str, measured_shift: f64) -> Option<f64> {
        self.chemical_shift_calibrations.get(solvent)
            .map(|curve| curve.apply(measured_shift))
    }

    /// Calibrate wavelength
    pub fn calibrate_wavelength(&self, instrument: &str, measured_wavelength: f64) -> Option<f64> {
        self.wavelength_calibrations.get(instrument)
            .map(|curve| curve.apply(measured_wavelength))
    }

    /// Add standard reference
    pub fn add_standard_reference(&mut self, reference: StandardReference) {
        self.standard_references.push(reference);
    }

    /// Get standard reference
    pub fn get_standard_reference(&self, name: &str) -> Option<&StandardReference> {
        self.standard_references.iter()
            .find(|r| r.name == name)
    }
}

impl CalibrationCurve {
    /// Create new calibration curve
    pub fn new(points: Vec<CalibrationPoint>, method: CalibrationMethod) -> Self {
        let mut curve = Self {
            points,
            coefficients: Vec::new(),
            r_squared: 0.0,
            method,
        };
        curve.fit();
        curve
    }

    /// Fit calibration curve
    fn fit(&mut self) {
        match self.method {
            CalibrationMethod::Linear => self.fit_linear(),
            CalibrationMethod::Polynomial(order) => self.fit_polynomial(order),
            CalibrationMethod::CubicSpline => self.fit_cubic_spline(),
            CalibrationMethod::PiecewiseLinear => self.fit_piecewise_linear(),
        }
    }

    /// Fit linear calibration
    fn fit_linear(&mut self) {
        if self.points.len() < 2 {
            return;
        }

        let n = self.points.len() as f64;
        let sum_x: f64 = self.points.iter().map(|p| p.measured).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.reference).sum();
        let sum_xy: f64 = self.points.iter().map(|p| p.measured * p.reference).sum();
        let sum_x2: f64 = self.points.iter().map(|p| p.measured * p.measured).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        self.coefficients = vec![intercept, slope];

        // Calculate R-squared
        let mean_y = sum_y / n;
        let ss_tot: f64 = self.points.iter()
            .map(|p| (p.reference - mean_y).powi(2))
            .sum();
        let ss_res: f64 = self.points.iter()
            .map(|p| (p.reference - self.apply(p.measured)).powi(2))
            .sum();

        self.r_squared = 1.0 - (ss_res / ss_tot);
    }

    /// Fit polynomial calibration (simplified)
    fn fit_polynomial(&mut self, _order: usize) {
        // Use linear for simplicity
        self.fit_linear();
    }

    /// Fit cubic spline (simplified)
    fn fit_cubic_spline(&mut self) {
        // Use linear for simplicity
        self.fit_linear();
    }

    /// Fit piecewise linear (simplified)
    fn fit_piecewise_linear(&mut self) {
        // Use linear for simplicity
        self.fit_linear();
    }

    /// Apply calibration
    pub fn apply(&self, measured_value: f64) -> f64 {
        match self.method {
            CalibrationMethod::Linear => {
                if self.coefficients.len() >= 2 {
                    self.coefficients[0] + self.coefficients[1] * measured_value
                } else {
                    measured_value
                }
            }
            _ => {
                // Simplified - use linear for all methods
                if self.coefficients.len() >= 2 {
                    self.coefficients[0] + self.coefficients[1] * measured_value
                } else {
                    measured_value
                }
            }
        }
    }
}

impl StandardReference {
    /// Create new standard reference
    pub fn new(name: String) -> Self {
        Self {
            name,
            values: HashMap::new(),
            uncertainty: 0.0,
        }
    }

    /// Add reference value
    pub fn add_value(&mut self, property: String, value: f64) {
        self.values.insert(property, value);
    }

    /// Get reference value
    pub fn get_value(&self, property: &str) -> Option<f64> {
        self.values.get(property).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_curve() {
        let points = vec![
            CalibrationPoint { measured: 100.0, reference: 99.8, weight: 1.0 },
            CalibrationPoint { measured: 200.0, reference: 199.6, weight: 1.0 },
            CalibrationPoint { measured: 300.0, reference: 299.4, weight: 1.0 },
        ];

        let curve = CalibrationCurve::new(points, CalibrationMethod::Linear);
        let calibrated = curve.apply(150.0);
        
        assert!(calibrated > 149.0 && calibrated < 151.0);
        assert!(curve.r_squared > 0.9);
    }

    #[test]
    fn test_calibration_manager() {
        let mut manager = CalibrationManager::new();
        
        let points = vec![
            CalibrationPoint { measured: 100.0, reference: 99.8, weight: 1.0 },
            CalibrationPoint { measured: 200.0, reference: 199.6, weight: 1.0 },
        ];

        let curve = CalibrationCurve::new(points, CalibrationMethod::Linear);
        manager.add_mass_calibration("Instrument1".to_string(), curve);

        let calibrated = manager.calibrate_mass("Instrument1", 150.0);
        assert!(calibrated.is_some());
    }
} 