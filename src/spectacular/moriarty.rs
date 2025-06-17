use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use super::{BiomechanicalResults, JointAngleSequence, VelocityData, ForceData, EnergyAnalysis, EfficiencyMetrics, InjuryRiskAssessment, RiskFactor};

pub struct MoriartyEngine {
    pub config: MoriartyConfig,
    pub analysis_models: HashMap<String, BiomechanicalModel>,
    pub calibration_data: CalibrationData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoriartyConfig {
    pub enable_real_time_analysis: bool,
    pub joint_tracking_precision: f64,
    pub force_calculation_method: ForceCalculationMethod,
    pub energy_analysis_enabled: bool,
    pub injury_assessment_enabled: bool,
    pub model_accuracy_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForceCalculationMethod {
    InverseDynamics,
    ForwardDynamics,
    Hybrid,
    GroundReactionForces,
}

#[derive(Debug, Clone)]
pub struct BiomechanicalModel {
    pub model_id: String,
    pub model_type: ModelType,
    pub accuracy_score: f64,
    pub training_data_size: usize,
    pub applicable_sports: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    JointAngleCalculation,
    ForceEstimation,
    EnergyAnalysis,
    InjuryPrediction,
    MotionEfficiency,
}

#[derive(Debug, Clone)]
pub struct CalibrationData {
    pub joint_limits: HashMap<String, (f64, f64)>, // min, max angles
    pub body_segment_parameters: HashMap<String, SegmentParameters>,
    pub reference_measurements: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct SegmentParameters {
    pub mass: f64,
    pub length: f64,
    pub center_of_mass: f64,
    pub moment_of_inertia: f64,
}

impl MoriartyEngine {
    pub fn new(config: MoriartyConfig) -> Self {
        let mut analysis_models = HashMap::new();
        
        // Initialize default models
        analysis_models.insert("joint_angles".to_string(), BiomechanicalModel {
            model_id: "joint_angles".to_string(),
            model_type: ModelType::JointAngleCalculation,
            accuracy_score: 0.95,
            training_data_size: 10000,
            applicable_sports: vec!["running".to_string(), "walking".to_string(), "jumping".to_string()],
        });

        analysis_models.insert("force_estimation".to_string(), BiomechanicalModel {
            model_id: "force_estimation".to_string(),
            model_type: ModelType::ForceEstimation,
            accuracy_score: 0.88,
            training_data_size: 5000,
            applicable_sports: vec!["running".to_string(), "jumping".to_string()],
        });

        let calibration_data = Self::create_default_calibration();

        Self {
            config,
            analysis_models,
            calibration_data,
        }
    }

    pub async fn analyze_biomechanics(&self, pose_data: &[super::PoseFrame]) -> Result<BiomechanicalResults> {
        let joint_angles = self.calculate_joint_angles(pose_data).await?;
        let velocities = self.calculate_velocities(pose_data).await?;
        let forces = self.calculate_forces(pose_data, &velocities).await?;
        let energy_analysis = self.analyze_energy(pose_data, &velocities).await?;
        let efficiency_metrics = self.calculate_efficiency_metrics(&joint_angles, &energy_analysis).await?;
        let injury_risk = self.assess_injury_risk(&joint_angles, &forces).await?;

        Ok(BiomechanicalResults {
            joint_angles,
            velocities,
            forces,
            energy_analysis,
            efficiency_metrics,
            injury_risk_assessment: injury_risk,
        })
    }

    async fn calculate_joint_angles(&self, pose_data: &[super::PoseFrame]) -> Result<Vec<JointAngleSequence>> {
        let mut joint_sequences = HashMap::new();
        
        for frame in pose_data {
            for (joint_name, joint) in &frame.joints {
                let entry = joint_sequences.entry(joint_name.clone()).or_insert_with(Vec::new);
                
                // Calculate angle based on joint position and adjacent joints
                let angle = self.calculate_joint_angle(joint_name, joint, &frame.joints)?;
                entry.push((angle, frame.timestamp_ms));
            }
        }

        let mut results = Vec::new();
        for (joint_name, angle_data) in joint_sequences {
            let (angles, timestamps): (Vec<f64>, Vec<u64>) = angle_data.into_iter().unzip();
            
            let range_of_motion = angles.iter().fold(0.0f64, |max, &val| max.max(val)) - 
                                 angles.iter().fold(f64::INFINITY, |min, &val| min.min(val));
            
            let peak_angles = Self::find_peaks(&angles);

            results.push(JointAngleSequence {
                joint_name,
                angles,
                timestamps,
                range_of_motion,
                peak_angles,
            });
        }

        Ok(results)
    }

    fn calculate_joint_angle(&self, joint_name: &str, joint: &super::Joint3D, all_joints: &HashMap<String, super::Joint3D>) -> Result<f64> {
        // Simplified joint angle calculation
        match joint_name {
            "knee" => {
                if let (Some(hip), Some(ankle)) = (all_joints.get("hip"), all_joints.get("ankle")) {
                    let v1 = (joint.x - hip.x, joint.y - hip.y);
                    let v2 = (ankle.x - joint.x, ankle.y - joint.y);
                    
                    let dot_product = v1.0 * v2.0 + v1.1 * v2.1;
                    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
                    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
                    
                    Ok((dot_product / (mag1 * mag2)).acos().to_degrees())
                } else {
                    Ok(90.0) // Default angle if adjacent joints not found
                }
            }
            "elbow" => {
                if let (Some(shoulder), Some(wrist)) = (all_joints.get("shoulder"), all_joints.get("wrist")) {
                    let v1 = (joint.x - shoulder.x, joint.y - shoulder.y);
                    let v2 = (wrist.x - joint.x, wrist.y - joint.y);
                    
                    let dot_product = v1.0 * v2.0 + v1.1 * v2.1;
                    let mag1 = (v1.0 * v1.0 + v1.1 * v1.1).sqrt();
                    let mag2 = (v2.0 * v2.0 + v2.1 * v2.1).sqrt();
                    
                    Ok((dot_product / (mag1 * mag2)).acos().to_degrees())
                } else {
                    Ok(90.0)
                }
            }
            _ => Ok(0.0), // Default for other joints
        }
    }

    async fn calculate_velocities(&self, pose_data: &[super::PoseFrame]) -> Result<Vec<VelocityData>> {
        let mut velocity_data = Vec::new();

        for joint_name in ["hip", "knee", "ankle", "shoulder", "elbow", "wrist"] {
            let mut linear_velocities = Vec::new();
            let mut angular_velocities = Vec::new();
            let mut accelerations = Vec::new();

            for i in 1..pose_data.len() {
                if let (Some(current_joint), Some(prev_joint)) = 
                    (pose_data[i].joints.get(joint_name), pose_data[i-1].joints.get(joint_name)) {
                    
                    let dt = (pose_data[i].timestamp_ms - pose_data[i-1].timestamp_ms) as f64 / 1000.0;
                    
                    let dx = current_joint.x - prev_joint.x;
                    let dy = current_joint.y - prev_joint.y;
                    let dz = current_joint.z - prev_joint.z;
                    
                    let linear_velocity = ((dx*dx + dy*dy + dz*dz).sqrt()) / dt;
                    linear_velocities.push((dx/dt, dy/dt, dz/dt));
                    
                    // Simplified angular velocity calculation
                    let angular_velocity = (dx + dy) / dt; // Simplified
                    angular_velocities.push(angular_velocity);
                    
                    // Calculate acceleration
                    if i >= 2 && !linear_velocities.is_empty() {
                        let prev_vel_mag = linear_velocities[linear_velocities.len()-2];
                        let curr_vel_mag = linear_velocities[linear_velocities.len()-1];
                        let acceleration = ((curr_vel_mag.0 - prev_vel_mag.0).powi(2) + 
                                          (curr_vel_mag.1 - prev_vel_mag.1).powi(2) + 
                                          (curr_vel_mag.2 - prev_vel_mag.2).powi(2)).sqrt() / dt;
                        accelerations.push(acceleration);
                    }
                }
            }

            let peak_velocity = linear_velocities.iter()
                .map(|(x, y, z)| (x*x + y*y + z*z).sqrt())
                .fold(0.0f64, |max, val| max.max(val));

            velocity_data.push(VelocityData {
                joint_name: joint_name.to_string(),
                linear_velocity: linear_velocities,
                angular_velocity: angular_velocities,
                peak_velocity,
                acceleration: accelerations,
            });
        }

        Ok(velocity_data)
    }

    async fn calculate_forces(&self, _pose_data: &[super::PoseFrame], velocities: &[VelocityData]) -> Result<Vec<ForceData>> {
        let mut force_data = Vec::new();

        for velocity in velocities {
            // Simplified force calculation using F = ma
            let mass = self.calibration_data.body_segment_parameters
                .get(&velocity.joint_name)
                .map(|params| params.mass)
                .unwrap_or(1.0); // Default mass

            let mut forces = Vec::new();
            let mut directions = Vec::new();

            for &acceleration in &velocity.acceleration {
                let force_magnitude = mass * acceleration;
                forces.push(force_magnitude);
                directions.push((1.0, 0.0, 0.0)); // Simplified direction
            }

            force_data.push(ForceData {
                force_type: "muscle_force".to_string(),
                magnitude: forces,
                direction: directions,
                application_points: vec![velocity.joint_name.clone()],
                ground_reaction_forces: None, // Would require force plates
            });
        }

        Ok(force_data)
    }

    async fn analyze_energy(&self, pose_data: &[super::PoseFrame], velocities: &[VelocityData]) -> Result<EnergyAnalysis> {
        let mut kinetic_energies = Vec::new();
        let mut potential_energies = Vec::new();
        let mut power_outputs = Vec::new();

        for (i, frame) in pose_data.iter().enumerate() {
            let mut total_kinetic = 0.0;
            let mut total_potential = 0.0;
            let mut total_power = 0.0;

            for velocity in velocities {
                if let Some(joint) = frame.joints.get(&velocity.joint_name) {
                    let mass = self.calibration_data.body_segment_parameters
                        .get(&velocity.joint_name)
                        .map(|params| params.mass)
                        .unwrap_or(1.0);

                    // Kinetic energy: KE = 0.5 * m * v^2
                    if i < velocity.linear_velocity.len() {
                        let v = &velocity.linear_velocity[i];
                        let v_mag_squared = v.0*v.0 + v.1*v.1 + v.2*v.2;
                        total_kinetic += 0.5 * mass * v_mag_squared;
                    }

                    // Potential energy: PE = mgh
                    let g = 9.81; // gravity
                    total_potential += mass * g * joint.y;

                    // Power output (simplified)
                    if i < velocity.acceleration.len() && i < velocity.linear_velocity.len() {
                        let v = &velocity.linear_velocity[i];
                        let v_mag = (v.0*v.0 + v.1*v.1 + v.2*v.2).sqrt();
                        let force = mass * velocity.acceleration[i];
                        total_power += force * v_mag;
                    }
                }
            }

            kinetic_energies.push(total_kinetic);
            potential_energies.push(total_potential);
            power_outputs.push(total_power);
        }

        let total_energies: Vec<f64> = kinetic_energies.iter()
            .zip(potential_energies.iter())
            .map(|(ke, pe)| ke + pe)
            .collect();

        let energy_transfer_efficiency = if total_energies.len() > 1 {
            let initial_energy = total_energies[0];
            let final_energy = total_energies[total_energies.len()-1];
            if initial_energy > 0.0 {
                (final_energy / initial_energy).min(1.0).max(0.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        Ok(EnergyAnalysis {
            kinetic_energy: kinetic_energies,
            potential_energy: potential_energies,
            total_energy: total_energies,
            energy_transfer_efficiency,
            power_output: power_outputs,
        })
    }

    async fn calculate_efficiency_metrics(&self, joint_angles: &[JointAngleSequence], energy_analysis: &EnergyAnalysis) -> Result<EfficiencyMetrics> {
        let movement_efficiency = self.calculate_movement_efficiency(joint_angles);
        let energy_efficiency = energy_analysis.energy_transfer_efficiency;
        let technique_score = self.calculate_technique_score(joint_angles);
        let optimization_suggestions = self.generate_optimization_suggestions(joint_angles, energy_analysis);

        Ok(EfficiencyMetrics {
            movement_efficiency,
            energy_efficiency,
            technique_score,
            optimization_suggestions,
        })
    }

    async fn assess_injury_risk(&self, joint_angles: &[JointAngleSequence], forces: &[ForceData]) -> Result<InjuryRiskAssessment> {
        let mut risk_factors = Vec::new();
        let mut high_risk_movements = Vec::new();
        let mut prevention_recommendations = Vec::new();

        // Check for extreme joint angles
        for joint_sequence in joint_angles {
            if let Some((min_angle, max_angle)) = self.calibration_data.joint_limits.get(&joint_sequence.joint_name) {
                for &angle in &joint_sequence.angles {
                    if angle < *min_angle || angle > *max_angle {
                        risk_factors.push(RiskFactor {
                            factor_name: format!("{}_extreme_angle", joint_sequence.joint_name),
                            risk_level: 0.8,
                            description: format!("Joint angle {} exceeds safe range", angle),
                            mitigation_strategies: vec![
                                "Improve flexibility training".to_string(),
                                "Focus on proper form".to_string(),
                            ],
                        });

                        high_risk_movements.push(format!("Extreme {} angle at {:.1}°", joint_sequence.joint_name, angle));
                    }
                }
            }
        }

        // Check for excessive forces
        for force_data in forces {
            let max_force = force_data.magnitude.iter().fold(0.0f64, |max, &val| max.max(val));
            if max_force > 1000.0 { // Arbitrary threshold
                risk_factors.push(RiskFactor {
                    factor_name: format!("{}_high_force", force_data.force_type),
                    risk_level: 0.6,
                    description: format!("High force detected: {:.1}N", max_force),
                    mitigation_strategies: vec![
                        "Gradual load progression".to_string(),
                        "Strength training".to_string(),
                    ],
                });
            }
        }

        // General prevention recommendations
        prevention_recommendations.extend(vec![
            "Regular flexibility and mobility work".to_string(),
            "Progressive overload in training".to_string(),
            "Proper warm-up and cool-down".to_string(),
            "Regular biomechanical assessments".to_string(),
        ]);

        let overall_risk_score = risk_factors.iter()
            .map(|rf| rf.risk_level)
            .fold(0.0f64, |sum, val| sum + val) / risk_factors.len().max(1) as f64;

        Ok(InjuryRiskAssessment {
            overall_risk_score,
            risk_factors,
            high_risk_movements,
            prevention_recommendations,
        })
    }

    fn find_peaks(data: &[f64]) -> Vec<f64> {
        let mut peaks = Vec::new();
        for i in 1..data.len()-1 {
            if data[i] > data[i-1] && data[i] > data[i+1] {
                peaks.push(data[i]);
            }
        }
        peaks
    }

    fn calculate_movement_efficiency(&self, joint_angles: &[JointAngleSequence]) -> f64 {
        // Simplified efficiency calculation based on smoothness of joint movements
        let mut total_smoothness = 0.0;
        let mut count = 0;

        for sequence in joint_angles {
            if sequence.angles.len() > 2 {
                let mut variations = Vec::new();
                for i in 1..sequence.angles.len() {
                    variations.push((sequence.angles[i] - sequence.angles[i-1]).abs());
                }
                
                let avg_variation = variations.iter().sum::<f64>() / variations.len() as f64;
                let smoothness = 1.0 / (1.0 + avg_variation / 10.0); // Normalize
                total_smoothness += smoothness;
                count += 1;
            }
        }

        if count > 0 {
            total_smoothness / count as f64
        } else {
            0.5
        }
    }

    fn calculate_technique_score(&self, joint_angles: &[JointAngleSequence]) -> f64 {
        // Simplified technique scoring based on joint angle patterns
        let mut score = 0.8; // Base score

        for sequence in joint_angles {
            // Check for excessive range of motion
            if sequence.range_of_motion > 180.0 {
                score -= 0.1;
            }
            
            // Check for consistency
            if sequence.angles.len() > 5 {
                let std_dev = Self::calculate_std_dev(&sequence.angles);
                if std_dev > 30.0 { // High variability
                    score -= 0.05;
                }
            }
        }

        score.max(0.0).min(1.0)
    }

    fn calculate_std_dev(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    fn generate_optimization_suggestions(&self, joint_angles: &[JointAngleSequence], _energy_analysis: &EnergyAnalysis) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Analyze joint angle patterns for suggestions
        for sequence in joint_angles {
            if sequence.range_of_motion < 30.0 {
                suggestions.push(format!("Increase {} range of motion through targeted mobility work", sequence.joint_name));
            }
            
            if sequence.peak_angles.len() > 10 {
                suggestions.push(format!("Work on {} stability to reduce excessive movement variations", sequence.joint_name));
            }
        }

        if suggestions.is_empty() {
            suggestions.push("Continue current training approach - biomechanics look good".to_string());
        }

        suggestions
    }

    fn create_default_calibration() -> CalibrationData {
        let mut joint_limits = HashMap::new();
        joint_limits.insert("knee".to_string(), (0.0, 160.0));
        joint_limits.insert("elbow".to_string(), (0.0, 150.0));
        joint_limits.insert("hip".to_string(), (-30.0, 120.0));
        joint_limits.insert("ankle".to_string(), (-45.0, 45.0));

        let mut body_segment_parameters = HashMap::new();
        body_segment_parameters.insert("thigh".to_string(), SegmentParameters {
            mass: 10.0,
            length: 0.4,
            center_of_mass: 0.2,
            moment_of_inertia: 0.133,
        });
        body_segment_parameters.insert("shank".to_string(), SegmentParameters {
            mass: 4.5,
            length: 0.35,
            center_of_mass: 0.175,
            moment_of_inertia: 0.045,
        });

        let mut reference_measurements = HashMap::new();
        reference_measurements.insert("height".to_string(), 1.75);
        reference_measurements.insert("weight".to_string(), 70.0);

        CalibrationData {
            joint_limits,
            body_segment_parameters,
            reference_measurements,
        }
    }
} 