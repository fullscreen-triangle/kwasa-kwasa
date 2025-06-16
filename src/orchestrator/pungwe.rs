use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use tokio::time::{Duration, Instant};
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};
use super::clothesline::{ClotheslineModule, ComprehensionResult};
use super::nicotine::NicotineContextValidator;

/// Represents the understanding gap between claimed and actual comprehension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderstandingGap {
    pub claimed_confidence: f64,
    pub actual_confidence: f64,
    pub gap_magnitude: f64,
    pub gap_type: GapType,
    pub remediation_priority: u8,
}

impl UnderstandingGap {
    pub fn new(claimed: f64, actual: f64) -> Self {
        let gap_magnitude = (claimed - actual).abs();
        let gap_type = if claimed > actual {
            GapType::Overconfidence
        } else if actual > claimed {
            GapType::Underconfidence
        } else {
            GapType::Calibrated
        };
        
        let remediation_priority = match gap_magnitude {
            g if g > 0.7 => 5, // Critical
            g if g > 0.5 => 4, // High
            g if g > 0.3 => 3, // Medium
            g if g > 0.1 => 2, // Low
            _ => 1,            // Minimal
        };
        
        Self {
            claimed_confidence: claimed,
            actual_confidence: actual,
            gap_magnitude,
            gap_type,
            remediation_priority,
        }
    }

    pub fn is_critical(&self) -> bool {
        self.remediation_priority >= 4
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapType {
    Overconfidence,   // Claimed > Actual (Dunning-Kruger effect)
    Underconfidence,  // Actual > Claimed (Imposter syndrome)
    Calibrated,       // Claimed ≈ Actual (Well-calibrated)
}

impl GapType {
    pub fn description(&self) -> &'static str {
        match self {
            GapType::Overconfidence => "Overestimating understanding (Dunning-Kruger pattern)",
            GapType::Underconfidence => "Underestimating understanding (Imposter syndrome pattern)",
            GapType::Calibrated => "Well-calibrated understanding assessment",
        }
    }
}

/// ATP synthesis tracking for truth energy production
#[derive(Debug, Clone)]
pub struct AtpSynthesis {
    pub process_id: Uuid,
    pub initial_substrate: String,
    pub clothesline_confidence: f64,
    pub nicotine_confidence: f64,
    pub mzekezeke_confidence: f64,
    pub hatata_confidence: f64,
    pub final_atp_yield: u32,
    pub synthesis_efficiency: f64,
    pub understanding_gap: UnderstandingGap,
    pub goal_distance: f64,
    pub created_at: Instant,
}

impl AtpSynthesis {
    pub fn new(process_id: Uuid, substrate: String) -> Self {
        Self {
            process_id,
            initial_substrate: substrate,
            clothesline_confidence: 0.0,
            nicotine_confidence: 0.0,
            mzekezeke_confidence: 0.0,
            hatata_confidence: 0.0,
            final_atp_yield: 0,
            synthesis_efficiency: 0.0,
            understanding_gap: UnderstandingGap::new(0.0, 0.0),
            goal_distance: 1.0, // Start at maximum distance
            created_at: Instant::now(),
        }
    }

    pub fn calculate_final_atp(&mut self) -> u32 {
        // ATP Synthase calculation: convert NADH/FADH₂ to ATP
        let base_atp = 32; // Theoretical maximum from electron transport
        
        // Efficiency based on module confidence integration
        let module_average = (self.clothesline_confidence + 
                            self.nicotine_confidence + 
                            self.mzekezeke_confidence + 
                            self.hatata_confidence) / 4.0;
        
        // Understanding gap penalty
        let gap_penalty = 1.0 - (self.understanding_gap.gap_magnitude * 0.5);
        
        // Goal distance penalty
        let goal_penalty = 1.0 - (self.goal_distance * 0.3);
        
        self.synthesis_efficiency = module_average * gap_penalty * goal_penalty;
        self.final_atp_yield = (base_atp as f64 * self.synthesis_efficiency) as u32;
        
        self.final_atp_yield
    }

    pub fn processing_time(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Represents a goal and the distance to achieving it
#[derive(Debug, Clone)]
pub struct GoalDistanceMetric {
    pub goal_description: String,
    pub current_state: String,
    pub target_state: String,
    pub distance: f64,
    pub progress_trajectory: Vec<f64>,
    pub estimated_completion_time: Option<Duration>,
}

impl GoalDistanceMetric {
    pub fn new(goal: String, current: String, target: String) -> Self {
        Self {
            goal_description: goal,
            current_state: current,
            target_state: target,
            distance: 1.0, // Start at maximum distance
            progress_trajectory: vec![1.0],
            estimated_completion_time: None,
        }
    }

    pub fn update_distance(&mut self, new_distance: f64) {
        self.distance = new_distance.clamp(0.0, 1.0);
        self.progress_trajectory.push(self.distance);
        
        // Estimate completion time based on trajectory
        if self.progress_trajectory.len() > 3 {
            let recent_progress: Vec<f64> = self.progress_trajectory.iter()
                .rev()
                .take(3)
                .cloned()
                .collect();
            
            let progress_rate = (recent_progress[2] - recent_progress[0]) / 2.0;
            if progress_rate > 0.01 {
                let remaining_distance = self.distance;
                let estimated_steps = remaining_distance / progress_rate;
                self.estimated_completion_time = Some(Duration::from_secs((estimated_steps * 60.0) as u64));
            }
        }
    }

    pub fn is_approaching_goal(&self) -> bool {
        self.distance < 0.3 && self.progress_trajectory.len() > 1 && 
        self.progress_trajectory.last().unwrap() < self.progress_trajectory[self.progress_trajectory.len() - 2]
    }
}

/// The Pungwe Module - ATP Synthase & Intuition Layer Workhorse
pub struct PungweModule {
    name: String,
    
    // ATP Synthesis Tracking
    active_syntheses: Arc<Mutex<HashMap<Uuid, AtpSynthesis>>>,
    completed_syntheses: Arc<Mutex<Vec<AtpSynthesis>>>,
    
    // Module Integration
    clothesline_module: Arc<Mutex<ClotheslineModule>>,
    nicotine_module: Arc<Mutex<NicotineContextValidator>>,
    
    // Goal Distance Tracking
    goal_metrics: Arc<Mutex<HashMap<String, GoalDistanceMetric>>>,
    
    // Understanding Gap Analysis
    understanding_gaps: Arc<Mutex<Vec<UnderstandingGap>>>,
    calibration_history: Arc<Mutex<Vec<(f64, f64, Instant)>>>, // claimed, actual, time
    
    // Configuration
    synthesis_threshold: f64,
    max_concurrent_syntheses: usize,
    gap_tolerance: f64,
    
    // Statistics
    stats: Arc<Mutex<ProcessorStats>>,
    total_atp_produced: Arc<Mutex<u64>>,
    average_efficiency: Arc<Mutex<f64>>,
}

impl PungweModule {
    pub fn new() -> Self {
        Self {
            name: "PungweModule".to_string(),
            active_syntheses: Arc::new(Mutex::new(HashMap::new())),
            completed_syntheses: Arc::new(Mutex::new(Vec::new())),
            clothesline_module: Arc::new(Mutex::new(ClotheslineModule::new())),
            nicotine_module: Arc::new(Mutex::new(NicotineContextValidator::new())),
            goal_metrics: Arc::new(Mutex::new(HashMap::new())),
            understanding_gaps: Arc::new(Mutex::new(Vec::new())),
            calibration_history: Arc::new(Mutex::new(Vec::new())),
            synthesis_threshold: 0.6,
            max_concurrent_syntheses: 15,
            gap_tolerance: 0.2,
            stats: Arc::new(Mutex::new(ProcessorStats::new())),
            total_atp_produced: Arc::new(Mutex::new(0)),
            average_efficiency: Arc::new(Mutex::new(0.0)),
        }
    }

    pub fn with_synthesis_threshold(mut self, threshold: f64) -> Self {
        self.synthesis_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_gap_tolerance(mut self, tolerance: f64) -> Self {
        self.gap_tolerance = tolerance.clamp(0.0, 1.0);
        self
    }

    /// Begin ATP synthesis process by combining module outputs
    pub async fn begin_atp_synthesis(&self, process_id: Uuid, content: &str) -> Result<f64, String> {
        let mut synthesis = AtpSynthesis::new(process_id, content.to_string());
        
        // Combine Clothesline output (comprehension validation)
        let clothesline_confidence = self.get_clothesline_confidence(content).await?;
        synthesis.clothesline_confidence = clothesline_confidence;
        
        // Combine Nicotine output (context validation)
        let nicotine_confidence = self.get_nicotine_confidence(content).await?;
        synthesis.nicotine_confidence = nicotine_confidence;
        
        // Get Mzekezeke confidence (Bayesian belief networks)
        synthesis.mzekezeke_confidence = self.get_mzekezeke_confidence(content).await?;
        
        // Get Hatata confidence (decision optimization)
        synthesis.hatata_confidence = self.get_hatata_confidence(content).await?;
        
        // Calculate understanding gap
        let claimed_confidence = (synthesis.clothesline_confidence + synthesis.nicotine_confidence) / 2.0;
        let actual_confidence = (synthesis.mzekezeke_confidence + synthesis.hatata_confidence) / 2.0;
        synthesis.understanding_gap = UnderstandingGap::new(claimed_confidence, actual_confidence);
        
        // Measure goal distance
        synthesis.goal_distance = self.measure_goal_distance(content).await?;
        
        // Calculate final ATP yield
        let final_atp = synthesis.calculate_final_atp();
        
        // Store synthesis
        self.active_syntheses.lock().unwrap().insert(process_id, synthesis);
        
        // Log results
        self.log_synthesis_results(process_id, final_atp).await?;
        
        Ok(final_atp as f64 / 32.0) // Return as confidence (0-1)
    }

    /// Measure distance between actual understanding vs claimed understanding
    pub async fn measure_understanding_gap(&self, content: &str, claimed_confidence: f64) -> Result<UnderstandingGap, String> {
        // Get actual understanding through rigorous testing
        let clothesline_confidence = self.get_clothesline_confidence(content).await?;
        let nicotine_confidence = self.get_nicotine_confidence(content).await?;
        let actual_confidence = (clothesline_confidence + nicotine_confidence) / 2.0;
        
        let gap = UnderstandingGap::new(claimed_confidence, actual_confidence);
        
        // Store gap for analysis
        self.understanding_gaps.lock().unwrap().push(gap.clone());
        self.calibration_history.lock().unwrap().push((claimed_confidence, actual_confidence, Instant::now()));
        
        // Log significant gaps
        if gap.is_critical() {
            warn!("🚨 Critical understanding gap detected: {} (Gap: {:.2})", 
                  gap.gap_type.description(), gap.gap_magnitude);
            self.initiate_gap_remediation(&gap).await?;
        } else {
            debug!("📊 Understanding gap: {} (Gap: {:.2})", 
                   gap.gap_type.description(), gap.gap_magnitude);
        }
        
        Ok(gap)
    }

    /// Measure distance from goals
    pub async fn measure_goal_distance(&self, content: &str) -> Result<f64, String> {
        // Simple goal distance calculation (would be enhanced with specific goal tracking)
        let goal_keywords = vec!["goal", "objective", "target", "achieve", "complete", "finish"];
        let content_lower = content.to_lowercase();
        
        let goal_mentions = goal_keywords.iter()
            .map(|&keyword| content_lower.matches(keyword).count())
            .sum::<usize>();
        
        // Simple heuristic: more goal-related content = closer to goal
        let goal_density = goal_mentions as f64 / content.split_whitespace().count() as f64;
        let distance = (1.0 - goal_density.min(1.0)).max(0.0);
        
        // Update goal metrics
        let mut goal_metrics = self.goal_metrics.lock().unwrap();
        let metric_key = format!("content_{}", content.len());
        
        if let Some(metric) = goal_metrics.get_mut(&metric_key) {
            metric.update_distance(distance);
        } else {
            let mut new_metric = GoalDistanceMetric::new(
                "Text processing goal".to_string(),
                content.chars().take(50).collect(),
                "Optimal understanding".to_string(),
            );
            new_metric.update_distance(distance);
            goal_metrics.insert(metric_key, new_metric);
        }
        
        Ok(distance)
    }

    /// Synthesize final truth energy (ATP production)
    pub async fn synthesize_final_truth_energy(&self, content: &str) -> Result<f64, String> {
        // Find active synthesis for this content
        let active_syntheses = self.active_syntheses.lock().unwrap();
        let synthesis = active_syntheses.values()
            .find(|s| s.initial_substrate.contains(content) || content.contains(&s.initial_substrate))
            .ok_or("No active synthesis found for content")?;
        
        let final_confidence = synthesis.synthesis_efficiency;
        
        // Update total ATP produced
        *self.total_atp_produced.lock().unwrap() += synthesis.final_atp_yield as u64;
        
        // Update average efficiency
        let completed_count = self.completed_syntheses.lock().unwrap().len() + 1;
        let current_average = *self.average_efficiency.lock().unwrap();
        let new_average = (current_average * (completed_count - 1) as f64 + final_confidence) / completed_count as f64;
        *self.average_efficiency.lock().unwrap() = new_average;
        
        info!("⚡ Final ATP synthesis: {:.2} confidence, {} ATP units", 
              final_confidence, synthesis.final_atp_yield);
        
        Ok(final_confidence)
    }

    async fn get_clothesline_confidence(&self, content: &str) -> Result<f64, String> {
        let clothesline = self.clothesline_module.lock().unwrap();
        clothesline.validate_genuine_understanding(content).await
    }

    async fn get_nicotine_confidence(&self, content: &str) -> Result<f64, String> {
        let nicotine = self.nicotine_module.lock().unwrap();
        // Simulate nicotine validation (would integrate with actual nicotine module)
        Ok(0.75) // Placeholder
    }

    async fn get_mzekezeke_confidence(&self, _content: &str) -> Result<f64, String> {
        // Simulate Mzekezeke Bayesian confidence
        Ok(0.8) // Placeholder
    }

    async fn get_hatata_confidence(&self, _content: &str) -> Result<f64, String> {
        // Simulate Hatata decision confidence
        Ok(0.7) // Placeholder
    }

    async fn log_synthesis_results(&self, process_id: Uuid, atp_yield: u32) -> Result<(), String> {
        let active_syntheses = self.active_syntheses.lock().unwrap();
        let synthesis = active_syntheses.get(&process_id)
            .ok_or("Synthesis not found")?;
        
        info!("🔬 ATP Synthesis Results for {}:", process_id);
        info!("   • Clothesline Confidence: {:.2}", synthesis.clothesline_confidence);
        info!("   • Nicotine Confidence: {:.2}", synthesis.nicotine_confidence);
        info!("   • Mzekezeke Confidence: {:.2}", synthesis.mzekezeke_confidence);
        info!("   • Hatata Confidence: {:.2}", synthesis.hatata_confidence);
        info!("   • Understanding Gap: {} ({:.2})", 
              synthesis.understanding_gap.gap_type.description(), 
              synthesis.understanding_gap.gap_magnitude);
        info!("   • Goal Distance: {:.2}", synthesis.goal_distance);
        info!("   • Synthesis Efficiency: {:.2}", synthesis.synthesis_efficiency);
        info!("   • Final ATP Yield: {} units", atp_yield);
        
        Ok(())
    }

    async fn initiate_gap_remediation(&self, gap: &UnderstandingGap) -> Result<(), String> {
        match gap.gap_type {
            GapType::Overconfidence => {
                warn!("🎯 Initiating overconfidence remediation (Dunning-Kruger mitigation)");
                // Implement overconfidence correction strategies
            }
            GapType::Underconfidence => {
                info!("💪 Initiating underconfidence remediation (confidence building)");
                // Implement confidence building strategies
            }
            GapType::Calibrated => {
                info!("✅ Understanding is well-calibrated, no remediation needed");
            }
        }
        
        Ok(())
    }

    /// Complete ATP synthesis and move to history
    pub async fn complete_synthesis(&self, process_id: Uuid) -> Result<AtpSynthesis, String> {
        let mut active_syntheses = self.active_syntheses.lock().unwrap();
        let synthesis = active_syntheses.remove(&process_id)
            .ok_or("Synthesis not found")?;
        
        // Move to completed syntheses
        self.completed_syntheses.lock().unwrap().push(synthesis.clone());
        
        // Update statistics
        let mut stats = self.stats.lock().unwrap();
        stats.increment_processed_count();
        stats.add_processing_time(synthesis.processing_time());
        
        info!("✅ Completed ATP synthesis {} with {} ATP yield", 
              process_id, synthesis.final_atp_yield);
        
        Ok(synthesis)
    }

    /// Get comprehensive Pungwe statistics
    pub fn get_pungwe_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let active_count = self.active_syntheses.lock().unwrap().len();
        let completed_count = self.completed_syntheses.lock().unwrap().len();
        let total_atp = *self.total_atp_produced.lock().unwrap();
        let average_efficiency = *self.average_efficiency.lock().unwrap();
        let gap_count = self.understanding_gaps.lock().unwrap().len();
        
        stats.insert("active_syntheses".to_string(), serde_json::Value::Number(active_count.into()));
        stats.insert("completed_syntheses".to_string(), serde_json::Value::Number(completed_count.into()));
        stats.insert("total_atp_produced".to_string(), serde_json::Value::Number(total_atp.into()));
        stats.insert("average_efficiency".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(average_efficiency).unwrap()));
        stats.insert("understanding_gaps_detected".to_string(), serde_json::Value::Number(gap_count.into()));
        
        // Calculate calibration accuracy
        let calibration_history = self.calibration_history.lock().unwrap();
        if !calibration_history.is_empty() {
            let total_calibration_error: f64 = calibration_history.iter()
                .map(|(claimed, actual, _)| (claimed - actual).abs())
                .sum();
            let average_calibration_error = total_calibration_error / calibration_history.len() as f64;
            stats.insert("average_calibration_error".to_string(), 
                       serde_json::Value::Number(serde_json::Number::from_f64(average_calibration_error).unwrap()));
        }
        
        stats
    }
}

#[async_trait]
impl StreamProcessor for PungweModule {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (output_tx, output_rx) = channel(100);
        
        let pungwe = Arc::new(self);
        tokio::spawn(async move {
            while let Some(data) = input.recv().await {
                match data {
                    StreamData::Text(content) => {
                        let process_id = Uuid::new_v4();
                        match pungwe.begin_atp_synthesis(process_id, &content).await {
                            Ok(confidence) => {
                                let gap_result = pungwe.measure_understanding_gap(&content, confidence).await;
                                let goal_distance = pungwe.measure_goal_distance(&content).await.unwrap_or(1.0);
                                
                                let output_data = StreamData::ProcessedText {
                                    content: content.clone(),
                                    metadata: {
                                        let mut meta = HashMap::new();
                                        meta.insert("process_id".to_string(), process_id.to_string());
                                        meta.insert("atp_confidence".to_string(), confidence.to_string());
                                        meta.insert("goal_distance".to_string(), goal_distance.to_string());
                                        meta.insert("processor".to_string(), "Pungwe".to_string());
                                        
                                        if let Ok(gap) = gap_result {
                                            meta.insert("understanding_gap".to_string(), gap.gap_magnitude.to_string());
                                            meta.insert("gap_type".to_string(), format!("{:?}", gap.gap_type));
                                        }
                                        
                                        meta
                                    },
                                    confidence: if confidence >= pungwe.synthesis_threshold {
                                        Confidence::High
                                    } else {
                                        Confidence::Medium
                                    },
                                };
                                
                                if output_tx.send(output_data).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Pungwe ATP synthesis error: {}", e);
                                if output_tx.send(StreamData::Error(e)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    other_data => {
                        if output_tx.send(other_data).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
        
        output_rx
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, data: &StreamData) -> bool {
        matches!(data, StreamData::Text(_))
    }

    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for PungweModule {
    fn default() -> Self {
        Self::new()
    }
} 