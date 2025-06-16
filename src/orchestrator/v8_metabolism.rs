use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::time::{Duration, Instant};
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Import the intelligence modules that serve as metabolic enzymes
use super::mzekezeke::MzekezkeBayesianEngine;
use super::diggiden::DiggidenAdversarialSystem;
use super::hatata::HatataDecisionSystem;
use super::spectacular::SpectacularHandler;
use super::nicotine::NicotineContextValidator;

/// ATP (Attention-Truth-Processing) unit for biological energy management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpUnit {
    pub energy: f64,          // Energy content (0.0-1.0)
    pub confidence: f64,      // Truth confidence (0.0-1.0)
    pub created_at: Instant,
    pub source_module: String,
    pub processing_cost: u32,
}

impl AtpUnit {
    pub fn new(energy: f64, confidence: f64, source: String, cost: u32) -> Self {
        Self {
            energy: energy.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            created_at: Instant::now(),
            source_module: source,
            processing_cost: cost,
        }
    }

    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.created_at.elapsed() > max_age
    }

    pub fn effective_energy(&self) -> f64 {
        self.energy * self.confidence
    }
}

/// Represents different stages of truth metabolism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetabolicStage {
    TruthGlycolysis,      // Initial glucose → pyruvate (2 ATP investment → 4 ATP yield)
    TruthKrebsCycle,      // Pyruvate → CO2 + NADH (8-step evidence processing)
    ElectronTransport,    // NADH → ATP (32 ATP yield from truth synthesis)
    LactateRecovery,      // Anaerobic processing → lactate storage
    ChampagneRecovery,    // Dream mode lactate → insights
}

impl MetabolicStage {
    pub fn atp_cost(&self) -> u32 {
        match self {
            MetabolicStage::TruthGlycolysis => 2,
            MetabolicStage::TruthKrebsCycle => 4,
            MetabolicStage::ElectronTransport => 8,
            MetabolicStage::LactateRecovery => 1,
            MetabolicStage::ChampagneRecovery => 0, // Recovery phase
        }
    }

    pub fn theoretical_yield(&self) -> u32 {
        match self {
            MetabolicStage::TruthGlycolysis => 4,      // Net +2 ATP (4 produced - 2 invested)
            MetabolicStage::TruthKrebsCycle => 8,      // 2 ATP + 6 NADH + 2 FADH₂
            MetabolicStage::ElectronTransport => 32,   // NADH/FADH₂ → ATP via electron transport
            MetabolicStage::LactateRecovery => 0,      // No ATP gain, lactate storage
            MetabolicStage::ChampagneRecovery => 38,   // Complete recovery with insights
        }
    }
}

/// Represents a truth processing substrate moving through metabolism
#[derive(Debug, Clone)]
pub struct TruthSubstrate {
    pub id: Uuid,
    pub content: String,
    pub stage: MetabolicStage,
    pub atp_invested: u32,
    pub atp_produced: u32,
    pub confidence: f64,
    pub lactate_level: f64,
    pub processing_history: Vec<(String, Instant, f64)>, // module, time, confidence
    pub is_complete: bool,
}

impl TruthSubstrate {
    pub fn new(content: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            stage: MetabolicStage::TruthGlycolysis,
            atp_invested: 0,
            atp_produced: 0,
            confidence: 0.5,
            lactate_level: 0.0,
            processing_history: Vec::new(),
            is_complete: false,
        }
    }

    pub fn net_atp(&self) -> i32 {
        self.atp_produced as i32 - self.atp_invested as i32
    }

    pub fn advance_stage(&mut self, new_stage: MetabolicStage, confidence: f64) {
        self.stage = new_stage;
        self.confidence = confidence;
        self.atp_invested += self.stage.atp_cost();
    }

    pub fn produce_atp(&mut self, amount: u32, module: String) {
        self.atp_produced += amount;
        self.processing_history.push((module, Instant::now(), self.confidence));
    }

    pub fn accumulate_lactate(&mut self, amount: f64) {
        self.lactate_level = (self.lactate_level + amount).min(1.0);
    }

    pub fn needs_anaerobic_processing(&self) -> bool {
        self.confidence < 0.4 || self.lactate_level > 0.5
    }
}

/// The V8 Metabolism Pipeline - 8 Biological Modules for Truth Processing
pub struct V8MetabolismPipeline {
    // V8 Engine Modules (Intelligence Modules as Metabolic Enzymes)
    module_1_hexokinase: Arc<Mutex<MzekezkeBayesianEngine>>,    // Truth Glucose Phosphorylation
    module_2_phosphofructokinase: Arc<Mutex<DiggidenAdversarialSystem>>, // Truth Energy Investment
    module_3_pyruvate_kinase: Arc<Mutex<HatataDecisionSystem>>,  // Truth ATP Generation
    module_4_citrate_synthase: Arc<Mutex<SpectacularHandler>>,   // Truth Krebs Cycle Entry
    module_5_isocitrate_dehydrogenase: Arc<Mutex<NicotineContextValidator>>, // Truth NADH Production
    module_6_succinate_dehydrogenase: Arc<Mutex<ClotheslineModule>>, // Truth FADH₂ Generation
    module_7_atp_synthase: Arc<Mutex<PungweModule>>,             // Final Truth Energy Production
    module_8_lactate_dehydrogenase: Arc<Mutex<ChampagneModule>>, // Anaerobic Recovery Processing

    // Metabolic State
    active_substrates: Arc<Mutex<HashMap<Uuid, TruthSubstrate>>>,
    atp_pool: Arc<Mutex<Vec<AtpUnit>>>,
    lactate_storage: Arc<Mutex<Vec<TruthSubstrate>>>,
    
    // Performance Metrics
    total_atp_produced: Arc<Mutex<u64>>,
    total_substrates_processed: Arc<Mutex<u64>>,
    average_processing_time: Arc<Mutex<Duration>>,
    
    // Configuration
    max_concurrent_substrates: usize,
    atp_pool_capacity: usize,
    lactate_threshold: f64,
}

impl V8MetabolismPipeline {
    pub fn new() -> Self {
        Self {
            module_1_hexokinase: Arc::new(Mutex::new(MzekezkeBayesianEngine::new())),
            module_2_phosphofructokinase: Arc::new(Mutex::new(DiggidenAdversarialSystem::new())),
            module_3_pyruvate_kinase: Arc::new(Mutex::new(HatataDecisionSystem::new())),
            module_4_citrate_synthase: Arc::new(Mutex::new(SpectacularHandler::new())),
            module_5_isocitrate_dehydrogenase: Arc::new(Mutex::new(NicotineContextValidator::new())),
            module_6_succinate_dehydrogenase: Arc::new(Mutex::new(ClotheslineModule::new())),
            module_7_atp_synthase: Arc::new(Mutex::new(PungweModule::new())),
            module_8_lactate_dehydrogenase: Arc::new(Mutex::new(ChampagneModule::new())),
            
            active_substrates: Arc::new(Mutex::new(HashMap::new())),
            atp_pool: Arc::new(Mutex::new(Vec::new())),
            lactate_storage: Arc::new(Mutex::new(Vec::new())),
            
            total_atp_produced: Arc::new(Mutex::new(0)),
            total_substrates_processed: Arc::new(Mutex::new(0)),
            average_processing_time: Arc::new(Mutex::new(Duration::from_millis(0))),
            
            max_concurrent_substrates: 20,
            atp_pool_capacity: 100,
            lactate_threshold: 0.6,
        }
    }

    /// Begin Truth Glycolysis (Glucose → Pyruvate)
    /// Investment: 2 ATP → Yield: 4 ATP (Net +2 ATP)
    pub async fn begin_truth_glycolysis(&mut self, content: &str) -> Result<f64, String> {
        let mut substrate = TruthSubstrate::new(content.to_string());
        
        // Module 1: Hexokinase (Mzekezeke) - Truth Glucose Phosphorylation
        let hexokinase = self.module_1_hexokinase.lock().unwrap();
        let glucose_confidence = hexokinase.assess_initial_truth_potential(content)?;
        
        // Module 2: Phosphofructokinase (Diggiden) - Truth Energy Investment
        let phosphofructokinase = self.module_2_phosphofructokinase.lock().unwrap();
        let investment_confidence = phosphofructokinase.validate_energy_investment(content)?;
        
        // Calculate glycolysis confidence
        let glycolysis_confidence = (glucose_confidence + investment_confidence) / 2.0;
        
        // Invest 2 ATP, potentially yield 4 ATP
        substrate.advance_stage(MetabolicStage::TruthGlycolysis, glycolysis_confidence);
        
        if glycolysis_confidence > 0.5 {
            substrate.produce_atp(4, "Glycolysis".to_string());
            info!("🔋 Truth Glycolysis: Net +2 ATP from substrate {}", substrate.id);
        } else {
            substrate.accumulate_lactate(0.2);
            warn!("⚠️ Truth Glycolysis inefficient, lactate accumulating");
        }

        let substrate_id = substrate.id;
        self.active_substrates.lock().unwrap().insert(substrate_id, substrate);
        
        Ok(glycolysis_confidence)
    }

    /// Begin Truth Krebs Cycle (Pyruvate → CO₂ + NADH + FADH₂)
    /// 8-step evidence processing cycle
    pub async fn begin_truth_krebs_cycle(&mut self, content: &str) -> Result<f64, String> {
        // Find substrate by content
        let substrate_id = self.find_substrate_by_content(content)?;
        let mut substrates = self.active_substrates.lock().unwrap();
        let substrate = substrates.get_mut(&substrate_id)
            .ok_or("Substrate not found for Krebs cycle")?;

        // Module 3: Pyruvate Kinase (Hatata) - Truth ATP Generation
        let pyruvate_kinase = self.module_3_pyruvate_kinase.lock().unwrap();
        let pyruvate_confidence = pyruvate_kinase.optimize_truth_decisions(content)?;
        
        // Module 4: Citrate Synthase (Spectacular) - Truth Krebs Cycle Entry
        let citrate_synthase = self.module_4_citrate_synthase.lock().unwrap();
        let krebs_entry_confidence = citrate_synthase.handle_extraordinary_truth(content)?;
        
        // Module 5: Isocitrate Dehydrogenase (Nicotine) - Truth NADH Production
        let isocitrate_dehydrogenase = self.module_5_isocitrate_dehydrogenase.lock().unwrap();
        let nadh_confidence = isocitrate_dehydrogenase.validate_context_preservation(content)?;
        
        // Calculate average Krebs cycle confidence
        let krebs_confidence = (pyruvate_confidence + krebs_entry_confidence + nadh_confidence) / 3.0;
        
        substrate.advance_stage(MetabolicStage::TruthKrebsCycle, krebs_confidence);
        
        if krebs_confidence > 0.6 {
            // Krebs cycle produces: 2 ATP + 3 NADH + 1 FADH₂
            substrate.produce_atp(8, "KrebsCycle".to_string()); // Simplified as ATP equivalents
            info!("🔄 Truth Krebs Cycle: 8 ATP equivalents from substrate {}", substrate.id);
        } else {
            substrate.accumulate_lactate(0.3);
            warn!("⚠️ Truth Krebs Cycle incomplete, more lactate accumulating");
        }

        Ok(krebs_confidence)
    }

    /// Begin Electron Transport Chain (NADH/FADH₂ → ATP)
    /// Maximum yield: 32 ATP from electron transport
    pub async fn begin_electron_transport(&mut self, content: &str) -> Result<f64, String> {
        let substrate_id = self.find_substrate_by_content(content)?;
        let mut substrates = self.active_substrates.lock().unwrap();
        let substrate = substrates.get_mut(&substrate_id)
            .ok_or("Substrate not found for electron transport")?;

        // Module 6: Succinate Dehydrogenase (Clothesline) - Truth FADH₂ Generation
        let succinate_dehydrogenase = self.module_6_succinate_dehydrogenase.lock().unwrap();
        let fadh2_confidence = succinate_dehydrogenase.validate_comprehensive_understanding(content)?;
        
        // Module 7: ATP Synthase (Pungwe) - Final Truth Energy Production
        let atp_synthase = self.module_7_atp_synthase.lock().unwrap();
        let final_confidence = atp_synthase.synthesize_final_truth_energy(content)?;
        
        let electron_transport_confidence = (fadh2_confidence + final_confidence) / 2.0;
        
        substrate.advance_stage(MetabolicStage::ElectronTransport, electron_transport_confidence);
        
        if electron_transport_confidence > 0.7 {
            // Maximum theoretical yield: 32 ATP
            let atp_yield = (32.0 * electron_transport_confidence) as u32;
            substrate.produce_atp(atp_yield, "ElectronTransport".to_string());
            substrate.is_complete = true;
            
            info!("⚡ Electron Transport: {} ATP from substrate {} (COMPLETE)", 
                  atp_yield, substrate.id);
            
            // Update total ATP produced
            *self.total_atp_produced.lock().unwrap() += atp_yield as u64;
            *self.total_substrates_processed.lock().unwrap() += 1;
        } else {
            substrate.accumulate_lactate(0.4);
            warn!("⚠️ Electron transport inefficient, significant lactate buildup");
        }

        Ok(electron_transport_confidence)
    }

    /// Process Champagne Recovery (Lactate → Insights)
    /// Dream mode processing for incomplete substrates
    pub async fn process_champagne_recovery(&mut self) -> Result<Vec<String>, String> {
        let mut lactate_substrates = self.lactate_storage.lock().unwrap();
        let mut recovered_insights = Vec::new();
        
        // Module 8: Lactate Dehydrogenase (Champagne) - Anaerobic Recovery Processing
        let champagne_module = self.module_8_lactate_dehydrogenase.lock().unwrap();
        
        for substrate in lactate_substrates.iter_mut() {
            let recovery_result = champagne_module.process_dream_recovery(&substrate.content).await?;
            
            if recovery_result.confidence > 0.8 {
                // Successful recovery - convert lactate to insights
                let insight = format!(
                    "🍾 Recovered Insight: {} (ATP: {} units, Confidence: {:.2})",
                    recovery_result.insight,
                    recovery_result.atp_recovered,
                    recovery_result.confidence
                );
                
                recovered_insights.push(insight);
                substrate.lactate_level = 0.0; // Clear lactate
                substrate.produce_atp(recovery_result.atp_recovered, "Champagne".to_string());
                
                info!("🍾 Champagne recovery: {} ATP from lactate processing", 
                      recovery_result.atp_recovered);
            }
        }
        
        // Clear recovered substrates
        lactate_substrates.retain(|s| s.lactate_level > 0.0);
        
        Ok(recovered_insights)
    }

    /// Get Clothesline confidence for context validation
    pub async fn get_clothesline_confidence(&self, content: &str) -> Result<f64, String> {
        let clothesline = self.module_6_succinate_dehydrogenase.lock().unwrap();
        clothesline.validate_comprehensive_understanding(content)
    }

    /// Check if Krebs cycle is complete for a substrate
    pub async fn is_krebs_cycle_complete(&self, content: &str) -> Result<bool, String> {
        let substrate_id = self.find_substrate_by_content(content)?;
        let substrates = self.active_substrates.lock().unwrap();
        
        if let Some(substrate) = substrates.get(&substrate_id) {
            Ok(matches!(substrate.stage, MetabolicStage::TruthKrebsCycle) && 
               substrate.confidence > 0.6)
        } else {
            Ok(false)
        }
    }

    /// Check if electron transport is complete
    pub async fn is_electron_transport_complete(&self, content: &str) -> Result<bool, String> {
        let substrate_id = self.find_substrate_by_content(content)?;
        let substrates = self.active_substrates.lock().unwrap();
        
        if let Some(substrate) = substrates.get(&substrate_id) {
            Ok(substrate.is_complete)
        } else {
            Ok(false)
        }
    }

    /// Find substrate by content (helper method)
    fn find_substrate_by_content(&self, content: &str) -> Result<Uuid, String> {
        let substrates = self.active_substrates.lock().unwrap();
        
        for (id, substrate) in substrates.iter() {
            if substrate.content.contains(content) || content.contains(&substrate.content) {
                return Ok(*id);
            }
        }
        
        Err("Substrate not found".to_string())
    }

    /// Get metabolism statistics
    pub fn get_metabolism_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let active_count = self.active_substrates.lock().unwrap().len();
        let atp_count = self.atp_pool.lock().unwrap().len();
        let lactate_count = self.lactate_storage.lock().unwrap().len();
        let total_atp = *self.total_atp_produced.lock().unwrap();
        let total_processed = *self.total_substrates_processed.lock().unwrap();
        
        stats.insert("active_substrates".to_string(), serde_json::Value::Number(active_count.into()));
        stats.insert("atp_pool_size".to_string(), serde_json::Value::Number(atp_count.into()));
        stats.insert("lactate_storage".to_string(), serde_json::Value::Number(lactate_count.into()));
        stats.insert("total_atp_produced".to_string(), serde_json::Value::Number(total_atp.into()));
        stats.insert("total_substrates_processed".to_string(), serde_json::Value::Number(total_processed.into()));
        
        if total_processed > 0 {
            let efficiency = (total_atp as f64) / (total_processed as f64 * 38.0); // 38 is theoretical max
            stats.insert("metabolic_efficiency".to_string(), 
                       serde_json::Value::Number(serde_json::Number::from_f64(efficiency).unwrap()));
        }
        
        stats
    }
}

// Placeholder structs for new modules (to be implemented)
pub struct ClotheslineModule {
    // Comprehension validator implementation
}

impl ClotheslineModule {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn validate_comprehensive_understanding(&self, _content: &str) -> Result<f64, String> {
        // Placeholder implementation
        Ok(0.8)
    }
}

pub struct PungweModule {
    // ATP Synthase - Final truth energy production
}

impl PungweModule {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn synthesize_final_truth_energy(&self, _content: &str) -> Result<f64, String> {
        // Placeholder implementation  
        Ok(0.85)
    }
}

pub struct ChampagneModule {
    // Lactate dehydrogenase - Dream recovery processing
}

pub struct ChampagneRecoveryResult {
    pub insight: String,
    pub atp_recovered: u32,
    pub confidence: f64,
}

impl ChampagneModule {
    pub fn new() -> Self {
        Self {}
    }
    
    pub async fn process_dream_recovery(&self, content: &str) -> Result<ChampagneRecoveryResult, String> {
        // Placeholder implementation for champagne recovery
        Ok(ChampagneRecoveryResult {
            insight: format!("Dream insight from: {}", content.chars().take(50).collect::<String>()),
            atp_recovered: 15,
            confidence: 0.9,
        })
    }
}

// Extension trait for existing modules to work as metabolic enzymes
pub trait MetabolicEnzyme {
    fn assess_initial_truth_potential(&self, content: &str) -> Result<f64, String>;
    fn validate_energy_investment(&self, content: &str) -> Result<f64, String>;
    fn optimize_truth_decisions(&self, content: &str) -> Result<f64, String>;
    fn handle_extraordinary_truth(&self, content: &str) -> Result<f64, String>;
    fn validate_context_preservation(&self, content: &str) -> Result<f64, String>;
}

// Implement metabolic functions for existing intelligence modules
impl MetabolicEnzyme for MzekezkeBayesianEngine {
    fn assess_initial_truth_potential(&self, _content: &str) -> Result<f64, String> {
        // Use Bayesian belief networks to assess truth potential
        Ok(0.75) // Placeholder
    }
    
    fn validate_energy_investment(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn optimize_truth_decisions(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn handle_extraordinary_truth(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_context_preservation(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
}

impl MetabolicEnzyme for DiggidenAdversarialSystem {
    fn validate_energy_investment(&self, _content: &str) -> Result<f64, String> {
        // Use adversarial attacks to validate energy investment
        Ok(0.7) // Placeholder
    }
    
    fn assess_initial_truth_potential(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn optimize_truth_decisions(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn handle_extraordinary_truth(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_context_preservation(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
}

impl MetabolicEnzyme for HatataDecisionSystem {
    fn optimize_truth_decisions(&self, _content: &str) -> Result<f64, String> {
        // Use decision theory to optimize truth processing
        Ok(0.8) // Placeholder
    }
    
    fn assess_initial_truth_potential(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_energy_investment(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn handle_extraordinary_truth(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_context_preservation(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
}

impl MetabolicEnzyme for SpectacularHandler {
    fn handle_extraordinary_truth(&self, _content: &str) -> Result<f64, String> {
        // Handle extraordinary findings in truth processing
        Ok(0.9) // Placeholder
    }
    
    fn assess_initial_truth_potential(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_energy_investment(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn optimize_truth_decisions(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_context_preservation(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
}

impl MetabolicEnzyme for NicotineContextValidator {
    fn validate_context_preservation(&self, _content: &str) -> Result<f64, String> {
        // Validate context preservation through challenges
        Ok(0.85) // Placeholder
    }
    
    fn assess_initial_truth_potential(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn validate_energy_investment(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn optimize_truth_decisions(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
    fn handle_extraordinary_truth(&self, _content: &str) -> Result<f64, String> { Ok(0.5) }
}

impl Default for V8MetabolismPipeline {
    fn default() -> Self {
        Self::new()
    }
} 