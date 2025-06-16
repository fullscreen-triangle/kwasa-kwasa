use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::{sleep, Duration, Instant};
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};
use super::v8_metabolism::V8MetabolismPipeline;

/// The three consciousness layers of the Tres Commas Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsciousnessLayer {
    Context,    // Foundation layer - understanding what is
    Reasoning,  // Processing layer - understanding why/how
    Intuition,  // Synthesis layer - understanding what could be
}

impl ConsciousnessLayer {
    pub fn depth(&self) -> u8 {
        match self {
            ConsciousnessLayer::Context => 1,
            ConsciousnessLayer::Reasoning => 2,
            ConsciousnessLayer::Intuition => 3,
        }
    }

    pub fn can_transition_to(&self, target: &ConsciousnessLayer) -> bool {
        match (self, target) {
            (ConsciousnessLayer::Context, ConsciousnessLayer::Reasoning) => true,
            (ConsciousnessLayer::Reasoning, ConsciousnessLayer::Intuition) => true,
            (ConsciousnessLayer::Intuition, ConsciousnessLayer::Context) => true, // Cycle back
            _ => false, // No direct jumps allowed
        }
    }
}

/// Represents the biological respiration cycle state
#[derive(Debug, Clone)]
pub struct RespirationState {
    pub oxygen_level: f64,        // Available processing capacity (0.0-1.0)
    pub carbon_dioxide: f64,      // Accumulated processing waste (0.0-1.0)
    pub breathing_rate: f64,      // Cycles per minute (10.0-30.0)
    pub lactate_level: f64,       // Incomplete processing buildup (0.0-1.0)
    pub is_anaerobic: bool,       // Emergency processing mode
    pub champagne_phase: bool,    // Dreaming/recovery mode
}

impl RespirationState {
    pub fn new() -> Self {
        Self {
            oxygen_level: 1.0,
            carbon_dioxide: 0.0,
            breathing_rate: 15.0,
            lactate_level: 0.0,
            is_anaerobic: false,
            champagne_phase: false,
        }
    }

    pub fn consume_oxygen(&mut self, amount: f64) {
        self.oxygen_level = (self.oxygen_level - amount).max(0.0);
        self.carbon_dioxide = (self.carbon_dioxide + amount * 0.8).min(1.0);
    }

    pub fn needs_respiration(&self) -> bool {
        self.oxygen_level < 0.3 || self.carbon_dioxide > 0.7
    }

    pub fn enter_anaerobic(&mut self) {
        self.is_anaerobic = true;
        self.breathing_rate = self.breathing_rate * 1.5;
    }

    pub fn accumulate_lactate(&mut self, amount: f64) {
        self.lactate_level = (self.lactate_level + amount).min(1.0);
    }

    pub fn needs_champagne_recovery(&self) -> bool {
        self.lactate_level > 0.6 || (self.is_anaerobic && self.carbon_dioxide > 0.8)
    }

    pub fn breathe(&mut self) {
        // Inspiration - fresh processing capacity
        let inspiration_amount = 0.3 * (self.breathing_rate / 15.0);
        self.oxygen_level = (self.oxygen_level + inspiration_amount).min(1.0);
        
        // Expiration - remove waste
        let expiration_amount = 0.4 * (self.breathing_rate / 15.0);
        self.carbon_dioxide = (self.carbon_dioxide - expiration_amount).max(0.0);
        
        // Recovery from anaerobic state
        if self.is_anaerobic && self.oxygen_level > 0.7 && self.carbon_dioxide < 0.3 {
            self.is_anaerobic = false;
            self.breathing_rate = 15.0; // Return to normal
        }
    }

    pub fn enter_champagne_phase(&mut self) {
        self.champagne_phase = true;
        self.breathing_rate = 8.0; // Slow deep breathing during dreams
        info!("🍾 Entering Champagne Phase - Dream Mode Activated");
    }

    pub fn exit_champagne_phase(&mut self) {
        self.champagne_phase = false;
        self.lactate_level = 0.0; // Complete recovery
        self.breathing_rate = 12.0; // Refreshed breathing
        info!("✨ Exiting Champagne Phase - Awakened with Insights");
    }
}

/// Represents a cognitive process moving through the trinity layers
#[derive(Debug, Clone)]
pub struct CognitiveProcess {
    pub id: Uuid,
    pub content: String,
    pub current_layer: ConsciousnessLayer,
    pub confidence: f64,
    pub atp_cost: u32,
    pub created_at: Instant,
    pub transitions: Vec<(ConsciousnessLayer, Instant, f64)>, // layer, time, confidence
    pub v8_processing_state: HashMap<String, serde_json::Value>,
}

impl CognitiveProcess {
    pub fn new(content: String) -> Self {
        let id = Uuid::new_v4();
        let now = Instant::now();
        
        Self {
            id,
            content,
            current_layer: ConsciousnessLayer::Context,
            confidence: 0.5,
            atp_cost: 2, // Base glycolysis cost
            created_at: now,
            transitions: vec![(ConsciousnessLayer::Context, now, 0.5)],
            v8_processing_state: HashMap::new(),
        }
    }

    pub fn transition_to(&mut self, layer: ConsciousnessLayer, confidence: f64) -> Result<(), String> {
        if !self.current_layer.can_transition_to(&layer) {
            return Err(format!(
                "Invalid transition from {:?} to {:?}", 
                self.current_layer, layer
            ));
        }

        self.current_layer = layer.clone();
        self.confidence = confidence;
        self.transitions.push((layer, Instant::now(), confidence));
        
        // ATP cost increases with layer depth
        self.atp_cost += match layer {
            ConsciousnessLayer::Context => 2,
            ConsciousnessLayer::Reasoning => 4,
            ConsciousnessLayer::Intuition => 8,
        };

        debug!("🧠 Process {} transitioned to {:?} (confidence: {:.2})", 
               self.id, self.current_layer, confidence);
        Ok(())
    }

    pub fn processing_time(&self) -> Duration {
        self.created_at.elapsed()
    }

    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.processing_time() > max_age
    }

    pub fn total_atp_yield(&self) -> u32 {
        // Theoretical maximum: 38 ATP from complete oxidation
        match self.transitions.len() {
            1 => 2,  // Glycolysis only
            2 => 8,  // Glycolysis + partial Krebs
            3 => 38, // Complete oxidation through all layers
            _ => 38, // Maximum theoretical yield
        }
    }
}

/// The Tres Commas Engine - Revolutionary Trinity-Based Cognitive Architecture
pub struct TresCommasEngine {
    name: String,
    
    // Trinity Layer Processors
    context_layer: Arc<Mutex<Vec<CognitiveProcess>>>,
    reasoning_layer: Arc<Mutex<Vec<CognitiveProcess>>>,
    intuition_layer: Arc<Mutex<Vec<CognitiveProcess>>>,
    
    // V8 Metabolism Pipeline
    v8_pipeline: Arc<Mutex<V8MetabolismPipeline>>,
    
    // Biological Systems
    respiration_state: Arc<Mutex<RespirationState>>,
    
    // Processing Configuration
    max_concurrent_processes: usize,
    layer_transition_threshold: f64,
    champagne_trigger_threshold: f64,
    
    // Monitoring
    stats: Arc<Mutex<ProcessorStats>>,
    process_history: Arc<Mutex<Vec<CognitiveProcess>>>,
}

impl TresCommasEngine {
    pub fn new() -> Self {
        Self {
            name: "TresCommasEngine".to_string(),
            context_layer: Arc::new(Mutex::new(Vec::new())),
            reasoning_layer: Arc::new(Mutex::new(Vec::new())),
            intuition_layer: Arc::new(Mutex::new(Vec::new())),
            v8_pipeline: Arc::new(Mutex::new(V8MetabolismPipeline::new())),
            respiration_state: Arc::new(Mutex::new(RespirationState::new())),
            max_concurrent_processes: 10,
            layer_transition_threshold: 0.7,
            champagne_trigger_threshold: 0.6,
            stats: Arc::new(Mutex::new(ProcessorStats::new())),
            process_history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_max_concurrent_processes(mut self, max: usize) -> Self {
        self.max_concurrent_processes = max;
        self
    }

    pub fn with_transition_threshold(mut self, threshold: f64) -> Self {
        self.layer_transition_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Start a new cognitive process in the Context layer
    pub async fn initiate_process(&self, content: String) -> Result<Uuid, String> {
        let process = CognitiveProcess::new(content.clone());
        let process_id = process.id;

        // Check if we can handle more processes
        let context_count = self.context_layer.lock().unwrap().len();
        if context_count >= self.max_concurrent_processes {
            return Err("Context layer at maximum capacity".to_string());
        }

        // Add to context layer
        self.context_layer.lock().unwrap().push(process);
        
        // Begin V8 metabolism processing
        let mut v8 = self.v8_pipeline.lock().unwrap();
        v8.begin_truth_glycolysis(&content).await?;

        // Consume oxygen for initial processing
        self.respiration_state.lock().unwrap().consume_oxygen(0.1);

        info!("🚀 Initiated cognitive process {} in Context layer", process_id);
        Ok(process_id)
    }

    /// Process transitions between consciousness layers
    pub async fn process_layer_transitions(&self) -> Result<(), String> {
        // Check Context → Reasoning transitions
        self.check_context_to_reasoning_transitions().await?;
        
        // Check Reasoning → Intuition transitions  
        self.check_reasoning_to_intuition_transitions().await?;
        
        // Check Intuition → Context completion cycles
        self.check_intuition_completion_cycles().await?;

        // Monitor respiration state
        self.monitor_respiration().await?;

        Ok(())
    }

    async fn check_context_to_reasoning_transitions(&self) -> Result<(), String> {
        let mut context_layer = self.context_layer.lock().unwrap();
        let mut reasoning_layer = self.reasoning_layer.lock().unwrap();
        let mut v8 = self.v8_pipeline.lock().unwrap();
        
        let mut transitions = Vec::new();
        
        for (index, process) in context_layer.iter_mut().enumerate() {
            // Check if Clothesline validation passed
            let clothesline_confidence = v8.get_clothesline_confidence(&process.content).await
                .unwrap_or(0.0);
                
            if clothesline_confidence > self.layer_transition_threshold {
                // Validated comprehension - can transition to Reasoning
                let reasoning_confidence = v8.begin_truth_krebs_cycle(&process.content).await?;
                
                process.transition_to(ConsciousnessLayer::Reasoning, reasoning_confidence)?;
                transitions.push(index);
                
                info!("🧠→💭 Process {} transitioned to Reasoning layer (confidence: {:.2})", 
                      process.id, reasoning_confidence);
            } else if process.processing_time() > Duration::from_secs(30) {
                // Timeout - move to anaerobic processing
                self.respiration_state.lock().unwrap().enter_anaerobic();
                self.respiration_state.lock().unwrap().accumulate_lactate(0.2);
                
                warn!("⚠️ Process {} failed Context validation, entering anaerobic mode", process.id);
            }
        }

        // Move validated processes to reasoning layer
        for &index in transitions.iter().rev() {
            let process = context_layer.remove(index);
            reasoning_layer.push(process);
        }

        Ok(())
    }

    async fn check_reasoning_to_intuition_transitions(&self) -> Result<(), String> {
        let mut reasoning_layer = self.reasoning_layer.lock().unwrap();
        let mut intuition_layer = self.intuition_layer.lock().unwrap();
        let mut v8 = self.v8_pipeline.lock().unwrap();
        
        let mut transitions = Vec::new();
        
        for (index, process) in reasoning_layer.iter_mut().enumerate() {
            // Check if reasoning processing is complete
            let reasoning_complete = v8.is_krebs_cycle_complete(&process.content).await
                .unwrap_or(false);
                
            if reasoning_complete {
                // Begin electron transport chain (Intuition layer)
                let intuition_confidence = v8.begin_electron_transport(&process.content).await?;
                
                process.transition_to(ConsciousnessLayer::Intuition, intuition_confidence)?;
                transitions.push(index);
                
                info!("💭→✨ Process {} transitioned to Intuition layer (confidence: {:.2})", 
                      process.id, intuition_confidence);
            }
        }

        // Move completed processes to intuition layer
        for &index in transitions.iter().rev() {
            let process = reasoning_layer.remove(index);
            intuition_layer.push(process);
        }

        Ok(())
    }

    async fn check_intuition_completion_cycles(&self) -> Result<(), String> {
        let mut intuition_layer = self.intuition_layer.lock().unwrap();
        let mut process_history = self.process_history.lock().unwrap();
        let mut v8 = self.v8_pipeline.lock().unwrap();
        
        let mut completed = Vec::new();
        
        for (index, process) in intuition_layer.iter_mut().enumerate() {
            // Check if intuition synthesis is complete
            let synthesis_complete = v8.is_electron_transport_complete(&process.content).await
                .unwrap_or(false);
                
            if synthesis_complete {
                // Calculate final ATP yield
                let final_atp = process.total_atp_yield();
                
                info!("✨ Process {} completed full trinity cycle! ATP yield: {} units", 
                      process.id, final_atp);
                
                // Move to completed history
                completed.push(index);
                
                // Update statistics
                let mut stats = self.stats.lock().unwrap();
                stats.increment_processed_count();
                stats.add_processing_time(process.processing_time());
            }
        }

        // Archive completed processes
        for &index in completed.iter().rev() {
            let process = intuition_layer.remove(index);
            process_history.push(process);
        }

        Ok(())
    }

    async fn monitor_respiration(&self) -> Result<(), String> {
        let mut respiration = self.respiration_state.lock().unwrap();
        
        // Natural breathing cycle
        respiration.breathe();
        
        // Check if champagne phase is needed
        if respiration.needs_champagne_recovery() && !respiration.champagne_phase {
            respiration.enter_champagne_phase();
            
            // Spawn champagne processing task
            let v8_pipeline = Arc::clone(&self.v8_pipeline);
            let respiration_state = Arc::clone(&self.respiration_state);
            
            tokio::spawn(async move {
                // Champagne phase processing
                sleep(Duration::from_secs(2)).await; // Dream time
                
                let mut v8 = v8_pipeline.lock().unwrap();
                let recovery_result = v8.process_champagne_recovery().await;
                
                match recovery_result {
                    Ok(insights) => {
                        info!("🍾 Champagne phase recovered {} insights", insights.len());
                        respiration_state.lock().unwrap().exit_champagne_phase();
                    }
                    Err(e) => {
                        warn!("Champagne phase error: {}", e);
                        respiration_state.lock().unwrap().exit_champagne_phase();
                    }
                }
            });
        }

        Ok(())
    }

    /// Get current engine status
    pub fn get_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();
        
        let context_count = self.context_layer.lock().unwrap().len();
        let reasoning_count = self.reasoning_layer.lock().unwrap().len();
        let intuition_count = self.intuition_layer.lock().unwrap().len();
        let history_count = self.process_history.lock().unwrap().len();
        
        let respiration = self.respiration_state.lock().unwrap();
        
        status.insert("name".to_string(), serde_json::Value::String(self.name.clone()));
        status.insert("context_processes".to_string(), serde_json::Value::Number(context_count.into()));
        status.insert("reasoning_processes".to_string(), serde_json::Value::Number(reasoning_count.into()));
        status.insert("intuition_processes".to_string(), serde_json::Value::Number(intuition_count.into()));
        status.insert("completed_processes".to_string(), serde_json::Value::Number(history_count.into()));
        
        status.insert("oxygen_level".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(respiration.oxygen_level).unwrap()));
        status.insert("lactate_level".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(respiration.lactate_level).unwrap()));
        status.insert("breathing_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(respiration.breathing_rate).unwrap()));
        status.insert("champagne_phase".to_string(), serde_json::Value::Bool(respiration.champagne_phase));
        
        status
    }
}

#[async_trait]
impl StreamProcessor for TresCommasEngine {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (output_tx, output_rx) = channel(100);
        
        let engine = Arc::new(self);
        let engine_clone = Arc::clone(&engine);
        
        tokio::spawn(async move {
            // Background task for layer transitions
            let transition_engine = Arc::clone(&engine_clone);
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(500));
                loop {
                    interval.tick().await;
                    if let Err(e) = transition_engine.process_layer_transitions().await {
                        error!("Layer transition error: {}", e);
                    }
                }
            });
            
            // Main processing loop
            while let Some(data) = input.recv().await {
                match data {
                    StreamData::Text(content) => {
                        match engine_clone.initiate_process(content.clone()).await {
                            Ok(process_id) => {
                                let output_data = StreamData::ProcessedText {
                                    content: content.clone(),
                                    metadata: {
                                        let mut meta = HashMap::new();
                                        meta.insert("process_id".to_string(), process_id.to_string());
                                        meta.insert("layer".to_string(), "Context".to_string());
                                        meta.insert("engine".to_string(), "TresCommas".to_string());
                                        meta
                                    },
                                    confidence: Confidence::Medium,
                                };
                                
                                if output_tx.send(output_data).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to initiate process: {}", e);
                                if output_tx.send(StreamData::Error(e)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    other_data => {
                        // Pass through non-text data
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

impl Default for TresCommasEngine {
    fn default() -> Self {
        Self::new()
    }
}

pub fn demonstrate_tres_commas() {
    println!("🚀 Tres Commas Engine - Revolutionary Trinity Architecture!");
    println!("🧠 Context → Reasoning → Intuition");
    println!("🔋 V8 Metabolism: Truth processing through biological cycles");
    println!("🍾 Champagne Phase: Dream mode for lactate recovery");
    println!("🫁 Biological Respiration: Oxygen/CO2 management");
    
    let mut engine = TresCommasEngine::new();
    let _ = engine.initiate_process("Test cognitive process".to_string());
    
    let status = engine.get_status();
    println!("📊 Engine Status: {:?}", status);
    
    println!("✨ The first truly biological artificial intelligence system!");
} 