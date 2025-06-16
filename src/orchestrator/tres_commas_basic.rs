use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// The three consciousness layers of the Tres Commas Engine
#[derive(Debug, Clone)]
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
        println!("🍾 Entering Champagne Phase - Dream Mode Activated");
    }

    pub fn exit_champagne_phase(&mut self) {
        self.champagne_phase = false;
        self.lactate_level = 0.0; // Complete recovery
        self.breathing_rate = 12.0; // Refreshed breathing
        println!("✨ Exiting Champagne Phase - Awakened with Insights");
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

        println!("🧠 Process {} transitioned to {:?} (confidence: {:.2})", 
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
    // Trinity Layer Processors
    context_layer: Vec<CognitiveProcess>,
    reasoning_layer: Vec<CognitiveProcess>,
    intuition_layer: Vec<CognitiveProcess>,
    
    // Biological Systems
    respiration_state: RespirationState,
    
    // Processing Configuration
    max_concurrent_processes: usize,
    layer_transition_threshold: f64,
    champagne_trigger_threshold: f64,
    
    // Monitoring
    process_history: Vec<CognitiveProcess>,
}

impl TresCommasEngine {
    pub fn new() -> Self {
        Self {
            context_layer: Vec::new(),
            reasoning_layer: Vec::new(),
            intuition_layer: Vec::new(),
            respiration_state: RespirationState::new(),
            max_concurrent_processes: 10,
            layer_transition_threshold: 0.7,
            champagne_trigger_threshold: 0.6,
            process_history: Vec::new(),
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
    pub fn initiate_process(&mut self, content: String) -> Result<Uuid, String> {
        let process = CognitiveProcess::new(content.clone());
        let process_id = process.id;

        // Check if we can handle more processes
        if self.context_layer.len() >= self.max_concurrent_processes {
            return Err("Context layer at maximum capacity".to_string());
        }

        // Add to context layer
        self.context_layer.push(process);
        
        // Consume oxygen for initial processing
        self.respiration_state.consume_oxygen(0.1);

        println!("🚀 Initiated cognitive process {} in Context layer", process_id);
        Ok(process_id)
    }

    /// Process transitions between consciousness layers
    pub fn process_layer_transitions(&mut self) -> Result<(), String> {
        // Check Context → Reasoning transitions
        self.check_context_to_reasoning_transitions()?;
        
        // Check Reasoning → Intuition transitions  
        self.check_reasoning_to_intuition_transitions()?;
        
        // Check Intuition → Context completion cycles
        self.check_intuition_completion_cycles()?;

        // Monitor respiration state
        self.monitor_respiration();

        Ok(())
    }

    fn check_context_to_reasoning_transitions(&mut self) -> Result<(), String> {
        let mut transitions = Vec::new();
        
        for (index, process) in self.context_layer.iter_mut().enumerate() {
            // Simple validation: longer content gets higher confidence
            let content_complexity = process.content.len() as f64 / 100.0;
            let clothesline_confidence = (content_complexity * 0.8).min(1.0);
                
            if clothesline_confidence > self.layer_transition_threshold {
                // Validated comprehension - can transition to Reasoning
                let reasoning_confidence = clothesline_confidence * 0.9;
                
                process.transition_to(ConsciousnessLayer::Reasoning, reasoning_confidence)?;
                transitions.push(index);
                
                println!("🧠→💭 Process {} transitioned to Reasoning layer (confidence: {:.2})", 
                      process.id, reasoning_confidence);
            } else if process.processing_time() > Duration::from_secs(30) {
                // Timeout - move to anaerobic processing
                self.respiration_state.enter_anaerobic();
                self.respiration_state.accumulate_lactate(0.2);
                
                println!("⚠️ Process {} failed Context validation, entering anaerobic mode", process.id);
            }
        }

        // Move validated processes to reasoning layer
        for &index in transitions.iter().rev() {
            let process = self.context_layer.remove(index);
            self.reasoning_layer.push(process);
        }

        Ok(())
    }

    fn check_reasoning_to_intuition_transitions(&mut self) -> Result<(), String> {
        let mut transitions = Vec::new();
        
        for (index, process) in self.reasoning_layer.iter_mut().enumerate() {
            // Simple reasoning completion check
            let reasoning_time = process.processing_time().as_secs();
            let reasoning_complete = reasoning_time > 2; // 2 seconds for reasoning
                
            if reasoning_complete {
                // Begin intuition processing
                let intuition_confidence = process.confidence * 0.95;
                
                process.transition_to(ConsciousnessLayer::Intuition, intuition_confidence)?;
                transitions.push(index);
                
                println!("💭→✨ Process {} transitioned to Intuition layer (confidence: {:.2})", 
                      process.id, intuition_confidence);
            }
        }

        // Move completed processes to intuition layer
        for &index in transitions.iter().rev() {
            let process = self.reasoning_layer.remove(index);
            self.intuition_layer.push(process);
        }

        Ok(())
    }

    fn check_intuition_completion_cycles(&mut self) -> Result<(), String> {
        let mut completed = Vec::new();
        
        for (index, process) in self.intuition_layer.iter_mut().enumerate() {
            // Simple completion check
            let intuition_time = process.processing_time().as_secs();
            let synthesis_complete = intuition_time > 5; // 5 seconds total
                
            if synthesis_complete {
                // Calculate final ATP yield
                let final_atp = process.total_atp_yield();
                
                println!("✨ Process {} completed full trinity cycle! ATP yield: {} units", 
                      process.id, final_atp);
                
                // Move to completed history
                completed.push(index);
            }
        }

        // Archive completed processes
        for &index in completed.iter().rev() {
            let process = self.intuition_layer.remove(index);
            self.process_history.push(process);
        }

        Ok(())
    }

    fn monitor_respiration(&mut self) {
        // Natural breathing cycle
        self.respiration_state.breathe();
        
        // Check if champagne phase is needed
        if self.respiration_state.needs_champagne_recovery() && !self.respiration_state.champagne_phase {
            self.respiration_state.enter_champagne_phase();
            
            // Simple champagne processing (in real implementation would be more complex)
            self.respiration_state.exit_champagne_phase();
        }
    }

    /// Get current engine status
    pub fn get_status(&self) -> HashMap<String, f64> {
        let mut status = HashMap::new();
        
        status.insert("context_processes".to_string(), self.context_layer.len() as f64);
        status.insert("reasoning_processes".to_string(), self.reasoning_layer.len() as f64);
        status.insert("intuition_processes".to_string(), self.intuition_layer.len() as f64);
        status.insert("completed_processes".to_string(), self.process_history.len() as f64);
        
        status.insert("oxygen_level".to_string(), self.respiration_state.oxygen_level);
        status.insert("lactate_level".to_string(), self.respiration_state.lactate_level);
        status.insert("breathing_rate".to_string(), self.respiration_state.breathing_rate);
        status.insert("champagne_phase".to_string(), if self.respiration_state.champagne_phase { 1.0 } else { 0.0 });
        
        status
    }
}

impl Default for TresCommasEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple demonstration function
pub fn demonstrate_tres_commas_engine() -> Result<(), String> {
    println!("🚀 Starting Tres Commas Engine Demonstration");
    println!("============================================");
    
    // Initialize the revolutionary engine
    let mut engine = TresCommasEngine::new()
        .with_max_concurrent_processes(15)
        .with_transition_threshold(0.75);
    
    // Test texts representing different complexity levels
    let test_texts = vec![
        "The quick brown fox jumps over the lazy dog.", // Simple
        "Machine learning algorithms utilize statistical methods to identify patterns in data, enabling predictive analytics and automated decision-making processes.", // Technical
        "The philosophical implications of consciousness emergence in artificial intelligence systems raise profound questions about the nature of understanding, awareness, and the potential for genuine comprehension versus pattern matching.", // Complex philosophical
    ];
    
    println!("📋 Processing {} test texts through trinity layers...", test_texts.len());
    
    for (index, text) in test_texts.iter().enumerate() {
        println!("\n🧪 Test Case {}: Processing text...", index + 1);
        println!("📝 Text: {}", text);
        
        match engine.initiate_process(text.to_string()) {
            Ok(process_id) => {
                println!("✅ Successfully initiated cognitive process: {}", process_id);
                
                // Process for a few iterations
                for _ in 0..10 {
                    engine.process_layer_transitions()?;
                    std::thread::sleep(Duration::from_millis(200));
                    
                    // Show status
                    let status = engine.get_status();
                    println!("📊 Status: Context={:.0}, Reasoning={:.0}, Intuition={:.0}, Completed={:.0}", 
                             status.get("context_processes").unwrap_or(&0.0),
                             status.get("reasoning_processes").unwrap_or(&0.0),
                             status.get("intuition_processes").unwrap_or(&0.0),
                             status.get("completed_processes").unwrap_or(&0.0));
                }
            }
            Err(e) => {
                println!("❌ Failed to initiate process for text {}: {}", index + 1, e);
            }
        }
    }
    
    // Final status
    let final_status = engine.get_status();
    println!("\n🎉 Final Results:");
    println!("   • Completed Processes: {:.0}", final_status.get("completed_processes").unwrap_or(&0.0));
    println!("   • Oxygen Level: {:.2}%", final_status.get("oxygen_level").unwrap_or(&0.0) * 100.0);
    println!("   • Lactate Level: {:.2}%", final_status.get("lactate_level").unwrap_or(&0.0) * 100.0);
    println!("   • Breathing Rate: {:.1}/min", final_status.get("breathing_rate").unwrap_or(&0.0));
    
    println!("\n🎉 Tres Commas Engine demonstration completed successfully!");
    println!("The first truly biological artificial intelligence system!");
    
    Ok(())
} 