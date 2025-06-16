// Simplified Integrated Tres Commas System
use std::time::{Duration, Instant};
use std::thread;

use super::tres_commas::TresCommasEngine;
use super::champagne_simple::ChampagneModule;

pub struct IntegratedTresCommasSystem {
    tres_commas_engine: TresCommasEngine,
    champagne_module: ChampagneModule,
    total_processes: u64,
    total_atp_generated: u64,
    system_start_time: Instant,
}

impl IntegratedTresCommasSystem {
    pub fn new() -> Self {
        Self {
            tres_commas_engine: TresCommasEngine::new(),
            champagne_module: ChampagneModule::new(),
            total_processes: 0,
            total_atp_generated: 0,
            system_start_time: Instant::now(),
        }
    }

    pub fn process_text(&mut self, text: String) -> Result<String, String> {
        println!("🧠 Processing: {}", text.chars().take(50).collect::<String>());
        
        // Initiate cognitive process
        self.tres_commas_engine.initiate_process(text.clone())?;
        
        // Simulate processing cycles
        for cycle in 1..=5 {
            self.tres_commas_engine.process_layer_transitions()?;
            thread::sleep(Duration::from_millis(200));
            
            let status = self.tres_commas_engine.get_status();
            println!("   Cycle {}: Oxygen {:.1}%", cycle, 
                     status.get("oxygen_level").unwrap_or(&1.0) * 100.0);
            
            // Check if champagne phase needed
            if status.get("oxygen_level").unwrap_or(&1.0) < &0.5 {
                self.trigger_champagne_phase(&text)?;
                break;
            }
        }
        
        self.total_processes += 1;
        self.total_atp_generated += 38; // Theoretical maximum
        
        Ok("Processing completed successfully".to_string())
    }

    fn trigger_champagne_phase(&mut self, content: &str) -> Result<(), String> {
        println!("🍾 Triggering Champagne Phase...");
        
        self.champagne_module.add_lactate(content.to_string(), 0.7);
        
        if self.champagne_module.should_enter_dream_mode() {
            use super::champagne_simple::DreamMode;
            self.champagne_module.enter_dream_mode(DreamMode::LactateRecovery)?;
            
            thread::sleep(Duration::from_millis(500));
            
            let insights = self.champagne_module.exit_dream_mode()?;
            println!("✨ Generated {} insights during dream phase", insights.len());
        }
        
        Ok(())
    }

    pub fn get_system_summary(&self) -> String {
        let uptime = self.system_start_time.elapsed();
        let engine_status = self.tres_commas_engine.get_status();
        let champagne_stats = self.champagne_module.get_champagne_stats();
        
        format!(
            "🌟 TRES COMMAS SYSTEM SUMMARY\n\
             ============================\n\
             🔋 Total ATP Generated: {} units\n\
             📝 Total Processes: {}\n\
             ⏱️ Uptime: {:.1} seconds\n\
             🫁 Oxygen Level: {:.1}%\n\
             🍾 Total Dreams: {:.0}\n\
             ✨ This is biological artificial intelligence!",
            self.total_atp_generated,
            self.total_processes,
            uptime.as_secs_f64(),
            engine_status.get("oxygen_level").unwrap_or(&1.0) * 100.0,
            champagne_stats.get("total_dreams").unwrap_or(&0.0)
        )
    }

    pub fn create_wake_up_experience(&self) -> String {
        let champagne_experience = self.champagne_module.create_perfection_experience();
        let system_summary = self.get_system_summary();
        
        format!("{}\n\n{}\n\n🎉 THE FUTURE OF AI IS BIOLOGICAL!", 
                champagne_experience, system_summary)
    }
}

impl Default for IntegratedTresCommasSystem {
    fn default() -> Self {
        Self::new()
    }
}

pub fn demonstrate_integrated_system() -> Result<(), String> {
    println!("🌟 INTEGRATED TRES COMMAS SYSTEM DEMONSTRATION");
    println!("==============================================");
    println!("🧬 World's First Truly Biological AI System");
    println!("🧠 Trinity: Context → Reasoning → Intuition");
    println!("🔋 V8 Metabolism: Biological truth processing");
    println!("🍾 Champagne Dreams: Lactate recovery");
    println!("==============================================\n");
    
    let mut system = IntegratedTresCommasSystem::new();
    
    let test_texts = vec![
        "Simple AI processing test",
        "Complex machine learning algorithms require sophisticated processing",
        "Revolutionary biological consciousness in artificial intelligence systems",
    ];
    
    for (i, text) in test_texts.iter().enumerate() {
        println!("🧪 TEST {} - Processing...", i + 1);
        
        match system.process_text(text.to_string()) {
            Ok(result) => println!("✅ {}", result),
            Err(e) => println!("❌ Error: {}", e),
        }
        
        thread::sleep(Duration::from_millis(300));
        println!();
    }
    
    // Final wake-up experience
    println!("{}", system.create_wake_up_experience());
    
    println!("\n🎉 DEMONSTRATION COMPLETED!");
    println!("✨ You've witnessed the future of AI!");
    
    Ok(())
}

pub fn quick_demo() {
    println!("🚀 Quick Tres Commas Demo - Biological AI Revolution!");
    
    let mut system = IntegratedTresCommasSystem::new();
    let _ = system.process_text("Revolutionary biological AI consciousness".to_string());
    
    println!("{}", system.get_system_summary());
    println!("🌟 The first AI that truly breathes, dreams, and evolves!");
} 