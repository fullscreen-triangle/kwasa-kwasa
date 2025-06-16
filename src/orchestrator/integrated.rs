// Integrated Metacognitive Orchestrator
// Combines all five intelligence modules

use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::info;

use super::stream::{StreamProcessor, StreamPipeline, ProcessorStats};
use super::types::StreamData;
use super::mzekezeke::MzekezkeBayesianEngine;
use super::diggiden::DiggidenAdversarialSystem;
use super::hatata::HatataDecisionSystem;
use super::spectacular::SpectacularHandler;
use super::nicotine::NicotineContextValidator;

/// The main integrated metacognitive orchestrator
pub struct IntegratedMetacognitiveOrchestrator {
    mzekezeke: Arc<MzekezkeBayesianEngine>,
    diggiden: Arc<DiggidenAdversarialSystem>,
    hatata: Arc<HatataDecisionSystem>,
    spectacular: Arc<SpectacularHandler>,
    nicotine: Arc<NicotineContextValidator>,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl IntegratedMetacognitiveOrchestrator {
    pub fn new() -> Self {
        Self {
            mzekezeke: Arc::new(MzekezkeBayesianEngine::new()),
            diggiden: Arc::new(DiggidenAdversarialSystem::new()),
            hatata: Arc::new(HatataDecisionSystem::new()),
            spectacular: Arc::new(SpectacularHandler::new()),
            nicotine: Arc::new(NicotineContextValidator::new()),
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub async fn process_text_intelligently(&self, input: String) -> String {
        info!("Processing text intelligently: {}", input);
        
        let initial_data = StreamData::new(input.clone()).with_confidence(0.5);
        
        let mut pipeline = StreamPipeline::new("Intelligence");
        pipeline
            .add_processor(MzekezkeBayesianEngine::new())
            .add_processor(DiggidenAdversarialSystem::new())
            .add_processor(HatataDecisionSystem::new())
            .add_processor(SpectacularHandler::new())
            .add_processor(NicotineContextValidator::new());
        
        let results = pipeline.execute(vec![initial_data]).await;
        
        results.into_iter().next()
            .map(|data| data.content)
            .unwrap_or(input)
    }
}

#[async_trait]
impl StreamProcessor for IntegratedMetacognitiveOrchestrator {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            // Create a sequential processing chain
            while let Some(mut data) = input.recv().await {
                // Process through Mzekezeke (Bayesian Learning)
                let mzekezeke = MzekezkeBayesianEngine::new();
                let (m_tx, m_rx) = channel(1);
                let _ = m_tx.send(data).await;
                drop(m_tx);
                let mut m_output = mzekezeke.process(m_rx).await;
                data = m_output.recv().await.unwrap_or(data);
                
                // Process through Diggiden (Adversarial System)
                let diggiden = DiggidenAdversarialSystem::new();
                let (d_tx, d_rx) = channel(1);
                let _ = d_tx.send(data).await;
                drop(d_tx);
                let mut d_output = diggiden.process(d_rx).await;
                data = d_output.recv().await.unwrap_or(data);
                
                // Process through Hatata (Decision System)
                let hatata = HatataDecisionSystem::new();
                let (h_tx, h_rx) = channel(1);
                let _ = h_tx.send(data).await;
                drop(h_tx);
                let mut h_output = hatata.process(h_rx).await;
                data = h_output.recv().await.unwrap_or(data);
                
                // Process through Spectacular (Extraordinary Handler)
                let spectacular = SpectacularHandler::new();
                let (s_tx, s_rx) = channel(1);
                let _ = s_tx.send(data).await;
                drop(s_tx);
                let mut s_output = spectacular.process(s_rx).await;
                data = s_output.recv().await.unwrap_or(data);
                
                // Process through Nicotine (Context Validator)
                let nicotine = NicotineContextValidator::new();
                let (n_tx, n_rx) = channel(1);
                let _ = n_tx.send(data).await;
                drop(n_tx);
                let mut n_output = nicotine.process(n_rx).await;
                data = n_output.recv().await.unwrap_or(data);
                
                if tx.send(data).await.is_err() {
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "IntegratedMetacognitiveOrchestrator"
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
} 