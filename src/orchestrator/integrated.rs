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
            .add_processor(self.mzekezeke.as_ref().clone())
            .add_processor(self.diggiden.as_ref().clone())
            .add_processor(self.hatata.as_ref().clone())
            .add_processor(self.spectacular.as_ref().clone())
            .add_processor(self.nicotine.as_ref().clone());
        
        let results = pipeline.execute(vec![initial_data]).await;
        
        results.into_iter().next()
            .map(|data| data.content)
            .unwrap_or(input)
    }
}

#[async_trait]
impl StreamProcessor for IntegratedMetacognitiveOrchestrator {
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        let mzekezeke = self.mzekezeke.clone();
        let diggiden = self.diggiden.clone();
        let hatata = self.hatata.clone();
        let spectacular = self.spectacular.clone();
        let nicotine = self.nicotine.clone();
        
        tokio::spawn(async move {
            let mut pipeline = StreamPipeline::new("Integrated");
            pipeline
                .add_processor(mzekezeke.as_ref().clone())
                .add_processor(diggiden.as_ref().clone())
                .add_processor(hatata.as_ref().clone())
                .add_processor(spectacular.as_ref().clone())
                .add_processor(nicotine.as_ref().clone());
            
            let output = pipeline.process(input).await;
            
            let mut output = output;
            while let Some(data) = output.recv().await {
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