use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use super::biomimetic::{GlycolicCycle, DreamingModule, LactateCycle};
use super::stream::{StreamProcessor, StreamPipeline};
use super::types::StreamData;

/// Confidence threshold for partial results
const CONFIDENCE_THRESHOLD: f64 = 0.8;

/// MetacognitiveOrchestrator manages the nested processing layers
pub struct MetacognitiveOrchestrator {
    /// Context layer for domain understanding
    context_layer: Arc<dyn StreamProcessor>,
    
    /// Reasoning layer for logical processing
    reasoning_layer: Arc<dyn StreamProcessor>,
    
    /// Intuition layer for pattern recognition
    intuition_layer: Arc<dyn StreamProcessor>,
    
    /// Glycolic cycle for resource management
    glycolytic: Arc<GlycolicCycle>,
    
    /// Dreaming module for edge case exploration
    dreaming: Arc<DreamingModule>,
    
    /// Lactate cycle for partial computations
    lactate_cycle: Arc<LactateCycle>,
    
    /// Knowledge base
    knowledge: Arc<Mutex<HashMap<String, String>>>,
    
    /// Processing pipeline
    pipeline: StreamPipeline,
}

impl MetacognitiveOrchestrator {
    /// Create a new metacognitive orchestrator
    pub fn new(
        context_layer: impl StreamProcessor + 'static,
        reasoning_layer: impl StreamProcessor + 'static,
        intuition_layer: impl StreamProcessor + 'static,
    ) -> Self {
        let glycolytic = Arc::new(GlycolicCycle::new());
        let dreaming = Arc::new(DreamingModule::new(0.7)); // medium diversity
        let lactate_cycle = Arc::new(LactateCycle::new(0.3)); // store less than 30% complete
        
        let mut pipeline = StreamPipeline::new("MetacognitiveOrchestrator");
        
        Self {
            context_layer: Arc::new(context_layer),
            reasoning_layer: Arc::new(reasoning_layer),
            intuition_layer: Arc::new(intuition_layer),
            glycolytic,
            dreaming,
            lactate_cycle,
            knowledge: Arc::new(Mutex::new(HashMap::new())),
            pipeline,
        }
    }
    
    /// Start processing with the orchestrator
    pub async fn process(&self, input: Vec<StreamData>) -> Vec<StreamData> {
        // Start biomimetic components
        self.glycolytic.start_monitoring();
        self.dreaming.start_dreaming();
        
        // Build the pipeline
        let mut pipeline = StreamPipeline::new("MetacognitiveOrchestrator");
        pipeline
            .add_processor(ContextLayerProcessor::new(self.context_layer.clone(), self.knowledge.clone()))
            .add_processor(ReasoningLayerProcessor::new(self.reasoning_layer.clone(), self.knowledge.clone()))
            .add_processor(IntuitionLayerProcessor::new(self.intuition_layer.clone(), self.knowledge.clone()));
        
        // Execute the pipeline
        let mut results = pipeline.execute(input).await;
        
        // Process incomplete tasks from lactate cycle
        for (id, data, completion) in self.lactate_cycle.get_all() {
            // If resources available, retry incomplete tasks
            if completion > 0.5 { // Retry if more than 50% done
                let retry_input = vec![data];
                let retry_results = pipeline.execute(retry_input).await;
                
                // If successful, add to results
                if !retry_results.is_empty() {
                    // Add retry results to final results with retry metadata
                    for retry_result in retry_results {
                        let mut retry_result = retry_result.with_metadata("retry_id", &id);
                        retry_result = retry_result.with_metadata("retry_completion", &completion.to_string());
                        results.push(retry_result);
                    }
                }
            }
        }
        
        results
    }
    
    /// Add knowledge to the knowledge base
    pub fn add_knowledge(&mut self, key: &str, value: &str) {
        let mut kb = self.knowledge.lock().unwrap();
        kb.insert(key.to_string(), value.to_string());
        
        // Also add to dreaming module
        self.dreaming.add_knowledge(value);
    }
    
    /// Get a buffered sender to stream data into the orchestrator
    pub async fn create_input_stream(&self) -> (Sender<StreamData>, Receiver<StreamData>) {
        // Build the pipeline
        let mut pipeline = StreamPipeline::new("MetacognitiveOrchestrator");
        pipeline
            .add_processor(ContextLayerProcessor::new(self.context_layer.clone(), self.knowledge.clone()))
            .add_processor(ReasoningLayerProcessor::new(self.reasoning_layer.clone(), self.knowledge.clone()))
            .add_processor(IntuitionLayerProcessor::new(self.intuition_layer.clone(), self.knowledge.clone()));
        
        // Create and return input channels
        let (tx, rx) = channel(32);
        (tx, rx)
    }
    
    /// Process streaming input and collect via callback
    pub async fn process_stream<F, Fut>(&self, input: Receiver<StreamData>, callback: F)
    where
        F: Fn(StreamData) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        // Build the pipeline
        let mut pipeline = StreamPipeline::new("MetacognitiveOrchestrator");
        pipeline
            .add_processor(ContextLayerProcessor::new(self.context_layer.clone(), self.knowledge.clone()))
            .add_processor(ReasoningLayerProcessor::new(self.reasoning_layer.clone(), self.knowledge.clone()))
            .add_processor(IntuitionLayerProcessor::new(self.intuition_layer.clone(), self.knowledge.clone()));
        
        // Process streaming
        pipeline.stream(input, callback).await;
    }
}

/// Processor that wraps the context layer
struct ContextLayerProcessor {
    processor: Arc<dyn StreamProcessor>,
    knowledge: Arc<Mutex<HashMap<String, String>>>,
}

impl ContextLayerProcessor {
    pub fn new(processor: Arc<dyn StreamProcessor>, knowledge: Arc<Mutex<HashMap<String, String>>>) -> Self {
        Self {
            processor,
            knowledge,
        }
    }
}

#[async_trait]
impl StreamProcessor for ContextLayerProcessor {
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        // Process through the underlying processor
        let mut processed = self.processor.process(input).await;
        
        // Create output channel
        let (tx, rx) = channel(32);
        
        // Clone knowledge for async block
        let knowledge = self.knowledge.clone();
        
        // Forward results with knowledge enrichment
        tokio::spawn(async move {
            while let Some(mut data) = processed.recv().await {
                // Enrich with knowledge context
                {
                    let kb = knowledge.lock().unwrap();
                    // Use kb here if needed
                }
                data = data.with_metadata("layer", "context");
                
                // Send enriched data
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "ContextLayerProcessor"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
}

/// Processor that wraps the reasoning layer
struct ReasoningLayerProcessor {
    processor: Arc<dyn StreamProcessor>,
    knowledge: Arc<Mutex<HashMap<String, String>>>,
}

impl ReasoningLayerProcessor {
    pub fn new(processor: Arc<dyn StreamProcessor>, knowledge: Arc<Mutex<HashMap<String, String>>>) -> Self {
        Self {
            processor,
            knowledge,
        }
    }
}

#[async_trait]
impl StreamProcessor for ReasoningLayerProcessor {
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        // Process through the underlying processor
        let mut processed = self.processor.process(input).await;
        
        // Create output channel
        let (tx, rx) = channel(32);
        
        // Clone knowledge for async block
        let knowledge = self.knowledge.clone();
        
        // Forward results with reasoning annotations
        tokio::spawn(async move {
            while let Some(mut data) = processed.recv().await {
                // Add reasoning layer annotations
                data = data.with_metadata("layer", "reasoning");
                
                // Send enriched data
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "ReasoningLayerProcessor"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
}

/// Processor that wraps the intuition layer
struct IntuitionLayerProcessor {
    processor: Arc<dyn StreamProcessor>,
    knowledge: Arc<Mutex<HashMap<String, String>>>,
}

impl IntuitionLayerProcessor {
    pub fn new(processor: Arc<dyn StreamProcessor>, knowledge: Arc<Mutex<HashMap<String, String>>>) -> Self {
        Self {
            processor,
            knowledge,
        }
    }
}

#[async_trait]
impl StreamProcessor for IntuitionLayerProcessor {
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        // Process through the underlying processor
        let mut processed = self.processor.process(input).await;
        
        // Create output channel
        let (tx, rx) = channel(32);
        
        // Clone knowledge for async block
        let knowledge = self.knowledge.clone();
        
        // Forward results with intuition annotations
        tokio::spawn(async move {
            while let Some(mut data) = processed.recv().await {
                // Add intuition layer annotations
                data = data.with_metadata("layer", "intuition");
                
                // Finalize result
                data.is_final = true;
                data.confidence = 1.0;
                
                // Send final data
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "IntuitionLayerProcessor"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
}

/// Default context layer implementation
pub struct DefaultContextLayer;

impl DefaultContextLayer {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StreamProcessor for DefaultContextLayer {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                // Simple context processing - analyze the content
                if !data.content.is_empty() {
                    // Partial processing example
                    data = data.with_confidence(0.5)
                        .with_metadata("context_processed", "true");
                    
                    let _ = tx.send(data).await;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "DefaultContextLayer"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
}

/// Default reasoning layer implementation
pub struct DefaultReasoningLayer;

impl DefaultReasoningLayer {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StreamProcessor for DefaultReasoningLayer {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                // Simple reasoning - enhance confidence
                data = data.with_confidence(0.8)
                    .with_metadata("reasoning_processed", "true");
                
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "DefaultReasoningLayer"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
}

/// Default intuition layer implementation
pub struct DefaultIntuitionLayer;

impl DefaultIntuitionLayer {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl StreamProcessor for DefaultIntuitionLayer {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                // Simple intuition - finalize the process
                data = data.with_confidence(1.0)
                    .with_metadata("intuition_processed", "true");
                
                data.is_final = true;
                
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "DefaultIntuitionLayer"
    }
    
    fn stats(&self) -> super::stream::ProcessorStats {
        super::stream::ProcessorStats::default()
    }
} 