use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use async_trait::async_trait;
use std::future::Future;

use super::types::StreamData;

/// Buffer size for channels between processors
const DEFAULT_BUFFER_SIZE: usize = 32;

/// StreamProcessor defines the interface for each processing layer
#[async_trait]
pub trait StreamProcessor: Send + Sync {
    /// Process incoming data and return a stream of results
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData>;
    
    /// Name of this processor for debugging
    fn name(&self) -> &str;
}

/// StreamPipeline combines multiple processors into a processing pipeline
pub struct StreamPipeline {
    processors: Vec<Arc<dyn StreamProcessor>>,
    name: String,
}

impl StreamPipeline {
    /// Create a new pipeline with the given name
    pub fn new(name: &str) -> Self {
        Self {
            processors: Vec::new(),
            name: name.to_string(),
        }
    }
    
    /// Add a processor to the pipeline
    pub fn add_processor<P: StreamProcessor + 'static>(&mut self, processor: P) -> &mut Self {
        self.processors.push(Arc::new(processor));
        self
    }
    
    /// Execute the pipeline with the given input
    pub async fn execute(&self, input: Vec<StreamData>) -> Vec<StreamData> {
        if self.processors.is_empty() {
            return input;
        }
        
        // Create input channel
        let (input_tx, mut input_rx) = channel::<StreamData>(DEFAULT_BUFFER_SIZE);
        
        // Send all input data
        for data in input {
            let _ = input_tx.send(data).await;
        }
        drop(input_tx); // Close the channel
        
        // Create a chain of processors
        let mut current_rx = input_rx;
        for processor in &self.processors {
            current_rx = processor.process(current_rx).await;
        }
        
        // Collect results
        let mut results = Vec::new();
        while let Some(data) = current_rx.recv().await {
            results.push(data);
        }
        
        results
    }
    
    /// Stream results through the pipeline and collect them via a callback
    pub async fn stream<F, Fut>(&self, input: Receiver<StreamData>, callback: F)
    where
        F: Fn(StreamData) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send,
    {
        let mut current_rx = input;
        
        // Chain processors
        for processor in &self.processors {
            current_rx = processor.process(current_rx).await;
        }
        
        // Forward results to callback
        while let Some(data) = current_rx.recv().await {
            callback(data).await;
        }
    }
    
    /// Create an input sender for streaming data into this pipeline
    pub fn create_input(&self) -> (Sender<StreamData>, Receiver<StreamData>) {
        let (tx, rx) = channel::<StreamData>(DEFAULT_BUFFER_SIZE);
        (tx, rx)
    }
}

/// A simple pass-through processor that doesn't modify the data
pub struct IdentityProcessor {
    name: String,
}

impl IdentityProcessor {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl StreamProcessor for IdentityProcessor {
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(DEFAULT_BUFFER_SIZE);
        
        tokio::spawn(async move {
            let mut input = input; // Make mutable locally
            while let Some(data) = input.recv().await {
                let _ = tx.send(data).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// Create a processor that applies a function to each item
pub struct FunctionProcessor<F> {
    name: String,
    func: F,
}

impl<F> FunctionProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + Clone + 'static,
{
    pub fn new(name: &str, func: F) -> Self {
        Self {
            name: name.to_string(),
            func,
        }
    }
}

#[async_trait]
impl<F> StreamProcessor for FunctionProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + Clone + 'static,
{
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(DEFAULT_BUFFER_SIZE);
        // Clone self into the closure to avoid reference issues
        let func = Arc::new(self.func.clone());
        
        tokio::spawn(async move {
            let mut input = input; // Make mutable locally
            while let Some(data) = input.recv().await {
                let result = func(data);
                let _ = tx.send(result).await;
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
} 