use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use async_trait::async_trait;
use log::{info, warn, error, debug};

use super::types::StreamData;

/// Buffer size for channels between processors
const DEFAULT_BUFFER_SIZE: usize = 32;

/// Trait for stream processors that can be chained together
#[async_trait]
pub trait StreamProcessor: Send + Sync {
    /// Process a stream of data
    async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData>;
    
    /// Get the name of this processor
    fn name(&self) -> &str;
    
    /// Check if this processor can handle the given input
    fn can_handle(&self, _data: &StreamData) -> bool {
        true // Default: can handle any data
    }
    
    /// Get processor statistics
    fn stats(&self) -> ProcessorStats;
}

/// Statistics for a processor
#[derive(Debug, Clone, Default)]
pub struct ProcessorStats {
    pub items_processed: u64,
    pub errors: u64,
    pub average_processing_time_ms: f64,
    pub last_processed: Option<u64>,
}

/// A pipeline of stream processors
pub struct StreamPipeline {
    name: String,
    processors: Vec<Arc<dyn StreamProcessor>>,
    buffer_size: usize,
    error_handler: Option<Box<dyn Fn(String) + Send + Sync>>,
    stats: PipelineStats,
}

/// Statistics for the entire pipeline
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_items: u64,
    pub successful_items: u64,
    pub failed_items: u64,
    pub average_latency_ms: f64,
    pub throughput_per_second: f64,
}

impl StreamPipeline {
    /// Create a new stream pipeline
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processors: Vec::new(),
            buffer_size: 32,
            error_handler: None,
            stats: PipelineStats::default(),
        }
    }
    
    /// Add a processor to the pipeline
    pub fn add_processor(&mut self, processor: impl StreamProcessor + 'static) -> &mut Self {
        self.processors.push(Arc::new(processor));
        self
    }
    
    /// Set the buffer size for channels
    pub fn with_buffer_size(&mut self, size: usize) -> &mut Self {
        self.buffer_size = size;
        self
    }
    
    /// Set an error handler
    pub fn with_error_handler<F>(&mut self, handler: F) -> &mut Self 
    where
        F: Fn(String) + Send + Sync + 'static,
    {
        self.error_handler = Some(Box::new(handler));
        self
    }
    
    /// Execute the pipeline with the given input
    pub async fn execute(&self, input: Vec<StreamData>) -> Vec<StreamData> {
        info!("Executing pipeline '{}' with {} items", self.name, input.len());
        
        let (tx, mut rx) = channel(self.buffer_size);
        
        // Send input data
        for item in input {
            if tx.send(item).await.is_err() {
                error!("Failed to send input data to pipeline");
                break;
            }
        }
        drop(tx); // Close the channel
        
        // Process through all processors
        for (i, processor) in self.processors.iter().enumerate() {
            debug!("Processing through processor {}: {}", i, processor.name());
            rx = processor.process(rx).await;
        }
        
        // Collect results
        let mut results = Vec::new();
        while let Some(item) = rx.recv().await {
            results.push(item);
        }
        
        info!("Pipeline '{}' completed with {} output items", self.name, results.len());
        results
    }
    
    /// Create an input channel for streaming data
    pub async fn create_input(&self) -> (Sender<StreamData>, Receiver<StreamData>) {
        channel(self.buffer_size)
    }
    
    /// Stream data through the pipeline with a callback
    pub async fn stream<F, Fut>(&self, input: Receiver<StreamData>, callback: F)
    where
        F: Fn(StreamData) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send,
    {
        info!("Starting stream processing for pipeline '{}'", self.name);
        
        let mut current_rx = input;
        
        // Process through all processors
        for processor in &self.processors {
            current_rx = processor.process(current_rx).await;
        }
        
        // Handle output with callback
        let callback = Arc::new(callback);
        while let Some(item) = current_rx.recv().await {
            let cb = callback.clone();
            let item_clone = item.clone();
            
            tokio::spawn(async move {
                cb(item_clone).await;
            });
        }
        
        info!("Stream processing completed for pipeline '{}'", self.name);
    }
    
    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }
    
    /// Get processor statistics
    pub fn processor_stats(&self) -> Vec<(String, ProcessorStats)> {
        self.processors
            .iter()
            .map(|p| (p.name().to_string(), p.stats()))
            .collect()
    }
}

/// A simple function-based processor
pub struct FunctionProcessor<F> {
    name: String,
    function: Arc<F>,
    stats: std::sync::Mutex<ProcessorStats>,
}

impl<F> FunctionProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + 'static,
{
    /// Create a new function processor
    pub fn new(name: &str, function: F) -> Self {
        Self {
            name: name.to_string(),
            function: Arc::new(function),
            stats: std::sync::Mutex::new(ProcessorStats::default()),
        }
    }
}

#[async_trait]
impl<F> StreamProcessor for FunctionProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + 'static,
{
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let function = self.function.clone();
        let name = self.name.clone();
        
        tokio::spawn(async move {
            let mut processed_count = 0;
            
            while let Some(data) = input.recv().await {
                let start_time = std::time::Instant::now();
                
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(data.clone())
                })) {
                    Ok(result) => {
                        processed_count += 1;
                        
                        if tx.send(result).await.is_err() {
                            warn!("Processor '{}' output channel closed", name);
                            break;
                        }
                    }
                    Err(_) => {
                        error!("Processor '{}' panicked while processing data", name);
                        // Send original data on error
                        if tx.send(data).await.is_err() {
                            break;
                        }
                    }
                }
                
                let processing_time = start_time.elapsed();
                debug!("Processor '{}' processed item in {:?}", name, processing_time);
            }
            
            debug!("Processor '{}' completed, processed {} items", name, processed_count);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

/// A filter processor that removes items based on a predicate
pub struct FilterProcessor<P> {
    name: String,
    predicate: Arc<P>,
}

impl<P> FilterProcessor<P>
where
    P: Fn(&StreamData) -> bool + Send + Sync + 'static,
{
    /// Create a new filter processor
    pub fn new(name: &str, predicate: P) -> Self {
        Self {
            name: name.to_string(),
            predicate: Arc::new(predicate),
        }
    }
}

#[async_trait]
impl<P> StreamProcessor for FilterProcessor<P>
where
    P: Fn(&StreamData) -> bool + Send + Sync + 'static,
{
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let predicate = self.predicate.clone();
        let name = self.name.clone();
        
        tokio::spawn(async move {
            let mut filtered_count = 0;
            let mut total_count = 0;
            
            while let Some(data) = input.recv().await {
                total_count += 1;
                
                if predicate(&data) {
                    filtered_count += 1;
                    if tx.send(data).await.is_err() {
                        warn!("Filter '{}' output channel closed", name);
                        break;
                    }
                }
            }
            
            debug!("Filter '{}' passed {}/{} items", name, filtered_count, total_count);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn stats(&self) -> ProcessorStats {
        ProcessorStats::default()
    }
}

/// A batch processor that groups items together
pub struct BatchProcessor {
    name: String,
    batch_size: usize,
    timeout_ms: u64,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(name: &str, batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            batch_size,
            timeout_ms,
        }
    }
}

#[async_trait]
impl StreamProcessor for BatchProcessor {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let batch_size = self.batch_size;
        let timeout_ms = self.timeout_ms;
        let name = self.name.clone();
        
        tokio::spawn(async move {
            let mut batch = Vec::new();
            let mut batch_count = 0;
            
            loop {
                let timeout = tokio::time::sleep(tokio::time::Duration::from_millis(timeout_ms));
                
                tokio::select! {
                    item = input.recv() => {
                        match item {
                            Some(data) => {
                                batch.push(data);
                                
                                if batch.len() >= batch_size {
                                    // Send batch as combined data
                                    let combined = combine_batch(&batch);
                                    batch_count += 1;
                                    
                                    if tx.send(combined).await.is_err() {
                                        warn!("Batch processor '{}' output channel closed", name);
                                        break;
                                    }
                                    
                                    batch.clear();
                                }
                            }
                            None => {
                                // Input closed, send remaining batch if any
                                if !batch.is_empty() {
                                    let combined = combine_batch(&batch);
                                    batch_count += 1;
                                    let _ = tx.send(combined).await;
                                }
                                break;
                            }
                        }
                    }
                    _ = timeout => {
                        // Timeout reached, send current batch if any
                        if !batch.is_empty() {
                            let combined = combine_batch(&batch);
                            batch_count += 1;
                            
                            if tx.send(combined).await.is_err() {
                                warn!("Batch processor '{}' output channel closed", name);
                                break;
                            }
                            
                            batch.clear();
                        }
                    }
                }
            }
            
            debug!("Batch processor '{}' completed, created {} batches", name, batch_count);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn stats(&self) -> ProcessorStats {
        ProcessorStats::default()
    }
}

/// Combine a batch of StreamData into a single StreamData
fn combine_batch(batch: &[StreamData]) -> StreamData {
    if batch.is_empty() {
        return StreamData::new(String::new());
    }
    
    if batch.len() == 1 {
        return batch[0].clone();
    }
    
    // Combine content
    let combined_content = batch
        .iter()
        .map(|d| d.content.as_str())
        .collect::<Vec<&str>>()
        .join("\n");
    
    // Average confidence
    let avg_confidence = batch.iter().map(|d| d.confidence).sum::<f64>() / batch.len() as f64;
    
    // Combine metadata
    let mut combined_metadata = std::collections::HashMap::new();
    for data in batch {
        for (k, v) in &data.metadata {
            combined_metadata.insert(k.clone(), v.clone());
        }
    }
    
    // Add batch info
    combined_metadata.insert("batch_size".to_string(), batch.len().to_string());
    combined_metadata.insert("batch_type".to_string(), "combined".to_string());
    
    StreamData {
        content: combined_content,
        confidence: avg_confidence,
        metadata: combined_metadata,
        is_final: batch.iter().all(|d| d.is_final),
        state: std::collections::HashMap::new(),
    }
}

/// A parallel processor that splits work across multiple workers
pub struct ParallelProcessor<F> {
    name: String,
    worker_count: usize,
    function: Arc<F>,
}

impl<F> ParallelProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + 'static,
{
    /// Create a new parallel processor
    pub fn new(name: &str, worker_count: usize, function: F) -> Self {
        Self {
            name: name.to_string(),
            worker_count,
            function: Arc::new(function),
        }
    }
}

#[async_trait]
impl<F> StreamProcessor for ParallelProcessor<F>
where
    F: Fn(StreamData) -> StreamData + Send + Sync + 'static,
{
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let worker_count = self.worker_count;
        let function = self.function.clone();
        let name = self.name.clone();
        
        tokio::spawn(async move {
            let mut workers = Vec::new();
            let (work_tx, work_rx) = channel::<StreamData>(worker_count * 2);
            let work_rx = Arc::new(tokio::sync::Mutex::new(work_rx));
            
            // Spawn workers
            for i in 0..worker_count {
                let work_rx = work_rx.clone();
                let tx = tx.clone();
                let function = function.clone();
                let worker_name = format!("{}_worker_{}", name, i);
                
                let worker = tokio::spawn(async move {
                    let mut processed = 0;
                    
                    loop {
                        let data = {
                            let mut rx = work_rx.lock().await;
                            rx.recv().await
                        };
                        
                        match data {
                            Some(data) => {
                                let result = function(data);
                                processed += 1;
                                
                                if tx.send(result).await.is_err() {
                                    debug!("Worker '{}' output channel closed", worker_name);
                                    break;
                                }
                            }
                            None => {
                                debug!("Worker '{}' completed, processed {} items", worker_name, processed);
                                break;
                            }
                        }
                    }
                });
                
                workers.push(worker);
            }
            
            // Distribute work
            let mut distributed = 0;
            while let Some(data) = input.recv().await {
                if work_tx.send(data).await.is_err() {
                    warn!("Failed to distribute work in parallel processor '{}'", name);
                    break;
                }
                distributed += 1;
            }
            
            drop(work_tx); // Signal workers to stop
            
            // Wait for all workers to complete
            for worker in workers {
                let _ = worker.await;
            }
            
            debug!("Parallel processor '{}' completed, distributed {} items", name, distributed);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn stats(&self) -> ProcessorStats {
        ProcessorStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_function_processor() {
        let processor = FunctionProcessor::new("test", |mut data: StreamData| {
            data.content = data.content.to_uppercase();
            data
        });
        
        let (tx, rx) = channel(10);
        tx.send(StreamData::new("hello world".to_string())).await.unwrap();
        drop(tx);
        
        let mut output = processor.process(rx).await;
        let result = output.recv().await.unwrap();
        
        assert_eq!(result.content, "HELLO WORLD");
    }
    
    #[tokio::test]
    async fn test_filter_processor() {
        let processor = FilterProcessor::new("test", |data: &StreamData| {
            data.content.len() > 5
        });
        
        let (tx, rx) = channel(10);
        tx.send(StreamData::new("hi".to_string())).await.unwrap();
        tx.send(StreamData::new("hello world".to_string())).await.unwrap();
        tx.send(StreamData::new("bye".to_string())).await.unwrap();
        drop(tx);
        
        let mut output = processor.process(rx).await;
        let result = output.recv().await.unwrap();
        
        assert_eq!(result.content, "hello world");
        assert!(output.recv().await.is_none()); // Only one item should pass
    }
    
    #[tokio::test]
    async fn test_pipeline() {
        let mut pipeline = StreamPipeline::new("test");
        
        pipeline
            .add_processor(FunctionProcessor::new("uppercase", |mut data: StreamData| {
                data.content = data.content.to_uppercase();
                data
            }))
            .add_processor(FilterProcessor::new("long_only", |data: &StreamData| {
                data.content.len() > 3
            }));
        
        let input = vec![
            StreamData::new("hi".to_string()),
            StreamData::new("hello".to_string()),
            StreamData::new("world".to_string()),
        ];
        
        let results = pipeline.execute(input).await;
        
        assert_eq!(results.len(), 2); // "hi" should be filtered out
        assert_eq!(results[0].content, "HELLO");
        assert_eq!(results[1].content, "WORLD");
    }
} 