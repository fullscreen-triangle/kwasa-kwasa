use std::collections::HashMap;

/// Confidence level for partial results
pub type Confidence = f64;

/// StreamData represents data flowing through the streaming pipeline
#[derive(Clone, Debug)]
pub struct StreamData {
    /// The content being processed
    pub content: String,
    
    /// Metadata associated with this data chunk
    pub metadata: HashMap<String, String>,
    
    /// Processing confidence - how confident we are in the partial result
    pub confidence: Confidence,
    
    /// Is this a final result or a partial one
    pub is_final: bool,
    
    /// Processing state - allows components to maintain state between operations
    pub state: HashMap<String, String>,
}

impl StreamData {
    /// Create new stream data with the given content
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
            confidence: 0.0,
            is_final: false,
            state: HashMap::new(),
        }
    }
    
    /// Create a final result
    pub fn final_result(content: String) -> Self {
        Self {
            content,
            metadata: HashMap::new(),
            confidence: 1.0,
            is_final: true,
            state: HashMap::new(),
        }
    }
    
    /// Set a metadata value
    pub fn with_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Set confidence level
    pub fn with_confidence(&mut self, confidence: Confidence) -> &mut Self {
        self.confidence = confidence;
        self
    }
    
    /// Add processing state
    pub fn with_state(&mut self, key: &str, value: &str) -> &mut Self {
        self.state.insert(key.to_string(), value.to_string());
        self
    }
} 