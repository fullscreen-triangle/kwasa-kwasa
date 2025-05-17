use std::collections::HashMap;
use crate::text_unit::{TextUnit, TextUnitType};
use crate::orchestrator::StreamData;

/// A Proposition is a container for ideas that can be analyzed and transformed
pub struct Proposition {
    /// The name of this proposition
    name: String,
    
    /// The motions (pieces of ideas) contained in this proposition
    motions: Vec<Motion>,
    
    /// Additional metadata about this proposition
    metadata: HashMap<String, String>,
}

/// A Motion is a piece of an idea, typically a text segment with semantic meaning
pub struct Motion {
    /// The content of this motion
    content: String,
    
    /// The type of this text unit
    unit_type: TextUnitType,
    
    /// Confidence in this motion (0.0-1.0)
    confidence: f64,
    
    /// Additional properties for this motion
    properties: HashMap<String, String>,
}

impl Proposition {
    /// Create a new proposition with the given name
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            motions: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add a motion to this proposition
    pub fn add_motion(&mut self, motion: Motion) {
        self.motions.push(motion);
    }
    
    /// Add a motion from content
    pub fn add_motion_from_content(&mut self, content: &str, unit_type: TextUnitType) {
        let motion = Motion::new(content, unit_type);
        self.add_motion(motion);
    }
    
    /// Get all motions in this proposition
    pub fn motions(&self) -> &[Motion] {
        &self.motions
    }
    
    /// Get the name of this proposition
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Add metadata to this proposition
    pub fn with_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
    
    /// Get metadata from this proposition
    pub fn metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// Convert to stream data for processing
    pub fn to_stream_data(&self) -> StreamData {
        let mut content = String::new();
        
        // Combine all motions into a single content string
        for motion in &self.motions {
            content.push_str(&motion.content);
            content.push('\n');
        }
        
        // Create stream data
        let mut stream_data = StreamData::new(content);
        
        // Add metadata
        for (key, value) in &self.metadata {
            stream_data = stream_data.with_metadata(key, value);
        }
        
        stream_data
    }
}

impl Motion {
    /// Create a new motion with the given content
    pub fn new(content: &str, unit_type: TextUnitType) -> Self {
        Self {
            content: content.to_string(),
            unit_type,
            confidence: 1.0,
            properties: HashMap::new(),
        }
    }
    
    /// Get the content of this motion
    pub fn content(&self) -> &str {
        &self.content
    }
    
    /// Get the type of this text unit
    pub fn unit_type(&self) -> TextUnitType {
        self.unit_type
    }
    
    /// Add a property to this motion
    pub fn with_property(&mut self, key: &str, value: &str) {
        self.properties.insert(key.to_string(), value.to_string());
    }
    
    /// Get a property from this motion
    pub fn property(&self, key: &str) -> Option<&String> {
        self.properties.get(key)
    }
    
    /// Convert to a text unit
    pub fn to_text_unit(&self, id: usize) -> TextUnit {
        TextUnit::new(
            self.content.clone(),
            0,
            self.content.len(),
            self.unit_type,
            id,
        )
    }
    
    /// Check spelling of this motion
    pub fn spelling(&self) -> SpellingResult {
        // Simple implementation - in a real system we would use a proper spell checker
        let mut misspelled = Vec::new();
        
        // For demonstration, just flag words shorter than 3 chars as misspelled
        for word in self.content.split_whitespace() {
            if word.len() < 3 {
                misspelled.push(word.to_string());
            }
        }
        
        SpellingResult {
            text: self.content.clone(),
            misspelled,
        }
    }
    
    /// Check capitalization issues
    pub fn capitalization(&self) -> CapitalizationResult {
        let mut issues = Vec::new();
        
        // Simple implementation - check if sentences start with capital letters
        for sentence in self.content.split('.') {
            if let Some(first_char) = sentence.trim().chars().next() {
                if !first_char.is_uppercase() && sentence.trim().len() > 0 {
                    issues.push(sentence.trim().to_string());
                }
            }
        }
        
        CapitalizationResult {
            text: self.content.clone(),
            issues,
        }
    }
    
    /// Check for cognitive biases (example of advanced analysis)
    pub fn check_sunken_cost_fallacy(&self) -> BiasAnalysisResult {
        // Keywords that might indicate sunken cost fallacy
        let sunken_cost_indicators = [
            "already invested",
            "too much time",
            "can't quit now",
            "give up",
            "wasted",
            "spent so much",
        ];
        
        let mut found_indicators = Vec::new();
        
        for indicator in &sunken_cost_indicators {
            if self.content.to_lowercase().contains(indicator) {
                found_indicators.push(indicator.to_string());
            }
        }
        
        BiasAnalysisResult {
            text: self.content.clone(),
            bias_type: "Sunken Cost Fallacy".to_string(),
            indicators: found_indicators.clone(),
            has_bias: !found_indicators.is_empty(),
        }
    }
    
    /// Generic method to check something exactly as requested
    pub fn check_this_exactly(&self, pattern: &str) -> CheckResult {
        CheckResult {
            text: self.content.clone(),
            pattern: pattern.to_string(),
            found: self.content.contains(pattern),
            count: self.content.matches(pattern).count(),
        }
    }
}

/// Result of a spelling check
pub struct SpellingResult {
    pub text: String,
    pub misspelled: Vec<String>,
}

/// Result of a capitalization check
pub struct CapitalizationResult {
    pub text: String,
    pub issues: Vec<String>,
}

/// Result of a bias analysis
pub struct BiasAnalysisResult {
    pub text: String,
    pub bias_type: String,
    pub indicators: Vec<String>,
    pub has_bias: bool,
}

/// Result of a custom check
pub struct CheckResult {
    pub text: String,
    pub pattern: String,
    pub found: bool,
    pub count: usize,
} 