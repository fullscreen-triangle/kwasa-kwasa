/// Streaming Text Processing for Kwasa-Kwasa
/// 
/// This module implements streaming text processing that treats text as a continuous
/// stream processed in sentence-level stages, maintaining positional semantics and
/// extracting Points for debate platform resolution.

use std::collections::{VecDeque, HashMap};
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::probabilistic::{TextPoint, ResolutionResult, ResolutionManager};
use crate::turbulance::positional_semantics::{PositionalSentence, PositionalAnalyzer};

/// A streaming text processor that handles continuous input
pub struct TextStream {
    /// Buffer for incomplete sentences
    sentence_buffer: VecDeque<PositionalSentence>,
    
    /// Size of context window (number of sentences to consider)
    context_window: usize,
    
    /// Positional analyzer for sentence processing
    positional_analyzer: PositionalAnalyzer,
    
    /// Point extractor for semantic content
    point_extractor: PointExtractor,
    
    /// Resolution manager for debate platforms
    resolution_manager: ResolutionManager,
    
    /// Current stream state
    stream_state: StreamState,
    
    /// Processing statistics
    stats: StreamStats,
    
    /// Configuration
    config: StreamConfig,
}

/// Configuration for streaming processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Maximum context window size
    pub max_context_window: usize,
    
    /// Minimum confidence threshold for point extraction
    pub min_point_confidence: f64,
    
    /// Whether to enable real-time processing
    pub real_time_processing: bool,
    
    /// Sentence boundary detection mode
    pub boundary_detection: BoundaryDetectionMode,
    
    /// Point extraction strategy
    pub extraction_strategy: ExtractionStrategy,
    
    /// Buffer size for sentence processing
    pub sentence_buffer_size: usize,
    
    /// Enable positional analysis
    pub enable_positional_analysis: bool,
    
    /// Enable debate platform creation
    pub enable_debate_platforms: bool,
}

/// Mode for detecting sentence boundaries
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BoundaryDetectionMode {
    /// Simple period/question/exclamation detection
    Simple,
    
    /// Advanced linguistic boundary detection
    Advanced,
    
    /// Machine learning-based detection
    ML,
    
    /// Custom boundary patterns
    Custom(Vec<String>),
}

/// Strategy for extracting points from sentences
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ExtractionStrategy {
    /// One point per sentence
    OnePerSentence,
    
    /// Multiple points per sentence based on clauses
    MultiplePerSentence,
    
    /// Points based on semantic density
    SemanticDensity,
    
    /// Custom extraction rules
    Custom,
}

/// Current state of the text stream
#[derive(Clone, Debug, PartialEq)]
pub enum StreamState {
    /// Stream is ready to receive input
    Ready,
    
    /// Stream is actively processing
    Processing,
    
    /// Stream is waiting for more input
    Buffering,
    
    /// Stream has completed processing
    Complete,
    
    /// Stream encountered an error
    Error(String),
    
    /// Stream is paused
    Paused,
}

/// Statistics about stream processing
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamStats {
    /// Total sentences processed
    pub sentences_processed: u64,
    
    /// Total points extracted
    pub points_extracted: u64,
    
    /// Total resolutions created
    pub resolutions_created: u64,
    
    /// Average processing time per sentence (ms)
    pub avg_processing_time_ms: f64,
    
    /// Current throughput (sentences/sec)
    pub current_throughput: f64,
    
    /// Total processing time (ms)
    pub total_processing_time_ms: u64,
    
    /// Errors encountered
    pub errors_encountered: u64,
    
    /// Context window utilization
    pub context_utilization: f64,
}

/// Result of processing a stream chunk
#[derive(Clone, Debug)]
pub struct StreamResult {
    /// Points extracted from this chunk
    pub points: Vec<TextPoint>,
    
    /// Resolutions created for the points
    pub resolutions: Vec<ResolutionResult>,
    
    /// Current context information
    pub context: StreamContext,
    
    /// Processing metadata
    pub metadata: StreamMetadata,
    
    /// Updated stream state
    pub state: StreamState,
}

/// Context maintained during streaming
#[derive(Clone, Debug)]
pub struct StreamContext {
    /// Recent sentences for context
    pub recent_sentences: VecDeque<PositionalSentence>,
    
    /// Accumulated semantic themes
    pub themes: HashMap<String, f64>,
    
    /// Discourse markers and transitions
    pub discourse_markers: Vec<String>,
    
    /// Temporal progression
    pub temporal_sequence: Vec<String>,
    
    /// Current topic focus
    pub current_topic: Option<String>,
    
    /// Confidence in current context
    pub context_confidence: f64,
}

/// Metadata about stream processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamMetadata {
    /// Timestamp of processing
    pub timestamp: u64,
    
    /// Processing duration (ms)
    pub duration_ms: u64,
    
    /// Chunk size processed
    pub chunk_size: usize,
    
    /// Context window size used
    pub context_window_used: usize,
    
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    
    /// Warnings generated
    pub warnings: Vec<String>,
    
    /// Debug information
    pub debug_info: HashMap<String, Value>,
}

/// Extractor for converting sentences to semantic points
pub struct PointExtractor {
    /// Extraction strategy
    strategy: ExtractionStrategy,
    
    /// Confidence threshold
    confidence_threshold: f64,
    
    /// Clause detection patterns
    clause_patterns: Vec<ClausePattern>,
    
    /// Cache for extracted points
    extraction_cache: HashMap<String, Vec<TextPoint>>,
}

/// Pattern for detecting clauses within sentences
#[derive(Clone, Debug)]
pub struct ClausePattern {
    /// Pattern to match
    pattern: String,
    
    /// Confidence in this pattern
    confidence: f64,
    
    /// Type of clause
    clause_type: ClauseType,
}

/// Types of clauses that can be extracted as points
#[derive(Clone, Debug, PartialEq)]
pub enum ClauseType {
    /// Main independent clause
    Main,
    
    /// Dependent subordinate clause
    Subordinate,
    
    /// Relative clause
    Relative,
    
    /// Conditional clause
    Conditional,
    
    /// Temporal clause
    Temporal,
    
    /// Causal clause
    Causal,
}

impl TextStream {
    /// Create a new text stream processor
    pub fn new(config: StreamConfig) -> Self {
        Self {
            sentence_buffer: VecDeque::with_capacity(config.sentence_buffer_size),
            context_window: config.max_context_window,
            positional_analyzer: PositionalAnalyzer::new(),
            point_extractor: PointExtractor::new(config.extraction_strategy.clone(), config.min_point_confidence),
            resolution_manager: ResolutionManager::new(),
            stream_state: StreamState::Ready,
            stats: StreamStats::default(),
            config,
        }
    }
    
    /// Process a chunk of text through the stream
    pub async fn process_chunk(&mut self, text_chunk: &str) -> Result<StreamResult> {
        let start_time = std::time::Instant::now();
        self.stream_state = StreamState::Processing;
        
        // Detect sentence boundaries
        let sentences = self.detect_sentences(text_chunk)?;
        let sentence_count = sentences.len();
        
        let mut all_points = Vec::new();
        let mut all_resolutions = Vec::new();
        
        // Process each sentence
        for sentence_text in sentences {
            // Skip empty sentences
            if sentence_text.trim().is_empty() {
                continue;
            }
            
            // Positional analysis
            let positional_sentence = if self.config.enable_positional_analysis {
                self.positional_analyzer.analyze(&sentence_text)?
            } else {
                // Create minimal positional sentence
                PositionalSentence {
                    original_text: sentence_text.clone(),
                    words: Vec::new(),
                    semantic_signature: "UNKNOWN".to_string(),
                    order_dependency_score: 0.5,
                    positional_hash: 0,
                    analysis_confidence: 0.5,
                    metadata: HashMap::new(),
                }
            };
            
            // Add to sentence buffer
            self.sentence_buffer.push_back(positional_sentence.clone());
            
            // Maintain context window size
            while self.sentence_buffer.len() > self.context_window {
                self.sentence_buffer.pop_front();
            }
            
            // Extract points from the sentence
            let points = self.point_extractor.extract_points(&positional_sentence, &self.sentence_buffer)?;
            
            // Create resolutions for new points
            let mut resolutions = Vec::new();
            if self.config.enable_debate_platforms {
                for point in &points {
                    // Create basic resolution (could be enhanced with actual debate platform)
                    let resolution = self.create_resolution_for_point(point).await?;
                    resolutions.push(resolution);
                }
            }
            
            all_points.extend(points);
            all_resolutions.extend(resolutions);
            
            // Update statistics
            self.stats.sentences_processed += 1;
        }
        
        // Update stats
        let duration = start_time.elapsed();
        self.stats.total_processing_time_ms += duration.as_millis() as u64;
        self.stats.points_extracted += all_points.len() as u64;
        self.stats.resolutions_created += all_resolutions.len() as u64;
        
        if self.stats.sentences_processed > 0 {
            self.stats.avg_processing_time_ms = self.stats.total_processing_time_ms as f64 / self.stats.sentences_processed as f64;
        }
        
        // Calculate throughput
        if duration.as_secs_f64() > 0.0 {
            self.stats.current_throughput = sentence_count as f64 / duration.as_secs_f64();
        }
        
        // Calculate context utilization
        self.stats.context_utilization = self.sentence_buffer.len() as f64 / self.context_window as f64;
        
        // Create context
        let context = StreamContext {
            recent_sentences: self.sentence_buffer.clone(),
            themes: self.extract_themes(),
            discourse_markers: self.extract_discourse_markers(),
            temporal_sequence: self.extract_temporal_sequence(),
            current_topic: self.identify_current_topic(),
            context_confidence: self.calculate_context_confidence(),
        };
        
        // Create metadata
        let metadata = StreamMetadata {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms: duration.as_millis() as u64,
            chunk_size: text_chunk.len(),
            context_window_used: self.sentence_buffer.len(),
            quality_metrics: self.calculate_quality_metrics(),
            warnings: Vec::new(),
            debug_info: HashMap::new(),
        };
        
        self.stream_state = StreamState::Ready;
        
        Ok(StreamResult {
            points: all_points,
            resolutions: all_resolutions,
            context,
            metadata,
            state: self.stream_state.clone(),
        })
    }
    
    /// Detect sentence boundaries in text
    fn detect_sentences(&self, text: &str) -> Result<Vec<String>> {
        match self.config.boundary_detection {
            BoundaryDetectionMode::Simple => {
                let sentences: Vec<String> = text
                    .split(&['.', '!', '?'])
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                Ok(sentences)
            },
            BoundaryDetectionMode::Advanced => {
                // More sophisticated sentence detection
                self.advanced_sentence_detection(text)
            },
            BoundaryDetectionMode::ML => {
                // Placeholder for ML-based detection
                self.ml_sentence_detection(text)
            },
            BoundaryDetectionMode::Custom(ref patterns) => {
                self.custom_sentence_detection(text, patterns)
            },
        }
    }
    
    /// Advanced linguistic sentence boundary detection
    fn advanced_sentence_detection(&self, text: &str) -> Result<Vec<String>> {
        let mut sentences = Vec::new();
        let mut current_sentence = String::new();
        let mut in_quotes = false;
        let mut paren_depth = 0;
        
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            let ch = chars[i];
            current_sentence.push(ch);
            
            match ch {
                '"' | '\'' => in_quotes = !in_quotes,
                '(' | '[' | '{' => paren_depth += 1,
                ')' | ']' | '}' => paren_depth = (paren_depth - 1).max(0),
                '.' | '!' | '?' => {
                    if !in_quotes && paren_depth == 0 {
                        // Check if this is likely end of sentence
                        if self.is_sentence_boundary(&chars, i) {
                            sentences.push(current_sentence.trim().to_string());
                            current_sentence.clear();
                        }
                    }
                },
                _ => {},
            }
            
            i += 1;
        }
        
        // Add any remaining text as a sentence
        if !current_sentence.trim().is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
        
        Ok(sentences.into_iter().filter(|s| !s.is_empty()).collect())
    }
    
    /// Check if a punctuation mark represents a sentence boundary
    fn is_sentence_boundary(&self, chars: &[char], pos: usize) -> bool {
        // Look ahead for capitalization or whitespace patterns
        let mut i = pos + 1;
        
        // Skip whitespace
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        
        // Check if next character is capitalized (likely new sentence)
        if i < chars.len() {
            chars[i].is_uppercase()
        } else {
            true // End of text
        }
    }
    
    /// ML-based sentence detection (placeholder)
    fn ml_sentence_detection(&self, text: &str) -> Result<Vec<String>> {
        // For now, fall back to advanced detection
        // In a real implementation, this would use a trained model
        self.advanced_sentence_detection(text)
    }
    
    /// Custom sentence detection using patterns
    fn custom_sentence_detection(&self, text: &str, patterns: &[String]) -> Result<Vec<String>> {
        let mut sentences = vec![text.to_string()];
        
        for pattern in patterns {
            let mut new_sentences = Vec::new();
            for sentence in sentences {
                new_sentences.extend(sentence.split(pattern).map(|s| s.trim().to_string()));
            }
            sentences = new_sentences;
        }
        
        Ok(sentences.into_iter().filter(|s| !s.is_empty()).collect())
    }
    
    /// Create a resolution for a point
    async fn create_resolution_for_point(&mut self, point: &TextPoint) -> Result<ResolutionResult> {
        // For now, create a simple resolution
        // In a full implementation, this would create actual debate platforms
        Ok(ResolutionResult::Certain(Value::String(format!(
            "Resolution for: {} (confidence: {:.2})",
            point.content, point.confidence
        ))))
    }
    
    /// Extract thematic content from recent sentences
    fn extract_themes(&self) -> HashMap<String, f64> {
        let mut themes = HashMap::new();
        
        for sentence in &self.sentence_buffer {
            // Simple keyword-based theme extraction
            for word in &sentence.words {
                if word.is_content_word && word.text.len() > 3 {
                    let theme_weight = word.positional_weight * word.order_dependency;
                    *themes.entry(word.text.clone()).or_insert(0.0) += theme_weight;
                }
            }
        }
        
        // Normalize themes
        let max_weight = themes.values().cloned().fold(0.0, f64::max);
        if max_weight > 0.0 {
            for weight in themes.values_mut() {
                *weight /= max_weight;
            }
        }
        
        themes
    }
    
    /// Extract discourse markers from recent sentences
    fn extract_discourse_markers(&self) -> Vec<String> {
        let discourse_words = [
            "however", "therefore", "furthermore", "moreover", "nevertheless",
            "consequently", "meanwhile", "finally", "firstly", "secondly",
            "in addition", "on the other hand", "as a result", "for example",
        ];
        
        let mut markers = Vec::new();
        for sentence in &self.sentence_buffer {
            for word in &sentence.words {
                let word_lower = word.text.to_lowercase();
                if discourse_words.contains(&word_lower.as_str()) {
                    markers.push(word.text.clone());
                }
            }
        }
        
        markers
    }
    
    /// Extract temporal sequence from recent sentences
    fn extract_temporal_sequence(&self) -> Vec<String> {
        let temporal_words = [
            "yesterday", "today", "tomorrow", "now", "then", "before", "after",
            "first", "next", "finally", "previously", "subsequently",
        ];
        
        let mut sequence = Vec::new();
        for sentence in &self.sentence_buffer {
            for word in &sentence.words {
                let word_lower = word.text.to_lowercase();
                if temporal_words.contains(&word_lower.as_str()) {
                    sequence.push(word.text.clone());
                }
            }
        }
        
        sequence
    }
    
    /// Identify current topic from context
    fn identify_current_topic(&self) -> Option<String> {
        let themes = self.extract_themes();
        themes.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(topic, _weight)| topic)
    }
    
    /// Calculate confidence in current context
    fn calculate_context_confidence(&self) -> f64 {
        if self.sentence_buffer.is_empty() {
            return 0.0;
        }
        
        let avg_confidence: f64 = self.sentence_buffer.iter()
            .map(|s| s.analysis_confidence)
            .sum::<f64>() / self.sentence_buffer.len() as f64;
            
        // Adjust based on context window utilization
        let utilization_factor = self.sentence_buffer.len() as f64 / self.context_window as f64;
        
        avg_confidence * utilization_factor
    }
    
    /// Calculate quality metrics for the stream
    fn calculate_quality_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        if !self.sentence_buffer.is_empty() {
            // Average order dependency
            let avg_order_dep: f64 = self.sentence_buffer.iter()
                .map(|s| s.order_dependency_score)
                .sum::<f64>() / self.sentence_buffer.len() as f64;
            metrics.insert("avg_order_dependency".to_string(), avg_order_dep);
            
            // Average analysis confidence
            let avg_confidence: f64 = self.sentence_buffer.iter()
                .map(|s| s.analysis_confidence)
                .sum::<f64>() / self.sentence_buffer.len() as f64;
            metrics.insert("avg_analysis_confidence".to_string(), avg_confidence);
            
            // Semantic diversity (number of unique semantic signatures)
            let unique_signatures: std::collections::HashSet<&String> = self.sentence_buffer.iter()
                .map(|s| &s.semantic_signature)
                .collect();
            let diversity = unique_signatures.len() as f64 / self.sentence_buffer.len() as f64;
            metrics.insert("semantic_diversity".to_string(), diversity);
        }
        
        // Processing efficiency
        if self.stats.sentences_processed > 0 {
            let efficiency = self.stats.sentences_processed as f64 / (self.stats.total_processing_time_ms as f64 / 1000.0);
            metrics.insert("processing_efficiency".to_string(), efficiency);
        }
        
        metrics
    }
    
    /// Get current stream statistics
    pub fn get_stats(&self) -> &StreamStats {
        &self.stats
    }
    
    /// Get current stream state
    pub fn get_state(&self) -> &StreamState {
        &self.stream_state
    }
    
    /// Reset the stream (clear buffers and stats)
    pub fn reset(&mut self) {
        self.sentence_buffer.clear();
        self.stats = StreamStats::default();
        self.stream_state = StreamState::Ready;
        self.point_extractor.extraction_cache.clear();
    }
    
    /// Configure the stream
    pub fn configure(&mut self, config: StreamConfig) {
        self.config = config;
        self.context_window = self.config.max_context_window;
        self.point_extractor.confidence_threshold = self.config.min_point_confidence;
        self.point_extractor.strategy = self.config.extraction_strategy.clone();
    }
}

impl PointExtractor {
    /// Create a new point extractor
    pub fn new(strategy: ExtractionStrategy, confidence_threshold: f64) -> Self {
        Self {
            strategy,
            confidence_threshold,
            clause_patterns: Self::create_default_clause_patterns(),
            extraction_cache: HashMap::new(),
        }
    }
    
    /// Extract points from a positional sentence
    pub fn extract_points(&mut self, sentence: &PositionalSentence, context: &VecDeque<PositionalSentence>) -> Result<Vec<TextPoint>> {
        // Check cache
        let cache_key = format!("{}:{}", sentence.original_text, sentence.positional_hash);
        if let Some(cached) = self.extraction_cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        let points = match self.strategy {
            ExtractionStrategy::OnePerSentence => {
                vec![self.extract_single_point(sentence)?]
            },
            ExtractionStrategy::MultiplePerSentence => {
                self.extract_multiple_points(sentence)?
            },
            ExtractionStrategy::SemanticDensity => {
                self.extract_by_semantic_density(sentence)?
            },
            ExtractionStrategy::Custom => {
                self.extract_custom_points(sentence)?
            },
        };
        
        // Filter by confidence threshold
        let filtered_points: Vec<TextPoint> = points.into_iter()
            .filter(|p| p.confidence >= self.confidence_threshold)
            .collect();
        
        // Cache result
        self.extraction_cache.insert(cache_key, filtered_points.clone());
        
        Ok(filtered_points)
    }
    
    /// Extract a single point from the sentence
    fn extract_single_point(&self, sentence: &PositionalSentence) -> Result<TextPoint> {
        let mut point = TextPoint::new(sentence.original_text.clone(), sentence.analysis_confidence);
        
        // Add positional metadata
        point.metadata.insert("semantic_signature".to_string(), 
            Value::String(sentence.semantic_signature.clone()));
        point.metadata.insert("order_dependency".to_string(), 
            Value::Number(sentence.order_dependency_score));
        point.metadata.insert("word_count".to_string(), 
            Value::Number(sentence.words.len() as f64));
            
        Ok(point)
    }
    
    /// Extract multiple points from clauses
    fn extract_multiple_points(&self, sentence: &PositionalSentence) -> Result<Vec<TextPoint>> {
        let mut points = Vec::new();
        
        // For now, just create one point per sentence
        // In a full implementation, this would detect clauses
        points.push(self.extract_single_point(sentence)?);
        
        Ok(points)
    }
    
    /// Extract points based on semantic density
    fn extract_by_semantic_density(&self, sentence: &PositionalSentence) -> Result<Vec<TextPoint>> {
        // Calculate semantic density based on content words and their weights
        let content_words: Vec<&crate::turbulance::positional_semantics::PositionalWord> = sentence.words.iter()
            .filter(|w| w.is_content_word)
            .collect();
            
        if content_words.len() >= 3 {
            // High semantic density - create multiple points
            self.extract_multiple_points(sentence)
        } else {
            // Low semantic density - create single point
            Ok(vec![self.extract_single_point(sentence)?])
        }
    }
    
    /// Extract points using custom rules
    fn extract_custom_points(&self, sentence: &PositionalSentence) -> Result<Vec<TextPoint>> {
        // Placeholder for custom extraction logic
        self.extract_single_point(sentence).map(|p| vec![p])
    }
    
    /// Create default clause detection patterns
    fn create_default_clause_patterns() -> Vec<ClausePattern> {
        vec![
            ClausePattern {
                pattern: r"\bthat\b".to_string(),
                confidence: 0.8,
                clause_type: ClauseType::Subordinate,
            },
            ClausePattern {
                pattern: r"\bwhich\b".to_string(),
                confidence: 0.9,
                clause_type: ClauseType::Relative,
            },
            ClausePattern {
                pattern: r"\bif\b".to_string(),
                confidence: 0.7,
                clause_type: ClauseType::Conditional,
            },
        ]
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_context_window: 5,
            min_point_confidence: 0.5,
            real_time_processing: true,
            boundary_detection: BoundaryDetectionMode::Advanced,
            extraction_strategy: ExtractionStrategy::OnePerSentence,
            sentence_buffer_size: 10,
            enable_positional_analysis: true,
            enable_debate_platforms: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stream_creation() {
        let config = StreamConfig::default();
        let stream = TextStream::new(config);
        assert_eq!(stream.get_state(), &StreamState::Ready);
    }
    
    #[tokio::test]
    async fn test_sentence_detection() {
        let config = StreamConfig::default();
        let stream = TextStream::new(config);
        
        let text = "Hello world. How are you? I am fine!";
        let sentences = stream.detect_sentences(text).unwrap();
        
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world");
        assert_eq!(sentences[1], "How are you");
        assert_eq!(sentences[2], "I am fine");
    }
    
    #[tokio::test]
    async fn test_stream_processing() {
        let config = StreamConfig::default();
        let mut stream = TextStream::new(config);
        
        let text = "The solution is optimal. It works very well.";
        let result = stream.process_chunk(text).await.unwrap();
        
        assert!(!result.points.is_empty());
        assert_eq!(result.state, StreamState::Ready);
        assert!(stream.get_stats().sentences_processed > 0);
    }
    
    #[tokio::test]
    async fn test_point_extraction() {
        let mut extractor = PointExtractor::new(ExtractionStrategy::OnePerSentence, 0.3);
        let mut analyzer = PositionalAnalyzer::new();
        
        let sentence = analyzer.analyze("The cat sat on the mat").unwrap();
        let context = VecDeque::new();
        
        let points = extractor.extract_points(&sentence, &context).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].content, "The cat sat on the mat");
    }
    
    #[tokio::test]
    async fn test_context_building() {
        let config = StreamConfig {
            max_context_window: 3,
            ..StreamConfig::default()
        };
        let mut stream = TextStream::new(config);
        
        // Process multiple chunks
        let texts = [
            "First sentence here.",
            "Second sentence follows.",
            "Third sentence continues.",
            "Fourth sentence concludes.",
        ];
        
        for text in &texts {
            let _result = stream.process_chunk(text).await.unwrap();
        }
        
        // Context window should be limited to 3
        assert!(stream.sentence_buffer.len() <= 3);
        assert!(stream.get_stats().context_utilization <= 1.0);
    }
} 