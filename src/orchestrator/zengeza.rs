use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn};

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Statistical metrics for noise analysis
#[derive(Debug, Clone)]
pub struct NoiseMetrics {
    pub word_frequency_variance: f64,
    pub positional_entropy: f64,
    pub semantic_coherence: f64,
    pub information_density: f64,
    pub syntactic_complexity: f64,
    pub redundancy_ratio: f64,
}

impl NoiseMetrics {
    pub fn new() -> Self {
        Self {
            word_frequency_variance: 0.0,
            positional_entropy: 0.0,
            semantic_coherence: 0.0,
            information_density: 0.0,
            syntactic_complexity: 0.0,
            redundancy_ratio: 0.0,
        }
    }

    pub fn compute_noise_score(&self) -> f64 {
        // Weighted combination of noise indicators
        let weights = [0.15, 0.20, 0.25, 0.20, 0.10, 0.10];
        let metrics = [
            1.0 - self.word_frequency_variance.min(1.0),
            1.0 - self.positional_entropy.min(1.0),
            self.semantic_coherence,
            self.information_density,
            1.0 - self.syntactic_complexity.min(1.0),
            1.0 - self.redundancy_ratio.min(1.0),
        ];
        
        metrics.iter()
            .zip(weights.iter())
            .map(|(metric, weight)| metric * weight)
            .sum()
    }
}

/// Position-based analysis for semantic importance
#[derive(Debug, Clone)]
pub struct PositionalImportance {
    pub position: usize,
    pub sentence_position: usize,
    pub paragraph_position: usize,
    pub proximity_to_key_terms: f64,
    pub syntactic_role: String,
    pub semantic_weight: f64,
}

impl PositionalImportance {
    pub fn calculate_importance(&self) -> f64 {
        // Position-based importance calculation
        let sentence_importance = match self.sentence_position {
            0 => 0.9,  // First word in sentence
            pos if pos < 3 => 0.8,  // Early in sentence
            _ => 0.4,  // Later in sentence
        };
        
        let paragraph_importance = match self.paragraph_position {
            0 => 0.95, // First sentence in paragraph
            1 => 0.85, // Second sentence
            _ => 0.6,  // Later sentences
        };
        
        let role_importance = match self.syntactic_role.as_str() {
            "NOUN" => 0.9,
            "VERB" => 0.8,
            "ADJ" => 0.6,
            "ADV" => 0.4,
            "PREP" => 0.2,
            "CONJ" => 0.1,
            _ => 0.5,
        };
        
        // Combine factors
        (sentence_importance * 0.3 + 
         paragraph_importance * 0.3 + 
         role_importance * 0.2 + 
         self.proximity_to_key_terms * 0.1 + 
         self.semantic_weight * 0.1)
    }
}

/// Statistical analysis results for noise reduction
#[derive(Debug, Clone)]
pub struct NoiseAnalysis {
    pub original_length: usize,
    pub noise_level: f64,
    pub word_importance: HashMap<String, f64>,
    pub positional_importance: Vec<PositionalImportance>,
    pub redundant_phrases: Vec<String>,
    pub preservation_score: f64,
}

impl NoiseAnalysis {
    pub fn should_preserve_word(&self, word: &str, position: usize) -> bool {
        let word_importance = self.word_importance.get(word).unwrap_or(&0.5);
        let positional_importance = self.positional_importance
            .get(position)
            .map(|p| p.calculate_importance())
            .unwrap_or(0.5);
        
        // Combine importance scores
        let combined_importance = (word_importance + positional_importance) / 2.0;
        combined_importance > 0.6
    }
    
    pub fn calculate_reduction_percentage(&self) -> f64 {
        let preserved_words = self.word_importance
            .values()
            .filter(|&&importance| importance > 0.6)
            .count();
        
        let total_words = self.word_importance.len();
        if total_words == 0 {
            return 0.0;
        }
        
        (1.0 - (preserved_words as f64 / total_words as f64)) * 100.0
    }
}

/// Machine learning model for noise detection
#[derive(Debug)]
pub struct MLNoiseDetector {
    word_frequency_model: HashMap<String, f64>,
    ngram_importance: HashMap<String, f64>,
    training_iterations: usize,
    learning_rate: f64,
}

impl MLNoiseDetector {
    pub fn new() -> Self {
        Self {
            word_frequency_model: HashMap::new(),
            ngram_importance: HashMap::new(),
            training_iterations: 0,
            learning_rate: 0.01,
        }
    }
    
    pub fn train_on_text(&mut self, text: &str, importance_labels: &HashMap<String, f64>) {
        // Update word frequency model
        for word in text.split_whitespace() {
            let current_freq = self.word_frequency_model.get(word).unwrap_or(&0.0);
            let importance = importance_labels.get(word).unwrap_or(&0.5);
            
            // Simple gradient descent update
            let new_freq = current_freq + self.learning_rate * (importance - current_freq);
            self.word_frequency_model.insert(word.to_string(), new_freq);
        }
        
        // Update n-gram importance
        let words: Vec<&str> = text.split_whitespace().collect();
        for window in words.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            let avg_importance = (importance_labels.get(window[0]).unwrap_or(&0.5) + 
                                importance_labels.get(window[1]).unwrap_or(&0.5)) / 2.0;
            
            let current_importance = self.ngram_importance.get(&bigram).unwrap_or(&0.0);
            let new_importance = current_importance + self.learning_rate * (avg_importance - current_importance);
            self.ngram_importance.insert(bigram, new_importance);
        }
        
        self.training_iterations += 1;
    }
    
    pub fn predict_word_importance(&self, word: &str) -> f64 {
        self.word_frequency_model.get(word).unwrap_or(&0.5).clone()
    }
    
    pub fn predict_phrase_importance(&self, phrase: &str) -> f64 {
        self.ngram_importance.get(phrase).unwrap_or(&0.5).clone()
    }
}

/// Zengeza intelligent noise reduction engine
pub struct ZengezaNoiseReduction {
    name: String,
    ml_detector: Arc<Mutex<MLNoiseDetector>>,
    noise_threshold: f64,
    preservation_mode: bool,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl ZengezaNoiseReduction {
    pub fn new() -> Self {
        Self {
            name: "Zengeza".to_string(),
            ml_detector: Arc::new(Mutex::new(MLNoiseDetector::new())),
            noise_threshold: 0.4,
            preservation_mode: false,
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
    
    pub fn with_noise_threshold(mut self, threshold: f64) -> Self {
        self.noise_threshold = threshold.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_preservation_mode(mut self, enabled: bool) -> Self {
        self.preservation_mode = enabled;
        self
    }
    
    fn analyze_text_noise(&self, text: &str) -> NoiseAnalysis {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_importance = HashMap::new();
        let mut positional_importance = Vec::new();
        
        // Analyze each word
        for (i, word) in words.iter().enumerate() {
            let ml_importance = self.ml_detector.lock().unwrap().predict_word_importance(word);
            
            // Calculate positional importance
            let sentence_pos = self.calculate_sentence_position(text, i);
            let paragraph_pos = self.calculate_paragraph_position(text, i);
            let proximity = self.calculate_proximity_to_key_terms(text, i);
            let syntactic_role = self.determine_syntactic_role(word);
            
            let pos_importance = PositionalImportance {
                position: i,
                sentence_position: sentence_pos,
                paragraph_position: paragraph_pos,
                proximity_to_key_terms: proximity,
                syntactic_role: syntactic_role.clone(),
                semantic_weight: ml_importance,
            };
            
            let final_importance = (ml_importance + pos_importance.calculate_importance()) / 2.0;
            word_importance.insert(word.to_string(), final_importance);
            positional_importance.push(pos_importance);
        }
        
        // Detect redundant phrases
        let redundant_phrases = self.detect_redundant_phrases(text);
        
        // Calculate noise level
        let noise_level = self.calculate_noise_metrics(text);
        
        NoiseAnalysis {
            original_length: text.len(),
            noise_level: noise_level.compute_noise_score(),
            word_importance,
            positional_importance,
            redundant_phrases,
            preservation_score: 0.85, // Default preservation target
        }
    }
    
    fn calculate_sentence_position(&self, _text: &str, word_index: usize) -> usize {
        // Simplified: assume each sentence has ~10 words
        word_index % 10
    }
    
    fn calculate_paragraph_position(&self, _text: &str, word_index: usize) -> usize {
        // Simplified: assume each paragraph has ~50 words
        word_index % 50
    }
    
    fn calculate_proximity_to_key_terms(&self, text: &str, word_index: usize) -> f64 {
        let key_terms = ["important", "key", "main", "critical", "essential"];
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let mut min_distance = f64::INFINITY;
        for (i, word) in words.iter().enumerate() {
            if key_terms.contains(&word.to_lowercase().as_str()) {
                let distance = (i as f64 - word_index as f64).abs();
                min_distance = min_distance.min(distance);
            }
        }
        
        if min_distance == f64::INFINITY {
            0.0
        } else {
            (1.0 / (1.0 + min_distance / 10.0)).min(1.0)
        }
    }
    
    fn determine_syntactic_role(&self, word: &str) -> String {
        // Simplified POS tagging
        if word.ends_with("ing") || word.ends_with("ed") {
            "VERB".to_string()
        } else if word.ends_with("ly") {
            "ADV".to_string()
        } else if ["the", "a", "an"].contains(&word.to_lowercase().as_str()) {
            "DET".to_string()
        } else if ["and", "or", "but"].contains(&word.to_lowercase().as_str()) {
            "CONJ".to_string()
        } else {
            "NOUN".to_string()
        }
    }
    
    fn detect_redundant_phrases(&self, text: &str) -> Vec<String> {
        let mut redundant_phrases = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // Look for repeated phrases
        for window_size in 2..=5 {
            for i in 0..words.len().saturating_sub(window_size) {
                let phrase = words[i..i + window_size].join(" ");
                let remaining_text = words[i + window_size..].join(" ");
                
                if remaining_text.contains(&phrase) {
                    redundant_phrases.push(phrase);
                }
            }
        }
        
        redundant_phrases
    }
    
    fn calculate_noise_metrics(&self, text: &str) -> NoiseMetrics {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_freq = HashMap::new();
        
        // Calculate word frequencies
        for word in &words {
            *word_freq.entry(word.to_string()).or_insert(0) += 1;
        }
        
        let variance = self.calculate_frequency_variance(&word_freq);
        let entropy = self.calculate_positional_entropy(text);
        let coherence = self.calculate_semantic_coherence(text);
        let density = self.calculate_information_density(text);
        let complexity = self.calculate_syntactic_complexity(text);
        let redundancy = self.calculate_redundancy_ratio(text);
        
        NoiseMetrics {
            word_frequency_variance: variance,
            positional_entropy: entropy,
            semantic_coherence: coherence,
            information_density: density,
            syntactic_complexity: complexity,
            redundancy_ratio: redundancy,
        }
    }
    
    fn calculate_frequency_variance(&self, word_freq: &HashMap<String, i32>) -> f64 {
        if word_freq.is_empty() {
            return 0.0;
        }
        
        let mean = word_freq.values().sum::<i32>() as f64 / word_freq.len() as f64;
        let variance = word_freq.values()
            .map(|&freq| (freq as f64 - mean).powi(2))
            .sum::<f64>() / word_freq.len() as f64;
        
        variance.sqrt() / mean
    }
    
    fn calculate_positional_entropy(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let mut position_freq = HashMap::new();
        for (i, word) in words.iter().enumerate() {
            let position_type = match i {
                0 => "start",
                i if i == words.len() - 1 => "end",
                _ => "middle",
            };
            *position_freq.entry((word.to_string(), position_type.to_string())).or_insert(0) += 1;
        }
        
        let total = position_freq.values().sum::<i32>() as f64;
        let entropy = position_freq.values()
            .map(|&freq| {
                let p = freq as f64 / total;
                -p * p.log2()
            })
            .sum::<f64>();
        
        entropy / (words.len() as f64).log2()
    }
    
    fn calculate_semantic_coherence(&self, text: &str) -> f64 {
        // Simplified semantic coherence based on sentence transitions
        let sentences: Vec<&str> = text.split('.').collect();
        if sentences.len() < 2 {
            return 1.0;
        }
        
        let mut coherence_scores = Vec::new();
        for i in 0..sentences.len() - 1 {
            let current_words: Vec<&str> = sentences[i].split_whitespace().collect();
            let next_words: Vec<&str> = sentences[i + 1].split_whitespace().collect();
            
            let overlap = current_words.iter()
                .filter(|word| next_words.contains(word))
                .count();
            
            let max_len = current_words.len().max(next_words.len());
            if max_len > 0 {
                coherence_scores.push(overlap as f64 / max_len as f64);
            }
        }
        
        coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
    }
    
    fn calculate_information_density(&self, text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        
        if words.is_empty() {
            0.0
        } else {
            unique_words.len() as f64 / words.len() as f64
        }
    }
    
    fn calculate_syntactic_complexity(&self, text: &str) -> f64 {
        let sentences: Vec<&str> = text.split('.').collect();
        if sentences.is_empty() {
            return 0.0;
        }
        
        let avg_sentence_length = sentences.iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f64 / sentences.len() as f64;
        
        // Normalize to 0-1 range (assuming 30 words is very complex)
        (avg_sentence_length / 30.0).min(1.0)
    }
    
    fn calculate_redundancy_ratio(&self, text: &str) -> f64 {
        let redundant_phrases = self.detect_redundant_phrases(text);
        let total_phrases = text.split_whitespace().count() / 2; // Approximate phrase count
        
        if total_phrases == 0 {
            0.0
        } else {
            redundant_phrases.len() as f64 / total_phrases as f64
        }
    }
    
    fn reduce_noise(&self, text: &str, analysis: &NoiseAnalysis) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut result = Vec::new();
        
        for (i, word) in words.iter().enumerate() {
            if analysis.should_preserve_word(word, i) {
                result.push(word.to_string());
            } else if self.preservation_mode {
                // In preservation mode, keep but mark as low importance
                result.push(format!("[{}]", word));
            }
            // Otherwise, remove the word
        }
        
        result.join(" ")
    }
}

#[async_trait]
impl StreamProcessor for ZengezaNoiseReduction {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let name = self.name.clone();
        let ml_detector = self.ml_detector.clone();
        let noise_threshold = self.noise_threshold;
        let preservation_mode = self.preservation_mode;
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut processed_count = 0;
            
            while let Some(mut data) = input.recv().await {
                let start_time = std::time::Instant::now();
                
                debug!("Zengeza processing text: {} chars", data.content.len());
                
                // Skip if already processed
                if data.state.contains_key("zengeza_processed") {
                    if tx.send(data).await.is_err() {
                        break;
                    }
                    continue;
                }
                
                // Perform noise analysis
                let detector = ZengezaNoiseReduction {
                    name: name.clone(),
                    ml_detector: ml_detector.clone(),
                    noise_threshold,
                    preservation_mode,
                    stats: stats.clone(),
                };
                
                let analysis = detector.analyze_text_noise(&data.content);
                
                // Only reduce noise if it's above threshold
                if analysis.noise_level > noise_threshold {
                    let reduced_text = detector.reduce_noise(&data.content, &analysis);
                    
                    // Update metadata
                    data.content = reduced_text;
                    data.metadata.insert("zengeza_noise_level".to_string(), 
                                       format!("{:.3}", analysis.noise_level));
                    data.metadata.insert("zengeza_reduction_percent".to_string(), 
                                       format!("{:.1}%", analysis.calculate_reduction_percentage()));
                    data.metadata.insert("zengeza_original_length".to_string(), 
                                       analysis.original_length.to_string());
                    data.state.insert("zengeza_processed".to_string(), "true".to_string());
                    
                    info!("Zengeza reduced noise by {:.1}% (noise level: {:.3})", 
                          analysis.calculate_reduction_percentage(), analysis.noise_level);
                } else {
                    data.metadata.insert("zengeza_noise_level".to_string(), "low".to_string());
                    data.state.insert("zengeza_processed".to_string(), "skipped".to_string());
                }
                
                // Update confidence based on noise reduction
                if analysis.noise_level > noise_threshold {
                    data.confidence = (data.confidence + 0.15).min(1.0);
                }
                
                processed_count += 1;
                
                // Update stats
                {
                    let mut stats_guard = stats.lock().unwrap();
                    stats_guard.items_processed += 1;
                    let processing_time = start_time.elapsed().as_millis() as f64;
                    stats_guard.average_processing_time_ms = 
                        (stats_guard.average_processing_time_ms * (processed_count - 1) as f64 + processing_time) / processed_count as f64;
                }
                
                if tx.send(data).await.is_err() {
                    break;
                }
            }
            
            info!("Zengeza processed {} items", processed_count);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn can_handle(&self, data: &StreamData) -> bool {
        // Can handle any text data
        !data.content.trim().is_empty()
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for ZengezaNoiseReduction {
    fn default() -> Self {
        Self::new()
    }
} 