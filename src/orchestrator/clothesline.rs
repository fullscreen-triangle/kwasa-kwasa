use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};
use rand::{thread_rng, Rng};
use uuid::Uuid;

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Different types of strategic text occlusion patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OcclusionPattern {
    Keyword,      // Hide key domain-specific terms
    Logical,      // Hide logical connectors (and, but, therefore)
    Positional,   // Hide words based on sentence position
    Semantic,     // Hide semantically important words
    Structural,   // Hide punctuation and structure
}

impl OcclusionPattern {
    pub fn description(&self) -> &'static str {
        match self {
            OcclusionPattern::Keyword => "Strategic removal of domain-specific terminology",
            OcclusionPattern::Logical => "Removal of logical connectors and reasoning links",
            OcclusionPattern::Positional => "Position-based word removal (first/last words)",
            OcclusionPattern::Semantic => "Removal of semantically dense words",
            OcclusionPattern::Structural => "Removal of punctuation and structural elements",
        }
    }

    pub fn difficulty_level(&self) -> u8 {
        match self {
            OcclusionPattern::Structural => 1,  // Easy - structure helps readability
            OcclusionPattern::Positional => 2, // Medium - position matters but predictable
            OcclusionPattern::Keyword => 3,    // Hard - domain knowledge required
            OcclusionPattern::Logical => 4,    // Very Hard - logical reasoning required
            OcclusionPattern::Semantic => 5,   // Extreme - deep understanding required
        }
    }
}

/// Represents an occlusion challenge for comprehension testing
#[derive(Debug, Clone)]
pub struct OcclusionChallenge {
    pub id: Uuid,
    pub original_text: String,
    pub occluded_text: String,
    pub hidden_words: Vec<String>,
    pub pattern_used: OcclusionPattern,
    pub expected_predictions: Vec<String>,
    pub difficulty_score: f64,
    pub context_preservation_score: f64,
}

impl OcclusionChallenge {
    pub fn new(original: String, pattern: OcclusionPattern) -> Self {
        let id = Uuid::new_v4();
        let difficulty_score = pattern.difficulty_level() as f64 / 5.0;
        
        Self {
            id,
            original_text: original.clone(),
            occluded_text: String::new(),
            hidden_words: Vec::new(),
            pattern_used: pattern,
            expected_predictions: Vec::new(),
            difficulty_score,
            context_preservation_score: 0.0,
        }
    }

    pub fn apply_occlusion(&mut self) -> Result<(), String> {
        match self.pattern_used {
            OcclusionPattern::Keyword => self.apply_keyword_occlusion(),
            OcclusionPattern::Logical => self.apply_logical_occlusion(),
            OcclusionPattern::Positional => self.apply_positional_occlusion(),
            OcclusionPattern::Semantic => self.apply_semantic_occlusion(),
            OcclusionPattern::Structural => self.apply_structural_occlusion(),
        }
    }

    fn apply_keyword_occlusion(&mut self) -> Result<(), String> {
        let keywords = self.extract_domain_keywords();
        let mut occluded = self.original_text.clone();
        
        // Hide 30% of domain-specific keywords
        let hide_count = (keywords.len() as f64 * 0.3).ceil() as usize;
        let mut rng = thread_rng();
        
        for _ in 0..hide_count.min(keywords.len()) {
            if let Some(keyword) = keywords.get(rng.gen_range(0..keywords.len())) {
                if !self.hidden_words.contains(keyword) {
                    occluded = occluded.replace(keyword, "_____");
                    self.hidden_words.push(keyword.clone());
                    self.expected_predictions.push(keyword.clone());
                }
            }
        }
        
        self.occluded_text = occluded;
        self.context_preservation_score = 0.7; // Keywords are challenging but context remains
        Ok(())
    }

    fn apply_logical_occlusion(&mut self) -> Result<(), String> {
        let logical_connectors = vec![
            "and", "but", "however", "therefore", "because", "since", "although",
            "while", "whereas", "moreover", "furthermore", "consequently", "thus",
            "hence", "nevertheless", "nonetheless", "meanwhile", "in contrast"
        ];
        
        let mut occluded = self.original_text.clone();
        
        for connector in &logical_connectors {
            if occluded.contains(connector) {
                occluded = occluded.replace(connector, "_____");
                self.hidden_words.push(connector.to_string());
                self.expected_predictions.push(connector.to_string());
            }
        }
        
        self.occluded_text = occluded;
        self.context_preservation_score = 0.5; // Logical flow is severely impacted
        Ok(())
    }

    fn apply_positional_occlusion(&mut self) -> Result<(), String> {
        let sentences: Vec<&str> = self.original_text.split('.').collect();
        let mut occluded_sentences = Vec::new();
        
        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            if words.len() > 2 {
                let mut occluded_words = words.clone();
                
                // Hide first word (often important for sentence meaning)
                if !words[0].is_empty() {
                    self.hidden_words.push(words[0].to_string());
                    self.expected_predictions.push(words[0].to_string());
                    occluded_words[0] = "_____";
                }
                
                // Hide last word (often contains key information)
                let last_idx = words.len() - 1;
                if !words[last_idx].is_empty() {
                    self.hidden_words.push(words[last_idx].to_string());
                    self.expected_predictions.push(words[last_idx].to_string());
                    occluded_words[last_idx] = "_____";
                }
                
                occluded_sentences.push(occluded_words.join(" "));
            } else {
                occluded_sentences.push(sentence.to_string());
            }
        }
        
        self.occluded_text = occluded_sentences.join(".");
        self.context_preservation_score = 0.8; // Positional occlusion preserves middle context
        Ok(())
    }

    fn apply_semantic_occlusion(&mut self) -> Result<(), String> {
        let words: Vec<&str> = self.original_text.split_whitespace().collect();
        let semantic_scores = self.calculate_semantic_importance(&words);
        
        // Hide top 20% semantically important words
        let mut semantic_pairs: Vec<(usize, f64)> = semantic_scores.iter().enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        semantic_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let hide_count = (words.len() as f64 * 0.2).ceil() as usize;
        let mut occluded_words = words.clone();
        
        for &(idx, _) in semantic_pairs.iter().take(hide_count) {
            self.hidden_words.push(words[idx].to_string());
            self.expected_predictions.push(words[idx].to_string());
            occluded_words[idx] = "_____";
        }
        
        self.occluded_text = occluded_words.join(" ");
        self.context_preservation_score = 0.3; // Semantic occlusion severely impacts meaning
        Ok(())
    }

    fn apply_structural_occlusion(&mut self) -> Result<(), String> {
        let mut occluded = self.original_text.clone();
        let structural_elements = vec![",", ".", ":", ";", "(", ")", "[", "]", "\"", "'"];
        
        for element in &structural_elements {
            occluded = occluded.replace(element, "");
        }
        
        // Remove double spaces
        while occluded.contains("  ") {
            occluded = occluded.replace("  ", " ");
        }
        
        self.occluded_text = occluded;
        self.context_preservation_score = 0.9; // Structure removal is challenging but meaning preserved
        Ok(())
    }

    fn extract_domain_keywords(&self) -> Vec<String> {
        // Simple keyword extraction based on word frequency and length
        let words: Vec<&str> = self.original_text.split_whitespace().collect();
        let mut word_freq = HashMap::new();
        
        for word in &words {
            if word.len() > 4 && !self.is_common_word(word) {
                *word_freq.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }
        
        word_freq.into_iter()
            .filter(|(_, freq)| *freq >= 1)
            .map(|(word, _)| word)
            .collect()
    }

    fn is_common_word(&self, word: &str) -> bool {
        let common_words = vec![
            "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
            "his", "from", "they", "she", "her", "been", "than", "its", "who", "did"
        ];
        common_words.contains(&word.to_lowercase().as_str())
    }

    fn calculate_semantic_importance(&self, words: &[&str]) -> Vec<f64> {
        words.iter().map(|word| {
            let mut score = 0.0;
            
            // Length-based importance
            score += word.len() as f64 * 0.1;
            
            // Part-of-speech importance (simplified)
            if word.chars().all(|c| c.is_uppercase()) {
                score += 0.8; // Acronyms/proper nouns
            }
            if word.chars().next().map_or(false, |c| c.is_uppercase()) {
                score += 0.4; // Capitalized words
            }
            
            // Domain-specific patterns
            if word.contains("_") || word.contains("-") {
                score += 0.6; // Technical terms
            }
            
            // Common word penalty
            if self.is_common_word(word) {
                score *= 0.2;
            }
            
            score.clamp(0.0, 1.0)
        }).collect()
    }
}

/// Comprehension validation result
#[derive(Debug, Clone)]
pub struct ComprehensionResult {
    pub challenge_id: Uuid,
    pub predicted_words: Vec<String>,
    pub accuracy: f64,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub validation_passed: bool,
    pub remediation_needed: bool,
}

impl ComprehensionResult {
    pub fn calculate_transition_confidence(&self) -> f64 {
        // Formula for determining if comprehension is sufficient for layer transition
        let accuracy_weight = 0.4;
        let confidence_weight = 0.3;
        let speed_weight = 0.2;
        let difficulty_weight = 0.1;
        
        let speed_score = if self.processing_time_ms < 1000 { 1.0 } 
                         else { 1.0 - (self.processing_time_ms as f64 / 10000.0).min(0.8) };
        
        let overall_confidence = 
            self.accuracy * accuracy_weight +
            self.confidence * confidence_weight +
            speed_score * speed_weight +
            (if self.validation_passed { 1.0 } else { 0.0 }) * difficulty_weight;
        
        overall_confidence.clamp(0.0, 1.0)
    }
}

/// The Clothesline Module - Comprehension Validator & Context Layer Gatekeeper
pub struct ClotheslineModule {
    name: String,
    
    // Challenge Management
    active_challenges: Arc<Mutex<HashMap<Uuid, OcclusionChallenge>>>,
    validation_history: Arc<Mutex<Vec<ComprehensionResult>>>,
    
    // Configuration
    validation_threshold: f64,
    max_attempts_per_challenge: u8,
    remediation_enabled: bool,
    
    // V8 Integration
    champagne_integration: bool,
    
    // Statistics
    stats: Arc<Mutex<ProcessorStats>>,
    success_rate: Arc<Mutex<f64>>,
    average_accuracy: Arc<Mutex<f64>>,
}

impl ClotheslineModule {
    pub fn new() -> Self {
        Self {
            name: "ClotheslineModule".to_string(),
            active_challenges: Arc::new(Mutex::new(HashMap::new())),
            validation_history: Arc::new(Mutex::new(Vec::new())),
            validation_threshold: 0.7,
            max_attempts_per_challenge: 3,
            remediation_enabled: true,
            champagne_integration: true,
            stats: Arc::new(Mutex::new(ProcessorStats::new())),
            success_rate: Arc::new(Mutex::new(0.0)),
            average_accuracy: Arc::new(Mutex::new(0.0)),
        }
    }

    pub fn with_validation_threshold(mut self, threshold: f64) -> Self {
        self.validation_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn with_remediation(mut self, enabled: bool) -> Self {
        self.remediation_enabled = enabled;
        self
    }

    pub fn with_champagne_integration(mut self, enabled: bool) -> Self {
        self.champagne_integration = enabled;
        self
    }

    /// Create a comprehension challenge from text
    pub async fn create_challenge(&self, text: &str, pattern: OcclusionPattern) -> Result<Uuid, String> {
        let mut challenge = OcclusionChallenge::new(text.to_string(), pattern);
        challenge.apply_occlusion()?;
        
        let challenge_id = challenge.id;
        self.active_challenges.lock().unwrap().insert(challenge_id, challenge);
        
        info!("🧪 Created {:?} occlusion challenge {}", pattern, challenge_id);
        Ok(challenge_id)
    }

    /// Validate comprehension through prediction
    pub async fn validate_comprehension(&self, challenge_id: Uuid, predictions: Vec<String>) -> Result<ComprehensionResult, String> {
        let start_time = std::time::Instant::now();
        
        let challenge = self.active_challenges.lock().unwrap()
            .get(&challenge_id)
            .ok_or("Challenge not found")?
            .clone();
        
        // Calculate prediction accuracy
        let accuracy = self.calculate_prediction_accuracy(&challenge.expected_predictions, &predictions);
        
        // Calculate confidence based on accuracy and pattern difficulty
        let confidence = accuracy * (1.0 - challenge.difficulty_score * 0.2);
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        let validation_passed = accuracy >= self.validation_threshold;
        
        let result = ComprehensionResult {
            challenge_id,
            predicted_words: predictions,
            accuracy,
            confidence,
            processing_time_ms: processing_time,
            validation_passed,
            remediation_needed: !validation_passed && self.remediation_enabled,
        };
        
        // Store result
        self.validation_history.lock().unwrap().push(result.clone());
        
        // Update statistics
        self.update_statistics(&result);
        
        if validation_passed {
            info!("✅ Comprehension validation passed for challenge {} (accuracy: {:.2})", 
                  challenge_id, accuracy);
        } else {
            warn!("❌ Comprehension validation failed for challenge {} (accuracy: {:.2})", 
                  challenge_id, accuracy);
            
            if result.remediation_needed {
                self.initiate_remediation(&challenge).await?;
            }
        }
        
        Ok(result)
    }

    /// Check if text understanding is genuine (not just pattern matching)
    pub async fn validate_genuine_understanding(&self, text: &str) -> Result<f64, String> {
        // Create multiple challenges with different patterns
        let patterns = vec![
            OcclusionPattern::Keyword,
            OcclusionPattern::Logical,
            OcclusionPattern::Semantic,
        ];
        
        let mut total_confidence = 0.0;
        let mut challenge_count = 0;
        
        for pattern in patterns {
            let challenge_id = self.create_challenge(text, pattern).await?;
            
            // Simulate AI prediction (in real implementation, this would call the AI)
            let predictions = self.simulate_ai_predictions(&challenge_id).await?;
            
            let result = self.validate_comprehension(challenge_id, predictions).await?;
            total_confidence += result.calculate_transition_confidence();
            challenge_count += 1;
        }
        
        let average_confidence = if challenge_count > 0 {
            total_confidence / challenge_count as f64
        } else {
            0.0
        };
        
        // Context layer gatekeeper decision
        if average_confidence >= self.validation_threshold {
            info!("🚪 Context layer gatekeeper: ALLOW transition (confidence: {:.2})", 
                  average_confidence);
        } else {
            warn!("🚪 Context layer gatekeeper: BLOCK transition (confidence: {:.2})", 
                  average_confidence);
        }
        
        Ok(average_confidence)
    }

    /// Initiate remediation for failed comprehension
    async fn initiate_remediation(&self, challenge: &OcclusionChallenge) -> Result<(), String> {
        info!("🔧 Initiating remediation for failed comprehension challenge {}", challenge.id);
        
        // Remediation strategies based on pattern type
        match challenge.pattern_used {
            OcclusionPattern::Keyword => {
                info!("📚 Remediation: Providing domain-specific vocabulary training");
            }
            OcclusionPattern::Logical => {
                info!("🔗 Remediation: Strengthening logical reasoning connections");
            }
            OcclusionPattern::Semantic => {
                info!("🧠 Remediation: Deep semantic analysis training");
            }
            OcclusionPattern::Positional => {
                info!("📍 Remediation: Position-based context training");
            }
            OcclusionPattern::Structural => {
                info!("🏗️ Remediation: Structural pattern recognition training");
            }
        }
        
        // If champagne integration is enabled, store for dream processing
        if self.champagne_integration {
            info!("🍾 Scheduling challenge for champagne phase deep learning");
        }
        
        Ok(())
    }

    fn calculate_prediction_accuracy(&self, expected: &[String], predicted: &[String]) -> f64 {
        if expected.is_empty() {
            return 1.0;
        }
        
        let mut correct = 0;
        let mut total = 0;
        
        for exp_word in expected {
            total += 1;
            for pred_word in predicted {
                if exp_word.to_lowercase() == pred_word.to_lowercase() {
                    correct += 1;
                    break;
                }
                // Partial match for similar words
                if self.words_similar(exp_word, pred_word) {
                    correct += 1;
                    break;
                }
            }
        }
        
        if total == 0 {
            1.0
        } else {
            correct as f64 / total as f64
        }
    }

    fn words_similar(&self, word1: &str, word2: &str) -> bool {
        // Simple similarity check (could be enhanced with edit distance)
        let w1 = word1.to_lowercase();
        let w2 = word2.to_lowercase();
        
        if w1.len() < 3 || w2.len() < 3 {
            return false;
        }
        
        // Check if one word contains the other (for variations)
        w1.contains(&w2) || w2.contains(&w1)
    }

    // Simulate AI predictions for testing (in real implementation, integrate with AI)
    async fn simulate_ai_predictions(&self, challenge_id: &Uuid) -> Result<Vec<String>, String> {
        let challenge = self.active_challenges.lock().unwrap()
            .get(challenge_id)
            .ok_or("Challenge not found")?
            .clone();
        
        // Simulate predictions with some accuracy
        let mut predictions = Vec::new();
        let mut rng = thread_rng();
        
        for expected in &challenge.expected_predictions {
            if rng.gen_bool(0.8) { // 80% accuracy simulation
                predictions.push(expected.clone());
            } else {
                predictions.push("wrong".to_string());
            }
        }
        
        Ok(predictions)
    }

    fn update_statistics(&self, result: &ComprehensionResult) {
        let mut stats = self.stats.lock().unwrap();
        stats.increment_processed_count();
        
        // Update success rate
        let history = self.validation_history.lock().unwrap();
        let total_validations = history.len();
        let successful_validations = history.iter()
            .filter(|r| r.validation_passed)
            .count();
        
        if total_validations > 0 {
            *self.success_rate.lock().unwrap() = successful_validations as f64 / total_validations as f64;
        }
        
        // Update average accuracy
        let total_accuracy: f64 = history.iter().map(|r| r.accuracy).sum();
        if total_validations > 0 {
            *self.average_accuracy.lock().unwrap() = total_accuracy / total_validations as f64;
        }
    }

    /// Get comprehensive statistics
    pub fn get_clothesline_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let active_challenges = self.active_challenges.lock().unwrap().len();
        let total_validations = self.validation_history.lock().unwrap().len();
        let success_rate = *self.success_rate.lock().unwrap();
        let average_accuracy = *self.average_accuracy.lock().unwrap();
        
        stats.insert("active_challenges".to_string(), serde_json::Value::Number(active_challenges.into()));
        stats.insert("total_validations".to_string(), serde_json::Value::Number(total_validations.into()));
        stats.insert("success_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(success_rate).unwrap()));
        stats.insert("average_accuracy".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(average_accuracy).unwrap()));
        stats.insert("validation_threshold".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.validation_threshold).unwrap()));
        
        stats
    }
}

#[async_trait]
impl StreamProcessor for ClotheslineModule {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (output_tx, output_rx) = channel(100);
        
        let clothesline = Arc::new(self);
        tokio::spawn(async move {
            while let Some(data) = input.recv().await {
                match data {
                    StreamData::Text(content) => {
                        match clothesline.validate_genuine_understanding(&content).await {
                            Ok(confidence) => {
                                let output_data = StreamData::ProcessedText {
                                    content: content.clone(),
                                    metadata: {
                                        let mut meta = HashMap::new();
                                        meta.insert("comprehension_confidence".to_string(), confidence.to_string());
                                        meta.insert("validation_passed".to_string(), (confidence >= clothesline.validation_threshold).to_string());
                                        meta.insert("processor".to_string(), "Clothesline".to_string());
                                        meta
                                    },
                                    confidence: if confidence >= clothesline.validation_threshold {
                                        Confidence::High
                                    } else {
                                        Confidence::Low
                                    },
                                };
                                
                                if output_tx.send(output_data).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Clothesline validation error: {}", e);
                                if output_tx.send(StreamData::Error(e)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    other_data => {
                        if output_tx.send(other_data).await.is_err() {
                            break;
                        }
                    }
                }
            }
        });
        
        output_rx
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, data: &StreamData) -> bool {
        matches!(data, StreamData::Text(_))
    }

    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for ClotheslineModule {
    fn default() -> Self {
        Self::new()
    }
} 