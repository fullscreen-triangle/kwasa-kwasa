// Spectacular - The Extraordinary Handler
// Named after the English word denoting something extraordinary that demands special attention

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn};

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Indicators that content might be extraordinary
#[derive(Debug, Clone)]
pub enum ExtraordinaryIndicator {
    UnexpectedSemanticClarity {
        clarity_score: f64,
        baseline_comparison: f64,
    },
    ParadigmShiftingContent {
        shift_magnitude: f64,
        affected_domains: Vec<String>,
    },
    CrossDomainResonance {
        primary_domain: String,
        resonant_domains: Vec<String>,
        resonance_strength: f64,
    },
    NovelConceptualPattern {
        pattern_type: String,
        novelty_score: f64,
    },
    UnusualCoherence {
        coherence_score: f64,
        coherence_type: String,
        statistical_significance: f64,
    },
}

/// Significance scoring for extraordinary content
#[derive(Debug, Clone)]
pub struct SignificanceScore {
    pub composite_score: f64,
    pub impact_dimensions: Vec<f64>,
    pub historical_percentile: f64,
    pub expert_consensus: f64,
}

impl SignificanceScore {
    pub fn new() -> Self {
        Self {
            composite_score: 0.0,
            impact_dimensions: Vec::new(),
            historical_percentile: 0.5,
            expert_consensus: 0.5,
        }
    }
}

/// Registry for extraordinary discoveries
#[derive(Debug)]
pub struct SpectacularRegistry {
    discoveries: Vec<(Instant, String, SignificanceScore)>,
    max_entries: usize,
}

impl SpectacularRegistry {
    pub fn new() -> Self {
        Self {
            discoveries: Vec::new(),
            max_entries: 100,
        }
    }

    pub fn register_discovery(&mut self, content: String, significance: SignificanceScore) {
        self.discoveries.push((Instant::now(), content, significance));
        
        // Keep registry manageable
        if self.discoveries.len() > self.max_entries {
            self.discoveries.remove(0);
        }
    }

    pub fn get_most_significant(&self, count: usize) -> Vec<&(Instant, String, SignificanceScore)> {
        let mut discoveries = self.discoveries.iter().collect::<Vec<_>>();
        discoveries.sort_by(|a, b| b.2.composite_score.partial_cmp(&a.2.composite_score).unwrap_or(std::cmp::Ordering::Equal));
        discoveries.into_iter().take(count).collect()
    }
}

/// Main Spectacular handler
pub struct SpectacularHandler {
    significance_threshold: f64,
    atp_investment_base: f64,
    registry: Arc<Mutex<SpectacularRegistry>>,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl SpectacularHandler {
    pub fn new() -> Self {
        Self {
            significance_threshold: 0.8, // High threshold for extraordinary content
            atp_investment_base: 500.0,  // Base ATP for extraordinary processing
            registry: Arc::new(Mutex::new(SpectacularRegistry::new())),
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn with_significance_threshold(mut self, threshold: f64) -> Self {
        self.significance_threshold = threshold;
        self
    }

    pub fn with_atp_investment(mut self, investment: f64) -> Self {
        self.atp_investment_base = investment;
        self
    }

    /// Assess if content is extraordinary
    fn assess_extraordinariness(&self, content: &str) -> SignificanceScore {
        let mut score = SignificanceScore::new();
        
        // Simple heuristics for extraordinary content detection
        let word_count = content.split_whitespace().count();
        let unique_words = content.split_whitespace().collect::<std::collections::HashSet<_>>().len();
        let word_diversity = unique_words as f64 / word_count as f64;
        
        // Check for paradigm-shifting language
        let paradigm_indicators = [
            "breakthrough", "revolutionary", "unprecedented", "paradigm shift",
            "groundbreaking", "novel approach", "fundamental change", "innovative",
            "disruptive", "transformative"
        ];
        
        let paradigm_score = paradigm_indicators.iter()
            .map(|indicator| if content.to_lowercase().contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / paradigm_indicators.len() as f64;
        
        // Check for cross-domain indicators
        let domains = ["technology", "science", "medicine", "economics", "philosophy", "psychology"];
        let domain_mentions = domains.iter()
            .filter(|&&domain| content.to_lowercase().contains(domain))
            .count();
        
        let cross_domain_score = if domain_mentions > 1 { 0.8 } else { 0.2 };
        
        // Check for clarity indicators
        let clarity_indicators = ["clear", "obvious", "evident", "demonstrated", "proven"];
        let clarity_score = clarity_indicators.iter()
            .map(|indicator| if content.to_lowercase().contains(indicator) { 1.0 } else { 0.0 })
            .sum::<f64>() / clarity_indicators.len() as f64;
        
        // Compute composite score
        score.composite_score = (paradigm_score * 0.4 + cross_domain_score * 0.3 + clarity_score * 0.2 + word_diversity * 0.1).min(1.0);
        score.impact_dimensions = vec![paradigm_score, cross_domain_score, clarity_score, word_diversity];
        score.historical_percentile = score.composite_score; // Simplified
        score.expert_consensus = score.composite_score * 0.8; // Assume some agreement
        
        score
    }

    /// Compute ATP investment for extraordinary content
    fn compute_atp_investment(&self, significance: &SignificanceScore) -> f64 {
        let base = self.atp_investment_base;
        
        let significance_multiplier = match significance.composite_score {
            score if score > 0.95 => 10.0,  // Revolutionary insights
            score if score > 0.90 => 5.0,   // Major breakthroughs
            score if score > 0.80 => 3.0,   // Significant insights
            score if score > 0.70 => 2.0,   // Notable findings
            _ => 1.0,                        // Standard processing
        };
        
        let domain_multiplier = (significance.impact_dimensions.len() as f64).sqrt();
        
        (base * significance_multiplier * domain_multiplier).min(5000.0) // Cap at 5000 ATP
    }
}

#[async_trait]
impl StreamProcessor for SpectacularHandler {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        
        let registry = self.registry.clone();
        let stats = self.stats.clone();
        let significance_threshold = self.significance_threshold;
        let atp_investment_base = self.atp_investment_base;
        
        tokio::spawn(async move {
            while let Some(mut data) = input.recv().await {
                let start_time = Instant::now();
                
                debug!("Spectacular processing: {}", data.content);
                
                let spectacular = SpectacularHandler {
                    significance_threshold,
                    atp_investment_base,
                    registry: registry.clone(),
                    stats: stats.clone(),
                };
                
                // Assess extraordinariness
                let significance = spectacular.assess_extraordinariness(&data.content);
                
                // Check if content meets extraordinary threshold
                let is_extraordinary = significance.composite_score >= significance_threshold;
                
                if is_extraordinary {
                    // Compute ATP investment
                    let atp_investment = spectacular.compute_atp_investment(&significance);
                    
                    // Register discovery
                    {
                        let mut reg = registry.lock().unwrap();
                        reg.register_discovery(data.content.clone(), significance.clone());
                    }
                    
                    // Update metadata for extraordinary content
                    data = data.with_metadata("spectacular_extraordinary", "true");
                    data = data.with_metadata("spectacular_significance", &significance.composite_score.to_string());
                    data = data.with_metadata("spectacular_atp_investment", &atp_investment.to_string());
                    data = data.with_metadata("spectacular_historical_percentile", &significance.historical_percentile.to_string());
                    
                    // Boost confidence for extraordinary content
                    let confidence_boost = significance.composite_score * 0.2;
                    let new_confidence = (data.confidence + confidence_boost).min(1.0);
                    data = data.with_confidence(new_confidence);
                    
                    info!("Spectacular: Extraordinary content detected! Significance: {:.3}, ATP: {:.0}", 
                          significance.composite_score, atp_investment);
                } else {
                    data = data.with_metadata("spectacular_extraordinary", "false");
                    data = data.with_metadata("spectacular_significance", &significance.composite_score.to_string());
                }
                
                // Update processing statistics
                {
                    let mut stats_guard = stats.lock().unwrap();
                    stats_guard.items_processed += 1;
                    let processing_time = start_time.elapsed().as_millis() as f64;
                    stats_guard.average_processing_time_ms = 
                        (stats_guard.average_processing_time_ms + processing_time) / 2.0;
                    stats_guard.last_processed = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs()
                    );
                }
                
                if tx.send(data).await.is_err() {
                    warn!("Spectacular: Failed to send processed data");
                    break;
                }
            }
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        "SpectacularHandler"
    }
    
    fn can_handle(&self, _data: &StreamData) -> bool {
        true // Can assess any content for extraordinariness
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for SpectacularHandler {
    fn default() -> Self {
        Self::new()
    }
} 