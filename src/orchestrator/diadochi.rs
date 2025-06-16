use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use tokio::sync::mpsc::{channel, Receiver};
use log::{info, debug, warn, error};
use serde::{Deserialize, Serialize};

use super::stream::{StreamProcessor, ProcessorStats};
use super::types::{StreamData, Confidence};

/// Domain specialization for different types of content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainSpecialty {
    Scientific,
    Technical,
    Creative,
    Legal,
    Medical,
    Financial,
    Educational,
    Marketing,
    Research,
    Translation,
    CodeGeneration,
    DataAnalysis,
}

impl DomainSpecialty {
    pub fn from_content(content: &str) -> Vec<DomainSpecialty> {
        let mut domains = Vec::new();
        let content_lower = content.to_lowercase();
        
        // Pattern matching for domain detection
        if content_lower.contains("research") || content_lower.contains("study") || content_lower.contains("hypothesis") {
            domains.push(DomainSpecialty::Research);
        }
        if content_lower.contains("code") || content_lower.contains("programming") || content_lower.contains("function") {
            domains.push(DomainSpecialty::CodeGeneration);
        }
        if content_lower.contains("data") || content_lower.contains("statistics") || content_lower.contains("analysis") {
            domains.push(DomainSpecialty::DataAnalysis);
        }
        if content_lower.contains("creative") || content_lower.contains("story") || content_lower.contains("poem") {
            domains.push(DomainSpecialty::Creative);
        }
        if content_lower.contains("medical") || content_lower.contains("health") || content_lower.contains("diagnosis") {
            domains.push(DomainSpecialty::Medical);
        }
        if content_lower.contains("legal") || content_lower.contains("law") || content_lower.contains("contract") {
            domains.push(DomainSpecialty::Legal);
        }
        if content_lower.contains("financial") || content_lower.contains("money") || content_lower.contains("investment") {
            domains.push(DomainSpecialty::Financial);
        }
        if content_lower.contains("education") || content_lower.contains("learning") || content_lower.contains("teaching") {
            domains.push(DomainSpecialty::Educational);
        }
        if content_lower.contains("marketing") || content_lower.contains("advertising") || content_lower.contains("promotion") {
            domains.push(DomainSpecialty::Marketing);
        }
        if content_lower.contains("translate") || content_lower.contains("language") {
            domains.push(DomainSpecialty::Translation);
        }
        
        // Default to Technical if no specific domain detected
        if domains.is_empty() {
            domains.push(DomainSpecialty::Technical);
        }
        
        domains
    }
    
    pub fn get_recommended_models(&self) -> Vec<&'static str> {
        match self {
            DomainSpecialty::Scientific => vec!["microsoft/DialoGPT-medium", "allenai/scibert_scivocab_uncased"],
            DomainSpecialty::Technical => vec!["microsoft/CodeBERT-base", "huggingface/CodeBERTa-small-v1"],
            DomainSpecialty::Creative => vec!["gpt2", "EleutherAI/gpt-neo-1.3B"],
            DomainSpecialty::Legal => vec!["nlpaueb/legal-bert-base-uncased", "law-ai/InLegalBERT"],
            DomainSpecialty::Medical => vec!["dmis-lab/biobert-base-cased-v1.1", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"],
            DomainSpecialty::Financial => vec!["ProsusAI/finbert", "yiyanghkust/finbert-tone"],
            DomainSpecialty::Educational => vec!["microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"],
            DomainSpecialty::Marketing => vec!["cardiffnlp/twitter-roberta-base-sentiment", "j-hartmann/emotion-english-distilroberta-base"],
            DomainSpecialty::Research => vec!["allenai/scibert_scivocab_uncased", "microsoft/DialoGPT-medium"],
            DomainSpecialty::Translation => vec!["Helsinki-NLP/opus-mt-en-de", "facebook/mbart-large-50-many-to-many-mmt"],
            DomainSpecialty::CodeGeneration => vec!["microsoft/CodeBERT-base", "Salesforce/codegen-350M-multi"],
            DomainSpecialty::DataAnalysis => vec!["microsoft/DialoGPT-medium", "huggingface/CodeBERTa-small-v1"],
        }
    }
}

/// Configuration for external LLM connection
#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub model_name: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub timeout_seconds: u64,
    pub specialty: DomainSpecialty,
    pub confidence_weight: f64,
}

impl LLMConfig {
    pub fn new(model_name: String, specialty: DomainSpecialty) -> Self {
        Self {
            model_name: model_name.clone(),
            api_endpoint: format!("https://api-inference.huggingface.co/models/{}", model_name),
            api_key: None,
            max_tokens: 512,
            temperature: 0.7,
            timeout_seconds: 30,
            specialty,
            confidence_weight: 1.0,
        }
    }
    
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }
    
    pub fn with_endpoint(mut self, endpoint: String) -> Self {
        self.api_endpoint = endpoint;
        self
    }
    
    pub fn with_confidence_weight(mut self, weight: f64) -> Self {
        self.confidence_weight = weight.clamp(0.0, 2.0);
        self
    }
}

/// Response from an external LLM
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub model_name: String,
    pub content: String,
    pub confidence: f64,
    pub processing_time_ms: u64,
    pub tokens_used: usize,
    pub specialty: DomainSpecialty,
    pub metadata: HashMap<String, String>,
}

impl LLMResponse {
    pub fn calculate_weighted_confidence(&self, config: &LLMConfig) -> f64 {
        (self.confidence * config.confidence_weight).clamp(0.0, 1.0)
    }
}

/// Strategy for combining multiple LLM responses
#[derive(Debug, Clone)]
pub enum CombinationStrategy {
    HighestConfidence,
    WeightedAverage,
    Consensus,
    SpecialtyBased,
    Ensemble,
}

impl CombinationStrategy {
    pub fn combine_responses(&self, responses: Vec<LLMResponse>, configs: &HashMap<String, LLMConfig>) -> String {
        if responses.is_empty() {
            return String::new();
        }
        
        match self {
            CombinationStrategy::HighestConfidence => {
                responses.into_iter()
                    .max_by(|a, b| {
                        let a_weighted = if let Some(config) = configs.get(&a.model_name) {
                            a.calculate_weighted_confidence(config)
                        } else {
                            a.confidence
                        };
                        let b_weighted = if let Some(config) = configs.get(&b.model_name) {
                            b.calculate_weighted_confidence(config)
                        } else {
                            b.confidence
                        };
                        a_weighted.partial_cmp(&b_weighted).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.content)
                    .unwrap_or_default()
            }
            
            CombinationStrategy::WeightedAverage => {
                // For text, we'll use the highest confidence response but note the averaging
                let best_response = responses.into_iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                
                format!("{}\n\n[Combined from {} expert sources]", best_response.content, responses.len())
            }
            
            CombinationStrategy::Consensus => {
                // Find common themes/words and build consensus
                let mut word_counts = HashMap::new();
                for response in &responses {
                    for word in response.content.split_whitespace() {
                        *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
                    }
                }
                
                // Use the response that has the most common words
                responses.into_iter()
                    .max_by(|a, b| {
                        let a_score = a.content.split_whitespace()
                            .map(|w| word_counts.get(&w.to_lowercase()).unwrap_or(&0))
                            .sum::<i32>();
                        let b_score = b.content.split_whitespace()
                            .map(|w| word_counts.get(&w.to_lowercase()).unwrap_or(&0))
                            .sum::<i32>();
                        a_score.cmp(&b_score)
                    })
                    .map(|r| r.content)
                    .unwrap_or_default()
            }
            
            CombinationStrategy::SpecialtyBased => {
                // Prefer responses from models that match the detected domain
                responses.into_iter()
                    .max_by(|a, b| {
                        // This is a simplified specialty matching
                        let a_specialty_bonus = match a.specialty {
                            DomainSpecialty::Technical => 0.1,
                            DomainSpecialty::Creative => 0.05,
                            _ => 0.0,
                        };
                        let b_specialty_bonus = match b.specialty {
                            DomainSpecialty::Technical => 0.1,
                            DomainSpecialty::Creative => 0.05,
                            _ => 0.0,
                        };
                        
                        (a.confidence + a_specialty_bonus).partial_cmp(&(b.confidence + b_specialty_bonus))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|r| r.content)
                    .unwrap_or_default()
            }
            
            CombinationStrategy::Ensemble => {
                // Combine all responses with annotations
                let mut combined = String::new();
                for (i, response) in responses.iter().enumerate() {
                    combined.push_str(&format!("=== Expert {} ({:?}) ===\n", i + 1, response.specialty));
                    combined.push_str(&response.content);
                    combined.push_str("\n\n");
                }
                combined
            }
        }
    }
}

/// Query routing logic for intelligent model selection
#[derive(Debug)]
pub struct QueryRouter {
    domain_preferences: HashMap<DomainSpecialty, Vec<String>>,
    model_performance_history: HashMap<String, f64>,
    fallback_models: Vec<String>,
}

impl QueryRouter {
    pub fn new() -> Self {
        let mut domain_preferences = HashMap::new();
        
        // Set up domain preferences
        for specialty in [
            DomainSpecialty::Scientific,
            DomainSpecialty::Technical,
            DomainSpecialty::Creative,
            DomainSpecialty::Legal,
            DomainSpecialty::Medical,
            DomainSpecialty::Financial,
            DomainSpecialty::Educational,
            DomainSpecialty::Marketing,
            DomainSpecialty::Research,
            DomainSpecialty::Translation,
            DomainSpecialty::CodeGeneration,
            DomainSpecialty::DataAnalysis,
        ] {
            domain_preferences.insert(
                specialty.clone(),
                specialty.get_recommended_models().iter().map(|s| s.to_string()).collect()
            );
        }
        
        Self {
            domain_preferences,
            model_performance_history: HashMap::new(),
            fallback_models: vec![
                "gpt2".to_string(),
                "microsoft/DialoGPT-medium".to_string(),
                "EleutherAI/gpt-neo-125M".to_string(),
            ],
        }
    }
    
    pub fn route_query(&self, content: &str, available_models: &[String]) -> Vec<String> {
        let domains = DomainSpecialty::from_content(content);
        let mut selected_models = Vec::new();
        
        // Select models based on detected domains
        for domain in domains {
            if let Some(preferred_models) = self.domain_preferences.get(&domain) {
                for model in preferred_models {
                    if available_models.contains(model) && !selected_models.contains(model) {
                        selected_models.push(model.clone());
                    }
                }
            }
        }
        
        // Add fallback models if no domain-specific models are available
        if selected_models.is_empty() {
            for fallback in &self.fallback_models {
                if available_models.contains(fallback) {
                    selected_models.push(fallback.clone());
                    break;
                }
            }
        }
        
        // Limit to top 3 models to avoid overwhelming
        selected_models.truncate(3);
        selected_models
    }
    
    pub fn update_model_performance(&mut self, model_name: &str, performance_score: f64) {
        self.model_performance_history.insert(model_name.to_string(), performance_score);
    }
}

/// Mock HTTP client for demonstration (replace with actual HTTP client in production)
#[derive(Debug)]
pub struct MockHttpClient;

impl MockHttpClient {
    pub async fn call_huggingface_api(
        &self,
        config: &LLMConfig,
        prompt: &str,
    ) -> Result<LLMResponse, String> {
        // Simulate API call delay
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Mock response generation
        let mock_response = format!(
            "Mock response from {} for prompt: '{}'...\n\
            This is a simulated response demonstrating multi-domain orchestration.",
            config.model_name,
            if prompt.len() > 50 { &prompt[..50] } else { prompt }
        );
        
        Ok(LLMResponse {
            model_name: config.model_name.clone(),
            content: mock_response,
            confidence: 0.75 + (rand::random::<f64>() * 0.2), // 0.75-0.95
            processing_time_ms: 100 + (rand::random::<u64>() % 200), // 100-300ms
            tokens_used: prompt.split_whitespace().count() + 50,
            specialty: config.specialty.clone(),
            metadata: HashMap::new(),
        })
    }
}

/// Diadochi multi-domain LLM orchestration framework
pub struct DiadochiFramework {
    name: String,
    available_models: Arc<Mutex<HashMap<String, LLMConfig>>>,
    query_router: Arc<Mutex<QueryRouter>>,
    combination_strategy: CombinationStrategy,
    http_client: MockHttpClient,
    parallel_execution: bool,
    max_concurrent_requests: usize,
    stats: Arc<Mutex<ProcessorStats>>,
}

impl DiadochiFramework {
    pub fn new() -> Self {
        let mut available_models = HashMap::new();
        
        // Add some default models
        available_models.insert(
            "gpt2".to_string(),
            LLMConfig::new("gpt2".to_string(), DomainSpecialty::Creative)
        );
        available_models.insert(
            "microsoft/DialoGPT-medium".to_string(),
            LLMConfig::new("microsoft/DialoGPT-medium".to_string(), DomainSpecialty::Technical)
        );
        available_models.insert(
            "microsoft/CodeBERT-base".to_string(),
            LLMConfig::new("microsoft/CodeBERT-base".to_string(), DomainSpecialty::CodeGeneration)
        );
        
        Self {
            name: "Diadochi".to_string(),
            available_models: Arc::new(Mutex::new(available_models)),
            query_router: Arc::new(Mutex::new(QueryRouter::new())),
            combination_strategy: CombinationStrategy::HighestConfidence,
            http_client: MockHttpClient,
            parallel_execution: true,
            max_concurrent_requests: 3,
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
    
    pub fn add_model(&self, model_name: String, config: LLMConfig) {
        self.available_models.lock().unwrap().insert(model_name, config);
    }
    
    pub fn with_combination_strategy(mut self, strategy: CombinationStrategy) -> Self {
        self.combination_strategy = strategy;
        self
    }
    
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }
    
    async fn orchestrate_query(&self, content: &str) -> Result<String, String> {
        let available_models = self.available_models.lock().unwrap();
        let model_names: Vec<String> = available_models.keys().cloned().collect();
        
        // Route the query to appropriate models
        let selected_models = {
            let router = self.query_router.lock().unwrap();
            router.route_query(content, &model_names)
        };
        
        if selected_models.is_empty() {
            return Err("No suitable models available for this query".to_string());
        }
        
        debug!("Diadochi routing query to models: {:?}", selected_models);
        
        // Execute queries
        let mut responses = Vec::new();
        
        if self.parallel_execution {
            // Parallel execution
            let mut tasks = Vec::new();
            
            for model_name in selected_models {
                if let Some(config) = available_models.get(&model_name) {
                    let config_clone = config.clone();
                    let content_clone = content.to_string();
                    let client = &self.http_client;
                    
                    // In a real implementation, you'd use proper async HTTP client
                    let task = async move {
                        client.call_huggingface_api(&config_clone, &content_clone).await
                    };
                    
                    tasks.push(task);
                }
            }
            
            // Wait for all tasks to complete
            for task in tasks {
                match task.await {
                    Ok(response) => responses.push(response),
                    Err(e) => warn!("Model call failed: {}", e),
                }
            }
        } else {
            // Sequential execution
            for model_name in selected_models {
                if let Some(config) = available_models.get(&model_name) {
                    match self.http_client.call_huggingface_api(config, content).await {
                        Ok(response) => responses.push(response),
                        Err(e) => warn!("Model call failed: {}", e),
                    }
                }
            }
        }
        
        if responses.is_empty() {
            return Err("All model calls failed".to_string());
        }
        
        // Combine responses using the configured strategy
        let combined_result = self.combination_strategy.combine_responses(responses, &available_models);
        
        Ok(combined_result)
    }
}

#[async_trait]
impl StreamProcessor for DiadochiFramework {
    async fn process(&self, mut input: Receiver<StreamData>) -> Receiver<StreamData> {
        let (tx, rx) = channel(32);
        let name = self.name.clone();
        let available_models = self.available_models.clone();
        let query_router = self.query_router.clone();
        let combination_strategy = self.combination_strategy.clone();
        let http_client = MockHttpClient;
        let parallel_execution = self.parallel_execution;
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut processed_count = 0;
            
            while let Some(mut data) = input.recv().await {
                let start_time = std::time::Instant::now();
                
                debug!("Diadochi processing query: {} chars", data.content.len());
                
                // Skip if already processed by Diadochi
                if data.state.contains_key("diadochi_processed") {
                    if tx.send(data).await.is_err() {
                        break;
                    }
                    continue;
                }
                
                // Check if this requires multi-domain expertise
                let domains = DomainSpecialty::from_content(&data.content);
                let needs_expertise = domains.len() > 1 || 
                    matches!(domains.first(), Some(DomainSpecialty::Scientific) | Some(DomainSpecialty::Medical) | Some(DomainSpecialty::Legal));
                
                if needs_expertise {
                    // Create framework instance for this query
                    let framework = DiadochiFramework {
                        name: name.clone(),
                        available_models: available_models.clone(),
                        query_router: query_router.clone(),
                        combination_strategy: combination_strategy.clone(),
                        http_client,
                        parallel_execution,
                        max_concurrent_requests: 3,
                        stats: stats.clone(),
                    };
                    
                    match framework.orchestrate_query(&data.content).await {
                        Ok(enhanced_content) => {
                            data.content = enhanced_content;
                            data.metadata.insert("diadochi_domains".to_string(), 
                                               format!("{:?}", domains));
                            data.metadata.insert("diadochi_models_used".to_string(), 
                                               domains.len().to_string());
                            data.confidence = (data.confidence + 0.2).min(1.0);
                            
                            info!("Diadochi enhanced content using {} domain experts", domains.len());
                        }
                        Err(e) => {
                            warn!("Diadochi orchestration failed: {}", e);
                            data.metadata.insert("diadochi_error".to_string(), e);
                        }
                    }
                } else {
                    debug!("Diadochi: Single domain detected, skipping orchestration");
                    data.metadata.insert("diadochi_domains".to_string(), "single".to_string());
                }
                
                data.state.insert("diadochi_processed".to_string(), "true".to_string());
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
            
            info!("Diadochi processed {} items", processed_count);
        });
        
        rx
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn can_handle(&self, data: &StreamData) -> bool {
        // Can handle any text data, but prioritizes complex queries
        !data.content.trim().is_empty() && data.content.len() > 10
    }
    
    fn stats(&self) -> ProcessorStats {
        self.stats.lock().unwrap().clone()
    }
}

impl Default for DiadochiFramework {
    fn default() -> Self {
        Self::new()
    }
} 