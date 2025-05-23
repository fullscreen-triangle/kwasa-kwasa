//! External API Integrations for Kwasa-Kwasa Framework
//! 
//! This module provides integrations with external APIs and services for
//! enhanced functionality including research databases, language services,
//! and scientific data repositories.

use std::collections::HashMap;
use std::time::Duration;
use reqwest::{Client, Response};
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};

pub mod research;
pub mod language;
pub mod scientific;
pub mod knowledge;

/// Re-exports for easy access
pub mod prelude {
    pub use super::{
        ApiClient, ApiConfig, ApiResponse, ApiError,
        ResearchApi, LanguageApi, ScientificApi, KnowledgeApi
    };
    pub use super::research::*;
    pub use super::language::*;
    pub use super::scientific::*;
    pub use super::knowledge::*;
}

/// Generic API client for external service integration
pub struct ApiClient {
    /// HTTP client for making requests
    client: Client,
    /// Base configuration
    config: ApiConfig,
    /// Authentication tokens by service
    auth_tokens: HashMap<String, String>,
}

/// Configuration for API clients
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Request timeout in seconds
    pub timeout: Duration,
    /// Maximum number of retries
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay: Duration,
    /// User agent string
    pub user_agent: String,
    /// Rate limiting: requests per minute
    pub rate_limit: u32,
}

/// Generic API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Response data
    pub data: T,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Indicates if request was successful
    pub success: bool,
    /// Error message if any
    pub error: Option<String>,
}

/// Metadata about API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Rate limit remaining
    pub rate_limit_remaining: Option<u32>,
    /// API version used
    pub api_version: Option<String>,
    /// Request ID for tracking
    pub request_id: Option<String>,
}

/// API-specific error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiError {
    /// Network/connection error
    Network(String),
    /// Authentication error
    Authentication(String),
    /// Rate limit exceeded
    RateLimit(String),
    /// Invalid request format
    BadRequest(String),
    /// Service temporarily unavailable
    ServiceUnavailable(String),
    /// Unknown error
    Unknown(String),
}

impl ApiClient {
    /// Creates a new API client with default configuration
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            client,
            config: ApiConfig::default(),
            auth_tokens: HashMap::new(),
        }
    }
    
    /// Creates a new API client with custom configuration
    pub fn with_config(config: ApiConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            client,
            config,
            auth_tokens: HashMap::new(),
        }
    }
    
    /// Sets authentication token for a specific service
    pub fn set_auth_token(&mut self, service: &str, token: &str) {
        self.auth_tokens.insert(service.to_string(), token.to_string());
    }
    
    /// Makes a GET request to the specified URL
    pub async fn get<T>(&self, url: &str, service: &str) -> Result<ApiResponse<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start_time = std::time::Instant::now();
        
        let mut request = self.client.get(url);
        
        // Add authentication if available
        if let Some(token) = self.auth_tokens.get(service) {
            request = request.bearer_auth(token);
        }
        
        let response = self.execute_with_retry(request).await?;
        let response_time = start_time.elapsed().as_millis() as u64;
        
        self.parse_response(response, response_time).await
    }
    
    /// Makes a POST request with JSON body
    pub async fn post<T, B>(&self, url: &str, service: &str, body: &B) -> Result<ApiResponse<T>>
    where
        T: for<'de> Deserialize<'de>,
        B: Serialize,
    {
        let start_time = std::time::Instant::now();
        
        let mut request = self.client.post(url).json(body);
        
        // Add authentication if available
        if let Some(token) = self.auth_tokens.get(service) {
            request = request.bearer_auth(token);
        }
        
        let response = self.execute_with_retry(request).await?;
        let response_time = start_time.elapsed().as_millis() as u64;
        
        self.parse_response(response, response_time).await
    }
    
    /// Executes request with retry logic
    async fn execute_with_retry(&self, mut request: reqwest::RequestBuilder) -> Result<Response> {
        let mut last_error = None;
        
        for attempt in 0..=self.config.max_retries {
            match request.try_clone() {
                Some(cloned_request) => {
                    match cloned_request.send().await {
                        Ok(response) => return Ok(response),
                        Err(e) => {
                            last_error = Some(e);
                            if attempt < self.config.max_retries {
                                tokio::time::sleep(self.config.retry_delay).await;
                            }
                        }
                    }
                }
                None => {
                    return Err(Error::external_api("Failed to clone request for retry"));
                }
            }
        }
        
        Err(Error::external_api(format!(
            "Request failed after {} retries: {}",
            self.config.max_retries,
            last_error.map(|e| e.to_string()).unwrap_or_else(|| "Unknown error".to_string())
        )))
    }
    
    /// Parses HTTP response into ApiResponse
    async fn parse_response<T>(&self, response: Response, response_time: u64) -> Result<ApiResponse<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let status = response.status();
        let headers = response.headers().clone();
        
        // Extract metadata from headers
        let rate_limit_remaining = headers.get("x-ratelimit-remaining")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse().ok());
        
        let api_version = headers.get("api-version")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());
        
        let request_id = headers.get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());
        
        let metadata = ResponseMetadata {
            response_time_ms: response_time,
            rate_limit_remaining,
            api_version,
            request_id,
        };
        
        if status.is_success() {
            match response.json::<T>().await {
                Ok(data) => Ok(ApiResponse {
                    data,
                    metadata,
                    success: true,
                    error: None,
                }),
                Err(e) => Err(Error::external_api(format!("Failed to parse response: {}", e))),
            }
        } else {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            Err(Error::external_api(format!("HTTP {}: {}", status, error_text)))
        }
    }
}

/// Research API for academic databases and publications
pub struct ResearchApi {
    /// API client
    client: ApiClient,
}

impl ResearchApi {
    /// Creates a new research API client
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }
    
    /// Searches for academic papers
    pub async fn search_papers(&self, query: &str, limit: Option<u32>) -> Result<Vec<ResearchPaper>> {
        let url = format!(
            "https://api.semanticscholar.org/graph/v1/paper/search?query={}&limit={}",
            urlencoding::encode(query),
            limit.unwrap_or(10)
        );
        
        let response: ApiResponse<SearchResults> = self.client.get(&url, "semantic_scholar").await?;
        Ok(response.data.papers)
    }
    
    /// Gets detailed information about a specific paper
    pub async fn get_paper_details(&self, paper_id: &str) -> Result<ResearchPaper> {
        let url = format!(
            "https://api.semanticscholar.org/graph/v1/paper/{}?fields=title,authors,abstract,citations,references",
            paper_id
        );
        
        let response: ApiResponse<ResearchPaper> = self.client.get(&url, "semantic_scholar").await?;
        Ok(response.data)
    }
}

/// Language processing API for translation and analysis
pub struct LanguageApi {
    /// API client
    client: ApiClient,
}

impl LanguageApi {
    /// Creates a new language API client
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }
    
    /// Translates text between languages
    pub async fn translate(&self, text: &str, from_lang: &str, to_lang: &str) -> Result<TranslationResult> {
        let body = TranslationRequest {
            text: text.to_string(),
            from_language: from_lang.to_string(),
            to_language: to_lang.to_string(),
        };
        
        let url = "https://api.mymemory.translated.net/get";
        let response: ApiResponse<TranslationResult> = self.client.post(url, "mymemory", &body).await?;
        Ok(response.data)
    }
    
    /// Analyzes sentiment of text
    pub async fn analyze_sentiment(&self, text: &str) -> Result<SentimentAnalysis> {
        let body = SentimentRequest {
            text: text.to_string(),
        };
        
        let url = "https://api.textrazor.com/sentiment";
        let response: ApiResponse<SentimentAnalysis> = self.client.post(url, "textrazor", &body).await?;
        Ok(response.data)
    }
}

/// Scientific data API for databases and repositories
pub struct ScientificApi {
    /// API client
    client: ApiClient,
}

impl ScientificApi {
    /// Creates a new scientific API client
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }
    
    /// Searches for genomic sequences
    pub async fn search_genomic_sequences(&self, organism: &str, gene: &str) -> Result<Vec<GenomicSequence>> {
        let url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nuccore&term={}[ORGN]+AND+{}[GENE]&retmode=json",
            urlencoding::encode(organism),
            urlencoding::encode(gene)
        );
        
        let response: ApiResponse<NcbiSearchResult> = self.client.get(&url, "ncbi").await?;
        // Convert search results to genomic sequences
        Ok(self.convert_ncbi_results(response.data).await?)
    }
    
    /// Gets protein information from UniProt
    pub async fn get_protein_info(&self, protein_id: &str) -> Result<ProteinInfo> {
        let url = format!(
            "https://rest.uniprot.org/uniprotkb/{}.json",
            protein_id
        );
        
        let response: ApiResponse<ProteinInfo> = self.client.get(&url, "uniprot").await?;
        Ok(response.data)
    }
    
    async fn convert_ncbi_results(&self, _results: NcbiSearchResult) -> Result<Vec<GenomicSequence>> {
        // Placeholder implementation
        Ok(Vec::new())
    }
}

/// Knowledge base API for factual information
pub struct KnowledgeApi {
    /// API client
    client: ApiClient,
}

impl KnowledgeApi {
    /// Creates a new knowledge API client
    pub fn new(client: ApiClient) -> Self {
        Self { client }
    }
    
    /// Searches for factual information
    pub async fn search_facts(&self, query: &str) -> Result<Vec<FactualResult>> {
        let url = format!(
            "https://api.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&format=json",
            urlencoding::encode(query)
        );
        
        let response: ApiResponse<WikidataSearchResult> = self.client.get(&url, "wikidata").await?;
        Ok(self.convert_wikidata_results(response.data))
    }
    
    fn convert_wikidata_results(&self, _results: WikidataSearchResult) -> Vec<FactualResult> {
        // Placeholder implementation
        Vec::new()
    }
}

// Data structures for API responses

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub papers: Vec<ResearchPaper>,
    pub total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchPaper {
    pub id: String,
    pub title: String,
    pub authors: Vec<Author>,
    pub abstract_text: Option<String>,
    pub publication_year: Option<u32>,
    pub citation_count: Option<u32>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    pub name: String,
    pub author_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub text: String,
    pub from_language: String,
    pub to_language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub translated_text: String,
    pub confidence: f64,
    pub detected_language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentRequest {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    pub sentiment: String, // positive, negative, neutral
    pub confidence: f64,
    pub scores: SentimentScores,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScores {
    pub positive: f64,
    pub negative: f64,
    pub neutral: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomicSequence {
    pub id: String,
    pub organism: String,
    pub gene: String,
    pub sequence: String,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcbiSearchResult {
    pub id_list: Vec<String>,
    pub count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinInfo {
    pub id: String,
    pub name: String,
    pub organism: String,
    pub function: Option<String>,
    pub sequence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualResult {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikidataSearchResult {
    pub search: Vec<WikidataEntity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WikidataEntity {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_millis(1000),
            user_agent: "Kwasa-Kwasa Framework/1.0".to_string(),
            rate_limit: 60,
        }
    }
}

impl Default for ApiClient {
    fn default() -> Self {
        Self::new()
    }
} 