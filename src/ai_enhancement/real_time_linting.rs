use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tokio::sync::RwLock;
use std::collections::HashMap;
use crate::external_apis::HuggingFaceClient;
use crate::atomic_clock_system::AtomicClockProcessor;
use super::{RealTimeEnhancement, EnhancementType, AIEnhancementError};

/// Real-time enhancement engine for interactive script editing
/// Provides live suggestions as users type Turbulance code
pub struct RealTimeEnhancementEngine {
    huggingface_client: Arc<HuggingFaceClient>,
    atomic_clock_processor: Arc<AtomicClockProcessor>,

    /// Cache for quick suggestions
    suggestion_cache: Arc<RwLock<SuggestionCache>>,

    /// Fast models for real-time processing
    fast_models: Vec<String>,

    /// Configuration for real-time operation
    config: RealTimeConfig,
}

#[derive(Debug, Clone)]
struct RealTimeConfig {
    max_response_time: Duration,
    cache_size: usize,
    min_confidence_threshold: f64,
    debounce_delay: Duration,
    max_suggestions_per_request: usize,
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            max_response_time: Duration::from_millis(500),
            cache_size: 1000,
            min_confidence_threshold: 0.6,
            debounce_delay: Duration::from_millis(200),
            max_suggestions_per_request: 5,
        }
    }
}

/// Cache for storing recent suggestions
struct SuggestionCache {
    entries: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    max_size: usize,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    suggestions: Vec<RealTimeEnhancement>,
    timestamp: Instant,
    hit_count: usize,
}

impl SuggestionCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            max_size,
        }
    }

    fn get(&mut self, key: &str) -> Option<&Vec<RealTimeEnhancement>> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.hit_count += 1;

            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|x| x == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());

            Some(&entry.suggestions)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, suggestions: Vec<RealTimeEnhancement>) {
        // Remove oldest entries if cache is full
        while self.entries.len() >= self.max_size {
            if let Some(oldest_key) = self.access_order.first().cloned() {
                self.entries.remove(&oldest_key);
                self.access_order.remove(0);
            } else {
                break;
            }
        }

        self.entries.insert(key.clone(), CacheEntry {
            suggestions,
            timestamp: Instant::now(),
            hit_count: 1,
        });

        self.access_order.push(key);
    }
}

impl RealTimeEnhancementEngine {
    pub async fn new(
        huggingface_client: Arc<HuggingFaceClient>,
        atomic_clock_processor: Arc<AtomicClockProcessor>,
    ) -> Result<Self, AIEnhancementError> {
        let config = RealTimeConfig::default();

        Ok(Self {
            huggingface_client,
            atomic_clock_processor,
            suggestion_cache: Arc::new(RwLock::new(
                SuggestionCache::new(config.cache_size)
            )),
            fast_models: vec![
                "microsoft/CodeBERT-base".to_string(),
                "salesforce/codet5-small".to_string(),
            ],
            config,
        })
    }

    /// Main entry point for real-time enhancement
    /// Provides suggestions as user types
    pub async fn enhance_real_time(
        &self,
        partial_script: &str,
        cursor_position: usize,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let start_time = Instant::now();

        // Create cache key from script content and cursor position
        let cache_key = self.create_cache_key(partial_script, cursor_position);

        // Check cache first for fast response
        {
            let mut cache = self.suggestion_cache.write().await;
            if let Some(cached_suggestions) = cache.get(&cache_key) {
                return Ok(cached_suggestions.clone());
            }
        }

        // Get temporal coordinate for enhancement coordination
        let temporal_coordinate = self.atomic_clock_processor
            .get_current_temporal_coordinate().await?;

        // Analyze context around cursor position
        let context = self.extract_context(partial_script, cursor_position);

        // Generate suggestions using multiple strategies
        let suggestions = self.generate_real_time_suggestions(
            &context,
            temporal_coordinate,
            start_time,
        ).await?;

        // Cache results for future requests
        {
            let mut cache = self.suggestion_cache.write().await;
            cache.insert(cache_key, suggestions.clone());
        }

        Ok(suggestions)
    }

    /// Extract relevant context around cursor position
    fn extract_context(&self, script: &str, cursor_position: usize) -> EnhancementContext {
        let lines: Vec<&str> = script.lines().collect();
        let mut current_line = 0;
        let mut current_column = 0;
        let mut chars_counted = 0;

        // Find cursor line and column
        for (line_idx, line) in lines.iter().enumerate() {
            if chars_counted + line.len() >= cursor_position {
                current_line = line_idx;
                current_column = cursor_position - chars_counted;
                break;
            }
            chars_counted += line.len() + 1; // +1 for newline
        }

        // Extract context around cursor
        let context_radius = 3;
        let start_line = current_line.saturating_sub(context_radius);
        let end_line = (current_line + context_radius).min(lines.len());

        let context_lines = lines[start_line..end_line].to_vec();
        let current_line_text = lines.get(current_line).unwrap_or(&"").to_string();

        // Analyze what user is typing
        let word_at_cursor = self.extract_word_at_cursor(&current_line_text, current_column);
        let context_type = self.detect_context_type(&context_lines, current_line - start_line);

        EnhancementContext {
            lines: context_lines.into_iter().map(|s| s.to_string()).collect(),
            cursor_line: current_line - start_line,
            cursor_column: current_column,
            word_at_cursor,
            context_type,
            full_script: script.to_string(),
        }
    }

    fn extract_word_at_cursor(&self, line: &str, column: usize) -> String {
        let chars: Vec<char> = line.chars().collect();
        if column >= chars.len() {
            return String::new();
        }

        // Find word boundaries
        let mut start = column;
        let mut end = column;

        // Go backwards to find start of word
        while start > 0 && chars[start - 1].is_alphanumeric() {
            start -= 1;
        }

        // Go forwards to find end of word
        while end < chars.len() && chars[end].is_alphanumeric() {
            end += 1;
        }

        chars[start..end].iter().collect()
    }

    fn detect_context_type(&self, context_lines: &[String], cursor_line: usize) -> ContextType {
        let current_line = context_lines.get(cursor_line).unwrap_or(&String::new());

        if current_line.trim_start().starts_with("funxn") {
            ContextType::FunctionDefinition
        } else if current_line.contains("=") && !current_line.contains("==") {
            ContextType::VariableAssignment
        } else if current_line.trim_start().starts_with("if") ||
                  current_line.trim_start().starts_with("while") {
            ContextType::ControlFlow
        } else if current_line.contains("(") && current_line.contains(")") {
            ContextType::FunctionCall
        } else if current_line.trim().is_empty() {
            ContextType::EmptyLine
        } else {
            ContextType::General
        }
    }

    async fn generate_real_time_suggestions(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
        start_time: Instant,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        // Generate different types of suggestions based on context
        match context.context_type {
            ContextType::FunctionDefinition => {
                suggestions.extend(self.suggest_function_improvements(context, temporal_coordinate).await?);
            },
            ContextType::VariableAssignment => {
                suggestions.extend(self.suggest_variable_improvements(context, temporal_coordinate).await?);
            },
            ContextType::ControlFlow => {
                suggestions.extend(self.suggest_control_flow_improvements(context, temporal_coordinate).await?);
            },
            ContextType::FunctionCall => {
                suggestions.extend(self.suggest_function_call_improvements(context, temporal_coordinate).await?);
            },
            ContextType::EmptyLine => {
                suggestions.extend(self.suggest_next_statements(context, temporal_coordinate).await?);
            },
            ContextType::General => {
                suggestions.extend(self.suggest_general_improvements(context, temporal_coordinate).await?);
            },
        }

        // Check if we're running within time limit
        if start_time.elapsed() > self.config.max_response_time {
            // Return what we have so far
            suggestions.truncate(self.config.max_suggestions_per_request);
            return Ok(suggestions);
        }

        // Generate syntax and autocomplete suggestions
        suggestions.extend(self.generate_autocomplete_suggestions(context, temporal_coordinate).await?);

        // Filter by confidence and limit count
        suggestions.retain(|s| s.confidence >= self.config.min_confidence_threshold);
        suggestions.truncate(self.config.max_suggestions_per_request);

        // Sort by confidence (highest first)
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        Ok(suggestions)
    }

    async fn suggest_function_improvements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        let current_line = &context.lines[context.cursor_line];

        // Suggest parameter types if missing
        if current_line.contains("funxn") && !current_line.contains(":") {
            suggestions.push(RealTimeEnhancement {
                suggestion_type: EnhancementType::SyntaxCorrection,
                position: context.cursor_column,
                suggested_text: ": Type".to_string(),
                confidence: 0.8,
                preview: "Add parameter type annotation".to_string(),
            });
        }

        // Suggest return type if missing
        if current_line.contains("funxn") && current_line.contains(")") && !current_line.contains("->") {
            suggestions.push(RealTimeEnhancement {
                suggestion_type: EnhancementType::SyntaxCorrection,
                position: current_line.find(")").unwrap_or(0) + 1,
                suggested_text: " -> ReturnType".to_string(),
                confidence: 0.75,
                preview: "Add return type annotation".to_string(),
            });
        }

        Ok(suggestions)
    }

    async fn suggest_variable_improvements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        let current_line = &context.lines[context.cursor_line];

        // Suggest variable naming improvements
        if let Some(var_name) = self.extract_variable_name(current_line) {
            if var_name.len() < 3 || !self.is_descriptive_name(&var_name) {
                suggestions.push(RealTimeEnhancement {
                    suggestion_type: EnhancementType::SemanticOptimization,
                    position: 0,
                    suggested_text: format!("descriptive_{}", var_name),
                    confidence: 0.7,
                    preview: "Use more descriptive variable name".to_string(),
                });
            }
        }

        Ok(suggestions)
    }

    async fn suggest_control_flow_improvements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        let current_line = &context.lines[context.cursor_line];

        // Suggest braces if missing
        if (current_line.contains("if") || current_line.contains("while")) &&
           !current_line.contains("{") {
            suggestions.push(RealTimeEnhancement {
                suggestion_type: EnhancementType::SyntaxCorrection,
                position: current_line.len(),
                suggested_text: " {".to_string(),
                confidence: 0.9,
                preview: "Add opening brace".to_string(),
            });
        }

        Ok(suggestions)
    }

    async fn suggest_function_call_improvements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        // Suggest error handling
        suggestions.push(RealTimeEnhancement {
            suggestion_type: EnhancementType::SecurityEnhancement,
            position: 0,
            suggested_text: "try { ".to_string(),
            confidence: 0.6,
            preview: "Add error handling".to_string(),
        });

        Ok(suggestions)
    }

    async fn suggest_next_statements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        // Analyze previous lines to suggest next logical statements
        if context.cursor_line > 0 {
            let prev_line = &context.lines[context.cursor_line - 1];

            if prev_line.contains("funxn") {
                suggestions.push(RealTimeEnhancement {
                    suggestion_type: EnhancementType::CodeGeneration,
                    position: 0,
                    suggested_text: "    // Implementation here".to_string(),
                    confidence: 0.7,
                    preview: "Add function body".to_string(),
                });
            }
        }

        Ok(suggestions)
    }

    async fn suggest_general_improvements(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        // General syntax and style suggestions
        let current_line = &context.lines[context.cursor_line];

        // Suggest semicolon if missing
        if !current_line.trim().is_empty() &&
           !current_line.trim_end().ends_with(';') &&
           !current_line.trim_end().ends_with('{') &&
           !current_line.trim_end().ends_with('}') {
            suggestions.push(RealTimeEnhancement {
                suggestion_type: EnhancementType::SyntaxCorrection,
                position: current_line.len(),
                suggested_text: ";".to_string(),
                confidence: 0.8,
                preview: "Add semicolon".to_string(),
            });
        }

        Ok(suggestions)
    }

    async fn generate_autocomplete_suggestions(
        &self,
        context: &EnhancementContext,
        temporal_coordinate: f64,
    ) -> Result<Vec<RealTimeEnhancement>, AIEnhancementError> {
        let mut suggestions = Vec::new();

        // Use fast model for autocomplete
        if !context.word_at_cursor.is_empty() {
            let prompt = format!(
                "Complete this Turbulance code:\n{}\n\nCurrent word: {}\nSuggest completion:",
                context.lines.join("\n"),
                context.word_at_cursor
            );

            // Use smaller, faster model for real-time response
            let response = tokio::time::timeout(
                Duration::from_millis(300),
                self.huggingface_client.query_model(
                    &self.fast_models[0],
                    &prompt,
                    Some(50), // Small token limit for fast response
                )
            ).await;

            if let Ok(Ok(model_response)) = response {
                if let Some(completion) = model_response.get("generated_text")
                    .and_then(|v| v.as_str()) {
                    suggestions.push(RealTimeEnhancement {
                        suggestion_type: EnhancementType::CodeGeneration,
                        position: context.cursor_column,
                        suggested_text: completion.to_string(),
                        confidence: 0.6,
                        preview: "Autocomplete suggestion".to_string(),
                    });
                }
            }
        }

        Ok(suggestions)
    }

    fn create_cache_key(&self, script: &str, cursor_position: usize) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        script.hash(&mut hasher);
        cursor_position.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn extract_variable_name(&self, line: &str) -> Option<String> {
        if let Some(eq_pos) = line.find('=') {
            let var_part = &line[..eq_pos];
            let var_name = var_part.trim().split_whitespace().last()?;
            Some(var_name.to_string())
        } else {
            None
        }
    }

    fn is_descriptive_name(&self, name: &str) -> bool {
        // Simple heuristic for descriptive names
        name.len() >= 3 && !name.chars().all(|c| c.is_ascii_lowercase())
    }
}

#[derive(Debug, Clone)]
struct EnhancementContext {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_column: usize,
    word_at_cursor: String,
    context_type: ContextType,
    full_script: String,
}

#[derive(Debug, Clone)]
enum ContextType {
    FunctionDefinition,
    VariableAssignment,
    ControlFlow,
    FunctionCall,
    EmptyLine,
    General,
}
