use std::fmt::Debug;
use log::{debug, warn};
use crate::orchestrator::goal::Goal;
use crate::orchestrator::context::Context;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Trait for implementing different kinds of interventions
pub trait Intervention: Debug + Send + Sync {
    /// Get the name of the intervention
    fn name(&self) -> &str;
    
    /// Determine if this intervention should be applied
    fn should_intervene(&self, goal: &Goal, context: &Context) -> bool;
    
    /// Process text with this intervention
    fn process_text(&self, text: &str, goal: &Goal, context: &Context) -> Result<String, String>;
}

/// Intervention to improve readability
#[derive(Debug)]
pub struct ReadabilityIntervention {
    /// Name of the intervention
    name: String,
    
    /// Threshold for when to intervene (lower values mean more interventions)
    threshold: f64,
}

impl ReadabilityIntervention {
    /// Create a new readability intervention
    pub fn new() -> Self {
        Self {
            name: "ReadabilityIntervention".to_string(),
            threshold: 0.4,
        }
    }
    
    /// Create a new readability intervention with a custom threshold
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            name: "ReadabilityIntervention".to_string(),
            threshold,
        }
    }
    
    /// Calculate readability score
    fn calculate_readability(&self, text: &str) -> f64 {
        // Simple readability measurement (higher score = more readable)
        // Based on sentence and word length
        
        let sentences: Vec<&str> = text
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
            
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let sentence_count = sentences.len();
        let word_count = words.len();
        
        if sentence_count == 0 || word_count == 0 {
            return 1.0; // Perfect score for empty text
        }
        
        let avg_sentence_length = word_count as f64 / sentence_count as f64;
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = total_chars as f64 / word_count as f64;
        
        // Convert to a 0-1 scale (1 is most readable)
        let raw_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length);
        (raw_score / 100.0).max(0.0).min(1.0)
    }
    
    /// Simplify text to improve readability
    fn simplify_text(&self, text: &str) -> String {
        // This is a placeholder implementation
        // In a real system, we'd use NLP to actually simplify the text
        
        let sentences: Vec<&str> = text
            .split(&['.', '!', '?'][..])
            .filter(|s| !s.trim().is_empty())
            .collect();
        
        // For demonstration, just break up long sentences
        let mut result = String::new();
        
        for sentence in sentences {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            
            if words.len() > 15 {
                // Break into two parts
                let mid = words.len() / 2;
                let first_half = words[..mid].join(" ");
                let second_half = words[mid..].join(" ");
                
                result.push_str(&first_half);
                result.push_str(". ");
                result.push_str(&second_half);
                result.push_str(". ");
            } else {
                result.push_str(sentence);
                result.push_str(". ");
            }
        }
        
        result.trim().to_string()
    }
}

impl Intervention for ReadabilityIntervention {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn should_intervene(&self, _goal: &Goal, _context: &Context) -> bool {
        // For simplicity, always consider this intervention
        // In a real system, this would be more nuanced
        true
    }
    
    fn process_text(&self, text: &str, _goal: &Goal, _context: &Context) -> Result<String, String> {
        let readability = self.calculate_readability(text);
        
        if readability < self.threshold {
            debug!("Readability score {} below threshold {}, intervening", readability, self.threshold);
            Ok(self.simplify_text(text))
        } else {
            // No need to intervene
            Ok(text.to_string())
        }
    }
}

/// Intervention to improve text coherence
#[derive(Debug)]
pub struct CoherenceIntervention {
    /// Name of the intervention
    name: String,
}

impl CoherenceIntervention {
    /// Create a new coherence intervention
    pub fn new() -> Self {
        Self {
            name: "CoherenceIntervention".to_string(),
        }
    }
    
    /// Check if the text is coherent with the context
    fn is_coherent_with_context(&self, text: &str, context: &Context) -> bool {
        // Simple coherence check: do the keywords in the text match recent context keywords?
        
        // Extract keywords from text
        let text_keywords = extract_keywords(text);
        
        // Get recent keywords from context
        let context_keywords = context.get_recent_keywords();
        
        // Calculate overlap
        let mut matches = 0;
        for keyword in &text_keywords {
            if context_keywords.contains(keyword) {
                matches += 1;
            }
        }
        
        if text_keywords.is_empty() {
            return true; // Consider empty text coherent
        }
        
        // Require at least 20% of text keywords to match context
        matches as f64 / text_keywords.len() as f64 >= 0.2
    }
    
    /// Suggest transitions to improve coherence
    fn suggest_transitions(&self, text: &str, context: &Context) -> String {
        // Extract the main topic of the text
        let text_keywords = extract_keywords(text);
        
        if text_keywords.is_empty() {
            return text.to_string();
        }
        
        let primary_topic = &text_keywords[0];
        
        // Check if this is a transition from a previous topic
        let recent_keywords = context.get_recent_keywords();
        
        if recent_keywords.is_empty() {
            return text.to_string();
        }
        
        let previous_topic = &recent_keywords[0];
        
        // If transitioning to a new topic, suggest a transition phrase
        if primary_topic != previous_topic {
            let transition = format!("Moving from {} to {}, ", previous_topic, primary_topic);
            return format!("{}{}", transition, text);
        }
        
        text.to_string()
    }
}

impl Intervention for CoherenceIntervention {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn should_intervene(&self, _goal: &Goal, context: &Context) -> bool {
        // Intervene if the context has enough keywords to work with
        context.get_keywords().len() >= 3
    }
    
    fn process_text(&self, text: &str, _goal: &Goal, context: &Context) -> Result<String, String> {
        if !self.is_coherent_with_context(text, context) {
            debug!("Text not coherent with context, suggesting transitions");
            Ok(self.suggest_transitions(text, context))
        } else {
            // Already coherent
            Ok(text.to_string())
        }
    }
}

/// Intervention to suggest research when needed
#[derive(Debug)]
pub struct ResearchIntervention {
    /// Name of the intervention
    name: String,
}

impl ResearchIntervention {
    /// Create a new research intervention
    pub fn new() -> Self {
        Self {
            name: "ResearchIntervention".to_string(),
        }
    }
    
    /// Check if text appears to need more research
    fn needs_research(&self, text: &str) -> bool {
        // Simple heuristic: text contains phrases that suggest uncertainty
        let uncertainty_markers = [
            "not sure",
            "might be",
            "possibly",
            "perhaps",
            "I think",
            "maybe",
            "uncertain",
            "unclear",
        ];
        
        for marker in &uncertainty_markers {
            if text.to_lowercase().contains(marker) {
                return true;
            }
        }
        
        false
    }
    
    /// Identify topics that need more research
    fn identify_research_topics(&self, text: &str) -> Vec<String> {
        // Extract keywords that might need research
        // In a real system, this would use entity extraction and other NLP techniques
        
        let keywords = extract_keywords(text);
        
        // For now, just return up to 3 keywords
        keywords.into_iter().take(3).collect()
    }
}

impl Intervention for ResearchIntervention {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn should_intervene(&self, _goal: &Goal, _context: &Context) -> bool {
        // For simplicity, always consider this intervention
        true
    }
    
    fn process_text(&self, text: &str, _goal: &Goal, context: &Context) -> Result<String, String> {
        if self.needs_research(text) {
            debug!("Text appears to need more research");
            
            let topics = self.identify_research_topics(text);
            
            // Add topics to research context
            for topic in &topics {
                warn!("Consider researching: {}", topic);
                // In a real implementation, this would actually add to the context
                // but that would require a mutable context, which we don't have here
            }
            
            // This intervention doesn't modify text directly
            Ok(text.to_string())
        } else {
            // No research needed
            Ok(text.to_string())
        }
    }
}

/// Extract keywords from text
fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "from", "of", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "must", "that", "this", "these", "those",
    ];
    
    text.split_whitespace()
        .map(|word| word.to_lowercase())
        .filter(|word| {
            // Remove punctuation
            let clean_word = word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();
            
            // Filter out stop words and very short words
            clean_word.len() > 3 && !stop_words.contains(&clean_word.as_str())
        })
        .map(|word| {
            // Remove punctuation
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .collect()
}

/// Intervention system that monitors user behavior and provides assistance
#[derive(Debug)]
pub struct InterventionSystem {
    /// Currently active interventions
    active_interventions: Vec<ActiveIntervention>,
    
    /// Intervention history
    history: Vec<InterventionRecord>,
    
    /// Thresholds for triggering interventions
    thresholds: InterventionThresholds,
    
    /// Available intervention strategies
    strategies: HashMap<InterventionType, InterventionStrategy>,
    
    /// User preferences for interventions
    user_preferences: UserInterventionPreferences,
}

/// An active intervention that is currently being applied
#[derive(Debug, Clone)]
pub struct ActiveIntervention {
    /// Unique identifier
    id: String,
    
    /// Type of intervention
    intervention_type: InterventionType,
    
    /// The strategy being used
    strategy: InterventionStrategy,
    
    /// When this intervention was triggered
    triggered_at: u64,
    
    /// Context that triggered this intervention
    trigger_context: String,
    
    /// Current stage of the intervention
    stage: InterventionStage,
    
    /// Effectiveness so far (0.0 - 1.0)
    effectiveness: Option<f64>,
    
    /// Maximum duration for this intervention (seconds)
    max_duration: u64,
}

/// Record of a past intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionRecord {
    /// Type of intervention
    intervention_type: InterventionType,
    
    /// When it was triggered
    triggered_at: u64,
    
    /// When it was completed/ended
    ended_at: u64,
    
    /// Duration in seconds
    duration: u64,
    
    /// Outcome of the intervention
    outcome: InterventionOutcome,
    
    /// Effectiveness score (0.0 - 1.0)
    effectiveness: f64,
    
    /// User feedback (if any)
    user_feedback: Option<String>,
    
    /// Context that triggered it
    trigger_context: String,
}

/// Types of interventions available
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InterventionType {
    /// When user seems stuck or blocked
    WritersBlock,
    
    /// When writing quality is degrading
    QualityDegradation,
    
    /// When user is off-topic from their goal
    OffTopic,
    
    /// When productivity is low
    ProductivityDrop,
    
    /// When user shows signs of fatigue
    Fatigue,
    
    /// When goal progress is stalled
    GoalStagnation,
    
    /// When user seems frustrated
    Frustration,
    
    /// When writing patterns suggest confusion
    Confusion,
    
    /// When user needs motivational support
    Motivation,
    
    /// When technical issues are detected
    TechnicalIssues,
    
    /// Custom intervention type
    Custom(String),
}

/// Strategies for interventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionStrategy {
    /// Name of the strategy
    name: String,
    
    /// Description of what this strategy does
    description: String,
    
    /// Steps to execute for this strategy
    steps: Vec<InterventionStep>,
    
    /// Expected duration (seconds)
    expected_duration: u64,
    
    /// Success criteria
    success_criteria: Vec<String>,
    
    /// Parameters for customization
    parameters: HashMap<String, String>,
}

/// Individual step in an intervention strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterventionStep {
    /// Name of this step
    name: String,
    
    /// Action to take
    action: InterventionAction,
    
    /// Message to display to user (optional)
    message: Option<String>,
    
    /// Duration for this step (seconds)
    duration: u64,
    
    /// Whether this step requires user interaction
    requires_interaction: bool,
    
    /// Conditions for proceeding to next step
    proceed_conditions: Vec<String>,
}

/// Actions that can be taken during intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionAction {
    /// Display a message to the user
    ShowMessage(String),
    
    /// Suggest a writing prompt
    SuggestPrompt(String),
    
    /// Recommend taking a break
    SuggestBreak(u64), // Duration in seconds
    
    /// Provide research suggestions
    SuggestResearch(Vec<String>),
    
    /// Offer goal refinement
    RefineGoal,
    
    /// Suggest content reorganization
    ReorganizeContent,
    
    /// Provide writing techniques
    SuggestTechnique(String),
    
    /// Offer motivational content
    Motivate(String),
    
    /// Suggest environmental changes
    EnvironmentAdjustment(String),
    
    /// Redirect to goal
    RedirectToGoal,
    
    /// Custom action
    Custom(String, HashMap<String, String>),
}

/// Current stage of intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionStage {
    /// Initial detection phase
    Detection,
    
    /// Preparing intervention
    Preparation,
    
    /// Actively intervening
    Active,
    
    /// Monitoring effectiveness
    Monitoring,
    
    /// Winding down intervention
    Completion,
    
    /// Intervention completed
    Completed,
}

/// Outcome of an intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionOutcome {
    /// Intervention was successful
    Success,
    
    /// Intervention partially successful
    PartialSuccess,
    
    /// Intervention had no effect
    NoEffect,
    
    /// Intervention was counterproductive
    Negative,
    
    /// User cancelled intervention
    UserCancelled,
    
    /// Intervention timed out
    Timeout,
    
    /// Technical failure
    TechnicalFailure,
}

/// Thresholds for triggering different interventions
#[derive(Debug, Clone)]
pub struct InterventionThresholds {
    /// Minimum time without progress before intervention (seconds)
    pub stagnation_time: u64,
    
    /// Quality drop threshold (percentage)
    pub quality_drop_threshold: f64,
    
    /// Productivity drop threshold (words per minute)
    pub productivity_threshold: f64,
    
    /// Maximum time off-topic before intervention (seconds)
    pub off_topic_threshold: u64,
    
    /// Fatigue indicators threshold
    pub fatigue_threshold: f64,
    
    /// Frustration level threshold
    pub frustration_threshold: f64,
}

/// User preferences for interventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInterventionPreferences {
    /// Whether interventions are enabled
    pub enabled: bool,
    
    /// Preferred intervention style
    pub style: InterventionStyle,
    
    /// Maximum frequency (interventions per hour)
    pub max_frequency: u32,
    
    /// Disabled intervention types
    pub disabled_types: Vec<InterventionType>,
    
    /// Minimum severity before intervention
    pub min_severity: f64,
    
    /// Preferred timing for interventions
    pub timing_preference: TimingPreference,
}

/// Style of intervention delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterventionStyle {
    /// Gentle, non-intrusive suggestions
    Gentle,
    
    /// Direct, clear recommendations
    Direct,
    
    /// Motivational and encouraging
    Motivational,
    
    /// Technical and analytical
    Analytical,
    
    /// Minimal, only critical interventions
    Minimal,
    
    /// Custom style
    Custom(String),
}

/// When to deliver interventions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimingPreference {
    /// Immediate when detected
    Immediate,
    
    /// At natural pause points
    NaturalPauses,
    
    /// After current sentence/paragraph
    AfterCompletion,
    
    /// Only when user is idle
    WhenIdle,
    
    /// Ask before intervening
    AskFirst,
}

impl InterventionSystem {
    /// Create a new intervention system
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        
        // Initialize default strategies
        strategies.insert(
            InterventionType::WritersBlock,
            Self::create_writers_block_strategy(),
        );
        
        strategies.insert(
            InterventionType::QualityDegradation,
            Self::create_quality_strategy(),
        );
        
        strategies.insert(
            InterventionType::OffTopic,
            Self::create_off_topic_strategy(),
        );
        
        strategies.insert(
            InterventionType::ProductivityDrop,
            Self::create_productivity_strategy(),
        );
        
        strategies.insert(
            InterventionType::Fatigue,
            Self::create_fatigue_strategy(),
        );
        
        strategies.insert(
            InterventionType::GoalStagnation,
            Self::create_goal_stagnation_strategy(),
        );
        
        strategies.insert(
            InterventionType::Motivation,
            Self::create_motivation_strategy(),
        );
        
        Self {
            active_interventions: Vec::new(),
            history: Vec::new(),
            thresholds: InterventionThresholds::default(),
            strategies,
            user_preferences: UserInterventionPreferences::default(),
        }
    }
    
    /// Analyze context and determine if intervention is needed
    pub fn analyze_context(&mut self, context: &Context, goal: Option<&Goal>) -> Vec<InterventionType> {
        let mut needed_interventions = Vec::new();
        
        if !self.user_preferences.enabled {
            return needed_interventions;
        }
        
        // Check for stagnation
        if self.check_stagnation(context) {
            needed_interventions.push(InterventionType::GoalStagnation);
        }
        
        // Check for quality issues
        if self.check_quality_degradation(context) {
            needed_interventions.push(InterventionType::QualityDegradation);
        }
        
        // Check if off-topic
        if let Some(goal) = goal {
            if self.check_off_topic(context, goal) {
                needed_interventions.push(InterventionType::OffTopic);
            }
        }
        
        // Check productivity
        if self.check_productivity_drop(context) {
            needed_interventions.push(InterventionType::ProductivityDrop);
        }
        
        // Check for fatigue
        if self.check_fatigue(context) {
            needed_interventions.push(InterventionType::Fatigue);
        }
        
        // Check for writer's block
        if self.check_writers_block(context) {
            needed_interventions.push(InterventionType::WritersBlock);
        }
        
        // Filter based on user preferences
        needed_interventions.retain(|intervention_type| {
            !self.user_preferences.disabled_types.contains(intervention_type)
        });
        
        needed_interventions
    }
    
    /// Trigger an intervention
    pub fn trigger_intervention(&mut self, intervention_type: InterventionType, context: &str) -> Result<String, String> {
        // Check if we can trigger this intervention
        if !self.can_trigger_intervention(&intervention_type) {
            return Err("Intervention cannot be triggered at this time".to_string());
        }
        
        // Get the strategy for this intervention type
        let strategy = self.strategies.get(&intervention_type)
            .ok_or_else(|| format!("No strategy found for intervention type: {:?}", intervention_type))?
            .clone();
        
        // Create active intervention
        let intervention = ActiveIntervention {
            id: generate_intervention_id(),
            intervention_type: intervention_type.clone(),
            strategy,
            triggered_at: current_timestamp(),
            trigger_context: context.to_string(),
            stage: InterventionStage::Detection,
            effectiveness: None,
            max_duration: 300, // 5 minutes default
        };
        
        let intervention_id = intervention.id.clone();
        self.active_interventions.push(intervention);
        
        Ok(intervention_id)
    }
    
    /// Process an intervention step
    pub fn process_intervention(&mut self, intervention_id: &str) -> Option<InterventionAction> {
        let intervention = self.active_interventions.iter_mut()
            .find(|i| i.id == intervention_id)?;
        
        // Get current step
        let current_step_index = match intervention.stage {
            InterventionStage::Detection => 0,
            InterventionStage::Preparation => 1,
            InterventionStage::Active => 2,
            InterventionStage::Monitoring => 3,
            InterventionStage::Completion => 4,
            InterventionStage::Completed => return None,
        };
        
        if current_step_index >= intervention.strategy.steps.len() {
            intervention.stage = InterventionStage::Completed;
            return None;
        }
        
        let step = &intervention.strategy.steps[current_step_index];
        
        // Advance to next stage
        intervention.stage = match intervention.stage {
            InterventionStage::Detection => InterventionStage::Preparation,
            InterventionStage::Preparation => InterventionStage::Active,
            InterventionStage::Active => InterventionStage::Monitoring,
            InterventionStage::Monitoring => InterventionStage::Completion,
            InterventionStage::Completion => InterventionStage::Completed,
            InterventionStage::Completed => InterventionStage::Completed,
        };
        
        Some(step.action.clone())
    }
    
    /// Complete an intervention
    pub fn complete_intervention(&mut self, intervention_id: &str, outcome: InterventionOutcome, effectiveness: f64) {
        if let Some(pos) = self.active_interventions.iter().position(|i| i.id == intervention_id) {
            let intervention = self.active_interventions.remove(pos);
            
            // Create record
            let record = InterventionRecord {
                intervention_type: intervention.intervention_type,
                triggered_at: intervention.triggered_at,
                ended_at: current_timestamp(),
                duration: current_timestamp() - intervention.triggered_at,
                outcome,
                effectiveness,
                user_feedback: None,
                trigger_context: intervention.trigger_context,
            };
            
            self.history.push(record);
        }
    }
    
    /// Get intervention suggestions for current context
    pub fn get_suggestions(&self, context: &Context, goal: Option<&Goal>) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Analyze recent patterns
        let patterns = context.get_patterns("productivity");
        if patterns.len() > 5 {
            let recent_avg = patterns.iter().rev().take(5)
                .map(|p| p.value())
                .sum::<f64>() / 5.0;
            
            if recent_avg < 20.0 {
                suggestions.push("Consider taking a short break to refresh your mind".to_string());
            }
        }
        
        // Goal-based suggestions
        if let Some(goal) = goal {
            if goal.completion() < 0.1 {
                suggestions.push("Break down your goal into smaller, manageable sub-tasks".to_string());
            }
            
            if goal.is_overdue() {
                suggestions.push("Your goal is overdue. Consider adjusting the timeline or scope".to_string());
            }
        }
        
        // Context-based suggestions
        if context.keywords().len() < 3 {
            suggestions.push("Try adding more specific keywords to better define your focus".to_string());
        }
        
        suggestions
    }
    
    /// Get intervention history
    pub fn get_history(&self) -> &[InterventionRecord] {
        &self.history
    }
    
    /// Update user preferences
    pub fn update_preferences(&mut self, preferences: UserInterventionPreferences) {
        self.user_preferences = preferences;
    }
    
    /// Check if intervention can be triggered
    fn can_trigger_intervention(&self, intervention_type: &InterventionType) -> bool {
        // Check frequency limits
        let recent_interventions = self.history.iter()
            .filter(|r| r.intervention_type == *intervention_type)
            .filter(|r| current_timestamp() - r.triggered_at < 3600) // Last hour
            .count();
        
        if recent_interventions >= self.user_preferences.max_frequency as usize {
            return false;
        }
        
        // Check if already active
        let active_same_type = self.active_interventions.iter()
            .any(|i| i.intervention_type == *intervention_type);
        
        !active_same_type
    }
    
    // Detection methods
    fn check_stagnation(&self, context: &Context) -> bool {
        let now = current_timestamp();
        now - context.last_updated() > self.thresholds.stagnation_time
    }
    
    fn check_quality_degradation(&self, context: &Context) -> bool {
        let patterns = context.get_patterns("quality_score");
        if patterns.len() < 5 {
            return false;
        }
        
        let recent_avg = patterns.iter().rev().take(3)
            .map(|p| p.value())
            .sum::<f64>() / 3.0;
        
        let baseline_avg = patterns.iter().rev().skip(3).take(5)
            .map(|p| p.value())
            .sum::<f64>() / 5.0;
        
        (baseline_avg - recent_avg) / baseline_avg > self.thresholds.quality_drop_threshold
    }
    
    fn check_off_topic(&self, context: &Context, goal: &Goal) -> bool {
        // Check if recent focus areas align with goal keywords
        let focus_alignment = context.focus_areas().iter()
            .any(|area| goal.keywords().iter()
                .any(|keyword| area.to_lowercase().contains(&keyword.to_lowercase())));
        
        !focus_alignment
    }
    
    fn check_productivity_drop(&self, context: &Context) -> bool {
        let patterns = context.get_patterns("words_per_minute");
        if patterns.len() < 3 {
            return false;
        }
        
        let recent_avg = patterns.iter().rev().take(3)
            .map(|p| p.value())
            .sum::<f64>() / 3.0;
        
        recent_avg < self.thresholds.productivity_threshold
    }
    
    fn check_fatigue(&self, context: &Context) -> bool {
        // Check multiple fatigue indicators
        let patterns = context.get_patterns("sentence_length");
        if patterns.len() < 5 {
            return false;
        }
        
        let recent_variance = Self::calculate_variance(
            &patterns.iter().rev().take(5).map(|p| p.value()).collect::<Vec<_>>()
        );
        
        recent_variance > self.thresholds.fatigue_threshold
    }
    
    fn check_writers_block(&self, context: &Context) -> bool {
        // Multiple indicators of writer's block
        let patterns = context.get_patterns("writing_flow");
        if patterns.is_empty() {
            return false;
        }
        
        let latest_flow = patterns.last().unwrap().value();
        latest_flow < 0.3 // Low flow score indicates block
    }
    
    fn calculate_variance(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        variance
    }
    
    // Strategy creation methods
    fn create_writers_block_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Writer's Block Relief".to_string(),
            description: "Help overcome writer's block with prompts and techniques".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Detect Block".to_string(),
                    action: InterventionAction::ShowMessage(
                        "It seems like you might be experiencing writer's block. Let me help!".to_string()
                    ),
                    message: None,
                    duration: 5,
                    requires_interaction: false,
                    proceed_conditions: vec!["user_acknowledged".to_string()],
                },
                InterventionStep {
                    name: "Suggest Prompt".to_string(),
                    action: InterventionAction::SuggestPrompt(
                        "Try writing about why this topic matters to you personally".to_string()
                    ),
                    message: Some("Here's a writing prompt to get you started:".to_string()),
                    duration: 60,
                    requires_interaction: true,
                    proceed_conditions: vec!["writing_resumed".to_string()],
                },
            ],
            expected_duration: 120,
            success_criteria: vec!["writing_resumed".to_string(), "flow_improved".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_quality_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Quality Improvement".to_string(),
            description: "Help improve writing quality through techniques and suggestions".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Quality Check".to_string(),
                    action: InterventionAction::ShowMessage(
                        "I've noticed the writing quality could be improved. Let me suggest some techniques.".to_string()
                    ),
                    message: None,
                    duration: 5,
                    requires_interaction: false,
                    proceed_conditions: vec!["user_ready".to_string()],
                },
                InterventionStep {
                    name: "Suggest Technique".to_string(),
                    action: InterventionAction::SuggestTechnique(
                        "Try reading your last paragraph aloud to catch unclear sentences".to_string()
                    ),
                    message: None,
                    duration: 180,
                    requires_interaction: true,
                    proceed_conditions: vec!["quality_improved".to_string()],
                },
            ],
            expected_duration: 240,
            success_criteria: vec!["quality_score_increased".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_off_topic_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Goal Refocus".to_string(),
            description: "Help user get back on track with their writing goal".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Redirect".to_string(),
                    action: InterventionAction::RedirectToGoal,
                    message: Some("Let's refocus on your main writing goal".to_string()),
                    duration: 30,
                    requires_interaction: false,
                    proceed_conditions: vec!["goal_acknowledged".to_string()],
                },
            ],
            expected_duration: 60,
            success_criteria: vec!["back_on_topic".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_productivity_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Productivity Boost".to_string(),
            description: "Help increase writing productivity through environment and technique adjustments".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Environment Check".to_string(),
                    action: InterventionAction::EnvironmentAdjustment(
                        "Consider adjusting your environment - lighting, noise, posture".to_string()
                    ),
                    message: None,
                    duration: 60,
                    requires_interaction: false,
                    proceed_conditions: vec!["environment_adjusted".to_string()],
                },
            ],
            expected_duration: 120,
            success_criteria: vec!["productivity_increased".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_fatigue_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Fatigue Management".to_string(),
            description: "Help manage fatigue and restore energy for writing".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Break Suggestion".to_string(),
                    action: InterventionAction::SuggestBreak(300), // 5 minutes
                    message: Some("You seem tired. A short break might help refresh your mind.".to_string()),
                    duration: 300,
                    requires_interaction: true,
                    proceed_conditions: vec!["break_taken".to_string()],
                },
            ],
            expected_duration: 300,
            success_criteria: vec!["energy_restored".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_goal_stagnation_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Goal Progress".to_string(),
            description: "Help overcome goal stagnation through strategy adjustment".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Goal Review".to_string(),
                    action: InterventionAction::RefineGoal,
                    message: Some("Let's review and possibly adjust your writing goal".to_string()),
                    duration: 120,
                    requires_interaction: true,
                    proceed_conditions: vec!["goal_refined".to_string()],
                },
            ],
            expected_duration: 180,
            success_criteria: vec!["progress_resumed".to_string()],
            parameters: HashMap::new(),
        }
    }
    
    fn create_motivation_strategy() -> InterventionStrategy {
        InterventionStrategy {
            name: "Motivation Boost".to_string(),
            description: "Provide motivational support and encouragement".to_string(),
            steps: vec![
                InterventionStep {
                    name: "Motivate".to_string(),
                    action: InterventionAction::Motivate(
                        "Remember why you started this writing journey. Every word you write brings you closer to your goal!".to_string()
                    ),
                    message: None,
                    duration: 30,
                    requires_interaction: false,
                    proceed_conditions: vec!["motivation_restored".to_string()],
                },
            ],
            expected_duration: 60,
            success_criteria: vec!["motivation_improved".to_string()],
            parameters: HashMap::new(),
        }
    }
}

impl Default for InterventionThresholds {
    fn default() -> Self {
        Self {
            stagnation_time: 300,        // 5 minutes
            quality_drop_threshold: 0.2,  // 20% drop
            productivity_threshold: 10.0,  // 10 words per minute
            off_topic_threshold: 180,     // 3 minutes
            fatigue_threshold: 2.0,       // Variance threshold
            frustration_threshold: 0.7,   // 70% frustration
        }
    }
}

impl Default for UserInterventionPreferences {
    fn default() -> Self {
        Self {
            enabled: true,
            style: InterventionStyle::Gentle,
            max_frequency: 3, // 3 per hour
            disabled_types: Vec::new(),
            min_severity: 0.5,
            timing_preference: TimingPreference::NaturalPauses,
        }
    }
}

/// Generate a unique intervention ID
fn generate_intervention_id() -> String {
    format!("intervention_{}", current_timestamp())
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_intervention_system_creation() {
        let system = InterventionSystem::new();
        
        assert!(system.active_interventions.is_empty());
        assert!(system.history.is_empty());
        assert!(system.user_preferences.enabled);
        assert!(system.strategies.len() > 0);
    }
    
    #[test]
    fn test_trigger_intervention() {
        let mut system = InterventionSystem::new();
        
        let result = system.trigger_intervention(
            InterventionType::WritersBlock,
            "User seems stuck"
        );
        
        assert!(result.is_ok());
        assert_eq!(system.active_interventions.len(), 1);
    }
    
    #[test]
    fn test_complete_intervention() {
        let mut system = InterventionSystem::new();
        
        let intervention_id = system.trigger_intervention(
            InterventionType::WritersBlock,
            "Test context"
        ).unwrap();
        
        system.complete_intervention(
            &intervention_id,
            InterventionOutcome::Success,
            0.8
        );
        
        assert!(system.active_interventions.is_empty());
        assert_eq!(system.history.len(), 1);
        assert_eq!(system.history[0].effectiveness, 0.8);
    }
} 