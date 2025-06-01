use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::orchestrator::goal::Goal;

/// Maximum number of recent keywords to track
const MAX_RECENT_KEYWORDS: usize = 50;

/// Maximum number of recent research terms to track
const MAX_RECENT_RESEARCH: usize = 20;

/// The current context of the writing process
#[derive(Debug, Clone)]
pub struct Context {
    /// Current writing session ID
    session_id: String,
    
    /// Active keywords being tracked
    keywords: HashSet<String>,
    
    /// Research terms that have been explored
    research_terms: HashSet<String>,
    
    /// Recent writing patterns
    writing_patterns: VecDeque<WritingPattern>,
    
    /// Current focus areas
    focus_areas: Vec<String>,
    
    /// User preferences
    preferences: HashMap<String, String>,
    
    /// Goal state tracking
    goal_progress: GoalProgress,
    
    /// Context metadata
    metadata: HashMap<String, String>,
    
    /// Timestamp of last update
    last_updated: u64,
}

/// Represents a pattern in the user's writing
#[derive(Debug, Clone)]
pub struct WritingPattern {
    /// Type of pattern (e.g., "sentence_length", "vocabulary_level")
    pattern_type: String,
    
    /// The observed value
    value: f64,
    
    /// Confidence in this observation
    confidence: f64,
    
    /// When this pattern was observed
    timestamp: u64,
}

/// Tracks progress toward current goals
#[derive(Debug, Clone)]
pub struct GoalProgress {
    /// Current completion percentage (0.0-1.0)
    completion: f64,
    
    /// Sub-goals that have been completed
    completed_subgoals: Vec<String>,
    
    /// Current active sub-goal
    active_subgoal: Option<String>,
    
    /// Obstacles encountered
    obstacles: Vec<String>,
    
    /// Strategies being employed
    strategies: Vec<String>,
}

impl Context {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            session_id: generate_session_id(),
            keywords: HashSet::new(),
            research_terms: HashSet::new(),
            writing_patterns: VecDeque::with_capacity(100), // Keep last 100 patterns
            focus_areas: Vec::new(),
            preferences: HashMap::new(),
            goal_progress: GoalProgress::new(),
            metadata: HashMap::new(),
            last_updated: current_timestamp(),
        }
    }
    
    /// Add a keyword to track
    pub fn add_keyword(&mut self, keyword: String) {
        self.keywords.insert(keyword);
        self.update_timestamp();
    }
    
    /// Remove a keyword
    pub fn remove_keyword(&mut self, keyword: &str) -> bool {
        let removed = self.keywords.remove(keyword);
        if removed {
            self.update_timestamp();
        }
        removed
    }
    
    /// Get all tracked keywords
    pub fn keywords(&self) -> &HashSet<String> {
        &self.keywords
    }
    
    /// Add a research term
    pub fn add_research_term(&mut self, term: &str) {
        self.research_terms.insert(term.to_string());
        self.update_timestamp();
    }
    
    /// Get all research terms
    pub fn research_terms(&self) -> &HashSet<String> {
        &self.research_terms
    }
    
    /// Add a writing pattern observation
    pub fn add_writing_pattern(&mut self, pattern: WritingPattern) {
        // Keep only the most recent patterns
        if self.writing_patterns.len() >= 100 {
            self.writing_patterns.pop_front();
        }
        
        self.writing_patterns.push_back(pattern);
        self.update_timestamp();
    }
    
    /// Get recent patterns of a specific type
    pub fn get_patterns(&self, pattern_type: &str) -> Vec<&WritingPattern> {
        self.writing_patterns
            .iter()
            .filter(|p| p.pattern_type == pattern_type)
            .collect()
    }
    
    /// Add a focus area
    pub fn add_focus_area(&mut self, area: String) {
        if !self.focus_areas.contains(&area) {
            self.focus_areas.push(area);
            self.update_timestamp();
        }
    }
    
    /// Remove a focus area
    pub fn remove_focus_area(&mut self, area: &str) {
        self.focus_areas.retain(|a| a != area);
        self.update_timestamp();
    }
    
    /// Get current focus areas
    pub fn focus_areas(&self) -> &[String] {
        &self.focus_areas
    }
    
    /// Set a user preference
    pub fn set_preference(&mut self, key: String, value: String) {
        self.preferences.insert(key, value);
        self.update_timestamp();
    }
    
    /// Get a user preference
    pub fn get_preference(&self, key: &str) -> Option<&String> {
        self.preferences.get(key)
    }
    
    /// Update goal state
    pub fn update_goal_state(&mut self, goal: &Goal) {
        // Reset goal progress for new goal
        self.goal_progress = GoalProgress::new();
        
        // Add goal-related focus areas
        self.add_focus_area(goal.description().to_string());
        
        self.update_timestamp();
    }
    
    /// Update goal progress
    pub fn update_goal_progress(&mut self, completion: f64) {
        self.goal_progress.completion = completion.max(0.0).min(1.0);
        self.update_timestamp();
    }
    
    /// Mark a sub-goal as completed
    pub fn complete_subgoal(&mut self, subgoal: String) {
        if !self.goal_progress.completed_subgoals.contains(&subgoal) {
            self.goal_progress.completed_subgoals.push(subgoal);
            self.update_timestamp();
        }
    }
    
    /// Set the active sub-goal
    pub fn set_active_subgoal(&mut self, subgoal: Option<String>) {
        self.goal_progress.active_subgoal = subgoal;
        self.update_timestamp();
    }
    
    /// Add an obstacle
    pub fn add_obstacle(&mut self, obstacle: String) {
        if !self.goal_progress.obstacles.contains(&obstacle) {
            self.goal_progress.obstacles.push(obstacle);
            self.update_timestamp();
        }
    }
    
    /// Add a strategy
    pub fn add_strategy(&mut self, strategy: String) {
        if !self.goal_progress.strategies.contains(&strategy) {
            self.goal_progress.strategies.push(strategy);
            self.update_timestamp();
        }
    }
    
    /// Get goal progress
    pub fn goal_progress(&self) -> &GoalProgress {
        &self.goal_progress
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
        self.update_timestamp();
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    
    /// Get last updated timestamp
    pub fn last_updated(&self) -> u64 {
        self.last_updated
    }
    
    /// Generate a summary of the current context
    pub fn summarize(&self) -> String {
        let mut summary = format!("Session: {}\n", self.session_id);
        summary.push_str(&format!("Keywords: {:?}\n", self.keywords));
        summary.push_str(&format!("Focus Areas: {:?}\n", self.focus_areas));
        summary.push_str(&format!("Goal Progress: {:.1}%\n", self.goal_progress.completion * 100.0));
        
        if let Some(ref subgoal) = self.goal_progress.active_subgoal {
            summary.push_str(&format!("Active Sub-goal: {}\n", subgoal));
        }
        
        summary.push_str(&format!("Recent Patterns: {}\n", self.writing_patterns.len()));
        summary.push_str(&format!("Research Terms: {}\n", self.research_terms.len()));
        
        summary
    }
    
    /// Check if context is stale (hasn't been updated recently)
    pub fn is_stale(&self, threshold_seconds: u64) -> bool {
        let now = current_timestamp();
        now - self.last_updated > threshold_seconds
    }
    
    /// Reset the context for a new session
    pub fn reset(&mut self) {
        self.session_id = generate_session_id();
        self.keywords.clear();
        self.research_terms.clear();
        self.writing_patterns.clear();
        self.focus_areas.clear();
        self.goal_progress = GoalProgress::new();
        self.metadata.clear();
        self.update_timestamp();
    }
    
    /// Update the timestamp
    fn update_timestamp(&mut self) {
        self.last_updated = current_timestamp();
    }
}

impl WritingPattern {
    /// Create a new writing pattern
    pub fn new(pattern_type: String, value: f64, confidence: f64) -> Self {
        Self {
            pattern_type,
            value,
            confidence,
            timestamp: current_timestamp(),
        }
    }
    
    /// Get pattern type
    pub fn pattern_type(&self) -> &str {
        &self.pattern_type
    }
    
    /// Get pattern value
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Get confidence
    pub fn confidence(&self) -> f64 {
        self.confidence
    }
    
    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
    
    /// Check if pattern is recent
    pub fn is_recent(&self, threshold_seconds: u64) -> bool {
        let now = current_timestamp();
        now - self.timestamp <= threshold_seconds
    }
}

impl GoalProgress {
    /// Create new goal progress
    pub fn new() -> Self {
        Self {
            completion: 0.0,
            completed_subgoals: Vec::new(),
            active_subgoal: None,
            obstacles: Vec::new(),
            strategies: Vec::new(),
        }
    }
    
    /// Get completion percentage
    pub fn completion(&self) -> f64 {
        self.completion
    }
    
    /// Get completed sub-goals
    pub fn completed_subgoals(&self) -> &[String] {
        &self.completed_subgoals
    }
    
    /// Get active sub-goal
    pub fn active_subgoal(&self) -> Option<&String> {
        self.active_subgoal.as_ref()
    }
    
    /// Get obstacles
    pub fn obstacles(&self) -> &[String] {
        &self.obstacles
    }
    
    /// Get strategies
    pub fn strategies(&self) -> &[String] {
        &self.strategies
    }
}

/// Generate a unique session ID
fn generate_session_id() -> String {
    format!("session_{}", current_timestamp())
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
    fn test_context_creation() {
        let context = Context::new();
        
        assert!(!context.session_id().is_empty());
        assert_eq!(context.keywords().len(), 0);
        assert_eq!(context.focus_areas().len(), 0);
        assert_eq!(context.goal_progress().completion(), 0.0);
    }
    
    #[test]
    fn test_keyword_management() {
        let mut context = Context::new();
        
        context.add_keyword("test".to_string());
        assert!(context.keywords().contains("test"));
        
        assert!(context.remove_keyword("test"));
        assert!(!context.keywords().contains("test"));
        
        assert!(!context.remove_keyword("nonexistent"));
    }
    
    #[test]
    fn test_writing_patterns() {
        let mut context = Context::new();
        
        let pattern = WritingPattern::new("sentence_length".to_string(), 15.5, 0.8);
        context.add_writing_pattern(pattern);
        
        let patterns = context.get_patterns("sentence_length");
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].value(), 15.5);
    }
    
    #[test]
    fn test_goal_progress() {
        let mut context = Context::new();
        
        context.update_goal_progress(0.5);
        assert_eq!(context.goal_progress().completion(), 0.5);
        
        context.complete_subgoal("introduction".to_string());
        assert!(context.goal_progress().completed_subgoals().contains(&"introduction".to_string()));
        
        context.set_active_subgoal(Some("body".to_string()));
        assert_eq!(context.goal_progress().active_subgoal(), Some(&"body".to_string()));
    }
} 