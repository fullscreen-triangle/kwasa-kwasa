use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use crate::orchestrator::goal::Goal;

/// Maximum number of recent keywords to track
const MAX_RECENT_KEYWORDS: usize = 50;

/// Maximum number of recent research terms to track
const MAX_RECENT_RESEARCH: usize = 20;

/// Represents the writing context
#[derive(Debug, Clone)]
pub struct Context {
    /// Currently active keywords
    keywords: HashSet<String>,
    
    /// Recent keywords with timestamps (for recency weighting)
    recent_keywords: VecDeque<(String, Instant)>,
    
    /// Currently active research terms
    research_terms: HashSet<String>,
    
    /// Recent research terms with timestamps
    recent_research: VecDeque<(String, Instant)>,
    
    /// Topic transitions (from -> to) with counts
    topic_transitions: HashMap<String, HashMap<String, usize>>,
    
    /// Goal state (relevance to current goal)
    goal_relevance: f64,
    
    /// Flow state tracking
    writing_speed: Option<f64>, // words per minute
    
    /// Creation time
    created_at: Instant,
    
    /// Last activity time
    last_activity: Instant,
}

impl Context {
    /// Create a new context
    pub fn new() -> Self {
        let now = Instant::now();
        
        Self {
            keywords: HashSet::new(),
            recent_keywords: VecDeque::new(),
            research_terms: HashSet::new(),
            recent_research: VecDeque::new(),
            topic_transitions: HashMap::new(),
            goal_relevance: 0.0,
            writing_speed: None,
            created_at: now,
            last_activity: now,
        }
    }
    
    /// Add a keyword to the context
    pub fn add_keyword(&mut self, keyword: String) {
        let now = Instant::now();
        
        // Add to recent keywords
        if !self.recent_keywords.iter().any(|(k, _)| k == &keyword) {
            self.recent_keywords.push_back((keyword.clone(), now));
            
            // Maintain maximum size
            if self.recent_keywords.len() > MAX_RECENT_KEYWORDS {
                self.recent_keywords.pop_front();
            }
        }
        
        // Add to active keywords
        self.keywords.insert(keyword);
        
        self.last_activity = now;
    }
    
    /// Add multiple keywords to the context
    pub fn add_keywords(&mut self, keywords: Vec<String>) {
        for keyword in keywords {
            self.add_keyword(keyword);
        }
    }
    
    /// Get all active keywords
    pub fn get_keywords(&self) -> &HashSet<String> {
        &self.keywords
    }
    
    /// Get recent keywords sorted by recency
    pub fn get_recent_keywords(&self) -> Vec<String> {
        let mut recent: Vec<_> = self.recent_keywords.iter().collect();
        recent.sort_by(|(_, a_time), (_, b_time)| b_time.cmp(a_time));
        
        recent.into_iter()
            .map(|(keyword, _)| keyword.clone())
            .collect()
    }
    
    /// Add a research term to the context
    pub fn add_research_term(&mut self, term: &str) {
        let now = Instant::now();
        let term = term.to_lowercase();
        
        // Add to recent research
        if !self.recent_research.iter().any(|(t, _)| t == &term) {
            self.recent_research.push_back((term.clone(), now));
            
            // Maintain maximum size
            if self.recent_research.len() > MAX_RECENT_RESEARCH {
                self.recent_research.pop_front();
            }
        }
        
        // Add to active research terms
        self.research_terms.insert(term);
        
        self.last_activity = now;
    }
    
    /// Get all active research terms
    pub fn get_research_terms(&self) -> &HashSet<String> {
        &self.research_terms
    }
    
    /// Get recent research terms sorted by recency
    pub fn get_recent_research(&self) -> Vec<String> {
        let mut recent: Vec<_> = self.recent_research.iter().collect();
        recent.sort_by(|(_, a_time), (_, b_time)| b_time.cmp(a_time));
        
        recent.into_iter()
            .map(|(term, _)| term.clone())
            .collect()
    }
    
    /// Register a topic transition
    pub fn register_topic_transition(&mut self, from: &str, to: &str) {
        let from = from.to_lowercase();
        let to = to.to_lowercase();
        
        let transitions = self.topic_transitions
            .entry(from)
            .or_insert_with(HashMap::new);
        
        let count = transitions.entry(to).or_insert(0);
        *count += 1;
        
        self.last_activity = Instant::now();
    }
    
    /// Get the most common transition from a topic
    pub fn most_common_transition(&self, from: &str) -> Option<String> {
        let from = from.to_lowercase();
        
        if let Some(transitions) = self.topic_transitions.get(&from) {
            transitions.iter()
                .max_by_key(|(_, &count)| count)
                .map(|(to, _)| to.clone())
        } else {
            None
        }
    }
    
    /// Update the goal state based on the current goal
    pub fn update_goal_state(&mut self, goal: &Goal) {
        // Calculate relevance between context keywords and goal keywords
        let mut matching_count = 0;
        
        for keyword in &self.keywords {
            if goal.keywords().contains(keyword) {
                matching_count += 1;
            }
        }
        
        if self.keywords.is_empty() || goal.keywords().is_empty() {
            self.goal_relevance = 0.0;
        } else {
            // Simple relevance score: percentage of context keywords matching goal keywords
            self.goal_relevance = matching_count as f64 / self.keywords.len() as f64;
        }
        
        self.last_activity = Instant::now();
    }
    
    /// Get the goal relevance score
    pub fn goal_relevance(&self) -> f64 {
        self.goal_relevance
    }
    
    /// Set the current writing speed
    pub fn set_writing_speed(&mut self, words_per_minute: f64) {
        self.writing_speed = Some(words_per_minute);
        self.last_activity = Instant::now();
    }
    
    /// Get the current writing speed
    pub fn writing_speed(&self) -> Option<f64> {
        self.writing_speed
    }
    
    /// Get the time since the context was created
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// Get the time since the last activity
    pub fn time_since_last_activity(&self) -> Duration {
        self.last_activity.elapsed()
    }
    
    /// Check if the context is stale (inactive for a while)
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.time_since_last_activity() > timeout
    }
    
    /// Get the most relevant keywords based on recency and frequency
    pub fn most_relevant_keywords(&self, count: usize) -> Vec<String> {
        // Score keywords based on recency and frequency
        let mut keyword_scores: HashMap<String, f64> = HashMap::new();
        
        // Consider recency
        for (i, (keyword, _)) in self.recent_keywords.iter().enumerate() {
            let recency_score = 1.0 - (i as f64 / self.recent_keywords.len() as f64);
            let score = keyword_scores.entry(keyword.clone()).or_insert(0.0);
            *score += recency_score;
        }
        
        // Sort by score
        let mut scored_keywords: Vec<_> = keyword_scores.into_iter().collect();
        scored_keywords.sort_by(|(_, a_score), (_, b_score)| b_score.partial_cmp(a_score).unwrap());
        
        // Return top keywords
        scored_keywords.into_iter()
            .take(count)
            .map(|(keyword, _)| keyword)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    
    #[test]
    fn test_context_creation() {
        let context = Context::new();
        
        assert!(context.get_keywords().is_empty());
        assert!(context.get_research_terms().is_empty());
        assert_eq!(context.goal_relevance(), 0.0);
        assert!(context.writing_speed().is_none());
    }
    
    #[test]
    fn test_adding_keywords() {
        let mut context = Context::new();
        
        context.add_keyword("machine".to_string());
        context.add_keyword("learning".to_string());
        context.add_keyword("algorithm".to_string());
        
        assert_eq!(context.get_keywords().len(), 3);
        assert!(context.get_keywords().contains("machine"));
        assert!(context.get_keywords().contains("learning"));
        assert!(context.get_keywords().contains("algorithm"));
        
        let recent = context.get_recent_keywords();
        assert_eq!(recent.len(), 3);
        // Most recent should be last added
        assert_eq!(recent[0], "algorithm");
    }
    
    #[test]
    fn test_research_terms() {
        let mut context = Context::new();
        
        context.add_research_term("neural networks");
        context.add_research_term("deep learning");
        
        assert_eq!(context.get_research_terms().len(), 2);
        assert!(context.get_research_terms().contains("neural networks"));
        assert!(context.get_research_terms().contains("deep learning"));
        
        let recent = context.get_recent_research();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0], "deep learning");
    }
    
    #[test]
    fn test_topic_transitions() {
        let mut context = Context::new();
        
        context.register_topic_transition("intro", "method");
        context.register_topic_transition("method", "results");
        context.register_topic_transition("intro", "related work");
        context.register_topic_transition("intro", "method");
        
        assert_eq!(context.most_common_transition("intro"), Some("method".to_string()));
        assert_eq!(context.most_common_transition("method"), Some("results".to_string()));
        assert_eq!(context.most_common_transition("nonexistent"), None);
    }
    
    #[test]
    fn test_goal_state_update() {
        let mut context = Context::new();
        
        // Add some keywords to the context
        context.add_keywords(vec![
            "machine".to_string(),
            "learning".to_string(),
            "neural".to_string(),
            "networks".to_string(),
            "unrelated".to_string(),
        ]);
        
        // Create a goal with some matching keywords
        let goal = Goal::with_keywords(
            "ML Research",
            vec!["machine".to_string(), "learning".to_string(), "algorithm".to_string()],
            0.3
        );
        
        // Update goal state
        context.update_goal_state(&goal);
        
        // 2 out of 5 keywords match
        assert!((context.goal_relevance() - 0.4).abs() < 0.001);
    }
    
    #[test]
    fn test_staleness() {
        let mut context = Context::new();
        
        assert!(!context.is_stale(Duration::from_secs(1)));
        
        // Sleep briefly
        sleep(Duration::from_millis(50));
        
        // Should still be fresh
        assert!(!context.is_stale(Duration::from_millis(100)));
        
        // But stale with a shorter timeout
        assert!(context.is_stale(Duration::from_millis(10)));
        
        // Update to refresh
        context.add_keyword("refresh".to_string());
        assert!(!context.is_stale(Duration::from_millis(10)));
    }
    
    #[test]
    fn test_most_relevant_keywords() {
        let mut context = Context::new();
        
        // Add keywords in a specific order
        context.add_keyword("first".to_string());
        context.add_keyword("second".to_string());
        context.add_keyword("third".to_string());
        
        // Most recent should be most relevant
        let relevant = context.most_relevant_keywords(2);
        assert_eq!(relevant.len(), 2);
        assert_eq!(relevant[0], "third");
        assert_eq!(relevant[1], "second");
    }
} 