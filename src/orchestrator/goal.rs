use std::collections::HashSet;
use std::fmt;

/// Represents a writing goal
#[derive(Debug, Clone)]
pub struct Goal {
    /// The description of the goal
    description: String,
    
    /// Keywords relevant to the goal
    keywords: HashSet<String>,
    
    /// Threshold for relevance matching (0.0 to 1.0)
    relevance_threshold: f64,
    
    /// Current progress towards the goal (0.0 to 1.0)
    progress: f64,
}

impl Goal {
    /// Create a new goal with a description and relevance threshold
    pub fn new(description: &str, relevance_threshold: f64) -> Self {
        let keywords = extract_keywords(description);
        
        Self {
            description: description.to_string(),
            keywords,
            relevance_threshold,
            progress: 0.0,
        }
    }
    
    /// Create a new goal with explicit keywords
    pub fn with_keywords(description: &str, keywords: Vec<String>, relevance_threshold: f64) -> Self {
        let mut keyword_set = HashSet::new();
        for keyword in keywords {
            keyword_set.insert(keyword.to_lowercase());
        }
        
        Self {
            description: description.to_string(),
            keywords: keyword_set,
            relevance_threshold,
            progress: 0.0,
        }
    }
    
    /// Get the goal description
    pub fn description(&self) -> &str {
        &self.description
    }
    
    /// Get the goal keywords
    pub fn keywords(&self) -> &HashSet<String> {
        &self.keywords
    }
    
    /// Get the current progress
    pub fn progress(&self) -> f64 {
        self.progress
    }
    
    /// Set the progress
    pub fn set_progress(&mut self, progress: f64) {
        self.progress = progress.max(0.0).min(1.0);
    }
    
    /// Update the progress
    pub fn update_progress(&mut self, increment: f64) {
        self.progress = (self.progress + increment).max(0.0).min(1.0);
    }
    
    /// Add a keyword to the goal
    pub fn add_keyword(&mut self, keyword: &str) {
        self.keywords.insert(keyword.to_lowercase());
    }
    
    /// Check if text is relevant to the goal
    pub fn is_relevant(&self, text: &str) -> bool {
        let text_keywords = extract_keywords(text);
        let relevance_score = self.calculate_relevance(&text_keywords);
        
        relevance_score >= self.relevance_threshold
    }
    
    /// Calculate the relevance score of text to the goal
    fn calculate_relevance(&self, text_keywords: &HashSet<String>) -> f64 {
        if self.keywords.is_empty() || text_keywords.is_empty() {
            return 0.0;
        }
        
        // Count matching keywords
        let matching_count = text_keywords.iter()
            .filter(|kw| self.keywords.contains(*kw))
            .count();
        
        // Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
        let union_count = self.keywords.len() + text_keywords.len() - matching_count;
        
        if union_count == 0 {
            return 0.0;
        }
        
        matching_count as f64 / union_count as f64
    }
    
    /// Evaluate how well text aligns with the goal
    pub fn evaluate_alignment(&self, text: &str) -> f64 {
        let text_keywords = extract_keywords(text);
        self.calculate_relevance(&text_keywords)
    }
    
    /// Check if the goal is complete
    pub fn is_complete(&self) -> bool {
        self.progress >= 1.0
    }
}

impl fmt::Display for Goal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Goal: {} (progress: {:.1}%)", 
            self.description,
            self.progress * 100.0
        )
    }
}

/// Extract keywords from text
fn extract_keywords(text: &str) -> HashSet<String> {
    let stop_words = vec![
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "from", "of", "as", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "must", "that", "this", "these", "those",
    ];
    
    let mut keywords = HashSet::new();
    
    for word in text.split_whitespace() {
        let cleaned = word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        
        if cleaned.len() > 3 && !stop_words.contains(&cleaned.as_str()) {
            keywords.insert(cleaned);
        }
    }
    
    keywords
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_goal_creation() {
        let goal = Goal::new("Write an article about machine learning algorithms", 0.3);
        
        assert_eq!(goal.description(), "Write an article about machine learning algorithms");
        assert!(goal.keywords().contains("machine"));
        assert!(goal.keywords().contains("learning"));
        assert!(goal.keywords().contains("algorithms"));
        assert!(!goal.keywords().contains("write"));
        assert!(!goal.keywords().contains("about"));
        
        assert_eq!(goal.progress(), 0.0);
        assert!(!goal.is_complete());
    }
    
    #[test]
    fn test_goal_with_keywords() {
        let goal = Goal::with_keywords(
            "ML Research", 
            vec!["machine".to_string(), "learning".to_string(), "neural".to_string(), "networks".to_string()],
            0.3
        );
        
        assert_eq!(goal.description(), "ML Research");
        assert!(goal.keywords().contains("machine"));
        assert!(goal.keywords().contains("learning"));
        assert!(goal.keywords().contains("neural"));
        assert!(goal.keywords().contains("networks"));
    }
    
    #[test]
    fn test_goal_progress() {
        let mut goal = Goal::new("Test goal", 0.3);
        
        assert_eq!(goal.progress(), 0.0);
        
        goal.update_progress(0.5);
        assert_eq!(goal.progress(), 0.5);
        
        goal.update_progress(0.6);
        assert_eq!(goal.progress(), 1.0); // Should cap at 1.0
        
        goal.set_progress(0.7);
        assert_eq!(goal.progress(), 0.7);
        
        goal.update_progress(-0.8);
        assert_eq!(goal.progress(), 0.0); // Should floor at 0.0
    }
    
    #[test]
    fn test_goal_relevance() {
        let goal = Goal::with_keywords(
            "Machine Learning Research", 
            vec!["machine".to_string(), "learning".to_string(), "neural".to_string(), "networks".to_string()],
            0.2
        );
        
        // Highly relevant text
        let relevant_text = "Recent advances in neural networks have revolutionized machine learning research.";
        assert!(goal.is_relevant(relevant_text));
        
        // Somewhat relevant text
        let somewhat_relevant = "The learning process involves mathematical optimization.";
        // May or may not be relevant depending on threshold
        
        // Irrelevant text
        let irrelevant_text = "The weather today is sunny with a slight chance of rain.";
        assert!(!goal.is_relevant(irrelevant_text));
    }
    
    #[test]
    fn test_alignment_evaluation() {
        let goal = Goal::with_keywords(
            "Climate Change Research", 
            vec!["climate".to_string(), "change".to_string(), "warming".to_string(), "global".to_string(), "carbon".to_string()],
            0.2
        );
        
        let high_alignment = "Global warming and climate change are affected by carbon emissions.";
        let medium_alignment = "Global temperatures have been rising in recent decades.";
        let low_alignment = "Environmental policies are important for sustainable development.";
        let no_alignment = "The latest smartphone models were announced yesterday.";
        
        let high_score = goal.evaluate_alignment(high_alignment);
        let medium_score = goal.evaluate_alignment(medium_alignment);
        let low_score = goal.evaluate_alignment(low_alignment);
        let no_score = goal.evaluate_alignment(no_alignment);
        
        assert!(high_score > medium_score);
        assert!(medium_score > low_score);
        assert!(low_score > no_score);
    }
} 