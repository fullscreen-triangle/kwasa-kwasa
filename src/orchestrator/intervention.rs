use std::fmt::Debug;
use log::{debug, warn};
use crate::orchestrator::goal::Goal;
use crate::orchestrator::context::Context;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    // Helper function to create a test context
    fn create_test_context() -> Context {
        let mut context = Context::new();
        context.add_keywords(vec![
            "machine".to_string(),
            "learning".to_string(),
            "algorithm".to_string(),
            "neural".to_string(),
            "network".to_string(),
        ]);
        context
    }
    
    // Helper function to create a test goal
    fn create_test_goal() -> Goal {
        Goal::new("Write about machine learning", 0.3)
    }
    
    #[test]
    fn test_readability_intervention() {
        let intervention = ReadabilityIntervention::new();
        let goal = create_test_goal();
        let context = create_test_context();
        
        // Simple, readable text
        let simple_text = "This is a simple text. It has short sentences. Words are small.";
        let result = intervention.process_text(simple_text, &goal, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), simple_text);
        
        // Complex, less readable text
        let complex_text = "The implementation of metacognitive orchestration within the framework necessitates sophisticated algorithms for contextual awareness and semantic understanding of multifaceted textual structures.";
        let result = intervention.process_text(complex_text, &goal, &context);
        assert!(result.is_ok());
        
        // Should be simplified (broken into shorter sentences)
        let simplified = result.unwrap();
        assert_ne!(simplified, complex_text);
        assert!(simplified.contains(". "));
    }
    
    #[test]
    fn test_coherence_intervention() {
        let intervention = CoherenceIntervention::new();
        let goal = create_test_goal();
        let context = create_test_context();
        
        // Coherent text (contains context keywords)
        let coherent_text = "Neural networks are a type of machine learning algorithm.";
        let result = intervention.process_text(coherent_text, &goal, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), coherent_text);
        
        // Non-coherent text (new topic)
        let non_coherent = "Climate change is affecting global temperatures.";
        let result = intervention.process_text(non_coherent, &goal, &context);
        assert!(result.is_ok());
        
        // Should add a transition
        let with_transition = result.unwrap();
        assert_ne!(with_transition, non_coherent);
        assert!(with_transition.contains("Moving from"));
    }
    
    #[test]
    fn test_research_intervention() {
        let intervention = ResearchIntervention::new();
        let goal = create_test_goal();
        let context = create_test_context();
        
        // Text that needs research
        let uncertain_text = "I'm not sure how transformers work, but maybe they use attention mechanisms.";
        let result = intervention.process_text(uncertain_text, &goal, &context);
        assert!(result.is_ok());
        
        // This intervention doesn't modify text, it just suggests research
        assert_eq!(result.unwrap(), uncertain_text);
        
        // Text that doesn't need research
        let certain_text = "Transformers use attention mechanisms to process sequential data.";
        let result = intervention.process_text(certain_text, &goal, &context);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), certain_text);
    }
    
    #[test]
    fn test_extract_keywords() {
        let text = "The quick brown fox jumps over the lazy dog";
        let keywords = extract_keywords(text);
        
        assert!(keywords.contains(&"quick".to_string()));
        assert!(keywords.contains(&"brown".to_string()));
        assert!(keywords.contains(&"jumps".to_string()));
        assert!(keywords.contains(&"lazy".to_string()));
        
        // Should not contain stop words
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"over".to_string()));
    }
} 