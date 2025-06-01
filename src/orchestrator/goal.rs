use std::collections::HashSet;
use std::fmt;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Represents a writing goal with hierarchical structure and progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Unique identifier for this goal
    id: String,
    
    /// Human-readable description of the goal
    description: String,
    
    /// Type of goal (e.g., "essay", "research", "creative", "technical")
    goal_type: GoalType,
    
    /// Priority level (0.0 = low, 1.0 = high)
    priority: f64,
    
    /// Current completion percentage (0.0 - 1.0)
    completion: f64,
    
    /// Target metrics for this goal
    target_metrics: GoalMetrics,
    
    /// Current metrics achieved
    current_metrics: GoalMetrics,
    
    /// Keywords that define this goal's domain
    keywords: Vec<String>,
    
    /// Sub-goals that make up this goal
    sub_goals: Vec<Goal>,
    
    /// Dependencies on other goals
    dependencies: Vec<String>,
    
    /// Success criteria for completion
    success_criteria: Vec<SuccessCriterion>,
    
    /// Strategies to achieve this goal
    strategies: Vec<Strategy>,
    
    /// Current status
    status: GoalStatus,
    
    /// Creation timestamp
    created_at: u64,
    
    /// Last updated timestamp
    updated_at: u64,
    
    /// Due date (optional)
    due_date: Option<u64>,
    
    /// Additional metadata
    metadata: HashMap<String, String>,
}

/// Types of writing goals
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalType {
    /// Academic essay or paper
    Academic,
    /// Creative writing (fiction, poetry, etc.)
    Creative,
    /// Technical documentation
    Technical,
    /// Research paper or report
    Research,
    /// Business writing (proposals, reports)
    Business,
    /// Personal writing (journal, blog)
    Personal,
    /// Educational content
    Educational,
    /// Marketing content
    Marketing,
    /// Custom goal type
    Custom(String),
}

/// Metrics that define goal parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalMetrics {
    /// Target word count
    pub word_count: Option<usize>,
    
    /// Target page count
    pub page_count: Option<usize>,
    
    /// Reading level (Flesch-Kincaid grade level)
    pub reading_level: Option<f64>,
    
    /// Readability score (0.0 - 1.0, higher is more readable)
    pub readability_score: Option<f64>,
    
    /// Quality score (0.0 - 1.0)
    pub quality_score: Option<f64>,
    
    /// Coherence score (0.0 - 1.0)
    pub coherence_score: Option<f64>,
    
    /// Citation count (for academic writing)
    pub citation_count: Option<usize>,
    
    /// Section count
    pub section_count: Option<usize>,
    
    /// Average sentence length
    pub avg_sentence_length: Option<f64>,
    
    /// Vocabulary diversity (unique words / total words)
    pub vocabulary_diversity: Option<f64>,
}

/// Success criteria for goal completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    /// Name of the criterion
    pub name: String,
    
    /// Description of what constitutes success
    pub description: String,
    
    /// Metric to evaluate
    pub metric: String,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Target value
    pub target_value: f64,
    
    /// Weight in overall success calculation (0.0 - 1.0)
    pub weight: f64,
    
    /// Whether this criterion is currently met
    pub is_met: bool,
}

/// Comparison operators for success criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
    Within(f64), // Within X% of target
}

/// Strategies for achieving goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Name of the strategy
    pub name: String,
    
    /// Description of the strategy
    pub description: String,
    
    /// Type of strategy
    pub strategy_type: StrategyType,
    
    /// Parameters for the strategy
    pub parameters: HashMap<String, String>,
    
    /// Whether this strategy is currently active
    pub is_active: bool,
    
    /// Effectiveness score (0.0 - 1.0)
    pub effectiveness: Option<f64>,
}

/// Types of strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    /// Research-based strategy
    Research,
    /// Writing technique
    Technique,
    /// Time management
    TimeManagement,
    /// Quality improvement
    QualityImprovement,
    /// Content organization
    Organization,
    /// Style adjustment
    StyleAdjustment,
    /// Custom strategy
    Custom(String),
}

/// Current status of a goal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GoalStatus {
    /// Goal is planned but not started
    Planned,
    /// Goal is currently being worked on
    InProgress,
    /// Goal is on hold
    OnHold,
    /// Goal is completed successfully
    Completed,
    /// Goal was cancelled
    Cancelled,
    /// Goal is blocked by dependencies
    Blocked,
}

impl Goal {
    /// Create a new goal with basic information
    pub fn new(description: &str, priority: f64) -> Self {
        let id = generate_goal_id();
        let now = current_timestamp();
        
        Self {
            id,
            description: description.to_string(),
            goal_type: GoalType::Personal,
            priority: priority.max(0.0).min(1.0),
            completion: 0.0,
            target_metrics: GoalMetrics::default(),
            current_metrics: GoalMetrics::default(),
            keywords: Vec::new(),
            sub_goals: Vec::new(),
            dependencies: Vec::new(),
            success_criteria: Vec::new(),
            strategies: Vec::new(),
            status: GoalStatus::Planned,
            created_at: now,
            updated_at: now,
            due_date: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a goal with specific type and keywords
    pub fn with_type_and_keywords(
        description: &str,
        goal_type: GoalType,
        keywords: Vec<String>,
        priority: f64,
    ) -> Self {
        let mut goal = Self::new(description, priority);
        goal.goal_type = goal_type;
        goal.keywords = keywords;
        goal
    }
    
    /// Get goal ID
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Get goal description
    pub fn description(&self) -> &str {
        &self.description
    }
    
    /// Get goal type
    pub fn goal_type(&self) -> &GoalType {
        &self.goal_type
    }
    
    /// Get keywords
    pub fn keywords(&self) -> &[String] {
        &self.keywords
    }
    
    /// Get current completion percentage
    pub fn completion(&self) -> f64 {
        self.completion
    }
    
    /// Get priority
    pub fn priority(&self) -> f64 {
        self.priority
    }
    
    /// Get status
    pub fn status(&self) -> &GoalStatus {
        &self.status
    }
    
    /// Get target metrics
    pub fn target_metrics(&self) -> &GoalMetrics {
        &self.target_metrics
    }
    
    /// Get current metrics
    pub fn current_metrics(&self) -> &GoalMetrics {
        &self.current_metrics
    }
    
    /// Get sub-goals
    pub fn sub_goals(&self) -> &[Goal] {
        &self.sub_goals
    }
    
    /// Get success criteria
    pub fn success_criteria(&self) -> &[SuccessCriterion] {
        &self.success_criteria
    }
    
    /// Get strategies
    pub fn strategies(&self) -> &[Strategy] {
        &self.strategies
    }
    
    /// Update completion percentage
    pub fn update_completion(&mut self, completion: f64) {
        self.completion = completion.max(0.0).min(1.0);
        self.updated_at = current_timestamp();
        
        // Auto-update status based on completion
        if self.completion >= 1.0 && self.status == GoalStatus::InProgress {
            self.status = GoalStatus::Completed;
        } else if self.completion > 0.0 && self.status == GoalStatus::Planned {
            self.status = GoalStatus::InProgress;
        }
    }
    
    /// Set goal status
    pub fn set_status(&mut self, status: GoalStatus) {
        self.status = status;
        self.updated_at = current_timestamp();
    }
    
    /// Add a sub-goal
    pub fn add_sub_goal(&mut self, sub_goal: Goal) {
        self.sub_goals.push(sub_goal);
        self.updated_at = current_timestamp();
    }
    
    /// Add a success criterion
    pub fn add_success_criterion(&mut self, criterion: SuccessCriterion) {
        self.success_criteria.push(criterion);
        self.updated_at = current_timestamp();
    }
    
    /// Add a strategy
    pub fn add_strategy(&mut self, strategy: Strategy) {
        self.strategies.push(strategy);
        self.updated_at = current_timestamp();
    }
    
    /// Set target metrics
    pub fn set_target_metrics(&mut self, metrics: GoalMetrics) {
        self.target_metrics = metrics;
        self.updated_at = current_timestamp();
    }
    
    /// Update current metrics
    pub fn update_current_metrics(&mut self, metrics: GoalMetrics) {
        self.current_metrics = metrics;
        self.updated_at = current_timestamp();
    }
    
    /// Set due date
    pub fn set_due_date(&mut self, due_date: Option<u64>) {
        self.due_date = due_date;
        self.updated_at = current_timestamp();
    }
    
    /// Check if text content is relevant to this goal
    pub fn is_relevant(&self, text: &str) -> bool {
        if self.keywords.is_empty() {
            return true; // No keywords means everything is relevant
        }
        
        let text_lower = text.to_lowercase();
        let matching_keywords = self.keywords.iter()
            .filter(|keyword| text_lower.contains(&keyword.to_lowercase()))
            .count();
        
        // Consider relevant if at least 20% of keywords match
        matching_keywords as f64 / self.keywords.len() as f64 >= 0.2
    }
    
    /// Evaluate alignment between text and goal (0.0 - 1.0)
    pub fn evaluate_alignment(&self, text: &str) -> f64 {
        if self.keywords.is_empty() {
            return 0.5; // Neutral alignment when no keywords
        }
        
        let text_lower = text.to_lowercase();
        let matching_keywords = self.keywords.iter()
            .filter(|keyword| text_lower.contains(&keyword.to_lowercase()))
            .count();
        
        matching_keywords as f64 / self.keywords.len() as f64
    }
    
    /// Calculate overall success score based on criteria
    pub fn calculate_success_score(&self) -> f64 {
        if self.success_criteria.is_empty() {
            return self.completion; // Use completion as fallback
        }
        
        let total_weight: f64 = self.success_criteria.iter()
            .map(|c| c.weight)
            .sum();
        
        if total_weight == 0.0 {
            return 0.0;
        }
        
        let weighted_score: f64 = self.success_criteria.iter()
            .map(|c| if c.is_met { c.weight } else { 0.0 })
            .sum();
        
        weighted_score / total_weight
    }
    
    /// Update success criteria based on current metrics
    pub fn update_success_criteria(&mut self) {
        for criterion in &mut self.success_criteria {
            let current_value = self.get_metric_value(&criterion.metric);
            
            criterion.is_met = match criterion.operator {
                ComparisonOperator::GreaterThan => current_value > criterion.target_value,
                ComparisonOperator::GreaterThanOrEqual => current_value >= criterion.target_value,
                ComparisonOperator::LessThan => current_value < criterion.target_value,
                ComparisonOperator::LessThanOrEqual => current_value <= criterion.target_value,
                ComparisonOperator::Equal => (current_value - criterion.target_value).abs() < 0.001,
                ComparisonOperator::NotEqual => (current_value - criterion.target_value).abs() >= 0.001,
                ComparisonOperator::Within(percentage) => {
                    let tolerance = criterion.target_value * percentage / 100.0;
                    (current_value - criterion.target_value).abs() <= tolerance
                }
            };
        }
        
        self.updated_at = current_timestamp();
    }
    
    /// Get the value of a specific metric
    fn get_metric_value(&self, metric: &str) -> f64 {
        match metric {
            "word_count" => self.current_metrics.word_count.unwrap_or(0) as f64,
            "page_count" => self.current_metrics.page_count.unwrap_or(0) as f64,
            "reading_level" => self.current_metrics.reading_level.unwrap_or(0.0),
            "readability_score" => self.current_metrics.readability_score.unwrap_or(0.0),
            "quality_score" => self.current_metrics.quality_score.unwrap_or(0.0),
            "coherence_score" => self.current_metrics.coherence_score.unwrap_or(0.0),
            "citation_count" => self.current_metrics.citation_count.unwrap_or(0) as f64,
            "section_count" => self.current_metrics.section_count.unwrap_or(0) as f64,
            "avg_sentence_length" => self.current_metrics.avg_sentence_length.unwrap_or(0.0),
            "vocabulary_diversity" => self.current_metrics.vocabulary_diversity.unwrap_or(0.0),
            "completion" => self.completion,
            _ => 0.0,
        }
    }
    
    /// Check if goal is overdue
    pub fn is_overdue(&self) -> bool {
        if let Some(due_date) = self.due_date {
            let now = current_timestamp();
            now > due_date && self.status != GoalStatus::Completed
        } else {
            false
        }
    }
    
    /// Get progress summary
    pub fn progress_summary(&self) -> String {
        let mut summary = format!("Goal: {}\n", self.description);
        summary.push_str(&format!("Status: {:?}\n", self.status));
        summary.push_str(&format!("Completion: {:.1}%\n", self.completion * 100.0));
        summary.push_str(&format!("Priority: {:.1}\n", self.priority));
        
        if !self.sub_goals.is_empty() {
            summary.push_str(&format!("Sub-goals: {}\n", self.sub_goals.len()));
        }
        
        if !self.success_criteria.is_empty() {
            let met_criteria = self.success_criteria.iter().filter(|c| c.is_met).count();
            summary.push_str(&format!("Criteria met: {}/{}\n", met_criteria, self.success_criteria.len()));
        }
        
        if self.is_overdue() {
            summary.push_str("⚠️ OVERDUE\n");
        }
        
        summary
    }
}

impl SuccessCriterion {
    /// Create a new success criterion
    pub fn new(
        name: String,
        description: String,
        metric: String,
        operator: ComparisonOperator,
        target_value: f64,
        weight: f64,
    ) -> Self {
        Self {
            name,
            description,
            metric,
            operator,
            target_value,
            weight: weight.max(0.0).min(1.0),
            is_met: false,
        }
    }
}

impl Strategy {
    /// Create a new strategy
    pub fn new(
        name: String,
        description: String,
        strategy_type: StrategyType,
    ) -> Self {
        Self {
            name,
            description,
            strategy_type,
            parameters: HashMap::new(),
            is_active: false,
            effectiveness: None,
        }
    }
    
    /// Activate this strategy
    pub fn activate(&mut self) {
        self.is_active = true;
    }
    
    /// Deactivate this strategy
    pub fn deactivate(&mut self) {
        self.is_active = false;
    }
    
    /// Set effectiveness score
    pub fn set_effectiveness(&mut self, score: f64) {
        self.effectiveness = Some(score.max(0.0).min(1.0));
    }
}

/// Generate a unique goal ID
fn generate_goal_id() -> String {
    format!("goal_{}", current_timestamp())
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_goal_creation() {
        let goal = Goal::new("Write an essay about AI", 0.8);
        
        assert!(!goal.id().is_empty());
        assert_eq!(goal.description(), "Write an essay about AI");
        assert_eq!(goal.priority(), 0.8);
        assert_eq!(goal.completion(), 0.0);
        assert_eq!(goal.status(), &GoalStatus::Planned);
    }
    
    #[test]
    fn test_goal_with_keywords() {
        let keywords = vec!["AI".to_string(), "machine learning".to_string()];
        let goal = Goal::with_type_and_keywords(
            "AI research paper",
            GoalType::Academic,
            keywords.clone(),
            0.9,
        );
        
        assert_eq!(goal.goal_type(), &GoalType::Academic);
        assert_eq!(goal.keywords(), &keywords);
    }
    
    #[test]
    fn test_relevance_check() {
        let keywords = vec!["AI".to_string(), "machine learning".to_string()];
        let goal = Goal::with_type_and_keywords(
            "AI research",
            GoalType::Academic,
            keywords,
            0.8,
        );
        
        assert!(goal.is_relevant("This text discusses AI and its applications"));
        assert!(!goal.is_relevant("This text is about cooking recipes"));
    }
    
    #[test]
    fn test_completion_update() {
        let mut goal = Goal::new("Test goal", 0.5);
        
        goal.update_completion(0.5);
        assert_eq!(goal.completion(), 0.5);
        assert_eq!(goal.status(), &GoalStatus::InProgress);
        
        goal.update_completion(1.0);
        assert_eq!(goal.completion(), 1.0);
        assert_eq!(goal.status(), &GoalStatus::Completed);
    }
    
    #[test]
    fn test_success_criteria() {
        let mut goal = Goal::new("Test goal", 0.5);
        
        let criterion = SuccessCriterion::new(
            "Word count".to_string(),
            "Must have at least 1000 words".to_string(),
            "word_count".to_string(),
            ComparisonOperator::GreaterThanOrEqual,
            1000.0,
            1.0,
        );
        
        goal.add_success_criterion(criterion);
        
        // Set current metrics
        let mut metrics = GoalMetrics::default();
        metrics.word_count = Some(1200);
        goal.update_current_metrics(metrics);
        
        goal.update_success_criteria();
        
        assert_eq!(goal.calculate_success_score(), 1.0);
    }
} 