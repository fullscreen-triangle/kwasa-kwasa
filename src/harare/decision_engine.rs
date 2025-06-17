use std::collections::HashMap;
use std::time::SystemTime;
use anyhow::Result;
use uuid::Uuid;
use super::{Decision, Task, Priority, ModuleInfo};

pub struct DecisionEngine {
    pub id: Uuid,
    pub decision_history: Vec<Decision>,
    pub decision_criteria: HashMap<String, f64>,
}

impl DecisionEngine {
    pub fn new() -> Self {
        let mut criteria = HashMap::new();
        criteria.insert("performance".to_string(), 0.3);
        criteria.insert("resource_usage".to_string(), 0.2);
        criteria.insert("priority".to_string(), 0.4);
        criteria.insert("dependencies".to_string(), 0.1);

        Self {
            id: Uuid::new_v4(),
            decision_history: Vec::new(),
            decision_criteria: criteria,
        }
    }

    pub async fn make_execution_decision(
        &mut self,
        tasks: &[Task],
        available_modules: &[ModuleInfo],
    ) -> Result<Decision> {
        let decision_id = Uuid::new_v4();
        
        // Simple decision making logic
        let best_task = self.select_best_task(tasks);
        let best_module = self.select_best_module(available_modules, &best_task);
        
        let decision = Decision {
            decision_id,
            decision_type: "task_execution".to_string(),
            rationale: format!(
                "Selected task {} for execution on module {}",
                best_task.map(|t| t.name.clone()).unwrap_or_else(|| "none".to_string()),
                best_module.map(|m| m.name.clone()).unwrap_or_else(|| "none".to_string())
            ),
            confidence: 0.8,
            alternatives_considered: tasks.iter().map(|t| t.name.clone()).collect(),
            expected_impact: 0.7,
        };

        self.decision_history.push(decision.clone());
        Ok(decision)
    }

    fn select_best_task(&self, tasks: &[Task]) -> Option<&Task> {
        tasks
            .iter()
            .filter(|t| matches!(t.status, super::TaskStatus::Pending))
            .max_by(|a, b| {
                let score_a = self.calculate_task_score(a);
                let score_b = self.calculate_task_score(b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    fn select_best_module(&self, modules: &[ModuleInfo], _task: Option<&Task>) -> Option<&ModuleInfo> {
        modules.first() // Simplified selection
    }

    fn calculate_task_score(&self, task: &Task) -> f64 {
        let priority_score = match task.priority {
            Priority::Critical => 1.0,
            Priority::High => 0.8,
            Priority::Normal => 0.6,
            Priority::Low => 0.4,
            Priority::Background => 0.2,
        };

        let urgency_score = if let Some(deadline) = task.deadline {
            let now = SystemTime::now();
            match deadline.duration_since(now) {
                Ok(duration) => {
                    let hours_left = duration.as_secs() as f64 / 3600.0;
                    if hours_left < 1.0 { 1.0 } else { 1.0 / hours_left.log10() }
                }
                Err(_) => 0.0, // Past deadline
            }
        } else {
            0.5 // No deadline
        };

        let dependency_score = 1.0 / (task.dependencies.len() as f64 + 1.0);

        priority_score * 0.5 + urgency_score * 0.3 + dependency_score * 0.2
    }

    pub fn get_decision_history(&self) -> &[Decision] {
        &self.decision_history
    }
} 