use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use anyhow::Result;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use super::{Task, ModuleInfo, UserIntent};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub plan_id: Uuid,
    pub tasks: Vec<Uuid>,
    pub steps: Vec<ExecutionStep>,
    pub estimated_duration: Duration,
    pub resource_requirements: PlanResourceRequirements,
    pub dependencies: Vec<PlanDependency>,
    pub contingencies: Vec<Contingency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_id: Uuid,
    pub step_type: StepType,
    pub target_module: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub expected_duration: Duration,
    pub dependencies: Vec<Uuid>,
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    Initialize,
    Process,
    Transform,
    Analyze,
    Validate,
    Finalize,
    Cleanup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanResourceRequirements {
    pub total_cpu_cores: u32,
    pub total_memory_mb: u64,
    pub total_disk_mb: u64,
    pub estimated_duration: Duration,
    pub parallel_execution_slots: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDependency {
    pub dependent_step: Uuid,
    pub prerequisite_step: Uuid,
    pub dependency_type: DependencyType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    DataFlow,
    ControlFlow,
    ResourceLock,
    Synchronization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contingency {
    pub trigger_condition: String,
    pub alternative_steps: Vec<ExecutionStep>,
    pub fallback_plan: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_strategy: BackoffStrategy,
    pub retry_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Linear(Duration),
    Exponential(Duration),
    Fixed(Duration),
    None,
}

pub struct ExecutionPlanner {
    pub planner_id: Uuid,
    pub active_plans: HashMap<Uuid, ExecutionPlan>,
    pub plan_templates: HashMap<String, ExecutionPlan>,
}

impl ExecutionPlanner {
    pub fn new() -> Self {
        Self {
            planner_id: Uuid::new_v4(),
            active_plans: HashMap::new(),
            plan_templates: HashMap::new(),
        }
    }

    pub async fn create_plan(
        &mut self,
        tasks: &[Task],
        available_modules: &[ModuleInfo],
        user_intent: Option<&UserIntent>,
    ) -> Result<ExecutionPlan> {
        let plan_id = Uuid::new_v4();
        
        let steps = self.generate_execution_steps(tasks, available_modules, user_intent).await?;
        let dependencies = self.analyze_dependencies(&steps)?;
        let resource_requirements = self.calculate_resource_requirements(&steps, available_modules)?;
        let estimated_duration = self.estimate_total_duration(&steps)?;
        let contingencies = self.generate_contingencies(&steps)?;

        let plan = ExecutionPlan {
            plan_id,
            tasks: tasks.iter().map(|t| t.id).collect(),
            steps,
            estimated_duration,
            resource_requirements,
            dependencies,
            contingencies,
        };

        self.active_plans.insert(plan_id, plan.clone());
        Ok(plan)
    }

    async fn generate_execution_steps(
        &self,
        tasks: &[Task],
        available_modules: &[ModuleInfo],
        _user_intent: Option<&UserIntent>,
    ) -> Result<Vec<ExecutionStep>> {
        let mut steps = Vec::new();

        for task in tasks {
            // Find appropriate module for this task
            let target_module = self.select_module_for_task(task, available_modules)?;
            
            let step = ExecutionStep {
                step_id: Uuid::new_v4(),
                step_type: self.determine_step_type(task),
                target_module: target_module.name.clone(),
                parameters: task.parameters.clone(),
                expected_duration: Duration::from_secs(60), // Default estimate
                dependencies: Vec::new(), // Will be filled by analyze_dependencies
                retry_policy: RetryPolicy {
                    max_attempts: 3,
                    backoff_strategy: BackoffStrategy::Exponential(Duration::from_secs(1)),
                    retry_conditions: vec!["timeout".to_string(), "resource_unavailable".to_string()],
                },
            };

            steps.push(step);
        }

        Ok(steps)
    }

    fn select_module_for_task(&self, task: &Task, available_modules: &[ModuleInfo]) -> Result<&ModuleInfo> {
        // Simple selection logic - in practice this would be more sophisticated
        for target in &task.target_modules {
            if let Some(module) = available_modules.iter().find(|m| &m.name == target) {
                return Ok(module);
            }
        }
        
        available_modules.first()
            .ok_or_else(|| anyhow::anyhow!("No available modules"))
    }

    fn determine_step_type(&self, task: &Task) -> StepType {
        match task.task_type {
            super::TaskType::Analysis => StepType::Analyze,
            super::TaskType::Processing => StepType::Process,
            super::TaskType::Generation => StepType::Transform,
            super::TaskType::Orchestration => StepType::Initialize,
            super::TaskType::Monitoring => StepType::Validate,
            super::TaskType::Recovery => StepType::Cleanup,
            super::TaskType::Learning => StepType::Analyze,
        }
    }

    fn analyze_dependencies(&self, steps: &[ExecutionStep]) -> Result<Vec<PlanDependency>> {
        let mut dependencies = Vec::new();
        
        // Simple dependency analysis - in practice this would be more complex
        for (i, step) in steps.iter().enumerate() {
            if i > 0 {
                dependencies.push(PlanDependency {
                    dependent_step: step.step_id,
                    prerequisite_step: steps[i-1].step_id,
                    dependency_type: DependencyType::ControlFlow,
                });
            }
        }

        Ok(dependencies)
    }

    fn calculate_resource_requirements(
        &self,
        steps: &[ExecutionStep],
        _available_modules: &[ModuleInfo],
    ) -> Result<PlanResourceRequirements> {
        let total_duration: Duration = steps.iter()
            .map(|s| s.expected_duration)
            .sum();

        Ok(PlanResourceRequirements {
            total_cpu_cores: steps.len() as u32,
            total_memory_mb: steps.len() as u64 * 512, // Rough estimate
            total_disk_mb: steps.len() as u64 * 100,
            estimated_duration: total_duration,
            parallel_execution_slots: (steps.len() / 2).max(1) as u32,
        })
    }

    fn estimate_total_duration(&self, steps: &[ExecutionStep]) -> Result<Duration> {
        Ok(steps.iter().map(|s| s.expected_duration).sum())
    }

    fn generate_contingencies(&self, _steps: &[ExecutionStep]) -> Result<Vec<Contingency>> {
        // Simplified contingency generation
        Ok(vec![
            Contingency {
                trigger_condition: "module_failure".to_string(),
                alternative_steps: Vec::new(),
                fallback_plan: None,
            }
        ])
    }

    pub fn get_plan(&self, plan_id: &Uuid) -> Option<&ExecutionPlan> {
        self.active_plans.get(plan_id)
    }
} 