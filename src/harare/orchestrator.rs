use std::collections::HashMap;
use std::time::SystemTime;
use anyhow::Result;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use super::{Task, TaskStatus, ExecutionRecord, GlobalContext};

pub struct CoreOrchestrator {
    pub id: Uuid,
    pub tasks: HashMap<Uuid, Task>,
    pub execution_records: Vec<ExecutionRecord>,
    pub context: GlobalContext,
}

impl CoreOrchestrator {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            tasks: HashMap::new(),
            execution_records: Vec::new(),
            context: GlobalContext {
                current_task: None,
                active_modules: Vec::new(),
                execution_history: std::collections::VecDeque::new(),
                environment_state: HashMap::new(),
                resource_usage: super::ResourceMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0,
                    disk_usage_mb: 0,
                    network_in_mbps: 0.0,
                    network_out_mbps: 0.0,
                    gpu_usage_percent: None,
                    timestamp: SystemTime::now(),
                },
                user_intent: None,
                semantic_workspace: super::SemanticWorkspace {
                    active_items: HashMap::new(),
                    relationships: Vec::new(),
                    context_stack: std::collections::VecDeque::new(),
                    attention_focus: None,
                },
            },
        }
    }

    pub async fn submit_task(&mut self, task: Task) -> Result<Uuid> {
        let task_id = task.id;
        self.tasks.insert(task_id, task);
        Ok(task_id)
    }

    pub async fn execute_task(&mut self, task_id: Uuid) -> Result<ExecutionRecord> {
        if let Some(task) = self.tasks.get_mut(&task_id) {
            task.status = TaskStatus::Executing;
            
            let start_time = SystemTime::now();
            
            // Simulate task execution
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            task.status = TaskStatus::Completed;
            
            let record = ExecutionRecord {
                task_id,
                module_sequence: vec!["harare".to_string()],
                start_time,
                end_time: Some(SystemTime::now()),
                status: TaskStatus::Completed,
                resource_usage: super::ResourceMetrics {
                    cpu_usage_percent: 10.0,
                    memory_usage_mb: 100,
                    disk_usage_mb: 10,
                    network_in_mbps: 1.0,
                    network_out_mbps: 1.0,
                    gpu_usage_percent: None,
                    timestamp: SystemTime::now(),
                },
                outputs: HashMap::new(),
                errors: Vec::new(),
            };
            
            self.execution_records.push(record.clone());
            Ok(record)
        } else {
            Err(anyhow::anyhow!("Task not found"))
        }
    }

    pub fn get_task_status(&self, task_id: &Uuid) -> Option<TaskStatus> {
        self.tasks.get(task_id).map(|task| task.status.clone())
    }

    pub fn get_execution_history(&self) -> &[ExecutionRecord] {
        &self.execution_records
    }
} 