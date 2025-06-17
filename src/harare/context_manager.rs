use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;
use anyhow::Result;
use super::{GlobalContext, SemanticWorkspace, SemanticContext, SemanticItem, SemanticRelationship};

pub struct ContextManager {
    pub contexts: HashMap<String, GlobalContext>,
    pub active_context: Option<String>,
    pub context_history: VecDeque<String>,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            active_context: None,
            context_history: VecDeque::new(),
        }
    }

    pub fn create_context(&mut self, name: String) -> Result<()> {
        let context = GlobalContext {
            current_task: None,
            active_modules: Vec::new(),
            execution_history: VecDeque::new(),
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
            semantic_workspace: SemanticWorkspace {
                active_items: HashMap::new(),
                relationships: Vec::new(),
                context_stack: VecDeque::new(),
                attention_focus: None,
            },
        };

        self.contexts.insert(name.clone(), context);
        Ok(())
    }

    pub fn switch_context(&mut self, name: &str) -> Result<()> {
        if self.contexts.contains_key(name) {
            if let Some(current) = &self.active_context {
                self.context_history.push_back(current.clone());
            }
            self.active_context = Some(name.to_string());
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found: {}", name))
        }
    }

    pub fn get_active_context(&self) -> Option<&GlobalContext> {
        self.active_context
            .as_ref()
            .and_then(|name| self.contexts.get(name))
    }

    pub fn get_active_context_mut(&mut self) -> Option<&mut GlobalContext> {
        let name = self.active_context.clone()?;
        self.contexts.get_mut(&name)
    }

    pub fn update_semantic_workspace(&mut self, updates: SemanticWorkspaceUpdate) -> Result<()> {
        if let Some(context) = self.get_active_context_mut() {
            match updates {
                SemanticWorkspaceUpdate::AddItem(item) => {
                    context.semantic_workspace.active_items.insert(item.id.clone(), item);
                }
                SemanticWorkspaceUpdate::RemoveItem(id) => {
                    context.semantic_workspace.active_items.remove(&id);
                }
                SemanticWorkspaceUpdate::AddRelationship(rel) => {
                    context.semantic_workspace.relationships.push(rel);
                }
                SemanticWorkspaceUpdate::SetFocus(focus) => {
                    context.semantic_workspace.attention_focus = Some(focus);
                }
                SemanticWorkspaceUpdate::PushContext(semantic_context) => {
                    context.semantic_workspace.context_stack.push_back(semantic_context);
                }
                SemanticWorkspaceUpdate::PopContext => {
                    context.semantic_workspace.context_stack.pop_back();
                }
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("No active context"))
        }
    }

    pub fn merge_contexts(&mut self, source: &str, target: &str) -> Result<()> {
        if let (Some(source_ctx), Some(target_ctx)) = 
            (self.contexts.get(source).cloned(), self.contexts.get_mut(target)) {
            
            // Merge semantic workspaces
            for (id, item) in source_ctx.semantic_workspace.active_items {
                target_ctx.semantic_workspace.active_items.insert(id, item);
            }
            
            target_ctx.semantic_workspace.relationships.extend(
                source_ctx.semantic_workspace.relationships
            );
            
            Ok(())
        } else {
            Err(anyhow::anyhow!("Source or target context not found"))
        }
    }
}

pub enum SemanticWorkspaceUpdate {
    AddItem(SemanticItem),
    RemoveItem(String),
    AddRelationship(SemanticRelationship),
    SetFocus(String),
    PushContext(SemanticContext),
    PopContext,
} 