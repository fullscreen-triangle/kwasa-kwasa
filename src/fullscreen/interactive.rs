use std::collections::HashMap;
use anyhow::Result;
use uuid::Uuid;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveSession {
    pub session_id: Uuid,
    pub user_id: String,
    pub active_nodes: Vec<Uuid>,
    pub zoom_level: f64,
    pub center_position: (f64, f64),
    pub filters: Vec<InteractiveFilter>,
    pub highlights: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveFilter {
    pub filter_type: FilterType,
    pub value: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterType {
    ModuleType,
    DependencyLevel,
    SemanticWeight,
    ComplexityScore,
    Custom,
}

pub struct InteractiveManager {
    sessions: HashMap<Uuid, InteractiveSession>,
}

impl InteractiveManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    pub fn create_session(&mut self, user_id: String) -> Uuid {
        let session_id = Uuid::new_v4();
        let session = InteractiveSession {
            session_id,
            user_id,
            active_nodes: Vec::new(),
            zoom_level: 1.0,
            center_position: (0.0, 0.0),
            filters: Vec::new(),
            highlights: Vec::new(),
        };
        
        self.sessions.insert(session_id, session);
        session_id
    }

    pub fn get_session(&self, session_id: &Uuid) -> Option<&InteractiveSession> {
        self.sessions.get(session_id)
    }

    pub fn update_zoom(&mut self, session_id: &Uuid, zoom_level: f64) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.zoom_level = zoom_level;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found"))
        }
    }

    pub fn update_center(&mut self, session_id: &Uuid, position: (f64, f64)) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.center_position = position;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found"))
        }
    }

    pub fn add_filter(&mut self, session_id: &Uuid, filter: InteractiveFilter) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.filters.push(filter);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found"))
        }
    }

    pub fn highlight_nodes(&mut self, session_id: &Uuid, node_ids: Vec<Uuid>) -> Result<()> {
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.highlights = node_ids;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Session not found"))
        }
    }
} 