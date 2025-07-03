//! Templates module for Kwasa-Kwasa framework
//!
//! Provides template loading and management functionality

use std::path::Path;
use std::fs;
use anyhow::Result;

/// Available template types
#[derive(Debug, Clone)]
pub enum TemplateType {
    Default,
    Research,
    Analysis,
    NLP,
}

impl TemplateType {
    /// Get the template filename
    pub fn filename(&self) -> &'static str {
        match self {
            TemplateType::Default => "default_main.turb",
            TemplateType::Research => "research_main.turb",
            TemplateType::Analysis => "analysis_main.turb",
            TemplateType::NLP => "nlp_main.turb",
        }
    }

    /// Get template description
    pub fn description(&self) -> &'static str {
        match self {
            TemplateType::Default => "Basic Turbulance template",
            TemplateType::Research => "Research and data analysis template",
            TemplateType::Analysis => "Scientific analysis template",
            TemplateType::NLP => "Natural language processing template",
        }
    }
}

/// Load a template by type
pub fn load_template(template_type: TemplateType) -> Result<String> {
    let template_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/templates");
    let template_path = template_dir.join(template_type.filename());

    fs::read_to_string(template_path)
        .map_err(|e| anyhow::anyhow!("Failed to load template {}: {}", template_type.filename(), e))
}

/// Get all available templates
pub fn list_templates() -> Vec<TemplateType> {
    vec![
        TemplateType::Default,
        TemplateType::Research,
        TemplateType::Analysis,
        TemplateType::NLP,
    ]
}
