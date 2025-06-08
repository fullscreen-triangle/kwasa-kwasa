pub mod boundary;
pub mod operations;
pub mod registry;
pub mod hierarchy;
pub mod utils;
pub mod transform;
pub mod advanced_processing;
pub mod types;

// Re-export common types for convenience
pub use boundary::BoundaryType;
pub use hierarchy::{HierarchyNode, HierarchyNodeType};
pub use transform::{TransformationPipeline, PipelineStage, TransformationMetrics};
pub use advanced_processing::{AdvancedTextProcessor, SemanticAnalysis, StyleAnalysis, ReadabilityMetrics};
pub use types::{TextUnit, TextUnitId, TextUnitType, Boundary};
pub use registry::{TextUnitRegistry, BoundaryDetectionOptions};
pub use boundary::{detect_paragraph_boundaries, detect_sentence_boundaries, detect_word_boundaries};

// All implementations are in their respective modules - no duplicates here
