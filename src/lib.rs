pub mod turbulance;
pub mod text_unit;
pub mod orchestrator;
pub mod knowledge;
pub mod cli;
pub mod utils;
pub mod genomic;
pub mod spectrometry;
pub mod chemistry;
pub mod pattern;
pub mod evidence;

// Re-export important modules
pub use turbulance::run;
pub use turbulance::validate;
pub use orchestrator::Orchestrator;
pub use text_unit::TextUnit;
pub use genomic::prelude::*;
pub use spectrometry::prelude::*;
pub use chemistry::prelude::*;
pub use pattern::prelude::*;

// Prelude for easy imports
pub mod prelude {
    pub use crate::turbulance::prelude::*;
    pub use crate::genomic::prelude::*;
    pub use crate::spectrometry::prelude::*;
    pub use crate::evidence::{
        EvidenceIntegration, ConflictReport, CriticalEvidence, 
        VisGraph, VisNode, VisEdge
    };
} 