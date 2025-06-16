pub mod text_analysis;
pub mod text_transform;
pub mod research;
pub mod utils;
pub mod cross_domain_analysis;
pub mod image_processing;

use crate::turbulance::interpreter::Value;
use crate::turbulance::Result;

/// StdLib represents the Turbulance Standard Library
pub struct StdLib {
    // Library state and configuration
}

impl StdLib {
    /// Create a new StdLib instance
    pub fn new() -> Self {
        StdLib {}
    }
    
    /// Initialize the standard library functions
    pub fn load_functions(&self) -> std::collections::HashMap<&'static str, crate::turbulance::interpreter::NativeFunction> {
        // This is now handled by the generated code in build.rs
        crate::turbulance::stdlib_functions()
    }
}

// Re-export the implementation modules to make them accessible
// from the generated code in the build script 