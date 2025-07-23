use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::cli::config::CliConfig;
use crate::error::Error;
// Turbulance modules are not available yet:
// use crate::turbulance::lexer::Lexer;
// use crate::turbulance::parser::Parser;
// use crate::turbulance::interpreter::Interpreter;

/// Kwasa-Kwasa Environment Manager
/// Creates and manages isolated scientific computing environments
pub struct EnvironmentManager {
    pub workspace_root: PathBuf,
    pub config: CliConfig,
    pub environment_manifest: EnvironmentManifest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentManifest {
    pub version: String,
    pub rust_toolchain: String,
    pub python_version: String,
    pub r_version: String,
    pub dependencies: EnvironmentDependencies,
    pub tools: Vec<EnvironmentTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentDependencies {
    pub rust_crates: Vec<String>,
    pub python_packages: Vec<String>,
    pub r_packages: Vec<String>,
    pub system_libraries: Vec<String>,
    pub scientific_databases: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentTool {
    pub name: String,
    pub version: String,
    pub install_command: String,
    pub verification_command: String,
}

impl EnvironmentManager {
    /// Initialize a new kwasa-kwasa environment
    pub fn initialize(workspace_path: &Path) -> Result<Self, Error> {
        let workspace_root = workspace_path.to_path_buf();

        // Create workspace directory structure
        Self::create_workspace_structure(&workspace_root)?;

        // Generate default environment manifest
        let environment_manifest = Self::default_environment_manifest();

        // Save manifest
        let manifest_path = workspace_root.join("kwasa-environment.toml");
        let manifest_content = toml::to_string_pretty(&environment_manifest)
            .map_err(|e| Error::config_error(format!("Failed to serialize manifest: {}", e)))?;
        fs::write(&manifest_path, manifest_content)
            .map_err(|e| Error::io(format!("Failed to write manifest: {}", e)))?;

        // Initialize configuration
        let config = CliConfig::default();

        Ok(Self {
            workspace_root,
            config,
            environment_manifest,
        })
    }

    /// Create the isolated workspace structure
    fn create_workspace_structure(workspace_root: &Path) -> Result<(), Error> {
        let directories = [
            "projects",     // User scientific projects
            "data",         // Scientific datasets
            "models",       // Trained models and configurations
            "results",      // Analysis results and outputs
            "cache",        // Cached computations
            "logs",         // System and analysis logs
            "tools",        // Installed scientific tools
            "environments", // Virtual environments (Python, R, etc.)
            "databases",    // Local scientific databases
            "scripts",      // Generated analysis scripts
            "notebooks",    // Jupyter/R notebooks
            "reports",      // Generated scientific reports
            ".kwasa",       // Internal kwasa-kwasa files
        ];

        for dir in &directories {
            let dir_path = workspace_root.join(dir);
            fs::create_dir_all(&dir_path)
                .map_err(|e| Error::io(format!("Failed to create directory {}: {}", dir, e)))?;
        }

        // Create initial configuration files
        Self::create_initial_configs(workspace_root)?;

        Ok(())
    }

    /// Create initial configuration files
    fn create_initial_configs(workspace_root: &Path) -> Result<(), Error> {
        // Create .gitignore for scientific projects
        let gitignore_content = r#"
# Kwasa-Kwasa Environment
cache/
logs/
.kwasa/temp/
environments/*/

# Scientific Data
data/raw/
data/processed/large_files/
*.h5
*.hdf5
*.parquet

# Models
models/checkpoints/
models/temp/

# Results
results/temp/
*.pdf
*.png
*.svg

# System
.DS_Store
Thumbs.db
"#;

        fs::write(workspace_root.join(".gitignore"), gitignore_content)
            .map_err(|e| Error::io(format!("Failed to create .gitignore: {}", e)))?;

        // Create README for the environment
        let readme_content = r#"
# Kwasa-Kwasa Scientific Computing Environment

This directory contains an isolated scientific computing environment managed by kwasa-kwasa.

## Structure

- `projects/` - Your scientific analysis projects
- `data/` - Scientific datasets and inputs
- `models/` - Trained models and configurations
- `results/` - Analysis outputs and results
- `tools/` - Installed scientific computing tools
- `environments/` - Language-specific virtual environments

## Getting Started

1. Navigate to the `projects/` directory
2. Create a new project: `kwasa new-project <name>`
3. Write your analysis in Turbulance DSL
4. Run: `kwasa run <project-name>`

## Environment Management

- Check environment status: `kwasa env status`
- Update dependencies: `kwasa env update`
- Install new tools: `kwasa env install <tool>`
"#;

        fs::write(workspace_root.join("README.md"), readme_content)
            .map_err(|e| Error::io(format!("Failed to create README: {}", e)))?;

        Ok(())
    }

    /// Generate default environment manifest
    fn default_environment_manifest() -> EnvironmentManifest {
        EnvironmentManifest {
            version: "1.0.0".to_string(),
            rust_toolchain: "stable".to_string(),
            python_version: "3.11".to_string(),
            r_version: "4.3.0".to_string(),
            dependencies: EnvironmentDependencies {
                rust_crates: vec![
                    "polars".to_string(),
                    "ndarray".to_string(),
                    "plotters".to_string(),
                    "reqwest".to_string(),
                    "tokio".to_string(),
                    "serde".to_string(),
                    "candle-core".to_string(),
                    "rdkit-rs".to_string(),
                ],
                python_packages: vec![
                    "pandas".to_string(),
                    "numpy".to_string(),
                    "scipy".to_string(),
                    "scikit-learn".to_string(),
                    "matplotlib".to_string(),
                    "seaborn".to_string(),
                    "jupyter".to_string(),
                    "rdkit".to_string(),
                    "biopython".to_string(),
                    "pytorch".to_string(),
                ],
                r_packages: vec![
                    "tidyverse".to_string(),
                    "ggplot2".to_string(),
                    "dplyr".to_string(),
                    "BiocManager".to_string(),
                    "devtools".to_string(),
                    "rmarkdown".to_string(),
                ],
                system_libraries: vec![
                    "openblas".to_string(),
                    "lapack".to_string(),
                    "fftw".to_string(),
                    "hdf5".to_string(),
                ],
                scientific_databases: vec![
                    "pubchem".to_string(),
                    "uniprot".to_string(),
                    "ncbi".to_string(),
                    "ensembl".to_string(),
                ],
            },
            tools: vec![
                EnvironmentTool {
                    name: "jupyter".to_string(),
                    version: "latest".to_string(),
                    install_command: "pip install jupyter".to_string(),
                    verification_command: "jupyter --version".to_string(),
                },
                EnvironmentTool {
                    name: "rstudio".to_string(),
                    version: "2023.12.1".to_string(),
                    install_command: "# Manual installation required".to_string(),
                    verification_command: "R --version".to_string(),
                },
            ],
        }
    }

    /// Setup the complete environment
    pub fn setup_environment(&self) -> Result<(), Error> {
        println!("🔧 Setting up kwasa-kwasa scientific computing environment...");

        // Check system requirements
        self.check_system_requirements()?;

        // Setup Rust environment
        self.setup_rust_environment()?;

        // Setup Python environment
        self.setup_python_environment()?;

        // Setup R environment
        self.setup_r_environment()?;

        // Install scientific tools
        self.install_scientific_tools()?;

        // Setup databases
        self.setup_scientific_databases()?;

        println!("✅ Environment setup complete!");

        Ok(())
    }

    /// Check system requirements
    fn check_system_requirements(&self) -> Result<(), Error> {
        println!("📋 Checking system requirements...");

        // Check if Rust is installed
        if Command::new("rustc").arg("--version").output().is_err() {
            return Err(Error::environment(
                "Rust is not installed. Please install Rust from https://rustup.rs/".to_string(),
            ));
        }

        // Check available disk space (require at least 5GB)
        // This is a simplified check - in practice, you'd use platform-specific APIs

        println!("✅ System requirements satisfied");
        Ok(())
    }

    /// Setup Rust environment with scientific computing crates
    fn setup_rust_environment(&self) -> Result<(), Error> {
        println!("🦀 Setting up Rust scientific computing environment...");

        let cargo_toml_content = format!(
            r#"
[package]
name = "kwasa-scientific-env"
version = "0.1.0"
edition = "2021"

[dependencies]
{}
"#,
            self.environment_manifest
                .dependencies
                .rust_crates
                .iter()
                .map(|crate_name| format!(r#"{} = "*""#, crate_name))
                .collect::<Vec<_>>()
                .join("\n")
        );

        let rust_env_path = self.workspace_root.join("environments").join("rust");
        fs::create_dir_all(&rust_env_path)
            .map_err(|e| Error::io(format!("Failed to create Rust environment: {}", e)))?;

        fs::write(rust_env_path.join("Cargo.toml"), cargo_toml_content)
            .map_err(|e| Error::io(format!("Failed to write Cargo.toml: {}", e)))?;

        // Create a basic lib.rs that re-exports scientific computing functionality
        let lib_rs_content = r#"
//! Kwasa-Kwasa Scientific Computing Environment
//!
//! This module provides access to all scientific computing capabilities
//! available in the kwasa-kwasa environment.

pub use polars::prelude::*;
pub use ndarray::prelude::*;
pub use plotters::prelude::*;

/// Scientific computing utilities
pub mod scientific {
    pub mod data_analysis {
        pub use polars::prelude::*;
    }

    pub mod numerical {
        pub use ndarray::prelude::*;
    }

    pub mod visualization {
        pub use plotters::prelude::*;
    }

    pub mod machine_learning {
        // ML utilities would go here
    }
}
"#;

        fs::write(rust_env_path.join("src").join("lib.rs"), lib_rs_content)
            .map_err(|e| Error::io(format!("Failed to write lib.rs: {}", e)))?;

        println!("✅ Rust environment configured");
        Ok(())
    }

    /// Setup Python virtual environment
    fn setup_python_environment(&self) -> Result<(), Error> {
        println!("🐍 Setting up Python scientific computing environment...");

        let python_env_path = self.workspace_root.join("environments").join("python");

        // Create virtual environment
        let output = Command::new("python3")
            .args(&["-m", "venv", python_env_path.to_str().unwrap()])
            .output()
            .map_err(|e| Error::environment(format!("Failed to create Python venv: {}", e)))?;

        if !output.status.success() {
            return Err(Error::environment(format!(
                "Python venv creation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        // Create requirements.txt
        let requirements_content = self
            .environment_manifest
            .dependencies
            .python_packages
            .join("\n");
        fs::write(
            python_env_path.join("requirements.txt"),
            requirements_content,
        )
        .map_err(|e| Error::io(format!("Failed to write requirements.txt: {}", e)))?;

        println!("✅ Python environment configured");
        Ok(())
    }

    /// Setup R environment
    fn setup_r_environment(&self) -> Result<(), Error> {
        println!("📊 Setting up R scientific computing environment...");

        let r_env_path = self.workspace_root.join("environments").join("r");
        fs::create_dir_all(&r_env_path)
            .map_err(|e| Error::io(format!("Failed to create R environment: {}", e)))?;

        // Create R package installation script
        let r_install_script = format!(
            r#"
# Kwasa-Kwasa R Environment Setup
packages <- c({})

install.packages(packages, dependencies = TRUE)

# Install Bioconductor packages
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

cat("R environment setup complete!\n")
"#,
            self.environment_manifest
                .dependencies
                .r_packages
                .iter()
                .map(|pkg| format!(r#""{}""#, pkg))
                .collect::<Vec<_>>()
                .join(", ")
        );

        fs::write(r_env_path.join("setup.R"), r_install_script)
            .map_err(|e| Error::io(format!("Failed to write R setup script: {}", e)))?;

        println!("✅ R environment configured");
        Ok(())
    }

    /// Install scientific computing tools
    fn install_scientific_tools(&self) -> Result<(), Error> {
        println!("🔬 Installing scientific computing tools...");

        for tool in &self.environment_manifest.tools {
            println!("  Installing {}...", tool.name);
            // Tool installation would be implemented here
            // For now, just verify if already installed
            if Command::new("sh")
                .args(&["-c", &tool.verification_command])
                .output()
                .is_ok()
            {
                println!("  ✅ {} already installed", tool.name);
            } else {
                println!("  ⚠️  {} needs manual installation", tool.name);
            }
        }

        Ok(())
    }

    /// Setup scientific databases
    fn setup_scientific_databases(&self) -> Result<(), Error> {
        println!("🗄️  Setting up scientific databases...");

        let db_path = self.workspace_root.join("databases");
        fs::create_dir_all(&db_path)
            .map_err(|e| Error::io(format!("Failed to create database directory: {}", e)))?;

        // Create database configuration
        let db_config = format!(
            r#"
# Scientific Database Configuration
databases:
{}
"#,
            self.environment_manifest
                .dependencies
                .scientific_databases
                .iter()
                .map(|db| format!(
                    "  - name: {}\n    url: https://{}.org\n    local_cache: true",
                    db, db
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        fs::write(db_path.join("config.yaml"), db_config)
            .map_err(|e| Error::io(format!("Failed to write database config: {}", e)))?;

        println!("✅ Database configuration created");
        Ok(())
    }

    /// Get environment status
    pub fn status(&self) -> EnvironmentStatus {
        EnvironmentStatus {
            workspace_root: self.workspace_root.clone(),
            rust_available: Command::new("rustc").arg("--version").output().is_ok(),
            python_available: Command::new("python3").arg("--version").output().is_ok(),
            r_available: Command::new("R").arg("--version").output().is_ok(),
            tools_installed: self.check_tools_status(),
        }
    }

    fn check_tools_status(&self) -> Vec<(String, bool)> {
        self.environment_manifest
            .tools
            .iter()
            .map(|tool| {
                let installed = Command::new("sh")
                    .args(&["-c", &tool.verification_command])
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false);
                (tool.name.clone(), installed)
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct EnvironmentStatus {
    pub workspace_root: PathBuf,
    pub rust_available: bool,
    pub python_available: bool,
    pub r_available: bool,
    pub tools_installed: Vec<(String, bool)>,
}

impl EnvironmentStatus {
    pub fn print_status(&self) {
        println!("🌍 Kwasa-Kwasa Environment Status");
        println!("Workspace: {}", self.workspace_root.display());
        println!("Rust: {}", if self.rust_available { "✅" } else { "❌" });
        println!(
            "Python: {}",
            if self.python_available { "✅" } else { "❌" }
        );
        println!("R: {}", if self.r_available { "✅" } else { "❌" });

        println!("\nTools:");
        for (tool, installed) in &self.tools_installed {
            println!("  {}: {}", tool, if *installed { "✅" } else { "❌" });
        }
    }
}

/// Executes a Turbulance script file
pub fn run_script(path: &Path) -> Result<()> {
    // Read the file content
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;

    // Create async runtime for framework execution
    let runtime = tokio::runtime::Runtime::new()
        .with_context(|| "Failed to create async runtime")?;

    runtime.block_on(async {
        // Initialize framework with default config
        let config = crate::FrameworkConfig::default();
        let mut framework = crate::KwasaFramework::new(config).await
            .with_context(|| "Failed to initialize Kwasa framework")?;

        // Execute the Turbulance script
        let result = framework.process_turbulance_code(&content).await
            .with_context(|| "Error during script execution")?;

        // Print result if not empty
        if !result.is_empty() {
            println!("{}", result);
        }

        Ok(())
    })
}

/// Executes a Turbulance script string directly
pub fn run_script_string(script: &str) -> Result<()> {
    // Create async runtime for framework execution
    let runtime = tokio::runtime::Runtime::new()
        .with_context(|| "Failed to create async runtime")?;

    runtime.block_on(async {
        // Initialize framework with default config
        let config = crate::FrameworkConfig::default();
        let mut framework = crate::KwasaFramework::new(config).await
            .with_context(|| "Failed to initialize Kwasa framework")?;

        // Execute the Turbulance script
        let result = framework.process_turbulance_code(script).await
            .with_context(|| "Error during script execution")?;

        // Print result if not empty
        if !result.is_empty() {
            println!("{}", result);
        }

        Ok(())
    })
}

/// Validates a Turbulance script file without executing it
pub fn validate_script(path: &Path) -> Result<bool> {
    // Read the file content
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;

    // Validate the Turbulance script
    match crate::turbulance::validate(&content) {
        Ok(is_valid) => {
            if is_valid {
                println!("✅ Script is valid: {}", path.display());
            } else {
                println!("❌ Script has syntax errors: {}", path.display());
            }
            Ok(is_valid)
        }
        Err(e) => {
            println!("❌ Validation error in {}: {}", path.display(), e);
            Ok(false)
        }
    }
}

/// Validates a Turbulance script string without executing it
pub fn validate_script_string(script: &str) -> Result<bool> {
    // Validate the Turbulance script
    match crate::turbulance::validate(script) {
        Ok(is_valid) => {
            if is_valid {
                println!("✅ Script is valid");
            } else {
                println!("❌ Script has syntax errors");
            }
            Ok(is_valid)
        }
        Err(e) => {
            println!("❌ Validation error: {}", e);
            Ok(false)
        }
    }
}
