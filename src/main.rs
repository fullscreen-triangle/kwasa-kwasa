use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;

mod chemistry;
mod cli;
mod error;
mod genomic;
mod knowledge;
mod orchestrator;
mod pattern;
mod spectrometry;
mod text_unit;
mod turbulance;
mod utils;
mod wasm;

use cli::run::EnvironmentManager;
use error::KwasaError;
use turbulance::{ArgumentValidationReport, ScientificArgumentValidator};

fn main() {
    let cli = cli::Cli::parse();

    if let Err(e) = run_command(cli) {
        eprintln!("❌ Error: {}", e);
        process::exit(1);
    }
}

fn run_command(cli: cli::Cli) -> Result<(), KwasaError> {
    match cli.command {
        cli::Commands::Init(args) => {
            println!("🚀 Initializing kwasa-kwasa environment...");

            let env_manager = EnvironmentManager::initialize(&args.path)?;

            if !args.yes {
                println!("📋 Environment initialized at: {}", args.path.display());
                println!("💡 Run 'kwasa env setup' to install dependencies");
            }

            if !args.minimal {
                println!("🔧 Setting up complete environment...");
                env_manager.setup_environment()?;
            }

            println!("✅ Kwasa-kwasa environment ready!");
            println!("📖 Run 'kwasa project new <name>' to create your first project");
        }

        cli::Commands::Env(env_cmd) => {
            handle_env_command(env_cmd)?;
        }

        cli::Commands::Project(proj_cmd) => {
            handle_project_command(proj_cmd)?;
        }

        cli::Commands::Run(args) => {
            // Load and parse the Turbulance script
            let source = std::fs::read_to_string(&args.file).map_err(|e| {
                KwasaError::IoError(format!(
                    "Failed to read file {}: {}",
                    args.file.display(),
                    e
                ))
            })?;

            let mut lexer = turbulance::lexer::Lexer::new(&source);
            let tokens = lexer
                .tokenize()
                .map_err(|e| KwasaError::ParseError(format!("Lexer error: {:?}", e)))?;

            let mut parser = turbulance::parser::Parser::new(tokens);
            let ast = parser
                .parse()
                .map_err(|e| KwasaError::ParseError(format!("Parser error: {:?}", e)))?;

            // Perform scientific validation if requested
            if args.validate {
                println!("🔬 Performing scientific argument validation...");
                let mut validator = ScientificArgumentValidator::new();
                let report = validator.validate_argument(&ast)?;

                // Print validation report
                report.print_report();

                // Save report if requested
                if let Some(report_file) = &args.report_file {
                    save_validation_report(&report, report_file, &args.output)?;
                }

                // Check if we should continue based on strictness
                let strictness = cli::commands::ValidationStrictness::from_str(&args.strictness)
                    .map_err(|e| KwasaError::ConfigError(e))?;

                if should_stop_execution(&report, &strictness) {
                    println!("🚫 Execution stopped due to validation issues");
                    return Ok(());
                }
            }

            // Execute the script
            println!("🚀 Executing Turbulance script...");
            let mut interpreter = turbulance::interpreter::Interpreter::new();
            interpreter
                .execute(&ast)
                .map_err(|e| KwasaError::RuntimeError(format!("Execution error: {:?}", e)))?;

            println!("✅ Script execution completed");
        }

        cli::Commands::Validate(args) => {
            // Load and parse the Turbulance script
            let source = std::fs::read_to_string(&args.file).map_err(|e| {
                KwasaError::IoError(format!(
                    "Failed to read file {}: {}",
                    args.file.display(),
                    e
                ))
            })?;

            let mut lexer = turbulance::lexer::Lexer::new(&source);
            let tokens = lexer
                .tokenize()
                .map_err(|e| KwasaError::ParseError(format!("Lexer error: {:?}", e)))?;

            let mut parser = turbulance::parser::Parser::new(tokens);
            let ast = parser
                .parse()
                .map_err(|e| KwasaError::ParseError(format!("Parser error: {:?}", e)))?;

            // Perform scientific validation
            println!("🔬 Validating scientific arguments...");
            let mut validator = ScientificArgumentValidator::new();
            let report = validator.validate_argument(&ast)?;

            // Print validation report
            report.print_report();

            // Save report if requested
            if let Some(report_file) = &args.report_file {
                save_validation_report(&report, report_file, &args.output)?;
            }

            // Generate recommendations if requested
            if args.recommend {
                println!("\n🎯 Specific Recommendations:");
                for (i, rec) in report.recommendations.iter().enumerate() {
                    println!("  {}. {}", i + 1, rec);
                }
            }
        }

        cli::Commands::Repl(args) => {
            println!("🎮 Starting Turbulance REPL...");
            let mut repl = cli::repl::TurbulanceRepl::new(args.validate);

            if let Some(load_file) = &args.load {
                repl.load_file(load_file)?;
            }

            repl.run()?;
        }
    }

    Ok(())
}

fn handle_env_command(cmd: cli::EnvCommands) -> Result<(), KwasaError> {
    match cmd {
        cli::EnvCommands::Init { path, yes } => {
            println!("🔧 Initializing environment at: {}", path.display());
            let env_manager = EnvironmentManager::initialize(&path)?;

            if !yes {
                println!("✅ Environment initialized");
                println!("💡 Run 'kwasa env setup' to install dependencies");
            }
        }

        cli::EnvCommands::Status => {
            // Find the nearest kwasa environment
            let current_dir = env::current_dir().map_err(|e| {
                KwasaError::IoError(format!("Failed to get current directory: {}", e))
            })?;

            let env_manager = find_environment(&current_dir)?;
            let status = env_manager.status();
            status.print_status();
        }

        cli::EnvCommands::Setup { force } => {
            let current_dir = env::current_dir().map_err(|e| {
                KwasaError::IoError(format!("Failed to get current directory: {}", e))
            })?;

            let env_manager = find_environment(&current_dir)?;

            if force {
                println!("🔄 Force reinstalling environment...");
            }

            env_manager.setup_environment()?;
        }

        cli::EnvCommands::Update { component } => {
            println!("🔄 Updating environment...");
            if let Some(comp) = component {
                println!("  Updating component: {}", comp);
            } else {
                println!("  Updating all components");
            }
            // Implementation would go here
        }

        cli::EnvCommands::Install { tool, version } => {
            println!("📦 Installing tool: {}", tool);
            if let Some(ver) = version {
                println!("  Version: {}", ver);
            }
            // Implementation would go here
        }

        cli::EnvCommands::List => {
            println!("📋 Available scientific tools:");
            println!("  • jupyter - Jupyter notebooks");
            println!("  • rstudio - R development environment");
            println!("  • pymol - Molecular visualization");
            println!("  • vmd - Visual molecular dynamics");
            println!("  • chimera - Molecular modeling");
            // More tools would be listed here
        }

        cli::EnvCommands::Clean { deep } => {
            println!("🧹 Cleaning environment...");
            if deep {
                println!("  Deep cleaning (including dependencies)");
            }
            // Implementation would go here
        }
    }

    Ok(())
}

fn handle_project_command(cmd: cli::ProjectCommands) -> Result<(), KwasaError> {
    match cmd {
        cli::ProjectCommands::New {
            name,
            template,
            git,
        } => {
            println!("📂 Creating new project: {}", name);

            let template_type = cli::commands::ProjectTemplate::from_str(&template)
                .map_err(|e| KwasaError::ConfigError(e))?;

            println!("  Template: {} - {}", template, template_type.description());

            // Create project directory and files
            create_project(&name, &template_type, git)?;

            println!("✅ Project '{}' created successfully!", name);
            println!("📖 Navigate to projects/{} to get started", name);
        }

        cli::ProjectCommands::List => {
            println!("📋 Available projects:");
            list_projects()?;
        }

        cli::ProjectCommands::Info { name } => {
            println!("📊 Project information: {}", name);
            show_project_info(&name)?;
        }

        cli::ProjectCommands::Delete { name, yes } => {
            if !yes {
                print!(
                    "⚠️  Are you sure you want to delete project '{}'? [y/N]: ",
                    name
                );
                use std::io::{self, Write};
                io::stdout().flush().unwrap();

                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();

                if !input.trim().to_lowercase().starts_with('y') {
                    println!("❌ Deletion cancelled");
                    return Ok(());
                }
            }

            delete_project(&name)?;
            println!("✅ Project '{}' deleted", name);
        }
    }

    Ok(())
}

fn find_environment(start_path: &PathBuf) -> Result<EnvironmentManager, KwasaError> {
    let mut current_path = start_path.clone();

    loop {
        let manifest_path = current_path.join("kwasa-environment.toml");
        if manifest_path.exists() {
            return EnvironmentManager::initialize(&current_path);
        }

        if let Some(parent) = current_path.parent() {
            current_path = parent.to_path_buf();
        } else {
            break;
        }
    }

    Err(KwasaError::EnvironmentError(
        "No kwasa-kwasa environment found. Run 'kwasa init' to create one.".to_string(),
    ))
}

fn save_validation_report(
    report: &ArgumentValidationReport,
    file_path: &PathBuf,
    format: &str,
) -> Result<(), KwasaError> {
    let output_format =
        cli::commands::OutputFormat::from_str(format).map_err(|e| KwasaError::ConfigError(e))?;

    let content = match output_format {
        cli::commands::OutputFormat::Text => format_report_text(report),
        cli::commands::OutputFormat::Json => format_report_json(report)?,
        cli::commands::OutputFormat::Html => format_report_html(report),
    };

    std::fs::write(file_path, content)
        .map_err(|e| KwasaError::IoError(format!("Failed to write report: {}", e)))?;

    println!("📄 Validation report saved to: {}", file_path.display());
    Ok(())
}

fn should_stop_execution(
    report: &ArgumentValidationReport,
    strictness: &cli::commands::ValidationStrictness,
) -> bool {
    use turbulance::{FallacySeverity, OverallValidity};

    match strictness {
        cli::commands::ValidationStrictness::Critical => {
            matches!(report.overall_validity, OverallValidity::Invalid)
        }
        cli::commands::ValidationStrictness::Error => {
            matches!(
                report.overall_validity,
                OverallValidity::Invalid | OverallValidity::RequiresRevision
            )
        }
        cli::commands::ValidationStrictness::Warning => {
            !matches!(report.overall_validity, OverallValidity::Valid)
        }
    }
}

fn format_report_text(report: &ArgumentValidationReport) -> String {
    let mut output = String::new();
    output.push_str("Scientific Argument Validation Report\n");
    output.push_str("=====================================\n\n");

    output.push_str(&format!(
        "Overall Validity: {:?}\n\n",
        report.overall_validity
    ));

    if !report.logical_fallacies.is_empty() {
        output.push_str("Logical Issues:\n");
        for fallacy in &report.logical_fallacies {
            output.push_str(&format!(
                "  - {:?}: {} ({})\n",
                fallacy.severity, fallacy.description, fallacy.location
            ));
        }
        output.push('\n');
    }

    if !report.recommendations.is_empty() {
        output.push_str("Recommendations:\n");
        for rec in &report.recommendations {
            output.push_str(&format!("  • {}\n", rec));
        }
    }

    output
}

fn format_report_json(report: &ArgumentValidationReport) -> Result<String, KwasaError> {
    serde_json::to_string_pretty(report)
        .map_err(|e| KwasaError::ConfigError(format!("Failed to serialize report: {}", e)))
}

fn format_report_html(report: &ArgumentValidationReport) -> String {
    format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>Scientific Argument Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .valid {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        .critical {{ color: darkred; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🔬 Scientific Argument Validation Report</h1>
    <h2>Overall Validity: {:?}</h2>

    <h3>Logical Issues</h3>
    <ul>
    {}
    </ul>

    <h3>Recommendations</h3>
    <ul>
    {}
    </ul>
</body>
</html>
"#,
        report.overall_validity,
        report
            .logical_fallacies
            .iter()
            .map(|f| format!(
                "<li class=\"{:?}\">{}: {} ({})</li>",
                f.severity, f.severity, f.description, f.location
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        report
            .recommendations
            .iter()
            .map(|r| format!("<li>{}</li>", r))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

fn create_project(
    name: &str,
    template: &cli::commands::ProjectTemplate,
    with_git: bool,
) -> Result<(), KwasaError> {
    let projects_dir = PathBuf::from("projects");
    let project_dir = projects_dir.join(name);

    std::fs::create_dir_all(&project_dir)
        .map_err(|e| KwasaError::IoError(format!("Failed to create project directory: {}", e)))?;

    // Create basic project structure
    std::fs::create_dir_all(project_dir.join("src"))?;
    std::fs::create_dir_all(project_dir.join("data"))?;
    std::fs::create_dir_all(project_dir.join("results"))?;
    std::fs::create_dir_all(project_dir.join("docs"))?;

    // Create main.turb based on template
    let main_content = match template {
        cli::commands::ProjectTemplate::Basic => include_str!("../templates/default_main.turb"),
        cli::commands::ProjectTemplate::Chemistry => {
            include_str!("../templates/analysis_main.turb")
        }
        cli::commands::ProjectTemplate::Research => include_str!("../templates/research_main.turb"),
        _ => include_str!("../templates/default_main.turb"),
    };

    std::fs::write(project_dir.join("src").join("main.turb"), main_content)
        .map_err(|e| KwasaError::IoError(format!("Failed to create main.turb: {}", e)))?;

    // Create README
    let readme_content = format!(
        r#"# {}

{} project created with kwasa-kwasa.

## Getting Started

1. Navigate to the `src/` directory
2. Edit `main.turb` with your analysis
3. Run: `kwasa run src/main.turb --validate`

## Project Structure

- `src/` - Turbulance source files
- `data/` - Input data files
- `results/` - Analysis outputs
- `docs/` - Documentation

## Validation

This project includes scientific argument validation. Run:

```bash
kwasa validate src/main.turb --recommend
```
"#,
        name,
        template.description()
    );

    std::fs::write(project_dir.join("README.md"), readme_content)
        .map_err(|e| KwasaError::IoError(format!("Failed to create README: {}", e)))?;

    // Initialize git if requested
    if with_git {
        use std::process::Command;
        Command::new("git")
            .args(&["init"])
            .current_dir(&project_dir)
            .output()
            .map_err(|e| {
                KwasaError::EnvironmentError(format!("Failed to initialize git: {}", e))
            })?;
    }

    Ok(())
}

fn list_projects() -> Result<(), KwasaError> {
    let projects_dir = PathBuf::from("projects");

    if !projects_dir.exists() {
        println!("  No projects found. Create one with 'kwasa project new <name>'");
        return Ok(());
    }

    let entries = std::fs::read_dir(&projects_dir)
        .map_err(|e| KwasaError::IoError(format!("Failed to read projects directory: {}", e)))?;

    for entry in entries {
        let entry = entry
            .map_err(|e| KwasaError::IoError(format!("Failed to read directory entry: {}", e)))?;
        if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            println!("  📂 {}", entry.file_name().to_string_lossy());
        }
    }

    Ok(())
}

fn show_project_info(name: &str) -> Result<(), KwasaError> {
    let project_dir = PathBuf::from("projects").join(name);

    if !project_dir.exists() {
        return Err(KwasaError::ConfigError(format!(
            "Project '{}' not found",
            name
        )));
    }

    println!("  📍 Location: {}", project_dir.display());

    // Check for main.turb
    let main_file = project_dir.join("src").join("main.turb");
    if main_file.exists() {
        println!("  📄 Main file: src/main.turb");
    }

    // Check for README
    let readme_file = project_dir.join("README.md");
    if readme_file.exists() {
        println!("  📖 Documentation: README.md");
    }

    // Check for git
    let git_dir = project_dir.join(".git");
    if git_dir.exists() {
        println!("  🔄 Version control: Git");
    }

    Ok(())
}

fn delete_project(name: &str) -> Result<(), KwasaError> {
    let project_dir = PathBuf::from("projects").join(name);

    if !project_dir.exists() {
        return Err(KwasaError::ConfigError(format!(
            "Project '{}' not found",
            name
        )));
    }

    std::fs::remove_dir_all(&project_dir)
        .map_err(|e| KwasaError::IoError(format!("Failed to delete project: {}", e)))?;

    Ok(())
}
