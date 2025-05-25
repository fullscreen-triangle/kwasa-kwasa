use std::fs;
use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use colored::Colorize;
use anyhow::{Result, Context};

mod cli;
mod turbulance;
mod text_unit;
mod orchestrator;
mod knowledge;
mod utils;
mod wasm;
mod genomic;
mod spectrometry;
mod chemistry;
mod pattern;

use cli::{CliConfig, CliCommands};

#[derive(Parser)]
#[command(name = "kwasa-kwasa")]
#[command(author = "Kundai")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "A metacognitive text processing framework with Turbulance syntax", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a Turbulance script
    Run {
        /// Path to the script file
        #[arg(value_name = "SCRIPT")]
        script: PathBuf,
        
        /// Enable debug mode
        #[arg(short, long)]
        debug: bool,
        
        /// Set verbosity level (0-3)
        #[arg(short, long, default_value = "1")]
        verbose: u8,
    },
    
    /// Validate a Turbulance script
    Validate {
        /// Path to the script file
        #[arg(value_name = "SCRIPT")]
        script: PathBuf,
    },
    
    /// Process a document with embedded Turbulance functions
    Process {
        /// Path to the document file
        #[arg(value_name = "DOCUMENT")]
        document: PathBuf,
        
        /// Enable interactive mode
        #[arg(short, long)]
        interactive: bool,
        
        /// Output format (text, json, yaml, html)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    
    /// Start an interactive REPL for Turbulance
    Repl {
        /// Load initial script
        #[arg(short, long)]
        load: Option<PathBuf>,
        
        /// Enable debug mode
        #[arg(short, long)]
        debug: bool,
    },
    
    /// Initialize a new Kwasa-Kwasa project
    Init {
        /// Project name
        #[arg(value_name = "NAME")]
        name: String,
        
        /// Project template (default, research, analysis, nlp)
        #[arg(short, long, default_value = "default")]
        template: String,
    },
    
    /// Show project information
    Info {
        /// Project path (current directory if not specified)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,
    },
    
    /// Analyze project complexity and dependencies
    Analyze {
        /// Project path (current directory if not specified)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,
    },
    
    /// Format Turbulance code files
    Format {
        /// Path to file or directory
        #[arg(value_name = "PATH")]
        path: PathBuf,
        
        /// Check formatting without making changes
        #[arg(short, long)]
        check: bool,
    },
    
    /// Generate documentation
    Docs {
        /// Output format (markdown, html)
        #[arg(short, long, default_value = "markdown")]
        format: String,
        
        /// Project path (current directory if not specified)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,
    },
    
    /// Run project tests
    Test {
        /// Filter tests by name
        #[arg(short, long)]
        filter: Option<String>,
        
        /// Project path (current directory if not specified)
        #[arg(value_name = "PATH")]
        path: Option<PathBuf>,
    },
    
    /// Show or modify configuration
    Config {
        /// Configuration subcommand
        #[command(subcommand)]
        action: ConfigAction,
    },
    
    /// Run benchmark tests
    Bench {
        /// Benchmark filter
        #[arg(short, long)]
        filter: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    
    /// Set a configuration value
    Set {
        /// Configuration key
        key: String,
        /// Configuration value
        value: String,
    },
    
    /// Get a configuration value
    Get {
        /// Configuration key
        key: String,
    },
    
    /// Reset configuration to defaults
    Reset,
}

fn main() -> Result<()> {
    // Initialize environment
    dotenv::dotenv().ok();
    env_logger::init();
    
    let cli = Cli::parse();
    
    // Load configuration
    let config = CliConfig::load().unwrap_or_else(|_| {
        eprintln!("{}", "Warning: Could not load config, using defaults".yellow());
        CliConfig::default()
    });
    
    let commands = CliCommands::new(config.clone());
    
    match cli.command {
        Commands::Run { script, debug, verbose } => {
            run_script(&script, debug, verbose)?;
        },
        Commands::Validate { script } => {
            validate_script(&script)?;
        },
        Commands::Process { document, interactive, format } => {
            process_document(&document, interactive, &format)?;
        },
        Commands::Repl { load, debug } => {
            start_repl(load, debug)?;
        },
        Commands::Init { name, template } => {
            let template_opt = if template == "default" { None } else { Some(template.as_str()) };
            commands.init_project(&name, template_opt)?;
        },
        Commands::Info { path } => {
            commands.project_info(path.as_deref())?;
        },
        Commands::Analyze { path } => {
            commands.analyze_project(path.as_deref())?;
        },
        Commands::Format { path, check } => {
            commands.format_code(&path, check)?;
        },
        Commands::Docs { format, path } => {
            commands.generate_docs(path.as_deref(), &format)?;
        },
        Commands::Test { filter, path } => {
            commands.run_tests(path.as_deref(), filter.as_deref())?;
        },
        Commands::Config { action } => {
            handle_config_command(action, config)?;
        },
        Commands::Bench { filter } => {
            run_benchmarks(filter.as_deref())?;
        },
    }
    
    Ok(())
}

fn run_script(script_path: &PathBuf, debug: bool, verbose: u8) -> Result<()> {
    if verbose > 0 {
        println!("{} {}", "Running script:".green().bold(), script_path.display());
    }
    
    let source = fs::read_to_string(script_path)
        .with_context(|| format!("Failed to read script file: {}", script_path.display()))?;
    
    if debug {
        println!("{}", "Debug mode enabled".blue());
        println!("{}", "Script content:".blue());
        println!("{}", source.dimmed());
        println!();
    }
    
    match turbulance::run(&source) {
        Ok(_) => {
            if verbose > 0 {
                println!("{}", "Script executed successfully".green());
            }
            Ok(())
        },
        Err(err) => {
            eprintln!("{} {}", "Error:".red().bold(), err);
            process::exit(1);
        }
    }
}

fn validate_script(script_path: &PathBuf) -> Result<()> {
    println!("{} {}", "Validating script:".green().bold(), script_path.display());
    
    let source = fs::read_to_string(script_path)
        .with_context(|| format!("Failed to read script file: {}", script_path.display()))?;
    
    match turbulance::validate(&source) {
        Ok(valid) => {
            if valid {
                println!("{}", "Script is valid".green());
            } else {
                println!("{}", "Script contains errors".red());
                process::exit(1);
            }
            Ok(())
        },
        Err(err) => {
            eprintln!("{} {}", "Error:".red().bold(), err);
            process::exit(1);
        }
    }
}

fn process_document(document_path: &PathBuf, interactive: bool, format: &str) -> Result<()> {
    println!("{} {}", "Processing document:".green().bold(), document_path.display());
    
    if interactive {
        println!("{}", "Interactive mode enabled".blue());
    }
    
    println!("{} {}", "Output format:".blue(), format);
    
    // Load document content
    let content = fs::read_to_string(document_path)
        .with_context(|| format!("Failed to read document: {}", document_path.display()))?;
    
    // Process with text_unit operations
    println!("{}", "Analyzing document structure...".blue());
    
    // Basic text analysis
    let word_count = content.split_whitespace().count();
    let sentence_count = content.matches('.').count() + content.matches('!').count() + content.matches('?').count();
    let paragraph_count = content.split("\n\n").count();
    
    let analysis = serde_json::json!({
        "file": document_path.display().to_string(),
        "stats": {
            "words": word_count,
            "sentences": sentence_count,
            "paragraphs": paragraph_count,
            "characters": content.len()
        },
        "readability": {
            "avg_words_per_sentence": if sentence_count > 0 { word_count as f64 / sentence_count as f64 } else { 0.0 },
            "avg_sentences_per_paragraph": if paragraph_count > 0 { sentence_count as f64 / paragraph_count as f64 } else { 0.0 }
        }
    });
    
    match format {
        "json" => println!("{}", serde_json::to_string_pretty(&analysis)?),
        "yaml" => println!("{}", serde_yaml::to_string(&analysis)?),
        "text" => {
            println!("Document Analysis Results:");
            println!("  Words: {}", word_count);
            println!("  Sentences: {}", sentence_count);
            println!("  Paragraphs: {}", paragraph_count);
            println!("  Characters: {}", content.len());
            println!("  Avg words per sentence: {:.1}", analysis["readability"]["avg_words_per_sentence"]);
        },
        _ => return Err(anyhow::anyhow!("Unsupported output format: {}", format)),
    }
    
    Ok(())
}

fn start_repl(load_script: Option<PathBuf>, debug: bool) -> Result<()> {
    let mut repl = cli::Repl::new().with_context(|| "Failed to initialize REPL")?;
    
    if let Some(script_path) = load_script {
        println!("{} {}", "Loading script:".blue(), script_path.display());
        // Load script content into REPL
        let content = fs::read_to_string(&script_path)?;
        // Execute the loaded script
        if let Err(e) = turbulance::run(&content) {
            eprintln!("{} {}", "Error loading script:".red(), e);
        }
    }
    
    if debug {
        println!("{}", "Debug mode enabled for REPL".blue());
    }
    
    repl.start().with_context(|| "Error during REPL execution")?;
    Ok(())
}

fn handle_config_command(action: ConfigAction, mut config: CliConfig) -> Result<()> {
    match action {
        ConfigAction::Show => {
            let commands = CliCommands::new(config);
            commands.show_config()?;
        },
        ConfigAction::Set { key, value } => {
            // Parse key to update the appropriate config section
            let parts: Vec<&str> = key.split('.').collect();
            match parts.as_slice() {
                ["repl", "prompt"] => config.repl.prompt = value,
                ["repl", "syntax_highlighting"] => config.repl.syntax_highlighting = value.parse()?,
                ["repl", "auto_completion"] => config.repl.auto_completion = value.parse()?,
                ["repl", "history_size"] => config.repl.history_size = value.parse()?,
                ["output", "colored"] => config.output.colored = value.parse()?,
                ["output", "verbosity"] => config.output.verbosity = value.parse()?,
                ["editor", "command"] => config.editor.editor_command = value,
                ["editor", "tab_width"] => config.editor.tab_width = value.parse()?,
                ["performance", "parallel_processing"] => config.performance.parallel_processing = value.parse()?,
                ["performance", "thread_count"] => config.performance.thread_count = value.parse()?,
                ["performance", "timeout"] => config.performance.timeout = value.parse()?,
                _ => {
                    // Store as custom setting
                    config.set_custom(key, value);
                }
            }
            config.save()?;
            println!("{} Configuration updated", "✓".green());
        },
        ConfigAction::Get { key } => {
            let parts: Vec<&str> = key.split('.').collect();
            let value = match parts.as_slice() {
                ["repl", "prompt"] => config.repl.prompt.clone(),
                ["repl", "syntax_highlighting"] => config.repl.syntax_highlighting.to_string(),
                ["repl", "auto_completion"] => config.repl.auto_completion.to_string(),
                ["repl", "history_size"] => config.repl.history_size.to_string(),
                ["output", "colored"] => config.output.colored.to_string(),
                ["output", "verbosity"] => config.output.verbosity.to_string(),
                ["editor", "command"] => config.editor.editor_command.clone(),
                ["editor", "tab_width"] => config.editor.tab_width.to_string(),
                ["performance", "parallel_processing"] => config.performance.parallel_processing.to_string(),
                ["performance", "thread_count"] => config.performance.thread_count.to_string(),
                ["performance", "timeout"] => config.performance.timeout.to_string(),
                _ => config.get_custom(&key).cloned().unwrap_or_else(|| "Not found".to_string()),
            };
            println!("{}: {}", key, value);
        },
        ConfigAction::Reset => {
            let default_config = CliConfig::default();
            default_config.save()?;
            println!("{} Configuration reset to defaults", "✓".green());
        },
    }
    Ok(())
}

fn run_benchmarks(filter: Option<&str>) -> Result<()> {
    println!("{}", "Running benchmarks...".blue());
    
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("bench");
    
    if let Some(filter) = filter {
        cmd.arg("--").arg(filter);
    }
    
    let status = cmd.status()?;
    
    if !status.success() {
        return Err(anyhow::anyhow!("Benchmark execution failed"));
    }
    
    Ok(())
}
