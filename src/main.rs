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
    },
    
    /// Start an interactive REPL for Turbulance
    Repl,
}

fn main() -> Result<()> {
    // Initialize environment
    dotenv::dotenv().ok();
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run { script } => {
            run_script(&script)?;
        },
        Commands::Validate { script } => {
            validate_script(&script)?;
        },
        Commands::Process { document, interactive } => {
            process_document(&document, interactive)?;
        },
        Commands::Repl => {
            start_repl()?;
        },
    }
    
    Ok(())
}

fn run_script(script_path: &PathBuf) -> Result<()> {
    println!("{} {}", "Running script:".green().bold(), script_path.display());
    
    let source = fs::read_to_string(script_path)
        .with_context(|| format!("Failed to read script file: {}", script_path.display()))?;
    
    match turbulance::run(&source) {
        Ok(_) => {
            println!("{}", "Script executed successfully".green());
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

fn process_document(document_path: &PathBuf, interactive: bool) -> Result<()> {
    println!("{} {}", "Processing document:".green().bold(), document_path.display());
    
    if interactive {
        println!("{}", "Interactive mode enabled".blue());
    }
    
    // This is a placeholder for now - we'll implement document processing
    // in a future version
    println!("{}", "Document processing not yet implemented".yellow());
    
    Ok(())
}

fn start_repl() -> Result<()> {
    let mut repl = cli::Repl::new().with_context(|| "Failed to initialize REPL")?;
    repl.start().with_context(|| "Error during REPL execution")?;
    Ok(())
}
