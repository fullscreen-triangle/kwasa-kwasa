//! Turbulance CLI - Universal Scientific Experiment DSL
//!
//! Command-line interface for the Turbulance language, enabling scientists to run
//! experiments, validate scripts, and interact with the language directly.

use clap::{Parser, Subcommand};
use colored::*;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use turbulance::{Engine, Script, Context, Result, TurbulanceError, Value};

#[derive(Parser)]
#[command(name = "turbulance")]
#[command(about = "Universal Scientific Experiment DSL")]
#[command(version = turbulance::VERSION)]
#[command(long_about = "
Turbulance is a domain-specific language for formalizing scientific methods and experiments.
It enables scientists to express experimental procedures, hypotheses, and data transformations
in a programmatic yet natural way while preserving semantic meaning.

Examples:
  turbulance run experiment.turb         # Run a Turbulance script
  turbulance validate analysis.turb      # Validate syntax without running
  turbulance repl                       # Start interactive mode
  turbulance new hypothesis            # Create a new experiment template
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a Turbulance script
    Run {
        /// Input file to run
        file: PathBuf,
        
        /// Additional context files to load
        #[arg(short, long)]
        context: Vec<PathBuf>,
        
        /// Output format (json, yaml, text)
        #[arg(short, long, default_value = "text")]
        output: String,
        
        /// Enable interactive mode after execution
        #[arg(short, long)]
        interactive: bool,
    },
    
    /// Validate Turbulance syntax without execution
    Validate {
        /// Files to validate
        files: Vec<PathBuf>,
        
        /// Check semantic validity
        #[arg(short, long)]
        semantic: bool,
        
        /// Output validation report
        #[arg(short, long)]
        report: bool,
    },
    
    /// Start interactive REPL mode
    Repl {
        /// Load context from file
        #[arg(short, long)]
        context: Option<PathBuf>,
        
        /// History file location
        #[arg(long)]
        history: Option<PathBuf>,
    },
    
    /// Create new experiment templates
    New {
        /// Template name
        name: String,
        
        /// Template type (hypothesis, experiment, analysis)
        #[arg(short, long, default_value = "experiment")]
        template_type: String,
        
        /// Output directory
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    
    /// Show information about Turbulance
    Info {
        /// Show detailed system information
        #[arg(short, long)]
        detailed: bool,
    },
    
    /// Format Turbulance code
    Format {
        /// Files to format
        files: Vec<PathBuf>,
        
        /// Format in place
        #[arg(short, long)]
        in_place: bool,
        
        /// Check if files are formatted
        #[arg(short, long)]
        check: bool,
    },
}

fn main() {
    let cli = Cli::parse();
    
    if cli.no_color {
        colored::control::set_override(false);
    }
    
    env_logger::Builder::new()
        .filter_level(if cli.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    if let Err(e) = run_command(cli) {
        eprintln!("{} {}", "Error:".red().bold(), e.user_message());
        if cli.verbose {
            eprintln!("\n{} {:?}", "Debug info:".yellow(), e);
        }
        std::process::exit(1);
    }
}

fn run_command(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run { file, context, output, interactive } => {
            run_file(&file, &context, &output, interactive)
        }
        Commands::Validate { files, semantic, report } => {
            validate_files(&files, semantic, report)
        }
        Commands::Repl { context, history } => {
            start_repl(context, history)
        }
        Commands::New { name, template_type, output } => {
            create_template(&name, &template_type, &output)
        }
        Commands::Info { detailed } => {
            show_info(detailed)
        }
        Commands::Format { files, in_place, check } => {
            format_files(&files, in_place, check)
        }
    }
}

fn run_file(file: &Path, context_files: &[PathBuf], output_format: &str, interactive: bool) -> Result<()> {
    println!("{} {}", "Running".green().bold(), file.display());
    
    let source = fs::read_to_string(file)
        .map_err(|e| TurbulanceError::IoError { message: e.to_string() })?;
    
    let mut engine = Engine::new();
    
    // Load context files
    for context_file in context_files {
        let context_source = fs::read_to_string(context_file)
            .map_err(|e| TurbulanceError::IoError { message: e.to_string() })?;
        let context_script = Script::from_source(&context_source)?;
        engine.execute(&context_script)?;
    }
    
    let script = Script::from_source(&source)?;
    let result = engine.execute(&script)?;
    
    // Output result
    match output_format {
        "json" => println!("{}", serde_json::to_string_pretty(&result)?),
        "yaml" => println!("{}", serde_yaml::to_string(&result)?),
        "text" | _ => print_value(&result),
    }
    
    if interactive {
        println!("\n{}", "Entering interactive mode...".cyan());
        start_interactive_session(engine)?;
    }
    
    Ok(())
}

fn validate_files(files: &[PathBuf], semantic: bool, report: bool) -> Result<()> {
    let mut all_valid = true;
    let mut validation_results = Vec::new();
    
    for file in files {
        println!("{} {}", "Validating".blue().bold(), file.display());
        
        let source = fs::read_to_string(file)
            .map_err(|e| TurbulanceError::IoError { message: e.to_string() })?;
        
        match turbulance::validate(&source) {
            Ok(is_valid) => {
                if is_valid {
                    println!("  {} Valid syntax", "✓".green());
                    
                    if semantic {
                        // TODO: Add semantic validation
                        println!("  {} Semantic validation not yet implemented", "⚠".yellow());
                    }
                } else {
                    println!("  {} Invalid syntax", "✗".red());
                    all_valid = false;
                }
                
                validation_results.push((file.clone(), is_valid));
            }
            Err(e) => {
                println!("  {} {}", "✗".red(), e.user_message());
                all_valid = false;
                validation_results.push((file.clone(), false));
            }
        }
    }
    
    if report {
        println!("\n{}", "Validation Report".bold().underline());
        for (file, valid) in &validation_results {
            let status = if *valid { "PASS".green() } else { "FAIL".red() };
            println!("{}: {}", file.display(), status);
        }
    }
    
    if all_valid {
        println!("\n{} All files are valid!", "✓".green().bold());
    } else {
        println!("\n{} Some files have errors!", "✗".red().bold());
        std::process::exit(1);
    }
    
    Ok(())
}

fn start_repl(_context: Option<PathBuf>, _history: Option<PathBuf>) -> Result<()> {
    println!("{}", format!("Turbulance v{} Interactive Mode", turbulance::VERSION).bold());
    println!("Type 'exit' to quit, 'help' for commands\n");
    
    let engine = Engine::new();
    start_interactive_session(engine)?;
    
    Ok(())
}

fn start_interactive_session(mut engine: Engine) -> Result<()> {
    loop {
        print!("turbulance> ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();
                
                if input.is_empty() {
                    continue;
                }
                
                match input {
                    "exit" | "quit" => break,
                    "help" => show_repl_help(),
                    "version" => println!("Turbulance v{}", turbulance::VERSION),
                    "clear" => print!("\x1B[2J\x1B[1;1H"),
                    _ => {
                        match turbulance::execute(input) {
                            Ok(value) => print_value(&value),
                            Err(e) => println!("{} {}", "Error:".red(), e.user_message()),
                        }
                    }
                }
            }
            Err(e) => {
                println!("Error reading input: {}", e);
                break;
            }
        }
    }
    
    Ok(())
}

fn show_repl_help() {
    println!("Turbulance Interactive Commands:");
    println!("  help     - Show this help message");
    println!("  version  - Show version information");
    println!("  clear    - Clear the screen");
    println!("  exit     - Exit the REPL");
    println!("\nYou can also enter any Turbulance expression or statement.");
}

fn create_template(name: &str, template_type: &str, output_dir: &Path) -> Result<()> {
    let template_content = match template_type {
        "hypothesis" => create_hypothesis_template(name),
        "experiment" => create_experiment_template(name),
        "analysis" => create_analysis_template(name),
        _ => return Err(TurbulanceError::InvalidInput(
            format!("Unknown template type: {}", template_type)
        )),
    };
    
    let filename = format!("{}.turb", name);
    let filepath = output_dir.join(&filename);
    
    fs::write(&filepath, template_content)
        .map_err(|e| TurbulanceError::IoError { message: e.to_string() })?;
    
    println!("{} Created {} template: {}", 
             "✓".green().bold(), 
             template_type, 
             filepath.display());
    
    Ok(())
}

fn create_hypothesis_template(name: &str) -> String {
    format!(r#"// Turbulance Hypothesis Template: {}
// Generated by Turbulance v{}

proposition {}Hypothesis:
    motion Hypothesis("State your hypothesis here")
    
    sources:
        local("data/primary_sources.csv")
        web_search(engines = ["pubmed", "arxiv"])
    
    within experiment:
        given data_quality(primary_sources) > 0.8:
            item analysis = analyze_correlation(primary_sources)
            ensure analysis.confidence > 0.95
        
        alternatively:
            research "alternative methodologies for {}"

funxn validate_hypothesis():
    item evidence = collect_evidence()
    item statistical_power = calculate_power(evidence)
    
    given statistical_power > 0.8:
        return "Hypothesis is testable"
    alternatively:
        return "Need more data for proper validation"

// Run the validation
validate_hypothesis()
"#, name, turbulance::VERSION, name, name)
}

fn create_experiment_template(name: &str) -> String {
    format!(r#"// Turbulance Experiment Template: {}
// Generated by Turbulance v{}

project {}Experiment(
    title: "{}",
    version: "1.0",
    reproducible: true
):
    sources:
        local("data/experimental_data.csv")
    
    funxn setup_experiment():
        item control_group = select_controls()
        item treatment_group = select_treatments()
        ensure groups_balanced(control_group, treatment_group)
        return (control_group, treatment_group)
    
    funxn run_experiment(control, treatment):
        item baseline = measure_baseline(control)
        item intervention_result = apply_treatment(treatment)
        item follow_up = measure_outcomes(control + treatment)
        
        return analysis {{
            control: baseline,
            treatment: intervention_result,
            outcomes: follow_up
        }}
    
    funxn analyze_results(results):
        item statistical_test = t_test(results.control, results.treatment)
        item effect_size = cohen_d(results.control, results.treatment)
        
        given statistical_test.p_value < 0.05:
            return "Significant effect detected"
        alternatively:
            return "No significant effect"

// Execute the experiment
item (control, treatment) = setup_experiment()
item results = run_experiment(control, treatment)
item conclusion = analyze_results(results)

conclusion
"#, name, turbulance::VERSION, name, name)
}

fn create_analysis_template(name: &str) -> String {
    format!(r#"// Turbulance Analysis Template: {}
// Generated by Turbulance v{}

funxn load_and_clean_data(filepath):
    item raw_data = load_data(filepath)
    item cleaned = raw_data / missing_values / outliers
    ensure data_quality(cleaned) > 0.9
    return cleaned

funxn exploratory_analysis(dataset):
    item summary_stats = describe(dataset)
    item correlations = correlation_matrix(dataset)
    item distributions = plot_distributions(dataset)
    
    return {{
        statistics: summary_stats,
        correlations: correlations,
        visualizations: distributions
    }}

funxn statistical_modeling(dataset):
    item features = select_features(dataset)
    item model = fit_model(features, dataset.target)
    item validation = cross_validate(model, features)
    
    given validation.accuracy > 0.85:
        return model
    alternatively:
        research "alternative modeling approaches"

// Main analysis workflow
item data = load_and_clean_data("data/{}_data.csv")
item exploration = exploratory_analysis(data)
item model = statistical_modeling(data)

// Generate report
item report = {{
    data_summary: exploration.statistics,
    key_findings: exploration.correlations,
    model_performance: model.metrics,
    recommendations: generate_recommendations(model)
}}

report
"#, name, turbulance::VERSION, name)
}

fn show_info(detailed: bool) -> Result<()> {
    println!("{}", "Turbulance Information".bold().underline());
    println!("Version: {}", turbulance::VERSION);
    println!("Language: Universal Scientific Experiment DSL");
    println!("License: MIT");
    
    if detailed {
        println!("\n{}", "System Information".bold());
        println!("Platform: {}", std::env::consts::OS);
        println!("Architecture: {}", std::env::consts::ARCH);
        println!("Rust version: {}", env!("RUSTC_VERSION", "unknown"));
        
        println!("\n{}", "Features Enabled".bold());
        #[cfg(feature = "wasm")]
        println!("- WebAssembly support");
        #[cfg(feature = "scientific-stdlib")]
        println!("- Extended scientific library");
        #[cfg(not(any(feature = "wasm", feature = "scientific-stdlib")))]
        println!("- Standard features only");
        
        println!("\n{}", "Usage Examples".bold());
        println!("  turbulance run experiment.turb");
        println!("  turbulance validate analysis.turb");
        println!("  turbulance new my-hypothesis --type hypothesis");
        println!("  turbulance repl");
    }
    
    Ok(())
}

fn format_files(_files: &[PathBuf], _in_place: bool, _check: bool) -> Result<()> {
    println!("{} Code formatting is not yet implemented", "⚠".yellow());
    println!("This feature will be available in a future release.");
    Ok(())
}

fn print_value(value: &Value) {
    match value {
        Value::Number(n) => println!("{}", n),
        Value::String(s) => println!("{}", s),
        Value::Boolean(b) => println!("{}", b),
        Value::Array(arr) => {
            println!("[");
            for (i, item) in arr.iter().enumerate() {
                print!("  ");
                print_value(item);
                if i < arr.len() - 1 {
                    print!(",");
                }
            }
            println!("]");
        }
        Value::Object(obj) => {
            println!("{{");
            for (key, val) in obj {
                print!("  {}: ", key);
                print_value(val);
            }
            println!("}}");
        }
        Value::Null => println!("null"),
        _ => println!("{:?}", value),
    }
} 