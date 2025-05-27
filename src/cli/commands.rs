use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use colored::Colorize;
use anyhow::{Result, Context};

use crate::cli::config::CliConfig;
use crate::turbulance;
use crate::text_unit;
use crate::knowledge;

/// Available CLI commands beyond the basic subcommands
pub struct CliCommands {
    config: CliConfig,
}

impl CliCommands {
    /// Create a new CLI commands handler
    pub fn new(config: CliConfig) -> Self {
        Self { config }
    }
    
    /// Initialize a new Kwasa-Kwasa project
    pub fn init_project(&self, name: &str, template: Option<&str>) -> Result<()> {
        let project_path = PathBuf::from(name);
        
        if project_path.exists() {
            return Err(anyhow::anyhow!("Directory '{}' already exists", name));
        }
        
        println!("{} {}", "Creating project:".green().bold(), name);
        
        // Create project directory structure
        fs::create_dir_all(&project_path)?;
        fs::create_dir_all(project_path.join("src"))?;
        fs::create_dir_all(project_path.join("docs"))?;
        fs::create_dir_all(project_path.join("examples"))?;
        fs::create_dir_all(project_path.join("tests"))?;
        
        // Create main Turbulance file
        let main_content = match template {
            Some("research") => include_str!("../templates/research_main.turb"),
            Some("analysis") => include_str!("../templates/analysis_main.turb"), 
            Some("nlp") => include_str!("../templates/nlp_main.turb"),
            _ => include_str!("../templates/default_main.turb"),
        };
        
        fs::write(project_path.join("src/main.turb"), main_content)?;
        
        // Create project configuration
        let project_config = format!(
            r#"[project]
name = "{}"
version = "0.1.0"
authors = ["Your Name <your.email@example.com>"]
description = "A Kwasa-Kwasa text processing project"

[dependencies]
kwasa-kwasa = "0.1.0"

[build]
turbulance_version = "0.1.0"
"#,
            name
        );
        
        fs::write(project_path.join("project.toml"), project_config)?;
        
        // Create README
        let readme_content = format!(
            r#"# {}

A text processing project built with the Kwasa-Kwasa framework.

## Getting Started

Run the main script:
```bash
kwasa-kwasa run src/main.turb
```

Start an interactive session:
```bash
kwasa-kwasa repl
```

## Project Structure

- `src/` - Turbulance source files
- `docs/` - Documentation
- `examples/` - Example scripts
- `tests/` - Test files

## Learn More

- [Kwasa-Kwasa Documentation](https://github.com/yourusername/kwasa-kwasa)
- [Turbulance Language Guide](https://github.com/yourusername/kwasa-kwasa/docs/turbulance.md)
"#,
            name
        );
        
        fs::write(project_path.join("README.md"), readme_content)?;
        
        // Create example file
        let example_content = r#"// Example: Simple text analysis
project "example-analysis":
    source text_data: "The quick brown fox jumps over the lazy dog."
    
    funxn analyze_text(text):
        let words = text / word
        let sentences = text / sentence
        
        within words considering all:
            ensure len(this) > 2
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "readability": readability_score(text)
        }
    
    let result = analyze_text(text_data)
    print("Analysis results:", result)
"#;
        
        fs::write(project_path.join("examples/simple_analysis.turb"), example_content)?;
        
        println!("{}", "Project created successfully!".green());
        println!("Next steps:");
        println!("  cd {}", name);
        println!("  kwasa-kwasa run src/main.turb");
        
        Ok(())
    }
    
    /// Show project information
    pub fn project_info(&self, path: Option<&Path>) -> Result<()> {
        let project_path = path.unwrap_or_else(|| Path::new("."));
        let config_path = project_path.join("project.toml");
        
        if !config_path.exists() {
            return Err(anyhow::anyhow!("No project.toml found. Not a Kwasa-Kwasa project?"));
        }
        
        let config_content = fs::read_to_string(&config_path)?;
        let config: toml::Value = toml::from_str(&config_content)?;
        
        println!("{}", "Project Information".cyan().bold());
        println!("==================");
        
        if let Some(project) = config.get("project") {
            if let Some(name) = project.get("name") {
                println!("Name: {}", name.as_str().unwrap_or("Unknown"));
            }
            if let Some(version) = project.get("version") {
                println!("Version: {}", version.as_str().unwrap_or("Unknown"));
            }
            if let Some(description) = project.get("description") {
                println!("Description: {}", description.as_str().unwrap_or("No description"));
            }
        }
        
        // Show file statistics
        let src_path = project_path.join("src");
        if src_path.exists() {
            let turb_files: Vec<_> = fs::read_dir(&src_path)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry.path().extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "turb")
                        .unwrap_or(false)
                })
                .collect();
            
            println!("Turbulance files: {}", turb_files.len());
        }
        
        Ok(())
    }
    
    /// Analyze project dependencies and complexity
    pub fn analyze_project(&self, path: Option<&Path>) -> Result<()> {
        let project_path = path.unwrap_or_else(|| Path::new("."));
        let src_path = project_path.join("src");
        
        if !src_path.exists() {
            return Err(anyhow::anyhow!("No src directory found"));
        }
        
        println!("{}", "Project Analysis".cyan().bold());
        println!("================");
        
        let mut total_lines = 0;
        let mut total_functions = 0;
        let mut total_projects = 0;
        
        // Analyze all .turb files
        for entry in fs::read_dir(&src_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|ext| ext.to_str()) == Some("turb") {
                let content = fs::read_to_string(&path)?;
                let lines = content.lines().count();
                total_lines += lines;
                
                // Simple pattern matching for functions and projects
                let functions = content.matches("funxn ").count();
                let projects = content.matches("project ").count();
                
                total_functions += functions;
                total_projects += projects;
                
                println!("  {}: {} lines, {} functions", 
                    path.file_name().unwrap().to_string_lossy(),
                    lines,
                    functions
                );
            }
        }
        
        println!();
        println!("Summary:");
        println!("  Total lines: {}", total_lines);
        println!("  Total functions: {}", total_functions);
        println!("  Total projects: {}", total_projects);
        
        Ok(())
    }
    
    /// Format Turbulance code files
    pub fn format_code(&self, path: &Path, check_only: bool) -> Result<()> {
        if check_only {
            println!("{}", "Checking code formatting...".blue());
        } else {
            println!("{}", "Formatting code...".blue());
        }
        
        let files = if path.is_file() {
            vec![path.to_path_buf()]
        } else {
            self.find_turbulence_files(path)?
        };
        
        let mut needs_formatting = Vec::new();
        
        for file_path in files {
            let content = fs::read_to_string(&file_path)?;
            let formatted = self.format_turbulance_code(&content)?;
            
            if content != formatted {
                needs_formatting.push(file_path.clone());
                
                if !check_only {
                    fs::write(&file_path, formatted)?;
                    println!("  {} {}", "Formatted:".green(), file_path.display());
                }
            }
        }
        
        if check_only {
            if needs_formatting.is_empty() {
                println!("{}", "All files are properly formatted".green());
            } else {
                println!("{}", "Files that need formatting:".yellow());
                for file in &needs_formatting {
                    println!("  {}", file.display());
                }
                return Err(anyhow::anyhow!("Some files need formatting"));
            }
        }
        
        Ok(())
    }
    
    /// Generate documentation for a project
    pub fn generate_docs(&self, path: Option<&Path>, format: &str) -> Result<()> {
        let project_path = path.unwrap_or_else(|| Path::new("."));
        let src_path = project_path.join("src");
        let docs_path = project_path.join("docs");
        
        fs::create_dir_all(&docs_path)?;
        
        println!("{}", "Generating documentation...".blue());
        
        let files = self.find_turbulence_files(&src_path)?;
        let mut documentation = String::new();
        
        match format {
            "markdown" => {
                documentation.push_str("# Project Documentation\n\n");
                
                for file_path in files {
                    let content = fs::read_to_string(&file_path)?;
                    let filename = file_path.file_name().unwrap().to_string_lossy();
                    
                    documentation.push_str(&format!("## {}\n\n", filename));
                    
                    // Extract and document functions
                    let functions = self.extract_functions(&content);
                    for function in functions {
                        documentation.push_str(&format!("### `{}`\n\n", function.name));
                        if !function.description.is_empty() {
                            documentation.push_str(&format!("{}\n\n", function.description));
                        }
                        documentation.push_str("```turbulance\n");
                        documentation.push_str(&function.signature);
                        documentation.push_str("\n```\n\n");
                    }
                }
                
                fs::write(docs_path.join("README.md"), documentation)?;
            }
            "html" => {
                // Generate HTML documentation
                let html = format!(
                    r#"<!DOCTYPE html>
<html>
<head>
    <title>Project Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .function {{ border: 1px solid #ccc; padding: 20px; margin: 20px 0; }}
        code {{ background: #f5f5f5; padding: 2px 4px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Project Documentation</h1>
    <div id="content">
        <!-- Documentation content would be generated here -->
    </div>
</body>
</html>"#
                );
                fs::write(docs_path.join("index.html"), html)?;
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported documentation format: {}", format));
            }
        }
        
        println!("{}", "Documentation generated successfully!".green());
        
        Ok(())
    }
    
    /// Run project tests
    pub fn run_tests(&self, path: Option<&Path>, filter: Option<&str>) -> Result<()> {
        let project_path = path.unwrap_or_else(|| Path::new("."));
        let tests_path = project_path.join("tests");
        
        if !tests_path.exists() {
            println!("{}", "No tests directory found".yellow());
            return Ok(());
        }
        
        println!("{}", "Running tests...".blue());
        
        let test_files = self.find_turbulence_files(&tests_path)?;
        let mut passed = 0;
        let mut failed = 0;
        
        for test_file in test_files {
            let filename = test_file.file_name().unwrap().to_string_lossy();
            
            // Apply filter if specified
            if let Some(filter) = filter {
                if !filename.contains(filter) {
                    continue;
                }
            }
            
            print!("  Running {}: ", filename);
            
            let content = fs::read_to_string(&test_file)?;
            match turbulance::run(&content) {
                Ok(_) => {
                    println!("{}", "PASSED".green());
                    passed += 1;
                }
                Err(e) => {
                    println!("{}", "FAILED".red());
                    println!("    Error: {}", e);
                    failed += 1;
                }
            }
        }
        
        println!();
        println!("Test results: {} passed, {} failed", 
            passed.to_string().green(), 
            failed.to_string().red()
        );
        
        if failed > 0 {
            return Err(anyhow::anyhow!("Some tests failed"));
        }
        
        Ok(())
    }
    
    /// Show configuration
    pub fn show_config(&self) -> Result<()> {
        println!("{}", "Current Configuration".cyan().bold());
        println!("=====================");
        
        println!("REPL Settings:");
        println!("  Syntax highlighting: {}", self.config.repl.syntax_highlighting);
        println!("  Auto-completion: {}", self.config.repl.auto_completion);
        println!("  History size: {}", self.config.repl.history_size);
        println!("  Prompt: {}", self.config.repl.prompt);
        
        println!("\nOutput Settings:");
        println!("  Colored output: {}", self.config.output.colored);
        println!("  Verbosity: {}", self.config.output.verbosity);
        println!("  Pretty print: {}", self.config.output.pretty_print);
        
        println!("\nEditor Settings:");
        println!("  Editor command: {}", self.config.editor.editor_command);
        println!("  Tab width: {}", self.config.editor.tab_width);
        println!("  Use spaces: {}", self.config.editor.use_spaces);
        
        println!("\nPerformance Settings:");
        println!("  Parallel processing: {}", self.config.performance.parallel_processing);
        println!("  Thread count: {}", 
            if self.config.performance.thread_count == 0 { 
                "auto".to_string() 
            } else { 
                self.config.performance.thread_count.to_string() 
            }
        );
        
        if !self.config.custom.is_empty() {
            println!("\nCustom Settings:");
            for (key, value) in &self.config.custom {
                println!("  {}: {}", key, value);
            }
        }
        
        println!("\nConfig file: {}", CliConfig::config_path().display());
        
        Ok(())
    }
    
    // Helper methods
    
    fn find_turbulence_files(&self, path: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("turb") {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                
                if entry_path.is_dir() {
                    files.extend(self.find_turbulence_files(&entry_path)?);
                } else if entry_path.extension().and_then(|ext| ext.to_str()) == Some("turb") {
                    files.push(entry_path);
                }
            }
        }
        
        Ok(files)
    }
    
    fn format_turbulance_code(&self, content: &str) -> Result<String> {
        // Simple formatter - this would be expanded with proper AST-based formatting
        let mut formatted = String::new();
        let mut indent_level: usize = 0;
        let indent_str = if self.config.editor.use_spaces {
            " ".repeat(self.config.editor.tab_width as usize)
        } else {
            "\t".to_string()
        };
        
        for line in content.lines() {
            let trimmed = line.trim();
            
            if trimmed.is_empty() {
                formatted.push('\n');
                continue;
            }
            
            // Decrease indent for closing braces
            if trimmed.starts_with('}') {
                indent_level = indent_level.saturating_sub(1);
            }
            
            // Add indentation
            for _ in 0..indent_level {
                formatted.push_str(&indent_str);
            }
            formatted.push_str(trimmed);
            formatted.push('\n');
            
            // Increase indent for opening braces
            if trimmed.ends_with('{') {
                indent_level += 1;
            }
        }
        
        Ok(formatted)
    }
    
    fn extract_functions(&self, content: &str) -> Vec<FunctionDoc> {
        let mut functions = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            if line.trim().starts_with("funxn ") {
                let signature = line.trim().to_string();
                let name = signature
                    .strip_prefix("funxn ")
                    .and_then(|s| s.split('(').next())
                    .unwrap_or("unknown")
                    .to_string();
                
                // Look for preceding comments as description
                let mut description = String::new();
                let mut j = i;
                while j > 0 {
                    j -= 1;
                    let prev_line = lines[j].trim();
                    if prev_line.starts_with("//") {
                        let comment = prev_line.strip_prefix("//").unwrap_or("").trim();
                        if !comment.is_empty() {
                            if !description.is_empty() {
                                description = format!("{}\n{}", comment, description);
                            } else {
                                description = comment.to_string();
                            }
                        }
                    } else if !prev_line.is_empty() {
                        break;
                    }
                }
                
                functions.push(FunctionDoc {
                    name,
                    signature,
                    description,
                });
            }
        }
        
        functions
    }
}

#[derive(Debug)]
struct FunctionDoc {
    name: String,
    signature: String,
    description: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_project_init() {
        let temp_dir = TempDir::new().unwrap();
        let config = CliConfig::default();
        let commands = CliCommands::new(config);
        
        let project_name = "test_project";
        let project_path = temp_dir.path().join(project_name);
        
        std::env::set_current_dir(temp_dir.path()).unwrap();
        
        let result = commands.init_project(project_name, None);
        assert!(result.is_ok());
        
        // Verify project structure
        assert!(project_path.exists());
        assert!(project_path.join("src").exists());
        assert!(project_path.join("src/main.turb").exists());
        assert!(project_path.join("project.toml").exists());
        assert!(project_path.join("README.md").exists());
    }
    
    #[test]
    fn test_find_turbulence_files() {
        let temp_dir = TempDir::new().unwrap();
        let config = CliConfig::default();
        let commands = CliCommands::new(config);
        
        // Create test files
        let src_dir = temp_dir.path().join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(src_dir.join("main.turb"), "// main file").unwrap();
        fs::write(src_dir.join("lib.turb"), "// lib file").unwrap();
        fs::write(src_dir.join("readme.txt"), "not a turb file").unwrap();
        
        let files = commands.find_turbulence_files(&src_dir).unwrap();
        assert_eq!(files.len(), 2);
        
        let filenames: Vec<_> = files.iter()
            .map(|p| p.file_name().unwrap().to_string_lossy())
            .collect();
        assert!(filenames.contains(&"main.turb".into()));
        assert!(filenames.contains(&"lib.turb".into()));
    }
}

