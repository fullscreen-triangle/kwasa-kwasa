use std::fs;
use std::path::Path;
use anyhow::{Context, Result};

use crate::turbulance::lexer::Lexer;
use crate::turbulance::parser::Parser;
use crate::turbulance::interpreter::Interpreter;

/// Executes a Turbulance script file
pub fn run_script(path: &Path) -> Result<()> {
    // Read the file content
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;
    
    // Parse the script
    let tokens = Lexer::tokenize(&content)
        .with_context(|| "Failed to tokenize script")?;
    
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()
        .with_context(|| "Failed to parse script")?;
    
    // Execute the script
    let mut interpreter = Interpreter::new();
    interpreter.execute(&ast)
        .with_context(|| "Error during script execution")?;
    
    Ok(())
}

/// Executes a Turbulance script string directly
pub fn run_script_string(script: &str) -> Result<()> {
    // Parse the script
    let tokens = Lexer::tokenize(script)
        .with_context(|| "Failed to tokenize script")?;
    
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()
        .with_context(|| "Failed to parse script")?;
    
    // Execute the script
    let mut interpreter = Interpreter::new();
    interpreter.execute(&ast)
        .with_context(|| "Error during script execution")?;
    
    Ok(())
}

/// Validates a Turbulance script file without executing it
pub fn validate_script(path: &Path) -> Result<()> {
    // Read the file content
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;
    
    // Parse the script
    let tokens = Lexer::tokenize(&content)
        .with_context(|| "Failed to tokenize script")?;
    
    let mut parser = Parser::new(tokens);
    parser.parse()
        .with_context(|| "Failed to parse script")?;
    
    println!("Script validation successful: {}", path.display());
    Ok(())
} 