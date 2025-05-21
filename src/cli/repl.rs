use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use rustyline_derive::{Completer, Helper, Highlighter, Hinter, Validator};

use crate::error::{Error, Result};
use crate::turbulance;

#[derive(Completer, Helper, Highlighter, Hinter, Validator)]
struct ReplHelper;

/// The REPL environment for interactive Turbulance usage
pub struct Repl {
    /// The rustyline editor
    editor: DefaultEditor,
    /// The REPL history
    history: Vec<String>,
    /// The execution context (persisted between commands)
    context: turbulance::Context,
}

impl Repl {
    /// Create a new REPL
    pub fn new() -> Result<Self> {
        let mut editor = DefaultEditor::new().map_err(|e| Error::cli(format!("Failed to create editor: {}", e)))?;

        // Load history if it exists
        let _ = editor.load_history("turbulance_history.txt");

        Ok(Self {
            editor,
            history: Vec::new(),
            context: turbulance::Context::new(),
        })
    }

    /// Start the REPL
    pub fn start(&mut self) -> Result<()> {
        println!("{} v{}", "Turbulance REPL".green().bold(), turbulance::VERSION);
        println!("Type {} to see available commands, {} to exit", ":help".cyan(), ":exit".cyan());
        println!();

        loop {
            let readline = self.editor.readline("turbulance> ");

            match readline {
                Ok(line) => {
                    self.editor.add_history_entry(&line)?;
                    self.history.push(line.clone());

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(cmd) = line.strip_prefix(':') {
                        if !self.handle_command(cmd)? {
                            break;
                        }
                    } else {
                        self.execute_code(&line)?;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D");
                    break;
                }
                Err(err) => {
                    return Err(Error::cli(format!("Error: {}", err)));
                }
            }
        }

        // Save history
        let _ = self.editor.save_history("turbulance_history.txt");

        Ok(())
    }

    /// Handle REPL commands
    fn handle_command(&mut self, cmd: &str) -> Result<bool> {
        match cmd.trim() {
            "exit" | "quit" => return Ok(false),
            "help" => self.show_help(),
            "history" => self.show_history(),
            "clear" => self.clear_screen(),
            "reset" => self.reset_context(),
            cmd if cmd.starts_with("load ") => {
                let filename = cmd.trim_start_matches("load ").trim();
                self.load_file(filename)?;
            }
            cmd if cmd.starts_with("save ") => {
                let parts: Vec<&str> = cmd.splitn(3, ' ').collect();
                if parts.len() < 3 {
                    println!("{} Usage: :save <filename> <code>", "Error:".red());
                } else {
                    self.save_to_file(parts[1], parts[2])?;
                }
            }
            _ => {
                println!("{} Unknown command: {}", "Error:".red(), cmd);
                println!("Type {} for a list of commands", ":help".cyan());
            }
        }

        Ok(true)
    }

    /// Execute Turbulance code
    fn execute_code(&mut self, code: &str) -> Result<()> {
        match turbulance::run_with_context(code, &mut self.context) {
            Ok(result) => {
                if !result.is_empty() {
                    println!("{}", result);
                }
            }
            Err(err) => {
                println!("{} {}", "Error:".red().bold(), err);
            }
        }

        Ok(())
    }

    /// Show help
    fn show_help(&self) -> Result<bool> {
        println!("Available commands:");
        println!("  {}  - Exit the REPL", ":exit".cyan());
        println!("  {}  - Show this help", ":help".cyan());
        println!("  {} - Show command history", ":history".cyan());
        println!("  {}  - Clear the screen", ":clear".cyan());
        println!("  {}  - Reset the execution context", ":reset".cyan());
        println!("  {} - Load and execute a file", ":load <filename>".cyan());
        println!("  {} - Save code to a file", ":save <filename> <code>".cyan());
        println!();
        println!("You can also type Turbulance code directly to execute it.");
        Ok(true)
    }

    /// Show command history
    fn show_history(&self) -> Result<bool> {
        if self.history.is_empty() {
            println!("No history yet");
            return Ok(true);
        }

        for (i, cmd) in self.history.iter().enumerate() {
            println!("{}: {}", i + 1, cmd);
        }
        Ok(true)
    }

    /// Clear the screen
    fn clear_screen(&self) -> Result<bool> {
        print!("\x1B[2J\x1B[1;1H");
        Ok(true)
    }

    /// Reset the execution context
    fn reset_context(&mut self) -> Result<bool> {
        self.context = turbulance::Context::new();
        println!("Context reset");
        Ok(true)
    }

    /// Load and execute a file
    fn load_file(&mut self, filename: &str) -> Result<bool> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| Error::cli(format!("Failed to read file: {}", e)))?;

        println!("{} {}", "Loading file:".green(), filename);
        self.execute_code(&content)?;
        Ok(true)
    }

    /// Save code to a file
    fn save_to_file(&self, filename: &str, code: &str) -> Result<bool> {
        std::fs::write(filename, code)
            .map_err(|e| Error::cli(format!("Failed to write file: {}", e)))?;

        println!("{} {}", "Saved to file:".green(), filename);
        Ok(true)
    }
} 