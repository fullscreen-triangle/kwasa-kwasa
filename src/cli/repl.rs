use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::{Config, Editor, CompletionType};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::validate::{Validator, ValidationContext, ValidationResult};
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::{Hinter, HistoryHinter};
use rustyline_derive::{Helper};
use std::borrow::Cow::{self, Borrowed, Owned};
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::error::{Error, Result, ErrorReporter};
use crate::turbulance;

/// Custom prompt helper with syntax highlighting and completion
#[derive(Helper)]
struct ReplHelper {
    completer: FilenameCompleter,
    highlighter: MatchingBracketHighlighter,
    hinter: HistoryHinter,
    colored_prompt: String,
    keywords: Vec<String>,
}

impl ReplHelper {
    fn new() -> Self {
        // Define Turbulance language keywords for completion
        let keywords = vec![
            "funxn".to_string(), "project".to_string(), "source".to_string(),
            "within".to_string(), "ensure".to_string(), "given".to_string(),
            "return".to_string(), "if".to_string(), "else".to_string(),
            "for".to_string(), "while".to_string(), "let".to_string(),
            "considering".to_string(), "all".to_string(), "these".to_string(),
            "each".to_string(), "continue".to_string(), "break".to_string(),
        ];
        
        Self {
            completer: FilenameCompleter::new(),
            highlighter: MatchingBracketHighlighter::new(),
            hinter: HistoryHinter {},
            colored_prompt: "turbulance> ".green().to_string(),
            keywords,
        }
    }
}

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        // First try command completion if the line starts with ':'
        if line.starts_with(':') {
            let commands = vec![
                ":exit", ":help", ":history", ":clear", ":reset",
                ":load ", ":save ", ":run", ":debug",
            ];
            
            let matches: Vec<Pair> = commands
                .iter()
                .filter(|cmd| cmd.starts_with(&line[1..]))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            
            if !matches.is_empty() {
                return Ok((1, matches));
            }
        }
        
        // Then try keyword completion
        let word_start = line[..pos].rfind(|c: char| c.is_whitespace() || c == '(' || c == '{' || c == '[')
            .map(|i| i + 1)
            .unwrap_or(0);
        
        if pos > word_start {
            let current_word = &line[word_start..pos];
            
            // Complete keywords
            let keyword_matches: Vec<Pair> = self.keywords
                .iter()
                .filter(|kw| kw.starts_with(current_word))
                .map(|kw| Pair {
                    display: kw.clone(),
                    replacement: kw.clone(),
                })
                .collect();
            
            if !keyword_matches.is_empty() {
                return Ok((word_start, keyword_matches));
            }
        }
        
        // Fall back to filename completion for :load and :save commands
        if line.starts_with(":load ") || line.starts_with(":save ") {
            let command_prefix_len = if line.starts_with(":load ") { 6 } else { 6 };
            let (file_start, file_matches) = self.completer.complete(
                &line[command_prefix_len..], 
                pos - command_prefix_len, 
                _ctx
            )?;
            return Ok((command_prefix_len + file_start, file_matches));
        }
        
        Ok((pos, vec![]))
    }
}

impl Highlighter for ReplHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        // Do standard bracket highlighting
        let bracket_highlighted = self.highlighter.highlight(line, pos);
        
        // Tokenize the line and apply syntax highlighting
        let mut result = String::with_capacity(line.len() * 2); // Estimating extra chars for ANSI codes
        
        let mut in_string = false;
        let mut in_comment = false;
        let mut token_start = 0;
        
        for (i, c) in line.char_indices() {
            // Handle string literals
            if c == '"' && !in_comment {
                if in_string {
                    // End of string
                    result.push_str(&line[token_start..=i].yellow().to_string());
                    in_string = false;
                    token_start = i + 1;
                } else {
                    // Start of string
                    if token_start < i {
                        result.push_str(&self.highlight_token(&line[token_start..i]));
                    }
                    in_string = true;
                    token_start = i;
                }
                continue;
            }
            
            // Handle comments
            if c == '/' && i + 1 < line.len() && &line[i..i+2] == "//" && !in_string {
                // Process any token before the comment
                if token_start < i {
                    result.push_str(&self.highlight_token(&line[token_start..i]));
                }
                // Add the comment (rest of the line)
                result.push_str(&line[i..].bright_black().to_string());
                in_comment = true;
                break; // Done with the line
            }
            
            // Don't process other tokens if in a string or comment
            if in_string || in_comment {
                continue;
            }
            
            // Handle whitespace and delimiters
            if c.is_whitespace() || "(){}[],;:".contains(c) {
                if token_start < i {
                    result.push_str(&self.highlight_token(&line[token_start..i]));
                }
                if "(){}[]".contains(c) {
                    result.push_str(&c.to_string().cyan().to_string());
                } else if c == ',' || c == ';' || c == ':' {
                    result.push_str(&c.to_string().bright_black().to_string());
                } else {
                    result.push(c);
                }
                token_start = i + 1;
            }
        }
        
        // Add the last token if not processed
        if !in_comment && token_start < line.len() {
            result.push_str(&self.highlight_token(&line[token_start..]));
        }
        
        if result.is_empty() {
            bracket_highlighted
        } else {
            Owned(result)
        }
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if default {
            Borrowed(&self.colored_prompt)
        } else {
            Borrowed(prompt)
        }
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Owned(hint.bright_black().to_string())
    }

    fn highlight_char(&self, line: &str, pos: usize) -> bool {
        self.highlighter.highlight_char(line, pos)
    }
}

impl ReplHelper {
    fn highlight_token(&self, token: &str) -> String {
        // Keywords - bold blue
        if self.keywords.contains(&token.to_string()) {
            return token.blue().bold().to_string();
        }
        
        // Numbers - cyan
        if token.parse::<f64>().is_ok() {
            return token.cyan().to_string();
        }
        
        // Boolean literals - magenta
        if token == "true" || token == "false" {
            return token.magenta().to_string();
        }
        
        // Special values - magenta
        if token == "null" || token == "undefined" {
            return token.magenta().italic().to_string();
        }
        
        // Default - regular text
        token.to_string()
    }
}

impl Hinter for ReplHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        self.hinter.hint(line, pos, ctx)
    }
}

impl Validator for ReplHelper {
    fn validate(
        &self,
        ctx: &mut ValidationContext,
    ) -> Result<ValidationResult, ReadlineError> {
        // Simple validation for unclosed brackets
        let input = ctx.input();
        
        let mut brackets = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        
        for c in input.chars() {
            if c == '\\' && in_string {
                escaped = !escaped;
                continue;
            }
            
            if c == '"' && !escaped {
                in_string = !in_string;
            }
            
            if !in_string {
                match c {
                    '(' => brackets.push(')'),
                    '{' => brackets.push('}'),
                    '[' => brackets.push(']'),
                    ')' | '}' | ']' => {
                        if brackets.pop() != Some(c) {
                            return Ok(ValidationResult::Invalid(
                                Some("Mismatched brackets".to_string())
                            ));
                        }
                    }
                    _ => {}
                }
            }
            
            escaped = false;
        }
        
        if in_string {
            Ok(ValidationResult::Invalid(
                Some("Unclosed string literal".to_string())
            ))
        } else if !brackets.is_empty() {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

/// Session data stored to disk
#[derive(serde::Serialize, serde::Deserialize)]
struct ReplSession {
    variables: Vec<(String, String)>,
    functions: Vec<(String, String)>,
    history: Vec<String>,
    working_directory: Option<String>,
}

/// The REPL environment for interactive Turbulance usage
pub struct Repl {
    /// The rustyline editor
    editor: Editor<ReplHelper>,
    /// The REPL history
    history: Vec<String>,
    /// The execution context (persisted between commands)
    context: turbulance::Context,
    /// Error reporter
    error_reporter: ErrorReporter,
    /// Current working directory for file operations
    working_directory: Option<String>,
    /// Session file path
    session_file: String,
}

impl Repl {
    /// Create a new REPL
    pub fn new() -> Result<Self> {
        // Configure editor with validation and completion
        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(CompletionType::List)
            .build();
        
        let helper = ReplHelper::new();
        let mut editor = Editor::with_config(config)?;
        editor.set_helper(Some(helper));
        
        // Load history if it exists
        let home_dir = dirs::home_dir()
            .ok_or_else(|| Error::cli("Could not determine home directory"))?;
        
        let history_file = home_dir.join(".turbulance_history");
        let _ = editor.load_history(&history_file);
        
        // Determine session file path
        let session_file = home_dir.join(".turbulance_session.json").to_string_lossy().to_string();
        
        // Try to load previous session
        let mut context = turbulance::Context::new();
        let mut history = Vec::new();
        let mut working_directory = None;
        
        if let Ok(session_data) = fs::read_to_string(&session_file) {
            if let Ok(session) = serde_json::from_str::<ReplSession>(&session_data) {
                // Restore session data
                for (name, value) in session.variables {
                    // Simplified restoration - in a real implementation, would need to parse the value string
                    context.set_variable(&name, turbulance::context::Value::String(value));
                }
                
                for (name, body) in session.functions {
                    context.define_function(turbulance::context::Function {
                        name: name.clone(),
                        parameters: Vec::new(), // Simplified - would need to parse function signature
                        body,
                        native: false,
                    });
                }
                
                history = session.history;
                working_directory = session.working_directory;
            }
        }
        
        Ok(Self {
            editor,
            history,
            context,
            error_reporter: ErrorReporter::new(),
            working_directory,
            session_file,
        })
    }

    /// Start the REPL
    pub fn start(&mut self) -> Result<()> {
        self.print_welcome_message();
        
        loop {
            // Update prompt with context information
            let prompt = self.create_prompt();
            
            let readline = self.editor.readline(&prompt);

            match readline {
                Ok(line) => {
                    // Skip empty lines
                    if line.trim().is_empty() {
                        continue;
                    }
                    
                    // Add to history
                    self.editor.add_history_entry(&line)?;
                    self.history.push(line.clone());
                    
                    // Process the line
                    if line.starts_with(':') {
                        if !self.handle_command(&line[1..])? {
                            break;
                        }
                    } else {
                        self.execute_code(&line)?;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("CTRL-C pressed, use :exit to quit");
                }
                Err(ReadlineError::Eof) => {
                    println!("CTRL-D pressed, exiting...");
                    break;
                }
                Err(err) => {
                    return Err(Error::cli(format!("Error: {}", err)));
                }
            }
        }

        // Save session before exiting
        self.save_session()?;
        
        // Save history
        let home_dir = dirs::home_dir()
            .ok_or_else(|| Error::cli("Could not determine home directory"))?;
        let history_file = home_dir.join(".turbulance_history");
        let _ = self.editor.save_history(&history_file);

        Ok(())
    }
    
    /// Create a context-aware prompt
    fn create_prompt(&self) -> String {
        let mut prompt = String::new();
        
        // Add working directory if available
        if let Some(ref wd) = self.working_directory {
            let path = Path::new(wd);
            let dir_name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            
            prompt.push_str(&format!("[{}] ", dir_name.blue()));
        }
        
        // Add execution context indicator if in recovery mode
        if self.context.is_in_recovery_mode() {
            prompt.push_str("[recovery] ");
        }
        
        // Add the standard prompt
        prompt.push_str("turbulance> ");
        
        prompt
    }
    
    /// Print the welcome message
    fn print_welcome_message(&self) {
        println!("{} v{}", "Turbulance REPL".green().bold(), turbulance::VERSION);
        println!("Type {} to see available commands, {} to exit", ":help".cyan(), ":exit".cyan());
        println!();
    }

    /// Handle REPL commands
    fn handle_command(&mut self, cmd: &str) -> Result<bool> {
        let parts: Vec<&str> = cmd.trim().splitn(3, ' ').collect();
        let command = parts[0];
        
        match command {
            "exit" | "quit" => return Ok(false),
            
            "help" => self.show_help(),
            
            "history" => self.show_history(),
            
            "clear" => self.clear_screen(),
            
            "reset" => self.reset_context(),
            
            "load" if parts.len() >= 2 => {
                let filename = parts[1].trim();
                return self.load_file(filename);
            }
            
            "save" if parts.len() >= 3 => {
                let filename = parts[1].trim();
                let code = parts[2];
                return self.save_to_file(filename, code);
            }
            
            "cd" if parts.len() >= 2 => {
                let path = parts[1].trim();
                return self.change_directory(path);
            }
            
            "ls" => return self.list_directory(),
            
            "pwd" => return self.print_working_directory(),
            
            "run" if parts.len() >= 2 => {
                let filename = parts[1].trim();
                return self.run_file(filename);
            }
            
            "debug" if parts.len() >= 2 => {
                let filename = parts[1].trim();
                return self.debug_file(filename);
            }
            
            "vars" => return self.show_variables(),
            
            "funcs" => return self.show_functions(),
            
            "time" if parts.len() >= 2 => {
                let code = parts[1];
                return self.time_execution(code);
            }
            
            _ => {
                println!("{} Unknown command: {}", "Error:".red(), command);
                println!("Type {} for a list of commands", ":help".cyan());
            }
        }

        Ok(true)
    }

    /// Execute Turbulance code
    fn execute_code(&mut self, code: &str) -> Result<()> {
        // Begin execution tracking
        self.context.begin_execution();
        
        // Run the code
        let result = turbulance::run_with_context(code, &mut self.context);
        
        // End execution tracking
        self.context.end_execution();
        
        match result {
            Ok(output) => {
                if !output.is_empty() {
                    println!("{}", output);
                }
                
                // Check if there were non-fatal errors
                if self.context.error_reporter().has_errors() {
                    println!("{} {} (non-fatal)", "Warning:".yellow().bold(), 
                        self.context.error_reporter().report());
                }
            }
            Err(err) => {
                // Add the error to our context
                self.context.add_error(err.clone());
                
                // Print error with fancy formatting
                self.print_error(&err);
                
                // Check if we can recover
                if err.is_recoverable() {
                    println!("{} Entering recovery mode", "Recovery:".yellow());
                    self.context.enter_recovery_mode();
                }
            }
        }

        Ok(())
    }
    
    /// Print an error with fancy formatting
    fn print_error(&self, err: &Error) {
        let error_type = match err {
            Error::Parse { .. } => "Syntax Error".red().bold(),
            Error::Lexical { .. } => "Lexical Error".red().bold(),
            Error::Syntax { .. } => "Syntax Error".red().bold(),
            Error::Semantic(..) => "Semantic Error".red().bold(),
            Error::Runtime(..) => "Runtime Error".red().bold(),
            _ => "Error".red().bold(),
        };
        
        println!("{}: {}", error_type, err);
        
        // Add call stack if available
        let call_stack = self.context.get_call_stack();
        if !call_stack.is_empty() {
            println!("{} {}", "Call Stack:".bright_black(), call_stack);
        }
    }

    /// Show help
    fn show_help(&self) -> Result<bool> {
        println!("Available commands:");
        
        // Basic commands
        println!("  {}              - Exit the REPL", ":exit".cyan());
        println!("  {}              - Show this help", ":help".cyan());
        println!("  {}          - Show command history", ":history".cyan());
        println!("  {}             - Clear the screen", ":clear".cyan());
        println!("  {}             - Reset the execution context", ":reset".cyan());
        
        // File operations
        println!("\nFile operations:");
        println!("  {} {}     - Load and execute a file", ":load".cyan(), "<filename>".yellow());
        println!("  {} {} {}  - Save code to a file", ":save".cyan(), "<filename>".yellow(), "<code>".yellow());
        println!("  {} {}       - Run a file", ":run".cyan(), "<filename>".yellow());
        println!("  {} {}     - Debug a file", ":debug".cyan(), "<filename>".yellow());
        
        // Directory operations
        println!("\nDirectory operations:");
        println!("  {} {}          - Change working directory", ":cd".cyan(), "<path>".yellow());
        println!("  {}               - List files in current directory", ":ls".cyan());
        println!("  {}              - Show current working directory", ":pwd".cyan());
        
        // Context inspection
        println!("\nContext inspection:");
        println!("  {}              - Show all variables", ":vars".cyan());
        println!("  {}             - Show all functions", ":funcs".cyan());
        
        // Utilities
        println!("\nUtilities:");
        println!("  {} {}       - Time execution of code", ":time".cyan(), "<code>".yellow());
        
        println!("\nYou can also type Turbulance code directly to execute it.");
        Ok(true)
    }

    /// Show command history
    fn show_history(&self) -> Result<bool> {
        if self.history.is_empty() {
            println!("No history yet");
            return Ok(true);
        }

        let max_display = 20; // Limit history display
        let start_idx = if self.history.len() > max_display {
            self.history.len() - max_display
        } else {
            0
        };

        for (i, cmd) in self.history.iter().enumerate().skip(start_idx) {
            println!("{}: {}", (i + 1).to_string().bright_black(), cmd);
        }
        
        if self.history.len() > max_display {
            println!("(Showing last {} of {} entries)", max_display, self.history.len());
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
        let resolved_path = self.resolve_path(filename);
        
        let content = fs::read_to_string(&resolved_path)
            .map_err(|e| Error::cli(format!("Failed to read file: {}", e)))?;

        println!("{} {}", "Loading file:".green(), resolved_path.display());
        self.execute_code(&content)?;
        Ok(true)
    }

    /// Save code to a file
    fn save_to_file(&self, filename: &str, code: &str) -> Result<bool> {
        let resolved_path = self.resolve_path(filename);
        
        fs::write(&resolved_path, code)
            .map_err(|e| Error::cli(format!("Failed to write file: {}", e)))?;

        println!("{} {}", "Saved to file:".green(), resolved_path.display());
        Ok(true)
    }
    
    /// Change working directory
    fn change_directory(&mut self, path: &str) -> Result<bool> {
        let new_path = if path == "~" {
            dirs::home_dir().ok_or_else(|| Error::cli("Home directory not found"))?
        } else if path.starts_with('~') && path.chars().nth(1) == Some('/') {
            let home = dirs::home_dir().ok_or_else(|| Error::cli("Home directory not found"))?;
            home.join(&path[2..])
        } else {
            self.resolve_path(path)
        };
        
        if !new_path.exists() {
            return Err(Error::cli(format!("Directory does not exist: {}", new_path.display())));
        }
        
        if !new_path.is_dir() {
            return Err(Error::cli(format!("Not a directory: {}", new_path.display())));
        }
        
        self.working_directory = Some(new_path.to_string_lossy().to_string());
        println!("Working directory: {}", new_path.display());
        
        Ok(true)
    }
    
    /// List files in the current directory
    fn list_directory(&self) -> Result<bool> {
        let dir_path = match &self.working_directory {
            Some(path) => Path::new(path).to_path_buf(),
            None => std::env::current_dir().map_err(|e| Error::cli(format!("Failed to get current directory: {}", e)))?,
        };
        
        let entries = fs::read_dir(&dir_path)
            .map_err(|e| Error::cli(format!("Failed to read directory: {}", e)))?;
        
        println!("Contents of {}:", dir_path.display());
        println!();
        
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        
        for entry in entries {
            let entry = entry.map_err(|e| Error::cli(format!("Failed to read entry: {}", e)))?;
            let path = entry.path();
            let metadata = entry.metadata().map_err(|e| Error::cli(format!("Failed to read metadata: {}", e)))?;
            
            let name = path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            
            if metadata.is_dir() {
                dirs.push(name.to_string());
            } else {
                files.push(name.to_string());
            }
        }
        
        // Sort alphabetically
        dirs.sort();
        files.sort();
        
        // Print directories first
        for dir in dirs {
            println!("{} {}", "dir".blue().bold(), dir.blue());
        }
        
        // Then print files
        for file in files {
            // Color based on file extension
            let colored_name = if file.ends_with(".turb") {
                file.green().to_string()
            } else if file.ends_with(".txt") || file.ends_with(".md") {
                file.cyan().to_string()
            } else if file.ends_with(".json") || file.ends_with(".yaml") || file.ends_with(".yml") {
                file.yellow().to_string()
            } else {
                file.normal().to_string()
            };
            
            println!("{} {}", "file".normal(), colored_name);
        }
        
        println!();
        Ok(true)
    }
    
    /// Print current working directory
    fn print_working_directory(&self) -> Result<bool> {
        let dir_path = match &self.working_directory {
            Some(path) => Path::new(path).to_path_buf(),
            None => std::env::current_dir().map_err(|e| Error::cli(format!("Failed to get current directory: {}", e)))?,
        };
        
        println!("{}", dir_path.display());
        Ok(true)
    }
    
    /// Run a Turbulance file
    fn run_file(&mut self, filename: &str) -> Result<bool> {
        self.load_file(filename)
    }
    
    /// Debug a Turbulance file (run with additional information)
    fn debug_file(&mut self, filename: &str) -> Result<bool> {
        let resolved_path = self.resolve_path(filename);
        
        let content = fs::read_to_string(&resolved_path)
            .map_err(|e| Error::cli(format!("Failed to read file: {}", e)))?;

        println!("{} {}", "Debugging file:".green(), resolved_path.display());
        
        // Print code with line numbers
        println!("\n{}", "Code:".bright_black());
        for (i, line) in content.lines().enumerate() {
            println!("{:4}: {}", (i+1).to_string().bright_black(), line);
        }
        println!();
        
        // Create error reporter with source
        let reporter = ErrorReporter::new().with_source(content.clone());
        
        // Begin execution tracking
        self.context.begin_execution();
        
        // Time the execution
        let start_time = Instant::now();
        
        // Run the code
        let result = turbulance::run_with_context(&content, &mut self.context);
        
        // End execution tracking
        self.context.end_execution();
        
        let duration = start_time.elapsed();
        
        // Print execution metrics
        println!("\n{}", "Execution metrics:".bright_black());
        println!("Execution time: {:?}", duration);
        println!("Call stack: {}", self.context.get_call_stack());
        
        // Print performance report
        println!("{}", self.context.get_performance_report());
        
        match result {
            Ok(output) => {
                println!("\n{}", "Output:".green());
                if !output.is_empty() {
                    println!("{}", output);
                } else {
                    println!("(No output)");
                }
            }
            Err(err) => {
                println!("\n{}", "Error:".red().bold());
                self.print_error(&err);
            }
        }
        
        Ok(true)
    }
    
    /// Show all variables in the current context
    fn show_variables(&self) -> Result<bool> {
        // This is a simplified implementation
        println!("Variables in current context:");
        println!("(Detailed implementation would iterate through context.variables)");
        Ok(true)
    }
    
    /// Show all functions in the current context
    fn show_functions(&self) -> Result<bool> {
        // This is a simplified implementation
        println!("Functions in current context:");
        println!("(Detailed implementation would iterate through context.functions)");
        Ok(true)
    }
    
    /// Time the execution of a code snippet
    fn time_execution(&mut self, code: &str) -> Result<bool> {
        println!("Timing execution of: {}", code);
        
        // Begin execution tracking
        self.context.begin_execution();
        
        // Time the execution
        let start_time = Instant::now();
        
        // Run the code
        let result = turbulance::run_with_context(code, &mut self.context);
        
        // Calculate elapsed time
        let duration = start_time.elapsed();
        
        // End execution tracking
        self.context.end_execution();
        
        // Print timing information
        println!("Execution time: {:?}", duration);
        
        // Print result
        match result {
            Ok(output) => {
                if !output.is_empty() {
                    println!("Output: {}", output);
                }
            }
            Err(err) => {
                self.print_error(&err);
            }
        }
        
        Ok(true)
    }
    
    /// Resolve a path relative to the working directory
    fn resolve_path(&self, path: &str) -> std::path::PathBuf {
        let path_buf = Path::new(path);
        
        if path_buf.is_absolute() {
            path_buf.to_path_buf()
        } else {
            match &self.working_directory {
                Some(wd) => Path::new(wd).join(path),
                None => path_buf.to_path_buf(),
            }
        }
    }
    
    /// Save the current session to disk
    fn save_session(&self) -> Result<()> {
        // This is a simplified implementation that would need more detail
        // to fully serialize the context
        let session = ReplSession {
            variables: Vec::new(),  // Would iterate context.variables 
            functions: Vec::new(),  // Would iterate context.functions
            history: self.history.clone(),
            working_directory: self.working_directory.clone(),
        };
        
        let session_json = serde_json::to_string_pretty(&session)
            .map_err(|e| Error::cli(format!("Failed to serialize session: {}", e)))?;
        
        fs::write(&self.session_file, session_json)
            .map_err(|e| Error::cli(format!("Failed to write session file: {}", e)))?;
        
        Ok(())
    }
} 