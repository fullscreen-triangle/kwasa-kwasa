// General utility functions for Kwasa-Kwasa

use std::path::Path;
use std::fs;

/// Checks if a file exists
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Reads a file to string
pub fn read_file(path: &str) -> Result<String, std::io::Error> {
    fs::read_to_string(path)
}

/// Writes a string to a file
pub fn write_file(path: &str, content: &str) -> Result<(), std::io::Error> {
    fs::write(path, content)
}

/// Ensures a directory exists, creating it if necessary
pub fn ensure_dir(path: &str) -> Result<(), std::io::Error> {
    if !Path::new(path).exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Calculates cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut a_norm = 0.0;
    let mut b_norm = 0.0;
    
    for i in 0..a.len() {
        dot_product += a[i] as f32 * b[i] as f32;
        a_norm += a[i] as f32 * a[i] as f32;
        b_norm += b[i] as f32 * b[i] as f32;
    }
    
    a_norm = a_norm.sqrt();
    b_norm = b_norm.sqrt();
    
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    
    dot_product / (a_norm * b_norm)
}

/// Get file extension from a path
pub fn get_extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase())
}

/// Normalize a path string
pub fn normalize_path(path: &str) -> String {
    let path = Path::new(path);
    match path.to_str() {
        Some(p) => p.replace("\\", "/"),
        None => path.to_string_lossy().to_string(),
    }
}

/// Join path segments
pub fn join_paths(base: &str, paths: &[&str]) -> String {
    let mut path = Path::new(base).to_path_buf();
    for p in paths {
        path = path.join(p);
    }
    normalize_path(&path.to_string_lossy())
} 