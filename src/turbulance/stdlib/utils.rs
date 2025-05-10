use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;

/// Prints a value to the console/output
pub fn print(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        println!();
        return Ok(Value::Null);
    }
    
    // Print each argument
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        
        match arg {
            Value::String(s) => print!("{}", s),
            Value::Number(n) => {
                // Format numbers nicely - if integer, print without decimal
                if n.fract() == 0.0 {
                    print!("{}", n.trunc() as i64);
                } else {
                    print!("{}", n);
                }
            },
            Value::Boolean(b) => print!("{}", b),
            Value::Function(_) => print!("<function>"),
            Value::NativeFunction(_) => print!("<native function>"),
            Value::Array(arr) => {
                print!("[");
                for (j, item) in arr.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    // Recursive call to handle nested values, but don't print
                    let _ = print(vec![item.clone()]);
                }
                print!("]");
            },
            Value::Object(obj) => {
                print!("{{");
                let mut first = true;
                for (key, val) in obj {
                    if !first {
                        print!(", ");
                    }
                    print!("{}: ", key);
                    // Recursive call for the value
                    let _ = print(vec![val.clone()]);
                    first = false;
                }
                print!("}}");
            },
            Value::Null => print!("null"),
        }
    }
    
    println!();
    Ok(Value::Null)
}

/// Returns the length of a collection (string, array, object)
pub fn len(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "len() requires one argument".to_string(),
        });
    }

    // Get length based on value type
    let length = match &args[0] {
        Value::String(s) => s.len(),
        Value::Array(arr) => arr.len(),
        Value::Object(obj) => obj.len(),
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "len() can only be called on strings, arrays, or objects".to_string(),
            });
        }
    };
    
    Ok(Value::Number(length as f64))
}

/// Returns the type of a value
pub fn typeof_fn(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "typeof() requires one argument".to_string(),
        });
    }

    // Get the type name
    let type_name = match &args[0] {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Boolean(_) => "boolean",
        Value::Function(_) => "function",
        Value::NativeFunction(_) => "function",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        Value::Null => "null",
    };
    
    Ok(Value::String(type_name.to_string()))
}

/// Extends standard library with additional utility functions
pub fn register_extended_utils() -> HashMap<&'static str, fn(Vec<Value>) -> Result<Value>> {
    let mut utils = HashMap::new();
    
    // Register core utilities
    utils.insert("print", print);
    utils.insert("len", len);
    utils.insert("typeof", typeof_fn);
    
    // Add additional utilities
    utils.insert("json_stringify", json_stringify);
    utils.insert("json_parse", json_parse);
    utils.insert("time", time_fn);
    
    utils
}

/// Converts a value to its JSON string representation
pub fn json_stringify(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "json_stringify() requires one argument".to_string(),
        });
    }
    
    // Convert the value to a JSON string
    let json_str = match &args[0] {
        Value::String(s) => format!("\"{}\"", s.replace("\"", "\\\"")),
        Value::Number(n) => {
            if n.is_finite() {
                n.to_string()
            } else {
                "null".to_string()
            }
        },
        Value::Boolean(b) => b.to_string(),
        Value::Array(arr) => {
            let mut json = String::from("[");
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    json.push_str(", ");
                }
                
                match json_stringify(vec![item.clone()]) {
                    Ok(Value::String(s)) => json.push_str(&s),
                    _ => json.push_str("null"),
                }
            }
            json.push_str("]");
            json
        },
        Value::Object(obj) => {
            let mut json = String::from("{");
            let mut first = true;
            for (key, val) in obj {
                if !first {
                    json.push_str(", ");
                }
                
                // Key is always a string in JSON
                json.push_str(&format!("\"{}\":", key));
                
                // Recursively stringify the value
                match json_stringify(vec![val.clone()]) {
                    Ok(Value::String(s)) => json.push_str(&s),
                    _ => json.push_str("null"),
                }
                
                first = false;
            }
            json.push_str("}");
            json
        },
        Value::Function(_) | Value::NativeFunction(_) => "null".to_string(),
        Value::Null => "null".to_string(),
    };
    
    Ok(Value::String(json_str))
}

/// Parses a JSON string into a value
pub fn json_parse(args: Vec<Value>) -> Result<Value> {
    // Validate arguments
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "json_parse() requires one argument".to_string(),
        });
    }
    
    // Extract JSON string from argument
    let json_str = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(TurbulanceError::RuntimeError {
                message: "json_parse() argument must be a string".to_string(),
            });
        }
    };
    
    // This would use serde_json in a real implementation
    // For this simple example, we'll just handle a few basic cases
    let trimmed = json_str.trim();
    
    // Parse simple values
    match trimmed {
        "null" => return Ok(Value::Null),
        "true" => return Ok(Value::Boolean(true)),
        "false" => return Ok(Value::Boolean(false)),
        _ => {}
    }
    
    // Parse a number
    if let Ok(n) = trimmed.parse::<f64>() {
        return Ok(Value::Number(n));
    }
    
    // Parse a string
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let content = &trimmed[1..trimmed.len()-1];
        return Ok(Value::String(content.replace("\\\"", "\"").to_string()));
    }
    
    // Parse an array
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        // Very simplified parsing - in reality, this would use a proper JSON parser
        if trimmed == "[]" {
            return Ok(Value::Array(vec![]));
        }
        
        // For this example, we'll just return a mock array
        let mock_array = vec![
            Value::String("This is a placeholder".to_string()),
            Value::String("A real implementation would use serde_json".to_string()),
        ];
        return Ok(Value::Array(mock_array));
    }
    
    // Parse an object
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        // Very simplified parsing - in reality, this would use a proper JSON parser
        if trimmed == "{}" {
            return Ok(Value::Object(HashMap::new()));
        }
        
        // For this example, we'll just return a mock object
        let mut mock_obj = HashMap::new();
        mock_obj.insert("message".to_string(), 
                       Value::String("This is a placeholder object".to_string()));
        mock_obj.insert("note".to_string(),
                       Value::String("A real implementation would use serde_json".to_string()));
        return Ok(Value::Object(mock_obj));
    }
    
    // If we get here, the JSON is invalid or too complex for our simple parser
    Err(TurbulanceError::RuntimeError {
        message: "Invalid JSON or unsupported JSON structure".to_string(),
    })
}

/// Returns the current time (Unix timestamp in seconds)
pub fn time_fn(args: Vec<Value>) -> Result<Value> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    // Get current time since UNIX epoch
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    
    Ok(Value::Number(now))
} 