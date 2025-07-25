use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;

/// Print values to the console
pub fn print(args: Vec<Value>) -> Result<Value> {
    let mut output = String::new();
    
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            output.push(' ');
        }
        
        let formatted = format_value_for_display(arg);
        output.push_str(&formatted);
    }
    
    println!("{}", output);
    Ok(Value::Null)
}

/// Get the length of a collection or string
pub fn len(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "len requires exactly 1 argument".to_string(),
        });
    }

    let length = match &args[0] {
        Value::String(s) => s.chars().count() as f64, // Use char count for Unicode safety
        Value::TextUnit(tu) => tu.content.chars().count() as f64,
        Value::Array(arr) => arr.len() as f64,
        Value::List(list) => list.len() as f64,
        Value::Map(map) => map.len() as f64,
        Value::Object(obj) => obj.len() as f64,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "len can only be applied to strings, arrays, lists, maps, or objects".to_string(),
        }),
    };

    Ok(Value::Number(length))
}

/// Get the type of a value as a string
pub fn typeof_fn(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "typeof requires exactly 1 argument".to_string(),
        });
    }

    let type_name = match &args[0] {
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Boolean(_) => "boolean",
        Value::TextUnit(_) => "text_unit",
        Value::Array(_) => "array",
        Value::List(_) => "list",
        Value::Map(_) => "map",
        Value::Object(_) => "object",
        Value::Function(_) => "function",
        Value::NativeFunction(_) => "native_function",
        Value::Module(_) => "module",
        Value::Null => "null",
    };

    Ok(Value::String(type_name.to_string()))
}

/// Convert a value to a string representation
pub fn to_string(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "to_string requires exactly 1 argument".to_string(),
        });
    }

    let string_repr = format_value_for_display(&args[0]);
    Ok(Value::String(string_repr))
}

/// Convert a string to a number
pub fn to_number(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "to_number requires exactly 1 argument".to_string(),
        });
    }

    match &args[0] {
        Value::String(s) => {
            match s.trim().parse::<f64>() {
                Ok(n) => Ok(Value::Number(n)),
                Err(_) => Err(TurbulanceError::RuntimeError {
                    message: format!("Cannot convert '{}' to number", s),
                }),
            }
        },
        Value::Number(n) => Ok(Value::Number(*n)),
        Value::Boolean(b) => Ok(Value::Number(if *b { 1.0 } else { 0.0 })),
        _ => Err(TurbulanceError::RuntimeError {
            message: "Cannot convert value to number".to_string(),
        }),
    }
}

/// Convert a value to a boolean
pub fn to_boolean(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "to_boolean requires exactly 1 argument".to_string(),
        });
    }

    let boolean_value = is_truthy(&args[0]);
    Ok(Value::Boolean(boolean_value))
}

/// Check if a value is null
pub fn is_null(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "is_null requires exactly 1 argument".to_string(),
        });
    }

    let is_null = matches!(&args[0], Value::Null);
    Ok(Value::Boolean(is_null))
}

/// Check if a value is defined (not null)
pub fn is_defined(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "is_defined requires exactly 1 argument".to_string(),
        });
    }

    let is_defined = !matches!(&args[0], Value::Null);
    Ok(Value::Boolean(is_defined))
}

/// Get the keys of a map or object
pub fn keys(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "keys requires exactly 1 argument".to_string(),
        });
    }

    match &args[0] {
        Value::Map(map) => {
            let key_list: Vec<Value> = map.keys()
                .map(|k| Value::String(k.clone()))
                .collect();
            Ok(Value::List(key_list))
        },
        Value::Object(obj) => {
            let key_list: Vec<Value> = obj.keys()
                .map(|k| Value::String(k.clone()))
                .collect();
            Ok(Value::List(key_list))
        },
        _ => Err(TurbulanceError::RuntimeError {
            message: "keys can only be applied to maps or objects".to_string(),
        }),
    }
}

/// Get the values of a map or object
pub fn values(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "values requires exactly 1 argument".to_string(),
        });
    }

    match &args[0] {
        Value::Map(map) => {
            let value_list: Vec<Value> = map.values().cloned().collect();
            Ok(Value::List(value_list))
        },
        Value::Object(obj) => {
            let value_list: Vec<Value> = obj.values().cloned().collect();
            Ok(Value::List(value_list))
        },
        _ => Err(TurbulanceError::RuntimeError {
            message: "values can only be applied to maps or objects".to_string(),
        }),
    }
}

/// Check if a map or object has a specific key
pub fn has_key(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "has_key requires exactly 2 arguments (collection, key)".to_string(),
        });
    }

    let key = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "has_key second argument must be a string".to_string(),
        }),
    };

    match &args[0] {
        Value::Map(map) => {
            let has_key = map.contains_key(key);
            Ok(Value::Boolean(has_key))
        },
        Value::Object(obj) => {
            let has_key = obj.contains_key(key);
            Ok(Value::Boolean(has_key))
        },
        _ => Err(TurbulanceError::RuntimeError {
            message: "has_key first argument must be a map or object".to_string(),
        }),
    }
}

/// Get a value from a map with an optional default
pub fn get(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "get requires 2-3 arguments (collection, key, optional_default)".to_string(),
        });
    }

    let key = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "get second argument must be a string".to_string(),
        }),
    };

    let default_value = if args.len() > 2 {
        args[2].clone()
    } else {
        Value::Null
    };

    match &args[0] {
        Value::Map(map) => {
            let value = map.get(key).cloned().unwrap_or(default_value);
            Ok(value)
        },
        Value::Object(obj) => {
            let value = obj.get(key).cloned().unwrap_or(default_value);
            Ok(value)
        },
        _ => Err(TurbulanceError::RuntimeError {
            message: "get first argument must be a map or object".to_string(),
        }),
    }
}

/// Merge two maps or objects
pub fn merge(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "merge requires exactly 2 arguments".to_string(),
        });
    }

    match (&args[0], &args[1]) {
        (Value::Map(map1), Value::Map(map2)) => {
            let mut merged = map1.clone();
            for (key, value) in map2 {
                merged.insert(key.clone(), value.clone());
            }
            Ok(Value::Map(merged))
        },
        (Value::Object(obj1), Value::Object(obj2)) => {
            let mut merged = obj1.clone();
            for (key, value) in obj2 {
                merged.insert(key.clone(), value.clone());
            }
            Ok(Value::Object(merged))
        },
        _ => Err(TurbulanceError::RuntimeError {
            message: "merge arguments must be maps or objects of the same type".to_string(),
        }),
    }
}

/// Join an array of strings with a separator
pub fn join(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "join requires exactly 2 arguments (array, separator)".to_string(),
        });
    }

    let separator = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "join second argument must be a string".to_string(),
        }),
    };

    let strings = match &args[0] {
        Value::Array(arr) | Value::List(arr) => {
            let mut string_parts = Vec::new();
            for item in arr {
                string_parts.push(format_value_for_display(item));
            }
            string_parts
        },
        _ => return Err(TurbulanceError::RuntimeError {
            message: "join first argument must be an array or list".to_string(),
        }),
    };

    let joined = strings.join(separator);
    Ok(Value::String(joined))
}

/// Split a string by a separator
pub fn split(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "split requires exactly 2 arguments (string, separator)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "split first argument must be a string or TextUnit".to_string(),
        }),
    };

    let separator = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "split second argument must be a string".to_string(),
        }),
    };

    let parts: Vec<Value> = text.split(separator)
        .map(|s| Value::String(s.to_string()))
        .collect();

    Ok(Value::List(parts))
}

/// Get a substring
pub fn substring(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(TurbulanceError::RuntimeError {
            message: "substring requires 2-3 arguments (string, start, optional_end)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "substring first argument must be a string or TextUnit".to_string(),
        }),
    };

    let start = match &args[1] {
        Value::Number(n) => *n as usize,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "substring second argument must be a number".to_string(),
        }),
    };

    let end = if args.len() > 2 {
        match &args[2] {
            Value::Number(n) => Some(*n as usize),
            _ => return Err(TurbulanceError::RuntimeError {
                message: "substring third argument must be a number".to_string(),
            }),
        }
    } else {
        None
    };

    let chars: Vec<char> = text.chars().collect();
    let start_idx = start.min(chars.len());
    let end_idx = end.unwrap_or(chars.len()).min(chars.len());

    if start_idx > end_idx {
        return Ok(Value::String(String::new()));
    }

    let substring: String = chars[start_idx..end_idx].iter().collect();
    Ok(Value::String(substring))
}

/// Convert string to uppercase
pub fn to_upper(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "to_upper requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s.to_uppercase(),
        Value::TextUnit(tu) => tu.content.to_uppercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "to_upper requires a string or TextUnit".to_string(),
        }),
    };

    Ok(Value::String(text))
}

/// Convert string to lowercase
pub fn to_lower(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "to_lower requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s.to_lowercase(),
        Value::TextUnit(tu) => tu.content.to_lowercase(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "to_lower requires a string or TextUnit".to_string(),
        }),
    };

    Ok(Value::String(text))
}

/// Trim whitespace from a string
pub fn trim(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "trim requires exactly 1 argument".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s.trim().to_string(),
        Value::TextUnit(tu) => tu.content.trim().to_string(),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "trim requires a string or TextUnit".to_string(),
        }),
    };

    Ok(Value::String(text))
}

/// Get the minimum value from a list of numbers
pub fn min(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "min requires at least 1 argument".to_string(),
        });
    }

    let mut min_val = f64::INFINITY;
    let mut found_number = false;

    for arg in &args {
        match arg {
            Value::Number(n) => {
                min_val = min_val.min(*n);
                found_number = true;
            },
            Value::List(list) | Value::Array(list) => {
                for item in list {
                    if let Value::Number(n) = item {
                        min_val = min_val.min(*n);
                        found_number = true;
                    }
                }
            },
            _ => return Err(TurbulanceError::RuntimeError {
                message: "min can only be applied to numbers or lists of numbers".to_string(),
            }),
        }
    }

    if !found_number {
        return Err(TurbulanceError::RuntimeError {
            message: "min requires at least one numeric value".to_string(),
        });
    }

    Ok(Value::Number(min_val))
}

/// Get the maximum value from a list of numbers
pub fn max(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "max requires at least 1 argument".to_string(),
        });
    }

    let mut max_val = f64::NEG_INFINITY;
    let mut found_number = false;

    for arg in &args {
        match arg {
            Value::Number(n) => {
                max_val = max_val.max(*n);
                found_number = true;
            },
            Value::List(list) | Value::Array(list) => {
                for item in list {
                    if let Value::Number(n) = item {
                        max_val = max_val.max(*n);
                        found_number = true;
                    }
                }
            },
            _ => return Err(TurbulanceError::RuntimeError {
                message: "max can only be applied to numbers or lists of numbers".to_string(),
            }),
        }
    }

    if !found_number {
        return Err(TurbulanceError::RuntimeError {
            message: "max requires at least one numeric value".to_string(),
        });
    }

    Ok(Value::Number(max_val))
}

/// Calculate the sum of numbers
pub fn sum(args: Vec<Value>) -> Result<Value> {
    let mut total = 0.0;
    let mut found_number = false;

    for arg in &args {
        match arg {
            Value::Number(n) => {
                total += n;
                found_number = true;
            },
            Value::List(list) | Value::Array(list) => {
                for item in list {
                    if let Value::Number(n) = item {
                        total += n;
                        found_number = true;
                    }
                }
            },
            _ => return Err(TurbulanceError::RuntimeError {
                message: "sum can only be applied to numbers or lists of numbers".to_string(),
            }),
        }
    }

    if !found_number {
        total = 0.0; // Sum of empty set is 0
    }

    Ok(Value::Number(total))
}

/// Calculate the average of numbers
pub fn average(args: Vec<Value>) -> Result<Value> {
    let mut total = 0.0;
    let mut count = 0;

    for arg in &args {
        match arg {
            Value::Number(n) => {
                total += n;
                count += 1;
            },
            Value::List(list) | Value::Array(list) => {
                for item in list {
                    if let Value::Number(n) = item {
                        total += n;
                        count += 1;
                    }
                }
            },
            _ => return Err(TurbulanceError::RuntimeError {
                message: "average can only be applied to numbers or lists of numbers".to_string(),
            }),
        }
    }

    if count == 0 {
        return Err(TurbulanceError::RuntimeError {
            message: "average requires at least one numeric value".to_string(),
        });
    }

    Ok(Value::Number(total / count as f64))
}

/// Round a number to the nearest integer
pub fn round(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "round requires exactly 1 argument".to_string(),
        });
    }

    match &args[0] {
        Value::Number(n) => Ok(Value::Number(n.round())),
        _ => Err(TurbulanceError::RuntimeError {
            message: "round can only be applied to numbers".to_string(),
        }),
    }
}

/// Get the absolute value of a number
pub fn abs(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "abs requires exactly 1 argument".to_string(),
        });
    }

    match &args[0] {
        Value::Number(n) => Ok(Value::Number(n.abs())),
        _ => Err(TurbulanceError::RuntimeError {
            message: "abs can only be applied to numbers".to_string(),
        }),
    }
}

/// Check if a string contains a substring
pub fn contains(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "contains requires exactly 2 arguments (string, substring)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contains first argument must be a string or TextUnit".to_string(),
        }),
    };

    let substring = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "contains second argument must be a string".to_string(),
        }),
    };

    let contains = text.contains(substring);
    Ok(Value::Boolean(contains))
}

/// Check if a string starts with a prefix
pub fn starts_with(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "starts_with requires exactly 2 arguments (string, prefix)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "starts_with first argument must be a string or TextUnit".to_string(),
        }),
    };

    let prefix = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "starts_with second argument must be a string".to_string(),
        }),
    };

    let starts_with = text.starts_with(prefix);
    Ok(Value::Boolean(starts_with))
}

/// Check if a string ends with a suffix
pub fn ends_with(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "ends_with requires exactly 2 arguments (string, suffix)".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ends_with first argument must be a string or TextUnit".to_string(),
        }),
    };

    let suffix = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ends_with second argument must be a string".to_string(),
        }),
    };

    let ends_with = text.ends_with(suffix);
    Ok(Value::Boolean(ends_with))
}

// Helper functions

/// Helper function to determine if a value is truthy
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Boolean(b) => *b,
        Value::Number(n) => *n != 0.0 && !n.is_nan(),
        Value::String(s) => !s.is_empty(),
        Value::TextUnit(tu) => !tu.content.is_empty(),
        Value::Array(arr) => !arr.is_empty(),
        Value::List(list) => !list.is_empty(),
        Value::Map(map) => !map.is_empty(),
        Value::Object(obj) => !obj.is_empty(),
        Value::Function(_) => true,
        Value::NativeFunction(_) => true,
        Value::Module(_) => true,
        Value::Null => false,
    }
}

/// Helper function to format values for display
fn format_value_for_display(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        },
        Value::Boolean(b) => b.to_string(),
        Value::TextUnit(tu) => tu.content.clone(),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_value_for_display).collect();
            format!("[{}]", items.join(", "))
        },
        Value::List(list) => {
            let items: Vec<String> = list.iter().map(format_value_for_display).collect();
            format!("[{}]", items.join(", "))
        },
        Value::Map(map) => {
            let items: Vec<String> = map.iter()
                .map(|(k, v)| format!("{}: {}", k, format_value_for_display(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        },
        Value::Object(obj) => {
            let items: Vec<String> = obj.iter()
                .map(|(k, v)| format!("{}: {}", k, format_value_for_display(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        },
        Value::Function(_) => "<function>".to_string(),
        Value::NativeFunction(_) => "<native_function>".to_string(),
        Value::Module(_) => "<module>".to_string(),
        Value::Null => "null".to_string(),
    }
} 