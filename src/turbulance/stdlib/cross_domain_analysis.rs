use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;

/// Calculate statistical enrichment of genomic motifs
pub fn motif_enrichment(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "motif_enrichment() requires genomic_sequence and motif".to_string(),
        });
    }

    let sequence = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be genomic sequence string".to_string(),
        }),
    };

    let motif = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be motif string".to_string(),
        }),
    };

    let enrichment = calculate_motif_enrichment(sequence, motif);
    Ok(Value::Number(enrichment))
}

/// Compute correlation between mass spectral patterns
pub fn spectral_correlation(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "spectral_correlation() requires two spectra".to_string(),
        });
    }

    let spectrum1 = match &args[0] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be spectrum array".to_string(),
        }),
    };

    let spectrum2 = match &args[1] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be spectrum array".to_string(),
        }),
    };

    let correlation = calculate_spectral_correlation(&spectrum1, &spectrum2);
    Ok(Value::Number(correlation))
}

/// Calculate probability of hypothesis given evidence
pub fn evidence_likelihood(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "evidence_likelihood() requires evidence_network and hypothesis".to_string(),
        });
    }

    let evidence_network = match &args[0] {
        Value::Object(obj) => obj,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be evidence network object".to_string(),
        }),
    };

    let hypothesis = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be hypothesis string".to_string(),
        }),
    };

    let likelihood = calculate_evidence_likelihood(evidence_network, hypothesis);
    Ok(Value::Number(likelihood))
}

/// Model how uncertainty propagates through evidence
pub fn uncertainty_propagation(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "uncertainty_propagation() requires evidence_network and node_id".to_string(),
        });
    }

    let evidence_network = match &args[0] {
        Value::Object(obj) => obj,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be evidence network object".to_string(),
        }),
    };

    let node_id = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be node ID string".to_string(),
        }),
    };

    let uncertainty = calculate_uncertainty_propagation(evidence_network, node_id);
    Ok(Value::Number(uncertainty))
}

/// Update belief based on new evidence using Bayes' theorem
pub fn bayesian_update(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "bayesian_update() requires prior_belief and new_evidence".to_string(),
        });
    }

    let prior_belief = match &args[0] {
        Value::Number(n) => *n,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be prior belief probability".to_string(),
        }),
    };

    let new_evidence = match &args[1] {
        Value::Number(n) => *n,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be evidence likelihood".to_string(),
        }),
    };

    let updated_belief = calculate_bayesian_update(prior_belief, new_evidence);
    Ok(Value::Number(updated_belief))
}

/// Calculate confidence intervals for measurements
pub fn confidence_interval(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "confidence_interval() requires measurement and confidence_level".to_string(),
        });
    }

    let measurement = match &args[0] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be measurement array".to_string(),
        }),
    };

    let confidence_level = match &args[1] {
        Value::Number(n) => *n,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be confidence level".to_string(),
        }),
    };

    let (lower, upper) = calculate_confidence_interval(&measurement, confidence_level);
    
    let mut result = HashMap::new();
    result.insert("lower".to_string(), Value::Number(lower));
    result.insert("upper".to_string(), Value::Number(upper));
    result.insert("confidence_level".to_string(), Value::Number(confidence_level));
    
    Ok(Value::Object(result))
}

/// Find correlations between multi-domain datasets
pub fn cross_domain_correlation(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "cross_domain_correlation() requires genomic_data and spectral_data".to_string(),
        });
    }

    let genomic_data = match &args[0] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be genomic data array".to_string(),
        }),
    };

    let spectral_data = match &args[1] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be spectral data array".to_string(),
        }),
    };

    let correlation = calculate_cross_domain_correlation(&genomic_data, &spectral_data);
    Ok(Value::Number(correlation))
}

/// Estimate false discovery rate in pattern matching results
pub fn false_discovery_rate(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "false_discovery_rate() requires matches and null_model".to_string(),
        });
    }

    let matches = match &args[0] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be matches array".to_string(),
        }),
    };

    let null_model = match &args[1] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be null model array".to_string(),
        }),
    };

    let fdr = calculate_false_discovery_rate(&matches, &null_model);
    Ok(Value::Number(fdr))
}

/// Calculate significance through permutation testing
pub fn permutation_significance(args: Vec<Value>) -> Result<Value> {
    if args.len() < 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "permutation_significance() requires observed and randomized".to_string(),
        });
    }

    let observed = match &args[0] {
        Value::Number(n) => *n,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be observed value".to_string(),
        }),
    };

    let randomized = match &args[1] {
        Value::Array(arr) => extract_numbers_from_array(arr)?,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "Second argument must be randomized values array".to_string(),
        }),
    };

    let significance = calculate_permutation_significance(observed, &randomized);
    Ok(Value::Number(significance))
}

// Positional importance analysis functions

/// Calculate importance score based on position within document
pub fn positional_importance(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "positional_importance() requires text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let unit = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "paragraph",
        }
    } else {
        "paragraph"
    };

    let importance_scores = calculate_positional_importance(text, unit);
    let result: Vec<Value> = importance_scores.into_iter()
        .enumerate()
        .map(|(idx, score)| {
            let mut map = HashMap::new();
            map.insert("index".to_string(), Value::Number(idx as f64));
            map.insert("importance".to_string(), Value::Number(score));
            Value::Object(map)
        })
        .collect();

    Ok(Value::Array(result))
}

/// Create heatmap of importance weights across document sections
pub fn section_weight_map(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "section_weight_map() requires document".to_string(),
        });
    }

    let document = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be document string".to_string(),
        }),
    };

    let weight_map = calculate_section_weight_map(document);
    Ok(Value::Object(weight_map))
}

/// Measure text importance based on structural context
pub fn structural_prominence(args: Vec<Value>) -> Result<Value> {
    if args.is_empty() {
        return Err(TurbulanceError::RuntimeError {
            message: "structural_prominence() requires text".to_string(),
        });
    }

    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "First argument must be text string".to_string(),
        }),
    };

    let structure_type = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "heading",
        }
    } else {
        "heading"
    };

    let prominence = calculate_structural_prominence(text, structure_type);
    Ok(Value::Number(prominence))
}

// Helper functions for implementation

fn extract_numbers_from_array(arr: &[Value]) -> Result<Vec<f64>> {
    let mut numbers = Vec::new();
    for value in arr {
        match value {
            Value::Number(n) => numbers.push(*n),
            _ => return Err(TurbulanceError::RuntimeError {
                message: "Array must contain only numbers".to_string(),
            }),
        }
    }
    Ok(numbers)
}

fn calculate_motif_enrichment(sequence: &str, motif: &str) -> f64 {
    let sequence_len = sequence.len();
    let motif_len = motif.len();
    
    if motif_len == 0 || sequence_len < motif_len {
        return 0.0;
    }
    
    let observed_count = sequence.matches(motif).count();
    let possible_positions = sequence_len - motif_len + 1;
    
    // Calculate expected frequency based on nucleotide composition
    let mut char_freqs = HashMap::new();
    for c in sequence.chars() {
        *char_freqs.entry(c).or_insert(0) += 1;
    }
    
    let expected_prob: f64 = motif.chars()
        .map(|c| char_freqs.get(&c).unwrap_or(&0))
        .map(|&count| count as f64 / sequence_len as f64)
        .product();
    
    let expected_count = expected_prob * possible_positions as f64;
    
    if expected_count == 0.0 {
        0.0
    } else {
        observed_count as f64 / expected_count
    }
}

fn calculate_spectral_correlation(spectrum1: &[f64], spectrum2: &[f64]) -> f64 {
    if spectrum1.len() != spectrum2.len() || spectrum1.is_empty() {
        return 0.0;
    }
    
    let n = spectrum1.len() as f64;
    
    let mean1 = spectrum1.iter().sum::<f64>() / n;
    let mean2 = spectrum2.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    
    for i in 0..spectrum1.len() {
        let diff1 = spectrum1[i] - mean1;
        let diff2 = spectrum2[i] - mean2;
        
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    
    let denominator = (sum_sq1 * sum_sq2).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn calculate_evidence_likelihood(evidence_network: &HashMap<String, Value>, hypothesis: &str) -> f64 {
    // Simple implementation - in practice would use more sophisticated Bayesian network
    let mut total_support = 0.0;
    let mut evidence_count = 0;
    
    for (key, value) in evidence_network {
        if key.contains(hypothesis) {
            if let Value::Number(weight) = value {
                total_support += weight;
                evidence_count += 1;
            }
        }
    }
    
    if evidence_count == 0 {
        0.5 // Neutral prior
    } else {
        (total_support / evidence_count as f64).max(0.0).min(1.0)
    }
}

fn calculate_uncertainty_propagation(evidence_network: &HashMap<String, Value>, node_id: &str) -> f64 {
    // Simple uncertainty propagation model
    let mut uncertainty_sum = 0.0;
    let mut connection_count = 0;
    
    for (key, value) in evidence_network {
        if key.contains(node_id) {
            if let Value::Number(uncertainty) = value {
                uncertainty_sum += (1.0 - uncertainty).abs(); // Convert certainty to uncertainty
                connection_count += 1;
            }
        }
    }
    
    if connection_count == 0 {
        1.0 // Maximum uncertainty
    } else {
        uncertainty_sum / connection_count as f64
    }
}

fn calculate_bayesian_update(prior: f64, likelihood: f64) -> f64 {
    // Simplified Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
    // Assuming P(E) = 0.5 for simplicity
    let evidence_prob = 0.5;
    
    let posterior = (likelihood * prior) / evidence_prob;
    posterior.max(0.0).min(1.0)
}

fn calculate_confidence_interval(data: &[f64], confidence_level: f64) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    
    let variance = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (n - 1.0);
    
    let std_dev = variance.sqrt();
    let std_error = std_dev / n.sqrt();
    
    // Using normal approximation (t-distribution would be more accurate for small samples)
    let z_score = match confidence_level {
        level if level >= 0.99 => 2.576,
        level if level >= 0.95 => 1.96,
        level if level >= 0.90 => 1.645,
        _ => 1.96, // Default to 95%
    };
    
    let margin_error = z_score * std_error;
    
    (mean - margin_error, mean + margin_error)
}

fn calculate_cross_domain_correlation(data1: &[f64], data2: &[f64]) -> f64 {
    calculate_spectral_correlation(data1, data2)
}

fn calculate_false_discovery_rate(matches: &[f64], null_model: &[f64]) -> f64 {
    if matches.is_empty() || null_model.is_empty() {
        return 0.0;
    }
    
    let null_mean = null_model.iter().sum::<f64>() / null_model.len() as f64;
    let false_positives = matches.iter().filter(|&&x| x <= null_mean).count();
    
    false_positives as f64 / matches.len() as f64
}

fn calculate_permutation_significance(observed: f64, randomized: &[f64]) -> f64 {
    if randomized.is_empty() {
        return 0.5;
    }
    
    let extreme_count = randomized.iter().filter(|&&x| x >= observed).count();
    extreme_count as f64 / randomized.len() as f64
}

fn calculate_positional_importance(text: &str, unit: &str) -> Vec<f64> {
    let units: Vec<&str> = match unit {
        "sentence" => text.split(&['.', '!', '?']).filter(|s| !s.trim().is_empty()).collect(),
        "paragraph" => text.split("\n\n").filter(|s| !s.trim().is_empty()).collect(),
        "word" => text.split_whitespace().collect(),
        _ => text.split("\n\n").filter(|s| !s.trim().is_empty()).collect(),
    };
    
    let total_units = units.len();
    if total_units == 0 {
        return Vec::new();
    }
    
    units.into_iter()
        .enumerate()
        .map(|(idx, unit)| {
            let position_weight = if total_units == 1 {
                1.0
            } else {
                // Higher weight for beginning and end
                let normalized_pos = idx as f64 / (total_units - 1) as f64;
                if normalized_pos <= 0.2 || normalized_pos >= 0.8 {
                    1.0
                } else {
                    0.7
                }
            };
            
            // Length-based importance
            let length_weight = (unit.len() as f64).log(2.0).max(0.1);
            
            position_weight * length_weight
        })
        .collect()
}

fn calculate_section_weight_map(document: &str) -> HashMap<String, Value> {
    let sections: Vec<&str> = document.split("\n\n").collect();
    let mut weight_map = HashMap::new();
    
    for (idx, section) in sections.iter().enumerate() {
        let importance = calculate_positional_importance(section, "paragraph");
        let avg_importance = if importance.is_empty() {
            0.0
        } else {
            importance.iter().sum::<f64>() / importance.len() as f64
        };
        
        weight_map.insert(
            format!("section_{}", idx),
            Value::Number(avg_importance)
        );
    }
    
    weight_map
}

fn calculate_structural_prominence(text: &str, structure_type: &str) -> f64 {
    match structure_type {
        "heading" => {
            // Count heading indicators
            let heading_indicators = ["#", "=", "-", "*"];
            let lines: Vec<&str> = text.lines().collect();
            
            let heading_count = lines.iter()
                .filter(|line| {
                    heading_indicators.iter().any(|indicator| line.starts_with(indicator))
                })
                .count();
            
            if lines.is_empty() {
                0.0
            } else {
                heading_count as f64 / lines.len() as f64
            }
        },
        "list" => {
            let list_indicators = ["-", "*", "+", "1.", "2.", "3."];
            let lines: Vec<&str> = text.lines().collect();
            
            let list_count = lines.iter()
                .filter(|line| {
                    let trimmed = line.trim();
                    list_indicators.iter().any(|indicator| trimmed.starts_with(indicator))
                })
                .count();
            
            if lines.is_empty() {
                0.0
            } else {
                list_count as f64 / lines.len() as f64
            }
        },
        _ => 0.0,
    }
} 