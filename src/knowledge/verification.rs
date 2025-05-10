use std::collections::HashMap;
use crate::knowledge::{FactVerification, KnowledgeDomain};
use crate::text_unit::utils::string_similarity;

/// Fact verifier for checking the truthfulness of statements
pub struct FactVerifier {
    /// Name of this verifier
    name: String,
    /// Fact database (statement -> (is_factual, confidence, source))
    fact_database: HashMap<String, (bool, f64, String)>,
    /// Domains this verifier supports
    supported_domains: Vec<KnowledgeDomain>,
}

impl FactVerifier {
    /// Create a new fact verifier
    pub fn new(name: &str) -> Self {
        let mut verifier = Self {
            name: name.to_string(),
            fact_database: HashMap::new(),
            supported_domains: vec![KnowledgeDomain::General],
        };
        
        // Add some sample fact verifications
        verifier.add_fact(
            "Rust was first released in 2010.",
            true,
            0.95,
            "Official Rust documentation",
        );
        
        verifier.add_fact(
            "Python is a compiled language.",
            false,
            0.9,
            "Python language specification",
        );
        
        verifier.add_fact(
            "The Earth is flat.",
            false,
            0.99,
            "Scientific consensus",
        );
        
        verifier
    }
    
    /// Add a fact to the verifier database
    pub fn add_fact(&mut self, statement: &str, is_factual: bool, confidence: f64, source: &str) {
        self.fact_database.insert(
            statement.to_lowercase(),
            (is_factual, confidence, source.to_string()),
        );
    }
    
    /// Verify a statement
    pub fn verify(&self, statement: &str) -> Option<FactVerification> {
        // Check for exact match in the database
        if let Some((is_factual, confidence, source)) = self.fact_database.get(&statement.to_lowercase()) {
            return Some(FactVerification {
                is_factual: *is_factual,
                confidence: *confidence,
                evidence: Some(format!("Based on stored fact verification from {}", source)),
                source: source.clone(),
            });
        }
        
        // Check for partial matches
        let mut best_match: Option<(f64, &(bool, f64, String))> = None;
        
        for (stored_statement, fact_info) in &self.fact_database {
            let similarity = string_similarity(statement, stored_statement);
            
            // If similarity is high enough, consider this a potential match
            if similarity > 0.8 {
                if let Some((best_similarity, _)) = best_match {
                    if similarity > best_similarity {
                        best_match = Some((similarity, fact_info));
                    }
                } else {
                    best_match = Some((similarity, fact_info));
                }
            }
        }
        
        // If we found a good match, return a result with adjusted confidence
        if let Some((similarity, (is_factual, confidence, source))) = best_match {
            // Adjust confidence based on similarity
            let adjusted_confidence = confidence * similarity;
            
            return Some(FactVerification {
                is_factual: *is_factual,
                confidence: adjusted_confidence,
                evidence: Some(format!(
                    "Based on similar statement with {}% match. Original confidence: {}",
                    (similarity * 100.0) as u32,
                    confidence
                )),
                source: source.clone(),
            });
        }
        
        // No match found
        None
    }
    
    /// Add support for a domain
    pub fn add_supported_domain(&mut self, domain: KnowledgeDomain) {
        if !self.supported_domains.contains(&domain) {
            self.supported_domains.push(domain);
        }
    }
    
    /// Check if this verifier supports a domain
    pub fn supports_domain(&self, domain: &KnowledgeDomain) -> bool {
        self.supported_domains.contains(domain)
    }
    
    /// Get the name of this verifier
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// More sophisticated FactVerifier that can compose multiple sources
pub struct CompositeFactVerifier {
    /// Name of this verifier
    name: String,
    /// Component verifiers
    verifiers: Vec<FactVerifier>,
}

impl CompositeFactVerifier {
    /// Create a new composite fact verifier
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            verifiers: Vec::new(),
        }
    }
    
    /// Add a component verifier
    pub fn add_verifier(&mut self, verifier: FactVerifier) {
        self.verifiers.push(verifier);
    }
    
    /// Verify a statement using all available verifiers
    pub fn verify(&self, statement: &str, domain: Option<KnowledgeDomain>) -> Option<FactVerification> {
        let mut results = Vec::new();
        
        // Query all verifiers
        for verifier in &self.verifiers {
            // If domain is specified, only use verifiers that support it
            if let Some(ref d) = domain {
                if !verifier.supports_domain(d) {
                    continue;
                }
            }
            
            if let Some(result) = verifier.verify(statement) {
                results.push(result);
            }
        }
        
        if results.is_empty() {
            return None;
        }
        
        // Aggregate results using weighted average based on confidence
        let total_confidence: f64 = results.iter().map(|r| r.confidence).sum();
        
        if total_confidence == 0.0 {
            return None; // Avoid division by zero
        }
        
        // Calculate weighted agreement on factuality
        let weighted_factual: f64 = results.iter()
            .map(|r| if r.is_factual { r.confidence } else { 0.0 })
            .sum();
        
        let is_factual = (weighted_factual / total_confidence) >= 0.5;
        
        // Calculate aggregate confidence
        // Higher when verifiers agree, lower when they disagree
        let agreement_factor = if results.len() > 1 {
            let agreement = results.iter()
                .filter(|r| r.is_factual == is_factual)
                .count() as f64 / results.len() as f64;
            0.5 + (0.5 * agreement) // Scale agreement to 0.5-1.0 range
        } else {
            1.0 // Only one verifier, so full agreement
        };
        
        let average_confidence = total_confidence / results.len() as f64;
        let final_confidence = average_confidence * agreement_factor;
        
        // Collect sources and evidence
        let sources: Vec<String> = results.iter()
            .map(|r| r.source.clone())
            .collect();
        
        let evidence: Vec<String> = results.iter()
            .filter_map(|r| r.evidence.clone())
            .collect();
        
        Some(FactVerification {
            is_factual,
            confidence: final_confidence,
            evidence: if evidence.is_empty() {
                None
            } else {
                Some(evidence.join(" | "))
            },
            source: sources.join(", "),
        })
    }
    
    /// Get the name of this composite verifier
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fact_verifier_exact_match() {
        let verifier = FactVerifier::new("Test Verifier");
        
        let result = verifier.verify("Rust was first released in 2010.");
        assert!(result.is_some());
        
        let verification = result.unwrap();
        assert_eq!(verification.is_factual, true);
        assert!(verification.confidence > 0.9);
    }
    
    #[test]
    fn test_fact_verifier_similar_match() {
        let verifier = FactVerifier::new("Test Verifier");
        
        let result = verifier.verify("Rust was released in 2010.");
        assert!(result.is_some());
        
        let verification = result.unwrap();
        assert_eq!(verification.is_factual, true);
        // Confidence should be adjusted down due to partial match
        assert!(verification.confidence < 0.95);
    }
    
    #[test]
    fn test_fact_verifier_no_match() {
        let verifier = FactVerifier::new("Test Verifier");
        
        let result = verifier.verify("JavaScript was released in 1995.");
        assert!(result.is_none());
    }
    
    // Use the tests for string_similarity from the utils module
    
    #[test]
    fn test_composite_verifier() {
        let mut verifier1 = FactVerifier::new("Verifier 1");
        verifier1.add_fact(
            "Water boils at 100 degrees Celsius at sea level.",
            true,
            0.95,
            "Physics textbook",
        );
        
        let mut verifier2 = FactVerifier::new("Verifier 2");
        verifier2.add_fact(
            "Water boils at 100 degrees Celsius.",
            true,
            0.9,
            "Chemistry reference",
        );
        
        let mut composite = CompositeFactVerifier::new("Composite Verifier");
        composite.add_verifier(verifier1);
        composite.add_verifier(verifier2);
        
        let result = composite.verify("Water boils at 100 degrees Celsius at sea level.", None);
        assert!(result.is_some());
        
        let verification = result.unwrap();
        assert_eq!(verification.is_factual, true);
        assert!(verification.confidence > 0.9);
    }
} 