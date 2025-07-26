use super::*;
use async_trait::async_trait;

/// Default global viability checker implementing the Complexity Coherence Principle
pub struct DefaultGlobalViabilityChecker {
    config: SEntropyConfig,
}

impl DefaultGlobalViabilityChecker {
    pub fn new() -> Self {
        Self {
            config: SEntropyConfig::default(),
        }
    }

    pub fn with_config(config: SEntropyConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl GlobalViabilityChecker for DefaultGlobalViabilityChecker {
    async fn is_globally_viable(
        &self,
        s_values: &SConstantTriDimensional,
        context: &ProblemContext,
    ) -> SEntropyResult<ViabilityResult> {
        // Check if impossible local S values can maintain global viability
        let total_impossibility = self.calculate_total_impossibility(s_values).await?;
        let reality_complexity = self.estimate_reality_complexity(&context.domain).await?;
        let complexity_buffer = reality_complexity / total_impossibility.max(1e-10);

        // Apply Complexity Coherence Principle
        if complexity_buffer > 1000.0 {
            // Sufficient complexity buffer for impossible local S values
            Ok(ViabilityResult {
                viable: true,
                reason: format!(
                    "Global viability maintained through Complexity Coherence Principle: \
                    Reality complexity ({:.2e}) exceeds local impossibility ({:.2e}) \
                    by factor of {:.1e}×, providing sufficient buffer for impossible \
                    local S values to align to viable global S optimization.",
                    reality_complexity, total_impossibility, complexity_buffer
                ),
                confidence_level: 0.95,
            })
        } else if s_values.has_impossible_local_values() && self.config.enable_miracle_engineering {
            // Miracle engineering for impossible values
            Ok(ViabilityResult {
                viable: true,
                reason: format!(
                    "Global viability achieved through systematic miracle engineering: \
                    Impossible local S values (knowledge: {:.3}, time: {:.3}, entropy: {:.3}) \
                    create optimization resonance patterns that transcend traditional \
                    viability constraints through {}× impossibility amplification.",
                    s_values.s_knowledge, s_values.s_time, s_values.s_entropy,
                    self.config.max_impossibility_factor
                ),
                confidence_level: 0.85,
            })
        } else if s_values.global_s_distance() < self.config.global_viability_threshold {
            // Normal viability through low global S-distance
            Ok(ViabilityResult {
                viable: true,
                reason: format!(
                    "Global viability achieved through S-distance minimization: \
                    Global S-distance ({:.6}) below viability threshold ({:.6}).",
                    s_values.global_s_distance(), self.config.global_viability_threshold
                ),
                confidence_level: 0.99,
            })
        } else {
            // Viability failure
            Ok(ViabilityResult {
                viable: false,
                reason: format!(
                    "Global viability failure: Insufficient complexity buffer ({:.2e}) \
                    and global S-distance ({:.6}) exceeds threshold ({:.6}). \
                    Reality complexity cannot absorb local impossibility level.",
                    complexity_buffer, s_values.global_s_distance(), self.config.global_viability_threshold
                ),
                confidence_level: 0.80,
            })
        }
    }
}

impl DefaultGlobalViabilityChecker {
    /// Calculate total impossibility level from S values
    async fn calculate_total_impossibility(&self, s_values: &SConstantTriDimensional) -> SEntropyResult<f64> {
        let mut total_impossibility = 0.0;

        // Add impossibility from negative S values
        if s_values.s_knowledge < 0.0 {
            total_impossibility += s_values.s_knowledge.abs() * 10.0;
        }
        if s_values.s_time < 0.0 {
            total_impossibility += s_values.s_time.abs() * 10.0;
        }
        if s_values.s_entropy < 0.0 {
            total_impossibility += s_values.s_entropy.abs() * 10.0;
        }

        // Add impossibility from infinite S values
        if s_values.s_knowledge.is_infinite() {
            total_impossibility += 1e20;
        }
        if s_values.s_time.is_infinite() {
            total_impossibility += 1e20;
        }
        if s_values.s_entropy.is_infinite() {
            total_impossibility += 1e20;
        }

        // Add impossibility from NaN/paradoxical S values
        if s_values.s_knowledge.is_nan() {
            total_impossibility += 1e25; // Paradoxical values are extremely impossible
        }
        if s_values.s_time.is_nan() {
            total_impossibility += 1e25;
        }
        if s_values.s_entropy.is_nan() {
            total_impossibility += 1e25;
        }

        // If no impossibility detected, use global S-distance as baseline
        if total_impossibility == 0.0 {
            total_impossibility = s_values.global_s_distance();
        }

        Ok(total_impossibility)
    }

    /// Estimate reality complexity for domain-specific viability calculations
    async fn estimate_reality_complexity(&self, domain: &str) -> SEntropyResult<f64> {
        let base_complexity = match domain {
            "quantum_computing" => 1e50, // Quantum state space complexity
            "financial_optimization" => 1e30, // Global market complexity
            "scientific_discovery" => 1e45, // Natural phenomena complexity  
            "business_strategy" => 1e25, // Human behavior complexity
            "personal_development" => 1e20, // Individual psychology complexity
            "ai_system_training" => 1e40, // Neural network complexity
            _ => 1e35, // Default universal complexity
        };

        // Reality is always more complex than we estimate
        Ok(base_complexity * 1.5)
    }
}

/// Enhanced viability checker with domain-specific rules
pub struct EnhancedGlobalViabilityChecker {
    default_checker: DefaultGlobalViabilityChecker,
    domain_rules: std::collections::HashMap<String, DomainViabilityRule>,
}

impl EnhancedGlobalViabilityChecker {
    pub fn new() -> Self {
        let mut domain_rules = std::collections::HashMap::new();
        
        // Quantum computing domain rules
        domain_rules.insert("quantum_computing".to_string(), DomainViabilityRule {
            allow_negative_entropy: true,
            allow_infinite_time: true,
            allow_paradoxical_knowledge: true,
            impossibility_amplification_factor: 10000.0,
            complexity_buffer_requirement: 100.0,
        });

        // Financial optimization domain rules
        domain_rules.insert("financial_optimization".to_string(), DomainViabilityRule {
            allow_negative_entropy: false,
            allow_infinite_time: false,
            allow_paradoxical_knowledge: true,
            impossibility_amplification_factor: 1000.0,
            complexity_buffer_requirement: 500.0,
        });

        // Scientific discovery domain rules
        domain_rules.insert("scientific_discovery".to_string(), DomainViabilityRule {
            allow_negative_entropy: true,
            allow_infinite_time: true,
            allow_paradoxical_knowledge: true,
            impossibility_amplification_factor: 50000.0, // Science allows more impossibility
            complexity_buffer_requirement: 50.0,
        });

        Self {
            default_checker: DefaultGlobalViabilityChecker::new(),
            domain_rules,
        }
    }
}

#[async_trait]
impl GlobalViabilityChecker for EnhancedGlobalViabilityChecker {
    async fn is_globally_viable(
        &self,
        s_values: &SConstantTriDimensional,
        context: &ProblemContext,
    ) -> SEntropyResult<ViabilityResult> {
        // Apply domain-specific rules if available
        if let Some(domain_rule) = self.domain_rules.get(&context.domain) {
            let domain_viability = self.check_domain_viability(s_values, domain_rule).await?;
            if !domain_viability.viable {
                return Ok(domain_viability);
            }
        }

        // Fall back to default viability checking
        self.default_checker.is_globally_viable(s_values, context).await
    }
}

impl EnhancedGlobalViabilityChecker {
    async fn check_domain_viability(
        &self,
        s_values: &SConstantTriDimensional,
        rule: &DomainViabilityRule,
    ) -> SEntropyResult<ViabilityResult> {
        // Check domain-specific constraints
        if s_values.s_entropy < 0.0 && !rule.allow_negative_entropy {
            return Ok(ViabilityResult {
                viable: false,
                reason: "Domain does not allow negative entropy values".to_string(),
                confidence_level: 0.95,
            });
        }

        if s_values.s_time.is_infinite() && !rule.allow_infinite_time {
            return Ok(ViabilityResult {
                viable: false,
                reason: "Domain does not allow infinite time values".to_string(),
                confidence_level: 0.95,
            });
        }

        if s_values.s_knowledge.is_nan() && !rule.allow_paradoxical_knowledge {
            return Ok(ViabilityResult {
                viable: false,
                reason: "Domain does not allow paradoxical knowledge values".to_string(),
                confidence_level: 0.95,
            });
        }

        // Domain-specific viability passed
        Ok(ViabilityResult {
            viable: true,
            reason: "Domain-specific viability constraints satisfied".to_string(),
            confidence_level: 0.90,
        })
    }
}

/// Domain-specific viability rules
#[derive(Debug, Clone)]
pub struct DomainViabilityRule {
    pub allow_negative_entropy: bool,
    pub allow_infinite_time: bool,
    pub allow_paradoxical_knowledge: bool,
    pub impossibility_amplification_factor: f64,
    pub complexity_buffer_requirement: f64,
}

/// Statistical viability checker that learns from solution outcomes
pub struct StatisticalViabilityChecker {
    base_checker: DefaultGlobalViabilityChecker,
    viability_statistics: std::collections::HashMap<String, ViabilityStatistics>,
}

impl StatisticalViabilityChecker {
    pub fn new() -> Self {
        Self {
            base_checker: DefaultGlobalViabilityChecker::new(),
            viability_statistics: std::collections::HashMap::new(),
        }
    }

    /// Record the outcome of a viability prediction
    pub async fn record_outcome(
        &mut self,
        domain: &str,
        predicted_viable: bool,
        actual_outcome: bool,
        s_values: &SConstantTriDimensional,
    ) -> SEntropyResult<()> {
        let stats = self.viability_statistics
            .entry(domain.to_string())
            .or_insert_with(ViabilityStatistics::new);

        stats.record_prediction(predicted_viable, actual_outcome, s_values);
        Ok(())
    }

    /// Get prediction accuracy for domain
    pub fn get_accuracy(&self, domain: &str) -> f64 {
        self.viability_statistics
            .get(domain)
            .map(|stats| stats.accuracy())
            .unwrap_or(0.0)
    }
}

#[async_trait]
impl GlobalViabilityChecker for StatisticalViabilityChecker {
    async fn is_globally_viable(
        &self,
        s_values: &SConstantTriDimensional,
        context: &ProblemContext,
    ) -> SEntropyResult<ViabilityResult> {
        // Use statistical learning to adjust viability predictions
        let base_result = self.base_checker.is_globally_viable(s_values, context).await?;
        
        if let Some(stats) = self.viability_statistics.get(&context.domain) {
            let statistical_adjustment = stats.calculate_adjustment(s_values);
            let adjusted_confidence = (base_result.confidence_level + statistical_adjustment).min(1.0).max(0.0);
            
            Ok(ViabilityResult {
                viable: base_result.viable,
                reason: format!("{} (Statistically adjusted: {:.3})", base_result.reason, statistical_adjustment),
                confidence_level: adjusted_confidence,
            })
        } else {
            Ok(base_result)
        }
    }
}

/// Statistics for viability predictions
#[derive(Debug, Clone)]
pub struct ViabilityStatistics {
    total_predictions: usize,
    correct_predictions: usize,
    impossible_s_success_rate: f64,
    normal_s_success_rate: f64,
}

impl ViabilityStatistics {
    pub fn new() -> Self {
        Self {
            total_predictions: 0,
            correct_predictions: 0,
            impossible_s_success_rate: 0.0,
            normal_s_success_rate: 0.0,
        }
    }

    pub fn record_prediction(
        &mut self,
        predicted_viable: bool,
        actual_outcome: bool,
        s_values: &SConstantTriDimensional,
    ) {
        self.total_predictions += 1;
        if predicted_viable == actual_outcome {
            self.correct_predictions += 1;
        }

        // Update success rates based on S-value types
        if s_values.has_impossible_local_values() {
            self.impossible_s_success_rate = self.impossible_s_success_rate * 0.9 + 
                (if actual_outcome { 1.0 } else { 0.0 }) * 0.1;
        } else {
            self.normal_s_success_rate = self.normal_s_success_rate * 0.9 + 
                (if actual_outcome { 1.0 } else { 0.0 }) * 0.1;
        }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }

    pub fn calculate_adjustment(&self, s_values: &SConstantTriDimensional) -> f64 {
        if s_values.has_impossible_local_values() {
            (self.impossible_s_success_rate - 0.5) * 0.2 // Adjust confidence based on impossible S success rate
        } else {
            (self.normal_s_success_rate - 0.5) * 0.1 // Smaller adjustment for normal S values
        }
    }
} 