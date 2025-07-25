use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use std::collections::HashMap;

/// Provide research context for a given topic
pub fn research_context(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "research_context requires exactly 1 argument".to_string(),
        });
    }

    let query = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "research_context requires a string or TextUnit".to_string(),
        }),
    };

    // Analyze the query to provide contextual research information
    let mut research_data = HashMap::new();
    
    let concepts = extract_key_concepts(query);
    let context = generate_contextual_information(&concepts);
    let related_topics = find_related_topics(&concepts);
    let methodological_considerations = suggest_methodologies(&concepts);
    let evidence_requirements = determine_evidence_requirements(&concepts);
    
    research_data.insert("query".to_string(), Value::String(query.clone()));
    research_data.insert("key_concepts".to_string(), Value::List(
        concepts.into_iter().map(|c| Value::String(c)).collect()
    ));
    research_data.insert("context".to_string(), Value::String(context));
    research_data.insert("related_topics".to_string(), Value::List(
        related_topics.into_iter().map(|t| Value::String(t)).collect()
    ));
    research_data.insert("methodological_considerations".to_string(), Value::List(
        methodological_considerations.into_iter().map(|m| Value::String(m)).collect()
    ));
    research_data.insert("evidence_requirements".to_string(), Value::List(
        evidence_requirements.into_iter().map(|e| Value::String(e)).collect()
    ));
    research_data.insert("confidence".to_string(), Value::Number(0.78));
    
    Ok(Value::Map(research_data))
}

/// Perform fact-checking on claims
pub fn fact_check(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "fact_check requires exactly 1 argument".to_string(),
        });
    }

    let claim = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "fact_check requires a string or TextUnit".to_string(),
        }),
    };

    let claim_analysis = analyze_claim_structure(claim);
    let verifiability = assess_verifiability(claim);
    let evidence_requirements = determine_evidence_requirements_for_claim(claim);
    let potential_biases = identify_potential_biases(claim);
    let fact_check_status = determine_fact_check_status(claim);
    
    let mut fact_check_result = HashMap::new();
    fact_check_result.insert("claim".to_string(), Value::String(claim.clone()));
    fact_check_result.insert("claim_type".to_string(), Value::String(claim_analysis.claim_type));
    fact_check_result.insert("specificity".to_string(), Value::Number(claim_analysis.specificity));
    fact_check_result.insert("verifiability_status".to_string(), Value::String(verifiability.status));
    fact_check_result.insert("verifiability_score".to_string(), Value::Number(verifiability.score));
    fact_check_result.insert("evidence_requirements".to_string(), Value::List(
        evidence_requirements.into_iter().map(|req| Value::String(req)).collect()
    ));
    fact_check_result.insert("potential_biases".to_string(), Value::List(
        potential_biases.into_iter().map(|bias| Value::String(bias)).collect()
    ));
    fact_check_result.insert("fact_check_status".to_string(), Value::String(fact_check_status.status));
    fact_check_result.insert("confidence".to_string(), Value::Number(fact_check_status.confidence));
    fact_check_result.insert("explanation".to_string(), Value::String(fact_check_status.explanation));
    
    Ok(Value::Map(fact_check_result))
}

/// Ensure explanation follows logical structure
pub fn ensure_explanation_follows(args: Vec<Value>) -> Result<Value> {
    if args.len() != 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows requires exactly 2 arguments (explanation, logical_pattern)".to_string(),
        });
    }

    let explanation = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows first argument must be a string or TextUnit".to_string(),
        }),
    };

    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "ensure_explanation_follows second argument must be a string".to_string(),
        }),
    };

    let logical_analysis = analyze_logical_structure(explanation, pattern);
    
    let mut result = HashMap::new();
    result.insert("explanation".to_string(), Value::String(explanation.clone()));
    result.insert("expected_pattern".to_string(), Value::String(pattern.clone()));
    result.insert("follows_pattern".to_string(), Value::Boolean(logical_analysis.follows_pattern));
    result.insert("pattern_adherence_score".to_string(), Value::Number(logical_analysis.adherence_score));
    result.insert("missing_elements".to_string(), Value::List(
        logical_analysis.missing_elements.into_iter().map(|e| Value::String(e)).collect()
    ));
    result.insert("structural_issues".to_string(), Value::List(
        logical_analysis.structural_issues.into_iter().map(|i| Value::String(i)).collect()
    ));
    result.insert("recommendations".to_string(), Value::List(
        logical_analysis.recommendations.into_iter().map(|r| Value::String(r)).collect()
    ));
    
    Ok(Value::Map(result))
}

/// Generate research hypotheses based on data
pub fn generate_hypotheses(args: Vec<Value>) -> Result<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "generate_hypotheses requires 1-2 arguments (data, optional context)".to_string(),
        });
    }

    let data = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        Value::Map(m) => &format!("{:?}", m),
        _ => return Err(TurbulanceError::RuntimeError {
            message: "generate_hypotheses first argument must be data (string, TextUnit, or map)".to_string(),
        }),
    };

    let context = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => "general".to_string(),
        }
    } else {
        "general".to_string()
    };

    let hypotheses = generate_hypotheses_from_data(data, &context);
    let confidence_scores = calculate_hypothesis_confidence(&hypotheses);
    let testability_scores = assess_hypothesis_testability(&hypotheses);
    
    let mut result = HashMap::new();
    result.insert("hypotheses".to_string(), Value::List(
        hypotheses.into_iter().map(|h| Value::String(h)).collect()
    ));
    result.insert("confidence_scores".to_string(), Value::List(
        confidence_scores.into_iter().map(|c| Value::Number(c)).collect()
    ));
    result.insert("testability_scores".to_string(), Value::List(
        testability_scores.into_iter().map(|t| Value::Number(t)).collect()
    ));
    result.insert("context".to_string(), Value::String(context));
    
    Ok(Value::Map(result))
}

/// Analyze research validity
pub fn analyze_research_validity(args: Vec<Value>) -> Result<Value> {
    if args.len() != 1 {
        return Err(TurbulanceError::RuntimeError {
            message: "analyze_research_validity requires exactly 1 argument".to_string(),
        });
    }

    let research_text = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "analyze_research_validity requires a string or TextUnit".to_string(),
        }),
    };

    let validity_analysis = assess_research_validity(research_text);
    
    let mut result = HashMap::new();
    result.insert("overall_validity".to_string(), Value::String(validity_analysis.overall_score));
    result.insert("methodology_score".to_string(), Value::Number(validity_analysis.methodology_score));
    result.insert("evidence_quality".to_string(), Value::Number(validity_analysis.evidence_quality));
    result.insert("reasoning_coherence".to_string(), Value::Number(validity_analysis.reasoning_coherence));
    result.insert("bias_indicators".to_string(), Value::List(
        validity_analysis.bias_indicators.into_iter().map(|b| Value::String(b)).collect()
    ));
    result.insert("validity_threats".to_string(), Value::List(
        validity_analysis.validity_threats.into_iter().map(|t| Value::String(t)).collect()
    ));
    result.insert("recommendations".to_string(), Value::List(
        validity_analysis.recommendations.into_iter().map(|r| Value::String(r)).collect()
    ));
    
    Ok(Value::Map(result))
}

/// Suggest research methods for a topic
pub fn suggest_research_methods(args: Vec<Value>) -> Result<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(TurbulanceError::RuntimeError {
            message: "suggest_research_methods requires 1-2 arguments (topic, optional discipline)".to_string(),
        });
    }

    let topic = match &args[0] {
        Value::String(s) => s,
        Value::TextUnit(tu) => &tu.content,
        _ => return Err(TurbulanceError::RuntimeError {
            message: "suggest_research_methods first argument must be a string or TextUnit".to_string(),
        }),
    };

    let discipline = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => "general".to_string(),
        }
    } else {
        "general".to_string()
    };

    let methods = suggest_methods_for_topic(topic, &discipline);
    let strengths = analyze_method_strengths(&methods);
    let limitations = analyze_method_limitations(&methods);
    let combinations = suggest_method_combinations(&methods);
    
    let mut result = HashMap::new();
    result.insert("topic".to_string(), Value::String(topic.clone()));
    result.insert("discipline".to_string(), Value::String(discipline));
    result.insert("suggested_methods".to_string(), Value::List(
        methods.into_iter().map(|m| Value::String(m)).collect()
    ));
    result.insert("method_strengths".to_string(), Value::Map(strengths));
    result.insert("method_limitations".to_string(), Value::Map(limitations));
    result.insert("recommended_combinations".to_string(), Value::List(
        combinations.into_iter().map(|c| Value::String(c)).collect()
    ));
    
    Ok(Value::Map(result))
}

// Helper structures and functions

struct ClaimAnalysis {
    claim_type: String,
    specificity: f64,
}

struct VerifiabilityAssessment {
    status: String,
    score: f64,
}

struct FactCheckStatus {
    status: String,
    confidence: f64,
    explanation: String,
}

struct LogicalAnalysis {
    follows_pattern: bool,
    adherence_score: f64,
    missing_elements: Vec<String>,
    structural_issues: Vec<String>,
    recommendations: Vec<String>,
}

struct ValidityAnalysis {
    overall_score: String,
    methodology_score: f64,
    evidence_quality: f64,
    reasoning_coherence: f64,
    bias_indicators: Vec<String>,
    validity_threats: Vec<String>,
    recommendations: Vec<String>,
}

fn extract_key_concepts(query: &str) -> Vec<String> {
    let mut concepts = Vec::new();
    
    // Domain-specific keywords
    let domain_indicators = [
        ("biology", vec!["cell", "DNA", "protein", "gene", "organism", "evolution"]),
        ("chemistry", vec!["molecule", "atom", "reaction", "compound", "bond"]),
        ("physics", vec!["force", "energy", "quantum", "particle", "wave"]),
        ("psychology", vec!["behavior", "cognitive", "brain", "learning", "memory"]),
        ("medicine", vec!["patient", "disease", "treatment", "diagnosis", "therapy"]),
        ("computer_science", vec!["algorithm", "data", "programming", "software"]),
    ];
    
    let query_lower = query.to_lowercase();
    
    for (domain, keywords) in &domain_indicators {
        let mut domain_score = 0;
        for keyword in keywords {
            if query_lower.contains(keyword) {
                domain_score += 1;
                concepts.push(keyword.to_string());
            }
        }
        if domain_score > 0 {
            concepts.push(domain.to_string());
        }
    }
    
    // Extract noun phrases as concepts
    let words: Vec<&str> = query.split_whitespace().collect();
    for window in words.windows(2) {
        if window.len() == 2 {
            let phrase = format!("{} {}", window[0], window[1]);
            if phrase.len() > 6 {
                concepts.push(phrase);
            }
        }
    }
    
    concepts.truncate(8);
    concepts
}

fn generate_contextual_information(concepts: &[String]) -> String {
    let mut context_parts = Vec::new();
    
    for concept in concepts {
        let context = match concept.as_str() {
            "biology" => "Biological research involves studying living organisms and their processes.",
            "chemistry" => "Chemical research examines matter composition and reactions.",
            "physics" => "Physics research investigates fundamental forces and matter behavior.",
            "psychology" => "Psychological research studies mental processes and behavior.",
            "medicine" => "Medical research aims to understand diseases and develop treatments.",
            "computer_science" => "Computer science research develops computational methods and systems.",
            _ => "This research area requires domain-specific methodological considerations.",
        };
        context_parts.push(context);
    }
    
    if context_parts.is_empty() {
        "This research query requires interdisciplinary analysis and careful methodology.".to_string()
    } else {
        context_parts.join(" ")
    }
}

fn find_related_topics(concepts: &[String]) -> Vec<String> {
    let mut related = Vec::new();
    
    for concept in concepts {
        let related_topics = match concept.as_str() {
            "biology" => vec!["biochemistry", "ecology", "genetics", "microbiology"],
            "chemistry" => vec!["biochemistry", "physical chemistry", "organic chemistry"],
            "physics" => vec!["astrophysics", "quantum mechanics", "thermodynamics"],
            "psychology" => vec!["neuroscience", "cognitive science", "behavioral psychology"],
            "medicine" => vec!["pharmacology", "pathology", "epidemiology"],
            "computer_science" => vec!["machine learning", "data science", "algorithms"],
            _ => vec!["interdisciplinary studies", "methodology"],
        };
        related.extend(related_topics.into_iter().map(|s| s.to_string()));
    }
    
    related.sort();
    related.dedup();
    related.truncate(6);
    related
}

fn suggest_methodologies(concepts: &[String]) -> Vec<String> {
    let mut methodologies = Vec::new();
    
    for concept in concepts {
        let methods = match concept.as_str() {
            "biology" => vec!["experimental design", "field studies", "molecular techniques"],
            "chemistry" => vec!["spectroscopy", "synthesis protocols", "analytical methods"],
            "physics" => vec!["mathematical modeling", "experimental physics", "simulations"],
            "psychology" => vec!["experimental psychology", "surveys", "behavioral analysis"],
            "medicine" => vec!["clinical trials", "case studies", "epidemiological studies"],
            "computer_science" => vec!["algorithm analysis", "empirical evaluation", "benchmarking"],
            _ => vec!["literature review", "systematic analysis"],
        };
        methodologies.extend(methods.into_iter().map(|s| s.to_string()));
    }
    
    methodologies.sort();
    methodologies.dedup();
    methodologies.truncate(5);
    methodologies
}

fn determine_evidence_requirements(concepts: &[String]) -> Vec<String> {
    let mut requirements = Vec::new();
    
    for concept in concepts {
        let evidence = match concept.as_str() {
            "biology" => vec!["peer-reviewed studies", "reproducible experiments"],
            "chemistry" => vec!["analytical data", "spectroscopic evidence"],
            "physics" => vec!["mathematical proofs", "experimental validation"],
            "psychology" => vec!["statistical significance", "control groups"],
            "medicine" => vec!["clinical data", "safety profiles"],
            "computer_science" => vec!["performance metrics", "comparative analysis"],
            _ => vec!["documented methodology", "transparent data"],
        };
        requirements.extend(evidence.into_iter().map(|s| s.to_string()));
    }
    
    requirements.sort();
    requirements.dedup();
    requirements.truncate(4);
    requirements
}

fn analyze_claim_structure(claim: &str) -> ClaimAnalysis {
    let claim_lower = claim.to_lowercase();
    
    let claim_type = if claim_lower.contains("cause") || claim_lower.contains("because") {
        "Causal Claim"
    } else if claim_lower.contains("correlat") || claim_lower.contains("relationship") {
        "Correlational Claim"
    } else if claim_lower.contains("all") || claim_lower.contains("every") {
        "Universal Claim"
    } else if claim_lower.contains("will") || claim_lower.contains("predict") {
        "Predictive Claim"
    } else {
        "Descriptive Claim"
    }.to_string();
    
    let mut specificity = 0.5;
    if claim.chars().any(|c| c.is_numeric()) {
        specificity += 0.2;
    }
    if claim_lower.contains("when") || claim_lower.contains("if") {
        specificity += 0.1;
    }
    if claim_lower.contains("might") || claim_lower.contains("could") {
        specificity -= 0.1;
    }
    
    specificity = specificity.max(0.0).min(1.0);
    
    ClaimAnalysis {
        claim_type,
        specificity,
    }
}

fn assess_verifiability(claim: &str) -> VerifiabilityAssessment {
    let claim_lower = claim.to_lowercase();
    let mut score = 0.5;
    
    let verifiable_indicators = ["measure", "observe", "test", "data", "study", "research"];
    for indicator in &verifiable_indicators {
        if claim_lower.contains(indicator) {
            score += 0.1;
        }
    }
    
    if claim.chars().any(|c| c.is_numeric()) {
        score += 0.2;
    }
    
    let unverifiable_indicators = ["always", "never", "all", "none", "impossible"];
    for indicator in &unverifiable_indicators {
        if claim_lower.contains(indicator) {
            score -= 0.15;
        }
    }
    
    score = score.max(0.0).min(1.0);
    
    let status = if score > 0.7 {
        "Highly Verifiable"
    } else if score > 0.4 {
        "Moderately Verifiable"
    } else {
        "Difficult to Verify"
    }.to_string();
    
    VerifiabilityAssessment { status, score }
}

fn determine_evidence_requirements_for_claim(claim: &str) -> Vec<String> {
    let claim_lower = claim.to_lowercase();
    let mut requirements = Vec::new();
    
    if claim_lower.contains("cause") {
        requirements.push("Controlled experiments".to_string());
        requirements.push("Control for confounding variables".to_string());
    }
    
    if claim_lower.contains("effect") {
        requirements.push("Before-and-after measurements".to_string());
        requirements.push("Statistical significance testing".to_string());
    }
    
    if claim.chars().any(|c| c.is_numeric()) {
        requirements.push("Quantitative measurements".to_string());
        requirements.push("Confidence intervals".to_string());
    }
    
    if requirements.is_empty() {
        requirements.push("Primary source documentation".to_string());
        requirements.push("Peer-reviewed evidence".to_string());
    }
    
    requirements
}

fn identify_potential_biases(claim: &str) -> Vec<String> {
    let claim_lower = claim.to_lowercase();
    let mut biases = Vec::new();
    
    if claim_lower.contains("obviously") || claim_lower.contains("clearly") {
        biases.push("Overconfidence bias".to_string());
    }
    
    if claim_lower.contains("always") || claim_lower.contains("never") {
        biases.push("Overgeneralization bias".to_string());
    }
    
    if claim_lower.contains("studies show") || claim_lower.contains("research proves") {
        biases.push("Appeal to authority without citation".to_string());
    }
    
    if biases.is_empty() {
        biases.push("No obvious bias indicators detected".to_string());
    }
    
    biases
}

fn determine_fact_check_status(claim: &str) -> FactCheckStatus {
    let claim_lower = claim.to_lowercase();
    
    // Simple fact-checking for common claims
    match claim_lower.as_str() {
        s if s.contains("earth is round") => {
            FactCheckStatus {
                status: "True".to_string(),
                confidence: 0.99,
                explanation: "The Earth is an oblate spheroid, confirmed by scientific evidence.".to_string(),
            }
        },
        s if s.contains("water boils at 100") => {
            FactCheckStatus {
                status: "Partially True".to_string(),
                confidence: 0.85,
                explanation: "Water boils at 100°C at standard atmospheric pressure only.".to_string(),
            }
        },
        _ => {
            FactCheckStatus {
                status: "Needs Verification".to_string(),
                confidence: 0.0,
                explanation: "This claim requires verification against reliable sources.".to_string(),
            }
        }
    }
}

fn analyze_logical_structure(explanation: &str, pattern: &str) -> LogicalAnalysis {
    let mut missing_elements = Vec::new();
    let mut structural_issues = Vec::new();
    let mut recommendations = Vec::new();
    let mut adherence_score = 0.0;
    
    match pattern.to_lowercase().as_str() {
        "scientific_method" => {
            let required_elements = ["hypothesis", "method", "result", "conclusion"];
            let exp_lower = explanation.to_lowercase();
            
            for element in &required_elements {
                if exp_lower.contains(element) {
                    adherence_score += 0.25;
                } else {
                    missing_elements.push(format!("Missing {}", element));
                }
            }
        },
        "argument" => {
            let exp_lower = explanation.to_lowercase();
            
            if exp_lower.contains("premise") || exp_lower.contains("assumption") {
                adherence_score += 0.33;
            } else {
                missing_elements.push("Missing premise statement".to_string());
            }
            
            if exp_lower.contains("evidence") || exp_lower.contains("support") {
                adherence_score += 0.33;
            } else {
                missing_elements.push("Missing supporting evidence".to_string());
            }
            
            if exp_lower.contains("therefore") || exp_lower.contains("conclusion") {
                adherence_score += 0.34;
            } else {
                missing_elements.push("Missing conclusion".to_string());
            }
        },
        _ => {
            adherence_score = 0.5;
            structural_issues.push("Unknown logical pattern specified".to_string());
        }
    }
    
    if !missing_elements.is_empty() {
        recommendations.push("Add missing structural elements".to_string());
    }
    
    if adherence_score < 0.6 {
        recommendations.push("Restructure to better follow the logical pattern".to_string());
    }
    
    LogicalAnalysis {
        follows_pattern: adherence_score >= 0.7,
        adherence_score,
        missing_elements,
        structural_issues,
        recommendations,
    }
}

fn generate_hypotheses_from_data(data: &str, context: &str) -> Vec<String> {
    let mut hypotheses = Vec::new();
    
    // Simple hypothesis generation based on data content
    if data.contains("increase") || data.contains("higher") {
        hypotheses.push("There is a positive correlation between the measured variables".to_string());
    }
    
    if data.contains("decrease") || data.contains("lower") {
        hypotheses.push("There is a negative correlation between the measured variables".to_string());
    }
    
    if data.contains("differ") || data.contains("significant") {
        hypotheses.push("There are significant differences between groups or conditions".to_string());
    }
    
    match context {
        "medical" => {
            hypotheses.push("The intervention may have therapeutic effects".to_string());
        },
        "psychological" => {
            hypotheses.push("Cognitive or behavioral factors may mediate the observed effects".to_string());
        },
        _ => {
            hypotheses.push("The observed patterns suggest underlying systematic relationships".to_string());
        }
    }
    
    if hypotheses.is_empty() {
        hypotheses.push("Further investigation is needed to identify specific hypotheses".to_string());
    }
    
    hypotheses
}

fn calculate_hypothesis_confidence(hypotheses: &[String]) -> Vec<f64> {
    hypotheses.iter().map(|h| {
        if h.contains("may") || h.contains("might") {
            0.6
        } else if h.contains("significant") || h.contains("correlation") {
            0.8
        } else {
            0.7
        }
    }).collect()
}

fn assess_hypothesis_testability(hypotheses: &[String]) -> Vec<f64> {
    hypotheses.iter().map(|h| {
        if h.contains("measure") || h.contains("test") || h.contains("compare") {
            0.9
        } else if h.contains("correlation") || h.contains("difference") {
            0.8
        } else {
            0.6
        }
    }).collect()
}

fn assess_research_validity(research_text: &str) -> ValidityAnalysis {
    let text_lower = research_text.to_lowercase();
    
    let mut methodology_score = 0.5;
    let mut evidence_quality = 0.5;
    let mut reasoning_coherence = 0.5;
    
    // Check methodology indicators
    if text_lower.contains("control") || text_lower.contains("random") {
        methodology_score += 0.2;
    }
    if text_lower.contains("sample size") || text_lower.contains("power") {
        methodology_score += 0.1;
    }
    
    // Check evidence quality
    if text_lower.contains("peer review") || text_lower.contains("published") {
        evidence_quality += 0.2;
    }
    if text_lower.contains("data") || text_lower.contains("measurement") {
        evidence_quality += 0.1;
    }
    
    // Check reasoning coherence
    if text_lower.contains("therefore") || text_lower.contains("conclusion") {
        reasoning_coherence += 0.2;
    }
    if text_lower.contains("because") || text_lower.contains("evidence") {
        reasoning_coherence += 0.1;
    }
    
    let overall_score = if (methodology_score + evidence_quality + reasoning_coherence) / 3.0 > 0.8 {
        "High Validity"
    } else if (methodology_score + evidence_quality + reasoning_coherence) / 3.0 > 0.6 {
        "Moderate Validity"
    } else {
        "Low Validity"
    }.to_string();
    
    ValidityAnalysis {
        overall_score,
        methodology_score,
        evidence_quality,
        reasoning_coherence,
        bias_indicators: vec!["Selection bias potential".to_string()],
        validity_threats: vec!["External validity limitations".to_string()],
        recommendations: vec!["Strengthen methodology description".to_string()],
    }
}

fn suggest_methods_for_topic(topic: &str, discipline: &str) -> Vec<String> {
    let topic_lower = topic.to_lowercase();
    let mut methods = Vec::new();
    
    match discipline {
        "psychology" => {
            methods.extend(vec![
                "Experimental design".to_string(),
                "Survey research".to_string(),
                "Observational studies".to_string(),
            ]);
        },
        "medicine" => {
            methods.extend(vec![
                "Randomized controlled trial".to_string(),
                "Case-control study".to_string(),
                "Cohort study".to_string(),
            ]);
        },
        "biology" => {
            methods.extend(vec![
                "Laboratory experiments".to_string(),
                "Field studies".to_string(),
                "Molecular analysis".to_string(),
            ]);
        },
        _ => {
            methods.extend(vec![
                "Literature review".to_string(),
                "Qualitative analysis".to_string(),
                "Quantitative analysis".to_string(),
            ]);
        }
    }
    
    if topic_lower.contains("behavior") {
        methods.push("Behavioral observation".to_string());
    }
    if topic_lower.contains("intervention") {
        methods.push("Intervention study".to_string());
    }
    
    methods
}

fn analyze_method_strengths(methods: &[String]) -> HashMap<String, Value> {
    let mut strengths = HashMap::new();
    
    for method in methods {
        let strength = match method.as_str() {
            "Experimental design" => "High internal validity",
            "Survey research" => "Large sample sizes possible",
            "Randomized controlled trial" => "Gold standard for causation",
            "Literature review" => "Comprehensive overview",
            _ => "Systematic approach",
        };
        strengths.insert(method.clone(), Value::String(strength.to_string()));
    }
    
    strengths
}

fn analyze_method_limitations(methods: &[String]) -> HashMap<String, Value> {
    let mut limitations = HashMap::new();
    
    for method in methods {
        let limitation = match method.as_str() {
            "Experimental design" => "Limited external validity",
            "Survey research" => "Response bias potential",
            "Randomized controlled trial" => "Ethical constraints",
            "Literature review" => "Publication bias",
            _ => "Context-dependent limitations",
        };
        limitations.insert(method.clone(), Value::String(limitation.to_string()));
    }
    
    limitations
}

fn suggest_method_combinations(methods: &[String]) -> Vec<String> {
    let mut combinations = Vec::new();
    
    if methods.contains(&"Experimental design".to_string()) && methods.contains(&"Survey research".to_string()) {
        combinations.push("Mixed methods: Experiments followed by surveys".to_string());
    }
    
    if methods.contains(&"Literature review".to_string()) {
        combinations.push("Systematic review followed by primary research".to_string());
    }
    
    if combinations.is_empty() {
        combinations.push("Sequential mixed methods approach".to_string());
    }
    
    combinations
} 