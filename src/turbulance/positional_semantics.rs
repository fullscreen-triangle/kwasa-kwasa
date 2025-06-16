/// Positional Semantics Processing for Kwasa-Kwasa
/// 
/// This module implements position-aware text processing that treats word order
/// as a first-class semantic feature, recognizing that "position IS meaning"
/// in human language.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::turbulance::interpreter::Value;
use crate::turbulance::{Result, TurbulanceError};
use crate::turbulance::probabilistic::TextPoint;

/// A word with its positional semantic information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalWord {
    /// The text content of the word
    pub text: String,
    
    /// Position in the sentence (1-indexed)
    pub position: usize,
    
    /// Semantic weight based on position (0.0-1.0)
    pub positional_weight: f64,
    
    /// How much this word's meaning depends on its position (0.0-1.0)
    pub order_dependency: f64,
    
    /// Semantic role assigned based on position
    pub semantic_role: SemanticRole,
    
    /// Relationships to other words based on position
    pub relationships: Vec<PositionalRelationship>,
    
    /// Part of speech (helps determine positional importance)
    pub pos_tag: Option<String>,
    
    /// Whether this word is a content word (vs function word)
    pub is_content_word: bool,
}

/// Semantic roles that positions can play in sentences
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SemanticRole {
    /// Introduces or modifies the subject
    Determiner,
    
    /// Agent or subject of the sentence
    Subject,
    
    /// Main predicate or action
    Predicate,
    
    /// Object or recipient of action
    Object,
    
    /// Temporal modifier
    TemporalModifier,
    
    /// Manner or method modifier
    MannerModifier,
    
    /// Spatial or locative information
    SpatialModifier,
    
    /// Connects clauses or phrases
    Connector,
    
    /// Auxiliary or supporting element
    Auxiliary,
    
    /// Unclear or ambiguous role
    Ambiguous,
}

/// Relationship between words based on their positions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionalRelationship {
    /// Position of the target word
    pub target_position: usize,
    
    /// Type of relationship
    pub relationship_type: RelationType,
    
    /// Strength of the relationship (0.0-1.0)
    pub strength: f64,
    
    /// Distance between positions (affects relationship strength)
    pub distance: usize,
}

/// Types of positional relationships
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RelationType {
    /// Grammatical dependency (subject-verb, verb-object, etc.)
    Grammatical,
    
    /// Spatial or temporal proximity
    Proximity,
    
    /// Cause and effect relationship
    Causal,
    
    /// Modification relationship (adjective-noun, adverb-verb)
    Modification,
    
    /// Coordination (parallel elements)
    Coordination,
    
    /// Hierarchical relationship (main-subordinate)
    Hierarchical,
}

/// A sentence analyzed for positional semantics
#[derive(Clone, Debug)]
pub struct PositionalSentence {
    /// Original sentence text
    pub original_text: String,
    
    /// Words with positional analysis
    pub words: Vec<PositionalWord>,
    
    /// Overall semantic signature of position pattern
    pub semantic_signature: String,
    
    /// How dependent this sentence's meaning is on word order (0.0-1.0)
    pub order_dependency_score: f64,
    
    /// Hash for rapid comparison of positional structures
    pub positional_hash: u64,
    
    /// Confidence in the positional analysis (0.0-1.0)
    pub analysis_confidence: f64,
    
    /// Metadata about the analysis
    pub metadata: HashMap<String, Value>,
}

/// Analyzer for extracting positional semantic features
pub struct PositionalAnalyzer {
    /// Language model for POS tagging
    pos_tagger: POSTagger,
    
    /// Patterns for semantic role assignment
    role_patterns: HashMap<String, Vec<RolePattern>>,
    
    /// Weights for different positions in sentences
    position_weights: PositionWeightScheme,
    
    /// Cache for analysis results
    analysis_cache: HashMap<String, PositionalSentence>,
}

/// Part-of-speech tagger (simplified implementation)
struct POSTagger {
    word_pos_map: HashMap<String, String>,
    pos_patterns: Vec<POSPattern>,
}

/// Pattern for part-of-speech assignment
#[allow(dead_code)]
struct POSPattern {
    pattern: String,
    pos: String,
    confidence: f64,
}

/// Pattern for semantic role assignment based on position and POS
#[derive(Clone, Debug)]
struct RolePattern {
    /// Position range where this pattern applies
    position_range: (usize, usize),
    
    /// Required or preferred POS tags
    pos_requirements: Vec<String>,
    
    /// Resulting semantic role
    role: SemanticRole,
    
    /// Confidence in this assignment
    confidence: f64,
    
    /// Context requirements
    context: Vec<String>,
}

/// Scheme for weighting positions in sentences
#[derive(Clone, Debug)]
struct PositionWeightScheme {
    /// Weights for sentence start positions
    start_weights: Vec<f64>,
    
    /// Weights for sentence middle positions
    middle_weight: f64,
    
    /// Weights for sentence end positions
    end_weights: Vec<f64>,
    
    /// Special weights for function words
    function_word_weight: f64,
    
    /// Special weights for content words
    content_word_weight: f64,
}

impl PositionalAnalyzer {
    /// Create a new positional analyzer
    pub fn new() -> Self {
        Self {
            pos_tagger: POSTagger::new(),
            role_patterns: Self::create_default_role_patterns(),
            position_weights: PositionWeightScheme::default(),
            analysis_cache: HashMap::new(),
        }
    }
    
    /// Analyze a sentence for positional semantics
    pub fn analyze(&mut self, sentence: &str) -> Result<PositionalSentence> {
        // Check cache first
        if let Some(cached) = self.analysis_cache.get(sentence) {
            return Ok(cached.clone());
        }
        
        // Tokenize and clean the sentence
        let tokens = self.tokenize(sentence)?;
        if tokens.is_empty() {
            return Err(TurbulanceError::InvalidInput("Empty sentence".to_string()));
        }
        
        // Assign part-of-speech tags
        let pos_tags = self.pos_tagger.tag(&tokens)?;
        
        // Analyze each word's positional properties
        let mut words = Vec::new();
        for (i, (token, pos)) in tokens.iter().zip(pos_tags.iter()).enumerate() {
            let position = i + 1; // 1-indexed positions
            
            let positional_word = PositionalWord {
                text: token.clone(),
                position,
                positional_weight: self.calculate_positional_weight(position, pos, tokens.len()),
                order_dependency: self.calculate_order_dependency(token, pos, position, tokens.len()),
                semantic_role: self.assign_semantic_role(position, pos, &tokens, &pos_tags),
                relationships: self.find_relationships(position, &tokens, &pos_tags),
                pos_tag: Some(pos.clone()),
                is_content_word: self.is_content_word(pos),
            };
            
            words.push(positional_word);
        }
        
        // Calculate overall sentence properties
        let semantic_signature = self.generate_semantic_signature(&words);
        let order_dependency_score = self.calculate_sentence_order_dependency(&words);
        let positional_hash = self.calculate_positional_hash(&words);
        let analysis_confidence = self.calculate_analysis_confidence(&words);
        
        let result = PositionalSentence {
            original_text: sentence.to_string(),
            words,
            semantic_signature,
            order_dependency_score,
            positional_hash,
            analysis_confidence,
            metadata: HashMap::new(),
        };
        
        // Cache the result
        self.analysis_cache.insert(sentence.to_string(), result.clone());
        
        Ok(result)
    }
    
    /// Tokenize a sentence into words
    fn tokenize(&self, sentence: &str) -> Result<Vec<String>> {
        let tokens: Vec<String> = sentence
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
            
        Ok(tokens)
    }
    
    /// Calculate positional weight for a word
    fn calculate_positional_weight(&self, position: usize, pos: &str, sentence_length: usize) -> f64 {
        // Base weight from position scheme
        let base_weight = if position <= 3 {
            // Sentence start - high weight for temporal markers, subjects
            self.position_weights.start_weights.get(position - 1).unwrap_or(&0.5)
        } else if position > sentence_length - 3 {
            // Sentence end - high weight for objects, conclusions
            let end_index = position - (sentence_length - 2);
            self.position_weights.end_weights.get(end_index).unwrap_or(&0.5)
        } else {
            // Middle of sentence
            &self.position_weights.middle_weight
        };
        
        // Adjust based on part of speech
        let pos_multiplier = match pos {
            "NOUN" | "VERB" | "ADJ" | "ADV" => self.position_weights.content_word_weight,
            "DET" | "PREP" | "CONJ" | "PRON" => self.position_weights.function_word_weight,
            _ => 1.0,
        };
        
        (base_weight * pos_multiplier).max(0.0_f64).min(1.0_f64)
    }
    
    /// Calculate how dependent a word's meaning is on its position
    fn calculate_order_dependency(&self, word: &str, pos: &str, position: usize, sentence_length: usize) -> f64 {
        let mut dependency = 0.5; // Base dependency
        
        // High order dependency for certain word types
        match pos {
            "PRON" => dependency += 0.3, // Pronouns are highly order-dependent
            "DET" => dependency += 0.4,  // Determiners must be positioned correctly
            "PREP" => dependency += 0.4, // Prepositions are order-sensitive
            "VERB" => dependency += 0.2, // Verbs have positional constraints
            "ADJ" => dependency += 0.1,  // Adjectives have some positional flexibility
            _ => {},
        }
        
        // Dependency varies by position in sentence
        if position == 1 {
            dependency += 0.2; // First position has strong constraints
        } else if position == sentence_length {
            dependency += 0.1; // Last position has some constraints
        }
        
        // Check for words that are particularly order-sensitive
        let order_sensitive_words = ["not", "very", "too", "quite", "the", "a", "an"];
        if order_sensitive_words.contains(&word) {
            dependency += 0.2;
        }
        
        dependency.max(0.0).min(1.0)
    }
    
    /// Assign semantic role based on position and POS
    fn assign_semantic_role(&self, position: usize, pos: &str, tokens: &[String], pos_tags: &[String]) -> SemanticRole {
        // Simple rule-based assignment (could be improved with ML)
        match pos {
            "DET" => SemanticRole::Determiner,
            "NOUN" | "PRON" => {
                if position <= 2 || (position <= 3 && pos_tags.get(0).map_or(false, |p| p == "DET")) {
                    SemanticRole::Subject
                } else {
                    SemanticRole::Object
                }
            },
            "VERB" => SemanticRole::Predicate,
            "ADJ" | "ADV" => {
                if tokens.get(position - 1).map_or(false, |w| self.is_temporal_word(w)) {
                    SemanticRole::TemporalModifier
                } else if tokens.get(position - 1).map_or(false, |w| self.is_spatial_word(w)) {
                    SemanticRole::SpatialModifier
                } else {
                    SemanticRole::MannerModifier
                }
            },
            "PREP" => SemanticRole::SpatialModifier,
            "CONJ" => SemanticRole::Connector,
            "AUX" => SemanticRole::Auxiliary,
            _ => SemanticRole::Ambiguous,
        }
    }
    
    /// Find positional relationships between words
    fn find_relationships(&self, position: usize, tokens: &[String], pos_tags: &[String]) -> Vec<PositionalRelationship> {
        let mut relationships = Vec::new();
        
        // Look for grammatical relationships
        for (i, (token, pos)) in tokens.iter().zip(pos_tags.iter()).enumerate() {
            let target_position = i + 1;
            if target_position == position {
                continue; // Skip self
            }
            
            let distance = if target_position > position {
                target_position - position
            } else {
                position - target_position
            };
            
            // Only consider nearby words for most relationships
            if distance <= 3 {
                let relationship_type = self.determine_relationship_type(
                    position, target_position, pos_tags.get(position - 1).unwrap(), pos
                );
                
                if let Some(rel_type) = relationship_type {
                    let strength = 1.0 / (distance as f64).sqrt(); // Closer = stronger
                    
                    relationships.push(PositionalRelationship {
                        target_position,
                        relationship_type: rel_type,
                        strength,
                        distance,
                    });
                }
            }
        }
        
        relationships
    }
    
    /// Determine the type of relationship between two positions
    fn determine_relationship_type(&self, pos1: usize, pos2: usize, pos_tag1: &str, pos_tag2: &str) -> Option<RelationType> {
        // Subject-verb relationship
        if (pos_tag1 == "NOUN" || pos_tag1 == "PRON") && pos_tag2 == "VERB" && pos2 == pos1 + 1 {
            return Some(RelationType::Grammatical);
        }
        
        // Verb-object relationship
        if pos_tag1 == "VERB" && (pos_tag2 == "NOUN" || pos_tag2 == "PRON") && pos2 == pos1 + 1 {
            return Some(RelationType::Grammatical);
        }
        
        // Determiner-noun relationship
        if pos_tag1 == "DET" && pos_tag2 == "NOUN" && pos2 == pos1 + 1 {
            return Some(RelationType::Grammatical);
        }
        
        // Adjective-noun modification
        if pos_tag1 == "ADJ" && pos_tag2 == "NOUN" {
            return Some(RelationType::Modification);
        }
        
        // Adverb-verb modification
        if pos_tag1 == "ADV" && pos_tag2 == "VERB" {
            return Some(RelationType::Modification);
        }
        
        // Proximity relationship for adjacent words
        if (pos2 as i32 - pos1 as i32).abs() == 1 {
            return Some(RelationType::Proximity);
        }
        
        None
    }
    
    /// Generate a semantic signature for the sentence
    fn generate_semantic_signature(&self, words: &[PositionalWord]) -> String {
        let roles: Vec<String> = words.iter()
            .map(|w| format!("{:?}", w.semantic_role))
            .collect();
            
        format!("{}", roles.join("→"))
    }
    
    /// Calculate overall order dependency for the sentence
    fn calculate_sentence_order_dependency(&self, words: &[PositionalWord]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }
        
        let total_dependency: f64 = words.iter()
            .map(|w| w.order_dependency * w.positional_weight)
            .sum();
            
        let total_weight: f64 = words.iter()
            .map(|w| w.positional_weight)
            .sum();
            
        if total_weight > 0.0 {
            total_dependency / total_weight
        } else {
            0.0
        }
    }
    
    /// Calculate a hash for the positional structure
    fn calculate_positional_hash(&self, words: &[PositionalWord]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        for word in words {
            word.semantic_role.hash(&mut hasher);
            word.is_content_word.hash(&mut hasher);
            word.pos_tag.hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    /// Calculate confidence in the analysis
    fn calculate_analysis_confidence(&self, words: &[PositionalWord]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }
        
        // Confidence based on clarity of roles and relationships
        let role_clarity: f64 = words.iter()
            .map(|w| if w.semantic_role == SemanticRole::Ambiguous { 0.0 } else { 1.0 })
            .sum::<f64>() / words.len() as f64;
            
        let relationship_strength: f64 = words.iter()
            .map(|w| w.relationships.iter().map(|r| r.strength).sum::<f64>() / (w.relationships.len().max(1) as f64))
            .sum::<f64>() / words.len() as f64;
            
        (role_clarity + relationship_strength) / 2.0
    }
    
    /// Check if a word is temporal
    fn is_temporal_word(&self, word: &str) -> bool {
        let temporal_words = ["yesterday", "today", "tomorrow", "now", "then", "when", "before", "after"];
        temporal_words.contains(&word)
    }
    
    /// Check if a word is spatial
    fn is_spatial_word(&self, word: &str) -> bool {
        let spatial_words = ["here", "there", "where", "above", "below", "near", "far", "on", "in", "at"];
        spatial_words.contains(&word)
    }
    
    /// Check if a POS tag represents a content word
    fn is_content_word(&self, pos: &str) -> bool {
        matches!(pos, "NOUN" | "VERB" | "ADJ" | "ADV")
    }
    
    /// Create default role patterns
    fn create_default_role_patterns() -> HashMap<String, Vec<RolePattern>> {
        let mut patterns = HashMap::new();
        
        // English patterns
        let english_patterns = vec![
            RolePattern {
                position_range: (1, 1),
                pos_requirements: vec!["DET".to_string()],
                role: SemanticRole::Determiner,
                confidence: 0.9,
                context: vec![],
            },
            RolePattern {
                position_range: (1, 3),
                pos_requirements: vec!["NOUN".to_string(), "PRON".to_string()],
                role: SemanticRole::Subject,
                confidence: 0.8,
                context: vec![],
            },
            RolePattern {
                position_range: (2, 5),
                pos_requirements: vec!["VERB".to_string()],
                role: SemanticRole::Predicate,
                confidence: 0.9,
                context: vec![],
            },
        ];
        
        patterns.insert("english".to_string(), english_patterns);
        patterns
    }
}

impl POSTagger {
    /// Create a new POS tagger
    fn new() -> Self {
        Self {
            word_pos_map: Self::create_default_pos_map(),
            pos_patterns: Self::create_default_patterns(),
        }
    }
    
    /// Tag words with parts of speech
    fn tag(&self, words: &[String]) -> Result<Vec<String>> {
        let mut tags = Vec::new();
        
        for word in words {
            let tag = self.word_pos_map.get(word)
                .cloned()
                .unwrap_or_else(|| self.guess_pos(word));
            tags.push(tag);
        }
        
        Ok(tags)
    }
    
    /// Guess POS for unknown words
    fn guess_pos(&self, word: &str) -> String {
        // Simple heuristics
        if word.ends_with("ing") || word.ends_with("ed") {
            "VERB".to_string()
        } else if word.ends_with("ly") {
            "ADV".to_string()
        } else if word.ends_with("tion") || word.ends_with("ness") {
            "NOUN".to_string()
        } else if ["the", "a", "an"].contains(&word) {
            "DET".to_string()
        } else if ["is", "are", "was", "were", "be", "been", "being"].contains(&word) {
            "VERB".to_string()
        } else {
            "NOUN".to_string() // Default to noun
        }
    }
    
    /// Create default POS mappings
    fn create_default_pos_map() -> HashMap<String, String> {
        let mut map = HashMap::new();
        
        // Common determiners
        map.insert("the".to_string(), "DET".to_string());
        map.insert("a".to_string(), "DET".to_string());
        map.insert("an".to_string(), "DET".to_string());
        
        // Common verbs
        map.insert("is".to_string(), "VERB".to_string());
        map.insert("are".to_string(), "VERB".to_string());
        map.insert("was".to_string(), "VERB".to_string());
        map.insert("were".to_string(), "VERB".to_string());
        map.insert("have".to_string(), "VERB".to_string());
        map.insert("has".to_string(), "VERB".to_string());
        map.insert("had".to_string(), "VERB".to_string());
        map.insert("will".to_string(), "AUX".to_string());
        map.insert("would".to_string(), "AUX".to_string());
        map.insert("can".to_string(), "AUX".to_string());
        map.insert("could".to_string(), "AUX".to_string());
        map.insert("should".to_string(), "AUX".to_string());
        
        // Common pronouns
        map.insert("i".to_string(), "PRON".to_string());
        map.insert("you".to_string(), "PRON".to_string());
        map.insert("he".to_string(), "PRON".to_string());
        map.insert("she".to_string(), "PRON".to_string());
        map.insert("it".to_string(), "PRON".to_string());
        map.insert("we".to_string(), "PRON".to_string());
        map.insert("they".to_string(), "PRON".to_string());
        
        // Common prepositions
        map.insert("on".to_string(), "PREP".to_string());
        map.insert("in".to_string(), "PREP".to_string());
        map.insert("at".to_string(), "PREP".to_string());
        map.insert("by".to_string(), "PREP".to_string());
        map.insert("for".to_string(), "PREP".to_string());
        map.insert("with".to_string(), "PREP".to_string());
        map.insert("from".to_string(), "PREP".to_string());
        map.insert("to".to_string(), "PREP".to_string());
        
        // Common conjunctions
        map.insert("and".to_string(), "CONJ".to_string());
        map.insert("or".to_string(), "CONJ".to_string());
        map.insert("but".to_string(), "CONJ".to_string());
        map.insert("because".to_string(), "CONJ".to_string());
        map.insert("if".to_string(), "CONJ".to_string());
        map.insert("when".to_string(), "CONJ".to_string());
        
        map
    }
    
    /// Create default POS patterns
    fn create_default_patterns() -> Vec<POSPattern> {
        vec![
            POSPattern {
                pattern: r".*ing$".to_string(),
                pos: "VERB".to_string(),
                confidence: 0.7,
            },
            POSPattern {
                pattern: r".*ed$".to_string(),
                pos: "VERB".to_string(),
                confidence: 0.6,
            },
            POSPattern {
                pattern: r".*ly$".to_string(),
                pos: "ADV".to_string(),
                confidence: 0.8,
            },
        ]
    }
}

impl Default for PositionWeightScheme {
    fn default() -> Self {
        Self {
            start_weights: vec![0.8, 0.7, 0.6], // High weight for first 3 positions
            middle_weight: 0.5,
            end_weights: vec![0.6, 0.7, 0.8], // Increasing weight toward end
            function_word_weight: 0.7,
            content_word_weight: 1.2,
        }
    }
}

impl PositionalSentence {
    /// Extract a TextPoint from this positional analysis
    pub fn to_text_point(&self) -> TextPoint {
        let mut point = TextPoint::new(self.original_text.clone(), self.analysis_confidence);
        
        // Add positional metadata
        point.metadata.insert("semantic_signature".to_string(), 
            Value::String(self.semantic_signature.clone()));
        point.metadata.insert("order_dependency".to_string(), 
            Value::Number(self.order_dependency_score));
        point.metadata.insert("positional_hash".to_string(), 
            Value::Number(self.positional_hash as f64));
            
        point
    }
    
    /// Calculate similarity to another positional sentence
    pub fn positional_similarity(&self, other: &PositionalSentence) -> f64 {
        // Quick hash comparison
        if self.positional_hash == other.positional_hash {
            return 1.0;
        }
        
        // Compare semantic signatures
        let sig_similarity = if self.semantic_signature == other.semantic_signature {
            1.0
        } else {
            0.0 // Could implement more sophisticated comparison
        };
        
        // Compare order dependencies
        let order_similarity = 1.0 - (self.order_dependency_score - other.order_dependency_score).abs();
        
        // Average the similarities
        (sig_similarity + order_similarity) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positional_analysis() {
        let mut analyzer = PositionalAnalyzer::new();
        let result = analyzer.analyze("The cat sat on the mat").unwrap();
        
        assert_eq!(result.words.len(), 6);
        assert!(result.order_dependency_score > 0.0);
        assert!(result.analysis_confidence > 0.0);
        assert!(!result.semantic_signature.is_empty());
    }
    
    #[test]
    fn test_semantic_roles() {
        let mut analyzer = PositionalAnalyzer::new();
        let result = analyzer.analyze("The solution is optimal").unwrap();
        
        // Check that we identify basic roles
        let roles: Vec<&SemanticRole> = result.words.iter().map(|w| &w.semantic_role).collect();
        assert!(roles.contains(&&SemanticRole::Determiner));
        assert!(roles.contains(&&SemanticRole::Subject));
        assert!(roles.contains(&&SemanticRole::Predicate));
    }
    
    #[test]
    fn test_positional_weights() {
        let mut analyzer = PositionalAnalyzer::new();
        let result = analyzer.analyze("Yesterday the solution was optimal").unwrap();
        
        // First word should have high weight (temporal modifier)
        assert!(result.words[0].positional_weight > 0.5);
        
        // Content words should generally have higher weights than function words
        let content_weights: Vec<f64> = result.words.iter()
            .filter(|w| w.is_content_word)
            .map(|w| w.positional_weight)
            .collect();
        let function_weights: Vec<f64> = result.words.iter()
            .filter(|w| !w.is_content_word)
            .map(|w| w.positional_weight)
            .collect();
            
        if !content_weights.is_empty() && !function_weights.is_empty() {
            let avg_content = content_weights.iter().sum::<f64>() / content_weights.len() as f64;
            let avg_function = function_weights.iter().sum::<f64>() / function_weights.len() as f64;
            assert!(avg_content >= avg_function);
        }
    }
    
    #[test]
    fn test_order_dependency() {
        let mut analyzer = PositionalAnalyzer::new();
        let result = analyzer.analyze("The big red car").unwrap();
        
        // Determiners should have high order dependency
        let det_word = result.words.iter().find(|w| w.semantic_role == SemanticRole::Determiner);
        if let Some(det) = det_word {
            assert!(det.order_dependency > 0.5);
        }
    }
    
    #[test]
    fn test_positional_relationships() {
        let mut analyzer = PositionalAnalyzer::new();
        let result = analyzer.analyze("The cat sleeps peacefully").unwrap();
        
        // Words should have relationships to nearby words
        for word in &result.words {
            if word.position > 1 && word.position < result.words.len() {
                assert!(!word.relationships.is_empty(), 
                    "Word at position {} should have relationships", word.position);
            }
        }
    }
} 