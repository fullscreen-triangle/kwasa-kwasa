//! Text visualization components

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};
use crate::text_unit::{TextUnit, TextUnitType};
use super::{
    Visualization, VisualizationType, VisualizationData, VisualizationConfig,
    TextAnalysisData, SentenceStructure, TimeSeriesPoint, CategoryData
};

/// Text-specific visualization types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TextVisualizationType {
    /// Word cloud visualization
    WordCloud,
    /// Text structure tree
    TextStructure,
    /// Reading difficulty progression
    ReadabilityProgression,
    /// Sentence complexity analysis
    SentenceComplexity,
    /// Topic distribution
    TopicDistribution,
    /// Text unit hierarchy
    TextUnitHierarchy,
}

/// Configuration for text visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextVisualizationConfig {
    /// Maximum words to show in word cloud
    pub max_words: usize,
    /// Minimum word frequency to include
    pub min_frequency: u32,
    /// Show readability scores
    pub show_readability: bool,
    /// Color coding for complexity
    pub complexity_colors: bool,
    /// Font size range for word cloud
    pub font_size_range: (u32, u32),
}

impl Default for TextVisualizationConfig {
    fn default() -> Self {
        Self {
            max_words: 100,
            min_frequency: 2,
            show_readability: true,
            complexity_colors: true,
            font_size_range: (10, 48),
        }
    }
}

/// Word cloud data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordCloudData {
    /// Words with their frequencies and sizes
    pub words: Vec<WordCloudWord>,
    /// Total word count
    pub total_words: usize,
    /// Unique word count
    pub unique_words: usize,
}

/// Individual word in word cloud
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordCloudWord {
    /// The word text
    pub text: String,
    /// Frequency count
    pub frequency: u32,
    /// Font size for display
    pub font_size: u32,
    /// Color (hex string)
    pub color: String,
    /// Position coordinates (optional)
    pub position: Option<(f64, f64)>,
}

/// Text structure visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStructureData {
    /// Hierarchical structure of text units
    pub hierarchy: Vec<TextStructureNode>,
    /// Total depth of the structure
    pub max_depth: usize,
    /// Statistics about each level
    pub level_stats: HashMap<usize, LevelStatistics>,
}

/// Node in text structure hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStructureNode {
    /// Unique identifier
    pub id: String,
    /// Node label
    pub label: String,
    /// Node type
    pub node_type: String,
    /// Children nodes
    pub children: Vec<TextStructureNode>,
    /// Node metadata
    pub metadata: HashMap<String, String>,
    /// Depth in hierarchy
    pub depth: usize,
    /// Word count
    pub word_count: usize,
    /// Complexity score
    pub complexity: f64,
}

/// Statistics for a hierarchy level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelStatistics {
    /// Number of nodes at this level
    pub node_count: usize,
    /// Average word count per node
    pub avg_word_count: f64,
    /// Average complexity per node
    pub avg_complexity: f64,
    /// Most common node type at this level
    pub common_type: String,
}

/// Main text visualization struct
pub struct TextVisualization {
    config: TextVisualizationConfig,
}

impl TextVisualization {
    /// Create a new text visualization with default config
    pub fn new() -> Self {
        Self {
            config: TextVisualizationConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TextVisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate word cloud from text units
    pub fn create_word_cloud(&self, text_units: &[TextUnit]) -> Result<Visualization> {
        let word_frequencies = self.calculate_word_frequencies(text_units)?;
        let word_cloud_data = self.build_word_cloud_data(word_frequencies)?;

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: "Word Cloud".to_string(),
            description: "Frequency-based word cloud visualization".to_string(),
            visualization_type: VisualizationType::TextVisualization,
            data: VisualizationData::Custom(serde_json::to_value(word_cloud_data)?),
            config: VisualizationConfig::default(),
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    /// Generate text structure visualization
    pub fn create_text_structure(&self, text_units: &[TextUnit]) -> Result<Visualization> {
        let structure_data = self.build_text_structure(text_units)?;

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: "Text Structure".to_string(),
            description: "Hierarchical structure of text organization".to_string(),
            visualization_type: VisualizationType::TreeDiagram,
            data: VisualizationData::Custom(serde_json::to_value(structure_data)?),
            config: VisualizationConfig::default(),
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    /// Generate readability progression visualization
    pub fn create_readability_progression(&self, text_units: &[TextUnit]) -> Result<Visualization> {
        let progression_data = self.calculate_readability_progression(text_units)?;

        let time_series: Vec<TimeSeriesPoint> = progression_data
            .into_iter()
            .enumerate()
            .map(|(i, score)| TimeSeriesPoint {
                x: i as f64,
                y: score,
                label: Some(format!("Unit {}", i + 1)),
            })
            .collect();

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: "Readability Progression".to_string(),
            description: "Reading difficulty progression through the text".to_string(),
            visualization_type: VisualizationType::LineChart,
            data: VisualizationData::TimeSeries(time_series),
            config: VisualizationConfig::default(),
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    /// Generate sentence complexity analysis
    pub fn create_sentence_complexity(&self, text_units: &[TextUnit]) -> Result<Visualization> {
        let complexity_data = self.analyze_sentence_complexity(text_units)?;

        let categories: Vec<CategoryData> = complexity_data
            .into_iter()
            .map(|(range, count)| CategoryData {
                category: range,
                value: count as f64,
                color: None,
            })
            .collect();

        let visualization = Visualization {
            id: uuid::Uuid::new_v4().to_string(),
            title: "Sentence Complexity Distribution".to_string(),
            description: "Distribution of sentence complexity scores".to_string(),
            visualization_type: VisualizationType::BarChart,
            data: VisualizationData::Categorical(categories),
            config: VisualizationConfig::default(),
            metadata: HashMap::new(),
        };

        Ok(visualization)
    }

    /// Generate comprehensive text analysis visualization
    pub fn create_text_analysis(&self, text_units: &[TextUnit]) -> Result<TextAnalysisData> {
        let word_frequencies = self.calculate_word_frequencies(text_units)?;
        let sentence_structures = self.extract_sentence_structures(text_units)?;
        let difficulty_progression = self.calculate_readability_progression(text_units)?;
        let topic_distribution = self.analyze_topic_distribution(text_units)?;

        Ok(TextAnalysisData {
            word_frequencies,
            sentence_structures,
            difficulty_progression,
            topic_distribution,
        })
    }

    // Helper methods

    fn calculate_word_frequencies(&self, text_units: &[TextUnit]) -> Result<HashMap<String, u32>> {
        let mut frequencies = HashMap::new();

        for unit in text_units {
            let words: Vec<&str> = unit.content
                .to_lowercase()
                .split_whitespace()
                .filter(|word| word.len() > 2) // Filter short words
                .collect();

            for word in words {
                // Remove punctuation
                let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
                if !clean_word.is_empty() {
                    *frequencies.entry(clean_word.to_string()).or_insert(0) += 1;
                }
            }
        }

        // Filter by minimum frequency
        frequencies.retain(|_, &mut freq| freq >= self.config.min_frequency);

        Ok(frequencies)
    }

    fn build_word_cloud_data(&self, word_frequencies: HashMap<String, u32>) -> Result<WordCloudData> {
        let total_words: u32 = word_frequencies.values().sum();
        let unique_words = word_frequencies.len();

        // Sort by frequency and take top words
        let mut sorted_words: Vec<(String, u32)> = word_frequencies.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_words.truncate(self.config.max_words);

        let max_freq = sorted_words.first().map(|(_, freq)| *freq).unwrap_or(1);
        let min_freq = sorted_words.last().map(|(_, freq)| *freq).unwrap_or(1);

        let words: Vec<WordCloudWord> = sorted_words
            .into_iter()
            .enumerate()
            .map(|(i, (word, freq))| {
                let normalized_freq = if max_freq == min_freq {
                    1.0
                } else {
                    (freq - min_freq) as f64 / (max_freq - min_freq) as f64
                };

                let font_size = self.config.font_size_range.0 + 
                    (normalized_freq * (self.config.font_size_range.1 - self.config.font_size_range.0) as f64) as u32;

                // Generate color based on frequency (blue to red scale)
                let color = if self.config.complexity_colors {
                    format!("#{:02x}{:02x}{:02x}", 
                        (50 + (normalized_freq * 200.0) as u8),
                        (100 - (normalized_freq * 50.0) as u8),
                        (200 - (normalized_freq * 150.0) as u8))
                } else {
                    "#2563eb".to_string() // Default blue
                };

                WordCloudWord {
                    text: word,
                    frequency: freq,
                    font_size,
                    color,
                    position: None, // Will be calculated by frontend
                }
            })
            .collect();

        Ok(WordCloudData {
            words,
            total_words: total_words as usize,
            unique_words,
        })
    }

    fn build_text_structure(&self, text_units: &[TextUnit]) -> Result<TextStructureData> {
        // Group units by type and build hierarchy
        let mut level_stats = HashMap::new();
        let hierarchy = self.build_hierarchy_nodes(text_units, 0, &mut level_stats)?;
        
        let max_depth = level_stats.keys().max().copied().unwrap_or(0);

        Ok(TextStructureData {
            hierarchy,
            max_depth,
            level_stats,
        })
    }

    fn build_hierarchy_nodes(
        &self, 
        text_units: &[TextUnit], 
        depth: usize,
        level_stats: &mut HashMap<usize, LevelStatistics>
    ) -> Result<Vec<TextStructureNode>> {
        let mut nodes = Vec::new();
        let mut type_counts = HashMap::new();
        let mut total_words = 0;
        let mut total_complexity = 0.0;

        for unit in text_units {
            let word_count = unit.content.split_whitespace().count();
            let complexity = unit.complexity();

            total_words += word_count;
            total_complexity += complexity;

            let type_name = format!("{:?}", unit.unit_type);
            *type_counts.entry(type_name.clone()).or_insert(0) += 1;

            let node = TextStructureNode {
                id: unit.id.to_string(),
                label: if unit.content.len() > 50 {
                    format!("{}...", &unit.content[..47])
                } else {
                    unit.content.clone()
                },
                node_type: type_name,
                children: Vec::new(), // Simplified - would need hierarchy analysis
                metadata: HashMap::new(),
                depth,
                word_count,
                complexity,
            };

            nodes.push(node);
        }

        // Calculate level statistics
        if !text_units.is_empty() {
            let common_type = type_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(type_name, _)| type_name)
                .unwrap_or_default();

            level_stats.insert(depth, LevelStatistics {
                node_count: text_units.len(),
                avg_word_count: total_words as f64 / text_units.len() as f64,
                avg_complexity: total_complexity / text_units.len() as f64,
                common_type,
            });
        }

        Ok(nodes)
    }

    fn calculate_readability_progression(&self, text_units: &[TextUnit]) -> Result<Vec<f64>> {
        Ok(text_units.iter().map(|unit| unit.readability_score()).collect())
    }

    fn analyze_sentence_complexity(&self, text_units: &[TextUnit]) -> Result<Vec<(String, usize)>> {
        let mut complexity_ranges = HashMap::new();

        for unit in text_units {
            let complexity = unit.complexity();
            let range = match complexity {
                x if x < 0.2 => "Very Simple (0.0-0.2)",
                x if x < 0.4 => "Simple (0.2-0.4)",
                x if x < 0.6 => "Moderate (0.4-0.6)",
                x if x < 0.8 => "Complex (0.6-0.8)",
                _ => "Very Complex (0.8-1.0)",
            };

            *complexity_ranges.entry(range.to_string()).or_insert(0) += 1;
        }

        Ok(complexity_ranges.into_iter().collect())
    }

    fn extract_sentence_structures(&self, text_units: &[TextUnit]) -> Result<Vec<SentenceStructure>> {
        let mut structures = Vec::new();

        for (position, unit) in text_units.iter().enumerate() {
            if unit.unit_type == TextUnitType::Sentence {
                let word_count = unit.content.split_whitespace().count();
                let complexity = unit.complexity();

                structures.push(SentenceStructure {
                    text: unit.content.clone(),
                    word_count,
                    complexity,
                    position,
                });
            }
        }

        Ok(structures)
    }

    fn analyze_topic_distribution(&self, text_units: &[TextUnit]) -> Result<HashMap<String, f64>> {
        // Simplified topic analysis - in practice, this would use more sophisticated NLP
        let mut topic_keywords: HashMap<String, Vec<String>> = HashMap::new();
        
        // Define some basic topic categories with keywords
        let topics = vec![
            ("Technology", vec!["computer", "software", "digital", "internet", "technology", "data"]),
            ("Science", vec!["research", "study", "analysis", "experiment", "scientific", "theory"]),
            ("Business", vec!["company", "market", "economic", "business", "financial", "revenue"]),
            ("Education", vec!["learn", "study", "education", "teaching", "academic", "knowledge"]),
            ("Health", vec!["health", "medical", "patient", "treatment", "medicine", "clinical"]),
        ];

        let mut topic_scores = HashMap::new();
        let mut total_words = 0;

        for unit in text_units {
            let words: Vec<&str> = unit.content.to_lowercase().split_whitespace().collect();
            total_words += words.len();

            for (topic, keywords) in &topics {
                let matches = words.iter()
                    .filter(|word| keywords.iter().any(|keyword| word.contains(keyword)))
                    .count();
                
                *topic_scores.entry(topic.to_string()).or_insert(0) += matches;
            }
        }

        // Convert to percentages
        let mut topic_distribution = HashMap::new();
        for (topic, score) in topic_scores {
            let percentage = if total_words > 0 {
                (score as f64 / total_words as f64) * 100.0
            } else {
                0.0
            };
            topic_distribution.insert(topic, percentage);
        }

        Ok(topic_distribution)
    }
}

impl Default for TextVisualization {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for text analysis
pub mod text_utils {
    use super::*;

    /// Calculate text statistics
    pub fn calculate_text_stats(text_units: &[TextUnit]) -> TextStatistics {
        let total_chars: usize = text_units.iter().map(|u| u.content.len()).sum();
        let total_words: usize = text_units.iter()
            .map(|u| u.content.split_whitespace().count())
            .sum();
        let total_sentences = text_units.iter()
            .filter(|u| u.unit_type == TextUnitType::Sentence)
            .count();
        
        let avg_complexity: f64 = if !text_units.is_empty() {
            text_units.iter().map(|u| u.complexity()).sum::<f64>() / text_units.len() as f64
        } else {
            0.0
        };

        let avg_readability: f64 = if !text_units.is_empty() {
            text_units.iter().map(|u| u.readability_score()).sum::<f64>() / text_units.len() as f64
        } else {
            0.0
        };

        TextStatistics {
            total_characters: total_chars,
            total_words,
            total_sentences,
            average_complexity: avg_complexity,
            average_readability: avg_readability,
            unit_count: text_units.len(),
        }
    }

    /// Extract key phrases from text
    pub fn extract_key_phrases(text: &str, max_phrases: usize) -> Vec<String> {
        // Simple phrase extraction based on common patterns
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut phrases = Vec::new();

        // Extract 2-3 word phrases
        for window_size in 2..=3 {
            for window in words.windows(window_size) {
                if window.len() >= 2 && 
                   window.iter().all(|w| w.len() > 2 && w.chars().all(|c| c.is_alphabetic() || c.is_whitespace())) {
                    phrases.push(window.join(" "));
                }
            }
        }

        // Remove duplicates and sort by length
        phrases.sort();
        phrases.dedup();
        phrases.sort_by(|a, b| b.len().cmp(&a.len()));
        phrases.truncate(max_phrases);

        phrases
    }
}

/// Text statistics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    pub total_characters: usize,
    pub total_words: usize,
    pub total_sentences: usize,
    pub average_complexity: f64,
    pub average_readability: f64,
    pub unit_count: usize,
} 