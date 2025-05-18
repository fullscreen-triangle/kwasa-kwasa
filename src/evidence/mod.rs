use std::collections::HashMap;
use crate::turbulance::datastructures::{EvidenceNetwork, EvidenceNode, EdgeType, UncertaintyQuantifier};
use crate::genomic::{NucleotideSequence, GeneUnit, MotifUnit, Position, UnitId as GenomicUnitId};
use crate::spectrometry::{MassSpectrum, Peak, UnitId as SpectrumUnitId};
use crate::turbulance::proposition::Motion;
use rayon::prelude::*;

/// Integration module for EvidenceNetwork with genomic and spectrometry data
pub struct EvidenceIntegration;

impl EvidenceIntegration {
    /// Create a new evidence integration instance
    pub fn new() -> Self {
        Self
    }
    
    /// Build an evidence network from genomic and mass spectrometry data
    pub fn build_network(&self, 
                        genomic_data: &[NucleotideSequence], 
                        spectral_data: &[MassSpectrum],
                        threshold: f64) -> EvidenceNetwork {
        let mut network = EvidenceNetwork::new();
        
        // Process genomic data
        self.add_genomic_evidence(&mut network, genomic_data);
        
        // Process spectral data
        self.add_spectral_evidence(&mut network, spectral_data);
        
        // Find relationships between genomic and spectral data
        self.connect_cross_domain_evidence(&mut network, genomic_data, spectral_data, threshold);
        
        // Propagate beliefs through the network
        network.propagate_beliefs();
        
        network
    }
    
    /// Add genomic sequences as evidence nodes
    fn add_genomic_evidence(&self, network: &mut EvidenceNetwork, sequences: &[NucleotideSequence]) {
        for (idx, sequence) in sequences.iter().enumerate() {
            let id = format!("genomic_{}", idx);
            
            // Create a Motion object with the sequence content
            let content = String::from_utf8_lossy(sequence.content()).to_string();
            let motion = Motion::new(content);
            
            // Add to network
            network.add_node(&id, EvidenceNode::GenomicFeature { 
                sequence: sequence.content().to_vec(), 
                position: sequence.metadata().position.as_ref().map(|p| 
                    format!("{}:{}-{}", p.reference, p.start, p.end)
                ),
                motion 
            });
            
            // Set initial belief (can be adjusted based on data quality, source reliability, etc.)
            network.set_belief(&id, 0.7);
        }
    }
    
    /// Add mass spectra as evidence nodes
    fn add_spectral_evidence(&self, network: &mut EvidenceNetwork, spectra: &[MassSpectrum]) {
        for (idx, spectrum) in spectra.iter().enumerate() {
            let id = format!("spectrum_{}", idx);
            
            // Extract peaks for storage
            let peaks: Vec<(f64, f64)> = spectrum.peaks()
                .iter()
                .map(|peak| (peak.mz, peak.intensity))
                .collect();
            
            // Get retention time if available
            let retention_time = spectrum.metadata().annotations
                .get("retention_time")
                .and_then(|rt_str| rt_str.parse::<f64>().ok())
                .unwrap_or(0.0);
            
            // Create a Motion object with spectral summary
            let content = format!("Spectrum with {} peaks, base peak m/z: {}", 
                peaks.len(), 
                spectrum.base_peak().map(|p| p.mz).unwrap_or(0.0)
            );
            let motion = Motion::new(content);
            
            // Add to network
            network.add_node(&id, EvidenceNode::Spectra { 
                peaks, 
                retention_time,
                motion 
            });
            
            // Set initial belief (can be adjusted based on spectral quality)
            network.set_belief(&id, 0.75);
        }
    }
    
    /// Find and create connections between genomic and spectral data
    fn connect_cross_domain_evidence(&self, 
                                    network: &mut EvidenceNetwork, 
                                    genomic_data: &[NucleotideSequence], 
                                    spectral_data: &[MassSpectrum],
                                    threshold: f64) {
        // This is a simplified implementation - a real one would use sophisticated
        // algorithms to find meaningful relationships between the data types
        
        // For demonstration, use a simple approach: connect spectra that could represent
        // products of genes (e.g., proteins, metabolites)
        
        // Process in parallel for better performance
        let connections: Vec<(String, String, EdgeType, f64)> = (0..genomic_data.len())
            .into_par_iter()
            .flat_map(|g_idx| {
                let sequence = &genomic_data[g_idx];
                let genomic_id = format!("genomic_{}", g_idx);
                
                // For each spectrum
                (0..spectral_data.len())
                    .filter_map(move |s_idx| {
                        let spectrum = &spectral_data[s_idx];
                        let spectrum_id = format!("spectrum_{}", s_idx);
                        
                        // Calculate a correlation score (simplified example)
                        let correlation = self.calculate_correlation(sequence, spectrum);
                        
                        if correlation >= threshold {
                            // Determine edge type based on correlation pattern
                            let edge_type = if correlation > 0.8 {
                                EdgeType::Supports { strength: correlation }
                            } else {
                                EdgeType::BindsTo { affinity: correlation }
                            };
                            
                            // Uncertainty decreases with higher correlation
                            let uncertainty = 1.0 - correlation;
                            
                            Some((genomic_id.clone(), spectrum_id, edge_type, uncertainty))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        
        // Add all connections to the network
        for (from, to, edge_type, uncertainty) in connections {
            network.add_edge(&from, &to, edge_type, uncertainty);
        }
    }
    
    /// Calculate correlation between a genomic sequence and a spectrum (simplified)
    fn calculate_correlation(&self, sequence: &NucleotideSequence, spectrum: &MassSpectrum) -> f64 {
        // This is a very simplified example - real implementations would use
        // domain-specific algorithms to calculate meaningful correlations
        
        // For demonstration, use a simplistic approach based on GC content and spectrum intensity
        let gc_content = sequence.gc_content();
        
        // Calculate average m/z (weighted by intensity)
        let weighted_mz = spectrum.peaks().iter()
            .map(|p| p.mz * p.intensity)
            .sum::<f64>();
            
        let total_intensity = spectrum.peaks().iter()
            .map(|p| p.intensity)
            .sum::<f64>();
            
        let avg_mz = if total_intensity > 0.0 { 
            weighted_mz / total_intensity 
        } else { 
            0.0 
        };
        
        // Normalize both values to 0-1 range
        let norm_gc = gc_content; // Already 0-1
        let norm_mz = (avg_mz / 2000.0).min(1.0); // Assuming max m/z of 2000
        
        // Simple correlation formula (for demonstration)
        let base_correlation = 0.5 + (norm_gc - 0.5) * (norm_mz - 0.5);
        
        // Add some randomness for demonstration
        let random_factor = rand::random::<f64>() * 0.2;
        
        (base_correlation + random_factor).max(0.0).min(1.0)
    }
    
    /// Analyze conflicts in the evidence network
    pub fn analyze_conflicts(&self, network: &EvidenceNetwork) -> Vec<ConflictReport> {
        let conflicts = network.identify_conflicts();
        
        conflicts.into_iter()
            .map(|(from, to, strength)| {
                let from_belief = network.get_belief(&from).unwrap_or(0.5);
                let to_belief = network.get_belief(&to).unwrap_or(0.5);
                
                ConflictReport {
                    source_id: from,
                    target_id: to,
                    conflict_strength: strength,
                    source_belief: from_belief,
                    target_belief: to_belief,
                }
            })
            .collect()
    }
    
    /// Find critical evidence nodes for a specific conclusion
    pub fn identify_critical_evidence(&self, 
                                     network: &EvidenceNetwork, 
                                     target_id: &str) -> Vec<CriticalEvidence> {
        let sensitivities = network.sensitivity_analysis(target_id);
        
        sensitivities.into_iter()
            .map(|(id, impact)| {
                let node = network.get_node(&id);
                let belief = network.get_belief(&id).unwrap_or(0.5);
                
                CriticalEvidence {
                    node_id: id,
                    node_type: get_node_type(node),
                    impact,
                    belief,
                }
            })
            .collect()
    }
    
    /// Export evidence network to a graph format suitable for visualization
    pub fn export_graph(&self, network: &EvidenceNetwork) -> VisGraph {
        let mut graph = VisGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        
        // Process all nodes
        for (id, node) in network.nodes().iter() {
            let node_type = get_node_type(Some(node));
            let belief = network.get_belief(id).unwrap_or(0.5);
            
            graph.nodes.push(VisNode {
                id: id.clone(),
                node_type,
                belief,
                label: get_node_label(node),
            });
        }
        
        // Process all edges
        for (source, edges) in network.edges().iter() {
            for (target, edge_type, uncertainty) in edges {
                graph.edges.push(VisEdge {
                    source: source.clone(),
                    target: target.clone(),
                    edge_type: format!("{:?}", edge_type),
                    weight: get_edge_weight(edge_type),
                    uncertainty: *uncertainty,
                });
            }
        }
        
        graph
    }
}

/// Helper function to get node type as string
fn get_node_type(node: Option<&EvidenceNode>) -> String {
    match node {
        Some(EvidenceNode::Molecule { .. }) => "Molecule".to_string(),
        Some(EvidenceNode::Spectra { .. }) => "Spectrum".to_string(),
        Some(EvidenceNode::GenomicFeature { .. }) => "Genomic".to_string(),
        Some(EvidenceNode::Evidence { .. }) => "Evidence".to_string(),
        None => "Unknown".to_string(),
    }
}

/// Helper function to get a human-readable label for a node
fn get_node_label(node: &EvidenceNode) -> String {
    match node {
        EvidenceNode::Molecule { formula, .. } => format!("Molecule: {}", formula),
        EvidenceNode::Spectra { peaks, .. } => format!("Spectrum: {} peaks", peaks.len()),
        EvidenceNode::GenomicFeature { sequence, position, .. } => {
            let seq_preview = if sequence.len() > 10 {
                format!("{}...", String::from_utf8_lossy(&sequence[0..10]))
            } else {
                String::from_utf8_lossy(sequence).to_string()
            };
            
            if let Some(pos) = position {
                format!("Sequence: {} at {}", seq_preview, pos)
            } else {
                format!("Sequence: {}", seq_preview)
            }
        },
        EvidenceNode::Evidence { source, timestamp, .. } => {
            format!("Evidence from {} at {}", source, timestamp)
        },
    }
}

/// Helper function to get edge weight from edge type
fn get_edge_weight(edge_type: &EdgeType) -> f64 {
    match edge_type {
        EdgeType::Supports { strength } => *strength,
        EdgeType::Contradicts { strength } => *strength,
        EdgeType::PartOf => 0.8,
        EdgeType::Catalyzes { rate } => *rate,
        EdgeType::Transforms => 0.6,
        EdgeType::BindsTo { affinity } => *affinity,
    }
}

/// Report of a conflict in the evidence network
#[derive(Debug, Clone)]
pub struct ConflictReport {
    /// ID of the contradicting source
    pub source_id: String,
    /// ID of the contradicted target
    pub target_id: String,
    /// Strength of the conflict
    pub conflict_strength: f64,
    /// Belief in the source
    pub source_belief: f64,
    /// Belief in the target
    pub target_belief: f64,
}

/// Information about a critical evidence node
#[derive(Debug, Clone)]
pub struct CriticalEvidence {
    /// ID of the node
    pub node_id: String,
    /// Type of the node
    pub node_type: String,
    /// Impact on the conclusion
    pub impact: f64,
    /// Current belief in this node
    pub belief: f64,
}

/// Visualization graph structure
#[derive(Debug, Clone)]
pub struct VisGraph {
    /// Nodes in the graph
    pub nodes: Vec<VisNode>,
    /// Edges in the graph
    pub edges: Vec<VisEdge>,
}

/// Visualization node
#[derive(Debug, Clone)]
pub struct VisNode {
    /// Node ID
    pub id: String,
    /// Node type
    pub node_type: String,
    /// Current belief
    pub belief: f64,
    /// Human-readable label
    pub label: String,
}

/// Visualization edge
#[derive(Debug, Clone)]
pub struct VisEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge type
    pub edge_type: String,
    /// Edge weight
    pub weight: f64,
    /// Edge uncertainty
    pub uncertainty: f64,
} 