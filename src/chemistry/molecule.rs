//! Molecule analysis module
//!
//! This module provides advanced molecular analysis functionality.

use std::collections::{HashMap, HashSet};
use super::{Molecule, Atom, Bond, BondType, Unit};

/// Molecular analyzer
#[derive(Debug, Clone)]
pub struct MolecularAnalyzer {
    /// Analysis configuration
    config: MolecularAnalysisConfig,
}

/// Configuration for molecular analysis
#[derive(Debug, Clone)]
pub struct MolecularAnalysisConfig {
    /// Enable ring detection
    pub enable_ring_detection: bool,
    /// Enable aromaticity detection
    pub enable_aromaticity_detection: bool,
    /// Enable stereochemistry analysis
    pub enable_stereochemistry: bool,
    /// Maximum ring size to detect
    pub max_ring_size: usize,
}

/// Molecular analysis result
#[derive(Debug, Clone)]
pub struct MolecularAnalysisResult {
    /// Molecule ID
    pub molecule_id: String,
    /// Basic properties
    pub basic_properties: BasicProperties,
    /// Ring analysis
    pub ring_analysis: RingAnalysis,
    /// Functional groups
    pub functional_groups: Vec<FunctionalGroup>,
    /// Stereochemistry
    pub stereochemistry: StereochemistryAnalysis,
    /// Aromaticity
    pub aromaticity: AromaticityAnalysis,
}

/// Basic molecular properties
#[derive(Debug, Clone)]
pub struct BasicProperties {
    /// Molecular formula
    pub molecular_formula: String,
    /// Molecular weight
    pub molecular_weight: f64,
    /// Number of atoms
    pub atom_count: usize,
    /// Number of bonds
    pub bond_count: usize,
    /// Number of heavy atoms
    pub heavy_atom_count: usize,
}

/// Ring analysis results
#[derive(Debug, Clone)]
pub struct RingAnalysis {
    /// Total number of rings
    pub total_rings: usize,
    /// Ring sizes
    pub ring_sizes: Vec<usize>,
    /// Aromatic rings
    pub aromatic_rings: usize,
    /// Aliphatic rings
    pub aliphatic_rings: usize,
    /// Ring systems
    pub ring_systems: Vec<RingSystem>,
}

/// Ring system
#[derive(Debug, Clone)]
pub struct RingSystem {
    /// Rings in the system
    pub rings: Vec<Ring>,
    /// System type
    pub system_type: RingSystemType,
    /// Is aromatic
    pub is_aromatic: bool,
}

/// Ring system types
#[derive(Debug, Clone)]
pub enum RingSystemType {
    /// Isolated ring
    Isolated,
    /// Fused rings
    Fused,
    /// Bridged rings
    Bridged,
    /// Spiro rings
    Spiro,
}

/// Individual ring
#[derive(Debug, Clone)]
pub struct Ring {
    /// Atom indices in the ring
    pub atoms: Vec<usize>,
    /// Ring size
    pub size: usize,
    /// Is aromatic
    pub is_aromatic: bool,
}

/// Functional group
#[derive(Debug, Clone)]
pub struct FunctionalGroup {
    /// Group name
    pub name: String,
    /// Group type
    pub group_type: FunctionalGroupType,
    /// Atom indices
    pub atoms: Vec<usize>,
    /// SMARTS pattern
    pub smarts: String,
    /// Count
    pub count: usize,
}

/// Functional group types
#[derive(Debug, Clone)]
pub enum FunctionalGroupType {
    /// Alcohol
    Alcohol,
    /// Aldehyde
    Aldehyde,
    /// Ketone
    Ketone,
    /// Carboxylic acid
    CarboxylicAcid,
    /// Ester
    Ester,
    /// Ether
    Ether,
    /// Amine
    Amine,
    /// Amide
    Amide,
    /// Aromatic ring
    AromaticRing,
    /// Halogen
    Halogen,
    /// Custom
    Custom(String),
}

/// Stereochemistry analysis
#[derive(Debug, Clone)]
pub struct StereochemistryAnalysis {
    /// Chiral centers
    pub chiral_centers: Vec<ChiralCenter>,
    /// Double bond stereochemistry
    pub double_bonds: Vec<DoubleBondStereo>,
    /// Is chiral
    pub is_chiral: bool,
    /// Number of stereoisomers
    pub stereoisomer_count: usize,
}

/// Chiral center
#[derive(Debug, Clone)]
pub struct ChiralCenter {
    /// Atom index
    pub atom_index: usize,
    /// Chirality (R/S)
    pub chirality: ChiralityDesignation,
    /// Confidence
    pub confidence: f64,
}

/// Chirality designation
#[derive(Debug, Clone)]
pub enum ChiralityDesignation {
    R,
    S,
    Unknown,
}

/// Double bond stereochemistry
#[derive(Debug, Clone)]
pub struct DoubleBondStereo {
    /// Bond atoms
    pub bond_atoms: (usize, usize),
    /// Stereochemistry (E/Z)
    pub stereochemistry: EZDesignation,
    /// Confidence
    pub confidence: f64,
}

/// E/Z designation
#[derive(Debug, Clone)]
pub enum EZDesignation {
    E,
    Z,
    Unknown,
}

/// Aromaticity analysis
#[derive(Debug, Clone)]
pub struct AromaticityAnalysis {
    /// Aromatic atoms
    pub aromatic_atoms: Vec<usize>,
    /// Aromatic bonds
    pub aromatic_bonds: Vec<(usize, usize)>,
    /// Aromatic rings
    pub aromatic_rings: Vec<usize>,
    /// Total aromatic atom count
    pub aromatic_atom_count: usize,
}

impl Default for MolecularAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_ring_detection: true,
            enable_aromaticity_detection: true,
            enable_stereochemistry: true,
            max_ring_size: 10,
        }
    }
}

impl MolecularAnalyzer {
    /// Create new molecular analyzer
    pub fn new(config: MolecularAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(MolecularAnalysisConfig::default())
    }

    /// Analyze molecule
    pub fn analyze(&self, molecule: &Molecule) -> MolecularAnalysisResult {
        let molecule_id = format!("{}", molecule.id());
        
        // Basic properties
        let basic_properties = self.analyze_basic_properties(molecule);
        
        // Ring analysis
        let ring_analysis = if self.config.enable_ring_detection {
            self.analyze_rings(molecule)
        } else {
            RingAnalysis {
                total_rings: 0,
                ring_sizes: Vec::new(),
                aromatic_rings: 0,
                aliphatic_rings: 0,
                ring_systems: Vec::new(),
            }
        };
        
        // Functional groups
        let functional_groups = self.find_functional_groups(molecule);
        
        // Stereochemistry
        let stereochemistry = if self.config.enable_stereochemistry {
            self.analyze_stereochemistry(molecule)
        } else {
            StereochemistryAnalysis {
                chiral_centers: Vec::new(),
                double_bonds: Vec::new(),
                is_chiral: false,
                stereoisomer_count: 1,
            }
        };
        
        // Aromaticity
        let aromaticity = if self.config.enable_aromaticity_detection {
            self.analyze_aromaticity(molecule)
        } else {
            AromaticityAnalysis {
                aromatic_atoms: Vec::new(),
                aromatic_bonds: Vec::new(),
                aromatic_rings: Vec::new(),
                aromatic_atom_count: 0,
            }
        };
        
        MolecularAnalysisResult {
            molecule_id,
            basic_properties,
            ring_analysis,
            functional_groups,
            stereochemistry,
            aromaticity,
        }
    }

    /// Analyze basic molecular properties
    fn analyze_basic_properties(&self, molecule: &Molecule) -> BasicProperties {
        let atoms = molecule.atoms();
        let bonds = molecule.bonds();
        
        let molecular_formula = molecule.formula();
        let molecular_weight = molecule.molecular_weight();
        let atom_count = atoms.len();
        let bond_count = bonds.len();
        
        // Count heavy atoms (non-hydrogen)
        let heavy_atom_count = atoms.iter()
            .filter(|atom| atom.symbol != "H")
            .count();
        
        BasicProperties {
            molecular_formula,
            molecular_weight,
            atom_count,
            bond_count,
            heavy_atom_count,
        }
    }

    /// Analyze rings in molecule
    fn analyze_rings(&self, molecule: &Molecule) -> RingAnalysis {
        let rings = self.find_rings(molecule);
        let total_rings = rings.len();
        let ring_sizes: Vec<usize> = rings.iter().map(|r| r.size).collect();
        
        let aromatic_rings = rings.iter().filter(|r| r.is_aromatic).count();
        let aliphatic_rings = total_rings - aromatic_rings;
        
        let ring_systems = self.find_ring_systems(&rings);
        
        RingAnalysis {
            total_rings,
            ring_sizes,
            aromatic_rings,
            aliphatic_rings,
            ring_systems,
        }
    }

    /// Find rings using simplified algorithm
    fn find_rings(&self, molecule: &Molecule) -> Vec<Ring> {
        let mut rings = Vec::new();
        let atoms = molecule.atoms();
        let bonds = molecule.bonds();
        
        // Build adjacency list
        let mut adjacency = HashMap::new();
        for (i, _) in atoms.iter().enumerate() {
            adjacency.insert(i, Vec::new());
        }
        
        for bond in bonds {
            adjacency.get_mut(&bond.start).unwrap().push(bond.end);
            adjacency.get_mut(&bond.end).unwrap().push(bond.start);
        }
        
        // Simple ring detection using DFS (simplified)
        for start_atom in 0..atoms.len() {
            if let Some(ring_atoms) = self.find_ring_from_atom(start_atom, &adjacency, self.config.max_ring_size) {
                if ring_atoms.len() >= 3 && ring_atoms.len() <= self.config.max_ring_size {
                    let is_aromatic = self.is_ring_aromatic(&ring_atoms, molecule);
                    rings.push(Ring {
                        atoms: ring_atoms.clone(),
                        size: ring_atoms.len(),
                        is_aromatic,
                    });
                }
            }
        }
        
        // Remove duplicate rings
        rings.sort_by(|a, b| a.atoms.cmp(&b.atoms));
        rings.dedup_by(|a, b| a.atoms == b.atoms);
        
        rings
    }

    /// Find ring starting from specific atom (simplified)
    fn find_ring_from_atom(&self, start: usize, adjacency: &HashMap<usize, Vec<usize>>, max_size: usize) -> Option<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        
        if self.dfs_ring_search(start, start, &mut visited, &mut path, adjacency, max_size) {
            Some(path)
        } else {
            None
        }
    }

    /// DFS for ring search (simplified)
    fn dfs_ring_search(&self, current: usize, target: usize, visited: &mut HashSet<usize>, 
                      path: &mut Vec<usize>, adjacency: &HashMap<usize, Vec<usize>>, max_size: usize) -> bool {
        if path.len() > max_size {
            return false;
        }
        
        path.push(current);
        
        if path.len() > 2 && current == target {
            return true;
        }
        
        visited.insert(current);
        
        if let Some(neighbors) = adjacency.get(&current) {
            for &neighbor in neighbors {
                if neighbor == target && path.len() >= 3 {
                    return true;
                } else if !visited.contains(&neighbor) {
                    if self.dfs_ring_search(neighbor, target, visited, path, adjacency, max_size) {
                        return true;
                    }
                }
            }
        }
        
        path.pop();
        visited.remove(&current);
        false
    }

    /// Check if ring is aromatic (simplified)
    fn is_ring_aromatic(&self, ring_atoms: &[usize], molecule: &Molecule) -> bool {
        let atoms = molecule.atoms();
        
        // Simple heuristic: if all atoms in ring are carbon and ring size is 6, consider aromatic
        if ring_atoms.len() == 6 {
            ring_atoms.iter().all(|&i| atoms[i].symbol == "C")
        } else {
            false
        }
    }

    /// Find ring systems
    fn find_ring_systems(&self, rings: &[Ring]) -> Vec<RingSystem> {
        let mut systems = Vec::new();
        
        // Simplified: each ring is its own system
        for ring in rings {
            systems.push(RingSystem {
                rings: vec![ring.clone()],
                system_type: RingSystemType::Isolated,
                is_aromatic: ring.is_aromatic,
            });
        }
        
        systems
    }

    /// Find functional groups
    fn find_functional_groups(&self, molecule: &Molecule) -> Vec<FunctionalGroup> {
        let mut groups = Vec::new();
        let atoms = molecule.atoms();
        
        // Simple functional group detection
        for (i, atom) in atoms.iter().enumerate() {
            match atom.symbol.as_str() {
                "O" => {
                    groups.push(FunctionalGroup {
                        name: "Oxygen-containing".to_string(),
                        group_type: FunctionalGroupType::Ether,
                        atoms: vec![i],
                        smarts: "[O]".to_string(),
                        count: 1,
                    });
                }
                "N" => {
                    groups.push(FunctionalGroup {
                        name: "Nitrogen-containing".to_string(),
                        group_type: FunctionalGroupType::Amine,
                        atoms: vec![i],
                        smarts: "[N]".to_string(),
                        count: 1,
                    });
                }
                _ => {}
            }
        }
        
        groups
    }

    /// Analyze stereochemistry
    fn analyze_stereochemistry(&self, _molecule: &Molecule) -> StereochemistryAnalysis {
        // Simplified stereochemistry analysis
        StereochemistryAnalysis {
            chiral_centers: Vec::new(),
            double_bonds: Vec::new(),
            is_chiral: false,
            stereoisomer_count: 1,
        }
    }

    /// Analyze aromaticity
    fn analyze_aromaticity(&self, molecule: &Molecule) -> AromaticityAnalysis {
        let atoms = molecule.atoms();
        let mut aromatic_atoms = Vec::new();
        
        // Simple aromaticity detection
        for (i, atom) in atoms.iter().enumerate() {
            if atom.is_aromatic {
                aromatic_atoms.push(i);
            }
        }
        
        AromaticityAnalysis {
            aromatic_atom_count: aromatic_atoms.len(),
            aromatic_atoms,
            aromatic_bonds: Vec::new(),
            aromatic_rings: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::Atom;

    #[test]
    fn test_molecular_analysis() {
        let analyzer = MolecularAnalyzer::default();
        let mut molecule = Molecule::new(b"test molecule", "test");
        
        // Add some atoms
        molecule.add_atom(Atom::new("C", 6));
        molecule.add_atom(Atom::new("C", 6));
        molecule.add_atom(Atom::new("O", 8));
        
        let result = analyzer.analyze(&molecule);
        
        assert_eq!(result.basic_properties.atom_count, 3);
        assert_eq!(result.basic_properties.heavy_atom_count, 3);
    }
} 