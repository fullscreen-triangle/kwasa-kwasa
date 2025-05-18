use kwasa_kwasa::prelude::*;
use kwasa_kwasa::genomic::{NucleotideSequence, GenomicMetadata, Strand, Position};
use kwasa_kwasa::spectrometry::{MassSpectrum, Peak, SpectrumMetadata};
use kwasa_kwasa::genomic::high_throughput::{HighThroughputGenomics, SequenceCompressor};
use kwasa_kwasa::spectrometry::high_throughput::HighThroughputSpectrometry;
use kwasa_kwasa::evidence::EvidenceIntegration;
use std::collections::HashMap;

fn main() {
    println!("Evidence Network Integration for Genomic and Mass Spectrometry Data");
    println!("===============================================================");
    
    // 1. Create some genomic data
    println!("1. Creating genomic data...");
    let genomic_data = create_sample_genomic_data();
    println!("   Created {} genomic sequences", genomic_data.len());
    
    // 2. Create some mass spectrometry data
    println!("2. Creating mass spectrometry data...");
    let spectral_data = create_sample_spectral_data();
    println!("   Created {} mass spectra", spectral_data.len());
    
    // 3. Demonstrate high-throughput genomic operations
    println!("3. Demonstrating high-throughput genomic operations...");
    demonstrate_genomic_operations(&genomic_data);
    
    // 4. Demonstrate high-throughput spectrometry operations
    println!("4. Demonstrating high-throughput spectrometry operations...");
    demonstrate_spectrometry_operations(&spectral_data);
    
    // 5. Build evidence network
    println!("5. Building evidence network...");
    let evidence_integration = EvidenceIntegration::new();
    let network = evidence_integration.build_network(&genomic_data, &spectral_data, 0.6);
    println!("   Evidence network built successfully");
    
    // 6. Analyze conflicts
    println!("6. Analyzing conflicts in evidence...");
    let conflicts = evidence_integration.analyze_conflicts(&network);
    println!("   Found {} conflicts in the evidence", conflicts.len());
    
    for (i, conflict) in conflicts.iter().enumerate().take(3) {
        println!("   Conflict {}: {} vs {} (strength: {:.2})", 
            i + 1,
            conflict.source_id,
            conflict.target_id,
            conflict.conflict_strength
        );
    }
    
    // 7. Identify critical evidence
    println!("7. Identifying critical evidence...");
    // Use the first genomic node as our target
    let target_id = "genomic_0";
    let critical_evidence = evidence_integration.identify_critical_evidence(&network, target_id);
    println!("   Found {} critical evidence nodes for {}", critical_evidence.len(), target_id);
    
    for (i, evidence) in critical_evidence.iter().enumerate().take(3) {
        println!("   Critical evidence {}: {} ({}) - impact: {:.2}", 
            i + 1,
            evidence.node_id,
            evidence.node_type,
            evidence.impact
        );
    }
    
    // 8. Export network graph for visualization
    println!("8. Exporting network for visualization...");
    let graph = evidence_integration.export_graph(&network);
    println!("   Graph exported with {} nodes and {} edges", 
        graph.nodes.len(), 
        graph.edges.len()
    );
    
    println!("\nDemonstration complete!");
}

/// Create sample genomic data for demonstration
fn create_sample_genomic_data() -> Vec<NucleotideSequence> {
    let mut sequences = Vec::new();
    
    // Sample gene sequence (simplified)
    let gene1 = "ATGGCGTACGCGTACTGACTCGATCGTAGCATCAGTCGATGCTACGTAGCTACGATCGACTGATCGATCGA";
    let gene2 = "ATGCTACGTACGACTGACTGCAGCAGCAGCAGCAGCAGCAGCAGCATGCTAGCTAGCTAGCTAGCTAGCTA";
    let gene3 = "GCTAGCTAGCTAGCTAGCTAGCATGCATGCATGCATGCTAGCTAGCTAGCTAGCTGACTGACTGACTGACT";
    
    // Create sequences with metadata
    for (i, gene) in [gene1, gene2, gene3].iter().enumerate() {
        let mut metadata = GenomicMetadata {
            source: Some(format!("Sample{}", i+1)),
            strand: Strand::Forward,
            position: Some(Position {
                start: i * 1000,
                end: i * 1000 + gene.len(),
                reference: format!("chr{}", i+1),
            }),
            annotations: HashMap::new(),
        };
        
        // Add some annotations
        metadata.annotations.insert("gene_id".to_string(), format!("GENE{}", i+1));
        metadata.annotations.insert("gene_name".to_string(), format!("Sample gene {}", i+1));
        
        let mut sequence = NucleotideSequence::new(gene.as_bytes(), format!("gene{}", i+1));
        sequence.set_metadata(metadata);
        
        sequences.push(sequence);
    }
    
    sequences
}

/// Create sample mass spectrometry data for demonstration
fn create_sample_spectral_data() -> Vec<MassSpectrum> {
    let mut spectra = Vec::new();
    
    // Create several spectra with different characteristics
    for i in 0..5 {
        let mut peaks = Vec::new();
        
        // Add some peaks - in a real scenario these would be actual mass spec data
        for j in 0..20 {
            let mz = 100.0 + j as f64 * 50.0 + (i as f64 * 7.5);
            let intensity = 1000.0 * ((j % 5) as f64 + 1.0) * (5.0 - (i % 3) as f64) / 5.0;
            
            peaks.push(Peak::new(mz, intensity));
        }
        
        // Create a spectrum
        let content = format!("Sample spectrum {}", i+1);
        let mut spectrum = MassSpectrum::new(content.as_bytes(), format!("spectrum{}", i+1));
        
        // Add peaks to the spectrum
        for peak in &peaks {
            spectrum.add_peak(peak.clone());
        }
        
        // Add metadata
        let mut metadata = SpectrumMetadata {
            source: Some(format!("Instrument{}", i % 3 + 1)),
            ionization_method: Some("ESI".to_string()),
            resolution: Some(30000.0),
            annotations: HashMap::new(),
        };
        
        metadata.annotations.insert("retention_time".to_string(), format!("{:.2}", 5.0 + i as f64 * 3.0));
        metadata.annotations.insert("collision_energy".to_string(), format!("{}", 20 + i * 5));
        
        spectrum.set_metadata(metadata);
        
        spectra.push(spectrum);
    }
    
    spectra
}

/// Demonstrate high-throughput genomic operations
fn demonstrate_genomic_operations(sequences: &[NucleotideSequence]) {
    let ht_genomics = HighThroughputGenomics::new();
    
    // 1. Compress sequences for storage efficiency
    let compressor = SequenceCompressor::new();
    
    println!("   Compressing genomic sequences...");
    for (i, seq) in sequences.iter().enumerate().take(1) {
        let compressed = compressor.compress(seq);
        let original_size = seq.content().len();
        let compressed_size = compressed.data().len();
        
        println!("   Sequence {}: {} bp -> {} bytes ({:.1}% reduction)",
            i+1, original_size, compressed_size,
            100.0 * (1.0 - (compressed_size as f64 / original_size as f64))
        );
    }
    
    // 2. Find motifs in parallel
    println!("   Finding motifs in sequences...");
    
    // Create sample motifs to look for
    let motifs = vec![
        create_motif("ATGC", "motif1"),
        create_motif("CGATCG", "motif2"),
        create_motif("GCTAGCTA", "motif3"),
    ];
    
    for seq in sequences.iter().take(1) {
        let motif_results = ht_genomics.find_motifs_parallel(seq, &motifs, 0.5);
        
        println!("   Found motifs in sequence {}:", seq.id());
        for (motif, positions) in &motif_results {
            println!("     {} found at {} positions", motif.id(), positions.len());
        }
    }
    
    // 3. Count k-mers in parallel
    println!("   Counting k-mers in sequences...");
    for seq in sequences.iter().take(1) {
        let kmers = ht_genomics.count_kmers_parallel(seq, 3);
        
        println!("   Found {} unique 3-mers in sequence {}", kmers.len(), seq.id());
        
        // Print top 3 k-mers by frequency
        let mut kmer_vec: Vec<_> = kmers.iter().collect();
        kmer_vec.sort_by(|a, b| b.1.cmp(a.1));
        
        for (i, (kmer, count)) in kmer_vec.iter().take(3).enumerate() {
            println!("     #{}: {} (count: {})", 
                i+1, 
                String::from_utf8_lossy(kmer),
                count
            );
        }
    }
}

/// Demonstrate high-throughput mass spectrometry operations
fn demonstrate_spectrometry_operations(spectra: &[MassSpectrum]) {
    let ht_spectrometry = HighThroughputSpectrometry::new();
    
    // 1. Find peaks in parallel
    println!("   Finding peaks above threshold...");
    let peak_results = ht_spectrometry.find_peaks_parallel(spectra, 500.0, Some(3.0));
    
    for (i, peaks) in peak_results.iter().enumerate().take(3) {
        println!("   Spectrum {}: {} peaks above threshold", i+1, peaks.len());
    }
    
    // 2. Extract chromatograms
    println!("   Extracting chromatograms...");
    let mz_values = vec![150.0, 250.0, 350.0];
    let chromatograms = ht_spectrometry.extract_chromatograms_parallel(
        spectra, 
        &mz_values, 
        0.5
    );
    
    for &mz in &mz_values {
        if let Some(chrom) = chromatograms.get(&mz) {
            println!("   Chromatogram for m/z {:.1}: {} data points", mz, chrom.len());
        }
    }
    
    // 3. Spectrum alignment
    if !spectra.is_empty() {
        println!("   Aligning spectra to reference...");
        let reference = &spectra[0];
        let aligned = ht_spectrometry.align_spectra_parallel(
            &spectra[1..], 
            reference, 
            0.1
        );
        
        for (i, result) in aligned.iter().enumerate() {
            println!("   Aligned spectrum {}: similarity score = {:.2}", 
                i+1, 
                result.similarity_score
            );
        }
    }
}

/// Helper to create a motif unit for testing
fn create_motif(sequence: &str, id: &str) -> MotifUnit {
    let nucleotide_seq = NucleotideSequence::new(sequence.as_bytes(), id);
    MotifUnit::new(nucleotide_seq)
} 