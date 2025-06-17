#!/usr/bin/env python3
"""
Lavoisier Mass Spectrometry Analysis Module
Used by Kwasa-Kwasa framework for actual computational work
"""

import lavoisier
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

def load_spectrum_data(file_path, format='mzML'):
    """Load mass spectrum data using Lavoisier"""
    try:
        spectrum = lavoisier.io.load_spectrum(file_path, format=format)
        return {
            'status': 'success',
            'spectrum': spectrum,
            'num_peaks': len(spectrum.peaks),
            'mz_range': (spectrum.mz_min, spectrum.mz_max),
            'intensity_range': (spectrum.intensity_min, spectrum.intensity_max)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def preprocess_spectrum(spectrum, noise_threshold=1000, normalize=True):
    """Preprocess spectrum using Lavoisier's signal processing"""
    # Noise filtering
    filtered = lavoisier.signal.adaptive_filter(
        spectrum, 
        threshold=noise_threshold,
        method='gaussian_filter'
    )
    
    # Peak detection
    peaks = lavoisier.peak_detection.centwave_algorithm(
        filtered,
        ppm_tolerance=5.0,
        signal_noise_ratio=3.0
    )
    
    # Normalization
    if normalize:
        normalized = lavoisier.preprocessing.total_ion_current_normalization(peaks)
    else:
        normalized = peaks
    
    return {
        'preprocessed_spectrum': normalized,
        'num_peaks_detected': len(peaks),
        'preprocessing_metadata': {
            'noise_threshold': noise_threshold,
            'normalized': normalize,
            'filter_method': 'adaptive_gaussian'
        }
    }

def identify_compounds(spectrum, databases=['HMDB', 'KEGG'], mass_tolerance=0.01):
    """Identify compounds using Lavoisier's database search"""
    identifications = []
    
    for db in databases:
        db_results = lavoisier.identification.database_search(
            spectrum.peaks,
            database=db,
            mass_tolerance=mass_tolerance,
            ionization_mode='positive'
        )
        
        for result in db_results:
            identifications.append({
                'compound_id': result.compound_id,
                'compound_name': result.name,
                'formula': result.molecular_formula,
                'mass': result.exact_mass,
                'score': result.confidence_score,
                'database': db,
                'matched_mz': result.matched_mz
            })
    
    # Sort by confidence score
    identifications.sort(key=lambda x: x['score'], reverse=True)
    
    return identifications

def perform_statistical_analysis(sample_data, control_data):
    """Statistical comparison using Lavoisier's stats module"""
    stats_results = lavoisier.statistics.compare_groups(
        sample_data, 
        control_data,
        tests=['t_test', 'mann_whitney', 'fold_change'],
        correction='fdr'
    )
    
    significant_features = lavoisier.statistics.extract_significant_features(
        stats_results,
        p_value_threshold=0.05,
        fold_change_threshold=1.5
    )
    
    return {
        'statistical_results': stats_results,
        'significant_features': significant_features,
        'num_significant': len(significant_features)
    }

def pathway_analysis(compound_ids, organism='human'):
    """Pathway enrichment using Lavoisier's pathway module"""
    pathways = lavoisier.pathway.enrichment_analysis(
        compound_ids,
        organism=organism,
        databases=['KEGG', 'BioCyc', 'Reactome'],
        correction_method='bonferroni'
    )
    
    enriched_pathways = [p for p in pathways if p.p_value_corrected < 0.05]
    
    return {
        'enriched_pathways': enriched_pathways,
        'num_enriched': len(enriched_pathways),
        'pathway_metadata': {
            'organism': organism,
            'databases_used': ['KEGG', 'BioCyc', 'Reactome']
        }
    }

def generate_quality_metrics(spectrum, preprocessing_results):
    """Generate quality control metrics"""
    metrics = {
        'signal_to_noise': lavoisier.qc.calculate_signal_noise_ratio(spectrum),
        'peak_quality': lavoisier.qc.assess_peak_quality(preprocessing_results['preprocessed_spectrum']),
        'data_completeness': lavoisier.qc.calculate_completeness(spectrum),
        'instrument_stability': lavoisier.qc.assess_stability(spectrum),
        'overall_quality_score': 0.0
    }
    
    # Calculate overall quality score
    weights = {'signal_to_noise': 0.3, 'peak_quality': 0.3, 'data_completeness': 0.2, 'instrument_stability': 0.2}
    metrics['overall_quality_score'] = sum(metrics[k] * weights[k] for k in weights.keys())
    
    return metrics

def diabetes_biomarker_analysis(spectrum_files, clinical_metadata):
    """
    Main analysis function for diabetes biomarker discovery
    Called by Kwasa-Kwasa orchestrator
    """
    results = {
        'analysis_metadata': {
            'num_samples': len(spectrum_files),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'lavoisier_version': lavoisier.__version__
        },
        'sample_results': [],
        'group_comparisons': {},
        'biomarker_candidates': [],
        'pathway_results': {},
        'quality_report': {}
    }
    
    # Process each spectrum file
    for i, file_path in enumerate(spectrum_files):
        print(f"Processing sample {i+1}/{len(spectrum_files)}: {file_path}")
        
        # Load and preprocess
        load_result = load_spectrum_data(file_path)
        if load_result['status'] == 'error':
            continue
            
        preprocess_result = preprocess_spectrum(load_result['spectrum'])
        
        # Compound identification
        identifications = identify_compounds(preprocess_result['preprocessed_spectrum'])
        
        # Quality metrics
        quality = generate_quality_metrics(load_result['spectrum'], preprocess_result)
        
        results['sample_results'].append({
            'file_path': str(file_path),
            'sample_id': clinical_metadata[i].get('sample_id', f'sample_{i}'),
            'group': clinical_metadata[i].get('group', 'unknown'),
            'identifications': identifications[:50],  # Top 50 matches
            'quality_metrics': quality,
            'preprocessing_summary': preprocess_result['preprocessing_metadata']
        })
    
    # Group comparison (diabetes vs control)
    diabetes_samples = [r for r in results['sample_results'] if r['group'] == 'diabetes']
    control_samples = [r for r in results['sample_results'] if r['group'] == 'control']
    
    if diabetes_samples and control_samples:
        # Aggregate data for statistical comparison
        diabetes_data = aggregate_sample_data(diabetes_samples)
        control_data = aggregate_sample_data(control_samples)
        
        # Statistical analysis
        stats_result = perform_statistical_analysis(diabetes_data, control_data)
        results['group_comparisons'] = stats_result
        
        # Extract biomarker candidates
        biomarkers = extract_biomarker_candidates(stats_result['significant_features'])
        results['biomarker_candidates'] = biomarkers
        
        # Pathway analysis on significant compounds
        significant_compounds = [f['compound_id'] for f in stats_result['significant_features']]
        pathway_result = pathway_analysis(significant_compounds)
        results['pathway_results'] = pathway_result
    
    # Overall quality report
    overall_quality = calculate_overall_quality(results['sample_results'])
    results['quality_report'] = overall_quality
    
    return results

def aggregate_sample_data(sample_results):
    """Aggregate identification data for statistical analysis"""
    # Create intensity matrix for all identified compounds
    all_compounds = set()
    for sample in sample_results:
        for identification in sample['identifications']:
            all_compounds.add(identification['compound_id'])
    
    intensity_matrix = []
    for sample in sample_results:
        sample_intensities = {}
        for identification in sample['identifications']:
            sample_intensities[identification['compound_id']] = identification['score']
        
        # Fill missing compounds with zero
        row = [sample_intensities.get(compound, 0.0) for compound in sorted(all_compounds)]
        intensity_matrix.append(row)
    
    return np.array(intensity_matrix)

def extract_biomarker_candidates(significant_features):
    """Extract top biomarker candidates from statistical results"""
    candidates = []
    
    for feature in significant_features[:20]:  # Top 20 candidates
        candidates.append({
            'compound_id': feature.compound_id,
            'compound_name': feature.compound_name,
            'p_value': feature.p_value,
            'fold_change': feature.fold_change,
            'effect_size': feature.effect_size,
            'confidence_score': feature.confidence_score,
            'biomarker_potential': assess_biomarker_potential(feature)
        })
    
    return candidates

def assess_biomarker_potential(feature):
    """Assess biomarker potential based on multiple criteria"""
    score = 0.0
    
    # P-value contribution
    if feature.p_value < 0.001:
        score += 0.3
    elif feature.p_value < 0.01:
        score += 0.2
    elif feature.p_value < 0.05:
        score += 0.1
    
    # Fold change contribution
    if abs(feature.fold_change) > 3.0:
        score += 0.3
    elif abs(feature.fold_change) > 2.0:
        score += 0.2
    elif abs(feature.fold_change) > 1.5:
        score += 0.1
    
    # Effect size contribution
    if feature.effect_size > 0.8:
        score += 0.2
    elif feature.effect_size > 0.5:
        score += 0.15
    elif feature.effect_size > 0.3:
        score += 0.1
    
    # Confidence contribution
    score += feature.confidence_score * 0.2
    
    return min(1.0, score)

def calculate_overall_quality(sample_results):
    """Calculate overall quality metrics for the entire analysis"""
    quality_scores = [r['quality_metrics']['overall_quality_score'] for r in sample_results]
    
    return {
        'mean_quality': np.mean(quality_scores),
        'min_quality': np.min(quality_scores),
        'max_quality': np.max(quality_scores),
        'std_quality': np.std(quality_scores),
        'samples_passing_qc': sum(1 for q in quality_scores if q > 0.7),
        'total_samples': len(quality_scores)
    }

if __name__ == "__main__":
    # Command line interface for Kwasa-Kwasa to call
    if len(sys.argv) < 3:
        print("Usage: python lavoisier_analysis.py <spectrum_files_json> <clinical_metadata_json>")
        sys.exit(1)
    
    # Load input files
    with open(sys.argv[1], 'r') as f:
        spectrum_files = json.load(f)
    
    with open(sys.argv[2], 'r') as f:
        clinical_metadata = json.load(f)
    
    # Run analysis
    results = diabetes_biomarker_analysis(spectrum_files, clinical_metadata)
    
    # Output results as JSON for Kwasa-Kwasa to process
    print(json.dumps(results, indent=2, default=str)) 