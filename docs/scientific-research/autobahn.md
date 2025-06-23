---
layout: default
title: "Turbulance Language Reference"
description: "Complete reference documentation for the Turbulance programming language syntax and semantics"
---

# Turbulance Language Reference

## Overview

The Turbulance programming language is designed around the principle of **Scientific Method Encoding**—a paradigm where scientific reasoning becomes a first-class programming construct. This reference provides comprehensive documentation of language syntax, semantics, and usage patterns.

---

## Table of Contents

1. [Basic Syntax](#basic-syntax)
2. [Data Types](#data-types)
3. [Variables and Constants](#variables-and-constants)
4. [Functions](#functions)
5. [Scientific Constructs](#scientific-constructs)
6. [Control Flow](#control-flow)
7. [Pattern Matching](#pattern-matching)
8. [Biological Operations](#biological-operations)
9. [Metacognitive Features](#metacognitive-features)
10. [Standard Library](#standard-library)

---

## Basic Syntax

### Comments
```turbulance
# Single line comment
# Multi-line comments use multiple # symbols
```

### Statements
Statements in Turbulance end with newlines and use indentation for scope:

```turbulance
statement_one
statement_two
    indented_block
    another_indented_statement
```

---

## Data Types

### Primitive Types

#### Numeric Types
```turbulance
item integer_value = 42
item float_value = 3.14159
item scientific_notation = 6.022e23
```

#### Boolean Type
```turbulance
item is_active = true
item is_complete = false
```

#### String Type
```turbulance
item message = "Hello, biological computing!"
item multiline = """
    This is a multiline string
    that preserves formatting
"""
```

#### None Type
```turbulance
item empty_value = none
```

### Collection Types

#### Arrays
```turbulance
item numbers = [1, 2, 3, 4, 5]
item mixed_array = [1, "text", true, 3.14]
item nested_arrays = [[1, 2], [3, 4], [5, 6]]
```

#### Dictionaries
```turbulance
item properties = {
    "energy_level": 2.5,
    "stability": 0.87,
    "active": true
}
```

### Scientific Types

#### Evidence
```turbulance
item evidence_piece = Evidence {
    id: "exp_001",
    source: "biological_sensor",
    value: 0.85,
    confidence: 0.92,
    timestamp: current_time()
}
```

#### Pattern
```turbulance
item efficiency_pattern = pattern("high_energy", oscillatory)
item stability_pattern = pattern("steady_state", temporal)
```

---

## Variables and Constants

### Variable Declaration
```turbulance
item variable_name = initial_value
item energy_level = 0.0
item molecule_count = harvest_energy("atp_synthesis")
```

### Multiple Assignment
```turbulance
item x, y, z = [1, 2, 3]
item energy, efficiency = process_molecule("glucose")
```

### Variable Scope
Variables follow lexical scoping rules:

```turbulance
item global_var = "accessible everywhere"

funxn example_function():
    item local_var = "only accessible in function"
    
    within "scope_block":
        item block_var = "only accessible in this block"
        # Can access global_var and local_var here
    
    # block_var not accessible here
```

---

## Functions

### Function Declaration
```turbulance
funxn function_name(parameter1, parameter2):
    # function body
    return result
```

### Examples

#### Simple Function
```turbulance
funxn add_numbers(a, b):
    return a + b

item result = add_numbers(5, 10)
```

#### Biological Processing Function
```turbulance
funxn optimize_metabolism(substrate, target_efficiency):
    item current_efficiency = process_molecule(substrate)
    item energy_yield = harvest_energy("glycolysis")
    
    given current_efficiency < target_efficiency:
        item adjustment = target_efficiency - current_efficiency
        adjust_metabolic_rate(adjustment)
    
    return [current_efficiency, energy_yield]
```

#### Function with Default Parameters
```turbulance
funxn create_demon(demon_type="metabolic", threshold=2.5):
    item demon = BiologicalMaxwellDemon {
        type: demon_type,
        energy_threshold: threshold,
        state: "inactive"
    }
    return demon
```

---

## Scientific Constructs

### Propositions and Motions

#### Basic Proposition
```turbulance
proposition EnergyEfficiency:
    motion HighConversion("System achieves >90% energy conversion")
    motion StableOperation("Maintains consistent performance")
    motion ThermodynamicCompliance("Respects thermodynamic laws")
```

#### Proposition with Evidence Requirements
```turbulance
proposition MetabolicOptimization:
    motion ATPMaximization("Maximize ATP synthesis rate")
    motion WasteMinimization("Minimize metabolic waste products")
    
    # Evidence requirements
    requires_evidence from ["biosensor_array", "metabolic_analyzer"]
    
    # Support conditions
    given atp_rate > 0.9:
        support ATPMaximization with_weight(0.95)
    
    given waste_level < 0.1:
        support WasteMinimization with_weight(0.8)
```

### Evidence Collection

#### Basic Evidence Collector
```turbulance
evidence ExperimentalData from "sensor_network":
    collect energy_measurements
    collect efficiency_metrics
    validate data_quality
```

#### Advanced Evidence Collection
```turbulance
evidence ComprehensiveAnalysis from "multi_sensor_array":
    collect_batch:
        - temperature_readings
        - pressure_measurements  
        - chemical_concentrations
        - quantum_coherence_data
    
    validation_rules:
        - thermodynamic_consistency
        - measurement_uncertainty < 0.05
        - temporal_coherence > 0.9
    
    processing_pipeline:
        1. raw_data_filtering
        2. noise_reduction
        3. statistical_analysis
        4. confidence_calculation
```

### Goal Systems

#### Simple Goal
```turbulance
goal OptimizePerformance:
    description: "Achieve optimal system performance"
    success_threshold: 0.95
    metrics:
        efficiency: 0.0
        stability: 0.0
        throughput: 0.0
```

#### Complex Goal with Subgoals
```turbulance
goal SystemOptimization:
    description: "Complete system optimization with multiple objectives"
    success_threshold: 0.9
    
    subgoals:
        EnergyEfficiency:
            weight: 0.4
            threshold: 0.95
        
        ProcessingSpeed:
            weight: 0.3  
            threshold: 0.85
        
        Reliability:
            weight: 0.3
            threshold: 0.98
    
    constraints:
        - energy_consumption < max_energy_budget
        - temperature < critical_temperature
        - error_rate < 0.01
```

---

## Control Flow

### Conditional Statements

#### Basic Conditional
```turbulance
given condition:
    execute_if_true()
otherwise:
    execute_if_false()
```

#### Multiple Conditions
```turbulance
given energy_level > 0.8:
    operate_at_full_capacity()
given energy_level > 0.5:
    operate_at_reduced_capacity()
otherwise:
    enter_conservation_mode()
```

#### Complex Conditions
```turbulance
given (temperature < 310.0) and (pressure > 1.0) and not system_overload:
    continue_normal_operation()
otherwise:
    trigger_safety_protocols()
```

### Loops

#### While Loop
```turbulance
item counter = 0
while counter < 10:
    process_iteration(counter)
    counter = counter + 1
```

#### For Loop
```turbulance
for element in collection:
    process_element(element)

for i in range(10):
    perform_optimization_step(i)
```

#### Scientific Iteration Loop
```turbulance
optimize_until goal_achieved:
    item current_performance = measure_system_performance()
    item adjustment = calculate_optimization_step()
    apply_adjustment(adjustment)
    
    # Loop continues until goal is achieved
    check_goal_progress("SystemOptimization")
```

---

## Pattern Matching

### Basic Pattern Matching
```turbulance
item data = collect_sensor_data()
item efficiency_pattern = pattern("high_efficiency", oscillatory)

given data matches efficiency_pattern:
    apply_efficiency_optimization()
otherwise:
    investigate_anomaly()
```

### Advanced Pattern Matching
```turbulance
within "pattern_analysis":
    item patterns = {
        "efficiency": pattern("optimal_performance", temporal),
        "stability": pattern("steady_state", spatial),
        "anomaly": pattern("irregular_behavior", emergent)
    }
    
    for pattern_name, pattern_def in patterns.items():
        item match_result = sensor_data matches pattern_def
        given match_result:
            record_pattern_match(pattern_name, match_result.confidence)
```

### Pattern Types
```turbulance
# Temporal patterns - time-based sequences
item temporal_pattern = pattern("growth_cycle", temporal)

# Spatial patterns - geometric or structural
item spatial_pattern = pattern("molecular_arrangement", spatial)

# Oscillatory patterns - periodic behavior
item oscillatory_pattern = pattern("metabolic_rhythm", oscillatory)

# Emergent patterns - complex system behavior
item emergent_pattern = pattern("collective_behavior", emergent)
```

---

## Biological Operations

### Molecular Processing
```turbulance
# Basic molecule processing
item energy_yield = process_molecule("glucose")
item products = process_molecule("substrate", enzyme="catalase")

# Advanced molecular processing with parameters
item processing_result = process_molecule("complex_substrate") {
    temperature: 310.0,
    ph_level: 7.4,
    concentration: 0.1,
    catalyst: "biological_enzyme_x"
}
```

### Energy Harvesting
```turbulance
# Energy harvesting from various sources
item atp_energy = harvest_energy("atp_synthesis")
item glycolysis_energy = harvest_energy("glycolysis_pathway")
item photosynthetic_energy = harvest_energy("light_harvesting_complex")

# Energy harvesting with efficiency monitoring
item energy_data = harvest_energy("krebs_cycle") {
    monitor_efficiency: true,
    target_yield: 0.9,
    adaptive_optimization: true
}
```

### Information Extraction
```turbulance
# Extract information from biological processes
item metabolic_info = extract_information("metabolic_state")
item genetic_info = extract_information("gene_expression")
item structural_info = extract_information("protein_conformation")

# Information extraction with processing
item processed_info = extract_information("cellular_state") {
    processing_method: "shannon_entropy",
    noise_filtering: true,
    confidence_threshold: 0.8
}
```

### Membrane Operations
```turbulance
# Basic membrane state updates
update_membrane_state("high_permeability")
update_membrane_state("selective_transport")

# Advanced membrane control
configure_membrane {
    permeability: 0.7,
    selectivity: {
        "Na+": 0.9,
        "K+": 0.8,
        "Cl-": 0.6
    },
    transport_rate: 2.5,
    energy_requirement: 1.2
}
```

---

## Metacognitive Features

### Reasoning Monitoring
```turbulance
metacognitive ReasoningTracker:
    track_reasoning("optimization_process")
    track_reasoning("pattern_recognition")
    track_reasoning("decision_making")
    
    # Confidence evaluation
    item current_confidence = evaluate_confidence()
    
    # Bias detection
    item bias_detected = detect_bias("confirmation_bias")
    item availability_bias = detect_bias("availability_heuristic")
```

### Adaptive Behavior
```turbulance
metacognitive AdaptiveLearning:
    # Monitor system performance
    item performance_metrics = monitor_performance()
    
    # Adapt based on performance
    given performance_metrics.accuracy < 0.8:
        adapt_behavior("increase_evidence_collection")
    
    given performance_metrics.efficiency < 0.7:
        adapt_behavior("optimize_processing_pipeline")
    
    # Learn from past decisions
    analyze_decision_history()
    update_decision_strategies()
```

### Confidence Management
```turbulance
metacognitive ConfidenceManager:
    # Track confidence over time
    confidence_history = []
    
    funxn update_confidence():
        item current_confidence = evaluate_confidence()
        confidence_history.append(current_confidence)
        
        # Adaptive confidence thresholds
        given current_confidence < 0.6:
            increase_evidence_requirements()
        
        given current_confidence > 0.95:
            reduce_computational_overhead()
```

---

## Standard Library

### Mathematical Functions
```turbulance
# Basic math
item result = abs(-5)      # Absolute value
item power = pow(2, 8)     # Power function
item sqrt_val = sqrt(16)   # Square root
item log_val = log(10)     # Natural logarithm

# Statistical functions
item mean_val = mean([1, 2, 3, 4, 5])
item std_dev = stdev(data_array)
item correlation = corr(data_x, data_y)
```

### Scientific Functions
```turbulance
# Thermodynamic calculations
item entropy_change = calculate_entropy_change(initial_state, final_state)
item free_energy = gibbs_free_energy(enthalpy, entropy, temperature)

# Information theory
item shannon_entropy = shannon(probability_distribution)
item mutual_information = mutual_info(signal_x, signal_y)
item information_gain = info_gain(dataset, attribute)
```

### Biological Utility Functions
```turbulance
# Molecular calculations
item molecular_weight = calculate_mw("C6H12O6")  # Glucose
item binding_affinity = calculate_ka(concentration, bound_fraction)

# Metabolic pathway analysis
item pathway_flux = analyze_flux("glycolysis", metabolite_concentrations)
item enzyme_efficiency = calculate_kcat_km(enzyme_data)
```

### String and Data Processing
```turbulance
# String operations
item formatted = format("Energy level: {:.2f} kJ/mol", energy_value)
item sequence = reverse("ATCGGCAT")
item length = len(data_array)

# Data structure operations
item sorted_data = sort(measurements, key="timestamp")
item filtered_data = filter(data, lambda x: x.confidence > 0.8)
item mapped_data = map(data, lambda x: x * conversion_factor)
```

---

## Advanced Features

### Quantum Operations
```turbulance
quantum_state qubit_system:
    amplitude: 1.0
    phase: 0.0
    coherence_time: 1000.0

# Quantum gate operations
apply_hadamard(qubit_system)
apply_cnot(control_qubit, target_qubit)

# Quantum measurement
item measurement_result = measure(qubit_system)
item entanglement_degree = measure_entanglement(qubit_pair)
```

### Parallel Processing
```turbulance
# Parallel execution
parallel_execute:
    task_1: process_molecule_batch(batch_1)
    task_2: process_molecule_batch(batch_2)
    task_3: analyze_patterns(sensor_data)

# Wait for all tasks to complete
item results = await_all_tasks()
```

### Error Handling
```turbulance
try:
    item result = risky_biological_operation()
catch BiologicalError as e:
    handle_biological_failure(e)
    item result = fallback_operation()
catch QuantumDecoherenceError:
    restore_quantum_coherence()
    retry_operation()
finally:
    cleanup_resources()
```

---

## Language Conventions

### Naming Conventions
- Variables: `snake_case` (e.g., `energy_level`, `processing_rate`)
- Functions: `snake_case` (e.g., `process_molecule`, `harvest_energy`)
- Propositions: `PascalCase` (e.g., `EnergyEfficiency`, `SystemOptimization`)
- Constants: `UPPER_CASE` (e.g., `MAX_ENERGY`, `BOLTZMANN_CONSTANT`)

### Code Organization
```turbulance
# Import statements at the top
import biological_utils
import quantum_operations

# Constants
item TEMPERATURE_THRESHOLD = 310.0
item MAX_ITERATIONS = 1000

# Function definitions
funxn main():
    # Main program logic
    pass

# Execute main function
main()
```

### Best Practices

1. **Use descriptive variable names** that reflect biological or scientific meaning
2. **Group related operations** within `within` blocks for clarity
3. **Always validate evidence** before making scientific conclusions
4. **Monitor confidence levels** and adapt behavior accordingly
5. **Document complex propositions** with clear motion descriptions
6. **Use metacognitive features** to ensure robust reasoning

---

This completes the comprehensive Turbulance Language Reference. For implementation examples and advanced usage patterns, see the [Examples](/examples) section.

---

**© 2024 Autobahn Biological Computing Project. All rights reserved.** 