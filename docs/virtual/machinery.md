# Machinery

<div align="center">
  <img src="assets/machinery.png" alt="Machinery Logo" width="200"/>
  
  _Comprehensive Health Intelligence Ecosystem Through Multi-System Integration_

  ![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
  ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
  ![Status: Development](https://img.shields.io/badge/Status-Development-orange.svg?style=for-the-badge)
</div>

## Overview

**Machinery** is a comprehensive health intelligence ecosystem that combines multiple specialized systems to create robust, personalized health models through continuous prediction, adversarial testing, and stochastic optimization. The framework integrates four complementary systems:

- **🦀 Machinery (Rust)**: Core temporal dynamics and system orchestration
- **🐍 mzekezeke (Python)**: ML predictions and scientific health analysis  
- **⚔️ diggiden (Python)**: Adversarial challenges and robustness testing
- **🎲 hatata (Python)**: Stochastic decision processes and utility optimization

The core principle is simple yet powerful: **given some parts of an individual's biological system, predict the other parts while continuously challenging and optimizing those predictions under uncertainty**. This multi-layered approach ensures robust, evidence-based health intelligence that adapts to real-world complexity.

Unlike traditional health monitoring systems that rely on population averages or static thresholds, the Machinery ecosystem recognizes that biological measurements are inherently contextual and requires multi-layered validation. An 81 bpm heart rate means vastly different things depending on the individual's baseline, current activity, stress levels, sleep quality, metabolic state, and countless other factors. The integrated systems work together to:

- **Predict** health states using scientific ML models (mzekezeke)
- **Challenge** those predictions through adversarial testing (diggiden) 
- **Optimize** decisions under uncertainty using stochastic methods (hatata)
- **Orchestrate** the entire process through temporal dynamics (Machinery core)

## System Components

### 🦀 Machinery Core (Rust)
The central orchestration engine that provides:

- **Temporal Dynamics**: Manages health state evolution over time using Seke script engine
- **System Integration**: Coordinates between mzekezeke, diggiden, and hatata components
- **Contextual Modeling**: Understands that health measurements depend on individual patterns, time, and situation
- **Fuzzy Logic Processing**: Handles uncertainty and gradual transitions in health states
- **Real-time Orchestration**: Low-latency coordination of multi-system predictions and decisions

### 🐍 mzekezeke (Python)
*"The process engine"* - Scientific ML prediction and analysis system:

- **Health Metrics Analysis**: Comprehensive analysis of cardiovascular, metabolic, and other health systems
- **ML Predictions**: HuggingFace-powered models for health state forecasting
- **Scientific Validation**: Evidence-based health metric interpretation using established research
- **Heart Rate Variability**: Advanced HRV analysis with time-domain and frequency-domain metrics
- **Risk Assessment**: Personalized health risk evaluation with confidence scoring
- **Biomarker Integration**: Multi-modal health data processing and pattern recognition

### ⚔️ diggiden (Python)
*"The health antagonist"* - Adversarial system for robustness testing:

- **Health Balance Philosophy**: Models health as a complex mixture where systems operate at different efficiency levels
- **Adversarial Challenges**: Continuously tests system predictions against realistic health deterioration scenarios
- **System Vulnerability Discovery**: Identifies weak points in health optimization strategies
- **Multi-System Failure Simulation**: Tests robustness when multiple health systems are compromised
- **Challenge Evolution**: Learns and adapts adversarial strategies based on system responses
- **Entropy Modeling**: Simulates natural health degradation processes and external stressors

### 🎲 hatata (Python)
*"Step by step"* - Stochastic decision optimization system:

- **Markov Decision Processes**: Models health states and transitions with uncertainty quantification
- **Multi-Objective Utility Optimization**: Balances competing health goals using Pareto optimization
- **Monte Carlo Analysis**: Quantifies uncertainty in health projections using stochastic simulation
- **Evidence Integration**: Synthesizes predictions from mzekezeke and challenges from diggiden
- **Decision Confidence Scoring**: Provides uncertainty bounds and confidence intervals for recommendations
- **Policy Optimization**: Finds optimal health intervention strategies under uncertainty

## System Integration Philosophy

The four systems work in continuous interplay:

1. **mzekezeke** generates scientific predictions about health states
2. **diggiden** challenges those predictions with adversarial scenarios  
3. **hatata** optimizes decisions under the resulting uncertainty
4. **Machinery** orchestrates the entire process with temporal awareness

This creates a robust health intelligence system that doesn't just predict, but validates predictions against adversarial conditions and optimizes decisions for real-world uncertainty.

## Core Philosophy

### The Contextual Nature of Health Data

Traditional health systems treat measurements as absolute values:
- "Normal" heart rate: 60-100 bpm
- "Healthy" blood glucose: 70-99 mg/dL  
- "Optimal" sleep: 7-9 hours

**Machinery recognizes that health is fundamentally contextual:**
- 81 bpm at rest after 8 hours of sleep vs. 81 bpm after climbing stairs
- 85 mg/dL glucose fasting vs. 85 mg/dL two hours post-meal
- 6 hours of deep, restorative sleep vs. 8 hours of fragmented sleep

### Iterative System Prediction

Machinery operates on the principle that biological systems are predictable networks of relationships. By continuously observing patterns and testing predictions, the system builds increasingly accurate models of how each individual's biology functions:

1. **Observe**: Collect continuous data streams from multiple biological systems
2. **Predict**: Use current system state to predict other system components
3. **Validate**: Compare predictions against actual measurements
4. **Learn**: Refine models based on prediction accuracy
5. **Iterate**: Continuously improve understanding of the individual system

## Architecture

### Multi-System Framework

```
Machinery Ecosystem
├── 🦀 Machinery Core (Rust)
│   ├── Health AI Orchestrator          # Central coordination engine
│   ├── Seke Script Engine             # Hybrid logical/fuzzy programming runtime
│   ├── Continuous Data Collectors     # Multi-modal health data ingestion
│   ├── Temporal State Management      # Health state evolution over time
│   ├── Context-Aware Modeling         # Contextual interpretation of measurements
│   └── System Integration Layer       # Coordinates Python subsystems
│
├── 🐍 mzekezeke (Python)
│   ├── Health Data Processor          # Scientific health metrics analysis
│   ├── ML Prediction Pipeline         # HuggingFace model integration
│   ├── Heart Rate Analyzer           # Advanced HRV and cardiac analysis
│   ├── Biomarker Validation          # Evidence-based health interpretation
│   └── Risk Assessment Engine         # Personalized health risk evaluation
│
├── ⚔️ diggiden (Python)
│   ├── Adversarial Engine            # Health challenge generation
│   ├── Health Antagonist             # Complex balance testing
│   ├── System Vulnerability Scanner   # Weakness identification
│   ├── Multi-System Failure Simulator # Robustness testing
│   └── Challenge Evolution AI         # Adaptive adversarial strategies
│
└── 🎲 hatata (Python)
    ├── MDP Engine                    # Markov Decision Process modeling
    ├── Utility Optimizer             # Multi-objective goal optimization
    ├── Stochastic Processor          # Monte Carlo uncertainty analysis
    ├── Evidence Integrator           # Multi-source evidence synthesis
    └── Decision Policy Engine        # Optimal intervention strategies
```

### System Communication Flow

```
[Health Data] → Machinery Core → mzekezeke (Analysis) 
                     ↓              ↓
              hatata (Decisions) ← diggiden (Challenges)
                     ↓
              [Optimized Actions]
```

### Seke Scripts: Hybrid Logical-Fuzzy Programming

The heart of Machinery's flexibility lies in **Seke scripts** - a domain-specific language that combines logical programming with fuzzy logic systems. Seke scripts allow the system to handle the inherent uncertainty and context-dependency of biological measurements.

#### Example Seke Script: Heart Rate Context Analysis

```seke
// Heart rate interpretation with contextual fuzzy logic
rule heart_rate_analysis {
    input: hr_current, hr_baseline, activity_level, sleep_quality, stress_markers
    
    // Fuzzy sets for heart rate deviation
    fuzzy hr_deviation = (hr_current - hr_baseline) / hr_baseline
    
    context resting_state {
        if activity_level < 0.2 and time_since_activity > 10min {
            if hr_deviation in [-0.1, 0.1] -> "normal_resting"
            if hr_deviation in [0.1, 0.3] -> "elevated_resting" 
            if hr_deviation > 0.3 -> "concerning_elevation"
        }
    }
    
    context post_exercise {
        if time_since_activity < 30min {
            expected_recovery = calculate_recovery_curve(exercise_intensity, fitness_level)
            if hr_current within expected_recovery -> "normal_recovery"
            else -> "abnormal_recovery"
        }
    }
    
    // Logical rules with fuzzy confidence
    predict cardiovascular_stress {
        confidence = 0.8 if (hr_deviation > 0.2 and stress_markers.elevated)
        confidence = 0.6 if (hr_deviation > 0.15 and sleep_quality < 0.7)
        confidence = 0.9 if (hr_deviation > 0.3 and activity_level < 0.1)
    }
}
```

#### Example Seke Script: Metabolic State Prediction

```seke
// Predict glucose response based on multiple system inputs
rule glucose_prediction {
    input: current_glucose, meal_composition, exercise_history, sleep_debt, stress_level
    
    // Fuzzy modeling of insulin sensitivity
    fuzzy insulin_sensitivity {
        base_sensitivity = individual_baseline.insulin_sensitivity
        
        // Sleep impact on sensitivity
        sleep_modifier = if sleep_debt < 1hr -> 1.0
                        if sleep_debt in [1hr, 3hr] -> 0.85
                        if sleep_debt > 3hr -> 0.7
        
        // Exercise impact (time-dependent)
        exercise_modifier = calculate_exercise_sensitivity_boost(
            time_since_exercise, exercise_intensity, exercise_duration
        )
        
        current_sensitivity = base_sensitivity * sleep_modifier * exercise_modifier
    }
    
    // Predict glucose response to meal
    predict postprandial_glucose {
        carb_load = meal_composition.carbohydrates
        expected_peak = current_glucose + (carb_load / current_sensitivity)
        expected_time_to_peak = 45min + (stress_level * 15min)
        
        confidence = 0.9 if meal_composition.complete
        confidence = 0.7 if meal_composition.estimated
    }
    
    // Predict when glucose will return to baseline
    predict glucose_clearance {
        clearance_rate = current_sensitivity * 0.8
        time_to_baseline = (expected_peak - individual_baseline.glucose) / clearance_rate
    }
}
```

### Continuous Data Collection

Machinery integrates with multiple data sources to build comprehensive individual models:

#### Physiological Streams
- **Cardiovascular**: Heart rate, HRV, blood pressure, pulse wave velocity
- **Metabolic**: Continuous glucose monitoring, ketones, lactate
- **Respiratory**: Breathing rate, oxygen saturation, CO2 levels
- **Neurological**: EEG patterns, reaction times, cognitive performance
- **Hormonal**: Cortisol, melatonin, insulin, thyroid markers

#### Behavioral Streams  
- **Activity**: Movement patterns, exercise intensity, sedentary time
- **Sleep**: Sleep stages, duration, efficiency, timing
- **Nutrition**: Meal timing, macronutrient composition, hydration
- **Environmental**: Temperature, humidity, air quality, light exposure

#### Genomic and Metabolomic Integration
- **Genetic Variants**: SNPs affecting drug metabolism, nutrient processing, disease risk
- **Metabolomic Profiles**: Real-time metabolite concentrations, pathway activity
- **Epigenetic Markers**: Dynamic gene expression changes based on lifestyle

### Iterative Prediction Engine

The core of Machinery's learning system continuously tests and refines its understanding:

```rust
// Simplified representation of the prediction cycle
pub struct PredictionCycle {
    current_state: SystemState,
    prediction_models: Vec<PredictionModel>,
    validation_metrics: ValidationFramework,
}

impl PredictionCycle {
    pub async fn iterate(&mut self) -> PredictionResult {
        // 1. Generate predictions based on current system state
        let predictions = self.generate_predictions().await;
        
        // 2. Wait for actual measurements
        let actual_measurements = self.collect_measurements().await;
        
        // 3. Validate predictions against reality
        let validation_results = self.validate_predictions(
            &predictions, 
            &actual_measurements
        ).await;
        
        // 4. Update models based on accuracy
        self.update_models(&validation_results).await;
        
        // 5. Refine understanding of individual patterns
        self.refine_individual_patterns().await;
        
        validation_results
    }
}
```

### Context-Aware Modeling

Machinery's models understand that the same measurement can have different meanings in different contexts:

#### Temporal Context
- **Circadian Rhythms**: How measurements vary throughout the day
- **Seasonal Patterns**: Long-term cyclical changes
- **Menstrual Cycles**: Hormonal fluctuations affecting all systems
- **Training Cycles**: Adaptation and recovery patterns

#### Situational Context
- **Stress States**: How acute and chronic stress affect all measurements
- **Illness Recovery**: How the immune response changes normal patterns
- **Medication Effects**: How pharmaceuticals alter system relationships
- **Environmental Factors**: Temperature, altitude, air quality impacts

#### Individual Context
- **Genetic Background**: How genetic variants affect normal ranges
- **Fitness Level**: How conditioning changes physiological responses
- **Age and Development**: How patterns change over time
- **Personal History**: How past events influence current patterns

## Key Features

### 1. Personalized Normal Ranges

Instead of population-based "normal" values, Machinery establishes individual baselines:

```seke
// Individual baseline establishment
rule establish_baseline {
    input: measurement_history, context_data, validation_period
    
    // Calculate personal normal range with confidence intervals
    personal_baseline = calculate_individual_baseline(
        measurements = measurement_history.filter(context == "resting"),
        confidence_level = 0.95,
        minimum_samples = 100
    )
    
    // Account for natural variation
    normal_range = personal_baseline ± (2 * individual_standard_deviation)
    
    // Context-specific adjustments
    adjust_for_context(time_of_day, activity_level, stress_state)
}
```

### 2. Predictive Health Modeling

Machinery doesn't just monitor - it predicts:

- **Short-term predictions**: How will glucose respond to this meal?
- **Medium-term predictions**: How will sleep quality affect tomorrow's performance?
- **Long-term predictions**: How will this training program affect fitness markers?

### 3. Pattern Recognition and Anomaly Detection

The system learns individual patterns and identifies deviations:

```seke
// Anomaly detection with contextual awareness
rule detect_anomalies {
    input: current_measurements, historical_patterns, context
    
    // Calculate expected values based on context
    expected_values = predict_from_context(context, historical_patterns)
    
    // Fuzzy anomaly scoring
    fuzzy anomaly_score {
        deviation = abs(current_measurements - expected_values)
        normalized_deviation = deviation / historical_variance
        
        if normalized_deviation < 1.0 -> "normal"
        if normalized_deviation in [1.0, 2.0] -> "mild_anomaly"
        if normalized_deviation in [2.0, 3.0] -> "moderate_anomaly"
        if normalized_deviation > 3.0 -> "significant_anomaly"
    }
    
    // Context-aware confidence adjustment
    confidence = base_confidence * context_reliability * data_quality
}
```

### 4. Multi-System Integration

Machinery understands that biological systems are interconnected:

- **Cardiovascular-Metabolic**: How heart rate variability relates to glucose control
- **Sleep-Immune**: How sleep quality affects inflammatory markers
- **Stress-Digestive**: How cortisol levels impact gut health
- **Exercise-Recovery**: How training load affects all other systems

### 5. Continuous Learning and Adaptation

The system continuously improves its understanding:

- **Model Refinement**: Prediction accuracy improves over time
- **Pattern Discovery**: New relationships are automatically identified
- **Context Learning**: Understanding of situational factors deepens
- **Individual Adaptation**: Models become increasingly personalized

## Technical Implementation

### Multi-Language Architecture

The Machinery ecosystem leverages the strengths of different programming languages:

#### 🦀 Machinery Core (Rust)
Built for performance, safety, and real-time coordination:

```rust
// Core framework structure
pub mod machinery {
    pub mod orchestrator;      // Health AI coordination
    pub mod seke_engine;       // Script runtime
    pub mod data_collectors;   // Multi-modal data ingestion
    pub mod temporal_state;    // Time-aware state management
    pub mod modeling;          // Context-aware models
    pub mod integration;       // Python system coordination
    pub mod patterns;          // Individual pattern learning
}
```

#### 🐍 Python Subsystems
Scientific computing and ML capabilities:

```python
# mzekezeke - Scientific health analysis
from mzekezeke import HealthDataProcessor, PredictionPipeline, HeartRateAnalyzer
from mzekezeke.ml_models import HuggingFacePredictor

# diggiden - Adversarial testing
from diggiden import AdversarialEngine, HealthAntagonist
from diggiden.challenges import SystemChallenge, HealthBalance

# hatata - Stochastic optimization
from hatata import HatataCore, MDPEngine, UtilityOptimizer
from hatata.stochastic import StochasticProcessor, EvidenceIntegrator
```

### System Integration Patterns

```rust
// Rust-Python integration via FFI and messaging
pub struct SystemOrchestrator {
    mzekezeke_client: PythonMLClient,
    diggiden_client: AdversarialClient, 
    hatata_client: StochasticClient,
    coordination_bus: MessageBus,
}

impl SystemOrchestrator {
    pub async fn process_health_data(&mut self, data: HealthData) -> Decision {
        // 1. Scientific analysis
        let predictions = self.mzekezeke_client.analyze(data).await?;
        
        // 2. Adversarial testing
        let challenges = self.diggiden_client.challenge(predictions).await?;
        
        // 3. Stochastic optimization
        let decision = self.hatata_client.optimize(predictions, challenges).await?;
        
        decision
    }
}
```

### Seke Script Engine

The Seke script engine provides:

- **Fuzzy Logic Processing**: Handle uncertainty and gradual transitions
- **Logical Rule Evaluation**: Clear if-then reasoning
- **Context-Aware Execution**: Rules adapt based on situational factors
- **Confidence Propagation**: Uncertainty tracking through inference chains
- **Real-time Execution**: Low-latency script evaluation

### Data Pipeline

```rust
// Continuous data processing pipeline
pub struct DataPipeline {
    collectors: Vec<Box<dyn DataCollector>>,
    processors: Vec<Box<dyn DataProcessor>>,
    validators: Vec<Box<dyn DataValidator>>,
    storage: Box<dyn HealthDataStorage>,
}

impl DataPipeline {
    pub async fn process_continuous_stream(&mut self) {
        loop {
            // Collect from all sources
            let raw_data = self.collect_all_sources().await;
            
            // Process and contextualize
            let processed_data = self.process_data(raw_data).await;
            
            // Validate quality and consistency
            let validated_data = self.validate_data(processed_data).await;
            
            // Store with temporal and contextual metadata
            self.storage.store_with_context(validated_data).await;
            
            // Trigger prediction cycle
            self.trigger_prediction_cycle().await;
        }
    }
}
```

## Use Cases

### 1. Metabolic Health Optimization

The integrated systems work together for comprehensive metabolic analysis:

**mzekezeke** provides:
- Continuous glucose pattern analysis and ML-based prediction models
- Scientific validation of metabolic markers and biomarker interpretation
- Personalized glucose response forecasting based on meal composition

**diggiden** challenges with:
- Metabolic stress scenarios (missed meals, high-glycemic challenges)
- Multi-system failure simulation (sleep deprivation + high carb intake)
- Insulin resistance progression modeling

**hatata** optimizes by:
- Balancing competing goals (glucose control vs. dietary satisfaction)
- Stochastic meal timing optimization under uncertainty
- Multi-objective decision making for dietary interventions

**Machinery orchestrates** through:
- Temporal glucose pattern modeling with circadian awareness
- Real-time coordination of prediction, challenge, and optimization cycles

### 2. Cardiovascular Health Monitoring

Multi-system cardiovascular intelligence:

**mzekezeke** analyzes:
- Advanced HRV analysis with time-domain and frequency-domain metrics
- Heart rate zone optimization based on individual physiology
- Cardiovascular risk assessment with confidence scoring

**diggiden** tests resilience through:
- Cardiovascular stress simulation (sudden exercise, emotional stress)
- Recovery capacity challenges under various conditions
- Multi-system cardiovascular failure scenarios

**hatata** optimizes training by:
- Balancing performance goals with recovery needs using Markov Decision Processes
- Exercise intensity optimization under uncertainty
- Multi-objective training load management

**Machinery coordinates** with:
- Temporal HRV pattern recognition and circadian optimization
- Real-time exercise prescription based on integrated system feedback

### 3. Sleep and Circadian Optimization

By tracking sleep stages, timing, and quality:
- Predict optimal bedtimes for individual circadian rhythms
- Identify factors affecting sleep quality
- Optimize light exposure and meal timing
- Monitor the impact of sleep on other health markers

### 4. Stress and Mental Health

Through physiological stress markers and behavioral patterns:
- Identify early signs of chronic stress
- Predict stress responses to specific situations
- Optimize stress management interventions
- Monitor the effectiveness of mental health treatments

### 5. Athletic Performance and Recovery

For athletes and fitness enthusiasts:
- Predict optimal training loads
- Monitor recovery and adaptation
- Identify overtraining before it occurs
- Optimize nutrition and hydration strategies

## Getting Started

### Prerequisites

- **Rust 1.70+** with Cargo (for Machinery core)
- **Python 3.9+** with pip (for mzekezeke, diggiden, hatata)
- Compatible health monitoring devices
- Minimum 16GB RAM for multi-system processing
- GPU recommended for ML models (mzekezeke)
- Persistent storage for historical data

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/machinery.git
cd machinery

# Install Machinery core (Rust)
cargo build --release

# Install Python subsystems
cd mzekezeke && pip install -e . && cd ..
cd diggiden && pip install -e . && cd ..
cd hatata && pip install -e . && cd ..

# Run initial setup
cargo run --bin machinery-setup

# Start the integrated health ecosystem
cargo run --bin machinery-orchestrator
```

### Quick Start with Individual Systems

```bash
# Test mzekezeke (ML predictions)
cd mzekezeke/examples
python scientific_demo.py

# Test diggiden (adversarial challenges)
cd diggiden/examples  
python adversarial_demo.py

# Test hatata (stochastic optimization)
cd hatata/examples
python stochastic_demo.py

# Full integration test
cargo run --example full_system_demo
```

### Configuration

Machinery uses TOML configuration files for setup:

```toml
# machinery.toml
[orchestrator]
prediction_interval = "5min"
model_update_interval = "1hour"
data_retention_days = 365

[data_sources]
continuous_glucose = { enabled = true, device = "dexcom_g7" }
heart_rate = { enabled = true, device = "polar_h10" }
sleep_tracking = { enabled = true, device = "oura_ring" }

[seke_engine]
max_concurrent_scripts = 100
script_timeout = "30s"
confidence_threshold = 0.7

[modeling]
baseline_establishment_days = 30
prediction_horizon = "24hours"
anomaly_sensitivity = 0.8
```

### Writing Your First Seke Script

```seke
// my_first_health_rule.seke
rule morning_readiness {
    input: hrv_current, sleep_score, resting_hr
    
    // Calculate readiness based on multiple factors
    fuzzy readiness_score {
        hrv_factor = hrv_current / individual_baseline.hrv
        sleep_factor = sleep_score / 100.0
        hr_factor = individual_baseline.resting_hr / resting_hr
        
        readiness = (hrv_factor * 0.4) + (sleep_factor * 0.4) + (hr_factor * 0.2)
    }
    
    // Provide contextual recommendations
    recommend training_intensity {
        if readiness > 0.8 -> "high_intensity_ok"
        if readiness in [0.6, 0.8] -> "moderate_intensity"
        if readiness < 0.6 -> "recovery_day"
    }
}
```

## Contributing

Machinery is designed to be extensible and welcomes contributions:

- **Data Collectors**: Add support for new health monitoring devices
- **Seke Scripts**: Contribute domain-specific health rules
- **Prediction Models**: Implement new modeling approaches
- **Validation Methods**: Improve accuracy assessment techniques

## License

MIT License - see LICENSE file for details.

## Research and Citations

Machinery is built on decades of research in:
- Systems biology and network medicine
- Fuzzy logic and uncertainty reasoning
- Personalized medicine and precision health
- Continuous monitoring and digital biomarkers

Key research areas informing the framework:
- Individual variation in physiological responses
- Context-dependent interpretation of biomarkers
- Predictive modeling in healthcare
- Real-time health monitoring systems

---

**Machinery Ecosystem**: _Comprehensive health intelligence through multi-system integration - predicting with mzekezeke, challenging with diggiden, optimizing with hatata, and orchestrating with Machinery core._

## System Directory Structure

```
machinery/
├── 🦀 src/                          # Machinery core (Rust)
├── 🐍 mzekezeke/                    # ML prediction system
├── ⚔️ diggiden/                      # Adversarial testing system  
├── 🎲 hatata/                       # Stochastic optimization system
├── assets/                         # Documentation assets
├── examples/                       # Integration examples
├── Cargo.toml                      # Rust dependencies
└── README.md                       # This file
```

Each subsystem contains its own documentation, examples, and test suites while integrating seamlessly with the Machinery orchestrator.
