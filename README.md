<h1 align="center">Kwasa Kwasa</h1>
<p align="center"><em>There is no reason for your soul to be misunderstood</em></p>

<p align="center">
  <img src="horizontal_film.gif" alt="Logo">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-%23000000.svg?e&logo=rust&logoColor=white)](#)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=fff)](#)

## Table of Contents

- [Introduction](#introduction)
- [Historical Context](#historical-context)
- [Theoretical Foundation](#theoretical-foundation)
- [System Architecture](#system-architecture)
- [Turbulance Language](#turbulance-language)
- [Reasoning Integration](#reasoning-integration)
- [Implementation](#implementation)
- [Installation and Usage](#installation-and-usage)
- [Contributing](#contributing)

## Introduction

Kwasa-Kwasa implements a revolutionary semantic computation framework based on **Biological Maxwell's Demons (BMD)**—information catalysts that create order from the combinatorial chaos of natural language, audio, and visual inputs. The system processes textual, visual, and auditory data through **Information Catalysts (iCat)** that operate as pattern selectors and output channelers, enabling genuine semantic understanding rather than mere pattern matching.

The framework operates on the principle that **semantics emerge from catalytic interactions** between pattern recognition and output channeling. All probabilistic reasoning is delegated to the Autobahn engine, while Kwasa-Kwasa focuses exclusively on semantic information catalysis across multiple modalities and scales.

## Historical Context

### The Philosophy Behind Kwasa-Kwasa

Kwasa-Kwasa takes its name from the vibrant musical style that emerged in the Democratic Republic of Congo in the 1980s. During a period when many African nations had recently gained independence, kwasa-kwasa represented a form of expression that transcended language barriers. Despite lyrics often being in Lingala, the music achieved widespread popularity across Africa because it communicated something that required no translation.

### Understanding Without Translation

In the early 1970s across Africa, leaders faced the rising restlessness of Black youth born after independence. This generation knew nothing of the hardships of war or rural living—they had been born in bustling city hospitals, educated by the continent's finest experts, had disposable income, and free weekends. Music had always been a medium for dancing, but European customs of seated listening were fundamentally misaligned with how music was experienced on the continent.

The breakthrough came when a musician named Kanda Bongo Man broke the rules of soukous (modern "Congolese Rhumba") by making a consequential structural change: he encouraged his guitarist, known as Diblo "Machine Gun" Dibala, to play solo guitar riffs after every verse.

Just as DJ Kool Herc recognized the potential of extended breaks in "Amen Brother," a mechanic from Kinshasa named Jenoaro saw similar possibilities in these guitar breaks. The dance was intensely physical—deliberately so. In regions where political independence was still a distant dream, kwasa-kwasa became a covert meeting ground for insurgent groups. Instead of clandestine gatherings, people could congregate at venues playing this popular music.

The lyrics? No one fully understood them, nor did they need to—the souls of the performers were understood without their words being comprehended. Artists like Awilo Longomba, Papa Wemba, Pepe Kale, and Alan Nkuku weren't merely performing—they were expressing their souls in a way that needed no translation.

This framework aims to achieve a similar preservation of meaning across computational transformation, ensuring that the essential nature of expression survives the translation into algorithmic form.

## Theoretical Foundation

### Biological Maxwell's Demons and Information Catalysis

Kwasa-Kwasa implements the theoretical framework of **Biological Maxwell's Demons** as proposed by Eduardo Mizraji, treating semantic processing as **information catalysis**. Following pioneering biologists like J.B.S. Haldane, Jacques Monod, and François Jacob, we recognize that biological systems use information catalysts to create order from chaos—exactly what semantic understanding requires.

### Information Catalysts (iCat)

Every semantic processing operation is performed by an **Information Catalyst**:

```
iCat_semantic = ℑ_input ∘ ℑ_output
```

Where:
- **ℑ_input**: Pattern recognition filter that selects meaningful structures from input chaos
- **ℑ_output**: Channeling operator that directs understanding toward specific targets
- **∘**: Functional composition creating emergent semantic understanding

### Multi-Scale Semantic Architecture

The system operates at multiple scales, mirroring biological organization:

1. **Molecular-Level Semantics**: Token/phoneme processing (analogous to enzymes)
2. **Neural-Level Semantics**: Sentence/phrase understanding (analogous to neural networks)  
3. **Cognitive-Level Semantics**: Document/discourse processing (analogous to complex cognition)

#### Text Processing as Semantic BMD

Text processing operates through **Token-level BMDs** that function like molecular enzymes, selecting meaningful patterns from character chaos:

```turbulance
item paragraph = "Machine learning improves diagnosis. However, limitations exist."

// Semantic catalysis through pattern recognition and channeling
item semantic_patterns = recognize_patterns(paragraph)
item channeled_understanding = channel_to_targets(semantic_patterns)

// Information catalysts decompose meaning
item claims = paragraph / claim           // iCat filters claim patterns
item evidence = paragraph / evidence      // iCat filters evidence patterns  
item qualifications = paragraph / qualification // iCat filters qualification patterns

// Catalytic combination preserves semantic coherence
item enhanced = claims + supporting_research + evidence
```

#### Image Processing as Visual BMD

Images are processed through **Visual Semantic BMDs** that operate like biological pattern recognition systems:

**Helicopter Engine**: Autonomous reconstruction validation ensures genuine understanding
**Pakati Regional Processing**: Specialized semantic catalysts for different image regions

#### Audio Processing as Auditory BMD

Audio content is processed through **Temporal Semantic BMDs** that recognize rhythmic and harmonic patterns, decomposing sound into meaningful temporal structures through information catalysis.

### Cross-Modal Semantic BMD Networks

The framework implements **Cross-Modal BMD Networks** where different semantic catalysts coordinate to create unified understanding across modalities:

```turbulance
item clinical_notes = "Patient reports chest pain and shortness of breath"
item chest_xray = load_image("chest_xray.jpg")
item heart_sounds = load_audio("cardiac_auscultation.wav")

// Cross-modal information catalysis
item text_bmd = semantic_catalyst(clinical_notes)
item visual_bmd = semantic_catalyst(chest_xray)
item audio_bmd = semantic_catalyst(heart_sounds)

// BMD network coordination
item multimodal_analysis = orchestrate_bmds(text_bmd, visual_bmd, audio_bmd)
item semantic_coherence = ensure_cross_modal_consistency(multimodal_analysis)
```

## System Architecture

The framework implements a **Semantic BMD Network Architecture** with clear separation between information catalysis and probabilistic reasoning:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KWASA-KWASA FRAMEWORK                        │
│                 (Semantic Information Catalysis)               │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │  
│  │            SEMANTIC BMD NETWORK                           │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │  │
│  │  │ Text BMDs   │ │ Image BMDs  │ │ Audio BMDs          │  │  │
│  │  │ • Token     │ │ • Helicopter│ │ • Temporal          │  │  │
│  │  │   Catalysts │ │   Engine    │ │   Catalysts         │  │  │
│  │  │ • Sentence  │ │ • Pakati    │ │ • Rhythmic          │  │  │
│  │  │   BMDs      │ │   Regional  │ │   Pattern BMDs      │  │  │
│  │  │ • Document  │ │   BMDs      │ │ • Harmonic          │  │  │
│  │  │   BMDs      │ │             │ │   Recognition       │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                │                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │               TURBULANCE LANGUAGE ENGINE                  │  │
│  │  • Information Catalyst Operations (iCat)                │  │
│  │  • Cross-Modal BMD Orchestration                          │  │
│  │  • Semantic Thermodynamic Constraints                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                │                                │
│                                ▼                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                 AUTOBAHN REASONING ENGINE                 │  │
│  │        (All Probabilistic Reasoning Delegated)           │  │  
│  │  • Probabilistic State Management                        │  │
│  │  • Uncertainty Quantification                            │  │
│  │  • Temporal Reasoning                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Semantic BMD Network**: Multi-scale information catalysts for pattern recognition and output channeling
2. **Turbulance Language Engine**: Unified syntax for semantic BMD operations across all modalities  
3. **Autobahn Integration**: Handles all probabilistic reasoning while Kwasa-Kwasa focuses on semantic catalysis

## Turbulance Language

Turbulance is a domain-specific language designed for **semantic information catalysis**. The language provides constructs for operating with Information Catalysts (BMDs) rather than raw data processing.

### Basic Syntax

```turbulance
// Working with Semantic BMDs
item text = "The patient shows signs of improvement"
item text_bmd = semantic_catalyst(text)
item understanding = catalytic_cycle(text_bmd)

item image = load_image("medical_scan.jpg")
item visual_bmd = semantic_catalyst(image, specificity_threshold: 0.9)
item visual_understanding = catalytic_cycle(visual_bmd)

item audio = load_audio("cardiac_sounds.wav")
item audio_bmd = semantic_catalyst(audio, pattern_recognition_threshold: 0.9)
item audio_understanding = catalytic_cycle(audio_bmd)

// BMD Network orchestration
item comprehensive_analysis = orchestrate_bmds(text_bmd, visual_bmd, audio_bmd)
```

### Propositions and Semantic BMD Motions

The language includes constructs for expressing semantic propositions processed through BMD networks:

```turbulance
proposition MedicalClarity:
    motion SemanticCatalysis("Medical data should achieve catalytic understanding")
    
    within image:
        item image_bmd = semantic_catalyst(image)
        given catalytic_specificity(image_bmd) < "Excellent":
            item enhanced_bmd = enhance_recognition_patterns(image_bmd)
            image_bmd = enhanced_bmd
        
        given thermodynamic_efficiency(image_bmd) == "Optimal":
            item diagnostic_output = channel_to_diagnostic_targets(image_bmd)
            return diagnostic_output
        alternatively:
            delegate_to_autobahn(image_bmd, "Semantic catalysis insufficient")
```

### Positional Semantic Catalysis  

The language treats position as a semantic catalyst parameter, recognizing that element location affects catalytic efficiency:

```turbulance
item sentence = "Critically, the patient's condition has improved significantly."

item sentence_bmd = semantic_catalyst(sentence)
item positional_patterns = extract_positional_catalysts(sentence_bmd)

considering catalyst in positional_patterns:
    given catalyst.semantic_role == SemanticRole::Intensifier:
        item catalytic_importance = catalyst.position_weight * catalytic_efficiency
        enhance_channeling_specificity(sentence_bmd, catalytic_importance)
```

## Reasoning Integration

### Autobahn Engine Delegation

The framework **delegates all probabilistic reasoning** to the Autobahn engine, maintaining clear separation between semantic catalysis and uncertainty processing. This delegation provides:

- **Probabilistic State Management**: All uncertainty quantification handled by Autobahn
- **Temporal Reasoning**: Time-dependent logical relationships processed externally  
- **Causal Structure Analysis**: Causal inference delegated to specialized probabilistic systems
- **Uncertainty Propagation**: Semantic BMDs focus on catalysis while Autobahn handles uncertainty

### Semantic BMD Pipeline

```rust
use kwasa_kwasa::{KwasaFramework, FrameworkConfig};
use autobahn::probabilistic::ProbabilisticReasoning;

let framework = KwasaFramework::new(FrameworkConfig {
    autobahn_config: Some(autobahn::DelegationConfig {
        probabilistic_reasoning: true,
        uncertainty_quantification: true,
        temporal_reasoning: true,
        ..Default::default()
    }),
    semantic_bmd_config: SemanticBMDConfig {
        thermodynamic_constraints: true,
        multi_scale_processing: true,
        cross_modal_orchestration: true,
    },
    ..Default::default()
}).await?;

let result = framework.process_turbulance_code(
    "item analysis = orchestrate_bmds(text_bmd, image_bmd, audio_bmd)"
).await?;
```

## Implementation

### Framework Architecture

The implementation consists of:

**Core Framework Modules**:
- `turbulance/` - DSL language implementation with semantic BMD operations
- `text_unit/` - Text BMD processing and semantic catalysis
- `semantic_bmds/` - Information catalyst implementations across modalities
- `knowledge/` - Knowledge representation and retrieval
- `cli/` - Command line interface and REPL

**Integration Layer**:
- Autobahn probabilistic reasoning delegation
- Cross-modal BMD coordination
- Thermodynamic constraint enforcement

**Optional Modules** (conditionally compiled):
- Chemistry processing (`kwasa-cheminformatics`)
- Biology analysis (`kwasa-systems-biology`)
- Spectrometry processing (`kwasa-spectrometry`)
- Multimedia handling (`kwasa-multimedia`)

### Processing Paradigms

#### Information Catalysis Through Pattern Recognition

The system validates understanding through **catalytic efficiency** rather than reconstruction fidelity:

```turbulance
funxn validate_semantic_catalysis(input_data):
    item input_bmd = semantic_catalyst(input_data)
    item catalytic_efficiency = measure_catalytic_performance(input_bmd)
    item thermodynamic_cost = calculate_energy_cost(input_bmd)
    
    given catalytic_efficiency > 0.95 && thermodynamic_cost < threshold:
        accept_catalytic_understanding(input_bmd)
    alternatively:
        refine_pattern_recognition(input_bmd)
```

#### Points and Resolutions via BMD Networks

The system uses semantic BMD networks for handling complex semantic resolution:

```turbulance
point semantic_hypothesis = {
    content: "Patient has pneumonia based on multimodal analysis",
    catalytic_certainty: 0.73,
    cross_modal_coherence: 0.68,
    thermodynamic_efficiency: 0.82
}

resolution diagnose_condition(point: SemanticPoint) -> DiagnosticOutcome {
    item diagnostic_bmds = orchestrate_semantic_catalysts(point)
    item probabilistic_analysis = delegate_to_autobahn(diagnostic_bmds)
    return integrate_semantic_and_probabilistic(diagnostic_bmds, probabilistic_analysis)
}
```

## Installation and Usage

### Prerequisites

- Rust 1.70+
- Autobahn reasoning engine

### Installation

```bash
git clone https://github.com/yourusername/kwasa-kwasa.git
cd kwasa-kwasa

# Build with core features
cargo build --release

# Build with all modules
cargo build --release --features="full"
```

### Basic Usage

```bash
# Run Turbulance script
./target/release/kwasa-kwasa run script.turb

# Start interactive REPL
./target/release/kwasa-kwasa repl

# Validate syntax
./target/release/kwasa-kwasa validate script.turb
```

### Programming Interface

```rust
use kwasa_kwasa::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let framework = KwasaFramework::with_defaults().await?;
    
    let result = framework.process_text(
        "The patient shows improvement in respiratory function.",
        None
    ).await?;
    
    println!("Analysis complete: {:?}", result);
    Ok(())
}
```

## Technology Stack

- **Rust**: Core implementation language
- **Autobahn**: Probabilistic reasoning and consciousness-aware processing
- **WebAssembly**: Browser deployment capability
- **SQLite**: Knowledge persistence
- **Logos/Chumsky**: Language parsing infrastructure

## Contributing

Contributions are welcome in the following areas:

1. **Language Development**: Expanding Turbulance syntax and semantics
2. **Processing Engines**: Improving text, image, and audio processing
3. **Integration**: Enhancing Autobahn integration and external module support
4. **Documentation**: Expanding examples and use cases
5. **Performance**: Optimizing processing efficiency

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Kwasa-Kwasa implements semantic computation principles through structured processing of meaning-preserving transformations across textual, visual, and auditory modalities.*
