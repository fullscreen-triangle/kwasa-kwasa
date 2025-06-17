<h1 align="center">Lavoisier</h1>
<p align="center"><em> Only the extraordinary can beget the extraordinary</em></p>

<p align="center">
  <img src="assets/logos/Antoine_lavoisier.jpg" alt="Spectacular Logo" width="300"/>
</p>

[![Python Version](https://img.shields.io/pypi/pyversions/science-platform.svg)](https://pypi.org/project/science-platform/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)
[![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=fff)](#)
[![ChatGPT](https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white)](#)
[![IntelliJ IDEA](https://img.shields.io/badge/IntelliJIDEA-000000.svg?logo=intellij-idea&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)

Lavoisier is a high-performance computing solution for mass spectrometry-based metabolomics data analysis pipelines. It combines traditional numerical methods with advanced visualization and AI-driven analytics to provide comprehensive insights from high-volume MS data.

## Core Architecture

Lavoisier features a sophisticated AI-driven architecture that combines multiple specialized modules for mass spectrometry analysis:

1. **Diadochi Framework**: Multi-domain LLM orchestration system for intelligent query routing and expert collaboration
2. **Mzekezeke**: Bayesian Evidence Network with Fuzzy Logic for probabilistic MS annotations
3. **Hatata**: Markov Decision Process verification layer for stochastic validation
4. **Zengeza**: Intelligent noise reduction using statistical analysis and machine learning
5. **Nicotine**: Context verification system with cryptographic puzzles for AI integrity
6. **Diggiden**: Adversarial testing system for evidence network vulnerability assessment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Lavoisier AI Architecture                           │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │                 │    │                 │    │                 │        │
│  │   Diadochi      │◄──►│   Mzekezeke     │◄──►│    Hatata       │        │
│  │   (LLM Routing) │    │ (Bayesian Net)  │    │ (MDP Verify)    │        │
│  │                 │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           ▲                        ▲                        ▲              │
│           │                        │                        │              │
│           ▼                        ▼                        ▼              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │                 │    │                 │    │                 │        │
│  │    Zengeza      │◄──►│    Nicotine     │◄──►│    Diggiden     │        │
│  │ (Noise Reduce)  │    │ (Context Verify)│    │ (Adversarial)   │        │
│  │                 │    │                 │    │                 │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Command Line Interface

Lavoisier provides a high-performance CLI interface for seamless interaction with all system components:

- Built with modern CLI frameworks for visually pleasing, intuitive interaction
- Color-coded outputs, progress indicators, and interactive components
- Command completions and contextual help
- Workflow management and pipeline orchestration
- Integrated with LLM assistants for natural language interaction
- Configuration management and parameter customization
- Results visualization and reporting

## Numerical Processing Pipeline

The numerical pipeline processes raw mass spectrometry data through a distributed computing architecture, specifically designed for handling large-scale MS datasets:

### Raw Data Processing
- Extracts MS1 and MS2 spectra from mzML files
- Performs intensity thresholding (MS1: 1000.0, MS2: 100.0 by default)
- Applies m/z tolerance filtering (0.01 Da default)
- Handles retention time alignment (0.5 min tolerance)

### Comprehensive MS2 Annotation
- Multi-database annotation system integrating multiple complementary resources
- Spectral matching against libraries (MassBank, METLIN, MzCloud, in-house)
- Accurate mass search across HMDB, LipidMaps, KEGG, and PubChem
- Fragmentation tree generation for structural elucidation
- Pathway integration with KEGG and HumanCyc databases
- Multi-component confidence scoring system for reliable identifications
- Deep learning models for MS/MS prediction and interpretation

### Enhanced MS2 Analysis
- Deep learning models for spectral interpretation
- Transfer learning from large-scale metabolomics datasets
- Model serialization for all analytical outputs
- Automated hyperparameter optimization

### Distributed Computing
- Utilizes Ray for parallel processing
- Implements Dask for large dataset handling
- Automatic resource management based on system capabilities
- Dynamic workload distribution across available cores

### Data Management
- Efficient data storage using Zarr format
- Compressed data storage with LZ4 compression
- Parallel I/O operations for improved performance
- Hierarchical data organization

### Processing Features
- Automatic chunk size optimization
- Memory-efficient processing
- Progress tracking and reporting
- Comprehensive error handling and logging

## Visual Analysis Pipeline

The visualization pipeline transforms processed MS data into interpretable visual formats:

### Spectrum Analysis
- MS image database creation and management
- Feature extraction from spectral data
- Resolution-specific image generation (default 1024x1024)
- Feature dimension handling (128-dimensional by default)

### Visualization Generation
- Creates time-series visualizations of MS data
- Generates analysis videos showing spectral changes
- Supports multiple visualization formats
- Custom color mapping and scaling

### Data Integration
- Combines multiple spectra into cohesive visualizations
- Temporal alignment of spectral data
- Metadata integration into visualizations
- Batch processing capabilities

### Output Formats
- High-resolution image generation
- Video compilation of spectral changes
- Interactive visualization options
- Multiple export formats support

## LLM Integration & Continuous Learning

Lavoisier integrates commercial and open-source LLMs to enhance analytical capabilities and enable continuous learning:

### Assistive Intelligence
- Natural language interface through CLI
- Context-aware analytical assistance
- Automated report generation
- Expert knowledge integration

### Solver Architecture
- Integration with Claude, GPT, and other commercial LLMs
- Local models via Ollama for offline processing
- Numerical model API endpoints for LLM queries
- Pipeline result interpretation

### Continuous Learning System
- Feedback loop capturing new analytical results
- Incremental model updates via train-evaluate cycles
- Knowledge distillation from commercial LLMs to local models
- Versioned model repository with performance tracking

### Metacognitive Query Generation
- Auto-generated queries of increasing complexity
- Integration of numerical model outputs with LLM knowledge
- Comparative analysis between numeric and visual pipelines
- Knowledge extraction and synthesis

## Specialized Models Integration

Lavoisier incorporates domain-specific models for advanced analysis tasks:

### Biomedical Language Models
- BioMedLM integration for biomedical text analysis and generation
- Context-aware analysis of mass spectrometry data
- Biological pathway interpretation and metabolite identification
- Custom prompting templates for different analytical tasks

### Scientific Text Encoders
- SciBERT model for scientific literature processing and embedding
- Multiple pooling strategies for optimal text representation
- Similarity-based search across scientific documents
- Batch processing of large text collections

### Chemical Named Entity Recognition
- PubMedBERT-NER-Chemical for extracting chemical compounds from text
- Identification and normalization of chemical nomenclature
- Entity replacement for text preprocessing
- High-precision extraction with confidence scoring

### Proteomics Analysis
- InstaNovo model for de novo peptide sequencing
- Integration of proteomics and metabolomics data
- Cross-modal analysis for comprehensive biomolecule profiling
- Advanced protein identification workflows

## Advanced Model Architecture

Lavoisier features a comprehensive multi-tier model architecture that integrates cutting-edge AI technologies:

### 1. Models Module (`lavoisier.models`)

The models module provides a complete framework for managing, versioning, and deploying specialized AI models:

#### Chemical Language Models (`chemical_language_models.py`)
- **ChemBERTa Model**: Pre-trained transformer for molecular property prediction
  - SMILES encoding with multiple pooling strategies (CLS, mean, max)
  - Support for molecular embedding generation
  - Integration with DeepChem models
- **MoLFormer Model**: Large-scale molecular representation learning
  - Advanced molecular understanding through self-supervised learning
  - Custom tokenization for chemical structures
  - Transfer learning capabilities for downstream tasks
- **PubChemDeBERTa Model**: Specialized for chemical property prediction
  - Fine-tuned on PubChem data
  - Multi-task learning for property prediction
  - High-accuracy molecular classification

#### Spectral Transformer Models (`spectral_transformers.py`)
- **SpecTUS Model**: EI-MS spectra to SMILES conversion
  - Transformer-based spectrum interpretation
  - Direct structural elucidation from mass spectra
  - Batch processing and beam search optimization
  - Preprocessing pipelines for spectral data normalization

#### Embedding Models (`embedding_models.py`)
- **CMSSP Model**: Joint embeddings of MS/MS spectra and molecules
  - Cross-modal representation learning
  - Spectral similarity computation
  - Molecular structure-spectrum alignment
  - Batch processing for high-throughput analysis

#### Model Repository System (`repository.py`)
- Centralized model storage and retrieval
- Model versioning and metadata management
- Automatic model updates and synchronization
- Performance tracking and model comparison

#### Knowledge Distillation (`distillation.py`)
- Academic paper knowledge extraction
- Pipeline-specific model creation
- Ollama integration for local model deployment
- Progressive complexity training
- Model testing and validation frameworks

#### Model Registry (`registry.py`)
- Unified model discovery and management
- HuggingFace model integration
- Model type classification and organization
- Automatic model loading and configuration

#### Version Management (`versioning.py`)
- Comprehensive model versioning system
- Metadata tracking and validation
- Model performance history
- Rollback and comparison capabilities

### 2. LLM Integration Module (`lavoisier.llm`)

The LLM module provides comprehensive integration with large language models for enhanced analytical capabilities:

#### LLM Service Architecture (`service.py`)
- **Multi-Provider Support**: Seamless integration with commercial and local LLMs
- **Query Generation**: Automatic analytical query generation with increasing complexity
- **Caching System**: Intelligent result caching for improved performance
- **Asynchronous Processing**: Concurrent LLM request handling
- **Progressive Analysis**: Multi-stage analysis with escalating complexity
- **Pipeline Comparison**: Automated comparison between numerical and visual pipelines

#### API Client Layer (`api.py`)
- **Unified Interface**: Standardized API for different LLM providers
- **OpenAI Integration**: Direct integration with GPT models
- **Anthropic Integration**: Claude model support
- **Error Handling**: Robust error recovery and retry mechanisms
- **Rate Limiting**: Automatic rate limiting and quota management

#### Commercial LLM Proxy (`commercial.py`)
- **Provider Abstraction**: Unified interface for multiple commercial providers
- **Load Balancing**: Intelligent request distribution across providers
- **Cost Optimization**: Automatic provider selection based on cost and performance
- **API Key Management**: Secure credential handling

#### Local LLM Support (`ollama.py`)
- **Ollama Integration**: Complete integration with Ollama for local inference
- **Model Management**: Automatic model downloading and management
- **Offline Capabilities**: Full functionality without internet connectivity
- **Custom Model Support**: Support for fine-tuned and specialized models

#### Query Generation System (`query_gen.py`)
- **Adaptive Queries**: Context-aware query generation
- **Complexity Scaling**: Progressive query complexity increase
- **Domain-Specific Templates**: Specialized query templates for different analysis types
- **Multi-Modal Integration**: Queries combining numerical and visual analysis results

#### Chemical NER (`chemical_ner.py`)
- **PubMedBERT-NER**: Chemical entity recognition from text
- **Batch Processing**: High-throughput text processing
- **Entity Normalization**: Standardized chemical nomenclature
- **Confidence Scoring**: Reliability assessment for extracted entities

#### Text Encoders (`text_encoders.py`)
- **SciBERT Integration**: Scientific text understanding
- **Multiple Pooling Strategies**: Optimized text representation
- **Similarity Search**: Semantic similarity computation
- **Batch Processing**: Efficient large-scale text processing

#### Specialized LLM (`specialized_llm.py`)
- **Domain-Specific Models**: Models fine-tuned for mass spectrometry
- **Context-Aware Processing**: Specialized prompting strategies
- **Multi-Modal Understanding**: Integration of spectral and textual data

### 3. AI Integration Module (`lavoisier.ai_modules.integration`)

The integration module orchestrates all AI components into a cohesive analytical system:

#### Advanced MS Analysis System
- **Multi-Module Orchestration**: Coordinates all six AI modules (Diadochi, Mzekezeke, Hatata, Zengeza, Nicotine, Diggiden)
- **Parallel Processing**: Concurrent execution of multiple AI modules
- **Result Integration**: Unified analysis results from all modules
- **Quality Assessment**: Multi-layered validation and confidence scoring
- **Comprehensive Reporting**: Detailed analysis reports with all module outputs

#### Analysis Pipeline
1. **Stage 1 - Noise Reduction**: Zengeza intelligent noise removal
2. **Stage 2 - Evidence Networks**: Mzekezeke Bayesian network construction
3. **Stage 3 - Context Verification**: Nicotine cryptographic puzzle validation
4. **Stage 4 - MDP Validation**: Hatata stochastic verification
5. **Stage 5 - Security Assessment**: Diggiden adversarial testing
6. **Stage 6 - Integration**: Unified result compilation and confidence scoring

#### System Health Monitoring
- **Performance Metrics**: Real-time system performance tracking
- **Module Health**: Individual module status monitoring
- **Resource Utilization**: CPU, memory, and GPU usage tracking
- **Error Detection**: Automatic error detection and recovery

#### Export and Reporting
- **Multiple Formats**: JSON, CSV, HDF5, and custom formats
- **Visualization Ready**: Data formatted for immediate visualization
- **Audit Trail**: Complete analysis history and provenance
- **Quality Grades**: Automated quality assessment and grading

## Enhanced System Architecture

The complete Lavoisier architecture now includes these additional layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Enhanced Lavoisier Architecture                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          LLM Integration Layer                          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Commercial  │  │    Local    │  │   Query     │  │  Chemical   │   │ │
│  │  │     LLMs    │  │    LLMs     │  │  Generator  │  │     NER     │   │ │
│  │  │ (GPT/Claude)│  │  (Ollama)   │  │             │  │             │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Models Management Layer                        │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │  Chemical   │  │  Spectral   │  │ Embedding   │  │ Knowledge   │   │ │
│  │  │  Language   │  │Transformers │  │   Models    │  │Distillation │   │ │
│  │  │   Models    │  │             │  │             │  │             │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        AI Integration Layer                             │ │
│  │                    ┌─────────────────────────┐                         │ │
│  │                    │  Advanced MS Analysis   │                         │ │
│  │                    │        System           │                         │ │
│  │                    └─────────────────────────┘                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Core AI Modules                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │  Diadochi   │  │  Mzekezeke  │  │   Hatata    │  │   Zengeza   │   │ │
│  │  │ (LLM Route) │  │(Bayes Net)  │  │(MDP Verify) │  │(Noise Reduce│   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  │  ┌─────────────┐  ┌─────────────┐                                      │ │
│  │  │  Nicotine   │  │  Diggiden   │                                      │ │
│  │  │(Context Ver)│  │(Adversarial)│                                      │ │
│  │  └─────────────┘  └─────────────┘                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      ▲                                       │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      Processing Pipelines                              │ │
│  │  ┌─────────────────┐                    ┌─────────────────┐            │ │
│  │  │    Numerical    │                    │     Visual      │            │ │
│  │  │    Pipeline     │                    │    Pipeline     │            │ │
│  │  │                 │                    │                 │            │ │
│  │  └─────────────────┘                    └─────────────────┘            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Capabilities

### AI-Driven Analysis
- **Multi-Modal Reasoning**: Combines statistical, probabilistic, and logical approaches
- **Adversarial Robustness**: Built-in protection against data poisoning and attacks
- **Context Preservation**: Cryptographic verification of analysis context
- **Uncertainty Quantification**: Fuzzy logic and Bayesian approaches for uncertainty
- **Knowledge Distillation**: Academic literature integration into specialized models
- **Progressive Learning**: Continuous improvement through feedback loops

### Advanced Model Management
- **Model Repository**: Centralized storage and versioning system
- **Automatic Updates**: Self-updating model ecosystem
- **Performance Tracking**: Continuous model performance monitoring
- **Cross-Modal Learning**: Integration of spectral and chemical language models
- **Custom Model Creation**: Tools for creating domain-specific models

### LLM-Enhanced Analysis
- **Natural Language Interface**: Query mass spectrometry data using natural language
- **Automated Report Generation**: AI-generated analytical reports
- **Multi-Provider Support**: Seamless integration with various LLM providers
- **Local and Cloud**: Both offline and online LLM capabilities
- **Context-Aware Processing**: LLMs with domain-specific knowledge

### Advanced Annotation System
- **Evidence Network**: Graph-based evidence correlation and validation
- **Probabilistic Scoring**: Bayesian inference for annotation confidence
- **Fuzzy Membership**: Handles uncertainty in mass spectrometry measurements
- **Multi-Database Integration**: Combines evidence from multiple sources
- **Chemical Language Understanding**: Advanced chemical entity recognition
- **Spectral-Structure Alignment**: Direct spectrum-to-structure prediction

### Quality Assurance
- **MDP Verification**: Stochastic validation of analysis workflows
- **Adversarial Testing**: Systematic vulnerability assessment
- **Context Integrity**: Cryptographic verification of analysis context
- **Noise Characterization**: Advanced noise modeling and removal
- **Multi-Layer Validation**: Independent verification systems
- **System Health Monitoring**: Real-time performance and reliability tracking

### Performance Optimization
- **Intelligent Routing**: LLM-based query routing to appropriate experts
- **Parallel Processing**: Multi-expert parallel analysis
- **Adaptive Algorithms**: Self-optimizing analysis parameters
- **Resource Management**: Efficient computational resource allocation
- **Model Caching**: Intelligent model and result caching
- **Batch Processing**: Optimized high-throughput analysis

### Performance
- Processing speeds: Up to 1000 spectra/second (hardware dependent)
- Memory efficiency: Streaming processing for large datasets
- Scalability: Automatic adjustment to available resources
- Parallel processing: Multi-core utilization

### Data Handling
- Input formats: mzML (primary), with extensible format support
- Output formats: Zarr, HDF5, video (MP4), images (PNG/JPEG)
- Data volumes: Capable of handling datasets >100GB
- Batch processing: Multiple file handling

### Annotation Capabilities
- Multi-tiered annotation combining spectral matching and accurate mass search
- Integrated pathway analysis for biological context
- Confidence scoring system weighing multiple evidence sources
- Parallelized database searches for rapid compound identification
- Isotope pattern matching and fragmentation prediction
- RT prediction for additional identification confidence

### Analysis Features
- Peak detection and quantification
- Retention time alignment
- Mass accuracy verification
- Intensity normalization

## Use Cases

### Advanced MS Analysis
- **Intelligent Annotation**: AI-driven compound identification with uncertainty quantification
- **Robust Analysis**: Adversarially-tested analysis pipelines
- **Quality Validation**: Multi-layer verification of results
- **Context-Aware Processing**: Maintains analysis context throughout workflows

### AI Research Applications
- **Multi-Domain LLM Systems**: Template for combining specialized AI models
- **Adversarial ML Research**: Framework for testing ML robustness
- **Bayesian Network Applications**: Probabilistic reasoning in scientific domains
- **Context Verification**: Novel approaches to AI system integrity

### Security and Robustness
- **Adversarial Testing**: Systematic evaluation of AI system vulnerabilities
- **Data Integrity**: Cryptographic verification of analysis context
- **Noise Robustness**: Advanced noise modeling and mitigation
- **Quality Assurance**: Multi-modal validation of scientific results

### Proteomics Research
- Protein identification workflows
- Peptide quantification
- Post-translational modification analysis
- Comparative proteomics studies
- De novo peptide sequencing with InstaNovo integration
- Cross-analysis of proteomics and metabolomics datasets
- Protein-metabolite interaction mapping

### Metabolomics Studies
- Metabolite profiling
- Pathway analysis
- Biomarker discovery
- Time-series metabolomics

### Quality Control
- Instrument performance monitoring
- Method validation
- Batch effect detection
- System suitability testing

### Data Visualization
- Scientific presentation
- Publication-quality figures
- Time-course analysis
- Comparative analysis visualization

## Results & Validation

Our comprehensive validation demonstrates the effectiveness of Lavoisier's dual-pipeline approach through rigorous statistical analysis and performance metrics:

### Core Performance Metrics
- **Feature Extraction Accuracy**: 0.989 similarity score between pipelines, with complementarity index of 0.961
- **Vision Pipeline Robustness**: 0.954 stability score against noise/perturbations
- **Annotation Performance**: Numerical pipeline achieves perfect accuracy (1.0) for known compounds
- **Temporal Consistency**: 0.936 consistency score for time-series analysis
- **Anomaly Detection**: Low anomaly score of 0.02, indicating reliable performance

### Example Analysis Results

#### Mass Spectrometry Analysis
![Full MS Scan](assets/analytical_visualizations/20250527_094000/mass_spectra/full_scan.png)
*Full scan mass spectrum showing the comprehensive metabolite profile with high mass accuracy and resolution*

![MS/MS Analysis](assets/analytical_visualizations/20250527_094000/mass_spectra/glucose_msms.png)
*MS/MS fragmentation pattern analysis for glucose, demonstrating detailed structural elucidation*

![Feature Comparison](assets/analytical_visualizations/20250527_094000/feature_analysis/feature_comparison.png)
*Comparison of feature extraction between numerical and visual pipelines, showing high concordance and complementarity*

#### Visual Pipeline Output

The following video demonstrates our novel computer vision approach to mass spectrometry analysis:

<video width="100%" controls>
  <source src="public/output/visual/videos/analysis_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*The video above shows real-time mass spectrometry data analysis through our computer vision pipeline, demonstrating:*
- Real-time conversion of mass spectra to visual patterns
- Dynamic feature detection and tracking across time
- Metabolite intensity changes visualized as flowing patterns
- Structural similarities highlighted through visual clustering
- Pattern changes detected and analyzed in real-time

### Technical Details: Novel Visual Analysis Method

The visual pipeline represents a groundbreaking approach to mass spectrometry data analysis through computer vision techniques. This section details the mathematical foundations and implementation of the method demonstrated in the video above.

#### Mathematical Formulation

1. **Spectrum-to-Image Transformation**
   The conversion of mass spectra to visual representations follows:
   ```
   F(m/z, I) → R^(n×n)
   ```
   where:
   - m/z ∈ R^k: mass-to-charge ratio vector
   - I ∈ R^k: intensity vector
   - n: resolution dimension (default: 1024)
   
   The transformation is defined by:
   ```
   P(x,y) = G(σ) * ∑[δ(x - φ(m/z)) · ψ(I)]
   ```
   where:
   - P(x,y): pixel intensity at coordinates (x,y)
   - G(σ): Gaussian kernel with σ=1
   - φ: m/z mapping function to x-coordinate
   - ψ: intensity scaling function (log1p transform)
   - δ: Dirac delta function

2. **Temporal Integration**
   Sequential frames are processed using a sliding window approach:
   ```
   B_t = {F_i | i ∈ [t-w, t]}
   ```
   where:
   - B_t: frame buffer at time t
   - w: window size (default: 30 frames)
   - F_i: transformed frame at time i

#### Feature Detection and Tracking

1. **Scale-Invariant Feature Transform (SIFT)**
   - Keypoint detection using DoG (Difference of Gaussians)
   - Local extrema detection in scale space
   - Keypoint localization and filtering
   - Orientation assignment
   - Feature descriptor generation

2. **Temporal Pattern Analysis**
   - Optical flow computation using Farneback method
   - Flow magnitude and direction analysis:
     ```
     M(x,y) = √(fx² + fy²)
     θ(x,y) = arctan(fy/fx)
     ```
   where:
   - M: flow magnitude
   - θ: flow direction
   - fx, fy: flow vectors

#### Pattern Recognition

1. **Feature Correlation**
   Temporal patterns are analyzed using frame-to-frame correlation:
   ```
   C(i,j) = corr(F_i, F_j)
   ```
   where C(i,j) is the correlation coefficient between frames i and j.

2. **Significant Movement Detection**
   Features are tracked using a statistical threshold:
   ```
   T = μ(M) + 2σ(M)
   ```
   where:
   - T: movement threshold
   - μ(M): mean flow magnitude
   - σ(M): standard deviation of flow magnitude

#### Implementation Details

1. **Resolution and Parameters**
   - Frame resolution: 1024×1024 pixels
   - Feature vector dimension: 128
   - Gaussian blur σ: 1.0
   - Frame rate: 30 fps
   - Window size: 30 frames

2. **Processing Pipeline**
   a. Raw spectrum acquisition
   b. m/z and intensity normalization
   c. Coordinate mapping
   d. Gaussian smoothing
   e. Feature detection
   f. Temporal integration
   g. Video generation

3. **Quality Metrics**
   - Structural Similarity Index (SSIM)
   - Peak Signal-to-Noise Ratio (PSNR)
   - Feature stability across frames
   - Temporal consistency measures

This novel approach enables:
- Real-time visualization of spectral changes
- Pattern detection in complex MS data
- Intuitive interpretation of metabolomic profiles
- Enhanced feature detection through computer vision
- Temporal analysis of metabolite dynamics

#### Analysis Outputs
The system generates comprehensive analytical outputs organized in:

1. **Time Series Analysis** (`time_series/`)
   - Chromatographic peak tracking
   - Retention time alignment
   - Intensity variation monitoring

2. **Feature Analysis** (`feature_analysis/`)
   - Principal component analysis
   - Feature clustering
   - Pattern recognition results

3. **Interactive Dashboards** (`interactive_dashboards/`)
   - Real-time data exploration
   - Dynamic filtering capabilities
   - Interactive peak annotation

4. **Publication Quality Figures** (`publication_figures/`)
   - High-resolution spectral plots
   - Statistical analysis visualizations
   - Comparative analysis figures

### Pipeline Complementarity
The dual-pipeline approach shows strong synergistic effects:
- **Feature Comparison**: Multiple validation scores [1.0, 0.999, 0.999, 0.999, 0.932, 1.0] across different aspects
- **Vision Analysis**: Robust performance in both noise resistance (0.914) and temporal analysis (0.936)
- **Annotation Synergy**: While numerical pipeline excels in accuracy, visual pipeline provides complementary insights

### Validation Methodology
For detailed information about our validation approach and complete results, please refer to:
- [Visualization Documentation](docs/visualization.md) - Comprehensive analysis framework
- `validation_results/` - Raw validation data and metrics
- `validation_visualizations/` - Interactive visualizations and temporal analysis
- `assets/analytical_visualizations/` - Detailed analytical outputs

## Project Structure

```
lavoisier/
├── pyproject.toml            # Project metadata and dependencies
├── LICENSE                   # Project license
├── README.md                 # This file
├── docs/                     # Documentation
│   ├── ai-modules.md         # Comprehensive AI modules documentation
│   ├── user_guide.md         # User documentation
│   ├── developer_guide.md    # Developer documentation
│   ├── architecture.md       # System architecture details
│   └── performance.md        # Performance benchmarking
├── lavoisier/                # Main package
│   ├── __init__.py           # Package initialization
│   ├── diadochi/             # Multi-domain LLM framework
│   │   ├── __init__.py
│   │   ├── core.py           # Core framework components
│   │   ├── routers.py        # Query routing strategies
│   │   ├── chains.py         # Sequential processing chains
│   │   └── mixers.py         # Response mixing strategies
│   ├── ai_modules/           # Specialized AI modules
│   │   ├── __init__.py
│   │   ├── integration.py    # AI system orchestration
│   │   ├── mzekezeke.py      # Bayesian Evidence Network
│   │   ├── hatata.py         # MDP Verification Layer
│   │   ├── zengeza.py        # Intelligent Noise Reduction
│   │   ├── nicotine.py       # Context Verification System
│   │   └── diggiden.py       # Adversarial Testing Framework
│   ├── models/               # AI Model Management
│   │   ├── __init__.py
│   │   ├── chemical_language_models.py  # ChemBERTa, MoLFormer, PubChemDeBERTa
│   │   ├── spectral_transformers.py     # SpecTUS model
│   │   ├── embedding_models.py          # CMSSP model
│   │   ├── huggingface_models.py        # HuggingFace integration
│   │   ├── distillation.py              # Knowledge distillation
│   │   ├── registry.py                  # Model registry system
│   │   ├── repository.py                # Model repository
│   │   ├── versioning.py                # Model versioning
│   │   └── papers.py                    # Research papers integration
│   ├── llm/                  # LLM Integration Layer
│   │   ├── __init__.py
│   │   ├── service.py        # LLM service architecture
│   │   ├── api.py            # API client layer
│   │   ├── query_gen.py      # Query generation system
│   │   ├── commercial.py     # Commercial LLM proxy
│   │   ├── ollama.py         # Local LLM support
│   │   ├── chemical_ner.py   # Chemical NER
│   │   ├── text_encoders.py  # Scientific text encoders
│   │   └── specialized_llm.py # Specialized LLM implementations
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Logging utilities
│   │   └── registry.py       # Component registry
│   ├── numerical/            # Traditional MS analysis pipeline
│   │   ├── __init__.py
│   │   ├── numeric.py        # Main numerical analysis
│   │   ├── ms1.py            # MS1 spectra analysis
│   │   ├── ms2.py            # MS2 spectra analysis
│   │   └── io/               # Input/output operations
│   │       ├── __init__.py
│   │       ├── readers.py    # File format readers
│   │       └── writers.py    # File format writers
│   ├── visual/               # Computer vision pipeline
│   │   ├── __init__.py
│   │   ├── conversion.py     # Spectra to visual conversion
│   │   ├── processing.py     # Visual processing
│   │   ├── video.py          # Video generation
│   │   └── analysis.py       # Visual analysis
│   ├── proteomics/           # Proteomics analysis
│   │   └── __init__.py       # Proteomics module initialization
│   ├── cli/                  # Command-line interface
│   │   ├── __init__.py
│   │   ├── app.py            # CLI application entry point
│   │   ├── commands/         # CLI command implementations
│   │   └── ui/               # Terminal UI components
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── helpers.py        # General helpers
│       └── validation.py     # Validation utilities
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_ai_modules.py    # AI modules tests
│   ├── test_models.py        # Models module tests
│   ├── test_llm.py           # LLM integration tests
│   ├── test_diadochi.py      # Diadochi framework tests
│   ├── test_numerical.py     # Numerical pipeline tests
│   └── test_cli.py           # CLI tests
├── scripts/                  # Analysis scripts
│   ├── run_mtbls1707_analysis.py # MTBLS1707 benchmark
│   └── benchmark_analysis.py     # Performance benchmarking
└── examples/                 # Example workflows
    ├── ai_assisted_analysis.py   # AI-driven analysis
    ├── adversarial_testing.py    # Security testing
    ├── bayesian_annotation.py    # Bayesian network annotation
    ├── model_distillation.py     # Knowledge distillation example
    ├── llm_integration.py        # LLM integration example
    └── complete_pipeline.py      # Full pipeline example
```

## Installation & Usage

### Installation

```bash
pip install lavoisier
```

For development installation:

```bash
git clone https://github.com/username/lavoisier.git
cd lavoisier
pip install -e ".[dev]"
```

### Basic Usage

Process a single MS file:

```bash
lavoisier process --input sample.mzML --output results/
```

Run with LLM assistance:

```bash
lavoisier analyze --input sample.mzML --llm-assist
```