use logos::{Logos, Span};
use std::fmt;

/// TokenKind represents all possible token types in the Turbulance language
#[derive(Logos, Debug, Clone, Hash, Eq, PartialEq)]
pub enum TokenKind {
    // Keywords
    #[token("funxn")]
    FunctionDecl,

    #[token("project")]
    ProjectDecl,

    #[token("proposition")]
    Proposition,

    // New scientific reasoning keywords
    #[token("evidence")]
    Evidence,

    #[token("pattern")]
    Pattern,

    #[token("support")]
    Support,

    #[token("contradict")]
    Contradict,

    #[token("inconclusive")]
    Inconclusive,

    #[token("requirements")]
    Requirements,

    #[token("signature")]
    Signature,

    #[token("match")]
    Match,

    #[token("meta")]
    Meta,

    #[token("derive_hypotheses")]
    DeriveHypotheses,

    #[token("alternatively")]
    Alternatively,

    #[token("with")]
    With,

    #[token("classify_as")]
    ClassifyAs,

    #[token("confidence")]
    Confidence,

    #[token("emergent_behaviors")]
    EmergentBehaviors,

    #[token("mechanisms")]
    Mechanisms,

    #[token("clinical_relevance")]
    ClinicalRelevance,

    #[token("refined_hypotheses")]
    RefinedHypotheses,

    #[token("recommendations")]
    Recommendations,

    // Advanced orchestration keywords  
    #[token("flow")]
    Flow,

    #[token("on")]
    On,

    #[token("catalyze")]
    Catalyze,

    #[token("cross_scale")]
    CrossScale,

    #[token("coordinate")]
    Coordinate,

    #[token("drift")]
    Drift,

    #[token("until")]
    Until,

    #[token("cycle")]
    Cycle,

    #[token("roll")]
    Roll,

    #[token("resolve")]
    Resolve,

    #[token("execute_information_catalysis")]
    ExecuteInformationCatalysis,

    #[token("create_pattern_recognizer")]
    CreatePatternRecognizer,

    #[token("create_action_channeler")]
    CreateActionChanneler,

    #[token("capture_screen_pixels")]
    CaptureScreenPixels,

    #[token("point")]
    Point,

    #[token("content")]
    Content,

    #[token("certainty")]
    Certainty,

    #[token("evidence_strength")]
    EvidenceStrength,

    #[token("contextual_relevance")]
    ContextualRelevance,

    #[token("urgency_factor")]
    UrgencyFactor,

    // Autobahn reference keywords
    #[token("funxn")]
    Funxn,

    #[token("metacognitive")]
    Metacognitive,

    #[token("goal")]
    Goal,

    #[token("optimize_until")]
    OptimizeUntil,

    #[token("try")]
    Try,

    #[token("catch")]
    Catch,

    #[token("finally")]
    Finally,

    #[token("parallel")]
    Parallel,

    #[token("async")]
    Async,

    #[token("await")]
    Await,

    #[token("import")]
    Import,

    #[token("from")]
    From,

    #[token("otherwise")]
    Otherwise,

    #[token("within")]
    Within,

    #[token("each")]
    Each,

    #[token("all")]
    All,

    #[token("these")]
    These,

    #[token("for")]
    For,

    #[token("while")]
    While,

    #[token("break")]
    Break,

    #[token("continue")]
    Continue,

    #[token("description")]
    Description,

    #[token("success_threshold")]
    SuccessThreshold,

    #[token("metrics")]
    Metrics,

    #[token("subgoals")]
    Subgoals,

    #[token("weight")]
    Weight,

    #[token("threshold")]
    Threshold,

    #[token("constraints")]
    Constraints,

    #[token("requires_evidence")]
    RequiresEvidence,

    #[token("support")]
    Support,

    #[token("with_weight")]
    WithWeight,

    #[token("collect")]
    Collect,

    #[token("collect_batch")]
    CollectBatch,

    #[token("validation_rules")]
    ValidationRules,

    #[token("processing_pipeline")]
    ProcessingPipeline,

    #[token("track_reasoning")]
    TrackReasoning,

    #[token("evaluate_confidence")]
    EvaluateConfidence,

    #[token("detect_bias")]
    DetectBias,

    #[token("adapt_behavior")]
    AdaptBehavior,

    #[token("analyze_decision_history")]
    AnalyzeDecisionHistory,

    #[token("update_decision_strategies")]
    UpdateDecisionStrategies,

    #[token("increase_evidence_requirements")]
    IncreaseEvidenceRequirements,

    #[token("reduce_computational_overhead")]
    ReduceComputationalOverhead,

    // Biological operations
    #[token("process_molecule")]
    ProcessMolecule,

    #[token("harvest_energy")]
    HarvestEnergy,

    #[token("extract_information")]
    ExtractInformation,

    #[token("update_membrane_state")]
    UpdateMembraneState,

    #[token("configure_membrane")]
    ConfigureMembrane,

    // Scientific functions
    #[token("calculate_entropy_change")]
    CalculateEntropyChange,

    #[token("gibbs_free_energy")]
    GibbsFreeEnergy,

    #[token("shannon")]
    Shannon,

    #[token("mutual_info")]
    MutualInfo,

    #[token("info_gain")]
    InfoGain,

    #[token("calculate_mw")]
    CalculateMw,

    #[token("calculate_ka")]
    CalculateKa,

    #[token("analyze_flux")]
    AnalyzeFlux,

    #[token("calculate_kcat_km")]
    CalculateKcatKm,

    // Quantum operations
    #[token("quantum_state")]
    QuantumState,

    #[token("amplitude")]
    Amplitude,

    #[token("phase")]
    Phase,

    #[token("coherence_time")]
    CoherenceTime,

    #[token("apply_hadamard")]
    ApplyHadamard,

    #[token("apply_cnot")]
    ApplyCnot,

    #[token("measure")]
    Measure,

    #[token("measure_entanglement")]
    MeasureEntanglement,

    #[token("parallel_execute")]
    ParallelExecute,

    #[token("await_all_tasks")]
    AwaitAllTasks,

    // Pattern types
    #[token("temporal")]
    Temporal,

    #[token("spatial")]
    Spatial,

    #[token("oscillatory")]
    Oscillatory,

    #[token("emergent")]
    Emergent,

    // Additional scientific keywords
    #[token("matches")]
    Matches,

    #[token("contains")]
    Contains,

    #[token("temperature")]
    Temperature,

    #[token("ph_level")]
    PhLevel,

    #[token("concentration")]
    Concentration,

    #[token("catalyst")]
    Catalyst,

    #[token("monitor_efficiency")]
    MonitorEfficiency,

    #[token("target_yield")]
    TargetYield,

    #[token("adaptive_optimization")]
    AdaptiveOptimization,

    #[token("processing_method")]
    ProcessingMethod,

    #[token("noise_filtering")]
    NoiseFiltering,

    #[token("confidence_threshold")]
    ConfidenceThreshold,

    #[token("permeability")]
    Permeability,

    #[token("selectivity")]
    Selectivity,

    #[token("transport_rate")]
    TransportRate,

    #[token("energy_requirement")]
    EnergyRequirement,

    // Scale identifiers
    #[token("quantum")]
    Quantum,

    #[token("molecular")]
    Molecular,

    #[token("environmental")]
    Environmental,

    #[token("hardware")]
    Hardware,

    #[token("cognitive")]
    Cognitive,

    // Context and loading keywords
    #[token("load_sequence")]
    LoadSequence,

    #[token("load_molecules")]
    LoadMolecules,

    #[token("context")]
    Context,

    #[token("region")]
    Region,

    #[token("focus")]
    Focus,

    #[token("wavelength_range")]
    WavelengthRange,

    #[token("wavelength_scan")]
    WavelengthScan,

    #[token("sensitivity")]
    Sensitivity,

    #[token("specificity")]
    Specificity,

    #[token("amplification")]
    Amplification,

    #[token("duration")]
    Duration,

    #[token("size")]
    Size,

    #[token("diversity")]
    Diversity,

    #[token("var")]
    Var,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("sources")]
    SourcesDecl,

    #[token("given")]
    Given,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("considering")]
    Considering,

    #[token("item")]
    Item,

    #[token("in")]
    In,

    #[token("return")]
    Return,

    #[token("ensure")]
    Ensure,
    
    #[token("research")]
    Research,
    
    #[token("apply")]
    Apply,
    
    #[token("to_all")]
    ToAll,

    #[token("allow")]
    Allow,

    #[token("cause")]
    Cause,

    #[token("motion")]
    Motion,

    // Operators
    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Multiply,

    #[token("/")]
    Divide,

    #[token("|")]
    Pipe,
    
    #[token("|>")]
    PipeForward,

    #[token("=>")]
    Arrow,

    #[token("=")]
    Assign,

    #[token("==")]
    Equal,

    #[token("!=")]
    NotEqual,

    #[token("<")]
    LessThan,

    #[token(">")]
    GreaterThan,

    #[token("<=")]
    LessThanEqual,

    #[token(">=")]
    GreaterThanEqual,

    #[token("&&")]
    And,

    #[token("||")]
    Or,

    #[token("!")]
    Not,

    // Delimiters
    #[token("(")]
    LeftParen,

    #[token(")")]
    RightParen,

    #[token("{")]
    LeftBrace,

    #[token("}")]
    RightBrace,

    #[token("[")]
    LeftBracket,

    #[token("]")]
    RightBracket,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token(";")]
    Semicolon,

    #[token(".")]
    Dot,

    // Complex tokens
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Identifier,

    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLiteral,

    #[regex(r"[0-9]+(\.[0-9]+)?")]
    NumberLiteral,

    // Comments and whitespace
    #[regex(r"//[^\n]*", logos::skip)]
    Comment,

    #[regex(r"[ \t\n\r]+", logos::skip)]
    Whitespace,

    // Error token (without the error attribute as Logos 0.13+ doesn't require it)
    Error,

    // Bene Gesserit masterclass keywords
    #[token("success_framework")]
    SuccessFramework,

    #[token("primary_threshold")]
    PrimaryThreshold,

    #[token("secondary_threshold")]
    SecondaryThreshold,

    #[token("safety_threshold")]
    SafetyThreshold,

    #[token("evidence_quality_modulation")]
    EvidenceQualityModulation,

    #[token("uncertainty_penalty")]
    UncertaintyPenalty,

    #[token("fda_guidance_compliance")]
    FdaGuidanceCompliance,

    #[token("ema_scientific_advice_integration")]
    EmaScientificAdviceIntegration,

    #[token("biological_computer")]
    BiologicalComputer,

    #[token("atp_budget")]
    AtpBudget,

    #[token("time_horizon")]
    TimeHorizon,

    #[token("quantum_targets")]
    QuantumTargets,

    #[token("oscillatory_dynamics")]
    OscillatoryDynamics,

    #[token("atp_available")]
    AtpAvailable,

    #[token("quantum_coherence")]
    QuantumCoherence,

    #[token("quantum_enhanced")]
    QuantumEnhanced,

    #[token("quantum_mechanical")]
    QuantumMechanical,

    #[token("quantum_tunneling")]
    QuantumTunneling,

    #[token("enabled")]
    Enabled,

    #[token("biological_maxwells_demon")]
    BiologicalMaxwellsDemon,

    #[token("input_patterns")]
    InputPatterns,

    #[token("recognition_threshold")]
    RecognitionThreshold,

    #[token("catalysis_efficiency")]
    CatalysisEfficiency,

    #[token("optimize")]
    Optimize,

    #[token("atp_efficiency")]
    AtpEfficiency,

    #[token("track")]
    Track,

    #[token("oscillation_endpoints")]
    OscillationEndpoints,

    #[token("quantum_fidelity")]
    QuantumFidelity,

    #[token("calculate")]
    Calculate,

    #[token("information_catalysis_efficiency")]
    InformationCatalysisEfficiency,

    // Multi-source evidence integration
    #[token("molecular_sources")]
    MolecularSources,

    #[token("clinical_sources")]
    ClinicalSources,

    #[token("real_world_sources")]
    RealWorldSources,

    #[token("omics_sources")]
    OmicsSources,

    #[token("protein_structures")]
    ProteinStructures,

    #[token("molecular_dynamics")]
    MolecularDynamics,

    #[token("binding_affinity")]
    BindingAffinity,

    #[token("cellular_assays")]
    CellularAssays,

    #[token("animal_models")]
    AnimalModels,

    #[token("phase1_data")]
    Phase1Data,

    #[token("phase2_data")]
    Phase2Data,

    #[token("biomarker_data")]
    BiomarkerData,

    #[token("cognitive_assessments")]
    CognitiveAssessments,

    #[token("electronic_health_records")]
    ElectronicHealthRecords,

    #[token("insurance_claims")]
    InsuranceClaims,

    #[token("patient_registries")]
    PatientRegistries,

    #[token("wearable_data")]
    WearableData,

    #[token("gwas_data")]
    GwasData,

    #[token("transcriptomics")]
    Transcriptomics,

    #[token("proteomics")]
    Proteomics,

    #[token("metabolomics")]
    Metabolomics,

    // Data processing pipeline
    #[token("data_processing")]
    DataProcessing,

    #[token("quality_control")]
    QualityControl,

    #[token("missing_data_threshold")]
    MissingDataThreshold,

    #[token("adaptive_threshold")]
    AdaptiveThreshold,

    #[token("outlier_detection")]
    OutlierDetection,

    #[token("isolation_forest")]
    IsolationForest,

    #[token("contamination")]
    Contamination,

    #[token("batch_effect_correction")]
    BatchEffectCorrection,

    #[token("combat_seq")]
    CombatSeq,

    #[token("technical_replicate_correlation")]
    TechnicalReplicateCorrelation,

    #[token("harmonization")]
    Harmonization,

    #[token("unit_standardization")]
    UnitStandardization,

    #[token("si_units_conversion")]
    SiUnitsConversion,

    #[token("temporal_alignment")]
    TemporalAlignment,

    #[token("time_series_synchronization")]
    TimeSeriesSynchronization,

    #[token("population_stratification")]
    PopulationStratification,

    #[token("ancestry_matching")]
    AncestryMatching,

    #[token("covariate_adjustment")]
    CovariateAdjustment,

    #[token("propensity_score_matching")]
    PropensityScoreMatching,

    #[token("feature_engineering")]
    FeatureEngineering,

    #[token("molecular_descriptors")]
    MolecularDescriptors,

    #[token("rdkit_descriptors")]
    RdkitDescriptors,

    #[token("custom_descriptors")]
    CustomDescriptors,

    #[token("clinical_composite_scores")]
    ClinicalCompositeScores,

    #[token("principal_component_analysis")]
    PrincipalComponentAnalysis,

    #[token("time_series_features")]
    TimeSeriesFeatures,

    #[token("tsfresh_extraction")]
    TsfreshExtraction,

    #[token("network_features")]
    NetworkFeatures,

    #[token("protein_interaction_centrality")]
    ProteinInteractionCentrality,

    // Pattern analysis
    #[token("pattern_analysis")]
    PatternAnalysis,

    #[token("molecular_patterns")]
    MolecularPatterns,

    #[token("binding_pose_clustering")]
    BindingPoseClustering,

    #[token("dbscan")]
    Dbscan,

    #[token("eps")]
    Eps,

    #[token("min_samples")]
    MinSamples,

    #[token("pharmacophore_identification")]
    PharmacophoreIdentification,

    #[token("shape_based_clustering")]
    ShapeBasedClustering,

    #[token("admet_pattern_detection")]
    AdmetPatternDetection,

    #[token("random_forest_feature_importance")]
    RandomForestFeatureImportance,

    #[token("clinical_patterns")]
    ClinicalPatterns,

    #[token("responder_phenotyping")]
    ResponderPhenotyping,

    #[token("gaussian_mixture_models")]
    GaussianMixtureModels,

    #[token("n_components")]
    NComponents,

    #[token("disease_progression_trajectories")]
    DiseaseProgressionTrajectories,

    #[token("latent_class_growth_modeling")]
    LatentClassGrowthModeling,

    #[token("adverse_event_clustering")]
    AdverseEventClustering,

    #[token("network_analysis")]
    NetworkAnalysis,

    #[token("omics_integration")]
    OmicsIntegration,

    #[token("multi_block_pls")]
    MultiBlockPls,

    #[token("integrate_omics_blocks")]
    IntegrateOmicsBlocks,

    #[token("network_medicine_analysis")]
    NetworkMedicineAnalysis,

    #[token("disease_module_identification")]
    DiseaseModuleIdentification,

    #[token("pathway_enrichment")]
    PathwayEnrichment,

    #[token("hypergeometric_test_with_fdr")]
    HypergeometricTestWithFdr,

    // Advanced hypothesis testing
    #[token("confidence_interval")]
    ConfidenceInterval,

    #[token("bootstrap_confidence_interval")]
    BootstrapConfidenceInterval,

    #[token("n_bootstrap")]
    NBootstrap,

    #[token("lower_bound")]
    LowerBound,

    #[token("bbb_permeability")]
    BbbPermeability,

    #[token("efflux_ratio")]
    EffluxRatio,

    #[token("ensemble_prediction")]
    EnsemblePrediction,

    #[token("ensemble_vote")]
    EnsembleVote,

    #[token("random_forest_prediction")]
    RandomForestPrediction,

    #[token("svm_prediction")]
    SvmPrediction,

    #[token("neural_network_prediction")]
    NeuralNetworkPrediction,

    #[token("ensemble_agreement")]
    EnsembleAgreement,

    #[token("adas_cog_change")]
    AdasCogChange,

    #[token("p_value")]
    PValue,

    #[token("effect_size")]
    EffectSize,

    #[token("cohens_d")]
    CohensD,

    #[token("treatment_group")]
    TreatmentGroup,

    #[token("placebo_group")]
    PlaceboGroup,

    #[token("number_needed_to_treat")]
    NumberNeededToTreat,

    #[token("calculate_nnt")]
    CalculateNnt,

    #[token("response_rate")]
    ResponseRate,

    #[token("clinical_significance")]
    ClinicalSignificance,

    #[token("meaningful")]
    Meaningful,

    #[token("modest")]
    Modest,

    #[token("csf_tau_reduction")]
    CsfTauReduction,

    #[token("plasma_neurofilament_stable")]
    PlasmaNeurofilamentStable,

    #[token("longitudinal_model")]
    LongitudinalModel,

    #[token("mixed_effects_model")]
    MixedEffectsModel,

    #[token("fixed_effects")]
    FixedEffects,

    #[token("treatment")]
    Treatment,

    #[token("time")]
    Time,

    #[token("treatment_x_time")]
    TreatmentXTime,

    #[token("random_effects")]
    RandomEffects,

    #[token("patient_intercept")]
    PatientIntercept,

    #[token("patient_slope")]
    PatientSlope,

    #[token("treatment_effect")]
    TreatmentEffect,

    // Spatiotemporal analysis
    #[token("spatiotemporal_analysis")]
    SpatiotemporalAnalysis,

    #[token("spatial_modeling")]
    SpatialModeling,

    #[token("local_adaptation")]
    LocalAdaptation,

    #[token("isolation_by_distance_modeling")]
    IsolationByDistanceModeling,

    #[token("environmental_gradients")]
    EnvironmentalGradients,

    #[token("gradient_forest_analysis")]
    GradientForestAnalysis,

    #[token("population_structure")]
    PopulationStructure,

    #[token("spatial_principal_components")]
    SpatialPrincipalComponents,

    #[token("migration_patterns")]
    MigrationPatterns,

    #[token("gravity_model_migration")]
    GravityModelMigration,

    #[token("temporal_modeling")]
    TemporalModeling,

    #[token("evolutionary_trajectories")]
    EvolutionaryTrajectories,

    #[token("coalescent_simulation")]
    CoalescentSimulation,

    #[token("selection_dynamics")]
    SelectionDynamics,

    #[token("forward_simulation")]
    ForwardSimulation,

    #[token("demographic_inference")]
    DemographicInference,

    #[token("composite_likelihood")]
    CompositeLikelihood,

    #[token("cultural_evolution")]
    CulturalEvolution,

    #[token("dual_inheritance_modeling")]
    DualInheritanceModeling,

    #[token("association_analysis")]
    AssociationAnalysis,

    #[token("environmental_gwas")]
    EnvironmentalGwas,

    #[token("genome_environment_association")]
    GenomeEnvironmentAssociation,

    #[token("polygenic_adaptation")]
    PolygenicAdaptation,

    #[token("polygenic_score_evolution")]
    PolygenicScoreEvolution,

    #[token("balancing_selection")]
    BalancingSelection,

    #[token("tajimas_d_analysis")]
    TajimasDAnalysis,

    #[token("introgression_analysis")]
    IntrogressionAnalysis,

    #[token("admixture_mapping")]
    AdmixtureMapping,

    // Imhotep Framework: Revolutionary Self-Aware Neural Networks
    #[token("neural_consciousness")]
    NeuralConsciousness,

    #[token("session_name")]
    SessionName,

    #[token("consciousness_level")]
    ConsciousnessLevel,

    #[token("self_awareness")]
    SelfAwareness,

    #[token("metacognitive_monitoring")]
    MetacognitiveMonitoring,

    #[token("create_bmd_neuron")]
    CreateBmdNeuron,

    #[token("activation")]
    Activation,

    #[token("metacognitive_depth")]
    MetacognitiveDepth,

    #[token("subsystem")]
    Subsystem,

    #[token("question")]
    Question,

    #[token("depth")]
    Depth,

    #[token("precision")]
    Precision,

    #[token("consciousness_gated")]
    ConsciousnessGated,

    #[token("standards")]
    Standards,

    #[token("efficiency")]
    Efficiency,

    #[token("thoroughness")]
    Thoroughness,

    // Four-File System Neural Subsystems
    #[token("DecisionTrailLogger")]
    DecisionTrailLogger,

    #[token("MetacognitiveMonitor")]
    MetacognitiveMonitor,

    #[token("ReasoningChainTracker")]
    ReasoningChainTracker,

    #[token("SystemStateTracker")]
    SystemStateTracker,

    #[token("ThoughtQualityAssessor")]
    ThoughtQualityAssessor,

    #[token("KnowledgeNetworkManager")]
    KnowledgeNetworkManager,

    #[token("KnowledgeStateAuditor")]
    KnowledgeStateAuditor,

    #[token("SelfReflectionMonitor")]
    SelfReflectionMonitor,

    // Neural Connection Types
    #[token("connect_pattern")]
    ConnectPattern,

    #[token("ConsciousnessGated")]
    ConsciousnessGatedConnection,

    #[token("Excitatory")]
    Excitatory,

    #[token("Modulatory")]
    Modulatory,

    #[token("QuantumEntangled")]
    QuantumEntangled,

    // Self-Awareness Configuration
    #[token("configure_self_awareness")]
    ConfigureSelfAwareness,

    #[token("self_reflection_threshold")]
    SelfReflectionThreshold,

    #[token("thought_quality_standards")]
    ThoughtQualityStandards,

    #[token("knowledge_audit_frequency")]
    KnowledgeAuditFrequency,

    #[token("reasoning_chain_logging")]
    ReasoningChainLogging,

    #[token("decision_trail_persistence")]
    DecisionTrailPersistence,

    // Self-Aware Processing Operations
    #[token("activate_self_awareness")]
    ActivateSelfAwareness,

    #[token("get_metacognitive_state")]
    GetMetacognitiveState,

    #[token("current_thought_focus")]
    CurrentThoughtFocus,

    #[token("self_awareness_level")]
    SelfAwarenessLevel,

    #[token("process_with_metacognitive_monitoring")]
    ProcessWithMetacognitiveMonitoring,

    #[token("processing_steps")]
    ProcessingSteps,

    #[token("assess_reasoning_quality")]
    AssessReasoningQuality,

    #[token("overall_quality")]
    OverallQuality,

    #[token("enhance_metacognitive_monitoring")]
    EnhanceMetacognitiveMonitoring,

    #[token("reprocess_with_enhanced_awareness")]
    ReprocessWithEnhancedAwareness,

    // Scientific Self-Aware Reasoning
    #[token("begin_metacognitive_reasoning")]
    BeginMetacognitiveReasoning,

    #[token("analyze_with_metacognitive_oversight")]
    AnalyzeWithMetacognitiveOversight,

    #[token("analysis_type")]
    AnalysisType,

    #[token("get_current_reasoning_state")]
    GetCurrentReasoningState,

    #[token("focus")]
    Focus,

    #[token("statistical_quality")]
    StatisticalQuality,

    #[token("interpret_with_self_awareness")]
    InterpretWithSelfAwareness,

    #[token("interpretation_context")]
    InterpretationContext,

    #[token("uncertainty_tracking")]
    UncertaintyTracking,

    #[token("assess_biological_reasoning")]
    AssessBiologicalReasoning,

    #[token("uncertainties")]
    Uncertainties,

    #[token("analyze_pathways_with_metacognition")]
    AnalyzePathwaysWithMetacognition,

    #[token("metabolites")]
    Metabolites,

    #[token("self_reflection")]
    SelfReflection,

    #[token("knowledge_gap_detection")]
    KnowledgeGapDetection,

    #[token("identify_knowledge_gaps")]
    IdentifyKnowledgeGaps,

    #[token("significant_metabolites")]
    SignificantMetabolites,

    #[token("reasoning_quality")]
    ReasoningQuality,

    #[token("knowledge_gaps")]
    KnowledgeGaps,

    #[token("metacognitive_state")]
    MetacognitiveState,

    // Consciousness vs Self-Awareness Comparison
    #[token("demonstrate_self_awareness_vs_consciousness")]
    DemonstrateSelfAwarenessVsConsciousness,

    #[token("activate_consciousness")]
    ActivateConsciousness,

    #[token("analyze_metabolomics")]
    AnalyzeMetabolomics,

    #[token("conclusion")]
    Conclusion,

    #[token("confidence")]
    Confidence,

    #[token("analyze_with_metacognition")]
    AnalyzeWithMetacognition,

    #[token("reasoning_chain")]
    ReasoningChain,

    #[token("thought_quality_assessment")]
    ThoughtQualityAssessment,

    #[token("uncertainties_identified")]
    UncertaintiesIdentified,

    #[token("knowledge_gaps_identified")]
    KnowledgeGapsIdentified,

    #[token("decision_history")]
    DecisionHistory,

    #[token("decision")]
    Decision,

    #[token("reasoning")]
    Reasoning,

    #[token("external_knowledge_used")]
    ExternalKnowledgeUsed,

    #[token("metacognitive_insights")]
    MetacognitiveInsights,

    // Self-Aware Processing Methods
    #[token("noise_reduction_with_reasoning_tracking")]
    NoiseReductionWithReasoningTracking,

    #[token("peak_detection_with_uncertainty_assessment")]
    PeakDetectionWithUncertaintyAssessment,

    #[token("compound_identification_with_confidence_logging")]
    CompoundIdentificationWithConfidenceLogging,

    #[token("differential_metabolomics")]
    DifferentialMetabolomics,

    #[token("metabolic_pathways_diabetes")]
    MetabolicPathwaysDiabetes,

    // Knowledge Gap Analysis
    #[token("domain")]
    Domain,

    #[token("impact_level")]
    ImpactLevel,

    #[token("description")]
    Description,

    #[token("impact_on_conclusions")]
    ImpactOnConclusions,

    // Revolutionary Self-Awareness Achievement
    #[token("self_aware_system")]
    SelfAwareSystem,

    #[token("analysis_results")]
    AnalysisResults,

    #[token("consciousness_comparison")]
    ConsciousnessComparison,

    #[token("explicit_reasoning_chain_tracking")]
    ExplicitReasoningChainTracking,

    #[token("real_time_thought_quality_assessment")]
    RealTimeThoughtQualityAssessment,

    #[token("uncertainty_acknowledgment_and_quantification")]
    UncertaintyAcknowledgmentAndQuantification,

    #[token("knowledge_gap_identification")]
    KnowledgeGapIdentification,

    #[token("metacognitive_decision_logging")]
    MetacognitiveDecisionLogging,

    #[token("self_reflection_on_reasoning_quality")]
    SelfReflectionOnReasoningQuality,
}

/// Token represents a token with its type and span (location in source)
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub lexeme: String,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::FunctionDecl => write!(f, "funxn"),
            TokenKind::ProjectDecl => write!(f, "project"),
            TokenKind::Proposition => write!(f, "proposition"),
            TokenKind::Evidence => write!(f, "evidence"),
            TokenKind::Pattern => write!(f, "pattern"),
            TokenKind::Support => write!(f, "support"),
            TokenKind::Contradict => write!(f, "contradict"),
            TokenKind::Inconclusive => write!(f, "inconclusive"),
            TokenKind::Requirements => write!(f, "requirements"),
            TokenKind::Signature => write!(f, "signature"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::Meta => write!(f, "meta"),
            TokenKind::DeriveHypotheses => write!(f, "derive_hypotheses"),
            TokenKind::Alternatively => write!(f, "alternatively"),
            TokenKind::With => write!(f, "with"),
            TokenKind::ClassifyAs => write!(f, "classify_as"),
            TokenKind::Confidence => write!(f, "confidence"),
            TokenKind::EmergentBehaviors => write!(f, "emergent_behaviors"),
            TokenKind::Mechanisms => write!(f, "mechanisms"),
            TokenKind::ClinicalRelevance => write!(f, "clinical_relevance"),
            TokenKind::RefinedHypotheses => write!(f, "refined_hypotheses"),
            TokenKind::Recommendations => write!(f, "recommendations"),
            TokenKind::Var => write!(f, "var"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::SourcesDecl => write!(f, "sources"),
            TokenKind::Within => write!(f, "within"),
            TokenKind::Given => write!(f, "given"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::For => write!(f, "for"),
            TokenKind::Each => write!(f, "each"),
            TokenKind::Considering => write!(f, "considering"),
            TokenKind::All => write!(f, "all"),
            TokenKind::These => write!(f, "these"),
            TokenKind::Item => write!(f, "item"),
            TokenKind::In => write!(f, "in"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Ensure => write!(f, "ensure"),
            TokenKind::Research => write!(f, "research"),
            TokenKind::Apply => write!(f, "apply"),
            TokenKind::ToAll => write!(f, "to_all"),
            TokenKind::Allow => write!(f, "allow"),
            TokenKind::Cause => write!(f, "cause"),
            TokenKind::Motion => write!(f, "motion"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Multiply => write!(f, "*"),
            TokenKind::Divide => write!(f, "/"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::PipeForward => write!(f, "|>"),
            TokenKind::Arrow => write!(f, "=>"),
            TokenKind::Assign => write!(f, "="),
            TokenKind::Equal => write!(f, "=="),
            TokenKind::NotEqual => write!(f, "!="),
            TokenKind::LessThan => write!(f, "<"),
            TokenKind::GreaterThan => write!(f, ">"),
            TokenKind::LessThanEqual => write!(f, "<="),
            TokenKind::GreaterThanEqual => write!(f, ">="),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBrace => write!(f, "{{"),
            TokenKind::RightBrace => write!(f, "}}"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Identifier => write!(f, "identifier"),
            TokenKind::StringLiteral => write!(f, "string"),
            TokenKind::NumberLiteral => write!(f, "number"),
            TokenKind::Comment => write!(f, "comment"),
            TokenKind::Whitespace => write!(f, "whitespace"),
            TokenKind::Error => write!(f, "error"),
            TokenKind::Flow => write!(f, "flow"),
            TokenKind::On => write!(f, "on"),
            TokenKind::Catalyze => write!(f, "catalyze"),
            TokenKind::CrossScale => write!(f, "cross_scale"),
            TokenKind::Coordinate => write!(f, "coordinate"),
            TokenKind::Drift => write!(f, "drift"),
            TokenKind::Until => write!(f, "until"),
            TokenKind::Cycle => write!(f, "cycle"),
            TokenKind::Roll => write!(f, "roll"),
            TokenKind::Resolve => write!(f, "resolve"),
            TokenKind::ExecuteInformationCatalysis => write!(f, "execute_information_catalysis"),
            TokenKind::CreatePatternRecognizer => write!(f, "create_pattern_recognizer"),
            TokenKind::CreateActionChanneler => write!(f, "create_action_channeler"),
            TokenKind::CaptureScreenPixels => write!(f, "capture_screen_pixels"),
            TokenKind::Point => write!(f, "point"),
            TokenKind::Content => write!(f, "content"),
            TokenKind::Certainty => write!(f, "certainty"),
            TokenKind::EvidenceStrength => write!(f, "evidence_strength"),
            TokenKind::ContextualRelevance => write!(f, "contextual_relevance"),
            TokenKind::UrgencyFactor => write!(f, "urgency_factor"),
            TokenKind::Quantum => write!(f, "quantum"),
            TokenKind::Molecular => write!(f, "molecular"),
            TokenKind::Environmental => write!(f, "environmental"),
            TokenKind::Hardware => write!(f, "hardware"),
            TokenKind::Cognitive => write!(f, "cognitive"),
            TokenKind::LoadSequence => write!(f, "load_sequence"),
            TokenKind::LoadMolecules => write!(f, "load_molecules"),
            TokenKind::Context => write!(f, "context"),
            TokenKind::Region => write!(f, "region"),
            TokenKind::Focus => write!(f, "focus"),
            TokenKind::WavelengthRange => write!(f, "wavelength_range"),
            TokenKind::WavelengthScan => write!(f, "wavelength_scan"),
            TokenKind::Sensitivity => write!(f, "sensitivity"),
            TokenKind::Specificity => write!(f, "specificity"),
            TokenKind::Amplification => write!(f, "amplification"),
            TokenKind::Duration => write!(f, "duration"),
            TokenKind::Size => write!(f, "size"),
            TokenKind::Diversity => write!(f, "diversity"),
            TokenKind::Funxn => write!(f, "funxn"),
            TokenKind::Metacognitive => write!(f, "metacognitive"),
            TokenKind::Goal => write!(f, "goal"),
            TokenKind::OptimizeUntil => write!(f, "optimize_until"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Catch => write!(f, "catch"),
            TokenKind::Finally => write!(f, "finally"),
            TokenKind::Parallel => write!(f, "parallel"),
            TokenKind::Async => write!(f, "async"),
            TokenKind::Await => write!(f, "await"),
            TokenKind::Import => write!(f, "import"),
            TokenKind::From => write!(f, "from"),
            TokenKind::Otherwise => write!(f, "otherwise"),
            TokenKind::Description => write!(f, "description"),
            TokenKind::SuccessThreshold => write!(f, "success_threshold"),
            TokenKind::Metrics => write!(f, "metrics"),
            TokenKind::Subgoals => write!(f, "subgoals"),
            TokenKind::Weight => write!(f, "weight"),
            TokenKind::Threshold => write!(f, "threshold"),
            TokenKind::Constraints => write!(f, "constraints"),
            TokenKind::RequiresEvidence => write!(f, "requires_evidence"),
            TokenKind::WithWeight => write!(f, "with_weight"),
            TokenKind::Collect => write!(f, "collect"),
            TokenKind::CollectBatch => write!(f, "collect_batch"),
            TokenKind::ValidationRules => write!(f, "validation_rules"),
            TokenKind::ProcessingPipeline => write!(f, "processing_pipeline"),
            TokenKind::TrackReasoning => write!(f, "track_reasoning"),
            TokenKind::EvaluateConfidence => write!(f, "evaluate_confidence"),
            TokenKind::DetectBias => write!(f, "detect_bias"),
            TokenKind::AdaptBehavior => write!(f, "adapt_behavior"),
            TokenKind::AnalyzeDecisionHistory => write!(f, "analyze_decision_history"),
            TokenKind::UpdateDecisionStrategies => write!(f, "update_decision_strategies"),
            TokenKind::IncreaseEvidenceRequirements => write!(f, "increase_evidence_requirements"),
            TokenKind::ReduceComputationalOverhead => write!(f, "reduce_computational_overhead"),
            TokenKind::ProcessMolecule => write!(f, "process_molecule"),
            TokenKind::HarvestEnergy => write!(f, "harvest_energy"),
            TokenKind::ExtractInformation => write!(f, "extract_information"),
            TokenKind::UpdateMembraneState => write!(f, "update_membrane_state"),
            TokenKind::ConfigureMembrane => write!(f, "configure_membrane"),
            TokenKind::CalculateEntropyChange => write!(f, "calculate_entropy_change"),
            TokenKind::GibbsFreeEnergy => write!(f, "gibbs_free_energy"),
            TokenKind::Shannon => write!(f, "shannon"),
            TokenKind::MutualInfo => write!(f, "mutual_info"),
            TokenKind::InfoGain => write!(f, "info_gain"),
            TokenKind::CalculateMw => write!(f, "calculate_mw"),
            TokenKind::CalculateKa => write!(f, "calculate_ka"),
            TokenKind::AnalyzeFlux => write!(f, "analyze_flux"),
            TokenKind::CalculateKcatKm => write!(f, "calculate_kcat_km"),
            TokenKind::QuantumState => write!(f, "quantum_state"),
            TokenKind::Amplitude => write!(f, "amplitude"),
            TokenKind::Phase => write!(f, "phase"),
            TokenKind::CoherenceTime => write!(f, "coherence_time"),
            TokenKind::ApplyHadamard => write!(f, "apply_hadamard"),
            TokenKind::ApplyCnot => write!(f, "apply_cnot"),
            TokenKind::Measure => write!(f, "measure"),
            TokenKind::MeasureEntanglement => write!(f, "measure_entanglement"),
            TokenKind::ParallelExecute => write!(f, "parallel_execute"),
            TokenKind::AwaitAllTasks => write!(f, "await_all_tasks"),
            TokenKind::Temporal => write!(f, "temporal"),
            TokenKind::Spatial => write!(f, "spatial"),
            TokenKind::Oscillatory => write!(f, "oscillatory"),
            TokenKind::Emergent => write!(f, "emergent"),
            TokenKind::Matches => write!(f, "matches"),
            TokenKind::Contains => write!(f, "contains"),
            TokenKind::Temperature => write!(f, "temperature"),
            TokenKind::PhLevel => write!(f, "ph_level"),
            TokenKind::Concentration => write!(f, "concentration"),
            TokenKind::Catalyst => write!(f, "catalyst"),
            TokenKind::MonitorEfficiency => write!(f, "monitor_efficiency"),
            TokenKind::TargetYield => write!(f, "target_yield"),
            TokenKind::AdaptiveOptimization => write!(f, "adaptive_optimization"),
            TokenKind::ProcessingMethod => write!(f, "processing_method"),
            TokenKind::NoiseFiltering => write!(f, "noise_filtering"),
            TokenKind::ConfidenceThreshold => write!(f, "confidence_threshold"),
            TokenKind::Permeability => write!(f, "permeability"),
            TokenKind::Selectivity => write!(f, "selectivity"),
            TokenKind::TransportRate => write!(f, "transport_rate"),
            TokenKind::EnergyRequirement => write!(f, "energy_requirement"),
            TokenKind::SuccessFramework => write!(f, "success_framework"),
            TokenKind::PrimaryThreshold => write!(f, "primary_threshold"),
            TokenKind::SecondaryThreshold => write!(f, "secondary_threshold"),
            TokenKind::SafetyThreshold => write!(f, "safety_threshold"),
            TokenKind::EvidenceQualityModulation => write!(f, "evidence_quality_modulation"),
            TokenKind::UncertaintyPenalty => write!(f, "uncertainty_penalty"),
            TokenKind::FdaGuidanceCompliance => write!(f, "fda_guidance_compliance"),
            TokenKind::EmaScientificAdviceIntegration => write!(f, "ema_scientific_advice_integration"),
            TokenKind::BiologicalComputer => write!(f, "biological_computer"),
            TokenKind::AtpBudget => write!(f, "atp_budget"),
            TokenKind::TimeHorizon => write!(f, "time_horizon"),
            TokenKind::QuantumTargets => write!(f, "quantum_targets"),
            TokenKind::OscillatoryDynamics => write!(f, "oscillatory_dynamics"),
            TokenKind::AtpAvailable => write!(f, "atp_available"),
            TokenKind::QuantumCoherence => write!(f, "quantum_coherence"),
            TokenKind::QuantumEnhanced => write!(f, "quantum_enhanced"),
            TokenKind::QuantumMechanical => write!(f, "quantum_mechanical"),
            TokenKind::QuantumTunneling => write!(f, "quantum_tunneling"),
            TokenKind::Enabled => write!(f, "enabled"),
            TokenKind::BiologicalMaxwellsDemon => write!(f, "biological_maxwells_demon"),
            TokenKind::InputPatterns => write!(f, "input_patterns"),
            TokenKind::RecognitionThreshold => write!(f, "recognition_threshold"),
            TokenKind::CatalysisEfficiency => write!(f, "catalysis_efficiency"),
            TokenKind::Optimize => write!(f, "optimize"),
            TokenKind::AtpEfficiency => write!(f, "atp_efficiency"),
            TokenKind::Track => write!(f, "track"),
            TokenKind::OscillationEndpoints => write!(f, "oscillation_endpoints"),
            TokenKind::QuantumFidelity => write!(f, "quantum_fidelity"),
            TokenKind::Calculate => write!(f, "calculate"),
            TokenKind::InformationCatalysisEfficiency => write!(f, "information_catalysis_efficiency"),
            TokenKind::MolecularSources => write!(f, "molecular_sources"),
            TokenKind::ClinicalSources => write!(f, "clinical_sources"),
            TokenKind::RealWorldSources => write!(f, "real_world_sources"),
            TokenKind::OmicsSources => write!(f, "omics_sources"),
            TokenKind::ProteinStructures => write!(f, "protein_structures"),
            TokenKind::MolecularDynamics => write!(f, "molecular_dynamics"),
            TokenKind::BindingAffinity => write!(f, "binding_affinity"),
            TokenKind::CellularAssays => write!(f, "cellular_assays"),
            TokenKind::AnimalModels => write!(f, "animal_models"),
            TokenKind::Phase1Data => write!(f, "phase1_data"),
            TokenKind::Phase2Data => write!(f, "phase2_data"),
            TokenKind::BiomarkerData => write!(f, "biomarker_data"),
            TokenKind::CognitiveAssessments => write!(f, "cognitive_assessments"),
            TokenKind::ElectronicHealthRecords => write!(f, "electronic_health_records"),
            TokenKind::InsuranceClaims => write!(f, "insurance_claims"),
            TokenKind::PatientRegistries => write!(f, "patient_registries"),
            TokenKind::WearableData => write!(f, "wearable_data"),
            TokenKind::GwasData => write!(f, "gwas_data"),
            TokenKind::Transcriptomics => write!(f, "transcriptomics"),
            TokenKind::Proteomics => write!(f, "proteomics"),
            TokenKind::Metabolomics => write!(f, "metabolomics"),
            TokenKind::DataProcessing => write!(f, "data_processing"),
            TokenKind::QualityControl => write!(f, "quality_control"),
            TokenKind::MissingDataThreshold => write!(f, "missing_data_threshold"),
            TokenKind::AdaptiveThreshold => write!(f, "adaptive_threshold"),
            TokenKind::OutlierDetection => write!(f, "outlier_detection"),
            TokenKind::IsolationForest => write!(f, "isolation_forest"),
            TokenKind::Contamination => write!(f, "contamination"),
            TokenKind::BatchEffectCorrection => write!(f, "batch_effect_correction"),
            TokenKind::CombatSeq => write!(f, "combat_seq"),
            TokenKind::TechnicalReplicateCorrelation => write!(f, "technical_replicate_correlation"),
            TokenKind::Harmonization => write!(f, "harmonization"),
            TokenKind::UnitStandardization => write!(f, "unit_standardization"),
            TokenKind::SiUnitsConversion => write!(f, "si_units_conversion"),
            TokenKind::TemporalAlignment => write!(f, "temporal_alignment"),
            TokenKind::TimeSeriesSynchronization => write!(f, "time_series_synchronization"),
            TokenKind::PopulationStratification => write!(f, "population_stratification"),
            TokenKind::AncestryMatching => write!(f, "ancestry_matching"),
            TokenKind::CovariateAdjustment => write!(f, "covariate_adjustment"),
            TokenKind::PropensityScoreMatching => write!(f, "propensity_score_matching"),
            TokenKind::FeatureEngineering => write!(f, "feature_engineering"),
            TokenKind::MolecularDescriptors => write!(f, "molecular_descriptors"),
            TokenKind::RdkitDescriptors => write!(f, "rdkit_descriptors"),
            TokenKind::CustomDescriptors => write!(f, "custom_descriptors"),
            TokenKind::ClinicalCompositeScores => write!(f, "clinical_composite_scores"),
            TokenKind::PrincipalComponentAnalysis => write!(f, "principal_component_analysis"),
            TokenKind::TimeSeriesFeatures => write!(f, "time_series_features"),
            TokenKind::TsfreshExtraction => write!(f, "tsfresh_extraction"),
            TokenKind::NetworkFeatures => write!(f, "network_features"),
            TokenKind::ProteinInteractionCentrality => write!(f, "protein_interaction_centrality"),
            TokenKind::PatternAnalysis => write!(f, "pattern_analysis"),
            TokenKind::MolecularPatterns => write!(f, "molecular_patterns"),
            TokenKind::BindingPoseClustering => write!(f, "binding_pose_clustering"),
            TokenKind::Dbscan => write!(f, "dbscan"),
            TokenKind::Eps => write!(f, "eps"),
            TokenKind::MinSamples => write!(f, "min_samples"),
            TokenKind::PharmacophoreIdentification => write!(f, "pharmacophore_identification"),
            TokenKind::ShapeBasedClustering => write!(f, "shape_based_clustering"),
            TokenKind::AdmetPatternDetection => write!(f, "admet_pattern_detection"),
            TokenKind::RandomForestFeatureImportance => write!(f, "random_forest_feature_importance"),
            TokenKind::ClinicalPatterns => write!(f, "clinical_patterns"),
            TokenKind::ResponderPhenotyping => write!(f, "responder_phenotyping"),
            TokenKind::GaussianMixtureModels => write!(f, "gaussian_mixture_models"),
            TokenKind::NComponents => write!(f, "n_components"),
            TokenKind::DiseaseProgressionTrajectories => write!(f, "disease_progression_trajectories"),
            TokenKind::LatentClassGrowthModeling => write!(f, "latent_class_growth_modeling"),
            TokenKind::AdverseEventClustering => write!(f, "adverse_event_clustering"),
            TokenKind::NetworkAnalysis => write!(f, "network_analysis"),
            TokenKind::OmicsIntegration => write!(f, "omics_integration"),
            TokenKind::MultiBlockPls => write!(f, "multi_block_pls"),
            TokenKind::IntegrateOmicsBlocks => write!(f, "integrate_omics_blocks"),
            TokenKind::NetworkMedicineAnalysis => write!(f, "network_medicine_analysis"),
            TokenKind::DiseaseModuleIdentification => write!(f, "disease_module_identification"),
            TokenKind::PathwayEnrichment => write!(f, "pathway_enrichment"),
            TokenKind::HypergeometricTestWithFdr => write!(f, "hypergeometric_test_with_fdr"),
            TokenKind::ConfidenceInterval => write!(f, "confidence_interval"),
            TokenKind::BootstrapConfidenceInterval => write!(f, "bootstrap_confidence_interval"),
            TokenKind::NBootstrap => write!(f, "n_bootstrap"),
            TokenKind::LowerBound => write!(f, "lower_bound"),
            TokenKind::BbbPermeability => write!(f, "bbb_permeability"),
            TokenKind::EffluxRatio => write!(f, "efflux_ratio"),
            TokenKind::EnsemblePrediction => write!(f, "ensemble_prediction"),
            TokenKind::EnsembleVote => write!(f, "ensemble_vote"),
            TokenKind::RandomForestPrediction => write!(f, "random_forest_prediction"),
            TokenKind::SvmPrediction => write!(f, "svm_prediction"),
            TokenKind::NeuralNetworkPrediction => write!(f, "neural_network_prediction"),
            TokenKind::EnsembleAgreement => write!(f, "ensemble_agreement"),
            TokenKind::AdasCogChange => write!(f, "adas_cog_change"),
            TokenKind::PValue => write!(f, "p_value"),
            TokenKind::EffectSize => write!(f, "effect_size"),
            TokenKind::CohensD => write!(f, "cohens_d"),
            TokenKind::TreatmentGroup => write!(f, "treatment_group"),
            TokenKind::PlaceboGroup => write!(f, "placebo_group"),
            TokenKind::NumberNeededToTreat => write!(f, "number_needed_to_treat"),
            TokenKind::CalculateNnt => write!(f, "calculate_nnt"),
            TokenKind::ResponseRate => write!(f, "response_rate"),
            TokenKind::ClinicalSignificance => write!(f, "clinical_significance"),
            TokenKind::Meaningful => write!(f, "meaningful"),
            TokenKind::Modest => write!(f, "modest"),
            TokenKind::CsfTauReduction => write!(f, "csf_tau_reduction"),
            TokenKind::PlasmaNeurofilamentStable => write!(f, "plasma_neurofilament_stable"),
            TokenKind::LongitudinalModel => write!(f, "longitudinal_model"),
            TokenKind::MixedEffectsModel => write!(f, "mixed_effects_model"),
            TokenKind::FixedEffects => write!(f, "fixed_effects"),
            TokenKind::Treatment => write!(f, "treatment"),
            TokenKind::Time => write!(f, "time"),
            TokenKind::TreatmentXTime => write!(f, "treatment_x_time"),
            TokenKind::RandomEffects => write!(f, "random_effects"),
            TokenKind::PatientIntercept => write!(f, "patient_intercept"),
            TokenKind::PatientSlope => write!(f, "patient_slope"),
            TokenKind::TreatmentEffect => write!(f, "treatment_effect"),
            TokenKind::SpatiotemporalAnalysis => write!(f, "spatiotemporal_analysis"),
            TokenKind::SpatialModeling => write!(f, "spatial_modeling"),
            TokenKind::LocalAdaptation => write!(f, "local_adaptation"),
            TokenKind::IsolationByDistanceModeling => write!(f, "isolation_by_distance_modeling"),
            TokenKind::EnvironmentalGradients => write!(f, "environmental_gradients"),
            TokenKind::GradientForestAnalysis => write!(f, "gradient_forest_analysis"),
            TokenKind::PopulationStructure => write!(f, "population_structure"),
            TokenKind::SpatialPrincipalComponents => write!(f, "spatial_principal_components"),
            TokenKind::MigrationPatterns => write!(f, "migration_patterns"),
            TokenKind::GravityModelMigration => write!(f, "gravity_model_migration"),
            TokenKind::TemporalModeling => write!(f, "temporal_modeling"),
            TokenKind::EvolutionaryTrajectories => write!(f, "evolutionary_trajectories"),
            TokenKind::CoalescentSimulation => write!(f, "coalescent_simulation"),
            TokenKind::SelectionDynamics => write!(f, "selection_dynamics"),
            TokenKind::ForwardSimulation => write!(f, "forward_simulation"),
            TokenKind::DemographicInference => write!(f, "demographic_inference"),
            TokenKind::CompositeLikelihood => write!(f, "composite_likelihood"),
            TokenKind::CulturalEvolution => write!(f, "cultural_evolution"),
            TokenKind::DualInheritanceModeling => write!(f, "dual_inheritance_modeling"),
            TokenKind::AssociationAnalysis => write!(f, "association_analysis"),
            TokenKind::EnvironmentalGwas => write!(f, "environmental_gwas"),
            TokenKind::GenomeEnvironmentAssociation => write!(f, "genome_environment_association"),
            TokenKind::PolygenicAdaptation => write!(f, "polygenic_adaptation"),
            TokenKind::PolygenicScoreEvolution => write!(f, "polygenic_score_evolution"),
            TokenKind::BalancingSelection => write!(f, "balancing_selection"),
            TokenKind::TajimasDAnalysis => write!(f, "tajimas_d_analysis"),
            TokenKind::IntrogressionAnalysis => write!(f, "introgression_analysis"),
            TokenKind::AdmixtureMapping => write!(f, "admixture_mapping"),
            TokenKind::NeuralConsciousness => write!(f, "neural_consciousness"),
            TokenKind::SessionName => write!(f, "session_name"),
            TokenKind::ConsciousnessLevel => write!(f, "consciousness_level"),
            TokenKind::SelfAwareness => write!(f, "self_awareness"),
            TokenKind::MetacognitiveMonitoring => write!(f, "metacognitive_monitoring"),
            TokenKind::CreateBmdNeuron => write!(f, "create_bmd_neuron"),
            TokenKind::Activation => write!(f, "activation"),
            TokenKind::MetacognitiveDepth => write!(f, "metacognitive_depth"),
            TokenKind::Subsystem => write!(f, "subsystem"),
            TokenKind::Question => write!(f, "question"),
            TokenKind::Depth => write!(f, "depth"),
            TokenKind::Precision => write!(f, "precision"),
            TokenKind::ConsciousnessGated => write!(f, "consciousness_gated"),
            TokenKind::Standards => write!(f, "standards"),
            TokenKind::Efficiency => write!(f, "efficiency"),
            TokenKind::Thoroughness => write!(f, "thoroughness"),
            TokenKind::DecisionTrailLogger => write!(f, "DecisionTrailLogger"),
            TokenKind::MetacognitiveMonitor => write!(f, "MetacognitiveMonitor"),
            TokenKind::ReasoningChainTracker => write!(f, "ReasoningChainTracker"),
            TokenKind::SystemStateTracker => write!(f, "SystemStateTracker"),
            TokenKind::ThoughtQualityAssessor => write!(f, "ThoughtQualityAssessor"),
            TokenKind::KnowledgeNetworkManager => write!(f, "KnowledgeNetworkManager"),
            TokenKind::KnowledgeStateAuditor => write!(f, "KnowledgeStateAuditor"),
            TokenKind::SelfReflectionMonitor => write!(f, "SelfReflectionMonitor"),
            TokenKind::ConnectPattern => write!(f, "connect_pattern"),
            TokenKind::ConsciousnessGatedConnection => write!(f, "ConsciousnessGated"),
            TokenKind::Excitatory => write!(f, "Excitatory"),
            TokenKind::Modulatory => write!(f, "Modulatory"),
            TokenKind::QuantumEntangled => write!(f, "QuantumEntangled"),
            TokenKind::ConfigureSelfAwareness => write!(f, "configure_self_awareness"),
            TokenKind::SelfReflectionThreshold => write!(f, "self_reflection_threshold"),
            TokenKind::ThoughtQualityStandards => write!(f, "thought_quality_standards"),
            TokenKind::KnowledgeAuditFrequency => write!(f, "knowledge_audit_frequency"),
            TokenKind::ReasoningChainLogging => write!(f, "reasoning_chain_logging"),
            TokenKind::DecisionTrailPersistence => write!(f, "decision_trail_persistence"),
            TokenKind::ActivateSelfAwareness => write!(f, "activate_self_awareness"),
            TokenKind::GetMetacognitiveState => write!(f, "get_metacognitive_state"),
            TokenKind::CurrentThoughtFocus => write!(f, "current_thought_focus"),
            TokenKind::SelfAwarenessLevel => write!(f, "self_awareness_level"),
            TokenKind::ProcessWithMetacognitiveMonitoring => write!(f, "process_with_metacognitive_monitoring"),
            TokenKind::ProcessingSteps => write!(f, "processing_steps"),
            TokenKind::AssessReasoningQuality => write!(f, "assess_reasoning_quality"),
            TokenKind::OverallQuality => write!(f, "overall_quality"),
            TokenKind::EnhanceMetacognitiveMonitoring => write!(f, "enhance_metacognitive_monitoring"),
            TokenKind::ReprocessWithEnhancedAwareness => write!(f, "reprocess_with_enhanced_awareness"),
            TokenKind::BeginMetacognitiveReasoning => write!(f, "begin_metacognitive_reasoning"),
            TokenKind::AnalyzeWithMetacognitiveOversight => write!(f, "analyze_with_metacognitive_oversight"),
            TokenKind::AnalysisType => write!(f, "analysis_type"),
            TokenKind::GetCurrentReasoningState => write!(f, "get_current_reasoning_state"),
            TokenKind::Focus => write!(f, "focus"),
            TokenKind::StatisticalQuality => write!(f, "statistical_quality"),
            TokenKind::InterpretWithSelfAwareness => write!(f, "interpret_with_self_awareness"),
            TokenKind::InterpretationContext => write!(f, "interpretation_context"),
            TokenKind::UncertaintyTracking => write!(f, "uncertainty_tracking"),
            TokenKind::AssessBiologicalReasoning => write!(f, "assess_biological_reasoning"),
            TokenKind::Uncertainties => write!(f, "uncertainties"),
            TokenKind::AnalyzePathwaysWithMetacognition => write!(f, "analyze_pathways_with_metacognition"),
            TokenKind::Metabolites => write!(f, "metabolites"),
            TokenKind::SelfReflection => write!(f, "self_reflection"),
            TokenKind::KnowledgeGapDetection => write!(f, "knowledge_gap_detection"),
            TokenKind::IdentifyKnowledgeGaps => write!(f, "identify_knowledge_gaps"),
            TokenKind::SignificantMetabolites => write!(f, "significant_metabolites"),
            TokenKind::ReasoningQuality => write!(f, "reasoning_quality"),
            TokenKind::KnowledgeGaps => write!(f, "knowledge_gaps"),
            TokenKind::MetacognitiveState => write!(f, "metacognitive_state"),
            TokenKind::DemonstrateSelfAwarenessVsConsciousness => write!(f, "demonstrate_self_awareness_vs_consciousness"),
            TokenKind::ActivateConsciousness => write!(f, "activate_consciousness"),
            TokenKind::AnalyzeMetabolomics => write!(f, "analyze_metabolomics"),
            TokenKind::Conclusion => write!(f, "conclusion"),
            TokenKind::Confidence => write!(f, "confidence"),
            TokenKind::AnalyzeWithMetacognition => write!(f, "analyze_with_metacognition"),
            TokenKind::ReasoningChain => write!(f, "reasoning_chain"),
            TokenKind::ThoughtQualityAssessment => write!(f, "thought_quality_assessment"),
            TokenKind::UncertaintiesIdentified => write!(f, "uncertainties_identified"),
            TokenKind::KnowledgeGapsIdentified => write!(f, "knowledge_gaps_identified"),
            TokenKind::DecisionHistory => write!(f, "decision_history"),
            TokenKind::Decision => write!(f, "decision"),
            TokenKind::Reasoning => write!(f, "reasoning"),
            TokenKind::ExternalKnowledgeUsed => write!(f, "external_knowledge_used"),
            TokenKind::MetacognitiveInsights => write!(f, "metacognitive_insights"),
            TokenKind::NoiseReductionWithReasoningTracking => write!(f, "noise_reduction_with_reasoning_tracking"),
            TokenKind::PeakDetectionWithUncertaintyAssessment => write!(f, "peak_detection_with_uncertainty_assessment"),
            TokenKind::CompoundIdentificationWithConfidenceLogging => write!(f, "compound_identification_with_confidence_logging"),
            TokenKind::DifferentialMetabolomics => write!(f, "differential_metabolomics"),
            TokenKind::MetabolicPathwaysDiabetes => write!(f, "metabolic_pathways_diabetes"),
            TokenKind::Domain => write!(f, "domain"),
            TokenKind::ImpactLevel => write!(f, "impact_level"),
            TokenKind::ImpactOnConclusions => write!(f, "impact_on_conclusions"),
            TokenKind::SelfAwareSystem => write!(f, "self_aware_system"),
            TokenKind::AnalysisResults => write!(f, "analysis_results"),
            TokenKind::ConsciousnessComparison => write!(f, "consciousness_comparison"),
            TokenKind::ExplicitReasoningChainTracking => write!(f, "explicit_reasoning_chain_tracking"),
            TokenKind::RealTimeThoughtQualityAssessment => write!(f, "real_time_thought_quality_assessment"),
            TokenKind::UncertaintyAcknowledgmentAndQuantification => write!(f, "uncertainty_acknowledgment_and_quantification"),
            TokenKind::KnowledgeGapIdentification => write!(f, "knowledge_gap_identification"),
            TokenKind::MetacognitiveDecisionLogging => write!(f, "metacognitive_decision_logging"),
            TokenKind::SelfReflectionOnReasoningQuality => write!(f, "self_reflection_on_reasoning_quality"),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// The Lexer struct for tokenizing Turbulance source code
pub struct Lexer<'a> {
    lex: logos::Lexer<'a, TokenKind>,
    source: &'a str,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source code
    pub fn new(source: &'a str) -> Self {
        Self {
            lex: TokenKind::lexer(source),
            source,
        }
    }

    /// Get the current token's lexeme (actual text)
    fn get_lexeme(&self, span: Span) -> String {
        self.source[span.start..span.end].to_string()
    }

    /// Tokenize the entire source code into a vector of tokens
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        while let Some(result) = self.lex.next() {
            let span = self.lex.span();
            let lexeme = self.get_lexeme(span.clone());
            
            let token_kind = match result {
                Ok(kind) => kind,
                Err(_) => TokenKind::Error,
            };
            
            tokens.push(Token {
                kind: token_kind,
                span,
                lexeme,
            });
        }
        
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_keywords() {
        let source = "funxn project sources within given if else for each considering all these item in return ensure research apply to_all allow cause motion";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::FunctionDecl,
            TokenKind::ProjectDecl,
            TokenKind::SourcesDecl,
            TokenKind::Within,
            TokenKind::Given,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::For,
            TokenKind::Each,
            TokenKind::Considering,
            TokenKind::All,
            TokenKind::These,
            TokenKind::Item,
            TokenKind::In,
            TokenKind::Return,
            TokenKind::Ensure,
            TokenKind::Research,
            TokenKind::Apply,
            TokenKind::ToAll,
            TokenKind::Allow,
            TokenKind::Cause,
            TokenKind::Motion,
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_operators() {
        let source = "+ - * / | |> => = == != < > <= >= && || !";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Multiply,
            TokenKind::Divide,
            TokenKind::Pipe,
            TokenKind::PipeForward,
            TokenKind::Arrow,
            TokenKind::Assign,
            TokenKind::Equal,
            TokenKind::NotEqual,
            TokenKind::LessThan,
            TokenKind::GreaterThan,
            TokenKind::LessThanEqual,
            TokenKind::GreaterThanEqual,
            TokenKind::And,
            TokenKind::Or,
            TokenKind::Not,
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_identifiers_and_literals() {
        let source = r#"identifier 123 123.456 "string literal" another_id"#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        let expected_kinds = vec![
            TokenKind::Identifier,  // identifier
            TokenKind::NumberLiteral,  // 123
            TokenKind::NumberLiteral,  // 123.456
            TokenKind::StringLiteral,  // "string literal"
            TokenKind::Identifier,  // another_id
        ];
        
        assert_eq!(tokens.len(), expected_kinds.len());
        
        for (token, expected_kind) in tokens.iter().zip(expected_kinds.iter()) {
            assert_eq!(&token.kind, expected_kind);
        }
    }

    #[test]
    fn test_lexer_complex_code() {
        let source = r#"
            funxn enhance_paragraph(paragraph, domain="general"):
                within paragraph:
                    given contains("technical_term"):
                        research_context(domain)
                        ensure_explanation_follows()
                    given readability_score < 65:
                        simplify_sentences()
                        replace_jargon()
                    return processed
        "#;
        
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        
        // This is just a basic smoke test to ensure we don't panic on valid code
        assert!(tokens.len() > 20);  // Should have plenty of tokens
        
        // Check if specific important tokens are present
        let has_function = tokens.iter().any(|t| t.kind == TokenKind::FunctionDecl);
        let has_within = tokens.iter().any(|t| t.kind == TokenKind::Within);
        let has_given = tokens.iter().any(|t| t.kind == TokenKind::Given);
        let has_return = tokens.iter().any(|t| t.kind == TokenKind::Return);
        
        assert!(has_function && has_within && has_given && has_return);
    }
}
