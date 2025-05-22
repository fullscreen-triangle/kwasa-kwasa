use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwasa_kwasa::text_unit::{TextUnit, processor::TextProcessor};
use kwasa_kwasa::pattern::metacognitive::MetaCognitive;

// Benchmark for basic text unit operations
fn bench_text_unit_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("TextUnit Operations");
    
    // Create a sample text to use in benchmarks
    let sample_text = "This is a sample text for benchmarking purposes. \
                      It contains multiple sentences with varying complexity. \
                      The framework should be able to process this efficiently.";
    
    // Benchmark text unit creation
    group.bench_function("create_text_unit", |b| {
        b.iter(|| {
            let unit = TextUnit::new(black_box(sample_text.to_string()));
            black_box(unit)
        })
    });
    
    // Benchmark text unit splitting
    let text_unit = TextUnit::new(sample_text.to_string());
    group.bench_function("split_into_sentences", |b| {
        b.iter(|| {
            let sentences = black_box(&text_unit).split_by_sentences();
            black_box(sentences)
        })
    });
    
    // Benchmark text unit merging
    let sentences: Vec<TextUnit> = sample_text.split('.')
        .filter(|s| !s.trim().is_empty())
        .map(|s| TextUnit::new(format!("{}.", s)))
        .collect();
    
    group.bench_function("merge_text_units", |b| {
        b.iter(|| {
            let merged = TextUnit::merge(black_box(&sentences));
            black_box(merged)
        })
    });
    
    group.finish();
}

// Benchmark for TextProcessor operations
fn bench_text_processor(c: &mut Criterion) {
    let mut group = c.benchmark_group("TextProcessor");
    
    // Sample text with more content for processing
    let sample_text = "The Kwasa-kwasa framework provides metacognitive text processing capabilities. \
                      It can identify patterns and relationships between concepts. \
                      The framework uses advanced algorithms to extract meaning from unstructured text. \
                      It can also perform operations like summarization, simplification, and formalization. \
                      By analyzing semantic structures, it discovers hidden connections between ideas.";
    
    // Benchmark basic processing
    group.bench_function("process_text", |b| {
        b.iter(|| {
            let mut processor = TextProcessor::new();
            let result = processor.process(black_box(sample_text));
            black_box(result)
        })
    });
    
    // Benchmark pattern extraction
    group.bench_function("extract_patterns", |b| {
        b.iter(|| {
            let mut processor = TextProcessor::new();
            let result = processor.extract_patterns(black_box(sample_text));
            black_box(result)
        })
    });
    
    // Benchmark relationships discovery
    group.bench_function("discover_relationships", |b| {
        b.iter(|| {
            let mut processor = TextProcessor::new();
            let result = processor.discover_relationships(black_box(sample_text));
            black_box(result)
        })
    });
    
    group.finish();
}

// Benchmark for MetaCognitive operations
fn bench_metacognitive(c: &mut Criterion) {
    let mut group = c.benchmark_group("MetaCognitive");
    
    // Create a metacognitive engine and populate it
    let mut meta = MetaCognitive::new();
    
    // Prepare test data
    let concepts = vec![
        ("cloud", "Dark clouds in the sky"),
        ("rain", "Rainfall on the ground"),
        ("wet", "Wet surfaces after rain"),
        ("lightning", "Lightning bolt during storm"),
        ("thunder", "Thunder sound after lightning"),
    ];
    
    // Add nodes to metacognitive engine
    for (id, content) in &concepts {
        meta.add_node(kwasa_kwasa::pattern::metacognitive::MetaNode {
            id: id.to_string(),
            content: content.to_string(),
            confidence: 0.9,
            evidence: vec!["observation".to_string()],
            node_type: kwasa_kwasa::pattern::metacognitive::MetaNodeType::Concept,
            metadata: std::collections::HashMap::new(),
        }).unwrap();
    }
    
    // Add relationships
    let edges = vec![
        ("cloud", "rain", kwasa_kwasa::pattern::metacognitive::MetaEdgeType::Causes),
        ("rain", "wet", kwasa_kwasa::pattern::metacognitive::MetaEdgeType::Causes),
        ("lightning", "thunder", kwasa_kwasa::pattern::metacognitive::MetaEdgeType::Causes),
        ("cloud", "lightning", kwasa_kwasa::pattern::metacognitive::MetaEdgeType::Contains),
    ];
    
    for (src, tgt, edge_type) in &edges {
        meta.add_edge(kwasa_kwasa::pattern::metacognitive::MetaEdge {
            source: src.to_string(),
            target: tgt.to_string(),
            edge_type: edge_type.clone(),
            strength: 0.8,
            metadata: std::collections::HashMap::new(),
        }).unwrap();
    }
    
    // Clone the metacognitive engine for benchmarking
    let meta_clone = meta.clone();
    
    // Benchmark reasoning
    group.bench_function("reasoning", |b| {
        b.iter(|| {
            let result = meta_clone.reason(&["cloud".to_string()]);
            black_box(result)
        })
    });
    
    // Benchmark reflection
    let meta_clone2 = meta.clone();
    group.bench_function("reflection", |b| {
        b.iter(|| {
            let result = meta_clone2.reflect();
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_text_unit_operations,
    bench_text_processor,
    bench_metacognitive
);
criterion_main!(benches); 