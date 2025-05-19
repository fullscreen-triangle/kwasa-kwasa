# Kwasa-Kwasa Framework Compilation Errors & Solutions

This document catalogs all the compilation errors encountered in the Kwasa-kwasa framework and the solutions implemented to resolve them.

## 1. Unit Trait Implementation Issues

### ✅ Error: No method named `content()` found for reference `&NucleotideSequence` in `src/genomic/high_throughput.rs`
**Solution:** Add import for `Unit` trait from genomic module and change method calls to field access
```rust
// Add at the top of the file
use crate::genomic::Unit;

// Change from
let content = sequence.content();
// To
let content = sequence.content;
```

### ✅ Error: No method named `id()` found for reference `&NucleotideSequence` in `src/genomic/high_throughput.rs`
**Solution:** Change method call to field access
```rust
// Change from
id: UnitId::new(format!("compressed_{}", sequence.id())) 
// To
id: UnitId::new(format!("compressed_{}", sequence.id)) 
```

### ✅ Error: No method named `id()` found for reference `&MassSpectrum` in `src/spectrometry/high_throughput.rs`
**Solution:** Add import for `Unit` trait and change method call to field access
```rust
// Add at the top of the file
use crate::spectrometry::Unit;

// Change from
id: UnitId::new(format!("aligned_{}", spectrum.id())),
// To
id: UnitId::new(format!("aligned_{}", spectrum.id)),
```

### ✅ Error: No method named `content()` found for reference `&NucleotideSequence` in `src/evidence/mod.rs`
**Solution:** Add import for `Unit` trait and change method call to field access
```rust
// Add at the top of the file
use crate::genomic::Unit;

// Change from
let content = String::from_utf8_lossy(sequence.content()).to_string();
// To
let content = String::from_utf8_lossy(&sequence.content).to_string();
```

## 2. Collection Type Issues

### ✅ Error: Cannot build HashMap from iterator over elements of type `(&[u8; 3], u8)` in `src/genomic/mod.rs`
**Solution:** Convert `[u8; 3]` literals to `Vec<u8>` types in the HashMap creation
```rust
// Change from
let codon_table: HashMap<&[u8], u8> = [
    (b"TTT", b'F'), (b"TTC", b'F'), // ... more codons
].iter().cloned().collect();

// To
let codon_table: HashMap<Vec<u8>, u8> = [
    (b"TTT".to_vec(), b'F'), (b"TTC".to_vec(), b'F'), // ... more codons
].iter().cloned().collect();
```

### ✅ Error: The trait bound `f64: std::cmp::Eq` and `f64: Hash` is not satisfied in `src/spectrometry/high_throughput.rs`
**Solution:** Replace `HashMap` with `BTreeMap` for f64 keys, which doesn't require `Eq` or `Hash` traits
```rust
// Add at the top of the file
use std::collections::BTreeMap;

// Change from
let mut result = HashMap::new();
// To
let mut result = BTreeMap::new();
```

### ✅ Error: The trait bound `f64: std::cmp::Eq` and `f64: Hash` is not satisfied in `src/pattern/mod.rs`
**Solution:** Remove `Eq, Hash` derive macros from Pattern struct and add a proper PatternMetadata type
```rust
// Add PatternMetadata type definition
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PatternMetadata {
    /// Source of the pattern
    pub source: Option<String>,
    /// Additional key-value annotations
    pub annotations: HashMap<String, String>,
}

// Change from
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pattern {
    // Fields with f64 values
    significance: f64,
    // ...
}

// To
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    // Fields with f64 values
    pub significance: f64,
    // ...
}
```

## 3. Function Argument Issues

### ✅ Error: Function `Motion::new` takes 2 arguments but 1 argument was supplied in `src/evidence/mod.rs`
**Solution:** Add the missing `TextUnitType` argument
```rust
// Add import
use crate::turbulance::TextUnitType;

// Change from
let motion = Motion::new(content);
// To
let motion = Motion::new(&content, TextUnitType::Paragraph);
```

### ✅ Error: Mismatched types, expected `String` found `&str` in `src/evidence/mod.rs`
**Solution:** Convert `&str` to `String` using `to_string()` method
```rust
// Change from
source: source.clone(),
target: target.clone(),
// To
source: source.to_string(),
target: target.to_string(),
```

## 4. Borrow Checker Issues

### ✅ Error: Cannot borrow `*self` as mutable because it is also borrowed as immutable in `src/turbulance/parser.rs`
**Solution:** Store token before mutable borrow
```rust
// Change from
let equals = self.previous();
let value = Box::new(self.assignment()?);
// ... later ...
return Err(self.error_at_token(&equals, "Invalid assignment target"));

// To
let equals_token = self.previous().clone(); // Clone token to avoid borrow conflict
let value = Box::new(self.assignment()?);
// ... later ...
return Err(self.error_at_token(&equals_token, "Invalid assignment target"));
```

### ✅ Error: Cannot borrow `*registry` as mutable because it is also borrowed as immutable in `src/text_unit/boundary.rs`
**Solution:** Restructure the code to avoid simultaneous borrows
```rust
// Implementation steps:
// 1. Store immutable borrows in local variables before mutable borrowing
// 2. Clone or copy needed data from the registry before making the mutable borrow
// 3. Use separate scope blocks to isolate borrows

// Change from
let existing = registry.get(key);
if existing.is_some() {
    registry.insert(key, updated_value);
}

// To
let should_insert = {
    let existing = registry.get(key);
    existing.is_some()
};

if should_insert {
    registry.insert(key, updated_value);
}
```

### ✅ Error: Borrow of moved value `content` in `src/chemistry/mod.rs`
**Solution:** Clone the value before moving it
```rust
// Change from
let content = content.into();
Self {
    content,
    smiles: String::from_utf8_lossy(&content).to_string(), // Error: content moved
    // ...
}

// To
let content_vec = content.into();
let content_clone = content_vec.clone();
Self {
    content: content_vec,
    smiles: String::from_utf8_lossy(&content_clone).to_string(), // OK
    // ...
}
```

### ❌ Error: Cannot borrow `*node` as mutable more than once at a time in `src/text_unit/hierarchy.rs`
**Solution:** Restructure the recursive algorithm to avoid multiple mutable borrows
```rust
// Implementation steps:
// 1. Change the recursive approach to an iterative approach using a stack
// 2. Create a separate data structure to track changes
// 3. Process nodes one at a time

// Change from
fn process_node(&mut self, node: &mut Node) {
    // First recursive call borrowing node mutably
    self.process_children(&mut node.children);
    
    // Second recursive call borrowing node mutably again
    self.update_node_attributes(&mut node);
}

// To
fn process_node(&mut self, node: &mut Node) {
    // Store node IDs to process instead of recursive calls
    let mut node_ids_to_process = vec![node.id];
    
    // Process each node once
    while let Some(id) = node_ids_to_process.pop() {
        let current_node = self.get_node_by_id_mut(id);
        // Process this node...
        
        // Add children to the processing queue
        node_ids_to_process.extend(current_node.children.iter().map(|c| c.id));
    }
}
```

### ✅ Error: Cannot borrow as mutable, as it is not declared as mutable in `src/orchestrator/stream.rs`
**Solution:** Make parameter immutable in trait definition and add local mutability
```rust
// Change trait definition
async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData>;

// Change implementations
async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
    // ...
    tokio::spawn(async move {
        let mut input = input; // Make mutable locally
        // ...
    });
    // ...
}
```

### ❌ Error: Borrowed data escapes outside of method in `src/orchestrator/stream.rs`
**Solution:** Restructure the closure to avoid capturing `self` reference
```rust
// Implementation steps:
// 1. Extract all needed data from self before the closure
// 2. Clone or copy values that need to be moved into the closure
// 3. Use Arc to share ownership when necessary

// Change from
async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
    let (tx, rx) = channel(DEFAULT_BUFFER_SIZE);
    
    tokio::spawn(async move {
        // Error: closure captures &self but is returned from function
        while let Some(data) = input.recv().await {
            let result = self.transform(data); // Borrowed reference escapes here
            tx.send(result).await.unwrap();
        }
    });
    
    rx
}

// To
async fn process(&self, input: Receiver<StreamData>) -> Receiver<StreamData> {
    let (tx, rx) = channel(DEFAULT_BUFFER_SIZE);
    
    // Clone or extract any needed data from self
    let transformer = self.transformer.clone(); // Assuming transformer is Arc<T> or similar
    
    tokio::spawn(async move {
        while let Some(data) = input.recv().await {
            let result = transformer.transform(data); // Use cloned data instead of self
            tx.send(result).await.unwrap();
        }
    });
    
    rx
}
```

### ❌ Error: Cannot borrow `self.context`, `self.conn`, etc. as mutable, as they are behind `&` references
**Solution:** Change method signatures to take `&mut self`
```rust
// Implementation steps:
// 1. Update method signatures to use &mut self
// 2. Update all call sites to pass mutable references
// 3. Ensure thread safety if needed with proper locking mechanisms

// Change from
pub fn update_context_with_unit(&self, unit_id: usize) {
    self.context.add_keyword(keyword); // Error
}
// To
pub fn update_context_with_unit(&mut self, unit_id: usize) {
    self.context.add_keyword(keyword); // OK
}
```

## 5. Pattern Matching Issues

### ✅ Error: Non-exhaustive patterns: `&Node::ForEach { .. }` and others not covered in `src/turbulance/parser.rs`
**Solution:** Add all missing patterns to the match expression
```rust
// Implementation steps:
// 1. Identify all enum variants from the Node definition
// 2. Add cases for all missing variants
// 3. Add a catch-all pattern if new variants might be added in the future

// Change from
match self {
    Node::Number(_, span) => Some(*span),
    Node::String(_, span) => Some(*span),
    // Some patterns...
}

// To
match self {
    Node::Number(_, span) => Some(*span),
    Node::String(_, span) => Some(*span),
    // Original patterns...
    Node::ForEach { span, .. } => Some(*span),
    Node::ConsideringAll { span, .. } => Some(*span),
    Node::ConsideringThese { span, .. } => Some(*span),
    Node::ConsideringItem { span, .. } => Some(*span),
    Node::Motion { span, .. } => Some(*span),
    Node::Allow { span, .. } => Some(*span),
    Node::Cause { span, .. } => Some(*span),
    Node::PipeExpr { span, .. } => Some(*span),
    Node::ArrayExpr { span, .. } => Some(*span),
    Node::ObjectExpr { span, .. } => Some(*span),
    Node::PropertyAccess { span, .. } => Some(*span),
    Node::FormatString { span, .. } => Some(*span),
    Node::Range { span, .. } => Some(*span),
    Node::ListComprehension { span, .. } => Some(*span),
    Node::TypeAnnotation { span, .. } => Some(*span),
    Node::ImportStmt { span, .. } => Some(*span),
    Node::ExportStmt { span, .. } => Some(*span),
    // Add other missing patterns
    _ => None, // Catch-all for any future additions
}
```

## 6. Content Movement Issues

### ✅ Error: Borrow of moved value: `content` in `src/chemistry/mod.rs`
**Solution:** Clone content before using it
```rust
// Change from
let content = content.into();
Self {
    content,
    smiles: String::from_utf8_lossy(&content).to_string(), // Error: content moved
    // ...
}

// To
let content_vec = content.into();
let content_clone = content_vec.clone();
Self {
    content: content_vec,
    smiles: String::from_utf8_lossy(&content_clone).to_string(), // OK
    // ...
}
```

## Next Steps for Framework Completion

After addressing these errors, the following steps are required to complete the framework development:

### 1. ❌ Testing Framework Functionality
**Implementation Steps:**
- Create unit tests for each core module (turbulance, genomic, spectrometry, etc.)
- Develop integration tests for cross-module functionality
- Create end-to-end tests using the example scripts
- Test the compiler pipeline from parsing to execution
- Validate error handling and recovery mechanisms
- Develop specific test cases for edge cases identified during development

### 2. ❌ Error Handling Improvements
**Implementation Steps:**
- Add comprehensive error types for each module
- Implement proper error propagation between components
- Add context information to error messages
- Create recovery mechanisms for common error scenarios
- Add detailed debugging information in verbose mode
- Implement graceful degradation for non-critical errors

### 3. ❌ Performance Optimization
**Implementation Steps:**
- Profile code to identify bottlenecks
- Optimize memory usage in core data structures
- Implement caching for expensive operations
- Improve parallel processing capabilities
- Optimize parser for faster compilation
- Reduce unnecessary cloning and copying of data
- Implement lazy evaluation where appropriate

### 4. ❌ Documentation Expansion
**Implementation Steps:**
- Complete API documentation for all public interfaces
- Create user guides for each module
- Add comprehensive examples for common use cases
- Document the language syntax with a formal specification
- Create tutorials for beginners
- Add inline code examples
- Generate documentation website or PDF manual

### 5. ❌ Feature Completion
**Implementation Steps:**
- Implement remaining language features from the specification
- Complete the metacognitive reasoning engine
- Finish the pattern recognition algorithms
- Implement the advanced text processing capabilities
- Complete the scientific data analysis features
- Add visualization components
- Implement external API integrations

### 6. ❌ Usability Enhancements
**Implementation Steps:**
- Create a command-line interface with helpful commands
- Add auto-completion features
- Implement syntax highlighting definitions
- Create editor plugins for common IDEs
- Add interactive debugging capabilities
- Implement a REPL environment for experimentation
- Create project templates for common use cases

### 7. ❌ Distribution and Packaging
**Implementation Steps:**
- Create proper package structure for distribution
- Set up CI/CD pipeline for automated builds
- Implement versioning strategy
- Create installer for different platforms
- Set up package repository
- Create Docker images for containerized usage
- Write installation and upgrade documentation

These steps will bring the Kwasa-kwasa framework to a fully functional state where users can reliably use the text processing capabilities and the advanced features for scientific data analysis.
