# 🎯 Complete Usage Guide: Kwasa-Kwasa VSCode Extensions

## 🚀 Installation & Setup

### Prerequisites

Before you start, ensure you have:

-   **VSCode 1.80.0+**
-   **Node.js 18+** (`node --version`)
-   **npm** (`npm --version`)
-   **Kwasa-Kwasa framework** (optional for full integration)

### Quick Install

**On Linux/Mac:**

```bash
chmod +x install.sh
./install.sh
```

**On Windows:**

```cmd
install.bat
```

**Manual Install:**

```bash
# Install vsce if you don't have it
npm install -g vsce

# For each extension directory:
cd turbulance-language-server
npm install && npm run compile
vsce package
code --install-extension turbulance-language-server-0.1.0.vsix

# Repeat for all 5 extensions
```

## 📝 Getting Started - Your First Turbulance Project

### Step 1: Create a New Project

1. **Create a new folder** for your project
2. **Open it in VSCode**
3. **Create a new file** with extension `.turb` or `.trb`

### Step 2: Write Your First Turbulance Code

Create `example.turb`:

```turbulance
// Import scientific analysis functions
import from genomics import calculate_gc_content, analyze_sequence
import from statistics import mean, std, correlation

// Define a research proposition
proposition DNAComplexityHypothesis:
    motion HighGCCorrelatesComplexity("High GC content correlates with sequence complexity")
    motion RegionVariability("Variability exists across genomic regions")

    within sequence_data as sequence:
        given gc_content > 0.6:
            support HighGCCorrelatesComplexity
        given complexity_score > mean(complexity_scores):
            support HighGCCorrelatesComplexity

// Define evidence collection
evidence GenomicSequenceData:
    sources:
        - ncbi_data: DatabaseConnection("NCBI_GenBank")
        - local_sequences: FileSource("./data/sequences.fasta")

    collection:
        frequency: daily
        validation: cross_reference
        quality_threshold: 0.95

    processing:
        - filter_by_length(min_length=1000)
        - remove_ambiguous_bases()
        - calculate_quality_metrics()

// Analysis function
funxn analyze_genomic_complexity(sequences: List[String]) -> ComplexityReport:
    /// Analyze the relationship between GC content and sequence complexity

    item gc_contents = []
    item complexity_scores = []

    within sequences as seq:
        item gc = calculate_gc_content(seq)
        item complexity = calculate_sequence_complexity(seq)

        given gc > 0 and complexity > 0:
            collect gc into gc_contents
            collect complexity into complexity_scores

    item correlation_coef = correlation(gc_contents, complexity_scores)

    return ComplexityReport(
        gc_mean=mean(gc_contents),
        complexity_mean=mean(complexity_scores),
        correlation=correlation_coef,
        sample_size=len(sequences)
    )

// Metacognitive analysis
metacognitive QualityAssurance:
    track:
        - data_quality_metrics
        - statistical_significance
        - potential_confounding_variables

    evaluate:
        - sample_size_adequacy()
        - statistical_power_analysis()
        - bias_detection()

    reflect:
        given correlation_coef > 0.7:
            confidence_boost(DNAComplexityHypothesis, +0.2)
        given sample_size < 100:
            require_additional_data("genomic_sequences")
```

## 🔧 Using Each Extension

### 1. **Turbulance Language Server** - Language Support

**Automatic Features:**

-   ✅ **Syntax highlighting** appears automatically
-   ✅ **Error detection** shows red squiggles for syntax errors
-   ✅ **Auto-completion** triggers when you type

**Commands to try:**

```
Ctrl+Shift+P → "Turbulance: Validate File"
Ctrl+Shift+P → "Turbulance: Run Script"
Ctrl+Shift+P → "Turbulance: Format Document"
```

**Code Snippets:**

-   Type `funxn` + Tab → Function template
-   Type `proposition` + Tab → Proposition template
-   Type `evidence` + Tab → Evidence block template

### 2. **Knowledge Resolution Engine** - Validate Your Claims

**Testing Knowledge Claims:**

1. **Select text** containing a knowledge claim
2. **Right-click** → "Analyze Embedding Differentials"
3. **View results** in the new panel

**Commands to try:**

```
Ctrl+Shift+P → "Knowledge Engine: Analyze Embeddings"
Ctrl+Shift+P → "Knowledge Engine: Show Knowledge Provenance"
Ctrl+Shift+P → "Knowledge Engine: Compare with External Knowledge"
Ctrl+Shift+P → "Knowledge Engine: Show Confidence Map"
```

**What you'll see:**

-   🟢 **Green highlighting** = High confidence claims
-   🟡 **Yellow highlighting** = Medium confidence claims
-   🔴 **Red highlighting** = Low confidence claims
-   ✨ **Purple highlighting** = Novel insights detected

### 3. **Pattern Analysis & Evidence Workbench** - Build & Test

**Build a Proposition:**

1. **Open Command Palette**: `Ctrl+Shift+P`
2. **Run**: "Pattern Workbench: Build Proposition"
3. **Follow the interactive wizard** to build your proposition
4. **The code gets inserted** at your cursor

**Analyze Evidence Networks:**

1. **With your `.turb` file open**
2. **Run**: "Pattern Workbench: Analyze Evidence Network"
3. **View the visual graph** showing evidence relationships

**Commands to try:**

```
Ctrl+Shift+P → "Pattern Workbench: Detect Patterns"
Ctrl+Shift+P → "Pattern Workbench: Test Hypothesis"
Ctrl+Shift+P → "Pattern Workbench: Validate Evidence Quality"
Ctrl+Shift+P → "Pattern Workbench: Show Evidence Graph"
```

### 4. **Metacognitive Orchestration Debugger** - Debug Processing

**Start Debugging:**

1. **Open your `.turb` file**
2. **Press F5** or use "Debug: Start Debugging"
3. **Select "Turbulance Orchestration"** from the dropdown
4. **Set breakpoints** by clicking in the gutter

**Debug Views:**

-   🔄 **Orchestration Flow** - See how processing flows
-   🎯 **Goal Tracking** - Monitor objectives
-   ⚠️ **Intervention Monitor** - See when system intervenes
-   🧠 **Metacognitive State** - View self-aware processing

**Commands to try:**

```
Ctrl+Shift+P → "Orchestration Debugger: Show Orchestration Flow"
Ctrl+Shift+P → "Orchestration Debugger: Track Goals"
Ctrl+Shift+P → "Orchestration Debugger: Monitor Interventions"
```

### 5. **Boundary Detection Studio** - Analyze Text Structure

**Analyze Boundaries:**

1. **Select text** or use entire document
2. **Right-click** → "Analyze Text Boundaries"
3. **View multi-colored highlighting** showing different boundary types

**Compare Algorithms:**

1. **Open Command Palette**: `Ctrl+Shift+P`
2. **Run**: "Boundary Studio: Compare Boundary Algorithms"
3. **See side-by-side comparison** of 5 different detection methods

**Commands to try:**

```
Ctrl+Shift+P → "Boundary Studio: Analyze Boundaries"
Ctrl+Shift+P → "Boundary Studio: Show Boundary Hierarchy"
Ctrl+Shift+P → "Boundary Studio: Compare Algorithms"
Ctrl+Shift+P → "Boundary Studio: Open Studio"
```

## 🎨 Activity Bar Icons

After installation, you'll see new icons in your Activity Bar (left side):

-   🧠 **Knowledge Engine** - Embedding analysis and validation
-   🔬 **Pattern Workbench** - Pattern detection and evidence analysis
-   🐛 **Orchestration Debugger** - Debug views (appears during debugging)
-   📏 **Boundary Studio** - Text boundary analysis

## 🔧 Configuration

### Global Settings

Open **Settings** (`Ctrl+,`) and search for:

**Turbulance Language Server:**

```json
{
    "turbulanceLanguageServer.enableSemanticValidation": true,
    "turbulanceLanguageServer.scientificArgumentValidation": true
}
```

**Knowledge Resolution Engine:**

```json
{
    "knowledgeEngine.embeddingModel": "sentence-transformers/all-MiniLM-L6-v2",
    "knowledgeEngine.confidenceThreshold": 0.8,
    "knowledgeEngine.enableRealTimeAnalysis": true
}
```

**Pattern Workbench:**

```json
{
    "patternWorkbench.patternDetectionSensitivity": 0.7,
    "patternWorkbench.enableRealTimeAnalysis": false
}
```

### Workspace Settings

Create `.vscode/settings.json` in your project:

```json
{
    "turbulanceLanguageServer.enableSemanticValidation": true,
    "knowledgeEngine.externalSources": ["wikipedia", "pubmed", "arxiv"],
    "boundaryStudio.defaultAlgorithm": "hybrid",
    "orchestrationDebugger.traceDepth": 3
}
```

## 🧪 Testing the Extensions

### Test 1: Language Server

```turbulance
// Type this and watch for syntax highlighting and auto-completion
funxn test_function(data: String) -> Result:
    // Try typing 'calculate_' and see completion suggestions
```

### Test 2: Knowledge Engine

```turbulance
// Write a claim and test validation
proposition TestClaim:
    motion DNAReplication("DNA replication is semi-conservative")
    // Select this text and run "Analyze Embeddings"
```

### Test 3: Pattern Workbench

```turbulance
// Create evidence and test pattern detection
evidence TestEvidence:
    sources:
        - scientific_paper: Source("Watson & Crick, 1953")
    // Select all and run "Detect Patterns"
```

## 🚀 Using in Other Projects

### For Scientific Research Projects:

1. **Create `.turb` files** for your hypotheses
2. **Use propositions** to structure your research questions
3. **Define evidence blocks** for your data sources
4. **Use the Knowledge Engine** to validate claims against literature

### For Data Analysis Projects:

1. **Write analysis functions** in Turbulance
2. **Use boundary detection** to structure your text data
3. **Use pattern workbench** to find data patterns
4. **Debug with orchestration debugger** to understand processing

### For Academic Writing:

1. **Structure arguments** as propositions
2. **Validate citations** with the Knowledge Engine
3. **Analyze text boundaries** for better organization
4. **Use evidence networks** to ensure logical flow

## 🔍 Troubleshooting

### Extensions Not Appearing?

1. **Restart VSCode** completely
2. **Check Extensions view** (`Ctrl+Shift+X`) - search for "kwasa"
3. **Enable extensions** if disabled

### Commands Not Working?

1. **Open Command Palette** (`Ctrl+Shift+P`)
2. **Type the extension name** to see available commands
3. **Check if you have a `.turb` file open** (required for some commands)

### No Syntax Highlighting?

1. **Save your file** with `.turb` or `.trb` extension
2. **Check language mode** in bottom-right corner
3. **Manually select "Turbulance"** if needed

### Performance Issues?

1. **Disable real-time analysis** in settings:
    ```json
    {
        "knowledgeEngine.enableRealTimeAnalysis": false,
        "patternWorkbench.enableRealTimeAnalysis": false
    }
    ```

## 🎓 Next Steps

1. **Explore the example files** in the documentation
2. **Try the interactive tutorials** (coming soon)
3. **Join the community** for tips and tricks
4. **Contribute patterns** and custom rules
5. **Integrate with your existing workflows**

---

**🎉 You're now ready to use the full power of semantic text processing with Kwasa-Kwasa VSCode extensions!**
