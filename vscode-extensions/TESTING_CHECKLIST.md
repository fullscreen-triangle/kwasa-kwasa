# ✅ Extension Testing Checklist

Use this checklist to verify that all 5 extensions are working correctly after installation.

## 📋 Pre-Test Setup

-   [ ] VSCode version 1.80.0+ installed
-   [ ] All 5 extensions installed (check Extensions view: `Ctrl+Shift+X`)
-   [ ] Extensions are enabled (not grayed out)
-   [ ] VSCode has been restarted after installation

## 🔤 1. Turbulance Language Server Extension

### ✅ Basic Language Features

**Open:** `examples/basic-example.turb`

-   [ ] **Syntax Highlighting**: Keywords like `funxn`, `proposition`, `evidence` are colored
-   [ ] **Error Detection**: Add a syntax error (e.g., `funx` instead of `funxn`) - red squiggles appear
-   [ ] **Auto-completion**:
    -   Type `funxn ` and press `Tab` - template expands
    -   Type `calculate_` and see completion suggestions
    -   Type `sentiment_` and see analysis functions
-   [ ] **Hover Documentation**: Hover over `analyze_sentiment_patterns` - documentation appears
-   [ ] **Code Folding**: Click collapse arrow next to `proposition` block

### ✅ Commands

Press `Ctrl+Shift+P` and test these commands:

-   [ ] **"Turbulance: Validate File"** - Shows validation results
-   [ ] **"Turbulance: Format Document"** - Formats the code
-   [ ] **"Turbulance: Run Script"** - Opens terminal (even if kwasa-kwasa not installed)

### ✅ Code Lens

-   [ ] **Proposition Code Lens**: Click "🧪 Analyze Proposition" above proposition blocks
-   [ ] **Evidence Code Lens**: Click "✅ Validate Evidence" above evidence blocks

**Status**: 🟢 Pass / 🔴 Fail

---

## 🧠 2. Knowledge Resolution Engine

### ✅ Activity Bar Icon

-   [ ] **Knowledge Engine icon** appears in Activity Bar (brain/graph icon)
-   [ ] **Clicking icon** opens Knowledge Engine views panel

### ✅ Commands

**Open:** `examples/basic-example.turb`

Press `Ctrl+Shift+P` and test:

-   [ ] **"Knowledge Engine: Analyze Embeddings"** - Shows embedding analysis panel
-   [ ] **"Knowledge Engine: Show Knowledge Provenance"** - Opens provenance visualization
-   [ ] **"Knowledge Engine: Compare with External Knowledge"** - Shows reality comparison
-   [ ] **"Knowledge Engine: Show Confidence Map"** - Applies confidence highlighting to text

### ✅ Context Menu

-   [ ] **Right-click in editor** → See "Knowledge Engine" options in context menu
-   [ ] **Select text and right-click** → "Analyze Embedding Differentials" available

### ✅ Visual Features

Test on the `WellKnownFact` proposition:

-   [ ] **Confidence highlighting** appears on text (green/yellow/red)
-   [ ] **Webview panels** open with visualizations
-   [ ] **Activity bar views** populate with data

**Status**: 🟢 Pass / 🔴 Fail

---

## 🔬 3. Pattern Analysis & Evidence Workbench

### ✅ Activity Bar Icon

-   [ ] **Pattern Workbench icon** appears in Activity Bar
-   [ ] **Clicking icon** opens Pattern Workbench views

### ✅ Interactive Features

**Open:** `examples/basic-example.turb`

-   [ ] **"Pattern Workbench: Build Proposition"** - Opens interactive proposition builder
-   [ ] **"Pattern Workbench: Analyze Evidence Network"** - Shows evidence graph
-   [ ] **"Pattern Workbench: Detect Patterns"** - Highlights detected patterns
-   [ ] **"Pattern Workbench: Test Hypothesis"** - Shows hypothesis testing interface

### ✅ Pattern Detection

Select the `SentimentAccuracyHypothesis` proposition and run pattern detection:

-   [ ] **Pattern highlighting** appears in editor
-   [ ] **Pattern types** are correctly identified (structural, semantic, reasoning)
-   [ ] **Confidence scores** are shown for detected patterns

### ✅ Evidence Analysis

-   [ ] **Evidence network graph** displays relationships
-   [ ] **Evidence quality metrics** are calculated
-   [ ] **Source authority weighting** is shown

**Status**: 🟢 Pass / 🔴 Fail

---

## 🐛 4. Metacognitive Orchestration Debugger

### ✅ Debug Configuration

**Open:** `examples/basic-example.turb`

-   [ ] **Press F5** - Debug configuration dropdown appears
-   [ ] **"Turbulance Orchestration"** option is available
-   [ ] **Launch configuration** starts successfully (even if debug target not available)

### ✅ Debug Views

During a debug session:

-   [ ] **Orchestration Debugger icon** appears in Activity Bar
-   [ ] **Debug views** populate:
    -   Orchestration Flow
    -   Goal Tracking
    -   Intervention Monitor
    -   Context Analysis
    -   Metacognitive State

### ✅ Debug Commands

-   [ ] **"Orchestration Debugger: Show Orchestration Flow"** - Opens flow visualizer
-   [ ] **"Orchestration Debugger: Track Goals"** - Shows goal tracking dashboard
-   [ ] **"Orchestration Debugger: Monitor Interventions"** - Shows intervention timeline
-   [ ] **"Orchestration Debugger: Visualize Metacognition"** - Shows metacognitive trace

### ✅ Breakpoints

-   [ ] **Click in gutter** - Breakpoints can be set on Turbulance lines
-   [ ] **Breakpoint indicators** appear correctly

**Status**: 🟢 Pass / 🔴 Fail

---

## 📏 5. Advanced Boundary Detection Studio

### ✅ Activity Bar Icon

-   [ ] **Boundary Studio icon** appears in Activity Bar (split/hierarchy icon)
-   [ ] **Clicking icon** opens Boundary Studio views

### ✅ Boundary Analysis

**Open:** `examples/scientific-research.turb`

-   [ ] **"Boundary Studio: Analyze Boundaries"** - Shows comprehensive boundary analysis
-   [ ] **Multi-colored highlighting** appears showing different boundary types:
    -   🔵 Paragraph boundaries
    -   🟠 Sentence boundaries
    -   🟢 Clause boundaries
    -   🟣 Semantic boundaries

### ✅ Advanced Features

-   [ ] **"Boundary Studio: Compare Algorithms"** - Shows algorithm comparison panel
-   [ ] **"Boundary Studio: Show Boundary Hierarchy"** - Opens hierarchy tree visualization
-   [ ] **"Boundary Studio: Open Studio"** - Opens main studio interface

### ✅ Real-time Features (if enabled)

Enable in settings: `"boundaryStudio.enableRealTimeDetection": true`

-   [ ] **Type new text** - Boundary detection updates in real-time
-   [ ] **Edit existing text** - Boundary highlighting updates automatically

### ✅ Context Menu

Select text in the scientific example:

-   [ ] **Right-click** → "Boundary Studio" options appear
-   [ ] **"Detect Semantic Boundaries"** works on selected text

**Status**: 🟢 Pass / 🔴 Fail

---

## 🔄 Integration Testing

### ✅ Cross-Extension Features

**Open:** `examples/scientific-research.turb`

1. **Run pattern detection** → **Then run knowledge validation** → **Then analyze boundaries**

    - [ ] All three analyses work together without conflicts
    - [ ] Results complement each other
    - [ ] Performance remains acceptable

2. **Start debugging** → **Use other extensions during debug session**

    - [ ] Other extensions continue working during debugging
    - [ ] Debug views show information from other extensions

3. **Multi-file workflow**:
    - [ ] Open multiple `.turb` files
    - [ ] Extensions work correctly across all open files
    - [ ] Switching between files maintains extension state

### ✅ Performance Testing

-   [ ] **Large file handling**: Open `scientific-research.turb` (600+ lines)
    -   Extensions respond within reasonable time (<5 seconds)
-   [ ] **Real-time features**: Enable all real-time analysis
    -   Typing remains responsive
    -   CPU usage stays reasonable
-   [ ] **Memory usage**: With all extensions active
    -   VSCode memory usage increases moderately (<500MB additional)

**Status**: 🟢 Pass / 🔴 Fail

---

## 🚨 Troubleshooting

### Common Issues and Solutions:

#### ❌ Extensions not appearing

-   Restart VSCode completely
-   Check Extensions panel - ensure extensions are enabled
-   Look for error messages in Developer Console (`Help > Toggle Developer Tools`)

#### ❌ Commands not available

-   Ensure you have a `.turb` or `.trb` file open
-   Check that file language is set to "Turbulance" (bottom-right corner)
-   Try `Ctrl+Shift+P` and type full command name

#### ❌ No syntax highlighting

-   File extension must be `.turb` or `.trb`
-   Check language mode in bottom-right corner
-   Manually select "Turbulance" language if needed

#### ❌ Performance issues

-   Disable real-time analysis in settings
-   Close unused files
-   Reduce pattern detection sensitivity

#### ❌ Webview panels not opening

-   Check if popup blocker is interfering
-   Try different commands to open panels
-   Check VSCode version (requires 1.80.0+)

---

## ✅ Final Verification

**All Extensions Working**: 🟢 Pass / 🔴 Fail

If all tests pass, you have successfully installed and configured the complete Kwasa-Kwasa VSCode extension suite!

**Next Steps:**

1. Try the extensions on your own projects
2. Explore advanced configuration options
3. Experiment with different text types and languages
4. Join the community for tips and best practices

**Issues Found?**

-   Document any failing tests
-   Check the troubleshooting section
-   Report persistent issues on the project repository
