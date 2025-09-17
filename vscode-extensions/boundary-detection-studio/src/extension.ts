import * as vscode from 'vscode';
import { AlgorithmComparator } from './algorithm-comparator';
import { BoundaryDetectionEngine } from './boundary-detection-engine';
import { BoundaryStudio } from './boundary-studio';
import { BoundaryValidator } from './boundary-validator';
import { BoundaryVisualizer } from './boundary-visualizer';
import { CustomRulesManager } from './custom-rules-manager';
import { BoundaryHierarchyAnalyzer } from './hierarchy-analyzer';
import { ModelTrainer } from './model-trainer';

let boundaryDetectionEngine: BoundaryDetectionEngine;
let boundaryVisualizer: BoundaryVisualizer;
let hierarchyAnalyzer: BoundaryHierarchyAnalyzer;
let algorithmComparator: AlgorithmComparator;
let boundaryValidator: BoundaryValidator;
let customRulesManager: CustomRulesManager;
let boundaryStudio: BoundaryStudio;
let modelTrainer: ModelTrainer;

export function activate(context: vscode.ExtensionContext) {
    console.log('Advanced Boundary Detection Studio is now active!');

    // Initialize components
    initializeComponents(context);

    // Register commands
    const commands = [
        vscode.commands.registerCommand('boundaryStudio.analyzeBoundaries', analyzeBoundaries),
        vscode.commands.registerCommand('boundaryStudio.detectSemanticBoundaries', detectSemanticBoundaries),
        vscode.commands.registerCommand('boundaryStudio.showBoundaryHierarchy', showBoundaryHierarchy),
        vscode.commands.registerCommand('boundaryStudio.compareBoundaryAlgorithms', compareBoundaryAlgorithms),
        vscode.commands.registerCommand('boundaryStudio.validateBoundaries', validateBoundaries),
        vscode.commands.registerCommand('boundaryStudio.createCustomBoundaryRules', createCustomBoundaryRules),
        vscode.commands.registerCommand('boundaryStudio.exportBoundaryData', exportBoundaryData),
        vscode.commands.registerCommand('boundaryStudio.openStudio', openStudio),
        vscode.commands.registerCommand('boundaryStudio.trainBoundaryModel', trainBoundaryModel),
        
        // Additional utility commands
        vscode.commands.registerCommand('boundaryStudio.clearBoundaryVisualization', clearBoundaryVisualization),
        vscode.commands.registerCommand('boundaryStudio.toggleBoundaryType', toggleBoundaryType),
        vscode.commands.registerCommand('boundaryStudio.adjustConfidenceThreshold', adjustConfidenceThreshold),
        vscode.commands.registerCommand('boundaryStudio.batchAnalyzeFiles', batchAnalyzeFiles),
        vscode.commands.registerCommand('boundaryStudio.generateBoundaryReport', generateBoundaryReport)
    ];

    context.subscriptions.push(...commands);

    // Set up boundary decorations
    setupBoundaryDecorations(context);

    // Register tree providers
    registerTreeProviders(context);

    // Set up real-time detection if enabled
    if (vscode.workspace.getConfiguration('boundaryStudio').get('enableRealTimeDetection')) {
        setupRealTimeDetection(context);
    }

    // Set up status bar
    setupStatusBar(context);

    // Initialize studio interface
    boundaryStudio = new BoundaryStudio(context);

    vscode.window.showInformationMessage('🔍 Advanced Boundary Detection Studio activated!');
}

export function deactivate() {
    // Clean up resources
    boundaryDetectionEngine?.dispose();
    boundaryVisualizer?.dispose();
    hierarchyAnalyzer?.dispose();
    algorithmComparator?.dispose();
    boundaryValidator?.dispose();
    customRulesManager?.dispose();
    boundaryStudio?.dispose();
    modelTrainer?.dispose();
}

function initializeComponents(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('boundaryStudio');
    
    boundaryDetectionEngine = new BoundaryDetectionEngine({
        defaultAlgorithm: config.get('defaultAlgorithm'),
        confidenceThreshold: config.get('confidenceThreshold'),
        maxHierarchyDepth: config.get('maxHierarchyDepth'),
        enableMultiLanguageSupport: config.get('enableMultiLanguageSupport')
    });

    boundaryVisualizer = new BoundaryVisualizer(context, {
        visualizationMode: config.get('visualizationMode'),
        enableStatistics: config.get('enableStatistics')
    });

    hierarchyAnalyzer = new BoundaryHierarchyAnalyzer({
        maxDepth: config.get('maxHierarchyDepth')
    });

    algorithmComparator = new AlgorithmComparator();
    boundaryValidator = new BoundaryValidator();

    customRulesManager = new CustomRulesManager({
        customRulesPath: config.get('customRulesPath')
    });

    modelTrainer = new ModelTrainer(context);

    context.subscriptions.push(
        boundaryDetectionEngine,
        boundaryVisualizer,
        hierarchyAnalyzer,
        algorithmComparator,
        boundaryValidator,
        customRulesManager,
        modelTrainer
    );
}

async function analyzeBoundaries() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    const document = editor.document;
    let textToAnalyze: string;
    let analysisScope: string;

    if (!editor.selection.isEmpty) {
        textToAnalyze = document.getText(editor.selection);
        analysisScope = 'Selection';
    } else {
        textToAnalyze = document.getText();
        analysisScope = 'Entire Document';
    }

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Analyzing boundaries in ${analysisScope.toLowerCase()}...`,
        cancellable: true
    }, async (progress, token) => {
        try {
            progress.report({ message: "Detecting paragraph boundaries..." });
            const paragraphBoundaries = await boundaryDetectionEngine.detectParagraphBoundaries(textToAnalyze);

            if (token.isCancellationRequested) return;

            progress.report({ increment: 25, message: "Detecting sentence boundaries..." });
            const sentenceBoundaries = await boundaryDetectionEngine.detectSentenceBoundaries(textToAnalyze);

            if (token.isCancellationRequested) return;

            progress.report({ increment: 25, message: "Detecting clause boundaries..." });
            const clauseBoundaries = await boundaryDetectionEngine.detectClauseBoundaries(textToAnalyze);

            if (token.isCancellationRequested) return;

            progress.report({ increment: 25, message: "Detecting semantic boundaries..." });
            const semanticBoundaries = await boundaryDetectionEngine.detectSemanticBoundaries(textToAnalyze);

            if (token.isCancellationRequested) return;

            progress.report({ increment: 25, message: "Generating visualization..." });

            const boundaryAnalysis = {
                scope: analysisScope,
                paragraphs: paragraphBoundaries,
                sentences: sentenceBoundaries,
                clauses: clauseBoundaries,
                semantic: semanticBoundaries,
                text: textToAnalyze,
                totalBoundaries: paragraphBoundaries.length + sentenceBoundaries.length + clauseBoundaries.length + semanticBoundaries.length
            };

            // Apply boundary visualizations
            await boundaryVisualizer.visualizeBoundaries(editor, boundaryAnalysis);

            // Show analysis results
            showBoundaryAnalysisResults(boundaryAnalysis);

            // Set context for views
            vscode.commands.executeCommand('setContext', 'boundaryStudio:hasAnalysis', true);

        } catch (error) {
            vscode.window.showErrorMessage(`Boundary analysis failed: ${error}`);
        }
    });
}

async function detectSemanticBoundaries() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showWarningMessage('Please select text to analyze semantic boundaries.');
        return;
    }

    try {
        const selectedText = editor.document.getText(selection);
        const semanticBoundaries = await boundaryDetectionEngine.detectSemanticBoundariesAdvanced(selectedText);
        
        // Apply semantic boundary specific visualization
        await boundaryVisualizer.visualizeSemanticBoundaries(editor, semanticBoundaries, selection);
        
        // Show detailed semantic analysis
        showSemanticBoundaryResults(selectedText, semanticBoundaries);

    } catch (error) {
        vscode.window.showErrorMessage(`Semantic boundary detection failed: ${error}`);
    }
}

async function showBoundaryHierarchy() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    try {
        const document = editor.document;
        const text = document.getText();

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Building boundary hierarchy...",
            cancellable: false
        }, async (progress) => {
            progress.report({ message: "Analyzing text structure..." });
            
            const hierarchy = await hierarchyAnalyzer.buildBoundaryHierarchy(text);
            
            progress.report({ increment: 50, message: "Generating hierarchy visualization..." });
            
            showBoundaryHierarchyResults(hierarchy);
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Hierarchy analysis failed: ${error}`);
    }
}

async function compareBoundaryAlgorithms() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    try {
        const document = editor.document;
        const text = document.getText();

        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Comparing boundary detection algorithms...",
            cancellable: false
        }, async (progress) => {
            progress.report({ message: "Running semantic algorithm..." });
            const semanticResults = await algorithmComparator.runSemanticAlgorithm(text);
            
            progress.report({ increment: 20, message: "Running syntactic algorithm..." });
            const syntacticResults = await algorithmComparator.runSyntacticAlgorithm(text);
            
            progress.report({ increment: 20, message: "Running statistical algorithm..." });
            const statisticalResults = await algorithmComparator.runStatisticalAlgorithm(text);
            
            progress.report({ increment: 20, message: "Running neural algorithm..." });
            const neuralResults = await algorithmComparator.runNeuralAlgorithm(text);
            
            progress.report({ increment: 20, message: "Running hybrid algorithm..." });
            const hybridResults = await algorithmComparator.runHybridAlgorithm(text);
            
            progress.report({ increment: 20, message: "Generating comparison..." });
            
            const comparison = {
                semantic: semanticResults,
                syntactic: syntacticResults,
                statistical: statisticalResults,
                neural: neuralResults,
                hybrid: hybridResults
            };

            showAlgorithmComparisonResults(comparison);
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Algorithm comparison failed: ${error}`);
    }
}

async function validateBoundaries() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    try {
        const document = editor.document;
        const text = document.getText();

        // Get current boundary detections
        const boundaries = await boundaryDetectionEngine.detectAllBoundaries(text);
        
        // Validate boundary quality
        const validation = await boundaryValidator.validateBoundaries(text, boundaries);
        
        showBoundaryValidationResults(validation);

    } catch (error) {
        vscode.window.showErrorMessage(`Boundary validation failed: ${error}`);
    }
}

async function createCustomBoundaryRules() {
    try {
        await customRulesManager.openRulesEditor();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open custom rules editor: ${error}`);
    }
}

async function exportBoundaryData() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found.');
        return;
    }

    try {
        const document = editor.document;
        const text = document.getText();

        // Get all boundary data
        const boundaryData = await boundaryDetectionEngine.getAllBoundaryData(text);
        
        // Ask user for export format
        const format = await vscode.window.showQuickPick([
            { label: 'JSON', description: 'Export as JSON file' },
            { label: 'CSV', description: 'Export as CSV file' },
            { label: 'XML', description: 'Export as XML file' }
        ], { placeHolder: 'Select export format' });

        if (format) {
            const saveUri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file(`boundary-analysis.${format.label.toLowerCase()}`),
                filters: {
                    [format.label]: [format.label.toLowerCase()]
                }
            });

            if (saveUri) {
                await boundaryDetectionEngine.exportBoundaryData(boundaryData, saveUri.fsPath, format.label.toLowerCase());
                vscode.window.showInformationMessage(`✅ Boundary data exported to ${saveUri.fsPath}`);
            }
        }

    } catch (error) {
        vscode.window.showErrorMessage(`Export failed: ${error}`);
    }
}

async function openStudio() {
    try {
        await boundaryStudio.show();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open Boundary Studio: ${error}`);
    }
}

async function trainBoundaryModel() {
    try {
        const trainingData = await vscode.window.showOpenDialog({
            canSelectFiles: true,
            canSelectFolders: false,
            canSelectMany: true,
            filters: {
                'Text Files': ['txt', 'md', 'turb', 'trb']
            },
            openLabel: 'Select Training Data'
        });

        if (trainingData && trainingData.length > 0) {
            await modelTrainer.trainModel(trainingData);
            vscode.window.showInformationMessage('🤖 Model training completed!');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Model training failed: ${error}`);
    }
}

async function clearBoundaryVisualization() {
    const editor = vscode.window.activeTextEditor;
    if (editor) {
        boundaryVisualizer.clearVisualization(editor);
    }
}

async function toggleBoundaryType() {
    const boundaryTypes = ['paragraph', 'sentence', 'clause', 'semantic', 'all'];
    const selected = await vscode.window.showQuickPick(boundaryTypes, {
        placeHolder: 'Select boundary type to toggle'
    });

    if (selected) {
        boundaryVisualizer.toggleBoundaryType(selected);
    }
}

async function adjustConfidenceThreshold() {
    const currentThreshold = vscode.workspace.getConfiguration('boundaryStudio').get('confidenceThreshold', 0.7);
    
    const newThreshold = await vscode.window.showInputBox({
        prompt: 'Enter new confidence threshold (0.0 - 1.0)',
        value: currentThreshold.toString(),
        validateInput: (value) => {
            const num = parseFloat(value);
            if (isNaN(num) || num < 0 || num > 1) {
                return 'Please enter a number between 0.0 and 1.0';
            }
            return null;
        }
    });

    if (newThreshold) {
        const threshold = parseFloat(newThreshold);
        await vscode.workspace.getConfiguration('boundaryStudio').update('confidenceThreshold', threshold, vscode.ConfigurationTarget.Workspace);
        boundaryDetectionEngine.updateConfidenceThreshold(threshold);
        vscode.window.showInformationMessage(`✅ Confidence threshold updated to ${threshold}`);
    }
}

async function batchAnalyzeFiles() {
    const files = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: true,
        filters: {
            'Text Files': ['txt', 'md', 'turb', 'trb']
        },
        openLabel: 'Select Files for Batch Analysis'
    });

    if (files && files.length > 0) {
        await vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: `Batch analyzing ${files.length} files...`,
            cancellable: true
        }, async (progress, token) => {
            const results = await boundaryDetectionEngine.batchAnalyzeFiles(files, progress, token);
            showBatchAnalysisResults(results);
        });
    }
}

async function generateBoundaryReport() {
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showWarningMessage('No workspace folder found.');
        return;
    }

    try {
        const report = await boundaryDetectionEngine.generateWorkspaceReport(workspaceFolder.uri.fsPath);
        showBoundaryReportResults(report);
    } catch (error) {
        vscode.window.showErrorMessage(`Report generation failed: ${error}`);
    }
}

function setupBoundaryDecorations(context: vscode.ExtensionContext) {
    // Create decoration types for different boundary types
    const paragraphBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.paragraphBoundary'),
        border: '1px solid',
        borderColor: new vscode.ThemeColor('boundaryStudio.paragraphBoundary'),
        overviewRulerColor: new vscode.ThemeColor('boundaryStudio.paragraphBoundary'),
        overviewRulerLane: vscode.OverviewRulerLane.Right
    });

    const sentenceBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.sentenceBoundary'),
        border: '1px dotted',
        borderColor: new vscode.ThemeColor('boundaryStudio.sentenceBoundary'),
        overviewRulerColor: new vscode.ThemeColor('boundaryStudio.sentenceBoundary'),
        overviewRulerLane: vscode.OverviewRulerLane.Center
    });

    const clauseBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.clauseBoundary'),
        border: '1px dashed',
        borderColor: new vscode.ThemeColor('boundaryStudio.clauseBoundary'),
        overviewRulerColor: new vscode.ThemeColor('boundaryStudio.clauseBoundary'),
        overviewRulerLane: vscode.OverviewRulerLane.Left
    });

    const semanticBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.semanticBoundary'),
        border: '2px solid',
        borderColor: new vscode.ThemeColor('boundaryStudio.semanticBoundary'),
        overviewRulerColor: new vscode.ThemeColor('boundaryStudio.semanticBoundary'),
        overviewRulerLane: vscode.OverviewRulerLane.Full
    });

    const weakBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.weakBoundary'),
        opacity: '0.5'
    });

    const strongBoundaryDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('boundaryStudio.strongBoundary'),
        fontWeight: 'bold'
    });

    context.subscriptions.push(
        paragraphBoundaryDecoration,
        sentenceBoundaryDecoration,
        clauseBoundaryDecoration,
        semanticBoundaryDecoration,
        weakBoundaryDecoration,
        strongBoundaryDecoration
    );

    // Store decoration types for use by the visualizer
    (context as any).boundaryDecorations = {
        paragraph: paragraphBoundaryDecoration,
        sentence: sentenceBoundaryDecoration,
        clause: clauseBoundaryDecoration,
        semantic: semanticBoundaryDecoration,
        weak: weakBoundaryDecoration,
        strong: strongBoundaryDecoration
    };
}

function setupRealTimeDetection(context: vscode.ExtensionContext) {
    let detectionTimeout: NodeJS.Timeout | undefined;

    const onDocumentChange = vscode.workspace.onDidChangeTextDocument(event => {
        // Only process text-based documents
        if (!['turbulance', 'markdown', 'plaintext'].includes(event.document.languageId)) {
            return;
        }

        if (detectionTimeout) {
            clearTimeout(detectionTimeout);
        }

        detectionTimeout = setTimeout(async () => {
            try {
                const editor = vscode.window.activeTextEditor;
                if (editor && editor.document === event.document) {
                    // Perform lightweight boundary detection
                    const quickBoundaries = await boundaryDetectionEngine.quickBoundaryDetection(event.document.getText());
                    await boundaryVisualizer.updateQuickVisualization(editor, quickBoundaries);
                }
            } catch (error) {
                console.error('Real-time boundary detection error:', error);
            }
        }, 1500); // 1.5 second delay for real-time processing
    });

    context.subscriptions.push(onDocumentChange);
}

function registerTreeProviders(context: vscode.ExtensionContext) {
    const hierarchyProvider = new HierarchyTreeProvider();
    const algorithmProvider = new AlgorithmTreeProvider();
    const metricsProvider = new MetricsTreeProvider();
    const rulesProvider = new RulesTreeProvider();

    vscode.window.createTreeView('boundaryStudio.hierarchyView', {
        treeDataProvider: hierarchyProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('boundaryStudio.algorithmView', {
        treeDataProvider: algorithmProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('boundaryStudio.metricsView', {
        treeDataProvider: metricsProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('boundaryStudio.rulesView', {
        treeDataProvider: rulesProvider,
        showCollapseAll: true
    });

    context.subscriptions.push(hierarchyProvider, algorithmProvider, metricsProvider, rulesProvider);
}

function setupStatusBar(context: vscode.ExtensionContext) {
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 30);
    statusBar.text = "$(split-horizontal) Boundary Studio";
    statusBar.tooltip = "Advanced Boundary Detection Studio";
    statusBar.command = "boundaryStudio.analyzeBoundaries";
    statusBar.show();

    context.subscriptions.push(statusBar);

    // Update status bar when analysis is available
    vscode.window.onDidChangeActiveTextEditor(() => {
        if (vscode.window.activeTextEditor) {
            statusBar.show();
        } else {
            statusBar.hide();
        }
    });
}

// Result display functions

function showBoundaryAnalysisResults(analysis: any) {
    const panel = vscode.window.createWebviewPanel(
        'boundaryAnalysis',
        'Boundary Analysis Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getBoundaryAnalysisHtml(analysis);
}

function showSemanticBoundaryResults(text: string, boundaries: any) {
    const panel = vscode.window.createWebviewPanel(
        'semanticBoundaries',
        'Semantic Boundary Analysis',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getSemanticBoundaryHtml(text, boundaries);
}

function showBoundaryHierarchyResults(hierarchy: any) {
    const panel = vscode.window.createWebviewPanel(
        'boundaryHierarchy',
        'Boundary Hierarchy',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getBoundaryHierarchyHtml(hierarchy);
}

function showAlgorithmComparisonResults(comparison: any) {
    const panel = vscode.window.createWebviewPanel(
        'algorithmComparison',
        'Algorithm Comparison Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getAlgorithmComparisonHtml(comparison);
}

function showBoundaryValidationResults(validation: any) {
    const panel = vscode.window.createWebviewPanel(
        'boundaryValidation',
        'Boundary Validation Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getBoundaryValidationHtml(validation);
}

function showBatchAnalysisResults(results: any) {
    const panel = vscode.window.createWebviewPanel(
        'batchAnalysis',
        'Batch Analysis Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getBatchAnalysisHtml(results);
}

function showBoundaryReportResults(report: any) {
    const panel = vscode.window.createWebviewPanel(
        'boundaryReport',
        'Workspace Boundary Report',
        vscode.ViewColumn.One,
        { enableScripts: true }
    );

    panel.webview.html = getBoundaryReportHtml(report);
}

// HTML generation functions

function getBoundaryAnalysisHtml(analysis: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Boundary Analysis Results</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .summary { background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .boundary-type { margin: 15px 0; padding: 10px; border-left: 4px solid #007acc; }
            .boundary-item { background: #f9f9f9; padding: 8px; margin: 5px 0; border-radius: 3px; }
            .confidence-high { border-left-color: #00ff00; }
            .confidence-medium { border-left-color: #ff8000; }
            .confidence-low { border-left-color: #ff0000; }
            #boundaryChart { width: 100%; height: 300px; }
        </style>
    </head>
    <body>
        <h1>🔍 Boundary Analysis Results</h1>
        
        <div class="summary">
            <h2>Analysis Summary</h2>
            <p><strong>Scope:</strong> ${analysis.scope}</p>
            <p><strong>Total Boundaries Detected:</strong> ${analysis.totalBoundaries}</p>
            <p><strong>Paragraph Boundaries:</strong> ${analysis.paragraphs.length}</p>
            <p><strong>Sentence Boundaries:</strong> ${analysis.sentences.length}</p>
            <p><strong>Clause Boundaries:</strong> ${analysis.clauses.length}</p>
            <p><strong>Semantic Boundaries:</strong> ${analysis.semantic.length}</p>
        </div>

        <div id="boundaryChart"></div>

        <div class="boundary-type">
            <h3>Paragraph Boundaries</h3>
            ${analysis.paragraphs.map((boundary: any) => `
                <div class="boundary-item confidence-${boundary.confidence > 0.8 ? 'high' : boundary.confidence > 0.5 ? 'medium' : 'low'}">
                    Position: ${boundary.position} | Confidence: ${boundary.confidence?.toFixed(2) || 'N/A'}
                </div>
            `).join('')}
        </div>

        <div class="boundary-type">
            <h3>Sentence Boundaries</h3>
            ${analysis.sentences.map((boundary: any) => `
                <div class="boundary-item confidence-${boundary.confidence > 0.8 ? 'high' : boundary.confidence > 0.5 ? 'medium' : 'low'}">
                    Position: ${boundary.position} | Confidence: ${boundary.confidence?.toFixed(2) || 'N/A'}
                </div>
            `).join('')}
        </div>

        <div class="boundary-type">
            <h3>Semantic Boundaries</h3>
            ${analysis.semantic.map((boundary: any) => `
                <div class="boundary-item confidence-${boundary.confidence > 0.8 ? 'high' : boundary.confidence > 0.5 ? 'medium' : 'low'}">
                    Position: ${boundary.position} | Type: ${boundary.type || 'Unknown'} | Confidence: ${boundary.confidence?.toFixed(2) || 'N/A'}
                </div>
            `).join('')}
        </div>

        <script>
            // Create boundary distribution chart
            const data = [
                { type: 'Paragraphs', count: ${analysis.paragraphs.length} },
                { type: 'Sentences', count: ${analysis.sentences.length} },
                { type: 'Clauses', count: ${analysis.clauses.length} },
                { type: 'Semantic', count: ${analysis.semantic.length} }
            ];

            const margin = { top: 20, right: 30, bottom: 40, left: 90 };
            const width = 800 - margin.left - margin.right;
            const height = 300 - margin.top - margin.bottom;

            const svg = d3.select('#boundaryChart')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom);

            const g = svg.append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

            const x = d3.scaleLinear()
                .domain([0, d3.max(data, d => d.count)])
                .range([0, width]);

            const y = d3.scaleBand()
                .domain(data.map(d => d.type))
                .range([0, height])
                .padding(0.1);

            g.selectAll('.bar')
                .data(data)
                .enter().append('rect')
                .attr('class', 'bar')
                .attr('x', 0)
                .attr('height', y.bandwidth())
                .attr('y', d => y(d.type))
                .attr('width', d => x(d.count))
                .attr('fill', '#007acc');

            g.append('g')
                .attr('transform', 'translate(0,' + height + ')')
                .call(d3.axisBottom(x));

            g.append('g')
                .call(d3.axisLeft(y));

            g.append('text')
                .attr('transform', 'rotate(-90)')
                .attr('y', 0 - margin.left)
                .attr('x', 0 - (height / 2))
                .attr('dy', '1em')
                .style('text-anchor', 'middle')
                .text('Boundary Types');

            g.append('text')
                .attr('transform', 'translate(' + (width/2) + ' ,' + (height + margin.bottom) + ')')
                .style('text-anchor', 'middle')
                .text('Number of Boundaries');
        </script>
    </body>
    </html>
    `;
}

// Placeholder implementations for other HTML generation functions
function getSemanticBoundaryHtml(text: string, boundaries: any): string {
    return `Semantic boundary HTML would be implemented here...`;
}

function getBoundaryHierarchyHtml(hierarchy: any): string {
    return `Boundary hierarchy HTML would be implemented here...`;
}

function getAlgorithmComparisonHtml(comparison: any): string {
    return `Algorithm comparison HTML would be implemented here...`;
}

function getBoundaryValidationHtml(validation: any): string {
    return `Boundary validation HTML would be implemented here...`;
}

function getBatchAnalysisHtml(results: any): string {
    return `Batch analysis HTML would be implemented here...`;
}

function getBoundaryReportHtml(report: any): string {
    return `Boundary report HTML would be implemented here...`;
}

// Tree provider placeholder classes
class HierarchyTreeProvider implements vscode.TreeDataProvider<any> {
    getTreeItem(element: any): vscode.TreeItem { return element; }
    getChildren(element?: any): Thenable<any[]> { return Promise.resolve([]); }
    dispose() {}
}

class AlgorithmTreeProvider implements vscode.TreeDataProvider<any> {
    getTreeItem(element: any): vscode.TreeItem { return element; }
    getChildren(element?: any): Thenable<any[]> { return Promise.resolve([]); }
    dispose() {}
}

class MetricsTreeProvider implements vscode.TreeDataProvider<any> {
    getTreeItem(element: any): vscode.TreeItem { return element; }
    getChildren(element?: any): Thenable<any[]> { return Promise.resolve([]); }
    dispose() {}
}

class RulesTreeProvider implements vscode.TreeDataProvider<any> {
    getTreeItem(element: any): vscode.TreeItem { return element; }
    getChildren(element?: any): Thenable<any[]> { return Promise.resolve([]); }
    dispose() {}
}
