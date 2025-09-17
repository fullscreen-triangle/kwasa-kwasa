import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

let client: LanguageClient;
let kwasaProcess: ChildProcess | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('Turbulance Language Server extension is now active!');

    // Start the language server
    const serverOptions: ServerOptions = {
        run: { command: getKwasaCommand(), args: ['language-server'], transport: TransportKind.stdio },
        debug: { command: getKwasaCommand(), args: ['language-server', '--debug'], transport: TransportKind.stdio }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'turbulance' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{turb,trb}')
        },
        initializationOptions: {
            enableSemanticValidation: vscode.workspace.getConfiguration('turbulanceLanguageServer').get('enableSemanticValidation', true),
            scientificArgumentValidation: vscode.workspace.getConfiguration('turbulanceLanguageServer').get('scientificArgumentValidation', true)
        }
    };

    client = new LanguageClient(
        'turbulanceLanguageServer',
        'Turbulance Language Server',
        serverOptions,
        clientOptions
    );

    // Register commands
    const disposables = [
        vscode.commands.registerCommand('turbulance.validate', validateCurrentFile),
        vscode.commands.registerCommand('turbulance.runScript', runCurrentScript),
        vscode.commands.registerCommand('turbulance.formatDocument', formatCurrentDocument),
        
        // Advanced semantic commands
        vscode.commands.registerCommand('turbulance.analyzeProposition', analyzeProposition),
        vscode.commands.registerCommand('turbulance.validateEvidence', validateEvidence),
        vscode.commands.registerCommand('turbulance.checkMetacognitive', checkMetacognitive),
        
        // Text unit manipulation commands
        vscode.commands.registerCommand('turbulance.divideText', divideText),
        vscode.commands.registerCommand('turbulance.multiplyText', multiplyText),
        vscode.commands.registerCommand('turbulance.showTextHierarchy', showTextHierarchy)
    ];

    context.subscriptions.push(...disposables);

    // Start the client
    client.start();

    // Create status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = "$(check) Turbulance Ready";
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Register document formatting provider
    const formattingProvider = vscode.languages.registerDocumentFormattingProvider('turbulance', {
        provideDocumentFormattingEdits(document: vscode.TextDocument): vscode.ProviderResult<vscode.TextEdit[]> {
            return formatTurbulanceDocument(document);
        }
    });
    context.subscriptions.push(formattingProvider);

    // Register hover provider for semantic information
    const hoverProvider = vscode.languages.registerHoverProvider('turbulance', {
        provideHover(document: vscode.TextDocument, position: vscode.Position): vscode.ProviderResult<vscode.Hover> {
            return provideSemanticHover(document, position);
        }
    });
    context.subscriptions.push(hoverProvider);

    // Register code lens provider for propositions and evidence
    const codeLensProvider = vscode.languages.registerCodeLensProvider('turbulance', {
        provideCodeLenses(document: vscode.TextDocument): vscode.ProviderResult<vscode.CodeLens[]> {
            return provideSemanticCodeLenses(document);
        }
    });
    context.subscriptions.push(codeLensProvider);
}

export function deactivate(): Thenable<void> | undefined {
    if (kwasaProcess) {
        kwasaProcess.kill();
    }
    if (!client) {
        return undefined;
    }
    return client.stop();
}

function getKwasaCommand(): string {
    // Try to find kwasa-kwasa executable
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (workspaceFolder) {
        const localKwasa = path.join(workspaceFolder.uri.fsPath, 'target', 'release', 'kwasa-kwasa');
        if (require('fs').existsSync(localKwasa)) {
            return localKwasa;
        }
    }
    
    // Fallback to global installation
    return 'kwasa';
}

async function validateCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'turbulance') {
        vscode.window.showWarningMessage('Please open a Turbulance file (.turb or .trb) to validate.');
        return;
    }

    const document = editor.document;
    await document.save();

    // Show validation progress
    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Validating Turbulance file...",
        cancellable: false
    }, async (progress) => {
        try {
            const kwasaCommand = getKwasaCommand();
            const result = await executeKwasaCommand([
                'validate', 
                document.fileName, 
                '--report-file', 
                document.fileName + '.validation.json',
                '--output', 
                'json'
            ]);

            if (result.success) {
                vscode.window.showInformationMessage('✅ Turbulance file validation completed successfully!');
                
                // Parse and show detailed results
                const validationResults = JSON.parse(result.output);
                showValidationResults(validationResults);
            } else {
                vscode.window.showErrorMessage(`❌ Validation failed: ${result.error}`);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Validation error: ${error}`);
        }
    });
}

async function runCurrentScript() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'turbulance') {
        vscode.window.showWarningMessage('Please open a Turbulance file to run.');
        return;
    }

    const document = editor.document;
    await document.save();

    const terminal = vscode.window.createTerminal('Turbulance Execution');
    terminal.show();
    
    const kwasaCommand = getKwasaCommand();
    terminal.sendText(`${kwasaCommand} run "${document.fileName}"`);
}

async function formatCurrentDocument() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'turbulance') {
        return;
    }

    const document = editor.document;
    const edits = await formatTurbulanceDocument(document);
    
    if (edits && edits.length > 0) {
        const workspaceEdit = new vscode.WorkspaceEdit();
        workspaceEdit.set(document.uri, edits);
        await vscode.workspace.applyEdit(workspaceEdit);
    }
}

async function formatTurbulanceDocument(document: vscode.TextDocument): Promise<vscode.TextEdit[]> {
    try {
        const kwasaCommand = getKwasaCommand();
        const result = await executeKwasaCommand(['format', '--stdin'], document.getText());
        
        if (result.success && result.output !== document.getText()) {
            const fullRange = new vscode.Range(
                document.positionAt(0),
                document.positionAt(document.getText().length)
            );
            return [vscode.TextEdit.replace(fullRange, result.output)];
        }
    } catch (error) {
        console.error('Formatting error:', error);
    }
    
    return [];
}

async function provideSemanticHover(document: vscode.TextDocument, position: vscode.Position): Promise<vscode.Hover | undefined> {
    const wordRange = document.getWordRangeAtPosition(position);
    if (!wordRange) {
        return undefined;
    }

    const word = document.getText(wordRange);
    const line = document.lineAt(position.line);
    
    // Provide context-sensitive hover information
    if (line.text.includes('proposition') && word.match(/^[A-Z]/)) {
        return new vscode.Hover([
            `**Proposition**: ${word}`,
            'A testable hypothesis or claim that can be supported or refuted by evidence.',
            '',
            'Propositions contain motions that represent specific testable aspects.'
        ]);
    }
    
    if (line.text.includes('motion') && word.match(/^[A-Z]/)) {
        return new vscode.Hover([
            `**Motion**: ${word}`,
            'A specific testable component of a proposition.',
            '',
            'Motions can be supported or refuted through evidence within the proposition block.'
        ]);
    }
    
    if (line.text.includes('evidence') && word.match(/^[A-Z]/)) {
        return new vscode.Hover([
            `**Evidence Block**: ${word}`,
            'A structured collection of evidence sources, collection methods, and processing steps.',
            '',
            'Evidence blocks define how data is gathered and validated for hypothesis testing.'
        ]);
    }

    // Check if word is a built-in function
    const builtInFunctions = getBuiltInFunctions();
    const funcInfo = builtInFunctions.get(word);
    if (funcInfo) {
        return new vscode.Hover([
            `**${word}**(${funcInfo.params}) → ${funcInfo.returnType}`,
            '',
            funcInfo.description,
            '',
            `**Domain**: ${funcInfo.domain}`
        ]);
    }

    return undefined;
}

async function provideSemanticCodeLenses(document: vscode.TextDocument): Promise<vscode.CodeLens[]> {
    const codeLenses: vscode.CodeLens[] = [];
    const text = document.getText();
    const lines = text.split('\n');

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Add code lens for propositions
        const propMatch = line.match(/^\s*proposition\s+(\w+):/);
        if (propMatch) {
            const range = new vscode.Range(i, 0, i, line.length);
            codeLenses.push(
                new vscode.CodeLens(range, {
                    title: "🧪 Analyze Proposition",
                    command: "turbulance.analyzeProposition",
                    arguments: [propMatch[1], document.uri]
                }),
                new vscode.CodeLens(range, {
                    title: "📊 Show Evidence Graph",
                    command: "turbulance.showEvidenceGraph",
                    arguments: [propMatch[1], document.uri]
                })
            );
        }

        // Add code lens for evidence blocks
        const evidenceMatch = line.match(/^\s*evidence\s+(\w+):/);
        if (evidenceMatch) {
            const range = new vscode.Range(i, 0, i, line.length);
            codeLenses.push(
                new vscode.CodeLens(range, {
                    title: "✅ Validate Evidence",
                    command: "turbulance.validateEvidence",
                    arguments: [evidenceMatch[1], document.uri]
                }),
                new vscode.CodeLens(range, {
                    title: "🔗 Check Sources",
                    command: "turbulance.checkEvidenceSources",
                    arguments: [evidenceMatch[1], document.uri]
                })
            );
        }

        // Add code lens for functions with text processing
        const funcMatch = line.match(/^\s*funxn\s+(\w+)\s*\(/);
        if (funcMatch && (line.includes('text') || line.includes('Text'))) {
            const range = new vscode.Range(i, 0, i, line.length);
            codeLenses.push(
                new vscode.CodeLens(range, {
                    title: "📝 Show Text Hierarchy",
                    command: "turbulance.showTextHierarchy",
                    arguments: [funcMatch[1], document.uri]
                })
            );
        }
    }

    return codeLenses;
}

async function analyzeProposition(propositionName: string, documentUri: vscode.Uri) {
    try {
        const kwasaCommand = getKwasaCommand();
        const result = await executeKwasaCommand([
            'analyze-proposition',
            '--file', documentUri.fsPath,
            '--proposition', propositionName,
            '--output', 'json'
        ]);

        if (result.success) {
            const analysis = JSON.parse(result.output);
            showPropositionAnalysis(propositionName, analysis);
        } else {
            vscode.window.showErrorMessage(`Failed to analyze proposition: ${result.error}`);
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Proposition analysis error: ${error}`);
    }
}

async function validateEvidence(evidenceName: string, documentUri: vscode.Uri) {
    try {
        const kwasaCommand = getKwasaCommand();
        const result = await executeKwasaCommand([
            'validate-evidence',
            '--file', documentUri.fsPath,
            '--evidence', evidenceName,
            '--output', 'json'
        ]);

        if (result.success) {
            const validation = JSON.parse(result.output);
            showEvidenceValidation(evidenceName, validation);
        } else {
            vscode.window.showErrorMessage(`Failed to validate evidence: ${result.error}`);
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Evidence validation error: ${error}`);
    }
}

async function checkMetacognitive() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'turbulance') {
        return;
    }

    // Implementation for metacognitive analysis checking
    vscode.window.showInformationMessage('🧠 Metacognitive analysis checking not yet implemented');
}

async function divideText() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);
    
    if (!selectedText) {
        vscode.window.showWarningMessage('Please select text to divide.');
        return;
    }

    const result = await vscode.window.showQuickPick([
        'Semantic boundaries',
        'Paragraph boundaries', 
        'Sentence boundaries',
        'Clause boundaries'
    ], { placeHolder: 'Select division type' });

    if (result) {
        // Call kwasa-kwasa to divide the text
        try {
            const kwasaCommand = getKwasaCommand();
            const divisionResult = await executeKwasaCommand([
                'divide-text',
                '--type', result.toLowerCase().replace(' boundaries', ''),
                '--text', selectedText
            ]);

            if (divisionResult.success) {
                const divisions = JSON.parse(divisionResult.output);
                showTextDivisions(divisions);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Text division error: ${error}`);
        }
    }
}

async function multiplyText() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);
    
    if (!selectedText) {
        vscode.window.showWarningMessage('Please select text to multiply.');
        return;
    }

    // Implementation for text multiplication (semantic expansion)
    vscode.window.showInformationMessage('✖️ Text multiplication not yet implemented');
}

async function showTextHierarchy() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    try {
        const kwasaCommand = getKwasaCommand();
        const result = await executeKwasaCommand([
            'text-hierarchy',
            '--file', editor.document.fileName,
            '--output', 'json'
        ]);

        if (result.success) {
            const hierarchy = JSON.parse(result.output);
            showTextHierarchyView(hierarchy);
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Text hierarchy error: ${error}`);
    }
}

function showValidationResults(results: any) {
    const panel = vscode.window.createWebviewPanel(
        'turbulanceValidation',
        'Turbulance Validation Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getValidationResultsHtml(results);
}

function showPropositionAnalysis(name: string, analysis: any) {
    const panel = vscode.window.createWebviewPanel(
        'propositionAnalysis',
        `Proposition Analysis: ${name}`,
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getPropositionAnalysisHtml(name, analysis);
}

function showEvidenceValidation(name: string, validation: any) {
    const panel = vscode.window.createWebviewPanel(
        'evidenceValidation',
        `Evidence Validation: ${name}`,
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getEvidenceValidationHtml(name, validation);
}

function showTextDivisions(divisions: any) {
    const panel = vscode.window.createWebviewPanel(
        'textDivisions',
        'Text Division Results',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getTextDivisionsHtml(divisions);
}

function showTextHierarchyView(hierarchy: any) {
    const panel = vscode.window.createWebviewPanel(
        'textHierarchy',
        'Text Hierarchy',
        vscode.ViewColumn.Beside,
        { enableScripts: true }
    );

    panel.webview.html = getTextHierarchyHtml(hierarchy);
}

async function executeKwasaCommand(args: string[], stdin?: string): Promise<{ success: boolean, output: string, error: string }> {
    return new Promise((resolve) => {
        const command = getKwasaCommand();
        const process = spawn(command, args);
        
        let output = '';
        let error = '';

        process.stdout.on('data', (data) => {
            output += data.toString();
        });

        process.stderr.on('data', (data) => {
            error += data.toString();
        });

        process.on('close', (code) => {
            resolve({
                success: code === 0,
                output: output,
                error: error
            });
        });

        if (stdin) {
            process.stdin.write(stdin);
            process.stdin.end();
        }
    });
}

function getBuiltInFunctions(): Map<string, {params: string, returnType: string, description: string, domain: string}> {
    const functions = new Map();
    
    // Text processing functions
    functions.set('tokenize', {
        params: 'text: String',
        returnType: 'List[String]',
        description: 'Split text into tokens for analysis',
        domain: 'Text Processing'
    });
    
    functions.set('extract_keywords', {
        params: 'text: String, count?: Integer',
        returnType: 'List[String]',
        description: 'Extract important keywords from text',
        domain: 'Text Processing'
    });
    
    functions.set('sentiment_analysis', {
        params: 'text: String',
        returnType: 'SentimentResult',
        description: 'Analyze emotional sentiment of text',
        domain: 'Text Processing'
    });

    // Scientific functions
    functions.set('calculate_gc_content', {
        params: 'sequence: String',
        returnType: 'Float',
        description: 'Calculate GC content percentage of DNA sequence',
        domain: 'Genomics'
    });

    functions.set('analyze_spectrum', {
        params: 'spectrum_data: List[Float]',
        returnType: 'SpectrumAnalysis',
        description: 'Analyze spectroscopy data for peaks and patterns',
        domain: 'Spectrometry'
    });

    return functions;
}

function getValidationResultsHtml(results: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .success { color: green; }
            .warning { color: orange; }
            .error { color: red; }
            .section { margin-bottom: 20px; border: 1px solid #ccc; padding: 15px; }
            .fallacy { background-color: #fff3cd; padding: 10px; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>🔬 Turbulance Validation Results</h1>
        
        <div class="section">
            <h2>Overall Validity</h2>
            <p class="${results.overall_validity === 'Valid' ? 'success' : 'error'}">
                ${results.overall_validity}
            </p>
        </div>

        <div class="section">
            <h2>Logical Analysis</h2>
            ${results.logical_fallacies?.map((fallacy: any) => `
                <div class="fallacy">
                    <strong>${fallacy.severity}:</strong> ${fallacy.description}
                    <br><small>Location: ${fallacy.location}</small>
                </div>
            `).join('') || '<p>No logical issues detected.</p>'}
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            ${results.recommendations?.map((rec: string) => `
                <li>${rec}</li>
            `).join('') || '<p>No specific recommendations.</p>'}
        </div>
    </body>
    </html>
    `;
}

function getPropositionAnalysisHtml(name: string, analysis: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .motion { background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-left: 4px solid #0066cc; }
            .support { color: green; }
            .refute { color: red; }
        </style>
    </head>
    <body>
        <h1>🧪 Proposition Analysis: ${name}</h1>
        
        <div class="section">
            <h2>Motions</h2>
            ${analysis.motions?.map((motion: any) => `
                <div class="motion">
                    <h3>${motion.name}</h3>
                    <p>${motion.description}</p>
                    <p>Status: <span class="${motion.status}">${motion.status}</span></p>
                    <p>Confidence: ${motion.confidence}</p>
                </div>
            `).join('') || '<p>No motions found.</p>'}
        </div>
        
        <div class="section">
            <h2>Evidence Summary</h2>
            <p>Supporting Evidence: ${analysis.supporting_evidence_count || 0}</p>
            <p>Refuting Evidence: ${analysis.refuting_evidence_count || 0}</p>
            <p>Overall Confidence: ${analysis.overall_confidence || 'Unknown'}</p>
        </div>
    </body>
    </html>
    `;
}

function getEvidenceValidationHtml(name: string, validation: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .valid { color: green; }
            .invalid { color: red; }
            .source { background-color: #f9f9f9; padding: 10px; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>✅ Evidence Validation: ${name}</h1>
        
        <div class="section">
            <h2>Sources</h2>
            ${validation.sources?.map((source: any) => `
                <div class="source">
                    <strong>${source.name}</strong>: 
                    <span class="${source.valid ? 'valid' : 'invalid'}">
                        ${source.valid ? 'Valid' : 'Invalid'}
                    </span>
                    <br><small>${source.description}</small>
                    ${source.issues ? `<br><small style="color: red;">Issues: ${source.issues}</small>` : ''}
                </div>
            `).join('') || '<p>No sources found.</p>'}
        </div>
        
        <div class="section">
            <h2>Validation Summary</h2>
            <p>Quality Score: ${validation.quality_score || 'Unknown'}</p>
            <p>Reliability: ${validation.reliability || 'Unknown'}</p>
            <p>Completeness: ${validation.completeness || 'Unknown'}</p>
        </div>
    </body>
    </html>
    `;
}

function getTextDivisionsHtml(divisions: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .division { background-color: #f0f0f0; padding: 10px; margin: 5px 0; border-left: 3px solid #007acc; }
            .confidence { font-weight: bold; color: #007acc; }
        </style>
    </head>
    <body>
        <h1>📝 Text Division Results</h1>
        
        ${divisions.map((div: any, index: number) => `
            <div class="division">
                <h3>Division ${index + 1}</h3>
                <p><strong>Text:</strong> ${div.text}</p>
                <p><strong>Type:</strong> ${div.boundary_type}</p>
                <p><strong>Confidence:</strong> <span class="confidence">${div.confidence}</span></p>
            </div>
        `).join('')}
    </body>
    </html>
    `;
}

function getTextHierarchyHtml(hierarchy: any): string {
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .hierarchy-item { margin-left: 20px; padding: 5px; border-left: 2px solid #ccc; }
            .level-0 { border-left-color: #ff0000; }
            .level-1 { border-left-color: #ff8800; }
            .level-2 { border-left-color: #0088ff; }
            .level-3 { border-left-color: #00ff88; }
        </style>
    </head>
    <body>
        <h1>🌳 Text Hierarchy</h1>
        
        <div id="hierarchy">
            ${renderHierarchyLevel(hierarchy, 0)}
        </div>
    </body>
    </html>
    `;
}

function renderHierarchyLevel(items: any[], level: number): string {
    return items.map(item => `
        <div class="hierarchy-item level-${level}">
            <strong>${item.type}:</strong> ${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}
            ${item.children ? renderHierarchyLevel(item.children, level + 1) : ''}
        </div>
    `).join('');
}
