import * as vscode from 'vscode';
import { OrchestrationDebugSession } from './debug-session';
import { OrchestrationFlowVisualizer } from './flow-visualizer';
import { GoalTracker } from './goal-tracker';
import { InterventionMonitor } from './intervention-monitor';
import { ContextAnalyzer } from './context-analyzer';
import { MetacognitionVisualizer } from './metacognition-visualizer';
import { StreamPipelineVisualizer } from './stream-pipeline';
import { BiomimeticInspector } from './biomimetic-inspector';
import { DebugConsole } from './debug-console';

let orchestrationFlowVisualizer: OrchestrationFlowVisualizer;
let goalTracker: GoalTracker;
let interventionMonitor: InterventionMonitor;
let contextAnalyzer: ContextAnalyzer;
let metacognitionVisualizer: MetacognitionVisualizer;
let streamPipelineVisualizer: StreamPipelineVisualizer;
let biomimeticInspector: BiomimeticInspector;
let debugConsole: DebugConsole;

let currentDebugSession: OrchestrationDebugSession | null = null;

export function activate(context: vscode.ExtensionContext) {
    console.log('Metacognitive Orchestration Debugger is now active!');

    // Initialize components
    initializeComponents(context);

    // Register debug adapter
    registerDebugAdapter(context);

    // Register commands
    const commands = [
        vscode.commands.registerCommand('orchestrationDebugger.startDebugging', startDebugging),
        vscode.commands.registerCommand('orchestrationDebugger.showOrchestrationFlow', showOrchestrationFlow),
        vscode.commands.registerCommand('orchestrationDebugger.trackGoals', trackGoals),
        vscode.commands.registerCommand('orchestrationDebugger.monitorInterventions', monitorInterventions),
        vscode.commands.registerCommand('orchestrationDebugger.analyzeContext', analyzeContext),
        vscode.commands.registerCommand('orchestrationDebugger.visualizeMetacognition', visualizeMetacognition),
        vscode.commands.registerCommand('orchestrationDebugger.showStreamPipeline', showStreamPipeline),
        vscode.commands.registerCommand('orchestrationDebugger.inspectBiomimeticPatterns', inspectBiomimeticPatterns),
        vscode.commands.registerCommand('orchestrationDebugger.openDebugConsole', openDebugConsole),
        
        // Internal debug commands
        vscode.commands.registerCommand('orchestrationDebugger.stepIntoOrchestration', stepIntoOrchestration),
        vscode.commands.registerCommand('orchestrationDebugger.stepOverOperation', stepOverOperation),
        vscode.commands.registerCommand('orchestrationDebugger.stepOutOfContext', stepOutOfContext),
        vscode.commands.registerCommand('orchestrationDebugger.continueExecution', continueExecution),
        vscode.commands.registerCommand('orchestrationDebugger.pauseExecution', pauseExecution)
    ];

    context.subscriptions.push(...commands);

    // Register tree providers
    registerTreeProviders(context);

    // Set up status bar
    setupStatusBar(context);

    // Register configuration provider for debug launches
    vscode.debug.registerDebugConfigurationProvider('turbulance', new TurbulanceConfigurationProvider());

    vscode.window.showInformationMessage('🧠 Metacognitive Orchestration Debugger activated!');
}

export function deactivate() {
    // Clean up resources
    orchestrationFlowVisualizer?.dispose();
    goalTracker?.dispose();
    interventionMonitor?.dispose();
    contextAnalyzer?.dispose();
    metacognitionVisualizer?.dispose();
    streamPipelineVisualizer?.dispose();
    biomimeticInspector?.dispose();
    debugConsole?.dispose();
    
    if (currentDebugSession) {
        currentDebugSession.dispose();
    }
}

function initializeComponents(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('orchestrationDebugger');
    
    orchestrationFlowVisualizer = new OrchestrationFlowVisualizer(context, {
        updateFrequency: config.get('updateFrequency'),
        enableRealtimeVisualization: config.get('enableRealtimeVisualization')
    });

    goalTracker = new GoalTracker(context);
    interventionMonitor = new InterventionMonitor(context);
    contextAnalyzer = new ContextAnalyzer(context);

    metacognitionVisualizer = new MetacognitionVisualizer(context, {
        traceDepth: config.get('traceDepth')
    });

    streamPipelineVisualizer = new StreamPipelineVisualizer(context);
    biomimeticInspector = new BiomimeticInspector(context);
    debugConsole = new DebugConsole(context);

    context.subscriptions.push(
        orchestrationFlowVisualizer,
        goalTracker,
        interventionMonitor,
        contextAnalyzer,
        metacognitionVisualizer,
        streamPipelineVisualizer,
        biomimeticInspector,
        debugConsole
    );
}

function registerDebugAdapter(context: vscode.ExtensionContext) {
    // Register debug adapter factory
    vscode.debug.registerDebugAdapterDescriptorFactory('turbulance', {
        createDebugAdapterDescriptor(session: vscode.DebugSession): vscode.ProviderResult<vscode.DebugAdapterDescriptor> {
            // Return inline debug adapter
            return new vscode.DebugAdapterInlineImplementation(new OrchestrationDebugSession(context));
        }
    });

    // Track debug sessions
    vscode.debug.onDidStartDebugSession(session => {
        if (session.type === 'turbulance') {
            currentDebugSession = session.customRequest('getDebugSession') as any;
            vscode.commands.executeCommand('setContext', 'orchestrationDebugger:debugging', true);
        }
    });

    vscode.debug.onDidTerminateDebugSession(session => {
        if (session.type === 'turbulance') {
            currentDebugSession = null;
            vscode.commands.executeCommand('setContext', 'orchestrationDebugger:debugging', false);
        }
    });
}

async function startDebugging() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'turbulance') {
        vscode.window.showWarningMessage('Please open a Turbulance file to start debugging.');
        return;
    }

    const document = editor.document;
    await document.save();

    // Start debug session with orchestration debugging enabled
    const debugConfig: vscode.DebugConfiguration = {
        type: 'turbulance',
        request: 'launch',
        name: 'Debug Turbulance Orchestration',
        program: document.fileName,
        enableMetacognitiveTracing: true,
        enableOrchestrationFlow: true,
        enableGoalTracking: true,
        enableInterventionMonitoring: true,
        enableContextAnalysis: true
    };

    const started = await vscode.debug.startDebugging(undefined, debugConfig);
    
    if (started) {
        vscode.window.showInformationMessage('🚀 Orchestration debugging started!');
        // Reveal debug views
        vscode.commands.executeCommand('workbench.view.extension.orchestrationDebugger');
    } else {
        vscode.window.showErrorMessage('Failed to start orchestration debugging.');
    }
}

async function showOrchestrationFlow() {
    try {
        await orchestrationFlowVisualizer.show();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to show orchestration flow: ${error}`);
    }
}

async function trackGoals() {
    try {
        if (currentDebugSession) {
            const goalData = await currentDebugSession.getGoalTrackingData();
            goalTracker.updateGoalData(goalData);
            await goalTracker.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to track goals: ${error}`);
    }
}

async function monitorInterventions() {
    try {
        if (currentDebugSession) {
            const interventionData = await currentDebugSession.getInterventionData();
            interventionMonitor.updateInterventionData(interventionData);
            await interventionMonitor.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to monitor interventions: ${error}`);
    }
}

async function analyzeContext() {
    try {
        if (currentDebugSession) {
            const contextData = await currentDebugSession.getContextData();
            contextAnalyzer.updateContextData(contextData);
            await contextAnalyzer.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to analyze context: ${error}`);
    }
}

async function visualizeMetacognition() {
    try {
        if (currentDebugSession) {
            const metacognitiveTrace = await currentDebugSession.getMetacognitiveTrace();
            metacognitionVisualizer.updateTrace(metacognitiveTrace);
            await metacognitionVisualizer.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to visualize metacognition: ${error}`);
    }
}

async function showStreamPipeline() {
    try {
        if (currentDebugSession) {
            const streamData = await currentDebugSession.getStreamPipelineData();
            streamPipelineVisualizer.updateStreamData(streamData);
            await streamPipelineVisualizer.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to show stream pipeline: ${error}`);
    }
}

async function inspectBiomimeticPatterns() {
    try {
        if (currentDebugSession) {
            const biomimeticData = await currentDebugSession.getBiomimeticData();
            biomimeticInspector.updateBiomimeticData(biomimeticData);
            await biomimeticInspector.show();
        } else {
            vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to inspect biomimetic patterns: ${error}`);
    }
}

async function openDebugConsole() {
    try {
        await debugConsole.show();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open debug console: ${error}`);
    }
}

// Debug control commands
async function stepIntoOrchestration() {
    if (currentDebugSession) {
        await currentDebugSession.stepInto('orchestration');
    }
}

async function stepOverOperation() {
    if (currentDebugSession) {
        await currentDebugSession.stepOver('operation');
    }
}

async function stepOutOfContext() {
    if (currentDebugSession) {
        await currentDebugSession.stepOut('context');
    }
}

async function continueExecution() {
    if (currentDebugSession) {
        await currentDebugSession.continue();
    }
}

async function pauseExecution() {
    if (currentDebugSession) {
        await currentDebugSession.pause();
    }
}

function registerTreeProviders(context: vscode.ExtensionContext) {
    const flowProvider = new FlowTreeProvider();
    const goalsProvider = new GoalsTreeProvider();
    const interventionsProvider = new InterventionsTreeProvider();
    const contextProvider = new ContextTreeProvider();
    const metacognitionProvider = new MetacognitionTreeProvider();

    vscode.window.createTreeView('orchestrationDebugger.flowView', {
        treeDataProvider: flowProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('orchestrationDebugger.goalsView', {
        treeDataProvider: goalsProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('orchestrationDebugger.interventionsView', {
        treeDataProvider: interventionsProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('orchestrationDebugger.contextView', {
        treeDataProvider: contextProvider,
        showCollapseAll: true
    });

    vscode.window.createTreeView('orchestrationDebugger.metacognitionView', {
        treeDataProvider: metacognitionProvider,
        showCollapseAll: true
    });

    context.subscriptions.push(flowProvider, goalsProvider, interventionsProvider, contextProvider, metacognitionProvider);
}

function setupStatusBar(context: vscode.ExtensionContext) {
    const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 40);
    statusBar.text = "$(debug-alt) Orchestration Debugger";
    statusBar.tooltip = "Metacognitive Orchestration Debugger";
    statusBar.command = "orchestrationDebugger.startDebugging";
    statusBar.show();

    context.subscriptions.push(statusBar);

    // Update status bar based on debug state
    vscode.debug.onDidStartDebugSession(session => {
        if (session.type === 'turbulance') {
            statusBar.text = "$(debug-alt) $(play) Debugging Orchestration";
            statusBar.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        }
    });

    vscode.debug.onDidTerminateDebugSession(session => {
        if (session.type === 'turbulance') {
            statusBar.text = "$(debug-alt) Orchestration Debugger";
            statusBar.backgroundColor = undefined;
        }
    });
}

// Debug configuration provider
class TurbulanceConfigurationProvider implements vscode.DebugConfigurationProvider {
    
    resolveDebugConfiguration(
        folder: vscode.WorkspaceFolder | undefined,
        config: vscode.DebugConfiguration,
        token?: vscode.CancellationToken
    ): vscode.ProviderResult<vscode.DebugConfiguration> {
        
        // If no configuration provided, create default
        if (!config.type && !config.request && !config.name) {
            const editor = vscode.window.activeTextEditor;
            if (editor && editor.document.languageId === 'turbulance') {
                config.type = 'turbulance';
                config.name = 'Debug Turbulance Orchestration';
                config.request = 'launch';
                config.program = editor.document.fileName;
                config.enableMetacognitiveTracing = true;
                config.enableOrchestrationFlow = true;
            }
        }

        // Ensure required properties are set
        if (!config.program) {
            return vscode.window.showInformationMessage("Cannot find a program to debug").then(_ => {
                return undefined; // abort launch
            });
        }

        return config;
    }
}

// Tree provider placeholder classes
class FlowTreeProvider implements vscode.TreeDataProvider<any> {
    private _onDidChangeTreeData: vscode.EventEmitter<any | undefined | null | void> = new vscode.EventEmitter<any | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<any | undefined | null | void> = this._onDidChangeTreeData.event;

    getTreeItem(element: any): vscode.TreeItem {
        return element;
    }

    getChildren(element?: any): Thenable<any[]> {
        if (currentDebugSession) {
            return currentDebugSession.getFlowTreeData(element);
        }
        return Promise.resolve([]);
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    dispose() {}
}

class GoalsTreeProvider implements vscode.TreeDataProvider<any> {
    private _onDidChangeTreeData: vscode.EventEmitter<any | undefined | null | void> = new vscode.EventEmitter<any | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<any | undefined | null | void> = this._onDidChangeTreeData.event;

    getTreeItem(element: any): vscode.TreeItem {
        return element;
    }

    getChildren(element?: any): Thenable<any[]> {
        if (currentDebugSession) {
            return currentDebugSession.getGoalsTreeData(element);
        }
        return Promise.resolve([]);
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    dispose() {}
}

class InterventionsTreeProvider implements vscode.TreeDataProvider<any> {
    private _onDidChangeTreeData: vscode.EventEmitter<any | undefined | null | void> = new vscode.EventEmitter<any | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<any | undefined | null | void> = this._onDidChangeTreeData.event;

    getTreeItem(element: any): vscode.TreeItem {
        return element;
    }

    getChildren(element?: any): Thenable<any[]> {
        if (currentDebugSession) {
            return currentDebugSession.getInterventionsTreeData(element);
        }
        return Promise.resolve([]);
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    dispose() {}
}

class ContextTreeProvider implements vscode.TreeDataProvider<any> {
    private _onDidChangeTreeData: vscode.EventEmitter<any | undefined | null | void> = new vscode.EventEmitter<any | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<any | undefined | null | void> = this._onDidChangeTreeData.event;

    getTreeItem(element: any): vscode.TreeItem {
        return element;
    }

    getChildren(element?: any): Thenable<any[]> {
        if (currentDebugSession) {
            return currentDebugSession.getContextTreeData(element);
        }
        return Promise.resolve([]);
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    dispose() {}
}

class MetacognitionTreeProvider implements vscode.TreeDataProvider<any> {
    private _onDidChangeTreeData: vscode.EventEmitter<any | undefined | null | void> = new vscode.EventEmitter<any | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<any | undefined | null | void> = this._onDidChangeTreeData.event;

    getTreeItem(element: any): vscode.TreeItem {
        return element;
    }

    getChildren(element?: any): Thenable<any[]> {
        if (currentDebugSession) {
            return currentDebugSession.getMetacognitionTreeData(element);
        }
        return Promise.resolve([]);
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    dispose() {}
}
