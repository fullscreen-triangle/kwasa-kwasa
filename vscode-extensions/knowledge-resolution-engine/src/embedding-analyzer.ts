import * as vscode from 'vscode';
import * as tf from '@tensorflow/tfjs';

export interface EmbeddingAnalyzerConfig {
    model?: string;
    confidenceThreshold?: number;
    noveltyThreshold?: number;
}

export interface KnowledgeClaim {
    text: string;
    embedding: number[];
    confidence: number;
    startPosition: vscode.Position;
    endPosition: vscode.Position;
    context: string;
}

export interface EmbeddingDifferential {
    concept: string;
    projectEmbedding: number[];
    externalEmbedding: number[];
    divergenceScore: number;
    confidence: number;
    type: 'innovation' | 'error' | 'terminology' | 'perspective';
    explanation: string;
}

export interface NovelInsight {
    concept: string;
    description: string;
    noveltyScore: number;
    evidenceCount: number;
    supportingText: string[];
    confidence: number;
}

export interface ConfidenceRegion {
    start: vscode.Position;
    end: vscode.Position;
    confidence: number;
    isNovel: boolean;
    reasoning: string;
}

export class EmbeddingAnalyzer implements vscode.Disposable {
    private model: any;
    private config: EmbeddingAnalyzerConfig;
    private embeddingCache: Map<string, number[]> = new Map();
    
    constructor(config: EmbeddingAnalyzerConfig) {
        this.config = config;
        this.initializeModel();
    }

    private async initializeModel() {
        // Initialize the sentence transformer model
        // In a real implementation, this would load the actual model
        console.log(`Initializing embedding model: ${this.config.model}`);
        // Placeholder for model initialization
        this.model = { encode: this.mockEncode.bind(this) };
    }

    private async mockEncode(text: string): Promise<number[]> {
        // Mock embedding generation for demonstration
        // In reality, this would use a proper sentence transformer
        const hash = this.simpleHash(text);
        const embedding = Array.from({ length: 384 }, (_, i) => 
            Math.sin(hash + i) * Math.cos(hash * 0.7 + i * 0.3)
        );
        return embedding;
    }

    private simpleHash(str: string): number {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }

    async analyzeText(text: string): Promise<KnowledgeClaim[]> {
        const claims: KnowledgeClaim[] = [];
        
        // Split text into sentences for analysis
        const sentences = this.splitIntoSentences(text);
        
        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i];
            
            // Skip very short sentences
            if (sentence.trim().length < 10) continue;
            
            try {
                const embedding = await this.getEmbedding(sentence);
                const confidence = await this.calculateConfidence(sentence, embedding);
                
                // Create position information (simplified)
                const startPos = new vscode.Position(i, 0);
                const endPos = new vscode.Position(i, sentence.length);
                
                const claim: KnowledgeClaim = {
                    text: sentence.trim(),
                    embedding: embedding,
                    confidence: confidence,
                    startPosition: startPos,
                    endPosition: endPos,
                    context: this.getContext(sentences, i)
                };
                
                claims.push(claim);
            } catch (error) {
                console.error(`Error processing sentence: ${error}`);
            }
        }
        
        return claims;
    }

    async compareDifferentials(
        projectEmbeddings: KnowledgeClaim[],
        externalEmbeddings: any[]
    ): Promise<{
        divergences: EmbeddingDifferential[],
        novel_insights: NovelInsight[],
        project_embeddings: { x: number, y: number, text: string }[],
        external_embeddings: { x: number, y: number, text: string }[]
    }> {
        const divergences: EmbeddingDifferential[] = [];
        const novel_insights: NovelInsight[] = [];

        // Compare each project claim with external knowledge
        for (const projectClaim of projectEmbeddings) {
            const closestExternal = this.findClosestEmbedding(
                projectClaim.embedding,
                externalEmbeddings
            );

            if (closestExternal) {
                const divergenceScore = this.calculateDivergenceScore(
                    projectClaim.embedding,
                    closestExternal.embedding
                );

                if (divergenceScore > this.config.noveltyThreshold!) {
                    // Determine if this is a novel insight or potential error
                    const classification = await this.classifyDivergence(
                        projectClaim,
                        closestExternal,
                        divergenceScore
                    );

                    if (classification.type === 'innovation' && classification.confidence > 0.7) {
                        const insight: NovelInsight = {
                            concept: this.extractConcept(projectClaim.text),
                            description: `Novel perspective detected: ${projectClaim.text.substring(0, 100)}...`,
                            noveltyScore: divergenceScore,
                            evidenceCount: 1, // Would calculate supporting evidence
                            supportingText: [projectClaim.text],
                            confidence: classification.confidence
                        };
                        novel_insights.push(insight);
                    }

                    divergences.push({
                        concept: this.extractConcept(projectClaim.text),
                        projectEmbedding: projectClaim.embedding,
                        externalEmbedding: closestExternal.embedding,
                        divergenceScore: divergenceScore,
                        confidence: classification.confidence,
                        type: classification.type,
                        explanation: classification.explanation
                    });
                }
            }
        }

        // Generate 2D projections for visualization
        const projectVis = projectEmbeddings.map(claim => ({
            x: this.project2D(claim.embedding)[0],
            y: this.project2D(claim.embedding)[1], 
            text: claim.text.substring(0, 50) + '...'
        }));

        const externalVis = externalEmbeddings.map(ext => ({
            x: this.project2D(ext.embedding)[0],
            y: this.project2D(ext.embedding)[1],
            text: ext.text?.substring(0, 50) + '...' || 'External knowledge'
        }));

        return {
            divergences,
            novel_insights,
            project_embeddings: projectVis,
            external_embeddings: externalVis
        };
    }

    async generateConfidenceMap(document: vscode.TextDocument): Promise<{
        regions: ConfidenceRegion[],
        overallScore: number
    }> {
        const regions: ConfidenceRegion[] = [];
        const text = document.getText();
        const lines = text.split('\n');

        let totalConfidence = 0;
        let regionCount = 0;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.length < 10) continue;

            try {
                const embedding = await this.getEmbedding(line);
                const confidence = await this.calculateConfidence(line, embedding);
                const isNovel = await this.isNovelInsight(line, embedding);

                const region: ConfidenceRegion = {
                    start: new vscode.Position(i, 0),
                    end: new vscode.Position(i, lines[i].length),
                    confidence: confidence,
                    isNovel: isNovel,
                    reasoning: this.generateReasoningExplanation(line, confidence, isNovel)
                };

                regions.push(region);
                totalConfidence += confidence;
                regionCount++;
            } catch (error) {
                console.error(`Error processing line ${i}: ${error}`);
            }
        }

        const overallScore = regionCount > 0 ? totalConfidence / regionCount : 0;

        return { regions, overallScore };
    }

    async extractKnowledgeClaims(text: string): Promise<KnowledgeClaim[]> {
        // Extract specific knowledge claims from text
        const claims: KnowledgeClaim[] = [];
        
        // Look for proposition statements, evidence claims, etc.
        const propositionRegex = /proposition\s+(\w+):\s*(.+?)(?=proposition|\n\n|$)/gis;
        const evidenceRegex = /evidence\s+(\w+):\s*(.+?)(?=evidence|\n\n|$)/gis;
        const assertionRegex = /assert\s+(.+?)(?=\n|$)/gi;

        // Extract propositions
        let match;
        while ((match = propositionRegex.exec(text)) !== null) {
            const claimText = match[2].trim();
            const embedding = await this.getEmbedding(claimText);
            const confidence = await this.calculateConfidence(claimText, embedding);

            claims.push({
                text: claimText,
                embedding: embedding,
                confidence: confidence,
                startPosition: new vscode.Position(0, 0), // Would calculate actual position
                endPosition: new vscode.Position(0, claimText.length),
                context: `Proposition: ${match[1]}`
            });
        }

        // Extract evidence claims
        while ((match = evidenceRegex.exec(text)) !== null) {
            const claimText = match[2].trim();
            const embedding = await this.getEmbedding(claimText);
            const confidence = await this.calculateConfidence(claimText, embedding);

            claims.push({
                text: claimText,
                embedding: embedding,
                confidence: confidence,
                startPosition: new vscode.Position(0, 0),
                endPosition: new vscode.Position(0, claimText.length),
                context: `Evidence: ${match[1]}`
            });
        }

        // Extract assertions
        while ((match = assertionRegex.exec(text)) !== null) {
            const claimText = match[1].trim();
            const embedding = await this.getEmbedding(claimText);
            const confidence = await this.calculateConfidence(claimText, embedding);

            claims.push({
                text: claimText,
                embedding: embedding,
                confidence: confidence,
                startPosition: new vscode.Position(0, 0),
                endPosition: new vscode.Position(0, claimText.length),
                context: 'Assertion'
            });
        }

        return claims;
    }

    async analyzeCorrespondence(claims: KnowledgeClaim[], validationResults: any[]): Promise<any> {
        const correspondenceAnalysis = {
            totalClaims: claims.length,
            validatedClaims: 0,
            contradictedClaims: 0,
            novelClaims: 0,
            averageConfidence: 0,
            correspondenceDetails: [] as any[]
        };

        let totalConfidence = 0;

        for (let i = 0; i < claims.length; i++) {
            const claim = claims[i];
            const validation = validationResults[i];

            if (validation) {
                const correspondence = {
                    claim: claim.text,
                    validation: validation.result,
                    confidence: validation.confidence,
                    sources: validation.sources,
                    type: this.classifyCorrespondence(claim, validation)
                };

                correspondenceAnalysis.correspondenceDetails.push(correspondence);

                if (validation.result === 'validated') {
                    correspondenceAnalysis.validatedClaims++;
                } else if (validation.result === 'contradicted') {
                    correspondenceAnalysis.contradictedClaims++;
                } else if (validation.result === 'novel') {
                    correspondenceAnalysis.novelClaims++;
                }

                totalConfidence += validation.confidence;
            }
        }

        correspondenceAnalysis.averageConfidence = totalConfidence / claims.length;

        return correspondenceAnalysis;
    }

    async analyzeSemanticDrift(text: string): Promise<any> {
        const drift = {
            overallDrift: 0,
            driftRegions: [] as any[],
            terminology: [] as any[],
            conceptualShifts: [] as any[]
        };

        // Analyze how terminology and concepts shift from standard usage
        const sentences = this.splitIntoSentences(text);
        const standardEmbeddings = await this.getStandardEmbeddings(sentences);

        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i];
            const projectEmbedding = await this.getEmbedding(sentence);
            const standardEmbedding = standardEmbeddings[i];

            if (standardEmbedding) {
                const driftScore = this.calculateDivergenceScore(projectEmbedding, standardEmbedding);
                
                if (driftScore > 0.3) {
                    drift.driftRegions.push({
                        text: sentence,
                        driftScore: driftScore,
                        type: driftScore > 0.7 ? 'major' : 'minor'
                    });
                }
            }
        }

        drift.overallDrift = drift.driftRegions.reduce((sum, region) => sum + region.driftScore, 0) / drift.driftRegions.length || 0;

        return drift;
    }

    private async getEmbedding(text: string): Promise<number[]> {
        // Check cache first
        if (this.embeddingCache.has(text)) {
            return this.embeddingCache.get(text)!;
        }

        const embedding = await this.model.encode(text);
        this.embeddingCache.set(text, embedding);
        return embedding;
    }

    private async calculateConfidence(text: string, embedding: number[]): Promise<number> {
        // Calculate confidence based on various factors
        let confidence = 0.5; // Base confidence

        // Factor 1: Text length and complexity
        const lengthFactor = Math.min(text.length / 100, 1.0);
        confidence += lengthFactor * 0.1;

        // Factor 2: Presence of specific terms that indicate certainty
        const certaintyTerms = ['proven', 'demonstrated', 'established', 'confirmed'];
        const uncertaintyTerms = ['possibly', 'might', 'could', 'perhaps', 'potentially'];
        
        const certaintyCount = certaintyTerms.filter(term => text.toLowerCase().includes(term)).length;
        const uncertaintyCount = uncertaintyTerms.filter(term => text.toLowerCase().includes(term)).length;
        
        confidence += (certaintyCount - uncertaintyCount) * 0.1;

        // Factor 3: Embedding magnitude (assuming more specific concepts have higher magnitude)
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        confidence += Math.min(magnitude / 10, 0.2);

        // Clamp to [0, 1]
        return Math.max(0, Math.min(1, confidence));
    }

    private async isNovelInsight(text: string, embedding: number[]): Promise<boolean> {
        // Determine if this represents a novel insight
        // This would compare against a database of known knowledge
        
        // For now, use heuristics
        const noveltyIndicators = ['novel', 'new', 'unprecedented', 'breakthrough', 'discovery'];
        const hasNoveltyIndicators = noveltyIndicators.some(indicator => 
            text.toLowerCase().includes(indicator)
        );

        // Also check embedding divergence from "standard" embeddings
        const divergenceFromStandard = await this.calculateDivergenceFromStandard(embedding);
        
        return hasNoveltyIndicators || divergenceFromStandard > this.config.noveltyThreshold!;
    }

    private async calculateDivergenceFromStandard(embedding: number[]): Promise<number> {
        // This would compare against standard knowledge base embeddings
        // For now, return a mock divergence based on embedding characteristics
        const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return Math.random() * 0.5 + (magnitude > 15 ? 0.3 : 0);
    }

    private splitIntoSentences(text: string): string[] {
        // Simple sentence splitting - in reality would use more sophisticated NLP
        return text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
    }

    private getContext(sentences: string[], index: number): string {
        const contextRange = 2;
        const start = Math.max(0, index - contextRange);
        const end = Math.min(sentences.length, index + contextRange + 1);
        return sentences.slice(start, end).join(' ');
    }

    private findClosestEmbedding(targetEmbedding: number[], candidates: any[]): any {
        let closest = null;
        let minDistance = Infinity;

        for (const candidate of candidates) {
            const distance = this.calculateEuclideanDistance(targetEmbedding, candidate.embedding);
            if (distance < minDistance) {
                minDistance = distance;
                closest = candidate;
            }
        }

        return closest;
    }

    private calculateDivergenceScore(embedding1: number[], embedding2: number[]): number {
        return 1 - this.calculateCosineSimilarity(embedding1, embedding2);
    }

    private calculateEuclideanDistance(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same length');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            const diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    private calculateCosineSimilarity(vec1: number[], vec2: number[]): number {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same length');
        }

        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    private async classifyDivergence(
        projectClaim: KnowledgeClaim,
        externalClaim: any,
        divergenceScore: number
    ): Promise<{ type: 'innovation' | 'error' | 'terminology' | 'perspective', confidence: number, explanation: string }> {
        // Classify the type of divergence
        let type: 'innovation' | 'error' | 'terminology' | 'perspective' = 'perspective';
        let confidence = 0.5;
        let explanation = '';

        if (divergenceScore > 0.8) {
            // Very high divergence - could be innovation or error
            if (projectClaim.confidence > 0.8) {
                type = 'innovation';
                confidence = 0.8;
                explanation = 'High confidence claim with significant divergence from external sources suggests potential novel insight.';
            } else {
                type = 'error';
                confidence = 0.7;
                explanation = 'Low confidence claim with high divergence may indicate error or misconception.';
            }
        } else if (divergenceScore > 0.5) {
            type = 'terminology';
            confidence = 0.6;
            explanation = 'Moderate divergence suggests different terminology or framing for similar concepts.';
        } else {
            type = 'perspective';
            confidence = 0.7;
            explanation = 'Minor divergence indicates different perspective on established knowledge.';
        }

        return { type, confidence, explanation };
    }

    private extractConcept(text: string): string {
        // Extract the main concept from text
        // Simple implementation - would use more sophisticated NLP in reality
        const words = text.split(' ').filter(word => word.length > 3);
        return words.slice(0, 3).join(' ');
    }

    private project2D(embedding: number[]): [number, number] {
        // Simple 2D projection using PCA-like approach
        // In reality would use proper dimensionality reduction
        const x = embedding.slice(0, 100).reduce((sum, val, i) => sum + val * Math.cos(i), 0);
        const y = embedding.slice(0, 100).reduce((sum, val, i) => sum + val * Math.sin(i), 0);
        return [x / 100, y / 100];
    }

    private generateReasoningExplanation(text: string, confidence: number, isNovel: boolean): string {
        let reasoning = `Confidence: ${confidence.toFixed(2)} `;
        
        if (confidence > 0.8) {
            reasoning += '(High - well-supported claim)';
        } else if (confidence > 0.5) {
            reasoning += '(Medium - moderately supported)';
        } else {
            reasoning += '(Low - requires additional evidence)';
        }

        if (isNovel) {
            reasoning += ' ✨ Novel insight detected';
        }

        return reasoning;
    }

    private async getStandardEmbeddings(sentences: string[]): Promise<(number[] | null)[]> {
        // Get embeddings from standard knowledge sources
        // For now, return mock embeddings
        return sentences.map(() => null); // Would implement actual standard embeddings lookup
    }

    private classifyCorrespondence(claim: KnowledgeClaim, validation: any): string {
        if (validation.confidence > 0.8 && validation.result === 'validated') {
            return 'strong_correspondence';
        } else if (validation.confidence > 0.5 && validation.result === 'validated') {
            return 'moderate_correspondence';
        } else if (validation.result === 'contradicted') {
            return 'contradiction';
        } else if (validation.result === 'novel') {
            return 'novel_insight';
        } else {
            return 'uncertain';
        }
    }

    dispose() {
        this.embeddingCache.clear();
        // Clean up model resources
    }
}
