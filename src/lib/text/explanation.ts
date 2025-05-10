/**
 * Explanation Analysis and Enhancement Module
 * 
 * Provides functions to ensure text properly explains concepts
 * and includes appropriate supporting details.
 */

interface ExplanationOptions {
  checkLevel: 'basic' | 'detailed' | 'comprehensive';
  requiredElements?: string[];
  minSentences: number;
  requireExamples?: boolean;
  domainContext?: string;
}

interface ExplanationAnalysis {
  isValid: boolean;
  missingElements: string[];
  suggestions: string[];
  score: number;
  enhancedExplanation?: string;
}

/**
 * Verifies that an explanation properly follows a concept or statement
 * and provides suggested improvements if it doesn't
 * 
 * @param concept - The concept or statement being explained
 * @param explanation - The explanation text to verify
 * @param options - Options to customize verification behavior
 * @returns Analysis results with validity and suggestions
 */
export function ensure_explanation_follows(
  concept: string,
  explanation: string,
  options?: Partial<ExplanationOptions>
): ExplanationAnalysis {
  // Default options
  const opts: ExplanationOptions = {
    checkLevel: 'detailed',
    minSentences: 2,
    requireExamples: false,
    ...options
  };

  // Initialize result
  const result: ExplanationAnalysis = {
    isValid: true,
    missingElements: [],
    suggestions: [],
    score: 0
  };

  // Analyze semantic connection between concept and explanation
  const connectionScore = analyzeConceptExplanationConnection(concept, explanation);
  
  // Check if concept is actually referenced in explanation
  if (!explanationReferencesConceptProperly(concept, explanation)) {
    result.isValid = false;
    result.missingElements.push('concept reference');
    result.suggestions.push('Ensure the explanation directly references the concept being explained');
  }
  
  // Check for explanation length/depth
  const sentences = splitIntoSentences(explanation);
  if (sentences.length < opts.minSentences) {
    result.isValid = false;
    result.missingElements.push('explanation depth');
    result.suggestions.push(`Expand explanation to at least ${opts.minSentences} sentences for adequate depth`);
  }
  
  // Check for required explanation elements based on check level
  const requiredElements = determineRequiredElements(opts);
  const missingRequiredElements = findMissingExplanationElements(explanation, requiredElements);
  
  if (missingRequiredElements.length > 0) {
    result.isValid = false;
    result.missingElements = [...result.missingElements, ...missingRequiredElements];
    
    missingRequiredElements.forEach(element => {
      switch(element) {
        case 'definition':
          result.suggestions.push('Include a clear definition of the concept');
          break;
        case 'example':
          result.suggestions.push('Add at least one concrete example to illustrate the concept');
          break;
        case 'context':
          result.suggestions.push('Provide context for when/where this concept applies');
          break;
        case 'implications':
          result.suggestions.push('Discuss implications or consequences of the concept');
          break;
        default:
          result.suggestions.push(`Include missing element: ${element}`);
      }
    });
  }
  
  // Check logical flow of explanation
  const flowScore = analyzeLogicalFlow(explanation);
  if (flowScore < 0.7) {
    result.isValid = false;
    result.missingElements.push('logical flow');
    result.suggestions.push('Improve the logical flow between sentences in the explanation');
  }
  
  // Calculate overall score (0-100)
  // Weights: connection (40%), elements (30%), flow (30%)
  result.score = Math.round(
    (connectionScore * 0.4 + 
    ((requiredElements.length - missingRequiredElements.length) / requiredElements.length) * 0.3 + 
    flowScore * 0.3) * 100
  );
  
  // If requested and needed, generate enhanced explanation
  if (!result.isValid && options?.checkLevel === 'comprehensive') {
    result.enhancedExplanation = generateEnhancedExplanation(concept, explanation, result);
  }
  
  return result;
}

/**
 * Analyzes how well the explanation relates to the concept
 * Returns a score from 0.0 to 1.0
 */
function analyzeConceptExplanationConnection(concept: string, explanation: string): number {
  // Simple implementation - checks for keyword presence and relevance
  // A more sophisticated version would use semantic analysis
  
  // Extract key terms from concept
  const conceptTerms = extractKeyTerms(concept);
  
  // Check for presence of terms in explanation
  let matchCount = 0;
  conceptTerms.forEach(term => {
    const regex = new RegExp(`\\b${term}\\b`, 'i');
    if (regex.test(explanation)) {
      matchCount++;
    }
  });
  
  // Calculate basic connection score
  let score = conceptTerms.length > 0 ? matchCount / conceptTerms.length : 0;
  
  // Adjust score based on position of first reference
  // Concepts referenced earlier get a bonus
  const firstMentionPosition = findFirstMentionPosition(concept, explanation);
  if (firstMentionPosition < 0.2) { // In first 20% of text
    score = Math.min(1.0, score + 0.2);
  } else if (firstMentionPosition > 0.5) { // After halfway point
    score = Math.max(0.0, score - 0.1);
  }
  
  return score;
}

/**
 * Extract important terms from a concept statement
 */
function extractKeyTerms(text: string): string[] {
  // Remove common stop words and extract significant terms
  const stopWords = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'];
  
  // Split into words, filter out stop words and short words, convert to lowercase
  const terms = text.split(/\s+/)
    .map(word => word.toLowerCase().replace(/[^a-z0-9]/g, ''))
    .filter(word => word.length > 2 && !stopWords.includes(word));
  
  // Return unique terms
  return [...new Set(terms)];
}

/**
 * Determine where in the explanation the concept is first mentioned
 * Returns a value from 0.0 (beginning) to 1.0 (end)
 */
function findFirstMentionPosition(concept: string, explanation: string): number {
  const conceptTerms = extractKeyTerms(concept);
  
  // Find first occurrence of any concept term
  let firstPosition = explanation.length;
  
  conceptTerms.forEach(term => {
    const regex = new RegExp(`\\b${term}\\b`, 'i');
    const match = regex.exec(explanation);
    if (match && match.index < firstPosition) {
      firstPosition = match.index;
    }
  });
  
  // If no terms found, return 1.0 (end)
  if (firstPosition === explanation.length) {
    return 1.0;
  }
  
  // Return normalized position
  return firstPosition / explanation.length;
}

/**
 * Checks if the explanation directly references the concept
 */
function explanationReferencesConceptProperly(concept: string, explanation: string): boolean {
  // Basic implementation - extract main subject from concept and check if it appears in explanation
  // This would be more sophisticated with NLP parsing in a full implementation
  
  const mainTerms = extractKeyTerms(concept).slice(0, 2); // Take first two key terms
  
  // At least one of the main terms should appear in the explanation
  return mainTerms.some(term => {
    const regex = new RegExp(`\\b${term}\\b`, 'i');
    return regex.test(explanation);
  });
}

/**
 * Split text into sentences using basic rules
 */
function splitIntoSentences(text: string): string[] {
  // Basic sentence splitting - would be enhanced with NLP in production
  return text.split(/[.!?]+\s+|\n+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

/**
 * Determine required elements based on check level
 */
function determineRequiredElements(options: ExplanationOptions): string[] {
  // If explicit required elements provided, use those
  if (options.requiredElements && options.requiredElements.length > 0) {
    return options.requiredElements;
  }
  
  // Otherwise determine based on check level
  switch (options.checkLevel) {
    case 'basic':
      return ['definition'];
    case 'detailed':
      const elements = ['definition', 'context'];
      if (options.requireExamples) {
        elements.push('example');
      }
      return elements;
    case 'comprehensive':
      return ['definition', 'context', 'example', 'implications'];
    default:
      return ['definition'];
  }
}

/**
 * Find which required elements are missing from the explanation
 */
function findMissingExplanationElements(explanation: string, requiredElements: string[]): string[] {
  const missing: string[] = [];
  
  // Check for definition (usually at beginning, contains "is", "refers to", etc.)
  if (requiredElements.includes('definition')) {
    const hasDefinition = /\b(is|are|refers to|defined as|means)\b/i.test(
      splitIntoSentences(explanation)[0] || ''
    );
    if (!hasDefinition) {
      missing.push('definition');
    }
  }
  
  // Check for examples (typically include "for example", "such as", etc.)
  if (requiredElements.includes('example')) {
    const hasExample = /\b(for example|such as|instance|e\.g\.|like|including)\b/i.test(explanation);
    if (!hasExample) {
      missing.push('example');
    }
  }
  
  // Check for context (when, where, who uses it)
  if (requiredElements.includes('context')) {
    const hasContext = /\b(when|where|used in|applied to|during|in the context)\b/i.test(explanation);
    if (!hasContext) {
      missing.push('context');
    }
  }
  
  // Check for implications/consequences
  if (requiredElements.includes('implications')) {
    const hasImplications = /\b(therefore|thus|as a result|consequently|leads to|results in|implies|meaning that)\b/i.test(explanation);
    if (!hasImplications) {
      missing.push('implications');
    }
  }
  
  return missing;
}

/**
 * Analyze the logical flow of the explanation
 * Returns a score from 0.0 to 1.0
 */
function analyzeLogicalFlow(explanation: string): number {
  const sentences = splitIntoSentences(explanation);
  
  // Need at least 2 sentences to check flow
  if (sentences.length < 2) {
    return 0.5; // Neutral score for single sentence
  }
  
  // Count logical connectors between sentences
  let connectorCount = 0;
  
  // Check for connectors at beginning of sentences
  const connectors = [
    'therefore', 'thus', 'hence', 'so', 'consequently', 'as a result',
    'furthermore', 'moreover', 'in addition', 'additionally',
    'however', 'nevertheless', 'nonetheless', 'on the other hand',
    'similarly', 'likewise', 'in the same way',
    'for instance', 'for example', 'specifically',
    'in conclusion', 'to summarize', 'in summary'
  ];
  
  // Skip first sentence and check others for connectors
  for (let i = 1; i < sentences.length; i++) {
    // Check if sentence starts with a connector
    if (connectors.some(conn => sentences[i].toLowerCase().startsWith(conn))) {
      connectorCount++;
      continue;
    }
    
    // Check for pronouns referring to previous content
    if (/^(this|these|those|it|they|such)\b/i.test(sentences[i])) {
      connectorCount++;
      continue;
    }
  }
  
  // Calculate flow score - a perfect score would have every sentence after the first
  // containing a connector, but we don't want to be too strict
  const idealConnectorCount = sentences.length - 1;
  const flowScore = Math.min(1.0, (connectorCount / idealConnectorCount) * 1.5); // Allow some leeway
  
  return flowScore;
}

/**
 * Generate an enhanced version of the explanation based on analysis
 */
function generateEnhancedExplanation(
  concept: string, 
  originalExplanation: string, 
  analysis: ExplanationAnalysis
): string {
  // This would ideally use NLP/LLM to generate improvements
  // For this implementation, we'll use a template-based approach
  
  const sentences = splitIntoSentences(originalExplanation);
  let enhanced = originalExplanation;
  
  // Add definition if missing
  if (analysis.missingElements.includes('definition')) {
    const conceptTerms = extractKeyTerms(concept);
    const mainTerm = conceptTerms[0] || 'This concept';
    enhanced = `${mainTerm.charAt(0).toUpperCase() + mainTerm.slice(1)} refers to a key idea in this domain. ${enhanced}`;
  }
  
  // Add example if missing
  if (analysis.missingElements.includes('example')) {
    enhanced += ` For example, this concept can be illustrated in practical scenarios.`;
  }
  
  // Add context if missing
  if (analysis.missingElements.includes('context')) {
    enhanced += ` This is particularly relevant in certain contexts where the concept is commonly applied.`;
  }
  
  // Add implications if missing
  if (analysis.missingElements.includes('implications')) {
    enhanced += ` As a result, understanding this concept has important implications for the broader domain.`;
  }
  
  // Improve flow if needed
  if (analysis.missingElements.includes('logical flow') && sentences.length >= 2) {
    // Simple improvement - just connect sentences with transitions
    const improvedSentences = [sentences[0]];
    const transitions = ['Moreover', 'Additionally', 'Furthermore', 'In addition'];
    
    for (let i = 1; i < sentences.length; i++) {
      // Add transitions only where needed
      if (!/^(therefore|thus|hence|so|consequently|furthermore|moreover|however|nevertheless|for instance|specifically|in conclusion)/i.test(sentences[i])) {
        const transition = transitions[i % transitions.length];
        improvedSentences.push(`${transition}, ${sentences[i].charAt(0).toLowerCase() + sentences[i].slice(1)}`);
      } else {
        improvedSentences.push(sentences[i]);
      }
    }
    
    enhanced = improvedSentences.join('. ').replace(/\.\./g, '.');
  }
  
  return enhanced;
} 