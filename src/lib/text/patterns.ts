/**
 * Pattern Analysis Module
 * 
 * Provides functions for identifying and extracting text patterns
 * using regular expressions and semantic analysis.
 */

interface PatternOptions {
  caseSensitive?: boolean;     // Whether to match case sensitively
  matchWholeWord?: boolean;    // Whether to match whole words only
  maxMatches?: number;         // Maximum matches to return
  includeContext?: boolean;    // Include surrounding context with matches
  contextWords?: number;       // Number of words of context to include
  patterns?: RegExp[];         // Additional custom patterns
  strictMode?: boolean;        // Whether to require exact matches
}

interface PatternMatch {
  pattern: string;             // The pattern that matched
  matched: string;             // The text that matched
  index: number;               // Position in the original text
  context?: string;            // Surrounding context if requested
}

/**
 * Checks if text contains a specific pattern
 * 
 * @param text - The text to search in
 * @param pattern - The pattern to search for (string or regex)
 * @param options - Search configuration options
 * @returns Boolean indicating if the pattern was found
 */
export function contains(
  text: string,
  pattern: string | RegExp,
  options?: Partial<PatternOptions>
): boolean {
  // Default options
  const opts: PatternOptions = {
    caseSensitive: false,
    matchWholeWord: false,
    strictMode: false,
    ...options
  };
  
  // Convert pattern to regex if it's a string
  const regex = convertPatternToRegex(pattern, opts);
  
  // Test for match
  return regex.test(text);
}

/**
 * Extracts patterns from text based on provided criteria
 * 
 * @param text - The text to extract patterns from
 * @param patterns - Patterns to extract (strings or regexes)
 * @param options - Extraction configuration options
 * @returns Array of pattern matches
 */
export function extract_patterns(
  text: string,
  patterns: (string | RegExp)[],
  options?: Partial<PatternOptions>
): PatternMatch[] {
  // Default options
  const opts: PatternOptions = {
    caseSensitive: false,
    matchWholeWord: false,
    maxMatches: 100,
    includeContext: false,
    contextWords: 5,
    strictMode: false,
    ...options
  };
  
  const results: PatternMatch[] = [];
  
  // Process each pattern
  patterns.forEach(patternItem => {
    // Skip empty patterns
    if (typeof patternItem === 'string' && patternItem.trim() === '') {
      return;
    }
    
    // Convert pattern to regex
    const regex = convertPatternToRegex(patternItem, opts);
    
    // Set regex to be global to find all matches
    regex.lastIndex = 0;
    
    // Find matches
    let match;
    let matchCount = 0;
    const maxMatches = opts.maxMatches || 100;
    
    while ((match = regex.exec(text)) !== null && matchCount < maxMatches) {
      // Avoid infinite loops with zero-length matches
      if (match.index === regex.lastIndex) {
        regex.lastIndex++;
        continue;
      }
      
      matchCount++;
      
      // Create match result
      const matchResult: PatternMatch = {
        pattern: patternItem.toString(),
        matched: match[0],
        index: match.index
      };
      
      // Add context if requested
      if (opts.includeContext) {
        matchResult.context = extractContext(
          text, 
          match.index, 
          match[0].length, 
          opts.contextWords || 5
        );
      }
      
      results.push(matchResult);
      
      // Non-global regex will cause infinite loop
      if (!regex.global) break;
    }
  });
  
  // Sort results by position in text
  return results.sort((a, b) => a.index - b.index);
}

/**
 * Converts a pattern (string or regex) to a proper regex object
 * based on the provided options
 */
function convertPatternToRegex(
  pattern: string | RegExp,
  options: PatternOptions
): RegExp {
  // If already a regex, ensure global flag is set
  if (pattern instanceof RegExp) {
    const flags = (pattern.global ? 'g' : 'g') + 
                 (pattern.ignoreCase || !options.caseSensitive ? 'i' : '') +
                 (pattern.multiline ? 'm' : '');
    return new RegExp(pattern.source, flags);
  }
  
  // Escape special regex characters if strictMode
  let patternString = options.strictMode ? 
    escapeRegExp(pattern as string) : 
    pattern as string;
  
  // Add word boundaries if matching whole words
  if (options.matchWholeWord) {
    patternString = `\\b${patternString}\\b`;
  }
  
  // Create regex with appropriate flags
  const flags = 'g' + (options.caseSensitive ? '' : 'i');
  return new RegExp(patternString, flags);
}

/**
 * Escapes special regex characters in a string
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Extracts context around a match
 */
function extractContext(
  text: string,
  matchIndex: number,
  matchLength: number,
  contextWords: number
): string {
  // Split text into words
  const words = text.split(/\s+/);
  
  // Find which word contains the match
  let charCount = 0;
  let matchWordIndex = 0;
  
  for (let i = 0; i < words.length; i++) {
    const wordLength = words[i].length + 1; // +1 for the space
    
    if (charCount <= matchIndex && matchIndex < charCount + wordLength) {
      matchWordIndex = i;
      break;
    }
    
    charCount += wordLength;
  }
  
  // Calculate start and end word indices for context
  const startWord = Math.max(0, matchWordIndex - contextWords);
  const endWord = Math.min(words.length, matchWordIndex + contextWords + 1);
  
  // Extract context words
  const contextArray = words.slice(startWord, endWord);
  
  return contextArray.join(' ');
}

/**
 * Predefined pattern collections for common use cases
 */
export const CommonPatterns = {
  Email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/,
  URL: /https?:\/\/[^\s]+/,
  Phone: /\b(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b/,
  Date: /\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b/,
  Time: /\b\d{1,2}:\d{2}(:\d{2})?(\s*[ap]m)?\b/i,
  CreditCard: /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/,
  SSN: /\b\d{3}[- ]?\d{2}[- ]?\d{4}\b/,
  IPAddress: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/,
  Currency: /\b[$€£¥]\s?\d+(\.\d{2})?\b/,
  Hashtag: /\B#[a-zA-Z0-9_]+\b/
};

/**
 * Pattern collections for specific domains
 */
export const DomainPatterns = {
  // Programming-related patterns
  Programming: {
    Function: /\b\w+\s*\([^)]*\)/,
    Variable: /\b[a-zA-Z_]\w*\b/,
    Comment: /\/\/.*|\/\*[\s\S]*?\*\//,
    ClassDefinition: /\bclass\s+\w+/i
  },
  
  // Medical patterns
  Medical: {
    BloodPressure: /\b\d{2,3}\/\d{2,3}\s*mm\s*Hg\b/i,
    Height: /\b\d{1,2}'\s*\d{1,2}"\b|\b\d{1,3}\s*cm\b/i,
    Weight: /\b\d{2,3}\s*(lbs|pounds|kg|kilograms)\b/i,
    Temperature: /\b\d{2,3}(\.\d)?\s*(°[CF]|degrees [CF])\b/i
  },
  
  // Financial patterns
  Financial: {
    StockTicker: /\b[A-Z]{1,5}\b/,
    Percentage: /\b\d+(\.\d+)?%\b/,
    MoneyAmount: /\b\$\d{1,3}(,\d{3})*(\.\d{2})?\b/
  }
};

/**
 * Advanced semantic pattern matching capabilities
 * These functions use more sophisticated techniques beyond regex
 */

/**
 * Finds semantic patterns in text (conceptual matches)
 * @param text - Text to analyze
 * @param concept - The conceptual pattern to find
 * @returns Array of matches
 */
export function findSemanticPatterns(
  text: string,
  concept: string
): PatternMatch[] {
  // This would ideally use NLP/ML techniques
  // Simple implementation just looks for keyword matches
  
  const conceptKeywords = concept.toLowerCase().split(/\s+/);
  const results: PatternMatch[] = [];
  
  // Split text into sentences
  const sentences = text.split(/[.!?]+\s+/);
  let currentIndex = 0;
  
  sentences.forEach(sentence => {
    const lowerSentence = sentence.toLowerCase();
    
    // Check if sentence contains enough concept keywords
    const matchedKeywords = conceptKeywords.filter(keyword => 
      lowerSentence.includes(keyword)
    );
    
    // If at least half the keywords match, consider it a match
    if (matchedKeywords.length >= Math.ceil(conceptKeywords.length / 2)) {
      results.push({
        pattern: concept,
        matched: sentence,
        index: currentIndex
      });
    }
    
    currentIndex += sentence.length + 2; // +2 for the ending punctuation and space
  });
  
  return results;
} 