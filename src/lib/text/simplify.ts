/**
 * Text Simplification Module
 * 
 * Simplifies sentences to improve readability and comprehension
 * for a wider audience.
 */

interface SimplificationOptions {
  targetReadingLevel?: number; // Target Flesch-Kincaid grade level (1-12)
  maxSentenceLength?: number; // Maximum words per sentence
  simplifyVocabulary?: boolean; // Replace complex words with simpler alternatives
  splitComplexSentences?: boolean; // Break complex sentences into multiple simpler ones
  preserveKeyTerms?: string[]; // Terms to preserve even if complex
  returnAnalysis?: boolean; // Include analysis with the result
}

interface SimplificationResult {
  original: string;
  simplified: string;
  analysis?: {
    originalGradeLevel: number;
    simplifiedGradeLevel: number;
    readabilityImprovement: number;
    wordsReplaced: number;
    sentencesSplit: number;
  };
}

/**
 * Simplifies sentences in text to improve readability
 * 
 * @param text - The text to simplify
 * @param options - Configuration options for simplification
 * @returns The simplified text or a detailed result object
 */
export function simplify_sentences(
  text: string,
  options?: Partial<SimplificationOptions>
): string | SimplificationResult {
  // Default options
  const opts: SimplificationOptions = {
    targetReadingLevel: 8, // 8th grade reading level
    maxSentenceLength: 20,
    simplifyVocabulary: true,
    splitComplexSentences: true,
    preserveKeyTerms: [],
    returnAnalysis: false,
    ...options
  };

  // Initialize metrics for analysis
  let wordsReplaced = 0;
  let sentencesSplit = 0;
  
  // Original text metrics
  const originalGradeLevel = estimateReadingLevel(text);
  
  // Process the text
  // 1. Split text into sentences
  let sentences = splitIntoSentences(text);
  
  // 2. Apply transformations to each sentence
  for (let i = 0; i < sentences.length; i++) {
    // Skip already simple sentences
    if (isSentenceSimpleEnough(sentences[i], opts.targetReadingLevel || 8)) {
      continue;
    }
    
    // 2a. Split complex sentences if enabled
    if (opts.splitComplexSentences) {
      const splitResult = splitComplexSentence(sentences[i], opts);
      if (splitResult.sentences.length > 1) {
        // Replace the original sentence with the split ones
        sentences.splice(i, 1, ...splitResult.sentences);
        sentencesSplit++;
        // Adjust loop index to account for inserted sentences
        i += splitResult.sentences.length - 1;
      }
    }
  }
  
  // 3. Simplify vocabulary in each sentence (even if already split)
  if (opts.simplifyVocabulary) {
    for (let i = 0; i < sentences.length; i++) {
      const simplificationResult = simplifyVocabulary(sentences[i], opts.preserveKeyTerms || []);
      sentences[i] = simplificationResult.simplified;
      wordsReplaced += simplificationResult.replacements;
    }
  }
  
  // 4. Ensure proper sentence length
  sentences = shortenLongSentences(sentences, opts.maxSentenceLength || 20);
  
  // Combine sentences back to text
  const simplified = sentences.join(' ');
  
  // Calculate simplified text metrics
  const simplifiedGradeLevel = estimateReadingLevel(simplified);
  
  // Return appropriate result
  if (opts.returnAnalysis) {
    return {
      original: text,
      simplified,
      analysis: {
        originalGradeLevel,
        simplifiedGradeLevel,
        readabilityImprovement: originalGradeLevel - simplifiedGradeLevel,
        wordsReplaced,
        sentencesSplit
      }
    };
  }
  
  return simplified;
}

/**
 * Splits text into individual sentences
 */
function splitIntoSentences(text: string): string[] {
  // Basic sentence splitting
  const rawSentences = text.split(/(?<=[.!?])\s+/);
  
  // Clean up and filter empty sentences
  return rawSentences
    .map(s => s.trim())
    .filter(s => s.length > 0);
}

/**
 * Estimates reading level using simplified Flesch-Kincaid grade formula
 */
function estimateReadingLevel(text: string): number {
  // Count syllables, words, and sentences
  const sentences = splitIntoSentences(text);
  const words = text.split(/\s+/).filter(w => w.length > 0);
  
  if (words.length === 0 || sentences.length === 0) {
    return 0;
  }
  
  // Count syllables (very basic estimation)
  let syllableCount = 0;
  words.forEach(word => {
    syllableCount += countSyllables(word);
  });
  
  // Calculate averages
  const asl = words.length / sentences.length; // average sentence length
  const asw = syllableCount / words.length; // average syllables per word
  
  // Simplified Flesch-Kincaid Grade Level formula
  const readingLevel = 0.39 * asl + 11.8 * asw - 15.59;
  
  // Clamp to reasonable range
  return Math.max(0, Math.min(16, Math.round(readingLevel * 10) / 10));
}

/**
 * Very basic syllable counter
 */
function countSyllables(word: string): number {
  word = word.toLowerCase().replace(/[^a-z]/g, '');
  if (word.length <= 3) return 1;
  
  // Count vowel groups
  const vowelGroups = word.match(/[aeiouy]{1,}/g);
  let count = vowelGroups ? vowelGroups.length : 1;
  
  // Adjust for common patterns
  if (word.endsWith('e') && !word.endsWith('le')) {
    count--;
  }
  
  return Math.max(1, count);
}

/**
 * Checks if a sentence is already simple enough
 */
function isSentenceSimpleEnough(sentence: string, targetLevel: number): boolean {
  const words = sentence.split(/\s+/).filter(w => w.length > 0);
  
  // Too few words to worry about
  if (words.length <= 8) return true;
  
  // Quick check based on sentence length and word complexity
  const complexWords = words.filter(w => w.length > 8 || countSyllables(w) >= 3);
  
  // If very few complex words and short sentence, it's simple enough
  if (complexWords.length <= 1 && words.length <= 15) return true;
  
  // Detailed check using reading level
  const sentenceLevel = estimateReadingLevel(sentence);
  return sentenceLevel <= targetLevel;
}

/**
 * Splits a complex sentence into multiple simpler ones
 */
function splitComplexSentence(
  sentence: string, 
  options: SimplificationOptions
): { sentences: string[], complexity: number } {
  const result: string[] = [];
  let complexity = 0;
  
  // Identify splitting points
  // Common conjunctions and transition phrases
  const splittingPatterns = [
    // Coordinating conjunctions
    /,\s*(and|but|or|so|for|nor|yet)\s+/i,
    // Subordinating conjunctions
    /;\s*/,
    // Relative clauses 
    /,\s*(which|who|whom|where|that)\s+/,
    // Other transition points
    /,\s*(however|therefore|moreover|furthermore|consequently|as a result)\s+/i
  ];
  
  // Check if the sentence has any candidate splitting points
  let hasSplitPoints = false;
  for (const pattern of splittingPatterns) {
    if (pattern.test(sentence)) {
      hasSplitPoints = true;
      break;
    }
  }
  
  // If no obvious split points or sentence is already short, return as is
  if (!hasSplitPoints || sentence.split(/\s+/).length <= options.maxSentenceLength!) {
    return { sentences: [sentence], complexity: estimateReadingLevel(sentence) };
  }
  
  // Choose the best splitting strategy based on the sentence structure
  let splits: string[] = [];
  
  // Try splitting by semicolons first (usually clean breaks)
  if (/;\s*/.test(sentence)) {
    splits = sentence.split(/;\s*/);
    // Cap each part with proper punctuation
    splits = splits.map(s => s.endsWith('.') ? s : s + '.');
  } 
  // Try splitting by coordinating conjunctions
  else if (/,\s*(and|but|or|so|for|nor|yet)\s+/i.test(sentence)) {
    // Find the conjunction to preserve it in the right split
    const match = sentence.match(/,\s*(and|but|or|so|for|nor|yet)\s+/i);
    if (match && match.index !== undefined) {
      const firstPart = sentence.substring(0, match.index);
      const secondPart = sentence.substring(match.index + 1).trim();
      
      // Extract the conjunction
      const conjMatch = secondPart.match(/^(and|but|or|so|for|nor|yet)\s+/i);
      const conjunction = conjMatch ? conjMatch[1] : '';
      
      // Create proper sentences
      splits = [
        firstPart.endsWith('.') ? firstPart : firstPart + '.',
        // Capitalize first letter after removing conjunction
        secondPart.substring(conjunction.length, 1).toUpperCase() + 
        secondPart.substring(conjunction.length + 1)
      ];
    } else {
      // Fallback if regex match behaves unexpectedly
      splits = [sentence];
    }
  }
  // Try splitting by relative clauses
  else if (/,\s*(which|who|whom|where|that)\s+/i.test(sentence)) {
    // This is more complex as we need to rewrite to eliminate the relative pronoun
    // Simple approach: replace with a period and "This" or "It"
    const match = sentence.match(/,\s*(which|who|whom|where|that)\s+/i);
    if (match && match.index !== undefined) {
      const firstPart = sentence.substring(0, match.index);
      let secondPart = sentence.substring(match.index + match[0].length);
      
      // Replace pronouns and adjust for proper sentence
      const pronoun = match[1].toLowerCase();
      if (pronoun === 'which' || pronoun === 'that') {
        secondPart = 'This ' + secondPart;
      } else if (pronoun === 'who' || pronoun === 'whom') {
        secondPart = 'They ' + secondPart;
      } else {
        secondPart = 'It ' + secondPart;
      }
      
      // Create proper sentences
      splits = [
        firstPart.endsWith('.') ? firstPart : firstPart + '.',
        secondPart.charAt(0).toUpperCase() + secondPart.slice(1)
      ];
    } else {
      // Fallback
      splits = [sentence];
    }
  }
  else {
    // If we can't find a good splitting strategy, leave as is
    splits = [sentence];
  }
  
  // Process each split to ensure they're proper sentences
  splits.forEach(split => {
    // Ensure proper capitalization and ending punctuation
    let processed = split.trim();
    processed = processed.charAt(0).toUpperCase() + processed.slice(1);
    
    if (!/[.!?]$/.test(processed)) {
      processed += '.';
    }
    
    result.push(processed);
  });
  
  // Calculate complexity of the most complex resulting sentence
  let maxComplexity = 0;
  result.forEach(s => {
    const level = estimateReadingLevel(s);
    maxComplexity = Math.max(maxComplexity, level);
  });
  complexity = maxComplexity;
  
  return { sentences: result, complexity };
}

/**
 * Simplifies vocabulary in a sentence by replacing complex words
 */
function simplifyVocabulary(
  sentence: string, 
  preserveTerms: string[] = []
): { simplified: string, replacements: number } {
  let replacements = 0;
  let words = sentence.split(/\s+/);
  
  // Process each word
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    const cleanWord = word.replace(/[^\w]|_/g, '').toLowerCase();
    
    // Skip short words, preserved terms, and words with punctuation attached
    if (cleanWord.length <= 6 || 
        countSyllables(cleanWord) <= 2 || 
        preserveTerms.includes(cleanWord) ||
        word !== cleanWord) {
      continue;
    }
    
    // Check if word is in our simplification dictionary
    const simpler = getSimpleAlternative(cleanWord);
    if (simpler && simpler !== cleanWord) {
      // Replace while preserving capitalization and punctuation
      const punctuation = word.match(/[^\w]|_/g) || [];
      const isCapitalized = word.charAt(0) === word.charAt(0).toUpperCase();
      
      let replacement = isCapitalized 
        ? simpler.charAt(0).toUpperCase() + simpler.slice(1) 
        : simpler;
      
      // Add back any punctuation
      punctuation.forEach(p => {
        if (word.startsWith(p)) replacement = p + replacement;
        if (word.endsWith(p)) replacement = replacement + p;
      });
      
      words[i] = replacement;
      replacements++;
    }
  }
  
  return { 
    simplified: words.join(' '),
    replacements
  };
}

/**
 * Returns a simpler alternative for a complex word
 */
function getSimpleAlternative(word: string): string {
  // Simple dictionary of complex words and their simpler alternatives
  const simplifications: Record<string, string> = {
    "utilize": "use",
    "implement": "use",
    "sufficient": "enough",
    "assist": "help",
    "obtain": "get",
    "purchase": "buy",
    "demonstrate": "show",
    "regarding": "about",
    "consequently": "so",
    "furthermore": "also",
    "therefore": "so",
    "however": "but",
    "nevertheless": "still",
    "require": "need",
    "additional": "more",
    "numerous": "many",
    "commence": "start",
    "terminate": "end",
    "inquire": "ask",
    "ascertain": "find out",
    "comprehend": "understand",
    "encounter": "meet",
    "endeavor": "try",
    "excessive": "too much",
    "beneficial": "helpful",
    "location": "place",
    "residence": "home",
    "initiate": "begin",
    "approximately": "about",
    "subsequently": "later",
    "prioritize": "focus on",
    "visualize": "see",
    "facilitate": "help",
    "indicate": "show",
    "modification": "change",
    "requirement": "need",
    "communicate": "talk",
    "attempt": "try",
    "currently": "now",
    "frequently": "often",
    "previously": "before",
    "occasionally": "sometimes",
    "initially": "first",
    "ultimately": "finally",
    "apparently": "seems like",
    "essentially": "mainly",
    "virtually": "almost"
  };
  
  return simplifications[word] || word;
}

/**
 * Ensures sentences don't exceed the maximum length by breaking them into shorter ones
 */
function shortenLongSentences(sentences: string[], maxLength: number): string[] {
  const result: string[] = [];
  
  sentences.forEach(sentence => {
    const words = sentence.split(/\s+/);
    
    // If sentence is short enough, add it as is
    if (words.length <= maxLength) {
      result.push(sentence);
      return;
    }
    
    // Break into chunks of maxLength words
    let currentChunk: string[] = [];
    let chunks: string[][] = [];
    
    for (let i = 0; i < words.length; i++) {
      currentChunk.push(words[i]);
      
      // When chunk reaches max length, look for good break point
      if (currentChunk.length >= maxLength) {
        // Try to find a natural break point (comma, etc.)
        let breakIndex = -1;
        for (let j = currentChunk.length - 1; j >= Math.max(currentChunk.length - 5, 0); j--) {
          if (currentChunk[j].includes(',') || 
              currentChunk[j].includes(';') || 
              currentChunk[j] === 'and' ||
              currentChunk[j] === 'but' ||
              currentChunk[j] === 'or') {
            breakIndex = j;
            break;
          }
        }
        
        if (breakIndex >= 0) {
          // Split at the natural break
          const newChunk = currentChunk.slice(0, breakIndex + 1);
          chunks.push(newChunk);
          currentChunk = currentChunk.slice(breakIndex + 1);
        } else {
          // No natural break found, split at max length
          chunks.push(currentChunk);
          currentChunk = [];
        }
      }
    }
    
    // Add any remaining words
    if (currentChunk.length > 0) {
      chunks.push(currentChunk);
    }
    
    // Convert chunks to proper sentences
    chunks.forEach((chunk, index) => {
      let chunkText = chunk.join(' ');
      
      // Replace ending punctuation of all but the last chunk
      if (index < chunks.length - 1) {
        chunkText = chunkText.replace(/[,;]$/, '.').replace(/[^.!?]$/, '$&.');
      }
      
      // Ensure proper capitalization
      chunkText = chunkText.charAt(0).toUpperCase() + chunkText.slice(1);
      
      result.push(chunkText);
    });
  });
  
  return result;
} 