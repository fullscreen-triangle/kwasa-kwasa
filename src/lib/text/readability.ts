/**
 * Readability Score Calculator
 * 
 * Calculates the readability of text using multiple formulas including:
 * - Flesch-Kincaid Reading Ease
 * - Flesch-Kincaid Grade Level
 * - Gunning Fog Index
 * - SMOG Index
 * - Coleman-Liau Index
 * 
 * Returns an object with scores and a composite score
 */

interface ReadabilityScores {
  fleschKincaidReadingEase: number;
  fleschKincaidGradeLevel: number;
  gunningFogIndex: number;
  smogIndex: number;
  colemanLiauIndex: number;
  automatedReadabilityIndex: number;
  averageGradeLevel: number;
  textStandard: string;
}

/**
 * Calculates various readability scores for the provided text
 * @param text - The text to analyze
 * @param options - Optional configuration options
 * @returns An object containing various readability metrics
 */
export function readability_score(text: string, options?: { 
  detailed?: boolean, 
  lang?: 'en' | 'fr' | 'es' | 'de'
}): ReadabilityScores | number {
  // Default options
  const opts = {
    detailed: false,
    lang: 'en',
    ...options
  };

  // Extract text statistics
  const stats = extractTextStatistics(text, opts.lang);
  
  // Calculate individual scores
  const fleschKincaidReadingEase = calculateFleschKincaidReadingEase(stats);
  const fleschKincaidGradeLevel = calculateFleschKincaidGradeLevel(stats);
  const gunningFogIndex = calculateGunningFogIndex(stats);
  const smogIndex = calculateSMOGIndex(stats);
  const colemanLiauIndex = calculateColemanLiauIndex(stats);
  const automatedReadabilityIndex = calculateAutomatedReadabilityIndex(stats);
  
  // Calculate average grade level (normalized)
  const gradeScores = [
    normalizeGradeLevel(fleschKincaidGradeLevel),
    normalizeGradeLevel(gunningFogIndex),
    normalizeGradeLevel(smogIndex),
    normalizeGradeLevel(colemanLiauIndex),
    normalizeGradeLevel(automatedReadabilityIndex)
  ];
  
  const averageGradeLevel = gradeScores.reduce((sum, score) => sum + score, 0) / gradeScores.length;
  
  // Determine text standard (grade level as text)
  const textStandard = gradeToTextStandard(averageGradeLevel);
  
  // Return detailed scores or just the composite score
  if (opts.detailed) {
    return {
      fleschKincaidReadingEase,
      fleschKincaidGradeLevel,
      gunningFogIndex,
      smogIndex,
      colemanLiauIndex,
      automatedReadabilityIndex,
      averageGradeLevel,
      textStandard
    };
  } else {
    // Return just the reading ease score (higher = more readable)
    return fleschKincaidReadingEase;
  }
}

/**
 * Extracts text statistics needed for readability calculations
 */
function extractTextStatistics(text: string, lang: string = 'en') {
  // Remove extra whitespace
  const cleanText = text.replace(/\s+/g, ' ').trim();
  
  // Count sentences
  // This is a simple implementation; a more sophisticated one would handle
  // abbreviations like "Dr.", "Mrs.", etc.
  const sentenceCount = countSentences(cleanText);
  
  // Count words
  const words = cleanText.split(/\s+/).filter(word => word.length > 0);
  const wordCount = words.length;
  
  // Count syllables
  const syllableCount = words.reduce((count, word) => count + countSyllables(word, lang), 0);
  
  // Count complex words (3+ syllables)
  const complexWords = words.filter(word => countSyllables(word, lang) >= 3);
  const complexWordCount = complexWords.length;
  
  // Count characters
  const charCount = cleanText.replace(/\s/g, '').length;
  
  // Calculate averages
  const avgSentenceLength = wordCount / Math.max(1, sentenceCount);
  const avgSyllablesPerWord = syllableCount / Math.max(1, wordCount);
  const percentComplexWords = complexWordCount / Math.max(1, wordCount) * 100;
  
  return {
    text: cleanText,
    sentenceCount,
    wordCount,
    syllableCount,
    complexWordCount,
    charCount,
    avgSentenceLength,
    avgSyllablesPerWord,
    percentComplexWords
  };
}

/**
 * Counts sentences in text using regex for common sentence endings
 */
function countSentences(text: string): number {
  // Count by sentence-ending punctuation
  const sentenceEnds = text.match(/[.!?]+(\s|$)/g);
  return sentenceEnds ? sentenceEnds.length : 1;
}

/**
 * Counts syllables in a word
 */
function countSyllables(word: string, lang: string = 'en'): number {
  // Strip punctuation
  const cleanWord = word.toLowerCase().replace(/[.,?!;:()\-'"]/g, '');
  
  if (cleanWord.length <= 0) {
    return 0;
  }
  
  // Language-specific syllable counting
  switch (lang) {
    case 'en':
      return countEnglishSyllables(cleanWord);
    // Additional languages could be implemented here
    default:
      return countEnglishSyllables(cleanWord);
  }
}

/**
 * Counts syllables using English language rules
 */
function countEnglishSyllables(word: string): number {
  // Edge case - empty string
  if (word.length === 0) return 0;
  
  // Rule: Single-letter words
  if (word.length === 1) return 1;
  
  // Lower case and remove trailing e (silent e rule)
  let adjusted = word.toLowerCase();
  
  // Remove common suffixes for counting
  adjusted = adjusted.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
  adjusted = adjusted.replace(/^y/, '');
  
  // Count vowel groups as syllables
  const syllableCount = adjusted.match(/[aeiouy]{1,2}/g);
  
  // Ensure at least one syllable is counted
  return syllableCount ? syllableCount.length : 1;
}

/**
 * Calculates Flesch-Kincaid Reading Ease
 * Higher scores indicate text that is easier to read
 * 90-100: Very easy, 80-89: Easy, 70-79: Fairly easy, 
 * 60-69: Standard, 50-59: Fairly difficult, 30-49: Difficult, 0-29: Very confusing
 */
function calculateFleschKincaidReadingEase(stats: any): number {
  const score = 206.835 - (1.015 * stats.avgSentenceLength) - (84.6 * stats.avgSyllablesPerWord);
  return Math.max(0, Math.min(100, Math.round(score * 10) / 10));
}

/**
 * Calculates Flesch-Kincaid Grade Level
 * Returns approximate US grade level needed to comprehend the text
 */
function calculateFleschKincaidGradeLevel(stats: any): number {
  const score = (0.39 * stats.avgSentenceLength) + (11.8 * stats.avgSyllablesPerWord) - 15.59;
  return Math.max(0, Math.round(score * 10) / 10);
}

/**
 * Calculates Gunning Fog Index
 * Estimates years of formal education needed to understand text on first reading
 */
function calculateGunningFogIndex(stats: any): number {
  const score = 0.4 * (stats.avgSentenceLength + stats.percentComplexWords);
  return Math.max(0, Math.round(score * 10) / 10);
}

/**
 * Calculates SMOG Index
 * Simple Measure of Gobbledygook - rough estimate of years of education needed
 */
function calculateSMOGIndex(stats: any): number {
  if (stats.sentenceCount < 30) {
    // SMOG generally requires 30+ sentences for accuracy
    // This is a modified formula for shorter texts
    const score = 1.043 * Math.sqrt(stats.complexWordCount * (30 / stats.sentenceCount)) + 3.1291;
    return Math.max(0, Math.round(score * 10) / 10);
  }
  
  const score = 1.043 * Math.sqrt(stats.complexWordCount * (30 / stats.sentenceCount)) + 3.1291;
  return Math.max(0, Math.round(score * 10) / 10);
}

/**
 * Calculates Coleman-Liau Index
 * Based on characters rather than syllables
 */
function calculateColemanLiauIndex(stats: any): number {
  const L = (stats.charCount / stats.wordCount) * 100; // Avg number of characters per 100 words
  const S = (stats.sentenceCount / stats.wordCount) * 100; // Avg number of sentences per 100 words
  const score = 0.0588 * L - 0.296 * S - 15.8;
  return Math.max(0, Math.round(score * 10) / 10);
}

/**
 * Calculates Automated Readability Index
 * Based on characters per word and words per sentence
 */
function calculateAutomatedReadabilityIndex(stats: any): number {
  const score = 4.71 * (stats.charCount / stats.wordCount) + 0.5 * stats.avgSentenceLength - 21.43;
  return Math.max(0, Math.round(score * 10) / 10);
}

/**
 * Normalizes different grade level metrics to a consistent scale
 */
function normalizeGradeLevel(score: number): number {
  // Clamp between 1-16 (1st grade through college senior)
  return Math.max(1, Math.min(16, score));
}

/**
 * Converts numerical grade level to text representation
 */
function gradeToTextStandard(grade: number): string {
  // Round to nearest 0.5
  const roundedGrade = Math.round(grade * 2) / 2;
  
  if (roundedGrade <= 1) return '1st Grade';
  if (roundedGrade <= 2) return '2nd Grade';
  if (roundedGrade <= 3) return '3rd Grade';
  if (roundedGrade <= 12) return `${Math.floor(roundedGrade)}th Grade`;
  if (roundedGrade <= 13) return 'College Freshman';
  if (roundedGrade <= 14) return 'College Sophomore';
  if (roundedGrade <= 15) return 'College Junior';
  return 'College Senior or Graduate Level';
} 