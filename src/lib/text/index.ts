/**
 * Text Manipulation Library
 * 
 * Provides a comprehensive set of tools for text analysis,
 * manipulation, and enhancement.
 */

// Re-export all functions from individual modules
export { readability_score } from './readability';
export { ensure_explanation_follows } from './explanation';
export { simplify_sentences } from './simplify';
export { replace_jargon } from './jargon';
export { 
  contains, 
  extract_patterns, 
  CommonPatterns, 
  DomainPatterns,
  findSemanticPatterns
} from './patterns';

// Additional utility functions
export function capitalize(text: string): string {
  if (!text) return text;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function titleCase(text: string): string {
  if (!text) return text;
  return text
    .split(/\s+/)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

export function truncate(text: string, length: number, ellipsis = '...'): string {
  if (!text || text.length <= length) return text;
  return text.slice(0, length - ellipsis.length) + ellipsis;
}

export function wordCount(text: string): number {
  if (!text) return 0;
  return text.split(/\s+/).filter(word => word.length > 0).length;
}

export function sentenceCount(text: string): number {
  if (!text) return 0;
  const sentences = text.match(/[^.!?]+[.!?]+/g);
  return sentences ? sentences.length : 0;
}

export function paragraphCount(text: string): number {
  if (!text) return 0;
  return text.split(/\n\s*\n/).filter(p => p.trim().length > 0).length;
}

// String manipulation utilities
export function removeExtraWhitespace(text: string): string {
  if (!text) return text;
  return text.replace(/\s+/g, ' ').trim();
}

export function extractKeywords(text: string, count = 5): string[] {
  if (!text) return [];
  
  // Simple implementation - count word frequency and return most common
  const stopWords = [
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
    'by', 'about', 'as', 'of', 'that', 'this', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'it', 'its', 'their',
    'there', 'they', 'them', 'he', 'she', 'his', 'her', 'hers', 'him', 'we', 'us',
    'our', 'you', 'your', 'yours'
  ];
  
  // Split text into words, convert to lowercase, remove punctuation
  const words = text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(word => word.length > 2 && !stopWords.includes(word));
  
  // Count word frequency
  const wordCounts: Record<string, number> = {};
  words.forEach(word => {
    wordCounts[word] = (wordCounts[word] || 0) + 1;
  });
  
  // Sort by frequency
  return Object.entries(wordCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, count)
    .map(entry => entry[0]);
}

// Export the version
export const VERSION = '1.0.0'; 