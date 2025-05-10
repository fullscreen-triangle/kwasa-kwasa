/**
 * Jargon Replacement Module
 * 
 * Identifies and replaces technical jargon with more accessible terminology
 * for improved clarity and readability.
 */

interface JargonOptions {
  domains?: string[]; // Specific domains to target for jargon replacement
  preserveTerms?: string[]; // Terms to preserve even if they're jargon
  replacementStyle?: 'replace' | 'annotate' | 'define'; // How to handle replacements
  includeDefinitions?: boolean; // Whether to include definitions for first use
  targetAudience?: 'general' | 'beginner' | 'intermediate'; // Who the text is for
  aggressiveness?: number; // 0-1 scale for how aggressively to replace
}

interface JargonResult {
  original: string;
  simplified: string;
  replacements: {
    original: string;
    replacement: string;
    definition?: string;
    occurrences: number;
  }[];
}

/**
 * Replaces technical jargon with more accessible alternatives
 * 
 * @param text - The text containing jargon to replace
 * @param options - Configuration options for jargon replacement
 * @returns Either the simplified text or a detailed result object
 */
export function replace_jargon(
  text: string,
  options?: Partial<JargonOptions>
): string | JargonResult {
  // Default options
  const opts: JargonOptions = {
    domains: ['general'],
    preserveTerms: [],
    replacementStyle: 'replace',
    includeDefinitions: true,
    targetAudience: 'general',
    aggressiveness: 0.7, // Higher = more aggressive replacement
    ...options
  };

  // Track replacements for reporting
  const replacementMap: Map<string, { 
    replacement: string; 
    definition?: string; 
    occurrences: number;
    isFirstOccurrence: boolean;
  }> = new Map();
  
  // Process jargon domain dictionaries
  const jargonDictionaries = loadJargonDictionaries(opts.domains || ['general']);
  
  // Prepare regex pattern for matching
  const allJargonTerms = Object.keys(jargonDictionaries)
    .filter(term => !opts.preserveTerms?.includes(term.toLowerCase()));
    
  // No jargon to replace
  if (allJargonTerms.length === 0) {
    return text;
  }
  
  // Sort by length (longest first) to prevent partial replacements
  allJargonTerms.sort((a, b) => b.length - a.length);
  
  // Generate regex for word boundaries
  const jargonPattern = new RegExp(`\\b(${allJargonTerms.map(escapeRegExp).join('|')})\\b`, 'gi');
  
  // Process the text
  let simplified = text;
  let match;
  
  // Reset regex each time
  jargonPattern.lastIndex = 0;
  
  // First pass: identify all jargon terms and track them
  while ((match = jargonPattern.exec(text)) !== null) {
    const term = match[0];
    const lowerTerm = term.toLowerCase();
    
    // Skip if in preserve list
    if (opts.preserveTerms?.includes(lowerTerm)) {
      continue;
    }
    
    // Find the replacement based on the term
    const replacementInfo = jargonDictionaries[lowerTerm];
    if (!replacementInfo) continue;
    
    // Skip replacement if below aggressiveness threshold
    // More common/less specialized terms only replaced at higher aggressiveness
    if (replacementInfo.specialization < (opts.aggressiveness ?? 0.7)) {
      continue;
    }
    
    // Get the appropriate replacement based on audience
    let replacement: string;
    switch (opts.targetAudience) {
      case 'beginner':
        replacement = replacementInfo.beginnerAlternative || replacementInfo.alternative;
        break;
      case 'intermediate':
        replacement = replacementInfo.intermediateAlternative || replacementInfo.alternative;
        break;
      default:
        replacement = replacementInfo.alternative;
    }
    
    // Track this replacement
    if (replacementMap.has(lowerTerm)) {
      const info = replacementMap.get(lowerTerm)!;
      info.occurrences++;
    } else {
      replacementMap.set(lowerTerm, {
        replacement,
        definition: replacementInfo.definition,
        occurrences: 1,
        isFirstOccurrence: true
      });
    }
  }
  
  // Second pass: apply replacements
  // We need a fresh regex instance
  const replacementPattern = new RegExp(`\\b(${allJargonTerms.map(escapeRegExp).join('|')})\\b`, 'gi');
  
  // Apply replacements based on style
  simplified = text.replace(replacementPattern, (match) => {
    const lowerMatch = match.toLowerCase();
    
    // Skip if in preserve list or not in our map
    if (opts.preserveTerms?.includes(lowerMatch) || !replacementMap.has(lowerMatch)) {
      return match;
    }
    
    const info = replacementMap.get(lowerMatch)!;
    const isFirstOccurrence = info.isFirstOccurrence;
    
    // Mark as no longer first occurrence for next time
    if (isFirstOccurrence) {
      info.isFirstOccurrence = false;
    }
    
    // Handle case preservation
    const replacement = preserveCase(match, info.replacement);
    
    // Apply different replacement styles
    switch (opts.replacementStyle) {
      case 'annotate':
        return isFirstOccurrence && opts.includeDefinitions && info.definition
          ? `${replacement} (${match})`
          : replacement;
          
      case 'define':
        return isFirstOccurrence && opts.includeDefinitions && info.definition
          ? `${replacement} (${info.definition})`
          : replacement;
          
      case 'replace':
      default:
        return replacement;
    }
  });
  
  // Prepare result
  if (opts.replacementStyle !== 'replace' || options?.domains) {
    // Detailed result with replacements info
    const replacements = Array.from(replacementMap.entries()).map(([original, info]) => ({
      original,
      replacement: info.replacement,
      definition: info.definition,
      occurrences: info.occurrences
    }));
    
    return {
      original: text,
      simplified,
      replacements
    };
  }
  
  // Just return the simplified text
  return simplified;
}

/**
 * Escapes special characters in strings for regex
 */
function escapeRegExp(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Preserves the case pattern from original word to replacement
 */
function preserveCase(original: string, replacement: string): string {
  // All uppercase
  if (original === original.toUpperCase()) {
    return replacement.toUpperCase();
  }
  
  // First letter uppercase
  if (original[0] === original[0].toUpperCase() && original.slice(1) === original.slice(1).toLowerCase()) {
    return replacement.charAt(0).toUpperCase() + replacement.slice(1).toLowerCase();
  }
  
  // Default: use replacement as is
  return replacement;
}

/**
 * Loads jargon dictionaries based on selected domains
 */
function loadJargonDictionaries(domains: string[]): Record<string, {
  alternative: string;
  beginnerAlternative?: string;
  intermediateAlternative?: string;
  definition?: string;
  specialization: number; // 0-1 scale, 1 = very specialized
}> {
  // Combined dictionary across all selected domains
  const combinedDictionary: Record<string, {
    alternative: string;
    beginnerAlternative?: string;
    intermediateAlternative?: string;
    definition?: string;
    specialization: number;
  }> = {};
  
  // General terms (cross-domain)
  const generalJargon: Record<string, any> = {
    "utilize": {
      alternative: "use",
      specialization: 0.3,
      definition: "to make use of"
    },
    "leverage": {
      alternative: "use",
      definition: "to use something to maximum advantage",
      specialization: 0.4
    },
    "facilitate": {
      alternative: "help",
      intermediateAlternative: "make easier",
      definition: "to make an action or process easier",
      specialization: 0.5
    },
    "paradigm": {
      alternative: "model",
      beginnerAlternative: "way of thinking",
      definition: "a typical example or pattern of something",
      specialization: 0.8
    },
    "synergy": {
      alternative: "combined effect",
      definition: "interaction of multiple elements producing a combined effect greater than the sum of their separate effects",
      specialization: 0.7
    },
    "robust": {
      alternative: "strong",
      definition: "able to withstand or overcome adverse conditions",
      specialization: 0.4
    }
  };
  
  // Technical/Computer terms
  const techJargon: Record<string, any> = {
    "algorithm": {
      alternative: "procedure",
      beginnerAlternative: "step-by-step process",
      definition: "a process or set of rules to be followed in calculations or problem-solving",
      specialization: 0.6
    },
    "api": {
      alternative: "interface",
      beginnerAlternative: "connection method",
      definition: "Application Programming Interface - a way for different programs to communicate",
      specialization: 0.7
    },
    "backend": {
      alternative: "server-side",
      beginnerAlternative: "behind-the-scenes processing",
      definition: "the part of software not directly accessed by users, responsible for data processing",
      specialization: 0.6
    },
    "frontend": {
      alternative: "user interface",
      beginnerAlternative: "what you see on screen",
      definition: "the part of software users directly interact with",
      specialization: 0.6
    },
    "cache": {
      alternative: "temporary storage",
      definition: "a collection of data stored temporarily for quick access",
      specialization: 0.7
    },
    "framework": {
      alternative: "structure",
      beginnerAlternative: "pre-built system",
      definition: "a basic structure underlying a system or concept",
      specialization: 0.5
    },
    "latency": {
      alternative: "delay",
      definition: "the delay before a transfer of data begins following an instruction",
      specialization: 0.7
    },
    "bandwidth": {
      alternative: "data capacity",
      beginnerAlternative: "connection speed",
      definition: "the maximum rate of data transfer across a given path",
      specialization: 0.6
    }
  };
  
  // Medical jargon
  const medicalJargon: Record<string, any> = {
    "hypertension": {
      alternative: "high blood pressure",
      definition: "abnormally high blood pressure in the arteries",
      specialization: 0.8
    },
    "myocardial infarction": {
      alternative: "heart attack",
      definition: "the death of heart muscle due to blocked blood flow",
      specialization: 0.9
    },
    "benign": {
      alternative: "not harmful",
      definition: "not cancerous, not likely to spread or get worse",
      specialization: 0.7
    },
    "malignant": {
      alternative: "cancerous",
      definition: "tending to spread and worsen, often referring to cancer",
      specialization: 0.8
    },
    "prognosis": {
      alternative: "outlook",
      definition: "the likely course of a medical condition",
      specialization: 0.6
    },
    "edema": {
      alternative: "swelling",
      definition: "excess fluid in body tissues causing swelling",
      specialization: 0.8
    }
  };
  
  // Legal jargon
  const legalJargon: Record<string, any> = {
    "pursuant to": {
      alternative: "according to",
      beginnerAlternative: "following",
      definition: "in accordance with or following",
      specialization: 0.8
    },
    "herein": {
      alternative: "in this document",
      definition: "in this document or statement",
      specialization: 0.7
    },
    "aforementioned": {
      alternative: "previously mentioned",
      definition: "referring to something mentioned earlier in the document",
      specialization: 0.7
    },
    "tort": {
      alternative: "wrongful act",
      definition: "a wrongful act leading to civil legal liability",
      specialization: 0.9
    },
    "litigation": {
      alternative: "legal process",
      beginnerAlternative: "lawsuit",
      definition: "the process of taking legal action",
      specialization: 0.7
    },
    "statute": {
      alternative: "law",
      definition: "a written law passed by a legislative body",
      specialization: 0.6
    }
  };
  
  // Add general jargon dictionary
  Object.assign(combinedDictionary, generalJargon);
  
  // Add domain-specific dictionaries
  domains.forEach(domain => {
    switch (domain.toLowerCase()) {
      case 'tech':
      case 'technology':
      case 'computer':
      case 'programming':
        Object.assign(combinedDictionary, techJargon);
        break;
        
      case 'medical':
      case 'healthcare':
      case 'medicine':
        Object.assign(combinedDictionary, medicalJargon);
        break;
        
      case 'legal':
      case 'law':
        Object.assign(combinedDictionary, legalJargon);
        break;
        
      // Default to general, which is already added
    }
  });
  
  return combinedDictionary;
} 