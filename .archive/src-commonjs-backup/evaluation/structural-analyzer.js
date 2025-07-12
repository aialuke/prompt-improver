/**
 * Structural Analysis Engine
 * Automated quality measurement through prompt structure analysis
 */

class StructuralAnalyzer {
  constructor(config = {}) {
    this.config = {
      // Quality dimension weights
      qualityWeights: {
        clarity: 0.25,        // How clear and specific the prompt is
        completeness: 0.25,   // How complete the requirements are
        specificity: 0.25,    // How specific vs vague the instructions are
        structure: 0.25       // How well organized the prompt is
      },

      // Scoring thresholds
      thresholds: {
        excellent: 0.8,
        good: 0.6,
        fair: 0.4,
        poor: 0.2
      },

      // Text analysis parameters
      analysis: {
        minWordCount: 5,
        maxWordCount: 1000,
        sentenceThresholds: {
          short: 10,
          medium: 20,
          long: 30
        },
        readabilityTarget: 12 // Grade level target
      },

      ...config
    };

    // Initialize pattern libraries
    this.patterns = this.initializePatterns();
    this.stopWords = this.initializeStopWords();
  }

  /**
   * Analyze prompt structure and return quality scores
   * @param {string} prompt - Prompt text to analyze
   * @param {Object} context - Optional context for analysis
   * @returns {Object} Structural analysis results
   */
  async analyzePrompt(prompt, context = {}) {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Invalid prompt: must be a non-empty string');
    }

    const analysis = {
      prompt: prompt.trim(),
      timestamp: new Date().toISOString(),
      context: context,
      
      // Core metrics
      textMetrics: this.analyzeTextMetrics(prompt),
      clarityScore: this.analyzeClarityScore(prompt),
      completenessScore: this.analyzeCompletenessScore(prompt),
      specificityScore: this.analyzeSpecificityScore(prompt),
      structureScore: this.analyzeStructureScore(prompt),
      
      // Derived metrics
      readabilityScore: this.calculateReadabilityScore(prompt),
      actionabilityScore: this.analyzeActionabilityScore(prompt),
      ambiguityScore: this.analyzeAmbiguityScore(prompt),
      
      // Overall quality
      overallQuality: 0,
      qualityGrade: '',
      
      // Detailed analysis
      issues: [],
      suggestions: [],
      strengths: []
    };

    // Calculate overall quality score
    analysis.overallQuality = this.calculateOverallQuality(analysis);
    analysis.qualityGrade = this.getQualityGrade(analysis.overallQuality);

    // Generate detailed feedback
    analysis.issues = this.identifyIssues(analysis);
    analysis.suggestions = this.generateSuggestions(analysis);
    analysis.strengths = this.identifyStrengths(analysis);

    return analysis;
  }

  /**
   * Analyze basic text metrics
   * @param {string} prompt - Prompt text
   * @returns {Object} Text metrics
   */
  analyzeTextMetrics(prompt) {
    const words = this.getWords(prompt);
    const sentences = this.getSentences(prompt);
    const characters = prompt.length;
    const charactersNoSpaces = prompt.replace(/\s/g, '').length;

    return {
      wordCount: words.length,
      sentenceCount: sentences.length,
      characterCount: characters,
      characterCountNoSpaces: charactersNoSpaces,
      averageWordsPerSentence: sentences.length > 0 ? words.length / sentences.length : 0,
      averageSyllablesPerWord: this.calculateAverageSyllables(words),
      uniqueWordRatio: this.calculateUniqueWordRatio(words),
      complexWordCount: this.countComplexWords(words),
      
      // Structure indicators
      questionCount: (prompt.match(/\?/g) || []).length,
      exclamationCount: (prompt.match(/!/g) || []).length,
      listIndicators: this.countListIndicators(prompt),
      emphasisMarkers: this.countEmphasisMarkers(prompt)
    };
  }

  /**
   * Analyze clarity score
   * @param {string} prompt - Prompt text
   * @returns {number} Clarity score (0-1)
   */
  analyzeClarityScore(prompt) {
    let score = 1.0;
    const words = this.getWords(prompt);
    const sentences = this.getSentences(prompt);

    // Penalty for vague words
    const vagueWords = this.countVagueWords(words);
    score -= Math.min(vagueWords * 0.1, 0.4);

    // Penalty for overly complex sentences
    const avgWordsPerSentence = sentences.length > 0 ? words.length / sentences.length : 0;
    if (avgWordsPerSentence > 25) {
      score -= Math.min((avgWordsPerSentence - 25) * 0.02, 0.3);
    }

    // Penalty for ambiguous pronouns without clear referents
    const ambiguousPronouns = this.countAmbiguousPronouns(prompt);
    score -= Math.min(ambiguousPronouns * 0.05, 0.2);

    // Bonus for specific technical terms (in context)
    const technicalTerms = this.countTechnicalTerms(words);
    score += Math.min(technicalTerms * 0.02, 0.1);

    // Penalty for contradictory statements
    const contradictions = this.detectContradictions(prompt);
    score -= Math.min(contradictions * 0.1, 0.2);

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Analyze completeness score
   * @param {string} prompt - Prompt text
   * @returns {number} Completeness score (0-1)
   */
  analyzeCompletenessScore(prompt) {
    let score = 0.0;
    const elements = {
      hasTask: this.hasTaskDescription(prompt),
      hasContext: this.hasContext(prompt),
      hasFormat: this.hasOutputFormat(prompt),
      hasConstraints: this.hasConstraints(prompt),
      hasExamples: this.hasExamples(prompt),
      hasSuccessCriteria: this.hasSuccessCriteria(prompt)
    };

    // Base score from essential elements
    if (elements.hasTask) score += 0.4;
    if (elements.hasContext) score += 0.2;
    if (elements.hasFormat) score += 0.15;
    if (elements.hasConstraints) score += 0.1;
    if (elements.hasExamples) score += 0.1;
    if (elements.hasSuccessCriteria) score += 0.05;

    // Bonus for comprehensive prompts
    const completedElements = Object.values(elements).filter(Boolean).length;
    if (completedElements >= 5) score += 0.1;

    // Penalty for missing critical elements
    if (!elements.hasTask) score = Math.min(score, 0.3);

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Analyze specificity score
   * @param {string} prompt - Prompt text
   * @returns {number} Specificity score (0-1)
   */
  analyzeSpecificityScore(prompt) {
    let score = 0.5; // Start at middle
    const words = this.getWords(prompt);

    // Count specific vs vague terms
    const specificTerms = this.countSpecificTerms(words);
    const vagueTerms = this.countVagueWords(words);
    
    // Adjust based on specificity ratio
    const totalRelevantTerms = specificTerms + vagueTerms;
    if (totalRelevantTerms > 0) {
      const specificityRatio = specificTerms / totalRelevantTerms;
      score = specificityRatio;
    }

    // Bonus for quantitative details
    const quantities = this.countQuantitativeDetails(prompt);
    score += Math.min(quantities * 0.05, 0.2);

    // Bonus for concrete examples
    const concreteExamples = this.countConcreteExamples(prompt);
    score += Math.min(concreteExamples * 0.1, 0.2);

    // Penalty for placeholder text
    const placeholders = this.countPlaceholders(prompt);
    score -= Math.min(placeholders * 0.1, 0.3);

    // Bonus for domain-specific terminology
    const domainTerms = this.countDomainSpecificTerms(words);
    score += Math.min(domainTerms * 0.02, 0.1);

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Analyze structure score
   * @param {string} prompt - Prompt text
   * @returns {number} Structure score (0-1)
   */
  analyzeStructureScore(prompt) {
    let score = 0.0;

    // Check for logical flow
    const hasIntroduction = this.hasIntroduction(prompt);
    const hasMainContent = this.hasMainContent(prompt);
    const hasConclusion = this.hasConclusion(prompt);

    if (hasIntroduction) score += 0.2;
    if (hasMainContent) score += 0.4;
    if (hasConclusion) score += 0.2;

    // Check for organization markers
    const organizationMarkers = this.countOrganizationMarkers(prompt);
    score += Math.min(organizationMarkers * 0.05, 0.2);

    // Check for consistent formatting
    const consistentFormatting = this.hasConsistentFormatting(prompt);
    if (consistentFormatting) score += 0.1;

    // Penalty for run-on structure
    const isRunOn = this.isRunOnStructure(prompt);
    if (isRunOn) score -= 0.2;

    // Bonus for step-by-step structure
    const hasStepStructure = this.hasStepByStepStructure(prompt);
    if (hasStepStructure) score += 0.1;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Calculate readability score using Flesch Reading Ease
   * @param {string} prompt - Prompt text
   * @returns {number} Readability score (0-100)
   */
  calculateReadabilityScore(prompt) {
    const words = this.getWords(prompt);
    const sentences = this.getSentences(prompt);
    const syllables = words.reduce((total, word) => total + this.countSyllables(word), 0);

    if (sentences.length === 0 || words.length === 0) return 0;

    const avgSentenceLength = words.length / sentences.length;
    const avgSyllablesPerWord = syllables / words.length;

    // Flesch Reading Ease formula
    const score = 206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord);
    
    return Math.max(0, Math.min(100, score));
  }

  /**
   * Analyze actionability score
   * @param {string} prompt - Prompt text
   * @returns {number} Actionability score (0-1)
   */
  analyzeActionabilityScore(prompt) {
    let score = 0.0;

    // Count action verbs
    const actionVerbs = this.countActionVerbs(prompt);
    score += Math.min(actionVerbs * 0.1, 0.4);

    // Check for imperative mood
    const hasImperativeMood = this.hasImperativeMood(prompt);
    if (hasImperativeMood) score += 0.2;

    // Check for clear deliverables
    const hasDeliverables = this.hasDeliverables(prompt);
    if (hasDeliverables) score += 0.2;

    // Check for step-by-step instructions
    const hasSteps = this.hasStepByStepStructure(prompt);
    if (hasSteps) score += 0.2;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Analyze ambiguity score (lower is better)
   * @param {string} prompt - Prompt text
   * @returns {number} Ambiguity score (0-1, where 0 is unambiguous)
   */
  analyzeAmbiguityScore(prompt) {
    let score = 0.0;
    const words = this.getWords(prompt);

    // Count ambiguous terms
    const ambiguousTerms = this.countAmbiguousTerms(words);
    score += Math.min(ambiguousTerms * 0.05, 0.3);

    // Count modal verbs that create uncertainty
    const modalVerbs = this.countModalVerbs(words);
    score += Math.min(modalVerbs * 0.03, 0.2);

    // Count subjective adjectives
    const subjectiveAdjectives = this.countSubjectiveAdjectives(words);
    score += Math.min(subjectiveAdjectives * 0.02, 0.2);

    // Penalty for multiple interpretations
    const interpretations = this.countPossibleInterpretations(prompt);
    score += Math.min((interpretations - 1) * 0.1, 0.3);

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Calculate overall quality score
   * @param {Object} analysis - Analysis results
   * @returns {number} Overall quality score (0-1)
   */
  calculateOverallQuality(analysis) {
    const weights = this.config.qualityWeights;
    
    return (
      analysis.clarityScore * weights.clarity +
      analysis.completenessScore * weights.completeness +
      analysis.specificityScore * weights.specificity +
      analysis.structureScore * weights.structure
    );
  }

  /**
   * Get quality grade from score
   * @param {number} score - Quality score (0-1)
   * @returns {string} Quality grade
   */
  getQualityGrade(score) {
    const thresholds = this.config.thresholds;
    
    if (score >= thresholds.excellent) return 'Excellent';
    if (score >= thresholds.good) return 'Good';
    if (score >= thresholds.fair) return 'Fair';
    if (score >= thresholds.poor) return 'Poor';
    return 'Very Poor';
  }

  /**
   * Helper method: Get words from text
   * @param {string} text - Input text
   * @returns {Array} Array of words
   */
  getWords(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  /**
   * Helper method: Get sentences from text
   * @param {string} text - Input text
   * @returns {Array} Array of sentences
   */
  getSentences(text) {
    return text.split(/[.!?]+/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  /**
   * Helper method: Count syllables in a word
   * @param {string} word - Word to analyze
   * @returns {number} Syllable count
   */
  countSyllables(word) {
    word = word.toLowerCase();
    if (word.length <= 3) return 1;
    
    const vowels = 'aeiouy';
    let count = 0;
    let previousWasVowel = false;
    
    for (let i = 0; i < word.length; i++) {
      const isVowel = vowels.includes(word[i]);
      if (isVowel && !previousWasVowel) {
        count++;
      }
      previousWasVowel = isVowel;
    }
    
    // Adjust for silent e
    if (word.endsWith('e')) count--;
    
    return Math.max(1, count);
  }

  /**
   * Helper method: Calculate average syllables per word
   * @param {Array} words - Array of words
   * @returns {number} Average syllables per word
   */
  calculateAverageSyllables(words) {
    if (words.length === 0) return 0;
    
    const totalSyllables = words.reduce((total, word) => total + this.countSyllables(word), 0);
    return totalSyllables / words.length;
  }

  /**
   * Helper method: Calculate unique word ratio
   * @param {Array} words - Array of words
   * @returns {number} Ratio of unique words (0-1)
   */
  calculateUniqueWordRatio(words) {
    if (words.length === 0) return 0;
    
    const uniqueWords = new Set(words);
    return uniqueWords.size / words.length;
  }

  /**
   * Helper method: Count complex words (3+ syllables)
   * @param {Array} words - Array of words
   * @returns {number} Count of complex words
   */
  countComplexWords(words) {
    return words.filter(word => this.countSyllables(word) >= 3).length;
  }

  /**
   * Initialize pattern libraries for analysis
   * @returns {Object} Pattern libraries
   */
  initializePatterns() {
    return {
      vague: [
        'thing', 'stuff', 'something', 'anything', 'everything', 'nothing',
        'good', 'bad', 'nice', 'great', 'awesome', 'cool', 'interesting',
        'some', 'many', 'few', 'several', 'various', 'different',
        'big', 'small', 'large', 'huge', 'tiny', 'massive',
        'important', 'relevant', 'significant', 'appropriate', 'suitable'
      ],
      
      specific: [
        'implement', 'create', 'build', 'design', 'develop', 'write',
        'analyze', 'evaluate', 'measure', 'calculate', 'determine',
        'optimize', 'refactor', 'debug', 'test', 'validate',
        'integrate', 'deploy', 'configure', 'install', 'setup'
      ],
      
      action: [
        'create', 'build', 'make', 'develop', 'write', 'code', 'implement',
        'design', 'plan', 'analyze', 'review', 'test', 'debug', 'fix',
        'optimize', 'improve', 'enhance', 'refactor', 'update', 'modify',
        'install', 'configure', 'setup', 'deploy', 'integrate', 'connect'
      ],
      
      technical: [
        'function', 'class', 'method', 'variable', 'parameter', 'argument',
        'api', 'endpoint', 'database', 'table', 'query', 'schema',
        'component', 'module', 'package', 'library', 'framework',
        'algorithm', 'data structure', 'interface', 'protocol'
      ],
      
      ambiguous: [
        'this', 'that', 'these', 'those', 'it', 'they', 'them',
        'might', 'could', 'should', 'would', 'maybe', 'perhaps',
        'probably', 'possibly', 'likely', 'seems', 'appears'
      ],
      
      modal: [
        'can', 'could', 'may', 'might', 'must', 'shall', 'should',
        'will', 'would', 'ought'
      ],
      
      subjective: [
        'beautiful', 'ugly', 'nice', 'good', 'bad', 'better', 'worse',
        'best', 'worst', 'amazing', 'terrible', 'wonderful', 'awful',
        'excellent', 'poor', 'great', 'horrible', 'fantastic', 'dreadful'
      ],
      
      organization: [
        'first', 'second', 'third', 'next', 'then', 'finally', 'lastly',
        'before', 'after', 'during', 'while', 'meanwhile', 'however',
        'therefore', 'thus', 'consequently', 'furthermore', 'moreover'
      ],
      
      quantitative: [
        'exactly', 'precisely', 'approximately', 'about', 'around',
        'minimum', 'maximum', 'at least', 'no more than', 'between'
      ]
    };
  }

  /**
   * Initialize stop words list
   * @returns {Set} Set of stop words
   */
  initializeStopWords() {
    return new Set([
      'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
      'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
      'to', 'was', 'will', 'with', 'or', 'but', 'not', 'this', 'have',
      'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    ]);
  }

  // Pattern matching helper methods
  countVagueWords(words) {
    return words.filter(word => this.patterns.vague.includes(word)).length;
  }

  countSpecificTerms(words) {
    return words.filter(word => this.patterns.specific.includes(word)).length;
  }

  countActionVerbs(prompt) {
    const words = this.getWords(prompt);
    return words.filter(word => this.patterns.action.includes(word)).length;
  }

  countTechnicalTerms(words) {
    return words.filter(word => this.patterns.technical.includes(word)).length;
  }

  countAmbiguousTerms(words) {
    return words.filter(word => this.patterns.ambiguous.includes(word)).length;
  }

  countModalVerbs(words) {
    return words.filter(word => this.patterns.modal.includes(word)).length;
  }

  countSubjectiveAdjectives(words) {
    return words.filter(word => this.patterns.subjective.includes(word)).length;
  }

  countOrganizationMarkers(prompt) {
    const words = this.getWords(prompt);
    return words.filter(word => this.patterns.organization.includes(word)).length;
  }

  countQuantitativeDetails(prompt) {
    const words = this.getWords(prompt);
    const quantWords = words.filter(word => this.patterns.quantitative.includes(word)).length;
    const numbers = (prompt.match(/\d+/g) || []).length;
    return quantWords + numbers;
  }

  // Content analysis helper methods
  countListIndicators(prompt) {
    const bulletPoints = (prompt.match(/^\s*[-*•]/gm) || []).length;
    const numberedLists = (prompt.match(/^\s*\d+\./gm) || []).length;
    return bulletPoints + numberedLists;
  }

  countEmphasisMarkers(prompt) {
    const bold = (prompt.match(/\*\*[^*]+\*\*/g) || []).length;
    const italic = (prompt.match(/\*[^*]+\*/g) || []).length;
    const caps = (prompt.match(/\b[A-Z]{2,}\b/g) || []).length;
    return bold + italic + caps;
  }

  countAmbiguousPronouns(prompt) {
    // Simplified heuristic: count pronouns without clear recent referents
    const pronouns = prompt.match(/\b(it|this|that|they|them)\b/gi) || [];
    return pronouns.length;
  }

  countConcreteExamples(prompt) {
    const exampleMarkers = [
      'for example', 'e.g.', 'such as', 'like', 'including',
      'for instance', 'specifically', 'namely'
    ];
    
    return exampleMarkers.reduce((count, marker) => {
      const regex = new RegExp(marker, 'gi');
      return count + (prompt.match(regex) || []).length;
    }, 0);
  }

  countPlaceholders(prompt) {
    const placeholders = prompt.match(/\[.*?\]|\{.*?\}|<.*?>|XXX|TODO|TBD/gi) || [];
    return placeholders.length;
  }

  countDomainSpecificTerms(words) {
    // This would ideally be context-aware based on detected domain
    // For now, using general programming/technical terms
    return this.countTechnicalTerms(words);
  }

  // Structure analysis helper methods
  hasTaskDescription(prompt) {
    const taskMarkers = ['create', 'build', 'write', 'implement', 'design', 'develop'];
    return taskMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  hasContext(prompt) {
    const contextMarkers = ['for', 'using', 'with', 'in', 'given', 'considering'];
    return contextMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  hasOutputFormat(prompt) {
    const formatMarkers = ['format', 'output', 'return', 'display', 'show', 'json', 'xml', 'csv'];
    return formatMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  hasConstraints(prompt) {
    const constraintMarkers = ['must', 'should', 'cannot', 'require', 'ensure', 'without'];
    return constraintMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  hasExamples(prompt) {
    return this.countConcreteExamples(prompt) > 0;
  }

  hasSuccessCriteria(prompt) {
    const criteriaMarkers = ['success', 'complete', 'done', 'finished', 'achieve', 'goal'];
    return criteriaMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  hasIntroduction(prompt) {
    // Simple heuristic: check if first sentence sets context
    const sentences = this.getSentences(prompt);
    if (sentences.length === 0) return false;
    
    const firstSentence = sentences[0].toLowerCase();
    const introMarkers = ['need', 'want', 'require', 'looking', 'help', 'create'];
    return introMarkers.some(marker => firstSentence.includes(marker));
  }

  hasMainContent(prompt) {
    // Main content should contain action verbs and specific instructions
    return this.countActionVerbs(prompt) > 0 && this.hasTaskDescription(prompt);
  }

  hasConclusion(prompt) {
    // Simple heuristic: check if last sentence mentions deliverables or expectations
    const sentences = this.getSentences(prompt);
    if (sentences.length === 0) return false;
    
    const lastSentence = sentences[sentences.length - 1].toLowerCase();
    const conclusionMarkers = ['result', 'output', 'deliver', 'complete', 'finish'];
    return conclusionMarkers.some(marker => lastSentence.includes(marker));
  }

  hasConsistentFormatting(prompt) {
    // Check for consistent use of formatting elements
    const bulletCount = (prompt.match(/^\s*[-*•]/gm) || []).length;
    const numberCount = (prompt.match(/^\s*\d+\./gm) || []).length;
    
    // If using lists, should be consistent style
    if (bulletCount > 0 && numberCount > 0) return false;
    
    return true;
  }

  isRunOnStructure(prompt) {
    const sentences = this.getSentences(prompt);
    const avgWordsPerSentence = sentences.length > 0 ? this.getWords(prompt).length / sentences.length : 0;
    
    // Consider run-on if average sentence length > 30 words and no organization markers
    return avgWordsPerSentence > 30 && this.countOrganizationMarkers(prompt) === 0;
  }

  hasStepByStepStructure(prompt) {
    const stepMarkers = [
      /step \d+/gi,
      /^\s*\d+\./gm,
      /first.*second.*third/gi,
      /then.*next.*finally/gi
    ];
    
    return stepMarkers.some(pattern => pattern.test(prompt));
  }

  hasImperativeMood(prompt) {
    // Check if prompt uses imperative verbs (commands)
    const sentences = this.getSentences(prompt);
    const imperativeCount = sentences.filter(sentence => {
      const words = this.getWords(sentence);
      return words.length > 0 && this.patterns.action.includes(words[0]);
    }).length;
    
    return imperativeCount / sentences.length > 0.5;
  }

  hasDeliverables(prompt) {
    const deliverableMarkers = ['output', 'result', 'return', 'provide', 'generate', 'produce'];
    return deliverableMarkers.some(marker => prompt.toLowerCase().includes(marker));
  }

  detectContradictions(prompt) {
    // Simplified contradiction detection
    // Look for conflicting requirements or negations
    const negationWords = ['not', 'never', 'without', 'except', 'unless'];
    const requirements = ['must', 'should', 'require', 'need'];
    
    let contradictions = 0;
    const sentences = this.getSentences(prompt);
    
    for (let i = 0; i < sentences.length - 1; i++) {
      const current = sentences[i].toLowerCase();
      const next = sentences[i + 1].toLowerCase();
      
      const currentHasReq = requirements.some(req => current.includes(req));
      const nextHasNeg = negationWords.some(neg => next.includes(neg));
      
      if (currentHasReq && nextHasNeg) {
        contradictions++;
      }
    }
    
    return contradictions;
  }

  countPossibleInterpretations(prompt) {
    // Heuristic: count potential ambiguities that could lead to multiple interpretations
    let interpretations = 1; // Base interpretation
    
    // Add interpretation for each ambiguous pronoun
    interpretations += this.countAmbiguousPronouns(prompt);
    
    // Add interpretation for each modal verb
    interpretations += this.countModalVerbs(this.getWords(prompt)) * 0.5;
    
    // Add interpretation for each vague term
    interpretations += this.countVagueWords(this.getWords(prompt)) * 0.3;
    
    return Math.round(interpretations);
  }

  /**
   * Identify issues in the prompt
   * @param {Object} analysis - Analysis results
   * @returns {Array} Array of identified issues
   */
  identifyIssues(analysis) {
    const issues = [];
    
    if (analysis.clarityScore < 0.6) {
      issues.push({
        category: 'clarity',
        severity: 'medium',
        description: 'Prompt contains vague language that may lead to unclear results',
        score: analysis.clarityScore
      });
    }
    
    if (analysis.completenessScore < 0.5) {
      issues.push({
        category: 'completeness',
        severity: 'high',
        description: 'Prompt is missing essential elements like context or output format',
        score: analysis.completenessScore
      });
    }
    
    if (analysis.specificityScore < 0.5) {
      issues.push({
        category: 'specificity',
        severity: 'medium',
        description: 'Prompt lacks specific details and uses too many vague terms',
        score: analysis.specificityScore
      });
    }
    
    if (analysis.structureScore < 0.5) {
      issues.push({
        category: 'structure',
        severity: 'medium',
        description: 'Prompt lacks clear organization and logical flow',
        score: analysis.structureScore
      });
    }
    
    if (analysis.ambiguityScore > 0.6) {
      issues.push({
        category: 'ambiguity',
        severity: 'high',
        description: 'Prompt contains ambiguous language that may cause confusion',
        score: analysis.ambiguityScore
      });
    }
    
    if (analysis.textMetrics.wordCount < 10) {
      issues.push({
        category: 'length',
        severity: 'high',
        description: 'Prompt is too short and likely lacks necessary detail',
        wordCount: analysis.textMetrics.wordCount
      });
    }
    
    if (analysis.textMetrics.averageWordsPerSentence > 30) {
      issues.push({
        category: 'readability',
        severity: 'medium',
        description: 'Sentences are too long and may be difficult to parse',
        averageLength: analysis.textMetrics.averageWordsPerSentence
      });
    }
    
    return issues;
  }

  /**
   * Generate improvement suggestions
   * @param {Object} analysis - Analysis results
   * @returns {Array} Array of suggestions
   */
  generateSuggestions(analysis) {
    const suggestions = [];
    
    if (analysis.clarityScore < 0.7) {
      suggestions.push({
        category: 'clarity',
        priority: 'high',
        suggestion: 'Replace vague words with specific terms and add concrete details'
      });
    }
    
    if (analysis.completenessScore < 0.7) {
      suggestions.push({
        category: 'completeness',
        priority: 'high',
        suggestion: 'Add context, specify output format, and include constraints or requirements'
      });
    }
    
    if (analysis.specificityScore < 0.6) {
      suggestions.push({
        category: 'specificity',
        priority: 'medium',
        suggestion: 'Include specific examples, quantitative details, and technical specifications'
      });
    }
    
    if (analysis.structureScore < 0.6) {
      suggestions.push({
        category: 'structure',
        priority: 'medium',
        suggestion: 'Organize content with clear sections, use step-by-step format, and add logical flow markers'
      });
    }
    
    if (analysis.actionabilityScore < 0.6) {
      suggestions.push({
        category: 'actionability',
        priority: 'high',
        suggestion: 'Use action verbs, provide clear deliverables, and specify what should be done'
      });
    }
    
    if (analysis.readabilityScore < 30) {
      suggestions.push({
        category: 'readability',
        priority: 'medium',
        suggestion: 'Simplify sentence structure and use shorter sentences for better clarity'
      });
    }
    
    return suggestions;
  }

  /**
   * Identify strengths in the prompt
   * @param {Object} analysis - Analysis results
   * @returns {Array} Array of strengths
   */
  identifyStrengths(analysis) {
    const strengths = [];
    
    if (analysis.clarityScore >= 0.8) {
      strengths.push({
        category: 'clarity',
        strength: 'Prompt uses clear, specific language with minimal ambiguity'
      });
    }
    
    if (analysis.completenessScore >= 0.8) {
      strengths.push({
        category: 'completeness',
        strength: 'Prompt includes all essential elements: task, context, format, and constraints'
      });
    }
    
    if (analysis.specificityScore >= 0.8) {
      strengths.push({
        category: 'specificity',
        strength: 'Prompt provides specific details and concrete examples'
      });
    }
    
    if (analysis.structureScore >= 0.8) {
      strengths.push({
        category: 'structure',
        strength: 'Prompt is well-organized with clear logical flow'
      });
    }
    
    if (analysis.actionabilityScore >= 0.8) {
      strengths.push({
        category: 'actionability',
        strength: 'Prompt provides clear, actionable instructions'
      });
    }
    
    if (analysis.textMetrics.wordCount >= 20 && analysis.textMetrics.wordCount <= 200) {
      strengths.push({
        category: 'length',
        strength: 'Prompt has appropriate length - detailed but concise'
      });
    }
    
    return strengths;
  }

  /**
   * Generate detailed analysis report
   * @param {Object} analysis - Analysis results
   * @returns {Object} Formatted report
   */
  generateAnalysisReport(analysis) {
    return {
      summary: {
        overallQuality: analysis.overallQuality,
        qualityGrade: analysis.qualityGrade,
        primaryStrengths: analysis.strengths.map(s => s.category),
        primaryIssues: analysis.issues.map(i => i.category),
        recommendedActions: analysis.suggestions.length
      },
      
      scores: {
        clarity: analysis.clarityScore,
        completeness: analysis.completenessScore,
        specificity: analysis.specificityScore,
        structure: analysis.structureScore,
        actionability: analysis.actionabilityScore,
        readability: analysis.readabilityScore / 100, // Normalize to 0-1
        ambiguity: 1 - analysis.ambiguityScore // Invert for consistency (higher = better)
      },
      
      metrics: analysis.textMetrics,
      issues: analysis.issues,
      suggestions: analysis.suggestions,
      strengths: analysis.strengths,
      
      metadata: {
        analyzedAt: analysis.timestamp,
        analysisVersion: '1.0.0',
        promptLength: analysis.prompt.length
      }
    };
  }
}

module.exports = StructuralAnalyzer;