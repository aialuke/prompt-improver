/**
 * Enhanced Structural Analysis Logic
 * Extracted useful components for MCP server integration
 * Simplified for practical prompt evaluation
 */

export class StructuralAnalyzer {
  constructor() {
    this.patterns = this.initializePatterns();
  }

  /**
   * Main analysis method for MCP server integration
   * @param {string} prompt - Prompt text to analyze
   * @param {Object} context - Optional context (domain, etc.)
   * @returns {Object} Analysis results
   */
  async analyzePrompt(prompt, context = {}) {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Invalid prompt: must be a non-empty string');
    }

    return {
      clarity: this.assessClarity(prompt),
      completeness: this.assessCompleteness(prompt),
      specificity: this.assessSpecificity(prompt),
      actionability: this.assessActionability(prompt),
      domainRelevance: this.assessDomainRelevance(prompt, context.domain),
      complexity: this.assessComplexity(prompt),
      overallScore: 0 // Will be calculated by combining above scores
    };
  }

  /**
   * Assess how clear and unambiguous the prompt is
   * @param {string} prompt - Prompt text
   * @returns {number} Score 0-1
   */
  assessClarity(prompt) {
    const words = this.getWords(prompt);
    const ambiguousCount = this.countAmbiguousTerms(words);
    const pronounCount = this.countAmbiguousPronouns(prompt);
    
    // Penalty for ambiguous language
    const ambiguityPenalty = (ambiguousCount + pronounCount) / words.length;
    
    // Bonus for clear structure
    const structureBonus = this.hasGoodStructure(prompt) ? 0.2 : 0;
    
    return Math.max(0, Math.min(1, 0.8 - ambiguityPenalty + structureBonus));
  }

  /**
   * Assess if the prompt has complete requirements
   * @param {string} prompt - Prompt text
   * @returns {number} Score 0-1
   */
  assessCompleteness(prompt) {
    const words = this.getWords(prompt);
    const hasObjective = this.countActionVerbs(prompt) > 0;
    const hasContext = words.length > 10;
    const hasSpecifics = this.countSpecificTerms(words) > 0;
    const hasConstraints = this.hasConstraints(prompt);
    
    const completenessFactors = [hasObjective, hasContext, hasSpecifics, hasConstraints];
    return completenessFactors.filter(Boolean).length / completenessFactors.length;
  }

  /**
   * Assess how specific vs vague the prompt is
   * @param {string} prompt - Prompt text
   * @returns {number} Score 0-1
   */
  assessSpecificity(prompt) {
    const words = this.getWords(prompt);
    const vagueCount = this.countVagueWords(words);
    const specificCount = this.countSpecificTerms(words);
    const technicalCount = this.countTechnicalTerms(words);
    const quantitativeCount = this.countQuantitativeDetails(prompt);
    
    if (words.length === 0) return 0;
    
    const vagueRatio = vagueCount / words.length;
    const specificRatio = (specificCount + technicalCount + quantitativeCount) / words.length;
    
    return Math.max(0, Math.min(1, specificRatio - vagueRatio + 0.5));
  }

  /**
   * Assess how actionable/executable the prompt is
   * @param {string} prompt - Prompt text
   * @returns {number} Score 0-1
   */
  assessActionability(prompt) {
    const actionVerbCount = this.countActionVerbs(prompt);
    const hasImperative = /^(create|build|write|implement|design|analyze)/i.test(prompt.trim());
    const hasDeliverables = this.hasDeliverables(prompt);
    const hasSteps = this.hasSteps(prompt);
    
    // Base score from action verbs
    let score = Math.min(0.4, actionVerbCount * 0.1);
    
    // Bonuses for actionable elements
    if (hasImperative) score += 0.3;
    if (hasDeliverables) score += 0.2;
    if (hasSteps) score += 0.1;
    
    return Math.min(1, score);
  }

  /**
   * Assess domain relevance based on context
   * @param {string} prompt - Prompt text
   * @param {string} domain - Target domain
   * @returns {number} Score 0-1
   */
  assessDomainRelevance(prompt, domain) {
    if (!domain) return 0.5; // Neutral if no domain specified
    
    const domainKeywords = this.getDomainKeywords(domain);
    const words = this.getWords(prompt);
    const relevantCount = words.filter(word => 
      domainKeywords.includes(word.toLowerCase())
    ).length;
    
    if (words.length === 0) return 0;
    return Math.min(1, relevantCount / words.length * 10); // Scale up for visibility
  }

  /**
   * Assess complexity level of the prompt
   * @param {string} prompt - Prompt text
   * @returns {number} Score 0-1 (0=simple, 1=complex)
   */
  assessComplexity(prompt) {
    const words = this.getWords(prompt);
    const sentences = this.getSentences(prompt);
    
    // Length-based complexity
    const wordComplexity = Math.min(1, words.length / 100);
    const sentenceComplexity = sentences.length > 3 ? 0.3 : 0;
    
    // Technical term complexity
    const technicalRatio = this.countTechnicalTerms(words) / Math.max(1, words.length);
    
    // Multi-step complexity
    const hasMultipleSteps = this.hasMultipleSteps(prompt) ? 0.2 : 0;
    
    return Math.min(1, wordComplexity + sentenceComplexity + technicalRatio + hasMultipleSteps);
  }

  // Helper methods
  getWords(text) {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  getSentences(text) {
    return text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  }

  hasGoodStructure(prompt) {
    // Check for lists, sections, or clear organization
    const hasLists = /^\s*[-*•]\s+/m.test(prompt) || /^\s*\d+\.\s+/m.test(prompt);
    const hasSections = /\n\s*\n/.test(prompt);
    const hasHeaders = /^#{1,6}\s+/m.test(prompt) || /[A-Z][A-Za-z\s]+:/.test(prompt);
    
    return hasLists || hasSections || hasHeaders;
  }

  hasConstraints(prompt) {
    const constraintWords = ['must', 'should', 'require', 'need', 'limit', 'maximum', 'minimum', 'within'];
    return constraintWords.some(word => prompt.toLowerCase().includes(word));
  }

  hasDeliverables(prompt) {
    const deliverableWords = ['output', 'result', 'return', 'provide', 'generate', 'produce'];
    return deliverableWords.some(word => prompt.toLowerCase().includes(word));
  }

  hasSteps(prompt) {
    const stepIndicators = ['step', 'first', 'then', 'next', 'finally'];
    return stepIndicators.some(word => prompt.toLowerCase().includes(word)) ||
           /\d+\.\s+/.test(prompt);
  }

  hasMultipleSteps(prompt) {
    const stepCount = (prompt.match(/\d+\.\s+/g) || []).length;
    const sequenceWords = ['then', 'next', 'after', 'before', 'finally'].filter(word => 
      prompt.toLowerCase().includes(word)
    ).length;
    
    return stepCount > 1 || sequenceWords > 1;
  }

  getDomainKeywords(domain) {
    const keywords = {
      'web-development': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 'dom', 'api', 'rest', 'http', 'frontend', 'backend'],
      'machine-learning': ['model', 'training', 'dataset', 'algorithm', 'neural', 'tensorflow', 'pytorch', 'sklearn', 'prediction', 'classification'],
      'data-analysis': ['data', 'analysis', 'pandas', 'numpy', 'visualization', 'statistics', 'correlation', 'regression', 'chart', 'graph'],
      'backend': ['server', 'database', 'api', 'microservice', 'authentication', 'authorization', 'middleware', 'deployment', 'scaling'],
      'general': ['function', 'variable', 'code', 'programming', 'software', 'development', 'algorithm', 'logic']
    };
    
    return keywords[domain] || keywords.general;
  }

  // Pattern matching methods
  initializePatterns() {
    return {
      vague: ['thing', 'stuff', 'something', 'good', 'bad', 'nice', 'great', 'some', 'many', 'big', 'small'],
      specific: ['implement', 'create', 'build', 'design', 'analyze', 'optimize', 'configure', 'deploy'],
      action: ['create', 'build', 'make', 'develop', 'write', 'implement', 'design', 'test', 'fix', 'update'],
      technical: ['function', 'class', 'method', 'api', 'database', 'component', 'algorithm', 'interface'],
      ambiguous: ['this', 'that', 'it', 'they', 'might', 'could', 'maybe', 'probably', 'seems']
    };
  }

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

  countAmbiguousPronouns(prompt) {
    const pronouns = prompt.match(/\b(it|this|that|they|them)\b/gi) || [];
    return pronouns.length;
  }

  countQuantitativeDetails(prompt) {
    const numbers = (prompt.match(/\d+/g) || []).length;
    const quantWords = ['exactly', 'precisely', 'minimum', 'maximum', 'between'];
    const quantCount = quantWords.filter(word => prompt.toLowerCase().includes(word)).length;
    return numbers + quantCount;
  }
}

export default StructuralAnalyzer;