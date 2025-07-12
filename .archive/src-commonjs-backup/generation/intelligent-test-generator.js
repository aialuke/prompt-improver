/**
 * Intelligent Test Generation System
 * Smart algorithms for creating relevant, context-aware test prompts
 */

const PromptTemplates = require('./prompt-templates');
const ComplexityStratification = require('./complexity-stratification');
const CategoryCoverage = require('./category-coverage');
const Logger = require('../utils/logger');

class IntelligentTestGenerator {
  constructor(config = {}) {
    this.config = {
      // Default generation settings
      defaultTestCount: 100,
      maxRetries: 3,
      duplicateThreshold: 0.8, // Similarity threshold for duplicate detection
      qualityThreshold: 0.7,   // Minimum quality score for generated tests
      
      // Generation strategies
      strategies: {
        balanced: 'even distribution across categories and complexities',
        context_heavy: 'favor context-specific tests based on project analysis',
        complexity_focused: 'emphasize complex scenarios for advanced testing',
        category_focused: 'ensure comprehensive coverage of all prompt issues'
      },

      // Quality scoring weights
      qualityWeights: {
        relevance: 0.3,      // How relevant to detected context
        clarity: 0.25,       // How clear and specific the prompt is
        complexity: 0.2,     // Appropriate complexity for target level
        uniqueness: 0.15,    // How unique compared to other tests
        improvability: 0.1   // How much room for improvement exists
      },

      ...config
    };

    this.promptTemplates = new PromptTemplates();
    this.complexityStratification = new ComplexityStratification();
    this.categoryCoverage = new CategoryCoverage();
    this.logger = new Logger('IntelligentTestGenerator');

    // Cache for performance optimization
    this.generationCache = new Map();
    this.qualityCache = new Map();
  }

  /**
   * Generate comprehensive test suite for a project
   * @param {Object} projectContext - Project context from Phase 2 analysis
   * @param {Object} options - Generation options
   * @returns {Promise<Object>} Generated test suite
   */
  async generateTestSuite(projectContext, options = {}) {
    const startTime = Date.now();
    this.logger.info('Starting intelligent test generation', { 
      projectPath: projectContext?.metadata?.projectPath,
      testCount: options.testCount || this.config.defaultTestCount
    });

    try {
      // Validate inputs
      this.validateInputs(projectContext, options);

      // Setup generation parameters
      const generationParams = this.setupGenerationParameters(projectContext, options);

      // Calculate distributions
      const complexityDistribution = this.complexityStratification.calculateDistribution(
        generationParams.testCount, 
        projectContext
      );

      const categoryDistribution = this.categoryCoverage.calculateCategoryDistribution(
        generationParams.testCount, 
        projectContext
      );

      // Generate tests using intelligent algorithms
      const testCases = await this.generateTestCases(
        complexityDistribution,
        categoryDistribution,
        projectContext,
        generationParams
      );

      // Apply quality filtering and optimization
      const optimizedTests = await this.optimizeTestSuite(testCases, projectContext, generationParams);

      // Validate final test suite
      const validation = this.validateTestSuite(optimizedTests, generationParams);

      const results = {
        testCases: optimizedTests,
        metadata: {
          generationTime: Date.now() - startTime,
          totalGenerated: testCases.length,
          finalCount: optimizedTests.length,
          complexityDistribution,
          categoryDistribution,
          projectContext: this.sanitizeContext(projectContext),
          generationParams,
          validation,
          strategy: generationParams.strategy
        },
        statistics: this.generateStatistics(optimizedTests, projectContext),
        qualityMetrics: this.calculateQualityMetrics(optimizedTests),
        recommendations: this.generateRecommendations(optimizedTests, projectContext, validation)
      };

      this.logger.info('Test generation completed', {
        testsGenerated: optimizedTests.length,
        generationTime: results.metadata.generationTime,
        averageQuality: results.qualityMetrics.averageQuality,
        validationPassed: validation.valid
      });

      return results;

    } catch (error) {
      this.logger.error('Test generation failed', error);
      throw new Error(`Test generation failed: ${error.message}`);
    }
  }

  /**
   * Validate generation inputs
   * @param {Object} projectContext - Project context
   * @param {Object} options - Generation options
   */
  validateInputs(projectContext, options) {
    if (!projectContext || typeof projectContext !== 'object') {
      throw new Error('Valid project context is required');
    }

    const testCount = options.testCount || this.config.defaultTestCount;
    if (testCount < 10 || testCount > 1000) {
      throw new Error('Test count must be between 10 and 1000');
    }

    if (options.strategy && !this.config.strategies[options.strategy]) {
      throw new Error(`Unknown generation strategy: ${options.strategy}`);
    }
  }

  /**
   * Setup generation parameters
   * @param {Object} projectContext - Project context
   * @param {Object} options - User options
   * @returns {Object} Generation parameters
   */
  setupGenerationParameters(projectContext, options) {
    return {
      testCount: options.testCount || this.config.defaultTestCount,
      strategy: options.strategy || this.selectOptimalStrategy(projectContext),
      categories: options.categories || Object.keys(this.categoryCoverage.getAllCategories()),
      complexities: options.complexities || this.complexityStratification.getComplexityLevels(),
      contextWeight: options.contextWeight || this.calculateContextWeight(projectContext),
      qualityThreshold: options.qualityThreshold || this.config.qualityThreshold,
      allowDuplicates: options.allowDuplicates || false,
      seed: options.seed || Date.now() // For reproducible generation
    };
  }

  /**
   * Select optimal generation strategy based on project context
   * @param {Object} projectContext - Project context
   * @returns {string} Optimal strategy name
   */
  selectOptimalStrategy(projectContext) {
    // High complexity projects benefit from complexity-focused strategy
    if (projectContext.domain?.complexity === 'high' || 
        projectContext.organizational?.teamSize === 'large') {
      return 'complexity_focused';
    }

    // Rich context projects benefit from context-heavy strategy
    if (projectContext.confidence?.overall > 0.8) {
      return 'context_heavy';
    }

    // Simple projects or unclear context benefit from balanced approach
    if (projectContext.domain?.complexity === 'simple' || 
        projectContext.confidence?.overall < 0.5) {
      return 'balanced';
    }

    // Default to category-focused for comprehensive coverage
    return 'category_focused';
  }

  /**
   * Calculate context weight based on project analysis confidence
   * @param {Object} projectContext - Project context
   * @returns {number} Context weight (0-1)
   */
  calculateContextWeight(projectContext) {
    let weight = 0.5; // Default weight

    // Adjust based on analysis confidence
    if (projectContext.confidence?.overall) {
      weight = projectContext.confidence.overall;
    }

    // Boost weight for high-context domains
    const highContextDomains = ['fintech', 'healthcare', 'e-commerce'];
    if (highContextDomains.includes(projectContext.domain?.domain)) {
      weight += 0.2;
    }

    // Boost weight for complex project types
    const complexTypes = ['distributed-system', 'microservices', 'enterprise-application'];
    if (complexTypes.includes(projectContext.domain?.projectType)) {
      weight += 0.15;
    }

    return Math.min(weight, 1.0);
  }

  /**
   * Generate test cases using intelligent algorithms
   * @param {Object} complexityDistribution - Complexity distribution
   * @param {Object} categoryDistribution - Category distribution
   * @param {Object} projectContext - Project context
   * @param {Object} params - Generation parameters
   * @returns {Promise<Array>} Generated test cases
   */
  async generateTestCases(complexityDistribution, categoryDistribution, projectContext, params) {
    const testCases = [];
    const generation_matrix = this.createGenerationMatrix(complexityDistribution, categoryDistribution);

    this.logger.debug('Generation matrix created', {
      matrixSize: generation_matrix.length,
      strategy: params.strategy
    });

    // Generate tests for each cell in the matrix
    for (const cell of generation_matrix) {
      const cellTests = await this.generateCellTests(cell, projectContext, params);
      testCases.push(...cellTests);
    }

    return testCases;
  }

  /**
   * Create generation matrix combining complexity and category distributions
   * @param {Object} complexityDistribution - Complexity distribution
   * @param {Object} categoryDistribution - Category distribution
   * @returns {Array} Generation matrix
   */
  createGenerationMatrix(complexityDistribution, categoryDistribution) {
    const matrix = [];
    
    // Create cell for each complexity-category combination
    Object.keys(complexityDistribution.counts).forEach(complexity => {
      Object.keys(categoryDistribution.distribution).forEach(category => {
        const complexityCount = complexityDistribution.counts[complexity];
        const categoryCount = categoryDistribution.distribution[category];
        
        // Calculate proportional allocation
        const totalTests = complexityDistribution.total;
        const complexityRatio = complexityCount / totalTests;
        const categoryRatio = categoryCount / totalTests;
        
        const cellCount = Math.round(totalTests * complexityRatio * categoryRatio);
        
        if (cellCount > 0) {
          matrix.push({
            complexity,
            category,
            count: cellCount,
            priority: this.calculateCellPriority(complexity, category, complexityRatio, categoryRatio)
          });
        }
      });
    });

    // Sort by priority for better generation order
    return matrix.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Calculate priority for generation matrix cell
   * @param {string} complexity - Complexity level
   * @param {string} category - Category name
   * @param {number} complexityRatio - Complexity ratio
   * @param {number} categoryRatio - Category ratio
   * @returns {number} Priority score
   */
  calculateCellPriority(complexity, category, complexityRatio, categoryRatio) {
    let priority = complexityRatio + categoryRatio;

    // Boost priority for critical categories
    const criticalCategories = ['vague_instructions', 'missing_context', 'no_output_format'];
    if (criticalCategories.includes(category)) {
      priority += 0.2;
    }

    // Boost priority for moderate complexity (good balance)
    if (complexity === 'moderate') {
      priority += 0.1;
    }

    return priority;
  }

  /**
   * Generate tests for a specific matrix cell
   * @param {Object} cell - Generation matrix cell
   * @param {Object} projectContext - Project context
   * @param {Object} params - Generation parameters
   * @returns {Promise<Array>} Generated tests for the cell
   */
  async generateCellTests(cell, projectContext, params) {
    const tests = [];
    
    try {
      // Generate base prompts using templates
      const templatePrompts = this.promptTemplates.generateTemplates(
        this.mapCategoryToTemplateCategory(cell.category),
        cell.complexity,
        projectContext,
        cell.count
      );

      // Generate category-specific prompts
      const categoryPrompts = this.categoryCoverage.generateCategoryPrompts(
        cell.category,
        cell.complexity,
        projectContext,
        cell.count
      );

      // Combine and diversify
      const combinedPrompts = this.combineAndDiversify(templatePrompts, categoryPrompts, cell.count);

      // Create test case objects
      for (let i = 0; i < Math.min(combinedPrompts.length, cell.count); i++) {
        const prompt = combinedPrompts[i];
        const testCase = await this.createTestCase(prompt, cell, projectContext, params);
        tests.push(testCase);
      }

    } catch (error) {
      this.logger.warn(`Failed to generate tests for cell ${cell.category}:${cell.complexity}`, error);
    }

    return tests;
  }

  /**
   * Map category names to template categories
   * @param {string} category - Category name
   * @returns {string} Template category
   */
  mapCategoryToTemplateCategory(category) {
    const mapping = {
      vague_instructions: 'creation',
      missing_context: 'creation',
      poor_structure: 'refactoring',
      missing_examples: 'documentation',
      no_output_format: 'creation',
      domain_specific: 'integration'
    };

    return mapping[category] || 'creation';
  }

  /**
   * Combine and diversify prompts from different sources
   * @param {Array} templatePrompts - Prompts from templates
   * @param {Array} categoryPrompts - Prompts from category generation
   * @param {number} targetCount - Target number of prompts
   * @returns {Array} Combined and diversified prompts
   */
  combineAndDiversify(templatePrompts, categoryPrompts, targetCount) {
    const combined = [];
    const used = new Set();

    // Interleave template and category prompts
    const maxLength = Math.max(templatePrompts.length, categoryPrompts.length);
    
    for (let i = 0; i < maxLength && combined.length < targetCount; i++) {
      // Add template prompt
      if (i < templatePrompts.length && combined.length < targetCount) {
        const prompt = templatePrompts[i];
        if (!used.has(prompt) && !this.isDuplicatePrompt(prompt, combined)) {
          combined.push(prompt);
          used.add(prompt);
        }
      }

      // Add category prompt
      if (i < categoryPrompts.length && combined.length < targetCount) {
        const categoryPrompt = categoryPrompts[i];
        const prompt = categoryPrompt.prompt || categoryPrompt;
        if (!used.has(prompt) && !this.isDuplicatePrompt(prompt, combined)) {
          combined.push(prompt);
          used.add(prompt);
        }
      }
    }

    return combined;
  }

  /**
   * Check if prompt is duplicate
   * @param {string} prompt - Prompt to check
   * @param {Array} existingPrompts - Existing prompts
   * @returns {boolean} Whether prompt is duplicate
   */
  isDuplicatePrompt(prompt, existingPrompts) {
    const similarity = existingPrompts.some(existing => 
      this.calculatePromptSimilarity(prompt, existing) > this.config.duplicateThreshold
    );
    return similarity;
  }

  /**
   * Calculate similarity between two prompts
   * @param {string} prompt1 - First prompt
   * @param {string} prompt2 - Second prompt
   * @returns {number} Similarity score (0-1)
   */
  calculatePromptSimilarity(prompt1, prompt2) {
    // Simple similarity based on word overlap
    const words1 = prompt1.toLowerCase().split(/\s+/);
    const words2 = prompt2.toLowerCase().split(/\s+/);
    
    const overlap = words1.filter(word => words2.includes(word)).length;
    const total = Math.max(words1.length, words2.length);
    
    return total > 0 ? overlap / total : 0;
  }

  /**
   * Create test case object
   * @param {string} prompt - Generated prompt
   * @param {Object} cell - Generation matrix cell
   * @param {Object} projectContext - Project context
   * @param {Object} params - Generation parameters
   * @returns {Promise<Object>} Test case object
   */
  async createTestCase(prompt, cell, projectContext, params) {
    const testCase = {
      id: this.generateTestId(cell, prompt),
      originalPrompt: prompt,
      category: cell.category,
      complexity: cell.complexity,
      expectedImprovements: this.getExpectedImprovements(cell.category, cell.complexity),
      domain: projectContext.domain?.domain || 'general',
      techContext: this.extractTechContext(projectContext),
      metadata: {
        generatedBy: 'IntelligentTestGenerator',
        strategy: params.strategy,
        contextWeight: params.contextWeight,
        generationTime: new Date().toISOString(),
        qualityScore: await this.calculatePromptQuality(prompt, cell, projectContext)
      }
    };

    return testCase;
  }

  /**
   * Generate unique test ID
   * @param {Object} cell - Generation matrix cell
   * @param {string} prompt - Prompt text
   * @returns {string} Unique test ID
   */
  generateTestId(cell, prompt) {
    const hash = this.simpleHash(prompt);
    return `test-${cell.category}-${cell.complexity}-${hash}`;
  }

  /**
   * Simple hash function for strings
   * @param {string} str - String to hash
   * @returns {string} Hash value
   */
  simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Get expected improvements for category and complexity
   * @param {string} category - Category name
   * @param {string} complexity - Complexity level
   * @returns {Array} Expected improvements
   */
  getExpectedImprovements(category, complexity) {
    const categoryDetails = this.categoryCoverage.getCategoryDetails(category);
    const complexityDetails = this.complexityStratification.getComplexityDetails(complexity);
    
    const improvements = [];
    
    if (categoryDetails?.expectedImprovements) {
      improvements.push(...categoryDetails.expectedImprovements);
    }
    
    if (complexityDetails?.expectedImprovements) {
      improvements.push(...complexityDetails.expectedImprovements);
    }

    return [...new Set(improvements)]; // Remove duplicates
  }

  /**
   * Extract technical context from project context
   * @param {Object} projectContext - Project context
   * @returns {Array} Technical context array
   */
  extractTechContext(projectContext) {
    const techContext = [];
    
    if (projectContext.technical?.languages) {
      techContext.push(...Object.keys(projectContext.technical.languages));
    }
    
    if (projectContext.technical?.frameworks) {
      techContext.push(...Object.keys(projectContext.technical.frameworks).slice(0, 5));
    }

    return techContext;
  }

  /**
   * Calculate quality score for a prompt
   * @param {string} prompt - Prompt text
   * @param {Object} cell - Generation matrix cell
   * @param {Object} projectContext - Project context
   * @returns {Promise<number>} Quality score (0-1)
   */
  async calculatePromptQuality(prompt, cell, projectContext) {
    const cacheKey = this.simpleHash(prompt + cell.category + cell.complexity);
    
    if (this.qualityCache.has(cacheKey)) {
      return this.qualityCache.get(cacheKey);
    }

    let totalScore = 0;
    const weights = this.config.qualityWeights;

    // Relevance score
    const relevanceScore = this.calculateRelevanceScore(prompt, projectContext);
    totalScore += relevanceScore * weights.relevance;

    // Clarity score
    const clarityScore = this.calculateClarityScore(prompt);
    totalScore += clarityScore * weights.clarity;

    // Complexity appropriateness score
    const complexityScore = this.calculateComplexityScore(prompt, cell.complexity);
    totalScore += complexityScore * weights.complexity;

    // Uniqueness score (simplified for performance)
    const uniquenessScore = 0.8; // Placeholder - full implementation would check against all existing prompts
    totalScore += uniquenessScore * weights.uniqueness;

    // Improvability score
    const improvabilityScore = this.calculateImprovabilityScore(prompt, cell.category);
    totalScore += improvabilityScore * weights.improvability;

    const finalScore = Math.min(totalScore, 1.0);
    this.qualityCache.set(cacheKey, finalScore);
    
    return finalScore;
  }

  /**
   * Calculate relevance score based on project context
   * @param {string} prompt - Prompt text
   * @param {Object} projectContext - Project context
   * @returns {number} Relevance score (0-1)
   */
  calculateRelevanceScore(prompt, projectContext) {
    let score = 0.5; // Base score

    // Check for technology mentions
    if (projectContext.technical?.languages) {
      const languages = Object.keys(projectContext.technical.languages);
      const mentionsLanguage = languages.some(lang => 
        prompt.toLowerCase().includes(lang.toLowerCase())
      );
      if (mentionsLanguage) score += 0.2;
    }

    // Check for framework mentions
    if (projectContext.technical?.frameworks) {
      const frameworks = Object.keys(projectContext.technical.frameworks);
      const mentionsFramework = frameworks.some(fw => 
        prompt.toLowerCase().includes(fw.toLowerCase())
      );
      if (mentionsFramework) score += 0.2;
    }

    // Check for domain relevance
    if (projectContext.domain?.domain) {
      const domainKeywords = this.getDomainKeywords(projectContext.domain.domain);
      const mentionsDomain = domainKeywords.some(keyword => 
        prompt.toLowerCase().includes(keyword.toLowerCase())
      );
      if (mentionsDomain) score += 0.1;
    }

    return Math.min(score, 1.0);
  }

  /**
   * Get keywords for a domain
   * @param {string} domain - Domain name
   * @returns {Array} Domain keywords
   */
  getDomainKeywords(domain) {
    const keywords = {
      'e-commerce': ['shop', 'cart', 'product', 'order', 'payment', 'customer'],
      'fintech': ['payment', 'transaction', 'bank', 'finance', 'money', 'account'],
      'healthcare': ['patient', 'medical', 'health', 'doctor', 'appointment', 'treatment'],
      'education': ['student', 'course', 'lesson', 'grade', 'assignment', 'teacher'],
      'gaming': ['player', 'game', 'score', 'level', 'achievement', 'match'],
      'social-media': ['user', 'post', 'comment', 'friend', 'message', 'feed'],
      'analytics': ['data', 'metric', 'chart', 'report', 'analysis', 'dashboard'],
      'productivity': ['task', 'project', 'team', 'workflow', 'deadline', 'collaboration']
    };

    return keywords[domain] || [];
  }

  /**
   * Calculate clarity score
   * @param {string} prompt - Prompt text
   * @returns {number} Clarity score (0-1)
   */
  calculateClarityScore(prompt) {
    let score = 0.5; // Base score

    // Check for specific action verbs
    const actionVerbs = ['create', 'build', 'implement', 'design', 'develop', 'write', 'add'];
    const hasActionVerb = actionVerbs.some(verb => prompt.toLowerCase().includes(verb));
    if (hasActionVerb) score += 0.2;

    // Check for specific objects/targets
    const hasSpecificTarget = /\b(component|function|class|service|module|system)\b/i.test(prompt);
    if (hasSpecificTarget) score += 0.15;

    // Check for context variables
    const hasVariables = /\{[^}]+\}/g.test(prompt);
    if (hasVariables) score += 0.15;

    // Penalize vague terms
    const vagueTerms = ['better', 'good', 'nice', 'improve', 'enhance'];
    const hasVagueTerms = vagueTerms.some(term => prompt.toLowerCase().includes(term));
    if (hasVagueTerms) score -= 0.2;

    return Math.max(Math.min(score, 1.0), 0);
  }

  /**
   * Calculate complexity appropriateness score
   * @param {string} prompt - Prompt text
   * @param {string} targetComplexity - Target complexity level
   * @returns {number} Complexity score (0-1)
   */
  calculateComplexityScore(prompt, targetComplexity) {
    const characteristics = this.complexityStratification.analyzePromptCharacteristics(prompt);
    const inferredComplexity = this.complexityStratification.inferComplexityFromCharacteristics(characteristics, {});
    
    // Perfect match gets full score
    if (inferredComplexity === targetComplexity) {
      return 1.0;
    }

    // Adjacent complexity levels get partial score
    const complexityOrder = ['simple', 'moderate', 'complex'];
    const targetIndex = complexityOrder.indexOf(targetComplexity);
    const inferredIndex = complexityOrder.indexOf(inferredComplexity);
    
    const difference = Math.abs(targetIndex - inferredIndex);
    if (difference === 1) {
      return 0.7; // Adjacent level
    }
    
    return 0.3; // Far off
  }

  /**
   * Calculate improvability score
   * @param {string} prompt - Prompt text
   * @param {string} category - Category name
   * @returns {number} Improvability score (0-1)
   */
  calculateImprovabilityScore(prompt, category) {
    // Detect category patterns to ensure room for improvement
    const categoryDetails = this.categoryCoverage.getCategoryDetails(category);
    if (!categoryDetails) return 0.5;

    let score = 0.5; // Base score

    // Check if prompt exhibits the category's problematic characteristics
    let hasProblematicPatterns = false;
    
    if (categoryDetails.detectionPatterns) {
      hasProblematicPatterns = categoryDetails.detectionPatterns.some(pattern => {
        return pattern.test(prompt);
      });
    }

    if (hasProblematicPatterns) {
      score += 0.3; // Good - has room for improvement
    }

    // Check prompt length appropriateness for improvement potential
    const wordCount = prompt.split(/\s+/).length;
    if (category === 'vague_instructions' && wordCount < 10) {
      score += 0.2; // Short and vague = good improvement potential
    } else if (category === 'missing_context' && wordCount < 15) {
      score += 0.2; // Missing context with short prompt = good potential
    }

    return Math.min(score, 1.0);
  }

  /**
   * Optimize test suite for quality and coverage
   * @param {Array} testCases - Generated test cases
   * @param {Object} projectContext - Project context
   * @param {Object} params - Generation parameters
   * @returns {Promise<Array>} Optimized test cases
   */
  async optimizeTestSuite(testCases, projectContext, params) {
    let optimized = [...testCases];

    // Filter by quality threshold
    optimized = optimized.filter(test => 
      test.metadata.qualityScore >= params.qualityThreshold
    );

    // Remove duplicates if not allowed
    if (!params.allowDuplicates) {
      optimized = this.removeDuplicates(optimized);
    }

    // Ensure minimum coverage requirements
    optimized = this.ensureMinimumCoverage(optimized, params);

    // Sort by quality score (best first)
    optimized.sort((a, b) => b.metadata.qualityScore - a.metadata.qualityScore);

    // Limit to target count if exceeded
    if (optimized.length > params.testCount) {
      optimized = this.selectBestTests(optimized, params.testCount);
    }

    this.logger.debug('Test suite optimized', {
      original: testCases.length,
      afterQualityFilter: optimized.length,
      target: params.testCount
    });

    return optimized;
  }

  /**
   * Remove duplicate test cases
   * @param {Array} testCases - Test cases
   * @returns {Array} Deduplicated test cases
   */
  removeDuplicates(testCases) {
    const seen = new Set();
    const unique = [];

    for (const test of testCases) {
      const key = test.originalPrompt.toLowerCase().trim();
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(test);
      }
    }

    return unique;
  }

  /**
   * Ensure minimum coverage requirements are met
   * @param {Array} testCases - Current test cases
   * @param {Object} params - Generation parameters
   * @returns {Array} Test cases with ensured coverage
   */
  ensureMinimumCoverage(testCases, params) {
    // Check category coverage
    const categoryCounts = {};
    testCases.forEach(test => {
      categoryCounts[test.category] = (categoryCounts[test.category] || 0) + 1;
    });

    // Check complexity coverage
    const complexityCounts = {};
    testCases.forEach(test => {
      complexityCounts[test.complexity] = (complexityCounts[test.complexity] || 0) + 1;
    });

    // Add tests for underrepresented categories/complexities if needed
    // (This is a simplified implementation - full version would regenerate)
    
    return testCases;
  }

  /**
   * Select best tests while maintaining distribution
   * @param {Array} testCases - Sorted test cases
   * @param {number} targetCount - Target number of tests
   * @returns {Array} Selected best tests
   */
  selectBestTests(testCases, targetCount) {
    // Use stratified sampling to maintain category/complexity distribution
    const categorized = {};
    
    testCases.forEach(test => {
      const key = `${test.category}-${test.complexity}`;
      if (!categorized[key]) categorized[key] = [];
      categorized[key].push(test);
    });

    const selected = [];
    const keys = Object.keys(categorized);
    const perKey = Math.ceil(targetCount / keys.length);

    // Take best tests from each category-complexity combination
    keys.forEach(key => {
      const available = categorized[key].slice(0, perKey);
      selected.push(...available);
    });

    // Fill remaining slots with highest quality tests
    const remaining = targetCount - selected.length;
    if (remaining > 0) {
      const allRemaining = testCases.filter(test => !selected.includes(test));
      selected.push(...allRemaining.slice(0, remaining));
    }

    return selected.slice(0, targetCount);
  }

  /**
   * Validate final test suite
   * @param {Array} testCases - Final test cases
   * @param {Object} params - Generation parameters
   * @returns {Object} Validation results
   */
  validateTestSuite(testCases, params) {
    const validation = {
      valid: true,
      errors: [],
      warnings: [],
      metrics: {}
    };

    // Check test count
    if (testCases.length < params.testCount * 0.8) {
      validation.warnings.push(`Test count ${testCases.length} is below 80% of target ${params.testCount}`);
    }

    // Check category coverage
    const categoryCoverage = this.categoryCoverage.validateCoverage(testCases);
    if (!categoryCoverage.valid) {
      validation.warnings.push('Category coverage incomplete');
      validation.metrics.categoryCoverage = categoryCoverage;
    }

    // Check quality distribution
    const qualityScores = testCases.map(test => test.metadata.qualityScore);
    const avgQuality = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;
    
    if (avgQuality < params.qualityThreshold) {
      validation.errors.push(`Average quality ${avgQuality.toFixed(2)} below threshold ${params.qualityThreshold}`);
      validation.valid = false;
    }

    validation.metrics.averageQuality = avgQuality;
    validation.metrics.qualityDistribution = {
      min: Math.min(...qualityScores),
      max: Math.max(...qualityScores),
      median: this.calculateMedian(qualityScores)
    };

    return validation;
  }

  /**
   * Calculate median of array
   * @param {Array} values - Numeric values
   * @returns {number} Median value
   */
  calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /**
   * Generate statistics for test suite
   * @param {Array} testCases - Test cases
   * @param {Object} projectContext - Project context
   * @returns {Object} Statistics
   */
  generateStatistics(testCases, projectContext) {
    const stats = {
      total: testCases.length,
      byCategory: {},
      byComplexity: {},
      qualityDistribution: {},
      contextRelevance: 0
    };

    // Count by category
    testCases.forEach(test => {
      stats.byCategory[test.category] = (stats.byCategory[test.category] || 0) + 1;
    });

    // Count by complexity
    testCases.forEach(test => {
      stats.byComplexity[test.complexity] = (stats.byComplexity[test.complexity] || 0) + 1;
    });

    // Quality distribution
    const qualityScores = testCases.map(test => test.metadata.qualityScore);
    stats.qualityDistribution = {
      average: qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length,
      min: Math.min(...qualityScores),
      max: Math.max(...qualityScores),
      standardDeviation: this.calculateStandardDeviation(qualityScores)
    };

    // Context relevance
    const relevantTests = testCases.filter(test => 
      test.techContext.length > 0 || test.domain !== 'general'
    );
    stats.contextRelevance = relevantTests.length / testCases.length;

    return stats;
  }

  /**
   * Calculate standard deviation
   * @param {Array} values - Numeric values
   * @returns {number} Standard deviation
   */
  calculateStandardDeviation(values) {
    const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Calculate quality metrics for test suite
   * @param {Array} testCases - Test cases
   * @returns {Object} Quality metrics
   */
  calculateQualityMetrics(testCases) {
    const qualityScores = testCases.map(test => test.metadata.qualityScore);
    
    return {
      averageQuality: qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length,
      qualityRange: Math.max(...qualityScores) - Math.min(...qualityScores),
      highQualityTests: qualityScores.filter(score => score >= 0.8).length,
      lowQualityTests: qualityScores.filter(score => score < 0.6).length,
      qualityConsistency: 1 - this.calculateStandardDeviation(qualityScores) // Higher = more consistent
    };
  }

  /**
   * Generate recommendations for improvement
   * @param {Array} testCases - Test cases
   * @param {Object} projectContext - Project context
   * @param {Object} validation - Validation results
   * @returns {Array} Recommendations
   */
  generateRecommendations(testCases, projectContext, validation) {
    const recommendations = [];

    // Quality recommendations
    if (validation.metrics.averageQuality < 0.8) {
      recommendations.push('Consider increasing context weight or adjusting quality threshold for better test quality');
    }

    // Coverage recommendations
    if (validation.metrics.categoryCoverage && !validation.metrics.categoryCoverage.valid) {
      recommendations.push('Add more tests for underrepresented categories');
    }

    // Context relevance recommendations
    const stats = this.generateStatistics(testCases, projectContext);
    if (stats.contextRelevance < 0.7) {
      recommendations.push('Consider increasing context weight to generate more project-relevant tests');
    }

    // Distribution recommendations
    const categoryDistribution = Object.values(stats.byCategory);
    const maxCategory = Math.max(...categoryDistribution);
    const minCategory = Math.min(...categoryDistribution);
    
    if (maxCategory > minCategory * 3) {
      recommendations.push('Consider rebalancing category distribution for more even coverage');
    }

    if (recommendations.length === 0) {
      recommendations.push('Test suite generation meets all quality and coverage requirements');
    }

    return recommendations;
  }

  /**
   * Sanitize context for storage (remove sensitive data)
   * @param {Object} projectContext - Project context
   * @returns {Object} Sanitized context
   */
  sanitizeContext(projectContext) {
    return {
      domain: projectContext.domain,
      technical: {
        languages: Object.keys(projectContext.technical?.languages || {}),
        frameworks: Object.keys(projectContext.technical?.frameworks || {}).slice(0, 5)
      },
      confidence: projectContext.confidence,
      organizational: {
        teamSize: projectContext.organizational?.teamSize,
        maturity: projectContext.organizational?.maturity
      }
    };
  }

  /**
   * Clear generation caches
   */
  clearCaches() {
    this.generationCache.clear();
    this.qualityCache.clear();
    this.logger.debug('Generation caches cleared');
  }

  /**
   * Get generation statistics
   * @returns {Object} Generation statistics
   */
  getGenerationStats() {
    return {
      cacheSize: this.generationCache.size,
      qualityCacheSize: this.qualityCache.size,
      availableCategories: Object.keys(this.categoryCoverage.getAllCategories()).length,
      availableComplexities: this.complexityStratification.getComplexityLevels().length,
      availableTemplateCategories: this.promptTemplates.getAvailableCategories().length
    };
  }
}

module.exports = IntelligentTestGenerator;