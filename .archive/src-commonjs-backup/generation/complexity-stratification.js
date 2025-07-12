/**
 * Complexity Stratification System
 * Manages balanced distribution of tests across skill levels
 */

class ComplexityStratification {
  constructor(config = {}) {
    this.config = {
      // Default distribution percentages
      distribution: {
        simple: 0.4,     // 40% simple prompts
        moderate: 0.4,   // 40% moderate prompts
        complex: 0.2     // 20% complex prompts
      },
      
      // Minimum counts to ensure coverage
      minimumCounts: {
        simple: 5,
        moderate: 5,
        complex: 2
      },

      // Context-aware adjustments
      contextAdjustments: {
        teamSize: {
          solo: { simple: 0.5, moderate: 0.35, complex: 0.15 },
          small: { simple: 0.4, moderate: 0.4, complex: 0.2 },
          medium: { simple: 0.35, moderate: 0.45, complex: 0.2 },
          large: { simple: 0.3, moderate: 0.45, complex: 0.25 }
        },
        maturity: {
          prototype: { simple: 0.6, moderate: 0.3, complex: 0.1 },
          development: { simple: 0.4, moderate: 0.4, complex: 0.2 },
          production: { simple: 0.25, moderate: 0.45, complex: 0.3 }
        },
        projectComplexity: {
          simple: { simple: 0.7, moderate: 0.25, complex: 0.05 },
          medium: { simple: 0.4, moderate: 0.4, complex: 0.2 },
          high: { simple: 0.2, moderate: 0.5, complex: 0.3 }
        }
      },

      ...config
    };

    // Complexity level definitions
    this.complexityLevels = {
      simple: {
        name: 'Simple',
        description: 'Basic operations with minimal context requirements',
        characteristics: [
          'Single concept focus',
          'Clear, direct instructions',
          'Minimal dependencies',
          'Basic error handling',
          'Standard patterns'
        ],
        targetAudience: 'Beginners, quick tasks, basic automation',
        typicalTokenRange: [20, 100],
        maxVariables: 3,
        contextDepth: 'minimal',
        expectedImprovements: ['add basic context', 'specify output format', 'clarify requirements']
      },

      moderate: {
        name: 'Moderate',
        description: 'Multi-step processes requiring some context and integration',
        characteristics: [
          'Multi-step workflows',
          'Integration between components',
          'Some context requirements',
          'Error handling considerations',
          'Performance awareness'
        ],
        targetAudience: 'Intermediate developers, integration tasks',
        typicalTokenRange: [100, 300],
        maxVariables: 6,
        contextDepth: 'moderate',
        expectedImprovements: ['add technical context', 'specify constraints', 'include examples', 'clarify dependencies']
      },

      complex: {
        name: 'Complex',
        description: 'Architecture decisions with rich context and multiple constraints',
        characteristics: [
          'Architectural decisions',
          'Performance considerations',
          'Security requirements',
          'Scale considerations',
          'Multiple stakeholder concerns'
        ],
        targetAudience: 'Senior developers, complex scenarios, architecture decisions',
        typicalTokenRange: [300, 1000],
        maxVariables: 10,
        contextDepth: 'rich',
        expectedImprovements: ['add architectural context', 'specify trade-offs', 'include constraints', 'provide examples', 'clarify success criteria']
      }
    };
  }

  /**
   * Calculate complexity distribution for a given test count and context
   * @param {number} totalTests - Total number of tests to generate
   * @param {Object} projectContext - Project context from Phase 2 analysis
   * @returns {Object} Distribution counts by complexity level
   */
  calculateDistribution(totalTests, projectContext = {}) {
    // Start with base distribution
    let distribution = { ...this.config.distribution };

    // Apply context-aware adjustments
    distribution = this.applyContextAdjustments(distribution, projectContext);

    // Calculate actual counts
    const counts = {
      simple: Math.round(totalTests * distribution.simple),
      moderate: Math.round(totalTests * distribution.moderate),
      complex: Math.round(totalTests * distribution.complex)
    };

    // Ensure minimum counts are met
    this.enforceMinimumCounts(counts, totalTests);

    // Adjust for exact total (handle rounding)
    this.adjustForExactTotal(counts, totalTests);

    return {
      distribution: distribution,
      counts: counts,
      total: totalTests,
      percentages: {
        simple: (counts.simple / totalTests) * 100,
        moderate: (counts.moderate / totalTests) * 100,
        complex: (counts.complex / totalTests) * 100
      },
      metadata: {
        adjustments: this.getAppliedAdjustments(projectContext),
        minimumEnforced: this.wasMinimumEnforced(counts),
        contextInfluence: this.calculateContextInfluence(projectContext)
      }
    };
  }

  /**
   * Apply context-aware adjustments to base distribution
   * @param {Object} distribution - Base distribution
   * @param {Object} projectContext - Project context
   * @returns {Object} Adjusted distribution
   */
  applyContextAdjustments(distribution, projectContext) {
    let adjusted = { ...distribution };

    // Team size adjustments
    if (projectContext.teamSize && this.config.contextAdjustments.teamSize[projectContext.teamSize]) {
      adjusted = this.blendDistributions(adjusted, this.config.contextAdjustments.teamSize[projectContext.teamSize], 0.3);
    }

    // Project maturity adjustments
    if (projectContext.maturity && this.config.contextAdjustments.maturity[projectContext.maturity]) {
      adjusted = this.blendDistributions(adjusted, this.config.contextAdjustments.maturity[projectContext.maturity], 0.25);
    }

    // Project complexity adjustments
    if (projectContext.complexity && this.config.contextAdjustments.projectComplexity[projectContext.complexity]) {
      adjusted = this.blendDistributions(adjusted, this.config.contextAdjustments.projectComplexity[projectContext.complexity], 0.4);
    }

    // Framework complexity adjustments
    if (projectContext.frameworks && projectContext.frameworks.length > 0) {
      const frameworkComplexity = this.assessFrameworkComplexity(projectContext.frameworks);
      if (frameworkComplexity === 'high') {
        adjusted.complex += 0.1;
        adjusted.simple -= 0.05;
        adjusted.moderate -= 0.05;
      } else if (frameworkComplexity === 'low') {
        adjusted.simple += 0.1;
        adjusted.complex -= 0.05;
        adjusted.moderate -= 0.05;
      }
    }

    // Normalize to ensure sum equals 1
    return this.normalizeDistribution(adjusted);
  }

  /**
   * Blend two distributions with a given weight
   * @param {Object} base - Base distribution
   * @param {Object} adjustment - Adjustment distribution
   * @param {number} weight - Weight of adjustment (0-1)
   * @returns {Object} Blended distribution
   */
  blendDistributions(base, adjustment, weight) {
    return {
      simple: base.simple * (1 - weight) + adjustment.simple * weight,
      moderate: base.moderate * (1 - weight) + adjustment.moderate * weight,
      complex: base.complex * (1 - weight) + adjustment.complex * weight
    };
  }

  /**
   * Normalize distribution to sum to 1
   * @param {Object} distribution - Distribution to normalize
   * @returns {Object} Normalized distribution
   */
  normalizeDistribution(distribution) {
    const total = distribution.simple + distribution.moderate + distribution.complex;
    return {
      simple: distribution.simple / total,
      moderate: distribution.moderate / total,
      complex: distribution.complex / total
    };
  }

  /**
   * Enforce minimum counts for each complexity level
   * @param {Object} counts - Current counts
   * @param {number} totalTests - Total test count
   */
  enforceMinimumCounts(counts, totalTests) {
    const minimums = this.config.minimumCounts;

    // Check if minimums can be satisfied
    const totalMinimum = minimums.simple + minimums.moderate + minimums.complex;
    if (totalMinimum > totalTests) {
      // Scale down minimums proportionally
      const scale = totalTests / totalMinimum;
      Object.keys(minimums).forEach(level => {
        minimums[level] = Math.max(1, Math.floor(minimums[level] * scale));
      });
    }

    // Enforce minimums
    let totalAdjustment = 0;
    Object.keys(counts).forEach(level => {
      if (counts[level] < minimums[level]) {
        totalAdjustment += minimums[level] - counts[level];
        counts[level] = minimums[level];
      }
    });

    // Reduce other levels to compensate
    if (totalAdjustment > 0) {
      this.redistributeExcess(counts, totalAdjustment, totalTests);
    }
  }

  /**
   * Redistribute excess counts to maintain total
   * @param {Object} counts - Current counts
   * @param {number} excess - Excess to redistribute
   * @param {number} totalTests - Target total
   */
  redistributeExcess(counts, excess, totalTests) {
    const levels = ['complex', 'moderate', 'simple']; // Order of reduction priority
    
    for (const level of levels) {
      if (excess <= 0) break;
      
      const available = counts[level] - this.config.minimumCounts[level];
      const reduction = Math.min(available, excess);
      
      if (reduction > 0) {
        counts[level] -= reduction;
        excess -= reduction;
      }
    }
  }

  /**
   * Adjust counts to match exact total (handle rounding errors)
   * @param {Object} counts - Current counts
   * @param {number} totalTests - Target total
   */
  adjustForExactTotal(counts, totalTests) {
    const currentTotal = counts.simple + counts.moderate + counts.complex;
    const difference = totalTests - currentTotal;

    if (difference === 0) return;

    // Distribute difference across levels based on their relative sizes
    if (difference > 0) {
      // Add tests to largest categories first
      const sortedLevels = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
      for (let i = 0; i < Math.abs(difference); i++) {
        counts[sortedLevels[i % sortedLevels.length]]++;
      }
    } else {
      // Remove tests from largest categories first
      const sortedLevels = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
      for (let i = 0; i < Math.abs(difference); i++) {
        const level = sortedLevels[i % sortedLevels.length];
        if (counts[level] > this.config.minimumCounts[level]) {
          counts[level]--;
        }
      }
    }
  }

  /**
   * Assess framework complexity based on known frameworks
   * @param {Array} frameworks - List of frameworks
   * @returns {string} Complexity level (low, medium, high)
   */
  assessFrameworkComplexity(frameworks) {
    const complexityScores = {
      // Frontend frameworks
      react: 3,
      vue: 2,
      angular: 4,
      svelte: 2,
      nextjs: 4,
      nuxtjs: 4,

      // Backend frameworks  
      express: 2,
      fastify: 3,
      nestjs: 4,
      django: 4,
      flask: 2,
      fastapi: 3,
      spring: 5,

      // Testing frameworks
      jest: 2,
      cypress: 3,
      playwright: 3,
      mocha: 2,

      // Build tools
      webpack: 4,
      vite: 2,
      rollup: 3,

      // Others
      graphql: 4,
      kubernetes: 5,
      docker: 3
    };

    const totalScore = frameworks.reduce((sum, framework) => {
      const normalizedName = framework.toLowerCase().replace(/[^a-z]/g, '');
      return sum + (complexityScores[normalizedName] || 2); // Default to medium
    }, 0);

    const averageScore = totalScore / frameworks.length;

    if (averageScore >= 4) return 'high';
    if (averageScore >= 3) return 'medium';
    return 'low';
  }

  /**
   * Get applied adjustments summary
   * @param {Object} projectContext - Project context
   * @returns {Array} Applied adjustments
   */
  getAppliedAdjustments(projectContext) {
    const adjustments = [];

    if (projectContext.teamSize) {
      adjustments.push(`Team size: ${projectContext.teamSize}`);
    }
    if (projectContext.maturity) {
      adjustments.push(`Project maturity: ${projectContext.maturity}`);
    }
    if (projectContext.complexity) {
      adjustments.push(`Project complexity: ${projectContext.complexity}`);
    }
    if (projectContext.frameworks?.length > 0) {
      const frameworkComplexity = this.assessFrameworkComplexity(projectContext.frameworks);
      adjustments.push(`Framework complexity: ${frameworkComplexity}`);
    }

    return adjustments;
  }

  /**
   * Check if minimum counts were enforced
   * @param {Object} counts - Final counts
   * @returns {boolean} Whether minimums were enforced
   */
  wasMinimumEnforced(counts) {
    return Object.keys(counts).some(level => 
      counts[level] === this.config.minimumCounts[level]
    );
  }

  /**
   * Calculate context influence score
   * @param {Object} projectContext - Project context
   * @returns {number} Influence score (0-1)
   */
  calculateContextInfluence(projectContext) {
    let influence = 0;
    let factors = 0;

    if (projectContext.teamSize) {
      influence += 0.25;
      factors++;
    }
    if (projectContext.maturity) {
      influence += 0.25;
      factors++;
    }
    if (projectContext.complexity) {
      influence += 0.3;
      factors++;
    }
    if (projectContext.frameworks?.length > 0) {
      influence += 0.2;
      factors++;
    }

    return factors > 0 ? influence / factors : 0;
  }

  /**
   * Validate complexity assignment for a prompt
   * @param {string} prompt - Prompt text
   * @param {string} assignedComplexity - Assigned complexity level
   * @param {Object} context - Project context
   * @returns {Object} Validation result
   */
  validateComplexityAssignment(prompt, assignedComplexity, context) {
    const characteristics = this.analyzePromptCharacteristics(prompt);
    const expectedComplexity = this.inferComplexityFromCharacteristics(characteristics, context);
    
    return {
      valid: expectedComplexity === assignedComplexity,
      assignedComplexity,
      inferredComplexity: expectedComplexity,
      characteristics,
      confidence: this.calculateValidationConfidence(characteristics, assignedComplexity),
      recommendations: this.generateComplexityRecommendations(characteristics, assignedComplexity, expectedComplexity)
    };
  }

  /**
   * Analyze prompt characteristics
   * @param {string} prompt - Prompt text
   * @returns {Object} Analyzed characteristics
   */
  analyzePromptCharacteristics(prompt) {
    const words = prompt.split(/\s+/).length;
    const sentences = prompt.split(/[.!?]+/).length - 1;
    const variables = (prompt.match(/\{[^}]+\}/g) || []).length;
    
    // Count complexity indicators
    const complexityIndicators = {
      architecture: /\b(architect|design|system|scalable|distributed|microservice)\b/gi,
      performance: /\b(optimize|performance|scale|latency|throughput|efficiency)\b/gi,
      security: /\b(secure|security|authentication|authorization|encryption)\b/gi,
      integration: /\b(integrate|api|service|database|external|third-party)\b/gi,
      advanced: /\b(enterprise|production|high-availability|fault-tolerant|real-time)\b/gi
    };

    const indicatorCounts = {};
    let totalIndicators = 0;

    Object.keys(complexityIndicators).forEach(category => {
      const matches = prompt.match(complexityIndicators[category]) || [];
      indicatorCounts[category] = matches.length;
      totalIndicators += matches.length;
    });

    return {
      wordCount: words,
      sentenceCount: sentences,
      variableCount: variables,
      indicatorCounts,
      totalIndicators,
      avgWordsPerSentence: sentences > 0 ? words / sentences : words,
      hasMultipleSteps: /\b(then|next|after|finally|also|and)\b/gi.test(prompt),
      hasConstraints: /\b(with|using|without|must|should|ensure|require)\b/gi.test(prompt),
      hasExamples: /\b(like|such as|for example|e\.g\.|including)\b/gi.test(prompt)
    };
  }

  /**
   * Infer complexity from characteristics
   * @param {Object} characteristics - Analyzed characteristics
   * @param {Object} context - Project context
   * @returns {string} Inferred complexity level
   */
  inferComplexityFromCharacteristics(characteristics, context) {
    let score = 0;

    // Word count scoring
    if (characteristics.wordCount > 50) score += 2;
    else if (characteristics.wordCount > 20) score += 1;

    // Variable count scoring
    if (characteristics.variableCount > 6) score += 2;
    else if (characteristics.variableCount > 3) score += 1;

    // Complexity indicators scoring
    score += Math.min(characteristics.totalIndicators, 4);

    // Structural complexity
    if (characteristics.hasMultipleSteps) score += 1;
    if (characteristics.hasConstraints) score += 1;
    if (characteristics.hasExamples) score += 1;

    // Context-based adjustments
    if (context?.complexity === 'high') score += 1;
    if (context?.maturity === 'production') score += 1;

    // Map score to complexity level
    if (score >= 6) return 'complex';
    if (score >= 3) return 'moderate';
    return 'simple';
  }

  /**
   * Calculate validation confidence
   * @param {Object} characteristics - Prompt characteristics
   * @param {string} assignedComplexity - Assigned complexity
   * @returns {number} Confidence score (0-1)
   */
  calculateValidationConfidence(characteristics, assignedComplexity) {
    const levelScores = {
      simple: 0,
      moderate: 3,
      complex: 6
    };

    const assignedScore = levelScores[assignedComplexity];
    const actualScore = this.calculateComplexityScore(characteristics);
    
    const difference = Math.abs(assignedScore - actualScore);
    return Math.max(0, 1 - (difference / 6));
  }

  /**
   * Calculate complexity score for characteristics
   * @param {Object} characteristics - Prompt characteristics
   * @returns {number} Complexity score
   */
  calculateComplexityScore(characteristics) {
    let score = 0;

    if (characteristics.wordCount > 50) score += 2;
    else if (characteristics.wordCount > 20) score += 1;

    if (characteristics.variableCount > 6) score += 2;
    else if (characteristics.variableCount > 3) score += 1;

    score += Math.min(characteristics.totalIndicators, 4);

    if (characteristics.hasMultipleSteps) score += 1;
    if (characteristics.hasConstraints) score += 1;

    return score;
  }

  /**
   * Generate complexity recommendations
   * @param {Object} characteristics - Prompt characteristics
   * @param {string} assigned - Assigned complexity
   * @param {string} inferred - Inferred complexity
   * @returns {Array} Recommendations
   */
  generateComplexityRecommendations(characteristics, assigned, inferred) {
    const recommendations = [];

    if (assigned !== inferred) {
      recommendations.push(`Consider reassigning from ${assigned} to ${inferred} based on prompt analysis`);
    }

    if (assigned === 'simple' && characteristics.totalIndicators > 2) {
      recommendations.push('High complexity indicators detected - consider moderate or complex classification');
    }

    if (assigned === 'complex' && characteristics.totalIndicators === 0) {
      recommendations.push('No complexity indicators found - consider simple or moderate classification');
    }

    if (characteristics.variableCount > this.complexityLevels[assigned].maxVariables) {
      recommendations.push(`Variable count (${characteristics.variableCount}) exceeds typical range for ${assigned} complexity`);
    }

    return recommendations;
  }

  /**
   * Get complexity level details
   * @param {string} level - Complexity level
   * @returns {Object} Level details
   */
  getComplexityDetails(level) {
    return this.complexityLevels[level] || null;
  }

  /**
   * Get complexity level names
   * @returns {Array} Array of complexity level names
   */
  getComplexityLevels() {
    return Object.keys(this.complexityLevels);
  }

  /**
   * Get all complexity levels
   * @returns {Object} All complexity level definitions
   */
  getAllComplexityLevels() {
    return this.complexityLevels;
  }

  /**
   * Generate distribution report
   * @param {Object} distribution - Distribution result
   * @returns {Object} Formatted report
   */
  generateDistributionReport(distribution) {
    return {
      summary: {
        totalTests: distribution.total,
        distribution: distribution.counts,
        percentages: distribution.percentages
      },
      details: {
        contextAdjustments: distribution.metadata.adjustments,
        minimumEnforced: distribution.metadata.minimumEnforced,
        contextInfluence: Math.round(distribution.metadata.contextInfluence * 100)
      },
      breakdown: Object.keys(distribution.counts).map(level => ({
        level,
        count: distribution.counts[level],
        percentage: Math.round(distribution.percentages[level]),
        description: this.complexityLevels[level].description,
        targetAudience: this.complexityLevels[level].targetAudience
      }))
    };
  }
}

module.exports = ComplexityStratification;