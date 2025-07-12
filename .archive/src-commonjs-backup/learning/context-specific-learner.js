/**
 * Context-Specific Learning Engine
 * Optimizes prompt improvement strategies for different project types and contexts
 */

class ContextSpecificLearner {
  constructor(config = {}) {
    this.config = {
      // Learning parameters
      significanceThreshold: 0.15,     // Minimum difference for context-specific optimization
      minSampleSize: 10,               // Minimum samples per context for reliable learning
      similarityThreshold: 0.8,        // Context similarity threshold for grouping
      
      // Performance tracking
      improvementThreshold: 0.1,       // Minimum improvement to consider successful
      consistencyThreshold: 0.7,       // Required consistency for pattern recognition
      
      // Specialization parameters
      maxSpecializations: 5,           // Maximum specializations per rule
      confidenceThreshold: 0.8,        // Minimum confidence for specialization
      
      ...config
    };

    // Learning state
    this.contextGroups = new Map();
    this.contextPatterns = new Map();
    this.specializationOpportunities = [];
    this.performanceBaseline = null;
  }

  /**
   * Analyze context effectiveness across test results
   * @param {Array} results - Test results with context information
   * @returns {Object} Context-specific learning analysis
   */
  async analyzeContextEffectiveness(results) {
    // Group results by context
    const contextGroups = this.groupResultsByContext(results);
    
    // Analyze each context group
    const contextInsights = {};
    for (const [contextKey, contextResults] of Object.entries(contextGroups)) {
      contextInsights[contextKey] = await this.analyzeContextGroup(contextKey, contextResults);
    }

    // Cross-context analysis
    const crossContextAnalysis = this.performCrossContextAnalysis(contextInsights);
    
    // Identify specialization opportunities
    const specializationOpportunities = this.identifySpecializationOpportunities(contextInsights);

    return {
      contextInsights,
      crossContextComparisons: crossContextAnalysis.comparisons,
      universalPatterns: crossContextAnalysis.universalPatterns,
      contextSpecificPatterns: crossContextAnalysis.contextSpecificPatterns,
      specializationOpportunities,
      learningRecommendations: this.generateLearningRecommendations(contextInsights, specializationOpportunities),
      metadata: {
        totalContexts: Object.keys(contextInsights).length,
        totalResults: results.length,
        analysisDate: new Date().toISOString()
      }
    };
  }

  /**
   * Group results by project context
   * @private
   */
  groupResultsByContext(results) {
    const groups = {};
    
    for (const result of results) {
      const contextKey = this.generateContextKey(result.context);
      
      if (!groups[contextKey]) {
        groups[contextKey] = [];
      }
      groups[contextKey].push(result);
    }

    // Filter out groups with insufficient samples
    const filteredGroups = {};
    for (const [key, group] of Object.entries(groups)) {
      if (group.length >= this.config.minSampleSize) {
        filteredGroups[key] = group;
      }
    }

    return filteredGroups;
  }

  /**
   * Generate a context key for grouping similar contexts
   * @private
   */
  generateContextKey(context) {
    if (!context) return 'unknown';

    const components = [];
    
    // Project type
    if (context.projectType) {
      components.push(`type:${context.projectType}`);
    }
    
    // Domain
    if (context.domain) {
      components.push(`domain:${context.domain}`);
    }
    
    // Primary language/framework
    if (context.languages && context.languages.length > 0) {
      components.push(`lang:${context.languages[0]}`);
    }
    
    if (context.frameworks && context.frameworks.length > 0) {
      components.push(`framework:${context.frameworks[0]}`);
    }
    
    // Complexity level
    if (context.complexity) {
      components.push(`complexity:${context.complexity}`);
    }

    return components.join('|') || 'general';
  }

  /**
   * Analyze a specific context group
   * @private
   */
  async analyzeContextGroup(contextKey, contextResults) {
    const context = this.parseContextKey(contextKey);
    
    // Performance metrics
    const performance = this.calculateContextPerformance(contextResults);
    
    // Rule effectiveness in this context
    const ruleEffectiveness = this.analyzeRuleEffectivenessInContext(contextResults);
    
    // Category performance
    const categoryPerformance = this.analyzeCategoryPerformance(contextResults);
    
    // Complexity trends
    const complexityTrends = this.analyzeComplexityTrends(contextResults);
    
    // Success and failure patterns
    const patterns = this.identifyContextPatterns(contextResults);
    
    // Generate context-specific recommendations
    const recommendations = this.generateContextRecommendations(
      contextResults, context, ruleEffectiveness, patterns
    );

    return {
      context,
      contextKey,
      sampleSize: contextResults.length,
      performance,
      ruleEffectiveness,
      categoryPerformance,
      complexityTrends,
      patterns,
      recommendations,
      
      // Detailed analysis
      mostEffectiveRules: ruleEffectiveness.topRules.slice(0, 5),
      leastEffectiveRules: ruleEffectiveness.bottomRules.slice(0, 5),
      commonSuccessPatterns: patterns.successPatterns,
      commonFailurePatterns: patterns.failurePatterns
    };
  }

  /**
   * Calculate performance metrics for a context
   * @private
   */
  calculateContextPerformance(results) {
    const improvements = results.map(r => r.overallImprovement || 0);
    const successes = results.filter(r => 
      (r.overallImprovement || 0) >= this.config.improvementThreshold
    );

    return {
      averageImprovement: improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length,
      medianImprovement: this.calculateMedian(improvements),
      successRate: successes.length / results.length,
      consistencyScore: this.calculateConsistency(improvements),
      improvementDistribution: this.calculateDistribution(improvements),
      
      // Detailed metrics
      minImprovement: Math.min(...improvements),
      maxImprovement: Math.max(...improvements),
      standardDeviation: this.calculateStandardDeviation(improvements),
      confidenceInterval: this.calculateConfidenceInterval(improvements)
    };
  }

  /**
   * Analyze rule effectiveness within a specific context
   * @private
   */
  analyzeRuleEffectivenessInContext(results) {
    const rulePerformance = new Map();
    
    // Track rule applications and outcomes
    for (const result of results) {
      if (result.appliedRules) {
        for (const rule of result.appliedRules) {
          if (!rulePerformance.has(rule.name)) {
            rulePerformance.set(rule.name, {
              name: rule.name,
              category: rule.category,
              applications: 0,
              successes: 0,
              totalImprovement: 0,
              improvements: []
            });
          }
          
          const stats = rulePerformance.get(rule.name);
          stats.applications++;
          stats.totalImprovement += result.overallImprovement || 0;
          stats.improvements.push(result.overallImprovement || 0);
          
          if ((result.overallImprovement || 0) >= this.config.improvementThreshold) {
            stats.successes++;
          }
        }
      }
    }

    // Calculate effectiveness metrics
    const rules = Array.from(rulePerformance.values()).map(rule => ({
      ...rule,
      successRate: rule.successes / rule.applications,
      averageImprovement: rule.totalImprovement / rule.applications,
      consistency: this.calculateConsistency(rule.improvements),
      reliability: this.calculateReliability(rule.improvements),
      contextPerformance: (rule.successes / rule.applications) * 0.6 + 
                         (rule.totalImprovement / rule.applications) * 0.4
    }));

    // Sort by context performance
    rules.sort((a, b) => b.contextPerformance - a.contextPerformance);

    return {
      totalRules: rules.length,
      topRules: rules.slice(0, 10),
      bottomRules: rules.slice(-5).reverse(),
      ruleDistribution: this.calculateRuleDistribution(rules),
      
      // Performance insights
      highPerformanceRules: rules.filter(r => r.contextPerformance > 0.8),
      consistentRules: rules.filter(r => r.consistency > this.config.consistencyThreshold),
      unreliableRules: rules.filter(r => r.reliability < 0.5)
    };
  }

  /**
   * Identify patterns specific to this context
   * @private
   */
  identifyContextPatterns(results) {
    const successResults = results.filter(r => 
      (r.overallImprovement || 0) >= this.config.improvementThreshold
    );
    const failureResults = results.filter(r => 
      (r.overallImprovement || 0) < this.config.improvementThreshold
    );

    return {
      successPatterns: this.extractPatterns(successResults, 'success'),
      failurePatterns: this.extractPatterns(failureResults, 'failure'),
      
      // Comparative analysis
      differentiatingFactors: this.findDifferentiatingFactors(successResults, failureResults),
      criticalSuccessFactors: this.identifyCriticalSuccessFactors(successResults),
      commonFailurePoints: this.identifyCommonFailurePoints(failureResults)
    };
  }

  /**
   * Extract patterns from a set of results
   * @private
   */
  extractPatterns(results, type) {
    const patterns = [];
    
    // Prompt characteristics patterns
    const characteristics = results.map(r => this.analyzePromptCharacteristics(r));
    const commonCharacteristics = this.findCommonCharacteristics(characteristics);
    
    if (commonCharacteristics.length > 0) {
      patterns.push({
        type: 'prompt_characteristics',
        description: `${type} cases often involve prompts with: ${commonCharacteristics.join(', ')}`,
        frequency: commonCharacteristics.length / characteristics.length,
        examples: results.slice(0, 3).map(r => ({
          prompt: r.originalPrompt,
          improvement: r.overallImprovement
        }))
      });
    }

    // Rule application patterns
    const rulePatterns = this.analyzeRuleApplicationPatterns(results);
    patterns.push(...rulePatterns.map(pattern => ({
      type: 'rule_application',
      ...pattern
    })));

    // Improvement magnitude patterns
    const improvementPatterns = this.analyzeImprovementPatterns(results);
    patterns.push(...improvementPatterns.map(pattern => ({
      type: 'improvement_magnitude',
      ...pattern
    })));

    return patterns;
  }

  /**
   * Perform cross-context analysis
   * @private
   */
  performCrossContextAnalysis(contextInsights) {
    const contexts = Object.values(contextInsights);
    
    // Universal patterns (consistent across contexts)
    const universalPatterns = this.findUniversalPatterns(contexts);
    
    // Context-specific patterns (unique to specific contexts)
    const contextSpecificPatterns = this.findContextSpecificPatterns(contexts);
    
    // Performance comparisons
    const comparisons = this.generatePerformanceComparisons(contexts);

    return {
      universalPatterns,
      contextSpecificPatterns,
      comparisons,
      
      // Cross-context insights
      bestPerformingContext: contexts.reduce((best, ctx) => 
        !best || ctx.performance.averageImprovement > best.performance.averageImprovement ? ctx : best
      ),
      mostConsistentContext: contexts.reduce((best, ctx) => 
        !best || ctx.performance.consistencyScore > best.performance.consistencyScore ? ctx : best
      ),
      contextClusters: this.clusterSimilarContexts(contexts)
    };
  }

  /**
   * Identify specialization opportunities
   * @private
   */
  identifySpecializationOpportunities(contextInsights) {
    const opportunities = [];
    
    for (const [contextKey, insights] of Object.entries(contextInsights)) {
      // Rules that perform significantly better in this context
      for (const rule of insights.mostEffectiveRules) {
        if (this.isSpecializationCandidate(rule, contextKey, contextInsights)) {
          opportunities.push({
            type: 'strengthen',
            context: contextKey,
            rule: rule.name,
            currentPerformance: rule.contextPerformance,
            opportunity: `Rule '${rule.name}' shows exceptional performance in ${contextKey}`,
            recommendation: `Create context-specific enhancement for ${contextKey}`,
            confidence: this.calculateSpecializationConfidence(rule, insights),
            priority: rule.contextPerformance > 0.9 ? 'high' : 'medium'
          });
        }
      }
      
      // Rules that underperform in this context
      for (const rule of insights.leastEffectiveRules) {
        if (rule.contextPerformance < 0.3 && rule.applications >= 5) {
          opportunities.push({
            type: 'adapt',
            context: contextKey,
            rule: rule.name,
            currentPerformance: rule.contextPerformance,
            opportunity: `Rule '${rule.name}' consistently underperforms in ${contextKey}`,
            recommendation: `Develop context-specific alternative for ${contextKey}`,
            confidence: 0.8,
            priority: 'medium'
          });
        }
      }
    }

    return opportunities.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return (priorityOrder[b.priority] - priorityOrder[a.priority]) || 
             (b.confidence - a.confidence);
    });
  }

  /**
   * Generate learning recommendations
   * @private
   */
  generateLearningRecommendations(contextInsights, specializationOpportunities) {
    const recommendations = [];
    
    // Global recommendations
    const allContexts = Object.values(contextInsights);
    const globalAverage = allContexts.reduce((sum, ctx) => 
      sum + ctx.performance.averageImprovement, 0) / allContexts.length;

    // Context optimization recommendations
    for (const [contextKey, insights] of Object.entries(contextInsights)) {
      if (insights.performance.averageImprovement < globalAverage * 0.8) {
        recommendations.push({
          type: 'context_optimization',
          context: contextKey,
          priority: 'high',
          recommendation: `Focus improvement efforts on ${contextKey} context`,
          actions: [
            `Analyze top-performing rules in similar contexts`,
            `Develop context-specific rule variants`,
            `Increase training data for this context type`
          ],
          expectedImpact: 'high'
        });
      }
    }

    // Rule specialization recommendations
    const highConfidenceOpportunities = specializationOpportunities
      .filter(opp => opp.confidence >= this.config.confidenceThreshold);
    
    if (highConfidenceOpportunities.length > 0) {
      recommendations.push({
        type: 'rule_specialization',
        priority: 'medium',
        recommendation: 'Implement context-specific rule specializations',
        opportunities: highConfidenceOpportunities.slice(0, 5),
        expectedImpact: 'medium'
      });
    }

    // Pattern-based recommendations
    const universalPatterns = this.findUniversalPatterns(allContexts);
    if (universalPatterns.length > 0) {
      recommendations.push({
        type: 'universal_enhancement',
        priority: 'low',
        recommendation: 'Strengthen universal improvement patterns',
        patterns: universalPatterns,
        expectedImpact: 'low'
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  // Utility methods
  parseContextKey(contextKey) {
    const parts = contextKey.split('|');
    const context = {};
    
    for (const part of parts) {
      const [key, value] = part.split(':');
      context[key] = value;
    }
    
    return context;
  }

  calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 
      ? (sorted[mid - 1] + sorted[mid]) / 2 
      : sorted[mid];
  }

  calculateConsistency(values) {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const standardDeviation = Math.sqrt(variance);
    
    // Normalize consistency score (lower std dev = higher consistency)
    return Math.max(0, 1 - (standardDeviation / Math.max(mean, 0.1)));
  }

  calculateStandardDeviation(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  calculateConfidenceInterval(values, confidence = 0.95) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const std = this.calculateStandardDeviation(values);
    const n = values.length;
    
    // t-distribution critical value (approximation for 95% confidence)
    const t = 1.96; // For large samples
    const margin = t * (std / Math.sqrt(n));
    
    return {
      lower: mean - margin,
      upper: mean + margin,
      margin: margin
    };
  }

  calculateDistribution(values) {
    const sorted = [...values].sort((a, b) => a - b);
    return {
      min: sorted[0],
      q1: sorted[Math.floor(sorted.length * 0.25)],
      median: this.calculateMedian(values),
      q3: sorted[Math.floor(sorted.length * 0.75)],
      max: sorted[sorted.length - 1]
    };
  }

  calculateReliability(values) {
    // Simple reliability measure based on consistency and sample size
    const consistency = this.calculateConsistency(values);
    const sampleSizeBonus = Math.min(values.length / 20, 1); // Bonus for larger samples
    return consistency * 0.8 + sampleSizeBonus * 0.2;
  }

  analyzePromptCharacteristics(result) {
    const prompt = result.originalPrompt || '';
    return {
      length: prompt.length,
      wordCount: prompt.split(/\s+/).length,
      hasExamples: /for example|e\.g\.|such as/i.test(prompt),
      hasStructure: /<[^>]+>/.test(prompt),
      hasConstraints: /must|should|require/i.test(prompt),
      hasFormat: /format|json|xml|csv/i.test(prompt),
      complexity: this.assessPromptComplexity(prompt)
    };
  }

  assessPromptComplexity(prompt) {
    const factors = [
      prompt.length > 200,
      (prompt.match(/\./g) || []).length > 3,
      /\b(and|or|but|however|therefore)\b/gi.test(prompt),
      /<[^>]+>/.test(prompt),
      /\{[^}]+\}/.test(prompt)
    ];
    
    const score = factors.filter(Boolean).length / factors.length;
    return score > 0.6 ? 'high' : score > 0.3 ? 'medium' : 'low';
  }

  findCommonCharacteristics(characteristics) {
    const common = [];
    const threshold = 0.7; // 70% of cases should have this characteristic
    
    const booleanProps = ['hasExamples', 'hasStructure', 'hasConstraints', 'hasFormat'];
    for (const prop of booleanProps) {
      const frequency = characteristics.filter(c => c[prop]).length / characteristics.length;
      if (frequency >= threshold) {
        common.push(prop.replace(/^has/, '').toLowerCase());
      }
    }
    
    return common;
  }

  isSpecializationCandidate(rule, contextKey, allContextInsights) {
    // Rule must perform significantly better in this context than globally
    const globalPerformance = this.calculateGlobalRulePerformance(rule.name, allContextInsights);
    const performanceDiff = rule.contextPerformance - globalPerformance;
    
    return performanceDiff >= this.config.significanceThreshold &&
           rule.applications >= 5 &&
           rule.consistency >= this.config.consistencyThreshold;
  }

  calculateGlobalRulePerformance(ruleName, allContextInsights) {
    let totalApplications = 0;
    let totalSuccesses = 0;
    let totalImprovement = 0;
    
    for (const insights of Object.values(allContextInsights)) {
      const rule = insights.ruleEffectiveness.topRules
        .concat(insights.ruleEffectiveness.bottomRules)
        .find(r => r.name === ruleName);
      
      if (rule) {
        totalApplications += rule.applications;
        totalSuccesses += rule.successes;
        totalImprovement += rule.totalImprovement;
      }
    }
    
    return totalApplications > 0 ? 
      (totalSuccesses / totalApplications) * 0.6 + 
      (totalImprovement / totalApplications) * 0.4 : 0;
  }

  calculateSpecializationConfidence(rule, insights) {
    const factors = [
      rule.consistency >= this.config.consistencyThreshold ? 1 : 0,
      rule.applications >= 10 ? 1 : Math.max(0, rule.applications / 10),
      rule.contextPerformance >= 0.8 ? 1 : rule.contextPerformance,
      insights.sampleSize >= 20 ? 1 : Math.max(0, insights.sampleSize / 20)
    ];
    
    return factors.reduce((sum, factor) => sum + factor, 0) / factors.length;
  }

  findUniversalPatterns(contexts) {
    // Patterns that appear consistently across contexts
    const patterns = [];
    const threshold = 0.8; // Must appear in 80% of contexts
    
    // Analyze rule effectiveness universality
    const ruleConsistency = new Map();
    for (const context of contexts) {
      for (const rule of context.ruleEffectiveness.topRules.slice(0, 5)) {
        if (!ruleConsistency.has(rule.name)) {
          ruleConsistency.set(rule.name, { contexts: 0, totalPerformance: 0 });
        }
        const stats = ruleConsistency.get(rule.name);
        stats.contexts++;
        stats.totalPerformance += rule.contextPerformance;
      }
    }
    
    for (const [ruleName, stats] of ruleConsistency.entries()) {
      const frequency = stats.contexts / contexts.length;
      if (frequency >= threshold) {
        patterns.push({
          type: 'universal_rule',
          rule: ruleName,
          frequency: frequency,
          averagePerformance: stats.totalPerformance / stats.contexts,
          description: `Rule '${ruleName}' consistently performs well across contexts`
        });
      }
    }
    
    return patterns;
  }

  findContextSpecificPatterns(contexts) {
    // Patterns unique to specific contexts
    const patterns = [];
    
    for (const context of contexts) {
      // Find rules that perform exceptionally well only in this context
      for (const rule of context.ruleEffectiveness.topRules.slice(0, 3)) {
        if (rule.contextPerformance > 0.9) {
          // Check if this rule performs poorly in other contexts
          const otherContextPerformance = contexts
            .filter(ctx => ctx.contextKey !== context.contextKey)
            .map(ctx => {
              const otherRule = ctx.ruleEffectiveness.topRules
                .concat(ctx.ruleEffectiveness.bottomRules)
                .find(r => r.name === rule.name);
              return otherRule ? otherRule.contextPerformance : 0;
            });
          
          const avgOtherPerformance = otherContextPerformance.length > 0 ?
            otherContextPerformance.reduce((sum, perf) => sum + perf, 0) / otherContextPerformance.length : 0;
          
          if (rule.contextPerformance - avgOtherPerformance >= this.config.significanceThreshold) {
            patterns.push({
              type: 'context_specific_rule',
              context: context.contextKey,
              rule: rule.name,
              contextPerformance: rule.contextPerformance,
              otherContextsPerformance: avgOtherPerformance,
              difference: rule.contextPerformance - avgOtherPerformance,
              description: `Rule '${rule.name}' is highly effective specifically in ${context.contextKey}`
            });
          }
        }
      }
    }
    
    return patterns;
  }

  generatePerformanceComparisons(contexts) {
    return contexts.map(context => ({
      contextKey: context.contextKey,
      averageImprovement: context.performance.averageImprovement,
      successRate: context.performance.successRate,
      consistencyScore: context.performance.consistencyScore,
      sampleSize: context.sampleSize,
      topRule: context.mostEffectiveRules[0]?.name || 'none',
      rank: 0 // Will be filled by caller
    })).sort((a, b) => b.averageImprovement - a.averageImprovement)
      .map((context, index) => ({ ...context, rank: index + 1 }));
  }

  clusterSimilarContexts(contexts) {
    // Simple clustering based on performance similarity
    const clusters = [];
    const threshold = 0.1; // Performance difference threshold for clustering
    
    for (const context of contexts) {
      let assignedCluster = null;
      
      for (const cluster of clusters) {
        const avgPerformance = cluster.contexts.reduce((sum, ctx) => 
          sum + ctx.performance.averageImprovement, 0) / cluster.contexts.length;
        
        if (Math.abs(context.performance.averageImprovement - avgPerformance) <= threshold) {
          assignedCluster = cluster;
          break;
        }
      }
      
      if (assignedCluster) {
        assignedCluster.contexts.push(context);
      } else {
        clusters.push({
          id: clusters.length + 1,
          contexts: [context],
          characteristics: this.identifyClusterCharacteristics([context])
        });
      }
    }
    
    // Update cluster characteristics
    for (const cluster of clusters) {
      cluster.characteristics = this.identifyClusterCharacteristics(cluster.contexts);
    }
    
    return clusters;
  }

  identifyClusterCharacteristics(contexts) {
    // Identify common characteristics across contexts in cluster
    const characteristics = [];
    
    // Common project types
    const projectTypes = contexts.map(ctx => ctx.context.type).filter(Boolean);
    const mostCommonType = this.findMostCommon(projectTypes);
    if (mostCommonType) {
      characteristics.push(`Primary project type: ${mostCommonType}`);
    }
    
    // Common languages/frameworks
    const languages = contexts.flatMap(ctx => 
      ctx.context.lang ? [ctx.context.lang] : []
    );
    const mostCommonLang = this.findMostCommon(languages);
    if (mostCommonLang) {
      characteristics.push(`Primary language: ${mostCommonLang}`);
    }
    
    // Performance level
    const avgPerformance = contexts.reduce((sum, ctx) => 
      sum + ctx.performance.averageImprovement, 0) / contexts.length;
    
    const performanceLevel = avgPerformance > 0.8 ? 'High' : 
                           avgPerformance > 0.5 ? 'Medium' : 'Low';
    characteristics.push(`Performance level: ${performanceLevel}`);
    
    return characteristics;
  }

  findMostCommon(array) {
    const frequency = {};
    let maxCount = 0;
    let mostCommon = null;
    
    for (const item of array) {
      frequency[item] = (frequency[item] || 0) + 1;
      if (frequency[item] > maxCount) {
        maxCount = frequency[item];
        mostCommon = item;
      }
    }
    
    return mostCommon;
  }

  // Additional helper methods for pattern analysis
  analyzeRuleApplicationPatterns(results) {
    // Analyze how rules are typically applied together
    const patterns = [];
    const ruleCombinations = new Map();
    
    for (const result of results) {
      if (result.appliedRules && result.appliedRules.length > 1) {
        const ruleNames = result.appliedRules.map(r => r.name).sort();
        const combinationKey = ruleNames.join('|');
        
        if (!ruleCombinations.has(combinationKey)) {
          ruleCombinations.set(combinationKey, {
            rules: ruleNames,
            frequency: 0,
            totalImprovement: 0,
            successes: 0
          });
        }
        
        const combo = ruleCombinations.get(combinationKey);
        combo.frequency++;
        combo.totalImprovement += result.overallImprovement || 0;
        if ((result.overallImprovement || 0) >= this.config.improvementThreshold) {
          combo.successes++;
        }
      }
    }
    
    // Find significant combinations
    for (const [key, combo] of ruleCombinations.entries()) {
      if (combo.frequency >= 3) { // Minimum frequency threshold
        patterns.push({
          description: `Rules often applied together: ${combo.rules.join(', ')}`,
          frequency: combo.frequency / results.length,
          successRate: combo.successes / combo.frequency,
          averageImprovement: combo.totalImprovement / combo.frequency
        });
      }
    }
    
    return patterns;
  }

  analyzeImprovementPatterns(results) {
    const patterns = [];
    const improvements = results.map(r => r.overallImprovement || 0);
    
    // High improvement pattern
    const highImprovements = improvements.filter(imp => imp > 0.8);
    if (highImprovements.length > 0) {
      patterns.push({
        description: `High improvement cases (>0.8) represent ${(highImprovements.length / improvements.length * 100).toFixed(1)}% of results`,
        frequency: highImprovements.length / improvements.length,
        averageImprovement: highImprovements.reduce((sum, imp) => sum + imp, 0) / highImprovements.length
      });
    }
    
    // Moderate improvement pattern
    const moderateImprovements = improvements.filter(imp => imp >= 0.5 && imp <= 0.8);
    if (moderateImprovements.length > 0) {
      patterns.push({
        description: `Moderate improvement cases (0.5-0.8) represent ${(moderateImprovements.length / improvements.length * 100).toFixed(1)}% of results`,
        frequency: moderateImprovements.length / improvements.length,
        averageImprovement: moderateImprovements.reduce((sum, imp) => sum + imp, 0) / moderateImprovements.length
      });
    }
    
    return patterns;
  }

  generateContextRecommendations(results, context, ruleEffectiveness, patterns) {
    const recommendations = [];
    
    // Rule-based recommendations
    if (ruleEffectiveness.topRules.length > 0) {
      const topRule = ruleEffectiveness.topRules[0];
      recommendations.push({
        type: 'rule_emphasis',
        priority: 'high',
        recommendation: `Emphasize '${topRule.name}' in ${context.type || 'this'} contexts`,
        rationale: `Shows ${(topRule.successRate * 100).toFixed(1)}% success rate with ${topRule.applications} applications`
      });
    }
    
    // Pattern-based recommendations
    if (patterns.successPatterns.length > 0) {
      const pattern = patterns.successPatterns[0];
      recommendations.push({
        type: 'pattern_optimization',
        priority: 'medium',
        recommendation: `Optimize for success pattern: ${pattern.description}`,
        rationale: `Observed in ${(pattern.frequency * 100).toFixed(1)}% of successful cases`
      });
    }
    
    // Performance-based recommendations
    const performance = this.calculateContextPerformance(results);
    if (performance.consistencyScore < 0.5) {
      recommendations.push({
        type: 'consistency_improvement',
        priority: 'medium',
        recommendation: `Improve consistency in ${context.type || 'this'} context`,
        rationale: `Current consistency score: ${(performance.consistencyScore * 100).toFixed(1)}%`
      });
    }
    
    return recommendations;
  }

  findDifferentiatingFactors(successResults, failureResults) {
    // Compare characteristics between success and failure cases
    const factors = [];
    
    const successChars = successResults.map(r => this.analyzePromptCharacteristics(r));
    const failureChars = failureResults.map(r => this.analyzePromptCharacteristics(r));
    
    // Compare boolean characteristics
    const booleanProps = ['hasExamples', 'hasStructure', 'hasConstraints', 'hasFormat'];
    for (const prop of booleanProps) {
      const successRate = successChars.filter(c => c[prop]).length / successChars.length;
      const failureRate = failureChars.filter(c => c[prop]).length / failureChars.length;
      const difference = successRate - failureRate;
      
      if (Math.abs(difference) > 0.3) { // Significant difference
        factors.push({
          factor: prop,
          successRate: successRate,
          failureRate: failureRate,
          difference: difference,
          significance: Math.abs(difference) > 0.5 ? 'high' : 'medium'
        });
      }
    }
    
    return factors.sort((a, b) => Math.abs(b.difference) - Math.abs(a.difference));
  }

  identifyCriticalSuccessFactors(successResults) {
    const characteristics = successResults.map(r => this.analyzePromptCharacteristics(r));
    const factors = [];
    
    // Find characteristics present in most successful cases
    const booleanProps = ['hasExamples', 'hasStructure', 'hasConstraints', 'hasFormat'];
    for (const prop of booleanProps) {
      const frequency = characteristics.filter(c => c[prop]).length / characteristics.length;
      if (frequency > 0.7) { // Present in >70% of successful cases
        factors.push({
          factor: prop,
          frequency: frequency,
          description: `${prop.replace(/^has/, '')} present in ${(frequency * 100).toFixed(1)}% of successful cases`
        });
      }
    }
    
    return factors.sort((a, b) => b.frequency - a.frequency);
  }

  identifyCommonFailurePoints(failureResults) {
    const characteristics = failureResults.map(r => this.analyzePromptCharacteristics(r));
    const points = [];
    
    // Find characteristics common in failures
    const avgLength = characteristics.reduce((sum, c) => sum + c.length, 0) / characteristics.length;
    const avgWordCount = characteristics.reduce((sum, c) => sum + c.wordCount, 0) / characteristics.length;
    
    if (avgLength < 50) {
      points.push({
        point: 'short_prompts',
        description: 'Prompts tend to be very short (average: ' + avgLength.toFixed(0) + ' characters)',
        frequency: characteristics.filter(c => c.length < 50).length / characteristics.length
      });
    }
    
    if (avgWordCount < 10) {
      points.push({
        point: 'insufficient_detail',
        description: 'Prompts lack sufficient detail (average: ' + avgWordCount.toFixed(0) + ' words)',
        frequency: characteristics.filter(c => c.wordCount < 10).length / characteristics.length
      });
    }
    
    // Check for missing structure
    const structureRate = characteristics.filter(c => c.hasStructure).length / characteristics.length;
    if (structureRate < 0.2) {
      points.push({
        point: 'missing_structure',
        description: 'Most prompts lack clear structure',
        frequency: 1 - structureRate
      });
    }
    
    return points.sort((a, b) => b.frequency - a.frequency);
  }

  analyzeCategoryPerformance(results) {
    // Analyze performance by category if available
    const categoryStats = new Map();
    
    for (const result of results) {
      const category = result.category || 'unknown';
      
      if (!categoryStats.has(category)) {
        categoryStats.set(category, {
          category,
          count: 0,
          totalImprovement: 0,
          successes: 0
        });
      }
      
      const stats = categoryStats.get(category);
      stats.count++;
      stats.totalImprovement += result.overallImprovement || 0;
      if ((result.overallImprovement || 0) >= this.config.improvementThreshold) {
        stats.successes++;
      }
    }
    
    return Array.from(categoryStats.values()).map(stats => ({
      ...stats,
      averageImprovement: stats.totalImprovement / stats.count,
      successRate: stats.successes / stats.count
    })).sort((a, b) => b.averageImprovement - a.averageImprovement);
  }

  analyzeComplexityTrends(results) {
    const complexityStats = new Map();
    
    for (const result of results) {
      const complexity = result.complexity || 'unknown';
      
      if (!complexityStats.has(complexity)) {
        complexityStats.set(complexity, {
          complexity,
          count: 0,
          totalImprovement: 0,
          successes: 0
        });
      }
      
      const stats = complexityStats.get(complexity);
      stats.count++;
      stats.totalImprovement += result.overallImprovement || 0;
      if ((result.overallImprovement || 0) >= this.config.improvementThreshold) {
        stats.successes++;
      }
    }
    
    return Array.from(complexityStats.values()).map(stats => ({
      ...stats,
      averageImprovement: stats.totalImprovement / stats.count,
      successRate: stats.successes / stats.count
    })).sort((a, b) => {
      const order = { simple: 1, medium: 2, complex: 3, unknown: 4 };
      return (order[a.complexity] || 4) - (order[b.complexity] || 4);
    });
  }

  calculateRuleDistribution(rules) {
    const total = rules.length;
    return {
      highPerformance: rules.filter(r => r.contextPerformance > 0.8).length / total,
      mediumPerformance: rules.filter(r => r.contextPerformance >= 0.5 && r.contextPerformance <= 0.8).length / total,
      lowPerformance: rules.filter(r => r.contextPerformance < 0.5).length / total
    };
  }
}

module.exports = ContextSpecificLearner;