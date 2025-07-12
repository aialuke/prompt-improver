/**
 * Rule Effectiveness Analysis Engine
 * Identifies top/bottom performing rules and tracks their effectiveness across contexts
 */

const fs = require('fs').promises;
const path = require('path');

class RuleEffectivenessAnalyzer {
  constructor(config = {}) {
    this.config = {
      // Analysis thresholds
      successThreshold: 0.7,           // Minimum improvement score to count as success
      significanceThreshold: 0.1,      // Minimum difference for statistical significance
      minApplications: 5,              // Minimum rule applications for reliable statistics
      
      // Rule ranking parameters
      topPerformersCount: 10,
      underperformersCount: 10,
      
      // Context analysis
      contextSimilarityThreshold: 0.8,
      
      ...config
    };

    // Initialize rule tracking
    this.ruleEffectiveness = new Map();
    this.ruleSet = null;
    this.globalStats = {
      totalApplications: 0,
      averageImprovement: 0,
      successRate: 0
    };
  }

  /**
   * Load prompt engineering rules from configuration
   * @param {string} rulesPath - Path to rules JSON file
   */
  async loadRuleSet(rulesPath) {
    try {
      const rulesContent = await fs.readFile(rulesPath, 'utf8');
      this.ruleSet = JSON.parse(rulesContent);
      return this.ruleSet;
    } catch (error) {
      throw new Error(`Failed to load rule set: ${error.message}`);
    }
  }

  /**
   * Analyze rule performance across test results
   * @param {Array} testResults - Array of test results with original/improved prompts
   * @returns {Object} Rule effectiveness analysis
   */
  async analyzeRulePerformance(testResults) {
    if (!this.ruleSet) {
      throw new Error('Rule set not loaded. Call loadRuleSet() first.');
    }

    // Reset tracking
    this.ruleEffectiveness.clear();
    
    // Process each test result
    for (const result of testResults) {
      await this.processTestResult(result);
    }

    // Calculate global statistics
    this.calculateGlobalStats(testResults);

    // Generate comprehensive analysis
    const analysis = {
      summary: this.generateSummary(),
      ruleRankings: this.generateRuleRankings(),
      categoryAnalysis: this.analyzeCategoryPerformance(),
      contextualAnalysis: this.analyzeContextualPerformance(),
      statisticalSignificance: this.calculateStatisticalSignificance(),
      recommendations: this.generateRecommendations(),
      detailedStats: this.getDetailedStats()
    };

    return analysis;
  }

  /**
   * Process individual test result to track rule applications
   * @private
   */
  async processTestResult(result) {
    const appliedRules = await this.identifyAppliedRules(
      result.originalPrompt, 
      result.improvedPrompt,
      result.context
    );

    for (const rule of appliedRules) {
      this.trackRuleApplication(rule, result);
    }
  }

  /**
   * Identify which rules were applied in a prompt improvement
   * @param {string} original - Original prompt
   * @param {string} improved - Improved prompt
   * @param {Object} context - Test context
   * @returns {Array} Applied rules with metadata
   */
  async identifyAppliedRules(original, improved, context = {}) {
    const appliedRules = [];

    // Check each rule category
    for (const [categoryName, category] of Object.entries(this.ruleSet.corePrinciples || {})) {
      if (category.checkFunctions) {
        for (const checkFunction of category.checkFunctions) {
          const ruleKey = `${categoryName}_${checkFunction}`;
          
          if (await this.wasRuleApplied(original, improved, categoryName, checkFunction, context)) {
            appliedRules.push({
              name: ruleKey,
              category: categoryName,
              checkFunction: checkFunction,
              rule: category,
              confidence: this.calculateRuleConfidence(original, improved, checkFunction)
            });
          }
        }
      }
    }

    // Check improvement patterns
    const improvementPatterns = this.detectImprovementPatterns(original, improved);
    for (const pattern of improvementPatterns) {
      appliedRules.push({
        name: `pattern_${pattern.type}`,
        category: 'patterns',
        pattern: pattern,
        confidence: pattern.confidence
      });
    }

    return appliedRules;
  }

  /**
   * Determine if a specific rule was applied
   * @private
   */
  async wasRuleApplied(original, improved, category, checkFunction, context) {
    const originalScore = this.evaluateRuleCompliance(original, category, checkFunction);
    const improvedScore = this.evaluateRuleCompliance(improved, category, checkFunction);
    
    // Rule was applied if improved prompt scores significantly higher
    const improvement = improvedScore - originalScore;
    const threshold = 0.2; // Minimum improvement to consider rule applied
    
    return improvement >= threshold;
  }

  /**
   * Evaluate how well a prompt complies with a specific rule
   * @private
   */
  evaluateRuleCompliance(prompt, category, checkFunction) {
    let score = 0;

    switch (checkFunction) {
      case 'Contains specific action verbs':
        score = this.countActionVerbs(prompt) > 0 ? 1 : 0;
        break;
        
      case 'Includes measurable constraints (word count, format, etc.)':
        score = this.hasMeasurableConstraints(prompt) ? 1 : 0;
        break;
        
      case 'Defines expected output structure':
        score = this.hasOutputStructure(prompt) ? 1 : 0;
        break;
        
      case 'Avoids vague adjectives like \'good\', \'better\', \'nice\'':
        score = 1 - Math.min(this.countVagueAdjectives(prompt) * 0.2, 1);
        break;
        
      case 'Provides concrete examples':
        score = this.hasConcreteExamples(prompt) ? 1 : 0;
        break;
        
      case 'Uses XML-style tags for organization':
        score = this.hasXMLTags(prompt) ? 1 : 0;
        break;
        
      case 'Includes role definition':
        score = this.hasRoleDefinition(prompt) ? 1 : 0;
        break;
        
      case 'Specifies output format':
        score = this.hasOutputFormat(prompt) ? 1 : 0;
        break;
        
      default:
        // Generic rule evaluation based on category
        score = this.evaluateGenericRule(prompt, category, checkFunction);
    }

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Track rule application and its outcome
   * @private
   */
  trackRuleApplication(rule, result) {
    const ruleKey = rule.name;
    
    if (!this.ruleEffectiveness.has(ruleKey)) {
      this.ruleEffectiveness.set(ruleKey, {
        name: ruleKey,
        category: rule.category,
        applications: 0,
        successes: 0,
        failures: 0,
        totalImprovement: 0,
        contexts: [],
        examples: [],
        improvements: [],
        confidence: 0
      });
    }

    const stats = this.ruleEffectiveness.get(ruleKey);
    stats.applications++;
    stats.totalImprovement += result.overallImprovement || 0;
    stats.contexts.push(result.context);
    stats.improvements.push(result.overallImprovement || 0);
    stats.confidence += rule.confidence || 0.5;

    // Track success/failure
    if ((result.overallImprovement || 0) >= this.config.successThreshold) {
      stats.successes++;
      stats.examples.push({
        type: 'success',
        original: result.originalPrompt,
        improved: result.improvedPrompt,
        improvement: result.overallImprovement,
        context: result.context
      });
    } else {
      stats.failures++;
      stats.examples.push({
        type: 'failure',
        original: result.originalPrompt,
        improved: result.improvedPrompt,
        improvement: result.overallImprovement,
        context: result.context
      });
    }
  }

  /**
   * Calculate global statistics across all rules
   * @private
   */
  calculateGlobalStats(testResults) {
    this.globalStats.totalApplications = testResults.length;
    this.globalStats.averageImprovement = testResults.reduce((sum, r) => 
      sum + (r.overallImprovement || 0), 0) / testResults.length;
    this.globalStats.successRate = testResults.filter(r => 
      (r.overallImprovement || 0) >= this.config.successThreshold).length / testResults.length;
  }

  /**
   * Generate rule rankings by effectiveness
   * @private
   */
  generateRuleRankings() {
    const rules = Array.from(this.ruleEffectiveness.values());
    
    // Filter rules with sufficient applications
    const reliableRules = rules.filter(rule => 
      rule.applications >= this.config.minApplications);

    // Calculate effectiveness scores
    for (const rule of reliableRules) {
      rule.successRate = rule.successes / rule.applications;
      rule.averageImprovement = rule.totalImprovement / rule.applications;
      rule.averageConfidence = rule.confidence / rule.applications;
      
      // Combined effectiveness score
      rule.effectivenessScore = (rule.successRate * 0.6) + 
                              (rule.averageImprovement * 0.3) + 
                              (rule.averageConfidence * 0.1);
    }

    // Sort by effectiveness
    reliableRules.sort((a, b) => b.effectivenessScore - a.effectivenessScore);

    return {
      topPerformers: reliableRules.slice(0, this.config.topPerformersCount),
      underperformers: reliableRules.slice(-this.config.underperformersCount).reverse(),
      allRules: reliableRules
    };
  }

  /**
   * Analyze performance by rule category
   * @private
   */
  analyzeCategoryPerformance() {
    const categoryStats = new Map();
    
    for (const rule of this.ruleEffectiveness.values()) {
      if (!categoryStats.has(rule.category)) {
        categoryStats.set(rule.category, {
          category: rule.category,
          totalApplications: 0,
          totalSuccesses: 0,
          totalImprovement: 0,
          rules: []
        });
      }
      
      const stats = categoryStats.get(rule.category);
      stats.totalApplications += rule.applications;
      stats.totalSuccesses += rule.successes;
      stats.totalImprovement += rule.totalImprovement;
      stats.rules.push(rule);
    }

    // Calculate category metrics
    const categories = Array.from(categoryStats.values()).map(cat => ({
      ...cat,
      successRate: cat.totalSuccesses / cat.totalApplications,
      averageImprovement: cat.totalImprovement / cat.totalApplications,
      topRule: cat.rules.reduce((best, rule) => 
        !best || rule.effectivenessScore > best.effectivenessScore ? rule : best, null)
    }));

    return categories.sort((a, b) => b.successRate - a.successRate);
  }

  /**
   * Analyze rule performance across different contexts
   * @private
   */
  analyzeContextualPerformance() {
    const contextPerformance = new Map();
    
    for (const rule of this.ruleEffectiveness.values()) {
      for (let i = 0; i < rule.contexts.length; i++) {
        const context = rule.contexts[i];
        const improvement = rule.improvements[i];
        const contextKey = this.getContextKey(context);
        
        if (!contextPerformance.has(contextKey)) {
          contextPerformance.set(contextKey, {
            context: contextKey,
            rules: new Map(),
            totalTests: 0,
            averageImprovement: 0
          });
        }
        
        const contextStats = contextPerformance.get(contextKey);
        contextStats.totalTests++;
        contextStats.averageImprovement += improvement;
        
        if (!contextStats.rules.has(rule.name)) {
          contextStats.rules.set(rule.name, {
            applications: 0,
            totalImprovement: 0,
            successes: 0
          });
        }
        
        const ruleStats = contextStats.rules.get(rule.name);
        ruleStats.applications++;
        ruleStats.totalImprovement += improvement;
        if (improvement >= this.config.successThreshold) {
          ruleStats.successes++;
        }
      }
    }

    // Calculate final context statistics
    const contexts = Array.from(contextPerformance.values()).map(ctx => {
      ctx.averageImprovement /= ctx.totalTests;
      
      // Find best performing rules for this context
      const contextRules = Array.from(ctx.rules.entries()).map(([name, stats]) => ({
        name,
        successRate: stats.successes / stats.applications,
        averageImprovement: stats.totalImprovement / stats.applications,
        applications: stats.applications
      }));
      
      contextRules.sort((a, b) => b.successRate - a.successRate);
      ctx.topRules = contextRules.slice(0, 5);
      ctx.rules = contextRules;
      
      return ctx;
    });

    return contexts.sort((a, b) => b.averageImprovement - a.averageImprovement);
  }

  /**
   * Calculate statistical significance of rule effectiveness
   * @private
   */
  calculateStatisticalSignificance() {
    const significantRules = [];
    
    for (const rule of this.ruleEffectiveness.values()) {
      if (rule.applications >= this.config.minApplications) {
        // Simple z-test for proportion difference
        const p1 = rule.successRate;
        const p2 = this.globalStats.successRate;
        const n = rule.applications;
        
        const pooledP = (rule.successes + this.globalStats.successRate * this.globalStats.totalApplications) / 
                       (n + this.globalStats.totalApplications);
        const se = Math.sqrt(pooledP * (1 - pooledP) * (1/n + 1/this.globalStats.totalApplications));
        
        if (se > 0) {
          const zScore = (p1 - p2) / se;
          const pValue = 2 * (1 - this.normalCDF(Math.abs(zScore)));
          
          if (pValue < 0.05) { // Significant at 5% level
            significantRules.push({
              rule: rule.name,
              zScore: zScore,
              pValue: pValue,
              effectSize: p1 - p2,
              significance: pValue < 0.01 ? 'highly_significant' : 'significant'
            });
          }
        }
      }
    }

    return significantRules.sort((a, b) => a.pValue - b.pValue);
  }

  /**
   * Generate actionable recommendations based on analysis
   * @private
   */
  generateRecommendations() {
    const recommendations = [];
    const rankings = this.generateRuleRankings();
    
    // High-performing rule recommendations
    if (rankings.topPerformers.length > 0) {
      const topRule = rankings.topPerformers[0];
      recommendations.push({
        type: 'strengthen',
        priority: 'high',
        rule: topRule.name,
        recommendation: `Emphasize '${topRule.name}' - it shows ${(topRule.successRate * 100).toFixed(1)}% success rate`,
        impact: 'high',
        evidence: `${topRule.applications} applications, ${topRule.averageImprovement.toFixed(3)} average improvement`
      });
    }

    // Underperforming rule recommendations
    if (rankings.underperformers.length > 0) {
      const worstRule = rankings.underperformers[0];
      recommendations.push({
        type: 'improve',
        priority: 'medium',
        rule: worstRule.name,
        recommendation: `Revise '${worstRule.name}' - only ${(worstRule.successRate * 100).toFixed(1)}% success rate`,
        impact: 'medium',
        evidence: `${worstRule.applications} applications, needs refinement`
      });
    }

    // Context-specific recommendations
    const contextAnalysis = this.analyzeContextualPerformance();
    if (contextAnalysis.length > 0) {
      const bestContext = contextAnalysis[0];
      recommendations.push({
        type: 'contextualize',
        priority: 'medium',
        context: bestContext.context,
        recommendation: `Optimize rules for '${bestContext.context}' context - shows highest improvement rates`,
        impact: 'medium',
        evidence: `${bestContext.totalTests} tests, ${bestContext.averageImprovement.toFixed(3)} average improvement`
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  // Helper methods for rule evaluation
  countActionVerbs(prompt) {
    const actionVerbs = [
      'create', 'build', 'write', 'implement', 'design', 'develop',
      'analyze', 'evaluate', 'calculate', 'determine', 'generate',
      'optimize', 'improve', 'enhance', 'refactor', 'debug'
    ];
    const words = prompt.toLowerCase().split(/\s+/);
    return actionVerbs.filter(verb => words.includes(verb)).length;
  }

  hasMeasurableConstraints(prompt) {
    const constraintPatterns = [
      /\d+\s*words?/i,
      /\d+\s*characters?/i,
      /\d+\s*pages?/i,
      /\d+\s*items?/i,
      /exactly|approximately|at least|no more than/i,
      /json|xml|csv|markdown/i
    ];
    return constraintPatterns.some(pattern => pattern.test(prompt));
  }

  hasOutputStructure(prompt) {
    const structureIndicators = [
      /format:|structure:|template:/i,
      /json|xml|csv|yaml/i,
      /headers?|sections?|columns?/i,
      /list|table|grid/i,
      /<[^>]+>/,  // XML-like tags
      /\{[^}]+\}/ // JSON-like structure
    ];
    return structureIndicators.some(indicator => indicator.test(prompt));
  }

  countVagueAdjectives(prompt) {
    const vagueAdjectives = [
      'good', 'better', 'best', 'nice', 'great', 'awesome', 'cool',
      'bad', 'worse', 'worst', 'terrible', 'horrible', 'awful',
      'big', 'small', 'large', 'huge', 'tiny', 'massive',
      'important', 'relevant', 'significant', 'appropriate'
    ];
    const words = prompt.toLowerCase().split(/\s+/);
    return vagueAdjectives.filter(adj => words.includes(adj)).length;
  }

  hasConcreteExamples(prompt) {
    const exampleMarkers = [
      /for example/i, /e\.g\./i, /such as/i, /like/i, /including/i,
      /for instance/i, /specifically/i, /namely/i
    ];
    return exampleMarkers.some(marker => marker.test(prompt));
  }

  hasXMLTags(prompt) {
    return /<[^>]+>/.test(prompt);
  }

  hasRoleDefinition(prompt) {
    const roleMarkers = [
      /as a|you are a|act as/i,
      /role:|position:|perspective:/i,
      /expert|specialist|professional|developer|engineer/i
    ];
    return roleMarkers.some(marker => marker.test(prompt));
  }

  hasOutputFormat(prompt) {
    const formatMarkers = [
      /format|output|return|respond|provide/i,
      /json|xml|csv|markdown|html/i,
      /table|list|bullet|numbered/i
    ];
    return formatMarkers.some(marker => marker.test(prompt));
  }

  detectImprovementPatterns(original, improved) {
    const patterns = [];
    
    // Length change pattern
    const lengthRatio = improved.length / original.length;
    if (lengthRatio > 1.5) {
      patterns.push({
        type: 'expansion',
        confidence: Math.min((lengthRatio - 1) / 2, 1),
        description: 'Significant content expansion'
      });
    } else if (lengthRatio < 0.7) {
      patterns.push({
        type: 'condensation',
        confidence: Math.min((1 - lengthRatio) / 0.3, 1),
        description: 'Content condensation'
      });
    }

    // Structure addition pattern
    if (!this.hasXMLTags(original) && this.hasXMLTags(improved)) {
      patterns.push({
        type: 'structure_addition',
        confidence: 0.9,
        description: 'Added XML structure tags'
      });
    }

    // Example addition pattern
    if (!this.hasConcreteExamples(original) && this.hasConcreteExamples(improved)) {
      patterns.push({
        type: 'example_addition',
        confidence: 0.8,
        description: 'Added concrete examples'
      });
    }

    return patterns;
  }

  calculateRuleConfidence(original, improved, checkFunction) {
    // Calculate confidence based on how definitively the rule was applied
    const originalCompliance = this.evaluateRuleCompliance(original, '', checkFunction);
    const improvedCompliance = this.evaluateRuleCompliance(improved, '', checkFunction);
    const improvement = improvedCompliance - originalCompliance;
    
    return Math.max(0.1, Math.min(1.0, improvement + 0.5));
  }

  evaluateGenericRule(prompt, category, checkFunction) {
    // Fallback evaluation for rules not explicitly handled
    return Math.random() * 0.5 + 0.25; // Random score between 0.25-0.75
  }

  getContextKey(context) {
    if (!context) return 'unknown';
    
    const parts = [];
    if (context.projectType) parts.push(context.projectType);
    if (context.domain) parts.push(context.domain);
    if (context.languages && context.languages.length > 0) {
      parts.push(context.languages[0]);
    }
    
    return parts.join('_') || 'general';
  }

  // Statistical helper methods
  normalCDF(x) {
    // Approximation of normal cumulative distribution function
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - prob : prob;
  }

  generateSummary() {
    const totalRules = this.ruleEffectiveness.size;
    const reliableRules = Array.from(this.ruleEffectiveness.values())
      .filter(rule => rule.applications >= this.config.minApplications);
    
    return {
      totalRulesTracked: totalRules,
      reliableRules: reliableRules.length,
      globalSuccessRate: this.globalStats.successRate,
      averageImprovement: this.globalStats.averageImprovement,
      totalApplications: this.globalStats.totalApplications,
      analysisDate: new Date().toISOString()
    };
  }

  getDetailedStats() {
    return {
      ruleEffectiveness: Object.fromEntries(this.ruleEffectiveness),
      globalStats: this.globalStats,
      config: this.config
    };
  }
}

module.exports = RuleEffectivenessAnalyzer;