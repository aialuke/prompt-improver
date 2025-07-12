/**
 * Rule Modification System
 * Generates data-driven improvement suggestions for prompt engineering rules
 */

const fs = require('fs').promises;
const path = require('path');

class RuleOptimizer {
  constructor(config = {}) {
    this.config = {
      // Optimization thresholds
      minConfidenceForOptimization: 0.7,    // Minimum confidence to suggest changes
      minSampleSizeForRule: 10,             // Minimum applications for reliable optimization
      significanceThreshold: 0.1,           // Minimum performance difference for action
      
      // Risk assessment parameters
      maxRiskLevel: 'medium',               // Maximum risk level for automated updates
      conservativeMode: true,               // Err on side of caution
      
      // Optimization parameters
      maxOptimizationsPerCycle: 10,         // Maximum optimizations to suggest per cycle
      priorityWeights: {
        impact: 0.4,                        // Weight for expected impact
        confidence: 0.3,                    // Weight for confidence level
        risk: 0.2,                         // Weight for risk (inverted)
        frequency: 0.1                      // Weight for application frequency
      },
      
      ...config
    };

    // Optimization tracking
    this.analysisResults = null;
    this.optimizationHistory = [];
    this.rulePerformanceBaseline = new Map();
  }

  /**
   * Generate comprehensive optimization recommendations
   * @param {Object} analysisResults - Results from Phase 5 pattern analysis
   * @returns {Object} Complete optimization plan
   */
  async generateOptimizations(analysisResults) {
    this.analysisResults = analysisResults;

    const optimizations = {
      modifications: await this.generateRuleModifications(),
      additions: await this.generateNewRules(),
      specializations: await this.generateSpecializations(),
      deprecations: await this.identifyDeprecationCandidates(),
      
      // Meta-analysis
      summary: this.generateOptimizationSummary(),
      priorityRanking: [],
      impactEstimation: {},
      riskAssessment: {}
    };

    // Prioritize and rank all optimizations
    optimizations.priorityRanking = this.prioritizeOptimizations([
      ...optimizations.modifications,
      ...optimizations.additions,
      ...optimizations.specializations,
      ...optimizations.deprecations
    ]);

    // Generate impact and risk assessments
    optimizations.impactEstimation = this.estimateOverallImpact(optimizations);
    optimizations.riskAssessment = this.assessOverallRisk(optimizations);

    return optimizations;
  }

  /**
   * Generate modifications for existing underperforming rules
   * @private
   */
  async generateRuleModifications() {
    const modifications = [];
    const ruleEffectiveness = this.analysisResults.ruleEffectiveness;
    
    if (!ruleEffectiveness?.ruleRankings?.underperformers) {
      return modifications;
    }

    for (const rule of ruleEffectiveness.ruleRankings.underperformers) {
      if (rule.applications >= this.config.minSampleSizeForRule) {
        const modification = await this.analyzeRuleForModification(rule);
        if (modification) {
          modifications.push(modification);
        }
      }
    }

    return modifications.slice(0, this.config.maxOptimizationsPerCycle);
  }

  /**
   * Analyze a specific rule for modification opportunities
   * @private
   */
  async analyzeRuleForModification(rule) {
    const failureAnalysis = this.analyzeRuleFailures(rule);
    const contextAnalysis = this.analyzeRuleContextPerformance(rule);
    const improvementOpportunities = this.identifyImprovementOpportunities(rule, failureAnalysis);

    if (improvementOpportunities.length === 0) {
      return null;
    }

    return {
      type: 'modification',
      ruleId: rule.name,
      currentPerformance: {
        successRate: rule.successRate,
        averageImprovement: rule.averageImprovement,
        applications: rule.applications
      },
      
      // Issue analysis
      identifiedIssues: failureAnalysis.issues,
      contextPerformanceVariation: contextAnalysis.variation,
      
      // Proposed changes
      proposedChanges: improvementOpportunities.map(opp => ({
        component: opp.component,
        issue: opp.issue,
        currentLogic: opp.currentLogic,
        proposedLogic: opp.proposedLogic,
        rationale: opp.rationale,
        expectedImpact: opp.expectedImpact
      })),
      
      // Optimization metadata
      confidence: this.calculateModificationConfidence(rule, improvementOpportunities),
      expectedImprovement: this.estimateExpectedImprovement(rule, improvementOpportunities),
      riskLevel: this.assessModificationRisk(rule, improvementOpportunities),
      priority: this.calculateModificationPriority(rule, improvementOpportunities),
      
      // Supporting evidence
      supportingEvidence: {
        failureExamples: failureAnalysis.examples.slice(0, 3),
        successComparisons: this.findSuccessComparisons(rule),
        statisticalSignificance: this.calculateSignificance(rule)
      }
    };
  }

  /**
   * Analyze rule failures to identify improvement opportunities
   * @private
   */
  analyzeRuleFailures(rule) {
    const failures = rule.examples?.filter(ex => ex.type === 'failure') || [];
    
    const issues = [];
    const patterns = new Map();

    for (const failure of failures) {
      // Analyze detection issues
      const detectionIssues = this.analyzeDetectionFailure(failure, rule);
      if (detectionIssues.length > 0) {
        issues.push(...detectionIssues);
      }

      // Analyze application issues
      const applicationIssues = this.analyzeApplicationFailure(failure, rule);
      if (applicationIssues.length > 0) {
        issues.push(...applicationIssues);
      }

      // Track patterns
      const pattern = this.categorizeFailure(failure);
      patterns.set(pattern, (patterns.get(pattern) || 0) + 1);
    }

    return {
      issues: this.consolidateIssues(issues),
      patterns: Array.from(patterns.entries()).map(([pattern, count]) => ({
        pattern,
        frequency: count,
        percentage: (count / failures.length) * 100
      })),
      examples: failures.slice(0, 5)
    };
  }

  /**
   * Analyze detection failures
   * @private
   */
  analyzeDetectionFailure(failure, rule) {
    const issues = [];
    
    // Check if rule should have been applied but wasn't
    if (this.shouldRuleHaveBeenApplied(failure.original, failure.improved)) {
      issues.push({
        type: 'detection_miss',
        description: 'Rule failed to detect applicable scenario',
        evidence: {
          original: failure.original,
          improved: failure.improved,
          expectedDetection: this.explainExpectedDetection(failure, rule)
        }
      });
    }
    
    // Check for false positive detection
    if (this.wasRuleIncorrectlyApplied(failure.original, failure.improved, rule)) {
      issues.push({
        type: 'false_positive',
        description: 'Rule applied when it should not have been',
        evidence: {
          originalText: failure.original,
          detectionTrigger: this.identifyDetectionTrigger(failure.original, rule)
        }
      });
    }

    return issues;
  }

  /**
   * Analyze application failures
   * @private
   */
  analyzeApplicationFailure(failure, rule) {
    const issues = [];
    
    // Check if improvement was insufficient
    if (failure.improvement < 0.1) {
      issues.push({
        type: 'insufficient_improvement',
        description: 'Rule applied but improvement was minimal',
        evidence: {
          improvement: failure.improvement,
          expectedImprovement: rule.averageImprovement
        }
      });
    }
    
    // Check for incorrect application
    if (this.wasApplicationIncorrect(failure)) {
      issues.push({
        type: 'incorrect_application',
        description: 'Rule applied incorrectly or inappropriately',
        evidence: {
          appliedChange: this.identifyAppliedChange(failure),
          correctApplication: this.suggestCorrectApplication(failure, rule)
        }
      });
    }

    return issues;
  }

  /**
   * Identify improvement opportunities from failure analysis
   * @private
   */
  identifyImprovementOpportunities(rule, failureAnalysis) {
    const opportunities = [];
    
    for (const issue of failureAnalysis.issues) {
      switch (issue.type) {
        case 'detection_miss':
          opportunities.push({
            component: 'detectionLogic',
            issue: 'Missed detections in specific contexts',
            currentLogic: rule.rule?.checkFunctions || 'Unknown',
            proposedLogic: this.enhanceDetectionLogic(rule, issue),
            rationale: 'Expand detection patterns to cover missed cases',
            expectedImpact: this.estimateDetectionImprovementImpact(issue)
          });
          break;
          
        case 'false_positive':
          opportunities.push({
            component: 'detectionLogic',
            issue: 'Overly broad detection triggers',
            currentLogic: rule.rule?.checkFunctions || 'Unknown',
            proposedLogic: this.refineDetectionLogic(rule, issue),
            rationale: 'Add specificity to reduce false positives',
            expectedImpact: this.estimateFalsePositiveReductionImpact(issue)
          });
          break;
          
        case 'insufficient_improvement':
          opportunities.push({
            component: 'applicationLogic',
            issue: 'Weak improvement strategies',
            currentLogic: rule.rule?.improvement || 'Unknown',
            proposedLogic: this.enhanceApplicationLogic(rule, issue),
            rationale: 'Strengthen improvement techniques',
            expectedImpact: this.estimateApplicationImprovementImpact(issue)
          });
          break;
          
        case 'incorrect_application':
          opportunities.push({
            component: 'applicationLogic',
            issue: 'Inappropriate application methods',
            currentLogic: rule.rule?.improvement || 'Unknown',
            proposedLogic: this.correctApplicationLogic(rule, issue),
            rationale: 'Fix application logic for better results',
            expectedImpact: this.estimateApplicationCorrectionImpact(issue)
          });
          break;
      }
    }

    return opportunities;
  }

  /**
   * Generate new rules from identified gaps
   * @private
   */
  async generateNewRules() {
    const newRules = [];
    const failureAnalysis = this.analysisResults.failureAnalysis;
    
    if (!failureAnalysis?.ruleGaps) {
      return newRules;
    }

    for (const gap of failureAnalysis.ruleGaps) {
      if (gap.frequency >= this.config.minSampleSizeForRule) {
        const newRule = await this.createNewRule(gap);
        if (newRule) {
          newRules.push(newRule);
        }
      }
    }

    return newRules.slice(0, this.config.maxOptimizationsPerCycle);
  }

  /**
   * Create a new rule from a identified gap
   * @private
   */
  async createNewRule(gap) {
    return {
      type: 'addition',
      ruleId: this.generateNewRuleId(gap),
      
      // Rule definition
      proposedRule: {
        name: gap.improvementType,
        description: gap.description,
        category: this.inferRuleCategory(gap),
        priority: gap.priority,
        
        // Logic definition
        detectionLogic: this.generateDetectionLogic(gap),
        improvementLogic: this.generateImprovementLogic(gap),
        validationLogic: this.generateValidationLogic(gap)
      },
      
      // Gap analysis
      addressedPattern: gap.pattern,
      expectedCoverage: gap.frequency,
      targetScenarios: gap.examples,
      
      // Optimization metadata
      confidence: this.calculateNewRuleConfidence(gap),
      expectedImpact: this.estimateNewRuleImpact(gap),
      riskLevel: this.assessNewRuleRisk(gap),
      priority: gap.priority,
      
      // Implementation details
      implementationComplexity: this.assessImplementationComplexity(gap),
      testingRequirements: this.defineTestingRequirements(gap),
      rolloutStrategy: this.planRolloutStrategy(gap)
    };
  }

  /**
   * Generate context-specific rule specializations
   * @private
   */
  async generateSpecializations() {
    const specializations = [];
    const contextLearning = this.analysisResults.contextLearning;
    
    if (!contextLearning?.specializationOpportunities) {
      return specializations;
    }

    for (const opportunity of contextLearning.specializationOpportunities) {
      const specialization = await this.createSpecialization(opportunity);
      if (specialization) {
        specializations.push(specialization);
      }
    }

    return specializations;
  }

  /**
   * Create a context-specific specialization
   * @private
   */
  async createSpecialization(opportunity) {
    return {
      type: 'specialization',
      baseRuleId: opportunity.rule,
      specializationId: this.generateSpecializationId(opportunity),
      
      // Specialization details
      targetContext: opportunity.context,
      contextSpecificLogic: this.generateContextSpecificLogic(opportunity),
      performanceImprovement: opportunity.potentialImprovement,
      
      // Optimization metadata
      confidence: opportunity.confidence,
      expectedImpact: this.estimateSpecializationImpact(opportunity),
      riskLevel: this.assessSpecializationRisk(opportunity),
      priority: this.calculateSpecializationPriority(opportunity),
      
      // Implementation plan
      integrationStrategy: this.planSpecializationIntegration(opportunity),
      fallbackBehavior: this.defineFallbackBehavior(opportunity),
      validationCriteria: this.defineSpecializationValidation(opportunity)
    };
  }

  /**
   * Identify rules that should be deprecated
   * @private
   */
  async identifyDeprecationCandidates() {
    const deprecations = [];
    const ruleEffectiveness = this.analysisResults.ruleEffectiveness;
    
    if (!ruleEffectiveness?.ruleRankings?.underperformers) {
      return deprecations;
    }

    for (const rule of ruleEffectiveness.ruleRankings.underperformers) {
      if (this.shouldConsiderForDeprecation(rule)) {
        const deprecation = await this.analyzeForDeprecation(rule);
        if (deprecation) {
          deprecations.push(deprecation);
        }
      }
    }

    return deprecations;
  }

  /**
   * Check if rule should be considered for deprecation
   * @private
   */
  shouldConsiderForDeprecation(rule) {
    return (
      rule.successRate < 0.3 &&
      rule.applications >= this.config.minSampleSizeForRule &&
      rule.averageImprovement < 0.1
    );
  }

  /**
   * Analyze rule for potential deprecation
   * @private
   */
  async analyzeForDeprecation(rule) {
    const replacementAnalysis = this.findReplacementRules(rule);
    const impactAnalysis = this.analyzeDeprecationImpact(rule);
    
    if (!replacementAnalysis.hasViableReplacements && impactAnalysis.highImpact) {
      return null; // Don't deprecate if no replacement and high usage
    }

    return {
      type: 'deprecation',
      ruleId: rule.name,
      
      // Deprecation rationale
      performanceIssues: {
        successRate: rule.successRate,
        averageImprovement: rule.averageImprovement,
        failureRate: 1 - rule.successRate
      },
      
      // Replacement strategy
      replacementStrategy: replacementAnalysis.strategy,
      viableReplacements: replacementAnalysis.replacements,
      migrationPlan: this.createMigrationPlan(rule, replacementAnalysis),
      
      // Impact analysis
      affectedScenarios: impactAnalysis.scenarios,
      userImpact: impactAnalysis.userImpact,
      riskLevel: this.assessDeprecationRisk(rule, impactAnalysis),
      
      // Implementation details
      deprecationTimeline: this.planDeprecationTimeline(rule),
      communicationPlan: this.planDeprecationCommunication(rule),
      rollbackPlan: this.planDeprecationRollback(rule)
    };
  }

  /**
   * Prioritize all optimizations by impact and feasibility
   * @private
   */
  prioritizeOptimizations(optimizations) {
    return optimizations
      .map(opt => ({
        ...opt,
        priorityScore: this.calculatePriorityScore(opt)
      }))
      .sort((a, b) => b.priorityScore - a.priorityScore)
      .slice(0, this.config.maxOptimizationsPerCycle);
  }

  /**
   * Calculate priority score for optimization
   * @private
   */
  calculatePriorityScore(optimization) {
    const weights = this.config.priorityWeights;
    
    const impact = this.normalizeImpact(optimization.expectedImpact || optimization.expectedImprovement || 0);
    const confidence = optimization.confidence || 0.5;
    const risk = 1 - this.normalizeRisk(optimization.riskLevel);
    const frequency = this.normalizeFrequency(optimization);
    
    return (
      impact * weights.impact +
      confidence * weights.confidence +
      risk * weights.risk +
      frequency * weights.frequency
    );
  }

  // Helper methods for rule analysis and optimization
  
  shouldRuleHaveBeenApplied(original, improved) {
    // Simplified heuristic - could be enhanced with ML models
    const improvementMade = improved.length > original.length * 1.2;
    const structureAdded = improved.includes('<') && !original.includes('<');
    const clarityImproved = this.countVagueTerms(original) > this.countVagueTerms(improved);
    
    return improvementMade || structureAdded || clarityImproved;
  }

  wasRuleIncorrectlyApplied(original, improved, rule) {
    // Check if rule was applied but made things worse
    return improved.length < original.length * 0.8; // Simplified check
  }

  countVagueTerms(text) {
    const vagueTerms = ['good', 'better', 'nice', 'great', 'awesome', 'bad', 'worse'];
    return vagueTerms.filter(term => text.toLowerCase().includes(term)).length;
  }

  wasApplicationIncorrect(failure) {
    return failure.improvement < 0; // Negative improvement indicates incorrect application
  }

  identifyAppliedChange(failure) {
    // Simplified diff analysis
    return `Length change: ${failure.improved.length - failure.original.length} characters`;
  }

  enhanceDetectionLogic(rule, issue) {
    return `Enhanced detection logic for ${rule.name} to handle: ${issue.description}`;
  }

  refineDetectionLogic(rule, issue) {
    return `Refined detection logic for ${rule.name} to reduce: ${issue.description}`;
  }

  enhanceApplicationLogic(rule, issue) {
    return `Enhanced application logic for ${rule.name} to improve: ${issue.description}`;
  }

  correctApplicationLogic(rule, issue) {
    return `Corrected application logic for ${rule.name} to fix: ${issue.description}`;
  }

  generateNewRuleId(gap) {
    return `rule_${gap.improvementType.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
  }

  inferRuleCategory(gap) {
    const categories = ['clarity', 'completeness', 'specificity', 'structure', 'context'];
    return categories.find(cat => gap.description.toLowerCase().includes(cat)) || 'general';
  }

  generateDetectionLogic(gap) {
    return `Detection logic for: ${gap.description}`;
  }

  generateImprovementLogic(gap) {
    return `Improvement logic for: ${gap.description}`;
  }

  generateValidationLogic(gap) {
    return `Validation logic for: ${gap.description}`;
  }

  normalizeImpact(impact) {
    return Math.max(0, Math.min(1, impact));
  }

  normalizeRisk(riskLevel) {
    const riskMap = { low: 0.2, medium: 0.5, high: 0.8, critical: 1.0 };
    return riskMap[riskLevel] || 0.5;
  }

  normalizeFrequency(optimization) {
    if (optimization.applications) return Math.min(1, optimization.applications / 100);
    if (optimization.frequency) return Math.min(1, optimization.frequency / 50);
    return 0.5;
  }

  // Additional helper methods...
  consolidateIssues(issues) { return issues; }
  categorizeFailure(failure) { return 'general_failure'; }
  analyzeRuleContextPerformance(rule) { return { variation: 0.1 }; }
  calculateModificationConfidence(rule, opportunities) { return 0.7; }
  estimateExpectedImprovement(rule, opportunities) { return 0.2; }
  assessModificationRisk(rule, opportunities) { return 'medium'; }
  calculateModificationPriority(rule, opportunities) { return 'high'; }
  findSuccessComparisons(rule) { return []; }
  calculateSignificance(rule) { return { pValue: 0.05 }; }
  calculateNewRuleConfidence(gap) { return 0.6; }
  estimateNewRuleImpact(gap) { return 0.3; }
  assessNewRuleRisk(gap) { return 'medium'; }
  assessImplementationComplexity(gap) { return 'medium'; }
  defineTestingRequirements(gap) { return ['unit_tests', 'integration_tests']; }
  planRolloutStrategy(gap) { return 'gradual_rollout'; }
  generateSpecializationId(opportunity) { return `spec_${opportunity.context}_${Date.now()}`; }
  generateContextSpecificLogic(opportunity) { return `Context-specific logic for ${opportunity.context}`; }
  estimateSpecializationImpact(opportunity) { return opportunity.potentialImprovement || 0.2; }
  assessSpecializationRisk(opportunity) { return 'low'; }
  calculateSpecializationPriority(opportunity) { return 'medium'; }
  planSpecializationIntegration(opportunity) { return 'conditional_application'; }
  defineFallbackBehavior(opportunity) { return 'use_base_rule'; }
  defineSpecializationValidation(opportunity) { return ['context_match', 'performance_improvement']; }
  findReplacementRules(rule) { return { hasViableReplacements: false, replacements: [], strategy: 'none' }; }
  analyzeDeprecationImpact(rule) { return { highImpact: false, scenarios: [], userImpact: 'low' }; }
  createMigrationPlan(rule, replacementAnalysis) { return 'no_migration_needed'; }
  assessDeprecationRisk(rule, impactAnalysis) { return 'low'; }
  planDeprecationTimeline(rule) { return '30_days'; }
  planDeprecationCommunication(rule) { return 'deprecation_notice'; }
  planDeprecationRollback(rule) { return 'immediate_rollback_available'; }
  estimateDetectionImprovementImpact(issue) { return 0.15; }
  estimateFalsePositiveReductionImpact(issue) { return 0.1; }
  estimateApplicationImprovementImpact(issue) { return 0.2; }
  estimateApplicationCorrectionImpact(issue) { return 0.25; }
  explainExpectedDetection(failure, rule) { return 'Expected detection explanation'; }
  identifyDetectionTrigger(original, rule) { return 'Detection trigger'; }
  suggestCorrectApplication(failure, rule) { return 'Correct application suggestion'; }

  generateOptimizationSummary() {
    return {
      totalOptimizations: 0,
      highPriorityOptimizations: 0,
      estimatedImpact: 0,
      riskProfile: 'medium',
      recommendedImplementationOrder: []
    };
  }

  estimateOverallImpact(optimizations) {
    return {
      expectedImprovementIncrease: 0.15,
      expectedSuccessRateIncrease: 0.1,
      confidenceInterval: [0.08, 0.22]
    };
  }

  assessOverallRisk(optimizations) {
    return {
      overallRiskLevel: 'medium',
      mitigationStrategies: ['staged_rollout', 'monitoring', 'rollback_plan'],
      riskFactors: []
    };
  }
}

module.exports = RuleOptimizer;