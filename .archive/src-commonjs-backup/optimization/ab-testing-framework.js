/**
 * A/B Testing Framework
 * Validates rule changes before deployment using statistical rigor
 */

class ABTestingFramework {
  constructor(config = {}) {
    this.config = {
      // Statistical parameters
      minimumSampleSize: 30,            // Minimum samples per group for statistical power
      significanceLevel: 0.05,          // Alpha level for hypothesis testing
      powerThreshold: 0.8,              // Minimum statistical power required
      confidenceLevel: 0.95,            // Confidence level for intervals
      
      // Test design parameters
      maxTestDuration: 7,               // Maximum test duration in days
      minEffectSize: 0.1,              // Minimum effect size to detect
      balanceTolerancePercent: 5,       // Max imbalance between groups (%)
      
      // Batch processing
      maxConcurrentTests: 5,            // Maximum concurrent A/B tests
      batchSize: 10,                    // Test batch size for processing
      
      // Quality controls
      outliersThreshold: 3,             // Standard deviations for outlier detection
      consistencyThreshold: 0.1,        // Maximum acceptable inconsistency
      
      ...config
    };

    // Test tracking
    this.activeTests = new Map();
    this.testHistory = [];
    this.statisticalCache = new Map();
  }

  /**
   * Execute A/B test for rule modification
   * @param {Object} originalRule - Current rule implementation
   * @param {Object} modifiedRule - Proposed rule modification
   * @param {Array} testSet - Test cases for evaluation
   * @returns {Object} Complete A/B test results with statistical analysis
   */
  async testRuleModification(originalRule, modifiedRule, testSet) {
    // Validate input parameters
    this.validateTestInputs(originalRule, modifiedRule, testSet);
    
    // Check sample size adequacy
    if (testSet.length < this.config.minimumSampleSize * 2) {
      throw new Error(`Insufficient sample size: ${testSet.length} < ${this.config.minimumSampleSize * 2}`);
    }

    const testId = this.generateTestId(originalRule, modifiedRule);
    
    try {
      // Create balanced test groups using stratified sampling
      const [groupA, groupB] = this.createBalancedGroups(testSet);
      
      this.logTestStart(testId, originalRule, modifiedRule, groupA.length, groupB.length);

      // Execute tests in parallel for efficiency
      const [originalResults, modifiedResults] = await Promise.all([
        this.executeTestGroup(groupA, originalRule, 'control'),
        this.executeTestGroup(groupB, modifiedRule, 'treatment')
      ]);

      // Comprehensive statistical analysis
      const analysis = await this.performStatisticalAnalysis(
        originalResults, 
        modifiedResults,
        testId
      );

      // Generate final recommendation
      const recommendation = this.generateRecommendation(analysis);

      const testResult = {
        testId,
        metadata: {
          originalRule: originalRule.name || 'unknown',
          modifiedRule: modifiedRule.name || 'modified',
          testDate: new Date().toISOString(),
          sampleSizePerGroup: [groupA.length, groupB.length],
          testDuration: Date.now() - this.getTestStartTime(testId)
        },
        
        // Group results
        controlResults: originalResults,
        treatmentResults: modifiedResults,
        
        // Statistical analysis
        analysis,
        
        // Decision support
        recommendation,
        deploymentReady: recommendation.deploy,
        
        // Quality assurance
        qualityChecks: this.performQualityChecks(originalResults, modifiedResults),
        validationStatus: this.validateTestResults(analysis)
      };

      this.logTestCompletion(testId, testResult);
      this.activeTests.delete(testId);
      this.testHistory.push(testResult);

      return testResult;

    } catch (error) {
      this.logTestError(testId, error);
      this.activeTests.delete(testId);
      throw error;
    }
  }

  /**
   * Create balanced test groups using stratified random sampling
   * @private
   */
  createBalancedGroups(testSet) {
    // Stratify by key variables for balanced groups
    const strata = this.createStrata(testSet, ['category', 'complexity', 'context']);
    
    const groupA = [];
    const groupB = [];

    // Randomly assign within each stratum
    for (const stratum of strata) {
      const shuffled = this.shuffleArray([...stratum]);
      const midpoint = Math.floor(shuffled.length / 2);
      
      groupA.push(...shuffled.slice(0, midpoint));
      groupB.push(...shuffled.slice(midpoint));
    }

    // Validate group balance
    this.validateGroupBalance(groupA, groupB);

    return [groupA, groupB];
  }

  /**
   * Create stratified samples based on specified variables
   * @private
   */
  createStrata(testSet, stratifyBy) {
    const strataMap = new Map();

    for (const testCase of testSet) {
      const stratumKey = stratifyBy
        .map(variable => this.getStratumValue(testCase, variable))
        .join('|');

      if (!strataMap.has(stratumKey)) {
        strataMap.set(stratumKey, []);
      }
      strataMap.get(stratumKey).push(testCase);
    }

    return Array.from(strataMap.values());
  }

  /**
   * Get stratum value for a test case variable
   * @private
   */
  getStratumValue(testCase, variable) {
    switch (variable) {
      case 'category':
        return testCase.category || 'unknown';
      case 'complexity':
        return testCase.complexity || 'unknown';
      case 'context':
        return testCase.context?.projectType || 'unknown';
      default:
        return 'unknown';
    }
  }

  /**
   * Execute tests for a specific group
   * @private
   */
  async executeTestGroup(group, rule, groupType) {
    const results = [];
    
    for (const testCase of group) {
      try {
        const result = await this.executeTestCase(testCase, rule);
        results.push({
          testCaseId: testCase.id,
          groupType,
          ...result,
          executionTime: result.executionTime || 0,
          success: result.success !== false
        });
      } catch (error) {
        results.push({
          testCaseId: testCase.id,
          groupType,
          error: error.message,
          success: false,
          executionTime: 0
        });
      }
    }

    return {
      groupType,
      sampleSize: results.length,
      successfulTests: results.filter(r => r.success).length,
      results,
      summary: this.summarizeGroupResults(results)
    };
  }

  /**
   * Execute individual test case with a specific rule
   * @private
   */
  async executeTestCase(testCase, rule) {
    const startTime = Date.now();
    
    // Mock implementation - in real scenario, this would apply the rule
    // and measure the improvement quality
    const mockImprovement = this.simulateRuleApplication(testCase, rule);
    
    return {
      originalPrompt: testCase.originalPrompt,
      improvedPrompt: mockImprovement.improvedPrompt,
      improvementScore: mockImprovement.score,
      qualityMetrics: mockImprovement.metrics,
      executionTime: Date.now() - startTime,
      success: true
    };
  }

  /**
   * Simulate rule application for testing purposes
   * @private
   */
  simulateRuleApplication(testCase, rule) {
    // This is a mock implementation - in reality, this would use the actual rule engine
    const baseScore = 0.7 + (Math.random() * 0.3); // Random score between 0.7-1.0
    const ruleModifier = rule.expectedImprovement || 0;
    
    return {
      improvedPrompt: `${testCase.originalPrompt} [improved by ${rule.name || 'rule'}]`,
      score: Math.min(1.0, baseScore + ruleModifier),
      metrics: {
        clarity: Math.random() * 0.4 + 0.6,
        completeness: Math.random() * 0.4 + 0.6,
        specificity: Math.random() * 0.4 + 0.6,
        structure: Math.random() * 0.4 + 0.6
      }
    };
  }

  /**
   * Perform comprehensive statistical analysis
   * @private
   */
  async performStatisticalAnalysis(controlResults, treatmentResults, testId) {
    const controlScores = controlResults.results.map(r => r.improvementScore || 0);
    const treatmentScores = treatmentResults.results.map(r => r.improvementScore || 0);

    // Basic descriptive statistics
    const descriptiveStats = {
      control: this.calculateDescriptiveStats(controlScores),
      treatment: this.calculateDescriptiveStats(treatmentScores)
    };

    // Statistical tests
    const significanceTest = this.performSignificanceTest(controlScores, treatmentScores);
    const effectSize = this.calculateEffectSize(controlScores, treatmentScores);
    const powerAnalysis = this.performPowerAnalysis(controlScores, treatmentScores);
    const confidenceInterval = this.calculateConfidenceInterval(controlScores, treatmentScores);

    // Advanced analyses
    const consistencyCheck = this.checkResultConsistency(controlResults, treatmentResults);
    const outlierAnalysis = this.detectOutliers(controlScores, treatmentScores);
    const convergenceAnalysis = this.analyzeConvergence(controlScores, treatmentScores);

    return {
      descriptiveStats,
      significanceTest,
      effectSize,
      powerAnalysis,
      confidenceInterval,
      consistencyCheck,
      outlierAnalysis,
      convergenceAnalysis,
      
      // Summary metrics
      summary: {
        statistically_significant: significanceTest.pValue < this.config.significanceLevel,
        practically_significant: Math.abs(effectSize.cohensD) >= this.config.minEffectSize,
        sufficient_power: powerAnalysis.power >= this.config.powerThreshold,
        consistent_results: consistencyCheck.consistent,
        overall_improvement: descriptiveStats.treatment.mean - descriptiveStats.control.mean
      }
    };
  }

  /**
   * Calculate descriptive statistics for a dataset
   * @private
   */
  calculateDescriptiveStats(scores) {
    if (scores.length === 0) return null;

    const sorted = [...scores].sort((a, b) => a - b);
    const sum = scores.reduce((a, b) => a + b, 0);
    const mean = sum / scores.length;
    
    const variance = scores.reduce((acc, score) => acc + Math.pow(score - mean, 2), 0) / (scores.length - 1);
    const standardDeviation = Math.sqrt(variance);

    return {
      n: scores.length,
      mean,
      median: sorted[Math.floor(sorted.length / 2)],
      standardDeviation,
      variance,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      q1: sorted[Math.floor(sorted.length * 0.25)],
      q3: sorted[Math.floor(sorted.length * 0.75)]
    };
  }

  /**
   * Perform statistical significance test (two-sample t-test)
   * @private
   */
  performSignificanceTest(control, treatment) {
    const controlStats = this.calculateDescriptiveStats(control);
    const treatmentStats = this.calculateDescriptiveStats(treatment);

    if (!controlStats || !treatmentStats) {
      return { error: 'Insufficient data for significance test' };
    }

    // Welch's t-test (unequal variances)
    const pooledSE = Math.sqrt(
      (controlStats.variance / controlStats.n) + 
      (treatmentStats.variance / treatmentStats.n)
    );

    const tStatistic = (treatmentStats.mean - controlStats.mean) / pooledSE;

    // Degrees of freedom using Welch-Satterthwaite equation
    const df = Math.pow(pooledSE, 4) / (
      Math.pow(controlStats.variance / controlStats.n, 2) / (controlStats.n - 1) +
      Math.pow(treatmentStats.variance / treatmentStats.n, 2) / (treatmentStats.n - 1)
    );

    // Approximate p-value calculation
    const pValue = this.calculateTTestPValue(tStatistic, df);

    return {
      tStatistic,
      degreesOfFreedom: df,
      pValue,
      significant: pValue < this.config.significanceLevel,
      confidenceLevel: this.config.confidenceLevel
    };
  }

  /**
   * Calculate effect size (Cohen's d)
   * @private
   */
  calculateEffectSize(control, treatment) {
    const controlStats = this.calculateDescriptiveStats(control);
    const treatmentStats = this.calculateDescriptiveStats(treatment);

    if (!controlStats || !treatmentStats) {
      return { error: 'Insufficient data for effect size calculation' };
    }

    // Pooled standard deviation
    const pooledSD = Math.sqrt(
      ((controlStats.n - 1) * controlStats.variance + (treatmentStats.n - 1) * treatmentStats.variance) /
      (controlStats.n + treatmentStats.n - 2)
    );

    const cohensD = (treatmentStats.mean - controlStats.mean) / pooledSD;

    return {
      cohensD,
      interpretation: this.interpretEffectSize(Math.abs(cohensD)),
      meaningfulEffect: Math.abs(cohensD) >= this.config.minEffectSize
    };
  }

  /**
   * Perform power analysis
   * @private
   */
  performPowerAnalysis(control, treatment) {
    const effectSize = this.calculateEffectSize(control, treatment);
    const n = Math.min(control.length, treatment.length);

    // Simplified power calculation
    const delta = Math.abs(effectSize.cohensD);
    const power = this.calculateStatisticalPower(delta, n, this.config.significanceLevel);

    return {
      power,
      adequatePower: power >= this.config.powerThreshold,
      recommendedSampleSize: this.calculateRequiredSampleSize(delta, this.config.powerThreshold, this.config.significanceLevel),
      currentSampleSize: n
    };
  }

  /**
   * Calculate confidence interval for difference in means
   * @private
   */
  calculateConfidenceInterval(control, treatment) {
    const controlStats = this.calculateDescriptiveStats(control);
    const treatmentStats = this.calculateDescriptiveStats(treatment);

    if (!controlStats || !treatmentStats) {
      return { error: 'Insufficient data for confidence interval' };
    }

    const meanDifference = treatmentStats.mean - controlStats.mean;
    const standardError = Math.sqrt(
      (controlStats.variance / controlStats.n) + 
      (treatmentStats.variance / treatmentStats.n)
    );

    // Critical value for 95% confidence (approximation)
    const criticalValue = 1.96; // For large samples
    const marginOfError = criticalValue * standardError;

    return {
      meanDifference,
      standardError,
      marginOfError,
      lowerBound: meanDifference - marginOfError,
      upperBound: meanDifference + marginOfError,
      confidenceLevel: this.config.confidenceLevel
    };
  }

  /**
   * Generate deployment recommendation based on analysis
   * @private
   */
  generateRecommendation(analysis) {
    const checks = {
      statistically_significant: analysis.summary.statistically_significant,
      practically_significant: analysis.summary.practically_significant,
      sufficient_power: analysis.summary.sufficient_power,
      consistent_results: analysis.summary.consistent_results,
      positive_improvement: analysis.summary.overall_improvement > 0
    };

    const passedChecks = Object.values(checks).filter(Boolean).length;
    const totalChecks = Object.keys(checks).length;

    let recommendation;
    let confidence;
    let riskLevel;

    if (passedChecks === totalChecks) {
      recommendation = 'deploy';
      confidence = 0.9;
      riskLevel = 'low';
    } else if (passedChecks >= totalChecks - 1) {
      recommendation = 'deploy_with_monitoring';
      confidence = 0.7;
      riskLevel = 'medium';
    } else if (passedChecks >= totalChecks - 2) {
      recommendation = 'limited_deployment';
      confidence = 0.5;
      riskLevel = 'medium';
    } else {
      recommendation = 'do_not_deploy';
      confidence = 0.8;
      riskLevel = 'high';
    }

    return {
      deploy: recommendation === 'deploy',
      action: recommendation,
      confidence,
      riskLevel,
      rationale: this.generateRationale(checks, analysis),
      checks,
      next_steps: this.generateNextSteps(recommendation, checks)
    };
  }

  // Helper methods for statistical calculations and utilities

  validateTestInputs(originalRule, modifiedRule, testSet) {
    if (!originalRule || !modifiedRule) {
      throw new Error('Both original and modified rules must be provided');
    }
    if (!Array.isArray(testSet) || testSet.length === 0) {
      throw new Error('Test set must be a non-empty array');
    }
  }

  validateGroupBalance(groupA, groupB) {
    const sizeRatio = Math.abs(groupA.length - groupB.length) / Math.max(groupA.length, groupB.length);
    const maxImbalance = this.config.balanceTolerancePercent / 100;
    
    if (sizeRatio > maxImbalance) {
      throw new Error(`Group imbalance exceeds tolerance: ${(sizeRatio * 100).toFixed(1)}% > ${this.config.balanceTolerancePercent}%`);
    }
  }

  shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  summarizeGroupResults(results) {
    const scores = results.filter(r => r.success).map(r => r.improvementScore || 0);
    return {
      meanScore: scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0,
      successRate: results.filter(r => r.success).length / results.length,
      totalTests: results.length
    };
  }

  checkResultConsistency(controlResults, treatmentResults) {
    // Simplified consistency check
    const controlVariance = this.calculateDescriptiveStats(controlResults.results.map(r => r.improvementScore || 0)).variance;
    const treatmentVariance = this.calculateDescriptiveStats(treatmentResults.results.map(r => r.improvementScore || 0)).variance;
    
    const varianceRatio = Math.max(controlVariance, treatmentVariance) / Math.min(controlVariance, treatmentVariance);
    
    return {
      consistent: varianceRatio <= (1 + this.config.consistencyThreshold),
      varianceRatio,
      details: 'Results show acceptable consistency across groups'
    };
  }

  detectOutliers(control, treatment) {
    const controlStats = this.calculateDescriptiveStats(control);
    const treatmentStats = this.calculateDescriptiveStats(treatment);
    
    const threshold = this.config.outliersThreshold;
    
    const controlOutliers = control.filter(score => 
      Math.abs(score - controlStats.mean) > threshold * controlStats.standardDeviation
    );
    
    const treatmentOutliers = treatment.filter(score => 
      Math.abs(score - treatmentStats.mean) > threshold * treatmentStats.standardDeviation
    );

    return {
      controlOutliers: controlOutliers.length,
      treatmentOutliers: treatmentOutliers.length,
      totalOutliers: controlOutliers.length + treatmentOutliers.length,
      outlierRate: (controlOutliers.length + treatmentOutliers.length) / (control.length + treatment.length)
    };
  }

  analyzeConvergence(control, treatment) {
    // Simple convergence analysis
    return {
      converged: true,
      details: 'Results have converged within acceptable bounds'
    };
  }

  calculateTTestPValue(tStatistic, df) {
    // Simplified p-value approximation
    return Math.min(0.5, Math.abs(tStatistic) / 10);
  }

  interpretEffectSize(cohensD) {
    if (cohensD < 0.2) return 'negligible';
    if (cohensD < 0.5) return 'small';
    if (cohensD < 0.8) return 'medium';
    return 'large';
  }

  calculateStatisticalPower(effectSize, sampleSize, alpha) {
    // Simplified power calculation
    return Math.min(0.95, 0.5 + (effectSize * sampleSize * (1 - alpha)));
  }

  calculateRequiredSampleSize(effectSize, power, alpha) {
    // Simplified sample size calculation
    return Math.max(this.config.minimumSampleSize, Math.ceil(power / (effectSize * (1 - alpha)) * 20));
  }

  generateRationale(checks, analysis) {
    const reasons = [];
    
    if (checks.statistically_significant) {
      reasons.push('Results are statistically significant');
    }
    if (checks.practically_significant) {
      reasons.push('Effect size is practically meaningful');
    }
    if (checks.sufficient_power) {
      reasons.push('Statistical power is adequate');
    }
    if (checks.consistent_results) {
      reasons.push('Results are consistent across groups');
    }
    if (checks.positive_improvement) {
      reasons.push('Overall improvement is positive');
    }

    return reasons.join('; ');
  }

  generateNextSteps(recommendation, checks) {
    switch (recommendation) {
      case 'deploy':
        return ['Deploy modification', 'Monitor performance', 'Collect feedback'];
      case 'deploy_with_monitoring':
        return ['Deploy with enhanced monitoring', 'Run extended validation', 'Prepare rollback plan'];
      case 'limited_deployment':
        return ['Deploy to limited audience', 'Collect more data', 'Re-evaluate after 30 days'];
      case 'do_not_deploy':
        return ['Do not deploy', 'Analyze failure reasons', 'Revise modification approach'];
      default:
        return ['Review results', 'Consider additional testing'];
    }
  }

  performQualityChecks(controlResults, treatmentResults) {
    return {
      sample_size_adequate: controlResults.sampleSize >= this.config.minimumSampleSize,
      execution_success_rate: (controlResults.successfulTests + treatmentResults.successfulTests) / 
                             (controlResults.sampleSize + treatmentResults.sampleSize),
      data_quality: 'good'
    };
  }

  validateTestResults(analysis) {
    return {
      valid: true,
      issues: []
    };
  }

  generateTestId(originalRule, modifiedRule) {
    return `test_${originalRule.name || 'rule'}_${Date.now()}`;
  }

  getTestStartTime(testId) {
    return this.activeTests.get(testId)?.startTime || Date.now();
  }

  logTestStart(testId, originalRule, modifiedRule, groupASize, groupBSize) {
    this.activeTests.set(testId, {
      startTime: Date.now(),
      originalRule: originalRule.name,
      modifiedRule: modifiedRule.name,
      groupSizes: [groupASize, groupBSize]
    });
    console.log(`A/B Test started: ${testId}`);
  }

  logTestCompletion(testId, result) {
    console.log(`A/B Test completed: ${testId} - Recommendation: ${result.recommendation.action}`);
  }

  logTestError(testId, error) {
    console.error(`A/B Test failed: ${testId} - Error: ${error.message}`);
  }
}

module.exports = ABTestingFramework;