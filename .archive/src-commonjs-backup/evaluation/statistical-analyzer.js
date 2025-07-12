/**
 * Statistical Analysis Framework
 * Significance testing and validation for evaluation results
 */

const Logger = require('../utils/logger');

class StatisticalAnalyzer {
  constructor(config = {}) {
    this.config = {
      // Significance thresholds
      significanceLevel: 0.05, // Alpha level for hypothesis testing
      confidenceLevel: 0.95,   // Confidence level for intervals
      
      // Sample size requirements
      minimumSampleSize: 5,
      recommendedSampleSize: 30,
      
      // Effect size thresholds (Cohen's d)
      effectSizes: {
        small: 0.2,
        medium: 0.5,
        large: 0.8
      },

      // Statistical tests configuration
      tests: {
        normalityTest: 'shapiro-wilk', // For small samples
        tTest: 'welch',                // Default t-test variant
        nonParametric: 'mann-whitney', // For non-normal distributions
        correlationTest: 'pearson'     // Default correlation test
      },

      // Validation thresholds
      validation: {
        reliabilityThreshold: 0.7,     // Minimum reliability
        consistencyThreshold: 0.8,     // Minimum consistency
        validityThreshold: 0.6         // Minimum validity
      },

      ...config
    };

    this.logger = new Logger('StatisticalAnalyzer');
  }

  /**
   * Perform comprehensive statistical analysis on evaluation results
   * @param {Array} results - Array of evaluation results
   * @param {Object} options - Analysis options
   * @returns {Object} Statistical analysis results
   */
  async performStatisticalAnalysis(results, options = {}) {
    const startTime = Date.now();
    this.logger.info('Starting statistical analysis', {
      sampleSize: results.length,
      analysisType: options.analysisType || 'comprehensive'
    });

    try {
      // Validate inputs
      this.validateInputs(results, options);

      // Prepare data for analysis
      const analysisData = this.prepareAnalysisData(results, options);

      // Perform statistical tests
      const analysis = {
        metadata: {
          sampleSize: results.length,
          analysisType: options.analysisType || 'comprehensive',
          significanceLevel: this.config.significanceLevel,
          confidenceLevel: this.config.confidenceLevel,
          analyzedAt: new Date().toISOString()
        },

        // Descriptive statistics
        descriptiveStats: this.calculateDescriptiveStatistics(analysisData),

        // Distribution analysis
        distributionAnalysis: this.analyzeDistributions(analysisData),

        // Hypothesis testing
        hypothesisTests: await this.performHypothesisTests(analysisData, options),

        // Effect size analysis
        effectSizeAnalysis: this.analyzeEffectSizes(analysisData, options),

        // Reliability analysis
        reliabilityAnalysis: this.analyzeReliability(analysisData, options),

        // Validity analysis
        validityAnalysis: this.analyzeValidity(analysisData, options),

        // Correlation analysis
        correlationAnalysis: this.analyzeCorrelations(analysisData),

        // Confidence intervals
        confidenceIntervals: this.calculateConfidenceIntervals(analysisData),

        // Statistical summary
        summary: {},
        
        // Recommendations
        recommendations: []
      };

      // Generate summary and recommendations
      analysis.summary = this.generateStatisticalSummary(analysis);
      analysis.recommendations = this.generateStatisticalRecommendations(analysis);

      const analysisTime = Date.now() - startTime;
      this.logger.info('Statistical analysis completed', {
        analysisTime,
        significantFindings: analysis.summary.significantFindings || 0
      });

      return analysis;

    } catch (error) {
      this.logger.error('Statistical analysis failed', error);
      throw new Error(`Statistical analysis failed: ${error.message}`);
    }
  }

  /**
   * Prepare data for statistical analysis
   * @param {Array} results - Raw evaluation results
   * @param {Object} options - Analysis options
   * @returns {Object} Prepared analysis data
   */
  prepareAnalysisData(results, options) {
    const data = {
      raw: results,
      
      // Extract numeric metrics
      overallScores: results.map(r => r.overallScore || r.overallQuality || 0),
      clarityScores: results.map(r => this.extractScore(r, 'clarity')),
      completenessScores: results.map(r => this.extractScore(r, 'completeness')),
      actionabilityScores: results.map(r => this.extractScore(r, 'actionability')),
      effectivenessScores: results.map(r => this.extractScore(r, 'effectiveness')),
      
      // Extract categorical data
      strategies: results.map(r => r.strategy || r.metadata?.strategy || 'unknown'),
      models: results.map(r => r.model || r.metadata?.model || 'unknown'),
      complexities: results.map(r => r.complexity || 'unknown'),
      
      // Extract metadata
      timestamps: results.map(r => r.timestamp || r.metadata?.timestamp || new Date().toISOString()),
      sampleSizes: results.map(r => r.sampleSize || 1),
      
      // Group data for comparative analysis
      groups: this.groupDataForAnalysis(results, options)
    };

    // Calculate derived metrics
    data.scoreDifferences = this.calculateScoreDifferences(data);
    data.consistencyMetrics = this.calculateConsistencyMetrics(data);
    data.qualityTrends = this.calculateQualityTrends(data);

    return data;
  }

  /**
   * Extract score from result object
   * @param {Object} result - Result object
   * @param {string} metric - Metric name
   * @returns {number} Extracted score
   */
  extractScore(result, metric) {
    // Try different possible locations for the score
    if (result[metric]?.score !== undefined) return result[metric].score;
    if (result[metric]?.mean !== undefined) return result[metric].mean;
    if (result.scores?.[metric] !== undefined) return result.scores[metric];
    if (result.metrics?.[metric]?.score !== undefined) return result.metrics[metric].score;
    
    // Fallback: try to extract from nested structures
    if (typeof result[metric] === 'number') return result[metric];
    
    return 0; // Default if not found
  }

  /**
   * Group data for comparative analysis
   * @param {Array} results - Evaluation results
   * @param {Object} options - Analysis options
   * @returns {Object} Grouped data
   */
  groupDataForAnalysis(results, options) {
    const groups = {};

    // Group by strategy
    groups.byStrategy = this.groupBy(results, r => r.strategy || 'default');
    
    // Group by model (if applicable)
    groups.byModel = this.groupBy(results, r => r.model || 'default');
    
    // Group by complexity
    groups.byComplexity = this.groupBy(results, r => r.complexity || 'unknown');
    
    // Group by quality grade
    groups.byQualityGrade = this.groupBy(results, r => this.getQualityGrade(r.overallScore || r.overallQuality || 0));
    
    // Custom grouping if specified
    if (options.groupBy) {
      groups.custom = this.groupBy(results, options.groupBy);
    }

    return groups;
  }

  /**
   * Group array by key function
   * @param {Array} array - Array to group
   * @param {Function} keyFn - Function to extract grouping key
   * @returns {Object} Grouped object
   */
  groupBy(array, keyFn) {
    return array.reduce((groups, item) => {
      const key = keyFn(item);
      if (!groups[key]) groups[key] = [];
      groups[key].push(item);
      return groups;
    }, {});
  }

  /**
   * Calculate descriptive statistics for all metrics
   * @param {Object} data - Analysis data
   * @returns {Object} Descriptive statistics
   */
  calculateDescriptiveStatistics(data) {
    const metrics = ['overallScores', 'clarityScores', 'completenessScores', 'actionabilityScores', 'effectivenessScores'];
    const stats = {};

    metrics.forEach(metric => {
      const values = data[metric].filter(v => v !== null && v !== undefined && !isNaN(v));
      
      if (values.length > 0) {
        stats[metric] = {
          count: values.length,
          mean: this.calculateMean(values),
          median: this.calculateMedian(values),
          mode: this.calculateMode(values),
          standardDeviation: this.calculateStandardDeviation(values),
          variance: this.calculateVariance(values),
          min: Math.min(...values),
          max: Math.max(...values),
          range: Math.max(...values) - Math.min(...values),
          quartiles: this.calculateQuartiles(values),
          skewness: this.calculateSkewness(values),
          kurtosis: this.calculateKurtosis(values),
          coefficientOfVariation: this.calculateCoefficientOfVariation(values)
        };
      } else {
        stats[metric] = this.getEmptyStats();
      }
    });

    return stats;
  }

  /**
   * Analyze distributions of metrics
   * @param {Object} data - Analysis data
   * @returns {Object} Distribution analysis
   */
  analyzeDistributions(data) {
    const metrics = ['overallScores', 'clarityScores', 'completenessScores', 'actionabilityScores', 'effectivenessScores'];
    const distributions = {};

    metrics.forEach(metric => {
      const values = data[metric].filter(v => v !== null && v !== undefined && !isNaN(v));
      
      if (values.length >= this.config.minimumSampleSize) {
        distributions[metric] = {
          normalityTest: this.testNormality(values),
          histogram: this.createHistogram(values),
          outliers: this.detectOutliers(values),
          distributionType: this.classifyDistribution(values)
        };
      }
    });

    return distributions;
  }

  /**
   * Perform hypothesis tests
   * @param {Object} data - Analysis data
   * @param {Object} options - Analysis options
   * @returns {Promise<Object>} Hypothesis test results
   */
  async performHypothesisTests(data, options) {
    const tests = {};

    // Test for significant differences between groups
    if (options.compareGroups) {
      tests.groupComparisons = await this.performGroupComparisons(data, options);
    }

    // Test for improvement over baseline
    if (options.baseline) {
      tests.baselineComparison = await this.compareToBaseline(data, options.baseline);
    }

    // Test for correlation between metrics
    tests.correlationTests = this.performCorrelationTests(data);

    // Test for consistency across evaluations
    tests.consistencyTests = this.performConsistencyTests(data);

    return tests;
  }

  /**
   * Perform group comparisons
   * @param {Object} data - Analysis data
   * @param {Object} options - Analysis options
   * @returns {Promise<Object>} Group comparison results
   */
  async performGroupComparisons(data, options) {
    const comparisons = {};
    const groupField = options.compareGroups || 'byStrategy';
    const groups = data.groups[groupField];

    if (!groups || Object.keys(groups).length < 2) {
      return { error: 'Insufficient groups for comparison' };
    }

    const groupNames = Object.keys(groups);
    
    // Pairwise comparisons
    for (let i = 0; i < groupNames.length; i++) {
      for (let j = i + 1; j < groupNames.length; j++) {
        const group1 = groupNames[i];
        const group2 = groupNames[j];
        
        const group1Scores = groups[group1].map(r => r.overallScore || r.overallQuality || 0);
        const group2Scores = groups[group2].map(r => r.overallScore || r.overallQuality || 0);
        
        const comparisonKey = `${group1}_vs_${group2}`;
        comparisons[comparisonKey] = this.compareTwoGroups(group1Scores, group2Scores);
      }
    }

    return comparisons;
  }

  /**
   * Compare two groups statistically
   * @param {Array} group1 - First group scores
   * @param {Array} group2 - Second group scores
   * @returns {Object} Comparison results
   */
  compareTwoGroups(group1, group2) {
    const comparison = {
      group1: {
        size: group1.length,
        mean: this.calculateMean(group1),
        std: this.calculateStandardDeviation(group1)
      },
      group2: {
        size: group2.length,
        mean: this.calculateMean(group2),
        std: this.calculateStandardDeviation(group2)
      }
    };

    // Calculate effect size (Cohen's d)
    comparison.effectSize = this.calculateCohensD(group1, group2);
    comparison.effectSizeCategory = this.categorizeEffectSize(comparison.effectSize);

    // Perform t-test
    comparison.tTest = this.performTTest(group1, group2);

    // Perform non-parametric test (Mann-Whitney U)
    comparison.mannWhitneyU = this.performMannWhitneyU(group1, group2);

    // Determine practical significance
    comparison.practicalSignificance = this.assessPracticalSignificance(comparison);

    return comparison;
  }

  /**
   * Analyze effect sizes
   * @param {Object} data - Analysis data
   * @param {Object} options - Analysis options
   * @returns {Object} Effect size analysis
   */
  analyzeEffectSizes(data, options) {
    const analysis = {
      cohensD: {},
      practicalSignificance: {},
      recommendations: []
    };

    // Calculate effect sizes for different comparisons
    if (data.groups.byStrategy && Object.keys(data.groups.byStrategy).length >= 2) {
      const strategies = Object.keys(data.groups.byStrategy);
      
      for (let i = 0; i < strategies.length; i++) {
        for (let j = i + 1; j < strategies.length; j++) {
          const strategy1 = strategies[i];
          const strategy2 = strategies[j];
          
          const group1Scores = data.groups.byStrategy[strategy1].map(r => r.overallScore || r.overallQuality || 0);
          const group2Scores = data.groups.byStrategy[strategy2].map(r => r.overallScore || r.overallQuality || 0);
          
          const effectSize = this.calculateCohensD(group1Scores, group2Scores);
          const comparisonKey = `${strategy1}_vs_${strategy2}`;
          
          analysis.cohensD[comparisonKey] = {
            effectSize: effectSize,
            category: this.categorizeEffectSize(effectSize),
            interpretation: this.interpretEffectSize(effectSize)
          };
        }
      }
    }

    return analysis;
  }

  /**
   * Analyze reliability of evaluations
   * @param {Object} data - Analysis data
   * @param {Object} options - Analysis options
   * @returns {Object} Reliability analysis
   */
  analyzeReliability(data, options) {
    const reliability = {
      consistency: {},
      stability: {},
      interRaterReliability: {},
      cronbachAlpha: null,
      recommendations: []
    };

    // Internal consistency (Cronbach's alpha) for multi-dimensional scores
    const scoreDimensions = [
      data.clarityScores,
      data.completenessScores,
      data.actionabilityScores,
      data.effectivenessScores
    ].filter(scores => scores.length > 0);

    if (scoreDimensions.length >= 2) {
      reliability.cronbachAlpha = this.calculateCronbachAlpha(scoreDimensions);
    }

    // Test-retest reliability (if multiple evaluations of same prompts)
    reliability.testRetest = this.analyzeTestRetestReliability(data);

    // Inter-rater reliability (if multiple judges)
    reliability.interRater = this.analyzeInterRaterReliability(data);

    // Overall reliability assessment
    reliability.overall = this.assessOverallReliability(reliability);

    return reliability;
  }

  /**
   * Analyze validity of evaluations
   * @param {Object} data - Analysis data
   * @param {Object} options - Analysis options
   * @returns {Object} Validity analysis
   */
  analyzeValidity(data, options) {
    const validity = {
      construct: {},
      content: {},
      criterion: {},
      convergent: {},
      discriminant: {},
      recommendations: []
    };

    // Construct validity - do related measures correlate appropriately?
    validity.construct = this.analyzeConstructValidity(data);

    // Convergent validity - do similar measures correlate highly?
    validity.convergent = this.analyzeConvergentValidity(data);

    // Discriminant validity - do different constructs have low correlation?
    validity.discriminant = this.analyzeDiscriminantValidity(data);

    // Overall validity assessment
    validity.overall = this.assessOverallValidity(validity);

    return validity;
  }

  /**
   * Analyze correlations between metrics
   * @param {Object} data - Analysis data
   * @returns {Object} Correlation analysis
   */
  analyzeCorrelations(data) {
    const metrics = ['overallScores', 'clarityScores', 'completenessScores', 'actionabilityScores', 'effectivenessScores'];
    const correlations = {};

    // Calculate pairwise correlations
    for (let i = 0; i < metrics.length; i++) {
      for (let j = i + 1; j < metrics.length; j++) {
        const metric1 = metrics[i];
        const metric2 = metrics[j];
        
        const values1 = data[metric1].filter(v => !isNaN(v));
        const values2 = data[metric2].filter(v => !isNaN(v));
        
        if (values1.length === values2.length && values1.length >= 3) {
          const correlation = this.calculatePearsonCorrelation(values1, values2);
          const significance = this.testCorrelationSignificance(correlation, values1.length);
          
          correlations[`${metric1}_${metric2}`] = {
            correlation: correlation,
            significance: significance,
            strength: this.interpretCorrelationStrength(Math.abs(correlation)),
            direction: correlation > 0 ? 'positive' : 'negative'
          };
        }
      }
    }

    return correlations;
  }

  /**
   * Calculate confidence intervals
   * @param {Object} data - Analysis data
   * @returns {Object} Confidence intervals
   */
  calculateConfidenceIntervals(data) {
    const metrics = ['overallScores', 'clarityScores', 'completenessScores', 'actionabilityScores', 'effectivenessScores'];
    const intervals = {};

    metrics.forEach(metric => {
      const values = data[metric].filter(v => !isNaN(v));
      
      if (values.length >= 3) {
        intervals[metric] = {
          mean: this.calculateMean(values),
          confidenceInterval: this.calculateMeanConfidenceInterval(values, this.config.confidenceLevel),
          marginOfError: this.calculateMarginOfError(values, this.config.confidenceLevel)
        };
      }
    });

    return intervals;
  }

  // Statistical calculation methods

  /**
   * Calculate mean
   * @param {Array} values - Numeric values
   * @returns {number} Mean value
   */
  calculateMean(values) {
    return values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0;
  }

  /**
   * Calculate median
   * @param {Array} values - Numeric values
   * @returns {number} Median value
   */
  calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
      return (sorted[middle - 1] + sorted[middle]) / 2;
    }
    return sorted[middle];
  }

  /**
   * Calculate mode
   * @param {Array} values - Numeric values
   * @returns {number|Array} Mode value(s)
   */
  calculateMode(values) {
    const frequency = {};
    let maxFreq = 0;
    
    values.forEach(val => {
      frequency[val] = (frequency[val] || 0) + 1;
      maxFreq = Math.max(maxFreq, frequency[val]);
    });
    
    const modes = Object.keys(frequency)
      .filter(val => frequency[val] === maxFreq)
      .map(val => parseFloat(val));
    
    return modes.length === 1 ? modes[0] : modes;
  }

  /**
   * Calculate standard deviation
   * @param {Array} values - Numeric values
   * @returns {number} Standard deviation
   */
  calculateStandardDeviation(values) {
    const mean = this.calculateMean(values);
    const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
    const variance = this.calculateMean(squaredDifferences);
    return Math.sqrt(variance);
  }

  /**
   * Calculate variance
   * @param {Array} values - Numeric values
   * @returns {number} Variance
   */
  calculateVariance(values) {
    const mean = this.calculateMean(values);
    const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
    return this.calculateMean(squaredDifferences);
  }

  /**
   * Calculate quartiles
   * @param {Array} values - Numeric values
   * @returns {Object} Quartiles
   */
  calculateQuartiles(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const n = sorted.length;
    
    return {
      q1: this.calculatePercentile(sorted, 25),
      q2: this.calculateMedian(sorted),
      q3: this.calculatePercentile(sorted, 75),
      iqr: this.calculatePercentile(sorted, 75) - this.calculatePercentile(sorted, 25)
    };
  }

  /**
   * Calculate percentile
   * @param {Array} sortedValues - Sorted numeric values
   * @param {number} percentile - Percentile (0-100)
   * @returns {number} Percentile value
   */
  calculatePercentile(sortedValues, percentile) {
    const index = (percentile / 100) * (sortedValues.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index % 1;
    
    if (upper >= sortedValues.length) return sortedValues[sortedValues.length - 1];
    
    return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
  }

  /**
   * Calculate skewness
   * @param {Array} values - Numeric values
   * @returns {number} Skewness
   */
  calculateSkewness(values) {
    const n = values.length;
    const mean = this.calculateMean(values);
    const std = this.calculateStandardDeviation(values);
    
    if (std === 0) return 0;
    
    const skewness = values.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / std, 3);
    }, 0);
    
    return (n / ((n - 1) * (n - 2))) * skewness;
  }

  /**
   * Calculate kurtosis
   * @param {Array} values - Numeric values
   * @returns {number} Kurtosis (excess kurtosis)
   */
  calculateKurtosis(values) {
    const n = values.length;
    const mean = this.calculateMean(values);
    const std = this.calculateStandardDeviation(values);
    
    if (std === 0) return 0;
    
    const kurtosis = values.reduce((sum, val) => {
      return sum + Math.pow((val - mean) / std, 4);
    }, 0);
    
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * kurtosis - 3 * Math.pow(n - 1, 2) / ((n - 2) * (n - 3));
  }

  /**
   * Calculate coefficient of variation
   * @param {Array} values - Numeric values
   * @returns {number} Coefficient of variation
   */
  calculateCoefficientOfVariation(values) {
    const mean = this.calculateMean(values);
    const std = this.calculateStandardDeviation(values);
    return mean !== 0 ? std / mean : 0;
  }

  /**
   * Calculate Cohen's d effect size
   * @param {Array} group1 - First group values
   * @param {Array} group2 - Second group values
   * @returns {number} Cohen's d
   */
  calculateCohensD(group1, group2) {
    const mean1 = this.calculateMean(group1);
    const mean2 = this.calculateMean(group2);
    const std1 = this.calculateStandardDeviation(group1);
    const std2 = this.calculateStandardDeviation(group2);
    
    // Pooled standard deviation
    const pooledStd = Math.sqrt(((group1.length - 1) * std1 * std1 + (group2.length - 1) * std2 * std2) / (group1.length + group2.length - 2));
    
    return pooledStd !== 0 ? (mean1 - mean2) / pooledStd : 0;
  }

  /**
   * Categorize effect size
   * @param {number} effectSize - Effect size value
   * @returns {string} Effect size category
   */
  categorizeEffectSize(effectSize) {
    const abs = Math.abs(effectSize);
    
    if (abs >= this.config.effectSizes.large) return 'large';
    if (abs >= this.config.effectSizes.medium) return 'medium';
    if (abs >= this.config.effectSizes.small) return 'small';
    return 'negligible';
  }

  /**
   * Interpret effect size
   * @param {number} effectSize - Effect size value
   * @returns {string} Effect size interpretation
   */
  interpretEffectSize(effectSize) {
    const category = this.categorizeEffectSize(effectSize);
    const direction = effectSize > 0 ? 'favors first group' : 'favors second group';
    
    return `${category} effect that ${direction}`;
  }

  /**
   * Perform t-test
   * @param {Array} group1 - First group values
   * @param {Array} group2 - Second group values
   * @returns {Object} T-test results
   */
  performTTest(group1, group2) {
    const mean1 = this.calculateMean(group1);
    const mean2 = this.calculateMean(group2);
    const var1 = this.calculateVariance(group1);
    const var2 = this.calculateVariance(group2);
    const n1 = group1.length;
    const n2 = group2.length;
    
    // Welch's t-test (unequal variances)
    const standardError = Math.sqrt(var1 / n1 + var2 / n2);
    const tStatistic = standardError !== 0 ? (mean1 - mean2) / standardError : 0;
    
    // Degrees of freedom for Welch's test
    const df = Math.pow(var1 / n1 + var2 / n2, 2) / (Math.pow(var1 / n1, 2) / (n1 - 1) + Math.pow(var2 / n2, 2) / (n2 - 1));
    
    return {
      tStatistic: tStatistic,
      degreesOfFreedom: df,
      pValue: this.approximatePValue(Math.abs(tStatistic), df),
      significant: this.approximatePValue(Math.abs(tStatistic), df) < this.config.significanceLevel,
      meanDifference: mean1 - mean2,
      standardError: standardError
    };
  }

  /**
   * Approximate p-value for t-test (simplified)
   * @param {number} tStat - T-statistic (absolute value)
   * @param {number} df - Degrees of freedom
   * @returns {number} Approximate p-value
   */
  approximatePValue(tStat, df) {
    // Simplified approximation for common cases
    if (df < 1) return 1;
    if (tStat < 1) return 0.5;
    if (tStat > 4) return 0.001;
    
    // Linear approximation for demonstration
    // In real implementation, use proper statistical libraries
    return Math.max(0.001, 0.5 / Math.pow(tStat, 2));
  }

  /**
   * Perform Mann-Whitney U test (simplified)
   * @param {Array} group1 - First group values
   * @param {Array} group2 - Second group values
   * @returns {Object} Mann-Whitney U test results
   */
  performMannWhitneyU(group1, group2) {
    const combined = [...group1.map(val => ({val, group: 1})), ...group2.map(val => ({val, group: 2}))];
    combined.sort((a, b) => a.val - b.val);
    
    // Assign ranks
    let rank = 1;
    for (let i = 0; i < combined.length; i++) {
      let j = i;
      while (j < combined.length - 1 && combined[j].val === combined[j + 1].val) {
        j++;
      }
      
      const avgRank = (rank + rank + j - i) / 2;
      for (let k = i; k <= j; k++) {
        combined[k].rank = avgRank;
      }
      
      rank = j + 2;
      i = j;
    }
    
    // Calculate U statistics
    const r1 = combined.filter(item => item.group === 1).reduce((sum, item) => sum + item.rank, 0);
    const u1 = r1 - (group1.length * (group1.length + 1)) / 2;
    const u2 = group1.length * group2.length - u1;
    
    const uStat = Math.min(u1, u2);
    
    return {
      uStatistic: uStat,
      u1: u1,
      u2: u2,
      significant: this.isMannWhitneySignificant(uStat, group1.length, group2.length)
    };
  }

  /**
   * Check if Mann-Whitney U result is significant (simplified)
   * @param {number} uStat - U statistic
   * @param {number} n1 - First group size
   * @param {number} n2 - Second group size
   * @returns {boolean} Whether result is significant
   */
  isMannWhitneySignificant(uStat, n1, n2) {
    // Simplified critical value approximation
    const criticalValue = Math.floor(n1 * n2 / 2 - 1.96 * Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12));
    return uStat <= criticalValue;
  }

  /**
   * Calculate Pearson correlation coefficient
   * @param {Array} x - First variable values
   * @param {Array} y - Second variable values
   * @returns {number} Correlation coefficient
   */
  calculatePearsonCorrelation(x, y) {
    const n = x.length;
    const meanX = this.calculateMean(x);
    const meanY = this.calculateMean(y);
    
    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;
    
    for (let i = 0; i < n; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      
      numerator += xDiff * yDiff;
      sumXSquared += xDiff * xDiff;
      sumYSquared += yDiff * yDiff;
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    return denominator !== 0 ? numerator / denominator : 0;
  }

  /**
   * Test correlation significance
   * @param {number} r - Correlation coefficient
   * @param {number} n - Sample size
   * @returns {Object} Significance test results
   */
  testCorrelationSignificance(r, n) {
    if (n <= 2) return { significant: false, pValue: 1 };
    
    const tStat = r * Math.sqrt((n - 2) / (1 - r * r));
    const df = n - 2;
    const pValue = this.approximatePValue(Math.abs(tStat), df);
    
    return {
      tStatistic: tStat,
      degreesOfFreedom: df,
      pValue: pValue,
      significant: pValue < this.config.significanceLevel
    };
  }

  /**
   * Interpret correlation strength
   * @param {number} r - Absolute correlation coefficient
   * @returns {string} Strength interpretation
   */
  interpretCorrelationStrength(r) {
    if (r >= 0.9) return 'very strong';
    if (r >= 0.7) return 'strong';
    if (r >= 0.5) return 'moderate';
    if (r >= 0.3) return 'weak';
    return 'very weak';
  }

  /**
   * Calculate Cronbach's alpha for internal consistency
   * @param {Array} items - Array of item score arrays
   * @returns {number} Cronbach's alpha
   */
  calculateCronbachAlpha(items) {
    const k = items.length; // Number of items
    const n = items[0].length; // Number of observations
    
    // Calculate variance of each item
    const itemVariances = items.map(item => this.calculateVariance(item));
    const sumItemVariances = itemVariances.reduce((sum, variance) => sum + variance, 0);
    
    // Calculate total scores for each observation
    const totalScores = [];
    for (let i = 0; i < n; i++) {
      totalScores.push(items.reduce((sum, item) => sum + item[i], 0));
    }
    
    const totalVariance = this.calculateVariance(totalScores);
    
    if (totalVariance === 0) return 0;
    
    return (k / (k - 1)) * (1 - sumItemVariances / totalVariance);
  }

  /**
   * Calculate confidence interval for mean
   * @param {Array} values - Sample values
   * @param {number} confidenceLevel - Confidence level (0-1)
   * @returns {Object} Confidence interval
   */
  calculateMeanConfidenceInterval(values, confidenceLevel) {
    const mean = this.calculateMean(values);
    const std = this.calculateStandardDeviation(values);
    const n = values.length;
    
    // Critical value (approximation for t-distribution)
    const alpha = 1 - confidenceLevel;
    const tCritical = this.approximateTCritical(alpha / 2, n - 1);
    
    const marginOfError = tCritical * (std / Math.sqrt(n));
    
    return {
      lower: mean - marginOfError,
      upper: mean + marginOfError,
      margin: marginOfError
    };
  }

  /**
   * Calculate margin of error
   * @param {Array} values - Sample values
   * @param {number} confidenceLevel - Confidence level (0-1)
   * @returns {number} Margin of error
   */
  calculateMarginOfError(values, confidenceLevel) {
    const interval = this.calculateMeanConfidenceInterval(values, confidenceLevel);
    return interval.margin;
  }

  /**
   * Approximate t-critical value (simplified)
   * @param {number} alpha - Alpha level
   * @param {number} df - Degrees of freedom
   * @returns {number} Approximate t-critical value
   */
  approximateTCritical(alpha, df) {
    // Simplified approximation
    if (df >= 30) return 1.96; // Normal approximation
    if (df >= 20) return 2.086;
    if (df >= 10) return 2.228;
    return 2.776; // Conservative estimate for small samples
  }

  /**
   * Get quality grade from score
   * @param {number} score - Quality score (0-1)
   * @returns {string} Quality grade
   */
  getQualityGrade(score) {
    if (score >= 0.85) return 'Excellent';
    if (score >= 0.7) return 'Good';
    if (score >= 0.55) return 'Acceptable';
    if (score >= 0.4) return 'Poor';
    return 'Very Poor';
  }

  /**
   * Generate statistical summary
   * @param {Object} analysis - Statistical analysis results
   * @returns {Object} Statistical summary
   */
  generateStatisticalSummary(analysis) {
    const summary = {
      sampleSize: analysis.metadata.sampleSize,
      significantFindings: 0,
      reliabilityLevel: 'unknown',
      validityLevel: 'unknown',
      overallAssessment: '',
      keyFindings: []
    };

    // Count significant findings
    if (analysis.hypothesisTests?.groupComparisons) {
      Object.values(analysis.hypothesisTests.groupComparisons).forEach(comparison => {
        if (comparison.tTest?.significant || comparison.mannWhitneyU?.significant) {
          summary.significantFindings++;
        }
      });
    }

    // Assess reliability
    if (analysis.reliabilityAnalysis?.cronbachAlpha !== null) {
      const alpha = analysis.reliabilityAnalysis.cronbachAlpha;
      if (alpha >= 0.9) summary.reliabilityLevel = 'excellent';
      else if (alpha >= 0.8) summary.reliabilityLevel = 'good';
      else if (alpha >= 0.7) summary.reliabilityLevel = 'acceptable';
      else summary.reliabilityLevel = 'poor';
    }

    // Generate overall assessment
    if (summary.sampleSize < this.config.minimumSampleSize) {
      summary.overallAssessment = 'Insufficient sample size for reliable statistical conclusions';
    } else if (summary.reliabilityLevel === 'poor') {
      summary.overallAssessment = 'Low reliability limits confidence in findings';
    } else if (summary.significantFindings > 0) {
      summary.overallAssessment = 'Statistically significant differences detected with adequate reliability';
    } else {
      summary.overallAssessment = 'No significant differences detected; evaluation methods appear consistent';
    }

    return summary;
  }

  /**
   * Generate statistical recommendations
   * @param {Object} analysis - Statistical analysis results
   * @returns {Array} Recommendations
   */
  generateStatisticalRecommendations(analysis) {
    const recommendations = [];

    // Sample size recommendations
    if (analysis.metadata.sampleSize < this.config.recommendedSampleSize) {
      recommendations.push({
        category: 'sample_size',
        priority: 'high',
        recommendation: `Increase sample size to at least ${this.config.recommendedSampleSize} for more reliable results`,
        currentValue: analysis.metadata.sampleSize,
        targetValue: this.config.recommendedSampleSize
      });
    }

    // Reliability recommendations
    if (analysis.reliabilityAnalysis?.cronbachAlpha !== null && analysis.reliabilityAnalysis.cronbachAlpha < 0.7) {
      recommendations.push({
        category: 'reliability',
        priority: 'high',
        recommendation: 'Improve internal consistency of evaluation metrics',
        currentValue: analysis.reliabilityAnalysis.cronbachAlpha,
        targetValue: 0.8
      });
    }

    // Effect size recommendations
    if (analysis.effectSizeAnalysis?.cohensD) {
      const largeEffects = Object.values(analysis.effectSizeAnalysis.cohensD)
        .filter(effect => effect.category === 'large').length;
      
      if (largeEffects > 0) {
        recommendations.push({
          category: 'effect_size',
          priority: 'medium',
          recommendation: 'Large effect sizes detected; investigate practical significance of differences',
          currentValue: largeEffects,
          targetValue: 'investigation needed'
        });
      }
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  // Helper methods for specialized analyses

  /**
   * Calculate score differences for trend analysis
   * @param {Object} data - Analysis data
   * @returns {Object} Score differences
   */
  calculateScoreDifferences(data) {
    return {
      clarityVsCompleteness: data.clarityScores.map((clarity, i) => clarity - data.completenessScores[i]),
      actionabilityVsEffectiveness: data.actionabilityScores.map((action, i) => action - data.effectivenessScores[i])
    };
  }

  /**
   * Calculate consistency metrics
   * @param {Object} data - Analysis data
   * @returns {Object} Consistency metrics
   */
  calculateConsistencyMetrics(data) {
    return {
      overallConsistency: 1 - this.calculateCoefficientOfVariation(data.overallScores),
      dimensionalConsistency: {
        clarity: 1 - this.calculateCoefficientOfVariation(data.clarityScores),
        completeness: 1 - this.calculateCoefficientOfVariation(data.completenessScores),
        actionability: 1 - this.calculateCoefficientOfVariation(data.actionabilityScores),
        effectiveness: 1 - this.calculateCoefficientOfVariation(data.effectivenessScores)
      }
    };
  }

  /**
   * Calculate quality trends
   * @param {Object} data - Analysis data
   * @returns {Object} Quality trends
   */
  calculateQualityTrends(data) {
    // Simple linear trend analysis
    const trends = {};
    
    if (data.timestamps.length > 1) {
      trends.temporal = this.analyzeTemporalTrend(data.overallScores, data.timestamps);
    }
    
    return trends;
  }

  /**
   * Analyze temporal trend
   * @param {Array} scores - Score values
   * @param {Array} timestamps - Timestamps
   * @returns {Object} Temporal trend analysis
   */
  analyzeTemporalTrend(scores, timestamps) {
    if (scores.length !== timestamps.length || scores.length < 2) {
      return { trend: 'insufficient_data' };
    }

    // Convert timestamps to numeric values (hours since first timestamp)
    const firstTime = new Date(timestamps[0]).getTime();
    const timeValues = timestamps.map(ts => (new Date(ts).getTime() - firstTime) / (1000 * 60 * 60));
    
    // Calculate linear regression
    const correlation = this.calculatePearsonCorrelation(timeValues, scores);
    
    return {
      trend: correlation > 0.1 ? 'improving' : correlation < -0.1 ? 'declining' : 'stable',
      correlation: correlation,
      strength: this.interpretCorrelationStrength(Math.abs(correlation))
    };
  }

  /**
   * Test normality (simplified Shapiro-Wilk approximation)
   * @param {Array} values - Values to test
   * @returns {Object} Normality test results
   */
  testNormality(values) {
    if (values.length < 3) return { normal: false, reason: 'insufficient_data' };
    
    // Simplified normality check based on skewness and kurtosis
    const skewness = Math.abs(this.calculateSkewness(values));
    const kurtosis = Math.abs(this.calculateKurtosis(values));
    
    const isNormal = skewness < 1 && kurtosis < 3;
    
    return {
      normal: isNormal,
      skewness: this.calculateSkewness(values),
      kurtosis: this.calculateKurtosis(values),
      assessment: isNormal ? 'approximately_normal' : 'non_normal'
    };
  }

  /**
   * Create histogram data
   * @param {Array} values - Values to histogram
   * @param {number} bins - Number of bins
   * @returns {Object} Histogram data
   */
  createHistogram(values, bins = 10) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;
    
    const histogram = new Array(bins).fill(0);
    const binEdges = [];
    
    for (let i = 0; i <= bins; i++) {
      binEdges.push(min + i * binWidth);
    }
    
    values.forEach(val => {
      const binIndex = Math.min(Math.floor((val - min) / binWidth), bins - 1);
      histogram[binIndex]++;
    });
    
    return {
      bins: histogram,
      edges: binEdges,
      binWidth: binWidth
    };
  }

  /**
   * Detect outliers using IQR method
   * @param {Array} values - Values to check for outliers
   * @returns {Object} Outlier analysis
   */
  detectOutliers(values) {
    const quartiles = this.calculateQuartiles(values);
    const iqr = quartiles.iqr;
    const lowerFence = quartiles.q1 - 1.5 * iqr;
    const upperFence = quartiles.q3 + 1.5 * iqr;
    
    const outliers = values.filter(val => val < lowerFence || val > upperFence);
    
    return {
      outliers: outliers,
      count: outliers.length,
      percentage: (outliers.length / values.length) * 100,
      lowerFence: lowerFence,
      upperFence: upperFence
    };
  }

  /**
   * Classify distribution type
   * @param {Array} values - Values to classify
   * @returns {string} Distribution type
   */
  classifyDistribution(values) {
    const normalityTest = this.testNormality(values);
    
    if (normalityTest.normal) return 'normal';
    
    const skewness = normalityTest.skewness;
    if (skewness > 1) return 'right_skewed';
    if (skewness < -1) return 'left_skewed';
    
    const kurtosis = normalityTest.kurtosis;
    if (kurtosis > 3) return 'heavy_tailed';
    if (kurtosis < -1) return 'light_tailed';
    
    return 'unknown';
  }

  /**
   * Get empty statistics object
   * @returns {Object} Empty statistics
   */
  getEmptyStats() {
    return {
      count: 0,
      mean: 0,
      median: 0,
      mode: 0,
      standardDeviation: 0,
      variance: 0,
      min: 0,
      max: 0,
      range: 0,
      quartiles: { q1: 0, q2: 0, q3: 0, iqr: 0 },
      skewness: 0,
      kurtosis: 0,
      coefficientOfVariation: 0
    };
  }

  // Placeholder methods for complex analyses (would be implemented in full system)

  analyzeTestRetestReliability(data) {
    return { available: false, reason: 'requires repeated evaluations of same prompts' };
  }

  analyzeInterRaterReliability(data) {
    return { available: false, reason: 'requires multiple judges for same prompts' };
  }

  assessOverallReliability(reliability) {
    if (reliability.cronbachAlpha !== null) {
      return reliability.cronbachAlpha >= this.config.validation.reliabilityThreshold ? 'acceptable' : 'poor';
    }
    return 'unknown';
  }

  analyzeConstructValidity(data) {
    return { available: false, reason: 'requires external criterion measures' };
  }

  analyzeConvergentValidity(data) {
    return { available: false, reason: 'requires alternative measures of same construct' };
  }

  analyzeDiscriminantValidity(data) {
    return { available: false, reason: 'requires measures of different constructs' };
  }

  assessOverallValidity(validity) {
    return 'unknown';
  }

  performCorrelationTests(data) {
    return { available: false, reason: 'basic correlation analysis performed separately' };
  }

  performConsistencyTests(data) {
    return { available: false, reason: 'consistency metrics calculated separately' };
  }

  compareToBaseline(data, baseline) {
    return { available: false, reason: 'baseline comparison not implemented' };
  }

  assessPracticalSignificance(comparison) {
    return comparison.effectSizeCategory !== 'negligible';
  }

  /**
   * Validate inputs for statistical analysis
   * @param {Array} results - Evaluation results
   * @param {Object} options - Analysis options
   */
  validateInputs(results, options) {
    if (!Array.isArray(results) || results.length === 0) {
      throw new Error('Invalid results: must be a non-empty array');
    }

    if (results.length < this.config.minimumSampleSize) {
      throw new Error(`Insufficient sample size: minimum ${this.config.minimumSampleSize} required`);
    }

    // Check that results have required numeric fields
    const hasValidScores = results.some(result => 
      typeof (result.overallScore || result.overallQuality) === 'number'
    );

    if (!hasValidScores) {
      throw new Error('Results must contain numeric quality scores');
    }
  }
}

module.exports = StatisticalAnalyzer;