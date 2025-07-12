/**
 * Insight Generation Engine
 * Automated insight extraction and actionable recommendation generation
 */

class InsightGenerationEngine {
  constructor(config = {}) {
    this.config = {
      // Insight detection thresholds
      significanceThreshold: 0.1,      // Minimum effect size for significant insights
      confidenceThreshold: 0.7,        // Minimum confidence for actionable insights
      minSampleSize: 10,               // Minimum samples for reliable insights
      
      // Insight categorization
      maxInsightsPerCategory: 10,      // Maximum insights per category
      insightCategories: [
        'performance', 'patterns', 'optimization', 
        'quality', 'efficiency', 'trends'
      ],
      
      // Recommendation generation
      maxRecommendations: 15,          // Maximum recommendations to generate
      priorityLevels: ['critical', 'high', 'medium', 'low'],
      
      ...config
    };

    // Insight tracking
    this.discoveredInsights = new Map();
    this.trendAnalysis = new Map();
    this.performanceBaseline = null;
  }

  /**
   * Generate comprehensive insights from multiple analysis sources
   * @param {Object} analysisData - Combined analysis data from all sources
   * @returns {Object} Generated insights and recommendations
   */
  async generateInsights(analysisData) {
    const {
      testResults,
      ruleEffectiveness,
      contextLearning,
      failureAnalysis,
      statisticalAnalysis,
      historicalData = []
    } = analysisData;

    // Core insight generation
    const insights = {
      performance: await this.generatePerformanceInsights(testResults, statisticalAnalysis),
      patterns: await this.generatePatternInsights(ruleEffectiveness, contextLearning),
      optimization: await this.generateOptimizationInsights(ruleEffectiveness, failureAnalysis),
      quality: await this.generateQualityInsights(testResults, failureAnalysis),
      efficiency: await this.generateEfficiencyInsights(testResults, ruleEffectiveness),
      trends: await this.generateTrendInsights(historicalData, testResults)
    };

    // Cross-category analysis
    const crossCategoryInsights = this.performCrossCategoryAnalysis(insights);
    
    // Generate actionable recommendations
    const recommendations = await this.generateActionableRecommendations(insights, analysisData);
    
    // Strategic insights
    const strategicInsights = this.generateStrategicInsights(insights, recommendations);
    
    // Insight validation and ranking
    const validatedInsights = this.validateAndRankInsights(insights);
    const prioritizedRecommendations = this.prioritizeRecommendations(recommendations);

    return {
      summary: this.generateInsightSummary(validatedInsights, prioritizedRecommendations),
      insights: validatedInsights,
      crossCategoryInsights,
      recommendations: prioritizedRecommendations,
      strategicInsights,
      
      // Detailed analysis
      insightsByCategory: insights,
      confidenceMetrics: this.calculateConfidenceMetrics(validatedInsights),
      actionabilityScores: this.calculateActionabilityScores(prioritizedRecommendations),
      
      // Metadata
      metadata: {
        totalInsights: Object.values(validatedInsights).reduce((sum, cat) => sum + cat.length, 0),
        totalRecommendations: prioritizedRecommendations.length,
        analysisDepth: this.calculateAnalysisDepth(analysisData),
        generationDate: new Date().toISOString()
      }
    };
  }

  /**
   * Generate performance insights
   * @private
   */
  async generatePerformanceInsights(testResults, statisticalAnalysis) {
    const insights = [];
    
    // Overall performance analysis
    const improvements = testResults.map(r => r.overallImprovement || 0);
    const avgImprovement = improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length;
    const successRate = testResults.filter(r => (r.overallImprovement || 0) >= 0.7).length / testResults.length;
    
    if (avgImprovement >= 0.8) {
      insights.push({
        type: 'high_performance',
        title: 'Exceptional Overall Performance',
        description: `Average improvement of ${(avgImprovement * 100).toFixed(1)}% indicates highly effective prompt engineering`,
        significance: 0.9,
        confidence: 0.95,
        evidence: {
          avgImprovement,
          successRate,
          sampleSize: testResults.length
        },
        implications: [
          'Current rule set is highly effective',
          'Focus on consistency rather than major changes',
          'Consider this as best practice baseline'
        ]
      });
    } else if (avgImprovement < 0.5) {
      insights.push({
        type: 'performance_concern',
        title: 'Below-Average Performance Detected',
        description: `Average improvement of ${(avgImprovement * 100).toFixed(1)}% suggests significant optimization opportunities`,
        significance: 0.8,
        confidence: 0.9,
        evidence: {
          avgImprovement,
          successRate,
          sampleSize: testResults.length
        },
        implications: [
          'Major rule set revision needed',
          'Consider fundamental approach changes',
          'Investigate root causes systematically'
        ]
      });
    }

    // Performance distribution insights
    const performanceDistribution = this.calculatePerformanceDistribution(improvements);
    if (performanceDistribution.highVariance) {
      insights.push({
        type: 'inconsistent_performance',
        title: 'High Performance Variability',
        description: 'Large variation in improvement scores indicates inconsistent rule effectiveness',
        significance: 0.7,
        confidence: 0.8,
        evidence: performanceDistribution,
        implications: [
          'Some rules work well, others poorly',
          'Context-specific optimization needed',
          'Consider conditional rule application'
        ]
      });
    }

    // Statistical significance insights
    if (statisticalAnalysis && statisticalAnalysis.significance) {
      insights.push({
        type: 'statistical_validation',
        title: 'Statistically Significant Improvements',
        description: `Results show statistical significance (p < ${statisticalAnalysis.pValue})`,
        significance: 0.8,
        confidence: 0.95,
        evidence: statisticalAnalysis,
        implications: [
          'Improvements are not due to chance',
          'Changes can be trusted for deployment',
          'Results are scientifically valid'
        ]
      });
    }

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Generate pattern insights
   * @private
   */
  async generatePatternInsights(ruleEffectiveness, contextLearning) {
    const insights = [];
    
    // Rule effectiveness patterns
    if (ruleEffectiveness && ruleEffectiveness.topPerformers) {
      const topRule = ruleEffectiveness.topPerformers[0];
      if (topRule && topRule.successRate > 0.9) {
        insights.push({
          type: 'dominant_rule_pattern',
          title: `${topRule.name} Shows Exceptional Effectiveness`,
          description: `Rule shows ${(topRule.successRate * 100).toFixed(1)}% success rate across ${topRule.applications} applications`,
          significance: 0.9,
          confidence: 0.9,
          evidence: {
            rule: topRule.name,
            successRate: topRule.successRate,
            applications: topRule.applications,
            category: topRule.category
          },
          implications: [
            'Prioritize this rule in future improvements',
            'Study why this rule is so effective',
            'Consider expanding this rule type'
          ]
        });
      }

      // Underperforming rule patterns
      const worstRule = ruleEffectiveness.underperformers?.[0];
      if (worstRule && worstRule.successRate < 0.3) {
        insights.push({
          type: 'failing_rule_pattern',
          title: `${worstRule.name} Consistently Underperforms`,
          description: `Rule shows only ${(worstRule.successRate * 100).toFixed(1)}% success rate - needs revision`,
          significance: 0.8,
          confidence: 0.85,
          evidence: {
            rule: worstRule.name,
            successRate: worstRule.successRate,
            applications: worstRule.applications,
            category: worstRule.category
          },
          implications: [
            'Rule should be revised or removed',
            'Investigate why this rule fails',
            'May need context-specific variants'
          ]
        });
      }
    }

    // Context-specific patterns
    if (contextLearning && contextLearning.contextInsights) {
      const contexts = Object.values(contextLearning.contextInsights);
      const bestContext = contexts.reduce((best, ctx) => 
        !best || ctx.performance.averageImprovement > best.performance.averageImprovement ? ctx : best
      );
      
      if (bestContext && bestContext.performance.averageImprovement > 0.8) {
        insights.push({
          type: 'context_excellence_pattern',
          title: `${bestContext.contextKey} Context Shows Superior Performance`,
          description: `${(bestContext.performance.averageImprovement * 100).toFixed(1)}% average improvement in this context`,
          significance: 0.8,
          confidence: 0.8,
          evidence: {
            context: bestContext.contextKey,
            performance: bestContext.performance,
            sampleSize: bestContext.sampleSize
          },
          implications: [
            'Apply successful patterns to other contexts',
            'Study what makes this context effective',
            'Develop context-specific best practices'
          ]
        });
      }

      // Universal vs context-specific patterns
      if (contextLearning.universalPatterns && contextLearning.universalPatterns.length > 0) {
        insights.push({
          type: 'universal_pattern_strength',
          title: 'Strong Universal Patterns Identified',
          description: `${contextLearning.universalPatterns.length} patterns work consistently across contexts`,
          significance: 0.7,
          confidence: 0.9,
          evidence: {
            patterns: contextLearning.universalPatterns,
            contextCount: contexts.length
          },
          implications: [
            'Focus on strengthening universal patterns',
            'These are reliable improvement strategies',
            'Good candidates for automation'
          ]
        });
      }
    }

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Generate optimization insights
   * @private
   */
  async generateOptimizationInsights(ruleEffectiveness, failureAnalysis) {
    const insights = [];
    
    // Rule optimization opportunities
    if (ruleEffectiveness && ruleEffectiveness.underperformers) {
      const improvableRules = ruleEffectiveness.underperformers.filter(rule => 
        rule.applications >= 10 && rule.successRate < 0.6
      );
      
      if (improvableRules.length > 0) {
        insights.push({
          type: 'rule_optimization_opportunity',
          title: `${improvableRules.length} Rules Show Clear Optimization Potential`,
          description: 'Multiple rules with sufficient data show suboptimal performance',
          significance: 0.8,
          confidence: 0.8,
          evidence: {
            improvableRules: improvableRules.slice(0, 5),
            totalCount: improvableRules.length
          },
          implications: [
            'Systematic rule improvement needed',
            'High-impact optimization targets identified',
            'Could significantly boost overall performance'
          ]
        });
      }
    }

    // Failure-based optimization insights
    if (failureAnalysis && failureAnalysis.patterns) {
      const significantPatterns = failureAnalysis.patterns.filter(p => p.frequency >= 5);
      
      if (significantPatterns.length > 0) {
        const topPattern = significantPatterns[0];
        insights.push({
          type: 'failure_pattern_optimization',
          title: `Address ${topPattern.characteristic} Failure Pattern`,
          description: `${topPattern.frequency} failures show ${topPattern.characteristic} pattern - clear optimization target`,
          significance: 0.9,
          confidence: 0.85,
          evidence: {
            pattern: topPattern,
            totalPatterns: significantPatterns.length
          },
          implications: [
            'Targeted fix could eliminate multiple failures',
            'High-impact, focused optimization opportunity',
            'Address root cause rather than symptoms'
          ]
        });
      }
    }

    // Gap analysis insights
    if (failureAnalysis && failureAnalysis.ruleGaps) {
      const criticalGaps = failureAnalysis.ruleGaps.filter(gap => 
        gap.frequency >= 3 && gap.priority === 'high'
      );
      
      if (criticalGaps.length > 0) {
        insights.push({
          type: 'critical_capability_gap',
          title: `${criticalGaps.length} Critical Rule Gaps Identified`,
          description: 'Missing rules for common failure patterns represent optimization opportunities',
          significance: 0.8,
          confidence: 0.9,
          evidence: {
            gaps: criticalGaps,
            totalGaps: failureAnalysis.ruleGaps.length
          },
          implications: [
            'Develop rules for identified gaps',
            'Could prevent recurring failure types',
            'Systematic capability enhancement needed'
          ]
        });
      }
    }

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Generate quality insights
   * @private
   */
  async generateQualityInsights(testResults, failureAnalysis) {
    const insights = [];
    
    // Quality consistency analysis
    const qualityScores = testResults.map(r => r.qualityScore || r.overallImprovement || 0);
    const qualityConsistency = this.calculateConsistency(qualityScores);
    
    if (qualityConsistency < 0.5) {
      insights.push({
        type: 'quality_inconsistency',
        title: 'Inconsistent Quality Outcomes',
        description: `Quality consistency score of ${(qualityConsistency * 100).toFixed(1)}% indicates unpredictable results`,
        significance: 0.7,
        confidence: 0.8,
        evidence: {
          consistencyScore: qualityConsistency,
          qualityDistribution: this.calculateDistribution(qualityScores)
        },
        implications: [
          'Need better quality prediction mechanisms',
          'Consider quality-based rule selection',
          'Improve quality measurement consistency'
        ]
      });
    }

    // Quality threshold insights
    const highQualityCount = testResults.filter(r => (r.overallImprovement || 0) >= 0.8).length;
    const lowQualityCount = testResults.filter(r => (r.overallImprovement || 0) < 0.3).length;
    
    if (highQualityCount / testResults.length > 0.7) {
      insights.push({
        type: 'high_quality_dominance',
        title: 'Majority of Results Achieve High Quality',
        description: `${(highQualityCount / testResults.length * 100).toFixed(1)}% of results exceed 80% improvement threshold`,
        significance: 0.8,
        confidence: 0.9,
        evidence: {
          highQualityRate: highQualityCount / testResults.length,
          threshold: 0.8,
          sampleSize: testResults.length
        },
        implications: [
          'Current approach is highly effective',
          'Focus on consistency rather than improvement',
          'Good baseline for future development'
        ]
      });
    }

    // Quality-failure correlation
    if (failureAnalysis && lowQualityCount > 0) {
      const qualityFailureCorrelation = this.calculateQualityFailureCorrelation(testResults, failureAnalysis);
      
      if (qualityFailureCorrelation.strength > 0.6) {
        insights.push({
          type: 'quality_failure_correlation',
          title: 'Strong Correlation Between Quality Metrics and Failures',
          description: `${(qualityFailureCorrelation.strength * 100).toFixed(1)}% correlation suggests quality metrics predict failures well`,
          significance: 0.8,
          confidence: 0.85,
          evidence: qualityFailureCorrelation,
          implications: [
            'Quality metrics are reliable failure predictors',
            'Can use quality thresholds for early detection',
            'Focus quality improvements on failure-prone areas'
          ]
        });
      }
    }

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Generate efficiency insights
   * @private
   */
  async generateEfficiencyInsights(testResults, ruleEffectiveness) {
    const insights = [];
    
    // Rule efficiency analysis
    if (ruleEffectiveness && ruleEffectiveness.topPerformers) {
      const highEfficiencyRules = ruleEffectiveness.topPerformers.filter(rule => 
        rule.applications >= 10 && rule.successRate >= 0.8
      );
      
      const ruleEfficiencyRatio = highEfficiencyRules.length / ruleEffectiveness.topPerformers.length;
      
      if (ruleEfficiencyRatio > 0.6) {
        insights.push({
          type: 'high_rule_efficiency',
          title: 'Majority of Top Rules Show High Efficiency',
          description: `${(ruleEfficiencyRatio * 100).toFixed(1)}% of top-performing rules also show high efficiency`,
          significance: 0.7,
          confidence: 0.8,
          evidence: {
            efficiencyRatio: ruleEfficiencyRatio,
            highEfficiencyCount: highEfficiencyRules.length,
            totalTopRules: ruleEffectiveness.topPerformers.length
          },
          implications: [
            'Rule set is well-optimized for efficiency',
            'Focus on scaling rather than optimization',
            'Good foundation for automated application'
          ]
        });
      }
    }

    // Processing efficiency
    const processingTimes = testResults.map(r => r.executionTime || 0).filter(t => t > 0);
    if (processingTimes.length > 0) {
      const avgProcessingTime = processingTimes.reduce((sum, t) => sum + t, 0) / processingTimes.length;
      const efficiency = this.calculateProcessingEfficiency(processingTimes, testResults);
      
      if (efficiency.score > 0.8) {
        insights.push({
          type: 'processing_efficiency',
          title: 'Excellent Processing Efficiency',
          description: `Average processing time of ${avgProcessingTime.toFixed(0)}ms with high consistency`,
          significance: 0.6,
          confidence: 0.9,
          evidence: efficiency,
          implications: [
            'System can handle increased load',
            'Processing pipeline is well-optimized',
            'Good candidate for real-time applications'
          ]
        });
      }
    }

    // Resource utilization insights
    const resourceUtilization = this.analyzeResourceUtilization(testResults, ruleEffectiveness);
    if (resourceUtilization.efficiency > 0.7) {
      insights.push({
        type: 'resource_efficiency',
        title: 'Optimal Resource Utilization Achieved',
        description: `${(resourceUtilization.efficiency * 100).toFixed(1)}% resource efficiency indicates well-balanced system`,
        significance: 0.6,
        confidence: 0.7,
        evidence: resourceUtilization,
        implications: [
          'Resource allocation is well-optimized',
          'System operates near optimal capacity',
          'Good foundation for scaling operations'
        ]
      });
    }

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Generate trend insights from historical data
   * @private
   */
  async generateTrendInsights(historicalData, currentResults) {
    const insights = [];
    
    if (!historicalData || historicalData.length < 2) {
      return [{
        type: 'insufficient_trend_data',
        title: 'Insufficient Historical Data for Trend Analysis',
        description: 'Need more data points for meaningful trend analysis',
        significance: 0.3,
        confidence: 0.9,
        evidence: { dataPoints: historicalData.length },
        implications: ['Continue collecting data for future trend analysis']
      }];
    }

    // Performance trend analysis
    const performanceTrend = this.calculatePerformanceTrend(historicalData, currentResults);
    
    if (performanceTrend.direction === 'improving' && performanceTrend.strength > 0.6) {
      insights.push({
        type: 'positive_performance_trend',
        title: 'Strong Positive Performance Trend',
        description: `Performance improving by ${(performanceTrend.rate * 100).toFixed(1)}% per period`,
        significance: 0.8,
        confidence: 0.8,
        evidence: performanceTrend,
        implications: [
          'Current optimization strategy is working',
          'Continue current improvement trajectory',
          'Performance gains are sustainable'
        ]
      });
    } else if (performanceTrend.direction === 'declining' && performanceTrend.strength > 0.5) {
      insights.push({
        type: 'concerning_performance_trend',
        title: 'Performance Decline Detected',
        description: `Performance declining by ${(Math.abs(performanceTrend.rate) * 100).toFixed(1)}% per period`,
        significance: 0.9,
        confidence: 0.8,
        evidence: performanceTrend,
        implications: [
          'Immediate investigation required',
          'Review recent changes for negative impacts',
          'Consider reverting to previous configuration'
        ]
      });
    }

    // Quality trend analysis
    const qualityTrend = this.calculateQualityTrend(historicalData, currentResults);
    if (qualityTrend.significance > 0.6) {
      insights.push({
        type: 'quality_trend_insight',
        title: `Quality ${qualityTrend.direction} Trend Identified`,
        description: `Quality metrics show ${qualityTrend.direction} trend over ${historicalData.length} periods`,
        significance: 0.7,
        confidence: 0.75,
        evidence: qualityTrend,
        implications: qualityTrend.implications
      });
    }

    // Usage pattern trends
    const usageTrends = this.analyzeUsageTrends(historicalData);
    insights.push(...usageTrends);

    return insights.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Perform cross-category analysis to find interconnected insights
   * @private
   */
  performCrossCategoryAnalysis(insights) {
    const crossCategoryInsights = [];
    
    // Performance-Quality correlation
    const performanceInsights = insights.performance || [];
    const qualityInsights = insights.quality || [];
    
    const highPerf = performanceInsights.some(i => i.type === 'high_performance');
    const highQuality = qualityInsights.some(i => i.type === 'high_quality_dominance');
    
    if (highPerf && highQuality) {
      crossCategoryInsights.push({
        type: 'performance_quality_alignment',
        title: 'Performance and Quality Metrics Align',
        description: 'Both performance and quality show consistently high results',
        significance: 0.8,
        confidence: 0.9,
        categories: ['performance', 'quality'],
        implications: [
          'System is operating at optimal levels',
          'Both speed and quality are well-balanced',
          'Excellent foundation for production deployment'
        ]
      });
    }

    // Pattern-Optimization synergy
    const patternInsights = insights.patterns || [];
    const optimizationInsights = insights.optimization || [];
    
    const strongPatterns = patternInsights.filter(i => i.significance > 0.8);
    const clearOptimizations = optimizationInsights.filter(i => i.significance > 0.7);
    
    if (strongPatterns.length > 0 && clearOptimizations.length > 0) {
      crossCategoryInsights.push({
        type: 'pattern_optimization_synergy',
        title: 'Clear Patterns Enable Targeted Optimization',
        description: 'Strong patterns provide clear optimization targets',
        significance: 0.7,
        confidence: 0.8,
        categories: ['patterns', 'optimization'],
        implications: [
          'Pattern-based optimization strategy recommended',
          'Focus optimization efforts on identified patterns',
          'High probability of successful optimization'
        ]
      });
    }

    // Efficiency-Trend correlation
    const efficiencyInsights = insights.efficiency || [];
    const trendInsights = insights.trends || [];
    
    const highEfficiency = efficiencyInsights.some(i => i.type.includes('efficiency'));
    const positiveTrend = trendInsights.some(i => i.type === 'positive_performance_trend');
    
    if (highEfficiency && positiveTrend) {
      crossCategoryInsights.push({
        type: 'sustainable_improvement',
        title: 'Efficient System Shows Sustainable Improvement',
        description: 'High efficiency combined with positive trends indicates sustainable growth',
        significance: 0.8,
        confidence: 0.8,
        categories: ['efficiency', 'trends'],
        implications: [
          'Current approach is both effective and sustainable',
          'System can maintain improvement trajectory',
          'Good candidate for long-term deployment'
        ]
      });
    }

    return crossCategoryInsights;
  }

  /**
   * Generate actionable recommendations based on insights
   * @private
   */
  async generateActionableRecommendations(insights, analysisData) {
    const recommendations = [];
    
    // High-priority recommendations from critical insights
    const criticalInsights = this.getCriticalInsights(insights);
    for (const insight of criticalInsights) {
      const rec = this.generateRecommendationFromInsight(insight, 'critical');
      if (rec) recommendations.push(rec);
    }

    // Performance-based recommendations
    if (analysisData.testResults) {
      const perfRecs = this.generatePerformanceRecommendations(analysisData.testResults, insights.performance);
      recommendations.push(...perfRecs);
    }

    // Rule-based recommendations
    if (analysisData.ruleEffectiveness) {
      const ruleRecs = this.generateRuleRecommendations(analysisData.ruleEffectiveness, insights.patterns);
      recommendations.push(...ruleRecs);
    }

    // Context-based recommendations
    if (analysisData.contextLearning) {
      const contextRecs = this.generateContextRecommendations(analysisData.contextLearning, insights.patterns);
      recommendations.push(...contextRecs);
    }

    // Failure-based recommendations
    if (analysisData.failureAnalysis) {
      const failureRecs = this.generateFailureRecommendations(analysisData.failureAnalysis, insights.optimization);
      recommendations.push(...failureRecs);
    }

    // Strategic recommendations
    const strategicRecs = this.generateStrategicRecommendations(insights);
    recommendations.push(...strategicRecs);

    return recommendations.slice(0, this.config.maxRecommendations);
  }

  /**
   * Generate strategic insights that guide long-term decisions
   * @private
   */
  generateStrategicInsights(insights, recommendations) {
    const strategicInsights = [];
    
    // System maturity assessment
    const maturityLevel = this.assessSystemMaturity(insights);
    strategicInsights.push({
      type: 'system_maturity',
      title: `System Maturity Level: ${maturityLevel.level}`,
      description: maturityLevel.description,
      recommendations: maturityLevel.recommendations,
      timeHorizon: 'long_term'
    });

    // Investment priority analysis
    const investmentPriorities = this.analyzeInvestmentPriorities(insights, recommendations);
    strategicInsights.push({
      type: 'investment_priorities',
      title: 'Strategic Investment Recommendations',
      description: 'Prioritized areas for development investment',
      priorities: investmentPriorities,
      timeHorizon: 'medium_term'
    });

    // Risk assessment
    const riskAssessment = this.assessSystemRisks(insights);
    if (riskAssessment.risks.length > 0) {
      strategicInsights.push({
        type: 'risk_assessment',
        title: 'System Risk Analysis',
        description: 'Identified risks and mitigation strategies',
        risks: riskAssessment.risks,
        mitigationStrategies: riskAssessment.mitigations,
        timeHorizon: 'immediate'
      });
    }

    // Scalability assessment
    const scalabilityAssessment = this.assessScalability(insights);
    strategicInsights.push({
      type: 'scalability_assessment',
      title: 'System Scalability Analysis',
      description: scalabilityAssessment.description,
      currentCapacity: scalabilityAssessment.currentCapacity,
      scalingRecommendations: scalabilityAssessment.recommendations,
      timeHorizon: 'medium_term'
    });

    return strategicInsights;
  }

  // Helper methods for calculations and analysis

  calculatePerformanceDistribution(improvements) {
    const mean = improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length;
    const variance = improvements.reduce((sum, imp) => sum + Math.pow(imp - mean, 2), 0) / improvements.length;
    const standardDeviation = Math.sqrt(variance);
    
    return {
      mean,
      standardDeviation,
      variance,
      highVariance: standardDeviation / mean > 0.3, // 30% coefficient of variation
      distribution: {
        q1: this.calculatePercentile(improvements, 25),
        median: this.calculatePercentile(improvements, 50),
        q3: this.calculatePercentile(improvements, 75)
      }
    };
  }

  calculateConsistency(values) {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const standardDeviation = Math.sqrt(variance);
    
    // Normalize consistency score (lower std dev = higher consistency)
    return Math.max(0, 1 - (standardDeviation / Math.max(mean, 0.1)));
  }

  calculateDistribution(values) {
    const sorted = [...values].sort((a, b) => a - b);
    return {
      min: sorted[0],
      q1: sorted[Math.floor(sorted.length * 0.25)],
      median: sorted[Math.floor(sorted.length * 0.5)],
      q3: sorted[Math.floor(sorted.length * 0.75)],
      max: sorted[sorted.length - 1],
      mean: values.reduce((sum, val) => sum + val, 0) / values.length
    };
  }

  calculatePercentile(values, percentile) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor((percentile / 100) * sorted.length);
    return sorted[Math.min(index, sorted.length - 1)];
  }

  calculateQualityFailureCorrelation(testResults, failureAnalysis) {
    // Simplified correlation calculation
    const qualityScores = testResults.map(r => r.overallImprovement || 0);
    const failureCount = failureAnalysis.metadata?.totalFailures || 0;
    const totalTests = testResults.length;
    
    return {
      strength: Math.max(0, 1 - (failureCount / totalTests)),
      qualityMean: qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length,
      failureRate: failureCount / totalTests,
      sampleSize: totalTests
    };
  }

  calculateProcessingEfficiency(processingTimes, testResults) {
    const avgTime = processingTimes.reduce((sum, t) => sum + t, 0) / processingTimes.length;
    const timeConsistency = this.calculateConsistency(processingTimes);
    
    return {
      score: Math.min(timeConsistency, 1000 / Math.max(avgTime, 100)), // Normalize against 1 second
      avgProcessingTime: avgTime,
      timeConsistency,
      throughput: testResults.length / (avgTime / 1000) // Tests per second
    };
  }

  analyzeResourceUtilization(testResults, ruleEffectiveness) {
    // Mock resource utilization analysis
    const ruleApplications = ruleEffectiveness?.topPerformers?.reduce((sum, rule) => 
      sum + rule.applications, 0) || 0;
    const efficiency = Math.min(1, ruleApplications / (testResults.length * 3)); // Assume 3 rules per test optimal
    
    return {
      efficiency,
      ruleApplicationRate: ruleApplications / testResults.length,
      utilizationScore: efficiency * 0.7 + (testResults.length > 50 ? 0.3 : testResults.length / 50 * 0.3)
    };
  }

  calculatePerformanceTrend(historicalData, currentResults) {
    if (historicalData.length < 2) {
      return { direction: 'unknown', strength: 0, rate: 0 };
    }

    const currentAvg = currentResults.reduce((sum, r) => sum + (r.overallImprovement || 0), 0) / currentResults.length;
    const previousAvg = historicalData[historicalData.length - 1].averageImprovement || 0;
    
    const rate = (currentAvg - previousAvg) / Math.max(previousAvg, 0.1);
    const direction = rate > 0.05 ? 'improving' : rate < -0.05 ? 'declining' : 'stable';
    const strength = Math.abs(rate);
    
    return {
      direction,
      strength,
      rate,
      currentAvg,
      previousAvg,
      dataPoints: historicalData.length + 1
    };
  }

  calculateQualityTrend(historicalData, currentResults) {
    // Mock quality trend calculation
    return {
      direction: 'stable',
      significance: 0.4,
      implications: ['Quality metrics remain consistent', 'No significant quality changes detected']
    };
  }

  analyzeUsageTrends(historicalData) {
    // Mock usage trend analysis
    return [{
      type: 'usage_trend',
      title: 'Stable Usage Patterns',
      description: 'Usage patterns remain consistent over time',
      significance: 0.5,
      confidence: 0.7,
      evidence: { periods: historicalData.length },
      implications: ['System usage is predictable', 'No significant demand changes']
    }];
  }

  getCriticalInsights(insights) {
    const critical = [];
    for (const category of Object.values(insights)) {
      for (const insight of category) {
        if (insight.significance >= 0.8 && insight.confidence >= 0.8) {
          critical.push(insight);
        }
      }
    }
    return critical.sort((a, b) => (b.significance * b.confidence) - (a.significance * a.confidence));
  }

  generateRecommendationFromInsight(insight, priority) {
    return {
      id: `rec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      priority,
      title: `Address: ${insight.title}`,
      description: insight.implications?.[0] || 'Take action based on this insight',
      actions: insight.implications || ['Review and take appropriate action'],
      basedOnInsight: insight.type,
      confidence: insight.confidence,
      impact: this.estimateImpact(insight),
      effort: this.estimateEffort(insight),
      timeframe: this.estimateTimeframe(insight)
    };
  }

  generatePerformanceRecommendations(testResults, performanceInsights) {
    const recommendations = [];
    
    const avgImprovement = testResults.reduce((sum, r) => sum + (r.overallImprovement || 0), 0) / testResults.length;
    
    if (avgImprovement < 0.6) {
      recommendations.push({
        id: `perf_rec_${Date.now()}`,
        priority: 'high',
        title: 'Improve Overall Performance',
        description: 'Average improvement below 60% indicates need for systematic enhancement',
        actions: [
          'Review and optimize underperforming rules',
          'Analyze top failure patterns',
          'Consider rule set restructuring'
        ],
        confidence: 0.9,
        impact: 'high',
        effort: 'medium',
        timeframe: 'short_term'
      });
    }
    
    return recommendations;
  }

  generateRuleRecommendations(ruleEffectiveness, patternInsights) {
    const recommendations = [];
    
    if (ruleEffectiveness.underperformers && ruleEffectiveness.underperformers.length > 0) {
      const worstRule = ruleEffectiveness.underperformers[0];
      recommendations.push({
        id: `rule_rec_${Date.now()}`,
        priority: 'medium',
        title: `Optimize ${worstRule.name} Rule`,
        description: `Rule shows ${(worstRule.successRate * 100).toFixed(1)}% success rate - needs improvement`,
        actions: [
          'Analyze rule failure cases',
          'Revise rule logic',
          'Test improved rule version'
        ],
        confidence: 0.8,
        impact: 'medium',
        effort: 'low',
        timeframe: 'short_term'
      });
    }
    
    return recommendations;
  }

  generateContextRecommendations(contextLearning, patternInsights) {
    const recommendations = [];
    
    if (contextLearning.specializationOpportunities && contextLearning.specializationOpportunities.length > 0) {
      const topOpportunity = contextLearning.specializationOpportunities[0];
      recommendations.push({
        id: `context_rec_${Date.now()}`,
        priority: topOpportunity.priority || 'medium',
        title: topOpportunity.recommendation,
        description: topOpportunity.opportunity,
        actions: [
          'Develop context-specific rule variant',
          'Test in target context',
          'Monitor specialized performance'
        ],
        confidence: topOpportunity.confidence || 0.7,
        impact: 'medium',
        effort: 'medium',
        timeframe: 'medium_term'
      });
    }
    
    return recommendations;
  }

  generateFailureRecommendations(failureAnalysis, optimizationInsights) {
    const recommendations = [];
    
    if (failureAnalysis.recommendations && failureAnalysis.recommendations.length > 0) {
      const topFailureRec = failureAnalysis.recommendations[0];
      recommendations.push({
        id: `failure_rec_${Date.now()}`,
        priority: topFailureRec.priority || 'high',
        title: topFailureRec.title,
        description: topFailureRec.description,
        actions: topFailureRec.actions || ['Address identified failure pattern'],
        confidence: 0.8,
        impact: topFailureRec.expectedImpact || 'medium',
        effort: topFailureRec.effort || 'medium',
        timeframe: 'short_term'
      });
    }
    
    return recommendations;
  }

  generateStrategicRecommendations(insights) {
    const recommendations = [];
    
    // Check if system is performing well overall
    const hasHighPerformance = insights.performance?.some(i => i.type === 'high_performance');
    const hasStrongPatterns = insights.patterns?.some(i => i.significance > 0.8);
    
    if (hasHighPerformance && hasStrongPatterns) {
      recommendations.push({
        id: `strategic_rec_${Date.now()}`,
        priority: 'low',
        title: 'Consider Production Deployment',
        description: 'System shows strong performance and clear patterns - ready for broader deployment',
        actions: [
          'Prepare production deployment plan',
          'Set up monitoring and alerting',
          'Plan gradual rollout strategy'
        ],
        confidence: 0.8,
        impact: 'high',
        effort: 'high',
        timeframe: 'long_term'
      });
    }
    
    return recommendations;
  }

  validateAndRankInsights(insights) {
    const validated = {};
    
    for (const [category, categoryInsights] of Object.entries(insights)) {
      validated[category] = categoryInsights
        .filter(insight => 
          insight.confidence >= this.config.confidenceThreshold * 0.8 && // Slightly lower threshold for insights
          insight.significance >= this.config.significanceThreshold
        )
        .sort((a, b) => (b.significance * b.confidence) - (a.significance * a.confidence))
        .slice(0, this.config.maxInsightsPerCategory);
    }
    
    return validated;
  }

  prioritizeRecommendations(recommendations) {
    const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
    
    return recommendations
      .filter(rec => rec.confidence >= this.config.confidenceThreshold * 0.9) // Higher threshold for recommendations
      .sort((a, b) => {
        // Primary sort: priority
        const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
        if (priorityDiff !== 0) return priorityDiff;
        
        // Secondary sort: confidence
        return b.confidence - a.confidence;
      });
  }

  generateInsightSummary(insights, recommendations) {
    const totalInsights = Object.values(insights).reduce((sum, cat) => sum + cat.length, 0);
    const highSignificanceInsights = Object.values(insights)
      .flat()
      .filter(insight => insight.significance >= 0.8).length;
    
    const criticalRecommendations = recommendations.filter(rec => rec.priority === 'critical').length;
    const highPriorityRecommendations = recommendations.filter(rec => rec.priority === 'high').length;
    
    return {
      totalInsights,
      highSignificanceInsights,
      totalRecommendations: recommendations.length,
      criticalRecommendations,
      highPriorityRecommendations,
      
      keyFindings: this.extractKeyFindings(insights),
      topRecommendations: recommendations.slice(0, 3),
      overallAssessment: this.generateOverallAssessment(insights, recommendations)
    };
  }

  extractKeyFindings(insights) {
    const keyFindings = [];
    
    for (const categoryInsights of Object.values(insights)) {
      const topInsight = categoryInsights[0];
      if (topInsight && topInsight.significance >= 0.8) {
        keyFindings.push({
          finding: topInsight.title,
          significance: topInsight.significance,
          category: topInsight.type
        });
      }
    }
    
    return keyFindings.slice(0, 5);
  }

  generateOverallAssessment(insights, recommendations) {
    const totalSignificance = Object.values(insights)
      .flat()
      .reduce((sum, insight) => sum + insight.significance, 0);
    
    const avgSignificance = totalSignificance / Math.max(Object.values(insights).flat().length, 1);
    const criticalIssues = recommendations.filter(rec => rec.priority === 'critical').length;
    
    if (avgSignificance >= 0.8 && criticalIssues === 0) {
      return {
        level: 'excellent',
        description: 'System shows excellent performance with clear, actionable insights',
        confidence: 0.9
      };
    } else if (avgSignificance >= 0.6 && criticalIssues <= 1) {
      return {
        level: 'good',
        description: 'System performs well with some optimization opportunities',
        confidence: 0.8
      };
    } else if (criticalIssues > 2) {
      return {
        level: 'needs_attention',
        description: 'System has critical issues requiring immediate attention',
        confidence: 0.9
      };
    } else {
      return {
        level: 'average',
        description: 'System shows average performance with room for improvement',
        confidence: 0.7
      };
    }
  }

  calculateConfidenceMetrics(insights) {
    const allInsights = Object.values(insights).flat();
    const confidences = allInsights.map(i => i.confidence);
    
    return {
      averageConfidence: confidences.reduce((sum, c) => sum + c, 0) / confidences.length,
      minConfidence: Math.min(...confidences),
      maxConfidence: Math.max(...confidences),
      highConfidenceCount: confidences.filter(c => c >= 0.8).length,
      totalInsights: allInsights.length
    };
  }

  calculateActionabilityScores(recommendations) {
    return {
      averageActionability: recommendations.reduce((sum, rec) => sum + (rec.confidence || 0.5), 0) / recommendations.length,
      immediateActions: recommendations.filter(rec => rec.timeframe === 'short_term').length,
      strategicActions: recommendations.filter(rec => rec.timeframe === 'long_term').length,
      highImpactActions: recommendations.filter(rec => rec.impact === 'high').length
    };
  }

  calculateAnalysisDepth(analysisData) {
    let depth = 0;
    if (analysisData.testResults) depth += 1;
    if (analysisData.ruleEffectiveness) depth += 1;
    if (analysisData.contextLearning) depth += 1;
    if (analysisData.failureAnalysis) depth += 1;
    if (analysisData.statisticalAnalysis) depth += 1;
    if (analysisData.historicalData && analysisData.historicalData.length > 0) depth += 1;
    
    return depth / 6; // Normalize to 0-1 scale
  }

  // Strategic analysis methods
  
  assessSystemMaturity(insights) {
    const allInsights = Object.values(insights).flat();
    const avgSignificance = allInsights.reduce((sum, i) => sum + i.significance, 0) / allInsights.length;
    const highConfidenceCount = allInsights.filter(i => i.confidence >= 0.8).length;
    
    if (avgSignificance >= 0.8 && highConfidenceCount >= allInsights.length * 0.7) {
      return {
        level: 'Mature',
        description: 'System demonstrates mature, predictable performance with high-confidence insights',
        recommendations: ['Focus on optimization and scaling', 'Implement automated monitoring', 'Plan production deployment']
      };
    } else if (avgSignificance >= 0.6) {
      return {
        level: 'Developing',
        description: 'System shows good progress with some areas needing development',
        recommendations: ['Continue targeted improvements', 'Enhance data collection', 'Strengthen weak areas']
      };
    } else {
      return {
        level: 'Early',
        description: 'System is in early stages requiring fundamental improvements',
        recommendations: ['Focus on core functionality', 'Improve data quality', 'Establish baselines']
      };
    }
  }

  analyzeInvestmentPriorities(insights, recommendations) {
    const priorities = [];
    
    // Count issues by category
    const categoryIssues = {};
    for (const rec of recommendations) {
      const category = this.categorizeRecommendation(rec);
      categoryIssues[category] = (categoryIssues[category] || 0) + 1;
    }
    
    // Prioritize based on issue count and impact
    for (const [category, count] of Object.entries(categoryIssues)) {
      priorities.push({
        area: category,
        issueCount: count,
        priority: count > 3 ? 'high' : count > 1 ? 'medium' : 'low',
        recommendedInvestment: this.estimateInvestmentLevel(category, count)
      });
    }
    
    return priorities.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  assessSystemRisks(insights) {
    const risks = [];
    const mitigations = [];
    
    // Check for performance risks
    const performanceInsights = insights.performance || [];
    const hasPerformanceConcerns = performanceInsights.some(i => i.type === 'performance_concern');
    
    if (hasPerformanceConcerns) {
      risks.push({
        type: 'performance_degradation',
        severity: 'medium',
        description: 'Performance metrics indicate potential degradation risk',
        probability: 0.6
      });
      mitigations.push('Implement performance monitoring and alerting');
    }
    
    // Check for consistency risks
    const qualityInsights = insights.quality || [];
    const hasConsistencyIssues = qualityInsights.some(i => i.type === 'quality_inconsistency');
    
    if (hasConsistencyIssues) {
      risks.push({
        type: 'quality_inconsistency',
        severity: 'medium',
        description: 'Quality inconsistency could affect user experience',
        probability: 0.7
      });
      mitigations.push('Establish quality gates and consistency checks');
    }
    
    return { risks, mitigations };
  }

  assessScalability(insights) {
    const efficiencyInsights = insights.efficiency || [];
    const hasHighEfficiency = efficiencyInsights.some(i => i.type.includes('efficiency'));
    
    if (hasHighEfficiency) {
      return {
        description: 'System shows good scalability potential with high efficiency metrics',
        currentCapacity: 'Good',
        recommendations: [
          'Plan for horizontal scaling',
          'Implement load balancing',
          'Monitor resource utilization'
        ]
      };
    } else {
      return {
        description: 'System may face scalability challenges due to efficiency concerns',
        currentCapacity: 'Limited',
        recommendations: [
          'Optimize performance before scaling',
          'Identify bottlenecks',
          'Implement caching strategies'
        ]
      };
    }
  }

  // Utility methods for estimation and categorization
  
  estimateImpact(insight) {
    if (insight.significance >= 0.8) return 'high';
    if (insight.significance >= 0.6) return 'medium';
    return 'low';
  }

  estimateEffort(insight) {
    // Simple heuristic based on insight type
    const highEffortTypes = ['systematic', 'fundamental', 'architecture'];
    const lowEffortTypes = ['configuration', 'parameter', 'threshold'];
    
    const isHighEffort = highEffortTypes.some(type => insight.type.includes(type));
    const isLowEffort = lowEffortTypes.some(type => insight.type.includes(type));
    
    if (isHighEffort) return 'high';
    if (isLowEffort) return 'low';
    return 'medium';
  }

  estimateTimeframe(insight) {
    if (insight.significance >= 0.8) return 'short_term';
    if (insight.significance >= 0.6) return 'medium_term';
    return 'long_term';
  }

  categorizeRecommendation(recommendation) {
    const title = recommendation.title.toLowerCase();
    
    if (title.includes('performance')) return 'performance';
    if (title.includes('rule') || title.includes('optimize')) return 'optimization';
    if (title.includes('quality')) return 'quality';
    if (title.includes('context')) return 'contextual';
    if (title.includes('failure')) return 'reliability';
    
    return 'general';
  }

  estimateInvestmentLevel(category, issueCount) {
    if (issueCount > 3) return 'significant';
    if (issueCount > 1) return 'moderate';
    return 'minimal';
  }
}

module.exports = InsightGenerationEngine;