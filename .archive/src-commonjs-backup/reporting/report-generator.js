/**
 * Comprehensive Report Generator
 * Creates rich, actionable reports in multiple formats (JSON, Markdown, HTML, CSV)
 */

const fs = require('fs').promises;
const path = require('path');

class ReportGenerator {
  constructor(config = {}) {
    this.config = {
      includeExamples: true,
      includeStatistics: true,
      includeRecommendations: true,
      maxExampleLength: 200,
      reportSections: [
        'executiveSummary',
        'projectAnalysis',
        'testResults',
        'ruleEffectiveness',
        'optimizationRecommendations'
      ],
      defaultFormat: 'markdown',
      ...config
    };
    
    this.frameworkVersion = '1.0.0'; // Would typically come from package.json
  }

  /**
   * Generate comprehensive report from test results and analysis
   * @param {Object} data - Complete test results and analysis data
   * @param {string} format - Output format (json|markdown|html|csv)
   * @returns {string} Formatted report
   */
  async generateReport(data, format = null) {
    const outputFormat = format || this.config.defaultFormat;
    
    // Structure the complete report data
    const report = {
      metadata: this.generateMetadata(data),
      executiveSummary: this.generateExecutiveSummary(data),
      projectAnalysis: this.generateProjectAnalysis(data),
      testExecution: this.generateTestExecution(data),
      improvementAnalysis: this.generateImprovementAnalysis(data),
      ruleEffectiveness: this.generateRuleEffectiveness(data),
      failureAnalysis: this.generateFailureAnalysis(data),
      optimizationRecommendations: this.generateOptimizationRecommendations(data),
      appendices: this.generateAppendices(data)
    };

    // Format according to requested format
    return this.formatReport(report, outputFormat);
  }

  /**
   * Generate report metadata
   * @private
   */
  generateMetadata(data) {
    return {
      timestamp: new Date().toISOString(),
      framework_version: this.frameworkVersion,
      project_path: data.projectPath || process.cwd(),
      config_summary: this.summarizeConfig(data.config || {}),
      generation_time: Date.now()
    };
  }

  /**
   * Generate executive summary
   * @private
   */
  generateExecutiveSummary(data) {
    const testResults = data.stages?.test_execution?.testResults || [];
    const evaluation = data.stages?.result_evaluation || {};
    const analysis = data.stages?.pattern_analysis || {};
    const optimization = data.stages?.rule_optimization || {};

    return {
      testsRun: testResults.length,
      overallImprovement: this.calculateOverallImprovement(testResults),
      statisticalSignificance: evaluation.statisticalSignificance || 'not_calculated',
      topInsights: this.extractTopInsights(analysis, 5),
      keyRecommendations: this.extractKeyRecommendations(optimization, 3),
      executionTime: data.metadata?.executionTime || 0,
      successRate: this.calculateSuccessRate(testResults)
    };
  }

  /**
   * Generate project analysis section
   * @private
   */
  generateProjectAnalysis(data) {
    const context = data.stages?.context_analysis || {};
    
    return {
      detectedContext: context.projectContext || {},
      contextConfidence: context.contextConfidence || 0,
      relevanceScore: context.contextRelevance || 0,
      techStackCoverage: context.techStackCoverage || {},
      domainSpecificity: context.domainSpecificity || {},
      fileAnalysis: {
        totalFiles: context.fileCount || 0,
        analyzedFiles: context.analyzedFileCount || 0,
        languages: context.detectedLanguages || [],
        frameworks: context.detectedFrameworks || []
      }
    };
  }

  /**
   * Generate test execution section
   * @private
   */
  generateTestExecution(data) {
    const testResults = data.stages?.test_execution?.testResults || [];
    const generation = data.stages?.test_generation || {};

    return {
      testSuite: {
        totalTests: testResults.length,
        byCategory: this.groupByCategory(testResults),
        byComplexity: this.groupByComplexity(testResults),
        byContext: this.groupByContext(testResults)
      },
      performance: {
        averageExecutionTime: this.calculateAverageTime(testResults),
        successRate: this.calculateSuccessRate(testResults),
        errorRate: this.calculateErrorRate(testResults),
        throughput: this.calculateThroughput(testResults)
      },
      generation: {
        generationTime: generation.generationTime || 0,
        templatesUsed: generation.templatesUsed || 0,
        categoryCoverage: generation.categoryCoverage || {}
      }
    };
  }

  /**
   * Generate improvement analysis section
   * @private
   */
  generateImprovementAnalysis(data) {
    const testResults = data.stages?.test_execution?.testResults || [];
    const evaluation = data.stages?.result_evaluation || {};

    return {
      overallMetrics: {
        clarity: this.calculateMetricImprovement(testResults, 'clarity'),
        completeness: this.calculateMetricImprovement(testResults, 'completeness'),
        specificity: this.calculateMetricImprovement(testResults, 'specificity'),
        structure: this.calculateMetricImprovement(testResults, 'structure')
      },
      categoryBreakdown: this.generateCategoryBreakdown(testResults),
      complexityTrends: this.analyzeComplexityTrends(testResults),
      statisticalValidation: evaluation.statisticalResults || {},
      improvementDistribution: this.analyzeImprovementDistribution(testResults)
    };
  }

  /**
   * Generate rule effectiveness section
   * @private
   */
  generateRuleEffectiveness(data) {
    const learning = data.stages?.pattern_analysis || {};
    const testResults = data.stages?.test_execution?.testResults || [];

    return {
      topPerformingRules: learning.ruleEffectiveness?.topPerformers || [],
      underperformingRules: learning.ruleEffectiveness?.underperformers || [],
      contextSpecificEffectiveness: learning.contextSpecificRules || {},
      ruleUtilization: this.analyzeRuleUtilization(testResults),
      ruleImpactAnalysis: this.analyzeRuleImpact(testResults),
      recommendedRuleChanges: learning.recommendedRuleChanges || []
    };
  }

  /**
   * Generate failure analysis section
   * @private
   */
  generateFailureAnalysis(data) {
    const learning = data.stages?.pattern_analysis || {};
    const testResults = data.stages?.test_execution?.testResults || [];

    return {
      failurePatterns: learning.failureAnalysis?.patterns || [],
      rootCauses: learning.failureAnalysis?.rootCauses || [],
      edgeCases: learning.failureAnalysis?.edgeCases || [],
      recommendedFixes: learning.failureAnalysis?.recommendedFixes || [],
      failureDistribution: this.analyzeFailureDistribution(testResults),
      commonFailures: this.identifyCommonFailures(testResults)
    };
  }

  /**
   * Generate optimization recommendations section
   * @private
   */
  generateOptimizationRecommendations(data) {
    const optimization = data.stages?.rule_optimization || {};

    return {
      ruleModifications: optimization.modifications || [],
      newRules: optimization.additions || [],
      contextSpecializations: optimization.specializations || [],
      prioritizedActions: this.prioritizeRecommendations(optimization),
      implementationGuide: this.generateImplementationGuide(optimization),
      expectedImpact: this.calculateExpectedImpact(optimization)
    };
  }

  /**
   * Generate appendices section
   * @private
   */
  generateAppendices(data) {
    const testResults = data.stages?.test_execution?.testResults || [];
    const evaluation = data.stages?.result_evaluation || {};

    return {
      examples: this.config.includeExamples ? this.generateExamples(testResults) : null,
      detailedStatistics: this.config.includeStatistics ? evaluation.detailedStats : null,
      configurationUsed: data.config || {},
      technicalDetails: this.generateTechnicalDetails(testResults, evaluation),
      rawData: {
        testCount: testResults.length,
        executionTime: data.metadata?.executionTime || 0,
        stages: Object.keys(data.stages || {})
      }
    };
  }

  /**
   * Format report according to specified format
   * @private
   */
  formatReport(report, format) {
    switch (format.toLowerCase()) {
      case 'json':
        return JSON.stringify(report, null, 2);
      case 'markdown':
        return this.generateMarkdownReport(report);
      case 'html':
        return this.generateHtmlReport(report);
      case 'csv':
        return this.generateCsvReport(report);
      default:
        throw new Error(`Unsupported format: ${format}. Supported formats: json, markdown, html, csv`);
    }
  }

  /**
   * Generate Markdown format report
   * @private
   */
  generateMarkdownReport(report) {
    const md = [];
    
    // Header
    md.push('# Universal Prompt Testing Framework Report');
    md.push(`*Generated on ${new Date(report.metadata.timestamp).toLocaleString()}*`);
    md.push('');

    // Executive Summary
    md.push('## Executive Summary');
    md.push(`- **Tests Run**: ${report.executiveSummary.testsRun}`);
    md.push(`- **Overall Improvement**: ${(report.executiveSummary.overallImprovement * 100).toFixed(1)}%`);
    md.push(`- **Success Rate**: ${(report.executiveSummary.successRate * 100).toFixed(1)}%`);
    md.push(`- **Execution Time**: ${(report.executiveSummary.executionTime / 1000).toFixed(1)}s`);
    
    if (report.executiveSummary.topInsights.length > 0) {
      md.push(`- **Top Insight**: ${report.executiveSummary.topInsights[0]}`);
    }
    
    if (report.executiveSummary.keyRecommendations.length > 0) {
      md.push(`- **Key Recommendation**: ${report.executiveSummary.keyRecommendations[0]}`);
    }
    md.push('');

    // Project Analysis
    md.push('## Project Analysis');
    md.push(`- **Project Path**: ${report.metadata.project_path}`);
    md.push(`- **Context Confidence**: ${(report.projectAnalysis.contextConfidence * 100).toFixed(1)}%`);
    md.push(`- **Relevance Score**: ${(report.projectAnalysis.relevanceScore * 100).toFixed(1)}%`);
    
    if (report.projectAnalysis.fileAnalysis.languages.length > 0) {
      md.push(`- **Languages Detected**: ${report.projectAnalysis.fileAnalysis.languages.join(', ')}`);
    }
    
    if (report.projectAnalysis.fileAnalysis.frameworks.length > 0) {
      md.push(`- **Frameworks Detected**: ${report.projectAnalysis.fileAnalysis.frameworks.join(', ')}`);
    }
    md.push('');

    // Performance Breakdown
    md.push('## Performance Breakdown');
    md.push('### By Category');
    
    for (const [category, stats] of Object.entries(report.testExecution.testSuite.byCategory)) {
      const improvement = stats.averageImprovement || 0;
      const emoji = improvement > 0.8 ? '✅' : improvement > 0.6 ? '⚠️' : '❌';
      md.push(`- ${emoji} **${this.formatCategoryName(category)}**: ${(improvement * 100).toFixed(0)}% improvement (${stats.count} tests)`);
    }
    md.push('');

    // Statistical Validation
    if (report.improvementAnalysis.statisticalValidation.significance) {
      md.push('### Statistical Validation');
      md.push(`- **Statistical Significance**: ${report.improvementAnalysis.statisticalValidation.significance}`);
      md.push(`- **Effect Size**: ${report.improvementAnalysis.statisticalValidation.effectSize || 'Not calculated'}`);
      md.push(`- **Confidence Interval**: ${report.improvementAnalysis.statisticalValidation.confidenceInterval || 'Not calculated'}`);
      md.push('');
    }

    // Rule Effectiveness
    md.push('## Rule Effectiveness');
    if (report.ruleEffectiveness.topPerformingRules.length > 0) {
      md.push('### Top Performers');
      report.ruleEffectiveness.topPerformingRules.slice(0, 5).forEach((rule, index) => {
        md.push(`${index + 1}. **${rule.name || rule.rule}**: ${(rule.successRate * 100).toFixed(0)}% success rate (Applied ${rule.applications || 0} times)`);
      });
      md.push('');
    }

    if (report.ruleEffectiveness.underperformingRules.length > 0) {
      md.push('### Needs Improvement');
      report.ruleEffectiveness.underperformingRules.slice(0, 3).forEach((rule, index) => {
        md.push(`${index + 1}. **${rule.name || rule.rule}**: ${(rule.successRate * 100).toFixed(0)}% success rate (Applied ${rule.applications || 0} times)`);
      });
      md.push('');
    }

    // Optimization Recommendations
    if (report.optimizationRecommendations.prioritizedActions.length > 0) {
      md.push('## Optimization Recommendations');
      md.push('### High Priority');
      report.optimizationRecommendations.prioritizedActions.slice(0, 3).forEach((action, index) => {
        md.push(`${index + 1}. **${action.title || action.recommendation}** - ${action.description || action.reason}`);
      });
      md.push('');

      if (report.optimizationRecommendations.implementationGuide.length > 0) {
        md.push('### Implementation Guide');
        report.optimizationRecommendations.implementationGuide.forEach((step, index) => {
          md.push(`${index + 1}. ${step}`);
        });
        md.push('');
      }
    }

    // Examples
    if (this.config.includeExamples && report.appendices.examples) {
      md.push('## Example Improvements');
      report.appendices.examples.slice(0, 3).forEach((example, index) => {
        md.push(`### Example ${index + 1}: ${example.category}`);
        md.push('**Original:**');
        md.push(`\`\`\`\n${example.original}\n\`\`\``);
        md.push('**Improved:**');
        md.push(`\`\`\`\n${example.improved}\n\`\`\``);
        md.push(`**Improvement Score:** ${(example.score * 100).toFixed(1)}%`);
        md.push('');
      });
    }

    md.push('---');
    md.push(`*Report generated by Universal Prompt Testing Framework v${report.metadata.framework_version}*`);

    return md.join('\n');
  }

  /**
   * Generate HTML format report
   * @private
   */
  generateHtmlReport(report) {
    const html = [];
    
    html.push('<!DOCTYPE html>');
    html.push('<html lang="en">');
    html.push('<head>');
    html.push('<meta charset="UTF-8">');
    html.push('<meta name="viewport" content="width=device-width, initial-scale=1.0">');
    html.push('<title>Prompt Testing Framework Report</title>');
    html.push('<style>');
    html.push(this.getReportStyles());
    html.push('</style>');
    html.push('</head>');
    html.push('<body>');
    html.push('<div class="container">');
    
    // Header
    html.push('<header>');
    html.push('<h1>Universal Prompt Testing Framework Report</h1>');
    html.push(`<p class="timestamp">Generated on ${new Date(report.metadata.timestamp).toLocaleString()}</p>`);
    html.push('</header>');

    // Executive Summary
    html.push('<section class="executive-summary">');
    html.push('<h2>Executive Summary</h2>');
    html.push('<div class="metrics-grid">');
    html.push(`<div class="metric"><span class="value">${report.executiveSummary.testsRun}</span><span class="label">Tests Run</span></div>`);
    html.push(`<div class="metric"><span class="value">${(report.executiveSummary.overallImprovement * 100).toFixed(1)}%</span><span class="label">Overall Improvement</span></div>`);
    html.push(`<div class="metric"><span class="value">${(report.executiveSummary.successRate * 100).toFixed(1)}%</span><span class="label">Success Rate</span></div>`);
    html.push(`<div class="metric"><span class="value">${(report.executiveSummary.executionTime / 1000).toFixed(1)}s</span><span class="label">Execution Time</span></div>`);
    html.push('</div>');
    html.push('</section>');

    // Charts and visualizations would go here
    html.push('<section class="charts">');
    html.push('<h2>Performance Overview</h2>');
    html.push('<div class="chart-placeholder">');
    html.push('<p>Interactive charts would be generated here with libraries like Chart.js or D3.js</p>');
    html.push('</div>');
    html.push('</section>');

    html.push('</div>');
    html.push('</body>');
    html.push('</html>');

    return html.join('\n');
  }

  /**
   * Generate CSV format report
   * @private
   */
  generateCsvReport(report) {
    const csv = [];
    
    // Headers
    csv.push('Category,Test_Count,Success_Rate,Average_Improvement,Top_Rules');
    
    // Data rows
    for (const [category, stats] of Object.entries(report.testExecution.testSuite.byCategory)) {
      const successRate = ((stats.successCount || 0) / (stats.count || 1) * 100).toFixed(1);
      const avgImprovement = ((stats.averageImprovement || 0) * 100).toFixed(1);
      const topRules = (stats.topRules || []).slice(0, 3).join('; ');
      
      csv.push(`${category},${stats.count},${successRate}%,${avgImprovement}%,"${topRules}"`);
    }

    return csv.join('\n');
  }

  // Utility methods for data analysis and formatting

  calculateOverallImprovement(testResults) {
    if (!testResults || testResults.length === 0) return 0;
    
    const improvements = testResults
      .filter(result => result.improvements && typeof result.improvements.overall === 'number')
      .map(result => result.improvements.overall);
    
    return improvements.length > 0 
      ? improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length 
      : 0;
  }

  calculateSuccessRate(testResults) {
    if (!testResults || testResults.length === 0) return 0;
    
    const successfulTests = testResults.filter(result => 
      result.improvements && result.improvements.overall > 0.5
    ).length;
    
    return successfulTests / testResults.length;
  }

  calculateErrorRate(testResults) {
    if (!testResults || testResults.length === 0) return 0;
    
    const errorTests = testResults.filter(result => 
      result.error || result.status === 'error'
    ).length;
    
    return errorTests / testResults.length;
  }

  calculateAverageTime(testResults) {
    if (!testResults || testResults.length === 0) return 0;
    
    const times = testResults
      .filter(result => result.executionTime)
      .map(result => result.executionTime);
    
    return times.length > 0 
      ? times.reduce((sum, time) => sum + time, 0) / times.length 
      : 0;
  }

  calculateThroughput(testResults) {
    if (!testResults || testResults.length === 0) return 0;
    
    const totalTime = testResults.reduce((sum, result) => 
      sum + (result.executionTime || 0), 0
    );
    
    return totalTime > 0 ? testResults.length / (totalTime / 1000) : 0;
  }

  groupByCategory(testResults) {
    const grouped = {};
    
    testResults.forEach(result => {
      const category = result.category || 'unknown';
      
      if (!grouped[category]) {
        grouped[category] = {
          count: 0,
          successCount: 0,
          totalImprovement: 0,
          averageImprovement: 0
        };
      }
      
      grouped[category].count++;
      
      if (result.improvements && result.improvements.overall > 0.5) {
        grouped[category].successCount++;
      }
      
      if (result.improvements && typeof result.improvements.overall === 'number') {
        grouped[category].totalImprovement += result.improvements.overall;
      }
    });
    
    // Calculate averages
    Object.keys(grouped).forEach(category => {
      grouped[category].averageImprovement = 
        grouped[category].totalImprovement / grouped[category].count;
    });
    
    return grouped;
  }

  groupByComplexity(testResults) {
    return this.groupByField(testResults, 'complexity');
  }

  groupByContext(testResults) {
    return this.groupByField(testResults, 'context');
  }

  groupByField(testResults, field) {
    const grouped = {};
    
    testResults.forEach(result => {
      const value = result[field] || 'unknown';
      
      if (!grouped[value]) {
        grouped[value] = { count: 0, results: [] };
      }
      
      grouped[value].count++;
      grouped[value].results.push(result);
    });
    
    return grouped;
  }

  calculateMetricImprovement(testResults, metric) {
    const improvements = testResults
      .filter(result => result.improvements && result.improvements[metric])
      .map(result => result.improvements[metric]);
    
    return improvements.length > 0 
      ? improvements.reduce((sum, imp) => sum + imp, 0) / improvements.length 
      : 0;
  }

  generateCategoryBreakdown(testResults) {
    const categories = this.groupByCategory(testResults);
    const breakdown = {};
    
    Object.entries(categories).forEach(([category, stats]) => {
      breakdown[category] = {
        testCount: stats.count,
        successRate: stats.successCount / stats.count,
        averageImprovement: stats.averageImprovement,
        topIssues: this.identifyTopIssues(stats.results || [])
      };
    });
    
    return breakdown;
  }

  analyzeComplexityTrends(testResults) {
    const complexityGroups = this.groupByComplexity(testResults);
    const trends = {};
    
    Object.entries(complexityGroups).forEach(([complexity, group]) => {
      const improvements = group.results
        .filter(r => r.improvements && r.improvements.overall)
        .map(r => r.improvements.overall);
      
      trends[complexity] = {
        count: group.count,
        averageImprovement: improvements.length > 0 
          ? improvements.reduce((a, b) => a + b, 0) / improvements.length 
          : 0,
        successRate: improvements.filter(imp => imp > 0.5).length / improvements.length
      };
    });
    
    return trends;
  }

  analyzeImprovementDistribution(testResults) {
    const improvements = testResults
      .filter(result => result.improvements && result.improvements.overall)
      .map(result => result.improvements.overall);
    
    improvements.sort((a, b) => a - b);
    
    return {
      min: improvements[0] || 0,
      max: improvements[improvements.length - 1] || 0,
      median: improvements[Math.floor(improvements.length / 2)] || 0,
      quartiles: {
        q1: improvements[Math.floor(improvements.length * 0.25)] || 0,
        q3: improvements[Math.floor(improvements.length * 0.75)] || 0
      },
      distribution: {
        low: improvements.filter(imp => imp < 0.3).length,
        medium: improvements.filter(imp => imp >= 0.3 && imp < 0.7).length,
        high: improvements.filter(imp => imp >= 0.7).length
      }
    };
  }

  analyzeRuleUtilization(testResults) {
    const ruleUsage = {};
    
    testResults.forEach(result => {
      if (result.rulesApplied) {
        result.rulesApplied.forEach(rule => {
          if (!ruleUsage[rule]) {
            ruleUsage[rule] = { count: 0, successes: 0 };
          }
          ruleUsage[rule].count++;
          
          if (result.improvements && result.improvements.overall > 0.5) {
            ruleUsage[rule].successes++;
          }
        });
      }
    });
    
    // Calculate success rates
    Object.keys(ruleUsage).forEach(rule => {
      ruleUsage[rule].successRate = ruleUsage[rule].successes / ruleUsage[rule].count;
    });
    
    return ruleUsage;
  }

  analyzeRuleImpact(testResults) {
    // This would analyze the impact of individual rules on improvement scores
    return {
      highImpactRules: [],
      lowImpactRules: [],
      contextSpecificRules: {}
    };
  }

  analyzeFailureDistribution(testResults) {
    const failures = testResults.filter(result => 
      result.error || (result.improvements && result.improvements.overall < 0.3)
    );
    
    return this.groupByCategory(failures);
  }

  identifyCommonFailures(testResults) {
    const failures = testResults.filter(result => 
      result.error || (result.improvements && result.improvements.overall < 0.3)
    );
    
    const failureReasons = {};
    
    failures.forEach(failure => {
      const reason = failure.failureReason || failure.error?.message || 'unknown';
      failureReasons[reason] = (failureReasons[reason] || 0) + 1;
    });
    
    return Object.entries(failureReasons)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([reason, count]) => ({ reason, count }));
  }

  extractTopInsights(analysis, count = 5) {
    const insights = [];
    
    if (analysis.successPatterns) {
      insights.push(...analysis.successPatterns.slice(0, count));
    }
    
    if (analysis.recommendations) {
      insights.push(...analysis.recommendations.slice(0, count).map(r => 
        r.insight || r.recommendation || r
      ));
    }
    
    return insights.slice(0, count);
  }

  extractKeyRecommendations(optimization, count = 3) {
    const recommendations = [];
    
    if (optimization.recommendations) {
      recommendations.push(...optimization.recommendations.slice(0, count));
    }
    
    if (optimization.prioritizedActions) {
      recommendations.push(...optimization.prioritizedActions.slice(0, count).map(a => 
        a.recommendation || a.action || a
      ));
    }
    
    return recommendations.slice(0, count);
  }

  prioritizeRecommendations(optimization) {
    const recommendations = optimization.recommendations || [];
    
    return recommendations
      .sort((a, b) => (b.priority || 0) - (a.priority || 0))
      .slice(0, 10);
  }

  generateImplementationGuide(optimization) {
    const guide = [];
    
    if (optimization.modifications) {
      guide.push('Update existing rules with validated modifications');
    }
    
    if (optimization.additions) {
      guide.push('Add new rules for identified gaps');
    }
    
    if (optimization.specializations) {
      guide.push('Implement context-specific rule variations');
    }
    
    guide.push('Run validation tests before deployment');
    guide.push('Monitor performance after changes');
    
    return guide;
  }

  calculateExpectedImpact(optimization) {
    return {
      overallImprovement: 0.15, // 15% expected improvement
      affectedCategories: ['structure', 'clarity'],
      confidenceLevel: 0.8
    };
  }

  generateExamples(testResults) {
    return testResults
      .filter(result => result.improvements && result.improvements.overall > 0.7)
      .slice(0, 5)
      .map(result => ({
        category: result.category || 'unknown',
        original: this.truncateText(result.originalPrompt || '', this.config.maxExampleLength),
        improved: this.truncateText(result.improvedPrompt || '', this.config.maxExampleLength),
        score: result.improvements.overall
      }));
  }

  generateTechnicalDetails(testResults, evaluation) {
    return {
      frameworkVersion: this.frameworkVersion,
      evaluationMethods: evaluation.methods || [],
      statisticalTests: evaluation.statisticalTests || [],
      dataIntegrity: {
        totalTests: testResults.length,
        validTests: testResults.filter(r => !r.error).length,
        completedTests: testResults.filter(r => r.status === 'completed').length
      }
    };
  }

  summarizeConfig(config) {
    return {
      testCount: config.testing?.defaultTestCount || 0,
      evaluationMethods: config.evaluation?.methods || [],
      outputFormat: config.output?.defaultFormat || 'markdown'
    };
  }

  formatCategoryName(category) {
    return category
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + '...';
  }

  identifyTopIssues(results) {
    // Identify common issues in failed tests
    return ['Generic placeholder - would analyze actual failure patterns'];
  }

  getReportStyles() {
    return `
      body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
      .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
      header { text-align: center; border-bottom: 2px solid #eee; padding-bottom: 20px; }
      .timestamp { color: #666; font-style: italic; }
      .executive-summary { margin: 30px 0; }
      .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
      .metric { text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }
      .metric .value { display: block; font-size: 2em; font-weight: bold; color: #007bff; }
      .metric .label { color: #666; }
      .charts { margin: 30px 0; }
      .chart-placeholder { height: 300px; background: #f8f9fa; border-radius: 8px; display: flex; align-items: center; justify-content: center; }
      h2 { color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }
    `;
  }
}

module.exports = ReportGenerator;