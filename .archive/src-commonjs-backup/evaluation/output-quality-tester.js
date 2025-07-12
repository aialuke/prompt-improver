/**
 * Output Quality Testing System
 * Empirical effectiveness measurement through actual output evaluation
 */

const Logger = require('../utils/logger');

class OutputQualityTester {
  constructor(config = {}) {
    this.config = {
      // Testing strategies
      strategies: {
        simulation: 'Simulate responses based on prompt characteristics',
        benchmark: 'Compare against known good/bad examples',
        criteria_based: 'Evaluate against specific success criteria',
        comparative: 'Compare outputs from different prompt versions'
      },

      // Quality metrics
      qualityMetrics: {
        accuracy: {
          weight: 0.3,
          description: 'How accurately the output addresses the prompt requirements'
        },
        completeness: {
          weight: 0.25,
          description: 'How complete the output is relative to requirements'
        },
        relevance: {
          weight: 0.2,
          description: 'How relevant the output is to the stated goal'
        },
        usability: {
          weight: 0.15,
          description: 'How usable the output is for the intended purpose'
        },
        efficiency: {
          weight: 0.1,
          description: 'How efficiently the output can be produced and used'
        }
      },

      // Simulation parameters
      simulation: {
        sampleSize: 10,
        responseVariability: 0.2, // How much responses vary
        contextSensitivity: 0.3,  // How much context affects responses
        complexityFactor: 0.25    // How complexity affects quality
      },

      // Benchmark thresholds
      thresholds: {
        excellent: 0.85,
        good: 0.7,
        acceptable: 0.55,
        poor: 0.4
      },

      ...config
    };

    this.logger = new Logger('OutputQualityTester');
    this.testCache = new Map();
    this.benchmarkLibrary = this.initializeBenchmarkLibrary();
  }

  /**
   * Test output quality for a prompt
   * @param {string} prompt - Prompt to test
   * @param {Object} context - Context for testing
   * @param {Object} options - Testing options
   * @returns {Promise<Object>} Quality test results
   */
  async testOutputQuality(prompt, context = {}, options = {}) {
    const startTime = Date.now();
    this.logger.info('Starting output quality testing', {
      promptLength: prompt.length,
      strategy: options.strategy || 'simulation'
    });

    try {
      // Validate inputs
      this.validateInputs(prompt, context, options);

      // Setup testing parameters
      const testParams = this.setupTestParameters(context, options);

      // Generate cache key
      const cacheKey = this.generateCacheKey(prompt, testParams);

      // Check cache
      if (this.testCache.has(cacheKey) && !options.bypassCache) {
        this.logger.info('Returning cached quality test result');
        return this.testCache.get(cacheKey);
      }

      // Perform quality testing based on strategy
      let qualityTest;
      switch (testParams.strategy) {
        case 'benchmark':
          qualityTest = await this.performBenchmarkTesting(prompt, testParams);
          break;
        case 'criteria_based':
          qualityTest = await this.performCriteriaBasedTesting(prompt, testParams);
          break;
        case 'comparative':
          qualityTest = await this.performComparativeTesting(prompt, testParams);
          break;
        default:
          qualityTest = await this.performSimulationTesting(prompt, testParams);
      }

      // Post-process results
      qualityTest = await this.postProcessQualityTest(qualityTest, prompt, testParams);

      // Cache results
      this.testCache.set(cacheKey, qualityTest);

      const testTime = Date.now() - startTime;
      this.logger.info('Output quality testing completed', {
        overallQuality: qualityTest.overallQuality,
        testTime,
        strategy: testParams.strategy
      });

      return qualityTest;

    } catch (error) {
      this.logger.error('Output quality testing failed', error);
      throw new Error(`Quality testing failed: ${error.message}`);
    }
  }

  /**
   * Perform simulation-based testing
   * @param {string} prompt - Prompt to test
   * @param {Object} params - Test parameters
   * @returns {Promise<Object>} Simulation test results
   */
  async performSimulationTesting(prompt, params) {
    const simulations = [];
    
    // Generate multiple simulated outputs
    for (let i = 0; i < params.sampleSize; i++) {
      const simulation = await this.simulateOutput(prompt, params, i);
      simulations.push(simulation);
    }

    // Evaluate each simulation
    const evaluations = simulations.map(simulation => 
      this.evaluateSimulatedOutput(simulation, prompt, params)
    );

    // Aggregate results
    return this.aggregateSimulationResults(evaluations, simulations, params);
  }

  /**
   * Simulate output based on prompt characteristics
   * @param {string} prompt - Input prompt
   * @param {Object} params - Test parameters
   * @param {number} iteration - Simulation iteration
   * @returns {Object} Simulated output
   */
  async simulateOutput(prompt, params, iteration) {
    // Analyze prompt characteristics
    const characteristics = this.analyzePromptCharacteristics(prompt);
    
    // Simulate response based on characteristics
    const simulation = {
      id: `sim_${iteration + 1}`,
      prompt: prompt,
      characteristics: characteristics,
      
      // Simulated output properties
      outputLength: this.simulateOutputLength(characteristics, params),
      outputStructure: this.simulateOutputStructure(characteristics, params),
      outputQuality: this.simulateOutputQuality(characteristics, params),
      outputRelevance: this.simulateOutputRelevance(characteristics, params),
      outputCompleteness: this.simulateOutputCompleteness(characteristics, params),
      
      // Metadata
      simulationParams: {
        iteration: iteration,
        variability: params.simulation.responseVariability,
        timestamp: new Date().toISOString()
      }
    };

    // Add simulated content based on prompt type
    simulation.simulatedContent = this.generateSimulatedContent(characteristics, simulation);

    return simulation;
  }

  /**
   * Analyze prompt characteristics for simulation
   * @param {string} prompt - Input prompt
   * @returns {Object} Prompt characteristics
   */
  analyzePromptCharacteristics(prompt) {
    const words = prompt.toLowerCase().split(/\s+/);
    const sentences = prompt.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    const characteristics = {
      // Basic metrics
      wordCount: words.length,
      sentenceCount: sentences.length,
      complexity: this.assessComplexity(prompt),
      
      // Content type indicators
      isCodeRequest: /\b(code|program|script|function|class|method)\b/i.test(prompt),
      isAnalysisRequest: /\b(analyze|review|evaluate|assess|examine)\b/i.test(prompt),
      isCreationRequest: /\b(create|build|make|design|develop)\b/i.test(prompt),
      isExplanationRequest: /\b(explain|describe|how|why|what)\b/i.test(prompt),
      
      // Specificity indicators
      hasExamples: /\b(example|such as|like|including|for instance)\b/i.test(prompt),
      hasConstraints: /\b(must|should|cannot|require|ensure|without)\b/i.test(prompt),
      hasFormat: /\b(format|output|return|json|xml|csv|list|table)\b/i.test(prompt),
      hasContext: /\b(using|with|for|in|given|considering)\b/i.test(prompt),
      
      // Quality indicators
      hasSpecificTerms: this.countSpecificTerms(words),
      hasVagueTerms: this.countVagueTerms(words),
      hasActionVerbs: this.countActionVerbs(words),
      hasTechnicalTerms: this.countTechnicalTerms(words),
      
      // Structure indicators
      hasSteps: /\b(step|first|then|next|finally)\b/i.test(prompt),
      hasPriorities: /\b(important|priority|critical|essential)\b/i.test(prompt),
      hasMetrics: /\b(measure|metric|score|rate|percentage)\b/i.test(prompt)
    };

    // Calculate derived metrics
    characteristics.specificityRatio = characteristics.hasSpecificTerms / Math.max(characteristics.hasVagueTerms, 1);
    characteristics.clarityScore = this.calculateClarityScore(characteristics);
    characteristics.actionabilityScore = this.calculateActionabilityScore(characteristics);

    return characteristics;
  }

  /**
   * Assess complexity of prompt
   * @param {string} prompt - Input prompt
   * @returns {string} Complexity level
   */
  assessComplexity(prompt) {
    const words = prompt.split(/\s+/).length;
    const complexTerms = (prompt.match(/\b(architecture|system|distributed|scalable|optimize|integrate)\b/gi) || []).length;
    const hasMultipleTasks = prompt.split(/[,;]/).length > 2;
    
    let complexityScore = 0;
    if (words > 50) complexityScore += 2;
    else if (words > 20) complexityScore += 1;
    
    complexityScore += complexTerms;
    if (hasMultipleTasks) complexityScore += 2;
    
    if (complexityScore >= 5) return 'complex';
    if (complexityScore >= 3) return 'moderate';
    return 'simple';
  }

  /**
   * Simulate output length based on characteristics
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} params - Test parameters
   * @returns {number} Simulated output length in words
   */
  simulateOutputLength(characteristics, params) {
    let baseLength = 100; // Default response length
    
    // Adjust based on request type
    if (characteristics.isCodeRequest) baseLength = 50;
    else if (characteristics.isAnalysisRequest) baseLength = 200;
    else if (characteristics.isCreationRequest) baseLength = 150;
    else if (characteristics.isExplanationRequest) baseLength = 175;
    
    // Adjust based on complexity
    const complexityMultiplier = {
      simple: 0.7,
      moderate: 1.0,
      complex: 1.5
    };
    baseLength *= complexityMultiplier[characteristics.complexity];
    
    // Adjust based on specificity
    if (characteristics.hasExamples) baseLength *= 1.2;
    if (characteristics.hasConstraints) baseLength *= 1.1;
    if (characteristics.hasFormat) baseLength *= 0.9; // Structured outputs often more concise
    
    // Add variability
    const variability = params.simulation.responseVariability;
    const variation = 1 + (Math.random() - 0.5) * variability * 2;
    
    return Math.round(baseLength * variation);
  }

  /**
   * Simulate output structure
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} params - Test parameters
   * @returns {Object} Simulated structure
   */
  simulateOutputStructure(characteristics, params) {
    const structure = {
      type: 'text',
      organization: 'paragraph',
      hasIntroduction: false,
      hasConclusion: false,
      hasSections: false,
      hasExamples: false,
      hasCode: false,
      hasList: false
    };

    // Determine structure based on request type
    if (characteristics.isCodeRequest) {
      structure.type = 'code';
      structure.hasCode = true;
      structure.organization = 'structured';
    } else if (characteristics.isAnalysisRequest) {
      structure.organization = 'analytical';
      structure.hasIntroduction = true;
      structure.hasConclusion = true;
      structure.hasSections = true;
    } else if (characteristics.hasSteps) {
      structure.organization = 'sequential';
      structure.hasList = true;
    }

    // Adjust based on format requirements
    if (characteristics.hasFormat) {
      structure.organization = 'structured';
      if (/json|xml/i.test(params.context.formatHint || '')) {
        structure.type = 'structured_data';
      }
    }

    // Add examples if prompt suggests them
    if (characteristics.hasExamples || characteristics.isExplanationRequest) {
      structure.hasExamples = Math.random() > 0.3; // 70% chance
    }

    return structure;
  }

  /**
   * Simulate output quality metrics
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} params - Test parameters
   * @returns {Object} Quality simulation
   */
  simulateOutputQuality(characteristics, params) {
    // Base quality from prompt clarity
    let baseQuality = characteristics.clarityScore;
    
    // Adjust for specificity
    const specificityBonus = Math.min(characteristics.specificityRatio * 0.1, 0.2);
    baseQuality += specificityBonus;
    
    // Adjust for actionability
    baseQuality += characteristics.actionabilityScore * 0.1;
    
    // Complexity penalty/bonus
    const complexityEffect = {
      simple: 0.1,    // Simple prompts often get better responses
      moderate: 0.0,  // No effect
      complex: -0.05  // Complex prompts may suffer slightly
    };
    baseQuality += complexityEffect[characteristics.complexity];
    
    // Context bonus
    if (characteristics.hasContext) baseQuality += 0.05;
    if (characteristics.hasConstraints) baseQuality += 0.05;
    if (characteristics.hasFormat) baseQuality += 0.1;
    
    // Add variability
    const variability = params.simulation.responseVariability;
    const variation = (Math.random() - 0.5) * variability;
    
    const finalQuality = Math.max(0.1, Math.min(1.0, baseQuality + variation));
    
    return {
      overall: finalQuality,
      accuracy: finalQuality + (Math.random() - 0.5) * 0.1,
      completeness: finalQuality + (Math.random() - 0.5) * 0.1,
      relevance: finalQuality + (Math.random() - 0.5) * 0.1,
      usability: finalQuality + (Math.random() - 0.5) * 0.1,
      efficiency: finalQuality + (Math.random() - 0.5) * 0.1
    };
  }

  /**
   * Simulate output relevance
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} params - Test parameters
   * @returns {number} Relevance score (0-1)
   */
  simulateOutputRelevance(characteristics, params) {
    let relevance = 0.7; // Base relevance
    
    // Higher specificity = higher relevance
    if (characteristics.specificityRatio > 2) relevance += 0.2;
    else if (characteristics.specificityRatio < 0.5) relevance -= 0.2;
    
    // Technical terms help relevance
    if (characteristics.hasTechnicalTerms > 3) relevance += 0.1;
    
    // Context helps relevance
    if (characteristics.hasContext) relevance += 0.1;
    
    // Add variability
    const variability = params.simulation.responseVariability;
    relevance += (Math.random() - 0.5) * variability;
    
    return Math.max(0.1, Math.min(1.0, relevance));
  }

  /**
   * Simulate output completeness
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} params - Test parameters
   * @returns {number} Completeness score (0-1)
   */
  simulateOutputCompleteness(characteristics, params) {
    let completeness = 0.6; // Base completeness
    
    // Well-structured prompts get more complete responses
    if (characteristics.hasSteps) completeness += 0.15;
    if (characteristics.hasConstraints) completeness += 0.1;
    if (characteristics.hasFormat) completeness += 0.1;
    
    // Complexity can hurt completeness
    if (characteristics.complexity === 'complex') completeness -= 0.05;
    
    // Examples help completeness
    if (characteristics.hasExamples) completeness += 0.1;
    
    // Add variability
    const variability = params.simulation.responseVariability;
    completeness += (Math.random() - 0.5) * variability;
    
    return Math.max(0.1, Math.min(1.0, completeness));
  }

  /**
   * Generate simulated content snippet
   * @param {Object} characteristics - Prompt characteristics
   * @param {Object} simulation - Simulation data
   * @returns {string} Simulated content
   */
  generateSimulatedContent(characteristics, simulation) {
    const contentTemplates = {
      code: [
        'function exampleFunction() {\n  // Implementation here\n  return result;\n}',
        'class ExampleClass {\n  constructor() {\n    // Initialize\n  }\n}',
        'const result = data.filter(item => item.active).map(item => item.value);'
      ],
      
      analysis: [
        'Analysis reveals several key findings: 1) Primary factor is X, 2) Secondary consideration is Y, 3) Recommendation is Z.',
        'The evaluation shows strong performance in areas A and B, with improvement needed in area C.',
        'Based on the data, the optimal approach would be to implement strategy X while considering constraints Y and Z.'
      ],
      
      creation: [
        'Here is a comprehensive solution that addresses the requirements: [detailed implementation]',
        'The proposed design includes the following components: Component A for X, Component B for Y, Component C for Z.',
        'Implementation plan: Phase 1 - Foundation, Phase 2 - Core features, Phase 3 - Optimization.'
      ],
      
      explanation: [
        'This concept works by utilizing principle X to achieve outcome Y through process Z.',
        'The key factors are: 1) Factor A which controls B, 2) Factor C which influences D, 3) Factor E which determines F.',
        'Step-by-step breakdown: First X happens, then Y occurs, finally Z is achieved.'
      ]
    };

    let contentType = 'explanation'; // Default
    if (characteristics.isCodeRequest) contentType = 'code';
    else if (characteristics.isAnalysisRequest) contentType = 'analysis';
    else if (characteristics.isCreationRequest) contentType = 'creation';

    const templates = contentTemplates[contentType];
    const selectedTemplate = templates[Math.floor(Math.random() * templates.length)];
    
    // Adjust content quality based on simulation quality
    if (simulation.outputQuality.overall < 0.5) {
      return selectedTemplate.replace(/comprehensive|detailed|optimal/gi, 'basic');
    } else if (simulation.outputQuality.overall > 0.8) {
      return selectedTemplate + '\n\nAdditional considerations: [extra detail based on high quality]';
    }
    
    return selectedTemplate;
  }

  /**
   * Evaluate simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {Object} Evaluation results
   */
  evaluateSimulatedOutput(simulation, prompt, params) {
    const metrics = this.config.qualityMetrics;
    const evaluation = {
      simulationId: simulation.id,
      
      // Core quality metrics
      accuracy: this.evaluateAccuracy(simulation, prompt, params),
      completeness: this.evaluateCompleteness(simulation, prompt, params),
      relevance: this.evaluateRelevance(simulation, prompt, params),
      usability: this.evaluateUsability(simulation, prompt, params),
      efficiency: this.evaluateEfficiency(simulation, prompt, params),
      
      // Metadata
      evaluatedAt: new Date().toISOString(),
      outputLength: simulation.outputLength,
      expectedLength: this.estimateExpectedLength(prompt),
      
      // Issues and strengths
      identifiedIssues: [],
      identifiedStrengths: []
    };

    // Calculate weighted overall score
    evaluation.overallScore = (
      evaluation.accuracy * metrics.accuracy.weight +
      evaluation.completeness * metrics.completeness.weight +
      evaluation.relevance * metrics.relevance.weight +
      evaluation.usability * metrics.usability.weight +
      evaluation.efficiency * metrics.efficiency.weight
    );

    // Identify issues and strengths
    evaluation.identifiedIssues = this.identifySimulationIssues(evaluation, simulation);
    evaluation.identifiedStrengths = this.identifySimulationStrengths(evaluation, simulation);

    return evaluation;
  }

  /**
   * Evaluate accuracy of simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {number} Accuracy score (0-1)
   */
  evaluateAccuracy(simulation, prompt, params) {
    // Use the simulated quality as base accuracy
    let accuracy = simulation.outputQuality.accuracy;
    
    // Adjust based on prompt-output alignment
    const characteristics = simulation.characteristics;
    
    // Code requests should have code structure
    if (characteristics.isCodeRequest && simulation.outputStructure.hasCode) {
      accuracy += 0.1;
    } else if (characteristics.isCodeRequest && !simulation.outputStructure.hasCode) {
      accuracy -= 0.2;
    }
    
    // Analysis requests should have analytical structure
    if (characteristics.isAnalysisRequest && simulation.outputStructure.organization === 'analytical') {
      accuracy += 0.1;
    }
    
    // Format requirements compliance
    if (characteristics.hasFormat && simulation.outputStructure.organization === 'structured') {
      accuracy += 0.1;
    } else if (characteristics.hasFormat && simulation.outputStructure.organization !== 'structured') {
      accuracy -= 0.1;
    }
    
    return Math.max(0.0, Math.min(1.0, accuracy));
  }

  /**
   * Evaluate completeness of simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {number} Completeness score (0-1)
   */
  evaluateCompleteness(simulation, prompt, params) {
    let completeness = simulation.outputCompleteness;
    
    // Length appropriateness affects completeness
    const expectedLength = this.estimateExpectedLength(prompt);
    const lengthRatio = simulation.outputLength / expectedLength;
    
    if (lengthRatio < 0.5) {
      completeness -= 0.2; // Too short
    } else if (lengthRatio > 2.0) {
      completeness -= 0.1; // Too long
    } else if (lengthRatio >= 0.8 && lengthRatio <= 1.2) {
      completeness += 0.1; // Just right
    }
    
    // Structure completeness
    const characteristics = simulation.characteristics;
    if (characteristics.hasSteps && simulation.outputStructure.hasList) {
      completeness += 0.05;
    }
    
    if (characteristics.hasExamples && simulation.outputStructure.hasExamples) {
      completeness += 0.05;
    }
    
    return Math.max(0.0, Math.min(1.0, completeness));
  }

  /**
   * Evaluate relevance of simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {number} Relevance score (0-1)
   */
  evaluateRelevance(simulation, prompt, params) {
    let relevance = simulation.outputRelevance;
    
    // Context utilization affects relevance
    if (params.context && Object.keys(params.context).length > 0) {
      // If context is provided, simulated output should reflect it
      if (simulation.outputQuality.overall > 0.7) {
        relevance += 0.1; // High quality suggests good context use
      }
    }
    
    // Technical content relevance
    const characteristics = simulation.characteristics;
    if (characteristics.hasTechnicalTerms > 2 && simulation.outputQuality.overall > 0.6) {
      relevance += 0.05;
    }
    
    return Math.max(0.0, Math.min(1.0, relevance));
  }

  /**
   * Evaluate usability of simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {number} Usability score (0-1)
   */
  evaluateUsability(simulation, prompt, params) {
    let usability = simulation.outputQuality.usability;
    
    // Structure affects usability
    const structure = simulation.outputStructure;
    
    if (structure.organization === 'structured') {
      usability += 0.1;
    }
    
    if (structure.hasList || structure.hasSections) {
      usability += 0.05;
    }
    
    if (structure.hasExamples) {
      usability += 0.05;
    }
    
    // Length appropriateness affects usability
    if (simulation.outputLength > 500) {
      usability -= 0.05; // Very long outputs may be harder to use
    } else if (simulation.outputLength < 50) {
      usability -= 0.1; // Very short outputs may lack detail
    }
    
    return Math.max(0.0, Math.min(1.0, usability));
  }

  /**
   * Evaluate efficiency of simulated output
   * @param {Object} simulation - Simulation data
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {number} Efficiency score (0-1)
   */
  evaluateEfficiency(simulation, prompt, params) {
    let efficiency = simulation.outputQuality.efficiency;
    
    // Shorter, higher-quality outputs are more efficient
    const qualityPerWord = simulation.outputQuality.overall / Math.max(simulation.outputLength / 100, 1);
    
    if (qualityPerWord > 0.8) {
      efficiency += 0.1;
    } else if (qualityPerWord < 0.4) {
      efficiency -= 0.1;
    }
    
    // Structured outputs are typically more efficient to consume
    if (simulation.outputStructure.organization === 'structured') {
      efficiency += 0.05;
    }
    
    return Math.max(0.0, Math.min(1.0, efficiency));
  }

  /**
   * Estimate expected output length for a prompt
   * @param {string} prompt - Input prompt
   * @returns {number} Expected length in words
   */
  estimateExpectedLength(prompt) {
    const words = prompt.split(/\s+/).length;
    let expectedLength = 100; // Base expectation
    
    // Adjust based on prompt length
    if (words > 50) expectedLength = 200;
    else if (words > 20) expectedLength = 150;
    else if (words < 10) expectedLength = 75;
    
    // Adjust based on request type
    if (/code|program|script/i.test(prompt)) expectedLength *= 0.6; // Code is more concise
    if (/explain|describe|analysis/i.test(prompt)) expectedLength *= 1.3; // Explanations are longer
    if (/list|steps/i.test(prompt)) expectedLength *= 0.8; // Lists are structured/shorter
    
    return expectedLength;
  }

  /**
   * Aggregate simulation results
   * @param {Array} evaluations - Individual evaluations
   * @param {Array} simulations - Simulation data
   * @param {Object} params - Test parameters
   * @returns {Object} Aggregated results
   */
  aggregateSimulationResults(evaluations, simulations, params) {
    const metrics = ['accuracy', 'completeness', 'relevance', 'usability', 'efficiency', 'overallScore'];
    
    const aggregated = {
      strategy: 'simulation',
      sampleSize: evaluations.length,
      timestamp: new Date().toISOString(),
      
      // Statistical summary
      statistics: {},
      
      // Quality assessment
      overallQuality: 0,
      qualityGrade: '',
      reliability: 0,
      
      // Detailed metrics
      metrics: {},
      
      // Issues and recommendations
      commonIssues: [],
      improvementRecommendations: [],
      
      // Raw data
      simulations: simulations,
      evaluations: evaluations
    };

    // Calculate statistics for each metric
    metrics.forEach(metric => {
      const values = evaluations.map(evaluation => evaluation[metric]);
      aggregated.statistics[metric] = {
        mean: values.reduce((sum, val) => sum + val, 0) / values.length,
        median: this.calculateMedian(values),
        standardDeviation: this.calculateStandardDeviation(values),
        min: Math.min(...values),
        max: Math.max(...values),
        range: Math.max(...values) - Math.min(...values)
      };
    });

    // Overall quality is the mean of overallScore
    aggregated.overallQuality = aggregated.statistics.overallScore.mean;
    aggregated.qualityGrade = this.getQualityGrade(aggregated.overallQuality);
    
    // Reliability is inversely related to variability
    const overallVariability = aggregated.statistics.overallScore.standardDeviation;
    aggregated.reliability = Math.max(0, 1 - overallVariability);

    // Aggregate detailed metrics
    Object.keys(this.config.qualityMetrics).forEach(metric => {
      aggregated.metrics[metric] = {
        score: aggregated.statistics[metric].mean,
        consistency: 1 - aggregated.statistics[metric].standardDeviation,
        weight: this.config.qualityMetrics[metric].weight
      };
    });

    // Identify common issues
    aggregated.commonIssues = this.identifyCommonIssues(evaluations);
    aggregated.improvementRecommendations = this.generateImprovementRecommendations(aggregated);

    return aggregated;
  }

  /**
   * Calculate median of array
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
   * Calculate standard deviation
   * @param {Array} values - Numeric values
   * @returns {number} Standard deviation
   */
  calculateStandardDeviation(values) {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDifferences = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDifferences.reduce((sum, sq) => sum + sq, 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Identify common issues across simulations
   * @param {Array} evaluations - Evaluation results
   * @returns {Array} Common issues
   */
  identifyCommonIssues(evaluations) {
    const allIssues = evaluations.flatMap(evaluation => evaluation.identifiedIssues);
    const issueCounts = new Map();

    allIssues.forEach(issue => {
      const key = issue.type || issue.category || 'unknown';
      issueCounts.set(key, (issueCounts.get(key) || 0) + 1);
    });

    const commonThreshold = Math.ceil(evaluations.length * 0.3); // 30% threshold
    
    return Array.from(issueCounts.entries())
      .filter(([issue, count]) => count >= commonThreshold)
      .map(([issue, count]) => ({
        issue: issue,
        frequency: count,
        percentage: Math.round((count / evaluations.length) * 100)
      }))
      .sort((a, b) => b.frequency - a.frequency);
  }

  /**
   * Generate improvement recommendations
   * @param {Object} aggregated - Aggregated results
   * @returns {Array} Improvement recommendations
   */
  generateImprovementRecommendations(aggregated) {
    const recommendations = [];
    const thresholds = this.config.thresholds;

    // Check each metric for improvement opportunities
    Object.entries(aggregated.metrics).forEach(([metric, data]) => {
      if (data.score < thresholds.acceptable) {
        recommendations.push({
          category: metric,
          priority: data.score < thresholds.poor ? 'high' : 'medium',
          recommendation: this.getMetricRecommendation(metric, data.score),
          currentScore: data.score,
          targetScore: thresholds.good
        });
      }
    });

    // Overall quality recommendations
    if (aggregated.overallQuality < thresholds.good) {
      recommendations.push({
        category: 'overall',
        priority: aggregated.overallQuality < thresholds.acceptable ? 'high' : 'medium',
        recommendation: 'Improve prompt clarity, add specific requirements, and provide better context',
        currentScore: aggregated.overallQuality,
        targetScore: thresholds.good
      });
    }

    // Reliability recommendations
    if (aggregated.reliability < 0.7) {
      recommendations.push({
        category: 'consistency',
        priority: 'medium',
        recommendation: 'Reduce ambiguity in prompt to improve response consistency',
        currentScore: aggregated.reliability,
        targetScore: 0.8
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  /**
   * Get metric-specific recommendation
   * @param {string} metric - Metric name
   * @param {number} score - Current score
   * @returns {string} Recommendation text
   */
  getMetricRecommendation(metric, score) {
    const recommendations = {
      accuracy: 'Add more specific requirements and clear success criteria to improve output accuracy',
      completeness: 'Provide comprehensive context and specify all required elements in the output',
      relevance: 'Include domain-specific context and clarify the intended use case',
      usability: 'Specify output format and structure to improve usability of generated content',
      efficiency: 'Optimize prompt length and structure for more efficient content generation'
    };

    return recommendations[metric] || 'Review and improve prompt specificity and clarity';
  }

  /**
   * Identify simulation issues
   * @param {Object} evaluation - Evaluation data
   * @param {Object} simulation - Simulation data
   * @returns {Array} Identified issues
   */
  identifySimulationIssues(evaluation, simulation) {
    const issues = [];

    if (evaluation.accuracy < 0.6) {
      issues.push({
        type: 'accuracy',
        severity: evaluation.accuracy < 0.4 ? 'high' : 'medium',
        description: 'Simulated output may not accurately address the prompt requirements'
      });
    }

    if (evaluation.completeness < 0.6) {
      issues.push({
        type: 'completeness',
        severity: evaluation.completeness < 0.4 ? 'high' : 'medium',
        description: 'Simulated output appears incomplete relative to prompt requirements'
      });
    }

    if (simulation.outputLength < 30) {
      issues.push({
        type: 'length',
        severity: 'medium',
        description: 'Output may be too brief to fully address the prompt'
      });
    }

    const expectedLength = this.estimateExpectedLength(simulation.prompt);
    if (simulation.outputLength > expectedLength * 2) {
      issues.push({
        type: 'verbosity',
        severity: 'low',
        description: 'Output may be unnecessarily verbose'
      });
    }

    return issues;
  }

  /**
   * Identify simulation strengths
   * @param {Object} evaluation - Evaluation data
   * @param {Object} simulation - Simulation data
   * @returns {Array} Identified strengths
   */
  identifySimulationStrengths(evaluation, simulation) {
    const strengths = [];

    if (evaluation.accuracy >= 0.8) {
      strengths.push({
        category: 'accuracy',
        strength: 'High accuracy in addressing prompt requirements'
      });
    }

    if (evaluation.usability >= 0.8) {
      strengths.push({
        category: 'usability',
        strength: 'Well-structured output with good usability'
      });
    }

    if (evaluation.efficiency >= 0.8) {
      strengths.push({
        category: 'efficiency',
        strength: 'Efficient balance of quality and conciseness'
      });
    }

    if (simulation.outputStructure.organization === 'structured') {
      strengths.push({
        category: 'structure',
        strength: 'Clear, well-organized output structure'
      });
    }

    return strengths;
  }

  /**
   * Helper method: Count specific terms
   * @param {Array} words - Array of words
   * @returns {number} Count of specific terms
   */
  countSpecificTerms(words) {
    const specificTerms = [
      'create', 'build', 'implement', 'analyze', 'optimize', 'integrate',
      'function', 'class', 'method', 'component', 'system', 'algorithm'
    ];
    return words.filter(word => specificTerms.includes(word)).length;
  }

  /**
   * Helper method: Count vague terms
   * @param {Array} words - Array of words
   * @returns {number} Count of vague terms
   */
  countVagueTerms(words) {
    const vagueTerms = [
      'good', 'bad', 'better', 'improve', 'enhance', 'optimize',
      'thing', 'stuff', 'nice', 'great', 'awesome', 'cool'
    ];
    return words.filter(word => vagueTerms.includes(word)).length;
  }

  /**
   * Helper method: Count action verbs
   * @param {Array} words - Array of words
   * @returns {number} Count of action verbs
   */
  countActionVerbs(words) {
    const actionVerbs = [
      'create', 'build', 'make', 'develop', 'write', 'design',
      'implement', 'test', 'analyze', 'review', 'update', 'fix'
    ];
    return words.filter(word => actionVerbs.includes(word)).length;
  }

  /**
   * Helper method: Count technical terms
   * @param {Array} words - Array of words
   * @returns {number} Count of technical terms
   */
  countTechnicalTerms(words) {
    const technicalTerms = [
      'api', 'database', 'server', 'client', 'function', 'class',
      'method', 'variable', 'array', 'object', 'json', 'xml'
    ];
    return words.filter(word => technicalTerms.includes(word)).length;
  }

  /**
   * Calculate clarity score from characteristics
   * @param {Object} characteristics - Prompt characteristics
   * @returns {number} Clarity score (0-1)
   */
  calculateClarityScore(characteristics) {
    let score = 0.5; // Base score
    
    // Positive factors
    if (characteristics.hasSpecificTerms > 2) score += 0.2;
    if (characteristics.hasConstraints) score += 0.1;
    if (characteristics.hasFormat) score += 0.1;
    if (characteristics.hasContext) score += 0.1;
    
    // Negative factors
    if (characteristics.hasVagueTerms > 2) score -= 0.2;
    if (characteristics.wordCount < 10) score -= 0.1;
    
    return Math.max(0.1, Math.min(1.0, score));
  }

  /**
   * Calculate actionability score from characteristics
   * @param {Object} characteristics - Prompt characteristics
   * @returns {number} Actionability score (0-1)
   */
  calculateActionabilityScore(characteristics) {
    let score = 0.5; // Base score
    
    // Positive factors
    if (characteristics.hasActionVerbs > 1) score += 0.2;
    if (characteristics.hasSteps) score += 0.2;
    if (characteristics.hasFormat) score += 0.1;
    
    // Negative factors
    if (characteristics.hasVagueTerms > characteristics.hasSpecificTerms) score -= 0.2;
    
    return Math.max(0.1, Math.min(1.0, score));
  }

  /**
   * Get quality grade from score
   * @param {number} score - Quality score (0-1)
   * @returns {string} Quality grade
   */
  getQualityGrade(score) {
    const thresholds = this.config.thresholds;
    
    if (score >= thresholds.excellent) return 'Excellent';
    if (score >= thresholds.good) return 'Good';
    if (score >= thresholds.acceptable) return 'Acceptable';
    if (score >= thresholds.poor) return 'Poor';
    return 'Very Poor';
  }

  /**
   * Setup test parameters
   * @param {Object} context - Test context
   * @param {Object} options - Test options
   * @returns {Object} Test parameters
   */
  setupTestParameters(context, options) {
    return {
      strategy: options.strategy || 'simulation',
      sampleSize: options.sampleSize || this.config.simulation.sampleSize,
      context: context,
      simulation: {
        ...this.config.simulation,
        ...options.simulation
      },
      ...options
    };
  }

  /**
   * Validate inputs
   * @param {string} prompt - Prompt to test
   * @param {Object} context - Test context
   * @param {Object} options - Test options
   */
  validateInputs(prompt, context, options) {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Invalid prompt: must be a non-empty string');
    }

    if (prompt.length < 5) {
      throw new Error('Prompt too short: must be at least 5 characters');
    }

    if (options.strategy && !this.config.strategies[options.strategy]) {
      throw new Error(`Invalid strategy: ${options.strategy}`);
    }
  }

  /**
   * Generate cache key
   * @param {string} prompt - Prompt text
   * @param {Object} params - Test parameters
   * @returns {string} Cache key
   */
  generateCacheKey(prompt, params) {
    const keyData = {
      prompt: prompt.trim(),
      strategy: params.strategy,
      sampleSize: params.sampleSize,
      context: JSON.stringify(params.context || {})
    };
    
    return Buffer.from(JSON.stringify(keyData)).toString('base64');
  }

  /**
   * Post-process quality test results
   * @param {Object} qualityTest - Raw test results
   * @param {string} prompt - Original prompt
   * @param {Object} params - Test parameters
   * @returns {Object} Post-processed results
   */
  async postProcessQualityTest(qualityTest, prompt, params) {
    // Add metadata
    qualityTest.metadata = {
      promptLength: prompt.length,
      testStrategy: params.strategy,
      sampleSize: params.sampleSize,
      testedAt: new Date().toISOString(),
      version: '1.0.0'
    };

    // Add summary
    qualityTest.summary = {
      grade: qualityTest.qualityGrade,
      recommendation: this.getTestRecommendation(qualityTest.overallQuality),
      confidence: qualityTest.reliability || 0.8,
      keyIssues: qualityTest.commonIssues?.slice(0, 3) || []
    };

    return qualityTest;
  }

  /**
   * Get test recommendation
   * @param {number} score - Overall quality score
   * @returns {string} Recommendation
   */
  getTestRecommendation(score) {
    const thresholds = this.config.thresholds;
    
    if (score >= thresholds.excellent) return 'Prompt is highly effective and ready for use';
    if (score >= thresholds.good) return 'Prompt is effective with minor improvements possible';
    if (score >= thresholds.acceptable) return 'Prompt is usable but would benefit from improvements';
    return 'Prompt needs significant improvement before use';
  }

  /**
   * Initialize benchmark library
   * @returns {Object} Benchmark library
   */
  initializeBenchmarkLibrary() {
    return {
      // This would contain reference prompts and their expected quality scores
      // For simulation purposes, we'll keep this simple
      categories: {
        'code-generation': [],
        'analysis': [],
        'creation': [],
        'explanation': []
      }
    };
  }

  /**
   * Clear test cache
   */
  clearCache() {
    this.testCache.clear();
    this.logger.info('Quality test cache cleared');
  }
}

module.exports = OutputQualityTester;