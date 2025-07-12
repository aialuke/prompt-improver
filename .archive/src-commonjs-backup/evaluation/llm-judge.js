/**
 * LLM-as-a-Judge Evaluation System
 * Subjective quality evaluation using language models as evaluators
 */

const Logger = require('../utils/logger');

class LLMJudge {
  constructor(config = {}) {
    this.config = {
      // Default LLM configuration
      model: 'claude-3-haiku-20240307', // Fallback model
      temperature: 0.1, // Low temperature for consistent evaluation
      maxTokens: 2000,
      
      // Evaluation criteria
      evaluationCriteria: {
        clarity: {
          weight: 0.25,
          description: 'How clear and unambiguous the prompt is',
          scale: '1-10 where 10 is crystal clear and 1 is very confusing'
        },
        completeness: {
          weight: 0.25,
          description: 'How complete the prompt requirements are',
          scale: '1-10 where 10 is fully specified and 1 is missing critical elements'
        },
        actionability: {
          weight: 0.25,
          description: 'How actionable and specific the instructions are',
          scale: '1-10 where 10 is highly actionable and 1 is vague/abstract'
        },
        effectiveness: {
          weight: 0.25,
          description: 'How likely the prompt is to produce the desired outcome',
          scale: '1-10 where 10 is highly effective and 1 is likely to fail'
        }
      },

      // Evaluation strategies
      strategies: {
        single_judge: 'One LLM evaluates the prompt',
        multi_judge: 'Multiple LLMs evaluate and scores are aggregated',
        comparative: 'LLM compares pairs of prompts',
        rubric_based: 'LLM follows detailed rubric for evaluation'
      },

      // Bias mitigation
      biasMitigation: {
        randomizeOrder: true,
        anonymizePrompts: true,
        useMultipleJudges: true,
        temperatureVariation: [0.0, 0.1, 0.2] // Test different temperatures
      },

      ...config
    };

    this.logger = new Logger('LLMJudge');
    this.evaluationCache = new Map();
  }

  /**
   * Evaluate a prompt using LLM-as-a-Judge
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} context - Context for evaluation
   * @param {Object} options - Evaluation options
   * @returns {Promise<Object>} Evaluation results
   */
  async evaluatePrompt(prompt, context = {}, options = {}) {
    const startTime = Date.now();
    this.logger.info('Starting LLM judge evaluation', { 
      promptLength: prompt.length,
      strategy: options.strategy || 'single_judge'
    });

    try {
      // Validate inputs
      this.validateInputs(prompt, context, options);

      // Setup evaluation parameters
      const evaluationParams = this.setupEvaluationParameters(context, options);

      // Generate cache key for potential reuse
      const cacheKey = this.generateCacheKey(prompt, evaluationParams);
      
      // Check cache first
      if (this.evaluationCache.has(cacheKey) && !options.bypassCache) {
        this.logger.info('Returning cached evaluation result');
        return this.evaluationCache.get(cacheKey);
      }

      // Perform evaluation based on strategy
      let evaluation;
      switch (evaluationParams.strategy) {
        case 'multi_judge':
          evaluation = await this.performMultiJudgeEvaluation(prompt, evaluationParams);
          break;
        case 'comparative':
          evaluation = await this.performComparativeEvaluation(prompt, evaluationParams);
          break;
        case 'rubric_based':
          evaluation = await this.performRubricBasedEvaluation(prompt, evaluationParams);
          break;
        default:
          evaluation = await this.performSingleJudgeEvaluation(prompt, evaluationParams);
      }

      // Post-process evaluation results
      evaluation = await this.postProcessEvaluation(evaluation, prompt, evaluationParams);

      // Cache results
      this.evaluationCache.set(cacheKey, evaluation);

      const evaluationTime = Date.now() - startTime;
      this.logger.info('LLM judge evaluation completed', {
        overallScore: evaluation.overallScore,
        evaluationTime,
        strategy: evaluationParams.strategy
      });

      return evaluation;

    } catch (error) {
      this.logger.error('LLM judge evaluation failed', error);
      throw new Error(`LLM evaluation failed: ${error.message}`);
    }
  }

  /**
   * Perform single judge evaluation
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} params - Evaluation parameters
   * @returns {Promise<Object>} Evaluation results
   */
  async performSingleJudgeEvaluation(prompt, params) {
    const evaluationPrompt = this.buildEvaluationPrompt(prompt, params);
    const response = await this.callLLMJudge(evaluationPrompt, params);
    
    return this.parseEvaluationResponse(response, params);
  }

  /**
   * Perform multi-judge evaluation with consensus
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} params - Evaluation parameters
   * @returns {Promise<Object>} Aggregated evaluation results
   */
  async performMultiJudgeEvaluation(prompt, params) {
    const numJudges = params.numJudges || 3;
    const evaluations = [];

    // Get evaluations from multiple judges with different temperatures
    for (let i = 0; i < numJudges; i++) {
      const judgeParams = {
        ...params,
        temperature: this.config.biasMitigation.temperatureVariation[i % this.config.biasMitigation.temperatureVariation.length],
        judgeId: i + 1
      };

      const evaluationPrompt = this.buildEvaluationPrompt(prompt, judgeParams);
      const response = await this.callLLMJudge(evaluationPrompt, judgeParams);
      const evaluation = this.parseEvaluationResponse(response, judgeParams);
      
      evaluations.push(evaluation);
    }

    // Aggregate results
    return this.aggregateMultiJudgeEvaluations(evaluations, params);
  }

  /**
   * Perform comparative evaluation against reference prompts
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} params - Evaluation parameters
   * @returns {Promise<Object>} Comparative evaluation results
   */
  async performComparativeEvaluation(prompt, params) {
    const referencePrompts = params.referencePrompts || this.getDefaultReferencePrompts();
    const comparisons = [];

    for (const referencePrompt of referencePrompts) {
      const comparisonPrompt = this.buildComparisonPrompt(prompt, referencePrompt, params);
      const response = await this.callLLMJudge(comparisonPrompt, params);
      const comparison = this.parseComparisonResponse(response, referencePrompt, params);
      
      comparisons.push(comparison);
    }

    return this.aggregateComparativeEvaluations(comparisons, params);
  }

  /**
   * Perform rubric-based evaluation with detailed criteria
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} params - Evaluation parameters
   * @returns {Promise<Object>} Rubric-based evaluation results
   */
  async performRubricBasedEvaluation(prompt, params) {
    const rubric = params.customRubric || this.buildDefaultRubric(params);
    const evaluationPrompt = this.buildRubricEvaluationPrompt(prompt, rubric, params);
    
    const response = await this.callLLMJudge(evaluationPrompt, params);
    return this.parseRubricEvaluationResponse(response, rubric, params);
  }

  /**
   * Build evaluation prompt for LLM judge
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} params - Evaluation parameters
   * @returns {string} Evaluation prompt
   */
  buildEvaluationPrompt(prompt, params) {
    const criteria = this.config.evaluationCriteria;
    const context = params.context || {};
    
    let evaluationPrompt = `You are an expert prompt evaluator. Your task is to assess the quality of the following prompt across multiple dimensions.

**PROMPT TO EVALUATE:**
"${prompt}"

**EVALUATION CONTEXT:**
${context.domain ? `Domain: ${context.domain}` : ''}
${context.projectType ? `Project Type: ${context.projectType}` : ''}
${context.frameworks ? `Technologies: ${context.frameworks.join(', ')}` : ''}
${context.targetAudience ? `Target Audience: ${context.targetAudience}` : ''}

**EVALUATION CRITERIA:**
Please rate the prompt on each of the following dimensions:

1. **Clarity** (${criteria.clarity.scale})
   ${criteria.clarity.description}

2. **Completeness** (${criteria.completeness.scale})
   ${criteria.completeness.description}

3. **Actionability** (${criteria.actionability.scale})
   ${criteria.actionability.description}

4. **Effectiveness** (${criteria.effectiveness.scale})
   ${criteria.effectiveness.description}

**RESPONSE FORMAT:**
Provide your evaluation in the following JSON format:
{
  "clarity": {
    "score": [1-10],
    "reasoning": "Detailed explanation of the clarity score"
  },
  "completeness": {
    "score": [1-10],
    "reasoning": "Detailed explanation of the completeness score"
  },
  "actionability": {
    "score": [1-10],
    "reasoning": "Detailed explanation of the actionability score"
  },
  "effectiveness": {
    "score": [1-10],
    "reasoning": "Detailed explanation of the effectiveness score"
  },
  "overall_assessment": "Summary of strengths and weaknesses",
  "improvement_suggestions": [
    "Specific suggestion 1",
    "Specific suggestion 2",
    "Specific suggestion 3"
  ],
  "estimated_success_probability": [0-100]
}

Be objective, specific, and provide actionable feedback. Focus on concrete aspects that can be improved.`;

    return evaluationPrompt;
  }

  /**
   * Build comparison prompt for comparative evaluation
   * @param {string} promptA - First prompt
   * @param {string} promptB - Second prompt (reference)
   * @param {Object} params - Evaluation parameters
   * @returns {string} Comparison prompt
   */
  buildComparisonPrompt(promptA, promptB, params) {
    return `You are an expert prompt evaluator. Compare these two prompts and determine which is better and why.

**PROMPT A:**
"${promptA}"

**PROMPT B (Reference):**
"${promptB}"

**COMPARISON CRITERIA:**
- Clarity and specificity
- Completeness of requirements
- Actionability of instructions
- Likelihood of producing desired results

**RESPONSE FORMAT:**
{
  "winner": "A" or "B",
  "confidence": [1-10],
  "reasoning": "Detailed explanation of why the winner is better",
  "score_difference": [1-10],
  "prompt_a_strengths": ["strength 1", "strength 2"],
  "prompt_a_weaknesses": ["weakness 1", "weakness 2"],
  "prompt_b_strengths": ["strength 1", "strength 2"],
  "prompt_b_weaknesses": ["weakness 1", "weakness 2"]
}

Be objective and provide specific reasoning for your comparison.`;
  }

  /**
   * Build rubric evaluation prompt
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} rubric - Evaluation rubric
   * @param {Object} params - Evaluation parameters
   * @returns {string} Rubric evaluation prompt
   */
  buildRubricEvaluationPrompt(prompt, rubric, params) {
    let rubricText = '';
    
    Object.entries(rubric).forEach(([criterion, details]) => {
      rubricText += `\n**${criterion.toUpperCase()}:**\n`;
      Object.entries(details.levels).forEach(([level, description]) => {
        rubricText += `${level}: ${description}\n`;
      });
    });

    return `You are an expert prompt evaluator. Evaluate the following prompt using the detailed rubric provided.

**PROMPT TO EVALUATE:**
"${prompt}"

**EVALUATION RUBRIC:**
${rubricText}

**INSTRUCTIONS:**
1. For each criterion, determine which level (1-4) best describes the prompt
2. Provide specific evidence from the prompt to support your rating
3. Suggest concrete improvements where applicable

**RESPONSE FORMAT:**
{
  "rubric_scores": {
    ${Object.keys(rubric).map(criterion => `"${criterion}": {"level": [1-4], "evidence": "specific evidence", "improvement": "suggestion"}`).join(',\n    ')}
  },
  "overall_level": [1-4],
  "strengths": ["strength 1", "strength 2"],
  "priority_improvements": ["improvement 1", "improvement 2"],
  "revision_suggestions": "Specific rewrite suggestions"
}

Be thorough and provide specific evidence for each rating.`;
  }

  /**
   * Simulate LLM judge call (in real implementation, this would call actual LLM API)
   * @param {string} evaluationPrompt - Prompt for the judge
   * @param {Object} params - Evaluation parameters
   * @returns {Promise<string>} LLM response
   */
  async callLLMJudge(evaluationPrompt, params) {
    // In a real implementation, this would make an API call to Claude, GPT, or other LLM
    // For this framework, we'll simulate intelligent responses based on the prompt content
    
    this.logger.info('Simulating LLM judge call', { 
      model: params.model || this.config.model,
      temperature: params.temperature,
      promptLength: evaluationPrompt.length
    });

    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));

    // Extract the original prompt being evaluated
    const promptMatch = evaluationPrompt.match(/\*\*PROMPT TO EVALUATE:\*\*\s*"([^"]+)"/);
    const originalPrompt = promptMatch ? promptMatch[1] : '';

    // Generate simulated evaluation based on prompt analysis
    return this.generateSimulatedEvaluation(originalPrompt, params);
  }

  /**
   * Generate simulated LLM evaluation response
   * @param {string} prompt - Original prompt being evaluated
   * @param {Object} params - Evaluation parameters
   * @returns {string} Simulated JSON response
   */
  generateSimulatedEvaluation(prompt, params) {
    const words = prompt.split(/\s+/).length;
    const hasSpecificTerms = /\b(create|build|implement|analyze|specific|exactly|must|should)\b/i.test(prompt);
    const hasExamples = /\b(example|such as|like|including|for instance)\b/i.test(prompt);
    const hasContext = /\b(using|with|for|in|given|considering)\b/i.test(prompt);
    const hasFormat = /\b(format|output|return|json|xml|csv|display)\b/i.test(prompt);
    const isVague = /\b(good|nice|better|improve|enhance|optimize|thing|stuff)\b/i.test(prompt);
    
    // Calculate scores based on prompt characteristics
    let clarityScore = 5;
    if (hasSpecificTerms) clarityScore += 2;
    if (isVague) clarityScore -= 2;
    if (words > 10) clarityScore += 1;
    if (words > 50) clarityScore += 1;
    clarityScore = Math.max(1, Math.min(10, clarityScore));

    let completenessScore = 4;
    if (hasContext) completenessScore += 2;
    if (hasFormat) completenessScore += 2;
    if (hasExamples) completenessScore += 1;
    if (words < 10) completenessScore -= 2;
    completenessScore = Math.max(1, Math.min(10, completenessScore));

    let actionabilityScore = 5;
    if (hasSpecificTerms) actionabilityScore += 2;
    if (/\b(step|process|procedure)\b/i.test(prompt)) actionabilityScore += 1;
    if (isVague) actionabilityScore -= 2;
    actionabilityScore = Math.max(1, Math.min(10, actionabilityScore));

    let effectivenessScore = Math.round((clarityScore + completenessScore + actionabilityScore) / 3);
    effectivenessScore = Math.max(1, Math.min(10, effectivenessScore));

    // Add some randomness for temperature simulation
    const randomFactor = (params.temperature || 0.1) * 2;
    clarityScore += Math.round((Math.random() - 0.5) * randomFactor);
    completenessScore += Math.round((Math.random() - 0.5) * randomFactor);
    actionabilityScore += Math.round((Math.random() - 0.5) * randomFactor);
    effectivenessScore += Math.round((Math.random() - 0.5) * randomFactor);

    // Ensure scores stay in bounds
    clarityScore = Math.max(1, Math.min(10, clarityScore));
    completenessScore = Math.max(1, Math.min(10, completenessScore));
    actionabilityScore = Math.max(1, Math.min(10, actionabilityScore));
    effectivenessScore = Math.max(1, Math.min(10, effectivenessScore));

    const successProbability = Math.round(((clarityScore + completenessScore + actionabilityScore + effectivenessScore) / 40) * 100);

    const evaluation = {
      clarity: {
        score: clarityScore,
        reasoning: this.generateScoreReasoning('clarity', clarityScore, { hasSpecificTerms, isVague, words })
      },
      completeness: {
        score: completenessScore,
        reasoning: this.generateScoreReasoning('completeness', completenessScore, { hasContext, hasFormat, hasExamples, words })
      },
      actionability: {
        score: actionabilityScore,
        reasoning: this.generateScoreReasoning('actionability', actionabilityScore, { hasSpecificTerms, isVague })
      },
      effectiveness: {
        score: effectivenessScore,
        reasoning: this.generateScoreReasoning('effectiveness', effectivenessScore, { clarityScore, completenessScore, actionabilityScore })
      },
      overall_assessment: this.generateOverallAssessment({ clarityScore, completenessScore, actionabilityScore, effectivenessScore }),
      improvement_suggestions: this.generateImprovementSuggestions({ hasContext, hasFormat, hasExamples, isVague, hasSpecificTerms, words }),
      estimated_success_probability: successProbability
    };

    return JSON.stringify(evaluation, null, 2);
  }

  /**
   * Generate reasoning for individual scores
   * @param {string} criterion - Evaluation criterion
   * @param {number} score - Score given
   * @param {Object} characteristics - Prompt characteristics
   * @returns {string} Reasoning text
   */
  generateScoreReasoning(criterion, score, characteristics) {
    const reasoningTemplates = {
      clarity: {
        high: 'The prompt uses specific, actionable language with clear technical terms and well-defined requirements.',
        medium: 'The prompt is generally clear but contains some vague terms that could be interpreted multiple ways.',
        low: 'The prompt uses vague language and ambiguous terms that make the intended outcome unclear.'
      },
      completeness: {
        high: 'The prompt includes comprehensive context, output format specifications, and detailed requirements.',
        medium: 'The prompt covers most essential elements but may be missing some context or format specifications.',
        low: 'The prompt lacks important context, output format, or key requirements needed for successful completion.'
      },
      actionability: {
        high: 'The prompt provides specific, actionable instructions that clearly define what needs to be done.',
        medium: 'The prompt gives some actionable guidance but could be more specific about the required actions.',
        low: 'The prompt is too abstract or vague to provide clear direction for action.'
      },
      effectiveness: {
        high: 'Based on the strong clarity, completeness, and actionability, this prompt is likely to produce excellent results.',
        medium: 'The prompt should produce reasonable results but may require some interpretation or clarification.',
        low: 'The prompt\'s vagueness and lack of specificity make it unlikely to produce the desired outcome reliably.'
      }
    };

    let level = 'medium';
    if (score >= 8) level = 'high';
    else if (score <= 4) level = 'low';

    let reasoning = reasoningTemplates[criterion][level];

    // Add specific details based on characteristics
    if (criterion === 'clarity' && characteristics.hasSpecificTerms) {
      reasoning += ' Strong use of specific technical terminology enhances clarity.';
    }
    if (criterion === 'clarity' && characteristics.isVague) {
      reasoning += ' However, vague terms like "good" or "better" reduce precision.';
    }
    if (criterion === 'completeness' && !characteristics.hasContext) {
      reasoning += ' The prompt would benefit from additional context about the use case.';
    }
    if (criterion === 'completeness' && !characteristics.hasFormat) {
      reasoning += ' Specifying the desired output format would improve completeness.';
    }

    return reasoning;
  }

  /**
   * Generate overall assessment
   * @param {Object} scores - Individual criterion scores
   * @returns {string} Overall assessment text
   */
  generateOverallAssessment(scores) {
    const average = (scores.clarityScore + scores.completenessScore + scores.actionabilityScore + scores.effectivenessScore) / 4;
    
    if (average >= 8) {
      return 'This is a high-quality prompt with clear instructions, comprehensive requirements, and strong potential for success. Minor refinements could make it excellent.';
    } else if (average >= 6) {
      return 'This is a solid prompt with good foundations, but it has room for improvement in specificity and completeness. With targeted enhancements, it could become highly effective.';
    } else if (average >= 4) {
      return 'This prompt has some positive elements but significant weaknesses that limit its effectiveness. Substantial revisions are needed to improve clarity and completeness.';
    } else {
      return 'This prompt requires major revision. The lack of clarity, specificity, and completeness makes it unlikely to produce desired results without significant improvement.';
    }
  }

  /**
   * Generate improvement suggestions
   * @param {Object} characteristics - Prompt characteristics
   * @returns {Array} Array of suggestions
   */
  generateImprovementSuggestions(characteristics) {
    const suggestions = [];
    
    if (!characteristics.hasContext) {
      suggestions.push('Add specific context about the use case, target audience, or project requirements');
    }
    
    if (!characteristics.hasFormat) {
      suggestions.push('Specify the desired output format (e.g., JSON, markdown, step-by-step list)');
    }
    
    if (!characteristics.hasExamples) {
      suggestions.push('Include concrete examples to illustrate the expected output or approach');
    }
    
    if (characteristics.isVague) {
      suggestions.push('Replace vague terms like "good", "better", or "improve" with specific, measurable criteria');
    }
    
    if (!characteristics.hasSpecificTerms) {
      suggestions.push('Use more specific action verbs and technical terminology relevant to the task');
    }
    
    if (characteristics.words < 15) {
      suggestions.push('Expand the prompt with more detailed requirements and constraints');
    }
    
    // Always include at least one suggestion
    if (suggestions.length === 0) {
      suggestions.push('Consider adding more specific technical details or constraints to enhance precision');
    }
    
    return suggestions.slice(0, 3); // Limit to top 3 suggestions
  }

  /**
   * Parse evaluation response from LLM
   * @param {string} response - JSON response from LLM
   * @param {Object} params - Evaluation parameters
   * @returns {Object} Parsed evaluation results
   */
  parseEvaluationResponse(response, params) {
    try {
      const evaluation = JSON.parse(response);
      
      // Calculate weighted overall score
      const weights = this.config.evaluationCriteria;
      const overallScore = (
        evaluation.clarity.score * weights.clarity.weight +
        evaluation.completeness.score * weights.completeness.weight +
        evaluation.actionability.score * weights.actionability.weight +
        evaluation.effectiveness.score * weights.effectiveness.weight
      );

      return {
        judgeId: params.judgeId || 1,
        model: params.model || this.config.model,
        temperature: params.temperature,
        timestamp: new Date().toISOString(),
        
        // Individual scores
        clarity: evaluation.clarity,
        completeness: evaluation.completeness,
        actionability: evaluation.actionability,
        effectiveness: evaluation.effectiveness,
        
        // Aggregate metrics
        overallScore: Math.round(overallScore * 10) / 10,
        successProbability: evaluation.estimated_success_probability,
        
        // Qualitative feedback
        overallAssessment: evaluation.overall_assessment,
        suggestions: evaluation.improvement_suggestions,
        
        // Metadata
        confidence: this.calculateEvaluationConfidence(evaluation),
        bias_indicators: this.detectBiasIndicators(evaluation)
      };
    } catch (error) {
      this.logger.error('Failed to parse LLM evaluation response', { error: error.message, response });
      throw new Error(`Failed to parse evaluation response: ${error.message}`);
    }
  }

  /**
   * Aggregate multi-judge evaluations
   * @param {Array} evaluations - Individual judge evaluations
   * @param {Object} params - Evaluation parameters
   * @returns {Object} Aggregated evaluation results
   */
  aggregateMultiJudgeEvaluations(evaluations, params) {
    const criteria = ['clarity', 'completeness', 'actionability', 'effectiveness'];
    const aggregated = {
      judgeCount: evaluations.length,
      strategy: 'multi_judge',
      timestamp: new Date().toISOString(),
      
      // Aggregate scores
      scores: {},
      overallScore: 0,
      successProbability: 0,
      
      // Consensus metrics
      consensus: {},
      reliability: 0,
      
      // Combined feedback
      combinedAssessment: '',
      consolidatedSuggestions: [],
      
      // Individual judge results
      individualJudges: evaluations
    };

    // Calculate mean and consensus for each criterion
    criteria.forEach(criterion => {
      const scores = evaluations.map(evaluation => evaluation[criterion].score);
      const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      const standardDeviation = Math.sqrt(scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length);
      const variance = standardDeviation * standardDeviation;
      
      aggregated.scores[criterion] = {
        mean: Math.round(mean * 10) / 10,
        median: this.calculateMedian(scores),
        standardDeviation: Math.round(standardDeviation * 100) / 100,
        range: { min: Math.min(...scores), max: Math.max(...scores) },
        consensus: standardDeviation < 1.5 ? 'high' : standardDeviation < 2.5 ? 'medium' : 'low'
      };
      
      aggregated.consensus[criterion] = 1 - Math.min(variance / 4, 1); // Normalized consensus score
    });

    // Calculate overall metrics
    aggregated.overallScore = (
      aggregated.scores.clarity.mean * this.config.evaluationCriteria.clarity.weight +
      aggregated.scores.completeness.mean * this.config.evaluationCriteria.completeness.weight +
      aggregated.scores.actionability.mean * this.config.evaluationCriteria.actionability.weight +
      aggregated.scores.effectiveness.mean * this.config.evaluationCriteria.effectiveness.weight
    );

    aggregated.successProbability = Math.round(evaluations.reduce((sum, evaluation) => sum + evaluation.successProbability, 0) / evaluations.length);
    
    // Calculate reliability (average consensus across criteria)
    aggregated.reliability = Object.values(aggregated.consensus).reduce((sum, consensus) => sum + consensus, 0) / criteria.length;

    // Combine qualitative feedback
    aggregated.combinedAssessment = this.combineAssessments(evaluations.map(evaluation => evaluation.overallAssessment));
    aggregated.consolidatedSuggestions = this.consolidateSuggestions(evaluations.map(evaluation => evaluation.suggestions));

    return aggregated;
  }

  /**
   * Calculate median of an array
   * @param {Array} values - Array of numbers
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
   * Combine multiple assessments into a coherent summary
   * @param {Array} assessments - Individual assessments
   * @returns {string} Combined assessment
   */
  combineAssessments(assessments) {
    // Simple approach: find common themes and create summary
    const themes = {
      positive: [],
      negative: [],
      neutral: []
    };

    assessments.forEach(assessment => {
      if (assessment.includes('high-quality') || assessment.includes('excellent') || assessment.includes('strong')) {
        themes.positive.push('high quality');
      } else if (assessment.includes('solid') || assessment.includes('good')) {
        themes.positive.push('good foundation');
      }
      
      if (assessment.includes('weaknesses') || assessment.includes('requires') || assessment.includes('lacks')) {
        themes.negative.push('needs improvement');
      }
    });

    let combined = 'Multi-judge consensus: ';
    if (themes.positive.length > themes.negative.length) {
      combined += 'The prompt demonstrates strong qualities with some areas for enhancement.';
    } else if (themes.negative.length > themes.positive.length) {
      combined += 'The prompt requires significant improvement to be effective.';
    } else {
      combined += 'The prompt has mixed qualities with balanced strengths and weaknesses.';
    }

    return combined;
  }

  /**
   * Consolidate suggestions from multiple judges
   * @param {Array} suggestionLists - Arrays of suggestions from each judge
   * @returns {Array} Consolidated unique suggestions
   */
  consolidateSuggestions(suggestionLists) {
    const allSuggestions = suggestionLists.flat();
    const suggestionCounts = new Map();

    // Count frequency of similar suggestions
    allSuggestions.forEach(suggestion => {
      const normalized = suggestion.toLowerCase();
      let found = false;
      
      for (const [existing, count] of suggestionCounts) {
        if (this.areSimilarSuggestions(normalized, existing.toLowerCase())) {
          suggestionCounts.set(existing, count + 1);
          found = true;
          break;
        }
      }
      
      if (!found) {
        suggestionCounts.set(suggestion, 1);
      }
    });

    // Return suggestions sorted by frequency
    return Array.from(suggestionCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5) // Top 5 suggestions
      .map(([suggestion, count]) => ({
        suggestion,
        support: count,
        confidence: count / suggestionLists.length
      }));
  }

  /**
   * Check if two suggestions are similar
   * @param {string} suggestionA - First suggestion
   * @param {string} suggestionB - Second suggestion
   * @returns {boolean} Whether suggestions are similar
   */
  areSimilarSuggestions(suggestionA, suggestionB) {
    const wordsA = new Set(suggestionA.split(/\s+/));
    const wordsB = new Set(suggestionB.split(/\s+/));
    
    const intersection = new Set([...wordsA].filter(word => wordsB.has(word)));
    const union = new Set([...wordsA, ...wordsB]);
    
    const similarity = intersection.size / union.size;
    return similarity > 0.5; // 50% word overlap threshold
  }

  /**
   * Calculate evaluation confidence
   * @param {Object} evaluation - Evaluation results
   * @returns {number} Confidence score (0-1)
   */
  calculateEvaluationConfidence(evaluation) {
    // Simple heuristic: higher confidence for more consistent scores and detailed reasoning
    const scores = [
      evaluation.clarity.score,
      evaluation.completeness.score,
      evaluation.actionability.score,
      evaluation.effectiveness.score
    ];
    
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    
    // Lower variance = higher confidence
    const consistencyConfidence = Math.max(0, 1 - variance / 10);
    
    // Longer reasoning = higher confidence
    const reasoningLengths = [
      evaluation.clarity.reasoning.length,
      evaluation.completeness.reasoning.length,
      evaluation.actionability.reasoning.length,
      evaluation.effectiveness.reasoning.length
    ];
    
    const avgReasoningLength = reasoningLengths.reduce((sum, length) => sum + length, 0) / reasoningLengths.length;
    const detailConfidence = Math.min(1, avgReasoningLength / 100); // Normalize to reasonable length
    
    return (consistencyConfidence + detailConfidence) / 2;
  }

  /**
   * Detect potential bias indicators in evaluation
   * @param {Object} evaluation - Evaluation results
   * @returns {Array} Array of detected bias indicators
   */
  detectBiasIndicators(evaluation) {
    const indicators = [];
    
    // Check for score clustering (all scores very similar)
    const scores = [
      evaluation.clarity.score,
      evaluation.completeness.score,
      evaluation.actionability.score,
      evaluation.effectiveness.score
    ];
    
    const maxScore = Math.max(...scores);
    const minScore = Math.min(...scores);
    
    if (maxScore - minScore <= 1) {
      indicators.push({
        type: 'score_clustering',
        description: 'All scores are very similar, which may indicate superficial evaluation',
        severity: 'medium'
      });
    }
    
    // Check for extreme scores (all very high or very low)
    const averageScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    
    if (averageScore >= 9) {
      indicators.push({
        type: 'positive_bias',
        description: 'Unusually high scores across all criteria',
        severity: 'low'
      });
    } else if (averageScore <= 3) {
      indicators.push({
        type: 'negative_bias',
        description: 'Unusually low scores across all criteria',
        severity: 'low'
      });
    }
    
    // Check for generic reasoning
    const reasoningTexts = [
      evaluation.clarity.reasoning,
      evaluation.completeness.reasoning,
      evaluation.actionability.reasoning,
      evaluation.effectiveness.reasoning
    ];
    
    const hasGenericReasoning = reasoningTexts.some(reasoning => 
      reasoning.length < 50 || 
      reasoning.includes('generally') ||
      reasoning.includes('seems')
    );
    
    if (hasGenericReasoning) {
      indicators.push({
        type: 'generic_reasoning',
        description: 'Some reasoning appears generic or superficial',
        severity: 'medium'
      });
    }
    
    return indicators;
  }

  /**
   * Setup evaluation parameters
   * @param {Object} context - Evaluation context
   * @param {Object} options - Evaluation options
   * @returns {Object} Evaluation parameters
   */
  setupEvaluationParameters(context, options) {
    return {
      strategy: options.strategy || 'single_judge',
      model: options.model || this.config.model,
      temperature: options.temperature !== undefined ? options.temperature : this.config.temperature,
      numJudges: options.numJudges || 3,
      context: context,
      customRubric: options.customRubric,
      referencePrompts: options.referencePrompts,
      ...options
    };
  }

  /**
   * Validate inputs for evaluation
   * @param {string} prompt - Prompt to evaluate
   * @param {Object} context - Evaluation context
   * @param {Object} options - Evaluation options
   */
  validateInputs(prompt, context, options) {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Invalid prompt: must be a non-empty string');
    }

    if (prompt.length < 5) {
      throw new Error('Prompt too short: must be at least 5 characters');
    }

    if (prompt.length > 10000) {
      throw new Error('Prompt too long: must be less than 10,000 characters');
    }

    if (options.strategy && !this.config.strategies[options.strategy]) {
      throw new Error(`Invalid strategy: ${options.strategy}. Available: ${Object.keys(this.config.strategies).join(', ')}`);
    }
  }

  /**
   * Generate cache key for evaluation
   * @param {string} prompt - Prompt text
   * @param {Object} params - Evaluation parameters
   * @returns {string} Cache key
   */
  generateCacheKey(prompt, params) {
    const keyData = {
      prompt: prompt.trim(),
      strategy: params.strategy,
      model: params.model,
      temperature: params.temperature,
      context: JSON.stringify(params.context || {})
    };
    
    return Buffer.from(JSON.stringify(keyData)).toString('base64');
  }

  /**
   * Post-process evaluation results
   * @param {Object} evaluation - Raw evaluation results
   * @param {string} prompt - Original prompt
   * @param {Object} params - Evaluation parameters
   * @returns {Object} Post-processed evaluation
   */
  async postProcessEvaluation(evaluation, prompt, params) {
    // Add metadata
    evaluation.metadata = {
      promptLength: prompt.length,
      evaluationStrategy: params.strategy,
      model: params.model,
      temperature: params.temperature,
      evaluatedAt: new Date().toISOString(),
      version: '1.0.0'
    };

    // Add evaluation summary
    evaluation.summary = {
      grade: this.getQualityGrade(evaluation.overallScore),
      recommendation: this.getRecommendation(evaluation.overallScore),
      priority: this.getPriorityLevel(evaluation)
    };

    return evaluation;
  }

  /**
   * Get quality grade from score
   * @param {number} score - Overall score
   * @returns {string} Quality grade
   */
  getQualityGrade(score) {
    if (score >= 8.5) return 'Excellent';
    if (score >= 7.0) return 'Good';
    if (score >= 5.5) return 'Fair';
    if (score >= 4.0) return 'Poor';
    return 'Very Poor';
  }

  /**
   * Get recommendation based on score
   * @param {number} score - Overall score
   * @returns {string} Recommendation
   */
  getRecommendation(score) {
    if (score >= 8.5) return 'Use as-is or with minor refinements';
    if (score >= 7.0) return 'Good foundation, implement suggested improvements';
    if (score >= 5.5) return 'Requires moderate revision before use';
    if (score >= 4.0) return 'Needs significant improvement';
    return 'Requires complete rewrite';
  }

  /**
   * Get priority level for improvements
   * @param {Object} evaluation - Evaluation results
   * @returns {string} Priority level
   */
  getPriorityLevel(evaluation) {
    if (evaluation.overallScore >= 7.5) return 'Low';
    if (evaluation.overallScore >= 5.5) return 'Medium';
    return 'High';
  }

  /**
   * Build default rubric for rubric-based evaluation
   * @param {Object} params - Evaluation parameters
   * @returns {Object} Default rubric
   */
  buildDefaultRubric(params) {
    return {
      clarity: {
        description: 'How clear and unambiguous the prompt is',
        levels: {
          1: 'Very unclear, multiple interpretations possible, vague language throughout',
          2: 'Somewhat unclear, some ambiguous terms, unclear intent in places',
          3: 'Mostly clear, minor ambiguities, generally understandable intent',
          4: 'Very clear, specific language, unambiguous intent and requirements'
        }
      },
      completeness: {
        description: 'How complete the requirements and context are',
        levels: {
          1: 'Missing critical elements, lacks context, no output format specified',
          2: 'Missing some important elements, limited context provided',
          3: 'Most essential elements present, adequate context and requirements',
          4: 'Comprehensive requirements, rich context, clear output format specified'
        }
      },
      actionability: {
        description: 'How actionable and specific the instructions are',
        levels: {
          1: 'Very abstract, no clear actions, impossible to follow',
          2: 'Some actionable elements, but many vague instructions',
          3: 'Mostly actionable, some specific steps provided',
          4: 'Highly actionable, specific steps and clear deliverables'
        }
      },
      effectiveness: {
        description: 'Likelihood of producing desired outcome',
        levels: {
          1: 'Very unlikely to succeed, fundamental issues present',
          2: 'May produce some results, but likely to miss the mark',
          3: 'Good chance of success with minor interpretation needed',
          4: 'Highly likely to produce excellent results as intended'
        }
      }
    };
  }

  /**
   * Get default reference prompts for comparative evaluation
   * @returns {Array} Array of reference prompts
   */
  getDefaultReferencePrompts() {
    return [
      'Create a responsive navigation component for a React web application. The component should include a logo, menu items for Home, About, Services, and Contact, and a mobile hamburger menu. Use modern CSS with flexbox or grid. Ensure accessibility with proper ARIA labels.',
      
      'Write a Python function that validates email addresses using regex. The function should return True for valid emails and False for invalid ones. Include proper error handling and comprehensive test cases.',
      
      'Design a database schema for an e-commerce platform. Include tables for users, products, orders, and order items. Define primary keys, foreign keys, and necessary indexes. Provide SQL CREATE statements and explain the relationships.'
    ];
  }

  /**
   * Clear evaluation cache
   */
  clearCache() {
    this.evaluationCache.clear();
    this.logger.info('Evaluation cache cleared');
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache statistics
   */
  getCacheStats() {
    return {
      size: this.evaluationCache.size,
      maxSize: 1000, // Theoretical max, could be configurable
      hitRate: this.cacheHits / (this.cacheHits + this.cacheMisses) || 0
    };
  }
}

module.exports = LLMJudge;