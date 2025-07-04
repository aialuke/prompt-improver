/**
 * Comprehensive Prompt Quality Assessment System
 * Based on research from Context7 and industry best practices
 * 
 * Feature Categories:
 * 1. Structural Features - Length, complexity, format
 * 2. Semantic Features - Clarity, specificity, context
 * 3. Task-Oriented Features - Instruction clarity, output specification
 * 4. Quality Indicators - Completeness, coherence, effectiveness
 */

export class PromptAnalyzer {
  constructor() {
    // Initialize NLP patterns and scoring weights
    this.patterns = {
      // Instruction patterns (based on prompt engineering research)
      instructions: [
        /^(write|create|generate|produce|develop|build|make)/i,
        /^(explain|describe|analyze|summarize|outline)/i,
        /^(list|enumerate|identify|find|locate)/i,
        /^(compare|contrast|evaluate|assess|judge)/i,
        /^(solve|calculate|compute|determine)/i,
        /^(translate|convert|transform|rewrite)/i
      ],
      
      // Clarity indicators
      clarity: {
        vague: /\b(something|anything|stuff|things|somehow|maybe|perhaps|might|could)\b/gi,
        specific: /\b(specifically|exactly|precisely|clearly|explicitly|detailed?ly?)\b/gi,
        quantifiers: /\b(\d+|first|second|third|few|several|many|all|each|every)\b/gi
      },
      
      // Context indicators
      context: {
        background: /\b(background|context|scenario|situation|given|assume|suppose)\b/gi,
        constraints: /\b(limit|restrict|constrain|must|should|required?|only|exactly)\b/gi,
        examples: /\b(example|instance|such as|like|including|for example|e\.g\.)\b/gi
      },
      
      // Output format specifications
      format: {
        structured: /\b(format|structure|organize|json|xml|table|list|bullet|numbered?)\b/gi,
        length: /\b(\d+\s*(words?|characters?|sentences?|paragraphs?|pages?|lines?))\b/gi,
        style: /\b(formal|informal|technical|casual|professional|academic|creative)\b/gi
      },
      
      // Quality indicators
      quality: {
        questions: /\?/g,
        exclamations: /!/g,
        imperatives: /\b(please|ensure|make sure|be sure|don't forget|remember)\b/gi,
        conditionals: /\b(if|when|unless|provided|given|assuming)\b/gi
      }
    };
    
    // Feature weights based on research findings
    this.weights = {
      structural: 0.25,
      semantic: 0.35,
      taskOriented: 0.25,
      qualityIndicators: 0.15
    };
  }

  /**
   * Extract comprehensive features from a prompt
   * @param {string} prompt - The input prompt text
   * @returns {Array<number>} Feature vector for ML models
   */
  extractFeatures(prompt) {
    if (!prompt || typeof prompt !== 'string') {
      return this.getDefaultFeatures();
    }

    const features = [];
    
    // 1. STRUCTURAL FEATURES (10 features)
    features.push(...this.extractStructuralFeatures(prompt));
    
    // 2. SEMANTIC FEATURES (12 features)
    features.push(...this.extractSemanticFeatures(prompt));
    
    // 3. TASK-ORIENTED FEATURES (8 features)
    features.push(...this.extractTaskOrientedFeatures(prompt));
    
    // 4. QUALITY INDICATORS (5 features)
    features.push(...this.extractQualityIndicators(prompt));
    
    return features; // 35 total features
  }

  /**
   * Extract structural features (length, complexity, format)
   */
  extractStructuralFeatures(prompt) {
    const words = prompt.trim().split(/\s+/);
    const sentences = prompt.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const chars = prompt.length;
    
    return [
      chars,                                    // Character count
      words.length,                            // Word count
      sentences.length,                        // Sentence count
      chars / words.length,                    // Avg chars per word
      words.length / sentences.length,         // Avg words per sentence
      (prompt.match(/[A-Z]/g) || []).length,  // Uppercase letters
      (prompt.match(/[0-9]/g) || []).length,  // Digits
      (prompt.match(/[^\w\s]/g) || []).length, // Special characters
      this.calculateComplexity(prompt),        // Lexical complexity
      this.calculateReadability(prompt)        // Readability score
    ];
  }

  /**
   * Extract semantic features (clarity, specificity, context)
   */
  extractSemanticFeatures(prompt) {
    const wordCount = prompt.split(/\s+/).length;
    
    return [
      this.countMatches(prompt, this.patterns.clarity.vague) / wordCount,      // Vagueness ratio
      this.countMatches(prompt, this.patterns.clarity.specific) / wordCount,   // Specificity ratio
      this.countMatches(prompt, this.patterns.clarity.quantifiers) / wordCount, // Quantifier ratio
      this.countMatches(prompt, this.patterns.context.background) / wordCount, // Background context
      this.countMatches(prompt, this.patterns.context.constraints) / wordCount, // Constraints
      this.countMatches(prompt, this.patterns.context.examples) / wordCount,   // Examples provided
      this.calculateSemanticDensity(prompt),                                   // Semantic density
      this.calculateAmbiguityScore(prompt),                                    // Ambiguity score
      this.calculateCoherenceScore(prompt),                                    // Coherence score
      this.calculateContextRichness(prompt),                                   // Context richness
      this.calculateConcretenesss(prompt),                                     // Concreteness
      this.calculateSpecificityScore(prompt)                                   // Overall specificity
    ];
  }

  /**
   * Extract task-oriented features (instruction clarity, output specification)
   */
  extractTaskOrientedFeatures(prompt) {
    const hasInstruction = this.patterns.instructions.some(pattern => pattern.test(prompt));
    const wordCount = prompt.split(/\s+/).length;
    
    return [
      hasInstruction ? 1 : 0,                                                  // Has clear instruction
      this.countMatches(prompt, this.patterns.format.structured) / wordCount, // Format specification
      this.countMatches(prompt, this.patterns.format.length) / wordCount,     // Length specification
      this.countMatches(prompt, this.patterns.format.style) / wordCount,      // Style specification
      this.calculateTaskClarity(prompt),                                       // Task clarity score
      this.calculateOutputSpecificity(prompt),                                 // Output specificity
      this.calculateActionableRatio(prompt),                                   // Actionable content ratio
      this.calculateCompleteness(prompt)                                       // Instruction completeness
    ];
  }

  /**
   * Extract quality indicators (effectiveness predictors)
   */
  extractQualityIndicators(prompt) {
    const wordCount = prompt.split(/\s+/).length;
    
    return [
      this.countMatches(prompt, this.patterns.quality.questions) / wordCount,    // Question ratio
      this.countMatches(prompt, this.patterns.quality.exclamations) / wordCount, // Exclamation ratio
      this.countMatches(prompt, this.patterns.quality.imperatives) / wordCount,  // Imperative ratio
      this.countMatches(prompt, this.patterns.quality.conditionals) / wordCount, // Conditional ratio
      this.calculateOverallQualityScore(prompt)                                  // Overall quality score
    ];
  }

  // Helper methods for advanced feature calculations
  
  calculateComplexity(prompt) {
    const words = prompt.split(/\s+/);
    const uniqueWords = new Set(words.map(w => w.toLowerCase())).size;
    return uniqueWords / words.length; // Lexical diversity
  }

  calculateReadability(prompt) {
    // Simplified Flesch-Kincaid grade level approximation
    const words = prompt.split(/\s+/).length;
    const sentences = prompt.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const syllables = this.countSyllables(prompt);
    
    if (sentences === 0) return 0;
    return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59;
  }

  calculateSemanticDensity(prompt) {
    // Ratio of content words to total words
    const contentWords = prompt.split(/\s+/).filter(word => 
      word.length > 3 && !/^(the|and|or|but|in|on|at|to|for|of|with|by)$/i.test(word)
    );
    const totalWords = prompt.split(/\s+/).length;
    return contentWords.length / totalWords;
  }

  calculateAmbiguityScore(prompt) {
    // Count ambiguous pronouns and vague terms
    const ambiguousTerms = /\b(this|that|it|they|them|such|those|these|one|thing|stuff)\b/gi;
    const matches = (prompt.match(ambiguousTerms) || []).length;
    const wordCount = prompt.split(/\s+/).length;
    return matches / wordCount;
  }

  calculateCoherenceScore(prompt) {
    // Simplified coherence based on sentence connections
    const sentences = prompt.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const connectors = /\b(therefore|however|moreover|furthermore|additionally|consequently|thus|hence|because|since|although|while|whereas)\b/gi;
    const connectorCount = (prompt.match(connectors) || []).length;
    return Math.min(1.0, connectorCount / Math.max(1, sentences.length - 1));
  }

  calculateContextRichness(prompt) {
    // Score based on context-providing elements
    const contextIndicators = [
      this.patterns.context.background,
      this.patterns.context.constraints,
      this.patterns.context.examples
    ];
    
    const totalMatches = contextIndicators.reduce((sum, pattern) => 
      sum + this.countMatches(prompt, pattern), 0);
    const wordCount = prompt.split(/\s+/).length;
    
    return Math.min(1.0, totalMatches / (wordCount * 0.1)); // Normalize
  }

  calculateConcretenesss(prompt) {
    // Ratio of concrete vs abstract terms
    const concreteTerms = /\b(see|hear|touch|feel|smell|taste|show|display|create|build|write|draw|calculate|measure)\b/gi;
    const abstractTerms = /\b(think|believe|understand|know|realize|consider|assume|suppose|imagine|concept|idea|theory)\b/gi;
    
    const concreteCount = (prompt.match(concreteTerms) || []).length;
    const abstractCount = (prompt.match(abstractTerms) || []).length;
    const total = concreteCount + abstractCount;
    
    return total === 0 ? 0.5 : concreteCount / total;
  }

  calculateSpecificityScore(prompt) {
    // Overall specificity based on multiple factors
    const specificTerms = /\b(exactly|specifically|precisely|detailed?|particular|specific|explicit)\b/gi;
    const vageTerms = /\b(some|any|various|general|overall|basically|essentially|somewhat)\b/gi;
    
    const specificCount = (prompt.match(specificTerms) || []).length;
    const vague = (prompt.match(vageTerms) || []).length;
    const wordCount = prompt.split(/\s+/).length;
    
    return (specificCount - vague) / wordCount + 0.5; // Normalize to 0-1
  }

  calculateTaskClarity(prompt) {
    // Score based on clear task identification
    const taskIndicators = this.patterns.instructions.filter(pattern => pattern.test(prompt)).length;
    const maxIndicators = this.patterns.instructions.length;
    return Math.min(1.0, taskIndicators / maxIndicators * 2); // Scale up
  }

  calculateOutputSpecificity(prompt) {
    // Score based on output format specifications
    const formatSpecs = [
      this.patterns.format.structured,
      this.patterns.format.length,
      this.patterns.format.style
    ];
    
    const totalMatches = formatSpecs.reduce((sum, pattern) => 
      sum + this.countMatches(prompt, pattern), 0);
    const wordCount = prompt.split(/\s+/).length;
    
    return Math.min(1.0, totalMatches / (wordCount * 0.05)); // Normalize
  }

  calculateActionableRatio(prompt) {
    // Ratio of actionable verbs to total verbs
    const actionableVerbs = /\b(create|write|generate|build|make|develop|design|implement|execute|perform|complete|finish|deliver|produce|construct)\b/gi;
    const allVerbs = /\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can|shall|must|create|write|generate|build|make|develop|design|implement|execute|perform|complete|finish|deliver|produce|construct)\b/gi;
    
    const actionableCount = (prompt.match(actionableVerbs) || []).length;
    const totalVerbs = (prompt.match(allVerbs) || []).length;
    
    return totalVerbs === 0 ? 0 : actionableCount / totalVerbs;
  }

  calculateCompleteness(prompt) {
    // Score based on presence of key prompt elements
    const elements = [
      /\b(what|how|why|when|where|who)\b/gi,      // Question words
      /\b(please|kindly|would you|could you)\b/gi, // Polite requests
      /\b(format|style|length|structure)\b/gi,      // Format specifications
      /\b(example|instance|such as)\b/gi,           // Examples
      /\b(context|background|given|assuming)\b/gi   // Context
    ];
    
    const presentElements = elements.filter(pattern => pattern.test(prompt)).length;
    return presentElements / elements.length;
  }

  calculateOverallQualityScore(prompt) {
    // Composite quality score combining multiple factors
    const factors = [
      this.calculateTaskClarity(prompt),
      this.calculateOutputSpecificity(prompt),
      this.calculateContextRichness(prompt),
      this.calculateSpecificityScore(prompt),
      this.calculateCompleteness(prompt)
    ];
    
    return factors.reduce((sum, factor) => sum + factor, 0) / factors.length;
  }

  // Utility methods
  
  countMatches(text, pattern) {
    if (!pattern) return 0;
    const matches = text.match(pattern);
    return matches ? matches.length : 0;
  }

  countSyllables(text) {
    // Simplified syllable counting
    const vowels = /[aeiouy]/gi;
    const words = text.toLowerCase().split(/\s+/);
    
    return words.reduce((total, word) => {
      const vowelMatches = word.match(vowels);
      let syllables = vowelMatches ? vowelMatches.length : 1;
      
      // Adjust for silent e
      if (word.endsWith('e') && syllables > 1) {
        syllables--;
      }
      
      return total + Math.max(1, syllables);
    }, 0);
  }

  getDefaultFeatures() {
    // Return default feature vector for invalid inputs
    return new Array(35).fill(0);
  }

  /**
   * Get feature names for interpretability
   */
  getFeatureNames() {
    return [
      // Structural (10)
      'char_count', 'word_count', 'sentence_count', 'avg_chars_per_word', 
      'avg_words_per_sentence', 'uppercase_count', 'digit_count', 'special_char_count',
      'lexical_complexity', 'readability_score',
      
      // Semantic (12)
      'vagueness_ratio', 'specificity_ratio', 'quantifier_ratio', 'background_context',
      'constraints_ratio', 'examples_ratio', 'semantic_density', 'ambiguity_score',
      'coherence_score', 'context_richness', 'concreteness', 'specificity_score',
      
      // Task-Oriented (8)
      'has_instruction', 'format_specification', 'length_specification', 'style_specification',
      'task_clarity', 'output_specificity', 'actionable_ratio', 'instruction_completeness',
      
      // Quality Indicators (5)
      'question_ratio', 'exclamation_ratio', 'imperative_ratio', 'conditional_ratio',
      'overall_quality_score'
    ];
  }

  /**
   * Analyze prompt and return detailed insights
   */
  analyzePrompt(prompt) {
    const features = this.extractFeatures(prompt);
    const featureNames = this.getFeatureNames();
    
    const analysis = {
      prompt: prompt,
      features: features,
      featureMap: {},
      insights: this.generateInsights(prompt, features),
      qualityScore: features[features.length - 1], // Overall quality score
      recommendations: this.generateRecommendations(prompt, features)
    };
    
    // Create feature map for interpretability
    featureNames.forEach((name, index) => {
      analysis.featureMap[name] = features[index];
    });
    
    return analysis;
  }

  generateInsights(prompt, features) {
    const insights = [];
    
    // Length analysis
    const wordCount = features[1];
    if (wordCount < 10) {
      insights.push("Prompt is quite short - consider adding more context or specificity");
    } else if (wordCount > 100) {
      insights.push("Prompt is quite long - consider breaking into steps or simplifying");
    }
    
    // Clarity analysis
    const vagueness = features[10];
    if (vagueness > 0.1) {
      insights.push("Prompt contains vague terms - consider being more specific");
    }
    
    // Task clarity
    const taskClarity = features[24];
    if (taskClarity < 0.5) {
      insights.push("Task instruction could be clearer - consider starting with an action verb");
    }
    
    // Context analysis
    const contextRichness = features[19];
    if (contextRichness < 0.3) {
      insights.push("Consider adding more background context or examples");
    }
    
    return insights;
  }

  generateRecommendations(prompt, features) {
    const recommendations = [];
    
    // Specific improvements based on feature analysis
    const hasInstruction = features[22];
    if (!hasInstruction) {
      recommendations.push("Add a clear instruction verb (e.g., 'Write', 'Create', 'Explain')");
    }
    
    const formatSpec = features[23];
    if (formatSpec === 0) {
      recommendations.push("Specify desired output format (e.g., 'in JSON format', 'as a bulleted list')");
    }
    
    const completeness = features[27];
    if (completeness < 0.6) {
      recommendations.push("Consider adding examples or constraints to make the request more complete");
    }
    
    return recommendations;
  }
}

export default PromptAnalyzer;