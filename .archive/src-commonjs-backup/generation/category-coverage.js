/**
 * Category Coverage System
 * Comprehensive prompt engineering scenarios covering all major issues
 */

class CategoryCoverage {
  constructor(config = {}) {
    this.config = {
      // Default category weights (should sum to 1.0)
      categoryWeights: {
        vague_instructions: 0.25,
        missing_context: 0.25,
        poor_structure: 0.20,
        missing_examples: 0.15,
        no_output_format: 0.15
      },

      // Minimum tests per category
      minimumPerCategory: 2,

      // Context-based weight adjustments
      contextAdjustments: {
        // Adjust weights based on project domain
        domainAdjustments: {
          'e-commerce': {
            vague_instructions: 1.2,
            missing_context: 1.3,
            poor_structure: 0.9
          },
          'fintech': {
            missing_context: 1.4,
            no_output_format: 1.3,
            poor_structure: 1.1
          },
          'healthcare': {
            missing_context: 1.3,
            no_output_format: 1.4,
            vague_instructions: 0.9
          },
          'education': {
            missing_examples: 1.4,
            poor_structure: 1.2,
            vague_instructions: 1.1
          }
        },

        // Adjust based on project type
        projectTypeAdjustments: {
          'web-application': {
            poor_structure: 1.2,
            missing_context: 1.1
          },
          'api-service': {
            no_output_format: 1.4,
            missing_context: 1.2,
            missing_examples: 1.1
          },
          'mobile-app': {
            vague_instructions: 1.2,
            missing_context: 1.1
          },
          'cli-tool': {
            no_output_format: 1.3,
            missing_examples: 1.2,
            poor_structure: 1.1
          }
        }
      },

      ...config
    };

    // Comprehensive category definitions
    this.categories = {
      vague_instructions: {
        name: 'Vague Instructions',
        description: 'Prompts with unclear, ambiguous, or non-specific requirements',
        purpose: 'Test ability to add clarity and specificity to unclear requests',
        priority: 'critical',
        
        characteristics: [
          'Use of ambiguous terms like "better", "good", "nice"',
          'Lack of specific requirements',
          'Missing success criteria',
          'Unclear scope boundaries',
          'Non-actionable language'
        ],

        examples: {
          simple: [
            'Make this better',
            'Fix the code',
            'Improve performance',
            'Add some features',
            'Clean up the project'
          ],
          moderate: [
            'Optimize the {component_type}',
            'Enhance the user experience',
            'Make the {framework} app more responsive',
            'Improve the {language} codebase',
            'Add better error handling'
          ],
          complex: [
            'Architect a scalable solution',
            'Design a robust system',
            'Implement best practices',
            'Create an enterprise-grade application',
            'Build a performant architecture'
          ]
        },

        expectedImprovements: [
          'Define specific goals and metrics',
          'Add concrete requirements',
          'Specify success criteria',
          'Clarify scope and boundaries',
          'Use actionable language'
        ],

        detectionPatterns: [
          /\b(better|good|nice|great|awesome|cool)\b/gi,
          /\b(improve|enhance|optimize|fix|clean)\b(?!\s+\w+\s+(by|to|with|using))/gi,
          /\bmake\s+\w+\s+(better|faster|cleaner|nicer)/gi,
          /\b(add some|create a|build a)\b(?!\s+specific)/gi
        ],

        contextualVariations: {
          'web-application': ['improve the UI', 'make it more user-friendly', 'enhance the frontend'],
          'api-service': ['optimize the API', 'improve response times', 'make it more robust'],
          'mobile-app': ['enhance the mobile experience', 'improve app performance', 'make it smoother'],
          'cli-tool': ['improve the CLI', 'make it more intuitive', 'enhance usability']
        }
      },

      missing_context: {
        name: 'Missing Context',
        description: 'Prompts lacking necessary background information and technical context',
        purpose: 'Test ability to identify and add relevant contextual information',
        priority: 'critical',

        characteristics: [
          'No mention of target technology or framework',
          'Missing project requirements',
          'Lack of environment specifications',
          'No mention of constraints or limitations',
          'Missing user/audience context'
        ],

        examples: {
          simple: [
            'Create a button',
            'Write a function',
            'Add validation',
            'Implement search',
            'Create a form'
          ],
          moderate: [
            'Build a {component_type}',
            'Implement {feature_name}',
            'Create a data processing pipeline',
            'Add authentication',
            'Build a dashboard'
          ],
          complex: [
            'Design a microservices architecture',
            'Implement a recommendation system',
            'Create a real-time analytics platform',
            'Build a distributed caching layer',
            'Design a multi-tenant application'
          ]
        },

        expectedImprovements: [
          'Add technology stack specifications',
          'Include project requirements and constraints',
          'Specify target environment',
          'Add user context and use cases',
          'Include performance and scale requirements'
        ],

        detectionPatterns: [
          /^(create|build|implement|add|write)\s+\w+$/gi,
          /\b(component|function|feature|system|app)\b(?!\s+(using|with|for|in))/gi,
          /^[^{]*\{[^}]+\}[^{]*$/,  // Only has template variables, no context
        ],

        contextualVariations: {
          'web-application': ['create a component', 'add a feature', 'implement routing'],
          'api-service': ['create an endpoint', 'add middleware', 'implement authentication'],
          'mobile-app': ['create a screen', 'add navigation', 'implement offline support'],
          'cli-tool': ['create a command', 'add a flag', 'implement configuration']
        }
      },

      poor_structure: {
        name: 'Poor Structure',
        description: 'Prompts with unclear organization, mixed instruction styles, or logical flow issues',
        purpose: 'Test ability to reorganize and structure prompts logically',
        priority: 'high',

        characteristics: [
          'Multiple unrelated instructions mixed together',
          'Inconsistent instruction format',
          'Poor logical flow or organization',
          'Missing clear sections or hierarchy',
          'Contradictory or conflicting requirements'
        ],

        examples: {
          simple: [
            'Create a button and also fix the CSS and make sure it validates input',
            'Write a function that handles errors and also add documentation',
            'Build a component, test it, and deploy to production',
            'Add authentication and improve the design and fix bugs',
            'Implement search, add pagination, style the results'
          ],
          moderate: [
            'Build a {component_type} with {feature_list} and ensure it works with {framework} but also make it responsive and add error handling while maintaining backwards compatibility',
            'Create an API endpoint for {use_case} and document it and add tests and implement rate limiting and also consider security',
            'Design a {system_component} handling {data_operations} with proper error handling and logging and also optimize for performance'
          ],
          complex: [
            'Architect a {architecture_pattern} for {complex_use_case} that scales to {user_count} users and handles {data_volume} while maintaining {performance_target} response times and implementing security best practices and monitoring and also considering cost optimization and team training requirements',
            'Design microservices for {domain} with event sourcing and CQRS patterns while ensuring data consistency and implementing observability and also handling failure scenarios and setting up CI/CD pipelines'
          ]
        },

        expectedImprovements: [
          'Separate different concerns into distinct sections',
          'Use consistent instruction formatting',
          'Organize content in logical sequence',
          'Use clear headings and structure',
          'Remove contradictions and conflicts'
        ],

        detectionPatterns: [
          /\band\s+(also|additionally|plus|furthermore)/gi,
          /\b(but|however|although)\s+.*\band\s+/gi,
          /(\w+)\s+and\s+(\w+)\s+and\s+(\w+)/gi,  // Multiple "and" connections
          /\b(create|build|implement|add|design)\b.*\band\s+(create|build|implement|add|design)/gi
        ],

        contextualVariations: {
          'web-application': ['build frontend and backend and database', 'create UI and add logic and implement API'],
          'api-service': ['create endpoints and add auth and implement rate limiting', 'build API and add docs and deploy'],
          'mobile-app': ['create screens and add navigation and implement offline mode', 'build app and add testing and publish'],
          'cli-tool': ['create commands and add help and implement config', 'build CLI and add documentation and distribute']
        }
      },

      missing_examples: {
        name: 'Missing Examples',
        description: 'Prompts lacking concrete examples, patterns, or demonstrations',
        purpose: 'Test ability to add relevant examples and few-shot learning patterns',
        priority: 'medium',

        characteristics: [
          'Abstract instructions without concrete examples',
          'Missing sample inputs and outputs',
          'No demonstration of expected patterns',
          'Lack of format specifications',
          'Missing reference implementations'
        ],

        examples: {
          simple: [
            'Format the data correctly',
            'Parse the input string',
            'Validate the form fields',
            'Transform the object structure',
            'Convert between formats'
          ],
          moderate: [
            'Implement {pattern_name} for {use_case}',
            'Create a {component_type} following best practices',
            'Build a parser for {data_format}',
            'Design a {architecture_pattern} implementation',
            'Transform {data_type} according to business rules'
          ],
          complex: [
            'Architect a plugin system with dynamic loading',
            'Implement a distributed consensus algorithm',
            'Design a schema migration system',
            'Create a code generation framework',
            'Build a reactive data synchronization system'
          ]
        },

        expectedImprovements: [
          'Add concrete input/output examples',
          'Include sample code or pseudocode',
          'Provide reference implementations',
          'Show expected format and structure',
          'Demonstrate patterns with examples'
        ],

        detectionPatterns: [
          /\b(format|parse|validate|transform|convert)\b(?!\s+(like|such as|for example))/gi,
          /\b(implement|create|build|design)\s+\w+(?!\s+(like|such as|example|e\.g\.))/gi,
          /\b(following|according to|based on)\s+\w+\s+(practices|rules|patterns)(?!\s+like)/gi
        ],

        contextualVariations: {
          'web-application': ['format the component props', 'parse the API response', 'validate user input'],
          'api-service': ['format the JSON response', 'parse request parameters', 'validate input data'],
          'mobile-app': ['format the data model', 'parse device information', 'validate form input'],
          'cli-tool': ['format the output', 'parse command arguments', 'validate configuration']
        }
      },

      no_output_format: {
        name: 'No Output Format',
        description: 'Prompts without specified output structure, format, or presentation requirements',
        purpose: 'Test ability to specify clear output formats and structures',
        priority: 'high',

        characteristics: [
          'No specification of output format',
          'Missing structure requirements',
          'Unclear presentation expectations',
          'No data type specifications',
          'Missing response format guidelines'
        ],

        examples: {
          simple: [
            'Analyze the data',
            'Generate a report',
            'Create documentation',
            'Provide recommendations',
            'Summarize the results'
          ],
          moderate: [
            'Analyze {data_source} for {business_metric}',
            'Generate insights from {analysis_type}',
            'Create technical documentation for {component_type}',
            'Provide optimization recommendations for {system_component}',
            'Summarize {research_findings} for stakeholders'
          ],
          complex: [
            'Conduct comprehensive analysis of {complex_system} performance',
            'Generate strategic recommendations for {business_domain} optimization',
            'Create architectural documentation for {distributed_system}',
            'Provide risk assessment for {enterprise_initiative}',
            'Summarize multi-dimensional analysis of {complex_data_set}'
          ]
        },

        expectedImprovements: [
          'Specify exact output format (JSON, XML, markdown, etc.)',
          'Define structure requirements and sections',
          'Add data type specifications',
          'Include presentation guidelines',
          'Specify response format and style'
        ],

        detectionPatterns: [
          /\b(analyze|generate|create|provide|summarize)\b(?!\s+(a|an|the|in|as))/gi,
          /\b(report|documentation|recommendations|insights|results)\b$/gi,
          /^[^{]*output[^{]*$/gi  // Mentions output but no format specified
        ],

        contextualVariations: {
          'web-application': ['generate component documentation', 'create API response', 'provide user feedback'],
          'api-service': ['generate API documentation', 'create response format', 'provide error messages'],
          'mobile-app': ['generate screen documentation', 'create data models', 'provide user notifications'],
          'cli-tool': ['generate command help', 'create output format', 'provide status messages']
        }
      },

      domain_specific: {
        name: 'Domain-Specific Context',
        description: 'Prompts requiring specialized domain knowledge and technical context',
        purpose: 'Test ability to add domain-specific technical context and requirements',
        priority: 'medium',

        characteristics: [
          'Generic requests for domain-specific implementations',
          'Missing domain-specific constraints',
          'Lack of industry-specific requirements',
          'No mention of domain patterns or practices',
          'Missing compliance or regulatory context'
        ],

        examples: {
          simple: [
            'Create a payment form',
            'Build a user profile',
            'Implement data storage',
            'Add search functionality',
            'Create a notification system'
          ],
          moderate: [
            'Implement {domain_feature} for {business_context}',
            'Create {business_component} with {industry_requirements}',
            'Build {domain_specific_workflow} handling {business_data}',
            'Design {compliance_feature} for {regulatory_context}',
            'Implement {business_logic} following {industry_standards}'
          ],
          complex: [
            'Architect {enterprise_system} for {industry_domain} with {compliance_requirements}',
            'Design {domain_specific_architecture} handling {complex_business_rules}',
            'Implement {regulatory_framework} for {financial_services}',
            'Create {healthcare_system} with {privacy_requirements}',
            'Build {educational_platform} supporting {learning_standards}'
          ]
        },

        expectedImprovements: [
          'Add domain-specific requirements and constraints',
          'Include industry-specific patterns and practices',
          'Specify compliance and regulatory requirements',
          'Add domain-specific data models and workflows',
          'Include domain expertise and context'
        ],

        detectionPatterns: [
          /\b(payment|user|data|search|notification)\b(?!\s+(using|with|for|in))/gi,
          /\b(create|build|implement)\s+\w+(?!\s+(using|with|for|in))/gi
        ],

        contextualVariations: {
          'e-commerce': ['payment processing', 'inventory management', 'order fulfillment', 'customer analytics'],
          'fintech': ['fraud detection', 'risk assessment', 'compliance monitoring', 'transaction processing'],
          'healthcare': ['patient data management', 'clinical workflows', 'HIPAA compliance', 'medical records'],
          'education': ['learning management', 'student assessment', 'curriculum tracking', 'accessibility compliance']
        }
      }
    };
  }

  /**
   * Calculate category distribution for test generation
   * @param {number} totalTests - Total number of tests to generate
   * @param {Object} projectContext - Project context from Phase 2 analysis
   * @returns {Object} Distribution counts by category
   */
  calculateCategoryDistribution(totalTests, projectContext = {}) {
    // Start with base weights
    let weights = { ...this.config.categoryWeights };

    // Apply context-based adjustments
    weights = this.applyContextAdjustments(weights, projectContext);

    // Normalize weights to sum to 1
    weights = this.normalizeWeights(weights);

    // Calculate initial distribution
    const distribution = {};
    Object.keys(weights).forEach(category => {
      distribution[category] = Math.round(totalTests * weights[category]);
    });

    // Ensure minimum counts per category
    this.enforceMinimumCounts(distribution, totalTests);

    // Adjust for exact total
    this.adjustForExactTotal(distribution, totalTests);

    return {
      distribution,
      weights,
      totalTests,
      percentages: this.calculatePercentages(distribution, totalTests),
      metadata: {
        adjustments: this.getAppliedAdjustments(projectContext),
        contextInfluence: this.calculateContextInfluence(projectContext),
        coverage: this.calculateCoverage(distribution)
      }
    };
  }

  /**
   * Apply context-based weight adjustments
   * @param {Object} weights - Base category weights
   * @param {Object} projectContext - Project context
   * @returns {Object} Adjusted weights
   */
  applyContextAdjustments(weights, projectContext) {
    let adjustedWeights = { ...weights };

    // Apply domain-specific adjustments
    if (projectContext.domain && this.config.contextAdjustments.domainAdjustments[projectContext.domain]) {
      const domainAdjustments = this.config.contextAdjustments.domainAdjustments[projectContext.domain];
      Object.keys(domainAdjustments).forEach(category => {
        if (adjustedWeights[category]) {
          adjustedWeights[category] *= domainAdjustments[category];
        }
      });
    }

    // Apply project type adjustments
    if (projectContext.projectType && this.config.contextAdjustments.projectTypeAdjustments[projectContext.projectType]) {
      const typeAdjustments = this.config.contextAdjustments.projectTypeAdjustments[projectContext.projectType];
      Object.keys(typeAdjustments).forEach(category => {
        if (adjustedWeights[category]) {
          adjustedWeights[category] *= typeAdjustments[category];
        }
      });
    }

    // Framework-specific adjustments
    if (projectContext.frameworks?.length > 0) {
      adjustedWeights = this.applyFrameworkAdjustments(adjustedWeights, projectContext.frameworks);
    }

    // Complexity-based adjustments
    if (projectContext.complexity) {
      adjustedWeights = this.applyComplexityAdjustments(adjustedWeights, projectContext.complexity);
    }

    return adjustedWeights;
  }

  /**
   * Apply framework-specific adjustments
   * @param {Object} weights - Current weights
   * @param {Array} frameworks - Project frameworks
   * @returns {Object} Adjusted weights
   */
  applyFrameworkAdjustments(weights, frameworks) {
    const frameworkAdjustments = {
      // Frontend frameworks need more structure guidance
      react: { poor_structure: 1.2, missing_context: 1.1 },
      vue: { poor_structure: 1.1, missing_examples: 1.2 },
      angular: { missing_context: 1.3, poor_structure: 1.1 },

      // Backend frameworks need more output format specification
      express: { no_output_format: 1.3, missing_context: 1.1 },
      django: { no_output_format: 1.2, missing_examples: 1.1 },
      flask: { missing_context: 1.2, vague_instructions: 1.1 },

      // Testing frameworks need more examples
      jest: { missing_examples: 1.4, no_output_format: 1.2 },
      cypress: { missing_examples: 1.3, poor_structure: 1.1 },

      // API frameworks need output format specification
      graphql: { no_output_format: 1.4, missing_examples: 1.2 },
      fastapi: { no_output_format: 1.3, missing_context: 1.1 }
    };

    frameworks.forEach(framework => {
      const normalizedFramework = framework.toLowerCase().replace(/[^a-z]/g, '');
      if (frameworkAdjustments[normalizedFramework]) {
        const adjustments = frameworkAdjustments[normalizedFramework];
        Object.keys(adjustments).forEach(category => {
          if (weights[category]) {
            weights[category] *= adjustments[category];
          }
        });
      }
    });

    return weights;
  }

  /**
   * Apply complexity-based adjustments
   * @param {Object} weights - Current weights
   * @param {string} complexity - Project complexity
   * @returns {Object} Adjusted weights
   */
  applyComplexityAdjustments(weights, complexity) {
    const complexityAdjustments = {
      simple: {
        vague_instructions: 1.3,
        missing_examples: 1.2,
        poor_structure: 0.8,
        missing_context: 0.9
      },
      medium: {
        missing_context: 1.2,
        poor_structure: 1.1,
        no_output_format: 1.1
      },
      high: {
        missing_context: 1.4,
        poor_structure: 1.3,
        domain_specific: 1.2,
        no_output_format: 1.2
      }
    };

    if (complexityAdjustments[complexity]) {
      const adjustments = complexityAdjustments[complexity];
      Object.keys(adjustments).forEach(category => {
        if (weights[category]) {
          weights[category] *= adjustments[category];
        }
      });
    }

    return weights;
  }

  /**
   * Normalize weights to sum to 1
   * @param {Object} weights - Weights to normalize
   * @returns {Object} Normalized weights
   */
  normalizeWeights(weights) {
    const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
    const normalized = {};
    Object.keys(weights).forEach(category => {
      normalized[category] = weights[category] / total;
    });
    return normalized;
  }

  /**
   * Enforce minimum counts per category
   * @param {Object} distribution - Current distribution
   * @param {number} totalTests - Total test count
   */
  enforceMinimumCounts(distribution, totalTests) {
    const categories = Object.keys(distribution);
    const minPerCategory = this.config.minimumPerCategory;
    
    // Check if minimums can be satisfied
    const totalMinimum = categories.length * minPerCategory;
    if (totalMinimum > totalTests) {
      // Reduce minimum proportionally
      const adjustedMin = Math.floor(totalTests / categories.length);
      categories.forEach(category => {
        distribution[category] = Math.max(1, adjustedMin);
      });
      return;
    }

    // Enforce minimums
    let totalAdjustment = 0;
    categories.forEach(category => {
      if (distribution[category] < minPerCategory) {
        totalAdjustment += minPerCategory - distribution[category];
        distribution[category] = minPerCategory;
      }
    });

    // Redistribute excess
    if (totalAdjustment > 0) {
      this.redistributeExcess(distribution, totalAdjustment, categories);
    }
  }

  /**
   * Redistribute excess to maintain total
   * @param {Object} distribution - Current distribution
   * @param {number} excess - Excess to redistribute
   * @param {Array} categories - Available categories
   */
  redistributeExcess(distribution, excess, categories) {
    // Remove from categories with counts above minimum, starting with largest
    const sortedCategories = categories
      .filter(cat => distribution[cat] > this.config.minimumPerCategory)
      .sort((a, b) => distribution[b] - distribution[a]);

    let remaining = excess;
    for (const category of sortedCategories) {
      if (remaining <= 0) break;
      
      const available = distribution[category] - this.config.minimumPerCategory;
      const reduction = Math.min(available, remaining);
      
      distribution[category] -= reduction;
      remaining -= reduction;
    }
  }

  /**
   * Adjust distribution for exact total
   * @param {Object} distribution - Current distribution
   * @param {number} totalTests - Target total
   */
  adjustForExactTotal(distribution, totalTests) {
    const currentTotal = Object.values(distribution).reduce((sum, count) => sum + count, 0);
    const difference = totalTests - currentTotal;

    if (difference === 0) return;

    const categories = Object.keys(distribution).sort((a, b) => distribution[b] - distribution[a]);

    if (difference > 0) {
      // Add tests to categories in round-robin fashion
      for (let i = 0; i < difference; i++) {
        distribution[categories[i % categories.length]]++;
      }
    } else {
      // Remove tests from largest categories
      for (let i = 0; i < Math.abs(difference); i++) {
        const category = categories[i % categories.length];
        if (distribution[category] > this.config.minimumPerCategory) {
          distribution[category]--;
        }
      }
    }
  }

  /**
   * Calculate percentages for distribution
   * @param {Object} distribution - Test distribution
   * @param {number} totalTests - Total test count
   * @returns {Object} Percentage distribution
   */
  calculatePercentages(distribution, totalTests) {
    const percentages = {};
    Object.keys(distribution).forEach(category => {
      percentages[category] = Math.round((distribution[category] / totalTests) * 100);
    });
    return percentages;
  }

  /**
   * Get applied adjustments summary
   * @param {Object} projectContext - Project context
   * @returns {Array} Applied adjustments
   */
  getAppliedAdjustments(projectContext) {
    const adjustments = [];

    if (projectContext.domain) {
      adjustments.push(`Domain: ${projectContext.domain}`);
    }
    if (projectContext.projectType) {
      adjustments.push(`Project type: ${projectContext.projectType}`);
    }
    if (projectContext.frameworks?.length > 0) {
      adjustments.push(`Frameworks: ${projectContext.frameworks.slice(0, 3).join(', ')}`);
    }
    if (projectContext.complexity) {
      adjustments.push(`Complexity: ${projectContext.complexity}`);
    }

    return adjustments;
  }

  /**
   * Calculate context influence on distribution
   * @param {Object} projectContext - Project context
   * @returns {number} Influence score (0-1)
   */
  calculateContextInfluence(projectContext) {
    let influence = 0;
    let factors = 0;

    if (projectContext.domain) {
      influence += 0.3;
      factors++;
    }
    if (projectContext.projectType) {
      influence += 0.25;
      factors++;
    }
    if (projectContext.frameworks?.length > 0) {
      influence += 0.25;
      factors++;
    }
    if (projectContext.complexity) {
      influence += 0.2;
      factors++;
    }

    return factors > 0 ? influence / factors : 0;
  }

  /**
   * Calculate coverage completeness
   * @param {Object} distribution - Test distribution
   * @returns {Object} Coverage metrics
   */
  calculateCoverage(distribution) {
    const totalCategories = Object.keys(this.categories).length;
    const coveredCategories = Object.keys(distribution).filter(cat => distribution[cat] > 0).length;
    
    return {
      totalCategories,
      coveredCategories,
      coveragePercentage: Math.round((coveredCategories / totalCategories) * 100),
      uncoveredCategories: Object.keys(this.categories).filter(cat => !distribution[cat] || distribution[cat] === 0)
    };
  }

  /**
   * Generate test prompts for a specific category
   * @param {string} category - Category name
   * @param {string} complexity - Complexity level
   * @param {Object} context - Project context
   * @param {number} count - Number of prompts to generate
   * @returns {Array} Generated prompts
   */
  generateCategoryPrompts(category, complexity, context, count = 1) {
    if (!this.categories[category]) {
      throw new Error(`Unknown category: ${category}`);
    }

    const categoryDef = this.categories[category];
    const examples = categoryDef.examples[complexity] || categoryDef.examples.simple;
    const contextualVariations = categoryDef.contextualVariations?.[context.projectType] || [];
    
    const prompts = [];
    const usedPrompts = new Set();

    // Generate from base examples
    while (prompts.length < count && prompts.length < examples.length * 2) {
      let prompt = this.selectRandomFromArray([...examples, ...contextualVariations]);
      
      // Apply context-specific adaptations
      prompt = this.adaptPromptToContext(prompt, context, category);
      
      if (!usedPrompts.has(prompt)) {
        prompts.push({
          prompt,
          category,
          complexity,
          expectedImprovements: categoryDef.expectedImprovements,
          characteristics: categoryDef.characteristics,
          detectionPatterns: categoryDef.detectionPatterns.map(p => p.source),
          purpose: categoryDef.purpose
        });
        usedPrompts.add(prompt);
      }
    }

    return prompts;
  }

  /**
   * Adapt prompt to project context
   * @param {string} prompt - Base prompt
   * @param {Object} context - Project context
   * @param {string} category - Category name
   * @returns {string} Adapted prompt
   */
  adaptPromptToContext(prompt, context, category) {
    let adaptedPrompt = prompt;

    // Replace basic context variables
    const contextMappings = {
      component_type: this.getContextualComponentType(context, category),
      feature_name: this.getContextualFeatureName(context, category),
      business_context: context.domain || 'business application',
      framework: context.frameworks?.[0] || 'appropriate framework',
      language: context.languages?.[0] || 'appropriate language'
    };

    // Apply context mappings
    Object.keys(contextMappings).forEach(variable => {
      const pattern = new RegExp(`\\{${variable}\\}`, 'gi');
      adaptedPrompt = adaptedPrompt.replace(pattern, contextMappings[variable]);
    });

    return adaptedPrompt;
  }

  /**
   * Get contextual component type
   * @param {Object} context - Project context
   * @param {string} category - Category name
   * @returns {string} Component type
   */
  getContextualComponentType(context, category) {
    const projectTypeComponents = {
      'web-application': ['React component', 'Vue component', 'form component', 'layout component'],
      'api-service': ['API endpoint', 'middleware', 'service class', 'data model'],
      'mobile-app': ['screen component', 'navigation component', 'native module', 'data service'],
      'cli-tool': ['command handler', 'configuration parser', 'utility function', 'plugin system']
    };

    const components = projectTypeComponents[context.projectType] || ['component', 'module', 'service'];
    return this.selectRandomFromArray(components);
  }

  /**
   * Get contextual feature name
   * @param {Object} context - Project context
   * @param {string} category - Category name
   * @returns {string} Feature name
   */
  getContextualFeatureName(context, category) {
    const domainFeatures = {
      'e-commerce': ['shopping cart', 'product catalog', 'checkout flow', 'payment processing'],
      'fintech': ['transaction processing', 'fraud detection', 'risk assessment', 'compliance monitoring'],
      'healthcare': ['patient management', 'appointment scheduling', 'medical records', 'billing system'],
      'education': ['course management', 'student assessment', 'grade tracking', 'learning analytics']
    };

    const features = domainFeatures[context.domain] || ['user management', 'data processing', 'notification system'];
    return this.selectRandomFromArray(features);
  }

  /**
   * Detect category from prompt text
   * @param {string} prompt - Prompt to analyze
   * @returns {Object} Detection results
   */
  detectCategory(prompt) {
    const results = {};
    const scores = {};

    Object.keys(this.categories).forEach(categoryName => {
      const category = this.categories[categoryName];
      let score = 0;

      // Check detection patterns
      category.detectionPatterns.forEach(pattern => {
        const matches = prompt.match(pattern);
        if (matches) {
          score += matches.length;
        }
      });

      scores[categoryName] = score;
      if (score > 0) {
        results[categoryName] = {
          score,
          confidence: Math.min(score / 3, 1), // Normalize to 0-1
          description: category.description,
          expectedImprovements: category.expectedImprovements
        };
      }
    });

    // Find best match
    const bestMatch = Object.keys(scores).reduce((best, category) => 
      scores[category] > scores[best] ? category : best
    );

    return {
      detectedCategories: results,
      bestMatch: scores[bestMatch] > 0 ? bestMatch : null,
      confidence: scores[bestMatch] / Math.max(...Object.values(scores), 1),
      allScores: scores
    };
  }

  /**
   * Validate category coverage for test set
   * @param {Array} testCases - Array of test cases
   * @returns {Object} Coverage validation results
   */
  validateCoverage(testCases) {
    const categoryCounts = {};
    const coverageGaps = [];
    
    // Count tests per category
    testCases.forEach(testCase => {
      const category = testCase.category || 'unknown';
      categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    });

    // Check for gaps
    Object.keys(this.categories).forEach(category => {
      if (!categoryCounts[category] || categoryCounts[category] < this.config.minimumPerCategory) {
        coverageGaps.push({
          category,
          current: categoryCounts[category] || 0,
          minimum: this.config.minimumPerCategory,
          deficit: this.config.minimumPerCategory - (categoryCounts[category] || 0)
        });
      }
    });

    return {
      valid: coverageGaps.length === 0,
      categoryCounts,
      coverageGaps,
      totalTests: testCases.length,
      coveragePercentage: Math.round((Object.keys(categoryCounts).length / Object.keys(this.categories).length) * 100),
      recommendations: this.generateCoverageRecommendations(coverageGaps, categoryCounts)
    };
  }

  /**
   * Generate coverage recommendations
   * @param {Array} gaps - Coverage gaps
   * @param {Object} counts - Current counts
   * @returns {Array} Recommendations
   */
  generateCoverageRecommendations(gaps, counts) {
    const recommendations = [];

    if (gaps.length > 0) {
      recommendations.push(`Add ${gaps.reduce((sum, gap) => sum + gap.deficit, 0)} more tests to cover all categories`);
      
      gaps.forEach(gap => {
        recommendations.push(`Generate ${gap.deficit} more tests for '${gap.category}' category`);
      });
    }

    // Check for imbalanced distribution
    const values = Object.values(counts);
    const max = Math.max(...values);
    const min = Math.min(...values);
    
    if (max > min * 3) {
      recommendations.push('Consider rebalancing test distribution - some categories are over-represented');
    }

    if (recommendations.length === 0) {
      recommendations.push('Category coverage meets all requirements');
    }

    return recommendations;
  }

  /**
   * Select random element from array
   * @param {Array} array - Input array
   * @returns {*} Random element
   */
  selectRandomFromArray(array) {
    if (!array || array.length === 0) return null;
    return array[Math.floor(Math.random() * array.length)];
  }

  /**
   * Get category details
   * @param {string} categoryName - Category name
   * @returns {Object} Category details
   */
  getCategoryDetails(categoryName) {
    return this.categories[categoryName] || null;
  }

  /**
   * Get all categories
   * @returns {Object} All category definitions
   */
  getAllCategories() {
    return this.categories;
  }

  /**
   * Generate category coverage report
   * @param {Object} distribution - Distribution result
   * @returns {Object} Coverage report
   */
  generateCoverageReport(distribution) {
    return {
      summary: {
        totalTests: distribution.totalTests,
        categoriesCovered: Object.keys(distribution.distribution).length,
        totalCategories: Object.keys(this.categories).length,
        coveragePercentage: distribution.metadata.coverage.coveragePercentage
      },
      distribution: Object.keys(distribution.distribution).map(category => ({
        category,
        count: distribution.distribution[category],
        percentage: distribution.percentages[category],
        description: this.categories[category]?.description || 'Unknown category',
        priority: this.categories[category]?.priority || 'medium'
      })),
      contextInfluence: {
        adjustments: distribution.metadata.adjustments,
        influenceScore: Math.round(distribution.metadata.contextInfluence * 100)
      },
      recommendations: this.generateDistributionRecommendations(distribution)
    };
  }

  /**
   * Generate distribution recommendations
   * @param {Object} distribution - Distribution result
   * @returns {Array} Recommendations
   */
  generateDistributionRecommendations(distribution) {
    const recommendations = [];
    const coverage = distribution.metadata.coverage;

    if (coverage.coveragePercentage < 100) {
      recommendations.push(`Add tests for uncovered categories: ${coverage.uncoveredCategories.join(', ')}`);
    }

    const contextInfluence = distribution.metadata.contextInfluence;
    if (contextInfluence > 0.7) {
      recommendations.push('High context influence detected - distribution well-adapted to project characteristics');
    } else if (contextInfluence < 0.3) {
      recommendations.push('Low context influence - consider adding more project-specific context');
    }

    if (recommendations.length === 0) {
      recommendations.push('Category distribution looks optimal for the given context');
    }

    return recommendations;
  }
}

module.exports = CategoryCoverage;