/**
 * Universal Prompt Templates
 * Flexible templates that adapt to detected project context
 */

class PromptTemplates {
  constructor() {
    // Variable substitution patterns
    this.variablePattern = /\{([^}]+)\}/g;
    
    // Core template categories with context-aware variations
    this.templates = {
      debugging: {
        simple: [
          "Fix this {language} error: {error_type}",
          "Debug this {language} issue: {problem_description}",
          "Resolve the {error_type} in this {language} code",
          "Help fix this {language} problem: {issue_details}"
        ],
        moderate: [
          "Debug a {language} {component_type} that {problem_description}",
          "Troubleshoot this {framework} {component_type} experiencing {issue_type}",
          "Fix the {language} {component_type} that fails when {failure_condition}",
          "Resolve performance issues in {language} {component_type} during {operation_context}"
        ],
        complex: [
          "Troubleshoot {framework} {architecture_pattern} with {complex_issue}",
          "Debug distributed {language} system where {complex_problem} across {system_components}",
          "Analyze and fix race condition in {language} {architecture_pattern} affecting {system_behavior}",
          "Resolve memory leaks in {framework} application with {complex_architecture} when {usage_pattern}"
        ]
      },

      creation: {
        simple: [
          "Create a {component_type} for {basic_use_case}",
          "Build a {language} {component_type} that {simple_action}",
          "Make a {component_type} for {simple_requirement}",
          "Write a {language} function that {basic_functionality}"
        ],
        moderate: [
          "Build a {component_type} that {functional_requirement} using {framework}",
          "Create a {language} {component_type} with {feature_list} for {use_case}",
          "Implement a {component_type} that {behavior_description} in {framework}",
          "Design a {language} {component_type} handling {data_operations} with {constraints}"
        ],
        complex: [
          "Design a {architecture_component} for {complex_use_case} with {constraints}",
          "Architect a scalable {language} {system_type} supporting {advanced_features} with {performance_requirements}",
          "Build a distributed {component_type} handling {complex_operations} across {system_scale}",
          "Create enterprise-grade {framework} {architecture_pattern} managing {complex_data_flow} with {security_requirements}"
        ]
      },

      optimization: {
        simple: [
          "Improve {code_element} performance",
          "Optimize this {language} {code_element}",
          "Make this {component_type} faster",
          "Enhance {code_element} efficiency"
        ],
        moderate: [
          "Optimize {component} for {performance_metric} in {framework}",
          "Improve {language} {component_type} to achieve {performance_target}",
          "Enhance {component} performance handling {data_volume} with {constraints}",
          "Optimize {framework} {component_type} for {specific_use_case} scenarios"
        ],
        complex: [
          "Architect {system_component} for {scale_requirement} with {trade_offs}",
          "Optimize {framework} application for {complex_performance_requirements} across {deployment_environment}",
          "Design high-performance {language} system handling {massive_scale} with {strict_constraints}",
          "Architect fault-tolerant {system_type} optimized for {complex_requirements} in {distributed_environment}"
        ]
      },

      testing: {
        simple: [
          "Write tests for {function_name}",
          "Create {language} tests for {component_type}",
          "Test this {language} function: {function_description}",
          "Add basic tests for {component_name}"
        ],
        moderate: [
          "Create {test_type} tests for {component} using {testing_framework}",
          "Write comprehensive tests for {language} {component_type} covering {test_scenarios}",
          "Implement {testing_strategy} for {framework} {component} with {test_requirements}",
          "Design test suite for {component_type} handling {data_operations} in {framework}"
        ],
        complex: [
          "Design comprehensive test strategy for {system} including {test_types}",
          "Create end-to-end testing framework for {complex_system} covering {test_scenarios}",
          "Architect testing infrastructure for {distributed_system} with {testing_requirements}",
          "Design performance testing suite for {large_scale_system} validating {complex_requirements}"
        ]
      },

      refactoring: {
        simple: [
          "Refactor this {language} code",
          "Clean up this {component_type}",
          "Improve this {language} function",
          "Simplify this {code_element}"
        ],
        moderate: [
          "Refactor {component_type} to follow {design_pattern} in {framework}",
          "Restructure {language} {component} for better {code_quality_aspect}",
          "Modernize {legacy_component} to use {modern_approach} in {framework}",
          "Refactor {component_type} to implement {architectural_pattern}"
        ],
        complex: [
          "Architect migration from {legacy_system} to {modern_architecture}",
          "Refactor monolithic {language} application into {microservices_architecture}",
          "Modernize legacy {framework} codebase implementing {architectural_patterns}",
          "Design refactoring strategy for {large_codebase} transitioning to {new_architecture}"
        ]
      },

      documentation: {
        simple: [
          "Document this {language} function",
          "Write docs for {component_type}",
          "Create README for {project_type}",
          "Add comments to {code_element}"
        ],
        moderate: [
          "Create API documentation for {service_type} using {documentation_format}",
          "Write comprehensive guide for {framework} {component} including {doc_sections}",
          "Document {architecture_pattern} implementation with {documentation_requirements}",
          "Create developer documentation for {system_component} covering {usage_scenarios}"
        ],
        complex: [
          "Design documentation strategy for {enterprise_system} including {stakeholder_types}",
          "Create comprehensive documentation suite for {complex_platform} supporting {user_types}",
          "Architect knowledge management system for {large_project} with {documentation_requirements}",
          "Design interactive documentation platform for {complex_api} with {advanced_features}"
        ]
      },

      security: {
        simple: [
          "Secure this {language} function",
          "Add security to {component_type}",
          "Fix security issue in {code_element}",
          "Validate input in {language} {component}"
        ],
        moderate: [
          "Implement {security_pattern} in {framework} {component}",
          "Secure {api_type} endpoints using {authentication_method}",
          "Add {security_measures} to {component_type} handling {sensitive_data}",
          "Implement secure {data_operation} in {framework} with {security_requirements}"
        ],
        complex: [
          "Design security architecture for {enterprise_system} with {security_requirements}",
          "Implement zero-trust security model for {distributed_system}",
          "Architect secure {microservices_system} with {advanced_security_features}",
          "Design comprehensive security framework for {complex_platform} handling {sensitive_operations}"
        ]
      },

      integration: {
        simple: [
          "Connect {service_a} with {service_b}",
          "Integrate {language} app with {external_service}",
          "Add {service_name} to {project_type}",
          "Link {component_a} and {component_b}"
        ],
        moderate: [
          "Integrate {framework} application with {external_api} using {integration_pattern}",
          "Connect {database_type} to {framework} app with {data_requirements}",
          "Implement {integration_type} between {system_a} and {system_b}",
          "Design API integration for {service_type} handling {data_operations}"
        ],
        complex: [
          "Architect enterprise integration platform connecting {multiple_systems}",
          "Design event-driven architecture integrating {distributed_services}",
          "Implement complex data pipeline connecting {data_sources} to {analytics_platform}",
          "Architect microservices integration layer with {advanced_patterns}"
        ]
      },

      deployment: {
        simple: [
          "Deploy {project_type} to {platform}",
          "Set up {language} app deployment",
          "Configure {framework} for production",
          "Deploy {component} to {environment}"
        ],
        moderate: [
          "Deploy {framework} application to {cloud_platform} with {deployment_requirements}",
          "Set up CI/CD pipeline for {project_type} using {deployment_tools}",
          "Configure {containerization} deployment for {framework} app",
          "Implement blue-green deployment for {service_type} on {platform}"
        ],
        complex: [
          "Design multi-region deployment strategy for {enterprise_application}",
          "Architect auto-scaling infrastructure for {high_traffic_system}",
          "Implement disaster recovery deployment for {critical_system}",
          "Design GitOps deployment pipeline for {microservices_architecture}"
        ]
      }
    };

    // Context-specific variable mappings
    this.contextMappings = {
      // Programming languages
      languages: {
        javascript: ['JavaScript', 'JS', 'Node.js'],
        typescript: ['TypeScript', 'TS'],
        python: ['Python', 'Python 3'],
        java: ['Java', 'OpenJDK'],
        go: ['Go', 'Golang'],
        rust: ['Rust'],
        php: ['PHP'],
        ruby: ['Ruby'],
        swift: ['Swift'],
        kotlin: ['Kotlin'],
        scala: ['Scala'],
        csharp: ['C#', '.NET']
      },

      // Framework-specific terms
      frameworks: {
        react: {
          component_type: ['React component', 'functional component', 'class component', 'custom hook'],
          architecture_pattern: ['component composition', 'render props', 'higher-order components', 'context pattern'],
          testing_framework: ['Jest', 'React Testing Library', 'Enzyme'],
          performance_metric: ['render performance', 'bundle size', 'time to interactive']
        },
        vue: {
          component_type: ['Vue component', 'composition API component', 'options API component'],
          architecture_pattern: ['single file components', 'composables', 'provide/inject'],
          testing_framework: ['Vue Test Utils', 'Jest'],
          performance_metric: ['reactivity performance', 'bundle size', 'hydration time']
        },
        angular: {
          component_type: ['Angular component', 'directive', 'pipe', 'service'],
          architecture_pattern: ['dependency injection', 'module federation', 'lazy loading'],
          testing_framework: ['Jasmine', 'Karma', 'Protractor'],
          performance_metric: ['change detection', 'bundle size', 'Time to Interactive']
        },
        express: {
          component_type: ['Express middleware', 'route handler', 'controller', 'service'],
          architecture_pattern: ['MVC pattern', 'middleware chain', 'router composition'],
          testing_framework: ['Mocha', 'Chai', 'Supertest'],
          performance_metric: ['response time', 'throughput', 'memory usage']
        },
        django: {
          component_type: ['Django view', 'model', 'serializer', 'middleware'],
          architecture_pattern: ['MVT pattern', 'class-based views', 'function-based views'],
          testing_framework: ['Django TestCase', 'pytest-django'],
          performance_metric: ['query performance', 'response time', 'memory usage']
        },
        flask: {
          component_type: ['Flask route', 'blueprint', 'extension', 'decorator'],
          architecture_pattern: ['application factory', 'blueprints', 'context locals'],
          testing_framework: ['pytest', 'Flask-Testing'],
          performance_metric: ['response time', 'memory usage', 'concurrent requests']
        }
      },

      // Domain-specific terms
      domains: {
        'e-commerce': {
          component_type: ['shopping cart', 'product catalog', 'payment processor', 'order management'],
          use_case: ['product search', 'checkout flow', 'inventory management', 'order tracking'],
          data_operations: ['product filtering', 'cart calculations', 'payment processing', 'order fulfillment']
        },
        fintech: {
          component_type: ['transaction processor', 'risk analyzer', 'compliance checker', 'payment gateway'],
          use_case: ['fraud detection', 'automated trading', 'loan processing', 'regulatory reporting'],
          data_operations: ['transaction validation', 'risk assessment', 'compliance monitoring', 'audit trailing']
        },
        healthcare: {
          component_type: ['patient portal', 'medical record system', 'appointment scheduler', 'billing system'],
          use_case: ['patient management', 'medical imaging', 'telemedicine', 'clinical workflows'],
          data_operations: ['patient data processing', 'medical record management', 'appointment scheduling', 'billing calculations']
        },
        education: {
          component_type: ['learning management system', 'gradebook', 'assignment tracker', 'student portal'],
          use_case: ['course management', 'student assessment', 'online learning', 'progress tracking'],
          data_operations: ['grade calculations', 'progress tracking', 'assignment submission', 'student analytics']
        }
      },

      // Project type specific terms
      projectTypes: {
        'web-application': {
          architecture_component: ['frontend application', 'backend API', 'database layer', 'authentication service'],
          system_type: ['SPA', 'PWA', 'SSR application', 'JAMstack site'],
          deployment_environment: ['cloud hosting', 'CDN', 'serverless platform', 'container orchestration']
        },
        'api-service': {
          architecture_component: ['REST API', 'GraphQL API', 'microservice', 'API gateway'],
          system_type: ['RESTful service', 'GraphQL server', 'microservice architecture', 'event-driven system'],
          deployment_environment: ['container platform', 'serverless functions', 'API management platform']
        },
        'mobile-app': {
          architecture_component: ['mobile app', 'native module', 'cross-platform component', 'mobile backend'],
          system_type: ['native app', 'hybrid app', 'cross-platform app', 'mobile-first PWA'],
          deployment_environment: ['app store', 'enterprise distribution', 'mobile device management']
        },
        'cli-tool': {
          architecture_component: ['command parser', 'configuration manager', 'plugin system', 'output formatter'],
          system_type: ['command line interface', 'developer tool', 'automation script', 'build tool'],
          deployment_environment: ['package manager', 'binary distribution', 'container image']
        }
      }
    };
  }

  /**
   * Generate template variations for a specific category and complexity
   * @param {string} category - Template category (debugging, creation, etc.)
   * @param {string} complexity - Complexity level (simple, moderate, complex)
   * @param {Object} context - Project context from Phase 2 analysis
   * @param {number} count - Number of templates to generate
   * @returns {Array} Array of template strings
   */
  generateTemplates(category, complexity, context, count = 5) {
    const baseTemplates = this.templates[category]?.[complexity] || [];
    if (baseTemplates.length === 0) {
      throw new Error(`No templates found for category: ${category}, complexity: ${complexity}`);
    }

    const templates = [];
    const usedTemplates = new Set();

    // Generate requested number of unique templates
    while (templates.length < count && templates.length < baseTemplates.length * 3) {
      const baseTemplate = this.selectRandomTemplate(baseTemplates);
      const contextualTemplate = this.adaptTemplateToContext(baseTemplate, context);
      
      if (!usedTemplates.has(contextualTemplate)) {
        templates.push(contextualTemplate);
        usedTemplates.add(contextualTemplate);
      }
    }

    return templates;
  }

  /**
   * Adapt a template to specific project context
   * @param {string} template - Base template string
   * @param {Object} context - Project context
   * @returns {string} Context-adapted template
   */
  adaptTemplateToContext(template, context) {
    let adaptedTemplate = template;

    // Replace placeholders with context-specific values
    adaptedTemplate = adaptedTemplate.replace(this.variablePattern, (match, variable) => {
      const replacement = this.getContextualReplacement(variable, context);
      return replacement || match; // Keep original if no replacement found
    });

    return adaptedTemplate;
  }

  /**
   * Get contextual replacement for a template variable
   * @param {string} variable - Variable name
   * @param {Object} context - Project context
   * @returns {string|null} Replacement value or null
   */
  getContextualReplacement(variable, context) {
    // Primary language
    if (variable === 'language' && context.languages?.length > 0) {
      return this.selectRandomFromArray(context.languages);
    }

    // Framework selection
    if (variable === 'framework' && context.frameworks?.length > 0) {
      return this.selectRandomFromArray(context.frameworks);
    }

    // Project type
    if (variable === 'project_type') {
      return context.projectType || 'application';
    }

    // Domain-specific replacements
    if (variable === 'domain') {
      return context.domain || 'general';
    }

    // Component type based on framework and domain
    if (variable === 'component_type') {
      return this.getComponentType(context);
    }

    // Architecture pattern based on project context
    if (variable === 'architecture_pattern') {
      return this.getArchitecturePattern(context);
    }

    // Testing framework based on tech stack
    if (variable === 'testing_framework') {
      return this.getTestingFramework(context);
    }

    // Performance metrics based on project type
    if (variable === 'performance_metric') {
      return this.getPerformanceMetric(context);
    }

    // Use case based on domain
    if (variable === 'use_case' || variable === 'basic_use_case') {
      return this.getUseCase(context);
    }

    // Data operations based on domain
    if (variable === 'data_operations') {
      return this.getDataOperations(context);
    }

    // Error types based on language
    if (variable === 'error_type') {
      return this.getErrorType(context);
    }

    // Generic fallbacks
    return this.getGenericReplacement(variable, context);
  }

  /**
   * Get component type based on context
   * @param {Object} context - Project context
   * @returns {string} Component type
   */
  getComponentType(context) {
    // Framework-specific component types
    const primaryFramework = context.frameworks?.[0]?.toLowerCase();
    if (primaryFramework && this.contextMappings.frameworks[primaryFramework]?.component_type) {
      return this.selectRandomFromArray(this.contextMappings.frameworks[primaryFramework].component_type);
    }

    // Domain-specific component types
    if (context.domain && this.contextMappings.domains[context.domain]?.component_type) {
      return this.selectRandomFromArray(this.contextMappings.domains[context.domain].component_type);
    }

    // Project type specific component types
    if (context.projectType && this.contextMappings.projectTypes[context.projectType]?.architecture_component) {
      return this.selectRandomFromArray(this.contextMappings.projectTypes[context.projectType].architecture_component);
    }

    // Generic fallbacks
    const genericComponents = ['component', 'module', 'service', 'utility', 'helper', 'class', 'function'];
    return this.selectRandomFromArray(genericComponents);
  }

  /**
   * Get architecture pattern based on context
   * @param {Object} context - Project context
   * @returns {string} Architecture pattern
   */
  getArchitecturePattern(context) {
    const primaryFramework = context.frameworks?.[0]?.toLowerCase();
    if (primaryFramework && this.contextMappings.frameworks[primaryFramework]?.architecture_pattern) {
      return this.selectRandomFromArray(this.contextMappings.frameworks[primaryFramework].architecture_pattern);
    }

    // Project type patterns
    const patterns = {
      'web-application': ['MVC pattern', 'component-based architecture', 'layered architecture'],
      'api-service': ['REST architecture', 'microservices', 'event-driven architecture'],
      'mobile-app': ['MVVM pattern', 'unidirectional data flow', 'modular architecture'],
      'cli-tool': ['command pattern', 'plugin architecture', 'pipeline pattern']
    };

    if (context.projectType && patterns[context.projectType]) {
      return this.selectRandomFromArray(patterns[context.projectType]);
    }

    const genericPatterns = ['modular design', 'separation of concerns', 'single responsibility principle'];
    return this.selectRandomFromArray(genericPatterns);
  }

  /**
   * Get testing framework based on context
   * @param {Object} context - Project context
   * @returns {string} Testing framework
   */
  getTestingFramework(context) {
    const primaryFramework = context.frameworks?.[0]?.toLowerCase();
    if (primaryFramework && this.contextMappings.frameworks[primaryFramework]?.testing_framework) {
      return this.selectRandomFromArray(this.contextMappings.frameworks[primaryFramework].testing_framework);
    }

    // Language-based testing frameworks
    const primaryLanguage = context.languages?.[0]?.toLowerCase();
    const languageFrameworks = {
      javascript: ['Jest', 'Mocha', 'Vitest', 'Cypress'],
      typescript: ['Jest', 'Vitest', 'Playwright'],
      python: ['pytest', 'unittest', 'nose2'],
      java: ['JUnit', 'TestNG', 'Mockito'],
      go: ['testing package', 'Testify', 'Ginkgo'],
      rust: ['built-in testing', 'Proptest'],
      php: ['PHPUnit', 'Pest'],
      ruby: ['RSpec', 'Minitest']
    };

    if (primaryLanguage && languageFrameworks[primaryLanguage]) {
      return this.selectRandomFromArray(languageFrameworks[primaryLanguage]);
    }

    return 'appropriate testing framework';
  }

  /**
   * Get performance metric based on context
   * @param {Object} context - Project context
   * @returns {string} Performance metric
   */
  getPerformanceMetric(context) {
    const primaryFramework = context.frameworks?.[0]?.toLowerCase();
    if (primaryFramework && this.contextMappings.frameworks[primaryFramework]?.performance_metric) {
      return this.selectRandomFromArray(this.contextMappings.frameworks[primaryFramework].performance_metric);
    }

    // Project type specific metrics
    const metrics = {
      'web-application': ['loading time', 'First Contentful Paint', 'Time to Interactive', 'Core Web Vitals'],
      'api-service': ['response time', 'throughput', 'latency', 'error rate'],
      'mobile-app': ['app startup time', 'battery usage', 'memory consumption', 'frame rate'],
      'cli-tool': ['execution time', 'memory usage', 'startup time', 'resource consumption']
    };

    if (context.projectType && metrics[context.projectType]) {
      return this.selectRandomFromArray(metrics[context.projectType]);
    }

    const genericMetrics = ['performance', 'speed', 'efficiency', 'resource usage'];
    return this.selectRandomFromArray(genericMetrics);
  }

  /**
   * Get use case based on domain
   * @param {Object} context - Project context
   * @returns {string} Use case
   */
  getUseCase(context) {
    if (context.domain && this.contextMappings.domains[context.domain]?.use_case) {
      return this.selectRandomFromArray(this.contextMappings.domains[context.domain].use_case);
    }

    // Generic use cases
    const genericUseCases = ['user interaction', 'data processing', 'business logic', 'workflow automation'];
    return this.selectRandomFromArray(genericUseCases);
  }

  /**
   * Get data operations based on domain
   * @param {Object} context - Project context
   * @returns {string} Data operations
   */
  getDataOperations(context) {
    if (context.domain && this.contextMappings.domains[context.domain]?.data_operations) {
      return this.selectRandomFromArray(this.contextMappings.domains[context.domain].data_operations);
    }

    const genericOperations = ['data validation', 'data transformation', 'data persistence', 'data retrieval'];
    return this.selectRandomFromArray(genericOperations);
  }

  /**
   * Get error type based on language
   * @param {Object} context - Project context
   * @returns {string} Error type
   */
  getErrorType(context) {
    const primaryLanguage = context.languages?.[0]?.toLowerCase();
    
    const languageErrors = {
      javascript: ['TypeError', 'ReferenceError', 'SyntaxError', 'RangeError'],
      typescript: ['Type error', 'Compilation error', 'Interface mismatch', 'Generic constraint error'],
      python: ['NameError', 'TypeError', 'ValueError', 'ImportError'],
      java: ['NullPointerException', 'ClassCastException', 'IllegalArgumentException'],
      go: ['nil pointer dereference', 'type assertion error', 'interface conversion'],
      rust: ['borrow checker error', 'lifetime error', 'type mismatch'],
      php: ['Fatal error', 'Parse error', 'Notice', 'Warning'],
      ruby: ['NoMethodError', 'ArgumentError', 'NameError']
    };

    if (primaryLanguage && languageErrors[primaryLanguage]) {
      return this.selectRandomFromArray(languageErrors[primaryLanguage]);
    }

    return 'runtime error';
  }

  /**
   * Get generic replacement for common variables
   * @param {string} variable - Variable name
   * @param {Object} context - Project context
   * @returns {string} Generic replacement
   */
  getGenericReplacement(variable, context) {
    const genericReplacements = {
      function_name: 'handleUserAction',
      component_name: 'UserComponent',
      service_name: 'DataService',
      problem_description: 'not working as expected',
      issue_details: 'unexpected behavior',
      basic_functionality: 'processes user input',
      simple_action: 'handles user requests',
      simple_requirement: 'basic functionality',
      failure_condition: 'user submits invalid data',
      operation_context: 'high load situations',
      functional_requirement: 'validates and processes data',
      feature_list: 'validation, processing, and error handling',
      behavior_description: 'responds to user interactions',
      constraints: 'maintaining data consistency',
      code_element: 'function logic',
      specific_use_case: 'production workloads',
      complex_use_case: 'enterprise-scale operations',
      performance_requirements: 'sub-second response times',
      advanced_features: 'real-time updates and offline support',
      system_scale: 'multiple data centers',
      security_requirements: 'end-to-end encryption',
      deployment_requirements: 'zero-downtime deployment',
      integration_requirements: 'seamless data synchronization'
    };

    return genericReplacements[variable] || variable.replace(/_/g, ' ');
  }

  /**
   * Select random template from array
   * @param {Array} templates - Array of templates
   * @returns {string} Selected template
   */
  selectRandomTemplate(templates) {
    return templates[Math.floor(Math.random() * templates.length)];
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
   * Get all available template categories
   * @returns {Array} Array of category names
   */
  getAvailableCategories() {
    return Object.keys(this.templates);
  }

  /**
   * Get available complexity levels
   * @returns {Array} Array of complexity levels
   */
  getComplexityLevels() {
    return ['simple', 'moderate', 'complex'];
  }

  /**
   * Generate template examples for documentation
   * @param {Object} context - Sample project context
   * @returns {Object} Examples by category and complexity
   */
  generateExamples(context) {
    const examples = {};
    
    for (const category of this.getAvailableCategories()) {
      examples[category] = {};
      for (const complexity of this.getComplexityLevels()) {
        try {
          examples[category][complexity] = this.generateTemplates(category, complexity, context, 2);
        } catch (error) {
          examples[category][complexity] = [`No templates available for ${category}:${complexity}`];
        }
      }
    }

    return examples;
  }

  /**
   * Validate template structure
   * @param {string} template - Template to validate
   * @returns {Object} Validation result
   */
  validateTemplate(template) {
    const variables = [];
    let match;
    const regex = new RegExp(this.variablePattern);
    
    while ((match = regex.exec(template)) !== null) {
      variables.push(match[1]);
    }

    return {
      valid: true,
      variables,
      hasVariables: variables.length > 0,
      templateLength: template.length
    };
  }
}

module.exports = PromptTemplates;