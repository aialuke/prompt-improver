/**
 * System configuration for the Universal Prompt Testing Framework
 * Centralized configuration management with validation and defaults
 */

const path = require('path');
const FileHandler = require('../utils/file-handler');
const Logger = require('../utils/logger');

class FrameworkConfig {
  constructor(configPath = null) {
    this.logger = new Logger('FrameworkConfig');
    this.fileHandler = new FileHandler();
    this.configPath = configPath;
    this.config = this.getDefaultConfig();
    
    // Load custom configuration if provided
    if (configPath) {
      this.loadConfig(configPath);
    }
  }

  /**
   * Get default framework configuration
   * @private
   */
  getDefaultConfig() {
    return {
      // Framework metadata
      framework: {
        name: 'Universal Prompt Testing Framework',
        version: '1.0.0',
        description: 'Autonomous testing framework for prompt engineering improvement'
      },

      // Core system settings
      system: {
        maxConcurrency: 3,
        defaultTimeout: 30000,
        retryAttempts: 2,
        retryDelay: 1000,
        batchSize: 10,
        maxFileSize: '10MB',
        tempDirectory: path.join(process.cwd(), 'tmp'),
        outputDirectory: path.join(process.cwd(), 'output')
      },

      // Logging configuration
      logging: {
        level: process.env.LOG_LEVEL || 'info',
        enableColors: process.env.NO_COLOR !== '1',
        logToFile: process.env.LOG_TO_FILE === '1',
        logFilePath: path.join(process.cwd(), 'logs', 'framework.log'),
        maxLogSize: '100MB',
        maxLogFiles: 5
      },

      // Project analysis settings
      analysis: {
        includeTests: true,
        includeDocs: true,
        maxFileSize: 10 * 1024 * 1024, // 10MB
        excludePatterns: [
          'node_modules/',
          '.git/',
          'dist/',
          'build/',
          '.next/',
          'coverage/',
          '*.min.js',
          '*.bundle.js'
        ],
        supportedLanguages: [
          'javascript', 'typescript', 'python', 'java', 'go', 'rust',
          'php', 'ruby', 'swift', 'kotlin', 'scala', 'clojure'
        ],
        contextInference: {
          domainKeywords: true,
          dependencyAnalysis: true,
          structureAnalysis: true,
          confidenceThreshold: 0.7
        }
      },

      // Test generation configuration
      testing: {
        defaultTestCount: 50,
        maxTestCount: 1000,
        complexityDistribution: {
          simple: 0.4,
          moderate: 0.4,
          complex: 0.2
        },
        categoryWeights: {
          vague_instructions: 0.25,
          missing_context: 0.25,
          poor_structure: 0.20,
          missing_examples: 0.15,
          no_output_format: 0.15
        },
        contextRelevanceWeight: 0.8,
        templateVariationCount: 5,
        seedPrompts: {
          simple: [
            'Create a {component_type}',
            'Fix this {language} error',
            'Write a function that {action}'
          ],
          moderate: [
            'Implement {feature} with error handling',
            'Create a {component_type} that {requirement}',
            'Build a {system_type} for {use_case}'
          ],
          complex: [
            'Design a scalable {architecture} for {domain}',
            'Optimize {component} for {performance_metric}',
            'Architect {system} with {constraints}'
          ]
        }
      },

      // Evaluation system settings
      evaluation: {
        methods: ['structural', 'llm-judge'],
        defaultMethod: 'structural',
        
        // Structural analysis weights
        structuralWeights: {
          clarity: 0.3,
          completeness: 0.25,
          specificity: 0.25,
          structure: 0.2
        },

        // LLM-as-a-judge settings
        llmJudge: {
          model: 'claude-3.5-sonnet',
          temperature: 0.1,
          maxTokens: 1000,
          batchSize: 10,
          rateLimitDelay: 1000,
          enableCaching: true,
          cacheExpiration: 3600000, // 1 hour
          fallbackToStructural: true
        },

        // Output quality testing
        outputQuality: {
          enabled: false, // Expensive, off by default
          testScenarios: 3,
          evaluationCriteria: [
            'relevance', 'completeness', 'accuracy', 
            'helpfulness', 'instruction_following'
          ]
        },

        // Quality thresholds
        thresholds: {
          minimumImprovement: 10, // 10% minimum improvement
          significanceLevel: 0.05,
          effectSizeThreshold: 0.2,
          successThreshold: 70 // 70+ score considered success
        }
      },

      // Statistical analysis settings
      statistics: {
        significanceLevel: 0.05,
        confidenceLevel: 0.95,
        minimumSampleSize: 30,
        effectSizeThreshold: 0.2,
        powerAnalysisTarget: 0.8,
        bootstrapSamples: 1000
      },

      // Rule optimization settings
      optimization: {
        autoUpdateRules: false,
        minConfidenceForUpdate: 0.8,
        maxRiskLevel: 'medium',
        backupBeforeUpdate: true,
        maxUpdatesPerCycle: 5,
        validationTestCount: 50,
        
        // Rule effectiveness tracking
        ruleTracking: {
          minApplications: 5, // Minimum applications before evaluating rule
          trackingWindow: 100, // Last N tests to consider
          performanceThreshold: 0.7 // 70% success rate threshold
        },

        // A/B testing for rule changes
        abTesting: {
          enabled: true,
          minimumSampleSize: 50,
          stratificationFactors: ['category', 'complexity', 'context'],
          maxTestDuration: 7 * 24 * 60 * 60 * 1000 // 7 days
        }
      },

      // Output and reporting settings
      output: {
        defaultFormat: 'markdown',
        supportedFormats: ['json', 'markdown', 'html', 'csv'],
        includeExamples: true,
        includeStatistics: true,
        includeRecommendations: true,
        maxExampleLength: 200,
        
        // Report sections to include
        reportSections: [
          'executiveSummary',
          'projectAnalysis',
          'testResults',
          'ruleEffectiveness',
          'optimizationRecommendations'
        ],

        // File naming patterns
        filePatterns: {
          report: 'prompt-test-report-{timestamp}',
          data: 'test-data-{timestamp}',
          backup: 'rules-backup-{timestamp}'
        }
      },

      // Integration settings
      integration: {
        // Existing improve-prompt.js integration
        improvePrompt: {
          scriptPath: './improve-prompt.js',
          rulesPath: './prompt-engineering-best-practices.json',
          timeout: 10000
        },

        // External API integrations
        apis: {
          anthropic: {
            baseUrl: 'https://api.anthropic.com',
            timeout: 30000,
            retryAttempts: 3
          },
          openai: {
            baseUrl: 'https://api.openai.com',
            timeout: 30000,
            retryAttempts: 3
          }
        },

        // CI/CD integration
        cicd: {
          enabled: false,
          reportFormats: ['json', 'markdown'],
          failOnRegression: true,
          thresholds: {
            minimumSuccessRate: 0.8,
            maximumRegressionCount: 5
          }
        }
      },

      // Performance and resource management
      performance: {
        memoryLimit: '1GB',
        cpuLimit: 80, // Percentage
        diskSpaceLimit: '5GB',
        cacheSize: '100MB',
        workerProcesses: require('os').cpus().length,
        
        // Resource monitoring
        monitoring: {
          enabled: true,
          interval: 30000, // 30 seconds
          alertThresholds: {
            memory: 0.9,   // 90% memory usage
            cpu: 0.8,      // 80% CPU usage
            disk: 0.9      // 90% disk usage
          }
        }
      },

      // Security settings
      security: {
        sanitizeInputs: true,
        maxPromptLength: 10000,
        allowedFileTypes: ['.js', '.ts', '.json', '.md', '.txt', '.yml', '.yaml'],
        blocklistPatterns: [
          'eval\\(',
          'Function\\(',
          'require\\([\'"]child_process[\'"]\\)',
          'require\\([\'"]fs[\'"]\\)',
          'process\\.exit'
        ]
      }
    };
  }

  /**
   * Load configuration from file
   * @param {string} configPath - Path to configuration file
   */
  async loadConfig(configPath) {
    try {
      this.logger.info('Loading configuration', { configPath });
      
      if (!(await this.fileHandler.exists(configPath))) {
        this.logger.warn('Configuration file not found, using defaults', { configPath });
        return;
      }

      const userConfig = await this.fileHandler.readJSON(configPath);
      this.config = this.mergeConfigs(this.config, userConfig);
      this.validateConfig();
      
      this.logger.info('Configuration loaded successfully');
    } catch (error) {
      this.logger.error('Failed to load configuration', error);
      throw new Error(`Configuration loading failed: ${error.message}`);
    }
  }

  /**
   * Deep merge configuration objects
   * @private
   */
  mergeConfigs(defaultConfig, userConfig) {
    const merged = { ...defaultConfig };
    
    for (const [key, value] of Object.entries(userConfig)) {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        merged[key] = this.mergeConfigs(merged[key] || {}, value);
      } else {
        merged[key] = value;
      }
    }
    
    return merged;
  }

  /**
   * Validate configuration values
   * @private
   */
  validateConfig() {
    const errors = [];

    // Validate test count
    if (this.config.testing.defaultTestCount < 1 || 
        this.config.testing.defaultTestCount > this.config.testing.maxTestCount) {
      errors.push(`testing.defaultTestCount must be between 1 and ${this.config.testing.maxTestCount}`);
    }

    // Validate statistical settings
    if (this.config.statistics.significanceLevel <= 0 || 
        this.config.statistics.significanceLevel >= 1) {
      errors.push('statistics.significanceLevel must be between 0 and 1');
    }

    if (this.config.statistics.confidenceLevel <= 0 || 
        this.config.statistics.confidenceLevel >= 1) {
      errors.push('statistics.confidenceLevel must be between 0 and 1');
    }

    // Validate complexity distribution
    const complexitySum = Object.values(this.config.testing.complexityDistribution)
      .reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(complexitySum - 1.0) > 0.01) {
      errors.push('testing.complexityDistribution weights must sum to 1.0');
    }

    // Validate category weights
    const categorySum = Object.values(this.config.testing.categoryWeights)
      .reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(categorySum - 1.0) > 0.01) {
      errors.push('testing.categoryWeights must sum to 1.0');
    }

    // Validate evaluation weights
    const evaluationSum = Object.values(this.config.evaluation.structuralWeights)
      .reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(evaluationSum - 1.0) > 0.01) {
      errors.push('evaluation.structuralWeights must sum to 1.0');
    }

    if (errors.length > 0) {
      throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
    }
  }

  /**
   * Get configuration value by path
   * @param {string} path - Dot-separated path to config value
   * @param {*} defaultValue - Default value if path not found
   * @returns {*} Configuration value
   */
  get(path, defaultValue = undefined) {
    const keys = path.split('.');
    let value = this.config;
    
    for (const key of keys) {
      if (value && typeof value === 'object' && key in value) {
        value = value[key];
      } else {
        return defaultValue;
      }
    }
    
    return value;
  }

  /**
   * Set configuration value by path
   * @param {string} path - Dot-separated path to config value
   * @param {*} value - Value to set
   */
  set(path, value) {
    const keys = path.split('.');
    let current = this.config;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!(key in current) || typeof current[key] !== 'object') {
        current[key] = {};
      }
      current = current[key];
    }
    
    current[keys[keys.length - 1]] = value;
  }

  /**
   * Save current configuration to file
   * @param {string} outputPath - Path to save configuration
   */
  async saveConfig(outputPath) {
    try {
      await this.fileHandler.writeJSON(outputPath, this.config, true);
      this.logger.info('Configuration saved', { outputPath });
    } catch (error) {
      this.logger.error('Failed to save configuration', error);
      throw error;
    }
  }

  /**
   * Get the complete configuration object
   * @returns {Object} Complete configuration
   */
  getAll() {
    return { ...this.config };
  }

  /**
   * Create a configuration template file
   * @param {string} templatePath - Path to save template
   */
  async createTemplate(templatePath) {
    const template = {
      testing: {
        defaultTestCount: 100,
        complexityDistribution: {
          simple: 0.3,
          moderate: 0.5,
          complex: 0.2
        }
      },
      evaluation: {
        methods: ['structural', 'llm-judge'],
        llmJudge: {
          model: 'claude-3.5-sonnet',
          batchSize: 5
        }
      },
      optimization: {
        autoUpdateRules: false,
        minConfidenceForUpdate: 0.85
      },
      output: {
        defaultFormat: 'markdown',
        includeExamples: true
      }
    };

    await this.fileHandler.writeJSON(templatePath, template, true);
    this.logger.info('Configuration template created', { templatePath });
  }
}

module.exports = FrameworkConfig;