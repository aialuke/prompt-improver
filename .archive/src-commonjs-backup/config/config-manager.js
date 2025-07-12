/**
 * Configuration Management System
 * Flexible settings management for different use cases
 */

const fs = require('fs').promises;
const path = require('path');

class ConfigManager {
  constructor() {
    // Default configuration
    this.defaultConfig = {
      // Project Analysis
      analysis: {
        includeTests: true,
        includeDocs: true,
        maxFileSize: '10MB',
        excludePatterns: ['node_modules/', '.git/', 'dist/', 'build/', 'coverage/'],
        contextInference: {
          domainKeywords: true,
          dependencyAnalysis: true,
          structureAnalysis: true
        }
      },
      
      // Test Generation
      testing: {
        defaultTestCount: 100,
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
        contextRelevanceWeight: 0.8
      },
      
      // Evaluation
      evaluation: {
        methods: ['structural', 'llm-judge'],
        llmJudge: {
          model: 'claude-3.5-sonnet',
          temperature: 0.1,
          maxTokens: 1000,
          batchSize: 10,
          rateLimitDelay: 1000
        },
        structuralWeights: {
          clarity: 0.3,
          completeness: 0.25,
          specificity: 0.25,
          structure: 0.2
        }
      },
      
      // Statistical Analysis
      statistics: {
        significanceLevel: 0.05,
        confidenceLevel: 0.95,
        minimumSampleSize: 30,
        effectSizeThreshold: 0.2
      },
      
      // Rule Optimization
      optimization: {
        autoUpdateRules: false,
        minConfidenceForUpdate: 0.8,
        maxRiskLevel: 'medium',
        backupBeforeUpdate: true,
        maxUpdatesPerCycle: 5,
        validationTestCount: 50
      },
      
      // Output and Reporting
      output: {
        defaultFormat: 'markdown',
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
        ]
      },
      
      // Framework paths
      paths: {
        rulesFile: './config/prompt-engineering-rules.json',
        backupDir: './backups',
        outputDir: './output',
        logsDir: './logs'
      }
    };
  }

  /**
   * Get default configuration
   * @returns {Object} Default configuration object
   */
  getDefaultConfig() {
    return JSON.parse(JSON.stringify(this.defaultConfig));
  }

  /**
   * Load configuration from file
   * @param {string} configPath - Path to configuration file
   * @returns {Promise<Object>} Merged configuration
   */
  async loadConfig(configPath) {
    try {
      const userConfig = await this.loadUserConfig(configPath);
      const mergedConfig = this.mergeConfigs(this.defaultConfig, userConfig);
      this.validateConfig(mergedConfig);
      return mergedConfig;
    } catch (error) {
      if (error.code === 'ENOENT') {
        // Config file doesn't exist, return defaults
        return this.getDefaultConfig();
      }
      throw error;
    }
  }

  /**
   * Load user configuration from file
   * @private
   */
  async loadUserConfig(configPath) {
    const configContent = await fs.readFile(configPath, 'utf8');
    
    // Support both JSON and YAML formats
    if (configPath.endsWith('.json')) {
      return JSON.parse(configContent);
    } else if (configPath.endsWith('.yaml') || configPath.endsWith('.yml')) {
      return this.parseYaml(configContent);
    } else {
      // Try to parse as JSON first
      try {
        return JSON.parse(configContent);
      } catch {
        // If JSON parsing fails, try YAML
        return this.parseYaml(configContent);
      }
    }
  }

  /**
   * Parse YAML configuration (simplified parser)
   * @private
   */
  parseYaml(content) {
    // This is a simplified YAML parser for basic configurations
    // In production, you'd use a proper YAML library like js-yaml
    const config = {};
    const lines = content.split('\n');
    const stack = [config];
    const indentStack = [-1];
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      
      const indent = line.search(/\S/);
      const keyMatch = line.match(/^(\s*)([^:]+):\s*(.*)$/);
      
      if (keyMatch) {
        const [, , key, value] = keyMatch;
        
        // Pop stack to correct level
        while (indentStack.length > 1 && indent <= indentStack[indentStack.length - 1]) {
          stack.pop();
          indentStack.pop();
        }
        
        const parent = stack[stack.length - 1];
        
        if (value) {
          // Simple value
          parent[key] = this.parseYamlValue(value);
        } else {
          // New object
          parent[key] = {};
          stack.push(parent[key]);
          indentStack.push(indent);
        }
      }
    }
    
    return config;
  }

  /**
   * Parse YAML value
   * @private
   */
  parseYamlValue(value) {
    value = value.trim();
    
    // Remove quotes if present
    if ((value.startsWith('"') && value.endsWith('"')) || 
        (value.startsWith("'") && value.endsWith("'"))) {
      return value.slice(1, -1);
    }
    
    // Boolean
    if (value === 'true') return true;
    if (value === 'false') return false;
    
    // Number
    if (/^-?\d+(\.\d+)?$/.test(value)) {
      return parseFloat(value);
    }
    
    // Array (simplified - only handles simple arrays)
    if (value.startsWith('[') && value.endsWith(']')) {
      return value.slice(1, -1).split(',').map(v => this.parseYamlValue(v.trim()));
    }
    
    return value;
  }

  /**
   * Merge configurations with deep merge
   * @private
   */
  mergeConfigs(defaultConfig, userConfig) {
    const merged = JSON.parse(JSON.stringify(defaultConfig));
    
    const merge = (target, source) => {
      for (const key in source) {
        if (source.hasOwnProperty(key)) {
          if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
            if (!target[key]) target[key] = {};
            merge(target[key], source[key]);
          } else {
            target[key] = source[key];
          }
        }
      }
    };
    
    merge(merged, userConfig);
    return merged;
  }

  /**
   * Validate configuration
   * @private
   */
  validateConfig(config) {
    const errors = [];
    
    // Testing validations
    if (config.testing.defaultTestCount < 10) {
      errors.push('testing.defaultTestCount must be at least 10');
    }
    
    if (config.testing.defaultTestCount > 1000) {
      errors.push('testing.defaultTestCount cannot exceed 1000');
    }
    
    const complexitySum = Object.values(config.testing.complexityDistribution).reduce((a, b) => a + b, 0);
    if (Math.abs(complexitySum - 1.0) > 0.01) {
      errors.push('testing.complexityDistribution values must sum to 1.0');
    }
    
    // Statistics validations
    if (config.statistics.significanceLevel <= 0 || config.statistics.significanceLevel >= 1) {
      errors.push('statistics.significanceLevel must be between 0 and 1');
    }
    
    if (config.statistics.confidenceLevel <= 0 || config.statistics.confidenceLevel >= 1) {
      errors.push('statistics.confidenceLevel must be between 0 and 1');
    }
    
    if (config.statistics.minimumSampleSize < 10) {
      errors.push('statistics.minimumSampleSize must be at least 10');
    }
    
    // Optimization validations
    if (config.optimization.minConfidenceForUpdate <= 0 || config.optimization.minConfidenceForUpdate > 1) {
      errors.push('optimization.minConfidenceForUpdate must be between 0 and 1');
    }
    
    const validRiskLevels = ['low', 'medium', 'high'];
    if (!validRiskLevels.includes(config.optimization.maxRiskLevel)) {
      errors.push(`optimization.maxRiskLevel must be one of: ${validRiskLevels.join(', ')}`);
    }
    
    // Evaluation validations
    const validMethods = ['structural', 'llm-judge', 'output-quality'];
    for (const method of config.evaluation.methods) {
      if (!validMethods.includes(method)) {
        errors.push(`Invalid evaluation method: ${method}. Must be one of: ${validMethods.join(', ')}`);
      }
    }
    
    // Output validations
    const validFormats = ['json', 'markdown', 'html', 'csv'];
    if (!validFormats.includes(config.output.defaultFormat)) {
      errors.push(`output.defaultFormat must be one of: ${validFormats.join(', ')}`);
    }
    
    if (errors.length > 0) {
      throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
    }
    
    return true;
  }

  /**
   * Save configuration to file
   * @param {Object} config - Configuration to save
   * @param {string} configPath - Path to save configuration
   * @returns {Promise<void>}
   */
  async saveConfig(config, configPath) {
    const configDir = path.dirname(configPath);
    await fs.mkdir(configDir, { recursive: true });
    
    if (configPath.endsWith('.json')) {
      await fs.writeFile(configPath, JSON.stringify(config, null, 2));
    } else {
      // For YAML, convert to simplified format
      const yaml = this.convertToYaml(config);
      await fs.writeFile(configPath, yaml);
    }
  }

  /**
   * Convert configuration to YAML format
   * @private
   */
  convertToYaml(config, indent = 0) {
    let yaml = '';
    const spaces = '  '.repeat(indent);
    
    for (const [key, value] of Object.entries(config)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        yaml += `${spaces}${key}:\n`;
        yaml += this.convertToYaml(value, indent + 1);
      } else if (Array.isArray(value)) {
        yaml += `${spaces}${key}: [${value.join(', ')}]\n`;
      } else {
        yaml += `${spaces}${key}: ${value}\n`;
      }
    }
    
    return yaml;
  }

  /**
   * Generate example configuration file
   * @param {string} format - Format of example (json or yaml)
   * @returns {string} Example configuration content
   */
  generateExampleConfig(format = 'json') {
    const example = {
      testing: {
        defaultTestCount: 150,
        complexityDistribution: {
          simple: 0.3,
          moderate: 0.5,
          complex: 0.2
        }
      },
      evaluation: {
        methods: ['structural', 'llm-judge', 'output-quality'],
        llmJudge: {
          model: 'gpt-4-turbo',
          batchSize: 5
        }
      },
      optimization: {
        autoUpdateRules: true,
        minConfidenceForUpdate: 0.85
      }
    };
    
    if (format === 'json') {
      return JSON.stringify(example, null, 2);
    } else {
      return this.convertToYaml(example);
    }
  }

  /**
   * Get configuration schema for validation
   * @returns {Object} Configuration schema
   */
  getConfigSchema() {
    return {
      type: 'object',
      properties: {
        analysis: {
          type: 'object',
          properties: {
            includeTests: { type: 'boolean' },
            includeDocs: { type: 'boolean' },
            maxFileSize: { type: 'string' },
            excludePatterns: { type: 'array', items: { type: 'string' } }
          }
        },
        testing: {
          type: 'object',
          properties: {
            defaultTestCount: { type: 'number', minimum: 10, maximum: 1000 },
            complexityDistribution: {
              type: 'object',
              properties: {
                simple: { type: 'number', minimum: 0, maximum: 1 },
                moderate: { type: 'number', minimum: 0, maximum: 1 },
                complex: { type: 'number', minimum: 0, maximum: 1 }
              }
            }
          }
        },
        evaluation: {
          type: 'object',
          properties: {
            methods: {
              type: 'array',
              items: { type: 'string', enum: ['structural', 'llm-judge', 'output-quality'] }
            }
          }
        }
      }
    };
  }
}

module.exports = ConfigManager;