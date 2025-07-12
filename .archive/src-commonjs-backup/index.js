/**
 * Main entry point for the Universal Prompt Testing Framework
 * Provides the primary API for running tests and managing the framework
 */

const TestRunner = require('./core/test-runner');
const FrameworkConfig = require('./config/framework-config');
const TestCase = require('./models/test-case');
const TestResult = require('./models/test-result');
const Logger = require('./utils/logger');

class UniversalPromptTestingFramework {
  constructor(config = {}) {
    this.config = new FrameworkConfig(config.configPath);
    this.logger = new Logger('Framework');
    this.testRunner = new TestRunner(this.config.getAll());
    
    this.logger.info('Universal Prompt Testing Framework initialized', {
      version: this.config.get('framework.version'),
      configLoaded: !!config.configPath
    });
  }

  /**
   * Run a complete testing cycle
   * @param {Object} options - Test execution options
   * @returns {Promise<Object>} Complete test results and analysis
   */
  async runTestCycle(options = {}) {
    this.logger.info('Starting test cycle', options);
    
    try {
      const results = await this.testRunner.runTestCycle(options);
      
      this.logger.info('Test cycle completed successfully', {
        testsRun: results.tests?.length || 0,
        executionTime: results.metadata?.executionTime || 0
      });

      return results;
    } catch (error) {
      this.logger.error('Test cycle failed', error);
      throw error;
    }
  }

  /**
   * Process custom test cases
   * @param {Array<Object>} testCaseData - Array of test case data
   * @param {Object} options - Processing options
   * @returns {Promise<Array<TestResult>>} Test results
   */
  async processCustomTests(testCaseData, options = {}) {
    this.logger.info('Processing custom tests', { 
      count: testCaseData.length 
    });

    try {
      // Create TestCase instances
      const testCases = TestCase.createBatch(testCaseData);
      
      // Process using batch processor
      const results = await this.testRunner.runBatchTests(testCases, options);
      
      return results.results || [];
    } catch (error) {
      this.logger.error('Custom test processing failed', error);
      throw error;
    }
  }

  /**
   * Analyze a specific project
   * @param {string} projectPath - Path to project directory
   * @returns {Promise<Object>} Project analysis results
   */
  async analyzeProject(projectPath = process.cwd()) {
    this.logger.info('Analyzing project', { projectPath });
    
    // TODO: Implement actual project analysis
    // For now, return mock analysis
    return {
      projectPath,
      context: {
        languages: ['JavaScript', 'TypeScript'],
        frameworks: ['React', 'Next.js'],
        projectType: 'Web Application',
        domain: 'General',
        complexity: 'Medium'
      },
      confidence: 0.85,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate test cases for a project
   * @param {Object} projectContext - Project context from analysis
   * @param {Object} options - Generation options
   * @returns {Promise<Array<TestCase>>} Generated test cases
   */
  async generateTests(projectContext, options = {}) {
    const {
      testCount = this.config.get('testing.defaultTestCount', 50),
      categories = null,
      complexity = 'all'
    } = options;

    this.logger.info('Generating test cases', { 
      testCount, 
      categories, 
      complexity 
    });

    // TODO: Implement actual test generation based on context
    // For now, return mock test cases
    const testCases = [];
    for (let i = 0; i < testCount; i++) {
      testCases.push(new TestCase({
        category: this.randomCategory(),
        originalPrompt: `Create a ${this.randomComponent()} component for ${projectContext.domain}`,
        expectedImprovements: ['add-context', 'specify-requirements'],
        complexity: this.randomComplexity(),
        domain: projectContext.domain || 'general',
        techContext: projectContext.frameworks || []
      }));
    }

    return testCases;
  }

  /**
   * Get framework status
   * @returns {Object} Current framework status
   */
  getStatus() {
    return {
      framework: {
        name: this.config.get('framework.name'),
        version: this.config.get('framework.version'),
        uptime: this.getUptime()
      },
      runner: this.testRunner.getStatus(),
      config: {
        testCount: this.config.get('testing.defaultTestCount'),
        evaluationMethods: this.config.get('evaluation.methods'),
        outputFormat: this.config.get('output.defaultFormat')
      }
    };
  }

  /**
   * Update configuration
   * @param {Object} newConfig - Configuration updates
   */
  updateConfig(newConfig) {
    for (const [key, value] of Object.entries(newConfig)) {
      this.config.set(key, value);
    }
    
    // Recreate test runner with new config
    this.testRunner = new TestRunner(this.config.getAll());
    
    this.logger.info('Configuration updated');
  }

  /**
   * Save current configuration
   * @param {string} outputPath - Path to save configuration
   */
  async saveConfig(outputPath) {
    await this.config.saveConfig(outputPath);
    this.logger.info('Configuration saved', { outputPath });
  }

  /**
   * Stop the framework gracefully
   */
  async stop() {
    this.logger.info('Stopping framework');
    await this.testRunner.stop();
    this.logger.info('Framework stopped');
  }

  /**
   * Get framework uptime
   * @private
   */
  getUptime() {
    if (!this.startTime) {
      this.startTime = Date.now();
    }
    return Date.now() - this.startTime;
  }

  // Helper methods for mock data generation
  randomCategory() {
    const categories = [
      'vague_instructions',
      'missing_context', 
      'poor_structure',
      'missing_examples',
      'no_output_format'
    ];
    return categories[Math.floor(Math.random() * categories.length)];
  }

  randomComponent() {
    const components = ['Button', 'Modal', 'Form', 'Table', 'Card', 'Navigation'];
    return components[Math.floor(Math.random() * components.length)];
  }

  randomComplexity() {
    const complexities = ['simple', 'moderate', 'complex'];
    const weights = [0.4, 0.4, 0.2];
    const random = Math.random();
    let sum = 0;
    for (let i = 0; i < weights.length; i++) {
      sum += weights[i];
      if (random <= sum) return complexities[i];
    }
    return 'simple';
  }
}

// Export both the class and convenience functions
module.exports = {
  UniversalPromptTestingFramework,
  TestCase,
  TestResult,
  FrameworkConfig,
  
  // Convenience factory function
  createFramework: (config) => new UniversalPromptTestingFramework(config),
  
  // Version info
  version: '1.0.0',
  
  // Quick test function
  quickTest: async (prompt, options = {}) => {
    const framework = new UniversalPromptTestingFramework();
    const testCase = new TestCase({
      originalPrompt: prompt,
      category: 'general',
      complexity: 'simple'
    });
    
    const results = await framework.processCustomTests([testCase.toObject()], options);
    return results[0];
  }
};