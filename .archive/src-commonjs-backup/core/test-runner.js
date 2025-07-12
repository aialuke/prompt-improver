/**
 * Main orchestration engine for the Universal Prompt Testing Framework
 * Manages the complete testing workflow and coordinates between components
 */

const BatchProcessor = require('./batch-processor');
const PipelineManager = require('./pipeline-manager');
const Logger = require('../utils/logger');
const ErrorHandler = require('../utils/error-handler');

class TestRunner {
  constructor(config = {}) {
    this.config = config;
    this.batchProcessor = new BatchProcessor(config);
    this.pipelineManager = new PipelineManager(config);
    this.logger = new Logger('TestRunner');
    this.errorHandler = new ErrorHandler();
  }

  /**
   * Execute a complete testing cycle
   * @param {Object} options - Test execution options
   * @returns {Promise<Object>} Complete test results and analysis
   */
  async runTestCycle(options = {}) {
    const startTime = Date.now();
    this.logger.info('Starting test cycle', { options });

    try {
      // Validate input parameters
      this.validateRunOptions(options);

      // Execute the testing pipeline
      const results = await this.pipelineManager.execute({
        projectPath: options.projectPath || process.cwd(),
        testCount: options.testCount || 50,
        categories: options.categories || null,
        complexity: options.complexity || 'all',
        evaluationMethods: options.evaluationMethods || ['structural']
      });

      const executionTime = Date.now() - startTime;
      this.logger.info('Test cycle completed', { 
        executionTime: `${executionTime}ms`,
        testsRun: results.tests?.length || 0,
        successRate: results.metrics?.successRate || 0
      });

      return {
        ...results,
        metadata: {
          executionTime,
          timestamp: new Date().toISOString(),
          version: this.getFrameworkVersion()
        }
      };

    } catch (error) {
      this.logger.error('Test cycle failed', error);
      return this.errorHandler.handleTestCycleError(error);
    }
  }

  /**
   * Run tests in batch mode for large test suites
   * @param {Array} testCases - Array of test cases to process
   * @param {Object} options - Batch processing options
   * @returns {Promise<Object>} Batch processing results
   */
  async runBatchTests(testCases, options = {}) {
    this.logger.info('Starting batch test processing', { 
      testCount: testCases.length,
      batchSize: options.batchSize || 10
    });

    try {
      const results = await this.batchProcessor.processBatch(testCases, options);
      
      this.logger.info('Batch processing completed', {
        processed: results.processed,
        failed: results.failed,
        successRate: results.successRate
      });

      return results;

    } catch (error) {
      this.logger.error('Batch processing failed', error);
      return this.errorHandler.handleBatchError(error);
    }
  }

  /**
   * Validate test run options
   * @private
   */
  validateRunOptions(options) {
    if (options.testCount && (options.testCount < 1 || options.testCount > 1000)) {
      throw new Error('Test count must be between 1 and 1000');
    }

    if (options.complexity && !['simple', 'moderate', 'complex', 'all'].includes(options.complexity)) {
      throw new Error('Complexity must be one of: simple, moderate, complex, all');
    }

    if (options.projectPath && typeof options.projectPath !== 'string') {
      throw new Error('Project path must be a string');
    }
  }

  /**
   * Get current framework version
   * @private
   */
  getFrameworkVersion() {
    try {
      const packageJson = require('../../package.json');
      return packageJson.version || '1.0.0';
    } catch {
      return '1.0.0';
    }
  }

  /**
   * Get current test runner status
   * @returns {Object} Status information
   */
  getStatus() {
    return {
      isRunning: this.pipelineManager.getIsRunning(),
      currentStage: this.pipelineManager.getCurrentStage(),
      queuedTests: this.batchProcessor.getQueueSize(),
      uptime: this.getUptime()
    };
  }

  /**
   * Stop all running tests gracefully
   * @returns {Promise<void>}
   */
  async stop() {
    this.logger.info('Stopping test runner');
    await this.pipelineManager.stop();
    await this.batchProcessor.stop();
    this.logger.info('Test runner stopped');
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
}

module.exports = TestRunner;