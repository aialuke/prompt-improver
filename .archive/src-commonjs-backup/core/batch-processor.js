/**
 * Batch processor for handling multiple tests efficiently
 * Provides concurrent processing with rate limiting and error recovery
 */

const Logger = require('../utils/logger');
const ErrorHandler = require('../utils/error-handler');

class BatchProcessor {
  constructor(config = {}) {
    this.config = {
      batchSize: config.batchSize || 10,
      concurrency: config.concurrency || 3,
      retryAttempts: config.retryAttempts || 2,
      retryDelay: config.retryDelay || 1000,
      timeout: config.timeout || 30000,
      ...config
    };
    
    this.logger = new Logger('BatchProcessor');
    this.errorHandler = new ErrorHandler();
    this.activeJobs = new Map();
    this.queue = [];
    this.isProcessing = false;
  }

  /**
   * Process a batch of test cases
   * @param {Array} testCases - Array of test cases to process
   * @param {Object} options - Processing options
   * @returns {Promise<Object>} Batch processing results
   */
  async processBatch(testCases, options = {}) {
    if (!Array.isArray(testCases) || testCases.length === 0) {
      throw new Error('Test cases must be a non-empty array');
    }

    const batchSize = options.batchSize || this.config.batchSize;
    const batches = this.createBatches(testCases, batchSize);
    
    this.logger.info('Processing batch', { 
      totalTests: testCases.length,
      batches: batches.length,
      batchSize 
    });

    const results = {
      processed: 0,
      failed: 0,
      results: [],
      errors: [],
      startTime: Date.now()
    };

    this.isProcessing = true;

    try {
      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        this.logger.debug(`Processing batch ${i + 1}/${batches.length}`, { 
          batchSize: batch.length 
        });

        const batchResults = await this.processSingleBatch(batch, i);
        
        // Aggregate results
        results.processed += batchResults.processed;
        results.failed += batchResults.failed;
        results.results.push(...batchResults.results);
        results.errors.push(...batchResults.errors);

        // Rate limiting between batches
        if (i < batches.length - 1) {
          await this.delay(this.config.retryDelay);
        }
      }

      results.endTime = Date.now();
      results.executionTime = results.endTime - results.startTime;
      results.successRate = results.processed / (results.processed + results.failed);

      this.logger.info('Batch processing completed', {
        processed: results.processed,
        failed: results.failed,
        successRate: results.successRate,
        executionTime: results.executionTime
      });

      return results;

    } catch (error) {
      this.logger.error('Batch processing failed', error);
      throw this.errorHandler.wrapError(error, 'BATCH_PROCESSING_FAILED');
    } finally {
      this.isProcessing = false;
    }
  }

  /**
   * Process a single batch with concurrency control
   * @private
   */
  async processSingleBatch(batch, batchIndex) {
    const concurrentGroups = this.createConcurrentGroups(batch, this.config.concurrency);
    const batchResults = {
      processed: 0,
      failed: 0,
      results: [],
      errors: []
    };

    for (const group of concurrentGroups) {
      const groupPromises = group.map(testCase => 
        this.processTestCase(testCase, batchIndex)
      );

      const groupResults = await Promise.allSettled(groupPromises);

      // Process group results
      groupResults.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          batchResults.processed++;
          batchResults.results.push(result.value);
        } else {
          batchResults.failed++;
          batchResults.errors.push({
            testCase: group[index],
            error: result.reason.message || 'Unknown error',
            batchIndex,
            timestamp: new Date().toISOString()
          });
        }
      });
    }

    return batchResults;
  }

  /**
   * Process a single test case with retry logic
   * @private
   */
  async processTestCase(testCase, batchIndex) {
    const jobId = `${batchIndex}-${testCase.id || Date.now()}`;
    
    try {
      this.activeJobs.set(jobId, {
        testCase,
        startTime: Date.now(),
        attempts: 0
      });

      const result = await this.executeWithRetry(
        () => this.executeTestCase(testCase),
        this.config.retryAttempts,
        this.config.retryDelay
      );

      this.activeJobs.delete(jobId);
      return result;

    } catch (error) {
      this.activeJobs.delete(jobId);
      this.logger.error('Test case processing failed', { 
        testCase: testCase.id || 'unknown',
        error: error.message 
      });
      throw error;
    }
  }

  /**
   * Execute a single test case (to be overridden by specific implementations)
   * @private
   */
  async executeTestCase(testCase) {
    // This is a placeholder - actual implementation will depend on the test type
    // For now, simulate processing
    await this.delay(Math.random() * 1000 + 500);
    
    return {
      testId: testCase.id,
      originalPrompt: testCase.originalPrompt,
      improvedPrompt: testCase.originalPrompt + ' [IMPROVED]',
      improvements: {
        structural: Math.random() * 100,
        clarity: Math.random() * 100,
        completeness: Math.random() * 100,
        relevance: Math.random() * 100
      },
      executionTime: Date.now(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Execute function with retry logic
   * @private
   */
  async executeWithRetry(fn, maxAttempts, delay) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        
        if (attempt < maxAttempts) {
          this.logger.warn(`Attempt ${attempt} failed, retrying in ${delay}ms`, {
            error: error.message
          });
          await this.delay(delay * attempt); // Exponential backoff
        }
      }
    }
    
    throw lastError;
  }

  /**
   * Create batches from test cases
   * @private
   */
  createBatches(testCases, batchSize) {
    const batches = [];
    for (let i = 0; i < testCases.length; i += batchSize) {
      batches.push(testCases.slice(i, i + batchSize));
    }
    return batches;
  }

  /**
   * Create concurrent groups within a batch
   * @private
   */
  createConcurrentGroups(batch, concurrency) {
    const groups = [];
    for (let i = 0; i < batch.length; i += concurrency) {
      groups.push(batch.slice(i, i + concurrency));
    }
    return groups;
  }

  /**
   * Delay utility
   * @private
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current queue size
   */
  getQueueSize() {
    return this.queue.length;
  }

  /**
   * Get active jobs count
   */
  getActiveJobsCount() {
    return this.activeJobs.size;
  }

  /**
   * Stop processing gracefully
   */
  async stop() {
    this.logger.info('Stopping batch processor');
    this.isProcessing = false;
    
    // Wait for active jobs to complete or timeout
    const timeout = 10000; // 10 seconds
    const startTime = Date.now();
    
    while (this.activeJobs.size > 0 && (Date.now() - startTime) < timeout) {
      await this.delay(100);
    }
    
    if (this.activeJobs.size > 0) {
      this.logger.warn(`Stopping with ${this.activeJobs.size} jobs still active`);
    }
    
    this.activeJobs.clear();
    this.queue = [];
  }
}

module.exports = BatchProcessor;