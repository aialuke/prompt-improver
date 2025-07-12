/**
 * Graceful error management system
 * Provides consistent error handling, recovery, and reporting
 */

const Logger = require('./logger');

class ErrorHandler {
  constructor() {
    this.logger = new Logger('ErrorHandler');
    this.errorCounts = new Map();
    this.lastErrors = [];
    this.maxLastErrorsCount = 50;
    
    // Error categories for better handling
    this.errorCategories = {
      VALIDATION: 'validation',
      FILE_SYSTEM: 'file_system',
      NETWORK: 'network',
      TIMEOUT: 'timeout',
      AUTHENTICATION: 'authentication',
      RATE_LIMIT: 'rate_limit',
      PARSING: 'parsing',
      BUSINESS_LOGIC: 'business_logic',
      UNKNOWN: 'unknown'
    };

    // Recovery strategies
    this.recoveryStrategies = new Map([
      ['ENOENT', this.handleFileNotFound],
      ['EACCES', this.handlePermissionError],
      ['ETIMEOUT', this.handleTimeout],
      ['RATE_LIMIT', this.handleRateLimit],
      ['VALIDATION_ERROR', this.handleValidationError]
    ]);
  }

  /**
   * Handle and categorize errors with recovery attempts
   * @param {Error} error - The error to handle
   * @param {Object} context - Additional context about the error
   * @returns {Object} Error handling result
   */
  handleError(error, context = {}) {
    const errorInfo = this.analyzeError(error, context);
    this.recordError(errorInfo);
    
    this.logger.error('Error occurred', {
      message: error.message,
      category: errorInfo.category,
      code: errorInfo.code,
      recoverable: errorInfo.recoverable,
      context
    });

    // Attempt recovery if possible
    let recoveryResult = null;
    if (errorInfo.recoverable && this.shouldAttemptRecovery(errorInfo)) {
      recoveryResult = this.attemptRecovery(errorInfo, context);
    }

    return {
      error: errorInfo,
      recovery: recoveryResult,
      handled: true,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Analyze error to determine category and recovery options
   * @private
   */
  analyzeError(error, context) {
    const errorInfo = {
      message: error.message,
      stack: error.stack,
      code: error.code || error.name || 'UNKNOWN',
      category: this.categorizeError(error),
      recoverable: this.isRecoverable(error),
      severity: this.getSeverity(error),
      retryable: this.isRetryable(error),
      context
    };

    return errorInfo;
  }

  /**
   * Categorize error based on type and message
   * @private
   */
  categorizeError(error) {
    if (error.code === 'ENOENT' || error.code === 'EACCES') {
      return this.errorCategories.FILE_SYSTEM;
    }
    
    if (error.code === 'ETIMEOUT' || error.message.includes('timeout')) {
      return this.errorCategories.TIMEOUT;
    }
    
    if (error.code === 'ENOTFOUND' || error.message.includes('network')) {
      return this.errorCategories.NETWORK;
    }
    
    if (error.message.includes('validation') || error.name === 'ValidationError') {
      return this.errorCategories.VALIDATION;
    }
    
    if (error.message.includes('rate limit') || error.code === 'RATE_LIMIT') {
      return this.errorCategories.RATE_LIMIT;
    }
    
    if (error.name === 'SyntaxError' || error.message.includes('parse')) {
      return this.errorCategories.PARSING;
    }
    
    if (error.message.includes('authentication') || error.code === 'UNAUTHORIZED') {
      return this.errorCategories.AUTHENTICATION;
    }
    
    return this.errorCategories.UNKNOWN;
  }

  /**
   * Determine if error is recoverable
   * @private
   */
  isRecoverable(error) {
    const recoverableCategories = [
      this.errorCategories.TIMEOUT,
      this.errorCategories.NETWORK,
      this.errorCategories.RATE_LIMIT,
      this.errorCategories.FILE_SYSTEM
    ];
    
    return recoverableCategories.includes(this.categorizeError(error));
  }

  /**
   * Determine error severity
   * @private
   */
  getSeverity(error) {
    if (error.message.includes('critical') || error.code === 'CRITICAL') {
      return 'critical';
    }
    
    if (this.categorizeError(error) === this.errorCategories.VALIDATION) {
      return 'high';
    }
    
    if (this.isRecoverable(error)) {
      return 'medium';
    }
    
    return 'low';
  }

  /**
   * Determine if error is retryable
   * @private
   */
  isRetryable(error) {
    const retryableCategories = [
      this.errorCategories.TIMEOUT,
      this.errorCategories.NETWORK,
      this.errorCategories.RATE_LIMIT
    ];
    
    return retryableCategories.includes(this.categorizeError(error));
  }

  /**
   * Record error for monitoring and analysis
   * @private
   */
  recordError(errorInfo) {
    // Count errors by type
    const key = `${errorInfo.category}:${errorInfo.code}`;
    this.errorCounts.set(key, (this.errorCounts.get(key) || 0) + 1);
    
    // Keep recent errors for analysis
    this.lastErrors.unshift(errorInfo);
    if (this.lastErrors.length > this.maxLastErrorsCount) {
      this.lastErrors.pop();
    }
  }

  /**
   * Determine if recovery should be attempted
   * @private
   */
  shouldAttemptRecovery(errorInfo) {
    // Don't attempt recovery if we've seen too many of the same error recently
    const recentSimilarErrors = this.lastErrors
      .filter(e => e.category === errorInfo.category && e.code === errorInfo.code)
      .length;
    
    return recentSimilarErrors < 3; // Max 3 recovery attempts for same error type
  }

  /**
   * Attempt to recover from error
   * @private
   */
  attemptRecovery(errorInfo, context) {
    const strategy = this.recoveryStrategies.get(errorInfo.code);
    
    if (strategy) {
      try {
        this.logger.info('Attempting error recovery', { 
          category: errorInfo.category,
          code: errorInfo.code 
        });
        
        const result = strategy.call(this, errorInfo, context);
        
        this.logger.info('Recovery attempt completed', { 
          success: result.success,
          strategy: result.strategy 
        });
        
        return result;
      } catch (recoveryError) {
        this.logger.warn('Recovery attempt failed', { 
          originalError: errorInfo.code,
          recoveryError: recoveryError.message 
        });
        
        return {
          success: false,
          strategy: 'recovery_failed',
          error: recoveryError.message
        };
      }
    }
    
    return {
      success: false,
      strategy: 'no_strategy_available'
    };
  }

  /**
   * Recovery strategy for file not found errors
   * @private
   */
  handleFileNotFound(errorInfo, context) {
    return {
      success: false,
      strategy: 'file_not_found',
      recommendation: 'Check file path and ensure file exists',
      actions: ['verify_path', 'create_missing_directories']
    };
  }

  /**
   * Recovery strategy for permission errors
   * @private
   */
  handlePermissionError(errorInfo, context) {
    return {
      success: false,
      strategy: 'permission_error',
      recommendation: 'Check file permissions or run with appropriate privileges',
      actions: ['check_permissions', 'use_alternative_path']
    };
  }

  /**
   * Recovery strategy for timeout errors
   * @private
   */
  handleTimeout(errorInfo, context) {
    return {
      success: true,
      strategy: 'timeout_retry',
      recommendation: 'Retry with increased timeout',
      actions: ['increase_timeout', 'retry_operation']
    };
  }

  /**
   * Recovery strategy for rate limit errors
   * @private
   */
  handleRateLimit(errorInfo, context) {
    const waitTime = this.calculateBackoffDelay();
    
    return {
      success: true,
      strategy: 'rate_limit_backoff',
      recommendation: `Wait ${waitTime}ms before retrying`,
      actions: ['wait_and_retry'],
      waitTime
    };
  }

  /**
   * Recovery strategy for validation errors
   * @private
   */
  handleValidationError(errorInfo, context) {
    return {
      success: false,
      strategy: 'validation_error',
      recommendation: 'Fix validation issues and retry',
      actions: ['validate_input', 'sanitize_data']
    };
  }

  /**
   * Calculate exponential backoff delay
   * @private
   */
  calculateBackoffDelay(baseDelay = 1000, maxDelay = 30000) {
    const rateLimitErrors = this.lastErrors
      .filter(e => e.category === this.errorCategories.RATE_LIMIT)
      .length;
    
    const delay = Math.min(baseDelay * Math.pow(2, rateLimitErrors), maxDelay);
    return delay + Math.random() * 1000; // Add jitter
  }

  /**
   * Wrap error with additional context
   * @param {Error} error - Original error
   * @param {string} code - Error code
   * @param {Object} context - Additional context
   * @returns {Error} Enhanced error
   */
  wrapError(error, code, context = {}) {
    const wrappedError = new Error(error.message);
    wrappedError.originalError = error;
    wrappedError.code = code;
    wrappedError.context = context;
    wrappedError.stack = error.stack;
    
    return wrappedError;
  }

  /**
   * Handle specific test cycle errors
   */
  handleTestCycleError(error) {
    const handled = this.handleError(error, { component: 'test_cycle' });
    
    return {
      success: false,
      error: {
        message: 'Test cycle execution failed',
        details: error.message,
        category: handled.error.category,
        recoverable: handled.error.recoverable,
        timestamp: handled.timestamp
      },
      recovery: handled.recovery
    };
  }

  /**
   * Handle batch processing errors
   */
  handleBatchError(error) {
    const handled = this.handleError(error, { component: 'batch_processor' });
    
    return {
      success: false,
      processed: 0,
      failed: 0,
      results: [],
      errors: [{
        message: 'Batch processing failed',
        details: error.message,
        category: handled.error.category,
        timestamp: handled.timestamp
      }],
      recovery: handled.recovery
    };
  }

  /**
   * Get error statistics
   */
  getErrorStats() {
    const stats = {
      totalErrors: this.lastErrors.length,
      errorsByCategory: {},
      errorsByCode: {},
      recentErrors: this.lastErrors.slice(0, 10),
      topErrors: []
    };

    // Group by category
    for (const error of this.lastErrors) {
      stats.errorsByCategory[error.category] = 
        (stats.errorsByCategory[error.category] || 0) + 1;
    }

    // Group by code
    for (const [key, count] of this.errorCounts.entries()) {
      const [category, code] = key.split(':');
      stats.errorsByCode[code] = count;
    }

    // Get top errors
    stats.topErrors = Array.from(this.errorCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([key, count]) => ({ error: key, count }));

    return stats;
  }

  /**
   * Clear error history
   */
  clearHistory() {
    this.errorCounts.clear();
    this.lastErrors = [];
    this.logger.info('Error history cleared');
  }
}

module.exports = ErrorHandler;