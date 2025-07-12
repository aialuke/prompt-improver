/**
 * Structured logging system with configurable levels and outputs
 * Provides consistent logging across the framework
 */

class Logger {
  constructor(component = 'System') {
    this.component = component;
    this.logLevel = process.env.LOG_LEVEL || 'info';
    this.enableColors = process.env.NO_COLOR !== '1';
    
    // Log levels in order of verbosity
    this.levels = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3
    };

    // Color codes for different log levels
    this.colors = {
      debug: '\x1b[36m',    // Cyan
      info: '\x1b[32m',     // Green
      warn: '\x1b[33m',     // Yellow
      error: '\x1b[31m',    // Red
      reset: '\x1b[0m'      // Reset
    };
  }

  /**
   * Log debug message
   * @param {string} message - Log message
   * @param {Object} metadata - Additional metadata
   */
  debug(message, metadata = {}) {
    this.log('debug', message, metadata);
  }

  /**
   * Log info message
   * @param {string} message - Log message
   * @param {Object} metadata - Additional metadata
   */
  info(message, metadata = {}) {
    this.log('info', message, metadata);
  }

  /**
   * Log warning message
   * @param {string} message - Log message
   * @param {Object} metadata - Additional metadata
   */
  warn(message, metadata = {}) {
    this.log('warn', message, metadata);
  }

  /**
   * Log error message
   * @param {string} message - Log message
   * @param {Error|Object} error - Error object or additional metadata
   */
  error(message, error = {}) {
    let metadata = {};
    
    if (error instanceof Error) {
      metadata = {
        error: error.message,
        stack: error.stack,
        ...metadata
      };
    } else {
      metadata = error;
    }

    this.log('error', message, metadata);
  }

  /**
   * Core logging method
   * @private
   */
  log(level, message, metadata = {}) {
    // Check if this log level should be output
    if (this.levels[level] < this.levels[this.logLevel]) {
      return;
    }

    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level: level.toUpperCase(),
      component: this.component,
      message,
      ...metadata
    };

    // Format for console output
    const consoleMessage = this.formatConsoleMessage(level, timestamp, message, metadata);
    
    // Output to appropriate stream
    if (level === 'error' || level === 'warn') {
      console.error(consoleMessage);
    } else {
      console.log(consoleMessage);
    }

    // TODO: Add file logging, external logging service integration
    this.writeToFile(logEntry);
  }

  /**
   * Format message for console output
   * @private
   */
  formatConsoleMessage(level, timestamp, message, metadata) {
    const color = this.enableColors ? this.colors[level] : '';
    const reset = this.enableColors ? this.colors.reset : '';
    
    let formattedMessage = `${color}[${timestamp}] ${level.toUpperCase()} [${this.component}] ${message}${reset}`;

    // Add metadata if present
    if (Object.keys(metadata).length > 0) {
      const metadataStr = JSON.stringify(metadata, null, 2);
      formattedMessage += `\n${color}${metadataStr}${reset}`;
    }

    return formattedMessage;
  }

  /**
   * Write log entry to file (placeholder implementation)
   * @private
   */
  writeToFile(logEntry) {
    // TODO: Implement file logging
    // For now, this is a placeholder
    if (process.env.LOG_TO_FILE === '1') {
      // Would write to log file here
    }
  }

  /**
   * Create a child logger with additional context
   * @param {string} childComponent - Child component name
   * @param {Object} context - Additional context to include in all logs
   * @returns {Logger} Child logger instance
   */
  child(childComponent, context = {}) {
    const childLogger = new Logger(`${this.component}:${childComponent}`);
    childLogger.context = context;
    
    // Override log method to include context
    const originalLog = childLogger.log.bind(childLogger);
    childLogger.log = (level, message, metadata = {}) => {
      originalLog(level, message, { ...context, ...metadata });
    };

    return childLogger;
  }

  /**
   * Set log level for this logger instance
   * @param {string} level - New log level (debug, info, warn, error)
   */
  setLevel(level) {
    if (this.levels.hasOwnProperty(level)) {
      this.logLevel = level;
    } else {
      this.warn('Invalid log level', { level, validLevels: Object.keys(this.levels) });
    }
  }

  /**
   * Create performance timer
   * @param {string} operation - Operation name
   * @returns {Function} Timer end function
   */
  timer(operation) {
    const startTime = Date.now();
    this.debug(`Starting operation: ${operation}`);

    return (metadata = {}) => {
      const duration = Date.now() - startTime;
      this.info(`Completed operation: ${operation}`, { 
        duration: `${duration}ms`,
        ...metadata 
      });
      return duration;
    };
  }

  /**
   * Log with structured data for metrics/monitoring
   * @param {string} metric - Metric name
   * @param {number} value - Metric value
   * @param {Object} tags - Additional tags/metadata
   */
  metric(metric, value, tags = {}) {
    this.info('METRIC', {
      metric,
      value,
      tags,
      timestamp: Date.now()
    });
  }

  /**
   * Log request/response for API calls
   * @param {string} operation - Operation name
   * @param {Object} request - Request details
   * @param {Object} response - Response details
   * @param {number} duration - Operation duration in ms
   */
  apiCall(operation, request, response, duration) {
    this.info(`API: ${operation}`, {
      request: {
        method: request.method,
        url: request.url,
        headers: this.sanitizeHeaders(request.headers),
        body: this.sanitizeBody(request.body)
      },
      response: {
        status: response.status,
        headers: this.sanitizeHeaders(response.headers),
        size: response.size
      },
      duration: `${duration}ms`
    });
  }

  /**
   * Sanitize headers for logging (remove sensitive data)
   * @private
   */
  sanitizeHeaders(headers = {}) {
    const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];
    const sanitized = { ...headers };
    
    for (const header of sensitiveHeaders) {
      if (sanitized[header]) {
        sanitized[header] = '[REDACTED]';
      }
    }
    
    return sanitized;
  }

  /**
   * Sanitize request/response body for logging
   * @private
   */
  sanitizeBody(body) {
    if (!body) return body;
    
    // If it's a string, truncate if too long
    if (typeof body === 'string') {
      return body.length > 1000 ? body.substring(0, 1000) + '...[truncated]' : body;
    }
    
    // If it's an object, remove sensitive fields
    if (typeof body === 'object') {
      const sensitiveFields = ['password', 'token', 'apiKey', 'secret'];
      const sanitized = { ...body };
      
      for (const field of sensitiveFields) {
        if (sanitized[field]) {
          sanitized[field] = '[REDACTED]';
        }
      }
      
      return sanitized;
    }
    
    return body;
  }
}

module.exports = Logger;