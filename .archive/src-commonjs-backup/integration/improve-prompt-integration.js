/**
 * Integration with the existing improve-prompt.js tool
 * Provides interface to execute prompt improvements and capture results
 */

const { spawn } = require('child_process');
const path = require('path');
const Logger = require('../utils/logger');
const TestResult = require('../models/test-result');
const ErrorHandler = require('../utils/error-handler');

class ImprovePromptIntegration {
  constructor(config = {}) {
    this.config = {
      scriptPath: config.scriptPath || './improve-prompt.js',
      rulesPath: config.rulesPath || './prompt-engineering-best-practices.json',
      timeout: config.timeout || 30000,
      retryAttempts: config.retryAttempts || 2,
      retryDelay: config.retryDelay || 1000,
      ...config
    };
    
    this.logger = new Logger('ImprovePromptIntegration');
    this.errorHandler = new ErrorHandler();
  }

  /**
   * Process a single prompt using improve-prompt.js
   * @param {TestCase} testCase - Test case to process
   * @returns {Promise<TestResult>} Test result
   */
  async processPrompt(testCase) {
    const startTime = Date.now();
    
    try {
      this.logger.debug('Processing prompt', { 
        testId: testCase.id,
        promptLength: testCase.originalPrompt.length 
      });

      const improvedPrompt = await this.executeImprovePrompt(testCase.originalPrompt);
      const executionTime = Date.now() - startTime;

      // Analyze the improvement
      const improvements = await this.analyzeImprovement(
        testCase.originalPrompt, 
        improvedPrompt
      );

      const result = new TestResult({
        testId: testCase.id,
        originalPrompt: testCase.originalPrompt,
        improvedPrompt: improvedPrompt,
        improvements: improvements,
        executionTime: executionTime,
        success: true,
        appliedRules: [], // TODO: Extract from improve-prompt.js output
        evaluationMethods: ['structural'],
        metadata: {
          category: testCase.category,
          complexity: testCase.complexity,
          domain: testCase.domain,
          techContext: testCase.techContext
        }
      });

      this.logger.debug('Prompt processed successfully', {
        testId: testCase.id,
        executionTime,
        overallScore: result.getOverallScore()
      });

      return result;

    } catch (error) {
      this.logger.error('Prompt processing failed', { 
        testId: testCase.id, 
        error: error.message 
      });

      return new TestResult({
        testId: testCase.id,
        originalPrompt: testCase.originalPrompt,
        improvedPrompt: testCase.originalPrompt, // Fallback to original
        improvements: { structural: 0, clarity: 0, completeness: 0, relevance: 0 },
        executionTime: Date.now() - startTime,
        success: false,
        errorMessage: error.message,
        metadata: {
          category: testCase.category,
          complexity: testCase.complexity
        }
      });
    }
  }

  /**
   * Process multiple prompts in batch
   * @param {Array<TestCase>} testCases - Array of test cases
   * @param {Object} options - Processing options
   * @returns {Promise<Array<TestResult>>} Array of test results
   */
  async processBatch(testCases, options = {}) {
    const {
      concurrency = 3,
      progressCallback = null
    } = options;

    this.logger.info('Processing batch of prompts', {
      count: testCases.length,
      concurrency
    });

    const results = [];
    const chunks = this.createChunks(testCases, concurrency);

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      if (progressCallback) {
        progressCallback({
          processed: i * concurrency,
          total: testCases.length,
          percentage: Math.round((i * concurrency / testCases.length) * 100)
        });
      }

      const chunkPromises = chunk.map(testCase => 
        this.processPrompt(testCase).catch(error => {
          this.logger.error('Chunk processing error', error);
          return this.createErrorResult(testCase, error);
        })
      );

      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults);

      // Rate limiting between chunks
      if (i < chunks.length - 1) {
        await this.delay(this.config.retryDelay);
      }
    }

    if (progressCallback) {
      progressCallback({
        processed: testCases.length,
        total: testCases.length,
        percentage: 100
      });
    }

    this.logger.info('Batch processing completed', {
      total: results.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length
    });

    return results;
  }

  /**
   * Execute improve-prompt.js as child process
   * @private
   */
  async executeImprovePrompt(prompt) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Improve prompt execution timed out'));
      }, this.config.timeout);

      const child = spawn('node', [this.config.scriptPath], {
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        clearTimeout(timeout);
        
        if (code === 0) {
          try {
            // Parse the output to extract improved prompt
            const improvedPrompt = this.parseImprovePromptOutput(stdout);
            resolve(improvedPrompt);
          } catch (parseError) {
            reject(new Error(`Failed to parse improve-prompt output: ${parseError.message}`));
          }
        } else {
          reject(new Error(`Improve-prompt exited with code ${code}: ${stderr}`));
        }
      });

      child.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to spawn improve-prompt: ${error.message}`));
      });

      // Send the prompt to stdin
      child.stdin.write(prompt);
      child.stdin.end();
    });
  }

  /**
   * Parse output from improve-prompt.js
   * @private
   */
  parseImprovePromptOutput(output) {
    try {
      // Try to parse as JSON first
      const parsed = JSON.parse(output);
      return parsed.improvedPrompt || parsed.result || output;
    } catch {
      // If not JSON, assume the entire output is the improved prompt
      // Look for common patterns that indicate the improved prompt
      const lines = output.split('\n');
      
      // Remove empty lines and potential metadata
      const cleanLines = lines.filter(line => 
        line.trim() && 
        !line.startsWith('//') && 
        !line.startsWith('#') &&
        !line.includes('Processing prompt') &&
        !line.includes('Applying rules')
      );

      return cleanLines.join('\n').trim() || output.trim();
    }
  }

  /**
   * Analyze improvement between original and improved prompts
   * @private
   */
  async analyzeImprovement(originalPrompt, improvedPrompt) {
    const improvements = {
      structural: 0,
      clarity: 0,
      completeness: 0,
      relevance: 0
    };

    // Basic structural analysis
    improvements.structural = this.calculateStructuralImprovement(originalPrompt, improvedPrompt);
    improvements.clarity = this.calculateClarityImprovement(originalPrompt, improvedPrompt);
    improvements.completeness = this.calculateCompletenessImprovement(originalPrompt, improvedPrompt);
    improvements.relevance = this.calculateRelevanceImprovement(originalPrompt, improvedPrompt);

    return improvements;
  }

  /**
   * Calculate structural improvement score
   * @private
   */
  calculateStructuralImprovement(original, improved) {
    let score = 0;
    
    // Check for added structure elements
    const originalStructure = this.countStructuralElements(original);
    const improvedStructure = this.countStructuralElements(improved);

    // XML tags (good for prompt structure)
    if (improvedStructure.xmlTags > originalStructure.xmlTags) {
      score += 25;
    }

    // Sections/headers
    if (improvedStructure.sections > originalStructure.sections) {
      score += 20;
    }

    // Lists (bullet points, numbered)
    if (improvedStructure.lists > originalStructure.lists) {
      score += 15;
    }

    // Examples
    if (improvedStructure.examples > originalStructure.examples) {
      score += 20;
    }

    // Length increase (moderate improvement)
    const lengthIncrease = (improved.length - original.length) / original.length;
    if (lengthIncrease > 0.1 && lengthIncrease < 2.0) { // 10% to 200% increase
      score += 20;
    }

    return Math.min(score, 100);
  }

  /**
   * Calculate clarity improvement score
   * @private
   */
  calculateClarityImprovement(original, improved) {
    let score = 0;

    // Check for clarity indicators
    const clarityIndicators = [
      /specifically/gi,
      /exactly/gi,
      /step.by.step/gi,
      /for example/gi,
      /such as/gi,
      /in particular/gi,
      /clearly/gi,
      /detailed/gi
    ];

    const originalClarityCount = this.countMatches(original, clarityIndicators);
    const improvedClarityCount = this.countMatches(improved, clarityIndicators);

    if (improvedClarityCount > originalClarityCount) {
      score += Math.min((improvedClarityCount - originalClarityCount) * 20, 60);
    }

    // Check for reduced ambiguous words
    const ambiguousWords = [
      /\bthis\b/gi,
      /\bthat\b/gi,
      /\bit\b/gi,
      /\bstuff\b/gi,
      /\bthings\b/gi,
      /\bsomething\b/gi
    ];

    const originalAmbiguous = this.countMatches(original, ambiguousWords);
    const improvedAmbiguous = this.countMatches(improved, ambiguousWords);

    if (improvedAmbiguous < originalAmbiguous) {
      score += 40;
    }

    return Math.min(score, 100);
  }

  /**
   * Calculate completeness improvement score
   * @private
   */
  calculateCompletenessImprovement(original, improved) {
    let score = 0;

    // Check for completeness indicators
    const completenessElements = [
      /context[:\s]/gi,
      /background[:\s]/gi,
      /requirements[:\s]/gi,
      /constraints[:\s]/gi,
      /output.format/gi,
      /expected.output/gi,
      /format[:\s]/gi,
      /style[:\s]/gi
    ];

    const originalCompleteness = this.countMatches(original, completenessElements);
    const improvedCompleteness = this.countMatches(improved, completenessElements);

    if (improvedCompleteness > originalCompleteness) {
      score += Math.min((improvedCompleteness - originalCompleteness) * 25, 100);
    }

    return Math.min(score, 100);
  }

  /**
   * Calculate relevance improvement score
   * @private
   */
  calculateRelevanceImprovement(original, improved) {
    // For now, use a simple heuristic
    // In a full implementation, this would use the project context
    let score = 50; // Default moderate relevance

    // If the improved prompt is significantly longer and more structured,
    // assume it's more relevant
    if (improved.length > original.length * 1.2 && 
        this.countStructuralElements(improved).total > this.countStructuralElements(original).total) {
      score += 30;
    }

    return Math.min(score, 100);
  }

  /**
   * Count structural elements in text
   * @private
   */
  countStructuralElements(text) {
    const xmlTags = (text.match(/<[^>]+>/g) || []).length;
    const sections = (text.match(/^#{1,6}\s/gm) || []).length;
    const bulletPoints = (text.match(/^[\s]*[-*+]\s/gm) || []).length;
    const numberedLists = (text.match(/^[\s]*\d+\.\s/gm) || []).length;
    const examples = (text.match(/example[s]?:/gi) || []).length;
    
    return {
      xmlTags,
      sections,
      lists: bulletPoints + numberedLists,
      examples,
      total: xmlTags + sections + bulletPoints + numberedLists + examples
    };
  }

  /**
   * Count pattern matches in text
   * @private
   */
  countMatches(text, patterns) {
    return patterns.reduce((count, pattern) => {
      const matches = text.match(pattern) || [];
      return count + matches.length;
    }, 0);
  }

  /**
   * Create chunks for batch processing
   * @private
   */
  createChunks(array, chunkSize) {
    const chunks = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  /**
   * Create error result for failed test case
   * @private
   */
  createErrorResult(testCase, error) {
    return new TestResult({
      testId: testCase.id,
      originalPrompt: testCase.originalPrompt,
      improvedPrompt: testCase.originalPrompt,
      improvements: { structural: 0, clarity: 0, completeness: 0, relevance: 0 },
      executionTime: 0,
      success: false,
      errorMessage: error.message,
      metadata: {
        category: testCase.category,
        complexity: testCase.complexity
      }
    });
  }

  /**
   * Delay utility
   * @private
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Test the integration with a simple prompt
   */
  async testIntegration() {
    try {
      const testPrompt = "Create a button component";
      this.logger.info('Testing improve-prompt integration');
      
      const improvedPrompt = await this.executeImprovePrompt(testPrompt);
      
      this.logger.info('Integration test successful', {
        originalLength: testPrompt.length,
        improvedLength: improvedPrompt.length,
        hasImprovement: improvedPrompt !== testPrompt
      });

      return {
        success: true,
        originalPrompt: testPrompt,
        improvedPrompt: improvedPrompt,
        hasImprovement: improvedPrompt !== testPrompt
      };

    } catch (error) {
      this.logger.error('Integration test failed', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = ImprovePromptIntegration;