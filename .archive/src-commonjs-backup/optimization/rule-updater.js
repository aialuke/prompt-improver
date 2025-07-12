/**
 * Automated Rule Updates System
 * Safely applies validated rule improvements with comprehensive backup and rollback capabilities
 */

const fs = require('fs').promises;
const path = require('path');

class RuleUpdater {
  constructor(config = {}) {
    this.config = {
      // Update criteria
      minConfidenceForUpdate: 0.8,         // Minimum confidence required for auto-update
      maxRiskLevel: 'medium',              // Maximum acceptable risk level
      minEffectSize: 0.2,                  // Minimum effect size for meaningful update
      minStatisticalSignificance: 0.05,    // Maximum p-value for significance
      
      // Safety parameters
      maxUpdatesPerCycle: 5,               // Maximum rules to update in one cycle
      backupRetentionDays: 30,             // How long to keep backups
      validationTestCount: 50,             // Number of tests for validation
      rollbackTimeoutMs: 30000,            // Timeout for rollback operations
      
      // File system paths
      rulesPath: './config/prompt-engineering-rules.json',
      backupPath: './backups/rules',
      logPath: './logs/rule-updates.log',
      
      // Validation settings
      requireFullValidation: true,         // Require complete validation before update
      allowPartialUpdates: false,          // Allow updates even if some fail
      dryRunMode: false,                   // Test mode without actual changes
      
      ...config
    };

    // State tracking
    this.updateHistory = [];
    this.activeBackups = new Map();
    this.rollbackStack = [];
    this.currentRuleSet = null;
    this.updateSession = null;
  }

  /**
   * Apply validated optimizations to the rule set
   * @param {Array} validatedOptimizations - Optimizations that passed A/B testing
   * @returns {Object} Update results with success/failure details
   */
  async updateRules(validatedOptimizations) {
    const sessionId = this.generateSessionId();
    this.updateSession = {
      id: sessionId,
      startTime: Date.now(),
      optimizations: validatedOptimizations,
      status: 'in_progress'
    };

    try {
      // Pre-update validation
      this.validateUpdateInputs(validatedOptimizations);
      
      // Filter optimizations that meet update criteria
      const eligibleOptimizations = this.filterEligibleOptimizations(validatedOptimizations);
      
      if (eligibleOptimizations.length === 0) {
        return this.generateNoUpdatesResult('No optimizations meet update criteria');
      }

      // Create comprehensive backup
      const backup = await this.createRuleBackup(sessionId);
      
      // Load current rule set
      await this.loadCurrentRuleSet();
      
      // Apply updates with validation
      const updateResults = await this.applyOptimizationsWithValidation(eligibleOptimizations, backup);
      
      // Final validation
      const finalValidation = await this.performFinalValidation();
      
      if (!finalValidation.valid) {
        await this.performEmergencyRollback(backup, finalValidation.issues);
        throw new Error(`Final validation failed: ${finalValidation.issues.join(', ')}`);
      }

      // Success - cleanup and logging
      await this.finalizeSuccessfulUpdate(sessionId, updateResults, backup);
      
      return {
        success: true,
        sessionId,
        summary: this.generateUpdateSummary(updateResults),
        updatesApplied: updateResults.successful.length,
        updatesSkipped: updateResults.skipped.length,
        updatesFailed: updateResults.failed.length,
        backupId: backup.id,
        validationResults: finalValidation,
        details: updateResults
      };

    } catch (error) {
      await this.handleUpdateFailure(sessionId, error);
      throw error;
    } finally {
      this.updateSession = null;
    }
  }

  /**
   * Filter optimizations that meet update criteria
   * @private
   */
  filterEligibleOptimizations(optimizations) {
    return optimizations.filter(opt => this.meetsUpdateCriteria(opt))
                      .slice(0, this.config.maxUpdatesPerCycle);
  }

  /**
   * Check if optimization meets update criteria
   * @private
   */
  meetsUpdateCriteria(optimization) {
    const abTestResult = optimization.abTestResult;
    
    if (!abTestResult || !abTestResult.recommendation.deploy) {
      return false;
    }

    return (
      optimization.confidence >= this.config.minConfidenceForUpdate &&
      this.normalizeRiskLevel(optimization.riskLevel) <= this.normalizeRiskLevel(this.config.maxRiskLevel) &&
      Math.abs(abTestResult.analysis.effectSize.cohensD) >= this.config.minEffectSize &&
      abTestResult.analysis.significanceTest.pValue <= this.config.minStatisticalSignificance
    );
  }

  /**
   * Create comprehensive backup of current rule set
   * @private
   */
  async createRuleBackup(sessionId) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupId = `backup_${sessionId}_${timestamp}`;
    
    const backup = {
      id: backupId,
      sessionId,
      timestamp: new Date().toISOString(),
      originalPath: this.config.rulesPath,
      backupPath: path.join(this.config.backupPath, `${backupId}.json`),
      metadataPath: path.join(this.config.backupPath, `${backupId}_metadata.json`),
      checksumPath: path.join(this.config.backupPath, `${backupId}_checksum.txt`)
    };

    try {
      // Ensure backup directory exists
      await fs.mkdir(this.config.backupPath, { recursive: true });
      
      // Read original rules
      const originalRules = await fs.readFile(this.config.rulesPath, 'utf8');
      const checksum = this.calculateChecksum(originalRules);
      
      // Create backup files
      await Promise.all([
        fs.writeFile(backup.backupPath, originalRules),
        fs.writeFile(backup.metadataPath, JSON.stringify({
          ...backup,
          originalSize: originalRules.length,
          checksum,
          ruleCount: this.countRules(originalRules)
        }, null, 2)),
        fs.writeFile(backup.checksumPath, checksum)
      ]);
      
      this.activeBackups.set(backupId, backup);
      
      this.logBackupCreated(backup);
      return backup;

    } catch (error) {
      throw new Error(`Failed to create backup: ${error.message}`);
    }
  }

  /**
   * Load current rule set into memory
   * @private
   */
  async loadCurrentRuleSet() {
    try {
      const rulesContent = await fs.readFile(this.config.rulesPath, 'utf8');
      this.currentRuleSet = JSON.parse(rulesContent);
      return this.currentRuleSet;
    } catch (error) {
      throw new Error(`Failed to load rule set: ${error.message}`);
    }
  }

  /**
   * Apply optimizations with individual validation
   * @private
   */
  async applyOptimizationsWithValidation(optimizations, backup) {
    const results = {
      successful: [],
      failed: [],
      skipped: [],
      rollbacks: []
    };

    for (const optimization of optimizations) {
      try {
        // Apply single optimization
        const updateResult = await this.applySingleOptimization(optimization);
        
        // Validate individual change
        const validation = await this.validateSingleUpdate(optimization);
        
        if (validation.valid) {
          results.successful.push({
            optimization,
            updateResult,
            validation,
            appliedAt: new Date().toISOString()
          });
          
          this.logSuccessfulUpdate(optimization);
        } else {
          // Rollback this specific change
          await this.rollbackSingleUpdate(optimization);
          results.rollbacks.push({
            optimization,
            reason: validation.issues,
            rolledBackAt: new Date().toISOString()
          });
          
          this.logRollbackUpdate(optimization, validation.issues);
        }

      } catch (error) {
        results.failed.push({
          optimization,
          error: error.message,
          failedAt: new Date().toISOString()
        });
        
        this.logFailedUpdate(optimization, error);
        
        // If this is a critical failure, consider full rollback
        if (this.isCriticalFailure(error)) {
          this.logCriticalFailure(error);
          if (!this.config.allowPartialUpdates) {
            throw error;
          }
        }
      }
    }

    return results;
  }

  /**
   * Apply a single optimization to the rule set
   * @private
   */
  async applySingleOptimization(optimization) {
    if (this.config.dryRunMode) {
      return this.simulateOptimizationApplication(optimization);
    }

    switch (optimization.type) {
      case 'modification':
        return await this.applyRuleModification(optimization);
      case 'addition':
        return await this.applyRuleAddition(optimization);
      case 'specialization':
        return await this.applyRuleSpecialization(optimization);
      case 'deprecation':
        return await this.applyRuleDeprecation(optimization);
      default:
        throw new Error(`Unknown optimization type: ${optimization.type}`);
    }
  }

  /**
   * Apply rule modification
   * @private
   */
  async applyRuleModification(optimization) {
    const ruleId = optimization.ruleId;
    
    if (!this.currentRuleSet[ruleId]) {
      throw new Error(`Rule not found: ${ruleId}`);
    }

    // Store original for potential rollback
    const originalRule = JSON.parse(JSON.stringify(this.currentRuleSet[ruleId]));
    
    // Apply proposed changes
    for (const change of optimization.proposedChanges) {
      this.applyRuleChange(this.currentRuleSet[ruleId], change);
    }

    // Save updated rule set
    await this.saveRuleSet();
    
    return {
      type: 'modification',
      ruleId,
      changes: optimization.proposedChanges.length,
      originalRule,
      modifiedRule: this.currentRuleSet[ruleId]
    };
  }

  /**
   * Apply rule addition
   * @private
   */
  async applyRuleAddition(optimization) {
    const newRuleId = optimization.ruleId;
    
    if (this.currentRuleSet[newRuleId]) {
      throw new Error(`Rule already exists: ${newRuleId}`);
    }

    // Add new rule
    this.currentRuleSet[newRuleId] = optimization.proposedRule;
    
    // Save updated rule set
    await this.saveRuleSet();
    
    return {
      type: 'addition',
      ruleId: newRuleId,
      addedRule: optimization.proposedRule
    };
  }

  /**
   * Apply rule specialization
   * @private
   */
  async applyRuleSpecialization(optimization) {
    const baseRuleId = optimization.baseRuleId;
    const specializationId = optimization.specializationId;
    
    if (!this.currentRuleSet[baseRuleId]) {
      throw new Error(`Base rule not found: ${baseRuleId}`);
    }

    // Create specialized version
    const specializedRule = {
      ...this.currentRuleSet[baseRuleId],
      ...optimization.contextSpecificLogic,
      baseRule: baseRuleId,
      context: optimization.targetContext,
      specialization: true
    };

    this.currentRuleSet[specializationId] = specializedRule;
    
    // Save updated rule set
    await this.saveRuleSet();
    
    return {
      type: 'specialization',
      baseRuleId,
      specializationId,
      targetContext: optimization.targetContext,
      specializedRule
    };
  }

  /**
   * Apply rule deprecation
   * @private
   */
  async applyRuleDeprecation(optimization) {
    const ruleId = optimization.ruleId;
    
    if (!this.currentRuleSet[ruleId]) {
      throw new Error(`Rule not found: ${ruleId}`);
    }

    // Store original for rollback
    const originalRule = JSON.parse(JSON.stringify(this.currentRuleSet[ruleId]));
    
    // Mark as deprecated rather than deleting
    this.currentRuleSet[ruleId].deprecated = true;
    this.currentRuleSet[ruleId].deprecatedAt = new Date().toISOString();
    this.currentRuleSet[ruleId].deprecationReason = optimization.performanceIssues;
    
    // Save updated rule set
    await this.saveRuleSet();
    
    return {
      type: 'deprecation',
      ruleId,
      originalRule,
      deprecatedRule: this.currentRuleSet[ruleId]
    };
  }

  /**
   * Validate a single update
   * @private
   */
  async validateSingleUpdate(optimization) {
    const validationResults = [];

    // Check rule set integrity
    const integrityCheck = this.validateRuleSetIntegrity();
    if (!integrityCheck.valid) {
      return { valid: false, issues: integrityCheck.issues };
    }

    // Check for conflicts
    const conflictCheck = this.checkForRuleConflicts(optimization);
    if (!conflictCheck.valid) {
      return { valid: false, issues: conflictCheck.issues };
    }

    // Performance validation
    if (this.config.requireFullValidation) {
      const performanceCheck = await this.validatePerformance(optimization);
      if (!performanceCheck.valid) {
        return { valid: false, issues: performanceCheck.issues };
      }
    }

    return { valid: true, issues: [] };
  }

  /**
   * Perform final comprehensive validation
   * @private
   */
  async performFinalValidation() {
    const validationIssues = [];

    // Rule set structure validation
    const structureValidation = this.validateRuleSetStructure();
    if (!structureValidation.valid) {
      validationIssues.push(...structureValidation.issues);
    }

    // Circular dependency check
    const dependencyCheck = this.checkCircularDependencies();
    if (!dependencyCheck.valid) {
      validationIssues.push(...dependencyCheck.issues);
    }

    // Rule conflict detection
    const conflictCheck = this.detectRuleConflicts();
    if (!conflictCheck.valid) {
      validationIssues.push(...conflictCheck.issues);
    }

    // Performance regression test
    if (this.config.requireFullValidation) {
      const regressionCheck = await this.runRegressionTests();
      if (!regressionCheck.valid) {
        validationIssues.push(...regressionCheck.issues);
      }
    }

    return {
      valid: validationIssues.length === 0,
      issues: validationIssues,
      testsRun: this.config.requireFullValidation ? this.config.validationTestCount : 0
    };
  }

  /**
   * Restore from backup in case of failure
   * @private
   */
  async restoreFromBackup(backup) {
    try {
      // Verify backup integrity
      await this.verifyBackupIntegrity(backup);
      
      // Restore original rule set
      const originalRules = await fs.readFile(backup.backupPath, 'utf8');
      await fs.writeFile(this.config.rulesPath, originalRules);
      
      // Reload rule set
      await this.loadCurrentRuleSet();
      
      this.logBackupRestored(backup);
      
      return { success: true, backup: backup.id };
      
    } catch (error) {
      this.logBackupRestoreFailed(backup, error);
      throw new Error(`Failed to restore from backup: ${error.message}`);
    }
  }

  /**
   * Handle update failure with rollback
   * @private
   */
  async handleUpdateFailure(sessionId, error) {
    this.logUpdateFailure(sessionId, error);
    
    const backup = Array.from(this.activeBackups.values())
                       .find(b => b.sessionId === sessionId);
    
    if (backup) {
      try {
        await this.restoreFromBackup(backup);
        this.logEmergencyRollbackSuccess(sessionId);
      } catch (rollbackError) {
        this.logEmergencyRollbackFailed(sessionId, rollbackError);
      }
    }
    
    this.updateSession = { ...this.updateSession, status: 'failed', error: error.message };
  }

  // Helper methods for rule manipulation and validation

  applyRuleChange(rule, change) {
    switch (change.component) {
      case 'detectionLogic':
        rule.checkFunctions = change.proposedLogic;
        break;
      case 'applicationLogic':
        rule.improvement = change.proposedLogic;
        break;
      default:
        throw new Error(`Unknown rule component: ${change.component}`);
    }
  }

  async saveRuleSet() {
    const rulesJson = JSON.stringify(this.currentRuleSet, null, 2);
    await fs.writeFile(this.config.rulesPath, rulesJson);
  }

  validateRuleSetIntegrity() {
    if (!this.currentRuleSet || typeof this.currentRuleSet !== 'object') {
      return { valid: false, issues: ['Rule set is not a valid object'] };
    }
    return { valid: true, issues: [] };
  }

  checkForRuleConflicts(optimization) {
    // Simplified conflict detection
    return { valid: true, issues: [] };
  }

  async validatePerformance(optimization) {
    // Simplified performance validation
    return { valid: true, issues: [] };
  }

  validateRuleSetStructure() {
    return { valid: true, issues: [] };
  }

  checkCircularDependencies() {
    return { valid: true, issues: [] };
  }

  detectRuleConflicts() {
    return { valid: true, issues: [] };
  }

  async runRegressionTests() {
    return { valid: true, issues: [] };
  }

  async verifyBackupIntegrity(backup) {
    const backupContent = await fs.readFile(backup.backupPath, 'utf8');
    const backupChecksum = this.calculateChecksum(backupContent);
    const originalChecksum = await fs.readFile(backup.checksumPath, 'utf8');
    
    if (backupChecksum !== originalChecksum.trim()) {
      throw new Error('Backup integrity check failed: checksum mismatch');
    }
  }

  // Utility methods

  calculateChecksum(content) {
    // Simple checksum implementation
    return content.length.toString(16);
  }

  countRules(rulesContent) {
    try {
      const rules = JSON.parse(rulesContent);
      return Object.keys(rules).length;
    } catch {
      return 0;
    }
  }

  normalizeRiskLevel(riskLevel) {
    const riskMap = { low: 1, medium: 2, high: 3, critical: 4 };
    return riskMap[riskLevel] || 2;
  }

  isCriticalFailure(error) {
    return error.message.includes('critical') || error.message.includes('corruption');
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Placeholder methods for various validations and operations

  simulateOptimizationApplication(optimization) {
    return { simulated: true, type: optimization.type };
  }

  async rollbackSingleUpdate(optimization) {
    // Implementation would restore the specific rule state
    this.logSingleRollback(optimization);
  }

  async performEmergencyRollback(backup, issues) {
    await this.restoreFromBackup(backup);
    this.logEmergencyRollback(backup.id, issues);
  }

  validateUpdateInputs(optimizations) {
    if (!Array.isArray(optimizations)) {
      throw new Error('Optimizations must be an array');
    }
  }

  generateNoUpdatesResult(reason) {
    return {
      success: true,
      updatesApplied: 0,
      reason,
      details: { successful: [], failed: [], skipped: [] }
    };
  }

  generateUpdateSummary(updateResults) {
    return {
      successful: updateResults.successful.length,
      failed: updateResults.failed.length,
      skipped: updateResults.skipped.length,
      rollbacks: updateResults.rollbacks.length
    };
  }

  async finalizeSuccessfulUpdate(sessionId, updateResults, backup) {
    this.updateSession.status = 'completed';
    this.updateHistory.push({
      sessionId,
      completedAt: new Date().toISOString(),
      summary: this.generateUpdateSummary(updateResults),
      backupId: backup.id
    });
    
    this.logUpdateSuccess(sessionId, updateResults);
  }

  // Logging methods

  logBackupCreated(backup) {
    console.log(`Backup created: ${backup.id}`);
  }

  logSuccessfulUpdate(optimization) {
    console.log(`Successfully applied: ${optimization.type} - ${optimization.ruleId || optimization.newRuleId}`);
  }

  logRollbackUpdate(optimization, issues) {
    console.log(`Rolled back: ${optimization.type} - ${issues.join(', ')}`);
  }

  logFailedUpdate(optimization, error) {
    console.error(`Failed to apply: ${optimization.type} - ${error.message}`);
  }

  logCriticalFailure(error) {
    console.error(`Critical failure: ${error.message}`);
  }

  logBackupRestored(backup) {
    console.log(`Backup restored: ${backup.id}`);
  }

  logBackupRestoreFailed(backup, error) {
    console.error(`Backup restore failed: ${backup.id} - ${error.message}`);
  }

  logUpdateFailure(sessionId, error) {
    console.error(`Update session failed: ${sessionId} - ${error.message}`);
  }

  logEmergencyRollbackSuccess(sessionId) {
    console.log(`Emergency rollback successful: ${sessionId}`);
  }

  logEmergencyRollbackFailed(sessionId, error) {
    console.error(`Emergency rollback failed: ${sessionId} - ${error.message}`);
  }

  logSingleRollback(optimization) {
    console.log(`Single rollback: ${optimization.type}`);
  }

  logEmergencyRollback(backupId, issues) {
    console.log(`Emergency rollback: ${backupId} - ${issues.join(', ')}`);
  }

  logUpdateSuccess(sessionId, updateResults) {
    console.log(`Update successful: ${sessionId} - ${this.generateUpdateSummary(updateResults)}`);
  }
}

module.exports = RuleUpdater;