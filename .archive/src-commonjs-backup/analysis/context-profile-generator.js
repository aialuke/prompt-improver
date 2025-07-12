/**
 * Context Profile Generation
 * Creates standardized project profiles with rich context information and confidence scores
 */

const FileAnalyzer = require('./file-analyzer');
const DependencyAnalyzer = require('./dependency-analyzer');
const DomainAnalyzer = require('./domain-analyzer');
const Logger = require('../utils/logger');

class ContextProfileGenerator {
  constructor() {
    this.fileAnalyzer = new FileAnalyzer();
    this.dependencyAnalyzer = new DependencyAnalyzer();
    this.domainAnalyzer = new DomainAnalyzer();
    this.logger = new Logger('ContextProfileGenerator');
  }

  /**
   * Generate comprehensive project context profile
   * @param {string} projectPath - Path to project directory
   * @param {Object} options - Analysis options
   * @returns {Promise<Object>} Complete project context profile
   */
  async generateProfile(projectPath, options = {}) {
    const startTime = Date.now();
    this.logger.info('Generating context profile', { projectPath });

    try {
      const profile = {
        metadata: {
          projectPath,
          analysisDate: new Date().toISOString(),
          analysisVersion: '2.0.0',
          options
        },
        technical: {},
        domain: {},
        organizational: {},
        confidence: {},
        recommendations: {},
        summary: {}
      };

      // Step 1: File Pattern Analysis
      this.logger.debug('Starting file pattern analysis');
      const languages = await this.fileAnalyzer.detectLanguages(projectPath);
      const frameworks = await this.fileAnalyzer.detectFrameworks(projectPath);
      const architecture = await this.fileAnalyzer.detectArchitecture(projectPath);

      profile.technical.languages = languages;
      profile.technical.frameworks = frameworks;
      profile.technical.architecture = architecture;

      // Step 2: Dependency Analysis
      this.logger.debug('Starting dependency analysis');
      const languageList = Object.keys(languages);
      const dependencyAnalysis = await this.dependencyAnalyzer.analyzeDependencies(
        projectPath, 
        languageList
      );

      profile.technical.dependencies = dependencyAnalysis;

      // Step 3: Domain Analysis
      this.logger.debug('Starting domain analysis');
      const domainAnalysis = await this.domainAnalyzer.analyzeDomain(
        projectPath,
        dependencyAnalysis,
        architecture
      );

      profile.domain = domainAnalysis;

      // Step 4: Generate Derived Insights
      this.logger.debug('Generating derived insights');
      profile.organizational = this.generateOrganizationalInsights(
        profile.technical,
        profile.domain
      );

      profile.confidence = this.calculateConfidenceScores(profile);
      profile.recommendations = this.generateRecommendations(profile);
      profile.summary = this.generateSummary(profile);

      const executionTime = Date.now() - startTime;
      profile.metadata.executionTime = executionTime;

      this.logger.info('Context profile generated successfully', {
        executionTime,
        languagesDetected: Object.keys(languages).length,
        frameworksDetected: Object.keys(frameworks).length,
        domain: profile.domain.domain,
        projectType: profile.domain.projectType
      });

      return profile;

    } catch (error) {
      this.logger.error('Context profile generation failed', error);
      throw new Error(`Context profile generation failed: ${error.message}`);
    }
  }

  /**
   * Generate organizational and team insights
   * @private
   */
  generateOrganizationalInsights(technical, domain) {
    const insights = {
      teamSize: this.inferTeamSize(technical, domain),
      developmentPhase: this.inferDevelopmentPhase(technical, domain),
      technicalDebt: this.assessTechnicalDebt(technical),
      scalabilityNeeds: this.assessScalabilityNeeds(technical, domain),
      securityConsiderations: this.assessSecurityNeeds(domain),
      performanceProfile: this.assessPerformanceProfile(technical, domain)
    };

    return insights;
  }

  /**
   * Infer team size from project characteristics
   * @private
   */
  inferTeamSize(technical, domain) {
    let sizeScore = 0;

    // Language diversity indicates larger teams
    const languageCount = Object.keys(technical.languages).length;
    if (languageCount >= 3) sizeScore += 0.4;
    else if (languageCount >= 2) sizeScore += 0.2;

    // Framework diversity
    const frameworkCount = Object.keys(technical.frameworks).length;
    if (frameworkCount >= 5) sizeScore += 0.3;
    else if (frameworkCount >= 3) sizeScore += 0.2;

    // Architecture complexity
    const architectureCount = Object.keys(technical.architecture).length;
    if (architectureCount >= 3) sizeScore += 0.3;

    // Domain complexity
    if (domain.complexity === 'high') sizeScore += 0.3;
    else if (domain.complexity === 'medium') sizeScore += 0.2;

    // Microservices indicates larger teams
    if (technical.dependencies.summary?.architecturalPatterns?.includes('microservices')) {
      sizeScore += 0.4;
    }

    // Project maturity
    if (domain.maturity === 'production') sizeScore += 0.2;

    // Categorize team size
    if (sizeScore >= 1.0) return 'large'; // 10+ developers
    if (sizeScore >= 0.6) return 'medium'; // 4-10 developers
    if (sizeScore >= 0.3) return 'small'; // 2-4 developers
    return 'solo'; // 1 developer
  }

  /**
   * Infer development phase
   * @private
   */
  inferDevelopmentPhase(technical, domain) {
    const phases = {
      prototype: 0,
      mvp: 0,
      development: 0,
      production: 0,
      maintenance: 0
    };

    // Maturity strongly influences phase
    switch (domain.maturity) {
      case 'prototype':
        phases.prototype += 0.6;
        phases.mvp += 0.3;
        break;
      case 'development':
        phases.development += 0.5;
        phases.mvp += 0.3;
        phases.production += 0.2;
        break;
      case 'production':
        phases.production += 0.6;
        phases.maintenance += 0.4;
        break;
    }

    // Framework maturity influences phase
    const hasModernFrameworks = Object.keys(technical.frameworks).some(fw =>
      ['react', 'vue', 'angular', 'nextjs', 'fastapi', 'nestjs'].includes(fw)
    );

    if (hasModernFrameworks) {
      phases.development += 0.2;
      phases.production += 0.1;
    }

    // Testing presence indicates mature development
    const hasTesting = Object.keys(technical.frameworks).some(fw =>
      ['jest', 'cypress', 'playwright', 'pytest'].includes(fw)
    );

    if (hasTesting) {
      phases.development += 0.2;
      phases.production += 0.2;
    }

    // Return phase with highest score
    return Object.entries(phases)
      .sort(([,a], [,b]) => b - a)[0][0];
  }

  /**
   * Assess technical debt indicators
   * @private
   */
  assessTechnicalDebt(technical) {
    const debt = {
      level: 'low',
      indicators: [],
      score: 0
    };

    // Legacy framework usage
    const legacyFrameworks = ['jquery', 'backbone', 'ember'];
    const hasLegacy = Object.keys(technical.frameworks).some(fw =>
      legacyFrameworks.includes(fw)
    );

    if (hasLegacy) {
      debt.score += 0.3;
      debt.indicators.push('Legacy framework usage detected');
    }

    // Mixed language ecosystems without clear separation
    const languageCount = Object.keys(technical.languages).length;
    if (languageCount > 2) {
      debt.score += 0.2;
      debt.indicators.push('Multiple language ecosystems may indicate fragmentation');
    }

    // Too many frameworks for project size
    const frameworkCount = Object.keys(technical.frameworks).length;
    if (frameworkCount > 8) {
      debt.score += 0.3;
      debt.indicators.push('High framework count may indicate over-engineering');
    }

    // Categorize debt level
    if (debt.score >= 0.6) debt.level = 'high';
    else if (debt.score >= 0.3) debt.level = 'medium';

    return debt;
  }

  /**
   * Assess scalability needs
   * @private
   */
  assessScalabilityNeeds(technical, domain) {
    const scalability = {
      level: 'low',
      concerns: [],
      recommendations: []
    };

    // Domain-based scalability needs
    const highScalabilityDomains = ['e-commerce', 'social-media', 'fintech', 'analytics'];
    if (highScalabilityDomains.includes(domain.domain)) {
      scalability.level = 'high';
      scalability.concerns.push(`${domain.domain} domain typically requires high scalability`);
    }

    // Architecture patterns indicating scalability needs
    const patterns = technical.dependencies.summary?.architecturalPatterns || [];
    if (patterns.includes('microservices')) {
      scalability.level = 'high';
      scalability.concerns.push('Microservices architecture suggests scalability focus');
    }

    // Database considerations
    const databases = technical.dependencies.databases || [];
    if (databases.length > 1) {
      scalability.concerns.push('Multiple databases may indicate scaling strategy');
      if (scalability.level === 'low') scalability.level = 'medium';
    }

    // Generate recommendations
    if (scalability.level === 'high') {
      scalability.recommendations.push('Consider caching strategies');
      scalability.recommendations.push('Implement database sharding/replication');
      scalability.recommendations.push('Use CDN for static assets');
      scalability.recommendations.push('Implement rate limiting');
    }

    return scalability;
  }

  /**
   * Assess security considerations based on domain
   * @private
   */
  assessSecurityNeeds(domain) {
    const security = {
      level: 'standard',
      concerns: [],
      recommendations: []
    };

    // High-security domains
    const highSecurityDomains = ['fintech', 'healthcare', 'e-commerce'];
    if (highSecurityDomains.includes(domain.domain)) {
      security.level = 'high';
      security.concerns.push(`${domain.domain} domain requires enhanced security measures`);
    }

    // Generate domain-specific recommendations
    switch (domain.domain) {
      case 'fintech':
        security.recommendations.push('Implement PCI DSS compliance');
        security.recommendations.push('Use multi-factor authentication');
        security.recommendations.push('Implement transaction monitoring');
        break;

      case 'healthcare':
        security.recommendations.push('Ensure HIPAA compliance');
        security.recommendations.push('Implement data encryption at rest and in transit');
        security.recommendations.push('Audit access logs regularly');
        break;

      case 'e-commerce':
        security.recommendations.push('Secure payment processing');
        security.recommendations.push('Implement input validation');
        security.recommendations.push('Use HTTPS everywhere');
        break;

      default:
        security.recommendations.push('Implement basic authentication');
        security.recommendations.push('Use HTTPS for production');
        security.recommendations.push('Sanitize user inputs');
    }

    return security;
  }

  /**
   * Assess performance profile
   * @private
   */
  assessPerformanceProfile(technical, domain) {
    const performance = {
      priority: 'medium',
      concerns: [],
      optimizations: []
    };

    // Performance-critical domains
    const performanceCriticalDomains = ['gaming', 'fintech', 'analytics', 'iot'];
    if (performanceCriticalDomains.includes(domain.domain)) {
      performance.priority = 'high';
      performance.concerns.push(`${domain.domain} domain is performance-critical`);
    }

    // Framework-based performance considerations
    const frameworks = Object.keys(technical.frameworks);
    
    if (frameworks.includes('react')) {
      performance.optimizations.push('Consider React.memo for expensive components');
      performance.optimizations.push('Implement code splitting with lazy loading');
    }

    if (frameworks.includes('vue')) {
      performance.optimizations.push('Use Vue 3 Composition API for better performance');
      performance.optimizations.push('Implement virtual scrolling for large lists');
    }

    if (frameworks.includes('nodejs') || frameworks.includes('express')) {
      performance.optimizations.push('Implement caching middleware');
      performance.optimizations.push('Use clustering for CPU-intensive tasks');
    }

    // Add general recommendations based on priority
    if (performance.priority === 'high') {
      performance.optimizations.push('Implement performance monitoring');
      performance.optimizations.push('Use compression for responses');
      performance.optimizations.push('Optimize database queries');
    }

    return performance;
  }

  /**
   * Calculate confidence scores for different aspects
   * @private
   */
  calculateConfidenceScores(profile) {
    const confidence = {
      overall: 0,
      technical: 0,
      domain: 0,
      organizational: 0,
      breakdown: {}
    };

    // Technical confidence
    const languageConfidences = Object.values(profile.technical.languages)
      .map(lang => lang.confidence);
    const frameworkConfidences = Object.values(profile.technical.frameworks)
      .map(fw => fw.confidence);
    const architectureConfidences = Object.values(profile.technical.architecture)
      .map(arch => arch.confidence);

    confidence.technical = this.calculateAverageConfidence([
      ...languageConfidences,
      ...frameworkConfidences,
      ...architectureConfidences
    ]);

    // Domain confidence
    confidence.domain = profile.domain.confidence || 0;

    // Organizational confidence (based on data availability)
    const orgDataPoints = [
      profile.technical.dependencies.summary?.frameworkCount || 0,
      Object.keys(profile.technical.languages).length,
      profile.domain.maturity ? 1 : 0,
      profile.domain.complexity ? 1 : 0
    ];

    confidence.organizational = Math.min(
      orgDataPoints.reduce((sum, point) => sum + (point > 0 ? 0.25 : 0), 0),
      1.0
    );

    // Overall confidence
    confidence.overall = (
      confidence.technical * 0.4 +
      confidence.domain * 0.4 +
      confidence.organizational * 0.2
    );

    // Detailed breakdown
    confidence.breakdown = {
      languages: this.calculateAverageConfidence(languageConfidences),
      frameworks: this.calculateAverageConfidence(frameworkConfidences),
      architecture: this.calculateAverageConfidence(architectureConfidences),
      domain: confidence.domain,
      projectType: profile.domain.projectTypeConfidence || 0
    };

    return confidence;
  }

  /**
   * Generate recommendations based on profile analysis
   * @private
   */
  generateRecommendations(profile) {
    const recommendations = {
      priority: {
        high: [],
        medium: [],
        low: []
      },
      categories: {
        technical: [],
        process: [],
        security: [],
        performance: [],
        scalability: []
      }
    };

    // Technical recommendations
    if (profile.confidence.technical < 0.7) {
      recommendations.priority.high.push('Improve technical documentation and standardization');
      recommendations.categories.technical.push('Consider consolidating technology stack');
    }

    // Domain-specific recommendations
    if (profile.domain.confidence < 0.6) {
      recommendations.priority.medium.push('Clarify project domain and business requirements');
      recommendations.categories.process.push('Document business context and use cases');
    }

    // Security recommendations
    if (profile.organizational.securityConsiderations.level === 'high') {
      recommendations.priority.high.push(...profile.organizational.securityConsiderations.recommendations);
      recommendations.categories.security.push(...profile.organizational.securityConsiderations.recommendations);
    }

    // Performance recommendations
    if (profile.organizational.performanceProfile.priority === 'high') {
      recommendations.priority.high.push('Implement performance monitoring');
      recommendations.categories.performance.push(...profile.organizational.performanceProfile.optimizations);
    }

    // Scalability recommendations
    if (profile.organizational.scalabilityNeeds.level === 'high') {
      recommendations.priority.medium.push(...profile.organizational.scalabilityNeeds.recommendations);
      recommendations.categories.scalability.push(...profile.organizational.scalabilityNeeds.recommendations);
    }

    // Technical debt recommendations
    if (profile.organizational.technicalDebt.level === 'high') {
      recommendations.priority.high.push('Address technical debt and legacy dependencies');
      recommendations.categories.technical.push('Migrate from legacy frameworks');
      recommendations.categories.technical.push('Consolidate overlapping dependencies');
    }

    return recommendations;
  }

  /**
   * Generate executive summary
   * @private
   */
  generateSummary(profile) {
    const summary = {
      title: this.generateProjectTitle(profile),
      description: this.generateProjectDescription(profile),
      keyCharacteristics: this.extractKeyCharacteristics(profile),
      riskFactors: this.identifyRiskFactors(profile),
      opportunities: this.identifyOpportunities(profile),
      nextSteps: this.suggestNextSteps(profile)
    };

    return summary;
  }

  /**
   * Generate project title
   * @private
   */
  generateProjectTitle(profile) {
    const domain = profile.domain.domain;
    const projectType = profile.domain.projectType;
    const primaryLanguage = Object.entries(profile.technical.languages)
      .sort(([,a], [,b]) => b.confidence - a.confidence)[0]?.[0];
    const primaryFramework = Object.entries(profile.technical.frameworks)
      .sort(([,a], [,b]) => b.confidence - a.confidence)[0]?.[0];

    let title = '';

    if (domain && domain !== 'general') {
      title += domain.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    if (projectType && projectType !== 'application') {
      if (title) title += ' ';
      title += projectType.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    if (primaryFramework) {
      if (title) title += ' using ';
      title += primaryFramework.charAt(0).toUpperCase() + primaryFramework.slice(1);
    } else if (primaryLanguage) {
      if (title) title += ' in ';
      title += primaryLanguage.charAt(0).toUpperCase() + primaryLanguage.slice(1);
    }

    return title || 'Software Application';
  }

  /**
   * Generate project description
   * @private
   */
  generateProjectDescription(profile) {
    const teamSize = profile.organizational.teamSize;
    const phase = profile.organizational.developmentPhase;
    const complexity = profile.domain.complexity;
    const maturity = profile.domain.maturity;

    const descriptions = [
      `A ${complexity}-complexity ${profile.domain.projectType || 'application'}`,
      `in the ${phase} phase`,
      `built for the ${profile.domain.domain || 'general'} domain`,
      `by a ${teamSize} team`,
      `with ${maturity} maturity level`
    ];

    return descriptions.join(' ');
  }

  /**
   * Extract key characteristics
   * @private
   */
  extractKeyCharacteristics(profile) {
    const characteristics = [];

    // Technical characteristics
    const languageCount = Object.keys(profile.technical.languages).length;
    if (languageCount > 1) {
      characteristics.push(`Multi-language (${languageCount} languages)`);
    }

    const primaryFrameworks = Object.entries(profile.technical.frameworks)
      .filter(([,config]) => config.confidence > 0.7)
      .map(([name]) => name);

    if (primaryFrameworks.length > 0) {
      characteristics.push(`Uses ${primaryFrameworks.slice(0, 3).join(', ')}`);
    }

    // Architecture characteristics
    const architecturePatterns = profile.technical.dependencies.summary?.architecturalPatterns || [];
    if (architecturePatterns.length > 0) {
      characteristics.push(`Architecture: ${architecturePatterns.join(', ')}`);
    }

    // Domain characteristics
    if (profile.domain.businessContext) {
      characteristics.push(`${profile.domain.businessContext} context`);
    }

    return characteristics;
  }

  /**
   * Identify risk factors
   * @private
   */
  identifyRiskFactors(profile) {
    const risks = [];

    // Low confidence risks
    if (profile.confidence.overall < 0.6) {
      risks.push('Low analysis confidence - may need more project information');
    }

    // Technical debt risks
    if (profile.organizational.technicalDebt.level === 'high') {
      risks.push('High technical debt level detected');
    }

    // Security risks
    if (profile.organizational.securityConsiderations.level === 'high' &&
        profile.organizational.securityConsiderations.concerns.length > 0) {
      risks.push('High security requirements for domain');
    }

    // Scalability risks
    if (profile.organizational.scalabilityNeeds.level === 'high') {
      risks.push('High scalability requirements detected');
    }

    return risks;
  }

  /**
   * Identify opportunities
   * @private
   */
  identifyOpportunities(profile) {
    const opportunities = [];

    // Modern framework opportunities
    const modernFrameworks = ['react', 'vue', 'angular', 'nextjs', 'fastapi'];
    const hasModernFramework = Object.keys(profile.technical.frameworks)
      .some(fw => modernFrameworks.includes(fw));

    if (hasModernFramework) {
      opportunities.push('Modern framework stack enables rapid development');
    }

    // Domain opportunities
    const growthDomains = ['e-commerce', 'fintech', 'analytics', 'iot'];
    if (growthDomains.includes(profile.domain.domain)) {
      opportunities.push(`${profile.domain.domain} is a high-growth domain`);
    }

    // Team size opportunities
    if (profile.organizational.teamSize === 'small') {
      opportunities.push('Small team enables rapid iteration and decision-making');
    }

    return opportunities;
  }

  /**
   * Suggest next steps
   * @private
   */
  suggestNextSteps(profile) {
    const steps = [];

    // Based on development phase
    switch (profile.organizational.developmentPhase) {
      case 'prototype':
        steps.push('Focus on core feature validation');
        steps.push('Prepare MVP roadmap');
        break;
      case 'mvp':
        steps.push('Gather user feedback');
        steps.push('Plan feature expansion');
        break;
      case 'development':
        steps.push('Implement testing strategy');
        steps.push('Plan production deployment');
        break;
      case 'production':
        steps.push('Monitor performance metrics');
        steps.push('Plan scaling strategy');
        break;
      case 'maintenance':
        steps.push('Assess modernization opportunities');
        steps.push('Plan technical debt reduction');
        break;
    }

    // High-priority recommendations
    const highPriorityRecs = profile.recommendations.priority.high.slice(0, 2);
    steps.push(...highPriorityRecs);

    return steps;
  }

  /**
   * Calculate average confidence from array of scores
   * @private
   */
  calculateAverageConfidence(confidences) {
    if (confidences.length === 0) return 0;
    return confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
  }

  /**
   * Validate and enrich profile data
   * @param {Object} profile - Generated profile
   * @returns {Object} Validated and enriched profile
   */
  validateProfile(profile) {
    // Ensure all required sections exist
    const requiredSections = ['metadata', 'technical', 'domain', 'organizational', 'confidence', 'recommendations', 'summary'];
    
    for (const section of requiredSections) {
      if (!profile[section]) {
        profile[section] = {};
      }
    }

    // Ensure confidence scores are valid
    for (const [key, value] of Object.entries(profile.confidence)) {
      if (typeof value === 'number') {
        profile.confidence[key] = Math.max(0, Math.min(1, value));
      }
    }

    // Add metadata
    profile.metadata.profileVersion = '2.0.0';
    profile.metadata.generatedBy = 'Universal Prompt Testing Framework';

    return profile;
  }
}

module.exports = ContextProfileGenerator;