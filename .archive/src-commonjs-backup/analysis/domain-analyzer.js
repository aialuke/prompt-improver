/**
 * Domain Context Inference Engine
 * Analyzes project documentation and code patterns to identify domain and project type
 */

const path = require('path');
const FileHandler = require('../utils/file-handler');
const Logger = require('../utils/logger');

class DomainAnalyzer {
  constructor() {
    this.fileHandler = new FileHandler();
    this.logger = new Logger('DomainAnalyzer');

    // Domain keyword patterns
    this.domainKeywords = {
      'e-commerce': {
        keywords: [
          'cart', 'checkout', 'payment', 'product', 'shop', 'store', 'order', 'invoice',
          'inventory', 'catalog', 'price', 'discount', 'coupon', 'shipping', 'customer',
          'vendor', 'marketplace', 'ecommerce', 'retail', 'sales', 'transaction'
        ],
        codePatterns: [
          /class.*Product/, /class.*Order/, /class.*Cart/, /class.*Payment/,
          /checkout/, /addToCart/, /removeFromCart/, /calculateTotal/,
          /stripe|paypal|square/, /\$price|\$total|\$amount/
        ],
        confidence: 0.9
      },

      'fintech': {
        keywords: [
          'bank', 'finance', 'payment', 'transaction', 'account', 'balance', 'credit',
          'debit', 'loan', 'investment', 'trading', 'portfolio', 'cryptocurrency',
          'blockchain', 'wallet', 'exchange', 'kyc', 'aml', 'compliance', 'audit'
        ],
        codePatterns: [
          /class.*Account/, /class.*Transaction/, /class.*Payment/, /class.*Wallet/,
          /calculateInterest/, /validateTransaction/, /processPayment/,
          /bitcoin|ethereum|crypto/, /\$balance|\$amount/, /kyc|aml/
        ],
        confidence: 0.95
      },

      'healthcare': {
        keywords: [
          'patient', 'doctor', 'medical', 'health', 'hospital', 'clinic', 'appointment',
          'prescription', 'diagnosis', 'treatment', 'medicine', 'pharmacy', 'lab',
          'insurance', 'hipaa', 'ehr', 'emr', 'telehealth', 'telemedicine'
        ],
        codePatterns: [
          /class.*Patient/, /class.*Doctor/, /class.*Appointment/, /class.*Prescription/,
          /scheduleAppointment/, /prescribeMedicine/, /patientRecord/,
          /hipaa|gdpr/, /medical.*record/
        ],
        confidence: 0.95
      },

      'education': {
        keywords: [
          'student', 'teacher', 'course', 'lesson', 'curriculum', 'grade', 'exam',
          'assignment', 'homework', 'quiz', 'school', 'university', 'college',
          'learning', 'education', 'tutor', 'classroom', 'lms', 'blackboard'
        ],
        codePatterns: [
          /class.*Student/, /class.*Teacher/, /class.*Course/, /class.*Assignment/,
          /enrollStudent/, /gradeAssignment/, /createCourse/,
          /grade|score/, /enrollment/
        ],
        confidence: 0.9
      },

      'gaming': {
        keywords: [
          'game', 'player', 'level', 'score', 'achievement', 'leaderboard', 'quest',
          'character', 'avatar', 'weapon', 'item', 'inventory', 'guild', 'clan',
          'tournament', 'match', 'unity', 'unreal', 'godot', 'phaser'
        ],
        codePatterns: [
          /class.*Player/, /class.*Game/, /class.*Level/, /class.*Character/,
          /updateScore/, /levelUp/, /spawnEnemy/, /collectItem/,
          /Unity|Unreal|Godot/, /GameObject|Entity/
        ],
        confidence: 0.9
      },

      'social-media': {
        keywords: [
          'user', 'post', 'comment', 'like', 'share', 'follow', 'friend', 'feed',
          'timeline', 'notification', 'message', 'chat', 'profile', 'social',
          'community', 'network', 'content', 'media', 'upload'
        ],
        codePatterns: [
          /class.*User/, /class.*Post/, /class.*Comment/, /class.*Message/,
          /createPost/, /likePost/, /followUser/, /sendMessage/,
          /timeline|feed/, /notification/
        ],
        confidence: 0.85
      },

      'analytics': {
        keywords: [
          'analytics', 'metrics', 'dashboard', 'report', 'chart', 'graph', 'data',
          'visualization', 'kpi', 'tracking', 'measurement', 'statistics', 'insights',
          'performance', 'monitoring', 'business intelligence', 'bi', 'etl'
        ],
        codePatterns: [
          /class.*Metric/, /class.*Report/, /class.*Dashboard/, /class.*Chart/,
          /trackEvent/, /generateReport/, /calculateMetrics/,
          /d3|chart\.js|plotly/, /aggregat|sum|count|average/
        ],
        confidence: 0.85
      },

      'productivity': {
        keywords: [
          'task', 'todo', 'project', 'team', 'collaboration', 'document', 'file',
          'workspace', 'calendar', 'schedule', 'meeting', 'note', 'reminder',
          'organization', 'productivity', 'workflow', 'automation'
        ],
        codePatterns: [
          /class.*Task/, /class.*Project/, /class.*Team/, /class.*Document/,
          /createTask/, /scheduleEvent/, /assignTask/, /uploadFile/,
          /calendar|schedule/, /workflow/
        ],
        confidence: 0.8
      },

      'iot': {
        keywords: [
          'sensor', 'device', 'hardware', 'embedded', 'arduino', 'raspberry',
          'mqtt', 'zigbee', 'bluetooth', 'wifi', 'gateway', 'edge', 'telemetry',
          'automation', 'smart', 'connected', 'protocol', 'firmware'
        ],
        codePatterns: [
          /class.*Sensor/, /class.*Device/, /class.*Gateway/,
          /readSensor/, /sendTelemetry/, /deviceConfig/,
          /mqtt|zigbee|bluetooth/, /Arduino|RaspberryPi/
        ],
        confidence: 0.9
      },

      'api-service': {
        keywords: [
          'api', 'endpoint', 'rest', 'graphql', 'microservice', 'service', 'server',
          'backend', 'middleware', 'authentication', 'authorization', 'jwt', 'oauth',
          'rate limiting', 'cors', 'webhook', 'integration'
        ],
        codePatterns: [
          /app\.get|app\.post|app\.put|app\.delete/, /router\.|route\./,
          /@app\.route|@api\.route/, /express\(\)|fastify\(\)/,
          /jwt|oauth/, /middleware/, /cors/
        ],
        confidence: 0.85
      }
    };

    // Project type patterns
    this.projectTypes = {
      'web-application': {
        indicators: [
          'html', 'css', 'javascript', 'react', 'vue', 'angular', 'frontend',
          'spa', 'webapp', 'website', 'web app', 'user interface', 'ui'
        ],
        architecturePatterns: ['frontend', 'spa', 'fullstack'],
        frameworks: ['react', 'vue', 'angular', 'svelte', 'nextjs'],
        confidence: 0.9
      },

      'api-service': {
        indicators: [
          'api', 'rest', 'graphql', 'server', 'backend', 'microservice',
          'endpoint', 'service', 'web service', 'http'
        ],
        architecturePatterns: ['backend', 'api', 'microservices'],
        frameworks: ['express', 'fastify', 'nestjs', 'django', 'flask', 'fastapi'],
        confidence: 0.9
      },

      'mobile-app': {
        indicators: [
          'mobile', 'app', 'ios', 'android', 'react native', 'flutter',
          'xamarin', 'cordova', 'ionic', 'phone', 'tablet'
        ],
        architecturePatterns: ['mobile'],
        frameworks: ['flutter', 'react-native'],
        files: ['pubspec.yaml', 'android/', 'ios/'],
        confidence: 0.95
      },

      'desktop-app': {
        indicators: [
          'desktop', 'electron', 'tauri', 'qt', 'tkinter', 'javafx',
          'wpf', 'winforms', 'native', 'cross-platform'
        ],
        architecturePatterns: ['desktop'],
        files: ['main.js', 'main.rs', 'tauri.conf.json'],
        confidence: 0.9
      },

      'cli-tool': {
        indicators: [
          'cli', 'command line', 'terminal', 'console', 'script', 'tool',
          'utility', 'automation', 'command', 'bin'
        ],
        architecturePatterns: ['cli'],
        files: ['bin/', 'cli.js', 'main.rs'],
        codePatterns: [/argparse|commander|clap/, /#!/],
        confidence: 0.85
      },

      'library': {
        indicators: [
          'library', 'package', 'module', 'sdk', 'framework', 'component',
          'utility', 'helper', 'npm package', 'pip package'
        ],
        architecturePatterns: ['library'],
        files: ['lib/', 'src/index.js', 'setup.py'],
        confidence: 0.8
      },

      'machine-learning': {
        indicators: [
          'machine learning', 'ml', 'ai', 'artificial intelligence', 'neural network',
          'deep learning', 'data science', 'model', 'training', 'inference'
        ],
        frameworks: ['tensorflow', 'pytorch', 'scikit_learn'],
        files: ['model.py', 'train.py', 'data/'],
        confidence: 0.95
      },

      'data-processing': {
        indicators: [
          'data', 'etl', 'pipeline', 'processing', 'analytics', 'warehouse',
          'lake', 'batch', 'stream', 'transform', 'extract', 'load'
        ],
        frameworks: ['pandas', 'numpy'],
        architecturePatterns: ['analytics'],
        confidence: 0.85
      },

      'devops-automation': {
        indicators: [
          'devops', 'automation', 'deployment', 'ci/cd', 'infrastructure',
          'terraform', 'ansible', 'kubernetes', 'docker', 'pipeline'
        ],
        architecturePatterns: ['devops'],
        files: ['Dockerfile', 'docker-compose.yml', '.github/workflows/'],
        confidence: 0.9
      },

      'game': {
        indicators: [
          'game', 'gaming', 'unity', 'unreal', 'godot', 'phaser', 'pygame',
          '2d', '3d', 'graphics', 'physics', 'rendering'
        ],
        frameworks: ['unity', 'unreal', 'godot', 'phaser'],
        confidence: 0.95
      }
    };

    // Business context indicators
    this.businessContexts = {
      'startup': {
        indicators: [
          'mvp', 'prototype', 'beta', 'early stage', 'seed', 'startup',
          'lean', 'agile', 'rapid development', 'iteration'
        ],
        codeIndicators: ['TODO', 'FIXME', 'HACK', 'temporary'],
        confidence: 0.7
      },

      'enterprise': {
        indicators: [
          'enterprise', 'corporate', 'large scale', 'compliance', 'security',
          'governance', 'audit', 'sla', 'enterprise grade'
        ],
        codeIndicators: ['interface', 'abstract', 'factory', 'strategy'],
        confidence: 0.8
      },

      'open-source': {
        indicators: [
          'open source', 'community', 'contributor', 'license', 'mit', 'apache',
          'gpl', 'public', 'github', 'contribution'
        ],
        files: ['LICENSE', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md'],
        confidence: 0.9
      }
    };
  }

  /**
   * Analyze project domain and context
   * @param {string} projectPath - Path to project directory
   * @param {Object} dependencies - Dependency analysis results
   * @param {Object} architecture - Architecture analysis results
   * @returns {Promise<Object>} Domain analysis results
   */
  async analyzeDomain(projectPath, dependencies = {}, architecture = {}) {
    this.logger.debug('Analyzing domain context', { projectPath });

    try {
      const analysis = {
        domain: null,
        confidence: 0,
        projectType: null,
        businessContext: null,
        indicators: [],
        codePatterns: [],
        documentationAnalysis: {},
        complexity: 'medium',
        maturity: 'development'
      };

      // Analyze documentation
      analysis.documentationAnalysis = await this.analyzeDocumentation(projectPath);

      // Analyze code patterns
      analysis.codePatterns = await this.analyzeCodePatterns(projectPath);

      // Infer domain from multiple sources
      const domainResults = this.inferDomain(
        analysis.documentationAnalysis,
        analysis.codePatterns,
        dependencies,
        architecture
      );

      analysis.domain = domainResults.domain;
      analysis.confidence = domainResults.confidence;
      analysis.indicators = domainResults.indicators;

      // Infer project type
      const projectTypeResults = this.inferProjectType(
        dependencies,
        architecture,
        analysis.documentationAnalysis
      );

      analysis.projectType = projectTypeResults.type;
      analysis.projectTypeConfidence = projectTypeResults.confidence;

      // Infer business context
      analysis.businessContext = this.inferBusinessContext(
        analysis.documentationAnalysis,
        projectPath
      );

      // Assess complexity and maturity
      analysis.complexity = this.assessComplexity(dependencies, architecture);
      analysis.maturity = await this.assessMaturity(projectPath);

      this.logger.debug('Domain analysis completed', {
        domain: analysis.domain,
        projectType: analysis.projectType,
        confidence: analysis.confidence
      });

      return analysis;

    } catch (error) {
      this.logger.error('Domain analysis failed', error);
      throw new Error(`Domain analysis failed: ${error.message}`);
    }
  }

  /**
   * Analyze project documentation for domain indicators
   * @private
   */
  async analyzeDocumentation(projectPath) {
    const analysis = {
      readme: {},
      packageInfo: {},
      comments: {},
      totalWordCount: 0,
      keywordMatches: {}
    };

    try {
      // Analyze README files
      const readmeFiles = ['README.md', 'README.txt', 'README.rst', 'readme.md'];
      for (const readmeFile of readmeFiles) {
        const readmePath = path.join(projectPath, readmeFile);
        if (await this.fileHandler.exists(readmePath)) {
          const content = await this.fileHandler.readFile(readmePath);
          analysis.readme = this.analyzeText(content);
          break;
        }
      }

      // Analyze package.json description
      const packageJsonPath = path.join(projectPath, 'package.json');
      if (await this.fileHandler.exists(packageJsonPath)) {
        const packageJson = await this.fileHandler.readJSON(packageJsonPath);
        if (packageJson.description) {
          analysis.packageInfo = this.analyzeText(packageJson.description);
        }
      }

      // Analyze code comments (sample)
      const files = await this.fileHandler.listFiles(projectPath, {
        recursive: true,
        extensions: ['.js', '.ts', '.py', '.java', '.go', '.rs'],
        excludePatterns: ['node_modules', '.git', 'dist', 'build']
      });

      let commentText = '';
      for (const file of files.slice(0, 10)) { // Sample first 10 files
        const content = await this.getFileContent(file.path);
        if (content) {
          const comments = this.extractComments(content, path.extname(file.name));
          commentText += comments + ' ';
        }
      }

      if (commentText.trim()) {
        analysis.comments = this.analyzeText(commentText);
      }

      // Aggregate keyword matches
      analysis.keywordMatches = this.aggregateKeywordMatches([
        analysis.readme,
        analysis.packageInfo,
        analysis.comments
      ]);

      analysis.totalWordCount = 
        (analysis.readme.wordCount || 0) +
        (analysis.packageInfo.wordCount || 0) +
        (analysis.comments.wordCount || 0);

    } catch (error) {
      this.logger.warn('Documentation analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze code patterns for domain indicators
   * @private
   */
  async analyzeCodePatterns(projectPath) {
    const patterns = {
      classNames: [],
      functionNames: [],
      variableNames: [],
      imports: [],
      domainMatches: {}
    };

    try {
      const files = await this.fileHandler.listFiles(projectPath, {
        recursive: true,
        extensions: ['.js', '.ts', '.py', '.java', '.go', '.rs'],
        excludePatterns: ['node_modules', '.git', 'dist', 'build', '__pycache__']
      });

      // Sample files for analysis
      const sampleFiles = files.slice(0, 20);

      for (const file of sampleFiles) {
        const content = await this.getFileContent(file.path);
        if (content) {
          const filePatterns = this.extractCodePatterns(content, path.extname(file.name));
          
          patterns.classNames.push(...filePatterns.classNames);
          patterns.functionNames.push(...filePatterns.functionNames);
          patterns.variableNames.push(...filePatterns.variableNames);
          patterns.imports.push(...filePatterns.imports);
        }
      }

      // Check for domain-specific code patterns
      patterns.domainMatches = this.matchDomainPatterns(patterns);

    } catch (error) {
      this.logger.warn('Code pattern analysis partial failure', error);
    }

    return patterns;
  }

  /**
   * Analyze text content for keywords and patterns
   * @private
   */
  analyzeText(text) {
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2);

    const analysis = {
      wordCount: words.length,
      uniqueWords: [...new Set(words)].length,
      domainKeywords: {},
      businessKeywords: {}
    };

    // Match domain keywords
    for (const [domain, config] of Object.entries(this.domainKeywords)) {
      const matches = config.keywords.filter(keyword => 
        text.toLowerCase().includes(keyword.toLowerCase())
      );
      
      if (matches.length > 0) {
        analysis.domainKeywords[domain] = {
          matches,
          score: matches.length / config.keywords.length,
          confidence: config.confidence
        };
      }
    }

    // Match business context keywords
    for (const [context, config] of Object.entries(this.businessContexts)) {
      const matches = config.indicators.filter(indicator =>
        text.toLowerCase().includes(indicator.toLowerCase())
      );

      if (matches.length > 0) {
        analysis.businessKeywords[context] = {
          matches,
          score: matches.length / config.indicators.length
        };
      }
    }

    return analysis;
  }

  /**
   * Extract comments from code
   * @private
   */
  extractComments(content, extension) {
    let comments = '';

    try {
      switch (extension) {
        case '.js':
        case '.ts':
        case '.java':
        case '.go':
        case '.rs':
          // Single line comments
          const singleLineComments = content.match(/\/\/.*$/gm) || [];
          // Multi-line comments
          const multiLineComments = content.match(/\/\*[\s\S]*?\*\//g) || [];
          comments = [...singleLineComments, ...multiLineComments].join(' ');
          break;

        case '.py':
          // Python comments
          const pythonComments = content.match(/#.*$/gm) || [];
          const pythonDocstrings = content.match(/"""[\s\S]*?"""/g) || [];
          comments = [...pythonComments, ...pythonDocstrings].join(' ');
          break;
      }
    } catch (error) {
      // Ignore regex errors
    }

    return comments;
  }

  /**
   * Extract code patterns (class names, function names, etc.)
   * @private
   */
  extractCodePatterns(content, extension) {
    const patterns = {
      classNames: [],
      functionNames: [],
      variableNames: [],
      imports: []
    };

    try {
      switch (extension) {
        case '.js':
        case '.ts':
          // Classes
          const jsClasses = content.match(/class\s+(\w+)/g) || [];
          patterns.classNames = jsClasses.map(match => match.replace('class ', ''));

          // Functions
          const jsFunctions = content.match(/function\s+(\w+)|(\w+)\s*:/g) || [];
          patterns.functionNames = jsFunctions.map(match => 
            match.replace(/function\s+|:\s*$/g, '')
          );

          // Imports
          const jsImports = content.match(/import.*from\s+['"]([^'"]+)['"]/g) || [];
          patterns.imports = jsImports.map(match => 
            match.match(/from\s+['"]([^'"]+)['"]/)?.[1] || ''
          );
          break;

        case '.py':
          // Classes
          const pyClasses = content.match(/class\s+(\w+)/g) || [];
          patterns.classNames = pyClasses.map(match => match.replace('class ', ''));

          // Functions
          const pyFunctions = content.match(/def\s+(\w+)/g) || [];
          patterns.functionNames = pyFunctions.map(match => match.replace('def ', ''));

          // Imports
          const pyImports = content.match(/import\s+(\w+)|from\s+(\w+)/g) || [];
          patterns.imports = pyImports.map(match => 
            match.replace(/import\s+|from\s+/, '')
          );
          break;
      }
    } catch (error) {
      // Ignore regex errors
    }

    return patterns;
  }

  /**
   * Match code patterns against domain patterns
   * @private
   */
  matchDomainPatterns(codePatterns) {
    const matches = {};

    for (const [domain, config] of Object.entries(this.domainKeywords)) {
      const allCodeText = [
        ...codePatterns.classNames,
        ...codePatterns.functionNames,
        ...codePatterns.variableNames,
        ...codePatterns.imports
      ].join(' ').toLowerCase();

      const patternMatches = config.codePatterns.filter(pattern =>
        pattern.test(allCodeText)
      );

      if (patternMatches.length > 0) {
        matches[domain] = {
          matchingPatterns: patternMatches.length,
          totalPatterns: config.codePatterns.length,
          confidence: config.confidence
        };
      }
    }

    return matches;
  }

  /**
   * Aggregate keyword matches from multiple sources
   * @private
   */
  aggregateKeywordMatches(analyses) {
    const aggregated = {};

    for (const analysis of analyses) {
      if (analysis.domainKeywords) {
        for (const [domain, data] of Object.entries(analysis.domainKeywords)) {
          if (!aggregated[domain]) {
            aggregated[domain] = {
              totalScore: 0,
              sources: 0,
              confidence: data.confidence,
              allMatches: []
            };
          }
          
          aggregated[domain].totalScore += data.score;
          aggregated[domain].sources += 1;
          aggregated[domain].allMatches.push(...data.matches);
        }
      }
    }

    // Calculate average scores
    for (const domain of Object.keys(aggregated)) {
      aggregated[domain].averageScore = 
        aggregated[domain].totalScore / aggregated[domain].sources;
    }

    return aggregated;
  }

  /**
   * Infer domain from all analysis sources
   * @private
   */
  inferDomain(documentation, codePatterns, dependencies, architecture) {
    const domainScores = {};

    // Score from documentation analysis
    for (const [domain, data] of Object.entries(documentation.keywordMatches || {})) {
      domainScores[domain] = (domainScores[domain] || 0) + data.averageScore * data.confidence;
    }

    // Score from code patterns
    for (const [domain, data] of Object.entries(codePatterns.domainMatches || {})) {
      const patternScore = data.matchingPatterns / data.totalPatterns;
      domainScores[domain] = (domainScores[domain] || 0) + patternScore * data.confidence;
    }

    // Score from framework analysis
    if (dependencies.frameworks) {
      for (const [framework, config] of Object.entries(dependencies.frameworks)) {
        // Map frameworks to likely domains
        const frameworkDomains = this.getFrameworkDomains(framework);
        for (const domain of frameworkDomains) {
          domainScores[domain] = (domainScores[domain] || 0) + config.confidence * 0.5;
        }
      }
    }

    // Find highest scoring domain
    const sortedDomains = Object.entries(domainScores)
      .sort(([,a], [,b]) => b - a);

    if (sortedDomains.length === 0) {
      return { domain: 'general', confidence: 0.1, indicators: [] };
    }

    const [topDomain, score] = sortedDomains[0];
    const maxPossibleScore = 3; // documentation + code + framework
    const normalizedConfidence = Math.min(score / maxPossibleScore, 1.0);

    return {
      domain: topDomain,
      confidence: normalizedConfidence,
      indicators: this.getDomainIndicators(topDomain, documentation, codePatterns)
    };
  }

  /**
   * Infer project type from multiple sources
   * @private
   */
  inferProjectType(dependencies, architecture, documentation) {
    const typeScores = {};

    // Score from architecture patterns
    if (architecture) {
      for (const [pattern] of Object.entries(architecture)) {
        for (const [type, config] of Object.entries(this.projectTypes)) {
          if (config.architecturePatterns?.includes(pattern)) {
            typeScores[type] = (typeScores[type] || 0) + config.confidence;
          }
        }
      }
    }

    // Score from frameworks
    if (dependencies.frameworks) {
      for (const [framework] of Object.entries(dependencies.frameworks)) {
        for (const [type, config] of Object.entries(this.projectTypes)) {
          if (config.frameworks?.includes(framework)) {
            typeScores[type] = (typeScores[type] || 0) + config.confidence;
          }
        }
      }
    }

    // Score from documentation keywords
    const docText = [
      documentation.readme?.wordCount ? Object.keys(documentation.readme.domainKeywords || {}).join(' ') : '',
      documentation.packageInfo?.wordCount ? Object.keys(documentation.packageInfo.domainKeywords || {}).join(' ') : ''
    ].join(' ').toLowerCase();

    for (const [type, config] of Object.entries(this.projectTypes)) {
      const keywordMatches = config.indicators.filter(indicator =>
        docText.includes(indicator.toLowerCase())
      );
      
      if (keywordMatches.length > 0) {
        typeScores[type] = (typeScores[type] || 0) + 
          (keywordMatches.length / config.indicators.length) * config.confidence;
      }
    }

    // Find highest scoring type
    const sortedTypes = Object.entries(typeScores)
      .sort(([,a], [,b]) => b - a);

    if (sortedTypes.length === 0) {
      return { type: 'application', confidence: 0.1 };
    }

    const [topType, score] = sortedTypes[0];
    const maxPossibleScore = 3; // architecture + frameworks + documentation
    const normalizedConfidence = Math.min(score / maxPossibleScore, 1.0);

    return {
      type: topType,
      confidence: normalizedConfidence
    };
  }

  /**
   * Infer business context from documentation and project structure
   * @private
   */
  inferBusinessContext(documentation, projectPath) {
    const contextScores = {};

    // Check documentation for business context indicators
    for (const [context, config] of Object.entries(this.businessContexts)) {
      const docText = [
        documentation.readme?.wordCount ? JSON.stringify(documentation.readme) : '',
        documentation.packageInfo?.wordCount ? JSON.stringify(documentation.packageInfo) : ''
      ].join(' ').toLowerCase();

      const matches = config.indicators.filter(indicator =>
        docText.includes(indicator.toLowerCase())
      );

      if (matches.length > 0) {
        contextScores[context] = (matches.length / config.indicators.length) * config.confidence;
      }
    }

    // Find highest scoring context
    const sortedContexts = Object.entries(contextScores)
      .sort(([,a], [,b]) => b - a);

    return sortedContexts.length > 0 ? sortedContexts[0][0] : null;
  }

  /**
   * Assess project complexity based on dependencies and architecture
   * @private
   */
  assessComplexity(dependencies, architecture) {
    let complexityScore = 0;

    // Framework count contributes to complexity
    const frameworkCount = Object.keys(dependencies.frameworks || {}).length;
    complexityScore += Math.min(frameworkCount * 0.2, 1.0);

    // Architecture patterns contribute to complexity
    const architectureCount = Object.keys(architecture || {}).length;
    complexityScore += Math.min(architectureCount * 0.3, 1.0);

    // Database count contributes to complexity
    const databaseCount = (dependencies.databases || []).length;
    complexityScore += Math.min(databaseCount * 0.3, 0.6);

    // Microservices pattern increases complexity
    if (dependencies.summary?.architecturalPatterns?.includes('microservices')) {
      complexityScore += 0.5;
    }

    // Normalize and categorize
    if (complexityScore >= 1.5) return 'high';
    if (complexityScore >= 0.8) return 'medium';
    return 'simple';
  }

  /**
   * Assess project maturity based on file structure and tooling
   * @private
   */
  async assessMaturity(projectPath) {
    let maturityScore = 0;

    try {
      // Check for CI/CD
      const ciFiles = ['.github/workflows/', '.gitlab-ci.yml', '.travis.yml', 'Jenkinsfile'];
      for (const file of ciFiles) {
        if (await this.fileHandler.exists(path.join(projectPath, file))) {
          maturityScore += 0.2;
          break;
        }
      }

      // Check for testing setup
      const testDirs = ['test/', 'tests/', '__tests__/', 'spec/'];
      for (const dir of testDirs) {
        if (await this.fileHandler.exists(path.join(projectPath, dir))) {
          maturityScore += 0.2;
          break;
        }
      }

      // Check for documentation
      const docFiles = ['README.md', 'docs/', 'CONTRIBUTING.md'];
      for (const file of docFiles) {
        if (await this.fileHandler.exists(path.join(projectPath, file))) {
          maturityScore += 0.1;
        }
      }

      // Check for containerization
      if (await this.fileHandler.exists(path.join(projectPath, 'Dockerfile'))) {
        maturityScore += 0.2;
      }

      // Check for linting/formatting
      const lintFiles = ['.eslintrc', '.prettierrc', 'pyproject.toml', 'tox.ini'];
      for (const file of lintFiles) {
        if (await this.fileHandler.exists(path.join(projectPath, file))) {
          maturityScore += 0.1;
          break;
        }
      }

    } catch (error) {
      this.logger.warn('Maturity assessment partial failure', error);
    }

    // Categorize maturity
    if (maturityScore >= 0.7) return 'production';
    if (maturityScore >= 0.4) return 'development';
    return 'prototype';
  }

  /**
   * Get file content safely
   * @private
   */
  async getFileContent(filePath) {
    try {
      const stats = await this.fileHandler.getStats(filePath);
      if (stats.size > 100000) return null; // Skip large files

      return await this.fileHandler.readFile(filePath);
    } catch {
      return null;
    }
  }

  /**
   * Map frameworks to likely domains
   * @private
   */
  getFrameworkDomains(framework) {
    const frameworkDomainMap = {
      'django': ['api-service', 'e-commerce'],
      'flask': ['api-service'],
      'express': ['api-service'],
      'react': ['social-media', 'e-commerce', 'productivity'],
      'vue': ['productivity', 'e-commerce'],
      'tensorflow': ['analytics'],
      'pytorch': ['analytics'],
      'unity': ['gaming'],
      'phaser': ['gaming']
    };

    return frameworkDomainMap[framework] || ['general'];
  }

  /**
   * Get domain indicators for reporting
   * @private
   */
  getDomainIndicators(domain, documentation, codePatterns) {
    const indicators = [];

    // Add documentation indicators
    if (documentation.keywordMatches?.[domain]) {
      indicators.push(...documentation.keywordMatches[domain].allMatches);
    }

    // Add code pattern indicators
    if (codePatterns.domainMatches?.[domain]) {
      indicators.push(`${codePatterns.domainMatches[domain].matchingPatterns} matching code patterns`);
    }

    return [...new Set(indicators)]; // Remove duplicates
  }
}

module.exports = DomainAnalyzer;