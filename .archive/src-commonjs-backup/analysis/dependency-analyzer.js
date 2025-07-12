/**
 * Dependency Analysis System
 * Multi-language dependency extraction and analysis
 */

const path = require('path');
const FileHandler = require('../utils/file-handler');
const Logger = require('../utils/logger');

class DependencyAnalyzer {
  constructor() {
    this.fileHandler = new FileHandler();
    this.logger = new Logger('DependencyAnalyzer');

    // Language-specific dependency analyzers
    this.analyzers = {
      javascript: this.analyzeJavaScript.bind(this),
      typescript: this.analyzeJavaScript.bind(this), // Same as JS
      python: this.analyzePython.bind(this),
      java: this.analyzeJava.bind(this),
      go: this.analyzeGo.bind(this),
      rust: this.analyzeRust.bind(this),
      php: this.analyzePHP.bind(this),
      ruby: this.analyzeRuby.bind(this),
      swift: this.analyzeSwift.bind(this),
      kotlin: this.analyzeKotlin.bind(this),
      csharp: this.analyzeCSharp.bind(this),
      dart: this.analyzeDart.bind(this)
    };

    // Framework/library categorization
    this.frameworkCategories = {
      // Frontend Frameworks
      frontend: {
        react: {
          keywords: ['react', 'react-dom', 'react-router', 'react-scripts'],
          type: 'ui-framework',
          ecosystem: 'react'
        },
        vue: {
          keywords: ['vue', '@vue/cli', 'vue-router', 'vuex', 'pinia'],
          type: 'ui-framework',
          ecosystem: 'vue'
        },
        angular: {
          keywords: ['@angular/core', '@angular/cli', '@angular/router', 'rxjs'],
          type: 'ui-framework',
          ecosystem: 'angular'
        },
        svelte: {
          keywords: ['svelte', 'svelte-kit', '@sveltejs/kit'],
          type: 'ui-framework',
          ecosystem: 'svelte'
        }
      },

      // Backend Frameworks
      backend: {
        express: {
          keywords: ['express', 'express-session', 'body-parser'],
          type: 'web-framework',
          ecosystem: 'node'
        },
        fastify: {
          keywords: ['fastify', '@fastify/'],
          type: 'web-framework',
          ecosystem: 'node'
        },
        nestjs: {
          keywords: ['@nestjs/core', '@nestjs/common', '@nestjs/platform-express'],
          type: 'web-framework',
          ecosystem: 'node'
        },
        django: {
          keywords: ['django', 'djangorestframework'],
          type: 'web-framework',
          ecosystem: 'python'
        },
        flask: {
          keywords: ['flask', 'flask-sqlalchemy', 'flask-migrate'],
          type: 'web-framework',
          ecosystem: 'python'
        },
        fastapi: {
          keywords: ['fastapi', 'uvicorn', 'pydantic'],
          type: 'web-framework',
          ecosystem: 'python'
        },
        spring: {
          keywords: ['spring-boot-starter', 'spring-core', 'spring-web'],
          type: 'web-framework',
          ecosystem: 'java'
        }
      },

      // Full-stack Frameworks
      fullstack: {
        nextjs: {
          keywords: ['next', 'next/router', 'next/image'],
          type: 'fullstack-framework',
          ecosystem: 'react'
        },
        nuxtjs: {
          keywords: ['nuxt', '@nuxt/'],
          type: 'fullstack-framework',
          ecosystem: 'vue'
        },
        sveltekit: {
          keywords: ['@sveltejs/kit', '@sveltejs/adapter'],
          type: 'fullstack-framework',
          ecosystem: 'svelte'
        }
      },

      // Database & ORM
      database: {
        mongodb: {
          keywords: ['mongodb', 'mongoose', 'mongo'],
          type: 'database',
          ecosystem: 'nosql'
        },
        postgresql: {
          keywords: ['pg', 'postgres', 'postgresql', 'psycopg2'],
          type: 'database',
          ecosystem: 'sql'
        },
        mysql: {
          keywords: ['mysql', 'mysql2', 'pymysql'],
          type: 'database',
          ecosystem: 'sql'
        },
        redis: {
          keywords: ['redis', 'ioredis', 'redis-py'],
          type: 'cache',
          ecosystem: 'inmemory'
        },
        prisma: {
          keywords: ['prisma', '@prisma/client'],
          type: 'orm',
          ecosystem: 'typescript'
        },
        sequelize: {
          keywords: ['sequelize'],
          type: 'orm',
          ecosystem: 'node'
        }
      },

      // Testing Frameworks
      testing: {
        jest: {
          keywords: ['jest', '@types/jest', 'jest-environment'],
          type: 'test-runner',
          ecosystem: 'javascript'
        },
        vitest: {
          keywords: ['vitest'],
          type: 'test-runner',
          ecosystem: 'vite'
        },
        cypress: {
          keywords: ['cypress'],
          type: 'e2e-testing',
          ecosystem: 'javascript'
        },
        playwright: {
          keywords: ['@playwright/test', 'playwright'],
          type: 'e2e-testing',
          ecosystem: 'javascript'
        },
        pytest: {
          keywords: ['pytest', 'pytest-cov'],
          type: 'test-runner',
          ecosystem: 'python'
        }
      },

      // Machine Learning
      ml: {
        tensorflow: {
          keywords: ['tensorflow', 'tensorflow-gpu', '@tensorflow/tfjs'],
          type: 'ml-framework',
          ecosystem: 'ml'
        },
        pytorch: {
          keywords: ['torch', 'torchvision', 'torchaudio'],
          type: 'ml-framework',
          ecosystem: 'ml'
        },
        scikit_learn: {
          keywords: ['scikit-learn', 'sklearn'],
          type: 'ml-library',
          ecosystem: 'python'
        },
        pandas: {
          keywords: ['pandas'],
          type: 'data-analysis',
          ecosystem: 'python'
        },
        numpy: {
          keywords: ['numpy'],
          type: 'data-computation',
          ecosystem: 'python'
        }
      },

      // DevOps & Tools
      devops: {
        docker: {
          keywords: ['docker', 'dockerfile'],
          type: 'containerization',
          ecosystem: 'devops'
        },
        kubernetes: {
          keywords: ['kubernetes', 'kubectl'],
          type: 'orchestration',
          ecosystem: 'devops'
        },
        terraform: {
          keywords: ['terraform'],
          type: 'infrastructure',
          ecosystem: 'devops'
        }
      }
    };
  }

  /**
   * Analyze dependencies for multiple languages in a project
   * @param {string} projectPath - Path to project directory
   * @param {Array<string>} detectedLanguages - List of detected languages
   * @returns {Promise<Object>} Comprehensive dependency analysis
   */
  async analyzeDependencies(projectPath, detectedLanguages = []) {
    this.logger.debug('Analyzing dependencies', { 
      projectPath, 
      languages: detectedLanguages 
    });

    try {
      const analysis = {
        languages: {},
        frameworks: {},
        libraries: {},
        tools: {},
        databases: [],
        summary: {}
      };

      // Analyze each detected language
      for (const language of detectedLanguages) {
        if (this.analyzers[language]) {
          this.logger.debug(`Analyzing ${language} dependencies`);
          const languageAnalysis = await this.analyzers[language](projectPath);
          analysis.languages[language] = languageAnalysis;
        }
      }

      // Aggregate and categorize all dependencies
      const allDependencies = this.aggregateDependencies(analysis.languages);
      const categorized = this.categorizeDependencies(allDependencies);

      analysis.frameworks = categorized.frameworks;
      analysis.libraries = categorized.libraries;
      analysis.tools = categorized.tools;
      analysis.databases = categorized.databases;
      analysis.summary = this.generateSummary(categorized);

      this.logger.debug('Dependency analysis completed', {
        languagesAnalyzed: Object.keys(analysis.languages).length,
        totalDependencies: allDependencies.length,
        frameworksFound: Object.keys(analysis.frameworks).length
      });

      return analysis;

    } catch (error) {
      this.logger.error('Dependency analysis failed', error);
      throw new Error(`Dependency analysis failed: ${error.message}`);
    }
  }

  /**
   * Analyze JavaScript/TypeScript dependencies
   * @private
   */
  async analyzeJavaScript(projectPath) {
    const analysis = {
      packageManagers: [],
      dependencies: {},
      devDependencies: {},
      scripts: {},
      engines: {}
    };

    try {
      // package.json analysis
      const packageJsonPath = path.join(projectPath, 'package.json');
      if (await this.fileHandler.exists(packageJsonPath)) {
        const packageJson = await this.fileHandler.readJSON(packageJsonPath);
        
        analysis.dependencies = packageJson.dependencies || {};
        analysis.devDependencies = packageJson.devDependencies || {};
        analysis.scripts = packageJson.scripts || {};
        analysis.engines = packageJson.engines || {};
        analysis.packageManagers.push('npm');
      }

      // yarn.lock detection
      if (await this.fileHandler.exists(path.join(projectPath, 'yarn.lock'))) {
        analysis.packageManagers.push('yarn');
      }

      // pnpm-lock.yaml detection
      if (await this.fileHandler.exists(path.join(projectPath, 'pnpm-lock.yaml'))) {
        analysis.packageManagers.push('pnpm');
      }

    } catch (error) {
      this.logger.warn('JavaScript dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Python dependencies
   * @private
   */
  async analyzePython(projectPath) {
    const analysis = {
      packageManagers: [],
      dependencies: {},
      devDependencies: {},
      pythonVersion: null
    };

    try {
      // requirements.txt
      const requirementsPath = path.join(projectPath, 'requirements.txt');
      if (await this.fileHandler.exists(requirementsPath)) {
        const requirements = await this.fileHandler.readFile(requirementsPath);
        analysis.dependencies = this.parseRequirementsTxt(requirements);
        analysis.packageManagers.push('pip');
      }

      // pyproject.toml
      const pyprojectPath = path.join(projectPath, 'pyproject.toml');
      if (await this.fileHandler.exists(pyprojectPath)) {
        const pyproject = await this.fileHandler.readFile(pyprojectPath);
        const parsed = this.parsePyprojectToml(pyproject);
        Object.assign(analysis.dependencies, parsed.dependencies);
        Object.assign(analysis.devDependencies, parsed.devDependencies);
        analysis.packageManagers.push('poetry');
      }

      // Pipfile
      const pipfilePath = path.join(projectPath, 'Pipfile');
      if (await this.fileHandler.exists(pipfilePath)) {
        analysis.packageManagers.push('pipenv');
      }

    } catch (error) {
      this.logger.warn('Python dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Java dependencies
   * @private
   */
  async analyzeJava(projectPath) {
    const analysis = {
      buildTools: [],
      dependencies: {},
      javaVersion: null
    };

    try {
      // Maven (pom.xml)
      const pomPath = path.join(projectPath, 'pom.xml');
      if (await this.fileHandler.exists(pomPath)) {
        analysis.buildTools.push('maven');
        // Note: Full XML parsing would require additional dependency
        // For now, we'll do basic text analysis
      }

      // Gradle
      const gradlePaths = ['build.gradle', 'build.gradle.kts'];
      for (const gradleFile of gradlePaths) {
        if (await this.fileHandler.exists(path.join(projectPath, gradleFile))) {
          analysis.buildTools.push('gradle');
          break;
        }
      }

    } catch (error) {
      this.logger.warn('Java dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Go dependencies
   * @private
   */
  async analyzeGo(projectPath) {
    const analysis = {
      packageManager: 'go mod',
      dependencies: {},
      goVersion: null
    };

    try {
      const goModPath = path.join(projectPath, 'go.mod');
      if (await this.fileHandler.exists(goModPath)) {
        const goMod = await this.fileHandler.readFile(goModPath);
        analysis.dependencies = this.parseGoMod(goMod);
      }

    } catch (error) {
      this.logger.warn('Go dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Rust dependencies
   * @private
   */
  async analyzeRust(projectPath) {
    const analysis = {
      packageManager: 'cargo',
      dependencies: {},
      devDependencies: {},
      rustVersion: null
    };

    try {
      const cargoPath = path.join(projectPath, 'Cargo.toml');
      if (await this.fileHandler.exists(cargoPath)) {
        const cargo = await this.fileHandler.readFile(cargoPath);
        const parsed = this.parseCargoToml(cargo);
        analysis.dependencies = parsed.dependencies;
        analysis.devDependencies = parsed.devDependencies;
      }

    } catch (error) {
      this.logger.warn('Rust dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze PHP dependencies
   * @private
   */
  async analyzePHP(projectPath) {
    const analysis = {
      packageManager: 'composer',
      dependencies: {},
      devDependencies: {}
    };

    try {
      const composerPath = path.join(projectPath, 'composer.json');
      if (await this.fileHandler.exists(composerPath)) {
        const composer = await this.fileHandler.readJSON(composerPath);
        analysis.dependencies = composer.require || {};
        analysis.devDependencies = composer['require-dev'] || {};
      }

    } catch (error) {
      this.logger.warn('PHP dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Ruby dependencies
   * @private
   */
  async analyzeRuby(projectPath) {
    const analysis = {
      packageManager: 'bundler',
      dependencies: {},
      rubyVersion: null
    };

    try {
      const gemfilePath = path.join(projectPath, 'Gemfile');
      if (await this.fileHandler.exists(gemfilePath)) {
        const gemfile = await this.fileHandler.readFile(gemfilePath);
        analysis.dependencies = this.parseGemfile(gemfile);
      }

    } catch (error) {
      this.logger.warn('Ruby dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Swift dependencies
   * @private
   */
  async analyzeSwift(projectPath) {
    const analysis = {
      packageManager: 'swift package manager',
      dependencies: {}
    };

    try {
      const packagePath = path.join(projectPath, 'Package.swift');
      if (await this.fileHandler.exists(packagePath)) {
        // Swift Package.swift parsing would require more complex logic
        analysis.dependencies = {};
      }

    } catch (error) {
      this.logger.warn('Swift dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Analyze Kotlin dependencies (similar to Java)
   * @private
   */
  async analyzeKotlin(projectPath) {
    return this.analyzeJava(projectPath); // Kotlin uses same build tools
  }

  /**
   * Analyze C# dependencies
   * @private
   */
  async analyzeCSharp(projectPath) {
    const analysis = {
      packageManager: 'nuget',
      dependencies: {}
    };

    // C# project analysis would require .csproj parsing
    return analysis;
  }

  /**
   * Analyze Dart dependencies
   * @private
   */
  async analyzeDart(projectPath) {
    const analysis = {
      packageManager: 'pub',
      dependencies: {},
      devDependencies: {}
    };

    try {
      const pubspecPath = path.join(projectPath, 'pubspec.yaml');
      if (await this.fileHandler.exists(pubspecPath)) {
        const pubspec = await this.fileHandler.readFile(pubspecPath);
        const parsed = this.parsePubspecYaml(pubspec);
        analysis.dependencies = parsed.dependencies;
        analysis.devDependencies = parsed.devDependencies;
      }

    } catch (error) {
      this.logger.warn('Dart dependency analysis partial failure', error);
    }

    return analysis;
  }

  /**
   * Parse requirements.txt file
   * @private
   */
  parseRequirementsTxt(content) {
    const dependencies = {};
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const match = trimmed.match(/^([a-zA-Z0-9_-]+)([>=<].+)?$/);
        if (match) {
          dependencies[match[1]] = match[2] || '*';
        }
      }
    }

    return dependencies;
  }

  /**
   * Parse pyproject.toml file (basic implementation)
   * @private
   */
  parsePyprojectToml(content) {
    // This is a simplified parser - a full implementation would use a TOML library
    const dependencies = {};
    const devDependencies = {};
    
    // Basic regex parsing for dependencies
    const depMatch = content.match(/\[tool\.poetry\.dependencies\]([\s\S]*?)(?=\[|$)/);
    if (depMatch) {
      const depSection = depMatch[1];
      const depLines = depSection.split('\n');
      for (const line of depLines) {
        const match = line.match(/^([a-zA-Z0-9_-]+)\s*=\s*"(.+)"/);
        if (match) {
          dependencies[match[1]] = match[2];
        }
      }
    }

    return { dependencies, devDependencies };
  }

  /**
   * Parse go.mod file
   * @private
   */
  parseGoMod(content) {
    const dependencies = {};
    const lines = content.split('\n');
    let inRequireBlock = false;

    for (const line of lines) {
      const trimmed = line.trim();
      
      if (trimmed === 'require (') {
        inRequireBlock = true;
        continue;
      }
      
      if (trimmed === ')') {
        inRequireBlock = false;
        continue;
      }

      if (inRequireBlock || trimmed.startsWith('require ')) {
        const match = trimmed.match(/([^\s]+)\s+v?([^\s]+)/);
        if (match) {
          dependencies[match[1]] = match[2];
        }
      }
    }

    return dependencies;
  }

  /**
   * Parse Cargo.toml file (basic implementation)
   * @private
   */
  parseCargoToml(content) {
    const dependencies = {};
    const devDependencies = {};
    
    // Basic parsing - would use proper TOML parser in production
    const sections = content.split(/\[([^\]]+)\]/);
    
    for (let i = 1; i < sections.length; i += 2) {
      const sectionName = sections[i];
      const sectionContent = sections[i + 1] || '';
      
      if (sectionName === 'dependencies') {
        Object.assign(dependencies, this.parseTomlDependencies(sectionContent));
      } else if (sectionName === 'dev-dependencies') {
        Object.assign(devDependencies, this.parseTomlDependencies(sectionContent));
      }
    }

    return { dependencies, devDependencies };
  }

  /**
   * Parse TOML dependencies section
   * @private
   */
  parseTomlDependencies(content) {
    const dependencies = {};
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const match = trimmed.match(/^([a-zA-Z0-9_-]+)\s*=\s*"(.+)"/);
        if (match) {
          dependencies[match[1]] = match[2];
        }
      }
    }

    return dependencies;
  }

  /**
   * Parse Gemfile (basic implementation)
   * @private
   */
  parseGemfile(content) {
    const dependencies = {};
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();
      const match = trimmed.match(/^gem\s+['"]([^'"]+)['"](?:,\s*['"]([^'"]+)['"])?/);
      if (match) {
        dependencies[match[1]] = match[2] || '*';
      }
    }

    return dependencies;
  }

  /**
   * Parse pubspec.yaml (basic implementation)
   * @private
   */
  parsePubspecYaml(content) {
    const dependencies = {};
    const devDependencies = {};
    
    // Very basic YAML parsing - would use proper YAML parser in production
    const lines = content.split('\n');
    let currentSection = null;

    for (const line of lines) {
      const trimmed = line.trim();
      
      if (trimmed === 'dependencies:') {
        currentSection = 'dependencies';
        continue;
      } else if (trimmed === 'dev_dependencies:') {
        currentSection = 'dev_dependencies';
        continue;
      } else if (trimmed.endsWith(':') && !trimmed.startsWith(' ')) {
        currentSection = null;
        continue;
      }

      if (currentSection && trimmed.startsWith(' ')) {
        const match = trimmed.match(/^([a-zA-Z0-9_]+):\s*(.+)?$/);
        if (match) {
          const target = currentSection === 'dependencies' ? dependencies : devDependencies;
          target[match[1]] = match[2] || '*';
        }
      }
    }

    return { dependencies, devDependencies };
  }

  /**
   * Aggregate dependencies from all languages
   * @private
   */
  aggregateDependencies(languageAnalysis) {
    const allDependencies = [];

    for (const [language, analysis] of Object.entries(languageAnalysis)) {
      // Add dependencies with language context
      const deps = analysis.dependencies || {};
      const devDeps = analysis.devDependencies || {};

      for (const [name, version] of Object.entries(deps)) {
        allDependencies.push({
          name: name.toLowerCase(),
          version,
          language,
          type: 'production'
        });
      }

      for (const [name, version] of Object.entries(devDeps)) {
        allDependencies.push({
          name: name.toLowerCase(),
          version,
          language,
          type: 'development'
        });
      }
    }

    return allDependencies;
  }

  /**
   * Categorize dependencies into frameworks, libraries, tools, etc.
   * @private
   */
  categorizeDependencies(dependencies) {
    const categorized = {
      frameworks: {},
      libraries: {},
      tools: {},
      databases: []
    };

    const dependencyNames = dependencies.map(d => d.name);

    // Categorize based on framework patterns
    for (const [categoryType, categories] of Object.entries(this.frameworkCategories)) {
      for (const [frameworkName, config] of Object.entries(categories)) {
        const matchingDeps = config.keywords.filter(keyword => 
          dependencyNames.some(dep => dep.includes(keyword.toLowerCase()))
        );

        if (matchingDeps.length > 0) {
          const confidence = matchingDeps.length / config.keywords.length;
          
          if (config.type.includes('framework')) {
            categorized.frameworks[frameworkName] = {
              type: config.type,
              ecosystem: config.ecosystem,
              confidence,
              matchingDependencies: matchingDeps,
              category: categoryType
            };
          } else if (config.type === 'database' || config.type === 'cache') {
            categorized.databases.push({
              name: frameworkName,
              type: config.type,
              ecosystem: config.ecosystem,
              confidence
            });
          } else {
            categorized.libraries[frameworkName] = {
              type: config.type,
              ecosystem: config.ecosystem,
              confidence,
              matchingDependencies: matchingDeps
            };
          }
        }
      }
    }

    return categorized;
  }

  /**
   * Generate dependency analysis summary
   * @private
   */
  generateSummary(categorized) {
    return {
      frameworkCount: Object.keys(categorized.frameworks).length,
      libraryCount: Object.keys(categorized.libraries).length,
      databaseCount: categorized.databases.length,
      primaryFrameworks: Object.entries(categorized.frameworks)
        .filter(([_, config]) => config.confidence > 0.5)
        .map(([name]) => name),
      primaryEcosystems: [...new Set(
        Object.values(categorized.frameworks)
          .concat(Object.values(categorized.libraries))
          .map(config => config.ecosystem)
      )],
      architecturalPatterns: this.inferArchitecturalPatterns(categorized)
    };
  }

  /**
   * Infer architectural patterns from dependencies
   * @private
   */
  inferArchitecturalPatterns(categorized) {
    const patterns = [];

    // Check for frontend frameworks
    const frontendFrameworks = Object.entries(categorized.frameworks)
      .filter(([_, config]) => config.category === 'frontend');
    if (frontendFrameworks.length > 0) {
      patterns.push('spa'); // Single Page Application
    }

    // Check for fullstack frameworks
    const fullstackFrameworks = Object.entries(categorized.frameworks)
      .filter(([_, config]) => config.category === 'fullstack');
    if (fullstackFrameworks.length > 0) {
      patterns.push('fullstack');
    }

    // Check for API frameworks
    const backendFrameworks = Object.entries(categorized.frameworks)
      .filter(([_, config]) => config.category === 'backend');
    if (backendFrameworks.length > 0) {
      patterns.push('api');
    }

    // Check for microservices indicators
    if (categorized.databases.length > 1 && backendFrameworks.length > 0) {
      patterns.push('microservices');
    }

    // Check for ML/AI patterns
    const mlFrameworks = Object.entries(categorized.frameworks)
      .filter(([_, config]) => config.ecosystem === 'ml');
    if (mlFrameworks.length > 0) {
      patterns.push('machine-learning');
    }

    return patterns;
  }
}

module.exports = DependencyAnalyzer;