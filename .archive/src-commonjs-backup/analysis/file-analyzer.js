/**
 * File Pattern Recognition Engine
 * Detects languages, frameworks, and architecture patterns from file system analysis
 */

const path = require('path');
const FileHandler = require('../utils/file-handler');
const Logger = require('../utils/logger');

class FileAnalyzer {
  constructor() {
    this.fileHandler = new FileHandler();
    this.logger = new Logger('FileAnalyzer');
    
    // Language detection patterns
    this.languagePatterns = {
      javascript: {
        extensions: ['.js', '.jsx', '.mjs', '.cjs'],
        indicators: ['package.json', 'node_modules', 'yarn.lock', 'npm-shrinkwrap.json'],
        confidence: 0.9
      },
      typescript: {
        extensions: ['.ts', '.tsx', '.d.ts'],
        indicators: ['tsconfig.json', 'tsc', 'typescript'],
        confidence: 0.95
      },
      python: {
        extensions: ['.py', '.pyw', '.pyx', '.pyz'],
        indicators: ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile', '__pycache__'],
        confidence: 0.9
      },
      java: {
        extensions: ['.java', '.jar', '.war', '.class'],
        indicators: ['pom.xml', 'build.gradle', 'gradle.properties', 'src/main/java'],
        confidence: 0.9
      },
      go: {
        extensions: ['.go'],
        indicators: ['go.mod', 'go.sum', 'Gopkg.toml', 'vendor'],
        confidence: 0.95
      },
      rust: {
        extensions: ['.rs'],
        indicators: ['Cargo.toml', 'Cargo.lock', 'src/main.rs', 'src/lib.rs'],
        confidence: 0.95
      },
      php: {
        extensions: ['.php', '.phtml', '.php3', '.php4', '.php5', '.phps'],
        indicators: ['composer.json', 'composer.lock', 'vendor/autoload.php'],
        confidence: 0.9
      },
      ruby: {
        extensions: ['.rb', '.rbw', '.rake', '.gemspec'],
        indicators: ['Gemfile', 'Gemfile.lock', 'Rakefile', '.ruby-version'],
        confidence: 0.9
      },
      swift: {
        extensions: ['.swift'],
        indicators: ['Package.swift', 'Sources/', '.swiftpm', 'xcodeproj'],
        confidence: 0.95
      },
      kotlin: {
        extensions: ['.kt', '.kts'],
        indicators: ['build.gradle.kts', 'kotlin', 'src/main/kotlin'],
        confidence: 0.9
      },
      scala: {
        extensions: ['.scala', '.sc'],
        indicators: ['build.sbt', 'project/build.properties', 'src/main/scala'],
        confidence: 0.9
      },
      clojure: {
        extensions: ['.clj', '.cljs', '.cljc', '.edn'],
        indicators: ['project.clj', 'deps.edn', 'shadow-cljs.edn'],
        confidence: 0.9
      },
      dart: {
        extensions: ['.dart'],
        indicators: ['pubspec.yaml', 'pubspec.lock', 'lib/', 'flutter'],
        confidence: 0.95
      },
      csharp: {
        extensions: ['.cs', '.csx'],
        indicators: ['.csproj', '.sln', 'packages.config', 'Program.cs'],
        confidence: 0.9
      }
    };

    // Framework detection patterns
    this.frameworkPatterns = {
      // JavaScript/TypeScript Frameworks
      react: {
        dependencies: ['react', 'react-dom', '@types/react'],
        files: ['src/App.jsx', 'src/App.tsx', 'public/index.html'],
        patterns: [/import.*from\s+['"]react['"]/],
        confidence: 0.95
      },
      nextjs: {
        dependencies: ['next'],
        files: ['next.config.js', 'pages/', 'app/', '_app.js'],
        patterns: [/from\s+['"]next\//],
        confidence: 0.95
      },
      vue: {
        dependencies: ['vue', '@vue/cli'],
        files: ['src/App.vue', 'vue.config.js'],
        patterns: [/<template>/, /<script>/, /<style>/],
        confidence: 0.95
      },
      angular: {
        dependencies: ['@angular/core', '@angular/cli'],
        files: ['angular.json', 'src/app/app.component.ts'],
        patterns: [/@Component\(/, /@NgModule\(/],
        confidence: 0.95
      },
      svelte: {
        dependencies: ['svelte'],
        files: ['src/App.svelte', 'svelte.config.js'],
        patterns: [/<script>/, /<\/script>/, /\.svelte$/],
        confidence: 0.95
      },
      express: {
        dependencies: ['express'],
        files: ['server.js', 'app.js', 'index.js'],
        patterns: [/require\(['"]express['"]\)/, /app\.listen\(/],
        confidence: 0.9
      },
      fastify: {
        dependencies: ['fastify'],
        patterns: [/require\(['"]fastify['"]\)/, /fastify\(/],
        confidence: 0.9
      },
      nestjs: {
        dependencies: ['@nestjs/core', '@nestjs/common'],
        files: ['nest-cli.json', 'src/main.ts'],
        patterns: [/@Controller\(/, /@Injectable\(/],
        confidence: 0.95
      },

      // Python Frameworks
      django: {
        dependencies: ['Django'],
        files: ['manage.py', 'settings.py', 'urls.py'],
        patterns: [/from django/, /DJANGO_SETTINGS_MODULE/],
        confidence: 0.95
      },
      flask: {
        dependencies: ['Flask'],
        files: ['app.py', 'wsgi.py'],
        patterns: [/from flask/, /Flask\(__name__\)/],
        confidence: 0.9
      },
      fastapi: {
        dependencies: ['fastapi'],
        patterns: [/from fastapi/, /FastAPI\(/],
        confidence: 0.9
      },
      tensorflow: {
        dependencies: ['tensorflow', 'tensorflow-gpu'],
        patterns: [/import tensorflow/, /tf\./],
        confidence: 0.9
      },
      pytorch: {
        dependencies: ['torch', 'torchvision'],
        patterns: [/import torch/, /torch\./],
        confidence: 0.9
      },

      // Java Frameworks
      spring: {
        dependencies: ['spring-boot-starter', 'spring-core'],
        files: ['src/main/java', 'application.properties'],
        patterns: [/@SpringBootApplication/, /@Controller/],
        confidence: 0.9
      },

      // Testing Frameworks
      jest: {
        dependencies: ['jest', '@types/jest'],
        files: ['jest.config.js', '__tests__/'],
        patterns: [/describe\(/, /test\(/, /it\(/],
        confidence: 0.9
      },
      cypress: {
        dependencies: ['cypress'],
        files: ['cypress.json', 'cypress/'],
        patterns: [/cy\./, /Cypress\./],
        confidence: 0.95
      },
      playwright: {
        dependencies: ['@playwright/test'],
        patterns: [/test\(.*page/, /playwright/],
        confidence: 0.9
      }
    };

    // Architecture patterns
    this.architecturePatterns = {
      frontend: {
        directories: ['src/components', 'src/pages', 'components/', 'pages/', 'public/', 'assets/', 'styles/'],
        files: ['index.html', 'App.js', 'App.tsx', 'App.vue'],
        confidence: 0.8
      },
      backend: {
        directories: ['src/routes', 'src/controllers', 'src/api', 'api/', 'routes/', 'controllers/', 'middleware/'],
        files: ['server.js', 'app.js', 'main.py', 'wsgi.py'],
        confidence: 0.8
      },
      fullstack: {
        directories: ['client/', 'server/', 'frontend/', 'backend/'],
        indicators: ['monorepo', 'workspace'],
        confidence: 0.7
      },
      mobile: {
        directories: ['ios/', 'android/', 'src/screens'],
        files: ['pubspec.yaml', 'Podfile', 'build.gradle'],
        confidence: 0.9
      },
      desktop: {
        directories: ['src-tauri/', 'main/', 'renderer/'],
        files: ['tauri.conf.json', 'electron.js', 'main.rs'],
        confidence: 0.8
      },
      cli: {
        directories: ['src/commands', 'bin/'],
        files: ['cli.js', 'main.rs', 'setup.py'],
        patterns: [/#!/, /argparse/, /commander/, /clap/],
        confidence: 0.7
      },
      library: {
        files: ['lib/', 'dist/', 'build/', 'index.js', 'index.ts'],
        patterns: [/module\.exports/, /export default/, /export \{/],
        confidence: 0.6
      },
      database: {
        directories: ['migrations/', 'models/', 'schemas/', 'db/'],
        files: ['schema.sql', 'models.py', 'database.js'],
        confidence: 0.8
      },
      devops: {
        files: ['Dockerfile', 'docker-compose.yml', 'kubernetes/', '.github/workflows/', 'terraform/'],
        confidence: 0.9
      }
    };
  }

  /**
   * Analyze project directory to detect languages
   * @param {string} projectPath - Path to project directory
   * @returns {Promise<Object>} Detected languages with confidence scores
   */
  async detectLanguages(projectPath) {
    this.logger.debug('Detecting languages', { projectPath });

    try {
      const files = await this.fileHandler.listFiles(projectPath, {
        recursive: true,
        includeStats: false,
        excludePatterns: ['node_modules', '.git', 'dist', 'build', '__pycache__', 'target']
      });

      const languageScores = {};

      // Analyze file extensions
      for (const file of files) {
        const ext = path.extname(file.name).toLowerCase();
        
        for (const [language, config] of Object.entries(this.languagePatterns)) {
          if (config.extensions.includes(ext)) {
            languageScores[language] = (languageScores[language] || 0) + config.confidence;
          }
        }
      }

      // Analyze indicator files
      const fileNames = files.map(f => f.name.toLowerCase());
      const filePaths = files.map(f => f.path.toLowerCase());

      for (const [language, config] of Object.entries(this.languagePatterns)) {
        for (const indicator of config.indicators) {
          if (fileNames.some(name => name.includes(indicator.toLowerCase())) ||
              filePaths.some(path => path.includes(indicator.toLowerCase()))) {
            languageScores[language] = (languageScores[language] || 0) + config.confidence * 2;
          }
        }
      }

      // Normalize scores and filter
      const detectedLanguages = {};
      const maxScore = Math.max(...Object.values(languageScores));

      for (const [language, score] of Object.entries(languageScores)) {
        const normalizedScore = maxScore > 0 ? score / maxScore : 0;
        if (normalizedScore > 0.3) { // Threshold for inclusion
          detectedLanguages[language] = {
            confidence: Math.min(normalizedScore, 1.0),
            indicators: this.getLanguageIndicators(language, files)
          };
        }
      }

      this.logger.debug('Languages detected', { 
        count: Object.keys(detectedLanguages).length,
        languages: Object.keys(detectedLanguages)
      });

      return detectedLanguages;

    } catch (error) {
      this.logger.error('Language detection failed', error);
      throw new Error(`Language detection failed: ${error.message}`);
    }
  }

  /**
   * Detect frameworks and libraries from dependencies and code patterns
   * @param {string} projectPath - Path to project directory
   * @returns {Promise<Object>} Detected frameworks with confidence scores
   */
  async detectFrameworks(projectPath) {
    this.logger.debug('Detecting frameworks', { projectPath });

    try {
      const dependencies = await this.extractDependencies(projectPath);
      const files = await this.fileHandler.listFiles(projectPath, {
        recursive: true,
        excludePatterns: ['node_modules', '.git', 'dist', 'build']
      });

      const frameworkScores = {};

      // Analyze dependencies
      for (const [framework, config] of Object.entries(this.frameworkPatterns)) {
        let dependencyScore = 0;
        
        for (const dep of config.dependencies || []) {
          if (dependencies.includes(dep.toLowerCase())) {
            dependencyScore += config.confidence;
          }
        }

        if (dependencyScore > 0) {
          frameworkScores[framework] = dependencyScore;
        }
      }

      // Analyze file patterns
      for (const file of files) {
        const content = await this.getFileContent(file.path);
        if (!content) continue;

        for (const [framework, config] of Object.entries(this.frameworkPatterns)) {
          // Check file existence
          if (config.files) {
            for (const expectedFile of config.files) {
              if (file.path.includes(expectedFile)) {
                frameworkScores[framework] = (frameworkScores[framework] || 0) + config.confidence;
              }
            }
          }

          // Check content patterns
          if (config.patterns) {
            for (const pattern of config.patterns) {
              if (pattern.test(content)) {
                frameworkScores[framework] = (frameworkScores[framework] || 0) + config.confidence * 0.5;
              }
            }
          }
        }
      }

      // Normalize and filter results
      const detectedFrameworks = {};
      const maxScore = Math.max(...Object.values(frameworkScores));

      for (const [framework, score] of Object.entries(frameworkScores)) {
        const normalizedScore = maxScore > 0 ? score / maxScore : 0;
        if (normalizedScore > 0.2) { // Lower threshold for frameworks
          detectedFrameworks[framework] = {
            confidence: Math.min(normalizedScore, 1.0),
            category: this.getFrameworkCategory(framework),
            indicators: this.getFrameworkIndicators(framework, dependencies, files)
          };
        }
      }

      this.logger.debug('Frameworks detected', {
        count: Object.keys(detectedFrameworks).length,
        frameworks: Object.keys(detectedFrameworks)
      });

      return detectedFrameworks;

    } catch (error) {
      this.logger.error('Framework detection failed', error);
      throw new Error(`Framework detection failed: ${error.message}`);
    }
  }

  /**
   * Detect project architecture patterns
   * @param {string} projectPath - Path to project directory
   * @returns {Promise<Object>} Detected architecture patterns
   */
  async detectArchitecture(projectPath) {
    this.logger.debug('Detecting architecture', { projectPath });

    try {
      const files = await this.fileHandler.listFiles(projectPath, {
        recursive: true,
        excludePatterns: ['node_modules', '.git', 'dist', 'build']
      });

      const architectureScores = {};

      // Analyze directory structure
      const directories = [...new Set(files.map(f => path.dirname(f.path)))];
      const fileNames = files.map(f => f.name);

      for (const [architecture, config] of Object.entries(this.architecturePatterns)) {
        let score = 0;

        // Check directories
        if (config.directories) {
          for (const expectedDir of config.directories) {
            if (directories.some(dir => dir.includes(expectedDir))) {
              score += config.confidence;
            }
          }
        }

        // Check files
        if (config.files) {
          for (const expectedFile of config.files) {
            if (fileNames.some(name => name.includes(expectedFile))) {
              score += config.confidence;
            }
          }
        }

        // Check patterns in code
        if (config.patterns) {
          for (const file of files.slice(0, 20)) { // Sample first 20 files for performance
            const content = await this.getFileContent(file.path);
            if (content) {
              for (const pattern of config.patterns) {
                if (pattern.test(content)) {
                  score += config.confidence * 0.3;
                }
              }
            }
          }
        }

        if (score > 0) {
          architectureScores[architecture] = score;
        }
      }

      // Normalize and return results
      const detectedArchitecture = {};
      const maxScore = Math.max(...Object.values(architectureScores));

      for (const [architecture, score] of Object.entries(architectureScores)) {
        const normalizedScore = maxScore > 0 ? score / maxScore : 0;
        if (normalizedScore > 0.3) {
          detectedArchitecture[architecture] = {
            confidence: Math.min(normalizedScore, 1.0),
            indicators: this.getArchitectureIndicators(architecture, directories, fileNames)
          };
        }
      }

      this.logger.debug('Architecture detected', {
        patterns: Object.keys(detectedArchitecture)
      });

      return detectedArchitecture;

    } catch (error) {
      this.logger.error('Architecture detection failed', error);
      throw new Error(`Architecture detection failed: ${error.message}`);
    }
  }

  /**
   * Extract dependencies from various dependency files
   * @private
   */
  async extractDependencies(projectPath) {
    const dependencies = [];

    try {
      // JavaScript/Node.js
      const packageJsonPath = path.join(projectPath, 'package.json');
      if (await this.fileHandler.exists(packageJsonPath)) {
        const packageJson = await this.fileHandler.readJSON(packageJsonPath);
        const deps = [
          ...Object.keys(packageJson.dependencies || {}),
          ...Object.keys(packageJson.devDependencies || {}),
          ...Object.keys(packageJson.peerDependencies || {})
        ];
        dependencies.push(...deps.map(d => d.toLowerCase()));
      }

      // Python
      const requirementsPath = path.join(projectPath, 'requirements.txt');
      if (await this.fileHandler.exists(requirementsPath)) {
        const requirements = await this.fileHandler.readFile(requirementsPath);
        const pythonDeps = requirements
          .split('\n')
          .map(line => line.trim().split(/[>=<]/)[0])
          .filter(dep => dep && !dep.startsWith('#'));
        dependencies.push(...pythonDeps.map(d => d.toLowerCase()));
      }

      // More dependency file types can be added here

    } catch (error) {
      this.logger.warn('Failed to extract some dependencies', error);
    }

    return [...new Set(dependencies)]; // Remove duplicates
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
   * Get language indicators for reporting
   * @private
   */
  getLanguageIndicators(language, files) {
    const config = this.languagePatterns[language];
    const indicators = [];

    // Find matching files
    const matchingFiles = files.filter(f => 
      config.extensions.includes(path.extname(f.name).toLowerCase())
    ).slice(0, 3);

    indicators.push(...matchingFiles.map(f => f.name));

    return indicators;
  }

  /**
   * Get framework indicators for reporting
   * @private
   */
  getFrameworkIndicators(framework, dependencies, files) {
    const config = this.frameworkPatterns[framework];
    const indicators = [];

    // Dependency indicators
    if (config.dependencies) {
      for (const dep of config.dependencies) {
        if (dependencies.includes(dep.toLowerCase())) {
          indicators.push(`dependency: ${dep}`);
        }
      }
    }

    // File indicators
    if (config.files) {
      for (const expectedFile of config.files) {
        const matchingFile = files.find(f => f.path.includes(expectedFile));
        if (matchingFile) {
          indicators.push(`file: ${path.basename(matchingFile.path)}`);
        }
      }
    }

    return indicators;
  }

  /**
   * Get framework category for classification
   * @private
   */
  getFrameworkCategory(framework) {
    const categories = {
      react: 'frontend',
      vue: 'frontend',
      angular: 'frontend',
      svelte: 'frontend',
      nextjs: 'fullstack',
      express: 'backend',
      fastify: 'backend',
      nestjs: 'backend',
      django: 'backend',
      flask: 'backend',
      fastapi: 'backend',
      spring: 'backend',
      tensorflow: 'ml',
      pytorch: 'ml',
      jest: 'testing',
      cypress: 'testing',
      playwright: 'testing'
    };

    return categories[framework] || 'other';
  }

  /**
   * Get architecture indicators for reporting
   * @private
   */
  getArchitectureIndicators(architecture, directories, fileNames) {
    const config = this.architecturePatterns[architecture];
    const indicators = [];

    if (config.directories) {
      for (const expectedDir of config.directories) {
        const matchingDir = directories.find(dir => dir.includes(expectedDir));
        if (matchingDir) {
          indicators.push(`directory: ${expectedDir}`);
        }
      }
    }

    if (config.files) {
      for (const expectedFile of config.files) {
        if (fileNames.some(name => name.includes(expectedFile))) {
          indicators.push(`file: ${expectedFile}`);
        }
      }
    }

    return indicators;
  }
}

module.exports = FileAnalyzer;