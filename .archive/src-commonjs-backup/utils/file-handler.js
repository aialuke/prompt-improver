/**
 * Robust file I/O operations utility
 * Handles file reading, writing, and manipulation with error recovery
 */

const fs = require('fs').promises;
const path = require('path');
const Logger = require('./logger');

class FileHandler {
  constructor() {
    this.logger = new Logger('FileHandler');
  }

  /**
   * Read file content with encoding support
   * @param {string} filePath - Path to the file
   * @param {string} encoding - File encoding (default: utf8)
   * @returns {Promise<string>} File content
   */
  async readFile(filePath, encoding = 'utf8') {
    try {
      this.logger.debug('Reading file', { filePath });
      const content = await fs.readFile(filePath, encoding);
      return content;
    } catch (error) {
      this.logger.error('Failed to read file', { filePath, error: error.message });
      throw new Error(`Failed to read file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Write content to file with directory creation
   * @param {string} filePath - Path to the file
   * @param {string} content - Content to write
   * @param {Object} options - Write options
   * @returns {Promise<void>}
   */
  async writeFile(filePath, content, options = {}) {
    try {
      // Ensure directory exists
      const directory = path.dirname(filePath);
      await this.ensureDirectory(directory);

      this.logger.debug('Writing file', { filePath, size: content.length });
      await fs.writeFile(filePath, content, { encoding: 'utf8', ...options });
    } catch (error) {
      this.logger.error('Failed to write file', { filePath, error: error.message });
      throw new Error(`Failed to write file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Append content to file
   * @param {string} filePath - Path to the file
   * @param {string} content - Content to append
   * @returns {Promise<void>}
   */
  async appendFile(filePath, content) {
    try {
      this.logger.debug('Appending to file', { filePath, size: content.length });
      await fs.appendFile(filePath, content, 'utf8');
    } catch (error) {
      this.logger.error('Failed to append to file', { filePath, error: error.message });
      throw new Error(`Failed to append to file ${filePath}: ${error.message}`);
    }
  }

  /**
   * Read JSON file and parse content
   * @param {string} filePath - Path to JSON file
   * @returns {Promise<Object>} Parsed JSON object
   */
  async readJSON(filePath) {
    try {
      const content = await this.readFile(filePath);
      return JSON.parse(content);
    } catch (error) {
      if (error.message.includes('JSON')) {
        throw new Error(`Invalid JSON in file ${filePath}: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * Write object to JSON file
   * @param {string} filePath - Path to JSON file
   * @param {Object} data - Data to write
   * @param {boolean} pretty - Whether to format JSON (default: true)
   * @returns {Promise<void>}
   */
  async writeJSON(filePath, data, pretty = true) {
    const content = pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
    await this.writeFile(filePath, content);
  }

  /**
   * Check if file or directory exists
   * @param {string} filePath - Path to check
   * @returns {Promise<boolean>} Whether path exists
   */
  async exists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get file stats
   * @param {string} filePath - Path to file
   * @returns {Promise<Object>} File stats
   */
  async getStats(filePath) {
    try {
      const stats = await fs.stat(filePath);
      return {
        size: stats.size,
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory(),
        modified: stats.mtime,
        created: stats.birthtime
      };
    } catch (error) {
      throw new Error(`Failed to get stats for ${filePath}: ${error.message}`);
    }
  }

  /**
   * List files in directory with optional filtering
   * @param {string} dirPath - Directory path
   * @param {Object} options - Filtering options
   * @returns {Promise<Array>} List of files
   */
  async listFiles(dirPath, options = {}) {
    try {
      const {
        recursive = false,
        extensions = null,
        excludePatterns = [],
        includeStats = false
      } = options;

      const files = [];
      const entries = await fs.readdir(dirPath, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        
        // Skip excluded patterns
        if (excludePatterns.some(pattern => entry.name.includes(pattern))) {
          continue;
        }

        if (entry.isFile()) {
          // Filter by extensions if specified
          if (extensions && !extensions.includes(path.extname(entry.name))) {
            continue;
          }

          const fileInfo = { name: entry.name, path: fullPath };
          if (includeStats) {
            fileInfo.stats = await this.getStats(fullPath);
          }
          files.push(fileInfo);

        } else if (entry.isDirectory() && recursive) {
          const subFiles = await this.listFiles(fullPath, options);
          files.push(...subFiles);
        }
      }

      return files;
    } catch (error) {
      throw new Error(`Failed to list files in ${dirPath}: ${error.message}`);
    }
  }

  /**
   * Ensure directory exists, create if necessary
   * @param {string} dirPath - Directory path
   * @returns {Promise<void>}
   */
  async ensureDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      if (error.code !== 'EEXIST') {
        throw new Error(`Failed to create directory ${dirPath}: ${error.message}`);
      }
    }
  }

  /**
   * Copy file to destination
   * @param {string} sourcePath - Source file path
   * @param {string} destPath - Destination file path
   * @returns {Promise<void>}
   */
  async copyFile(sourcePath, destPath) {
    try {
      const destDir = path.dirname(destPath);
      await this.ensureDirectory(destDir);
      await fs.copyFile(sourcePath, destPath);
      this.logger.debug('File copied', { sourcePath, destPath });
    } catch (error) {
      throw new Error(`Failed to copy file from ${sourcePath} to ${destPath}: ${error.message}`);
    }
  }

  /**
   * Delete file or directory
   * @param {string} filePath - Path to delete
   * @param {Object} options - Delete options
   * @returns {Promise<void>}
   */
  async delete(filePath, options = {}) {
    try {
      const { recursive = false } = options;
      const stats = await this.getStats(filePath);

      if (stats.isDirectory) {
        await fs.rmdir(filePath, { recursive });
      } else {
        await fs.unlink(filePath);
      }

      this.logger.debug('Path deleted', { filePath, recursive });
    } catch (error) {
      throw new Error(`Failed to delete ${filePath}: ${error.message}`);
    }
  }

  /**
   * Get file size in bytes
   * @param {string} filePath - Path to file
   * @returns {Promise<number>} File size in bytes
   */
  async getFileSize(filePath) {
    const stats = await this.getStats(filePath);
    return stats.size;
  }

  /**
   * Create backup of file
   * @param {string} filePath - Path to file to backup
   * @param {string} backupSuffix - Suffix for backup file (default: timestamp)
   * @returns {Promise<string>} Backup file path
   */
  async createBackup(filePath, backupSuffix = null) {
    if (!backupSuffix) {
      backupSuffix = new Date().toISOString().replace(/[:.]/g, '-');
    }

    const parsedPath = path.parse(filePath);
    const backupPath = path.join(
      parsedPath.dir,
      `${parsedPath.name}.backup.${backupSuffix}${parsedPath.ext}`
    );

    await this.copyFile(filePath, backupPath);
    this.logger.info('Backup created', { original: filePath, backup: backupPath });
    
    return backupPath;
  }
}

module.exports = FileHandler;