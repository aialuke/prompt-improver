import SklearnBridge from './client.js';

/**
 * SklearnModelWrapper – convenience layer around a single scikit-learn estimator
 */
export default class SklearnModelWrapper {
  /**
   * @param {SklearnBridge} bridge  existing bridge instance
   * @param {string} klass           estimator class name (e.g. 'RandomForestClassifier')
   * @param {object} defaultParams   baseline hyper-parameters
   */
  constructor(bridge, klass, defaultParams = {}) {
    this.bridge = bridge instanceof SklearnBridge ? bridge : new SklearnBridge();
    this.klass = klass;
    this.defaultParams = { ...defaultParams };
    this.modelId = null;
  }

  /** Train the underlying estimator */
  async fit(X, y, params = {}) {
    const combined = { ...this.defaultParams, ...params };
    this.modelId = await this.bridge.fitModel(this.klass, combined, X, y);
    this.params = combined;
    return this;
  }

  /** Predict class probabilities (if available) or labels */
  async predict(X) {
    if (!this.modelId) throw new Error('Model not trained/loaded');
    return await this.bridge.predict(this.modelId, X);
  }

  async save(path, compress = 3) {
    if (!this.modelId) throw new Error('Model not trained/loaded');
    await this.bridge.saveModel(this.modelId, path, compress);
  }

  async load(path) {
    this.modelId = await this.bridge.loadModel(path);
  }

  /** Update default hyper-parameters (Optuna integration) */
  updateParams(delta = {}) {
    this.defaultParams = { ...this.defaultParams, ...delta };
  }

  /**
   * Run Optuna HPO with nested CV via the bridge and merge best params.
   * @param {Array<Array<number>>} X feature matrix
   * @param {Array<number>} y labels
   * @param {number} nTrials number of Optuna trials (default 30)
   * @returns {Promise<object>} best params
   */
  async optimizeHyperparameters(X, y, nTrials = 30) {
    const minSamples = y.length;
    const outerFolds = Math.min(5, Math.max(2, Math.floor(minSamples / 2)));
    const innerFolds = Math.min(3, outerFolds);
    const result = await this.bridge.optimizeModel(this.klass, X, y, null, nTrials, innerFolds, outerFolds);
    
    // The bridge now returns a handle to the best, fully-fitted estimator
    this.modelId = result.model_id; 
    
    this.updateParams(result.best_params);
    this.paramsPerformance = result;
    return result;
  }
} 