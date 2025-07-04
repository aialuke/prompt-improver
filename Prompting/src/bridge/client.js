import { spawn } from 'child_process';
import path, { dirname } from 'path';
import { randomUUID } from 'crypto';
import { fileURLToPath } from 'url';

/**
 * SklearnBridge
 * --------------
 * Manages a persistent python subprocess running `python/sklearn_bridge.py` and
 * performs RPC over newline-delimited JSON.
 */
export default class SklearnBridge {
  /**
   * @param {string} [pythonBin] – path to python executable (default: python3)
   * @param {string} [scriptRelPath] – relative path to bridge script
   */
  constructor(pythonBin = 'python3', scriptRelPath = '../../ml/bridge.py') {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const scriptPath = path.resolve(__dirname, scriptRelPath);
    this.proc = spawn(pythonBin, [scriptPath], {
      stdio: ['pipe', 'pipe', 'inherit']
    });

    this._buffer = '';
    this._pending = new Map(); // reqId -> {resolve, reject}

    this.proc.stdout.on('data', chunk => {
      this._buffer += chunk.toString();
      let lines = this._buffer.split('\n');
      this._buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        let payload;
        try {
          payload = JSON.parse(line);
        } catch (e) {
          console.error('Failed to parse bridge output:', line);
          continue;
        }
        const { __reqid } = payload;
        if (__reqid && this._pending.has(__reqid)) {
          const { resolve, reject } = this._pending.get(__reqid);
          this._pending.delete(__reqid);
          if (payload.status === 'ok') {
            resolve(payload);
          } else {
            reject(new Error(payload.message || 'Bridge error'));
          }
        }
      }
    });
  }

  /**
   * Internal send helper that attaches a request id and returns a Promise.
   * @param {object} obj JSON payload to send
   * @returns {Promise<object>} bridge response
   */
  _send(obj) {
    const reqId = randomUUID();
    return new Promise((resolve, reject) => {
      this._pending.set(reqId, { resolve, reject });
      const payload = { ...obj, __reqid: reqId };
      this.proc.stdin.write(JSON.stringify(payload) + '\n');
    });
  }

  async fitModel(klass, params, X, y) {
    const res = await this._send({ cmd: 'fit_model', klass, params, X, y });
    return res.model_id;
  }

  async predict(modelId, X) {
    const res = await this._send({ cmd: 'predict', model_id: modelId, X });
    return res; // Return full response including predictions, probabilities, and model_type
  }

  async saveModel(modelId, path, compress = 3) {
    await this._send({ cmd: 'save_model', model_id: modelId, path, compress });
  }

  async loadModel(path) {
    const res = await this._send({ cmd: 'load_model', path });
    return res.model_id;
  }

  /**
   * Hyper-parameter optimisation with nested CV & bootstrap CI
   * @param {string} klass
   * @param {Array<Array<number>>} X
   * @param {Array<number>} y
   * @param {object|null} searchSpace optional explicit Optuna distributions (JSON serialisable)
   * @param {number} nTrials number of trials (default 30)
   * @param {number} innerFolds number of inner folds (default 3)
   * @param {number} outerFolds number of outer folds (default 5)
   * @returns {Promise<{best_params:object, mean_outer_score:number, ci_low:number, ci_high:number}>}
   */
  async optimizeModel(klass, X, y, searchSpace = null, nTrials = 30, innerFolds = 3, outerFolds = 5) {
    const res = await this._send({
      cmd: 'optimize_model',
      klass,
      X,
      y,
      search_space: searchSpace,
      n_trials: nTrials,
      inner_folds: innerFolds,
      outer_folds: outerFolds,
    });
    return res;
  }

  /** Dispose the python process */
  close() {
    if (this.proc) {
      this.proc.stdin.end();
      this.proc.kill();
      this.proc = null;
    }
  }
} 