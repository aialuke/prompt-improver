// New integration accuracy test
import { RealEnsembleOptimizer } from '../src/ensemble-optimizer.js';
import { spawnSync } from 'child_process';

function loadBreastCancerDataset() {
  const pyCode = `\nimport json, sys\nfrom sklearn.datasets import load_breast_cancer\ndata = load_breast_cancer()\njson.dump({\"X\": data.data.tolist(), \"y\": data.target.tolist()}, sys.stdout)\n`;
  const res = spawnSync('python3', ['-'], { input: pyCode, encoding: 'utf-8' });
  if (res.error) throw res.error;
  if (res.status !== 0) {
    console.error(res.stderr);
    throw new Error('Failed to load dataset via Python');
  }
  return JSON.parse(res.stdout);
}

(async () => {
  console.log('🔎 RealEnsembleOptimizer – breast cancer accuracy test');
  const optimizer = new RealEnsembleOptimizer();

  // Load dataset from scikit-learn via Python
  const { X, y } = loadBreastCancerDataset();

  // Prepare dataset objects with explicit numeric features
  const dataset = X.map((features, idx) => ({
    prompt: '', // not used when features are supplied
    context: {},
    expectedScore: y[idx],
    features,
    label: y[idx]
  }));

  const trainInfo = await optimizer.trainRealEnsemble(dataset);
  console.log('   ✅ Training finished:', trainInfo.models.join(','));

  // Collect outer scores from wrappers (saved in last optimise call)
  const perf = Array.from(optimizer.trainedModels.values())
    .map(m => m.paramsPerformance)
    .filter(Boolean);

  const meanScores = perf.map(p => p.mean_outer_score);
  const cis = perf.map(p => p.ci_high - p.ci_low);
  const meanScore = meanScores.reduce((a, b) => a + b, 0) / meanScores.length;
  const meanCiWidth = cis.reduce((a, b) => a + b, 0) / cis.length;

  console.log(`   📈 Mean outer CV accuracy: ${meanScore.toFixed(3)} (CI width ${meanCiWidth.toFixed(3)})`);

  if (meanScore < 0.9) throw new Error('Accuracy target not reached');
  if (meanCiWidth > 0.2) throw new Error('CI width too large');
  console.log('🎉 Accuracy test passed');
})(); 