/**
 * Integration test – RealEnsembleOptimizer with real scikit-learn wrappers
 * -----------------------------------------------------------------------
 * Verifies that:
 *   1. Python bridge boots successfully (child_process spawned)
 *   2. RealEnsembleOptimizer trains on a tiny synthetic dataset
 *   3. Prediction call works and returns averaged probability
 *   4. Bridge is terminated cleanly at the end
 */

import { RealEnsembleOptimizer } from '../src/ensemble-optimizer.js';

(async () => {
  console.log('\uD83E\uDD16  RealEnsembleOptimizer – integration test');

  // Step 1: construct optimizer
  const optimizer = new RealEnsembleOptimizer();
  console.log('   ✅ Optimizer instantiated');

  // Step 2: synthetic dataset (minimal)
  const dataset = [
    { prompt: 'Write hello world', features: [17, 1, 0], label: 1, context: { domain: 'generic' }, expectedScore: 0.9 },
    { prompt: 'Explain HTTP protocol', features: [22, 1, 0], label: 1, context: { domain: 'web-development' }, expectedScore: 0.8 },
    { prompt: 'Define photosynthesis', features: [22, 1, 0], label: 1, context: { domain: 'biology' }, expectedScore: 0.85 },
    { prompt: '2+2?', features: [4, 1, 0], label: 0, context: { domain: 'math' }, expectedScore: 0.4 },
    { prompt: 'Random joke', features: [11, 0, 0], label: 0, context: { domain: 'humor' }, expectedScore: 0.3 }
  ];

  // Step 3: train – this triggers Python scikit-learn fits
  console.log('   ⏳ Training real ensemble (this may take a few seconds)…');
  await optimizer.trainRealEnsemble(dataset);
  console.log('   ✅ Training complete');

  // Step 4: prediction - now pass a feature vector
  const predictionFeatures = [20, 2, 0]; // A feature vector for "Hello explain world"
  const pred = await optimizer.predictWithRealEnsemble(predictionFeatures);
  if (typeof pred.score !== 'number') throw new Error('Prediction did not return numeric score');
  console.log(`   ✅ Prediction OK – score = ${pred.score.toFixed(3)}`);

  // Cleanup bridge
  optimizer.bridge.close();
  console.log('   ✅ Bridge shut down');

  console.log('\n🎉  RealEnsembleOptimizer integration test passed');
  process.exit(0);
})().catch(err => {
  console.error('💥  Test failed:', err);
  process.exit(1);
}); 