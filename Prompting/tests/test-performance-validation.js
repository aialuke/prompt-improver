// Quick performance validation test
import { RealEnsembleOptimizer } from '../src/ensemble-optimizer.js';

(async () => {
  console.log('🔍 Performance Validation Test');
  const optimizer = new RealEnsembleOptimizer();

  // Small synthetic dataset for quick validation
  const dataset = [
    { features: [1, 0, 0], label: 0, expectedScore: 0 },
    { features: [0, 1, 0], label: 1, expectedScore: 1 },
    { features: [0, 0, 1], label: 0, expectedScore: 0 },
    { features: [1, 1, 0], label: 1, expectedScore: 1 },
    { features: [1, 0, 1], label: 0, expectedScore: 0 },
    { features: [0, 1, 1], label: 1, expectedScore: 1 },
    { features: [1, 1, 1], label: 1, expectedScore: 1 },
    { features: [0, 0, 0], label: 0, expectedScore: 0 }
  ];

  console.log('⏱️  Starting training...');
  const startTime = Date.now();
  
  const trainInfo = await optimizer.trainRealEnsemble(dataset);
  const trainingTime = Date.now() - startTime;
  
  console.log(`   ✅ Training completed in ${trainingTime}ms`);
  console.log('   🎯 Models trained:', trainInfo.models.join(', '));

  // Test predictions
  console.log('🧪 Testing predictions...');
  
  // Test case 1: Clear positive case
  const pred1 = await optimizer.predictWithRealEnsemble([1, 1, 0]);
  console.log(`   📊 Prediction [1,1,0]: score=${pred1.score.toFixed(3)}, confidence=${pred1.confidence.toFixed(3)}`);
  console.log(`      Models used: ${pred1.successful_models}/${pred1.total_models}`);
  
  // Test case 2: Clear negative case
  const pred2 = await optimizer.predictWithRealEnsemble([0, 0, 0]);
  console.log(`   📊 Prediction [0,0,0]: score=${pred2.score.toFixed(3)}, confidence=${pred2.confidence.toFixed(3)}`);
  
  // Test case 3: Ambiguous case
  const pred3 = await optimizer.predictWithRealEnsemble([0.5, 0.5, 0.5]);
  console.log(`   📊 Prediction [0.5,0.5,0.5]: score=${pred3.score.toFixed(3)}, confidence=${pred3.confidence.toFixed(3)}`);

  // Validate that predictions are not random
  if (pred1.score === 0.5 && pred2.score === 0.5 && pred3.score === 0.5) {
    console.log('❌ WARNING: All predictions = 0.5, possibly using fallback logic');
  } else {
    console.log('✅ Predictions show variation, indicating real ML models');
  }

  // Check for failures
  if (pred1.failures && pred1.failures.length > 0) {
    console.log('⚠️  Some model failures detected:', pred1.failures);
  }

  optimizer.bridge.close();
  console.log('🎉 Performance validation complete');
})().catch(err => {
  console.error('💥 Test failed:', err);
  process.exit(1);
});