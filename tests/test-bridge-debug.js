// Debug test to examine bridge response structure
import { RealEnsembleOptimizer } from '../src/ensemble-optimizer.js';

(async () => {
  console.log('🔍 Bridge Response Debug Test');
  const optimizer = new RealEnsembleOptimizer();

  // Small dataset for quick testing
  const dataset = [
    { features: [1, 0, 0], label: 0, expectedScore: 0 },
    { features: [0, 1, 0], label: 1, expectedScore: 1 },
    { features: [1, 1, 0], label: 1, expectedScore: 1 },
    { features: [0, 0, 0], label: 0, expectedScore: 0 }
  ];

  console.log('🎯 Training models...');
  await optimizer.trainRealEnsemble(dataset);
  
  console.log('🔍 Testing direct bridge calls...');
  
  // Test each trained model individually
  for (const [key, wrapper] of optimizer.trainedModels.entries()) {
    console.log(`\n--- Testing ${key} model ---`);
    console.log(`Model ID: ${wrapper.modelId}`);
    
    try {
      const rawResponse = await optimizer.bridge.predict(wrapper.modelId, [[1, 1, 0]]);
      console.log(`Raw bridge response:`, JSON.stringify(rawResponse, null, 2));
      
      // Check if this is binary classification
      if (rawResponse.probabilities && rawResponse.probabilities[0]) {
        console.log(`Probabilities array: [${rawResponse.probabilities[0].join(', ')}]`);
        console.log(`Positive class prob: ${rawResponse.probabilities[0][1]}`);
      }
      
      if (rawResponse.predictions) {
        console.log(`Raw predictions: [${rawResponse.predictions.join(', ')}]`);
      }
      
    } catch (error) {
      console.log(`❌ Error testing ${key}:`, error.message);
    }
  }

  // Test ensemble prediction with debugging
  console.log('\n🧪 Testing ensemble prediction...');
  
  // Temporarily modify the prediction logic to add debugging
  const originalPredict = optimizer.predictWithRealEnsemble;
  optimizer.predictWithRealEnsemble = async function(features) {
    console.log(`\n🔍 Ensemble debug for features: [${features.join(', ')}]`);
    
    const predictions = await Promise.all(
      Array.from(this.trainedModels.entries()).map(async ([key, modelWrapper]) => {
        try {
          const modelId = modelWrapper.modelId;
          const result = await this.bridge.predict(modelId, [features]);
          console.log(`${key} raw result:`, JSON.stringify(result, null, 2));
          return { key, result, modelType: result.model_type };
        } catch (error) {
          console.log(`${key} failed:`, error.message);
          return { key, result: null, error: error.message };
        }
      })
    );

    const successfulPreds = predictions.filter(p => p.result !== null);
    console.log(`Successful predictions: ${successfulPreds.length}/${predictions.length}`);
    
    const probabilities = successfulPreds.map(p => {
      const result = p.result;
      console.log(`Extracting from ${p.key}:`, {
        hasProbs: !!result.probabilities,
        probsLength: result.probabilities?.length,
        probs0: result.probabilities?.[0],
        hasPreds: !!result.predictions,
        predsLength: result.predictions?.length,
        preds0: result.predictions?.[0]
      });
      
      if (result.probabilities && result.probabilities[0] && result.probabilities[0][1] !== undefined) {
        const prob = result.probabilities[0][1];
        console.log(`${p.key} using probability: ${prob}`);
        return prob;
      } else if (result.predictions && result.predictions[0] !== undefined) {
        const pred = Math.max(0, Math.min(1, result.predictions[0]));
        console.log(`${p.key} using prediction: ${pred}`);
        return pred;
      } else {
        console.log(`${p.key} using fallback: 0.5`);
        return 0.5;
      }
    });

    console.log('Final probabilities array:', probabilities);
    const avgProb = probabilities.reduce((sum, prob) => sum + prob, 0) / probabilities.length;
    console.log('Average probability:', avgProb);
    
    return { score: avgProb, debug: true };
  };

  const result = await optimizer.predictWithRealEnsemble([1, 1, 0]);
  console.log('\n🎯 Final result:', result);

  optimizer.bridge.close();
  console.log('\n🎉 Debug test complete');
})().catch(err => {
  console.error('💥 Debug test failed:', err);
  process.exit(1);
});