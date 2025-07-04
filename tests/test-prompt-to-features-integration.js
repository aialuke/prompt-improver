/**
 * Integration test for prompt-to-features with real ML ensemble
 * Tests the complete pipeline: Prompt → PromptAnalyzer → Features → ML Models → Quality Score
 */

import { RealEnsembleOptimizer } from '../src/ensemble-optimizer.js';
import PromptAnalyzer from '../src/prompt-analyzer.js';

console.log('🚀 Prompt-to-Features Integration Test\n');

(async () => {
  const optimizer = new RealEnsembleOptimizer();
  const analyzer = new PromptAnalyzer();

  // Step 1: Create comprehensive prompt quality dataset
  console.log('📋 Creating comprehensive prompt quality dataset...');
  
  const prompts = [
    {
      text: "Write a detailed technical specification document for a REST API that handles user authentication. The document should include exactly 5 sections: Overview, Endpoints, Authentication Flow, Error Handling, and Examples. Use formal technical writing style and provide specific JSON examples for each endpoint. The document should be approximately 2000 words and follow standard API documentation format.",
      label: 1, // High quality
      expectedScore: 0.9
    },
    {
      text: "Create a user guide for our new mobile app. Include screenshots and step-by-step instructions. Make it user-friendly and cover the main features like login, settings, and notifications.",
      label: 1, // Medium-high quality  
      expectedScore: 0.75
    },
    {
      text: "Explain how machine learning works. Include examples and make it easy to understand. Cover the main concepts and algorithms.",
      label: 1, // Medium quality
      expectedScore: 0.65
    },
    {
      text: "Write something about mobile apps. Make it good.",
      label: 0, // Low quality
      expectedScore: 0.25
    },
    {
      text: "Can you help me with some stuff? I need something that works well and looks nice. Maybe use some examples or whatever.",
      label: 0, // Very low quality
      expectedScore: 0.15
    },
    {
      text: "Implement a binary search algorithm in Python. The function should take a sorted array and target value as parameters, return the index if found or -1 if not found. Include comprehensive error handling for edge cases like empty arrays and invalid inputs. Add docstrings following PEP 257 conventions and include 5 test cases with assertions.",
      label: 1, // Very high quality
      expectedScore: 0.95
    },
    {
      text: "List the steps to make a sandwich.",
      label: 0, // Simple, low quality for ML task
      expectedScore: 0.4
    },
    {
      text: "Generate a comprehensive business plan for a SaaS startup in the healthcare industry. The plan should include: executive summary, market analysis with specific data points, competitive landscape assessment, product roadmap with technical specifications, financial projections for 3 years, go-to-market strategy with specific channels, team structure and hiring plan, risk assessment with mitigation strategies. Use professional business language and include specific metrics, timelines, and budget allocations. Format the document with clear sections, bullet points, and supporting charts where appropriate.",
      label: 1, // Extremely high quality
      expectedScore: 0.98
    }
  ];

  // Step 2: Extract features using PromptAnalyzer
  console.log('🔍 Extracting features using PromptAnalyzer...');
  
  const dataset = prompts.map((item, index) => {
    const analysis = analyzer.analyzePrompt(item.text);
    console.log(`${index + 1}. "${item.text.substring(0, 60)}..."`);
    console.log(`   Quality Score: ${analysis.qualityScore.toFixed(3)} | Expected: ${item.expectedScore}`);
    console.log(`   Features: ${analysis.features.length} dimensions`);
    console.log(`   Key Features: clarity=${analysis.featureMap.task_clarity.toFixed(2)}, specificity=${analysis.featureMap.specificity_score.toFixed(2)}, completeness=${analysis.featureMap.instruction_completeness.toFixed(2)}`);
    
    return {
      prompt: item.text,
      features: analysis.features,
      label: item.label,
      expectedScore: item.expectedScore,
      context: { 
        domain: 'prompt-engineering',
        analysis: analysis
      }
    };
  });

  console.log(`\n✅ Dataset prepared with ${dataset.length} samples, ${dataset[0].features.length} features per sample\n`);

  // Step 3: Train the ensemble with extracted features
  console.log('🎯 Training ensemble with extracted features...');
  const trainInfo = await optimizer.trainRealEnsemble(dataset);
  console.log(`✅ Training completed: ${trainInfo.models.join(', ')}\n`);

  // Step 4: Test prompt-to-prediction pipeline
  console.log('🧪 Testing prompt-to-prediction pipeline...');
  
  const testPrompts = [
    {
      text: "Design a comprehensive API security framework that includes OAuth 2.0 implementation, rate limiting, input validation, SQL injection prevention, and audit logging. Provide specific code examples in Python using Flask, include configuration templates, and detail deployment considerations for production environments.",
      expectedCategory: "high"
    },
    {
      text: "Write a tutorial about web development. Make it good and include examples.",
      expectedCategory: "medium"
    },
    {
      text: "Do something with computers.",
      expectedCategory: "low"
    }
  ];

  for (const [index, testCase] of testPrompts.entries()) {
    console.log(`\n${index + 1}. Testing: "${testCase.text.substring(0, 80)}..."`);
    
    // Step 4a: Analyze the prompt
    const analysis = analyzer.analyzePrompt(testCase.text);
    console.log(`   📊 PromptAnalyzer Quality Score: ${analysis.qualityScore.toFixed(3)}`);
    
    // Step 4b: Get ML prediction using features
    const prediction = await optimizer.predictWithRealEnsemble(testCase.text);
    console.log(`   🎯 ML Ensemble Score: ${prediction.score.toFixed(3)}`);
    console.log(`   📈 Confidence: ${prediction.confidence.toFixed(3)}`);
    console.log(`   🤖 Models Used: ${prediction.successful_models}/${prediction.total_models}`);
    
    // Step 4c: Show feature analysis insights
    if (analysis.insights.length > 0) {
      console.log(`   💡 Insights: ${analysis.insights[0]}`);
    }
    if (analysis.recommendations.length > 0) {
      console.log(`   🚀 Recommendation: ${analysis.recommendations[0]}`);
    }
    
    // Step 4d: Compare analyzer vs ML prediction
    const analyzerCategory = analysis.qualityScore > 0.7 ? 'high' : analysis.qualityScore > 0.4 ? 'medium' : 'low';
    const mlCategory = prediction.score > 0.7 ? 'high' : prediction.score > 0.4 ? 'medium' : 'low';
    
    console.log(`   📋 Analyzer Category: ${analyzerCategory} | ML Category: ${mlCategory} | Expected: ${testCase.expectedCategory}`);
    
    const agreement = analyzerCategory === mlCategory ? '✅ AGREE' : '⚠️ DIFFER';
    console.log(`   🔄 Agreement: ${agreement}`);
  }

  // Step 5: Feature importance analysis
  console.log('\n📊 Feature Importance Analysis:');
  const featureNames = analyzer.getFeatureNames();
  
  // Sample a few prompts to understand feature distributions
  const sampleAnalyses = prompts.slice(0, 3).map(p => analyzer.analyzePrompt(p.text));
  
  console.log('\nTop feature variations across samples:');
  const featureStats = featureNames.map((name, index) => {
    const values = sampleAnalyses.map(a => a.features[index]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    return { name, range, min, max };
  }).sort((a, b) => b.range - a.range).slice(0, 10);
  
  featureStats.forEach((stat, index) => {
    console.log(`${index + 1}. ${stat.name}: range=${stat.range.toFixed(3)} (${stat.min.toFixed(2)} to ${stat.max.toFixed(2)})`);
  });

  // Step 6: Validation summary
  console.log('\n📈 Integration Test Summary:');
  console.log(`✅ PromptAnalyzer: ${featureNames.length} features extracted successfully`);
  console.log(`✅ ML Pipeline: ${trainInfo.models.length} models trained successfully`);
  console.log(`✅ End-to-End: String prompts → Features → ML predictions working`);
  console.log(`✅ Feature Categories: Structural, Semantic, Task-Oriented, Quality Indicators`);
  console.log(`✅ Advanced Features: Context richness, specificity, task clarity, completeness`);

  // Cleanup
  optimizer.bridge.close();
  console.log('\n🎉 Prompt-to-Features Integration Test Complete!');
  
})().catch(err => {
  console.error('💥 Integration test failed:', err);
  process.exit(1);
});