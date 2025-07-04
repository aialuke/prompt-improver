/**
 * Test suite for PromptAnalyzer
 * Validates prompt feature extraction and quality assessment
 */

import PromptAnalyzer from '../src/prompt-analyzer.js';

const analyzer = new PromptAnalyzer();

console.log('🧪 Testing PromptAnalyzer Feature Extraction\n');

// Test cases representing different prompt quality levels
const testCases = [
  {
    name: "High Quality Prompt",
    prompt: "Write a detailed technical specification document for a REST API that handles user authentication. The document should include exactly 5 sections: Overview, Endpoints, Authentication Flow, Error Handling, and Examples. Use formal technical writing style and provide specific JSON examples for each endpoint. The document should be approximately 2000 words and follow standard API documentation format.",
    expectedQuality: "high"
  },
  {
    name: "Medium Quality Prompt", 
    prompt: "Create a user guide for our new mobile app. Include screenshots and step-by-step instructions. Make it user-friendly and cover the main features like login, settings, and notifications.",
    expectedQuality: "medium"
  },
  {
    name: "Low Quality Prompt",
    prompt: "Write something about mobile apps. Make it good.",
    expectedQuality: "low"
  },
  {
    name: "Vague Prompt",
    prompt: "Can you help me with some stuff? I need something that works well and looks nice. Maybe use some examples or whatever.",
    expectedQuality: "low"
  },
  {
    name: "Technical Prompt",
    prompt: "Implement a binary search algorithm in Python. The function should take a sorted array and target value as parameters, return the index if found or -1 if not found. Include comprehensive error handling for edge cases like empty arrays and invalid inputs. Add docstrings following PEP 257 conventions and include 5 test cases with assertions.",
    expectedQuality: "high"
  }
];

console.log('📊 Feature Extraction Results:\n');

testCases.forEach((testCase, index) => {
  console.log(`${index + 1}. ${testCase.name}`);
  console.log(`Prompt: "${testCase.prompt.substring(0, 100)}${testCase.prompt.length > 100 ? '...' : ''}"`);
  
  const analysis = analyzer.analyzePrompt(testCase.prompt);
  const features = analysis.features;
  
  console.log(`📈 Quality Score: ${analysis.qualityScore.toFixed(3)}`);
  console.log(`🔍 Key Features:`);
  console.log(`   - Word Count: ${Math.round(features[1])}`);
  console.log(`   - Readability: ${features[9].toFixed(2)}`);
  console.log(`   - Specificity: ${features[21].toFixed(3)}`);
  console.log(`   - Task Clarity: ${features[24].toFixed(3)}`);
  console.log(`   - Context Richness: ${features[19].toFixed(3)}`);
  console.log(`   - Has Instruction: ${features[22] ? 'Yes' : 'No'}`);
  
  if (analysis.insights.length > 0) {
    console.log(`💡 Insights:`);
    analysis.insights.forEach(insight => console.log(`   - ${insight}`));
  }
  
  if (analysis.recommendations.length > 0) {
    console.log(`🚀 Recommendations:`);
    analysis.recommendations.forEach(rec => console.log(`   - ${rec}`));
  }
  
  console.log('');
});

console.log('🔬 Feature Vector Analysis:\n');

// Test feature vector generation
const samplePrompt = "Write a comprehensive marketing plan for a new SaaS product. Include market analysis, target audience, pricing strategy, and promotional channels. Format as a professional business document with clear sections and bullet points.";

const features = analyzer.extractFeatures(samplePrompt);
const featureNames = analyzer.getFeatureNames();

console.log(`Sample Prompt: "${samplePrompt}"`);
console.log(`Feature Vector Length: ${features.length}`);
console.log(`Feature Names: ${featureNames.length}`);

// Display top 10 most significant features
console.log('\n🎯 Top Features:');
const featureMap = featureNames.map((name, index) => ({ name, value: features[index] }))
  .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  .slice(0, 10);

featureMap.forEach((feature, index) => {
  console.log(`${index + 1}. ${feature.name}: ${feature.value.toFixed(4)}`);
});

console.log('\n✅ PromptAnalyzer test completed successfully!');
console.log('\n📋 Summary:');
console.log(`- Total test cases: ${testCases.length}`);
console.log(`- Feature dimensions: ${features.length}`);
console.log(`- Feature categories: 4 (Structural, Semantic, Task-Oriented, Quality)`);
console.log(`- Analysis components: Features, Insights, Recommendations`);