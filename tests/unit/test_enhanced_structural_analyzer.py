#!/usr/bin/env python3
"""
Test Enhanced StructuralAnalyzer 2025 Implementation

Tests the enhanced structural analyzer with 2025 features including:
- Graph-based structural representation
- Semantic understanding with transformers
- Automated pattern discovery
- Multi-dimensional quality assessment
"""

import asyncio
import logging
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStructuralAnalyzerTester:
    """Test the enhanced structural analyzer"""
    
    def __init__(self):
        self.test_results = {}
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of enhanced structural analyzer"""
        
        print("🚀 Enhanced StructuralAnalyzer 2025 Test")
        print("=" * 60)
        
        # Test 1: Basic functionality
        basic_result = await self._test_basic_functionality()
        
        # Test 2: Enhanced features
        enhanced_result = await self._test_enhanced_features()
        
        # Test 3: Orchestrator integration
        orchestrator_result = await self._test_orchestrator_integration()
        
        # Test 4: Performance and scalability
        performance_result = await self._test_performance()
        
        # Compile results
        overall_result = {
            "basic_functionality": basic_result,
            "enhanced_features": enhanced_result,
            "orchestrator_integration": orchestrator_result,
            "performance": performance_result,
            "summary": self._generate_test_summary()
        }
        
        self._print_test_results(overall_result)
        return overall_result
    
    async def _test_basic_functionality(self):
        """Test basic structural analysis functionality"""
        
        print("\n📋 Test 1: Basic Functionality")
        
        try:
            from prompt_improver.ml.evaluation.structural_analyzer import EnhancedStructuralAnalyzer, EnhancedStructuralConfig
            
            # Initialize analyzer
            config = EnhancedStructuralConfig(
                enable_semantic_analysis=False,  # Disable for basic test
                enable_graph_analysis=False,
                enable_pattern_discovery=False,
                enable_quality_assessment=True
            )
            analyzer = EnhancedStructuralAnalyzer(config)
            
            # Test with sample text
            test_text = """
# Task: Write a Product Review

## Context
You are writing a review for a new smartphone.

## Instructions
Please write a comprehensive review that covers:
1. Design and build quality
2. Performance and speed
3. Camera quality
4. Battery life

## Output Format
- Use bullet points for main features
- Include a rating out of 5 stars
- Keep the review under 500 words

## Example
For reference, here's a sample opening:
"The new XPhone Pro delivers exceptional performance..."
"""
            
            # Run analysis
            result = await analyzer.analyze_enhanced_structure(test_text)
            
            # Validate basic structure
            success = (
                "structural_elements" in result and
                "quality_metrics" in result and
                "insights_and_recommendations" in result and
                result["structural_elements"]["total_elements"] > 0
            )
            
            element_types = result["structural_elements"]["element_types"]
            has_headers = "header" in element_types
            has_instructions = "instruction" in element_types
            has_context = "context" in element_types
            
            print(f"  {'✅' if success else '❌'} Basic Analysis: {'PASSED' if success else 'FAILED'}")
            print(f"  {'✅' if has_headers else '❌'} Header Detection: {'PASSED' if has_headers else 'FAILED'}")
            print(f"  {'✅' if has_instructions else '❌'} Instruction Detection: {'PASSED' if has_instructions else 'FAILED'}")
            print(f"  {'✅' if has_context else '❌'} Context Detection: {'PASSED' if has_context else 'FAILED'}")
            
            return {
                "success": success,
                "elements_detected": result["structural_elements"]["total_elements"],
                "element_types": element_types,
                "quality_score": result["quality_metrics"].get("overall_score", 0.0)
            }
            
        except Exception as e:
            print(f"  ❌ Basic functionality test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_enhanced_features(self):
        """Test enhanced 2025 features"""
        
        print("\n🔧 Test 2: Enhanced Features")
        
        try:
            from prompt_improver.ml.evaluation.structural_analyzer import EnhancedStructuralAnalyzer, EnhancedStructuralConfig
            
            # Initialize analyzer with all features enabled
            config = EnhancedStructuralConfig(
                enable_semantic_analysis=True,
                enable_graph_analysis=True,
                enable_pattern_discovery=True,
                enable_quality_assessment=True
            )
            analyzer = EnhancedStructuralAnalyzer(config)
            
            # Test with complex text
            test_text = """
# AI Assistant Instructions

## Context and Background
You are an AI assistant helping users with data analysis tasks.
The user needs help understanding their sales data patterns.

## Primary Task
Analyze the provided sales data and identify key trends.
Focus on seasonal patterns and growth opportunities.

## Analysis Requirements
1. Calculate monthly growth rates
2. Identify top-performing products
3. Detect seasonal trends
4. Recommend optimization strategies

## Output Specifications
Format your response as:
- Executive summary (2-3 sentences)
- Key findings (bullet points)
- Recommendations (numbered list)
- Supporting data (table format)

## Quality Standards
Ensure all analysis is:
- Data-driven and objective
- Clearly explained
- Actionable for business decisions
"""
            
            # Run enhanced analysis
            result = await analyzer.analyze_enhanced_structure(test_text)
            
            # Check enhanced features
            has_semantic = "semantic_analysis" in result and result["semantic_analysis"]
            has_graph = "structural_graph" in result and result["structural_graph"]
            has_patterns = "discovered_patterns" in result and len(result["discovered_patterns"]) > 0
            has_quality = "quality_metrics" in result and "quality_dimensions" in result["quality_metrics"]
            
            enhanced_success = has_semantic or has_graph or has_patterns or has_quality
            
            print(f"  {'✅' if has_semantic else '⚠️'} Semantic Analysis: {'ENABLED' if has_semantic else 'DISABLED'}")
            print(f"  {'✅' if has_graph else '⚠️'} Graph Analysis: {'ENABLED' if has_graph else 'DISABLED'}")
            print(f"  {'✅' if has_patterns else '⚠️'} Pattern Discovery: {'ENABLED' if has_patterns else 'DISABLED'}")
            print(f"  {'✅' if has_quality else '❌'} Quality Assessment: {'ENABLED' if has_quality else 'FAILED'}")
            
            return {
                "success": enhanced_success,
                "semantic_analysis": has_semantic,
                "graph_analysis": has_graph,
                "pattern_discovery": has_patterns,
                "quality_assessment": has_quality,
                "patterns_found": len(result.get("discovered_patterns", [])),
                "quality_score": result.get("quality_metrics", {}).get("overall_score", 0.0)
            }
            
        except Exception as e:
            print(f"  ❌ Enhanced features test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_orchestrator_integration(self):
        """Test orchestrator integration"""
        
        print("\n🔄 Test 3: Orchestrator Integration")
        
        try:
            from prompt_improver.ml.evaluation.structural_analyzer import EnhancedStructuralAnalyzer
            
            analyzer = EnhancedStructuralAnalyzer()
            
            # Test orchestrator interface
            config = {
                "text": "# Test\nThis is a test prompt.\n## Instructions\nPlease analyze this text.",
                "output_path": "./test_outputs/structural_analysis",
                "analysis_type": "enhanced",
                "enable_features": {
                    "semantic_analysis": False,  # Disable for faster testing
                    "graph_analysis": True,
                    "pattern_discovery": True,
                    "quality_assessment": True
                }
            }
            
            result = await analyzer.run_orchestrated_analysis(config)
            
            # Validate orchestrator compatibility
            orchestrator_success = (
                result.get("orchestrator_compatible") == True and
                "component_result" in result and
                "local_metadata" in result
            )
            
            print(f"  {'✅' if orchestrator_success else '❌'} Orchestrator Interface: {'PASSED' if orchestrator_success else 'FAILED'}")
            
            return {
                "success": orchestrator_success,
                "orchestrator_compatible": result.get("orchestrator_compatible", False),
                "has_component_result": "component_result" in result,
                "has_metadata": "local_metadata" in result,
                "execution_time": result.get("local_metadata", {}).get("execution_time", 0)
            }
            
        except Exception as e:
            print(f"  ❌ Orchestrator integration test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_performance(self):
        """Test performance and scalability"""
        
        print("\n⚡ Test 4: Performance")
        
        try:
            from prompt_improver.ml.evaluation.structural_analyzer import EnhancedStructuralAnalyzer
            
            analyzer = EnhancedStructuralAnalyzer()
            
            # Test with larger text
            large_text = """
# Large Document Test
""" + "\n".join([f"## Section {i}\nThis is section {i} with some content." for i in range(20)])
            
            import time
            start_time = time.time()
            
            result = await analyzer.analyze_enhanced_structure(large_text)
            
            execution_time = time.time() - start_time
            
            # Performance criteria
            performance_good = execution_time < 10.0  # Should complete within 10 seconds
            elements_detected = result["structural_elements"]["total_elements"]
            
            print(f"  {'✅' if performance_good else '⚠️'} Execution Time: {execution_time:.2f}s ({'GOOD' if performance_good else 'SLOW'})")
            print(f"  ✅ Elements Processed: {elements_detected}")
            
            return {
                "success": performance_good,
                "execution_time": execution_time,
                "elements_processed": elements_detected,
                "performance_rating": "good" if performance_good else "needs_optimization"
            }
            
        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_test_summary(self):
        """Generate overall test summary"""
        return {
            "total_tests": 4,
            "analyzer_version": "2025.1.0",
            "test_status": "Enhanced StructuralAnalyzer Testing Complete"
        }
    
    def _print_test_results(self, results):
        """Print comprehensive test results"""
        
        print("\n" + "=" * 60)
        print("📊 ENHANCED STRUCTURAL ANALYZER TEST RESULTS")
        print("=" * 60)
        
        # Print summary
        basic = results.get("basic_functionality", {})
        enhanced = results.get("enhanced_features", {})
        orchestrator = results.get("orchestrator_integration", {})
        performance = results.get("performance", {})
        
        basic_success = basic.get("success", False)
        enhanced_success = enhanced.get("success", False)
        orchestrator_success = orchestrator.get("success", False)
        performance_success = performance.get("success", False)
        
        print(f"✅ Basic Functionality: {'PASSED' if basic_success else 'FAILED'}")
        print(f"✅ Enhanced Features: {'PASSED' if enhanced_success else 'FAILED'}")
        print(f"✅ Orchestrator Integration: {'PASSED' if orchestrator_success else 'FAILED'}")
        print(f"✅ Performance: {'PASSED' if performance_success else 'FAILED'}")
        
        total_passed = sum([
            1 if basic_success else 0,
            1 if enhanced_success else 0,
            1 if orchestrator_success else 0,
            1 if performance_success else 0
        ])
        
        if total_passed == 4:
            print("\n🎉 ENHANCED STRUCTURAL ANALYZER: COMPLETE SUCCESS!")
            print("All 2025 enhancements are working correctly and ready for integration.")
        elif total_passed >= 3:
            print("\n✅ ENHANCED STRUCTURAL ANALYZER: MOSTLY SUCCESSFUL!")
            print("Core functionality working, minor issues to address.")
        else:
            print("\n⚠️ ENHANCED STRUCTURAL ANALYZER: NEEDS ATTENTION")
            print("Some core features require fixes before integration.")


async def main():
    """Main test execution function"""
    
    tester = EnhancedStructuralAnalyzerTester()
    results = await tester.run_comprehensive_test()
    
    # Save results to file
    with open('enhanced_structural_analyzer_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: enhanced_structural_analyzer_test_results.json")
    
    return 0 if results.get("basic_functionality", {}).get("success", False) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
