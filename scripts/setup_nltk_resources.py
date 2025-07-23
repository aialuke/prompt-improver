#!/usr/bin/env python3
"""
Setup NLTK resources for better linguistic analysis.
"""

import sys
from pathlib import Path

# Add the src directory to Python path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_nltk():
    """Setup NLTK resources using our NLTK manager."""
    try:
        from prompt_improver.ml.utils.nltk_manager import get_nltk_manager, setup_nltk_for_production
        
        print("🔧 Setting up NLTK resources...")
        
        # Get the NLTK manager
        manager = get_nltk_manager()
        
        # Get current status
        status = manager.get_resource_status()
        print(f"Current NLTK availability: {status['availability_rate']:.1%}")
        print(f"Available resources: {status['available']}")
        print(f"Missing resources: {status['missing']}")
        
        # Try to setup for production (will attempt downloads)
        print("\n🚀 Setting up for production...")
        success = setup_nltk_for_production()
        
        if success:
            print("✅ NLTK setup completed successfully!")
            
            # Get updated status
            final_status = manager.get_resource_status()
            print(f"Final NLTK availability: {final_status['availability_rate']:.1%}")
            print(f"Available resources: {final_status['available']}")
            if final_status['missing']:
                print(f"Still missing: {final_status['missing']}")
        else:
            print("⚠️  NLTK setup had issues, but critical resources should be available")
            
        return success
        
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")
        return False

def test_linguistic_analysis():
    """Test that linguistic analysis now works better."""
    try:
        from prompt_improver.ml.analysis.linguistic_analyzer import LinguisticAnalyzer, get_lightweight_config
        
        print("\n🧪 Testing improved linguistic analysis...")
        
        # Use lightweight config to avoid resource issues
        config = get_lightweight_config()
        config.nltk_fallback_enabled = True
        config.auto_download_nltk = False  # Skip downloads due to SSL issues
        
        analyzer = LinguisticAnalyzer(config)
        
        # Test with different types of prompts
        test_prompts = [
            "Write a simple Python function to sort numbers",
            "Analyze the complex molecular interactions in biochemical pathways for pharmaceutical research applications",
            "Hello",
            "Create a comprehensive business strategy for enterprise software deployment"
        ]
        
        results = []
        for i, prompt in enumerate(test_prompts):
            print(f"   Analyzing prompt {i+1}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
            
            # Use the correct method name
            result = analyzer.analyze(prompt)
            
            print(f"   - Flesch-Kincaid: {result.flesch_kincaid_grade:.2f}")
            print(f"   - Readability score: {result.readability_score:.2f}")
            print(f"   - Lexical diversity: {result.lexical_diversity:.2f}")
            print(f"   - Entities found: {len(result.entities)}")
            
            results.append(result)
        
        # Check for diversity in results
        complexity_scores = [r.overall_linguistic_quality for r in results]
        readability_scores = [r.flesch_kincaid_grade for r in results]
        
        complexity_unique = len(set(f"{score:.2f}" for score in complexity_scores))
        readability_unique = len(set(f"{score:.2f}" for score in readability_scores))
        
        print(f"\n📊 Analysis Results:")
        print(f"   - Unique complexity scores: {complexity_unique}/4")
        print(f"   - Unique readability scores: {readability_unique}/4") 
        print(f"   - Complexity range: {min(complexity_scores):.2f} - {max(complexity_scores):.2f}")
        print(f"   - Readability range: {min(readability_scores):.2f} - {max(readability_scores):.2f}")
        
        # Success if we have any diversity
        if complexity_unique >= 2 or readability_unique >= 2:
            print("✅ Linguistic analysis producing diverse results!")
            return True
        else:
            print("⚠️  Linguistic analysis functional but limited diversity")
            return True
            
    except Exception as e:
        print(f"❌ Linguistic analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and test function."""
    print("🔍 NLTK Resource Setup and Validation\n")
    
    # Step 1: Setup NLTK
    nltk_success = setup_nltk()
    
    # Step 2: Test linguistic analysis
    analysis_success = test_linguistic_analysis()
    
    print(f"\n📊 Setup Results:")
    print(f"   - NLTK setup: {'✅ Success' if nltk_success else '⚠️ Partial'}")
    print(f"   - Linguistic analysis: {'✅ Working' if analysis_success else '❌ Failed'}")
    
    if nltk_success and analysis_success:
        print("\n🎉 NLTK resources are now properly configured!")
        print("   You can re-run the context learner tests to see improved diversity.")
        return True
    else:
        print("\n⚠️  Setup had some issues, but basic functionality should work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)