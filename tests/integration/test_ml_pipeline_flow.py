#!/usr/bin/env python3
"""Test script to verify ML pipeline processes training data correctly"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prompt_improver.database.connection import get_session_context
from prompt_improver.learning.training_data_loader import TrainingDataLoader, get_training_data_stats
from prompt_improver.services.prompt_improvement import PromptImprovementService


async def test_ml_pipeline():
    """Test the entire ML pipeline data flow"""
    print("🔍 Testing ML Pipeline Data Flow\n")
    
    async with get_session_context() as session:
        # Step 1: Check training data availability
        print("1️⃣ Checking training data availability...")
        stats = await get_training_data_stats(session)
        
        print(f"   Real data samples: {stats['real_data']['total_samples']}")
        print(f"   Synthetic data samples: {stats['synthetic_data']['total_samples']}")
        print(f"   Combined total: {stats['combined']['total_samples']}")
        print(f"   Real data ratio: {stats['combined']['real_ratio']:.1%}\n")
        
        # Step 2: Load training data
        print("2️⃣ Loading training data with TrainingDataLoader...")
        loader = TrainingDataLoader(
            real_data_priority=True,
            min_samples=20,
            lookback_days=30,
            synthetic_ratio=0.3
        )
        
        training_data = await loader.load_training_data(session)
        
        print(f"   Loaded {len(training_data['features'])} training samples")
        print(f"   Feature dimensions: {len(training_data['features'][0]) if training_data['features'] else 0}")
        print(f"   Data valid: {training_data['validation']['is_valid']}")
        
        if training_data['validation']['warnings']:
            print(f"   ⚠️  Warnings: {', '.join(training_data['validation']['warnings'])}")
        print()
        
        # Step 3: Initialize prompt improvement service
        print("3️⃣ Initializing PromptImprovementService...")
        service = PromptImprovementService(
            enable_bandit_optimization=True,
            enable_automl=True
        )
        print("   Service initialized with ML optimization enabled\n")
        
        # Step 4: Run ML optimization
        print("4️⃣ Running ML optimization...")
        print("   Note: This will use Optuna for hyperparameter tuning")
        print("   Expected processing time: 30-120 seconds\n")
        
        try:
            result = await service.run_ml_optimization(
                rule_ids=None,  # Optimize all rules
                db_session=session
            )
            
            print("5️⃣ ML Optimization Results:")
            print(f"   Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'success':
                print(f"   ✅ Model ID: {result.get('model_id', 'N/A')}")
                print(f"   ✅ Best Score: {result.get('best_score', 0):.3f}")
                print(f"   ✅ Accuracy: {result.get('accuracy', 0):.3f}")
                print(f"   ✅ Training Samples Used: {result.get('training_samples', 0)}")
                print(f"   ✅ Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
                
                if 'feature_importance' in result:
                    print("\n   📊 Feature Importance:")
                    for feature, importance in result['feature_importance'].items():
                        print(f"      {feature}: {importance:.3f}")
                
                if 'ensemble' in result:
                    print(f"\n   🎯 Ensemble Score: {result['ensemble'].get('ensemble_score', 0):.3f}")
                
                print("\n✨ ML Pipeline successfully processed training data!")
                
            elif result.get('status') == 'insufficient_data':
                print(f"   ⚠️  {result.get('message', 'Insufficient data')}")
                print(f"   Total samples: {result.get('samples_found', 0)}")
                print(f"   Real samples: {result.get('real_samples', 0)}")
                print(f"   Synthetic samples: {result.get('synthetic_samples', 0)}")
                
                if result.get('warnings'):
                    print(f"\n   Warnings:")
                    for warning in result['warnings']:
                        print(f"   - {warning}")
                        
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                print(f"   Message: {result.get('message', '')}")
                
        except Exception as e:
            print(f"   ❌ Exception during ML optimization: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n🏁 ML Pipeline test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("ML Pipeline Data Flow Test")
    print("=" * 60)
    asyncio.run(test_ml_pipeline())