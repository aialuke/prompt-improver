#!/usr/bin/env python3
"""Final verification that apriori_analyzer and context_learner are properly exposed."""

import asyncio
import logging
from src.prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
from src.prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def final_verification():
    """Final verification of the fix."""
    
    # Create orchestrator
    config = OrchestratorConfig()
    orchestrator = MLPipelineOrchestrator(config)
    
    # Initialize orchestrator
    logger.info("Initializing orchestrator...")
    await orchestrator.initialize()
    
    # Test apriori_analyzer
    logger.info("\n=== Testing apriori_analyzer ===")
    apriori_methods = orchestrator.get_component_methods("apriori_analyzer")
    if apriori_methods:
        logger.info(f"✅ SUCCESS: apriori_analyzer has {len(apriori_methods)} methods exposed")
        logger.info(f"   Methods: {', '.join(apriori_methods[:5])}...")
    else:
        logger.error("❌ FAIL: apriori_analyzer has no methods exposed")
    
    # Test context_learner  
    logger.info("\n=== Testing context_learner ===")
    context_methods = orchestrator.get_component_methods("context_learner")
    if context_methods:
        logger.info(f"✅ SUCCESS: context_learner has {len(context_methods)} methods exposed")
        logger.info(f"   Methods: {', '.join(context_methods[:5])}...")
    else:
        logger.error("❌ FAIL: context_learner has no methods exposed")
    
    # Test invoking a method
    logger.info("\n=== Testing method invocation ===")
    invocation_success = False
    
    try:
        # Test apriori_analyzer method - now it should work with mock data
        result = await orchestrator.invoke_component(
            "apriori_analyzer", 
            "analyze_patterns",
            window_days=7,
            save_to_database=False
        )
        # Check if we got a result dict (successful invocation)
        if isinstance(result, dict):
            logger.info("✅ SUCCESS: Successfully invoked apriori_analyzer.analyze_patterns")
            logger.info(f"   Result keys: {list(result.keys())[:5]}...")
            if 'error' in result:
                logger.warning(f"   Method returned error: {result['error']}")
            else:
                logger.info(f"   Analysis completed: {result.get('transaction_count', 0)} transactions processed")
                invocation_success = True
        else:
            logger.error(f"❌ FAIL: Unexpected result type: {type(result)}")
    except Exception as e:
        logger.error(f"❌ FAIL: Exception during invocation: {e}")
    
    # Final summary
    logger.info("\n=== FINAL SUMMARY ===")
    success = bool(apriori_methods) and bool(context_methods) and invocation_success
    if success:
        logger.info("✅ ALL TESTS PASSED: Both components are properly exposed and accessible!")
        logger.info("\nThe fix has been successfully implemented:")
        logger.info("1. Added ContextLearnerConnector to tier1_connectors.py")
        logger.info("2. Made db_manager optional in AprioriAnalyzer")
        logger.info("3. Fixed FeatureExtractionConfig compatibility in ContextLearner")
        logger.info("4. Added both components to core_components list for auto-initialization")
    else:
        logger.error("❌ TESTS FAILED: Components are still not properly exposed")
        
    await orchestrator.shutdown()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(final_verification())
    exit(0 if success else 1)