"""
Test script for modernized ML components with 2025 best practices.

Tests both AdvancedDimensionalityReducer and ProductionSyntheticDataGenerator
with their new neural network, transformer, and diffusion model capabilities.
"""
import asyncio
import logging
import os
import sys
import numpy as np
from prompt_improver.ml.optimization.algorithms.dimensionality_reducer import AdvancedDimensionalityReducer, DimensionalityConfig
from prompt_improver.ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dimensionality_reducer():
    """Test the modernized AdvancedDimensionalityReducer."""
    logger.info('Testing AdvancedDimensionalityReducer with modern methods...')
    np.random.seed(42)
    X = np.random.randn(500, 31)
    y = np.random.randint(0, 3, 500)
    configs_to_test = [('Statistical Methods', DimensionalityConfig(target_dimensions=10, enable_neural_methods=False, fast_mode=False)), ('Neural Methods', DimensionalityConfig(target_dimensions=10, enable_neural_methods=True, neural_epochs=50, neural_batch_size=32)), ('Fast Mode', DimensionalityConfig(target_dimensions=10, fast_mode=True, use_randomized_svd=True))]
    results = {}
    for config_name, config in configs_to_test:
        try:
            logger.info('Testing configuration: %s', config_name)
            reducer = AdvancedDimensionalityReducer(config=config)
            result = asyncio.run(reducer.reduce_dimensions(X, y))
            results[config_name] = {'method': result.method, 'original_dims': result.original_dimensions, 'reduced_dims': result.reduced_dimensions, 'variance_preserved': result.variance_preserved, 'quality_score': result.quality_score, 'processing_time': result.processing_time}
            logger.info('✅ %s: %s - %s→%s dims, quality: %s, time: %ss', config_name, result.method, result.original_dimensions, result.reduced_dimensions, format(result.quality_score, '.3f'), format(result.processing_time, '.2f'))
        except Exception as e:
            logger.error('❌ {config_name} failed: %s', e)
            results[config_name] = {'error': str(e)}
    return results

async def test_synthetic_data_generator():
    """Test the modernized ProductionSyntheticDataGenerator."""
    logger.info('Testing ProductionSyntheticDataGenerator with modern methods...')
    generators_to_test = [('Statistical', ProductionSyntheticDataGenerator(target_samples=100, generation_method='statistical', use_enhanced_scoring=True)), ('Neural VAE', ProductionSyntheticDataGenerator(target_samples=100, generation_method='neural', neural_model_type='vae', neural_epochs=50, neural_batch_size=32)), ('Hybrid', ProductionSyntheticDataGenerator(target_samples=100, generation_method='hybrid', neural_model_type='vae', neural_epochs=30, neural_batch_size=32))]
    results = {}
    for generator_name, generator in generators_to_test:
        try:
            logger.info('Testing generator: %s', generator_name)
            data = await generator.generate_data()
            results[generator_name] = {'total_samples': data['metadata']['total_samples'], 'generation_method': data['metadata']['generation_method'], 'source': data['metadata']['source'], 'domain_distribution': data['metadata']['domain_distribution'], 'quality_assessment': data['metadata']['quality_assessment_type']}
            logger.info('✅ %s: Generated %s samples using %s method', generator_name, data['metadata']['total_samples'], data['metadata']['generation_method'])
        except Exception as e:
            logger.error('❌ {generator_name} failed: %s', e)
            results[generator_name] = {'error': str(e)}
    return results

def main():
    """Run all tests."""
    logger.info('🚀 Starting modernized ML components test suite...')
    logger.info('\n' + '=' * 60)
    logger.info('TESTING ADVANCED DIMENSIONALITY REDUCER')
    logger.info('=' * 60)
    dim_results = test_dimensionality_reducer()
    logger.info('\n' + '=' * 60)
    logger.info('TESTING PRODUCTION SYNTHETIC DATA GENERATOR')
    logger.info('=' * 60)
    syn_results = asyncio.run(test_synthetic_data_generator())
    logger.info('\n' + '=' * 60)
    logger.info('TEST SUMMARY')
    logger.info('=' * 60)
    logger.info('\nDimensionality Reducer Results:')
    for config, result in dim_results.items():
        if 'error' in result:
            logger.info('  ❌ {config}: %s', result['error'])
        else:
            logger.info('  ✅ %s: %s method, quality: %s', config, result['method'], format(result['quality_score'], '.3f'))
    logger.info('\nSynthetic Data Generator Results:')
    for generator, result in syn_results.items():
        if 'error' in result:
            logger.info('  ❌ {generator}: %s', result['error'])
        else:
            logger.info('  ✅ %s: %s samples, method: %s', generator, result['total_samples'], result['generation_method'])
    logger.info('\n🎉 Test suite completed!')
if __name__ == '__main__':
    main()
