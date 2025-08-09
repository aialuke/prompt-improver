"""Debug script to isolate the feature extraction issue"""
import asyncio
import sys
import traceback
import numpy as np
sys.path.append('/Users/lukemckenzie/prompt-improver/src')

class SimpleExtractor:
    """Simplified version to test the basic functionality"""

    def __init__(self):
        self.logger = self._get_logger()

    def _get_logger(self):
        import logging
        return logging.getLogger(__name__)

    async def simple_extract(self, training_data: dict) -> np.ndarray:
        """Simplified extraction without complex features"""
        features = training_data.get('features', [])
        if not features:
            self.logger.warning('No features found in training data')
            return np.array([])
        feature_matrix = np.array(features)
        self.logger.info('Extracted %s context feature vectors of dimension %s', feature_matrix.shape[0], feature_matrix.shape[1])
        return feature_matrix

async def test_simple_extraction():
    print('Creating simple extractor...')
    extractor = SimpleExtractor()
    print('Creating test data...')
    test_data = {'features': [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'labels': [0.5, 0.7, 0.6], 'metadata': {'total_samples': 3}}
    print('Testing extraction...')
    try:
        result = await extractor.simple_extract(test_data)
        print(f'Success: {result.shape}')
        print(f'Result: {result}')
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
if __name__ == '__main__':
    asyncio.run(test_simple_extraction())