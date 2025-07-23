# ML Component Complexity Reduction - Migration Guide

## Overview

This guide documents the complexity reduction refactoring of ML learning components, providing migration paths from the original monolithic classes to the new specialized components.

## Architecture Changes

### Before: Monolithic Design
```
ContextLearner (3,127 lines)
├── Feature extraction (linguistic, domain, context)
├── Clustering algorithms (HDBSCAN, K-means)
├── Pattern analysis
├── Caching and performance monitoring
└── Database operations
```

### After: Specialized Components
```
CompositeFeatureExtractor
├── LinguisticFeatureExtractor (10 features)
├── DomainFeatureExtractor (15 features)
└── ContextFeatureExtractor (20 features)

ContextClusteringEngine
├── HDBSCAN clustering
├── K-means clustering
└── Quality metrics

ContextLearner (orchestration only)
├── Uses CompositeFeatureExtractor
├── Uses ContextClusteringEngine
└── Focuses on learning logic
```

## Migration Steps

### 1. Update Imports

**Updated (Current):**
```python
# Clean imports - no legacy names
from prompt_improver.ml.learning.algorithms import ContextLearner, ContextConfig
from prompt_improver.ml.learning.features import CompositeFeatureExtractor, FeatureExtractionConfig
from prompt_improver.ml.learning.clustering import ContextClusteringEngine, ClusteringConfig
```

### 2. Update Initialization

**Old:**
```python
learner = ContextSpecificLearner(config=context_config)
```

**New:**
```python
config = ContextConfig(
    enable_linguistic_features=True,
    enable_domain_features=True,
    enable_context_features=True,
    use_advanced_clustering=True
)
learner = ContextLearner(config)
```

### 3. Update Feature Extraction

**Old (embedded in ContextLearner):**
```python
# Features were extracted internally with complex methods
features = learner._extract_linguistic_features(text)
domain_features = learner._extract_domain_features(text)
```

**New (specialized extractors):**
```python
# Use individual extractors
linguistic_extractor = LinguisticFeatureExtractor()
domain_extractor = DomainFeatureExtractor()

linguistic_features = linguistic_extractor.extract_features(text)
domain_features = domain_extractor.extract_features(text)

# Or use composite extractor
composite_extractor = CompositeFeatureExtractor()
result = composite_extractor.extract_features(text, context_data)
all_features = result['features']
```

### 4. Update Clustering

**Old (embedded in ContextLearner):**
```python
# Clustering was done internally
cluster_result = learner._cluster_contexts_hdbscan(features)
```

**New (specialized engine):**
```python
clustering_engine = ContextClusteringEngine()
cluster_result = await clustering_engine.cluster_contexts(features)
```

### 5. Update Learning Workflow

**Old:**
```python
# Complex internal workflow
results = await learner.learn_context_patterns(training_data)
```

**New:**
```python
# Simplified orchestration
result = await learner.learn_from_data(training_data)
patterns = learner.get_context_patterns()
cluster_info = learner.get_cluster_info()
```

## Configuration Migration

### Feature Extraction Configuration

**Old (scattered throughout code):**
```python
# Configuration was hardcoded in methods
enable_linguistic = True
linguistic_weight = 1.0
cache_enabled = True
```

**New (centralized configuration):**
```python
feature_config = FeatureExtractionConfig(
    enable_linguistic=True,
    enable_domain=True,
    enable_context=True,
    linguistic_weight=1.0,
    domain_weight=1.0,
    context_weight=1.0,
    cache_enabled=True,
    deterministic=True
)
```

### Clustering Configuration

**Old (hardcoded parameters):**
```python
# Parameters scattered in clustering methods
min_cluster_size = 5
cluster_selection_epsilon = 0.1
```

**New (structured configuration):**
```python
clustering_config = ClusteringConfig(
    use_advanced_clustering=True,
    hdbscan_min_cluster_size=5,
    hdbscan_cluster_selection_epsilon=0.1,
    kmeans_max_clusters=8,
    min_silhouette_score=0.3
)
```

## Benefits Realized

### 1. Reduced Complexity
- **ContextLearner**: 3,127 lines → **ContextLearner (new)**: 300 lines
- **Cyclomatic Complexity**: Reduced from 15-25+ to 5-10 per method
- **Single Responsibility**: Each component has one clear purpose

### 2. Improved Testability
- **Individual Components**: Can be tested in isolation
- **Mock-friendly**: Easy to mock dependencies
- **Focused Tests**: Each test targets specific functionality

### 3. Enhanced Maintainability
- **Clear Interfaces**: Well-defined component boundaries
- **Easier Debugging**: Issues isolated to specific components
- **Better Documentation**: Each component has clear purpose

### 4. Performance Improvements
- **Specialized Caching**: Component-specific cache strategies
- **Parallel Processing**: Independent extractors can run concurrently
- **Memory Efficiency**: Smaller objects with focused responsibilities

## Testing the Migration

### Run Component Tests
```bash
# Test individual components
pytest tests/unit/ml/learning/test_refactored_components.py::TestLinguisticFeatureExtractor -v
pytest tests/unit/ml/learning/test_refactored_components.py::TestDomainFeatureExtractor -v
pytest tests/unit/ml/learning/test_refactored_components.py::TestContextClusteringEngine -v

# Test integrated workflow
pytest tests/unit/ml/learning/test_refactored_components.py::TestIntegratedWorkflow -v
```

### Performance Comparison
```python
import time
from prompt_improver.ml.learning.algorithms.context_learner import ContextSpecificLearner
from prompt_improver.ml.learning.algorithms.context_learner import ContextLearner

# Compare initialization time
start = time.time()
old_learner = ContextSpecificLearner()
old_init_time = time.time() - start

start = time.time()
new_learner = ContextLearner()
new_init_time = time.time() - start

print(f"Old initialization: {old_init_time:.3f}s")
print(f"New initialization: {new_init_time:.3f}s")
print(f"Improvement: {((old_init_time - new_init_time) / old_init_time * 100):.1f}%")
```

## Backward Compatibility

### Gradual Migration Strategy

1. **Phase 1**: Deploy new components alongside existing ones
2. **Phase 2**: Update new code to use refactored components
3. **Phase 3**: Migrate existing code module by module
4. **Phase 4**: Deprecate old monolithic classes

### Compatibility Layer (Optional)
```python
class ContextLearnerAdapter:
    """Adapter to maintain compatibility with old interface."""
    
    def __init__(self, config=None):
        # Map old config to new config
        new_config = self._map_config(config)
        self.context_learner = ContextLearner(new_config)

    async def learn_context_patterns(self, training_data):
        """Old interface method."""
        result = await self.context_learner.learn_from_data(training_data)
        # Map new result format to old format
        return self._map_result(result)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure new modules are in Python path
   - Check for circular import dependencies

2. **Configuration Mismatches**
   - Verify feature extraction configuration
   - Check clustering parameter mappings

3. **Performance Differences**
   - New components may have different caching behavior
   - Clustering algorithms may produce different results

### Debug Mode
```python
import logging
logging.getLogger('prompt_improver.ml.learning').setLevel(logging.DEBUG)

# Enable detailed logging for migration debugging
learner = ContextLearner()
result = await learner.learn_from_data(training_data)
```

## Next Steps

1. **Run Tests**: Execute the test suite to verify functionality
2. **Performance Benchmarks**: Compare performance with original implementation
3. **Gradual Rollout**: Start with non-critical components
4. **Monitor Metrics**: Track performance and quality metrics
5. **Team Training**: Educate team on new architecture

## Support

For questions or issues during migration:
- Review component documentation
- Check test examples for usage patterns
- Consult the refactored code for implementation details
