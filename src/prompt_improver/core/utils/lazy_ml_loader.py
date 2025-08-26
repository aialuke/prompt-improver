"""Centralized lazy loading utility for ML dependencies.

Eliminates system-wide dependency contamination by providing centralized lazy loading
for all heavy ML dependencies including NumPy, Torch, SciPy, and Scikit-learn.

This prevents the 1,000ms+ startup penalty and 126MB memory overhead from loading
heavy ML libraries during package import.

Usage:
    from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_torch

    # Instead of: import numpy as np
    def my_function():
        np = get_numpy()
        return np.array([1, 2, 3])
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module cache to avoid repeated imports
_ML_MODULE_CACHE: dict[str, Any] = {}


def _lazy_import_with_fallback(module_name: str, fallback_message: str | None = None) -> Any:
    """Generic lazy import with fallback handling."""
    if module_name in _ML_MODULE_CACHE:
        return _ML_MODULE_CACHE[module_name]

    try:
        if module_name == "numpy":
            import numpy
            _ML_MODULE_CACHE[module_name] = numpy
            return numpy
        if module_name == "torch":
            import torch
            _ML_MODULE_CACHE[module_name] = torch
            return torch
        if module_name == "scipy":
            import scipy
            _ML_MODULE_CACHE[module_name] = scipy
            return scipy
        if module_name == "sklearn":
            import sklearn
            _ML_MODULE_CACHE[module_name] = sklearn
            return sklearn
        if module_name == "scipy.stats":
            from scipy import stats
            _ML_MODULE_CACHE[module_name] = stats
            return stats
        if module_name == "sklearn.utils":
            from sklearn import utils
            _ML_MODULE_CACHE[module_name] = utils
            return utils
        if module_name == "sklearn.metrics":
            from sklearn import metrics
            _ML_MODULE_CACHE[module_name] = metrics
            return metrics
        if module_name == "transformers":
            import transformers
            _ML_MODULE_CACHE[module_name] = transformers
            return transformers
        # Dynamic import for other modules
        import importlib
        module = importlib.import_module(module_name)
        _ML_MODULE_CACHE[module_name] = module
        return module

    except ImportError as e:
        error_msg = fallback_message or f"{module_name} not available: {e}"
        logger.warning(error_msg)

        # Return mock object to prevent runtime errors
        class MockModule:
            def __getattr__(self, name):
                raise ImportError(f"{module_name} is not installed. Install with: pip install {module_name}")

        mock = MockModule()
        _ML_MODULE_CACHE[module_name] = mock
        return mock


# Primary ML dependencies
def get_numpy():
    """Lazy load NumPy with fallback handling."""
    return _lazy_import_with_fallback(
        "numpy",
        "NumPy not available. Install with: pip install numpy"
    )


def get_torch():
    """Lazy load PyTorch with fallback handling."""
    return _lazy_import_with_fallback(
        "torch",
        "PyTorch not available. Install with: pip install torch"
    )


def get_scipy():
    """Lazy load SciPy with fallback handling."""
    return _lazy_import_with_fallback(
        "scipy",
        "SciPy not available. Install with: pip install scipy"
    )


def get_sklearn():
    """Lazy load Scikit-learn with fallback handling."""
    return _lazy_import_with_fallback(
        "sklearn",
        "Scikit-learn not available. Install with: pip install scikit-learn"
    )


# Specific submodules
def get_scipy_stats():
    """Lazy load scipy.stats with fallback handling."""
    return _lazy_import_with_fallback(
        "scipy.stats",
        "SciPy stats not available. Install with: pip install scipy"
    )


def get_sklearn_utils():
    """Lazy load sklearn.utils with fallback handling."""
    return _lazy_import_with_fallback(
        "sklearn.utils",
        "Scikit-learn utils not available. Install with: pip install scikit-learn"
    )


def get_sklearn_metrics():
    """Lazy load sklearn.metrics with fallback handling."""
    return _lazy_import_with_fallback(
        "sklearn.metrics",
        "Scikit-learn metrics not available. Install with: pip install scikit-learn"
    )


# Transformers and NLP libraries
def get_transformers():
    """Lazy load transformers library with fallback handling."""
    return _lazy_import_with_fallback(
        "transformers",
        "Transformers not available. Install with: pip install transformers"
    )


def get_transformers_components():
    """Lazy load specific transformers components."""
    transformers = get_transformers()
    if hasattr(transformers, 'AutoModel'):
        return {
            'AutoModel': transformers.AutoModel,
            'AutoTokenizer': transformers.AutoTokenizer,
            'pipeline': transformers.pipeline,
            'available': True
        }
    return {'available': False}


# Utilities for common patterns
def check_ml_availability() -> dict[str, bool]:
    """Check availability of all ML dependencies."""
    availability = {}

    for module_name in ["numpy", "torch", "scipy", "sklearn", "transformers"]:
        try:
            _lazy_import_with_fallback(module_name)
            # Check if it's not a mock
            module = _ML_MODULE_CACHE.get(module_name)
            availability[module_name] = module and not hasattr(module, '__getattr__')
        except:
            availability[module_name] = False

    return availability


def clear_ml_cache():
    """Clear the ML module cache (useful for testing)."""
    global _ML_MODULE_CACHE
    _ML_MODULE_CACHE.clear()


# Convenience functions for common numpy operations
def create_numpy_array(*args, **kwargs):
    """Create numpy array using lazy loading."""
    np = get_numpy()
    return np.array(*args, **kwargs)


def create_numpy_zeros(*args, **kwargs):
    """Create numpy zeros using lazy loading."""
    np = get_numpy()
    return np.zeros(*args, **kwargs)


def create_numpy_ones(*args, **kwargs):
    """Create numpy ones using lazy loading."""
    np = get_numpy()
    return np.ones(*args, **kwargs)
