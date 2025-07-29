"""Synthetic Data Generation Modules

This package contains specialized generators for different types of synthetic data generation:

- statistical_generator: Traditional statistical methods with scikit-learn
- neural_generator: Neural network-based generation (VAE, basic neural approaches)  
- gan_generator: Advanced generative models (GANs, VAEs, Diffusion models)
- Base classes and utilities for generation

Decomposed from synthetic_data_generator.py (3,389 lines) into focused modules
for better maintainability and separation of concerns.
"""

from .statistical_generator import (
    GenerationMethodMetrics,
    MethodPerformanceTracker,
)

# Neural and GAN generators will be imported conditionally based on dependencies
try:
    from .neural_generator import NeuralSyntheticGenerator, DiffusionSyntheticGenerator
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from .gan_generator import (
        TabularGAN,
        TabularVAE, 
        TabularDiffusion,
        HybridGenerationSystem,
    )
    GAN_AVAILABLE = True
except ImportError:
    GAN_AVAILABLE = False

__all__ = [
    "GenerationMethodMetrics",
    "MethodPerformanceTracker",
    "NEURAL_AVAILABLE",
    "GAN_AVAILABLE",
]

if NEURAL_AVAILABLE:
    __all__.extend(["NeuralSyntheticGenerator", "DiffusionSyntheticGenerator"])

if GAN_AVAILABLE:
    __all__.extend([
        "TabularGAN",
        "TabularVAE",
        "TabularDiffusion", 
        "HybridGenerationSystem",
    ])