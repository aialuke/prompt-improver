"""GAN Data Generator Module

Advanced generative models for synthetic data generation including GANs, VAEs, and Diffusion models.
Extracted from synthetic_data_generator.py for focused functionality.

This module contains:
- TabularGAN: Enhanced GAN for tabular data (2025 best practices) 
- TabularVAE: Variational Autoencoder for tabular data
- TabularDiffusion: Diffusion model for tabular data (2025 best practice)
- HybridGenerationSystem: Combines multiple generation methods
"""
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from typing import TYPE_CHECKING
from prompt_improver.core.utils.lazy_ml_loader import get_numpy, get_sklearn, get_torch

if TYPE_CHECKING:
    from sklearn.datasets import make_classification
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
else:
    # Runtime lazy loading
    def _get_sklearn_imports():
        sklearn = get_sklearn()
        return sklearn.datasets.make_classification
    
    make_classification = _get_sklearn_imports()

logger = logging.getLogger(__name__)

# PyTorch imports with fallback
try:
    torch = get_torch()
    nn = torch.nn
    optim = torch.optim
    DataLoader = torch.utils.data.DataLoader
    TensorDataset = torch.utils.data.TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn('PyTorch not available. GAN/VAE/Diffusion models will be disabled. Install with: pip install torch')
try:
    from .statistical_generator import GenerationMethodMetrics, MethodPerformanceTracker
except ImportError:
    from dataclasses import dataclass
    from datetime import datetime

    @dataclass
    class GenerationMethodMetrics:
        method_name: str
        generation_time: float
        quality_score: float
        diversity_score: float
        memory_usage_mb: float
        success_rate: float
        samples_generated: int
        timestamp: datetime
        performance_gaps_addressed: dict[str, float]

    class MethodPerformanceTracker:

        def __init__(self):
            self.method_history = {}
            self.method_rankings = {}

        def record_performance(self, metrics):
            pass

        def get_best_method(self, gaps):
            return 'statistical'

class TabularGAN(nn.Module):
    """Enhanced Generative Adversarial Network for tabular data synthesis (2025 best practices)"""

    def __init__(self, data_dim: int, noise_dim: int=100, hidden_dims: list[int]=None):
        """Initialize TabularGAN
        
        Args:
            data_dim: Dimensionality of the data
            noise_dim: Dimensionality of noise input
            hidden_dims: Hidden layer dimensions for generator/discriminator
        """
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for GAN models')
        super().__init__()
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        gen_layers = []
        prev_dim = noise_dim
        for i, hidden_dim in enumerate(hidden_dims):
            gen_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim
        gen_layers.extend([nn.Linear(prev_dim, data_dim), nn.Tanh()])
        self.generator = nn.Sequential(*gen_layers)
        disc_layers = []
        prev_dim = data_dim
        for hidden_dim in reversed(hidden_dims):
            disc_layers.extend([nn.utils.spectral_norm(nn.Linear(prev_dim, hidden_dim)), nn.LeakyReLU(0.2), nn.Dropout(0.3)])
            prev_dim = hidden_dim
        disc_layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])
        self.discriminator = nn.Sequential(*disc_layers)

    def generate(self, batch_size: int, device: get_torch().device):
        """Generate synthetic samples"""
        noise = get_torch().randn(batch_size, self.noise_dim, device=device)
        return self.generator(noise)

    def discriminate(self, data: get_torch().Tensor):
        """Discriminate real vs fake data"""
        return self.discriminator(data)

class TabularVAE(nn.Module):
    """Variational Autoencoder for tabular data synthesis."""

    def __init__(self, data_dim: int, latent_dim: int=50, hidden_dims: list[int]=None, beta: float=1.0):
        """Initialize TabularVAE
        
        Args:
            data_dim: Dimensionality of the data
            latent_dim: Dimensionality of latent space
            hidden_dims: Hidden layer dimensions
            beta: Beta parameter for β-VAE regularization
        """
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for VAE models')
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.beta = beta
        if hidden_dims is None:
            hidden_dims = [128, 64]
        encoder_layers = []
        prev_dim = data_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, data_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return (mu, logvar)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = get_torch().exp(0.5 * logvar)
        eps = get_torch().randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent representation to data"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return (recon, mu, logvar, z)

    def generate(self, batch_size: int, device: get_torch().device):
        """Generate synthetic samples"""
        z = get_torch().randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)

    def loss_function(self, recon_x, x, mu, logvar):
        """Enhanced VAE loss with β-VAE regularization (2025 best practice)"""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * get_torch().sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * kl_loss, recon_loss, kl_loss)

class TabularDiffusion(nn.Module):
    """Diffusion model for tabular data synthesis (2025 best practice)"""

    def __init__(self, data_dim: int, timesteps: int=1000, hidden_dims: list[int]=None):
        """Initialize TabularDiffusion
        
        Args:
            data_dim: Dimensionality of the data
            timesteps: Number of diffusion timesteps
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for Diffusion models')
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps
        if hidden_dims is None:
            hidden_dims = [256, 512, 256]
        self.time_embed_dim = 128
        self.time_embedding = nn.Sequential(nn.Linear(1, self.time_embed_dim), nn.ReLU(), nn.Linear(self.time_embed_dim, self.time_embed_dim))
        layers = []
        prev_dim = data_dim + self.time_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GroupNorm(8, hidden_dim), nn.SiLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, data_dim))
        self.noise_predictor = nn.Sequential(*layers)
        self.register_buffer('betas', self._cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', get_torch().cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps: int, s: float=0.008):
        """Cosine beta schedule for improved training stability"""
        steps = timesteps + 1
        x = get_torch().linspace(0, timesteps, steps)
        alphas_cumprod = get_torch().cos((x / timesteps + s) / (1 + s) * get_torch().pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return get_torch().clip(betas, 0.0001, 0.9999)

    def forward(self, x, t):
        """Forward pass: predict noise given noisy data and timestep"""
        t_embed = self.time_embedding(t.float().unsqueeze(-1))
        x_t = get_torch().cat([x, t_embed], dim=-1)
        return self.noise_predictor(x_t)

    def add_noise(self, x, t, noise=None):
        """Add noise to data according to diffusion schedule"""
        if noise is None:
            noise = get_torch().randn_like(x)
        sqrt_alphas_cumprod = get_torch().sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = get_torch().sqrt(1.0 - self.alphas_cumprod[t])
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise

    def generate(self, batch_size: int, device: get_torch().device):
        """Generate samples using DDPM sampling"""
        x = get_torch().randn(batch_size, self.data_dim, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = get_torch().full((batch_size,), t, device=device, dtype=get_torch().long)
            predicted_noise = self.forward(x, t_batch)
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            if t > 0:
                noise = get_torch().randn_like(x)
            else:
                noise = get_torch().zeros_like(x)
            x = 1 / get_torch().sqrt(alpha) * (x - beta / get_torch().sqrt(1 - alpha_cumprod) * predicted_noise) + get_torch().sqrt(beta) * noise
        return x

class HybridGenerationSystem:
    """Hybrid generation system combining multiple methods (2025 best practice)"""

    def __init__(self, data_dim: int, device: str='auto'):
        """Initialize hybrid generation system
        
        Args:
            data_dim: Dimensionality of the data
            device: Device for computation ("auto", "cpu", "cuda")
        """
        self.data_dim = data_dim
        self.device = self._get_device(device)
        self.performance_tracker = MethodPerformanceTracker()
        self.methods = {}
        if TORCH_AVAILABLE:
            try:
                self.methods['gan'] = TabularGAN(data_dim).to(self.device)
                self.methods['vae'] = TabularVAE(data_dim).to(self.device)
                self.methods['diffusion'] = TabularDiffusion(data_dim).to(self.device)
            except Exception as e:
                logger.warning('Failed to initialize some neural methods: %s', e)
        self.method_weights = {'statistical': 0.3, 'gan': 0.25, 'vae': 0.25, 'diffusion': 0.2}

    def _get_device(self, device: str) -> get_torch().device:
        """Get appropriate device for computation"""
        if not TORCH_AVAILABLE:
            return None
        if device == 'auto':
            return get_torch().device('cuda' if get_torch().cuda.is_available() else 'cpu')
        return get_torch().device(device)

    async def generate_hybrid_data(self, batch_size: int, performance_gaps: dict[str, float], quality_threshold: float=0.7) -> dict[str, Any]:
        """Generate data using hybrid approach with quality filtering"""
        start_time = time.time()
        method_allocation = self._determine_method_allocation(performance_gaps, batch_size)
        generated_samples = []
        method_metrics = {}
        for method, sample_count in method_allocation.items():
            if sample_count == 0:
                continue
            method_start = time.time()
            try:
                if method == 'statistical':
                    samples = await self._generate_statistical_samples(sample_count)
                else:
                    samples = await self._generate_neural_samples(method, sample_count)
                quality_score = self._assess_sample_quality(samples)
                if quality_score >= quality_threshold:
                    generated_samples.extend(samples)
                method_time = time.time() - method_start
                method_metrics[method] = GenerationMethodMetrics(method_name=method, generation_time=method_time, quality_score=quality_score, diversity_score=self._calculate_diversity_score(samples), memory_usage_mb=self._get_memory_usage(), success_rate=1.0 if quality_score >= quality_threshold else 0.0, samples_generated=len(samples), timestamp=datetime.now(UTC), performance_gaps_addressed=performance_gaps)
                self.performance_tracker.record_performance(method_metrics[method])
            except Exception as e:
                logger.warning('Method {method} failed: %s', e)
                method_metrics[method] = GenerationMethodMetrics(method_name=method, generation_time=time.time() - method_start, quality_score=0.0, diversity_score=0.0, memory_usage_mb=self._get_memory_usage(), success_rate=0.0, samples_generated=0, timestamp=datetime.now(UTC), performance_gaps_addressed=performance_gaps)
        total_time = time.time() - start_time
        return {'samples': generated_samples, 'method_metrics': method_metrics, 'total_generation_time': total_time, 'method_allocation': method_allocation, 'quality_threshold': quality_threshold, 'samples_generated': len(generated_samples)}

    def _determine_method_allocation(self, performance_gaps: dict[str, float], total_samples: int) -> dict[str, int]:
        """Determine how many samples each method should generate"""
        best_method = self.performance_tracker.get_best_method(performance_gaps)
        adjusted_weights = self.method_weights.copy()
        if best_method in adjusted_weights:
            adjusted_weights[best_method] *= 1.5
        if performance_gaps.get('model_accuracy', 0) > 0.1:
            adjusted_weights['gan'] *= 1.3
            adjusted_weights['diffusion'] *= 1.2
        elif performance_gaps.get('diversity', 0) > 0.1:
            adjusted_weights['vae'] *= 1.4
        available_methods = {'statistical'}
        if TORCH_AVAILABLE and self.methods:
            available_methods.update(self.methods.keys())
        adjusted_weights = {k: v for k, v in adjusted_weights.items() if k in available_methods}
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            adjusted_weights = {'statistical': 1.0}
            total_weight = 1.0
        normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        allocation = {}
        for method, weight in normalized_weights.items():
            allocation[method] = int(total_samples * weight)
        allocated_total = sum(allocation.values())
        if allocated_total < total_samples:
            target_method = best_method if best_method in allocation else 'statistical'
            allocation[target_method] += total_samples - allocated_total
        return allocation

    async def _generate_statistical_samples(self, sample_count: int) -> list:
        """Generate samples using statistical methods"""
        try:
            X, _ = make_classification(n_samples=sample_count, n_features=self.data_dim, n_informative=self.data_dim, n_redundant=0, n_clusters_per_class=2, class_sep=0.8, random_state=42)
            return X.tolist()
        except Exception as e:
            logger.error('Statistical generation failed: %s', e)
            return get_numpy().random.randn(sample_count, self.data_dim).tolist()

    async def _generate_neural_samples(self, method: str, sample_count: int) -> list:
        """Generate samples using neural methods"""
        if not TORCH_AVAILABLE or method not in self.methods:
            raise ValueError(f'Neural method {method} not available')
        try:
            model = self.methods[method]
            model.eval()
            with get_torch().no_grad():
                synthetic_data = model.generate(sample_count, self.device)
                return synthetic_data.cpu().numpy().tolist()
        except Exception as e:
            logger.error('Neural generation with {method} failed: %s', e)
            return await self._generate_statistical_samples(sample_count)

    def _assess_sample_quality(self, samples: list) -> float:
        """Assess quality of generated samples"""
        if not samples:
            return 0.0
        try:
            data = get_numpy().array(samples)
            variances = get_numpy().var(data, axis=0)
            variance_quality = get_numpy().mean(variances > 0.01)
            means = get_numpy().mean(data, axis=0)
            mean_quality = get_numpy().mean(get_numpy().abs(means) < 5.0)
            return (variance_quality + mean_quality) / 2
        except Exception:
            return 0.5

    def _calculate_diversity_score(self, samples: list) -> float:
        """Calculate diversity score for samples"""
        if len(samples) < 2:
            return 0.0
        try:
            data = get_numpy().array(samples)
            n_samples = min(50, len(samples))
            subset = data[:n_samples]
            distances = []
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = get_numpy().linalg.norm(subset[i] - subset[j])
                    distances.append(dist)
            if not distances:
                return 0.0
            mean_distance = get_numpy().mean(distances)
            expected_distance = get_numpy().sqrt(self.data_dim)
            return min(1.0, mean_distance / expected_distance)
        except Exception:
            return 0.5

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0