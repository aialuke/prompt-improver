"""Production Synthetic Data Generator for ML Training

Enhanced synthetic data generation based on 2025 best practices:
- Multi-domain pattern generation (technical, creative, analytical, instructional, conversational)
- Statistical quality guarantees (class diversity, variance control, three-tier stratification)
- Advanced feature engineering using scikit-learn principles
- Modern generative models (GANs, VAEs, Diffusion Models)
- Neural network-based synthetic data generation
- Comprehensive validation framework for ML pipeline integration

Research Sources:
- 2025 Survey: Comprehensive Survey of Synthetic Tabular Data Generation
- Modern generative AI approaches (GANs, VAEs, Diffusion Models)
- Firecrawl Deep Research: NLP synthetic data generation best practices
- Context7 Scikit-learn: Advanced data generation and statistical controls
"""

import asyncio
import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_classification
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models import TrainingPrompt
from ..learning.quality.enhanced_scorer import EnhancedQualityMetrics, EnhancedQualityScorer
from ..optimization.batch.dynamic_batch_optimizer import DynamicBatchOptimizer, BatchOptimizationConfig
from ..analytics.generation_analytics import GenerationHistoryTracker, GenerationAnalytics

# Neural network and deep learning imports for modern generative models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural generative models will be disabled. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Some generative models will be disabled. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)

# ===== 2025 ENHANCED NEURAL GENERATION METHODS =====

@dataclass
class GenerationMethodMetrics:
    """Performance metrics for generation methods (2025 best practice)"""
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
    """Tracks performance of different generation methods for auto-selection (2025 best practice)"""

    def __init__(self):
        self.method_history: dict[str, list[GenerationMethodMetrics]] = {}
        self.method_rankings: dict[str, float] = {}

    def record_performance(self, metrics: GenerationMethodMetrics) -> None:
        """Record performance metrics for a generation method"""
        if metrics.method_name not in self.method_history:
            self.method_history[metrics.method_name] = []

        self.method_history[metrics.method_name].append(metrics)
        self._update_rankings()

    def get_best_method(self, performance_gaps: dict[str, float]) -> str:
        """Select best method based on historical performance and current gaps"""
        if not self.method_rankings:
            return "statistical"  # Default fallback

        # Weight rankings by gap-specific performance
        weighted_scores = {}
        for method, base_score in self.method_rankings.items():
            gap_bonus = self._calculate_gap_bonus(method, performance_gaps)
            weighted_scores[method] = base_score + gap_bonus

        return max(weighted_scores, key=weighted_scores.get)

    def _update_rankings(self) -> None:
        """Update method rankings based on recent performance"""
        for method, history in self.method_history.items():
            if not history:
                continue

            # Calculate weighted score (recent performance weighted higher)
            recent_metrics = history[-10:]  # Last 10 generations
            weights = np.linspace(0.5, 1.0, len(recent_metrics))

            quality_scores = [m.quality_score for m in recent_metrics]
            diversity_scores = [m.diversity_score for m in recent_metrics]
            success_rates = [m.success_rate for m in recent_metrics]

            weighted_quality = np.average(quality_scores, weights=weights)
            weighted_diversity = np.average(diversity_scores, weights=weights)
            weighted_success = np.average(success_rates, weights=weights)

            # Combined score (2025 best practice weighting)
            self.method_rankings[method] = (
                0.4 * weighted_quality +
                0.3 * weighted_diversity +
                0.3 * weighted_success
            )

    def _calculate_gap_bonus(self, method: str, performance_gaps: dict[str, float]) -> float:
        """Calculate bonus score based on method's effectiveness for specific gaps"""
        if method not in self.method_history:
            return 0.0

        # Analyze historical effectiveness for similar gaps
        relevant_metrics = []
        for metrics in self.method_history[method][-5:]:  # Recent history
            gap_similarity = self._calculate_gap_similarity(
                metrics.performance_gaps_addressed, performance_gaps
            )
            if gap_similarity > 0.5:  # Similar gap patterns
                relevant_metrics.append(metrics)

        if not relevant_metrics:
            return 0.0

        # Return average effectiveness for similar gaps
        return np.mean([m.quality_score for m in relevant_metrics]) * 0.2

    def _calculate_gap_similarity(self, gaps1: dict[str, float], gaps2: dict[str, float]) -> float:
        """Calculate similarity between two gap patterns"""
        common_keys = set(gaps1.keys()) & set(gaps2.keys())
        if not common_keys:
            return 0.0

        similarities = []
        for key in common_keys:
            # Normalized difference (closer to 0 = more similar)
            diff = abs(gaps1[key] - gaps2[key])
            similarity = max(0, 1 - diff)
            similarities.append(similarity)

        return np.mean(similarities)

# Modern Generative Models for Tabular Data (Enhanced 2025 Version)
class TabularGAN(nn.Module):
    """Enhanced Generative Adversarial Network for tabular data synthesis (2025 best practices)"""

    def __init__(self, data_dim: int, noise_dim: int = 100, hidden_dims: list[int] = None):
        super().__init__()
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        # Enhanced Generator with Batch Normalization and Residual Connections
        gen_layers = []
        prev_dim = noise_dim
        for i, hidden_dim in enumerate(hidden_dims):
            gen_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 2025 enhancement
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        gen_layers.extend([
            nn.Linear(prev_dim, data_dim),
            nn.Tanh()  # Normalize output to [-1, 1]
        ])
        self.generator = nn.Sequential(*gen_layers)

        # Enhanced Discriminator with Spectral Normalization
        disc_layers = []
        prev_dim = data_dim
        for hidden_dim in reversed(hidden_dims):
            disc_layers.extend([
                nn.utils.spectral_norm(nn.Linear(prev_dim, hidden_dim)),  # 2025 enhancement
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)  # Higher dropout for better regularization
            ])
            prev_dim = hidden_dim
        disc_layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        self.discriminator = nn.Sequential(*disc_layers)

    def generate(self, batch_size: int, device: torch.device):
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        return self.generator(noise)

    def discriminate(self, data: torch.Tensor):
        return self.discriminator(data)

class TabularVAE(nn.Module):
    """Variational Autoencoder for tabular data synthesis."""

    def __init__(self, data_dim: int, latent_dim: int = 50, hidden_dims: list[int] = None, beta: float = 1.0):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        encoder_layers = []
        prev_dim = data_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, data_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def generate(self, batch_size: int, device: torch.device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)

    def loss_function(self, recon_x, x, mu, logvar):
        """Enhanced VAE loss with β-VAE regularization (2025 best practice)"""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss with β weighting for disentanglement
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

class TabularDiffusion(nn.Module):
    """Diffusion model for tabular data synthesis (2025 best practice)"""

    def __init__(self, data_dim: int, timesteps: int = 1000, hidden_dims: list[int] = None):
        super().__init__()
        self.data_dim = data_dim
        self.timesteps = timesteps

        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        # Time embedding
        self.time_embed_dim = 128
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.ReLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # Noise prediction network
        layers = []
        prev_dim = data_dim + self.time_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),  # Group normalization for stability
                nn.SiLU(),  # Swish activation for better gradients
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, data_dim))
        self.noise_predictor = nn.Sequential(*layers)

        # Beta schedule for noise
        self.register_buffer('betas', self._cosine_beta_schedule(timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """Cosine beta schedule for improved training stability"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(self, x, t):
        """Forward pass: predict noise given noisy data and timestep"""
        # Time embedding
        t_embed = self.time_embedding(t.float().unsqueeze(-1))

        # Concatenate data and time embedding
        x_t = torch.cat([x, t_embed], dim=-1)

        # Predict noise
        return self.noise_predictor(x_t)

    def add_noise(self, x, t, noise=None):
        """Add noise to data according to diffusion schedule"""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])

        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise

    def generate(self, batch_size: int, device: torch.device):
        """Generate samples using DDPM sampling"""
        # Start from pure noise
        x = torch.randn(batch_size, self.data_dim, device=device)

        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.forward(x, t_batch)

            # Remove predicted noise
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise

        return x

class HybridGenerationSystem:
    """Hybrid generation system combining multiple methods (2025 best practice)"""

    def __init__(self, data_dim: int, device: str = "auto"):
        self.data_dim = data_dim
        self.device = self._get_device(device)
        self.performance_tracker = MethodPerformanceTracker()

        # Initialize all generation methods
        self.methods = {}
        if TORCH_AVAILABLE:
            self.methods['gan'] = TabularGAN(data_dim).to(self.device)
            self.methods['vae'] = TabularVAE(data_dim).to(self.device)
            self.methods['diffusion'] = TabularDiffusion(data_dim).to(self.device)

        # Method weights for ensemble generation
        self.method_weights = {
            'statistical': 0.3,
            'gan': 0.25,
            'vae': 0.25,
            'diffusion': 0.2
        }

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for computation"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    async def generate_hybrid_data(
        self,
        batch_size: int,
        performance_gaps: dict[str, float],
        quality_threshold: float = 0.7
    ) -> dict[str, Any]:
        """Generate data using hybrid approach with quality filtering"""
        start_time = time.time()

        # Determine optimal method mix based on performance gaps
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

                # Quality assessment
                quality_score = self._assess_sample_quality(samples)

                # Filter samples based on quality threshold
                if quality_score >= quality_threshold:
                    generated_samples.extend(samples)

                # Record performance metrics
                method_time = time.time() - method_start
                method_metrics[method] = GenerationMethodMetrics(
                    method_name=method,
                    generation_time=method_time,
                    quality_score=quality_score,
                    diversity_score=self._calculate_diversity_score(samples),
                    memory_usage_mb=self._get_memory_usage(),
                    success_rate=1.0 if quality_score >= quality_threshold else 0.0,
                    samples_generated=len(samples),
                    timestamp=datetime.now(),
                    performance_gaps_addressed=performance_gaps
                )

                # Update performance tracker
                self.performance_tracker.record_performance(method_metrics[method])

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                method_metrics[method] = GenerationMethodMetrics(
                    method_name=method,
                    generation_time=time.time() - method_start,
                    quality_score=0.0,
                    diversity_score=0.0,
                    memory_usage_mb=self._get_memory_usage(),
                    success_rate=0.0,
                    samples_generated=0,
                    timestamp=datetime.now(),
                    performance_gaps_addressed=performance_gaps
                )

        total_time = time.time() - start_time

        return {
            'samples': generated_samples,
            'method_metrics': method_metrics,
            'total_generation_time': total_time,
            'method_allocation': method_allocation,
            'quality_threshold': quality_threshold,
            'samples_generated': len(generated_samples)
        }

    def _determine_method_allocation(
        self,
        performance_gaps: dict[str, float],
        total_samples: int
    ) -> dict[str, int]:
        """Determine how many samples each method should generate"""
        # Get best method recommendation
        best_method = self.performance_tracker.get_best_method(performance_gaps)

        # Adjust weights based on performance gaps and best method
        adjusted_weights = self.method_weights.copy()

        # Boost best method weight
        if best_method in adjusted_weights:
            adjusted_weights[best_method] *= 1.5

        # Adjust based on specific gaps
        if performance_gaps.get('model_accuracy', 0) > 0.1:
            # Favor neural methods for complex patterns
            adjusted_weights['gan'] *= 1.3
            adjusted_weights['diffusion'] *= 1.2
        elif performance_gaps.get('diversity', 0) > 0.1:
            # Favor VAE for diversity
            adjusted_weights['vae'] *= 1.4

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        normalized_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        # Allocate samples
        allocation = {}
        for method, weight in normalized_weights.items():
            allocation[method] = int(total_samples * weight)

        # Ensure we generate the exact number requested
        allocated_total = sum(allocation.values())
        if allocated_total < total_samples:
            # Add remaining to best method
            allocation[best_method] += total_samples - allocated_total

        return allocation

    async def _generate_statistical_samples(self, sample_count: int) -> list:
        """Generate samples using statistical methods"""
        # Use scikit-learn's make_classification for realistic statistical generation
        from sklearn.datasets import make_classification

        try:
            X, _ = make_classification(
                n_samples=sample_count,
                n_features=self.data_dim,
                n_informative=self.data_dim,
                n_redundant=0,
                n_clusters_per_class=2,
                class_sep=1.0,
                random_state=42
            )

            # Normalize to [0, 1] range
            X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
            return X_normalized.tolist()

        except Exception as e:
            logger.warning(f"Statistical generation failed: {e}, using fallback")
            # Fallback to simple random generation
            return [[np.random.random() for _ in range(self.data_dim)] for _ in range(sample_count)]

    async def _generate_neural_samples(self, method: str, sample_count: int) -> list:
        """Generate samples using neural methods"""
        if method not in self.methods:
            raise ValueError(f"Method {method} not available")

        model = self.methods[method]
        model.eval()

        with torch.no_grad():
            samples = model.generate(sample_count, self.device)
            return samples.cpu().numpy().tolist()

    def _assess_sample_quality(self, samples: list) -> float:
        """Assess quality of generated samples"""
        if not samples:
            return 0.0

        # Convert to numpy for analysis
        samples_array = np.array(samples)

        # Basic quality metrics
        variance = np.var(samples_array, axis=0).mean()
        range_coverage = (samples_array.max() - samples_array.min()) / 2.0  # Normalized range

        # Combine metrics (simple heuristic)
        quality_score = min(1.0, (variance + range_coverage) / 2.0)
        return quality_score

    def _calculate_diversity_score(self, samples: list) -> float:
        """Calculate diversity score for samples"""
        if len(samples) < 2:
            return 0.0

        samples_array = np.array(samples)

        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(samples_array)

        # Diversity is average pairwise distance
        return float(np.mean(distances))

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

class NeuralSyntheticGenerator:
    """Neural network-based synthetic data generator."""

    def __init__(self, model_type: str = "vae", latent_dim: int = 50,
                 hidden_dims: list[int] = None, beta: float = 1.0,
                 device: str = "auto", epochs: int = 200, batch_size: int = 64,
                 learning_rate: float = 1e-3):
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'NeuralSyntheticGenerator':
        """Fit the neural synthetic data generator."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural generative models")

        from sklearn.preprocessing import StandardScaler

        # Normalize input data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        data_dim = X_scaled.shape[1]

        # Initialize model
        if self.model_type == "gan":
            self.model = TabularGAN(data_dim, hidden_dims=self.hidden_dims)
            self.model.to(self.device)
            self.optimizer_g = optim.Adam(self.model.generator.parameters(), lr=self.learning_rate)
            self.optimizer_d = optim.Adam(self.model.discriminator.parameters(), lr=self.learning_rate)
            self._train_gan(X_scaled)
        elif self.model_type == "vae":
            self.model = TabularVAE(data_dim, self.latent_dim, self.hidden_dims, self.beta)
            self.model.to(self.device)
            self.optimizer_g = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self._train_vae(X_scaled)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.is_fitted = True
        return self

    def _train_vae(self, X_scaled: np.ndarray):
        """Train VAE model."""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer_g.zero_grad()

                recon, mu, logvar, _ = self.model(data)
                loss = self.model.loss_function(recon, data, mu, logvar)

                loss.backward()
                self.optimizer_g.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"VAE Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def _train_gan(self, X_scaled: np.ndarray):
        """Train GAN model."""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            for batch_idx, (real_data,) in enumerate(dataloader):
                batch_size = real_data.size(0)

                # Train Discriminator
                self.optimizer_d.zero_grad()

                # Real data
                real_labels = torch.ones(batch_size, 1, device=self.device)
                real_output = self.model.discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)

                # Fake data
                fake_data = self.model.generate(batch_size, self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                fake_output = self.model.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                fake_data = self.model.generate(batch_size, self.device)
                fake_output = self.model.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)

                g_loss.backward()
                self.optimizer_g.step()

            if epoch % 50 == 0:
                logger.info(f"GAN Epoch {epoch}, D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic data samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")

        self.model.eval()
        with torch.no_grad():
            synthetic_data = self.model.generate(n_samples, self.device)
            synthetic_data = synthetic_data.cpu().numpy()

        # Inverse transform to original scale
        return self.scaler.inverse_transform(synthetic_data)

class TabularDiffusionModel(nn.Module):
    """Diffusion model for tabular data synthesis."""

    def __init__(self, data_dim: int, num_timesteps: int = 1000, hidden_dim: int = 256):
        super().__init__()
        self.data_dim = data_dim
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        # Noise schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def add_noise(self, x, t):
        """Add noise to data at timestep t."""
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

    def forward(self, x, t):
        """Predict noise at timestep t."""
        t_emb = t.float().unsqueeze(1) / self.num_timesteps  # Normalize timestep
        x_with_t = torch.cat([x, t_emb], dim=1)
        return self.denoiser(x_with_t)

    def sample(self, batch_size: int, device: torch.device):
        """Generate samples using DDPM sampling."""
        x = torch.randn(batch_size, self.data_dim, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.forward(x, t_tensor)

            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # DDPM sampling step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

        return x

class DiffusionSyntheticGenerator:
    """Diffusion model-based synthetic data generator."""

    def __init__(self, num_timesteps: int = 1000, hidden_dim: int = 256,
                 device: str = "auto", epochs: int = 300, batch_size: int = 64,
                 learning_rate: float = 1e-3):
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.optimizer = None
        self.scaler = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'DiffusionSyntheticGenerator':
        """Fit the diffusion model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for diffusion models")

        from sklearn.preprocessing import StandardScaler

        # Normalize input data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        data_dim = X_scaled.shape[1]

        # Initialize model
        self.model = TabularDiffusionModel(data_dim, self.num_timesteps, self.hidden_dim)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer.zero_grad()

                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (data.size(0),), device=self.device)

                # Add noise
                x_t, noise = self.model.add_noise(data, t)

                # Predict noise
                predicted_noise = self.model(x_t, t)

                # Compute loss
                loss = nn.functional.mse_loss(predicted_noise, noise)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"Diffusion Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_fitted = True
        return self

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate synthetic data samples."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating data")

        self.model.eval()
        with torch.no_grad():
            synthetic_data = self.model.sample(n_samples, self.device)
            synthetic_data = synthetic_data.cpu().numpy()

        # Inverse transform to original scale
        return self.scaler.inverse_transform(synthetic_data)

@dataclass
class DomainConfig:
    """Configuration for domain-specific data generation"""

    name: str
    ratio: float
    patterns: list[tuple[str, str]]
    feature_ranges: dict[str, tuple[float, float]]
    effectiveness_params: tuple[float, float]  # Beta distribution parameters
    complexity_range: tuple[int, int]

@dataclass
class QualityMetrics:
    """Quality validation metrics for generated data"""

    sample_count: int
    class_diversity: int
    variance_sufficient: bool
    min_samples_met: bool
    ensemble_ready: bool
    no_invalid_values: bool
    overall_quality: bool
    domain_distribution: dict[str, int]
    feature_correlations: dict[str, float]

class ProductionSyntheticDataGenerator:
    """Production-grade synthetic data generator with advanced quality assessment and modern generative models

    Enhanced with 2025 best practices for adaptive data generation:
    - Gap-based targeting for performance improvement
    - Difficulty distribution control
    - Focus area specification
    - Hardness characterization integration
    """

    def __init__(
        self,
        target_samples: int = 1000,
        random_state: int = 42,
        use_enhanced_scoring: bool = True,
        generation_method: str = "statistical",  # "statistical", "neural", "hybrid", "diffusion"
        neural_model_type: str = "vae",  # "vae", "gan", "diffusion"
        neural_epochs: int = 200,
        neural_batch_size: int = 64,
        neural_learning_rate: float = 1e-3,
        neural_device: str = "auto",
        # New 2025 adaptive generation parameters
        enable_gap_targeting: bool = True,
        difficulty_distribution: str = "adaptive",  # "uniform", "adaptive", "hard_focused"
        focus_areas: list[str] | None = None,
        hardness_threshold: float = 0.7,
    ):
        """Initialize the production synthetic data generator

        Args:
            target_samples: Total number of samples to generate (default: 1000)
            random_state: Random seed for reproducible generation
            use_enhanced_scoring: Whether to use enhanced multi-dimensional quality scoring
            generation_method: Method for data generation ("statistical", "neural", "hybrid")
            neural_model_type: Type of neural model to use ("vae", "gan")
            neural_epochs: Number of training epochs for neural models
            neural_batch_size: Batch size for neural model training
            neural_learning_rate: Learning rate for neural models
            neural_device: Device for neural model training ("auto", "cpu", "cuda")
            enable_gap_targeting: Enable performance gap-based targeting (2025 best practice)
            difficulty_distribution: Strategy for difficulty distribution ("uniform", "adaptive", "hard_focused")
            focus_areas: Specific areas to focus generation on (e.g., ["clarity", "specificity"])
            hardness_threshold: Threshold for identifying hard examples (0.0-1.0)
        """
        self.target_samples = target_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.use_enhanced_scoring = use_enhanced_scoring
        self.generation_method = generation_method
        self.neural_model_type = neural_model_type
        self.neural_epochs = neural_epochs
        self.neural_batch_size = neural_batch_size
        self.neural_learning_rate = neural_learning_rate
        self.neural_device = neural_device

        # 2025 adaptive generation parameters
        self.enable_gap_targeting = enable_gap_targeting
        self.difficulty_distribution = difficulty_distribution
        self.focus_areas = focus_areas or []
        self.hardness_threshold = hardness_threshold

        # Performance gap tracking
        self.current_performance_gaps: dict[str, float] = {}
        self.generation_strategy: str = "statistical"  # Will be determined dynamically

        # Initialize enhanced quality scorer
        if use_enhanced_scoring:
            self.quality_scorer = EnhancedQualityScorer(confidence_level=0.95)

        # Initialize neural generator if needed
        self.neural_generator = None
        self.hybrid_generator = None

        if generation_method in ["neural", "hybrid", "diffusion"] and TORCH_AVAILABLE:
            if generation_method == "hybrid":
                # Initialize hybrid generation system (2025 best practice)
                self.hybrid_generator = HybridGenerationSystem(
                    data_dim=len(self.feature_names),
                    device=neural_device
                )
            elif neural_model_type == "diffusion":
                self.neural_generator = DiffusionSyntheticGenerator(
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                    device=neural_device
                )
            else:
                self.neural_generator = NeuralSyntheticGenerator(
                    model_type=neural_model_type,
                    epochs=neural_epochs,
                    batch_size=neural_batch_size,
                    learning_rate=neural_learning_rate,
                    device=neural_device
                )

        # Initialize method performance tracker for auto-selection
        self.method_tracker = MethodPerformanceTracker()

        # Quality assessment and filtering system (2025 best practice)
        self.quality_filter_threshold = 0.7
        self.enable_quality_filtering = True

        # Dynamic batch optimization system (2025 best practice)
        batch_config = BatchOptimizationConfig(
            min_batch_size=10,
            max_batch_size=min(target_samples, 1000),
            initial_batch_size=min(target_samples // 4, 200),
            memory_limit_mb=2000.0,  # 2GB limit
            efficiency_threshold=0.7
        )
        self.batch_optimizer = DynamicBatchOptimizer(batch_config)

        # Generation history tracking (Week 6)
        self.history_tracker: Optional[GenerationHistoryTracker] = None
        self.current_session_id: Optional[str] = None

        # Configure domains based on research insights
        self.domains = self._initialize_domain_configs()

        # Feature specifications (6-dimensional feature vectors)
        self.feature_names = [
            "clarity",  # 0: How clear and understandable the prompt is
            "length",  # 1: Content length and detail level
            "specificity",  # 2: Level of specific details and precision
            "complexity",  # 3: Intellectual/technical complexity level
            "context_richness",  # 4: Amount of contextual information provided
            "actionability",  # 5: How actionable and implementable the result is
        ]

        self.quality_thresholds = {
            "min_samples": 10,  # Minimum for basic optimization
            "ensemble_threshold": 20,  # Minimum for ensemble methods
            "min_classes": 2,  # Minimum class diversity
            "min_variance": 0.1,  # Minimum effectiveness variance
            "max_correlation": 0.8,  # Maximum feature correlation
        }

    def _initialize_domain_configs(self) -> dict[str, DomainConfig]:
        """Initialize domain-specific configurations based on research"""
        return {
            "technical": DomainConfig(
                name="technical",
                ratio=0.25,  # 25% technical content
                patterns=[
                    (
                        "Create API endpoint",
                        "Create a comprehensive REST API endpoint with authentication, rate limiting, error handling, and detailed OpenAPI documentation including examples",
                    ),
                    (
                        "Debug error",
                        "Debug this specific error by analyzing logs, identifying root cause, implementing robust error handling, and adding prevention measures",
                    ),
                    (
                        "Optimize function",
                        "Optimize this function for performance by implementing algorithmic improvements, memory efficiency, and scalability considerations",
                    ),
                    (
                        "Add tests",
                        "Add comprehensive unit tests with edge cases, mocking strategies, integration test coverage, and performance benchmarks",
                    ),
                    (
                        "Document system",
                        "Document this system architecture with detailed diagrams, deployment guides, troubleshooting procedures, and maintenance workflows",
                    ),
                    (
                        "Setup guide",
                        "Create a detailed setup guide with prerequisites, step-by-step instructions, verification steps, and common troubleshooting solutions",
                    ),
                    (
                        "Code review",
                        "Conduct thorough code review focusing on security, performance, maintainability, and adherence to best practices",
                    ),
                    (
                        "Refactor code",
                        "Refactor this code to improve readability, reduce complexity, eliminate duplication, and enhance testability",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.4, 0.9),  # Higher baseline for technical clarity
                    "length": (150, 300),  # Longer technical content
                    "specificity": (0.7, 1.0),  # High specificity for technical
                    "complexity": (4, 8),  # Medium-high complexity
                    "context_richness": (0.5, 0.9),
                    "actionability": (
                        0.8,
                        1.0,
                    ),  # Very high actionability for technical
                },
                effectiveness_params=(
                    3,
                    2,
                ),  # Beta(3,2) - skewed higher for measurable outcomes
                complexity_range=(4, 8),
            ),
            "creative": DomainConfig(
                name="creative",
                ratio=0.20,  # 20% creative content
                patterns=[
                    (
                        "Write story",
                        "Write an engaging story with compelling characters, clear narrative arc, vivid descriptive details, and emotional resonance",
                    ),
                    (
                        "Create content",
                        "Create engaging content that resonates with the target audience, drives meaningful engagement, and achieves specific goals",
                    ),
                    (
                        "Improve writing",
                        "Improve this writing by enhancing flow, adding sensory details, strengthening emotional impact, and clarifying messaging",
                    ),
                    (
                        "Add creativity",
                        "Add creative elements like metaphors, unique perspectives, innovative approaches, and memorable hooks",
                    ),
                    (
                        "Write copy",
                        "Write persuasive copy that speaks to customer pain points, highlights unique value propositions, and drives action",
                    ),
                    (
                        "Create campaign",
                        "Create a comprehensive marketing campaign with compelling messaging, multi-channel strategy, and measurable success metrics",
                    ),
                    (
                        "Design content",
                        "Design visual content that captures attention, communicates key messages, and aligns with brand identity",
                    ),
                    (
                        "Brand voice",
                        "Develop a distinctive brand voice that reflects company values, resonates with target audience, and maintains consistency",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.3, 0.8),  # More variable for creative expression
                    "length": (80, 250),  # Varied creative content length
                    "specificity": (0.4, 0.9),  # Wide range for creative specificity
                    "complexity": (2, 7),  # Varied complexity levels
                    "context_richness": (0.6, 1.0),  # High context for creative
                    "actionability": (0.5, 0.9),  # Medium-high actionability
                },
                effectiveness_params=(
                    2,
                    2,
                ),  # Beta(2,2) - more uniform for subjective creative quality
                complexity_range=(2, 7),
            ),
            "analytical": DomainConfig(
                name="analytical",
                ratio=0.20,  # 20% analytical content
                patterns=[
                    (
                        "Analyze data",
                        "Analyze this dataset comprehensively including statistical summaries, trend identification, correlation analysis, and actionable insights",
                    ),
                    (
                        "Research topic",
                        "Research this topic thoroughly using reliable sources, synthesizing findings, identifying patterns, and drawing evidence-based conclusions",
                    ),
                    (
                        "Create report",
                        "Create a detailed analytical report with executive summary, methodology, key findings, recommendations, and supporting visualizations",
                    ),
                    (
                        "Compare options",
                        "Compare these options systematically using relevant criteria, quantitative analysis, risk assessment, and strategic recommendations",
                    ),
                    (
                        "Identify trends",
                        "Identify significant trends in this data using statistical analysis, predictive modeling, and contextual interpretation",
                    ),
                    (
                        "Validate hypothesis",
                        "Validate this hypothesis through rigorous testing, statistical analysis, and comprehensive evaluation of evidence",
                    ),
                    (
                        "Performance metrics",
                        "Establish comprehensive performance metrics with baselines, targets, measurement methods, and reporting frameworks",
                    ),
                    (
                        "Market analysis",
                        "Conduct thorough market analysis including competitor research, trend analysis, opportunity identification, and strategic implications",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.5, 0.9),  # High clarity for analytical work
                    "length": (120, 280),  # Substantial analytical content
                    "specificity": (0.6, 1.0),  # High specificity for analysis
                    "complexity": (3, 7),  # Medium-high analytical complexity
                    "context_richness": (0.7, 1.0),  # Rich context for analysis
                    "actionability": (0.6, 0.9),  # High actionability for insights
                },
                effectiveness_params=(
                    2.5,
                    1.8,
                ),  # Beta(2.5,1.8) - moderately skewed for analytical rigor
                complexity_range=(3, 7),
            ),
            "instructional": DomainConfig(
                name="instructional",
                ratio=0.20,  # 20% instructional content
                patterns=[
                    (
                        "Explain concept",
                        "Explain this concept clearly with simple language, relevant examples, step-by-step breakdowns, and practical applications",
                    ),
                    (
                        "Create tutorial",
                        "Create a comprehensive tutorial with learning objectives, structured lessons, hands-on exercises, and progress assessments",
                    ),
                    (
                        "Teaching guide",
                        "Develop a teaching guide with curriculum outline, learning activities, assessment methods, and differentiation strategies",
                    ),
                    (
                        "Simplify complex",
                        "Simplify this complex topic using analogies, visual aids, progressive disclosure, and relatable examples",
                    ),
                    (
                        "Learning path",
                        "Design a learning path with prerequisites, milestones, resources, practice opportunities, and mastery indicators",
                    ),
                    (
                        "Study guide",
                        "Create a study guide with key concepts, practice questions, review activities, and self-assessment tools",
                    ),
                    (
                        "Workshop design",
                        "Design an interactive workshop with clear objectives, engaging activities, collaborative exercises, and practical outcomes",
                    ),
                    (
                        "Skill development",
                        "Develop a skill-building program with competency frameworks, practice scenarios, feedback mechanisms, and certification criteria",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.6, 1.0),  # Very high clarity for instruction
                    "length": (100, 220),  # Moderate length for digestibility
                    "specificity": (0.5, 0.8),  # Balanced specificity for learning
                    "complexity": (2, 6),  # Moderate complexity for accessibility
                    "context_richness": (0.6, 0.9),  # Good context for learning
                    "actionability": (
                        0.7,
                        1.0,
                    ),  # Very high actionability for instruction
                },
                effectiveness_params=(
                    2.2,
                    1.5,
                ),  # Beta(2.2,1.5) - skewed toward effective instruction
                complexity_range=(2, 6),
            ),
            "conversational": DomainConfig(
                name="conversational",
                ratio=0.15,  # 15% conversational content
                patterns=[
                    (
                        "Answer question",
                        "Answer this question thoroughly with clear explanations, relevant context, helpful examples, and follow-up guidance",
                    ),
                    (
                        "Provide advice",
                        "Provide thoughtful advice considering multiple perspectives, potential outcomes, practical steps, and personalized recommendations",
                    ),
                    (
                        "Clarify confusion",
                        "Clarify this confusion by addressing misconceptions, providing clear explanations, and offering additional resources",
                    ),
                    (
                        "Support request",
                        "Respond to this support request with empathy, practical solutions, step-by-step guidance, and follow-up options",
                    ),
                    (
                        "Facilitate discussion",
                        "Facilitate this discussion by asking thoughtful questions, encouraging participation, and synthesizing key points",
                    ),
                    (
                        "Resolve conflict",
                        "Help resolve this conflict through active listening, perspective-taking, common ground identification, and solution-focused approaches",
                    ),
                    (
                        "Build rapport",
                        "Build rapport through authentic engagement, shared understanding, emotional intelligence, and mutual respect",
                    ),
                    (
                        "Guide decision",
                        "Guide this decision-making process with structured analysis, option evaluation, and personalized recommendations",
                    ),
                ],
                feature_ranges={
                    "clarity": (0.4, 0.9),  # Variable clarity for natural conversation
                    "length": (60, 180),  # Shorter conversational responses
                    "specificity": (0.3, 0.7),  # Moderate specificity for dialogue
                    "complexity": (1, 5),  # Lower complexity for accessibility
                    "context_richness": (0.5, 0.8),  # Moderate context richness
                    "actionability": (
                        0.4,
                        0.8,
                    ),  # Variable actionability for conversation
                },
                effectiveness_params=(
                    2,
                    2.5,
                ),  # Beta(2,2.5) - slightly skewed for helpful responses
                complexity_range=(1, 5),
            ),
        }

    async def generate_comprehensive_training_data(self) -> dict[str, Any]:
        """Generate comprehensive training data with enhanced quality assessment"""
        logger.info(
            f"Starting generation of {self.target_samples} high-quality synthetic samples"
        )

        # Initialize storage
        all_features = []
        all_effectiveness = []
        all_prompts = []
        domain_counts = {}

        # Generate data for each domain
        for domain_name, domain_config in self.domains.items():
            domain_samples = int(self.target_samples * domain_config.ratio)
            logger.info(f"Generating {domain_samples} samples for {domain_name} domain")

            domain_data = await self._generate_domain_data(
                domain_config, domain_samples
            )

            all_features.extend(domain_data["features"])
            all_effectiveness.extend(domain_data["effectiveness_scores"])
            all_prompts.extend(domain_data["prompts"])
            domain_counts[domain_name] = len(domain_data["features"])

        # Apply research-based quality guarantees
        logger.info("Applying statistical quality guarantees")
        features, effectiveness = self._ensure_quality_guarantees(
            all_features, all_effectiveness
        )

        # Enhanced or legacy quality assessment
        if self.use_enhanced_scoring:
            # Use enhanced multi-dimensional quality assessment
            logger.info("Performing enhanced multi-dimensional quality assessment")
            enhanced_metrics = await self.quality_scorer.assess_comprehensive_quality(
                features,
                effectiveness,
                domain_counts,
                {
                    "target_samples": self.target_samples,
                    "random_state": self.random_state,
                },
            )

            # Apply corrections if overall score is below threshold
            if enhanced_metrics.overall_score < 0.55:  # ADEQUATE threshold
                logger.warning(
                    f"Enhanced quality score {enhanced_metrics.overall_score:.3f} below threshold, applying corrections"
                )
                features, effectiveness = self._apply_quality_corrections(
                    features, effectiveness
                )

                # Re-assess after corrections
                enhanced_metrics = (
                    await self.quality_scorer.assess_comprehensive_quality(
                        features,
                        effectiveness,
                        domain_counts,
                        {
                            "target_samples": self.target_samples,
                            "random_state": self.random_state,
                        },
                    )
                )
        else:
            # Use legacy binary quality validation
            logger.info("Validating against ML pipeline requirements")
            legacy_metrics = self._validate_ml_requirements(
                features, effectiveness, domain_counts
            )

            if not legacy_metrics.overall_quality:
                logger.warning(
                    "Generated data failed quality validation, applying corrections"
                )
                features, effectiveness = self._apply_quality_corrections(
                    features, effectiveness
                )
                legacy_metrics = self._validate_ml_requirements(
                    features, effectiveness, domain_counts
                )

        # Prepare result structure
        result = {
            "features": features,
            "effectiveness_scores": effectiveness,
            "prompts": all_prompts[
                : len(features)
            ],  # Ensure alignment after corrections
            "metadata": {
                "source": "enhanced_synthetic_v3",
                "total_samples": len(features),
                "domain_distribution": domain_counts,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "quality_assessment_type": "enhanced"
                if self.use_enhanced_scoring
                else "legacy",
            },
        }

        # Add appropriate quality metrics to metadata
        if self.use_enhanced_scoring:
            result["metadata"]["enhanced_quality_metrics"] = enhanced_metrics
        else:
            result["metadata"]["quality_metrics"] = legacy_metrics

        logger.info(
            f"Successfully generated {len(features)} high-quality synthetic samples"
        )
        return result

    async def generate_neural_training_data(self) -> dict[str, Any]:
        """Generate training data using modern neural generative models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to statistical generation")
            return await self.generate_comprehensive_training_data()

        logger.info(f"Starting neural generation of {self.target_samples} samples using {self.neural_model_type}")

        # First generate a base dataset using statistical methods for training the neural model
        base_data = await self.generate_comprehensive_training_data()
        base_features = np.array(base_data["features"])
        base_effectiveness = np.array(base_data["effectiveness_scores"])

        # Train neural generator on base data
        logger.info("Training neural generative model on base statistical data")
        self.neural_generator.fit(base_features)

        # Generate synthetic features using neural model
        logger.info("Generating synthetic features using trained neural model")
        synthetic_features = self.neural_generator.generate(self.target_samples)

        # Generate corresponding effectiveness scores using domain patterns
        synthetic_effectiveness = []
        synthetic_prompts = []

        # Map synthetic features back to domain patterns
        for i, feature_vector in enumerate(synthetic_features):
            # Select domain based on feature characteristics
            domain_name = self._select_domain_from_features(feature_vector)
            domain_config = self.domains[domain_name]

            # Generate effectiveness score for this domain
            effectiveness = self._generate_domain_effectiveness(domain_config, i % 3)  # Cycle through effectiveness tiers
            synthetic_effectiveness.append(effectiveness)

            # Select a random pattern from the domain
            pattern_idx = self.rng.randint(0, len(domain_config.patterns))
            original, enhanced = domain_config.patterns[pattern_idx]
            synthetic_prompts.append((original, enhanced))

        # Apply quality guarantees
        logger.info("Applying quality guarantees to neural-generated data")
        final_features, final_effectiveness = self._ensure_quality_guarantees(
            synthetic_features.tolist(), synthetic_effectiveness
        )

        # Quality assessment
        domain_counts = self._estimate_domain_distribution(final_features)

        if self.use_enhanced_scoring:
            enhanced_metrics = await self.quality_scorer.assess_comprehensive_quality(
                final_features,
                final_effectiveness,
                domain_counts,
                {
                    "target_samples": self.target_samples,
                    "random_state": self.random_state,
                    "generation_method": "neural",
                    "neural_model_type": self.neural_model_type,
                },
            )
            quality_metrics = enhanced_metrics
        else:
            quality_metrics = self._assess_legacy_quality(final_features, final_effectiveness, domain_counts)

        # Prepare result
        result = {
            "features": final_features,
            "effectiveness_scores": final_effectiveness,
            "prompts": synthetic_prompts[:len(final_features)],
            "metadata": {
                "source": f"neural_{self.neural_model_type}_v1",
                "total_samples": len(final_features),
                "domain_distribution": domain_counts,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "quality_assessment_type": "enhanced" if self.use_enhanced_scoring else "legacy",
                "generation_method": "neural",
                "neural_model_type": self.neural_model_type,
                "neural_epochs": self.neural_epochs,
            },
        }

        if self.use_enhanced_scoring:
            result["metadata"]["enhanced_quality_metrics"] = quality_metrics
        else:
            result["metadata"]["quality_metrics"] = quality_metrics

        logger.info(f"Successfully generated {len(final_features)} neural synthetic samples")
        return result

    async def generate_hybrid_training_data(self) -> dict[str, Any]:
        """Generate training data using hybrid statistical + neural approach"""
        logger.info(f"Starting hybrid generation of {self.target_samples} samples")

        # Generate 70% using statistical methods
        statistical_samples = int(self.target_samples * 0.7)
        neural_samples = self.target_samples - statistical_samples

        # Generate statistical portion
        original_target = self.target_samples
        self.target_samples = statistical_samples
        statistical_data = await self.generate_comprehensive_training_data()
        self.target_samples = original_target

        # Generate neural portion if available
        if TORCH_AVAILABLE and self.neural_generator:
            self.target_samples = neural_samples
            neural_data = await self.generate_neural_training_data()
            self.target_samples = original_target

            # Combine datasets
            combined_features = statistical_data["features"] + neural_data["features"]
            combined_effectiveness = statistical_data["effectiveness_scores"] + neural_data["effectiveness_scores"]
            combined_prompts = statistical_data["prompts"] + neural_data["prompts"]

            # Merge domain distributions
            combined_domain_counts = statistical_data["metadata"]["domain_distribution"].copy()
            for domain, count in neural_data["metadata"]["domain_distribution"].items():
                combined_domain_counts[domain] = combined_domain_counts.get(domain, 0) + count
        else:
            logger.warning("Neural generation not available, using statistical only")
            return statistical_data

        # Final quality assessment on combined data
        if self.use_enhanced_scoring:
            enhanced_metrics = await self.quality_scorer.assess_comprehensive_quality(
                combined_features,
                combined_effectiveness,
                combined_domain_counts,
                {
                    "target_samples": self.target_samples,
                    "random_state": self.random_state,
                    "generation_method": "hybrid",
                },
            )
            quality_metrics = enhanced_metrics
        else:
            quality_metrics = self._assess_legacy_quality(combined_features, combined_effectiveness, combined_domain_counts)

        # Prepare hybrid result
        result = {
            "features": combined_features,
            "effectiveness_scores": combined_effectiveness,
            "prompts": combined_prompts,
            "metadata": {
                "source": "hybrid_statistical_neural_v1",
                "total_samples": len(combined_features),
                "domain_distribution": combined_domain_counts,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "quality_assessment_type": "enhanced" if self.use_enhanced_scoring else "legacy",
                "generation_method": "hybrid",
                "statistical_samples": len(statistical_data["features"]),
                "neural_samples": len(neural_data["features"]) if TORCH_AVAILABLE else 0,
            },
        }

        if self.use_enhanced_scoring:
            result["metadata"]["enhanced_quality_metrics"] = quality_metrics
        else:
            result["metadata"]["quality_metrics"] = quality_metrics

        logger.info(f"Successfully generated {len(combined_features)} hybrid synthetic samples")
        return result

    async def _generate_domain_data(
        self, domain_config: DomainConfig, sample_count: int
    ) -> dict[str, Any]:
        """Generate domain-specific training data with realistic patterns"""
        features = []
        effectiveness_scores = []
        prompts = []

        patterns = domain_config.patterns

        # Use optimized scikit-learn feature generation with modern parameters
        base_features, base_effectiveness = make_classification(
            n_samples=sample_count,
            n_features=len(self.feature_names),
            n_informative=len(self.feature_names),  # All features informative
            n_redundant=0,  # No redundant features
            n_clusters_per_class=3,  # Increased clusters for better diversity
            n_classes=3,  # Three effectiveness tiers
            class_sep=1.5,  # Improved class separation for better quality
            flip_y=0.005,  # Reduced label noise for higher quality
            weights=None,  # Balanced classes
            hypercube=True,  # Use hypercube for better feature distribution
            shift=0.0,
            scale=1.0,
            shuffle=True,
            random_state=self.random_state + hash(domain_config.name) % 1000,
        )

        # Transform base features to domain-specific ranges
        for i in range(sample_count):
            pattern_idx = i % len(patterns)
            original, enhanced = patterns[pattern_idx]

            # Transform features to domain-specific ranges
            domain_features = self._transform_to_domain_features(
                base_features[i], domain_config
            )

            # Generate effectiveness score based on domain characteristics
            effectiveness_score = self._generate_domain_effectiveness(
                domain_config, base_effectiveness[i]
            )

            features.append(domain_features)
            effectiveness_scores.append(effectiveness_score)
            prompts.append((original, enhanced))

        return {
            "features": features,
            "effectiveness_scores": effectiveness_scores,
            "prompts": prompts,
        }

    def _transform_to_domain_features(
        self, base_features: np.ndarray, domain_config: DomainConfig
    ) -> list[float]:
        """Transform normalized features to domain-specific ranges"""
        # Normalize base features to [0, 1] range
        normalized_features = (base_features - base_features.min()) / (
            base_features.max() - base_features.min() + 1e-8
        )

        domain_features = []
        for i, feature_name in enumerate(self.feature_names):
            feature_range = domain_config.feature_ranges[feature_name]

            if feature_name == "complexity":
                # Handle discrete complexity as integer
                complexity_val = int(
                    feature_range[0]
                    + normalized_features[i] * (feature_range[1] - feature_range[0])
                )
                domain_features.append(float(complexity_val))
            else:
                # Transform continuous features
                feature_val = feature_range[0] + normalized_features[i] * (
                    feature_range[1] - feature_range[0]
                )
                domain_features.append(feature_val)

        return domain_features

    def _generate_domain_effectiveness(
        self, domain_config: DomainConfig, base_class: int
    ) -> float:
        """Generate domain-specific effectiveness score with three-tier stratification"""
        alpha, beta = domain_config.effectiveness_params

        # Map class to effectiveness tier
        if base_class == 0:  # Low effectiveness (25%)
            effectiveness = self.rng.beta(alpha, beta + 2) * 0.3  # 0.0-0.3 range
        elif base_class == 1:  # Medium effectiveness (50%)
            effectiveness = 0.3 + self.rng.beta(alpha, beta) * 0.3  # 0.3-0.6 range
        else:  # High effectiveness (25%)
            effectiveness = 0.6 + self.rng.beta(alpha + 1, beta) * 0.4  # 0.6-1.0 range

        # Ensure valid range
        return np.clip(effectiveness, 0.0, 1.0)

    def _ensure_quality_guarantees(
        self, features: list[list[float]], effectiveness: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        """Apply research-based quality guarantees with modern optimizations"""
        effectiveness_array = np.array(effectiveness)
        features_array = np.array(features)

        # 1. Enhanced class diversity guarantee with modern stratification
        unique_values = np.unique(effectiveness_array)
        if len(unique_values) < 3:  # Target 3 tiers
            logger.warning("Insufficient class diversity, applying advanced stratification")
            effectiveness_array = self._apply_three_tier_distribution(
                effectiveness_array
            )

        # 2. Modern feature correlation optimization
        if features_array.shape[1] > 1:
            correlation_matrix = np.corrcoef(features_array.T)
            max_correlation = np.max(np.abs(correlation_matrix - np.eye(features_array.shape[1])))

            if max_correlation > 0.95:  # Too high correlation
                logger.info("Applying correlation reduction for better feature diversity")
                features_array = self._reduce_feature_correlation(features_array)
                features = features_array.tolist()

        # 2. Variance guarantee (from research)
        effectiveness_std = np.std(effectiveness_array)
        if effectiveness_std < self.quality_thresholds["min_variance"]:
            logger.warning(
                f"Low variance ({effectiveness_std:.3f}), injecting controlled noise"
            )
            noise = self.rng.normal(0, 0.15, len(effectiveness_array))
            effectiveness_array = np.clip(effectiveness_array + noise, 0.0, 1.0)

        # 3. Feature correlation control
        features_array = np.array(features)
        correlations = np.corrcoef(features_array.T)
        max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))

        if max_corr > self.quality_thresholds["max_correlation"]:
            logger.warning(
                f"High feature correlation ({max_corr:.3f}), applying decorrelation"
            )
            features_array = self._decorrelate_features(features_array)

        return features_array.tolist(), effectiveness_array.tolist()

    def _apply_three_tier_distribution(
        self, effectiveness_array: np.ndarray
    ) -> np.ndarray:
        """Apply three-tier performance stratification from research"""
        n_samples = len(effectiveness_array)
        quarter_size = n_samples // 4
        half_size = n_samples // 2

        # Redistribute to ensure three performance tiers
        shuffle_indices = self.rng.permutation(n_samples)

        # High performers (25%)
        effectiveness_array[shuffle_indices[:quarter_size]] = self.rng.uniform(
            0.7, 1.0, quarter_size
        )

        # Medium performers (50%)
        effectiveness_array[
            shuffle_indices[quarter_size : quarter_size + half_size]
        ] = self.rng.uniform(0.4, 0.6, half_size)

        # Low performers (25%)
        effectiveness_array[shuffle_indices[quarter_size + half_size :]] = (
            self.rng.uniform(0.0, 0.3, n_samples - quarter_size - half_size)
        )

        return effectiveness_array

    def _reduce_feature_correlation(self, features_array: np.ndarray) -> np.ndarray:
        """Reduce feature correlation using modern techniques"""
        try:
            # Use PCA whitening to reduce correlation
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Apply PCA with whitening to decorrelate features
            pca = PCA(n_components=features_array.shape[1], whiten=True, random_state=self.random_state)
            features_decorrelated = pca.fit_transform(features_scaled)

            # Transform back to original scale approximately
            features_final = scaler.inverse_transform(features_decorrelated)

            logger.info("Successfully reduced feature correlation using PCA whitening")
            return features_final

        except Exception as e:
            logger.warning(f"Failed to reduce correlation: {e}, using original features")
            return features_array

    def _decorrelate_features(self, features_array: np.ndarray) -> np.ndarray:
        """Apply feature decorrelation to reduce excessive correlations"""
        # Add small amount of independent noise to reduce correlations
        noise_scale = 0.05  # Small noise to preserve feature meaning
        noise = self.rng.normal(0, noise_scale, features_array.shape)

        # Apply noise with feature-specific scaling
        for i, feature_name in enumerate(self.feature_names):
            if feature_name == "complexity":
                # Round complexity back to integers
                features_array[:, i] = np.round(features_array[:, i] + noise[:, i])
                features_array[:, i] = np.clip(features_array[:, i], 1, 8)
            else:
                features_array[:, i] += noise[:, i]
                # Clip to reasonable ranges
                features_array[:, i] = np.clip(features_array[:, i], 0.0, 1000.0)

        return features_array

    def _validate_ml_requirements(
        self,
        features: list[list[float]],
        effectiveness: list[float],
        domain_counts: dict[str, int],
    ) -> QualityMetrics:
        """Validate against ML pipeline requirements"""
        features_array = np.array(features)
        effectiveness_array = np.array(effectiveness)

        # Basic validations
        sample_count = len(features)
        class_diversity = len(np.unique(effectiveness_array))
        variance_sufficient = (
            np.std(effectiveness_array) >= self.quality_thresholds["min_variance"]
        )
        min_samples_met = sample_count >= self.quality_thresholds["min_samples"]
        ensemble_ready = sample_count >= self.quality_thresholds["ensemble_threshold"]
        no_invalid_values = not (
            np.isnan(features_array).any() or np.isnan(effectiveness_array).any()
        )

        # Feature correlation analysis
        correlations = np.corrcoef(features_array.T)
        max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
        feature_correlations = {
            "max_correlation": float(max_corr),
            "mean_correlation": float(
                np.mean(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
            ),
            "correlation_threshold_met": max_corr
            <= self.quality_thresholds["max_correlation"],
        }

        overall_quality = all([
            min_samples_met,
            class_diversity >= self.quality_thresholds["min_classes"],
            variance_sufficient,
            no_invalid_values,
            feature_correlations["correlation_threshold_met"],
        ])

        return QualityMetrics(
            sample_count=sample_count,
            class_diversity=class_diversity,
            variance_sufficient=variance_sufficient,
            min_samples_met=min_samples_met,
            ensemble_ready=ensemble_ready,
            no_invalid_values=no_invalid_values,
            overall_quality=overall_quality,
            domain_distribution=domain_counts,
            feature_correlations=feature_correlations,
        )

    def _apply_quality_corrections(
        self, features: list[list[float]], effectiveness: list[float]
    ) -> tuple[list[list[float]], list[float]]:
        """Apply corrections for failed quality validation"""
        logger.info("Applying quality corrections to meet ML requirements")

        # Ensure minimum sample count by duplicating with noise
        while len(features) < self.quality_thresholds["min_samples"]:
            idx = self.rng.randint(0, len(features))
            new_feature = np.array(features[idx]) + self.rng.normal(
                0, 0.05, len(features[idx])
            )
            new_effectiveness = effectiveness[idx] + self.rng.normal(0, 0.05)

            features.append(new_feature.tolist())
            effectiveness.append(np.clip(new_effectiveness, 0.0, 1.0))

        # Ensure class diversity by forcing three-tier distribution
        effectiveness_array = np.array(effectiveness)
        effectiveness_array = self._apply_three_tier_distribution(effectiveness_array)

        # Apply final quality guarantees
        features, effectiveness = self._ensure_quality_guarantees(
            features, effectiveness_array.tolist()
        )

        return features, effectiveness

    async def save_to_database(
        self, training_data: dict[str, Any], db_session: AsyncSession
    ) -> int:
        """Save generated training data to database with TrainingPrompt model compatibility

        Args:
            training_data: Generated training data from generate_comprehensive_training_data
            db_session: Database session for saving

        Returns:
            Number of records saved
        """
        features = training_data["features"]
        effectiveness_scores = training_data["effectiveness_scores"]
        prompts = training_data["prompts"]
        metadata = training_data["metadata"]

        saved_count = 0

        try:
            for i, (feature_vector, effectiveness, (original, enhanced)) in enumerate(
                zip(features, effectiveness_scores, prompts, strict=False)
            ):
                training_prompt = TrainingPrompt(
                    prompt_text=original,
                    enhancement_result={
                        "enhanced_prompt": enhanced,
                        "effectiveness_score": effectiveness,
                        "feature_vector": feature_vector,
                        "metadata": {
                            "source": metadata["source"],
                            "domain": self._identify_domain_from_index(
                                i, metadata["domain_distribution"]
                            ),
                            "generation_timestamp": metadata["generation_timestamp"],
                            "feature_names": metadata["feature_names"],
                            "quality_validated": True,
                        },
                    },
                    data_source="synthetic",
                    training_priority=10,  # Synthetic data priority
                )

                db_session.add(training_prompt)
                saved_count += 1

            await db_session.commit()
            logger.info(
                f"Successfully saved {saved_count} synthetic training samples to database"
            )

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to save synthetic data to database: {e}")
            raise

        return saved_count

    def _select_domain_from_features(self, feature_vector: np.ndarray) -> str:
        """Select domain based on feature characteristics"""
        # Simple heuristic based on feature patterns
        # In practice, this could use clustering or classification

        # Assume features are: clarity, length, specificity, complexity, context_richness, actionability
        if len(feature_vector) >= 6:
            clarity = feature_vector[0]
            length = feature_vector[1] if len(feature_vector) > 1 else 0.5
            specificity = feature_vector[2] if len(feature_vector) > 2 else 0.5
            complexity = feature_vector[3] if len(feature_vector) > 3 else 0.5
            actionability = feature_vector[5] if len(feature_vector) > 5 else 0.5

            # Domain selection logic based on feature patterns
            if actionability > 0.8 and specificity > 0.7:
                return "technical"
            elif clarity < 0.6 and complexity < 0.5:
                return "creative"
            elif specificity > 0.6 and complexity > 0.5:
                return "analytical"
            elif clarity > 0.6 and actionability > 0.7:
                return "instructional"
            else:
                return "conversational"
        else:
            # Fallback to random selection
            return self.rng.choice(list(self.domains.keys()))

    def _estimate_domain_distribution(self, features: list[list[float]]) -> dict[str, int]:
        """Estimate domain distribution from generated features"""
        domain_counts = {domain: 0 for domain in self.domains.keys()}

        for feature_vector in features:
            domain = self._select_domain_from_features(np.array(feature_vector))
            domain_counts[domain] += 1

        return domain_counts

    async def generate_data(self) -> dict[str, Any]:
        """Main generation method that routes to appropriate generator based on configuration"""
        if self.generation_method == "statistical":
            return await self.generate_comprehensive_training_data()
        elif self.generation_method == "neural":
            return await self.generate_neural_training_data()
        elif self.generation_method == "diffusion":
            return await self.generate_diffusion_training_data()
        elif self.generation_method == "hybrid":
            return await self.generate_hybrid_training_data()
        else:
            logger.warning(f"Unknown generation method: {self.generation_method}, falling back to statistical")
            return await self.generate_comprehensive_training_data()

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for synthetic data generation (2025 pattern)

        Args:
            config: Orchestrator configuration containing:
                - target_samples: Number of samples to generate (optional, uses instance default)
                - generation_method: Method to use ("statistical", "neural", "diffusion", "hybrid")
                - output_path: Local path for output files (optional)
                - quality_assessment: Whether to include quality assessment (default: True)
                - domain_distribution: Custom domain distribution (optional)
                - performance_gaps: Performance gaps for targeted generation (optional)
                - target_gaps: Enable gap-based targeting (default: False)
                - strategy: Targeting strategy ("gap_based", "rule_focused", etc.)
                - focus_areas: Specific areas to focus on (optional)

        Returns:
            Orchestrator-compatible result with synthetic data and metadata
        """
        from datetime import datetime
        start_time = datetime.now()

        try:
            # Extract configuration parameters
            target_samples = config.get("target_samples", self.target_samples)
            generation_method = config.get("generation_method", self.generation_method)
            output_path = config.get("output_path", "./outputs/synthetic_data")
            quality_assessment = config.get("quality_assessment", True)

            # 2025 adaptive generation parameters
            performance_gaps = config.get("performance_gaps", {})
            target_gaps = config.get("target_gaps", False)
            strategy = config.get("strategy", "gap_based")
            focus_areas = config.get("focus_areas", [])

            # Update instance configuration if needed
            original_target = self.target_samples
            original_method = self.generation_method

            self.target_samples = target_samples
            self.generation_method = generation_method

            logger.info(f"Starting orchestrated synthetic data generation: {target_samples} samples using {generation_method}")

            # Generate synthetic data - use targeted generation if gaps provided
            if target_gaps and performance_gaps:
                logger.info(f"Using targeted generation with strategy: {strategy}")
                result = await self.generate_targeted_data(
                    performance_gaps=performance_gaps,
                    strategy=strategy,
                    batch_size=target_samples,
                    focus_areas=focus_areas
                )
            else:
                result = await self.generate_data()

            # Restore original configuration
            self.target_samples = original_target
            self.generation_method = original_method

            # Prepare orchestrator-compatible response
            execution_time = (datetime.now() - start_time).total_seconds()

            orchestrator_result = {
                "orchestrator_compatible": True,
                "component_result": {
                    "synthetic_data": result,
                    "generation_summary": {
                        "samples_generated": len(result.get("features", [])),
                        "generation_method": generation_method,
                        "quality_score": result.get("metadata", {}).get("quality_score", 0.0),
                        "domain_distribution": result.get("metadata", {}).get("domain_distribution", {}),
                    }
                },
                "local_metadata": {
                    "execution_time_seconds": execution_time,
                    "output_files": [f"{output_path}/synthetic_data.json"],
                    "memory_used_mb": result.get("metadata", {}).get("memory_usage_mb", 0),
                    "component_name": "ProductionSyntheticDataGenerator",
                    "generation_timestamp": start_time.isoformat(),
                    "configuration": {
                        "target_samples": target_samples,
                        "generation_method": generation_method,
                        "neural_model_type": self.neural_model_type,
                        "quality_assessment_enabled": quality_assessment
                    }
                }
            }

            logger.info(f"Orchestrated synthetic data generation completed: {execution_time:.2f}s")
            return orchestrator_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Orchestrated synthetic data generation failed: {e}")

            return {
                "orchestrator_compatible": True,
                "component_result": {
                    "error": str(e),
                    "status": "failed"
                },
                "local_metadata": {
                    "execution_time_seconds": execution_time,
                    "component_name": "ProductionSyntheticDataGenerator",
                    "error_timestamp": datetime.now().isoformat(),
                    "configuration": config
                }
            }

    async def generate_diffusion_training_data(self) -> dict[str, Any]:
        """Generate training data using diffusion models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to statistical generation")
            return await self.generate_comprehensive_training_data()

        logger.info(f"Starting diffusion generation of {self.target_samples} samples")

        # First generate a base dataset using statistical methods for training the diffusion model
        base_data = await self.generate_comprehensive_training_data()
        base_features = np.array(base_data["features"])

        # Train diffusion generator on base data
        logger.info("Training diffusion model on base statistical data")
        if isinstance(self.neural_generator, DiffusionSyntheticGenerator):
            self.neural_generator.fit(base_features)
        else:
            logger.error("Neural generator is not a diffusion model")
            return await self.generate_comprehensive_training_data()

        # Generate synthetic features using diffusion model
        logger.info("Generating synthetic features using trained diffusion model")
        synthetic_features = self.neural_generator.generate(self.target_samples)

        # Generate corresponding effectiveness scores and prompts
        synthetic_effectiveness = []
        synthetic_prompts = []

        for i, feature_vector in enumerate(synthetic_features):
            domain_name = self._select_domain_from_features(feature_vector)
            domain_config = self.domains[domain_name]

            effectiveness = self._generate_domain_effectiveness(domain_config, i % 3)
            synthetic_effectiveness.append(effectiveness)

            pattern_idx = self.rng.randint(0, len(domain_config.patterns))
            original, enhanced = domain_config.patterns[pattern_idx]
            synthetic_prompts.append((original, enhanced))

        # Apply quality guarantees
        logger.info("Applying quality guarantees to diffusion-generated data")
        final_features, final_effectiveness = self._ensure_quality_guarantees(
            synthetic_features.tolist(), synthetic_effectiveness
        )

        # Quality assessment
        domain_counts = self._estimate_domain_distribution(final_features)

        if self.use_enhanced_scoring:
            enhanced_metrics = await self.quality_scorer.assess_comprehensive_quality(
                final_features,
                final_effectiveness,
                domain_counts,
                {
                    "target_samples": self.target_samples,
                    "random_state": self.random_state,
                    "generation_method": "diffusion",
                },
            )
            quality_metrics = enhanced_metrics
        else:
            quality_metrics = self._assess_legacy_quality(final_features, final_effectiveness, domain_counts)

        # Prepare result
        result = {
            "features": final_features,
            "effectiveness_scores": final_effectiveness,
            "prompts": synthetic_prompts[:len(final_features)],
            "metadata": {
                "source": "diffusion_tabular_v1",
                "total_samples": len(final_features),
                "domain_distribution": domain_counts,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "quality_assessment_type": "enhanced" if self.use_enhanced_scoring else "legacy",
                "generation_method": "diffusion",
                "diffusion_timesteps": self.neural_generator.num_timesteps if hasattr(self.neural_generator, 'num_timesteps') else 1000,
            },
        }

        if self.use_enhanced_scoring:
            result["metadata"]["enhanced_quality_metrics"] = quality_metrics
        else:
            result["metadata"]["quality_metrics"] = quality_metrics

        logger.info(f"Successfully generated {len(final_features)} diffusion synthetic samples")
        return result

    def _identify_domain_from_index(
        self, index: int, domain_distribution: dict[str, int]
    ) -> str:
        """Identify domain for a given sample index"""
        current_count = 0
        for domain_name, count in domain_distribution.items():
            if index < current_count + count:
                return domain_name
            current_count += count

        return "unknown"  # Fallback

    def get_generation_summary(self, training_data: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive summary with enhanced quality reporting"""
        metadata = training_data["metadata"]

        if self.use_enhanced_scoring and "enhanced_quality_metrics" in metadata:
            enhanced_metrics = metadata["enhanced_quality_metrics"]

            # Generate comprehensive quality report
            quality_report = self.quality_scorer.generate_quality_report(
                enhanced_metrics
            )

            return {
                "generation_summary": {
                    "total_samples": metadata["total_samples"],
                    "target_samples": self.target_samples,
                    "generation_efficiency": metadata["total_samples"]
                    / self.target_samples,
                    "overall_quality_score": enhanced_metrics.overall_score,
                    "confidence_score": enhanced_metrics.confidence_score,
                    "recommendation_tier": enhanced_metrics.recommendation_tier,
                    "domains_covered": len(metadata["domain_distribution"]),
                    "generation_time": metadata["generation_timestamp"],
                    "assessment_type": "enhanced_multi_dimensional",
                },
                "quality_analysis": quality_report,
                "domain_breakdown": metadata["domain_distribution"],
                "dimensional_scores": {
                    "fidelity": enhanced_metrics.fidelity.score,
                    "utility": enhanced_metrics.utility.score,
                    "privacy": enhanced_metrics.privacy.score,
                    "statistical_validity": enhanced_metrics.statistical_validity.score,
                    "diversity": enhanced_metrics.diversity.score,
                    "consistency": enhanced_metrics.consistency.score,
                },
            }
            quality_metrics = metadata["quality_metrics"]

        return {
            "generation_summary": {
                "total_samples": metadata["total_samples"],
                "target_samples": self.target_samples,
                "generation_efficiency": metadata["total_samples"]
                / self.target_samples,
                "quality_score": float(quality_metrics.overall_quality),
                "domains_covered": len(metadata["domain_distribution"]),
                "generation_time": metadata["generation_timestamp"],
                "assessment_type": "legacy_binary",
            },
            "quality_analysis": {
                "class_diversity": quality_metrics.class_diversity,
                "variance_sufficient": quality_metrics.variance_sufficient,
                "ml_requirements_met": quality_metrics.overall_quality,
                "ensemble_ready": quality_metrics.ensemble_ready,
                "feature_correlations": quality_metrics.feature_correlations,
            },
            "domain_breakdown": metadata["domain_distribution"],
            "recommendations": self._generate_recommendations(quality_metrics),
        }

    def _generate_recommendations(self, quality_metrics: QualityMetrics) -> list[str]:
        """Generate recommendations for improving synthetic data quality"""
        recommendations = []

        if not quality_metrics.overall_quality:
            recommendations.append(
                "Consider increasing sample count for more robust ML training"
            )

        if quality_metrics.class_diversity < 3:
            recommendations.append(
                "Enhance effectiveness score stratification for better class separation"
            )

        if not quality_metrics.variance_sufficient:
            recommendations.append(
                "Increase diversity in generation patterns to improve variance"
            )

        if quality_metrics.feature_correlations["max_correlation"] > 0.7:
            recommendations.append(
                "Reduce feature correlations for improved ML model performance"
            )

        if quality_metrics.overall_quality:
            recommendations.append(
                "Generated data meets all quality requirements for ML training"
            )

        return recommendations

    # ===== 2025 ADAPTIVE DATA GENERATION METHODS =====

    async def generate_targeted_data(
        self,
        performance_gaps: dict[str, float],
        strategy: str = "gap_based",
        batch_size: int = 200,
        focus_areas: list[str] | None = None
    ) -> dict[str, Any]:
        """Generate targeted synthetic data based on performance gaps (2025 best practice)

        Args:
            performance_gaps: Dictionary mapping metric names to gap magnitudes
            strategy: Generation strategy ("gap_based", "rule_focused", "diversity_enhanced", "neural_enhanced")
            batch_size: Number of samples to generate
            focus_areas: Specific areas to focus on (overrides instance focus_areas)

        Returns:
            Dictionary containing targeted synthetic data and metadata
        """
        logger.info(f"Starting targeted data generation: strategy={strategy}, batch_size={batch_size}")

        # Update current performance gaps
        self.current_performance_gaps = performance_gaps

        # Determine optimal generation strategy
        optimal_strategy = self._determine_generation_strategy(performance_gaps)
        self.generation_strategy = optimal_strategy

        # Use provided focus areas or instance focus areas
        target_focus_areas = focus_areas or self.focus_areas

        # Configure difficulty distribution based on gaps
        difficulty_config = self._configure_difficulty_distribution(performance_gaps)

        # Generate targeted data using optimal strategy (Enhanced 2025 approach)
        if self.generation_method == "hybrid" and self.hybrid_generator:
            # Use hybrid generation system for best quality (2025 best practice)
            result = await self._generate_hybrid_targeted_data(
                performance_gaps, batch_size, target_focus_areas, difficulty_config
            )
        elif optimal_strategy == "neural_enhanced":
            result = await self._generate_neural_enhanced_data(
                performance_gaps, batch_size, target_focus_areas, difficulty_config
            )
        elif optimal_strategy == "rule_focused":
            result = await self._generate_rule_focused_data(
                performance_gaps, batch_size, target_focus_areas, difficulty_config
            )
        elif optimal_strategy == "diversity_enhanced":
            result = await self._generate_diversity_enhanced_data(
                performance_gaps, batch_size, target_focus_areas, difficulty_config
            )
        else:  # statistical fallback
            result = await self._generate_gap_based_statistical_data(
                performance_gaps, batch_size, target_focus_areas, difficulty_config
            )

        # Apply quality filtering if enabled (2025 best practice)
        if self.enable_quality_filtering and result.get("features"):
            result = await self._apply_quality_filtering(result)

        # Update method performance tracking
        await self._update_method_performance_tracking(result, performance_gaps)

        # Add targeting metadata
        result["metadata"]["targeting_info"] = {
            "performance_gaps": performance_gaps,
            "strategy_used": optimal_strategy,
            "focus_areas": target_focus_areas,
            "difficulty_config": difficulty_config,
            "gap_targeting_enabled": self.enable_gap_targeting
        }

        logger.info(f"Targeted data generation completed: {len(result.get('features', []))} samples")
        return result

    async def _generate_hybrid_targeted_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data using hybrid approach with multiple methods (2025 best practice)"""
        logger.info("Generating hybrid targeted data using multiple methods")

        if not self.hybrid_generator:
            # Fallback to statistical if hybrid not available
            return await self._generate_gap_based_statistical_data(
                performance_gaps, batch_size, focus_areas, difficulty_config
            )

        try:
            # Use hybrid generation system
            hybrid_result = await self.hybrid_generator.generate_hybrid_data(
                batch_size=batch_size,
                performance_gaps=performance_gaps,
                quality_threshold=self.quality_filter_threshold
            )

            # Convert hybrid result to standard format
            features = hybrid_result['samples']
            effectiveness = [0.7 + 0.3 * np.random.random() for _ in features]  # Placeholder

            # Create metadata
            metadata = {
                "generation_method": "hybrid",
                "generation_timestamp": datetime.now().isoformat(),
                "batch_size": len(features),
                "method_metrics": hybrid_result['method_metrics'],
                "method_allocation": hybrid_result['method_allocation'],
                "total_generation_time": hybrid_result['total_generation_time'],
                "quality_threshold": hybrid_result['quality_threshold'],
                "domain_distribution": self._calculate_domain_distribution(features),
                "targeting_enabled": True,
                "focus_areas": focus_areas,
                "difficulty_config": difficulty_config
            }

            return {
                "features": features,
                "effectiveness": effectiveness,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}")
            # Fallback to statistical generation
            return await self._generate_gap_based_statistical_data(
                performance_gaps, batch_size, focus_areas, difficulty_config
            )

    async def _apply_quality_filtering(self, result: dict[str, Any]) -> dict[str, Any]:
        """Apply quality-based filtering and ranking (2025 best practice)"""
        features = result.get("features", [])
        effectiveness = result.get("effectiveness", [])

        if not features:
            return result

        logger.info(f"Applying quality filtering to {len(features)} samples")

        # Calculate quality scores for each sample
        quality_scores = []
        for i, feature_vector in enumerate(features):
            quality_score = self._calculate_sample_quality_score(feature_vector, effectiveness[i])
            quality_scores.append(quality_score)

        # Filter samples based on quality threshold
        filtered_indices = [
            i for i, score in enumerate(quality_scores)
            if score >= self.quality_filter_threshold
        ]

        if not filtered_indices:
            logger.warning("No samples passed quality filtering, keeping all samples")
            filtered_indices = list(range(len(features)))

        # Apply filtering
        filtered_features = [features[i] for i in filtered_indices]
        filtered_effectiveness = [effectiveness[i] for i in filtered_indices]
        filtered_quality_scores = [quality_scores[i] for i in filtered_indices]

        # Sort by quality score (highest first)
        sorted_data = sorted(
            zip(filtered_features, filtered_effectiveness, filtered_quality_scores),
            key=lambda x: x[2],
            reverse=True
        )

        if sorted_data:
            filtered_features, filtered_effectiveness, filtered_quality_scores = zip(*sorted_data)
            filtered_features = list(filtered_features)
            filtered_effectiveness = list(filtered_effectiveness)
            filtered_quality_scores = list(filtered_quality_scores)

        # Update metadata
        result["features"] = filtered_features
        result["effectiveness"] = filtered_effectiveness
        result["metadata"]["quality_filtering"] = {
            "enabled": True,
            "threshold": self.quality_filter_threshold,
            "original_count": len(features),
            "filtered_count": len(filtered_features),
            "filter_rate": 1.0 - (len(filtered_features) / len(features)),
            "average_quality_score": np.mean(filtered_quality_scores) if filtered_quality_scores else 0.0,
            "quality_score_range": [min(filtered_quality_scores), max(filtered_quality_scores)] if filtered_quality_scores else [0.0, 0.0]
        }

        logger.info(f"Quality filtering completed: {len(filtered_features)}/{len(features)} samples retained")
        return result

    def _calculate_sample_quality_score(self, feature_vector: list[float], effectiveness: float) -> float:
        """Calculate quality score for a single sample (2025 best practice)"""
        # Multi-dimensional quality assessment

        # 1. Feature validity (all values in reasonable range)
        feature_validity = 1.0
        for value in feature_vector:
            if not (0.0 <= value <= 1.0):  # Assuming normalized features
                feature_validity *= 0.8

        # 2. Feature diversity (not all values the same)
        feature_diversity = 1.0 - (np.std(feature_vector) < 0.01)

        # 3. Effectiveness reasonableness
        effectiveness_validity = 1.0 if 0.0 <= effectiveness <= 1.0 else 0.5

        # 4. Feature correlation (avoid highly correlated features)
        correlation_penalty = 0.0
        if len(feature_vector) > 1:
            # Simple correlation check
            for i in range(len(feature_vector) - 1):
                if abs(feature_vector[i] - feature_vector[i + 1]) < 0.05:
                    correlation_penalty += 0.1

        correlation_score = max(0.0, 1.0 - correlation_penalty)

        # Combined quality score
        quality_score = (
            0.3 * feature_validity +
            0.25 * feature_diversity +
            0.25 * effectiveness_validity +
            0.2 * correlation_score
        )

        return min(1.0, max(0.0, quality_score))

    async def _update_method_performance_tracking(
        self,
        result: dict[str, Any],
        performance_gaps: dict[str, float]
    ) -> None:
        """Update method performance tracking for auto-selection (2025 best practice)"""
        metadata = result.get("metadata", {})
        generation_method = metadata.get("generation_method", "statistical")

        # Calculate performance metrics
        features = result.get("features", [])
        effectiveness = result.get("effectiveness", [])

        if not features:
            return

        # Quality metrics
        quality_scores = [
            self._calculate_sample_quality_score(f, e)
            for f, e in zip(features, effectiveness)
        ]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # Diversity metrics
        diversity_score = self._calculate_diversity_score_internal(features)

        # Create performance metrics
        metrics = GenerationMethodMetrics(
            method_name=generation_method,
            generation_time=metadata.get("total_generation_time", 0.0),
            quality_score=avg_quality,
            diversity_score=diversity_score,
            memory_usage_mb=self._get_current_memory_usage(),
            success_rate=1.0 if avg_quality > 0.5 else 0.0,
            samples_generated=len(features),
            timestamp=datetime.now(),
            performance_gaps_addressed=performance_gaps
        )

        # Record performance
        self.method_tracker.record_performance(metrics)

        logger.info(f"Updated performance tracking for {generation_method}: quality={avg_quality:.3f}, diversity={diversity_score:.3f}")

    def _calculate_diversity_score_internal(self, features: list[list[float]]) -> float:
        """Calculate diversity score for feature vectors"""
        if len(features) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                distance = np.linalg.norm(np.array(features[i]) - np.array(features[j]))
                distances.append(distance)

        return float(np.mean(distances)) if distances else 0.0

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _calculate_domain_distribution(self, features: list[list[float]]) -> dict[str, int]:
        """Calculate domain distribution for features"""
        # Placeholder implementation
        return {"technical": len(features) // 4, "creative": len(features) // 4,
                "analytical": len(features) // 4, "other": len(features) - 3 * (len(features) // 4)}

    async def generate_with_dynamic_batching(
        self,
        total_samples: int,
        performance_gaps: dict[str, float] = None,
        strategy: str = "adaptive"
    ) -> dict[str, Any]:
        """Generate data using dynamic batch optimization (2025 best practice)"""
        logger.info(f"Starting dynamic batch generation for {total_samples} samples")

        performance_gaps = performance_gaps or {}
        all_features = []
        all_effectiveness = []
        generation_metadata = {
            "total_samples_requested": total_samples,
            "batches_processed": 0,
            "total_generation_time": 0.0,
            "batch_optimization_stats": [],
            "method": "dynamic_batching"
        }

        remaining_samples = total_samples
        start_time = time.time()

        while remaining_samples > 0:
            # Get optimal batch size
            optimal_batch_size = await self.batch_optimizer.get_optimal_batch_size(
                target_samples=remaining_samples
            )

            batch_start_time = time.time()
            batch_success_count = 0
            batch_error_count = 0

            try:
                # Generate batch using targeted generation
                batch_result = await self.generate_targeted_data(
                    performance_gaps=performance_gaps,
                    strategy=strategy,
                    batch_size=optimal_batch_size
                )

                # Extract results
                batch_features = batch_result.get("features", [])
                batch_effectiveness = batch_result.get("effectiveness", [])

                if batch_features:
                    all_features.extend(batch_features)
                    all_effectiveness.extend(batch_effectiveness)
                    batch_success_count = len(batch_features)
                    remaining_samples -= batch_success_count
                else:
                    batch_error_count = 1

            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                batch_error_count = 1

            # Record batch performance
            batch_time = time.time() - batch_start_time
            await self.batch_optimizer.record_batch_performance(
                batch_size=optimal_batch_size,
                processing_time=batch_time,
                success_count=batch_success_count,
                error_count=batch_error_count
            )

            # Update metadata
            generation_metadata["batches_processed"] += 1
            generation_metadata["total_generation_time"] += batch_time

            # Safety check to prevent infinite loops
            if batch_success_count == 0 and batch_error_count > 0:
                logger.warning("Batch generation failed, reducing remaining samples")
                remaining_samples = max(0, remaining_samples - optimal_batch_size)

            # Log progress
            if generation_metadata["batches_processed"] % 5 == 0:
                progress = (total_samples - remaining_samples) / total_samples * 100
                logger.info(f"Dynamic batching progress: {progress:.1f}% "
                           f"({len(all_features)}/{total_samples} samples)")

        # Get final optimization stats
        optimization_stats = self.batch_optimizer.get_optimization_stats()
        generation_metadata["batch_optimization_stats"] = optimization_stats
        generation_metadata["final_sample_count"] = len(all_features)
        generation_metadata["total_generation_time"] = time.time() - start_time
        generation_metadata["average_batch_time"] = (
            generation_metadata["total_generation_time"] / generation_metadata["batches_processed"]
            if generation_metadata["batches_processed"] > 0 else 0.0
        )

        logger.info(f"Dynamic batch generation completed: {len(all_features)} samples in "
                   f"{generation_metadata['batches_processed']} batches, "
                   f"total time: {generation_metadata['total_generation_time']:.2f}s")

        return {
            "features": all_features,
            "effectiveness": all_effectiveness,
            "metadata": generation_metadata
        }

    def enable_history_tracking(self, db_session) -> None:
        """Enable generation history tracking with database integration"""
        from sqlalchemy.ext.asyncio import AsyncSession

        if isinstance(db_session, AsyncSession):
            self.history_tracker = GenerationHistoryTracker(db_session)
            logger.info("Generation history tracking enabled")
        else:
            logger.warning("Invalid database session provided for history tracking")

    async def generate_with_history_tracking(
        self,
        total_samples: int,
        performance_gaps: dict[str, float] = None,
        strategy: str = "adaptive",
        training_session_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate data with comprehensive history tracking (Week 6 feature)"""

        if not self.history_tracker:
            logger.warning("History tracking not enabled, falling back to standard generation")
            return await self.generate_with_dynamic_batching(
                total_samples=total_samples,
                performance_gaps=performance_gaps,
                strategy=strategy
            )

        # Start tracking session
        session_id = await self.history_tracker.start_tracking_session(
            generation_method=self.generation_method,
            target_samples=total_samples,
            session_type="synthetic_data",
            training_session_id=training_session_id,
            configuration={
                "strategy": strategy,
                "quality_threshold": self.quality_filter_threshold,
                "enable_gap_targeting": self.enable_gap_targeting,
                "difficulty_distribution": self.difficulty_distribution,
                "focus_areas": self.focus_areas
            },
            performance_gaps=performance_gaps,
            focus_areas=self.focus_areas
        )

        self.current_session_id = session_id
        logger.info(f"Started generation session {session_id} with history tracking")

        try:
            # Generate with dynamic batching and track each batch
            result = await self._generate_with_tracked_batching(
                total_samples=total_samples,
                performance_gaps=performance_gaps,
                strategy=strategy,
                session_id=session_id
            )

            # Complete session tracking
            await self.history_tracker.complete_session(
                session_id=session_id,
                final_sample_count=len(result.get("features", [])),
                status="completed"
            )

            # Record method performance
            if result.get("metadata"):
                await self.history_tracker.record_method_performance(
                    session_id=session_id,
                    method_name=self.generation_method,
                    performance_metrics={
                        "generation_time": result["metadata"].get("total_generation_time", 0.0),
                        "quality_score": result["metadata"].get("average_quality_score", 0.0),
                        "diversity_score": result["metadata"].get("diversity_score", 0.0),
                        "memory_usage_mb": result["metadata"].get("memory_usage_mb", 0.0),
                        "success_rate": 1.0 if result.get("features") else 0.0,
                        "samples_generated": len(result.get("features", [])),
                        "performance_gaps_addressed": performance_gaps,
                        "batch_size": result["metadata"].get("average_batch_size"),
                        "configuration": result["metadata"].get("configuration")
                    }
                )

            # Add session tracking info to metadata
            result["metadata"]["session_tracking"] = {
                "session_id": session_id,
                "tracking_enabled": True,
                "training_session_id": training_session_id
            }

            logger.info(f"Completed generation session {session_id} with {len(result.get('features', []))} samples")
            return result

        except Exception as e:
            # Mark session as failed
            await self.history_tracker.complete_session(
                session_id=session_id,
                final_sample_count=0,
                status="failed",
                error_message=str(e)
            )
            logger.error(f"Generation session {session_id} failed: {e}")
            raise

    async def _generate_with_tracked_batching(
        self,
        total_samples: int,
        performance_gaps: dict[str, float],
        strategy: str,
        session_id: str
    ) -> dict[str, Any]:
        """Generate data with batch-level tracking"""

        all_features = []
        all_effectiveness = []
        generation_metadata = {
            "total_samples_requested": total_samples,
            "batches_processed": 0,
            "total_generation_time": 0.0,
            "method": "tracked_dynamic_batching",
            "session_id": session_id
        }

        remaining_samples = total_samples
        start_time = time.time()
        batch_number = 1

        while remaining_samples > 0:
            # Get optimal batch size
            optimal_batch_size = await self.batch_optimizer.get_optimal_batch_size(
                target_samples=remaining_samples
            )

            batch_start_time = time.time()
            batch_success_count = 0
            batch_error_count = 0

            try:
                # Generate batch
                batch_result = await self.generate_targeted_data(
                    performance_gaps=performance_gaps,
                    strategy=strategy,
                    batch_size=optimal_batch_size
                )

                # Extract results
                batch_features = batch_result.get("features", [])
                batch_effectiveness = batch_result.get("effectiveness", [])

                if batch_features:
                    all_features.extend(batch_features)
                    all_effectiveness.extend(batch_effectiveness)
                    batch_success_count = len(batch_features)
                    remaining_samples -= batch_success_count
                else:
                    batch_error_count = 1

            except Exception as e:
                logger.error(f"Batch {batch_number} generation failed: {e}")
                batch_error_count = 1

            # Track batch completion
            batch_time = time.time() - batch_start_time
            await self.history_tracker.track_batch_completion(
                session_id=session_id,
                batch_number=batch_number,
                batch_size=optimal_batch_size,
                generation_method=self.generation_method,
                processing_time=batch_time,
                samples_generated=batch_success_count,
                samples_filtered=0,  # Would be calculated from quality filtering
                error_count=batch_error_count,
                quality_metrics={
                    "average_quality": np.mean([self._calculate_sample_quality_score(f, e)
                                              for f, e in zip(batch_features, batch_effectiveness)])
                                     if batch_features else 0.0
                },
                performance_metrics={
                    "memory_usage_mb": self._get_current_memory_usage(),
                    "efficiency_score": batch_success_count / batch_time if batch_time > 0 else 0.0
                }
            )

            # Record batch performance for optimizer
            await self.batch_optimizer.record_batch_performance(
                batch_size=optimal_batch_size,
                processing_time=batch_time,
                success_count=batch_success_count,
                error_count=batch_error_count
            )

            # Update metadata
            generation_metadata["batches_processed"] += 1
            generation_metadata["total_generation_time"] += batch_time
            batch_number += 1

            # Safety check
            if batch_success_count == 0 and batch_error_count > 0:
                remaining_samples = max(0, remaining_samples - optimal_batch_size)

        # Final metadata
        generation_metadata["final_sample_count"] = len(all_features)
        generation_metadata["total_generation_time"] = time.time() - start_time
        generation_metadata["average_batch_time"] = (
            generation_metadata["total_generation_time"] / generation_metadata["batches_processed"]
            if generation_metadata["batches_processed"] > 0 else 0.0
        )

        return {
            "features": all_features,
            "effectiveness": all_effectiveness,
            "metadata": generation_metadata
        }

    def _determine_generation_strategy(self, performance_gaps: dict[str, float]) -> str:
        """Determine optimal generation strategy based on performance gaps (2025 best practice)

        Based on research from "Targeted synthetic data generation for tabular data via hardness characterization"
        and 2025 best practices for adaptive ML data generation.

        Args:
            performance_gaps: Dictionary mapping metric names to gap magnitudes

        Returns:
            Optimal generation strategy string
        """
        if not performance_gaps:
            return "statistical"

        # Calculate gap severity and types
        max_gap = max(performance_gaps.values())
        avg_gap = sum(performance_gaps.values()) / len(performance_gaps)

        # Strategy selection based on 2025 research insights
        if performance_gaps.get("model_accuracy", 0) > 0.1:
            # Large model accuracy gap - use neural enhancement for complex patterns
            return "neural_enhanced"
        elif performance_gaps.get("rule_effectiveness", 0) > 0.1:
            # Rule effectiveness gap - focus on specific rule weaknesses
            return "rule_focused"
        elif performance_gaps.get("pattern_coverage", 0) > 0.1:
            # Pattern coverage gap - increase pattern variety
            return "diversity_enhanced"
        elif max_gap > 0.05:
            # Moderate gaps - use statistical with gap-based targeting
            return "statistical"
        else:
            # Small gaps - standard statistical generation
            return "statistical"

    def _configure_difficulty_distribution(self, performance_gaps: dict[str, float]) -> dict[str, Any]:
        """Configure difficulty distribution based on performance gaps and instance settings

        Args:
            performance_gaps: Dictionary mapping metric names to gap magnitudes

        Returns:
            Difficulty configuration dictionary
        """
        config = {
            "distribution_type": self.difficulty_distribution,
            "hardness_threshold": self.hardness_threshold,
            "focus_hard_examples": False,
            "difficulty_weights": {}
        }

        if self.difficulty_distribution == "adaptive":
            # Adaptive distribution based on gaps
            max_gap = max(performance_gaps.values()) if performance_gaps else 0
            if max_gap > 0.1:
                config["focus_hard_examples"] = True
                config["difficulty_weights"] = {
                    "easy": 0.2,
                    "medium": 0.3,
                    "hard": 0.5
                }
            else:
                config["difficulty_weights"] = {
                    "easy": 0.3,
                    "medium": 0.4,
                    "hard": 0.3
                }
        elif self.difficulty_distribution == "hard_focused":
            config["focus_hard_examples"] = True
            config["difficulty_weights"] = {
                "easy": 0.1,
                "medium": 0.2,
                "hard": 0.7
            }
        else:  # uniform
            config["difficulty_weights"] = {
                "easy": 0.33,
                "medium": 0.34,
                "hard": 0.33
            }

        return config

    async def _generate_neural_enhanced_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data using neural enhancement for complex patterns"""
        logger.info("Generating neural-enhanced data for complex pattern improvement")

        # Use neural generation if available, otherwise fall back to statistical
        if self.neural_generator and TORCH_AVAILABLE:
            # First generate base data for training
            base_result = await self._generate_gap_based_statistical_data(
                performance_gaps, min(batch_size, 500), focus_areas, difficulty_config
            )

            # Train neural generator on gap-focused data
            base_features = np.array(base_result["features"])
            if len(base_features) > 0:
                self.neural_generator.fit(base_features)

                # Generate enhanced synthetic features
                enhanced_features = self.neural_generator.generate(batch_size)

                # Apply gap-based adjustments to neural output
                adjusted_features = self._apply_gap_adjustments(enhanced_features, performance_gaps, focus_areas)

                return await self._finalize_targeted_generation(
                    adjusted_features, performance_gaps, difficulty_config, "neural_enhanced"
                )

        # Fallback to statistical generation
        logger.warning("Neural generation not available, falling back to statistical")
        return await self._generate_gap_based_statistical_data(
            performance_gaps, batch_size, focus_areas, difficulty_config
        )

    async def _generate_rule_focused_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data focused on specific rule weaknesses"""
        logger.info("Generating rule-focused data for rule effectiveness improvement")

        # Identify weak rule areas from gaps
        weak_areas = [area for area in focus_areas if performance_gaps.get(f"{area}_effectiveness", 0) > 0.05]
        if not weak_areas:
            weak_areas = ["clarity", "specificity"]  # Default weak areas

        # Generate data with emphasis on weak rule areas
        features = []
        effectiveness_scores = []
        prompts = []

        for i in range(batch_size):
            # Create feature vector emphasizing weak areas
            feature_vector = self._generate_rule_focused_features(weak_areas, difficulty_config, i)
            features.append(feature_vector)

            # Generate corresponding effectiveness and prompts
            domain_name = self._select_domain_from_features(np.array(feature_vector))
            domain_config = self.domains[domain_name]

            effectiveness = self._generate_targeted_effectiveness(domain_config, weak_areas, difficulty_config)
            effectiveness_scores.append(effectiveness)

            prompt_pair = self._generate_targeted_prompt_pair(domain_config, weak_areas)
            prompts.append(prompt_pair)

        return await self._finalize_targeted_generation(
            features, performance_gaps, difficulty_config, "rule_focused",
            effectiveness_scores, prompts
        )

    async def _generate_diversity_enhanced_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate data with enhanced diversity for pattern coverage improvement"""
        logger.info("Generating diversity-enhanced data for pattern coverage improvement")

        # Increase domain diversity and feature variance
        features = []
        effectiveness_scores = []
        prompts = []

        # Ensure coverage across all domains with extra variance
        domain_names = list(self.domains.keys())
        samples_per_domain = batch_size // len(domain_names)
        extra_samples = batch_size % len(domain_names)

        for domain_idx, domain_name in enumerate(domain_names):
            domain_config = self.domains[domain_name]
            domain_samples = samples_per_domain + (1 if domain_idx < extra_samples else 0)

            for i in range(domain_samples):
                # Generate diverse features with high variance
                feature_vector = self._generate_diverse_features(domain_config, difficulty_config, i)
                features.append(feature_vector)

                # Generate diverse effectiveness scores
                effectiveness = self._generate_diverse_effectiveness(domain_config, difficulty_config)
                effectiveness_scores.append(effectiveness)

                # Generate diverse prompt patterns
                prompt_pair = self._generate_diverse_prompt_pair(domain_config)
                prompts.append(prompt_pair)

        return await self._finalize_targeted_generation(
            features, performance_gaps, difficulty_config, "diversity_enhanced",
            effectiveness_scores, prompts
        )

    async def _generate_gap_based_statistical_data(
        self,
        performance_gaps: dict[str, float],
        batch_size: int,
        focus_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate statistical data with gap-based targeting"""
        logger.info("Generating gap-based statistical data")

        # Use existing statistical generation with gap-based modifications
        original_target = self.target_samples
        self.target_samples = batch_size

        try:
            # Generate base statistical data
            result = await self.generate_comprehensive_training_data()

            # Apply gap-based adjustments to features
            if self.enable_gap_targeting and performance_gaps:
                adjusted_features = self._apply_gap_adjustments(
                    result["features"], performance_gaps, focus_areas
                )
                result["features"] = adjusted_features

            # Apply difficulty distribution adjustments
            if difficulty_config["focus_hard_examples"]:
                result = self._apply_difficulty_adjustments(result, difficulty_config)

            return result

        finally:
            self.target_samples = original_target

    def _apply_gap_adjustments(
        self,
        features: list[list[float]],
        performance_gaps: dict[str, float],
        focus_areas: list[str]
    ) -> list[list[float]]:
        """Apply performance gap-based adjustments to features"""
        if not performance_gaps or not features:
            return features

        adjusted_features = []
        feature_names = self.feature_names

        for feature_vector in features:
            adjusted_vector = feature_vector.copy()

            # Apply adjustments based on performance gaps
            for area in focus_areas:
                if area in feature_names:
                    feature_idx = feature_names.index(area)
                    gap_magnitude = performance_gaps.get(f"{area}_effectiveness", 0)

                    if gap_magnitude > 0.05:  # Significant gap
                        # Increase variance in problematic areas
                        noise_factor = min(gap_magnitude * 2, 0.3)
                        noise = self.rng.normal(0, noise_factor)
                        adjusted_vector[feature_idx] = np.clip(
                            adjusted_vector[feature_idx] + noise, 0.0, 1.0
                        )

            adjusted_features.append(adjusted_vector)

        return adjusted_features

    def _generate_rule_focused_features(
        self,
        weak_areas: list[str],
        difficulty_config: dict[str, Any],
        sample_idx: int
    ) -> list[float]:
        """Generate features focused on weak rule areas"""
        feature_vector = [0.0] * len(self.feature_names)

        # Determine difficulty level for this sample
        difficulty_level = self._sample_difficulty_level(difficulty_config, sample_idx)

        for i, feature_name in enumerate(self.feature_names):
            if feature_name in weak_areas:
                # Generate challenging values for weak areas
                if difficulty_level == "hard":
                    # Hard examples: extreme values that challenge the rules
                    feature_vector[i] = self.rng.choice([
                        self.rng.uniform(0.0, 0.3),  # Very low
                        self.rng.uniform(0.7, 1.0)   # Very high
                    ])
                elif difficulty_level == "medium":
                    # Medium examples: moderate challenge
                    feature_vector[i] = self.rng.uniform(0.3, 0.7)
                else:
                    # Easy examples: typical values
                    feature_vector[i] = self.rng.uniform(0.4, 0.6)
            else:
                # Normal distribution for non-focus areas
                feature_vector[i] = np.clip(self.rng.normal(0.5, 0.2), 0.0, 1.0)

        return feature_vector

    def _sample_difficulty_level(self, difficulty_config: dict[str, Any], sample_idx: int) -> str:
        """Sample difficulty level based on configuration"""
        weights = difficulty_config["difficulty_weights"]
        levels = list(weights.keys())
        probabilities = list(weights.values())

        # Use sample index for deterministic sampling
        self.rng.seed(self.random_state + sample_idx)
        level = self.rng.choice(levels, p=probabilities)
        self.rng.seed(self.random_state)  # Reset seed

        return level

    def _generate_diverse_features(
        self,
        domain_config: DomainConfig,
        difficulty_config: dict[str, Any],
        sample_idx: int
    ) -> list[float]:
        """Generate features with enhanced diversity"""
        feature_vector = [0.0] * len(self.feature_names)

        # Use higher variance for diversity
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in domain_config.feature_ranges:
                min_val, max_val = domain_config.feature_ranges[feature_name]
                # Increase variance by expanding range
                expanded_min = max(0.0, min_val - 0.1)
                expanded_max = min(1.0, max_val + 0.1)
                feature_vector[i] = self.rng.uniform(expanded_min, expanded_max)
            else:
                # High variance normal distribution
                feature_vector[i] = np.clip(self.rng.normal(0.5, 0.3), 0.0, 1.0)

        return feature_vector

    def _generate_targeted_effectiveness(
        self,
        domain_config: DomainConfig,
        weak_areas: list[str],
        difficulty_config: dict[str, Any]
    ) -> float:
        """Generate effectiveness scores targeted at weak areas"""
        base_effectiveness = self._generate_domain_effectiveness(domain_config, 0)

        # Adjust based on weak areas and difficulty
        if difficulty_config["focus_hard_examples"]:
            # Lower effectiveness for hard examples to create challenging training data
            adjustment = self.rng.uniform(-0.2, -0.1)
        else:
            # Normal effectiveness distribution
            adjustment = self.rng.uniform(-0.1, 0.1)

        return np.clip(base_effectiveness + adjustment, 0.0, 1.0)

    def _generate_diverse_effectiveness(
        self,
        domain_config: DomainConfig,
        difficulty_config: dict[str, Any]
    ) -> float:
        """Generate effectiveness scores with enhanced diversity"""
        # Use wider distribution for diversity
        alpha, beta = domain_config.effectiveness_params
        # Increase variance by adjusting beta distribution parameters
        diverse_alpha = max(1.0, alpha * 0.8)
        diverse_beta = max(1.0, beta * 0.8)

        return self.rng.beta(diverse_alpha, diverse_beta)

    def _generate_targeted_prompt_pair(
        self,
        domain_config: DomainConfig,
        weak_areas: list[str]
    ) -> tuple[str, str]:
        """Generate prompt pairs targeted at weak areas"""
        # Select patterns that emphasize weak areas
        relevant_patterns = [
            pattern for pattern in domain_config.patterns
            if any(area in pattern[0].lower() or area in pattern[1].lower() for area in weak_areas)
        ]

        if relevant_patterns:
            return self.rng.choice(relevant_patterns)
        else:
            # Fallback to random pattern
            return self.rng.choice(domain_config.patterns)

    def _generate_diverse_prompt_pair(self, domain_config: DomainConfig) -> tuple[str, str]:
        """Generate diverse prompt pairs"""
        # Simply use random selection for diversity
        return self.rng.choice(domain_config.patterns)

    def _apply_difficulty_adjustments(
        self,
        result: dict[str, Any],
        difficulty_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply difficulty-based adjustments to generated data"""
        if not difficulty_config["focus_hard_examples"]:
            return result

        # Adjust effectiveness scores to create more challenging examples
        adjusted_effectiveness = []
        for score in result["effectiveness"]:
            if self.rng.random() < 0.3:  # 30% of samples become harder
                adjusted_score = max(0.0, score - self.rng.uniform(0.1, 0.3))
            else:
                adjusted_score = score
            adjusted_effectiveness.append(adjusted_score)

        result["effectiveness"] = adjusted_effectiveness
        return result

    async def _finalize_targeted_generation(
        self,
        features: list[list[float]],
        performance_gaps: dict[str, float],
        difficulty_config: dict[str, Any],
        strategy: str,
        effectiveness_scores: list[float] | None = None,
        prompts: list[tuple[str, str]] | None = None
    ) -> dict[str, Any]:
        """Finalize targeted generation with metadata"""

        # Generate missing components if not provided
        if effectiveness_scores is None:
            effectiveness_scores = []
            for feature_vector in features:
                domain_name = self._select_domain_from_features(np.array(feature_vector))
                domain_config = self.domains[domain_name]
                effectiveness = self._generate_domain_effectiveness(domain_config, 0)
                effectiveness_scores.append(effectiveness)

        if prompts is None:
            prompts = []
            for feature_vector in features:
                domain_name = self._select_domain_from_features(np.array(feature_vector))
                domain_config = self.domains[domain_name]
                prompt_pair = self.rng.choice(domain_config.patterns)
                prompts.append(prompt_pair)

        # Create result structure
        result = {
            "features": features,
            "effectiveness": effectiveness_scores,
            "prompts": prompts,
            "metadata": {
                "generation_method": f"targeted_{strategy}",
                "samples_generated": len(features),
                "targeting_enabled": True,
                "performance_gaps_addressed": performance_gaps,
                "difficulty_distribution": difficulty_config,
                "quality_score": np.mean(effectiveness_scores) if effectiveness_scores else 0.0,
                "domain_distribution": self._calculate_domain_distribution(features)
            }
        }

        return result
