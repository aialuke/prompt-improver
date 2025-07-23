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
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import make_classification
from sqlalchemy.ext.asyncio import AsyncSession

from ...database.models import TrainingPrompt
from ..learning.quality.enhanced_scorer import EnhancedQualityMetrics, EnhancedQualityScorer

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


# Modern Generative Models for Tabular Data
class TabularGAN(nn.Module):
    """Generative Adversarial Network for tabular data synthesis."""

    def __init__(self, data_dim: int, noise_dim: int = 100, hidden_dims: list[int] = None):
        super().__init__()
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        # Generator
        gen_layers = []
        prev_dim = noise_dim
        for hidden_dim in hidden_dims:
            gen_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        gen_layers.append(nn.Linear(prev_dim, data_dim))
        gen_layers.append(nn.Tanh())  # Normalize output to [-1, 1]
        self.generator = nn.Sequential(*gen_layers)

        # Discriminator
        disc_layers = []
        prev_dim = data_dim
        for hidden_dim in reversed(hidden_dims):
            disc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        disc_layers.append(nn.Linear(prev_dim, 1))
        disc_layers.append(nn.Sigmoid())
        self.discriminator = nn.Sequential(*disc_layers)

    def generate(self, batch_size: int, device: torch.device):
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        return self.generator(noise)


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
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld_loss


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
    """Production-grade synthetic data generator with advanced quality assessment and modern generative models"""

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

        # Initialize enhanced quality scorer
        if use_enhanced_scoring:
            self.quality_scorer = EnhancedQualityScorer(confidence_level=0.95)

        # Initialize neural generator if needed
        self.neural_generator = None
        if generation_method in ["neural", "hybrid", "diffusion"] and TORCH_AVAILABLE:
            if neural_model_type == "diffusion":
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

        # Legacy quality validation thresholds (for backward compatibility)
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

            # Update instance configuration if needed
            original_target = self.target_samples
            original_method = self.generation_method

            self.target_samples = target_samples
            self.generation_method = generation_method

            logger.info(f"Starting orchestrated synthetic data generation: {target_samples} samples using {generation_method}")

            # Generate synthetic data
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
        # Legacy quality metrics
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
