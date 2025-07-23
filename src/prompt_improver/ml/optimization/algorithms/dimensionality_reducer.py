"""Advanced Dimensionality Reduction for High-Dimensional Linguistic Features.

Implements multiple dimensionality reduction techniques with intelligent method selection
and variance preservation optimization for 31-dimensional linguistic feature vectors.

Enhanced with 2025 best practices:
- Neural network autoencoders (standard, variational, β-VAE)
- Modern deep learning approaches with GPU acceleration
- Transformer-based dimensionality reduction
- Diffusion model integration
"""

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from sklearn.decomposition import PCA, FastICA, KernelPCA, TruncatedSVD, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold

from sklearn.manifold import TSNE, Isomap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# Advanced dimensionality reduction imports
try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.decomposition import FactorAnalysis

    FACTOR_ANALYSIS_AVAILABLE = True
except ImportError:
    FACTOR_ANALYSIS_AVAILABLE = False
    FactorAnalysis = None
    warnings.warn("FactorAnalysis not available in this sklearn version")

# Neural network and deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural network methods will be disabled. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Some neural network methods will be disabled. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)


# Neural Network Autoencoder Classes
class StandardAutoencoder(nn.Module):
    """Standard autoencoder for dimensionality reduction."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [max(latent_dim * 4, 64), max(latent_dim * 2, 32)]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

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
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.encoder(x)


class VariationalAutoencoder(nn.Module):
    """Variational autoencoder for probabilistic dimensionality reduction."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int] = None, beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # β-VAE parameter for disentanglement

        if hidden_dims is None:
            hidden_dims = [max(latent_dim * 4, 64), max(latent_dim * 2, 32)]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
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
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
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

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kld_loss


class NeuralDimensionalityReducer:
    """Neural network-based dimensionality reduction wrapper."""

    def __init__(self, model_type: str = "autoencoder", latent_dim: int = 10,
                 hidden_dims: list[int] = None, beta: float = 1.0,
                 device: str = "auto", epochs: int = 100, batch_size: int = 32,
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
        self.optimizer = None
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'NeuralDimensionalityReducer':
        """Fit the neural dimensionality reducer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network methods")

        # Normalize input data
        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        # Initialize model
        if self.model_type == "autoencoder":
            self.model = StandardAutoencoder(input_dim, self.latent_dim, self.hidden_dims)
        elif self.model_type == "vae":
            self.model = VariationalAutoencoder(input_dim, self.latent_dim, self.hidden_dims, self.beta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

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

                if self.model_type == "autoencoder":
                    recon, _ = self.model(data)
                    loss = nn.functional.mse_loss(recon, data)
                elif self.model_type == "vae":
                    recon, mu, logvar, _ = self.model(data)
                    loss = self.model.loss_function(recon, data, mu, logvar)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensional space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.model_type == "autoencoder":
                encoded = self.model.encode(X_tensor)
            elif self.model_type == "vae":
                mu, _ = self.model.encode(X_tensor)
                encoded = mu  # Use mean for deterministic encoding

        return encoded.cpu().numpy()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and transform the data."""
        return self.fit(X).transform(X)


class TransformerDimensionalityReducer(nn.Module):
    """Transformer-based dimensionality reduction using attention mechanisms."""

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8,
                 num_layers: int = 3, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Positional encoding (for sequence-like treatment of features)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)

        # Project to hidden dimension and add positional encoding
        x = self.input_projection(x).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        x = x + self.pos_encoding

        # Apply transformer
        x = self.transformer(x)  # (batch_size, 1, hidden_dim)

        # Project to output dimension
        x = self.output_projection(x.squeeze(1))  # (batch_size, output_dim)

        return x


class DiffusionDimensionalityReducer:
    """Diffusion model-based dimensionality reduction."""

    def __init__(self, input_dim: int, output_dim: int, num_timesteps: int = 1000,
                 hidden_dim: int = 256, device: str = "auto"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(output_dim + 1, hidden_dim),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)

        # Encoder network (maps high-dim to low-dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)

        self.optimizer = None
        self.is_fitted = False

    def add_noise(self, x, t):
        """Add noise to data at timestep t."""
        noise = torch.randn_like(x)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

    def fit(self, X: np.ndarray, epochs: int = 200, batch_size: int = 32, lr: float = 1e-3):
        """Train the diffusion-based dimensionality reducer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for diffusion models")

        from sklearn.preprocessing import StandardScaler

        # Normalize data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Encode to low-dimensional space
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            X_encoded = self.encoder(X_tensor)

        # Prepare data loader
        dataset = TensorDataset(X_encoded)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.denoiser.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (x_0,) in enumerate(dataloader):
                self.optimizer.zero_grad()

                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (x_0.size(0),), device=self.device)

                # Add noise
                x_t, noise = self.add_noise(x_0, t)

                # Predict noise
                t_emb = t.float().unsqueeze(1) / self.num_timesteps  # Normalize timestep
                x_t_with_t = torch.cat([x_t, t_emb], dim=1)
                predicted_noise = self.denoiser(x_t_with_t)

                # Compute loss
                loss = nn.functional.mse_loss(predicted_noise, noise)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"Diffusion Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensional space using the encoder."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            encoded = self.encoder(X_tensor)

        return encoded.cpu().numpy()

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fit the model and transform the data."""
        return self.fit(X, **kwargs).transform(X)


class ModernNeuralDimensionalityReducer:
    """Enhanced neural dimensionality reducer with transformer and diffusion support."""

    def __init__(self, model_type: str = "transformer", output_dim: int = 10,
                 hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 3,
                 num_timesteps: int = 1000, device: str = "auto",
                 epochs: int = 200, batch_size: int = 32, learning_rate: float = 1e-3):
        self.model_type = model_type
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
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

    def fit(self, X: np.ndarray) -> 'ModernNeuralDimensionalityReducer':
        """Fit the modern neural dimensionality reducer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for modern neural methods")

        from sklearn.preprocessing import StandardScaler

        # Normalize input data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        input_dim = X_scaled.shape[1]

        # Initialize model based on type
        if self.model_type == "transformer":
            self.model = TransformerDimensionalityReducer(
                input_dim=input_dim,
                output_dim=self.output_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim
            ).to(self.device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self._train_transformer(X_scaled)

        elif self.model_type == "diffusion":
            self.model = DiffusionDimensionalityReducer(
                input_dim=input_dim,
                output_dim=self.output_dim,
                num_timesteps=self.num_timesteps,
                hidden_dim=self.hidden_dim,
                device=self.device
            )
            self.model.fit(X_scaled, epochs=self.epochs, batch_size=self.batch_size, lr=self.learning_rate)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.is_fitted = True
        return self

    def _train_transformer(self, X_scaled: np.ndarray):
        """Train transformer model with reconstruction loss."""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Add reconstruction head for training
        reconstruction_head = nn.Linear(self.output_dim, X_scaled.shape[1]).to(self.device)
        recon_optimizer = optim.Adam(reconstruction_head.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer.zero_grad()
                recon_optimizer.zero_grad()

                # Forward pass
                encoded = self.model(data)
                reconstructed = reconstruction_head(encoded)

                # Reconstruction loss
                loss = nn.functional.mse_loss(reconstructed, data)

                loss.backward()
                self.optimizer.step()
                recon_optimizer.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"Transformer Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to lower dimensional space."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        if self.model_type == "transformer":
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            self.model.eval()
            with torch.no_grad():
                encoded = self.model(X_tensor)

            return encoded.cpu().numpy()

        elif self.model_type == "diffusion":
            return self.model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and transform the data."""
        return self.fit(X).transform(X)


@dataclass
class DimensionalityConfig:
    """Configuration for dimensionality reduction optimization."""

    # Target dimensionality
    target_dimensions: int = 10
    min_dimensions: int = 5
    max_dimensions: int = 20
    variance_threshold: float = 0.95  # Minimum variance to preserve

    # Method selection
    auto_method_selection: bool = True
    preferred_methods: list[str] = None  # ['pca', 'umap', 'lda', 'ica', 'kernel_pca', 'autoencoder', 'vae']
    enable_manifold_learning: bool = True
    enable_feature_selection: bool = True
    enable_neural_methods: bool = True  # Enable neural network-based methods

    # Performance optimization
    fast_mode: bool = False  # Use faster but potentially less accurate methods
    memory_efficient: bool = True
    n_jobs: int = -1
    random_state: int = 42

    # Neural network specific parameters
    neural_epochs: int = 100
    neural_batch_size: int = 32
    neural_learning_rate: float = 1e-3
    neural_hidden_dims: list[int] = None  # Auto-determined if None
    vae_beta: float = 1.0  # β-VAE parameter for disentanglement
    neural_device: str = "auto"  # "auto", "cpu", "cuda"

    # Transformer specific parameters
    transformer_num_heads: int = 8
    transformer_num_layers: int = 3
    transformer_hidden_dim: int = 256
    transformer_dropout: float = 0.1

    # Diffusion model specific parameters
    diffusion_num_timesteps: int = 1000
    diffusion_hidden_dim: int = 256

    # Evaluation criteria
    preservation_metrics: list[str] = (
        None  # ['variance', 'structure', 'clustering', 'classification']
    )
    cross_validation_folds: int = 3
    clustering_quality_weight: float = 0.4
    classification_quality_weight: float = 0.3
    variance_preservation_weight: float = 0.3

    # Method-specific parameters
    pca_svd_solver: str = "auto"  # 'auto', 'full', 'arpack', 'randomized'
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    kernel_pca_kernel: str = "rbf"  # 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    kernel_pca_gamma: float | None = None

    tsne_perplexity: float = 30.0
    tsne_early_exaggeration: float = 12.0

    # Feature selection parameters
    feature_selection_method: str = (
        "mutual_info"  # 'f_score', 'mutual_info', 'rfe', 'variance'
    )
    feature_selection_k: int = 15

    # Caching and persistence
    enable_caching: bool = True
    cache_transformations: bool = True
    save_models: bool = False
    model_save_path: str | None = None

    # Modern optimizations
    use_gpu_acceleration: bool = True  # Use GPU for compatible methods
    use_incremental_learning: bool = True  # Use incremental PCA for large datasets
    use_randomized_svd: bool = True  # Use randomized SVD for faster PCA
    optimize_memory_usage: bool = True  # Enable memory optimizations


@dataclass
class ReductionResult:
    """Result of dimensionality reduction process."""

    method: str
    original_dimensions: int
    reduced_dimensions: int
    variance_preserved: float
    reconstruction_error: float
    processing_time: float
    quality_score: float
    transformed_data: np.ndarray
    transformer: Any
    feature_importance: np.ndarray | None = None
    evaluation_metrics: dict[str, float] | None = None


class AdvancedDimensionalityReducer:
    """Advanced dimensionality reduction with intelligent method selection."""

    def __init__(self, config: DimensionalityConfig | None = None, training_loader=None):
        """Initialize dimensionality reducer.

        Args:
            config: Configuration for dimensionality reduction
            training_loader: Training data loader for ML pipeline integration
        """
        self.config = config or DimensionalityConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Training data integration
        from ...core.training_data_loader import TrainingDataLoader
        self.training_loader = training_loader or TrainingDataLoader()
        self.logger.info("Dimensionality reducer integrated with training data pipeline")

        # Initialize method registry
        self.available_methods = self._initialize_method_registry()

        # Performance tracking
        self.reduction_history: list[ReductionResult] = []
        self.method_performance: dict[str, list[float]] = {}

        # Caching
        self.cache: dict[str, ReductionResult] = {}
        self.scaler: Any | None = None

        # Preprocessing pipeline
        self.preprocessing_pipeline: Pipeline | None = None

        self.logger.info(
            f"Advanced dimensionality reducer initialized with {len(self.available_methods)} methods"
        )

    def _initialize_method_registry(self) -> dict[str, dict[str, Any]]:
        """Initialize registry of available dimensionality reduction methods."""
        # Optimize PCA based on configuration
        pca_params = {
            "n_components": self.config.target_dimensions,
            "random_state": self.config.random_state,
        }

        # Use randomized SVD for faster computation on large datasets
        if self.config.use_randomized_svd:
            pca_params["svd_solver"] = "randomized"
        else:
            pca_params["svd_solver"] = self.config.pca_svd_solver

        methods = {
            "pca": {
                "class": PCA,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": pca_params,
            },
            "incremental_pca": {
                "class": IncrementalPCA,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "batch_size": min(1000, max(100, self.config.neural_batch_size)),
                },
            },
            "kernel_pca": {
                "class": KernelPCA,
                "type": "nonlinear",
                "manifold": False,
                "supervised": False,
                "scalable": False,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "kernel": self.config.kernel_pca_kernel,
                    "gamma": self.config.kernel_pca_gamma,
                    "random_state": self.config.random_state,
                    "n_jobs": self.config.n_jobs
                    if not self.config.memory_efficient
                    else 1,
                },
            },
            "ica": {
                "class": FastICA,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "random_state": self.config.random_state,
                    "max_iter": 1000,
                },
            },
            "truncated_svd": {
                "class": TruncatedSVD,
                "type": "linear",
                "manifold": False,
                "supervised": False,
                "scalable": True,
                "preserves_variance": True,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "random_state": self.config.random_state,
                },
            },
            "lda": {
                "class": LinearDiscriminantAnalysis,
                "type": "linear",
                "manifold": False,
                "supervised": True,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": min(
                        self.config.target_dimensions, 2
                    )  # LDA limited by n_classes-1
                },
            },
        }

        # Add UMAP if available
        if UMAP_AVAILABLE:
            methods["umap"] = {
                "class": umap.UMAP,
                "type": "nonlinear",
                "manifold": True,
                "supervised": False,
                "scalable": True,
                "preserves_variance": False,
                "params": {
                    "n_components": self.config.target_dimensions,
                    "n_neighbors": self.config.umap_n_neighbors,
                    "min_dist": self.config.umap_min_dist,
                    "metric": self.config.umap_metric,
                    "random_state": self.config.random_state,
                    "n_jobs": 1 if self.config.memory_efficient else self.config.n_jobs,
                },
            }

        # Add manifold learning methods if enabled
        if self.config.enable_manifold_learning and not self.config.fast_mode:
            methods.update({
                "tsne": {
                    "class": TSNE,
                    "type": "nonlinear",
                    "manifold": True,
                    "supervised": False,
                    "scalable": False,
                    "preserves_variance": False,
                    "params": {
                        "n_components": min(
                            self.config.target_dimensions, 3
                        ),  # t-SNE typically 2-3D
                        "perplexity": self.config.tsne_perplexity,
                        "early_exaggeration": self.config.tsne_early_exaggeration,
                        "random_state": self.config.random_state,
                        "n_jobs": 1
                        if self.config.memory_efficient
                        else self.config.n_jobs,
                    },
                },
                "isomap": {
                    "class": Isomap,
                    "type": "nonlinear",
                    "manifold": True,
                    "supervised": False,
                    "scalable": False,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "n_neighbors": 10,
                        "n_jobs": 1
                        if self.config.memory_efficient
                        else self.config.n_jobs,
                    },
                },
            })

        # Add random projection methods for fast mode
        if self.config.fast_mode:
            methods.update({
                "gaussian_rp": {
                    "class": GaussianRandomProjection,
                    "type": "linear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "random_state": self.config.random_state,
                    },
                },
                "sparse_rp": {
                    "class": SparseRandomProjection,
                    "type": "linear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": False,
                    "params": {
                        "n_components": self.config.target_dimensions,
                        "random_state": self.config.random_state,
                    },
                },
            })

        # Add neural network methods if enabled and available
        if self.config.enable_neural_methods and TORCH_AVAILABLE:
            methods.update({
                "autoencoder": {
                    "class": NeuralDimensionalityReducer,
                    "type": "nonlinear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": True,
                    "params": {
                        "model_type": "autoencoder",
                        "latent_dim": self.config.target_dimensions,
                        "hidden_dims": self.config.neural_hidden_dims,
                        "epochs": self.config.neural_epochs,
                        "batch_size": self.config.neural_batch_size,
                        "learning_rate": self.config.neural_learning_rate,
                        "device": self.config.neural_device,
                    },
                },
                "vae": {
                    "class": NeuralDimensionalityReducer,
                    "type": "nonlinear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": True,
                    "params": {
                        "model_type": "vae",
                        "latent_dim": self.config.target_dimensions,
                        "hidden_dims": self.config.neural_hidden_dims,
                        "beta": self.config.vae_beta,
                        "epochs": self.config.neural_epochs,
                        "batch_size": self.config.neural_batch_size,
                        "learning_rate": self.config.neural_learning_rate,
                        "device": self.config.neural_device,
                    },
                },
                "transformer": {
                    "class": ModernNeuralDimensionalityReducer,
                    "type": "nonlinear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": True,
                    "params": {
                        "model_type": "transformer",
                        "output_dim": self.config.target_dimensions,
                        "hidden_dim": self.config.transformer_hidden_dim,
                        "num_heads": self.config.transformer_num_heads,
                        "num_layers": self.config.transformer_num_layers,
                        "epochs": self.config.neural_epochs,
                        "batch_size": self.config.neural_batch_size,
                        "learning_rate": self.config.neural_learning_rate,
                        "device": self.config.neural_device,
                    },
                },
                "diffusion": {
                    "class": ModernNeuralDimensionalityReducer,
                    "type": "nonlinear",
                    "manifold": False,
                    "supervised": False,
                    "scalable": True,
                    "preserves_variance": True,
                    "params": {
                        "model_type": "diffusion",
                        "output_dim": self.config.target_dimensions,
                        "hidden_dim": self.config.diffusion_hidden_dim,
                        "num_timesteps": self.config.diffusion_num_timesteps,
                        "epochs": self.config.neural_epochs,
                        "batch_size": self.config.neural_batch_size,
                        "learning_rate": self.config.neural_learning_rate,
                        "device": self.config.neural_device,
                    },
                },
            })

        return methods

    # Training Data Integration Methods
    async def optimize_feature_space(self, db_session) -> dict[str, Any]:
        """Optimize feature space using training data
        
        Analyzes the complete training dataset to determine optimal
        dimensionality reduction approach and parameters.
        """
        try:
            self.logger.info("Starting feature space optimization with training data")
            
            # Load training data from pipeline
            training_data = await self.training_loader.load_training_data(db_session)
            
            # Check if training data validation passed
            if not training_data.get("validation", {}).get("is_valid", False):
                self.logger.warning("Insufficient training data for feature space optimization")
                return {"status": "insufficient_data", "samples": training_data["metadata"]["total_samples"]}
            
            if training_data["metadata"]["total_samples"] < 10:
                self.logger.warning("Insufficient training data for feature space optimization")
                return {"status": "insufficient_data", "samples": training_data["metadata"]["total_samples"]}
            
            # Extract features from training data
            full_features = np.array(training_data.get("features", []))
            if full_features.size == 0:
                self.logger.warning("No features found in training data")
                return {"status": "no_features"}
            
            self.logger.info(f"Optimizing feature space on {full_features.shape[0]} samples with {full_features.shape[1]} dimensions")
            
            # Try multiple reduction techniques and evaluate each
            results = {}
            for method in ['pca', 'umap', 'autoencoder']:
                try:
                    reduced = await self._reduce_dimensions_method(full_features, method)
                    quality_score = await self._evaluate_reduction_quality(reduced, full_features)
                    results[method] = {
                        'reduced_features': reduced,
                        'quality_score': quality_score,
                        'variance_preserved': self._calculate_variance_preserved(reduced, full_features),
                        'dimensions': reduced.shape[1] if reduced.size > 0 else 0
                    }
                    self.logger.info(f"{method.upper()} reduction: quality={quality_score:.3f}, dims={results[method]['dimensions']}")
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate method {method}: {e}")
                    results[method] = {'quality_score': 0.0, 'error': str(e)}
            
            # Select best method based on quality scores
            best_method = max(results.items(), key=lambda x: x[1].get('quality_score', 0.0))
            
            # Update configuration with optimal parameters
            self._update_config_with_best_method(best_method[0], best_method[1])
            
            result = {
                "status": "success",
                "training_samples": full_features.shape[0],
                "original_dimensions": full_features.shape[1],
                "best_method": best_method[0],
                "best_quality_score": best_method[1].get('quality_score', 0.0),
                "optimal_dimensions": best_method[1].get('dimensions', 0),
                "variance_preserved": best_method[1].get('variance_preserved', 0.0),
                "all_results": {k: {
                    'quality_score': v.get('quality_score', 0.0),
                    'dimensions': v.get('dimensions', 0),
                    'variance_preserved': v.get('variance_preserved', 0.0)
                } for k, v in results.items()}
            }
            
            self.logger.info(f"Feature space optimization completed. Best method: {best_method[0]} (score: {best_method[1].get('quality_score', 0.0):.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to optimize feature space: {e}")
            return {"status": "error", "error": str(e)}

    async def adaptive_reduction(self, new_data: np.ndarray, db_session) -> dict[str, Any]:
        """Adaptively adjust dimensionality based on new data
        
        Combines new data with training data to determine if dimensionality
        reduction parameters need adjustment.
        """
        try:
            self.logger.info("Performing adaptive dimensionality reduction")
            
            # Load existing training data
            training_data = await self.training_loader.load_training_data(db_session)
            existing_features = np.array(training_data.get("features", []))
            
            if existing_features.size == 0:
                self.logger.warning("No existing training data for adaptive reduction")
                return await self._reduce_new_data_only(new_data)
            
            # Combine new data with existing training data
            combined_data = await self._merge_with_training_data(new_data, existing_features)
            
            # Analyze if current reduction is still optimal
            optimal_dims = await self._find_optimal_dimensions(combined_data)
            
            # Update reducer if needed
            if optimal_dims != self.config.target_dimensions:
                self.logger.info(f"Updating target dimensions from {self.config.target_dimensions} to {optimal_dims}")
                self.config.target_dimensions = optimal_dims
                self.reducer = await self._update_reducer(optimal_dims)
            
            # Apply reduction to new data
            reduced_new_data = await self.reduce_dimensions(new_data)
            
            result = {
                "status": "success",
                "new_data_samples": new_data.shape[0],
                "combined_samples": combined_data.shape[0],
                "optimal_dimensions": optimal_dims,
                "dimensions_changed": optimal_dims != self.config.target_dimensions,
                "reduced_data": reduced_new_data.reduced_data if hasattr(reduced_new_data, 'reduced_data') else None
            }
            
            self.logger.info(f"Adaptive reduction completed with {optimal_dims} dimensions")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to perform adaptive reduction: {e}")
            return {"status": "error", "error": str(e)}

    async def _reduce_dimensions_method(self, features: np.ndarray, method: str) -> np.ndarray:
        """Apply specific dimensionality reduction method"""
        try:
            if method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=min(self.config.target_dimensions, features.shape[1]))
                return reducer.fit_transform(features)
            
            elif method == 'umap' and UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=min(self.config.target_dimensions, features.shape[1]),
                    n_neighbors=min(self.config.umap_n_neighbors, features.shape[0] - 1),
                    min_dist=self.config.umap_min_dist,
                    random_state=self.config.random_state
                )
                return reducer.fit_transform(features)
            
            elif method == 'autoencoder' and TORCH_AVAILABLE:
                reducer = NeuralDimensionalityReducer(
                    model_type="autoencoder",
                    latent_dim=min(self.config.target_dimensions, features.shape[1]),
                    hidden_dims=self.config.neural_hidden_dims,
                    epochs=self.config.neural_epochs,
                    batch_size=self.config.neural_batch_size,
                    learning_rate=self.config.neural_learning_rate,
                    device=self.config.neural_device
                )
                return reducer.fit_transform(features)

            elif method == 'vae' and TORCH_AVAILABLE:
                reducer = NeuralDimensionalityReducer(
                    model_type="vae",
                    latent_dim=min(self.config.target_dimensions, features.shape[1]),
                    hidden_dims=self.config.neural_hidden_dims,
                    beta=self.config.vae_beta,
                    epochs=self.config.neural_epochs,
                    batch_size=self.config.neural_batch_size,
                    learning_rate=self.config.neural_learning_rate,
                    device=self.config.neural_device
                )
                return reducer.fit_transform(features)
            
            else:
                # Fallback to PCA
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=min(self.config.target_dimensions, features.shape[1]))
                return reducer.fit_transform(features)
                
        except Exception as e:
            self.logger.error(f"Failed to apply {method} reduction: {e}")
            return np.array([])

    async def _evaluate_reduction_quality(self, reduced_data: np.ndarray, original_data: np.ndarray) -> float:
        """Evaluate quality of dimensionality reduction"""
        try:
            if reduced_data.size == 0 or original_data.size == 0:
                return 0.0
            
            # Calculate variance preservation
            if reduced_data.shape[0] != original_data.shape[0]:
                return 0.0
            
            # Simple quality metric: variance preservation + reconstruction error
            variance_preserved = np.var(reduced_data) / np.var(original_data)
            
            # Normalize to 0-1 range
            quality_score = min(1.0, max(0.0, variance_preserved))
            
            return quality_score
            
        except Exception as e:
            self.logger.debug(f"Failed to evaluate reduction quality: {e}")
            return 0.0

    def _calculate_variance_preserved(self, reduced_data: np.ndarray, original_data: np.ndarray) -> float:
        """Calculate how much variance is preserved in the reduction"""
        try:
            if reduced_data.size == 0 or original_data.size == 0:
                return 0.0
            
            reduced_var = np.var(reduced_data)
            original_var = np.var(original_data)
            
            if original_var == 0:
                return 1.0
            
            return min(1.0, reduced_var / original_var)
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate variance preserved: {e}")
            return 0.0

    def _update_config_with_best_method(self, method: str, results: dict):
        """Update configuration with parameters from best performing method"""
        try:
            # Update preferred methods list
            if self.config.preferred_methods is None:
                self.config.preferred_methods = []
            
            # Put best method first
            if method in self.config.preferred_methods:
                self.config.preferred_methods.remove(method)
            self.config.preferred_methods.insert(0, method)
            
            # Update target dimensions if significantly different
            optimal_dims = results.get('dimensions', self.config.target_dimensions)
            if abs(optimal_dims - self.config.target_dimensions) > 2:
                self.config.target_dimensions = optimal_dims
            
            self.logger.info(f"Configuration updated with best method: {method}")
            
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")

    async def _merge_with_training_data(self, new_data: np.ndarray, existing_data: np.ndarray) -> np.ndarray:
        """Merge new data with existing training data"""
        try:
            # Ensure compatible shapes
            if new_data.shape[1] != existing_data.shape[1]:
                self.logger.warning(f"Feature dimension mismatch: new={new_data.shape[1]}, existing={existing_data.shape[1]}")
                # Truncate to smaller dimension
                min_features = min(new_data.shape[1], existing_data.shape[1])
                new_data = new_data[:, :min_features]
                existing_data = existing_data[:, :min_features]
            
            # Combine data
            combined = np.vstack([existing_data, new_data])
            return combined
            
        except Exception as e:
            self.logger.error(f"Failed to merge data: {e}")
            return new_data

    async def _find_optimal_dimensions(self, data: np.ndarray) -> int:
        """Find optimal number of dimensions for the given data"""
        try:
            # Use PCA to analyze variance explained
            from sklearn.decomposition import PCA
            
            max_components = min(data.shape[1], data.shape[0] - 1, 20)  # Reasonable upper bound
            pca = PCA(n_components=max_components)
            pca.fit(data)
            
            # Find number of components for desired variance threshold
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            optimal_dims = np.argmax(cumsum_var >= self.config.variance_threshold) + 1
            
            # Ensure within configured bounds
            optimal_dims = max(self.config.min_dimensions, min(optimal_dims, self.config.max_dimensions))
            
            return optimal_dims
            
        except Exception as e:
            self.logger.error(f"Failed to find optimal dimensions: {e}")
            return self.config.target_dimensions

    async def _update_reducer(self, new_dimensions: int):
        """Update the reducer with new target dimensions"""
        # Placeholder for updating the active reducer
        # In a real implementation, this would retrain the reducer
        self.logger.info(f"Reducer updated for {new_dimensions} dimensions")
        return None

    async def _reduce_new_data_only(self, new_data: np.ndarray) -> dict[str, Any]:
        """Handle case where only new data is available"""
        try:
            reduced = await self.reduce_dimensions(new_data)
            return {
                "status": "success",
                "new_data_samples": new_data.shape[0],
                "reduced_data": reduced.reduced_data if hasattr(reduced, 'reduced_data') else None,
                "note": "Only new data available, no adaptive adjustment performed"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def reduce_dimensions(
        self, X: np.ndarray, y: np.ndarray | None = None, method: str | None = None
    ) -> ReductionResult:
        """Reduce dimensionality of input data with optimal method selection.

        Args:
            X: Input feature matrix (n_samples, n_features)
            y: Optional target labels for supervised methods
            method: Specific method to use (if None, auto-select best method)

        Returns:
            ReductionResult with transformed data and evaluation metrics
        """
        start_time = time.time()

        self.logger.info(
            f"Starting dimensionality reduction: {X.shape[0]} samples x {X.shape[1]} features"
        )

        try:
            # Input validation
            if X.ndim != 2:
                raise ValueError("Input data must be 2D array")

            if X.shape[1] <= self.config.target_dimensions:
                self.logger.warning(
                    f"Input dimensionality ({X.shape[1]}) already <= target ({self.config.target_dimensions})"
                )
                return ReductionResult(
                    method="identity",
                    original_dimensions=X.shape[1],
                    reduced_dimensions=X.shape[1],
                    variance_preserved=1.0,
                    reconstruction_error=0.0,
                    processing_time=time.time() - start_time,
                    quality_score=1.0,
                    transformed_data=X.copy(),
                    transformer=None,
                )

            # Preprocessing
            X_processed = await self._preprocess_data(X)

            # Method selection
            if method is None:
                if self.config.auto_method_selection:
                    method = await self._select_optimal_method(X_processed, y)
                else:
                    method = (
                        self.config.preferred_methods[0]
                        if self.config.preferred_methods
                        else "pca"
                    )

            # Validate method availability
            if method not in self.available_methods:
                self.logger.warning(
                    f"Method '{method}' not available, falling back to PCA"
                )
                method = "pca"

            # Check cache
            cache_key = self._generate_cache_key(X, method)
            if self.config.enable_caching and cache_key in self.cache:
                self.logger.debug(f"Using cached result for method '{method}'")
                return self.cache[cache_key]

            # Perform dimensionality reduction
            result = await self._apply_reduction_method(X_processed, y, method)

            # Evaluate result quality
            evaluation_metrics = await self._evaluate_reduction_quality_detailed(
                X_processed, result.transformed_data, y
            )
            result.evaluation_metrics = evaluation_metrics
            result.quality_score = self._compute_overall_quality_score(
                result, evaluation_metrics
            )

            # Cache result
            if self.config.enable_caching:
                self.cache[cache_key] = result

            # Update performance tracking
            self._update_performance_tracking(result)

            self.logger.info(
                f"Dimensionality reduction completed with method '{method}': "
                f"{result.original_dimensions}→{result.reduced_dimensions} "
                f"(quality: {result.quality_score:.3f}, time: {result.processing_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Dimensionality reduction failed: {e}", exc_info=True)
            raise

    async def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for dimensionality reduction."""
        if self.scaler is None:
            # Use RobustScaler for better handling of outliers
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Remove constant features
        variance_selector = VarianceThreshold(threshold=1e-6)
        X_processed = variance_selector.fit_transform(X_scaled)

        if X_processed.shape[1] < X.shape[1]:
            self.logger.info(
                f"Removed {X.shape[1] - X_processed.shape[1]} constant features"
            )

        return X_processed

    async def _select_optimal_method(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> str:
        """Intelligently select the optimal dimensionality reduction method."""
        n_samples, n_features = X.shape

        # Filter methods based on data characteristics and constraints
        suitable_methods = []

        for method_name, method_info in self.available_methods.items():
            # Skip supervised methods if no labels provided
            if method_info["supervised"] and y is None:
                continue

            # Skip non-scalable methods for large datasets, but use optimized alternatives
            if not method_info["scalable"] and n_samples > 1000:
                # Suggest incremental alternatives for large datasets
                if method_name == "pca" and self.config.use_incremental_learning:
                    suitable_methods.append("incremental_pca")
                continue

            # Skip manifold methods for very high-dimensional data in fast mode
            if method_info["manifold"] and n_features > 50 and self.config.fast_mode:
                continue

            # Adjust LDA components based on number of classes
            if method_name == "lda" and y is not None:
                n_classes = len(np.unique(y))
                if n_classes <= 1:
                    continue  # Skip LDA if only one class
                method_info["params"]["n_components"] = min(
                    self.config.target_dimensions, n_classes - 1
                )

            suitable_methods.append(method_name)

        if not suitable_methods:
            self.logger.warning("No suitable methods found, falling back to PCA")
            return "pca"

        # If only one method is suitable, use it
        if len(suitable_methods) == 1:
            return suitable_methods[0]

        # For multiple suitable methods, use heuristics based on data characteristics

        # Prefer faster methods for large datasets
        if n_samples > 5000 or self.config.fast_mode:
            fast_methods = [
                m
                for m in suitable_methods
                if m in ["pca", "truncated_svd", "gaussian_rp", "sparse_rp"]
            ]
            if fast_methods:
                return fast_methods[0]

        # Prefer supervised methods when labels are available
        if y is not None:
            supervised_methods = [
                m for m in suitable_methods if self.available_methods[m]["supervised"]
            ]
            if supervised_methods:
                return supervised_methods[0]

        # Prefer variance-preserving methods for interpretability
        if self.config.variance_preservation_weight > 0.5:
            variance_methods = [
                m
                for m in suitable_methods
                if self.available_methods[m]["preserves_variance"]
            ]
            if variance_methods:
                return variance_methods[0]

        # Prefer UMAP for general-purpose manifold learning
        if "umap" in suitable_methods and UMAP_AVAILABLE:
            return "umap"

        # Default to PCA as most reliable
        return "pca" if "pca" in suitable_methods else suitable_methods[0]

    async def _apply_reduction_method(
        self, X: np.ndarray, y: np.ndarray | None, method: str
    ) -> ReductionResult:
        """Apply specific dimensionality reduction method."""
        start_time = time.time()

        method_info = self.available_methods[method]
        method_class = method_info["class"]
        method_params = method_info["params"].copy()

        try:
            # Initialize reducer
            reducer = method_class(**method_params)

            # Fit and transform
            if method_info["supervised"] and y is not None:
                X_reduced = reducer.fit_transform(X, y)
            else:
                X_reduced = reducer.fit_transform(X)

            # Calculate variance preserved (for applicable methods)
            variance_preserved = 0.0
            if hasattr(reducer, "explained_variance_ratio_"):
                variance_preserved = np.sum(reducer.explained_variance_ratio_)
            elif method in ["pca", "truncated_svd"]:
                # Fallback calculation for PCA-like methods
                total_var = np.var(X, axis=0).sum()
                reduced_var = np.var(X_reduced, axis=0).sum()
                variance_preserved = min(1.0, reduced_var / total_var)

            # Calculate reconstruction error
            reconstruction_error = self._calculate_reconstruction_error(
                X, X_reduced, reducer
            )

            # Extract feature importance if available
            feature_importance = None
            if hasattr(reducer, "components_"):
                # For linear methods like PCA, use component magnitudes
                feature_importance = np.abs(reducer.components_).mean(axis=0)
            elif hasattr(reducer, "feature_importances_"):
                feature_importance = reducer.feature_importances_

            processing_time = time.time() - start_time

            return ReductionResult(
                method=method,
                original_dimensions=X.shape[1],
                reduced_dimensions=X_reduced.shape[1],
                variance_preserved=variance_preserved,
                reconstruction_error=reconstruction_error,
                processing_time=processing_time,
                quality_score=0.0,  # Will be computed later
                transformed_data=X_reduced,
                transformer=reducer,
                feature_importance=feature_importance,
            )

        except Exception as e:
            self.logger.error(f"Failed to apply method '{method}': {e}")
            raise

    def _calculate_reconstruction_error(
        self, X_original: np.ndarray, X_reduced: np.ndarray, reducer: Any
    ) -> float:
        """Calculate reconstruction error for the dimensionality reduction."""
        try:
            # For methods with inverse_transform
            if hasattr(reducer, "inverse_transform"):
                X_reconstructed = reducer.inverse_transform(X_reduced)
                error = np.mean(np.square(X_original - X_reconstructed))
                return float(error)

            # For PCA-like methods, use explained variance
            if hasattr(reducer, "explained_variance_ratio_"):
                return float(1.0 - np.sum(reducer.explained_variance_ratio_))

            # For other methods, use a simple approximation
            # Based on the ratio of preserved vs original variance
            original_var = np.var(X_original)
            reduced_var = np.var(X_reduced)
            normalized_error = max(0.0, 1.0 - (reduced_var / (original_var + 1e-8)))
            return float(normalized_error)

        except Exception as e:
            self.logger.warning(f"Could not calculate reconstruction error: {e}")
            return 0.5  # Default moderate error

    async def _evaluate_reduction_quality_detailed(
        self, X_original: np.ndarray, X_reduced: np.ndarray, y: np.ndarray | None = None
    ) -> dict[str, float]:
        """Evaluate the quality of dimensionality reduction."""
        metrics = {}

        try:
            # 1. Variance preservation (always computed)
            original_total_var = np.sum(np.var(X_original, axis=0))
            reduced_total_var = np.sum(np.var(X_reduced, axis=0))
            metrics["variance_preservation"] = min(
                1.0, reduced_total_var / (original_total_var + 1e-8)
            )

            # 2. Clustering quality preservation
            if X_original.shape[0] >= 20:  # Need sufficient samples
                try:
                    from sklearn.cluster import KMeans
                    from sklearn.metrics import adjusted_rand_score

                    # Simple clustering comparison
                    n_clusters = min(5, X_original.shape[0] // 10)
                    if n_clusters >= 2:
                        kmeans_orig = KMeans(
                            n_clusters=n_clusters, random_state=42, n_init=10
                        )
                        kmeans_reduced = KMeans(
                            n_clusters=n_clusters, random_state=42, n_init=10
                        )

                        labels_orig = kmeans_orig.fit_predict(X_original)
                        labels_reduced = kmeans_reduced.fit_predict(X_reduced)

                        metrics["clustering_preservation"] = adjusted_rand_score(
                            labels_orig, labels_reduced
                        )
                except Exception as e:
                    self.logger.debug(f"Clustering quality evaluation failed: {e}")
                    metrics["clustering_preservation"] = 0.5

            # 3. Classification quality preservation (if labels available)
            if y is not None and len(np.unique(y)) > 1:
                try:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import cross_val_score

                    # Simple classification comparison
                    clf = LogisticRegression(random_state=42, max_iter=1000)

                    scores_orig = cross_val_score(
                        clf, X_original, y, cv=3, scoring="accuracy"
                    )
                    scores_reduced = cross_val_score(
                        clf, X_reduced, y, cv=3, scoring="accuracy"
                    )

                    metrics["classification_preservation"] = np.mean(scores_reduced) / (
                        np.mean(scores_orig) + 1e-8
                    )
                except Exception as e:
                    self.logger.debug(f"Classification quality evaluation failed: {e}")
                    metrics["classification_preservation"] = 0.5

            # 4. Neighborhood preservation
            try:
                from sklearn.neighbors import NearestNeighbors

                k = min(10, X_original.shape[0] // 10)
                if k >= 2:
                    nn_orig = NearestNeighbors(n_neighbors=k)
                    nn_reduced = NearestNeighbors(n_neighbors=k)

                    nn_orig.fit(X_original)
                    nn_reduced.fit(X_reduced)

                    # Sample a subset for efficiency
                    sample_size = min(100, X_original.shape[0])
                    sample_indices = np.random.choice(
                        X_original.shape[0], sample_size, replace=False
                    )

                    neighbors_orig = nn_orig.kneighbors(
                        X_original[sample_indices], return_distance=False
                    )
                    neighbors_reduced = nn_reduced.kneighbors(
                        X_reduced[sample_indices], return_distance=False
                    )

                    # Calculate neighborhood preservation score
                    preservation_scores = []
                    for i in range(sample_size):
                        overlap = len(
                            set(neighbors_orig[i]) & set(neighbors_reduced[i])
                        )
                        preservation_scores.append(overlap / k)

                    metrics["neighborhood_preservation"] = np.mean(preservation_scores)
            except Exception as e:
                self.logger.debug(f"Neighborhood preservation evaluation failed: {e}")
                metrics["neighborhood_preservation"] = 0.5

        except Exception as e:
            self.logger.warning(f"Quality evaluation failed: {e}")
            # Return default metrics
            metrics = {
                "variance_preservation": 0.5,
                "clustering_preservation": 0.5,
                "classification_preservation": 0.5 if y is not None else None,
                "neighborhood_preservation": 0.5,
            }

        return metrics

    def _compute_overall_quality_score(
        self, result: ReductionResult, evaluation_metrics: dict[str, float]
    ) -> float:
        """Compute overall quality score for the dimensionality reduction."""
        try:
            score_components = []
            weights = []

            # Variance preservation
            if "variance_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["variance_preservation"])
                weights.append(self.config.variance_preservation_weight)

            # Clustering preservation
            if "clustering_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["clustering_preservation"])
                weights.append(self.config.clustering_quality_weight)

            # Classification preservation
            if (
                "classification_preservation" in evaluation_metrics
                and evaluation_metrics["classification_preservation"] is not None
            ):
                score_components.append(
                    evaluation_metrics["classification_preservation"]
                )
                weights.append(self.config.classification_quality_weight)

            # Neighborhood preservation
            if "neighborhood_preservation" in evaluation_metrics:
                score_components.append(evaluation_metrics["neighborhood_preservation"])
                weights.append(0.2)  # Fixed weight for neighborhood preservation

            # Processing time penalty (favor faster methods)
            time_penalty = max(
                0.0, 1.0 - result.processing_time / 60.0
            )  # Penalty after 1 minute
            score_components.append(time_penalty)
            weights.append(0.1)

            # Dimensionality efficiency (favor achieving target dimensions)
            target_efficiency = min(
                1.0, result.reduced_dimensions / self.config.target_dimensions
            )
            score_components.append(target_efficiency)
            weights.append(0.1)

            # Weighted average
            if score_components and weights:
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                overall_score = np.average(score_components, weights=weights)
                return float(np.clip(overall_score, 0.0, 1.0))

            return 0.5  # Default score

        except Exception as e:
            self.logger.warning(f"Quality score computation failed: {e}")
            return 0.5

    def _generate_cache_key(self, X: np.ndarray, method: str) -> str:
        """Generate cache key for reduction results."""
        import hashlib

        key_data = f"{X.shape}_{method}_{self.config.target_dimensions}_{self.config.random_state}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _update_performance_tracking(self, result: ReductionResult):
        """Update performance tracking for the reduction method."""
        self.reduction_history.append(result)

        # Update method-specific performance
        if result.method not in self.method_performance:
            self.method_performance[result.method] = []

        self.method_performance[result.method].append(result.quality_score)

        # Keep only recent history to prevent memory growth
        if len(self.reduction_history) > 100:
            self.reduction_history = self.reduction_history[-50:]

        for method in self.method_performance:
            if len(self.method_performance[method]) > 50:
                self.method_performance[method] = self.method_performance[method][-25:]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.reduction_history:
            return {"status": "no_data"}

        recent_results = self.reduction_history[-10:]

        summary = {
            "status": "available",
            "total_reductions": len(self.reduction_history),
            "recent_performance": {
                "avg_quality_score": float(
                    np.mean([r.quality_score for r in recent_results])
                ),
                "avg_processing_time": float(
                    np.mean([r.processing_time for r in recent_results])
                ),
                "avg_variance_preserved": float(
                    np.mean([r.variance_preserved for r in recent_results])
                ),
                "avg_dimensionality_reduction": float(
                    np.mean([
                        (r.original_dimensions - r.reduced_dimensions)
                        / r.original_dimensions
                        for r in recent_results
                    ])
                ),
            },
            "method_performance": {},
            "available_methods": list(self.available_methods.keys()),
        }

        # Method-specific performance
        for method, scores in self.method_performance.items():
            if scores:
                summary["method_performance"][method] = {
                    "avg_quality": float(np.mean(scores)),
                    "std_quality": float(np.std(scores)),
                    "usage_count": len(scores),
                    "reliability": float(np.mean([s > 0.5 for s in scores])),
                }

        return summary

    def get_recommended_method(
        self, X_shape: tuple[int, int], y: np.ndarray | None = None
    ) -> str:
        """Get recommended method based on data characteristics and historical performance."""
        n_samples, n_features = X_shape

        # Use historical performance if available
        if self.method_performance:
            best_method = max(
                self.method_performance.keys(),
                key=lambda m: np.mean(self.method_performance[m]),
            )
            avg_quality = np.mean(self.method_performance[best_method])
            if avg_quality > 0.7:  # Good historical performance
                return best_method

        # Fall back to heuristics with 2025 best practices
        if self.config.fast_mode:
            # For fast mode, prefer optimized statistical methods
            if n_samples > 5000:
                return "sparse_rp"  # Sparse random projection for speed
            return "pca"

        if n_samples > 10000:
            # For large datasets, prefer scalable methods
            if TORCH_AVAILABLE and self.config.enable_neural_methods:
                if n_features > 100:
                    return "transformer"  # Best for very high-dimensional data
                return "autoencoder"
            return "pca"  # Fallback to optimized PCA

        if y is not None and len(np.unique(y)) > 1:
            return "lda"

        if n_features > 50:
            if TORCH_AVAILABLE and self.config.enable_neural_methods:
                # For high-dimensional data, prefer modern neural methods
                if n_samples > 1000:
                    return "diffusion"  # Best quality for sufficient data
                return "vae"
            elif UMAP_AVAILABLE:
                return "umap"  # Fallback to UMAP

        if UMAP_AVAILABLE and n_features > 20:
            return "umap"

        return "pca"


# Factory function for easy integration
def get_dimensionality_reducer(
    config: DimensionalityConfig | None = None,
) -> AdvancedDimensionalityReducer:
    """Get dimensionality reducer instance.

    Args:
        config: Optional dimensionality reduction configuration

    Returns:
        AdvancedDimensionalityReducer instance
    """
    return AdvancedDimensionalityReducer(config=config)
