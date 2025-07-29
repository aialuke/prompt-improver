"""Neural Data Generator Module

Basic neural network-based synthetic data generation.
Extracted from synthetic_data_generator.py for focused functionality.

This module contains:
- NeuralSyntheticGenerator: Basic neural generation with VAE/GAN support
- TabularDiffusionModel: Diffusion model for tabular data
- DiffusionSyntheticGenerator: Diffusion-based generation

Note: This module requires PyTorch. Complex GAN/VAE models are in gan_generator.py
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Neural network and deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Neural generative models will be disabled. Install with: pip install torch")


class NeuralSyntheticGenerator:
    """Neural network-based synthetic data generator.
    
    This is a simplified neural generator that works with basic models.
    For advanced GAN/VAE implementations, use the gan_generator module.
    """

    def __init__(self, model_type: str = "vae", latent_dim: int = 50,
                 hidden_dims: list[int] = None, beta: float = 1.0,
                 device: str = "auto", epochs: int = 200, batch_size: int = 64,
                 learning_rate: float = 1e-3):
        """Initialize neural synthetic generator
        
        Args:
            model_type: Type of model ("vae", "gan", "diffusion") 
            latent_dim: Size of latent dimension for VAE
            hidden_dims: Hidden layer dimensions
            beta: Beta parameter for VAE
            device: Device for training ("auto", "cpu", "cuda")
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural generative models")
            
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

        # For basic neural generator, we'll use simple models
        # Complex GAN/VAE models are in gan_generator.py
        if self.model_type == "simple_vae":
            self.model = SimpleVAE(data_dim, self.latent_dim, self.hidden_dims, self.beta)
            self.model.to(self.device)
            self.optimizer_g = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self._train_simple_vae(X_scaled)
        elif self.model_type == "diffusion":
            # Use diffusion model from this module
            self.model = TabularDiffusionModel(data_dim)
            self.model.to(self.device)
            self.optimizer_g = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self._train_diffusion(X_scaled)
        else:
            # For full GAN/VAE models, defer to gan_generator
            try:
                from .gan_generator import TabularGAN, TabularVAE
                
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
                    
            except ImportError:
                logger.warning("Advanced GAN/VAE models not available, using simple VAE")
                self.model = SimpleVAE(data_dim, self.latent_dim, self.hidden_dims, self.beta)
                self.model.to(self.device)
                self.optimizer_g = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                self._train_simple_vae(X_scaled)

        self.is_fitted = True
        return self

    def _train_simple_vae(self, X_scaled: np.ndarray):
        """Train simple VAE model."""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer_g.zero_grad()

                recon, mu, logvar = self.model(data)
                loss = self.model.loss_function(recon, data, mu, logvar)

                loss.backward()
                self.optimizer_g.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"Simple VAE Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def _train_vae(self, X_scaled: np.ndarray):
        """Train VAE model (complex version from gan_generator)."""
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
                
    def _train_diffusion(self, X_scaled: np.ndarray):
        """Train diffusion model."""
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_idx, (data,) in enumerate(dataloader):
                self.optimizer_g.zero_grad()

                # Sample random timesteps
                t = torch.randint(0, self.model.num_timesteps, (data.size(0),), device=self.device)

                # Add noise
                x_t, noise = self.model.add_noise(data, t)

                # Predict noise
                predicted_noise = self.model(x_t, t)

                # Compute loss
                loss = nn.functional.mse_loss(predicted_noise, noise)

                loss.backward()
                self.optimizer_g.step()
                total_loss += loss.item()

            if epoch % 50 == 0:
                logger.info(f"Diffusion Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")

    def _train_gan(self, X_scaled: np.ndarray):
        """Train GAN model (complex version from gan_generator)."""
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
            if hasattr(self.model, 'sample') and callable(self.model.sample):
                # Diffusion model
                synthetic_data = self.model.sample(n_samples, self.device)
            elif hasattr(self.model, 'generate') and callable(self.model.generate):
                # GAN model
                synthetic_data = self.model.generate(n_samples, self.device)
            else:
                # VAE model
                z = torch.randn(n_samples, self.latent_dim, device=self.device)
                synthetic_data = self.model.decode(z)
            
            synthetic_data = synthetic_data.cpu().numpy()

        # Inverse transform to original scale
        return self.scaler.inverse_transform(synthetic_data)


class SimpleVAE(nn.Module):
    """Simple Variational Autoencoder for basic neural generation."""

    def __init__(self, data_dim: int, latent_dim: int = 50, hidden_dims: list[int] = None, beta: float = 1.0):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [64, 32]

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
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


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
        """Initialize diffusion generator
        
        Args:
            num_timesteps: Number of timesteps for diffusion process
            hidden_dim: Hidden dimension size
            device: Device for training ("auto", "cpu", "cuda")
            epochs: Number of training epochs
            batch_size: Training batch size  
            learning_rate: Learning rate for optimization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for diffusion models")
            
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