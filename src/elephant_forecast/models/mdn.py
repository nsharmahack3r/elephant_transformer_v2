from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureDensityHead(nn.Module):
    """
    Bivariate Gaussian Mixture Density Network head.
    Outputs K mixtures over (Δlat, Δlon).
    """
    def __init__(self, d_model: int, n_mixtures: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, n_mixtures)
        self.mu_head = nn.Linear(hidden_dim, n_mixtures * 2)
        self.sigma_head = nn.Linear(hidden_dim, n_mixtures * 2)
        self.rho_head = nn.Linear(hidden_dim, n_mixtures)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [..., d_model]
        Returns:
            pi: [..., n_mixtures]  mixture weights (softmax)
            mu: [..., n_mixtures, 2]  means
            sigma: [..., n_mixtures, 2]  standard deviations (>0)
            rho: [..., n_mixtures]  correlation (-1, 1)
        """
        h = self.mlp(x)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(*x.shape[:-1], self.n_mixtures, 2)
        sigma = F.softplus(self.sigma_head(h)).view(*x.shape[:-1], self.n_mixtures, 2) + 1e-6
        rho = torch.tanh(self.rho_head(h))
        return pi, mu, sigma, rho

    def nll(self, target: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor,
            sigma: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood for bivariate Gaussian mixture.
        target: [..., 2]
        """
        dx = target.unsqueeze(-2) - mu
        sx = sigma[..., 0]
        sy = sigma[..., 1]

        zx = dx[..., 0] / sx
        zy = dx[..., 1] / sy

        z = zx ** 2 + zy ** 2 - 2 * rho * zx * zy
        denom = 2 * math.pi * sx * sy * torch.sqrt(torch.clamp(1 - rho ** 2, min=1e-8))
        log_pdf = -0.5 * z - torch.log(denom)

        log_mix = torch.log(pi + 1e-10)
        nll = -torch.logsumexp(log_mix + log_pdf, dim=-1)
        return nll.mean()

    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        rho: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the mixture. Returns [..., n_samples, 2].
        """
        batch_shape = pi.shape[:-1]
        pi_flat = pi.reshape(-1, self.n_mixtures)
        mu_flat = mu.reshape(-1, self.n_mixtures, 2)
        sigma_flat = sigma.reshape(-1, self.n_mixtures, 2)
        rho_flat = rho.reshape(-1, self.n_mixtures)
        n_flat = pi_flat.shape[0]

        cats = torch.multinomial(pi_flat, num_samples=n_samples, replacement=True)
        chosen_mu = mu_flat[torch.arange(n_flat)[:, None], cats]
        chosen_sigma = sigma_flat[torch.arange(n_flat)[:, None], cats]
        chosen_rho = rho_flat[torch.arange(n_flat)[:, None], cats]

        eps = torch.randn(n_flat, n_samples, 2, device=pi.device)
        eps1 = eps[..., 0]
        eps2 = chosen_rho * eps[..., 0] + torch.sqrt(1 - chosen_rho ** 2) * eps[..., 1]

        samples = torch.stack([
            chosen_mu[..., 0] + chosen_sigma[..., 0] * eps1,
            chosen_mu[..., 1] + chosen_sigma[..., 1] * eps2,
        ], dim=-1)

        return samples.reshape(*batch_shape, n_samples, 2)

    def mode(self, pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Return the mean of the highest-weighted component. [..., 2]
        """
        best = pi.argmax(dim=-1)
        batch_shape = best.shape
        mu_flat = mu.reshape(-1, self.n_mixtures, 2)
        idx = best.reshape(-1)
        return mu_flat[torch.arange(idx.shape[0]), idx].reshape(*batch_shape, 2)
