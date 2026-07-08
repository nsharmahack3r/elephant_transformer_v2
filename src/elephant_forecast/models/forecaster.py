from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from elephant_forecast.models.backbone import DecoderBackbone
from elephant_forecast.models.mdn import MixtureDensityHead


class Time2Vec(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim))
        self.b = nn.Parameter(torch.randn(1, dim))
        self.linear = nn.Linear(dim + 1, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B, T, 1] log1p-scaled delta time
        """
        periodic = torch.sin(t * self.w + self.b)
        output = torch.cat([t, periodic], dim=-1)
        return self.linear(output)


class ElephantForecaster(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: float = 8.0 / 3.0,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        n_mixtures: int = 5,
        n_continuous_covars: int = 7,
        n_lulc_classes: int = 1,
        lulc_embed_dim: int = 16,
        time2vec_dim: int = 16,
        covariate_hidden: int = 64,
        fusion: str = "concat",
        aux_covariate_head: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion = fusion
        self.aux_covariate_head = aux_covariate_head

        self.motion_proj = nn.Linear(2, d_model)
        self.time2vec = Time2Vec(time2vec_dim)
        self.time_proj = nn.Linear(time2vec_dim + 4, d_model)
        self.lulc_embed = nn.Embedding(max(n_lulc_classes, 2), lulc_embed_dim)
        self.lulc_proj = nn.Linear(lulc_embed_dim, d_model)
        self.covar_mlp = nn.Sequential(
            nn.Linear(n_continuous_covars, covariate_hidden),
            nn.ReLU(),
            nn.Linear(covariate_hidden, d_model),
        )

        if fusion == "gated":
            self.gate_motion = nn.Linear(d_model * 2, d_model)
            self.gate_covar = nn.Linear(d_model * 2, d_model)
        else:
            self.fusion_proj = nn.Linear(d_model * 3, d_model)

        self.backbone = DecoderBackbone(d_model, n_layers, n_heads, ffn_mult, max_seq_len, dropout)
        self.mdn = MixtureDensityHead(d_model, n_mixtures)

        if aux_covariate_head:
            self.aux_head = nn.Linear(d_model, n_continuous_covars + 1)

    def _build_token(self, disp: torch.Tensor, dt: torch.Tensor, time_feat: torch.Tensor,
                     cov: torch.Tensor, lulc: torch.Tensor) -> torch.Tensor:
        motion = self.motion_proj(disp)
        t2v = self.time2vec(dt)
        time_emb = self.time_proj(torch.cat([t2v, time_feat], dim=-1))
        lulc_emb = self.lulc_proj(self.lulc_embed(lulc.clamp(0, self.lulc_embed.num_embeddings - 1)))
        covar = self.covar_mlp(cov)
        env = lulc_emb + covar

        if self.fusion == "gated":
            combined = torch.cat([motion, env], dim=-1)
            gate = torch.sigmoid(self.gate_motion(combined) + self.gate_covar(combined))
            return gate * motion + (1 - gate) * env
        else:
            return self.fusion_proj(torch.cat([motion, time_emb, env], dim=-1))

    def encode_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token = self._build_token(
            batch["disp_in"], batch["dt_in"], batch["time_in"],
            batch["cov_in"], batch["lulc_in"],
        )
        return self.backbone(token, batch.get("mask"))

    def forward(self, batch: dict[str, torch.Tensor], teacher_forcing: bool = True
                ) -> dict[str, torch.Tensor]:
        B, L, _ = batch["disp_in"].shape
        H = batch["target"].shape[1]

        ctx = self.encode_context(batch)
        last_hidden = ctx[:, -1]

        all_pi, all_mu, all_sigma, all_rho = [], [], [], []
        all_aux = []
        current_disp = batch["disp_in"][:, -1:].clone()
        current_pos = torch.zeros(B, H, 2, device=ctx.device)

        for t in range(H):
            token = self._build_token(
                current_disp,
                batch["dt_out"][:, t:t + 1],
                batch["time_out"][:, t:t + 1],
                batch["cov_out"][:, t:t + 1] if "cov_out" in batch else torch.zeros(B, 1, 0, device=ctx.device),
                batch["lulc_out"][:, t:t + 1] if "lulc_out" in batch else torch.zeros(B, 1, dtype=torch.long, device=ctx.device),
            )
            step_in = last_hidden.unsqueeze(1) + token
            hidden = self.backbone.final_norm(step_in)

            pi, mu, sigma, rho = self.mdn(hidden.squeeze(1))
            all_pi.append(pi)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_rho.append(rho)

            if self.aux_covariate_head:
                all_aux.append(self.aux_head(hidden.squeeze(1)))

            if teacher_forcing and t < H - 1:
                current_disp = batch["target"][:, t:t + 1]
            else:
                current_disp = self.mdn.mode(pi, mu).unsqueeze(1)

            last_hidden = hidden.squeeze(1)

        return {
            "pi": torch.stack(all_pi, dim=1),
            "mu": torch.stack(all_mu, dim=1),
            "sigma": torch.stack(all_sigma, dim=1),
            "rho": torch.stack(all_rho, dim=1),
            "aux": torch.stack(all_aux, dim=1) if all_aux else None,
        }

    def compute_loss(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nll = self.mdn.nll(
            batch["target"],
            output["pi"], output["mu"], output["sigma"], output["rho"],
        )
        loss = nll
        losses = {"nll": nll, "loss": loss}

        if self.aux_covariate_head and output["aux"] is not None and batch.get("cov_out") is not None:
            cov_cont = batch["cov_out"][:, :, :output["aux"].shape[-1]]
            aux_loss = F.mse_loss(output["aux"], cov_cont)
            losses["aux_loss"] = aux_loss
            losses["loss"] = loss + 0.1 * aux_loss

        return losses
