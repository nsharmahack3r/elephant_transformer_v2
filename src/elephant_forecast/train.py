from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import json

from elephant_forecast.config import Config


class Trainer:
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.train.use_amp)
        self.best_val_nll = float("inf")
        self.epoch = 0

        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_tf_ratio(self) -> float:
        cfg = self.config.train
        if self.epoch >= cfg.teacher_forcing_anneal_epochs:
            return cfg.teacher_forcing_ratio_end
        progress = min(self.epoch / max(cfg.teacher_forcing_anneal_epochs, 1), 1.0)
        return cfg.teacher_forcing_ratio_start - progress * (
            cfg.teacher_forcing_ratio_start - cfg.teacher_forcing_ratio_end
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_nll = 0.0
        n_batches = 0

        tf_ratio = self._get_tf_ratio()

        pbar = tqdm(loader, desc=f"Epoch {epoch} train")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            teacher_forcing = torch.rand(1).item() < tf_ratio

            with autocast(enabled=self.config.train.use_amp):
                output = self.model(batch, teacher_forcing=teacher_forcing)
                losses = self.model.compute_loss(output, batch)

            self.optimizer.zero_grad()
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses["loss"].item()
            total_nll += losses["nll"].item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}", "nll": f"{total_nll / n_batches:.4f}"})

        return {"train_loss": total_loss / n_batches, "train_nll": total_nll / n_batches}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_nll = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Validate"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with autocast(enabled=self.config.train.use_amp):
                output = self.model(batch, teacher_forcing=True)
                losses = self.model.compute_loss(output, batch)

            total_loss += losses["loss"].item()
            total_nll += losses["nll"].item()
            n_batches += 1

        return {"val_loss": total_loss / n_batches, "val_nll": total_nll / n_batches}

    def _cosine_warmup_scheduler(self) -> float:
        cfg = self.config.train
        step = self.epoch
        if step < cfg.warmup_steps:
            return float(step) / max(cfg.warmup_steps, 1)
        progress = float(step - cfg.warmup_steps) / max(cfg.max_epochs - cfg.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        for epoch in range(self.config.train.max_epochs):
            self.epoch = epoch

            lr = self._cosine_warmup_scheduler() * self.config.train.lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            train_metrics = self.train_epoch(train_loader, epoch)

            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)

                if val_metrics["val_nll"] < self.best_val_nll:
                    self.best_val_nll = val_metrics["val_nll"]
                    self.save_checkpoint("best.pt", {**train_metrics, **val_metrics})

                print(f"Epoch {epoch}: train_nll={train_metrics['train_nll']:.4f}, "
                      f"val_nll={val_metrics['val_nll']:.4f}, lr={lr:.2e}")

        self.save_checkpoint("last.pt", train_metrics)

    def save_checkpoint(self, name: str, metrics: dict[str, float]) -> None:
        path = self.checkpoint_dir / name
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_nll": self.best_val_nll,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_val_nll = ckpt["best_val_nll"]
        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']}, val_nll={self.best_val_nll:.4f})")
