from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import os
from dotenv import load_dotenv


@dataclass
class DataConfig:
    gap_threshold_hours: float = 6.0
    context_len: int = 32
    horizon: int = 16
    min_session_len: Optional[int] = None

    continuous_covars: tuple[str, ...] = (
        "aspect_deg", "human_settle", "elevation_m",
        "slope_deg", "EVI", "LST_celsius", "NDVI",
    )
    categorical_covars: tuple[str, ...] = ("LULC_class",)
    id_col: str = "individual-local-identifier"
    time_col: str = "timestamp"
    lon_col: str = "location-long"
    lat_col: str = "location-lat"

    covariate_carry_forward: bool = True

    def __post_init__(self) -> None:
        if self.min_session_len is None:
            self.min_session_len = self.context_len + self.horizon + 1


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_mult: float = 8.0 / 3.0
    dropout: float = 0.1
    max_seq_len: int = 512

    n_mixtures: int = 5
    lulc_embed_dim: int = 16
    time2vec_dim: int = 16
    covariate_hidden: int = 64

    fusion: str = "concat"
    aux_covariate_head: bool = False
    aux_loss_weight: float = 0.1


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    grad_clip: float = 1.0

    use_amp: bool = True
    teacher_forcing_ratio_start: float = 1.0
    teacher_forcing_ratio_end: float = 0.5
    teacher_forcing_anneal_epochs: int = 80

    augment_rotate: bool = True
    augment_jitter: bool = True
    augment_subsample: bool = True

    batch_size: int = 64
    num_workers: int = 4
    val_every_n_epochs: int = 1
    log_every_n_steps: int = 50

    fixed_val_elephants: tuple[str, ...] = ("LA2", "LA8")
    fixed_test_elephants: tuple[str, ...] = ("LA15",)

    checkpoint_dir: str = "checkpoints"
    best_metric: str = "val_nll"


@dataclass
class EvalConfig:
    n_samples: int = 50
    max_speed_mps: float = 7.0
    best_of_k: int = 10
    n_clusters: int = 20
    plot_n_examples: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    data_path: str = ""
    sample_path: str = ""

    seed: int = 42
    device: str = "cuda"


def load_config(yaml_path: str | Path) -> Config:
    load_dotenv()
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config()

    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(cfg.data, k):
                if k in ("continuous_covars", "categorical_covars"):
                    v = tuple(v)
                setattr(cfg.data, k, v)
        if cfg.data.min_session_len is None:
            cfg.data.min_session_len = cfg.data.context_len + cfg.data.horizon + 1

    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    if "train" in raw:
        for k, v in raw["train"].items():
            if hasattr(cfg.train, k):
                if k in ("fixed_val_elephants", "fixed_test_elephants"):
                    v = tuple(v)
                setattr(cfg.train, k, v)

    if "eval" in raw:
        for k, v in raw["eval"].items():
            if hasattr(cfg.eval, k):
                setattr(cfg.eval, k, v)

    if "seed" in raw:
        cfg.seed = raw["seed"]

    cfg.data_path = os.getenv("CLEANED_PATH", "data/cleaned.csv")
    cfg.sample_path = os.getenv("SAMPLE_PATH", "data/sample.csv")

    return cfg
