from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in no-torch envs
    raise ModuleNotFoundError(
        "dluxshera.ml.models requires PyTorch. Install the optional ML "
        "environment, for example `python -m pip install -e .[ml]`."
    ) from exc

__all__ = [
    "PairwiseCorrectionCNN",
    "SharedCNNEncoder",
    "SharedCNNModelConfig",
    "build_pairwise_correction_model",
    "count_parameters",
]


@dataclass(frozen=True)
class SharedCNNModelConfig:
    """Configure the deliberately small shared-CNN regression baseline."""

    input_channels: int = 1
    channels: tuple[int, ...] = (16, 32, 64, 128)
    embedding_dim: int = 128
    encoder_hidden_dim: int = 256
    head_hidden_dim: int = 256
    comparator: str = "concat_diff"
    dropout: float = 0.0
    normalization: str = "batch"
    adaptive_pool_shape: tuple[int, int] = (4, 4)

    def __post_init__(self) -> None:
        if self.comparator not in {"difference", "concat_diff"}:
            raise ValueError("comparator must be 'difference' or 'concat_diff'.")
        if self.normalization not in {"batch", "none"}:
            raise ValueError("normalization must be 'batch' or 'none'.")
        if int(self.input_channels) < 1:
            raise ValueError("input_channels must be >= 1.")
        if not self.channels:
            raise ValueError("channels must not be empty.")
        if min(int(v) for v in self.channels) < 1:
            raise ValueError("all channels must be >= 1.")
        if int(self.embedding_dim) < 1:
            raise ValueError("embedding_dim must be >= 1.")
        if int(self.encoder_hidden_dim) < 1 or int(self.head_hidden_dim) < 1:
            raise ValueError("hidden dimensions must be >= 1.")
        if float(self.dropout) < 0.0 or float(self.dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready model provenance."""
        return {
            "input_channels": int(self.input_channels),
            "channels": list(self.channels),
            "embedding_dim": int(self.embedding_dim),
            "encoder_hidden_dim": int(self.encoder_hidden_dim),
            "head_hidden_dim": int(self.head_hidden_dim),
            "comparator": self.comparator,
            "dropout": float(self.dropout),
            "normalization": self.normalization,
            "adaptive_pool_shape": list(self.adaptive_pool_shape),
            "shared_encoder": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SharedCNNModelConfig":
        """Build a config from a run-config mapping."""
        if payload is None:
            return cls()
        return cls(
            input_channels=int(payload.get("input_channels", 1)),
            channels=tuple(int(v) for v in payload.get("channels", (16, 32, 64, 128))),
            embedding_dim=int(payload.get("embedding_dim", 128)),
            encoder_hidden_dim=int(payload.get("encoder_hidden_dim", 256)),
            head_hidden_dim=int(payload.get("head_hidden_dim", 256)),
            comparator=str(payload.get("comparator", "concat_diff")),
            dropout=float(payload.get("dropout", 0.0)),
            normalization=str(payload.get("normalization", "batch")),
            adaptive_pool_shape=tuple(
                int(v) for v in payload.get("adaptive_pool_shape", (4, 4))
            ),
        )


class SharedCNNEncoder(nn.Module):
    """Encode one SHERA image into a compact learned embedding.

    The encoder uses a small convolutional stack inspired by the colleague
    classifier notebooks, but it keeps spatial structure through an adaptive
    ``4x4`` pooling stage before projecting to the embedding.  It does not
    perform per-image L2 normalization.
    """

    embedding_dim: int
    output_channels: int

    def __init__(self, config: SharedCNNModelConfig | None = None) -> None:
        super().__init__()
        self.config = SharedCNNModelConfig() if config is None else config
        layers: list[nn.Module] = []
        in_channels = int(self.config.input_channels)
        kernel_sizes = [5, 5, 3, 3]
        for index, out_channels in enumerate(self.config.channels):
            kernel_size = kernel_sizes[min(index, len(kernel_sizes) - 1)]
            layers.append(
                nn.Conv2d(
                    in_channels,
                    int(out_channels),
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                )
            )
            if self.config.normalization == "batch":
                layers.append(nn.BatchNorm2d(int(out_channels)))
            layers.append(nn.ReLU(inplace=True))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(tuple(self.config.adaptive_pool_shape))
        pooled_dim = (
            int(self.config.channels[-1])
            * int(self.config.adaptive_pool_shape[0])
            * int(self.config.adaptive_pool_shape[1])
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_dim, int(self.config.encoder_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.config.encoder_hidden_dim), int(self.config.embedding_dim)),
        )
        self.embedding_dim = int(self.config.embedding_dim)
        self.output_channels = int(self.config.channels[-1])

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_feature_map: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return an embedding, optionally with the final convolutional map."""
        feature_map = self.features(image)
        embedding = self.projection(self.pool(feature_map))
        if return_feature_map:
            return embedding, feature_map
        return embedding


class PairwiseCorrectionCNN(nn.Module):
    """Predict ``z_B - z_A`` from two images using one shared encoder instance."""

    output_dim: int
    comparator: str

    def __init__(
        self,
        *,
        output_dim: int,
        config: SharedCNNModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = SharedCNNModelConfig() if config is None else config
        self.encoder = SharedCNNEncoder(self.config)
        self.output_dim = int(output_dim)
        self.comparator = self.config.comparator
        comparator_dim = (
            self.config.embedding_dim
            if self.comparator == "difference"
            else 3 * self.config.embedding_dim
        )
        head_layers: list[nn.Module] = [
            nn.Linear(int(comparator_dim), int(self.config.head_hidden_dim)),
            nn.ReLU(inplace=True),
        ]
        if self.config.dropout > 0.0:
            head_layers.append(nn.Dropout(float(self.config.dropout)))
        head_layers.append(nn.Linear(int(self.config.head_hidden_dim), self.output_dim))
        self.regression_head = nn.Sequential(*head_layers)

    def compare(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """Return the configured comparator representation."""
        diff = h_b - h_a
        if self.comparator == "difference":
            return diff
        if self.comparator == "concat_diff":
            return torch.cat([h_a, h_b, diff], dim=1)
        raise ValueError(f"Unsupported comparator {self.comparator!r}.")

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict the Fisher-scaled correction from image ``A`` toward ``B``."""
        h_a = self.encoder(image_a)
        h_b = self.encoder(image_b)
        pred = self.regression_head(self.compare(h_a, h_b))
        if return_embeddings:
            return pred, h_a, h_b
        return pred


def build_pairwise_correction_model(
    output_dim: int,
    config: Mapping[str, Any] | SharedCNNModelConfig | None = None,
) -> PairwiseCorrectionCNN:
    """Build the baseline shared-CNN pairwise correction model."""
    cfg = config if isinstance(config, SharedCNNModelConfig) else SharedCNNModelConfig.from_dict(config)
    return PairwiseCorrectionCNN(output_dim=int(output_dim), config=cfg)


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Return model parameter count."""
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))
