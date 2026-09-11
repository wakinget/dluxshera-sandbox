from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "dluxshera.ml.losses requires PyTorch. Install the optional ML environment."
    ) from exc

from .eigenbasis import (
    ScienceEigenbasis,
    build_science_mode_weights,
    validate_science_mode_weights_nonuniform,
)

__all__ = [
    "PairConsistencyConfig",
    "ScienceLossConfig",
    "ScienceLossHelper",
    "build_mode_weights",
    "pair_consistency_losses",
]


@dataclass(frozen=True)
class ScienceLossConfig:
    mode: str = "ordinary"
    strength: float = 0.5
    eigenvalue_floor: float = 1.0e-6
    weight_cap: float = 10.0

    def __post_init__(self) -> None:
        if self.mode not in {"ordinary", "strong_mode_weighted", "weak_mode_weighted"}:
            raise ValueError(f"Unsupported science_loss.mode {self.mode!r}.")
        if not np.isfinite(float(self.strength)) or float(self.strength) < 0.0:
            raise ValueError("science_loss.strength must be finite and >= 0.")
        if not np.isfinite(float(self.eigenvalue_floor)) or float(self.eigenvalue_floor) <= 0.0:
            raise ValueError("science_loss.eigenvalue_floor must be finite and > 0.")
        if not np.isfinite(float(self.weight_cap)) or float(self.weight_cap) < 1.0:
            raise ValueError("science_loss.weight_cap must be finite and >= 1.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ScienceLossConfig":
        if payload is None:
            return cls()
        return cls(
            mode=str(payload.get("mode", "ordinary")),
            strength=float(payload.get("strength", 0.5)),
            eigenvalue_floor=float(payload.get("eigenvalue_floor", 1.0e-6)),
            weight_cap=float(payload.get("weight_cap", 10.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "strength": float(self.strength),
            "eigenvalue_floor": float(self.eigenvalue_floor),
            "weight_cap": float(self.weight_cap),
        }


def build_mode_weights(
    eigenvalues: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    eigenvalue_floor: float = 1.0e-6,
    weight_cap: float = 10.0,
) -> np.ndarray:
    """Return normalized eigenmode weights with mean active weight one."""
    return build_science_mode_weights(
        eigenvalues,
        mode=mode,
        strength=strength,
        eigenvalue_floor=eigenvalue_floor,
        weight_cap=weight_cap,
    )


class ScienceLossHelper(nn.Module):
    """Science MSE in canonical z coordinates, optionally weighted in a fixed eigenbasis."""

    def __init__(
        self,
        config: ScienceLossConfig | Mapping[str, Any] | None = None,
        *,
        eigenbasis: ScienceEigenbasis | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, ScienceLossConfig) else ScienceLossConfig.from_dict(config)
        if self.config.mode != "ordinary" and eigenbasis is None:
            raise ValueError(f"science_loss.mode={self.config.mode!r} requires a fixed eigenbasis artifact.")
        vectors = None if eigenbasis is None else torch.as_tensor(eigenbasis.eigenvectors, dtype=torch.float32, device=device)
        weights = (
            np.ones((0,), dtype=np.float32)
            if eigenbasis is None
            else build_mode_weights(
                eigenbasis.eigenvalues,
                mode=self.config.mode,
                strength=self.config.strength,
                eigenvalue_floor=self.config.eigenvalue_floor,
                weight_cap=self.config.weight_cap,
            ).astype(np.float32)
        )
        validate_science_mode_weights_nonuniform(weights, mode=self.config.mode)
        self.register_buffer("eigenvectors", torch.empty(0) if vectors is None else vectors)
        self.register_buffer("mode_weights", torch.as_tensor(weights, dtype=torch.float32, device=device))
        self.eigenbasis = eigenbasis

    @property
    def uses_eigenbasis(self) -> bool:
        return self.config.mode != "ordinary"

    def forward(self, pred_z: torch.Tensor, target_z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred_z.shape != target_z.shape:
            raise ValueError(f"Prediction shape {tuple(pred_z.shape)} does not match target {tuple(target_z.shape)}.")
        error = pred_z - target_z
        ordinary = error.pow(2).mean()
        if not self.uses_eigenbasis:
            return ordinary, {
                "ordinary_mse": ordinary.detach(),
                "weighted_mse": ordinary.detach(),
            }
        coeff = error @ self.eigenvectors
        per_mode = coeff.pow(2).mean(dim=0)
        weighted = (per_mode * self.mode_weights).mean()
        diagnostics = {
            "ordinary_mse": ordinary.detach(),
            "weighted_mse": weighted.detach(),
            "strong_mode_mse": _mode_group_mean(per_mode, "strong").detach(),
            "middle_mode_mse": _mode_group_mean(per_mode, "middle").detach(),
            "weak_mode_mse": _mode_group_mean(per_mode, "weak").detach(),
        }
        return weighted, diagnostics

    def metadata(self) -> dict[str, Any]:
        payload = self.config.to_dict()
        if self.eigenbasis is not None:
            payload["eigenbasis"] = {
                "artifact_id": self.eigenbasis.artifact_id,
                "content_sha256": self.eigenbasis.content_identity.get("sha256"),
                "coordinate_convention": self.eigenbasis.coordinate_convention,
                "source_matrix_coordinate_space": self.eigenbasis.source_matrix_coordinate_space,
                "eigenbasis_coordinate_space": self.eigenbasis.eigenbasis_coordinate_space,
                "parameter_labels": list(self.eigenbasis.parameter_labels),
                "normalization": self.eigenbasis.normalization,
                "mode_weights": self.mode_weights.detach().cpu().numpy().astype(float).tolist(),
            }
        return payload


def _mode_group_mean(values: torch.Tensor, group: str) -> torch.Tensor:
    n = int(values.shape[0])
    if n == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    third = max(n // 3, 1)
    if group == "strong":
        subset = values[:third]
    elif group == "weak":
        subset = values[-third:]
    else:
        subset = values[third : n - third] if n > 2 * third else values
    return subset.mean()


@dataclass(frozen=True)
class PairConsistencyConfig:
    antisymmetry_weight: float = 0.0
    identity_weight: float = 0.0
    noise_consistency_weight: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PairConsistencyConfig":
        if payload is None:
            return cls()
        return cls(
            antisymmetry_weight=float(payload.get("antisymmetry_weight", 0.0)),
            identity_weight=float(payload.get("identity_weight", 0.0)),
            noise_consistency_weight=float(payload.get("noise_consistency_weight", 0.0)),
        )

    def __post_init__(self) -> None:
        for name in ("antisymmetry_weight", "identity_weight", "noise_consistency_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"pair_consistency.{name} must be finite and >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "antisymmetry_weight": float(self.antisymmetry_weight),
            "identity_weight": float(self.identity_weight),
            "noise_consistency_weight": float(self.noise_consistency_weight),
        }


def _science_forward(model: nn.Module, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
    if getattr(model, "nuisance_head", None) is not None:
        return model.forward_multitask(image_a, image_b)["science"]
    return model(image_a, image_b)


def pair_consistency_losses(
    model: nn.Module,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    *,
    image_a_noise_view2: torch.Tensor | None = None,
    image_b_noise_view2: torch.Tensor | None = None,
    pred_ab: torch.Tensor | None = None,
    config: PairConsistencyConfig | Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config if isinstance(config, PairConsistencyConfig) else PairConsistencyConfig.from_dict(config)
    reference = pred_ab if pred_ab is not None else _science_forward(model, image_a, image_b)
    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    losses = {
        "antisymmetry": zero,
        "identity": zero,
        "noise_consistency": zero,
    }
    if cfg.antisymmetry_weight > 0.0:
        pred_ba = _science_forward(model, image_b, image_a)
        losses["antisymmetry"] = (reference + pred_ba).pow(2).mean()
    if cfg.identity_weight > 0.0:
        pred_aa = _science_forward(model, image_a, image_a)
        pred_bb = _science_forward(model, image_b, image_b)
        losses["identity"] = 0.5 * (pred_aa.pow(2).mean() + pred_bb.pow(2).mean())
    if cfg.noise_consistency_weight > 0.0:
        if image_a_noise_view2 is None or image_b_noise_view2 is None:
            raise ValueError("noise_consistency_weight > 0 requires second noisy pair views in the batch.")
        pred_view2 = _science_forward(model, image_a_noise_view2, image_b_noise_view2)
        losses["noise_consistency"] = (reference - pred_view2).pow(2).mean()
    return losses
