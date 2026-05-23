"""Shared CLI plumbing for observation sub-block Schur study wrappers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableSequence
from typing import Any


REFERENCE_OPTIMIZER_FLAG_MAP = {
    "reference_optimizer_kind": "--reference-optimizer-kind",
    "reference_base_lr": "--reference-base-lr",
    "reference_n_iter": "--reference-n-iter",
    "reference_optimizer_kwargs": "--reference-optimizer-kwarg",
    "reference_diagnostics_profile": "--reference-diagnostics-profile",
    "reference_init_mode": "--reference-init-mode",
    "reuse_reference_inference": "--reuse-reference-inference",
}
REFERENCE_SCHEDULE_FLAG_MAP = {
    "reference_schedule_kind": "--reference-schedule-kind",
    "reference_schedule_warmup_steps": "--reference-schedule-warmup-steps",
    "reference_schedule_start_factor": "--reference-schedule-start-factor",
    "reference_schedule_min_factor": "--reference-schedule-min-factor",
    "reference_schedule_boundaries": "--reference-schedule-boundaries",
    "reference_schedule_factors": "--reference-schedule-factors",
    "reference_schedule_decay_rate": "--reference-schedule-decay-rate",
    "reference_schedule_transition_steps": "--reference-schedule-transition-steps",
    "reference_schedule_staircase": "--reference-schedule-staircase",
}
REFERENCE_PRECONDITIONING_FLAG_MAP = {
    "reference_preconditioning_enabled": "--reference-preconditioning-enabled",
    "reference_preconditioning_disabled": "--reference-preconditioning-disabled",
    "reference_preconditioning_method": "--reference-preconditioning-method",
    "reference_preconditioning_reference": "--reference-preconditioning-reference",
    "reference_preconditioning_damping": "--reference-preconditioning-damping",
    "reference_preconditioning_eig_floor_rel": "--reference-preconditioning-eig-floor-rel",
    "reference_preconditioning_eig_floor_abs": "--reference-preconditioning-eig-floor-abs",
    "reference_preconditioning_lr_clip": "--reference-preconditioning-lr-clip",
}
REFERENCE_EARLY_STOPPING_FLAG_MAP = {
    "reference_early_stopping_enabled": "--reference-early-stopping",
    "reference_early_stopping_min_iter": "--reference-early-stopping-min-iter",
    "reference_early_stopping_patience": "--reference-early-stopping-patience",
    "reference_early_stopping_loss_rtol": "--reference-early-stopping-loss-rtol",
    "reference_early_stopping_loss_atol": "--reference-early-stopping-loss-atol",
    "reference_early_stopping_step_atol": "--reference-early-stopping-step-atol",
    "reference_early_stopping_grad_norm_atol": "--reference-early-stopping-grad-norm-atol",
}
SCHUR_FRAME_QUALITY_FLAG_MAP = {
    "schur_frame_quality_policy": "--schur-frame-quality-policy",
    "schur_frame_chi2_threshold": "--schur-frame-chi2-threshold",
    "schur_frame_quality_missing": "--schur-frame-quality-missing",
    "schur_frame_mask_denominator": "--schur-frame-mask-denominator",
    "schur_frame_mask_min_good_frames": "--schur-frame-mask-min-good-frames",
}


def add_reference_optimizer_args(parser: argparse.ArgumentParser) -> None:
    """Add recovered-reference optimizer flags accepted by the study script."""

    parser.add_argument("--reference-optimizer-kind", choices=("sgd", "adam"), default=None)
    parser.add_argument("--reference-base-lr", type=float, default=None)
    parser.add_argument("--reference-n-iter", type=int, default=None)
    parser.add_argument(
        "--reference-optimizer-kwarg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--reference-schedule-kind",
        choices=(
            "constant",
            "linear_warmup",
            "piecewise_constant",
            "exponential_decay",
            "cosine_decay",
            "linear_warmup_cosine_decay",
        ),
        default=None,
    )
    parser.add_argument("--reference-schedule-warmup-steps", type=int, default=None)
    parser.add_argument("--reference-schedule-start-factor", type=float, default=None)
    parser.add_argument("--reference-schedule-min-factor", type=float, default=None)
    parser.add_argument("--reference-schedule-boundaries", default=None)
    parser.add_argument("--reference-schedule-factors", default=None)
    parser.add_argument("--reference-schedule-decay-rate", type=float, default=None)
    parser.add_argument("--reference-schedule-transition-steps", type=int, default=None)
    parser.add_argument("--reference-schedule-staircase", action="store_true", default=False)

    preconditioning = parser.add_mutually_exclusive_group()
    preconditioning.add_argument(
        "--reference-preconditioning-enabled",
        dest="reference_preconditioning_enabled",
        action="store_const",
        const=True,
        default=None,
    )
    preconditioning.add_argument(
        "--reference-preconditioning-disabled",
        dest="reference_preconditioning_enabled",
        action="store_const",
        const=False,
    )
    parser.add_argument("--reference-preconditioning-method", default=None)
    parser.add_argument(
        "--reference-preconditioning-reference",
        choices=("initial", "truth_when_available"),
        default=None,
    )
    parser.add_argument("--reference-preconditioning-damping", type=float, default=None)
    parser.add_argument("--reference-preconditioning-eig-floor-rel", type=float, default=None)
    parser.add_argument("--reference-preconditioning-eig-floor-abs", type=float, default=None)
    parser.add_argument("--reference-preconditioning-lr-clip", default=None)

    parser.add_argument("--reference-early-stopping", action="store_true", default=False)
    parser.add_argument("--reference-early-stopping-min-iter", type=int, default=None)
    parser.add_argument("--reference-early-stopping-patience", type=int, default=None)
    parser.add_argument("--reference-early-stopping-loss-rtol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-loss-atol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-step-atol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-grad-norm-atol", type=float, default=None)
    parser.add_argument(
        "--reference-init-mode",
        choices=("initial", "truth_when_available"),
        default=None,
    )


def collect_reference_optimizer_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect optional reference fields without requiring a full parser namespace."""

    keys = (
        *REFERENCE_OPTIMIZER_FLAG_MAP,
        *REFERENCE_SCHEDULE_FLAG_MAP,
        "reference_preconditioning_enabled",
        *(
            key
            for key in REFERENCE_PRECONDITIONING_FLAG_MAP
            if key
            not in {
                "reference_preconditioning_enabled",
                "reference_preconditioning_disabled",
            }
        ),
        "reference_early_stopping",
        *tuple(key for key in REFERENCE_EARLY_STOPPING_FLAG_MAP if key != "reference_early_stopping_enabled"),
    )
    return {key: getattr(args, key, None) for key in keys}


def _csv_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _append_scalar_flags(
    command: MutableSequence[str],
    cfg: Mapping[str, Any],
    flag_map: Mapping[str, str],
) -> None:
    for key, flag in flag_map.items():
        value = cfg.get(key)
        if value is not None:
            command.extend([flag, _csv_value(value)])


def _schedule_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {key: cfg[key] for key in REFERENCE_SCHEDULE_FLAG_MAP if key in cfg}
    schedule = cfg.get("reference_schedule")
    if isinstance(schedule, Mapping):
        for key in (
            "kind",
            "warmup_steps",
            "start_factor",
            "min_factor",
            "boundaries",
            "factors",
            "decay_rate",
            "transition_steps",
            "staircase",
        ):
            if key in schedule:
                flattened[f"reference_schedule_{key}"] = schedule[key]
    return flattened


def append_reference_optimizer_flags(
    command: MutableSequence[str],
    cfg: Mapping[str, Any],
) -> None:
    """Forward recovered-reference settings from one config mapping."""

    _append_scalar_flags(
        command,
        cfg,
        {
            key: flag
            for key, flag in REFERENCE_OPTIMIZER_FLAG_MAP.items()
            if key != "reference_optimizer_kwargs"
        },
    )
    optimizer_kwargs = cfg.get("reference_optimizer_kwargs")
    if isinstance(optimizer_kwargs, Mapping):
        for key, value in sorted(optimizer_kwargs.items()):
            command.extend([REFERENCE_OPTIMIZER_FLAG_MAP["reference_optimizer_kwargs"], f"{key}={value}"])
    elif isinstance(optimizer_kwargs, (tuple, list)):
        for value in optimizer_kwargs:
            command.extend([REFERENCE_OPTIMIZER_FLAG_MAP["reference_optimizer_kwargs"], str(value)])

    schedule_cfg = _schedule_cfg(cfg)
    staircase = bool(schedule_cfg.pop("reference_schedule_staircase", False))
    _append_scalar_flags(command, schedule_cfg, REFERENCE_SCHEDULE_FLAG_MAP)
    if staircase:
        command.append(REFERENCE_SCHEDULE_FLAG_MAP["reference_schedule_staircase"])

    enabled = cfg.get("reference_preconditioning_enabled")
    if enabled is True:
        command.append(REFERENCE_PRECONDITIONING_FLAG_MAP["reference_preconditioning_enabled"])
    elif enabled is False:
        command.append(REFERENCE_PRECONDITIONING_FLAG_MAP["reference_preconditioning_disabled"])
    _append_scalar_flags(
        command,
        cfg,
        {
            key: flag
            for key, flag in REFERENCE_PRECONDITIONING_FLAG_MAP.items()
            if key not in {"reference_preconditioning_enabled", "reference_preconditioning_disabled"}
        },
    )

    if cfg.get("reference_early_stopping_enabled") is True:
        command.append(REFERENCE_EARLY_STOPPING_FLAG_MAP["reference_early_stopping_enabled"])
    _append_scalar_flags(
        command,
        cfg,
        {
            key: flag
            for key, flag in REFERENCE_EARLY_STOPPING_FLAG_MAP.items()
            if key != "reference_early_stopping_enabled"
        },
    )


def append_schur_frame_quality_flags(
    command: MutableSequence[str],
    cfg: Mapping[str, Any],
) -> None:
    """Forward optional Schur frame-quality settings."""

    _append_scalar_flags(command, cfg, SCHUR_FRAME_QUALITY_FLAG_MAP)
