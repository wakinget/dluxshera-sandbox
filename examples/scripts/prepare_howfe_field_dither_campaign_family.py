"""Prepare the 2026-07 HO-WFE/field-dither full-fidelity campaign family."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from dluxshera.utils.campaign_high_order_wfe import apply_high_order_wfe_campaign_config

from prepare_full_fidelity_campaign_shards import (
    Resources,
    prepare_shards_for_configs,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    ROOT
    / "examples/recipes/full_fidelity_next_campaigns_20260703/"
    "full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1.yaml"
)
OUTDIR = ROOT / "examples/recipes/full_fidelity_howfe_field_dither_20260720"
TRUTH_SEED = 20260610
KNOWLEDGE_ERROR_SEED = 20260720
AMPLITUDES_NM = (0.01, 0.05, 0.1, 0.5, 1.0)
FIELD_OFFSETS = (
    ("xp0p0_yp0p0", 0.0, 0.0),
    ("xp1p0_yp0p0", 1.0, 0.0),
    ("xm1p0_yp0p0", -1.0, 0.0),
    ("xp0p0_yp1p0", 0.0, 1.0),
    ("xp0p0_ym1p0", 0.0, -1.0),
)
MIRRORS = ("primary", "secondary")
RESULTS_ROOT = Path("/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return copy.deepcopy(dict(payload))


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _amp_token(value: float) -> str:
    if value < 0.1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.1f}"
    if "." not in text:
        text = f"{text}.0"
    if text.endswith("0") and not text.endswith(".0"):
        text = text.rstrip("0")
    return text.replace(".", "p") + "nm"


def _mirror_token(mirror: str | None) -> str:
    if mirror == "primary":
        return "m1"
    if mirror == "secondary":
        return "m2"
    return "noke"


def _condition_name(
    *,
    mirror: str | None,
    amplitude_nm: float,
    field_token: str,
    cadence_token: str,
) -> str:
    if mirror is None:
        return f"noke_{field_token}_{cadence_token}"
    return f"{_mirror_token(mirror)}_hoke_{_amp_token(amplitude_nm)}_{field_token}_{cadence_token}"


def _set_offsets(config: dict[str, Any], *, x_as: float, y_as: float, pa_deg: float) -> None:
    trace = config["experiment"]["subblocks"]["trace_source"]
    processing = dict(trace.get("processing", {}) or {})
    processing["offsets"] = {
        "source.x_position_as": float(x_as),
        "source.y_position_as": float(y_as),
        "source.position_angle_deg": float(pa_deg),
    }
    trace["processing"] = processing


def _set_cadence(config: dict[str, Any], *, windows: int, subblocks_per_window: int, draws: int) -> None:
    experiment = config["experiment"]
    experiment["iterative"]["windows_per_draw"] = int(windows)
    experiment["iterative"]["subblocks_per_window"] = int(subblocks_per_window)
    forecast = dict(experiment.get("iterative_forecast", {}) or {})
    forecast["actual_windows"] = int(windows)
    forecast["subblocks_per_window"] = int(subblocks_per_window)
    experiment["iterative_forecast"] = forecast
    prior = experiment["prior_draws"]
    prior["n_cases"] = int(draws)
    prior["draws_per_condition"] = int(draws)


def _set_single_condition(config: dict[str, Any], *, name: str, metadata: Mapping[str, Any]) -> None:
    prior = config["experiment"]["prior_draws"]
    condition = copy.deepcopy(prior["conditions"][0])
    condition["condition_name"] = name
    condition["metadata"] = dict(metadata)
    prior["conditions"] = [condition]
    config["experiment"]["campaign_family_condition"] = dict(metadata)


def _set_howfe(
    config: dict[str, Any],
    *,
    mirror: str | None,
    amplitude_nm: float,
) -> None:
    howfe = copy.deepcopy(config["experiment"]["high_order_wfe"])
    howfe["enabled"] = True
    howfe["truth"]["enabled"] = True
    howfe["truth"]["mirrors"] = ["primary", "secondary"]
    howfe["truth"]["seed"] = TRUTH_SEED
    knowledge = dict(howfe["inference"].get("knowledge_error", {}) or {})
    knowledge["enabled"] = mirror is not None
    knowledge["seed"] = KNOWLEDGE_ERROR_SEED
    knowledge["pairing"] = "independent"
    knowledge["power_law_alpha"] = "same_as_truth"
    knowledge["remove_low_order_zernikes"] = True
    knowledge["mirrors"] = {
        "primary": {
            "enabled": mirror == "primary",
            "amplitude_nm_rms": float(amplitude_nm if mirror == "primary" else 0.0),
        },
        "secondary": {
            "enabled": mirror == "secondary",
            "amplitude_nm_rms": float(amplitude_nm if mirror == "secondary" else 0.0),
        },
    }
    if mirror is None:
        howfe["validation"]["require_nonzero_difference_when_enabled"] = False
    howfe["inference"]["knowledge_error"] = knowledge
    config["experiment"]["high_order_wfe"] = howfe


def _hash_metadata(config: Mapping[str, Any], *, map_group: str) -> dict[str, Any]:
    result = apply_high_order_wfe_campaign_config(
        system_cfg={"source": {"kind": "binary_target"}, "optics": {}},
        high_order_wfe_cfg=config["experiment"]["high_order_wfe"],
        seed_context={"campaign_family": map_group},
        write_artifacts=False,
    )
    prov = result.provenance
    primary = prov["primary"]
    secondary = prov["secondary"]
    return {
        "truth_seed": prov.get("truth_seed"),
        "knowledge_error_seed": prov.get("knowledge_error_seed"),
        "primary_ke_map_hash": primary.get("normalised_knowledge_error_map_hash")
        or primary.get("knowledge_error_map_hash", ""),
        "secondary_ke_map_hash": secondary.get("normalised_knowledge_error_map_hash")
        or secondary.get("knowledge_error_map_hash", ""),
        "map_group": map_group,
    }


def _condition_config(
    base: Mapping[str, Any],
    *,
    mirror: str | None,
    amplitude_nm: float,
    field_token: str,
    x_as: float,
    y_as: float,
    pa_deg: float,
    cadence_token: str,
    windows: int,
    subblocks_per_window: int,
    draws: int,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    name = _condition_name(
        mirror=mirror,
        amplitude_nm=amplitude_nm,
        field_token=field_token,
        cadence_token=cadence_token,
    )
    config["experiment"]["run_name"] = f"ff_howfe_field_{name}_v1"
    _set_howfe(config, mirror=mirror, amplitude_nm=amplitude_nm)
    _set_offsets(config, x_as=x_as, y_as=y_as, pa_deg=pa_deg)
    _set_cadence(
        config,
        windows=windows,
        subblocks_per_window=subblocks_per_window,
        draws=draws,
    )
    map_group = f"truth{TRUTH_SEED}_ke{KNOWLEDGE_ERROR_SEED}_independent"
    metadata = {
        "campaign_family": "full_fidelity_howfe_field_dither_20260720",
        "ho_ke_active_mirror": "" if mirror is None else mirror,
        "ho_ke_primary_enabled": mirror == "primary",
        "ho_ke_secondary_enabled": mirror == "secondary",
        "ho_ke_primary_amplitude_nm_rms": float(amplitude_nm if mirror == "primary" else 0.0),
        "ho_ke_secondary_amplitude_nm_rms": float(amplitude_nm if mirror == "secondary" else 0.0),
        "field_offset_x_as": float(x_as),
        "field_offset_y_as": float(y_as),
        "field_offset_pa_deg": float(pa_deg),
    }
    metadata.update(_hash_metadata(config, map_group=map_group))
    _set_single_condition(config, name=name, metadata=metadata)
    return config


def _write_edge_submit_script(group_dir: Path) -> None:
    source = group_dir / "submit_draw_shards.sh"
    target = group_dir / "submit_draw_shards_edge.sh"
    text = source.read_text(encoding="utf-8")
    text = text.replace("\nsbatch ", "\nsbatch -M edge ")
    text = text.replace(
        "# conda activate dluxshera-py311",
        "source /cm/shared/apps/miniforge/etc/profile.d/conda.sh\n"
        "conda activate /scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311",
    )
    target.write_text(text, encoding="utf-8")
    target.chmod(0o755)


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_readme(root: Path, summary: Mapping[str, Any]) -> None:
    readme = f"""# HO-WFE Field-Dither Campaign Family

Prepared artifacts only. No validation or production jobs were submitted.

Base config: `{BASE_CONFIG.relative_to(ROOT).as_posix()}`

Fixed seeds:
- truth maps: `{TRUTH_SEED}`
- knowledge-error maps: `{KNOWLEDGE_ERROR_SEED}`

Production HO-WFE KE matrix:
- 2 mirrors x 5 amplitudes x 5 fields x 10 draws = 500 draw shards
- no-KE field controls: 5 fields x 10 draws = 50 draw shards

Priority launch order:
1. `shards/production_center/submit_draw_shards_edge.sh`
2. `shards/production_m2_offaxis/submit_draw_shards_edge.sh`
3. `shards/production_m1_offaxis/submit_draw_shards_edge.sh`
4. `shards/controls/submit_draw_shards_edge.sh`

Validation wave:
- `shards/validation/submit_draw_shards_edge.sh`

Cheap audits:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config {summary['audit_examples']['m1_center']}
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config {summary['audit_examples']['m2_center']}
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config {summary['audit_examples']['zero_offset']}
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config {summary['audit_examples']['off_axis']}
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config {summary['audit_examples']['m1_center']} --check-shard-manifest {summary['audit_examples']['manifest']}
```

Compute-node preflight commands are generated as `preflight_draw_shards.sh` in
each shard group. They are intentionally marked expensive and were not run.

TACC portability note: the scientific YAMLs avoid scheduler-specific fields.
A TACC wrapper still needs to provide module/environment activation, repository
and data paths, results root, Slurm account/partition, CPU/memory requests, JAX
cache path, and any submission wrapper differences.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def prepare_family(*, outdir: Path, overwrite: bool) -> dict[str, Any]:
    base = _read_yaml(BASE_CONFIG)
    production_dir = outdir / "configs" / "production"
    validation_dir = outdir / "configs" / "validation"
    controls_dir = outdir / "configs" / "controls"
    if outdir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing family directory: {outdir}")

    cadence_token = "w10x30"
    validation_cadence = "w1x30"
    production_paths: dict[str, Path] = {}
    center_paths: list[Path] = []
    m2_offaxis_paths: list[Path] = []
    m1_offaxis_paths: list[Path] = []
    control_paths: list[Path] = []

    for mirror in MIRRORS:
        for amplitude in AMPLITUDES_NM:
            for field_token, x_as, y_as in FIELD_OFFSETS:
                config = _condition_config(
                    base,
                    mirror=mirror,
                    amplitude_nm=amplitude,
                    field_token=field_token,
                    x_as=x_as,
                    y_as=y_as,
                    pa_deg=0.0,
                    cadence_token=cadence_token,
                    windows=10,
                    subblocks_per_window=30,
                    draws=10,
                )
                path = production_dir / f"{config['experiment']['run_name']}.yaml"
                _write_yaml(path, config)
                production_paths[config["experiment"]["run_name"]] = path
                if field_token == "xp0p0_yp0p0":
                    center_paths.append(path)
                elif mirror == "secondary":
                    m2_offaxis_paths.append(path)
                else:
                    m1_offaxis_paths.append(path)

    for field_token, x_as, y_as in FIELD_OFFSETS:
        config = _condition_config(
            base,
            mirror=None,
            amplitude_nm=0.0,
            field_token=field_token,
            x_as=x_as,
            y_as=y_as,
            pa_deg=0.0,
            cadence_token=cadence_token,
            windows=10,
            subblocks_per_window=30,
            draws=10,
        )
        path = controls_dir / f"{config['experiment']['run_name']}.yaml"
        _write_yaml(path, config)
        control_paths.append(path)

    validation_specs = [
        (None, 0.0, "xp0p0_yp0p0", 0.0, 0.0),
        ("primary", 0.1, "xp0p0_yp0p0", 0.0, 0.0),
        ("secondary", 0.1, "xp0p0_yp0p0", 0.0, 0.0),
        ("secondary", 0.1, "xp1p0_yp0p0", 1.0, 0.0),
        ("secondary", 0.1, "xm1p0_yp0p0", -1.0, 0.0),
        ("secondary", 0.1, "xp0p0_yp1p0", 0.0, 1.0),
        ("secondary", 0.1, "xp0p0_ym1p0", 0.0, -1.0),
        ("primary", 0.1, "xp1p0_yp0p0", 1.0, 0.0),
    ]
    validation_paths: list[Path] = []
    for mirror, amplitude, field_token, x_as, y_as in validation_specs:
        config = _condition_config(
            base,
            mirror=mirror,
            amplitude_nm=amplitude,
            field_token=field_token,
            x_as=x_as,
            y_as=y_as,
            pa_deg=0.0,
            cadence_token=validation_cadence,
            windows=1,
            subblocks_per_window=30,
            draws=1,
        )
        path = validation_dir / f"{config['experiment']['run_name']}.yaml"
        _write_yaml(path, config)
        validation_paths.append(path)

    spec = {
        "schema_version": "howfe_field_dither_campaign_family.v1",
        "base_config": BASE_CONFIG.relative_to(ROOT).as_posix(),
        "truth_seed": TRUTH_SEED,
        "knowledge_error_seed": KNOWLEDGE_ERROR_SEED,
        "amplitudes_nm_rms": list(AMPLITUDES_NM),
        "field_offsets": [
            {"token": token, "x_as": x, "y_as": y, "pa_deg": 0.0}
            for token, x, y in FIELD_OFFSETS
        ],
        "production_conditions": len(production_paths),
        "production_ho_ke_draw_shards": len(production_paths) * 10,
        "control_conditions": len(control_paths),
        "control_draw_shards": len(control_paths) * 10,
        "validation_conditions": len(validation_paths),
    }
    _write_yaml(outdir / "family_sweep.yaml", spec)

    resources = Resources(time="12:00:00", cpus_per_task=10, mem="128G", max_workers=5)
    groups = {
        "validation": validation_paths,
        "production_center": center_paths,
        "production_m2_offaxis": m2_offaxis_paths,
        "production_m1_offaxis": m1_offaxis_paths,
        "controls": control_paths,
    }
    group_rows: dict[str, int] = {}
    for group_name, paths in groups.items():
        group_dir = outdir / "shards" / group_name
        prepare_shards_for_configs(
            config_paths=paths,
            outdir=group_dir,
            run_name_prefix=f"ff_howfe_{group_name}",
            mode="draw",
            results_root=RESULTS_ROOT,
            resources=resources,
            dry_run=False,
            overwrite=True,
        )
        _write_edge_submit_script(group_dir)
        group_rows[group_name] = len(_manifest_rows(group_dir / "shard_manifest.csv"))

    launch_commands = {
        name: f"{shlex.quote(str((outdir / 'shards' / name / 'submit_draw_shards_edge.sh').relative_to(ROOT)))}"
        for name in groups
    }
    summary = {
        **spec,
        "groups": group_rows,
        "launch_commands": launch_commands,
        "audit_examples": {
            "m1_center": next(
                str(path.relative_to(ROOT))
                for key, path in production_paths.items()
                if "m1_hoke_0p1nm_xp0p0_yp0p0" in key
            ),
            "m2_center": next(
                str(path.relative_to(ROOT))
                for key, path in production_paths.items()
                if "m2_hoke_0p1nm_xp0p0_yp0p0" in key
            ),
            "zero_offset": str(control_paths[0].relative_to(ROOT)),
            "off_axis": next(
                str(path.relative_to(ROOT))
                for key, path in production_paths.items()
                if "m2_hoke_0p1nm_xp1p0_yp0p0" in key
            ),
            "manifest": str((outdir / "shards/production_center/shard_manifest.csv").relative_to(ROOT)),
        },
    }
    (outdir / "family_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_readme(outdir, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the HO-WFE KE field-dither campaign family artifacts."
    )
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = prepare_family(outdir=args.outdir, overwrite=bool(args.overwrite))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
