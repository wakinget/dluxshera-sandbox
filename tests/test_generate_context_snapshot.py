from devtools import generate_context_snapshot


def test_context_snapshot_reports_param_specs_and_transforms(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    meta = generate_context_snapshot.generate_context_snapshot(
        root=".", out_dir=str(snapshot_dir), max_depth=2
    )

    md_path = snapshot_dir / "context_snapshot.md"
    assert md_path.exists(), "Markdown summary should be written"

    md_text = md_path.read_text(encoding="utf-8")
    for expected in (
        "forward_threeplane",
        "forward_twoplane",
        "inference_basic",
        "system.plate_scale_as_per_pix",
        "binary.log_flux_total",
    ):
        assert expected in md_text

    specs = meta.get("param_specs", {}).get("specs", [])
    assert any(spec.get("system_id") == "shera_threeplane" for spec in specs)
    assert any(spec.get("system_id") == "shera_twoplane" for spec in specs)

    transforms = meta.get("transforms", {}).get("systems", {})
    assert "shera_threeplane" in transforms
    transform_keys = transforms["shera_threeplane"].get("transform_keys", [])
    assert "system.plate_scale_as_per_pix" in transform_keys
    assert "binary.log_flux_total" in transform_keys
