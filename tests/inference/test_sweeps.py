from pathlib import Path

from dluxshera.inference.sweeps import collect_runs, write_sweep_csv


def _write_run(tmp_dir: Path, name: str) -> Path:
    run_dir = tmp_dir / name
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        '{"run_id": "%s", "status": "ok", "created_at": "ts"}' % name, encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        '{"run_id": "%s", "theta": {"dim": 2}, "optimizer": {"name": "adam"}}' % name,
        encoding="utf-8",
    )
    return run_dir


def test_collect_and_write_csv(tmp_path: Path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_a = _write_run(runs_root, "run_a")
    run_b = _write_run(runs_root, "run_b")

    found = collect_runs(runs_root)
    assert run_a in found and run_b in found

    out_csv = tmp_path / "sweep.csv"
    count = write_sweep_csv(runs_root, out_csv)
    assert count == 2

    contents = out_csv.read_text(encoding="utf-8").splitlines()
    # header + two rows
    assert len(contents) == 3
    header = contents[0].split(",")
    assert "run_id" in header
    assert "optimizer.name" in header
    assert "has_checkpoint_best" in header
