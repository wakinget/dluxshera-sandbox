# notebook_setup.py
from pathlib import Path


def setup_paths():
    """Ensure notebooks run against the installed package, not path hacks."""

    repo_root = Path(__file__).resolve().parents[2]
    try:
        import dluxshera  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "dluxshera is not installed. Run `python -m pip install -e .` from the "
            "repository root before executing notebooks."
        ) from exc

    print("Repo root set to:", repo_root)
    print("CWD:", Path.cwd())
    return repo_root
