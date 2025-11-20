# notebook_setup.py
import sys
from pathlib import Path

def setup_paths(marker="Classes"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).is_dir():
            repo_root = parent
            break
    else:
        raise RuntimeError(f"Could not find repo root (no '{marker}' dir found)")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("Repo root set to:", repo_root)
    print("CWD:", current)
    return repo_root
