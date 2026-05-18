from pathlib import Path    

def _find_project_root(start: str) -> Path:
    start = Path(start).resolve().parent
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() or (path / "README.md").exists():
            return path
    raise FileNotFoundError("Could not find project root")