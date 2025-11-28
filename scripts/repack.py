from __future__ import annotations

import datetime
import os
from pathlib import Path
import zipfile

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "exports", "results", "data/cached_samples"}


def should_exclude(path: Path, project_root: Path) -> bool:
    rel_parts = path.relative_to(project_root).parts
    for part in rel_parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def repack(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    export_dir = project_root / "exports"
    export_dir.mkdir(exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = export_dir / f"nqs-sqd-transformer-lab-{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_root):
            root_path = Path(root)
            dirs[:] = [
                d for d in dirs
                if not should_exclude(root_path / d, project_root)
            ]
            for file in files:
                full_path = root_path / file
                if should_exclude(full_path, project_root):
                    continue
                rel_path = full_path.relative_to(project_root)
                zf.write(full_path, rel_path)

    print(f"[INFO] Wrote zip to: {zip_path}")
    return zip_path


if __name__ == "__main__":
    repack()
