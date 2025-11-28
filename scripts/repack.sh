#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_DIR="${PROJECT_ROOT}/exports"
mkdir -p "${EXPORT_DIR}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
ZIP_NAME="nqs-sqd-transformer-lab-${TIMESTAMP}.zip"
ZIP_PATH="${EXPORT_DIR}/${ZIP_NAME}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Exporting to: ${ZIP_PATH}"

cd "${PROJECT_ROOT}"

python - << 'EOF'
import os, zipfile, datetime, pathlib

project_root = pathlib.Path(os.getcwd())
export_dir = project_root / "exports"
export_dir.mkdir(exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
zip_path = export_dir / f"nqs-sqd-transformer-lab-{timestamp}.zip"

EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "exports", "results", "data/cached_samples"}
EXCLUDE_PREFIXES = tuple(str(project_root / d) for d in EXCLUDE_DIRS)

def should_exclude(path: pathlib.Path) -> bool:
    p_str = str(path)
    for prefix in EXCLUDE_PREFIXES:
        if p_str.startswith(prefix):
            return True
    return False

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(project_root):
        root_path = pathlib.Path(root)
        dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]
        for file in files:
            full_path = root_path / file
            if should_exclude(full_path):
                continue
            rel_path = full_path.relative_to(project_root)
            zf.write(full_path, rel_path)

print(f"[INFO] Wrote zip to: {zip_path}")
EOF

echo "[DONE] Repack complete."
