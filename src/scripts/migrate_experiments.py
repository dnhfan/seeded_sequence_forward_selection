from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable

SOURCE_DIR = Path("results")
DEST_DIR = Path("results")
DEFAULT_VARIANT = "raw"
DEFAULT_ALGO = "SFS"
DEFAULT_STAGE = "wrapper"
DRY_RUN = True

ALGO_PATTERNS = [
    r"sfs",
    r"rfe",
    r"lasso",
    r"mutual[_\s-]?information",
    r"mi",
]

HISTORY_NAMES = {
    "history.json",
    "history.txt",
    "history.csv",
    "sfs_report.txt",
    "report.txt",
}
FEATURE_NAMES = {"selected_features.csv"}
METRICS_NAMES = {"metrics.json", "metrics.csv"}


def detect_variant(path_text: str) -> str:
    if re.search(r"\braw\b", path_text, re.IGNORECASE):
        return "raw"
    if re.search(r"\bunion\b", path_text, re.IGNORECASE):
        return "union"
    return DEFAULT_VARIANT


def detect_algorithm(path_text: str) -> str:
    for pattern in ALGO_PATTERNS:
        if re.search(pattern, path_text, re.IGNORECASE):
            token = re.sub(r"[^a-zA-Z]", "", pattern).upper()
            if token == "MI" or "MUTUAL" in token:
                return "MI"
            return token
    return DEFAULT_ALGO


def classify_target_subdir(file_path: Path) -> str:
    name = file_path.name.lower()
    if name.startswith("history"):
        return "history"
    if name in HISTORY_NAMES:
        return "history"
    if name in FEATURE_NAMES:
        return "features"
    if name in METRICS_NAMES:
        return "metrics"
    if name == "config.json":
        return "artifacts"
    if name.endswith(".pkl"):
        return "artifacts"
    if name.endswith(".json"):
        return "metrics"
    if name.endswith(".csv"):
        return "features"
    return "artifacts"


def iter_old_runs(source_dir: Path) -> Iterable[Path]:
    for dataset_dir in source_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        for run_dir in dataset_dir.glob("run_*"):
            if run_dir.is_dir():
                yield run_dir


def migrate(dry_run: bool = True) -> None:
    for run_dir in iter_old_runs(SOURCE_DIR):
        dataset_name = run_dir.parent.name
        path_text = str(run_dir)

        dataset_variant = detect_variant(path_text)
        algorithm_name = detect_algorithm(path_text)

        run_id = run_dir.name
        if DEST_DIR.name == dataset_name:
            dataset_root = DEST_DIR
        else:
            dataset_root = DEST_DIR / dataset_name

        dest_run_root = (
            dataset_root / DEFAULT_STAGE / dataset_variant / algorithm_name / run_id
        )

        if dry_run:
            print(f"\n[PLAN] Run: {run_dir}")
            print(f"  Detected variant: {dataset_variant}")
            print(f"  Detected algo:    {algorithm_name}")
            print(f"  Target root:      {dest_run_root}")
        else:
            for sub in ["history", "features", "metrics", "artifacts"]:
                (dest_run_root / sub).mkdir(parents=True, exist_ok=True)

        for item in run_dir.iterdir():
            if item.is_dir():
                if dry_run:
                    print(f"  [SKIP DIR] {item}")
                continue

            subdir = classify_target_subdir(item)
            dest = dest_run_root / subdir / item.name

            if dry_run:
                print(f"  [COPY] {item} -> {dest}")
            else:
                shutil.copy2(item, dest)


if __name__ == "__main__":
    migrate(dry_run=DRY_RUN)
