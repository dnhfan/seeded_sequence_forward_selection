from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RunPaths:
    run_root: Path
    history_dir: Path
    features_dir: Path
    metrics_dir: Path
    artifacts_dir: Path

    history_json: Path
    history_txt: Path
    selected_features_csv: Path
    metrics_json: Path
    metrics_csv: Path

    def ensure_dirs(self) -> None:
        for path in [
            self.run_root,
            self.history_dir,
            self.features_dir,
            self.metrics_dir,
            self.artifacts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def build_run_paths(
    base_results_dir: Path,
    dataset_name: str,
    pipeline_stage: str,
    dataset_variant: str,
    algorithm_name: str,
    run_tag: Optional[str] = None,
) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    if run_tag:
        run_id = f"{run_id}_{run_tag}"

    if base_results_dir.name == dataset_name:
        dataset_root = base_results_dir
    else:
        dataset_root = base_results_dir / dataset_name

    run_root = dataset_root / pipeline_stage / dataset_variant / algorithm_name / run_id

    history_dir = run_root / "history"
    features_dir = run_root / "features"
    metrics_dir = run_root / "metrics"
    artifacts_dir = run_root / "artifacts"

    return RunPaths(
        run_root=run_root,
        history_dir=history_dir,
        features_dir=features_dir,
        metrics_dir=metrics_dir,
        artifacts_dir=artifacts_dir,
        history_json=history_dir / "history.json",
        history_txt=history_dir / "history.txt",
        selected_features_csv=features_dir / "selected_features.csv",
        metrics_json=metrics_dir / "metrics.json",
        metrics_csv=metrics_dir / "metrics.csv",
    )
