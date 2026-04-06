from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ProjectPath:
    """Centralized path management using pathlib.

    Rule A: data/processed/ is ONLY for machine-readable pipeline data.
    Rule B: results/ is strictly for human-readable experiment tracking.
    """

    data_name: str
    n_features: int = 50
    base_dir: Path = Path(".")

    @property
    def raw_path(self) -> Path:
        return self.base_dir / "data" / "raw" / f"{self.data_name}.csv"

    @property
    def processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed" / self.data_name

    @property
    def clean_dir(self) -> Path:
        return self.processed_dir / "01_clean"

    @property
    def filter_dir(self) -> Path:
        return self.processed_dir / "02_filter"

    @property
    def ensemble_dir(self) -> Path:
        return self.processed_dir / "03_ensemble"

    @property
    def wrapper_dir(self) -> Path:
        return self.processed_dir / "04_wrapper"

    @property
    def result_dir(self) -> Path:
        return self.base_dir / "results"

    @property
    def eda_result_dir(self) -> Path:
        """Thư mục lưu kết quả phân tích dữ liệu (Exploratory Data Analysis)"""
        return self.result_dir / self.data_name / "eda"

    @property
    def filter_result_dir(self) -> Path:
        """Thư mục lưu kết quả chạy Filter Methods"""
        return self.result_dir / self.data_name / "filter"

    @property
    def wrapper_result_dir(self) -> Path:
        """Thư mục lưu kết quả chạy Wrapper Methods (như con SFS của bồ)"""
        return self.result_dir / self.data_name / "wrapper"

    @property
    def ensemble_result_dir(self) -> Path:
        """Thư mục lưu kết quả chạy Ensemble Methods"""
        return self.result_dir / self.data_name / "ensemble"

    @property
    def evaluation_dir(self) -> Path:
        return self.result_dir / self.data_name / "evaluation"

    def clean_file(self, suffix: str = "") -> Path:
        name = f"{self.data_name}_preprocessed{suffix}.csv"
        return self.clean_dir / name

    def filter_file(self, method: str, suffix: str = "") -> Path:
        name = f"{self.data_name}_{method}_{self.n_features}features{suffix}.csv"
        return self.filter_dir / name

    def ensemble_file(self, file_type: str = "union", suffix: str = "") -> Path:
        if file_type == "union":
            name = f"{self.data_name}_Union_{self.n_features}features{suffix}.csv"
        elif file_type == "seeds":
            name = f"{self.data_name}_SFS_top{self.n_features}.csv"
        else:
            name = f"{self.data_name}_{file_type}_{self.n_features}{suffix}.csv"
        return self.ensemble_dir / name

    def wrapper_file(self, suffix: str = "", algorithsm_name: str = "SFS") -> Path:
        name = f"{self.data_name}_{algorithsm_name}_{suffix}.csv"
        return self.wrapper_dir / name

    @property
    def results_base_dir(self) -> Path:
        return self.base_dir / "results" / self.data_name

    @staticmethod
    def create_run_folder(base_path: Path, experiment_name: str) -> Path:
        """Create timestamped run folder: run_YYYYMMDD_HHMM_ExperimentName"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        folder_name = f"run_{timestamp}_{experiment_name}"
        run_path = base_path / folder_name
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def ensure_dirs(self) -> None:
        """Create all required directories for the pipeline."""
        for d in [
            self.clean_dir,
            self.filter_dir,
            self.ensemble_dir,
            self.wrapper_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def ensure_results_dir(self, experiment_name: str) -> Path:
        """Create timestamped results folder for an experiment."""
        return self.create_run_folder(self.results_base_dir, experiment_name)
