import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas

from src.config import ProjectPath
from src.utils.experiment_paths import RunPaths, build_run_paths
from src.wrapper.sfs_result import SFSResult


class BaseWrapperSelector(ABC):
    """
    Base class for wrapper-based feature selection
    Subclasses must implement `_execute_core`
    """

    def __init__(
        self,
        data_name: str,
        n_features: int,
        voting_csv_name: str,
        dataset_variant: str = "raw",
        pipeline_stage: str = "wrapper",
        run_tag: Optional[str] = None,
        run_folder: Optional[Path] = None,
        using_timer: bool = True,
        unit: str = "ms",
    ) -> None:

        self.data_name = data_name
        self.n_features = n_features
        self.voting_csv_name = voting_csv_name
        self.using_timer = using_timer
        self.unit = unit
        self.dataset_variant = dataset_variant
        self.pipeline_stage = pipeline_stage
        self.algorithm_name = self.__class__.__name__.lower()
        self.run_tag = run_tag

        self.path = ProjectPath(self.data_name, self.n_features)

        if run_folder is None:
            self.run_path = build_run_paths(
                base_results_dir=self.path.results_base_dir,
                dataset_name=self.data_name,
                pipeline_stage=self.pipeline_stage,
                dataset_variant=self.dataset_variant,
                algorithm_name=self.algorithm_name,
                run_tag=self.run_tag,
            )
        else:
            run_root = run_folder
            self.run_path = RunPaths(
                run_root=run_root,
                history_dir=run_root / "history",
                features_dir=run_root / "features",
                metrics_dir=run_root / "metrics",
                artifacts_dir=run_root / "artifacts",
                history_json=run_root / "history" / "history.json",
                history_txt=run_root / "history" / "history.txt",
                selected_features_csv=run_root / "features" / "selected_features.csv",
                metrics_json=run_root / "metrics" / "metrics.json",
                metrics_csv=run_root / "metrics" / "metrics.csv",
            )
        self.run_path.ensure_dirs()

    @abstractmethod
    def _execute_core(
        self,
        X_in: pandas.DataFrame,
        y_in: pandas.Series,
        sfs_params: dict,
        direction: str = "forward",
    ) -> SFSResult:
        raise NotImplementedError(" Subclasses must implement _execute_core")

    def _save_sfs_output(
        self,
        result: SFSResult,
        sfs_params: dict,
        file_suffix: str,
        max_features: int | str,
        cv: int,
        **kwargs,
    ) -> None:
        suffix_parts = []

        # 1. Cấu hình lõi: Model đánh giá
        if "model" in sfs_params:
            model_name = str(sfs_params["model"]).lower()
            suffix_parts.append(model_name)

        # 2. Cấu hình lõi: Tiêu chí tối ưu (Scoring)
        if "scoring" in sfs_params:
            scoring_name = str(sfs_params["scoring"]).lower()
            suffix_parts.append(scoring_name)

        # 3. Ràng buộc chính: Số lượng feature tối đa
        suffix_parts.append(f"{max_features}max")

        # 4. Tham số đặc thù của Seeded SFS
        if "n_seeds" in sfs_params:
            suffix_parts.append(f"{sfs_params['n_seeds']}seeds")
        if "patience" in sfs_params:
            suffix_parts.append(f"{sfs_params['patience']}pat")

        # 5. Chiến lược Validation
        suffix_parts.append(f"{cv}cv")

        # 6. Phiên bản dữ liệu đầu vào
        if file_suffix:
            suffix_parts.append(file_suffix)

        # Nối lại
        suffix_str = "_".join(suffix_parts)
        save_path = self.path.wrapper_file(suffix_str, self.algorithm_name)

        result.df_final.to_csv(save_path, index=False)
        print(f"\n Saved final data to: {save_path}")  # csv file handled

        # History handled
        if result.history_text:
            with open(self.run_path.history_txt, "w", encoding="utf-8") as f:
                f.write(result.history_text)

        # Selected features handled
        selected_features_df = pandas.DataFrame(
            {
                "feature": result.selected_features,
                "dataset": self.data_name,
                "dataset_variant": self.dataset_variant,
                "algorithm": self.algorithm_name,
            }
        )

        selected_features_df.to_csv(self.run_path.selected_features_csv, index=False)

        # Metrics handled
        metrics = {
            "dataset": self.data_name,
            "dataset_variant": self.dataset_variant,
            "n_features_selected": len(result.selected_features),
            "global_best_score": result.global_best_score,
            "total_fit_time_ms": result.total_fit_time_ms,
            "run_root": str(self.run_path.run_root),
            **kwargs,
        }

        with open(self.run_path.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        pandas.DataFrame([metrics]).to_csv(self.run_path.metrics_csv, index=False)

    def run_sfs(
        self,
        df: pandas.DataFrame,
        max_features: int | str = 20,
        cv: int = 5,
        verbose: int = 2,
        model: str = "logistic",
        scoring: str = "accuracy",
        direction: str = "forward",
        **kwargs,  # Key word arguments -> arguments with key-value
    ) -> pandas.DataFrame:
        print(f" Starting {self.algorithm_name}")

        X_in: pandas.DataFrame = df.iloc[:, 1:]
        y_in: pandas.Series = df.iloc[:, 0]

        sfs_params = {
            "max_features": max_features,
            "cv": cv,
            "model": model,
            "scoring": scoring,
            "verbose": verbose,
            **kwargs,
        }

        # 1. run the algorithm
        result = self._execute_core(X_in, y_in, sfs_params, direction=direction)

        print(
            f"\n {self.algorithm_name} completed! Final dataset shape: {result.df_final.shape}"
        )
        print("Selected features:", result.selected_features)

        # 2. save the result
        self._save_sfs_output(
            result=result,
            sfs_params=sfs_params,
            file_suffix=self.dataset_variant,
            max_features=max_features,
            cv=cv,
            **kwargs,
        )

        return result.df_final
