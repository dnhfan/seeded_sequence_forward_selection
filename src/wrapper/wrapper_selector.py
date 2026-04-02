import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import ProjectPath
from src.utils.experiment_paths import RunPaths, build_run_paths
from src.wrapper import SeededForwardSelection


class WrapperSelector:
    """
    A class to perform Wrapper-based Feature Selection (SFS).
    It first creates a Union of features from various filter methods,
    then runs Seeded Forward Selection to find the optimal subset.

    Rule A: Machine-readable outputs (CSVs) go to data/processed/<dataset>/04_wrapper/
    Rule B: Human-readable outputs (history.csv, report.txt, plots) go to results/<dataset>/run_YYYYMMDD_HHMM_ExperimentName/
    """

    def __init__(
        self,
        data_name: str,
        valid_method: List[str],
        n_features: int,
        voting_csv_name: str,
        experiment_name: str = "SFS",
        dataset_variant: str = "raw",
        pipeline_stage: str = "wrapper",
        algorithm_name: str = "SFS",
        run_tag: Optional[str] = None,
        run_folder: Optional[Path] = None,
        using_timer: bool = True,
        unit: str = "ms",
    ) -> None:

        self.data_name = data_name
        self.valid_method = valid_method
        self.n_features = n_features
        self.voting_csv_name = voting_csv_name
        self.experiment_name = experiment_name
        self.using_timer = using_timer
        self.unit = unit
        self.dataset_variant = dataset_variant
        self.pipeline_stage = pipeline_stage
        self.algorithm_name = algorithm_name
        self.run_tag = run_tag

        self.path = ProjectPath(data_name=data_name, n_features=n_features)

        self.path.wrapper_dir().mkdir(parents=True, exist_ok=True)

        if run_folder is None:
            self.run_paths = build_run_paths(
                base_results_dir=self.path.results_base_dir,
                dataset_name=self.data_name,
                pipeline_stage=self.pipeline_stage,
                dataset_variant=self.dataset_variant,
                algorithm_name=self.algorithm_name,
                run_tag=self.run_tag,
            )
        else:
            run_root = run_folder
            self.run_paths = RunPaths(
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
        self.run_paths.ensure_dirs()

    # def _create_union_features(self) -> pd.DataFrame:
    #     """
    #     [Private] Aggregates all features from the filter files into a unique Union set.
    #     Extracts the corresponding data from the raw dataset to construct a new DataFrame.
    #
    #     Returns:
    #         pd.DataFrame: A DataFrame containing the target variable and the union of all filtered features.
    #     """
    #
    #     return create_union_features(
    #         self.data_name,
    #         self.valid_method,
    #         self.n_features,
    #         self.filter_dir,
    #         self.raw_path,
    #         self.ensemble_dir,
    #     )

    def _execute_sfs_core(
        self, X_in: pd.DataFrame, y_in: pd.Series, sfs_params: dict
    ) -> tuple[pd.DataFrame, SeededForwardSelection, float]:
        """
        [Private] Execute the core SFS algorithm

        Returns:
            df_final: DataFrame which restore the results of sfs
            selector: Instance of SFS
        """
        voting_csv_path = str(self.path.ensemble_dir() / self.voting_csv_name)

        selector = SeededForwardSelection(
            seed_source=voting_csv_path,
            **sfs_params,
            using_timer=self.using_timer,
            unit=self.unit,
        )

        selector.fit(X_in, y_in)

        X_selected = selector.transform(X_in)
        X_selected_df = pd.DataFrame(
            X_selected, columns=selector.get_feature_names_out()
        )

        df_final = pd.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        return df_final, selector, selector.total_fit_time_ms_

    def _save_sfs_output(
        self,
        df_final: pd.DataFrame,
        selector: SeededForwardSelection,
        total_fit_time_ms: float,
        sfs_params: dict,
        file_suffix: str,
        n_seeds: int,
        patience: int,
        max_features: int,
    ) -> None:
        # Save machine-readable output to 04_wrapper (Rule A)
        save_path = self.path.wrapper_file(
            suffix=f"_{n_seeds}seed_{patience}p_{max_features}max_{file_suffix}"
        )
        df_final.to_csv(save_path, index=False)

        print(f" Saved Final data to: {save_path}")

        # Save human-readable logs to run_folder (Rule B)
        # Use .txt extension to get the detailed report from _generate_txt_report()
        selector.save_history(str(self.run_paths.history_txt))

        selected_features = list(selector.get_feature_names_out())
        selected_features_df = pd.DataFrame(
            {
                "feature": selected_features,
                "feature_index": list(range(len(selected_features))),
                "dataset": [self.data_name] * len(selected_features),
                "dataset_variant": [self.dataset_variant] * len(selected_features),
                "algorithm": [self.algorithm_name] * len(selected_features),
            }
        )
        selected_features_df.to_csv(self.run_paths.selected_features_csv, index=False)

        metrics = {
            "dataset": self.data_name,
            "dataset_variant": self.dataset_variant,
            "algorithm": self.algorithm_name,
            "n_features_selected": len(selector.get_feature_names_out()),
            "global_best_score": selector.global_best_score_,
            "total_fit_time_ms": total_fit_time_ms,
            "n_seeds": n_seeds,
            "patience": patience,
            "max_features": max_features,
            "cv": sfs_params.get("cv"),
            "scoring": sfs_params.get("scoring"),
            "model": sfs_params.get("model"),
            "run_root": str(self.run_paths.run_root),
        }
        with open(self.run_paths.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        pd.DataFrame([metrics]).to_csv(self.run_paths.metrics_csv, index=False)

    def run_sfs(
        self,
        df: pd.DataFrame,
        file_suffix: str = "custom",
        max_features: int = 20,
        patience: int = 3,
        n_seeds: int = 1,
        model: str = "logistic",
        scoring: str = "accuracy",
        cv: int = 5,
        verbose: int = 2,
    ) -> pd.DataFrame:
        """
        Executes Seeded Forward Selection (SFS) on the dataset.

        Args:
            max_features (int): The maximum number of features the SFS is allowed to select. Defaults to 20.
            patience (int): The number of iterations to wait for an improvement before early stopping. Defaults to 3.

        Returns:
            pd.DataFrame: The final DataFrame containing only the target and the SFS-selected features.
        """
        print("\n Starting Seeded Forward Selection (SFS)...")

        # 1. Prepare the data

        X_in: pd.DataFrame = df.iloc[:, 1:]
        y_in: pd.Series = df.iloc[:, 0]

        # 2. executes sfs
        sfs_params = {
            "max_features": max_features,
            "patience": patience,
            "n_seeds": n_seeds,
            "model": model,
            "scoring": scoring,
            "cv": cv,
            "verbose": verbose,
        }

        df_final, selector, total_fit_time_ms = self._execute_sfs_core(
            X_in, y_in, sfs_params
        )

        print(f"\n SFS completed! Final dataset shape: {df_final.shape}")
        print(" Selected Features: ", selector.get_feature_names_out())

        # 3. Save the result
        self._save_sfs_output(
            df_final,
            selector,
            total_fit_time_ms,
            sfs_params,
            file_suffix,
            n_seeds,
            patience,
            max_features,
        )

        return df_final
