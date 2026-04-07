import json
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

from src.config import ProjectPath
from src.utils.experiment_paths import RunPaths, build_run_paths
from src.wrapper import SeededForwardSelection

from .models import get_model


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
        self.using_timer = using_timer
        self.unit = unit
        self.dataset_variant = dataset_variant
        self.pipeline_stage = pipeline_stage
        self.algorithm_name = algorithm_name
        self.run_tag = run_tag

        self.path = ProjectPath(data_name=data_name, n_features=n_features)

        self.path.wrapper_dir.mkdir(parents=True, exist_ok=True)

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

    def _execute_sfs_core(
        self, X_in: pd.DataFrame, y_in: pd.Series, sfs_params: dict
    ) -> tuple[pd.DataFrame, SeededForwardSelection, float, List[str]]:
        """
        [Private] Execute the core SFS algorithm

        Returns:
            df_final: DataFrame which restore the results of sfs
            selector: Instance of SFS
        """
        voting_csv_path = str(self.path.ensemble_dir / self.voting_csv_name)

        sfs_kwargs = sfs_params.copy()
        sfs_kwargs.pop("engine", None)

        selector = SeededForwardSelection(
            seed_source=voting_csv_path,
            **sfs_kwargs,
            using_timer=self.using_timer,
            unit=self.unit,
        )

        selector.fit(X_in, y_in)

        X_selected = selector.transform(X_in)
        selected_features = list(selector.get_feature_names_out())
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

        df_final = pd.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        return df_final, selector, selector.total_fit_time_ms_, selected_features

    def _execute_sklearn_sfs_core(
        self,
        X_in: pd.DataFrame,
        y_in: pd.Series,
        sfs_params: dict,
        estimator,
        direction: str,
    ) -> tuple[pd.DataFrame, SequentialFeatureSelector, float, List[str]]:
        """
        [Private] Execute sklearn SequentialFeatureSelector

        Returns:
            df_final: DataFrame which restore the results of sfs
            selector: Instance of SequentialFeatureSelector
        """

        max_features = sfs_params.get("max_features") or 1

        if max_features == "auto":
            n_features_to_select = "auto"
        else:
            n_features_to_select = int(max_features)

        cv_value = int(sfs_params.get("cv") or 5)

        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features_to_select,  # type: ignore[arg-type]
            tol=0.001,
            direction=direction,
            scoring=sfs_params.get("scoring"),
            cv=cv_value,
            n_jobs=-1,
        )

        start_time = time.perf_counter()
        selector.fit(X_in, y_in)
        total_fit_time_ms = (time.perf_counter() - start_time) * 1000

        X_selected = selector.transform(X_in)
        selected_features = list(X_in.columns[selector.get_support()])
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

        df_final = pd.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        return df_final, selector, total_fit_time_ms, selected_features

    def _write_sklearn_history(
        self,
        selected_features: List[str],
        total_fit_time_ms: float,
        sfs_params: dict,
        direction: str,
    ) -> None:
        lines = [
            "Sklearn SequentialFeatureSelector report",
            f"direction: {direction}",
            f"max_features: {sfs_params.get('max_features')}",
            f"cv: {sfs_params.get('cv')}",
            f"scoring: {sfs_params.get('scoring')}",
            f"total_fit_time_ms: {total_fit_time_ms:.2f}",
            f"n_features_selected: {len(selected_features)}",
            f"selected_features: {selected_features}",
        ]
        with open(self.run_paths.history_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_sfs_output(
        self,
        df_final: pd.DataFrame,
        selector,
        total_fit_time_ms: float,
        sfs_params: dict,
        file_suffix: str,
        n_seeds: int,
        patience: int,
        max_features: int | str,
        selected_features: List[str],
        global_best_score: Optional[float],
        direction: str,
        cv: int,
    ) -> None:
        # Save machine-readable output to 04_wrapper (Rule A)
        save_path = self.path.wrapper_file(
            suffix=f"_{n_seeds}seed_{patience}p_{max_features}max_{cv}cv_{file_suffix}",
            algorithsm_name=self.algorithm_name,
        )
        df_final.to_csv(save_path, index=False)

        print(f" Saved Final data to: {save_path}")

        # Save human-readable logs to run_folder (Rule B)
        # Use .txt extension to get the detailed report from _generate_txt_report()
        if hasattr(selector, "save_history"):
            selector.save_history(str(self.run_paths.history_txt))
        else:
            self._write_sklearn_history(
                selected_features,
                total_fit_time_ms,
                sfs_params,
                direction,
            )

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
            "global_best_score": global_best_score,
            "total_fit_time_ms": total_fit_time_ms,
            "n_seeds": n_seeds,
            "patience": patience,
            "max_features": max_features,
            "cv": sfs_params.get("cv"),
            "scoring": sfs_params.get("scoring"),
            "model": sfs_params.get("model"),
            "engine": sfs_params.get("engine"),
            "direction": direction,
            "run_root": str(self.run_paths.run_root),
        }
        with open(self.run_paths.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        pd.DataFrame([metrics]).to_csv(self.run_paths.metrics_csv, index=False)

    def run_sfs(
        self,
        df: pd.DataFrame,
        max_features: int | str = 20,
        patience: int = 3,
        n_seeds: int = 1,
        model: str = "logistic",
        scoring: str = "accuracy",
        cv: int = 5,
        verbose: int = 2,
        engine: str = "custom",
        sklearn_estimator=None,
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
            "engine": engine,
        }
        direction = "forward"

        if engine == "sklearn":
            estimator = sklearn_estimator or get_model(model)
            sfs_params["model"] = estimator.__class__.__name__
            df_final, selector, total_fit_time_ms, selected_features = (
                self._execute_sklearn_sfs_core(
                    X_in,
                    y_in,
                    sfs_params,
                    estimator,
                    direction,
                )
            )
            X_selected = df_final.iloc[:, 1:]
            scores = cross_val_score(
                estimator,
                X_selected,
                y_in,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            global_best_score = float(scores.mean())
        else:
            df_final, selector, total_fit_time_ms, selected_features = (
                self._execute_sfs_core(X_in, y_in, sfs_params)
            )
            global_best_score = selector.global_best_score_

        print(f"\n SFS completed! Final dataset shape: {df_final.shape}")
        print(" Selected Features: ", selected_features)

        # 3. Save the result
        self._save_sfs_output(
            df_final,
            selector,
            total_fit_time_ms,
            sfs_params,
            self.dataset_variant,
            n_seeds,
            patience,
            max_features,
            selected_features,
            global_best_score,
            direction,
            cv,
        )

        return df_final
