import os
from pathlib import Path

import pandas as pd

from ..utils.Timer import TimerContext
from .filter_algorithms import (
    calc_anova,
    calc_chi_squared,
    calc_correlation,
    calc_mutual_info,
    calc_variance,
)


class FeatureFilter:
    """
    Filter method to select the feature, using 5 method

    - variance_threshold
    - correlation_based chi_squared
    - mutual information
    - anova_f_test
    """

    def __init__(self, method="variance", n_features=50, random_state=42) -> None:
        """
        initze the filter
        - method: 'variance', 'correlation', 'chi_squared', 'mutual_info', 'anova_f_test'.
        - n_features: number of gen want to keep.
        """
        self.method = method.lower()
        self.n_features = n_features
        self.random_state = random_state

        # State vars -> updated when using fit function
        self.selected_features_: list = []
        self.feature_scores_: dict = {}
        self.scaler_ = None
        self.metrics_: dict = {}  # Store timing metrics for each algorithm

        # raise error if we dont have that method
        valid_methods = [
            "variance",
            "correlation",
            "chi_squared",
            "mutual_information",
            "anova_f_test",
        ]
        if self.method not in valid_methods:
            raise ValueError(
                f" Not exist method '{self.method}', pls chosing exited method: {valid_methods}"
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureFilter":
        """
        Fit: running method aglorithsm to scoring and pick n_features
        """
        print(f"\n[FeatureFilter: {self.method}]")
        print(" → running filter...")

        if self.n_features > X.shape[1]:
            print(
                f"Warning: trying to take {self.n_features} but data only have {X.shape[1]}"
            )
            self.n_features = X.shape[1]

        # chose aglorithsm with self.method
        if self.method == "variance":
            self._variance_threshold(X)
        elif self.method == "correlation":
            self._correlation_based(X, y)
        elif self.method == "chi_squared":
            self._chi_squared(X, y)
        elif self.method == "mutual_information":
            self._mutual_information(X, y)
        elif self.method == "anova_f_test":
            self._anova_f_test(X, y)

        print(f"    󰄲 Successfully selected {len(self.selected_features_)} features")
        print(f"    󰓫 Top 5: {self.selected_features_[:5]}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform: save selected_feature to X
        """
        if not self.selected_features_:
            raise RuntimeError(" Error: pls run fit() before wanna transform.")

        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """fit and transform same time"""
        return self.fit(X, y).transform(X)

    # --- filter aglorithsm ---
    def _variance_threshold(self, X):
        print("󰃬 calculating variance...")
        with TimerContext(name="Variance Threshold", unit="ms") as timer:
            self.selected_features_, self.feature_scores_, _ = calc_variance(
                X, self.n_features
            )
        self.metrics_["variance"] = {
            "algorithm": "variance",
            "elapsed_ms": timer.elapsed,
            "n_features_selected": len(self.selected_features_),
        }

    def _correlation_based(self, X, y):
        print("󰃬 calculating correlation_based...")
        with TimerContext(name="Correlation Based", unit="ms") as timer:
            self.selected_features_, self.feature_scores_, _ = calc_correlation(
                X, y, self.n_features
            )
        self.metrics_["correlation"] = {
            "algorithm": "correlation",
            "elapsed_ms": timer.elapsed,
            "n_features_selected": len(self.selected_features_),
        }

    def _chi_squared(self, X, y):
        print("󰃬 calculating chi2...")
        with TimerContext(name="Chi-Squared", unit="ms") as timer:
            self.selected_features_, self.feature_scores_, _ = calc_chi_squared(
                X, y, self.n_features
            )
        self.metrics_["chi_squared"] = {
            "algorithm": "chi_squared",
            "elapsed_ms": timer.elapsed,
            "n_features_selected": len(self.selected_features_),
        }

    def _mutual_information(self, X, y):
        print("󰃬 calculating MI...")
        with TimerContext(name="Mutual Information", unit="ms") as timer:
            self.selected_features_, self.feature_scores_, _ = calc_mutual_info(
                X, y, self.n_features, random_state=self.random_state
            )
        self.metrics_["mutual_information"] = {
            "algorithm": "mutual_information",
            "elapsed_ms": timer.elapsed,
            "n_features_selected": len(self.selected_features_),
        }

    def _anova_f_test(self, X, y):
        print("󰃬 calculating anova_f_test...")
        with TimerContext(name="ANOVA F-test", unit="ms") as timer:
            self.selected_features_, self.feature_scores_, _ = calc_anova(
                X, y, self.n_features
            )
        self.metrics_["anova_f_test"] = {
            "algorithm": "anova_f_test",
            "elapsed_ms": timer.elapsed,
            "n_features_selected": len(self.selected_features_),
        }

    # --- Public Utilities ---
    def get_feature_scores(self) -> dict:
        """Get scores after fit"""
        return self.feature_scores_

    def get_selected_features(self) -> list:
        """Get selected_features"""
        return self.selected_features_

    def get_metrics(self) -> dict:
        """Get timing metrics collected during fit"""
        return self.metrics_

    def save_filtered_data(
        self, X_filtered: pd.DataFrame, y: pd.Series, save_path: str
    ):
        """Export filted feature to csv"""
        df_output = X_filtered.copy()

        target_name = y.name if hasattr(y, "name") and y.name else "V1"
        df_output.insert(0, target_name, y)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_output.to_csv(save_path, index=False)

        print(f" Saved filtered data csv in {save_path}")
        print(f"󰓫 New size of the dataset: {df_output.shape}")

    def save_metrics(self, data_name: str, output_dir: str = "results") -> Path:
        """
        Save metrics to a CSV file in results/{data_name}/filter/metrics/metrics.csv

        Args:
            data_name: Name of the dataset
            output_dir: Base output directory (default: "results")

        Returns:
            Path to the saved metrics file
        """
        metrics_dir = Path(output_dir) / data_name / "filter" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = metrics_dir / "metrics.csv"

        # Convert metrics dict to DataFrame
        metrics_list = []
        for algo_key, metrics_data in self.metrics_.items():
            metrics_list.append(metrics_data)

        if metrics_list:
            df_metrics = pd.DataFrame(metrics_list)
            df_metrics.to_csv(metrics_file, index=False)
            print(f" Saved metrics to {metrics_file}")
        else:
            print(" Warning: No metrics to save")

        return metrics_file
