import os
from collections import Counter
from datetime import datetime
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ProjectPath


class EnsembleFeatureSelector:
    """
    A class to perform Ensemble Feature Selection by voting.
    It collects selected features from multiple filtering methods,
    counts their occurrences (votes), and extracts the most robust features.

    Attributes:
        data_name (str): The name of the dataset (e.g., 'NCI', 'Lymphoma').
        valid_methods (List[str]): Feature selection methods to ensemble.
        n_features (int): The number of features expected from each method.
        data_dir (str): The directory containing the filtered CSV files.
    """

    def __init__(
        self, data_name: str, valid_methods: List[str], n_features: int, data_dir: str
    ) -> None:
        self.data_name = data_name
        self.valid_methods = valid_methods
        self.n_features = n_features
        self.data_dir = data_dir
        self.path = ProjectPath(data_name, n_features)

        self.df_counts: Optional[pd.DataFrame] = None

        # setup paths
        self.report_dir: str = str(self.path.ensemble_result_dir / "reports")
        self.plot_dir: str = str(self.path.ensemble_result_dir / "plots")
        self.csv_dir: str = str(self.path.ensemble_dir)

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

    def run_voting(self) -> pd.DataFrame:
        """
        Reads the top features from all methods, counts their frequencies,
        and sorts them by the number of votes.

        Returns:
            pd.DataFrame: A DataFrame containing 'Feature' and 'Votes', sorted descending.
        """
        all_selected_features: List[str] = []

        print("󰈞 Collecting filtered features from methods...\n")

        for m in self.valid_methods:
            data_path = (
                f"{self.data_dir}/{self.data_name}_{m}_{self.n_features}features.csv"
            )

            try:
                # nrow = 0, only take the cols for counting
                df_cols = pd.read_csv(data_path, nrows=0)

                features = df_cols.columns[1:].to_list()
                all_selected_features.extend(features)

            except FileNotFoundError:
                print(f" Error: not found {data_path}")
        print("󰄬 Done collecting features.")

        # Counts vote
        feature_count = Counter(all_selected_features)
        df_counts = pd.DataFrame.from_dict(
            feature_count, orient="index", columns=["Votes"]
        ).reset_index()

        df_counts.rename(columns={"index": "Feature"}, inplace=True)
        df_counts = df_counts.sort_values(by="Votes", ascending=False).reset_index(
            drop=True
        )

        self.df_counts = df_counts
        return self.df_counts

    def generate_report_and_plot(self, top_n_plot: int = 10):
        """
        Generates a bar chart for the top N voted features and saves the report
        (Text and CSV formats).

        Args:
            top_n_plot (int): The number of top features to display and save. Defaults to 10.

        Returns:
            Optional[pd.DataFrame]: The top N features DataFrame, or None if empty.
        """
        if self.df_counts is None or self.df_counts.empty:
            print(" No voting data available. Run `run_voting()` first.")
            return None

        # Take top N
        df_top_n = self.df_counts.head(top_n_plot)

        # File path
        plot_path = f"{self.plot_dir}/top{self.n_features}_features_voting.png"
        report_path = f"{self.report_dir}/top{self.n_features}_features_voting.txt"
        csv_path = f"{self.csv_dir}/top{self.n_features}_features_voting.csv"

        # --- 1. Plotting ---
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=df_top_n,
            x="Votes",
            y="Feature",
            hue="Feature",
            palette="magma",
            legend=False,
        )

        plt.title(f"Top {top_n_plot} Features, ({self.data_name})")
        plt.xticks(range(1, len(self.valid_methods) + 1))

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"󰋩 Plot saved to: {plot_path}")

        # --- 2. Text Report ---
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Ensemble Filter Selection Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total methods used: {len(self.valid_methods)}\n")
            f.write("-" * 60 + "\n")
            f.write(df_top_n.to_string(index=False))
        print(f"󰎞 Report saved to: {report_path}")

        # --- 3. CSV data ---
        df_top_n.to_csv(csv_path, index=False)
        print(f"󰈙 CSV saved to: {csv_path}")

        return df_top_n
