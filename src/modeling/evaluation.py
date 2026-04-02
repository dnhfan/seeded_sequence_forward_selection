import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import ProjectPath


class ModelEvaluator:
    """
    A dedicated class for training and evaluating Machine Learning models
    on both baseline (raw) datasets and feature-selected datasets.

    Using in part 3: modeling

    Attributes:
        data_name (str): The name of the dataset (e.g., 'Tumors9', 'Lymphoma').
        valid_methods (List[str]): A list of feature selection methods applied.
        n_features (int): The number of top features selected. Defaults to 50.
        max_iter (int): Maximum number of iterations for Logistic Regression. Defaults to 4000.
    """

    def __init__(
        self,
        data_name: str,
        valid_method: List[str],
        n_features: int = 50,
        max_iter: int = 4000,
    ) -> None:
        self.data_name = data_name
        self.valid_method = valid_method
        self.n_features = n_features
        self.max_iter = max_iter
        self.path = ProjectPath(data_name, n_features)

        self.results: List[Dict[str, Any]] = []
        self.timestamp: str = datetime.now().strftime("%Y-%m-%d")

        # path
        self.report_dir: str = str(self.path.filter_result_dir() / "reports")
        self.plot_dir: str = str(self.path.filter_result_dir() / "plots")

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _load_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        [Private] Reads a CSV file and splits it into Features (X) and Target (y).
        Assumes the first column (index 0) is the Target variable.
        """

        df = pd.read_csv(file_path)
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        return X, y

    def _train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series, method_name: str
    ) -> None:
        """
        [Private] Splits the data, trains Logistic Regression and Decision Tree models,
        evaluates their accuracy, and stores the results.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # 1. Train Logistic Regression
        log_model = LogisticRegression(max_iter=self.max_iter, random_state=42)
        log_model.fit(X_train, y_train)
        log_acc: float = accuracy_score(y_test, log_model.predict(X_test))

        # 2. Train Decision Tree
        tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        tree_model.fit(X_train, y_train)
        tree_acc: float = accuracy_score(y_test, tree_model.predict(X_test))

        # 3. Log the results
        self.results.append(
            {"Method": method_name, "Model": "LogisticRegression", "Acc": log_acc}
        )
        self.results.append(
            {"Method": method_name, "Model": "DecisionTreeClassifier", "Acc": tree_acc}
        )

        print(f"󰄭  {method_name:<12}: LogReg = {log_acc:.4f} | Tree = {tree_acc:.4f}")

    def evaluate_filtered_features(self, data_dir: str) -> None:
        """
        Evaluates models on datasets that have undergone Feature Selection.

        Args:
            data_dir (str): Path to the directory containing filtered CSV files.
        """

        print(f"\n[1] Training models with filtered data ({self.data_name})...")
        for m in self.valid_method:
            data_path = f"{data_dir}/{self.data_name}_{m}_{self.n_features}features.csv"
            try:
                X, y = self._load_data(data_path)
                self._train_and_evaluate(X, y, m.upper())
            except FileNotFoundError:
                print(f" File not found: {data_path}")

        pass

    def evaluate_baseline(self, raw_path: str) -> None:
        """
        Evaluates models on the original (unfiltered) dataset to establish a baseline.

        Args:
            raw_path (str): Path to the raw CSV file.
        """
        print("\n[2] Training models with baseline data (All features)...")
        try:
            X, y = self._load_data(raw_path)
            self._train_and_evaluate(X, y, method_name="None")
        except FileNotFoundError:
            print(f" Raw file not found: {raw_path}")

    def generate_report_and_plot(self) -> Optional[pd.DataFrame]:
        """
        Compiles the evaluation results into a DataFrame, generates a comparison bar chart,
        and exports a text report.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing all evaluation metrics, or None if no data.
        """

        if not self.results:
            print(" NO results. Aborting report generation.")
            return None
        result_df = pd.DataFrame(self.results)

        # --- Generates Chart ---
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 7))
        sns.barplot(data=result_df, x="Method", y="Acc", hue="Model", palette="pastel")

        plt.title(
            f"Comparison: Top {self.n_features} Features vs All Features, ({self.data_name})"
        )
        plt.ylim(0.1, 1.05)
        plt.ylabel("Accuracy")
        plt.xlabel("Feature Selection Method")
        plt.xticks(rotation=45, ha="right", fontweight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plot_path = f"{self.plot_dir}/model_comparison_top{self.n_features}_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Chart saved at: {plot_path}")

        # --- Generates Report ---
        report_path = f"{self.report_dir}/model_evaluation_top{self.n_features}_{self.timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Model Comparison Report - Dataset: {self.data_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Features: {self.n_features}\n")
            f.write("-" * 60 + "\n")
            f.write(result_df.to_string(index=False))

        print(f"󰎞 Report saved at: {report_path}")

        return result_df
