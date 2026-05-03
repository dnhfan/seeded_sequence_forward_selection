import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.container import BarContainer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
        valid_method: List[str] = [
            "variance",
            "correlation",
            "chi_squared",
            "mutual_information",
            "anova_f_test",
        ],
        n_features: int = 50,
        max_iter: int = 4000,
        use_scaler: bool = True,
        custom_base_dir: Optional[str | Path] = None,
    ) -> None:
        self.data_name = data_name
        self.valid_method = valid_method
        self.n_features = n_features
        self.max_iter = max_iter
        self.use_scaler = use_scaler
        self.path = ProjectPath(data_name, n_features)

        self.fold_results: List[Dict[str, Any]] = []
        self.model_results: List[Dict[str, Any]] = []

        self.timestamp: str = datetime.now().strftime("%Y-%m-%d")

        # path
        if custom_base_dir:
            base_dir = Path(custom_base_dir)
        else:
            base_dir = self.path.filter_result_dir

        self.report_dir: str = str(base_dir / "reports")
        self.plot_dir: str = str(base_dir / "plots")

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
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method_name: str,
        n_splits: int = 5,
    ) -> None:
        """
        [Private] Splits the data using Cross-Validation, trains Logistic Regression and Decision Tree models,
        evaluates their accuracy, and stores the results.
        """
        # 1. Init the CV
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # scoring
        scoring = ["accuracy"]

        # 2. Init model in a dict -> easy to add new one
        logreg_model = LogisticRegression(max_iter=self.max_iter, random_state=42)
        if self.use_scaler:
            logreg_model = make_pipeline(StandardScaler(), logreg_model)

        models = {
            "LogReg": logreg_model,
            "Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        }

        # 3. Runing each model
        for model_name, model in models.items():
            # cross_validate will auto fit and predict
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            test_accuracy = scores["test_accuracy"]
            mean_acc = test_accuracy.mean()

            # 4. Store the result of each fold
            for i in range(cv.get_n_splits()):
                self.fold_results.append(
                    {
                        "Method": method_name,
                        "Model": model_name,
                        "Fold": i + 1,
                        "Acc": scores["test_accuracy"][i],
                    }
                )
            self.model_results.append(
                {
                    "Method": method_name,
                    "Model": model_name,
                    "mean_acc": mean_acc,
                    "std": test_accuracy.std(),
                    "min": test_accuracy.min(),
                    "max": test_accuracy.max(),
                    "n_folds": len(test_accuracy),
                }
            )

            print(f"󰄭  [{method_name:<12}] {model_name:<8} | Acc: {mean_acc:.4f} ")

    def evaluate_filtered_features(self, data_dir: str, n_splits: int = 5) -> None:
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
                self._train_and_evaluate(X, y, m.upper(), n_splits=n_splits)
            except FileNotFoundError:
                print(f" File not found: {data_path}")

        pass

    def evaluate_baseline(self, raw_path: str, n_splits=3) -> None:
        """
        Evaluates models on the original (unfiltered) dataset to establish a baseline.

        Args:
            raw_path (str): Path to the raw CSV file.
        """
        print("\n[2] Training models with baseline data (All features)...")
        try:
            X, y = self._load_data(raw_path)
            self._train_and_evaluate(X, y, method_name="None", n_splits=n_splits)
        except FileNotFoundError:
            print(f" Raw file not found: {raw_path}")

    def generate_report_and_plot(
        self,
        experiment_prefix: str = "evaluation",
        chart_title: str = "Model Performance Comparison",
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Compiles the evaluation results into a DataFrame, generates a comparison bar chart,
        and exports a text report.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing all evaluation metrics, or None if no data.
        """

        if not self.fold_results:
            print(" NO results. Aborting report generation.")
            return None

        result_df = pd.DataFrame(self.fold_results)
        fold_level_df = cast(
            pd.DataFrame, result_df[["Method", "Model", "Fold", "Acc"]].dropna()
        )

        if fold_level_df.empty:
            print(" No fold-level results found. Aborting report generation.")
            return None

        save_prefix = str(experiment_prefix).replace(" ", "_").lower()

        plot_name = f"{save_prefix}_{self.data_name}"
        report_name = f"{save_prefix}_{self.data_name}"

        plot_path = Path(self.plot_dir) / f"{plot_name}.png"
        report_path = Path(self.report_dir) / f"{report_name}.txt"

        # --- Generates Chart ---
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(
            data=fold_level_df, x="Method", y="Acc", hue="Model", palette="pastel"
        )

        acc_series = cast(pd.Series, fold_level_df["Acc"])
        min_acc = float(acc_series.min())
        max_acc = float(acc_series.max())
        y_min = max(0.0, min_acc - 0.05)
        y_max = min(1.05, max_acc + 0.08)

        for container in ax.containers:
            if isinstance(container, BarContainer):
                ax.bar_label(
                    container, fmt="%.4f", padding=3, fontsize=9, color="black"
                )

        plt.title(f"{chart_title} ({self.data_name})")
        ax.set_ylim(y_min, y_max)
        plt.ylabel("Accuracy Score")
        plt.xlabel("Feature Selection Method")
        plt.xticks(rotation=45, ha="right", fontweight="bold")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f" Chart saved at: {plot_path}")

        summary_df = fold_level_df.groupby(["Method", "Model"], as_index=False).agg(
            mean_accuracy=("Acc", "mean"),
            std_accuracy=("Acc", "std"),
            min_accuracy=("Acc", "min"),
            max_accuracy=("Acc", "max"),
            median_accuracy=("Acc", "median"),
            n_folds=("Acc", "count"),
        )
        summary_df = cast(pd.DataFrame, summary_df)
        summary_df = summary_df.sort_values(by="std_accuracy", ascending=True)
        summary_df = summary_df.sort_values(
            by="mean_accuracy", ascending=False
        ).reset_index(drop=True)

        summary_df["std_accuracy"] = summary_df["std_accuracy"].fillna(0.0)
        summary_df["cv_stability"] = 1.0 - summary_df["std_accuracy"]
        summary_df["rank"] = (
            summary_df["mean_accuracy"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        best_row = summary_df.iloc[0]

        # --- Generates Report ---
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"experiment_id: {save_prefix}_{self.data_name}_{self.timestamp.replace('-', '')}\n"
            )
            f.write(f"dataset: {self.data_name}\n")
            f.write(f"experiment_prefix: {experiment_prefix}\n")
            f.write(f"feature_count: {self.n_features}\n")
            f.write(f"logreg_max_iter: {self.max_iter}\n")
            f.write(f"python_version: {platform.python_version()}\n")
            f.write(f"pandas_version: {pd.__version__}\n")
            f.write(f"sklearn_version: {sklearn.__version__}\n")
            if report_metadata:
                for key, value in report_metadata.items():
                    f.write(f"{key}: {value}\n")

            f.write("\nARTIFACTS\n")
            f.write(f"plot_path: {plot_path}\n")
            f.write(f"report_path: {report_path}\n")

            f.write("\nEXECUTIVE SUMMARY\n")
            f.write(
                f"best_configuration: method={best_row['Method']}, model={best_row['Model']}, "
                f"mean_accuracy={best_row['mean_accuracy']:.4f}, std={best_row['std_accuracy']:.4f}, "
                f"folds={int(best_row['n_folds'])}\n"
            )
            f.write(
                f"total_configurations: {len(summary_df)} | total_fold_evaluations: {len(fold_level_df)}\n"
            )

            f.write("\nCROSS-VALIDATION SUMMARY (ranked)\n")
            f.write("-" * 120 + "\n")
            summary_report_df = summary_df[
                [
                    "rank",
                    "Method",
                    "Model",
                    "mean_accuracy",
                    "std_accuracy",
                    "median_accuracy",
                    "min_accuracy",
                    "max_accuracy",
                    "n_folds",
                    "cv_stability",
                ]
            ].round(4)
            f.write(summary_report_df.to_string(index=False))
            f.write("\n")

            f.write("\nFOLD-LEVEL RESULTS (for auditability)\n")
            f.write("-" * 120 + "\n")
            fold_report_df = fold_level_df.sort_values(
                by=["Method", "Model", "Fold"]
            ).round(4)

            for method_name, group_df in fold_report_df.groupby("Method", sort=False):
                f.write(f" Method: {method_name} \n")

                clean_df = group_df.drop(columns=["Method"])
                f.write(clean_df.to_string(index=False))

                f.write("\n\n")

            f.write("\n" + "-" * 120 + "\n")
        print(f"󰎞 Report saved at: {report_path}")

        return result_df

    def evaluate_custom_file(
        self, file_path: str, method_label: str, n_splits: int = 5
    ) -> None:
        """
        Hàm vạn năng để đánh giá bất kỳ file CSV nào (SFS, Sklearn, PCA, v.v.)

        Args:
            file_path (str): Đường dẫn trực tiếp tới file CSV cần test.
            method_label (str): Tên bồ muốn hiển thị trên biểu đồ (VD: "Custom_SFS_Union", "Sklearn_SFS_Raw")
        """
        print(f"\n[*] Training models with custom data ({method_label})...")
        try:
            X, y = self._load_data(file_path)
            self._train_and_evaluate(X, y, method_name=method_label, n_splits=n_splits)
        except FileNotFoundError:
            print(f" Lỗi: Không tìm thấy file tại {file_path}")

    def plot_fit_time_comparison(
        self,
        seeded_metrics_path: str | Path,
        sklearn_metrics_path: str | Path,
        output_name: str = "time_comparison_seeded_vs_sklearn.png",
        algorithm_labels: Tuple[str, str] = ("SeededSFS", "SklearnSFS"),
        save_dir: Optional[str | Path] = None,
        show_plot: bool = True,
    ) -> pd.DataFrame:
        """
        Compare fit time between SeededSFS and SklearnSFS using metrics CSV files.

        Args:
            seeded_metrics_path (str | Path): Path to seeded selector metrics.csv file.
            sklearn_metrics_path (str | Path): Path to sklearn selector metrics.csv file.
            output_name (str): Output filename for the plot image.
            algorithm_labels (Tuple[str, str]): Labels shown on x-axis for seeded and sklearn.
            save_dir (Optional[str | Path]): Optional custom plot directory.
            show_plot (bool): Whether to display the chart after saving.

        Returns:
            pd.DataFrame: Comparison table with total fit time in milliseconds and seconds.
        """
        seeded_path = Path(seeded_metrics_path)
        sklearn_path = Path(sklearn_metrics_path)

        seeded_df = pd.read_csv(seeded_path)
        sklearn_df = pd.read_csv(sklearn_path)

        required_column = "total_fit_time_ms"
        if required_column not in seeded_df.columns:
            raise KeyError(f"Missing column '{required_column}' in: {seeded_path}")
        if required_column not in sklearn_df.columns:
            raise KeyError(f"Missing column '{required_column}' in: {sklearn_path}")

        comparison_df = pd.DataFrame(
            {
                "algorithm": [algorithm_labels[0], algorithm_labels[1]],
                "total_fit_time_ms": [
                    float(seeded_df.loc[0, required_column]),
                    float(sklearn_df.loc[0, required_column]),
                ],
            }
        )
        comparison_df["total_fit_time_sec"] = comparison_df["total_fit_time_ms"] / 1000

        target_plot_dir = (
            Path(save_dir) if save_dir else self.path.evaluation_dir / "plots"
        )
        target_plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_plot_dir / output_name

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(
            comparison_df["algorithm"],
            comparison_df["total_fit_time_sec"],
        )

        ax.set_title(f"{self.data_name} Dataset: Fit Time Comparison", fontsize=14)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Total Fit Time (seconds)")

        for bar, value in zip(bars, comparison_df["total_fit_time_sec"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        print(f" Chart saved at: {output_path}")

        return comparison_df

    def clear_results(self) -> None:
        """
        Dọn sạch danh sách kết quả cũ trong bộ nhớ (RAM).
        Dùng khi muốn dùng lại class này cho một thí nghiệm hoàn toàn mới.
        """
        self.fold_results = []
        self.model_results = []
        # Cập nhật lại timestamp để tên file mới không đè lên file cũ
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("\n Cleaned result. Ready for new experiment!")
