import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ProjectPath
from src.modeling.eval_strategies import (
    CustomCVStrategy,
    CVStrategy,
    EvalStrategy,
    TTSStrategy,
)
from src.modeling.plot_helper import (
    compute_summary_df,
    generate_performance_chart,
    write_evaluation_report,
)
from src.utils.models import get_model

_STRATEGY_MAP: Dict[str, type[EvalStrategy]] = {
    "cv": CVStrategy,
    "tts": TTSStrategy,
    "custom_cv": CustomCVStrategy,
}


class ModelEvaluator:
    """
    A dedicated class for training and evaluating Machine Learning models
    on both baseline (raw) datasets and feature-selected datasets.

    Using in part 3: modeling

    Attributes:
        data_name (str): The name of the dataset (e.g., 'Tumors9', 'Lymphoma').
        valid_methods (List[str]): A list of feature selection methods applied.
        n_features (int): The number of top features selected. Defaults to 50.
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
        custom_base_dir: Optional[str | Path] = None,
    ) -> None:
        self.data_name = data_name
        self.valid_method = valid_method
        self.n_features = n_features
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

    def _build_models(self):
        """
        [Private] Build the standard model dict (LogReg + Decision Tree). Used by every evaluation strategy, so all the strategy stay consistent.
        """
        logreg_model = get_model("log")
        dt_model = get_model("dt")

        return {
            "LogReg": logreg_model,
            "Tree": dt_model,
        }

    def _train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method_name: str,
        n_splits: int = 5,
        eval_strategy: str = "cv",
        n_iter: int = 100,
        test_size: float = 0.3,
    ) -> None:
        """Dispatch to the requested evaluation strategy and store results.

        Args:
            X: Feature matrix.
            y: Target vector.
            method_name: Label used for reporting (e.g. method/algorithm name).
            n_splits: Number of CV folds (used by "cv" and "custom_cv").
            eval_strategy: One of "cv" (default), "tts", or "custom_cv".
            n_iter: Number of repeats (used by "tts").
            test_size: Held-out fraction (used by "tts").
        """
        strategy_cls = _STRATEGY_MAP.get(eval_strategy)
        if strategy_cls is None:
            raise ValueError(
                f"Unknown eval_strategy: {eval_strategy!r}. "
                "Expected one of: 'cv', 'tts', 'custom_cv'."
            )

        models = self._build_models()

        if eval_strategy == "tts":
            strategy = strategy_cls(n_iter=n_iter, test_size=test_size)  # type: ignore[call-arg]
        else:
            strategy = strategy_cls(n_splits=n_splits)  # type: ignore[call-arg]

        fold_rows, model_rows = strategy.run(X, y, models, method_name)
        self.fold_results.extend(fold_rows)
        self.model_results.extend(model_rows)

    def evaluate_filtered_features(
        self,
        data_dir: str,
        n_splits: int = 5,
        eval_strategy: str = "cv",
        n_iter: int = 100,
        test_size: float = 0.3,
    ) -> None:
        """
        Evaluates models on datasets that have undergone Feature Selection.

        Args:
            data_dir (str): Path to the directory containing filtered CSV files.
            n_splits (int): Number of CV folds (for "cv"/"custom_cv" strategies).
            eval_strategy (str): "cv" (default), "tts", or "custom_cv".
            n_iter (int): Number of repeats (for "tts" strategy).
            test_size (float): Held-out fraction (for "tts" strategy).
        """

        print(f"\n[*] Training models with filtered data ({self.data_name})...")
        for m in self.valid_method:
            data_path = f"{data_dir}/{self.data_name}_{m}_{self.n_features}features.csv"
            try:
                X, y = self._load_data(data_path)
                self._train_and_evaluate(
                    X,
                    y,
                    m.upper(),
                    n_splits=n_splits,
                    eval_strategy=eval_strategy,
                    n_iter=n_iter,
                    test_size=test_size,
                )
            except FileNotFoundError:
                print(f" File not found: {data_path}")

        pass

    def evaluate_baseline(
        self,
        raw_path: str,
        n_splits: int = 5,
        eval_strategy: str = "cv",
        n_iter: int = 100,
        test_size: float = 0.3,
    ) -> None:
        """
        Evaluates models on the original (unfiltered) dataset to establish a baseline.

        Args:
            raw_path (str): Path to the raw CSV file.
            n_splits (int): Number of CV folds (for "cv"/"custom_cv" strategies).
            eval_strategy (str): "cv" (default), "tts", or "custom_cv".
            n_iter (int): Number of repeats (for "tts" strategy).
            test_size (float): Held-out fraction (for "tts" strategy).
        """
        print("\n[*] Training models with baseline data (All features)...")
        try:
            X, y = self._load_data(raw_path)
            self._train_and_evaluate(
                X,
                y,
                method_name="None",
                n_splits=n_splits,
                eval_strategy=eval_strategy,
                n_iter=n_iter,
                test_size=test_size,
            )
        except FileNotFoundError:
            print(f" Raw file not found: {raw_path}")

    def generate_report_and_plot(
        self,
        experiment_prefix: str = "evaluation",
        chart_title: str = "Model Performance Comparison",
        report_metadata: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        horizontal: Optional[bool] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Compiles the evaluation results into a DataFrame, generates a comparison bar chart,
        and exports a text report.

        Args:
            experiment_prefix (str): Prefix for saved artifacts.
            chart_title (str): Title of the chart.
            report_metadata (Optional[Dict[str, Any]]): Additional metadata for the report.
            figsize (Optional[Tuple[int, int]]): Custom figure size (width, height). Auto-sized if None.
            horizontal (Optional[bool]): Use horizontal barplot (y-axis = methods).
                                         Auto-detect if None: True if methods > 6, else False.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing all evaluation metrics, or None if no data.
        """

        if not self.fold_results:
            print(" NO results. Aborting report generation.")
            return None

        result_df = pd.DataFrame(self.fold_results)
        fold_level_df = cast(
            pd.DataFrame, result_df[["Method", "Model", "Fold", "Acc"]].dropna()
        )

        if fold_level_df.empty:
            print(" No fold-level results found. Aborting report generation.")
            return None

        save_prefix = str(experiment_prefix).replace(" ", "_").lower()

        plot_name = f"{save_prefix}_{self.data_name}_{self.n_features}"
        report_name = f"{save_prefix}_{self.data_name}_{self.n_features}"

        plot_path = Path(self.plot_dir) / f"{plot_name}.png"
        report_path = Path(self.report_dir) / f"{report_name}.txt"

        generate_performance_chart(
            fold_level_df=fold_level_df,
            chart_title=chart_title,
            data_name=self.data_name,
            save_path=plot_path,
            figsize=figsize,
            horizontal=horizontal,
        )

        summary_df = compute_summary_df(fold_level_df)

        write_evaluation_report(
            report_path=report_path,
            plot_path=plot_path,
            fold_level_df=fold_level_df,
            summary_df=summary_df,
            data_name=self.data_name,
            n_features=self.n_features,
            experiment_prefix=experiment_prefix,
            save_prefix=save_prefix,
            timestamp=self.timestamp,
            report_metadata=report_metadata,
        )

        return fold_level_df

    def evaluate_custom_file(
        self,
        file_path: str,
        method_label: str,
        n_splits: int = 5,
        eval_strategy: str = "cv",
        n_iter: int = 100,
        test_size: float = 0.3,
    ) -> None:
        """
        Hàm vạn năng để đánh giá bất kỳ file CSV nào (SFS, Sklearn, PCA, v.v.)

        Args:
            file_path (str): Đường dẫn trực tiếp tới file CSV cần test.
            method_label (str): Tên hiển thị trên biểu đồ (VD: "Custom_SFS_Union", "Sklearn_SFS_Raw")
            n_splits (int): Số fold (dùng cho "cv" và "custom_cv").
            eval_strategy (str): "cv" (default), "tts", hoặc "custom_cv".
            n_iter (int): Số lần lặp (dùng cho "tts").
            test_size (float): Tỷ lệ dữ liệu test (dùng cho "tts").
        """
        print(f"\n[*] Training models with custom data ({method_label})...")
        try:
            X, y = self._load_data(file_path)
            self._train_and_evaluate(
                X,
                y,
                method_name=method_label,
                n_splits=n_splits,
                eval_strategy=eval_strategy,
                n_iter=n_iter,
                test_size=test_size,
            )
        except FileNotFoundError:
            print(f" Lỗi: Không tìm thấy file tại {file_path}")

    def plot_fit_time_comparison(
        self,
        seeded_metrics_path: str | Path,
        sklearn_metrics_path: str | Path,
        output_name: str = "time_comparison_seeded_vs_sklearn.png",
        algorithm_labels: Tuple[str, str] = ("SeededSFS", "SklearnSFS"),
        save_dir: Optional[str | Path] = None,
        show_plot: bool = False,
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
                    float(seeded_df[required_column].iloc[0]),
                    float(sklearn_df[required_column].iloc[0]),
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
