import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ProjectPath
from src.modeling.evaluation import ModelEvaluator


def evaluate_benchmark(data_name: str, variant: str = "raw"):
    """
    Benchmark evaluation: read the metrics from the benchmark runs and aggregate them for comparison and visualization.
    """
    print(f" Evaluating benchmark for dataset: {data_name} (Variant: {variant})")

    path = ProjectPath(data_name=data_name)
    evaluator = ModelEvaluator(data_name=data_name, custom_base_dir=path.evaluation_dir)

    print("\nPHASE 1: EVALUATING BASELINE (All Features)")
    evaluator.evaluate_baseline(str(path.raw_path), n_splits=4)

    models = ["log", "dt", "rf", "svm"]
    df_metrics_list = []

    wrapper_dir = path.wrapper_result_dir / variant / "seededsfsselector"
    if not wrapper_dir.exists():
        print(
            f" No wrapper results found for dataset: {data_name} (Variant: {variant})"
        )
        return

    print("PHASE 2: COLLECTING SFS MODELS DATA & METRICS")

    for model in models:
        # Tìm thư mục run mới nhất của từng model
        pattern = f"*benchmark_{model}_1seeds*"
        run_folders = sorted(list(wrapper_dir.glob(pattern)))
        print(
            f"{run_folders[-1].name if run_folders else 'No runs found'} - {model.upper()}"
        )  # Debug: In ra tên thư mục run mới nhất hoặc thông báo nếu không tìm thấy

        if not run_folders:
            print(
                f" No run folders found for model: {model.upper()} in dataset: {data_name}"
            )
            continue

        latest_run = run_folders[-1]

        # --- 1. Đọc file thời gian (Fit time) ---
        metrics_file = latest_run / "metrics" / "metrics.csv"
        if metrics_file.exists():
            df_metrics = pandas.read_csv(metrics_file)
            model_name = (
                df_metrics["model_sfs"].iloc[0]
                if "model_sfs" in df_metrics.columns
                else model.upper()
            )
            df_metrics["model_name"] = model_name
            df_metrics_list.append(df_metrics)
        else:
            print(f" Error: Metrics file not found for {model.upper()}")

        # --- 2. Nạp dữ liệu đặc trưng đã qua SFS để chấm điểm chéo độc lập ---
        data_pattern = f"*{model.lower()}*_1seeds*{variant}.csv"
        data_files = sorted(list((path.wrapper_dir / variant / "seededsfsselector").glob(data_pattern)))

        if data_files:
            latest_data_file = data_files[-1]
            evaluator.evaluate_custom_file(
                file_path=str(latest_data_file),
                method_label=f"SFS_{model.upper()}",
                n_splits=4,
            )
            print(f"✓ Successfully loaded features dataset for {model.upper()}")
        else:
            print(f" No data files found for model: {model.upper()}")

    # --- ĐƯA RA NGOÀI VÒNG LẶP FOR: Tạo báo cáo tổng hợp từ tất cả các mô hình ---
    print("\nPHASE 3: GENERATING COMPREHENSIVE ACCURACY REPORT")
    if evaluator.fold_results:
        evaluator.generate_report_and_plot(
            experiment_prefix=f"benchmark_accuracy_{variant}",
            chart_title=f"Benchmark Models Accuracy ({variant.capitalize()}) - {data_name}",
        )
    else:
        print(" No evaluation results found to generate accuracy report.")

    # --- ĐƯA RA NGOÀI VÒNG LẶP FOR: Vẽ biểu đồ so sánh Fit Time & Số lượng đặc trưng ---
    if df_metrics_list:
        print("\nPHASE 4: VISUALIZING FIT TIME & FEATURE COUNT TRADE-OFF")
        df_metrics = pandas.concat(df_metrics_list, ignore_index=True)
        df_metrics["Time (s)"] = df_metrics["total_fit_time_ms"] / 1000

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.set_theme(style="whitegrid")

        # Biểu đồ 1: Fit Time
        sns.barplot(
            data=df_metrics, x="model_name", y="Time (s)", ax=axes[0], palette="pastel", hue="model_name", legend=False
        )
        axes[0].set_title(
            f"SFS Fit Time Comparison - {data_name}", fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("Time (seconds)", fontsize=11)
        axes[0].set_xlabel("SFS Internal Model", fontsize=11)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt="%.1fs", padding=3, fontsize=10)

        # Biểu đồ 2: Selected Features
        sns.barplot(
            data=df_metrics,
            x="model_name",
            y="n_features_selected",
            ax=axes[1],
            palette="pastel",
            hue="model_name",
            legend=False
        )
        axes[1].set_title(
            f"Number of Selected Features - {data_name}", fontsize=12, fontweight="bold"
        )
        axes[1].set_ylabel("Feature Count", fontsize=11)
        axes[1].set_xlabel("SFS Internal Model", fontsize=11)
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt="%.0f", padding=3, fontsize=10)

        plt.tight_layout()

        # Tiến hành lưu kết quả đồ họa
        plot_dir = path.evaluation_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        time_feat_plot_path = (
            plot_dir / f"benchmark_time_features_{variant}_{data_name}.png"
        )
        fig.savefig(time_feat_plot_path, dpi=300)
        plt.close(fig)
        print(f" Comparison plot successfully saved at:\n  {time_feat_plot_path}")

        # Tiến hành lưu kết quả bảng số liệu CSV
        report_dir = path.evaluation_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv_path = (
            report_dir / f"benchmark_metrics_summary_{variant}_{data_name}.csv"
        )

        # 1. Lọc lấy các cột cần thiết và tạo bản sao độc lập
        selected_cols = [
            "model_name",
            "n_features_selected",
            "global_best_score",
            "Time (s)",
        ]
        summary_csv = df_metrics[selected_cols].copy()

        # 2. Gán thẳng danh sách tên mới (phải chuẩn đúng thứ tự các cột ở trên)
        summary_csv.columns = [
            "Model",
            "Selected_Features",
            "Internal_SFS_Score",
            "Time (s)",
        ]

        summary_csv.to_csv(metrics_csv_path, index=False)
        print(f"Saved benchmark metrics summary CSV at:\n  {metrics_csv_path}")
        print("\n Benchmark evaluation completed successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI tool for Features Selection Benchmark Evaluation"
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset (e.g. 'Brain', 'Breast3classes', 'colon1', 'Leukemia_3c1')",
    )

    parser.add_argument(
        "variant",
        type=str,
        choices=["raw", "union"],
        help="Dataset variant to evaluate (default: 'raw')",
    )

    args = parser.parse_args()

    evaluate_benchmark(data_name=args.dataset, variant=args.variant)
