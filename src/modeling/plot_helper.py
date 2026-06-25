import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.container import BarContainer


def compute_method_order(fold_level_df: pd.DataFrame) -> list:
    """Sort feature-selection methods by mean accuracy (descending)."""
    mean_acc_df = fold_level_df.groupby("Method", as_index=False).agg(
        mean_acc=("Acc", "mean")
    )
    mean_acc_df = cast(pd.DataFrame, mean_acc_df)
    mean_acc_df = mean_acc_df.sort_values(by="mean_acc", ascending=False)
    return mean_acc_df["Method"].tolist()


def generate_performance_chart(
    fold_level_df: pd.DataFrame,
    chart_title: str,
    data_name: str,
    save_path: Path,
    figsize: Optional[Tuple[int, int]] = None,
    horizontal: Optional[bool] = None,
) -> Path:
    """Create and save a grouped bar chart of accuracy per method/model.

    Returns the Path where the plot was saved.
    """
    sns.set_theme(style="whitegrid")

    n_methods = fold_level_df["Method"].nunique()
    use_horizontal = horizontal if horizontal is not None else (n_methods > 6)
    method_order = compute_method_order(fold_level_df)

    if figsize is None:
        if use_horizontal:
            figsize = (10, int(max(4, 3 + n_methods * 0.8)))
        else:
            figsize = (int(max(12, 6 + n_methods * 1.2)), 7)

    plt.figure(figsize=figsize)

    if use_horizontal:
        ax = sns.barplot(
            data=fold_level_df,
            y="Method",
            x="Acc",
            hue="Model",
            palette="pastel",
            orient="h",
            order=method_order,
        )
        x_label = "Accuracy Score"
        y_label = "Feature Selection Method"
    else:
        ax = sns.barplot(
            data=fold_level_df,
            x="Method",
            y="Acc",
            hue="Model",
            palette="pastel",
            order=method_order,
        )
        x_label = "Feature Selection Method"
        y_label = "Accuracy Score"
        plt.xticks(rotation=45, ha="right", fontweight="bold")

    acc_series = cast(pd.Series, fold_level_df["Acc"])
    min_acc = float(acc_series.min())
    max_acc = float(acc_series.max())

    if use_horizontal:
        x_min = max(0.0, min_acc - 0.05)
        x_max = min(1.05, max_acc + 0.08)
        ax.set_xlim(x_min, x_max)
    else:
        y_min = max(0.0, min_acc - 0.05)
        y_max = min(1.1, max_acc + 0.12)
        ax.set_ylim(y_min, y_max)

    # Smart bar labels – avoids overlap with error bars
    for container in ax.containers:
        if not isinstance(container, BarContainer):
            continue
        for bar in container:
            if use_horizontal:
                x = bar.get_width()
                if not x or x != x:
                    continue

                y_center = bar.get_y() + bar.get_height() / 2
                label_x = x

                for line in ax.lines:
                    y_data = cast(Sequence[float], line.get_ydata())
                    if len(y_data) > 0 and abs(y_data[0] - y_center) < 1e-4:
                        x_data = cast(Sequence[float], line.get_xdata())
                        label_x = max(label_x, max(x_data))
                        break

                ax.annotate(
                    f"{x:.4f}",
                    xy=(label_x, y_center),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=7,
                    color="black",
                )
            else:
                y = bar.get_height()
                if not y or y != y:
                    continue

                x_center = bar.get_x() + bar.get_width() / 2
                label_y = y

                for line in ax.lines:
                    x_data = cast(Sequence[float], line.get_xdata())
                    if len(x_data) > 0 and abs(x_data[0] - x_center) < 1e-4:
                        y_data = cast(Sequence[float], line.get_ydata())
                        label_y = max(label_y, max(y_data))
                        break

                ax.annotate(
                    f"{y:.4f}",
                    xy=(x_center, label_y),
                    xytext=(0, 3),
                    textcoords="offset points",
                    va="bottom",
                    ha="left",
                    fontsize=7,
                    color="black",
                    rotation=45,
                    rotation_mode="anchor",
                )

    plt.title(f"{chart_title} ({data_name})")
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    if use_horizontal:
        plt.legend(
            bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=2, frameon=True
        )
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Chart saved at: {save_path}")

    return save_path


def generate_strategy_comparison_chart(
    fold_level_df: pd.DataFrame,
    chart_title: str,
    data_name: str,
    save_path: Path,
    figsize: Optional[Tuple[int, int]] = None,
    horizontal: Optional[bool] = None,
) -> Path:
    """Create and save a grouped bar chart comparing accuracy across evaluation strategies.

    The chart shows feature-selection methods on one axis, accuracy on the other,
    with bars grouped by Strategy (hue) and faceted by Model.

    Args:
        fold_level_df: Must contain columns ``Method``, ``Model``, ``Strategy``, ``Acc``.
        chart_title: Title shown above the chart.
        data_name: Dataset name (shown in title).
        save_path: File path for the saved PNG.
        figsize: Custom ``(width, height)``. Auto-sized if ``None``.
        horizontal: Use horizontal bars. Auto-detect if ``None``.

    Returns:
        The ``Path`` where the plot was saved.
    """
    sns.set_theme(style="whitegrid")

    required = {"Method", "Model", "Strategy", "Acc"}
    if not required.issubset(fold_level_df.columns):
        missing = required - set(fold_level_df.columns)
        raise ValueError(f"fold_level_df is missing columns: {missing}")

    n_methods = fold_level_df["Method"].nunique()
    use_horizontal = horizontal if horizontal is not None else (n_methods > 6)
    method_order = compute_method_order(fold_level_df)

    models = sorted(fold_level_df["Model"].unique())
    n_models = len(models)

    if figsize is None:
        if use_horizontal:
            figsize = (12, int(max(5, 4 + n_methods * 0.9)) * n_models)
        else:
            figsize = (int(max(14, 7 + n_methods * 1.4)), 5 * n_models)

    fig, axes = plt.subplots(n_models, 1, figsize=figsize, squeeze=False)
    fig.suptitle(f"{chart_title} ({data_name})", fontsize=14, y=1.01)

    for idx, model_name in enumerate(models):
        ax = axes[idx, 0]
        model_df = fold_level_df[fold_level_df["Model"] == model_name]

        if use_horizontal:
            sns.barplot(
                data=model_df,
                y="Method",
                x="Acc",
                hue="Strategy",
                palette="Set2",
                orient="h",
                order=method_order,
                ax=ax,
            )
            x_label = "Accuracy Score"
            y_label = "Feature Selection Method" if idx == n_models - 1 else ""
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

            acc_series = cast(pd.Series, model_df["Acc"])
            x_min = max(0.0, float(acc_series.min()) - 0.05)
            x_max = min(1.05, float(acc_series.max()) + 0.08)
            ax.set_xlim(x_min, x_max)
        else:
            sns.barplot(
                data=model_df,
                x="Method",
                y="Acc",
                hue="Strategy",
                palette="Set2",
                order=method_order,
                ax=ax,
            )
            x_label = "Feature Selection Method" if idx == n_models - 1 else ""
            y_label = "Accuracy Score"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=45)

            acc_series = cast(pd.Series, model_df["Acc"])
            y_min = max(0.0, float(acc_series.min()) - 0.05)
            y_max = min(1.1, float(acc_series.max()) + 0.12)
            ax.set_ylim(y_min, y_max)

        ax.set_title(f"Model: {model_name}", fontsize=11)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

        for container in ax.containers:
            if not isinstance(container, BarContainer):
                continue
            for bar in container:
                if use_horizontal:
                    w = bar.get_width()
                    if not w or w != w:
                        continue
                    y_center = bar.get_y() + bar.get_height() / 2
                    ax.annotate(
                        f"{w:.4f}",
                        xy=(w, y_center),
                        xytext=(4, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left",
                        fontsize=6,
                        color="black",
                    )
                else:
                    h = bar.get_height()
                    if not h or h != h:
                        continue
                    x_center = bar.get_x() + bar.get_width() / 2
                    ax.annotate(
                        f"{h:.4f}",
                        xy=(x_center, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        va="bottom",
                        ha="left",
                        fontsize=6,
                        color="black",
                        rotation=45,
                        rotation_mode="anchor",
                    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Chart saved at: {save_path}")

    return save_path


def compute_summary_df(fold_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level accuracy into per-method/model summary stats."""
    summary_df = fold_level_df.groupby(["Method", "Model"], as_index=False).agg(
        mean_accuracy=("Acc", "mean"),
        std_accuracy=("Acc", "std"),
        min_accuracy=("Acc", "min"),
        max_accuracy=("Acc", "max"),
        median_accuracy=("Acc", "median"),
        n_folds=("Acc", "count"),
    )
    summary_df = cast(pd.DataFrame, summary_df)
    summary_df["std_accuracy"] = summary_df["std_accuracy"].fillna(0.0)

    summary_df = summary_df.sort_values(
        by="mean_accuracy", ascending=False
    ).reset_index(drop=True)

    summary_df["cv_stability"] = 1.0 - summary_df["std_accuracy"]
    summary_df["rank"] = (
        summary_df["mean_accuracy"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return summary_df


def write_evaluation_report(
    report_path: Path,
    plot_path: Path,
    fold_level_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    data_name: str,
    n_features: int,
    experiment_prefix: str,
    save_prefix: str,
    timestamp: str,
    report_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the text evaluation report to *report_path*."""
    best_row = summary_df.iloc[0]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODEL EVALUATION REPORT\n")
        f.write(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"experiment_id: {save_prefix}_{data_name}_{timestamp.replace('-', '').replace('_', '')}\n"
        )
        f.write(f"dataset: {data_name}\n")
        f.write(f"experiment_prefix: {experiment_prefix}\n")
        f.write(f"feature_count: {n_features}\n")
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
            f.write(f" Method: {method_name} \n")

            clean_df = group_df.drop(columns=["Method"])
            f.write(clean_df.to_string(index=False))

            f.write("\n\n")

        f.write("\n" + "-" * 120 + "\n")
    print(f"󰎞 Report saved at: {report_path}")


def generate_time_comparison_chart(
    seeded_metrics_path: str | Path,
    sklearn_metrics_path: str | Path,
    data_name: str,
    output_name: str = "time_comparison_seeded_vs_sklearn.png",
    algorithm_labels: Tuple[str, str] = ("SeededSFS", "SklearnSFS"),
    save_dir: str | Path | None = None,
    show_plot: bool = False,
) -> pd.DataFrame:
    """Compare fit time between two algorithms using metrics CSV files.

    Args:
        seeded_metrics_path: Path to first algorithm's metrics.csv.
        sklearn_metrics_path: Path to second algorithm's metrics.csv.
        data_name: Dataset name (used in chart title).
        output_name: Output filename for the plot image.
        algorithm_labels: Labels shown on x-axis.
        save_dir: Directory to save the plot. Defaults to current dir.
        show_plot: Whether to display the chart after saving.

    Returns:
        pd.DataFrame: Comparison table with total fit time in ms and seconds.
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

    target_plot_dir = Path(save_dir) if save_dir else Path(".")
    target_plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_plot_dir / output_name

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        comparison_df["algorithm"],
        comparison_df["total_fit_time_sec"],
    )

    ax.set_title(f"{data_name} Dataset: Fit Time Comparison", fontsize=14)
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

    print(f" Chart saved at: {output_path}")

    return comparison_df
