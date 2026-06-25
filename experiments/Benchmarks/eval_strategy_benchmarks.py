"""CLI benchmark: evaluate SFS result files with all 3 strategies (cv, tts, custom_cv).

Usage:
    python experiments/Benchmarks/eval_strategy_benchmarks.py colon1
    python experiments/Benchmarks/eval_strategy_benchmarks.py colon1 -v union -k 5 -i 50
    python experiments/Benchmarks/eval_strategy_benchmarks.py Brain -v raw --skip-baseline
"""

import argparse
import sys
from pathlib import Path

# ── repo root on sys.path ──────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from experiments.Benchmarks.benchmark_helper import (
    build_smart_labels,
    discover_sfs_files,
)
from src.config import ProjectPath
from src.modeling.evaluation import ModelEvaluator
from src.modeling.plot_helper import generate_strategy_comparison_chart

# ── core ───────────────────────────────────────────────────────────────

STRATEGIES = ["cv", "tts", "custom_cv"]


def run_benchmark(
    data_name: str,
    variant: str = "union",
    n_features: int = 50,
    cv_splits: int = 5,
    n_iter: int = 100,
    test_size: float = 0.3,
    output_dir: str | None = None,
    skip_baseline: bool = False,
    algorithms: list[str] | None = None,
    models: list[str] | None = None,
) -> None:
    """Run the full strategy benchmark for one dataset."""

    path = ProjectPath(data_name=data_name, n_features=n_features)
    eval_dir = Path(output_dir) if output_dir else path.evaluation_dir

    print(f"\n{'='*60}")
    print(f"  Strategy Benchmark: {data_name}  (variant={variant})")
    print(f"  Output: {eval_dir}")
    print(f"{'='*60}")

    all_fold_rows: list[dict] = []
    summary_rows: list[dict] = []

    # ── Baseline results ─────────────────────────────────────────────
    if not skip_baseline:
        print(f"\n{'─'*40}")
        print("  Running baseline (raw data) with all strategies...")
        print(f"{'─'*40}")

        for strategy in STRATEGIES:
            print(f"\n--- Strategy: {strategy.upper()} ---")
            evaluator = ModelEvaluator(data_name, custom_base_dir=eval_dir)
            evaluator.evaluate_baseline(
                str(path.raw_path),
                n_splits=cv_splits,
                eval_strategy=strategy,
                n_iter=n_iter,
                test_size=test_size,
            )
            for row in evaluator.fold_results:
                row["Source"] = "Baseline"
                all_fold_rows.append(row)
            for row in evaluator.model_results:
                row["Source"] = "Baseline"
                summary_rows.append(row)

    # ── SFS file results ─────────────────────────────────────────────
    sfs_files = discover_sfs_files(
        path.wrapper_dir, variant, algorithms=algorithms, models=models
    )
    if not sfs_files:
        print(f"\n  No {variant} SFS files found in {path.wrapper_dir / variant}")
    else:
        print(f"\n{'─'*40}")
        print(f"  Running {len(sfs_files)} SFS file(s) with all strategies...")
        print(f"{'─'*40}")

        labels = build_smart_labels(sfs_files, data_name)

        for sfs_file, label in zip(sfs_files, labels):
            print(f"\n  File: {sfs_file.name}  (label={label})")

            for strategy in STRATEGIES:
                print(f"--- Strategy: {strategy.upper()} ---")
                evaluator = ModelEvaluator(data_name, custom_base_dir=eval_dir)
                evaluator.evaluate_custom_file(
                    str(sfs_file),
                    method_label=label,
                    n_splits=cv_splits,
                    eval_strategy=strategy,
                    n_iter=n_iter,
                    test_size=test_size,
                )
                for row in evaluator.fold_results:
                    row["Source"] = label
                    all_fold_rows.append(row)
                for row in evaluator.model_results:
                    row["Source"] = label
                    summary_rows.append(row)

    # ── Summary CSV + Combined Strategy Chart ────────────────────────
    print(f"\n{'─'*40}")
    print("  Generating summary + combined strategy chart...")
    print(f"{'─'*40}")

    if all_fold_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_cols: list[str] = [
            "Source",
            "Model",
            "Strategy",
            "mean_acc",
            "std",
            "min",
            "max",
            "n_fold",
        ]

        sort_cols = ["Source", "Model", "Strategy"]
        summary_df = summary_df[summary_cols].sort_values(by=sort_cols)  # type: ignore

        report_dir = eval_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = (
            report_dir / f"strategy_benchmark_summary_{variant}_{data_name}.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary saved: {summary_path}")
        print(summary_df.to_string(index=False))

        fold_df = pd.DataFrame(all_fold_rows)
        chart_df = fold_df[["Method", "Model", "Strategy", "Acc"]].dropna()

        if not chart_df.empty:
            plot_dir = eval_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"strategy_comparison_{variant}_{data_name}.png"
            generate_strategy_comparison_chart(
                fold_level_df=chart_df,
                chart_title=f"Strategy Comparison ({variant.upper()})",
                data_name=data_name,
                save_path=plot_path,
                horizontal=True,
            )
    else:
        print("\n  No results to summarize.")


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark: evaluate SFS results with cv / tts / custom_cv strategies"
    )
    parser.add_argument("dataset", type=str, help="Dataset name (e.g. colon1, Brain)")
    parser.add_argument(
        "-v",
        "--variant",
        type=str,
        default="union",
        choices=["raw", "union"],
        help="Data variant (default: union)",
    )
    parser.add_argument(
        "-n",
        "--n-features",
        type=int,
        default=50,
        help="Number of features (default: 50)",
    )
    parser.add_argument(
        "-k",
        "--cv-splits",
        type=int,
        default=5,
        help="CV folds for cv/custom_cv strategies (default: 5)",
    )
    parser.add_argument(
        "-i",
        "--n-iter",
        type=int,
        default=100,
        help="TTS iterations (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--test-size",
        type=float,
        default=0.3,
        help="TTS test fraction (default: 0.3)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (default: results/<dataset>/evaluation/)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip raw baseline evaluation",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Filter SFS files by algorithm name (e.g. seededsfsselector sklearnsfsselector)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter SFS files by model type (e.g. log dt rf)",
    )

    args = parser.parse_args()

    if args.cv_splits < 2:
        parser.error("--cv-splits must be >= 2")

    run_benchmark(
        data_name=args.dataset,
        variant=args.variant,
        n_features=args.n_features,
        cv_splits=args.cv_splits,
        n_iter=args.n_iter,
        test_size=args.test_size,
        output_dir=args.output_dir,
        skip_baseline=args.skip_baseline,
        algorithms=args.algorithms,
        models=args.models,
    )
