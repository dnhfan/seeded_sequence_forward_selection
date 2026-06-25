"""Evaluation strategies for model accuracy assessment.

Each strategy encapsulates a different way of splitting data and computing
accuracy scores.  The shared bookkeeping (fold rows, model summary rows,
printing) lives in ``EvalStrategy.run`` so that concrete subclasses only
need to implement ``generate_scores``.

Available strategies
--------------------
- ``CVStrategy``       – sklearn ``cross_validate`` with ``StratifiedKFold``
- ``TTSStrategy``      – repeated stratified train/test split
- ``CustomCVStrategy`` – manual fold loop (no ``cross_validate``)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Tuple

import numpy
import pandas
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


class EvalStrategy(ABC):
    """Base class for an evaluation strategy.

    Subclasses only need to implement :meth:`generate_scores`, which returns
    ``{model_name: [accuracy_per_fold_or_iter]}``.  The shared bookkeeping
    (``fold_results`` rows, ``model_results`` summary, printing) lives in
    :meth:`run` so it is not duplicated per strategy.
    """

    @abstractmethod
    def generate_scores(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        models: Mapping[str, Any],
    ) -> Dict[str, List[float]]:
        """Run the evaluation and return per-model accuracy lists.

        Returns:
            ``{model_name: [accuracy_score, ...]}`` where each score
            corresponds to one fold or one train/test iteration.
        """
        ...

    def run(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        models: Mapping[str, Any],
        method_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute the strategy and return fold-level + model-level results.

        Args:
            X: Feature matrix.
            y: Target vector.
            models: ``{model_name: estimator}`` dict.
            method_name: Label used in reporting (e.g. feature-selection method).

        Returns:
            ``(fold_rows, model_rows)`` – two lists of dicts compatible with
            ``ModelEvaluator.fold_results`` and ``ModelEvaluator.model_results``.
        """
        accs = self.generate_scores(X, y, models)
        fold_rows: List[Dict[str, Any]] = []
        model_rows: List[Dict[str, Any]] = []

        for model_name, acc_list in accs.items():
            # fold / iter rows
            for i, acc in enumerate(acc_list):
                fold_rows.append(
                    {
                        "Method": method_name,
                        "Model": model_name,
                        "Fold": i + 1,
                        "Acc": acc,
                    }
                )

            # model summary
            acc_arr = numpy.array(acc_list)
            mean_acc = acc_arr.mean()
            model_rows.append(
                {
                    "Method": method_name,
                    "Model": model_name,
                    "mean_acc": mean_acc,
                    "std": acc_arr.std(),
                    "min": acc_arr.min(),
                    "max": acc_arr.max(),
                    "n_fold": len(acc_arr),
                }
            )
            print(f"󰄭  [{method_name:<12}] {model_name:<8} | Acc: {mean_acc:.4f} ")

        return fold_rows, model_rows


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class CVStrategy(EvalStrategy):
    """Sklearn ``cross_validate`` with ``StratifiedKFold``.

    Uses ``cross_validate`` for efficient parallel fold evaluation.
    Each fold's test accuracy is returned as one entry in the score list.
    """

    def __init__(self, n_splits: int = 5) -> None:
        """
        Args:
            n_splits: Number of CV folds.  Must be >= 2.
        """
        self.n_splits = n_splits
        self.random_state = 42

    def generate_scores(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        models: Mapping[str, Any],
    ) -> Dict[str, List[float]]:
        """Run stratified k-fold CV via ``cross_validate``."""
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        accs: Dict[str, List[float]] = {name: [] for name in models}

        for model_name, model in models.items():
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=["accuracy"],
                n_jobs=-1,
            )
            accs[model_name].append(scores["test_accuracy"])

        return accs


class TTSStrategy(EvalStrategy):
    """Repeated stratified train/test split.

    Runs ``n_iter`` independent 70/30 (by default) stratified splits, each
    with a different ``random_state`` (= iteration index) for reproducibility,
    fits each model on the train portion and scores it on the held-out test
    portion.
    """

    def __init__(self, n_iter: int = 100, test_size: float = 0.3) -> None:
        """
        Args:
            n_iter: Number of independent split/evaluate repetitions.
            test_size: Fraction of data held out for testing (0.0–1.0).
        """
        self.n_iter = n_iter
        self.test_size = test_size

    def generate_scores(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        models: Mapping[str, Any],
    ) -> Dict[str, List[float]]:
        """Run repeated stratified train/test splits."""
        accs: Dict[str, List[float]] = {name: [] for name in models}

        for i in range(self.n_iter):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=i
            )

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                accs[model_name].append(acc)

        return accs


class CustomCVStrategy(EvalStrategy):
    """Manual cross-validation loop (no ``cross_validate``).

    Uses ``StratifiedKFold`` purely for generating the train/test index splits,
    then manually fits and scores each model per fold.  Functionally equivalent
    to :class:`CVStrategy` but gives explicit control over the per-fold loop
    (useful for future per-fold customization/instrumentation).
    """

    def __init__(self, n_splits: int = 5) -> None:
        """
        Args:
            n_splits: Number of CV folds.  Must be >= 2.
        """
        self.n_splits = n_splits
        self.random_state = 42

    def generate_scores(
        self,
        X: pandas.DataFrame,
        y: pandas.Series,
        models: Mapping[str, Any],
    ) -> Dict[str, List[float]]:
        """Run manual stratified k-fold evaluation."""
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        accs: Dict[str, List[float]] = {name: [] for name in models}

        # reset_index to guarantee integer iloc alignment
        X_arr = X.reset_index(drop=True)
        y_arr = y.reset_index(drop=True)

        for train_index, test_index in skf.split(X_arr, y_arr):
            X_train, X_test = X_arr.iloc[train_index], X_arr.iloc[test_index]
            y_train, y_test = y_arr.iloc[train_index], y_arr.iloc[test_index]

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                accs[model_name].append(acc)

        return accs
