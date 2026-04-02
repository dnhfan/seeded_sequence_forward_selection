import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from src.utils import load_seed_from_csv, validate_features
from src.utils.Timer import TimerContext

from .models import get_model


@dataclass
class ForwardSelectionState:
    # Selection inputs
    X_indices: list[int]
    selected_indices: list[int]

    # Score tracking
    current_score: float
    global_best_score: float
    global_best_indices: list[int]

    # Iteration tracking
    patience_counter: int = 0
    iteration: int = 0
    history: list[dict] = field(default_factory=list)

    # Timing tracking
    total_time_ms: float = 0


class SeededForwardSelection(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """
    Seeded Forward Selection: re write FS of wrapper method
    - add seed features
    - Using K-fold 1 time then evaluate
    """

    def __init__(
        self,
        seed_source: Union[str, List[str]],
        n_seeds: int = 1,
        model: Union[str, BaseEstimator] = "logistic",
        scoring: str = "accuracy",
        cv: int = 5,
        cv_shuffle: bool = True,
        cv_stratified: bool = True,
        max_features: Optional[int] = 100,
        patience: Optional[int] = 5,
        random_state: int = 42,
        verbose: int = 2,
        n_jobs: int = -1,
        using_timer: bool = True,
        unit: str = "ms",
    ) -> None:
        super().__init__()
        self.seed_source = seed_source
        self.n_seeds = n_seeds
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.cv_shuffle = cv_shuffle
        self.cv_stratified = cv_stratified
        self.max_features = max_features
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.using_timer = using_timer
        self.unit = unit

    def _initialize_seed_features(self, X_columns: list[str]) -> list[str]:
        """
        init seed features

        Return:
        -------------
        list of seed features
        """
        if isinstance(self.seed_source, list):
            seeds = self.seed_source[: self.n_seeds]
        elif isinstance(self.seed_source, str):
            seeds = load_seed_from_csv(self.seed_source, self.n_seeds)
        else:
            raise ValueError(f" Error: seed_source should be 'str' or 'list'.")

        validate_features(seeds, X_columns)

        return seeds

    def _build_cv(self) -> StratifiedKFold | KFold:
        """
        Build deterministic CV splitter from config.
        """
        if self.cv < 2:
            raise ValueError(" Error: cv val must be >= 2")
        if self.cv_stratified:
            return StratifiedKFold(
                n_splits=self.cv,
                shuffle=self.cv_shuffle,
                random_state=self.random_state if self.cv_shuffle else None,
            )
        return KFold(
            n_splits=self.cv,
            shuffle=self.cv_shuffle,
            random_state=self.random_state if self.cv_shuffle else None,
        )

    def _evaluate_feature_set(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        """
        evaluate features in each loop
        Returns
        -------------
        mean score
        """
        check_is_fitted(self, "_cv_splits_")

        score: np.ndarray = cross_val_score(
            self.model,
            X_np[:, selected_indices],
            y_np,
            cv=self._cv_splits_,
            scoring=self.scoring,
            n_jobs=1,
        )

        return float(score.mean())

    def _get_candidate_features(
        self,
        X_indices: list[int],
        selected_indices: list[int],
    ) -> list[int]:
        """
        return get the untaken (candidate) features
        """
        selected_set = set(selected_indices)

        return [i for i in X_indices if i not in selected_set]

    def _select_best_candidate(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        selected_indices: list[int],
        candidates: list[int],
    ) -> tuple[int, float]:
        """
        Evaluate all candidate features in parallel and return the best one.

        Parameters
        ----------
        X_np : np.ndarray
            Full feature matrix.
        y_np : np.ndarray
            Target vector.
        selected_indices : list[int]
            Currently selected feature indices (seed + previously added).
        candidates : list[int]
            Remaining feature indices to evaluate.

        Returns
        -------
        tuple[int, float]
            Best candidate feature index and its score.
        """

        # parallel (n) -> make a worker pool with n cpu core
        # parallel()(this) -> "this" is the work we need to do
        # delayed(function)(argm) -> delayed will put function to a tuple (task object)
        # evaluate return -> score, for candidate loop -> delay candidates time -> run candidate time -> return a list of score

        scores: list[float] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_feature_set)(X_np, y_np, selected_indices + [c])
            for c in candidates
        )  # type: ignore[assignment]

        best_idx = np.argmax(scores)
        best_feature = candidates[best_idx]
        best_score = scores[best_idx]

        return best_feature, best_score

    def _initialize_fit_state(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        X_columns: list[str],
    ) -> tuple[ForwardSelectionState, Union[str, BaseEstimator]]:
        """
        Initialize the fit's state
        """

        # Build model instance used during evaluation.
        if isinstance(self.model, str):
            self._model_instance = get_model(self.model, random_state=self.random_state)
        else:
            self._model_instance = self.model

        # Keep user configuration and swap runtime model.
        original_model = self.model
        self.model = self._model_instance

        # Initialize seed features.
        self._X_columns = X_columns
        self._name_to_idx = {name: i for i, name in enumerate(X_columns)}
        X_indices = list(range(X_np.shape[1]))

        seed_strings: list[str] = self._initialize_seed_features(X_columns)
        selected_indices: list[int] = [self._name_to_idx[s] for s in seed_strings]

        # Build and freeze CV splits to keep evaluation deterministic.
        self._cv_engine_ = self._build_cv()
        self._cv_splits_ = list(self._cv_engine_.split(X_np, y_np))

        # Evaluate baseline score from seed set.
        current_score: float = self._evaluate_feature_set(X_np, y_np, selected_indices)

        # Initialize tracking state.
        patience_counter = 0
        history: list[dict] = []
        iteration = 0

        # Initialize global best from baseline.
        global_best_score = current_score
        global_best_indices = list(selected_indices)

        state = ForwardSelectionState(
            X_indices=X_indices,
            selected_indices=selected_indices,
            current_score=current_score,
            global_best_score=global_best_score,
            global_best_indices=global_best_indices,
            patience_counter=patience_counter,
            iteration=iteration,
            history=history,
        )
        if self.verbose >= 1:
            seed_names = [X_columns[i] for i in selected_indices]
            print(
                f"  Start: seed={seed_names}, baseline score={state.current_score:.4f}"
            )

        return state, original_model

    def _update_global_best_and_patience(
        self,
        state: ForwardSelectionState,
        best_score: float,
    ) -> bool:
        """
        Update global-best tracking and patience.
        Returns:
            True  -> new global peak
            False -> no improvement
        """

        is_new_peak = best_score > state.global_best_score

        # Update global best and patience counter.
        if is_new_peak:
            state.global_best_score = best_score
            state.global_best_indices = list(state.selected_indices)
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            if self.verbose >= 2:
                print(
                    f"  No improvement. Patience{state.patience_counter}/{self.patience}"
                )

        return is_new_peak

    def _build_history_row(
        self,
        state: ForwardSelectionState,
        best_feature_index: int,
        best_score: float,
        step_improvement: float,
        is_new_peak: bool,
        elapsed_ms: float = 0.0,
    ) -> dict[str, object]:
        """
        Build one history row for the current iteration.
        """
        best_feature_name = self._X_columns[best_feature_index]
        selected_names = [self._X_columns[i] for i in state.selected_indices]

        return {
            "iteration": state.iteration,
            "best_candidate": best_feature_name,
            "best_score": round(best_score, 6),
            "improvement": round(step_improvement, 6),
            "n_selected": len(state.selected_indices),
            "selected_features": selected_names,
            "is_new_peak": is_new_peak,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def _run_single_iteration(
        self,
        state: ForwardSelectionState,
        X_np: np.ndarray,
        y_np: np.ndarray,
    ) -> bool:
        """
        Run one forward-selection iteration
        Return:
            True -> continue loop
            False -> stop loop (no candidates left)
        """
        with TimerContext(
            name=f"SFS.iter_{state.iteration + 1}",
            unit=self.unit,
            enabled=self.using_timer,
        ) as iter_timer:
            state.iteration += 1
            candidates = self._get_candidate_features(
                state.X_indices, state.selected_indices
            )

            # Stop when all features are already selected.
            if not candidates:
                if self.verbose >= 1:
                    print(" No more candidates. Stopping...")
                return False

            # Find best candidate for this iteration.
            best_feature_index, best_score = self._select_best_candidate(
                X_np, y_np, state.selected_indices, candidates
            )

            # Add the top candidate.
            state.selected_indices.append(best_feature_index)

            step_improvement = best_score - state.current_score
            is_new_peak = self._update_global_best_and_patience(state, best_score)

            state.current_score = best_score

            if self.verbose >= 2:
                print(
                    f"  Iter {state.iteration:>3}: + {self._X_columns[best_feature_index]}"
                    f"  score={best_score:.4f}"
                    f"  󰇂 = {step_improvement:+.4f} "
                )

        # Save iteration history.
        row = self._build_history_row(
            state,
            best_feature_index,
            best_score,
            step_improvement,
            is_new_peak,
            elapsed_ms=iter_timer.elapsed,
        )
        state.history.append(row)

        state.total_time_ms += iter_timer.elapsed

        return True

    def _should_stop(self, state: ForwardSelectionState) -> bool:
        """
        Check stopping criteria for forward-selection loop
        Returns:
            True  -> stop
            False -> continue
        """
        # Stop if max feature limit is reached.
        if (
            self.max_features is not None
            and len(state.selected_indices) >= self.max_features
        ):
            if self.verbose >= 1:
                print(f" Reached max_features={self.max_features}. Stopping...")
            return True

        # Stop when no improvement happens for configured patience.
        if self.patience is not None and state.patience_counter >= self.patience:
            if self.verbose >= 1:
                print(f" Patience exhausted ({self.patience}). Stopping...")
            return True
        return False

    def _finalize_fit(
        self,
        state: ForwardSelectionState,
        X: pd.DataFrame,
        original_model: Union[str, BaseEstimator],
    ) -> None:
        """
        Finalize fit by persisting fitted attributes and restoring model config.

        Parameters
        ----------
        state : ForwardSelectionState
            Selection state containing best features, score tracking, and history.
        X : pd.DataFrame
            Input feature matrix used during fitting.
        original_model : Union[str, BaseEstimator]
            User-provided model configuration to restore after internal evaluation.
        """
        # Persist fitted attributes.
        self.selected_features_ = [
            self._X_columns[i] for i in state.global_best_indices
        ]
        self.global_best_score_ = state.global_best_score
        self.n_features_in_ = X.shape[1]
        self.model = original_model
        self.history_ = state.history
        self.total_iter_time_ms_ = state.total_time_ms
        self.iteration_time_ms_ = [row["elapsed_ms"] for row in state.history]

        if self.verbose >= 1:
            print(
                f"\n Done: {len(state.global_best_indices)} features selected. Final score={state.global_best_score:.4f}"
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeededForwardSelection":
        """
        Run SFS - Seeded Forward Selection
            - X: features df
            - y: target Series
        """
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        with TimerContext(
            name="SFS.fit",
            unit=self.unit,
            enabled=self.using_timer,
        ) as total_timer:
            X_np = X.to_numpy(dtype=np.float32)
            y_np = y.to_numpy()
            X_columns = X.columns.tolist()
            state, original_model = self._initialize_fit_state(
                X_np=X_np, y_np=y_np, X_columns=X_columns
            )

            # Forward selection loop.
            while True:
                should_continue = self._run_single_iteration(state, X_np, y_np)
                if not should_continue:
                    break

                if self._should_stop(state):
                    break

            self._finalize_fit(state, X, original_model)
        self.total_fit_time_ms_ = total_timer.elapsed

        return self

    def _get_support_mask(self) -> np.ndarray:  # type: ignore[override]
        """
        Required by SelectorMixin -> auto-generates transform() and get_support().
        """

        check_is_fitted(self, "selected_features_")

        return np.array(
            [col in set(self.selected_features_) for col in self._X_columns]
        )

    def get_feature_names_out(self) -> np.ndarray:  # type: ignore[override]
        """
        Return selected features names as array.
        """

        check_is_fitted(self, "selected_features_")

        return np.array(self.selected_features_)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:  # type: ignore[override]
        """
        Fit and Return X with only selected features as DataFrame
        """
        self.fit(X, y)
        return pd.DataFrame(X[self.selected_features_])

    def _generate_txt_report(self) -> str:
        """
        Generate a human-readable TXT report of the SFS execution.
        Returns:
            str: Formatted report content
        """
        from datetime import datetime

        final_score = self.global_best_score_

        lines = []

        lines.append("=" * 60)
        lines.append(" SEEDED FORWARD SELECTION (SFS) - EXECUTION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        lines.append("-" * 40)
        lines.append(" CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  {'Model:':<20} {self.model}")
        lines.append(f"  {'Scoring:':<20} {self.scoring}")
        lines.append(f"  {'CV Folds:':<20} {self.cv}")
        lines.append(f"  {'CV Shuffle:':<20} {self.cv_shuffle}")
        lines.append(f"  {'CV Stratified:':<20} {self.cv_stratified}")
        lines.append(f"  {'Max Features:':<20} {self.max_features}")
        lines.append(f"  {'Patience:':<20} {self.patience}")
        lines.append(f"  {'N Seeds:':<20} {self.n_seeds}")
        lines.append(f"  {'Random State:':<20} {self.random_state}")
        lines.append(f"  {'N Jobs:':<20} {self.n_jobs}")
        lines.append(f"  {'Timing Enabled:':<20} {self.using_timer}")
        lines.append(f"  {'Time Unit:':<20} {self.unit}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("󰈙 SUMMARY")
        lines.append("-" * 40)
        lines.append(
            f"  {'Total Fit Time:':<20} {self.total_fit_time_ms_:.3f} {self.unit}"
        )
        lines.append(f"  {'Final Best Score:':<20} {final_score:.6f}")
        lines.append(f"  {'Total Selected:':<20} {len(self.selected_features_)}")
        lines.append(
            f"  {'Selected Features:':<20} {', '.join(self.selected_features_)}"
        )
        lines.append("")

        lines.append("-" * 40)
        lines.append(" ITERATION LOGS")
        lines.append("-" * 40)

        header = f"  {'Iter':<6} {'Candidate':<25} {'Score':<12} {'Improvement':<12} {'Time':<10}"
        lines.append(header)
        lines.append("  " + "-" * 60)

        for row in self.history_:
            iter_num = row.get("iteration", 0)
            candidate = row.get("best_candidate", "N/A")
            score = row.get("best_score", 0.0)
            improvement = row.get("improvement", 0.0)
            elapsed = row.get("elapsed_ms", 0.0)

            lines.append(
                f"  {iter_num:<6} {candidate[:22]:<25} {score:<12.6f} {improvement:<+12.6f} {elapsed:<10.3f}"
            )

        lines.append("")
        lines.append("=" * 60)
        lines.append(" END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_history(self, file_path: str) -> None:
        """
        Saving history (data processed) (JSON / csv / txt)
        """
        if not hasattr(self, "history_") or not self.history_:
            raise RuntimeError(" Error: no history_ found. Did you run fit()?")

        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            pd.DataFrame(self.history_).to_csv(file_path, index=False)
        elif ext == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.history_, f, indent=2, ensure_ascii=False)
        elif ext == ".txt":
            report = self._generate_txt_report()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)
        else:
            raise ValueError(
                f" Error: unsupported file extension {ext}. Choose .json, .csv, or .txt"
            )
        if self.verbose >= 1:
            print(f" saved history in {file_path}.")
