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
from sklearn.utils.validation import check_is_fitted

from src.utils import load_seed_from_csv, validate_features

from .models import get_model


@dataclass
class ForwardSelectionState:
    # for init seed
    X_columns: list[str]
    selected: list[str]

    # for evaluate
    current_score: float
    global_best_score: float
    global_best_features: list[str]

    # for while loop
    patience_counter: int = 0
    iteration: int = 0
    history: list[dict] = field(default_factory=list)


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

    def _initialize_seed_features(self, X_columns: list) -> list:
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

    def _evaluate_feature_set(self, X, y, features) -> float:
        """
        evaluate features in each loop
        Returns
        -------------
        mean score
        """
        check_is_fitted(self, "_cv_splits_")

        score: np.ndarray = cross_val_score(
            self.model,
            X[features],
            y,
            cv=self._cv_splits_,
            scoring=self.scoring,
            n_jobs=1,
        )

        return float(score.mean())

    def _get_candidate_features(self, X_columns: list, selected: list) -> list:
        """
        return get the untaken (candidate) features
        """
        selected_set = set(selected)

        return [c for c in X_columns if c not in selected_set]

    def _select_best_candidate(self, X, y, selected, candidates):
        """
             Evaluate all candidate features in parallel, return the best one.
                 - X: full feature DataFrame
                 - y: target Series
                 - selected: currently selected features (seed + previously
        added)
                 - candidates: remaining features to evaluate
             Returns: (best_feature_name, best_score)
        """

        # parallel (n) -> make a woker pool with n cpu core
        # parallel()(this) -> "this" is the work we need to do
        # delayed(function)(argm) -> delayed will put function to a tuple (task object)
        # evaluate return -> score, for candidate loop -> delay candidates time -> run candidate time -> return a list of score

        scores: list[float] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_feature_set)(X, y, selected + [c])
            for c in candidates
        )  # type: ignore[assignment]

        best_idx = np.argmax(scores)
        best_feature = candidates[best_idx]
        best_score = scores[best_idx]

        return best_feature, best_score

    def _initialize_fit_state(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[ForwardSelectionState, Union[str, BaseEstimator]]:
        """
        Initialize the fit's state
        """

        # 1. Build model instance
        if isinstance(self.model, str):
            self._model_instance = get_model(self.model, random_state=self.random_state)
        else:
            self._model_instance = self.model

        # Swap self.model
        original_model = self.model
        self.model = self._model_instance

        # 2. Init Seed
        X_columns = X.columns.tolist()
        selected: list[str] = self._initialize_seed_features(X_columns)

        # 3. Precompute and freeze CV splits (Fixed CV)

        # Build
        self._cv_engine_ = self._build_cv()

        # Freeze
        self._cv_splits_ = list(self._cv_engine_.split(X, y))

        # 4. Evaluate seed baseline
        current_score: float = self._evaluate_feature_set(X, y, selected)

        # 5. Init tracking
        patience_counter = 0
        history: list[dict] = []
        iteration = 0

        # Save the global best score + features
        global_best_score = current_score
        global_best_features = list(selected)

        state = ForwardSelectionState(
            X_columns=X_columns,
            selected=selected,
            current_score=current_score,
            global_best_score=global_best_score,
            global_best_features=global_best_features,
            patience_counter=patience_counter,
            iteration=iteration,
            history=history,
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

        # Accept or reject
        if is_new_peak:
            state.global_best_score = best_score
            state.global_best_features = list(state.selected)
            state.patience_counter = 0
        else:
            state.patience_counter += 1
            if self.verbose >= 2:
                print(
                    f"  No improvement. Patience{state.patience_counter}/{self.patience}"
                )

        return is_new_peak

    def _run_single_iteration(
        self,
        state: ForwardSelectionState,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> bool:
        """
        Run one forward-selection iteration
        Return:
            True -> continue loop
            False -> stop loop (no candicates left)
        """
        state.iteration += 1
        candidates = self._get_candidate_features(state.X_columns, state.selected)

        # No more candidates -> stop
        if not candidates:
            if self.verbose >= 1:
                print(" No more candidates. Stoping..")
            return False

        # Find best candidate this iteration
        best_feature, best_score = self._select_best_candidate(
            X, y, state.selected, candidates
        )

        # Always append the top 1 features
        state.selected.append(best_feature)

        step_improvement = best_score - state.current_score
        is_new_peak = self._update_global_best_and_patience(state, best_score)

        # Log iteration
        state.history.append(
            {
                "iteration": state.iteration,
                "best_candidate": best_feature,
                "best_score": round(best_score, 6),
                "improvement": round(step_improvement, 6),
                "n_selected": len(state.selected),
                "selected_features": list(state.selected),
                "is_new_peak": is_new_peak,
            }
        )

        state.current_score = best_score

        if self.verbose >= 2:
            print(
                f"  Iter {state.iteration:>3}: + {best_feature}"
                f"  score={best_score:.4f}"
                f"  󰇂 = {step_improvement:+.4f} "
            )

        return True

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Run SFS - Seeded Forward Selection
            - X: features df
            - y: target Series
        """
        #
        state, _original_model = self._initialize_fit_state(X=X, y=y)

        if self.verbose >= 1:
            print(
                f"  Start: seed={state.selected}, baseline score={state.current_score:.4f}"
            )

        # 6. Forward selection loop
        while True:
            should_continue = self._run_single_iteration(state, X, y)
            if not should_continue:
                break

            # Stoping criteria
            if (
                self.max_features is not None
                and len(state.selected) >= self.max_features
            ):
                if self.verbose >= 1:
                    print(f" Reached max_features={self.max_features}. Stoping...")
                break

            if self.patience is not None and state.patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f" Patience exhausted ({self.patience}). Stoping...")
                break

        # 7. Store Results
        self.selected_features_ = state.global_best_features
        self.n_features_in_ = X.shape[1]
        self._X_columns = state.X_columns
        self.model = _original_model
        self.history_ = state.history

        if self.verbose >= 1:
            print(
                f"\n Done: {len(state.global_best_features)} features selected. Final score={state.global_best_score:.4f}"
            )
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

    def save_history(self, file_path: str):
        """
        Saving history (data processed) (JSON / csv)
        """
        if not hasattr(self, "history_") or not self.history_:
            raise RuntimeError(" Error: we dont have history_, do u run fit() ?")

        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            pd.DataFrame(self.history_).to_csv(file_path, index=False)
        elif ext == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.history_, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(
                f" Error: not support {ext} file, pls chosing .json or .csv"
            )
        if self.verbose >= 1:
            print(f" saved history in {file_path}.")
