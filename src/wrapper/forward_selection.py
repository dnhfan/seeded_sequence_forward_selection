import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from .models import get_model
from .utils import load_seed_from_csv, validate_features


class SeededForwardSelection(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """
    Seeded Forward Selection: re write FS of wrapper method
    - add seed features
    """

    def __init__(
        self,
        seed_source,
        n_seeds=1,
        model="logistic",
        scoring="accuracy",
        cv=5,
        max_features=100,
        patience=5,
        random_state=42,
        verbose=2,
        n_jobs=-1,
    ) -> None:
        super().__init__()
        self.seed_source = seed_source
        self.n_seeds = n_seeds
        self.model = model
        self.scoring = scoring
        self.cv = cv
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

    def _evaluate_feature_set(self, X, y, features) -> float:
        """
        evaluate features in each loop
        Returns
        -------------
        mean score
        """
        score: np.ndarray = cross_val_score(
            self.model, X[features], y, cv=self.cv, scoring=self.scoring, n_jobs=1
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
        # parallel()(this) -> this is the work we need to do
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Run SFS - Seeded Forward Selection
            - X: features df
            - y: target Series
        """

        # 1. Build model instance
        if isinstance(self.model, str):
            self._model_instance = get_model(self.model, random_state=self.random_state)
        else:
            self._model_instance = self.model

        # Swap self.model
        _original_model = self.model
        self.model = self._model_instance

        # 2. Init Seed
        X_columns = X.columns.tolist()
        selected: list[str] = self._initialize_seed_features(X_columns)

        # 3. Evaluate seed baseline
        current_score: float = self._evaluate_feature_set(X, y, selected)

        # 4. Init tracking
        patience_counter = 0
        self._history: list[dict] = []
        iteration = 0

        # Save the global best score + features
        global_best_score = current_score
        global_best_features = list(selected)

        if self.verbose >= 1:
            print(f"  Start: seed={selected}, baseline score={current_score:.4f}")

        # 5. Forward selection loop
        while True:
            iteration += 1
            candidates = self._get_candidate_features(X_columns, selected)

            # No more candidates -> stop
            if not candidates:
                if self.verbose >= 1:
                    print(" No more candidates. Stoping..")
                break

            # Find best candidate this iteration
            best_feature, best_score = self._select_best_candidate(
                X, y, selected, candidates
            )

            # Always append the top 1 features
            selected.append(best_feature)

            step_improvement = best_score - current_score
            is_new_peak = best_score > global_best_score

            # Log iteration
            self._history.append(
                {
                    "iteration": iteration,
                    "best_candidate": best_feature,
                    "best_score": round(best_score, 6),
                    "improvement": round(step_improvement, 6),
                    "n_selected": len(selected),
                    "selected_features": list(selected),
                    "is_new_peak": is_new_peak,
                }
            )

            current_score = best_score

            if self.verbose >= 2:
                print(
                    f"  Iter {iteration:>3}: + {best_feature}"
                    f"  score={best_score:.4f}"
                    f"  󰇂 = {step_improvement:+.4f} "
                )

            # Accept or reject
            if is_new_peak:
                global_best_score = best_score
                global_best_features = list(selected)
                patience_counter = 0
            else:
                patience_counter += 1
                if self.verbose >= 2:
                    print(
                        f"  No improvement. Patience{patience_counter}/{self.patience}"
                    )

            # Stoping criteria
            if self.max_features is not None and len(selected) >= self.max_features:
                if self.verbose >= 1:
                    print(f" Reached max_features={self.max_features}. Stoping...")
                break

            if self.patience is not None and patience_counter >= self.patience:
                if self.verbose >= 1:
                    print(f" Patience exhausted ({self.patience}). Stoping...")
                break

        # 6. Store Results
        self.selected_features_ = global_best_features
        self.n_features_in_ = X.shape[1]
        self._X_columns = X_columns
        self.model = _original_model

        if self.verbose >= 1:
            print(
                f"\n Done: {len(selected)} features selected. Final score={current_score:.4f}"
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
        if not hasattr(self, "_history") or not self._history:
            raise RuntimeError(" Error: we dont have _history, do u run fit() ?")

        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            pd.DataFrame(self._history).to_csv(file_path, index=False)
        elif ext == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(
                f" Error: not support {ext} file, pls chosing .json or .csv"
            )
        if self.verbose >= 1:
            print(f" saved history in {file_path}.")
