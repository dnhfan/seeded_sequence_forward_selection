from datetime import datetime
from typing import List, Optional

import numpy
import pandas
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils.Timer import TimerContext
from src.wrapper.base import BaseWrapperSelector
from src.wrapper.models import get_model
from src.wrapper.sfs_result import SFSResult


class SklearnSFSSelector(BaseWrapperSelector):
    """
    Subclass for sklearn SequentialFeatureSelector engine
    """

    def __init__(
        self,
        *args,
        estimator: Optional[BaseEstimator] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.estimator = estimator

    def _execute_core(
        self,
        X_in: pandas.DataFrame,
        y_in: pandas.Series,
        sfs_params: dict,
        direction: str = "forward",
    ) -> SFSResult:

        # 1. taking needed params form sfs
        max_features: str | int = sfs_params.get("max_features", 1)
        cv_value: int = int(sfs_params.get("cv", 5))
        scoring: str = sfs_params.get("scoring", "accuracy")
        model_name = sfs_params.get("model", "logistic")

        # get the model + max_features
        estimator: BaseEstimator = self.estimator or get_model(model_name)
        sfs_params["model"] = estimator.__class__.__name__

        if max_features == "auto":
            n_features_to_select = "auto"
        else:
            n_features_to_select = int(max_features)

        # make fixed cv
        cv_splitter = StratifiedKFold(n_splits=cv_value, shuffle=True, random_state=42)

        # 2. using that params to build Sklearn_SFS
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features_to_select,  # type: ignore[arg-type]
            tol=0.001,
            direction=direction,
            scoring=scoring,
            cv=cv_splitter,  # type: ignore[arg-type]
            n_jobs=-1,
        )

        # 3. fit and timer
        with TimerContext(name="Sklearn_SFS", unit="ms") as fit_timer:
            selector.fit(X_in, y_in)

        total_fit_time_ms = fit_timer.elapsed

        # 4. store the result
        X_selected = selector.transform(X_in)  # this will return a numpy.ndarray
        selected_features = list(X_in.columns[selector.get_support()])
        X_selected_df = pandas.DataFrame(X_selected, columns=selected_features)

        # build the result df
        df_final = pandas.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        # global score (manual calc 😥)
        scores: numpy.ndarray = cross_val_score(
            estimator=estimator,
            X=X_selected_df,
            y=y_in,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
        )
        global_best_score = float(scores.mean())

        # history_text (manual wirte 😢)
        history_text = self._write_history(
            selected_features,
            global_best_score,
            total_fit_time_ms,
            sfs_params,
            direction,
        )

        return SFSResult(
            df_final=df_final,
            selected_features=selected_features,
            total_fit_time_ms=total_fit_time_ms,
            global_best_score=global_best_score,
            history_text=history_text,
        )

    def _write_history(
        self,
        selected_features: List[str],
        global_best_score: float,
        total_fit_time_ms: float,
        # for config
        sfs_params: dict,
        direction: str,
    ) -> str:

        lines = []

        lines.append("=" * 60)
        lines.append(" SKLEARN SEQUENTIAL FEATURE SELECTION (SFS) - REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        lines.append("-" * 40)
        lines.append(" CONFIGURATION")
        lines.append("-" * 40)
        lines.append(f"  {'Direction:':<20} {direction}")
        lines.append(f"  {'Model:':<20} {sfs_params.get('model', 'N/A')}")
        lines.append(f"  {'Scoring:':<20} {sfs_params.get('scoring', 'N/A')}")
        lines.append(f"  {'CV Folds:':<20} {sfs_params.get('cv', 'N/A')}")
        lines.append(f"  {'Max Features:':<20} {sfs_params.get('max_features', 'N/A')}")
        lines.append("")

        lines.append("-" * 40)
        lines.append("  SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  {'Total Fit Time:':<20} {total_fit_time_ms:.3f} ms")
        lines.append(f"  {'Final Best Score:':<20} {global_best_score:.6f}")
        lines.append(f"  {'Total Selected:':<20} {len(selected_features)}")
        lines.append(f"  {'Selected Features:':<20} {', '.join(selected_features)}")
        lines.append("")

        lines.append("-" * 40)
        lines.append(" ITERATION LOGS")
        lines.append("-" * 40)
        lines.append("  [!] Not available.")
        lines.append(
            "  Sklearn's SequentialFeatureSelector does not track per-iteration history."
        )
        lines.append("")

        lines.append("=" * 60)
        lines.append(" END OF REPORT")
        lines.append("=" * 60)

        return "\n".join(lines)
