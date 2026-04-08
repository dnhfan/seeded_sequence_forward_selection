import pandas

from src.wrapper.base import BaseWrapperSelector
from src.wrapper.forward_selection import SeededForwardSelection
from src.wrapper.sfs_result import SFSResult


class SeededSFSSelector(BaseWrapperSelector):
    """
    Sub class for runing custom SFS algorthm
    """

    def _execute_core(
        self,
        X_in: pandas.DataFrame,
        y_in: pandas.Series,
        sfs_params: dict,
        direction: str = "forward",
    ) -> SFSResult:

        if direction != "forward":
            raise ValueError(" Seeded sfs only support forward!")

        # 1. seed path
        voting_csv_path = str(self.path.ensemble_dir / self.voting_csv_name)

        # 2. taking params/args
        sfs_kwargs = sfs_params.copy()

        # 3. init SFS algorthm instance
        selector = SeededForwardSelection(
            seed_source=voting_csv_path,
            using_timer=self.using_timer,
            unit=self.unit,
            **sfs_kwargs
        )

        # 4. sfs.fit
        selector.fit(X_in, y_in)

        # 5. sfs.transform
        X_selected = selector.transform(X_in)
        selected_features = list(selector.get_feature_names_out())
        X_selected_df = pandas.DataFrame(X_selected, columns=selected_features)

        # 6. making df.final (concat: selected features + labels)
        df_final = pandas.concat([y_in.reset_index(drop=True), X_selected_df], axis=1)

        # 7. making history str
        history_str = selector.generate_txt_report()

        # 8. return the SFS
        return SFSResult(
            df_final=df_final,
            selected_features=selected_features,
            total_fit_time_ms=selector.total_fit_time_ms_,
            global_best_score=selector.global_best_score_,
            history_text=history_str,
        )
