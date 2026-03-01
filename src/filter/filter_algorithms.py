import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


def calc_variance(X: pd.DataFrame, n_features: int):
    """
    Calculate variance
    return:
        - n_features
        - scores
    """
    print(f"    󰃬 Computing variance for {X.shape[1]} features...")

    variances = X.var()
    scores = variances.to_dict()
    top_features = (
        variances.sort_values(ascending=False).head(n_features).index.tolist()
    )

    print(f"    󰄲 Selected {len(top_features)} features")
    print(f"    󰓫 Top 5 features: {top_features[:5]}")
    print(f"    󰓫 Variance range: [{variances.min():.4f}, {variances.max():.4f}]")

    return top_features, scores, None


def calc_anova(X: pd.DataFrame, y: pd.Series, n_features: int):
    """
    Calculate anova
    return:
        - n_features
        - scores
    """
    print(f"    󰃬 Computing ANOVA F-test for {X.shape[1]} features...")

    selector = SelectKBest(score_func=f_classif, k=n_features)
    selector.fit(X, y)

    scores = np.nan_to_num(selector.scores_)
    scores_dict = dict(zip(X.columns, scores))

    top_features = X.columns[selector.get_support()].tolist()

    # Get top 5 scores for display
    top_5_scores = [scores_dict[f] for f in top_features[:5]]

    print(f"    󰄲 Selected {len(top_features)} features")
    print(f"    󰓫 Top 5 features: {top_features[:5]}")
    print(f"    󰓫 Top 5 F-scores: {[f'{s:.2f}' for s in top_5_scores]}")
    print(f"    󰓫 F-score range: [{scores.min():.2f}, {scores.max():.2f}]")

    return top_features, scores_dict, None


def calc_chi_squared(X: pd.DataFrame, y, n_features: int):
    """
    Calculate chi-squared
    return:
        - n_features
        - scores
    """
    print(f"    󰃬 Computing Chi-squared test for {X.shape[1]} features...")

    scaler = None
    X_processed = X.copy()

    if (X < 0).any().any():
        print(f"    ⚠  Detected negative values, applying MinMaxScaler...")
        scaler = MinMaxScaler()
        X_processed_np = scaler.fit_transform(X)
        X_processed = pd.DataFrame(X_processed_np, columns=X.columns, index=X.index)
        print(f"    󰄲 Data scaled to [0, 1]")
    else:
        print(f"    󰓫 No negative values detected, using raw data")

    selector = SelectKBest(score_func=chi2, k=n_features)
    selector.fit(X_processed, y)

    scores = np.nan_to_num(selector.scores_)
    scores_dict = dict(zip(X.columns, scores))

    top_features = X.columns[selector.get_support()].tolist()

    # Get top 5 scores for display
    top_5_scores = [scores_dict[f] for f in top_features[:5]]

    print(f"    󰄲 Selected {len(top_features)} features")
    print(f"    󰓫 Top 5 features: {top_features[:5]}")
    print(f"    󰓫 Top 5 Chi2 scores: {[f'{s:.2f}' for s in top_5_scores]}")
    print(f"    󰓫 Chi2 score range: [{scores.min():.2f}, {scores.max():.2f}]")

    return top_features, scores_dict, scaler


def calc_correlation(
    X: pd.DataFrame, y: pd.Series, n_features: int, correlation_threshold: float = 0.95
):
    """
    Calculate Correlation (Pearson)

    Steps:
    1. Correlation matrix between features
    2. Drop highly correlated feature (|corr| > threshold)
    3. Ranking remaining features using ANOVA F TEST
    4. Select top N features with highest F-score

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_features : int
        Number of features to select
    correlation_threshold : float, default=0.95
        Correlation threshold for removing redundant features


    Return:
    -------
        - n_features
        - scores
    """
    print(f"    󰃬 Computing correlation matrix for {X.shape[1]} features...")

    # 1. corr_matrix calc
    corr_matrix = X.corr().abs()

    # 2. drop

    # upper triangle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # drop list
    to_drop = [
        column
        for column in upper_tri.columns
        if any(upper_tri[column] > correlation_threshold)
    ]

    print(
        f"    󰓫 Dropping {len(to_drop)} highly correlated features (|corr| > {correlation_threshold})"
    )

    X_filtered = X.drop(columns=to_drop)

    print(f"    󰓫 Remaining features: {len(X_filtered.columns)}")

    # 3. Ranking with f-score
    print(f"    󰃬 Ranking remaining features with ANOVA F-test...")

    # calc f-score
    f_scores, _ = f_classif(X_filtered, y)  # _ is p_value

    # create a df with remaining features + it's f_score
    scores_df = pd.DataFrame(
        {
            "feature": X_filtered.columns,
            "f_scores": f_scores,
        }
    )

    # sort with f-score
    scores_df = scores_df.sort_values("f_scores", ascending=False)

    # n_features arg can be > remaining features
    n_select = min(n_features, len(scores_df))

    if n_select < n_features:
        print(
            f"    ⚠  Requested {n_features} features but only {n_select} available after correlation filtering"
        )

    # take n features
    top_features_df = scores_df.head(n_select)

    scores_dict = dict(zip(top_features_df["feature"], top_features_df["f_scores"]))
    top_features = top_features_df["feature"].tolist()

    # Get top 5 scores for display
    top_5_scores = list(scores_dict.values())[:5]

    print(f"    󰄲 Selected {len(top_features)} features")
    print(f"    󰓫 Top 5 features: {top_features[:5]}")
    print(f"    󰓫 Top 5 F-scores: {[f'{s:.2f}' for s in top_5_scores]}")

    return top_features, scores_dict, None


def calc_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int,
    random_state: int = 42,
    n_neighbors: int = 3,
):
    """
    Calculate mutual information for feature selection

    MI = 0: feature and target are independent
    MI > 0: feature provides information about target

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (categorical)
    n_features : int
        Number of features to select
    random_state : int, default=42
        Random seed for reproducibility
    n_neighbors : int, default=3
        Number of neighbors for MI estimation (lower = faster)

    Returns:
    --------
    top_features : list
        Selected feature names
    scores_dict : dict
        Feature names mapped to MI scores
    None : None
        Placeholder for consistency
    """
    print(f"    󰃬 Computing mutual information for {X.shape[1]} features...")
    print(f"    ⚠  This may take 1-2 minutes with many features...")

    # 1. Compute MI score
    mi_scores = mutual_info_classif(
        X,
        y,
        discrete_features=False,  # type: ignore
        random_state=random_state,
        n_neighbors=n_neighbors,
    )

    print(f"    󰄲 MI computation completed")

    # 2. DataFrame and sort
    feature_scores_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mi_scores": mi_scores,
        }
    )

    feature_scores_df = feature_scores_df.sort_values("mi_scores", ascending=False)

    # 3. Select n features
    top_features_df = feature_scores_df.head(n_features)
    top_features = top_features_df["feature"].tolist()

    scores_dict = dict(zip(top_features_df["feature"], top_features_df["mi_scores"]))

    # Get top 5 scores for display
    top_5_scores = list(scores_dict.values())[:5]

    print(f"    󰄲 Selected {len(top_features)} features")
    print(f"    󰓫 Top 5 features: {top_features[:5]}")
    print(f"    󰓫 Top 5 MI scores: {[f'{s:.4f}' for s in top_5_scores]}")
    print(f"    󰓫 MI score range: [{mi_scores.min():.4f}, {mi_scores.max():.4f}]")

    return top_features, scores_dict, None
