import pandas as pd
from imblearn.over_sampling import SMOTE


def load_and_preprocess(filepath, target_index=0, save_path=None):
    """
    Pipeline for clean and balances the dataset:
        - filepath: data filepath
        - target_index: index of target collume
    """

    # 1. load data
    df = pd.read_csv(filepath)

    # 2. split X and Y
    y = df.iloc[:, target_index]

    x = df.drop(columns=df.columns[target_index])

    # 3. data balancing with SMOTE!
    try:
        smote = SMOTE(random_state=42)
        x_resampled, y_resampled = smote.fit_resample(x, y)  # type: ignore

    except ValueError as e:
        error_msg = str(e)
        if "n_neighbors" in error_msg or "n_sample_fit" in error_msg:
            print(" ValueError: n_sample_fit is > n_neighbors, cannot SMOTE!")
        else:
            print(f" ValueError: {e}")
        raise

    x_final = pd.DataFrame(x_resampled, columns=x.columns)

    print(f"󰄲 Preprocessed file: {filepath}")
    print(f"󰓫 X.Shape before preprocess: {x.shape}")
    print(f"󰓫 X.Shape after preprocess: {x_final.shape}")

    if save_path:
        df_processed = x_final.copy()

        df_processed.insert(0, "V1", y_resampled)

        df_processed.to_csv(save_path, index=False)

        print(f" Saved preprocess data in {save_path}")

    return x_final, y_resampled


# in file test function
if __name__ == "__main__":
    raw_path = "data/raw/Lung_cancer.csv"
    save_path = "data/processed/Lung_cancer_preprocessed.csv"

    x_train, y_train = load_and_preprocess(filepath=raw_path, save_path=save_path)
