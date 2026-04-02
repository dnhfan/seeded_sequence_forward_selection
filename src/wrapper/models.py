import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name: str, random_state: int = 42):
    """
    Model factory: return the chosing model
    """

    name = model_name.lower().strip()

    if name == "logistic" or name == "log":
        return LogisticRegression(max_iter=5000, random_state=random_state)

    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "svm":
        return SVC(kernel="linear", probability=True, random_state=random_state)

    elif name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "dt" or name == "decisiontree":
        return DecisionTreeClassifier(max_depth=5, random_state=random_state)
    else:
        raise ValueError(f" Model '{model_name}' is not supported.")
