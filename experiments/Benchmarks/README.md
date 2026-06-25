# Benchmark experiment

for acc evaluate, we using `src/modeling/evaluation.py`.

the core evaluate method we using is cross fold validation.

```python
    def _train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method_name: str,
        n_splits: int = 5,
    ) -> None:
        """
        [Private] Splits the data using Cross-Validation, trains Logistic Regression and Decision Tree models,
        evaluates their accuracy, and stores the results.
        """
        # 1. Init the CV
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # scoring
        scoring = ["accuracy"]

        # 2. Init model in a dict -> easy to add new one
        logreg_model = LogisticRegression(max_iter=self.max_iter, random_state=42)
        if self.use_scaler:
            logreg_model = make_pipeline(StandardScaler(), logreg_model)

        models = {
            "LogReg": logreg_model,
            "Tree": DecisionTreeClassifier(random_state=42),
        }

        # 3. Runing each model
        for model_name, model in models.items():
            # cross_validate will auto fit and predict
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            test_accuracy = scores["test_accuracy"]
            mean_acc = test_accuracy.mean()

            # 4. Store the result of each fold
            for i in range(cv.get_n_splits()):
                self.fold_results.append(
                    {
                        "Method": method_name,
                        "Model": model_name,
                        "Fold": i + 1,
                        "Acc": scores["test_accuracy"][i],
                    }
                )
            self.model_results.append(
                {
                    "Method": method_name,
                    "Model": model_name,
                    "mean_acc": mean_acc,
                    "std": test_accuracy.std(),
                    "min": test_accuracy.min(),
                    "max": test_accuracy.max(),
                    "n_folds": len(test_accuracy),
                }
            )

            print(f"󰄭  [{method_name:<12}] {model_name:<8} | Acc: {mean_acc:.4f} ")
```

## Problem

using only one way to evaluate, that may be one-sided.

## solution

we will evalute the reuslt data with:

- custom Train test split
- custom Cross-Validation

### Train test split

- 70/30
- 50 times
  -> avg

### Custom Cross-Validation

write cv from scratch
