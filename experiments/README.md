# Experients: runing sfs in all scenarios

## find all scenarios

### sfs args

```python
    def __init__(
        self,
        seed_source: Union[str, List[str]],
        n_seeds: int = 1,                                               # this
        model: Union[str, BaseEstimator] = "logistic",                  # this
        scoring: str = "accuracy",                                      # this
        cv: int = 5,
        cv_shuffle: bool = True,
        cv_stratified: bool = True,
        max_features: Optional[int] = 100,
        patience: Optional[int] = 5,                                    # this
        tol: float = 0.0,                                               # this
        random_state: int = 42,
        verbose: int = 2,
        n_jobs: int = -1,
        using_timer: bool = True,
        unit: str = "ms",
    ) -> None:
        super().__init__()
        self.seed_source = eed_source
        self.n_seeds = n_seeds
        self.model = model
        self.scoring = scoring
        self.cv = cv
        self.cv_shuffle = cv_shuffle
        self.cv_stratified = cv_stratified
        self.max_features = max_features
        self.patience = patience
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.using_timer = using_timer
        self.unit = unit

```

### insight

we can change:

- model
- seeds
- scoring

to get different scenarios.
we can also change patience and tol, but they are more related to the sfs process itself, not the scenario.
so we will keep them fixed for now.

## Running sfs in all scenarios

## Evaluating results

After running sfs in all scenarios, we will have a lot of results to compare.
I use `experiments/evaluate_benchmark.py` to evaluate the results and compare them for each dataset.

```bash
evaluate_benchmark.py <dataset> <variant>
```
