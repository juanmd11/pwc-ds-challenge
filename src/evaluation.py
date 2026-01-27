import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def bootstrap_mae_ci(y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05, random_state: int = 42):
    """
    Estimate a confidence interval for MAE using bootstrap resampling.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Model predictions.
    n_boot : int
        Number of bootstrap samples.
    alpha : float
        Significance level (0.05 -> 95% CI).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    mae_mean : float
        Mean MAE across bootstrap samples.
    ci_low : float
        Lower bound of the confidence interval.
    ci_high : float
        Upper bound of the confidence interval.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    maes = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        maes[i] = mean_absolute_error(y_true[idx], y_pred[idx])

    ci_low = float(np.quantile(maes, alpha / 2))
    ci_high = float(np.quantile(maes, 1 - alpha / 2))
    mae_mean = float(np.mean(maes))

    return mae_mean, ci_low, ci_high


def evaluate_holdout(df, target_col: str, pipeline, test_size: float = 0.2, random_state: int = 42, with_ci: bool = True):
    """
    Train/evaluate a pipeline using a holdout split and return MAE (and optional bootstrap CI).

    Parameters
    ----------
    df : pd.DataFrame
        Modeling dataset containing features + target.
    target_col : str
        Name of the target column (e.g., "Salary").
    pipeline : sklearn Pipeline
        Full pipeline: preprocess -> model.
    test_size : float
        Fraction of data for test split.
    random_state : int
        Random seed for reproducibility.
    with_ci : bool
        Whether to compute bootstrap CI on test predictions.

    Returns
    -------
    dict with metrics
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))

    results = {
        "mae": mae,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    if with_ci:
        mae_mean, ci_low, ci_high = bootstrap_mae_ci(y_test, y_pred)
        results.update({
            "mae_boot_mean": mae_mean,
            "mae_ci_low": ci_low,
            "mae_ci_high": ci_high,
        })

    return results
