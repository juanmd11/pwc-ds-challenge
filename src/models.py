from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

def build_model(model_name: str, preprocessor) -> Pipeline:
    """
    Build a full modeling pipeline: preprocessing + estimator.

    Parameters
    ----------
    model_name : str
        Which model to build. Supported values: "dummy", "linear".
    preprocessor :
        A scikit-learn transformer (e.g., ColumnTransformer) that will be fitted
        on training data and applied to both train and test before modeling.

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline with steps: ("preprocess" -> "model").
    """
    if model_name == "dummy":
        # Baseline model: ignores input features and predicts a constant value
        model = DummyRegressor(strategy="median")

    elif model_name == "linear":
        # Linear regression over the preprocessed features
        model = LinearRegression()

    else:
        raise ValueError(f"Unknown model_name='{model_name}'. Use 'dummy' or 'linear'.")

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipeline
