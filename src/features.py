from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(include_job_title: bool) -> ColumnTransformer:
    
    # 1) Define numerical and categorical feature groups
    numeric_features = ["Age", "Years of Experience"]
    gender_features = ["Gender"]
    education_features = ["Education Level"]

    # Job Title is included only if include_job_title=True (experimental feature)
    job_features = ["Job Title"] if include_job_title else []

    # 2) Pipeline for numerical features: imputation + scaling
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 3) Pipeline for Gender: imputation + one-hot
    gender_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    # 4) Pipeline for Education: imputation + ordinal
    edu_order = ["Bachelor's", "Master's", "PhD"]
    edu_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            categories=[edu_order],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # 5) Pipeline for Job Title (experimental): imputation + one-hot robusto
    job_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # 6) Combine all preprocessing steps using a ColumnTransformer
    transformers = [
        ("num", num_pipe, numeric_features),
        ("gender", gender_pipe, gender_features),
        ("edu", edu_pipe, education_features),
    ]

    if include_job_title:
        transformers.append(("job", job_pipe, job_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor
