from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(include_job_title: bool) -> ColumnTransformer:
    
    # 1) Definimos qué columnas son numéricas y cuáles categóricas
    numeric_features = ["Age", "Years of Experience"]
    gender_features = ["Gender"]
    education_features = ["Education Level"]

    # Job Title se agrega solo si include_job_title=True
    job_features = ["Job Title"] if include_job_title else []

    # 2) Pipeline para columnas numéricas: imputar + escalar
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 3) Pipeline para Gender: imputar + one-hot
    gender_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    # 4) Pipeline para Education: imputar + ordinal
    edu_order = ["Bachelor's", "Master's", "PhD"]
    edu_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            categories=[edu_order],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # 5) Pipeline para Job Title (solo experimento): imputar + one-hot robusto
    job_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # 6) Combinamos todo con ColumnTransformer
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
