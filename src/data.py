# src/data.py

import pandas as pd

def load_and_merge_data(people_path: str, salary_path: str, descriptions_path: str | None = None) -> pd.DataFrame:
    people = pd.read_csv(people_path)
    salary = pd.read_csv(salary_path)

    df = people.merge(salary, on="id", how="inner")

    if descriptions_path is not None:
        desc = pd.read_csv(descriptions_path)
        df = df.merge(desc, on="id", how="left")

    # Target is mandatory: rows without Salary cannot be used for training or evaluation
    df = df.dropna(subset=["Salary"]).reset_index(drop=True)

    return df
