# Salary Prediction Challenge

This project builds and evaluates regression models to predict **Salary** using structured tabular data.  
The goal is to compare a simple baseline against a more informative model and justify modeling decisions based on performance and generalization.

---

## Project structure

project1/
├── data/
│ └── raw/
│ ├── people.csv
│ ├── salary.csv
│ └── descriptions.csv
├── notebooks/
│ ├── 01_eda.ipynb
│ └── 02_final_report.ipynb
├── src/
│ ├── data.py
│ ├── features.py
│ ├── models.py
│ └── evaluation.py
├── requirements.txt
└── README.md


---

## Notebooks

- **01_eda.ipynb**  
  Exploratory Data Analysis: data inspection, distributions, and initial insights.

- **02_final_report.ipynb**  
  Final modeling report:
  - baseline vs linear regression
  - preprocessing pipelines
  - model evaluation using MAE and bootstrap confidence intervals
  - discussion of the Job Title feature

---

## Modeling approach

- **Baseline**: DummyRegressor to establish a minimum performance reference.
- **Linear Regression**: trained with numerical and categorical features using a preprocessing pipeline.
- **Evaluation**: Holdout split with MAE as the main metric.  
  Bootstrap resampling is used to estimate confidence intervals.

Job Title is evaluated as an additional experiment due to its high cardinality and potential impact on generalization.

