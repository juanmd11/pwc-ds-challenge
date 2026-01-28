# Salary Prediction Challenge

This project builds and evaluates regression models to predict **Salary** using structured tabular data.  
The goal is to compare a simple baseline against a more informative model and justify modeling decisions based on performance and generalization.

---

## Project structure

```
pwc-ds-challenge/
├── data/
│   └── raw/
│       ├── people.csv         # Personal information dataset
│       ├── salary.csv         # Salary information
│       └── descriptions.csv   # Job descriptions
├── models/                    # Serialized model artifacts
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_final_report.ipynb # Final modeling pipeline and evaluation
├── src/
│   ├── data.py               # Data loading and preprocessing
│   ├── features.py           # Feature engineering
│   ├── models.py             # Model training and utilities
│   └── evaluation.py         # Model evaluation metrics
├── pyproject.toml            # Project metadata and configuration
└── README.md                 # Project documentation
```

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

---

## Setup Instructions

### 1. Environment Setup

This project uses `uv` for fast Python package management. Follow these steps to set up your environment:

#### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Create virtual environment and install dependencies
```bash
# Create virtual environment and install all dependencies
uv sync --extra lab

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

#### Run Jupyter Notebooks

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to the `notebooks/` directory to access the notebooks.

---

## Running the Full Pipeline

To reproduce the complete analysis and model training:

1. **Execute all steps in the final report notebook**: [notebooks/02_final_report.ipynb](notebooks/02_final_report.ipynb)
   
   This notebook contains the complete pipeline:
   - Data loading and preprocessing
   - Feature engineering
   - Model training (baseline and linear regression)
   - Model evaluation with bootstrap confidence intervals
   - Model serialization for deployment

   Run all cells in sequence to generate the trained model artifacts.

---
