# Predicting Outcomes in Heart Failure

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project develops a predictive pipeline for patient outcome prediction in heart failure, using a publicly available dataset of clinical records. The goal is to design and evaluate machine learning models within a reproducible workflow that can be integrated into larger systems for clinical decision support. The workflow addresses data heterogeneity, defines consistent preprocessing and feature engineering strategies, and explores alternative modeling approaches with systematic evaluation using clinically relevant metrics. It also emphasizes model transparency and auditability, ensuring that the resulting pipeline can be deployed as a reliable, adaptable software component in healthcare applications. The project aims not only to improve baseline predictive performance but also to demonstrate how data-driven models can be effectively integrated into end-to-end AI-enabled healthcare systems.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         predicting_outcomes_in_heart_failure and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── predicting_outcomes_in_heart_failure   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes predicting_outcomes_in_heart_failure a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data               
    │   ├── __init__.py 
    │   ├── dataset.py          <- Scripts to download or generate data
    |   ├── preprocess.py       <- Data preprocessing code 
    │   └── split_data.py       <- Split dataset into train and test code
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## DVC Pipeline defined
```
+---------------+  
| download_data |  
+---------------+  
        *
        *
        *
+---------------+  
| preprocessing |  
+---------------+  
        *
        *
        *
  +------------+
  | split_data |
  +------------+
        *
        *
        *
  +----------+
  | training |
  +----------+
        *
        *
        *
  +------------+
  | evaluation |
  +------------+
```

