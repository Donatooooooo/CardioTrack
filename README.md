# Predicting Outcomes in Heart Failure

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Organization](#project-organization)  
3. [DVC Pipeline Defined](#dvc-pipeline-defined)  
4. [Milestones Summary](#milestones-summary)  
   - [Milestone 1 - Inception](#milestone-1---inception)  
   - [Milestone 2 - Reproducibility](#milestone-2---reproducibility)
   - [Milestone 3 - Quality Assurance](#milestone-3---quality-assurance)
   - [Milestone 4 - API Integration](#milestone-4---API-Integration)

## Project Overview
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
           ***          ***
          *                *
        **                  ***
+----------+                   *
| training |                ***
+----------+               *
           ***          ***
              *        *
               **    **
            +------------+
            | evaluation |
            +------------+
```

## Milestones Summary

### Milestone 1 - Inception
During this milestone, the **CCDS Project Template** was used as the foundation for organizing the project.
The main conceptual and structural components of the system were defined, following the template guidelines to ensure consistency and traceability.

Additionally, a **Machine Learning Canvas** has been added in the [`docs/`](./docs) folder.
It outlines the model objectives, the data to be used, and the key methodological aspects planned for the next phases of the project.

### Milestone 2 - Reproducibility
Milestone-2 introduces **reproducibility**, from **data management** to **model training and evaluation**. This includes a fully automated pipeline, experiment tracking, and model registry integration, ensuring every step can be consistently reproduced and monitored.

#### Exploratory Data Analysis (EDA)
As part of the early steps, we added and refined an **Exploratory Data Analysis** to better understand the dataset, its distribution, and relationships between variables. This helped define the preprocessing and modeling strategies used later.

#### DVC Initialization and Pipeline Setup
We initialized **DVC** and configured a full pipeline to automate the main steps of the ML workflow:
- Automatic data **download**
- **Preprocessing**
- **Data splitting**
- **Training** and **evaluation**

The pipeline is fully reproducible and version-controlled through DVC.

#### Model Training and Experiment Tracking
We implemented the **training scripts** and integrated **MLflow** for experiment tracking.  
Three models are trained and evaluated within this workflow:
- Decision Tree  
- Random Forest  
- Logistic Regression  

Each experiment is logged to MLflow.

#### Model Registry and Thresholds
Models that reach or exceed the predefined **performance thresholds** (as defined in the ML Canvas) are automatically **saved to the model registry**.  

### Milestone 3 – Quality Assurance

In this milestone, we introduced  **Quality Assurance** layer to the system.

#### Static Linters
Two static linters were added to improve code style and consistency:

- **Ruff** for Python files in the `predicting_outcomes_in_heart_failure` and `tests` folders.
  It checks formatting, syntax, and common anti-patterns, and is integrated into the GitHub workflow via an *action*.
- **Pynblint** for Jupyter notebooks, also integrated into the GitHub workflow through a dedicated *action*.

#### Data Quality
We implemented **data quality checks** on both raw and processed data using **Great Expectations**.
These validations help to:

- detect anomalies or invalid values at the data source
- prevent the propagation of data issues into downstream processes

#### Code Quality
We added automated **unit and integration tests** using **pytest**, covering the main modules and functionalities of the system.


#### ML Pipeline Enhancements
 we applied the following enhancements to the ML pipeline:

- Refactored preprocessing with gender-based dataset variants.
- Added validation (e.g., error on single-row datasets).
- Saved StandardScaler as preprocessing artifact.
- Updated split logic and DVC pipeline.
- Training now creates variant-specific MLflow experiments.
- Added RandomOverSampler to address class imbalance.
- Updated evaluation and inference to align with the new structure.

#### Explainability
We applied an explainability module:

- Added SHAP explainability module.
- Added tests for explainability functionality.


#### Risk Classification
We added a **Risk Classification** analysis for the system in accordance with **IMDRF** and **AI Act** regulations.
The documentation is available in the [`docs/`](./docs) folder.

Ecco la versione finale **in Markdown puro**, già formattata correttamente:


### Milestone 4 - API Integration

During Milestone 4, we implemented a fully functional API and Dataset Card and Model card for the champion model and the following used dataset. 
APIs are structured into four main routers:


#### **General Router**
- **GET /**  
  Returns a welcome message and confirms that the API is running.


#### **Prediction Router**
- **POST /predictions**  
  Generates a binary prediction (0/1) for a single patient sample.

- **POST /predict-batch**  
  Accepts a list of patient samples and returns a prediction for each element in the batch.

- **POST /explanations**  
  Produces SHAP-based explanations for a single input and returns the URL of the generated SHAP waterfall plot.


#### **Model Info Router**
- **GET /model/hyperparameters**  
  Returns the hyperparameters and cross-validation results of the model defined in `MODEL_PATH`.

- **GET /model/metrics**  
  Returns the test-set metrics stored during the model evaluation stage.


#### **Cards Router**
- **GET /card/{card_type}**  
  Returns the content of a “card” file (dataset card or model card).


### **Cards**

During this milestone, we also created:

- a **dataset card** describing the dataset used by the champion model  
- a **model card** documenting the champion model itself  


